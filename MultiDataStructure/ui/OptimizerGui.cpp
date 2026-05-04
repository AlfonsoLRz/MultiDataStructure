#include "OptimizerGui.h"

#include "../experiments/SchemaSearch.h"
#include "../workloads/points/LBVH.h"

namespace
{
	constexpr size_t TextBufferSize = 512;

	struct SchemaEntry
	{
		std::string label;
		std::string path;
		bool selected = true;
	};

	struct WorkloadEntry
	{
		std::string label;
		std::string path;
	};

	struct BestResult
	{
		std::string dataset;
		std::string workload;
		std::string schemaName;
		std::string schemaPath;
		double score = 0.0;
		double averageLatencyMs = 0.0;
		double buildTimeMs = 0.0;
		double gpuBuildMs = 0.0;
		double gpuQueryMs = 0.0;
		uint64_t memoryBytes = 0;
		size_t candidates = 0;
		std::string backend;
	};

	struct LiveRankingEntry
	{
		size_t order = 0;
		std::string dataset;
		std::string workload;
		std::string schemaName;
		std::string schemaPath;
		std::string backend;
		std::string cudaBuilder;
		double score = 0.0;
		double averageLatencyMs = 0.0;
		double p95LatencyMs = 0.0;
		double buildTimeMs = 0.0;
		double gpuBuildMs = 0.0;
		double gpuQueryMs = 0.0;
		double averageVisitedNodes = 0.0;
		double averageTestedPoints = 0.0;
		uint64_t memoryBytes = 0;
	};

	struct GuiState
	{
		std::array<char, TextBufferSize> inputPath{};
		std::array<char, TextBufferSize> rankModelPath{};
		std::array<char, TextBufferSize> csvPath{};
		std::array<char, TextBufferSize> bestCsvPath{};
		std::array<char, TextBufferSize> generatedSchemaDir{};

		std::vector<SchemaEntry> schemas;
		std::vector<WorkloadEntry> workloads;
		int selectedWorkload = 0;

		bool includeSynthetic = false;
		bool useBinaryCache = true;
		bool rebuildBinaryCache = false;
		bool generateSchemas = true;
		bool generatedOnly = false;
		bool generatedConditional = true;
		bool useRankModel = false;
		bool optimizeSchemas = false;
		int evaluator = 0;
		int cudaDevice = 0;
		int cudaBuilder = 0;
		int cudaQueryBatch = 0;
		int cudaMemoryBudgetMb = 0;
		int liveRankingTopN = 10;

		int queryCount = 128;
		int knnK = 16;
		int querySeed = 1337;
		int syntheticScale = 512;
		int generatedCount = 1000;
		int benchmarkTopK = 0;
		int generatedMaxBlocks = 3;
		int generatedMaxDepth = 12;
		int generatedMinLeaf = 32;
		int generatedMaxLeaf = 32768;
		int generatedSeed = 1337;
		int optimizerGenerations = 3;
		int optimizerPopulation = 64;
		int optimizerElites = 6;
		int optimizerSeed = 1337;
		float generatedConditionProbability = 0.5f;
		float optimizerMutationRate = 0.65f;
		float optimizerRandomFraction = 0.20f;
		float scoreBuildWeight = 0.0f;
		float scoreMemoryWeight = 0.0f;
		float scoreImbalanceWeight = 0.0f;
	};

	struct RunSession
	{
		std::thread worker;
		std::atomic<bool> running = false;
		std::atomic<bool> finished = false;
		std::mutex mutex;
		std::string status = "Idle";
		std::string log;
		std::string error;
		std::vector<BestResult> bestResults;
		std::vector<LiveRankingEntry> liveRanking;
		size_t evaluatedCandidates = 0;
		int exitCode = 0;
	};

	template <size_t N>
	void setText(std::array<char, N>& buffer, const std::string& text)
	{
		buffer.fill('\0');
		const size_t count = std::min(text.size(), N - 1);
		std::memcpy(buffer.data(), text.data(), count);
	}

	template <size_t N>
	std::string textValue(const std::array<char, N>& buffer)
	{
		return std::string(buffer.data());
	}

	bool filesystemExists(const std::filesystem::path& value)
	{
		if (value.empty())
			return false;

		std::error_code error;
		return std::filesystem::exists(value, error);
	}

	void addAncestorSearchRoots(std::vector<std::filesystem::path>& roots, std::filesystem::path start)
	{
		if (start.empty())
			return;

		std::error_code error;
		start = std::filesystem::absolute(start, error);
		if (error)
			return;

		if (start.has_filename() && start.extension() == ".exe")
			start = start.parent_path();

		for (;;)
		{
			const std::filesystem::path normalized = start.lexically_normal();
			if (std::find(roots.begin(), roots.end(), normalized) == roots.end())
				roots.push_back(normalized);

			if (!start.has_parent_path() || start == start.parent_path())
				break;

			start = start.parent_path();
		}
	}

	std::filesystem::path executablePath()
	{
#ifdef _WIN32
		char* programPath = nullptr;
		if (_get_pgmptr(&programPath) == 0 && programPath)
			return std::filesystem::path(programPath);
#endif
		return {};
	}

	std::vector<std::filesystem::path> searchRoots()
	{
		std::vector<std::filesystem::path> roots;
		addAncestorSearchRoots(roots, std::filesystem::current_path());
		addAncestorSearchRoots(roots, executablePath());
		return roots;
	}

	std::filesystem::path projectRoot()
	{
		for (const std::filesystem::path& root : searchRoots())
		{
			if (filesystemExists(root / "configs" / "schemas") && filesystemExists(root / "MultiDataStructure.sln"))
				return root;
		}

		return {};
	}

	std::string projectPath(const std::string& relativePath)
	{
		const std::filesystem::path root = projectRoot();
		if (root.empty())
			return relativePath;
		return (root / relativePath).lexically_normal().string();
	}

	std::string resolvePath(const std::string& configuredPath)
	{
		if (configuredPath.empty())
			return configuredPath;

		const std::filesystem::path path(configuredPath);
		if (filesystemExists(path))
			return path.lexically_normal().string();

		if (path.is_absolute())
			return configuredPath;

		for (const std::filesystem::path& root : searchRoots())
		{
			const std::filesystem::path candidate = (root / path).lexically_normal();
			if (filesystemExists(candidate))
				return candidate.string();
		}

		return configuredPath;
	}

	std::string schemaLabelFromPath(const std::string& path)
	{
		std::filesystem::path schemaPath(path);
		std::string stem = schemaPath.stem().string();
		for (char& c : stem)
		{
			if (c == '_' || c == '-')
				c = ' ';
		}
		return stem.empty() ? path : stem;
	}

	std::vector<std::string> splitCsvLine(const std::string& line)
	{
		std::vector<std::string> values;
		std::string current;
		bool quoted = false;

		for (size_t i = 0; i < line.size(); ++i)
		{
			const char c = line[i];
			if (quoted)
			{
				if (c == '"' && i + 1 < line.size() && line[i + 1] == '"')
				{
					current.push_back('"');
					++i;
				}
				else if (c == '"')
				{
					quoted = false;
				}
				else
				{
					current.push_back(c);
				}
				continue;
			}

			if (c == '"')
			{
				quoted = true;
			}
			else if (c == ',')
			{
				values.push_back(current);
				current.clear();
			}
			else
			{
				current.push_back(c);
			}
		}

		values.push_back(current);
		return values;
	}

	size_t columnIndex(const std::vector<std::string>& header, const std::string& name)
	{
		const auto found = std::find(header.begin(), header.end(), name);
		if (found == header.end())
			return std::numeric_limits<size_t>::max();
		return static_cast<size_t>(std::distance(header.begin(), found));
	}

	std::string csvValue(const std::vector<std::string>& row, size_t index)
	{
		if (index == std::numeric_limits<size_t>::max() || index >= row.size())
			return {};
		return row[index];
	}

	double parseDouble(const std::string& value)
	{
		if (value.empty())
			return 0.0;
		try
		{
			return std::stod(value);
		}
		catch (...)
		{
			return 0.0;
		}
	}

	uint64_t parseUint64(const std::string& value)
	{
		if (value.empty())
			return 0;
		try
		{
			return static_cast<uint64_t>(std::stoull(value));
		}
		catch (...)
		{
			return 0;
		}
	}

	std::vector<BestResult> loadBestResults(const std::string& bestCsvPath)
	{
		std::vector<BestResult> results;
		std::ifstream input(bestCsvPath);
		if (!input.is_open())
			return results;

		std::string headerLine;
		if (!std::getline(input, headerLine))
			return results;

		const std::vector<std::string> header = splitCsvLine(headerLine);
		const size_t datasetIndex = columnIndex(header, "dataset_name");
		const size_t workloadIndex = columnIndex(header, "workload_name");
		const size_t schemaNameIndex = columnIndex(header, "best_schema_name");
		const size_t schemaPathIndex = columnIndex(header, "best_schema_path");
		const size_t scoreIndex = columnIndex(header, "best_score");
		const size_t latencyIndex = columnIndex(header, "best_avg_latency_ms");
		const size_t buildIndex = columnIndex(header, "best_build_time_ms");
		const size_t memoryIndex = columnIndex(header, "best_memory_estimate_bytes");
		const size_t candidateIndex = columnIndex(header, "num_candidates");
		const size_t backendIndex = columnIndex(header, "backend");
		const size_t gpuBuildIndex = columnIndex(header, "gpu_build_ms");
		const size_t gpuQueryIndex = columnIndex(header, "gpu_query_ms");

		std::string rowLine;
		while (std::getline(input, rowLine))
		{
			if (rowLine.empty())
				continue;

			const std::vector<std::string> row = splitCsvLine(rowLine);
			BestResult result;
			result.dataset = csvValue(row, datasetIndex);
			result.workload = csvValue(row, workloadIndex);
			result.schemaName = csvValue(row, schemaNameIndex);
			result.schemaPath = csvValue(row, schemaPathIndex);
			result.score = parseDouble(csvValue(row, scoreIndex));
			result.averageLatencyMs = parseDouble(csvValue(row, latencyIndex));
			result.buildTimeMs = parseDouble(csvValue(row, buildIndex));
			result.gpuBuildMs = parseDouble(csvValue(row, gpuBuildIndex));
			result.gpuQueryMs = parseDouble(csvValue(row, gpuQueryIndex));
			result.memoryBytes = parseUint64(csvValue(row, memoryIndex));
			result.candidates = static_cast<size_t>(parseUint64(csvValue(row, candidateIndex)));
			result.backend = csvValue(row, backendIndex);
			results.push_back(std::move(result));
		}

		return results;
	}

	LiveRankingEntry makeLiveRankingEntry(const Experiments::SchemaSearchRecord& record, size_t order)
	{
		LiveRankingEntry entry;
		entry.order = order;
		entry.dataset = record.datasetName;
		entry.workload = record.workloadName;
		entry.schemaName = record.schemaName;
		entry.schemaPath = record.schemaPath;
		entry.backend = record.backend;
		entry.cudaBuilder = record.cudaBuilder;
		entry.score = record.score;
		entry.averageLatencyMs = record.queryMetrics.averageLatencyMs;
		entry.p95LatencyMs = record.queryMetrics.p95LatencyMs;
		entry.buildTimeMs = record.buildMetrics.buildTimeMs;
		entry.gpuBuildMs = record.gpuBuildMs;
		entry.gpuQueryMs = record.gpuQueryMs;
		entry.averageVisitedNodes = record.queryMetrics.averageVisitedNodes;
		entry.averageTestedPoints = record.queryMetrics.averageTestedPoints;
		entry.memoryBytes = static_cast<uint64_t>(record.buildMetrics.memoryEstimateBytes);
		return entry;
	}

	bool sameLiveCandidate(const LiveRankingEntry& entry, const Experiments::SchemaSearchRecord& record)
	{
		return entry.dataset == record.datasetName &&
			entry.workload == record.workloadName &&
			entry.schemaName == record.schemaName &&
			entry.schemaPath == record.schemaPath;
	}

	void sortAndTrimLiveRanking(std::vector<LiveRankingEntry>& ranking)
	{
		std::sort(ranking.begin(), ranking.end(), [](const LiveRankingEntry& left, const LiveRankingEntry& right) {
			if (left.score != right.score)
				return left.score < right.score;
			return left.order < right.order;
		});

		constexpr size_t MaxLiveRankingRows = 128;
		if (ranking.size() > MaxLiveRankingRows)
			ranking.resize(MaxLiveRankingRows);
	}

	void updateLiveRanking(RunSession& session, const Experiments::SchemaSearchRecord& record)
	{
		std::lock_guard<std::mutex> lock(session.mutex);
		const size_t order = ++session.evaluatedCandidates;
		const auto existing = std::find_if(session.liveRanking.begin(), session.liveRanking.end(), [&record](const LiveRankingEntry& entry) {
			return sameLiveCandidate(entry, record);
		});

		if (existing == session.liveRanking.end())
		{
			session.liveRanking.push_back(makeLiveRankingEntry(record, order));
		}
		else if (record.score < existing->score)
		{
			*existing = makeLiveRankingEntry(record, order);
		}

		sortAndTrimLiveRanking(session.liveRanking);
	}

	class SessionStreamBuffer : public std::streambuf
	{
	public:
		explicit SessionStreamBuffer(RunSession& session)
			: _session(session)
		{
		}

	protected:
		int overflow(int value) override
		{
			if (value == traits_type::eof())
				return traits_type::not_eof(value);

			const char c = traits_type::to_char_type(value);
			std::lock_guard<std::mutex> lock(_session.mutex);
			_session.log.push_back(c);
			return value;
		}

		std::streamsize xsputn(const char* text, std::streamsize count) override
		{
			std::lock_guard<std::mutex> lock(_session.mutex);
			_session.log.append(text, static_cast<size_t>(count));
			return count;
		}

	private:
		RunSession& _session;
	};

	class ScopedStreamCapture
	{
	public:
		explicit ScopedStreamCapture(RunSession& session)
			: _buffer(session),
			  _oldOut(std::cout.rdbuf(&_buffer)),
			  _oldErr(std::cerr.rdbuf(&_buffer))
		{
		}

		~ScopedStreamCapture()
		{
			std::cout.rdbuf(_oldOut);
			std::cerr.rdbuf(_oldErr);
		}

	private:
		SessionStreamBuffer _buffer;
		std::streambuf* _oldOut = nullptr;
		std::streambuf* _oldErr = nullptr;
	};

	std::vector<SchemaEntry> discoverSchemas()
	{
		const std::vector<std::string> preferred = {
			"configs/schemas/quadtree.json",
			"configs/schemas/octree.json",
			"configs/schemas/kdtree.json",
			"configs/schemas/bvh.json",
			"configs/schemas/quadtree_octree.json",
			"configs/schemas/octree_kdtree.json",
			"configs/schemas/urban_hybrid.json",
			"configs/schemas/adaptive_quadtree_octree.json",
		};

		std::vector<SchemaEntry> entries;
		for (const std::string& path : preferred)
		{
			const std::string resolved = resolvePath(path);
			if (!filesystemExists(resolved))
				continue;

			SchemaEntry entry;
			entry.label = schemaLabelFromPath(path);
			entry.path = resolved;
			entry.selected = path.find("adaptive") == std::string::npos;
			entries.push_back(std::move(entry));
		}

		std::error_code error;
		const std::filesystem::path schemaDir(resolvePath("configs/schemas"));
		if (std::filesystem::exists(schemaDir, error))
		{
			for (const std::filesystem::directory_entry& file : std::filesystem::directory_iterator(schemaDir, error))
			{
				if (error || !file.is_regular_file() || file.path().extension() != ".json")
					continue;

				const std::string path = file.path().lexically_normal().string();
				const auto existing = std::find_if(entries.begin(), entries.end(), [&path](const SchemaEntry& entry) {
					return entry.path == path || std::filesystem::path(entry.path).lexically_normal() == std::filesystem::path(path).lexically_normal();
				});
				if (existing != entries.end())
					continue;

				SchemaEntry entry;
				entry.label = schemaLabelFromPath(path);
				entry.path = path;
				entry.selected = false;
				entries.push_back(std::move(entry));
			}
		}

		return entries;
	}

	std::vector<WorkloadEntry> discoverWorkloads()
	{
		const std::vector<std::string> preferred = {
			"configs/workloads/volume_small_medium.json",
			"configs/workloads/mixed.json",
			"configs/workloads/range_heavy.json",
			"configs/workloads/knn_heavy.json",
		};

		std::vector<WorkloadEntry> entries;
		for (const std::string& path : preferred)
		{
			const std::string resolved = resolvePath(path);
			if (!filesystemExists(resolved))
				continue;

			WorkloadEntry entry;
			entry.label = schemaLabelFromPath(path);
			entry.path = resolved;
			entries.push_back(std::move(entry));
		}

		return entries;
	}

	void initializeState(GuiState& state)
	{
		setText(state.inputPath, "C:/Datasets/points/Alhambra_100M.las");
		setText(state.rankModelPath, resolvePath("models/schema_selector_onnx.json"));
		setText(state.csvPath, projectPath("results/gui_schema_search.csv"));
		setText(state.bestCsvPath, projectPath("results/gui_schema_search_best.csv"));
		setText(state.generatedSchemaDir, projectPath("results/generated_schemas"));
		state.schemas = discoverSchemas();
		state.workloads = discoverWorkloads();
	}

	void applyTheme()
	{
		ImGui::StyleColorsDark();
		ImGuiStyle& style = ImGui::GetStyle();
		style.WindowRounding = 0.0f;
		style.ChildRounding = 6.0f;
		style.FrameRounding = 4.0f;
		style.PopupRounding = 6.0f;
		style.ScrollbarRounding = 4.0f;
		style.GrabRounding = 4.0f;
		style.WindowBorderSize = 0.0f;
		style.ChildBorderSize = 1.0f;
		style.FrameBorderSize = 0.0f;
		style.ItemSpacing = ImVec2(10.0f, 8.0f);
		style.FramePadding = ImVec2(10.0f, 6.0f);

		ImVec4* colors = style.Colors;
		colors[ImGuiCol_WindowBg] = ImVec4(0.055f, 0.064f, 0.075f, 1.0f);
		colors[ImGuiCol_ChildBg] = ImVec4(0.080f, 0.092f, 0.108f, 1.0f);
		colors[ImGuiCol_Border] = ImVec4(0.180f, 0.205f, 0.235f, 1.0f);
		colors[ImGuiCol_FrameBg] = ImVec4(0.120f, 0.135f, 0.158f, 1.0f);
		colors[ImGuiCol_FrameBgHovered] = ImVec4(0.170f, 0.195f, 0.225f, 1.0f);
		colors[ImGuiCol_FrameBgActive] = ImVec4(0.210f, 0.245f, 0.285f, 1.0f);
		colors[ImGuiCol_Header] = ImVec4(0.160f, 0.230f, 0.245f, 1.0f);
		colors[ImGuiCol_HeaderHovered] = ImVec4(0.200f, 0.300f, 0.320f, 1.0f);
		colors[ImGuiCol_HeaderActive] = ImVec4(0.245f, 0.370f, 0.385f, 1.0f);
		colors[ImGuiCol_Button] = ImVec4(0.170f, 0.315f, 0.330f, 1.0f);
		colors[ImGuiCol_ButtonHovered] = ImVec4(0.220f, 0.405f, 0.420f, 1.0f);
		colors[ImGuiCol_ButtonActive] = ImVec4(0.270f, 0.480f, 0.500f, 1.0f);
		colors[ImGuiCol_CheckMark] = ImVec4(0.550f, 0.830f, 0.720f, 1.0f);
		colors[ImGuiCol_SliderGrab] = ImVec4(0.550f, 0.830f, 0.720f, 1.0f);
		colors[ImGuiCol_SliderGrabActive] = ImVec4(0.680f, 0.920f, 0.820f, 1.0f);
		colors[ImGuiCol_Tab] = ImVec4(0.110f, 0.130f, 0.150f, 1.0f);
		colors[ImGuiCol_TabHovered] = ImVec4(0.200f, 0.300f, 0.320f, 1.0f);
		colors[ImGuiCol_TabActive] = ImVec4(0.155f, 0.205f, 0.220f, 1.0f);
	}

	void drawSectionTitle(const char* label)
	{
		ImGui::Spacing();
		ImGui::TextUnformatted(label);
		ImGui::Separator();
	}

	void drawHelpMarker(const char* description)
	{
		ImGui::SameLine();
		ImGui::TextDisabled("?");
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::PushTextWrapPos(ImGui::GetFontSize() * 32.0f);
			ImGui::TextUnformatted(description);
			ImGui::PopTextWrapPos();
			ImGui::EndTooltip();
		}
	}

	void drawPathInput(const char* label, std::array<char, TextBufferSize>& value, const char* help = nullptr)
	{
		ImGui::PushID(label);
		ImGui::AlignTextToFramePadding();
		ImGui::TextUnformatted(label);
		if (help)
			drawHelpMarker(help);
		ImGui::SameLine(150.0f);
		ImGui::SetNextItemWidth(-1.0f);
		ImGui::InputText("##value", value.data(), value.size());
		ImGui::PopID();
	}

	void clampState(GuiState& state)
	{
		state.queryCount = std::max(1, state.queryCount);
		state.knnK = std::max(1, state.knnK);
		state.syntheticScale = std::max(1, state.syntheticScale);
		state.generatedCount = std::max(0, state.generatedCount);
		state.benchmarkTopK = std::max(0, state.benchmarkTopK);
		state.generatedMaxBlocks = std::max(1, state.generatedMaxBlocks);
		state.generatedMaxDepth = std::max(1, state.generatedMaxDepth);
		state.generatedMinLeaf = std::max(1, state.generatedMinLeaf);
		state.generatedMaxLeaf = std::max(state.generatedMinLeaf, state.generatedMaxLeaf);
		state.optimizerGenerations = std::max(0, state.optimizerGenerations);
		state.optimizerPopulation = std::max(1, state.optimizerPopulation);
		state.optimizerElites = std::max(1, state.optimizerElites);
		state.evaluator = std::clamp(state.evaluator, 0, 1);
		state.cudaDevice = std::max(0, state.cudaDevice);
		state.cudaBuilder = std::clamp(state.cudaBuilder, 0, 8);
		state.cudaQueryBatch = std::max(0, state.cudaQueryBatch);
		state.cudaMemoryBudgetMb = std::max(0, state.cudaMemoryBudgetMb);
		state.liveRankingTopN = std::clamp(state.liveRankingTopN, 1, 100);
		state.generatedConditionProbability = std::clamp(state.generatedConditionProbability, 0.0f, 1.0f);
		state.optimizerMutationRate = std::clamp(state.optimizerMutationRate, 0.0f, 1.0f);
		state.optimizerRandomFraction = std::clamp(state.optimizerRandomFraction, 0.0f, 1.0f);
		state.scoreBuildWeight = std::max(0.0f, state.scoreBuildWeight);
		state.scoreMemoryWeight = std::max(0.0f, state.scoreMemoryWeight);
		state.scoreImbalanceWeight = std::max(0.0f, state.scoreImbalanceWeight);
		if (state.selectedWorkload < 0 || state.selectedWorkload >= static_cast<int>(state.workloads.size()))
			state.selectedWorkload = 0;
	}

	std::optional<std::string> buildSearchOptions(const GuiState& state, Experiments::SchemaSearchOptions& options)
	{
		options = Experiments::SchemaSearchOptions();
		options.pauseAtEnd = false;
		options.useBinaryCache = state.useBinaryCache;
		options.rebuildBinaryCache = state.rebuildBinaryCache;
		options.includeSyntheticDatasets = state.includeSynthetic;
		options.syntheticScale = static_cast<size_t>(state.syntheticScale);
		options.queryCountOverride = static_cast<size_t>(state.queryCount);
		options.knnKOverride = static_cast<size_t>(state.knnK);
		options.querySeed = static_cast<uint32_t>(state.querySeed);
		options.querySeedOverride = true;
		options.csvPath = textValue(state.csvPath);
		options.bestCsvPath = textValue(state.bestCsvPath);

		const std::string inputPath = resolvePath(textValue(state.inputPath));
		if (!inputPath.empty())
			options.inputPaths.push_back(inputPath);

		if (options.inputPaths.empty() && !options.includeSyntheticDatasets)
			return "Select a point cloud input or enable synthetic datasets.";

		if (state.workloads.empty())
			return "No workload profiles were found under configs/workloads.";

		options.workloadPaths.push_back(state.workloads[static_cast<size_t>(state.selectedWorkload)].path);

		if (!state.generatedOnly)
		{
			for (const SchemaEntry& entry : state.schemas)
			{
				if (entry.selected)
					options.schemaPaths.push_back(entry.path);
			}
		}
		options.includeConfiguredSchemas = !state.generatedOnly;

		if (state.generateSchemas)
		{
			options.generation.count = static_cast<size_t>(state.generatedCount);
			options.generation.maxBlocks = static_cast<size_t>(state.generatedMaxBlocks);
			options.generation.maxDepth = static_cast<size_t>(state.generatedMaxDepth);
			options.generation.minLeafCapacity = static_cast<size_t>(state.generatedMinLeaf);
			options.generation.maxLeafCapacity = static_cast<size_t>(state.generatedMaxLeaf);
			options.generation.conditionalLevels = state.generatedConditional;
			options.generation.conditionalProbability = static_cast<double>(state.generatedConditionProbability);
			options.generation.seed = static_cast<uint32_t>(state.generatedSeed);
			options.generation.outputDirectory = textValue(state.generatedSchemaDir);
		}

		if (state.generatedOnly && options.generation.count == 0)
			return "Generated-only search needs a generated schema count greater than zero.";

		if (!state.generatedOnly && options.schemaPaths.empty() && options.generation.count == 0)
			return "Select at least one fixed schema or enable generated schemas.";

		options.benchmarkTopK = static_cast<size_t>(state.benchmarkTopK);
		if (state.useRankModel)
			options.rankModelPath = resolvePath(textValue(state.rankModelPath));

		options.weights.lambdaBuild = state.scoreBuildWeight;
		options.weights.lambdaMemory = state.scoreMemoryWeight;
		options.weights.lambdaImbalance = state.scoreImbalanceWeight;
		options.evolution.enabled = state.optimizeSchemas;
		options.evolution.generations = static_cast<size_t>(state.optimizerGenerations);
		options.evolution.populationSize = static_cast<size_t>(state.optimizerPopulation);
		options.evolution.eliteCount = static_cast<size_t>(state.optimizerElites);
		options.evolution.seed = static_cast<uint32_t>(state.optimizerSeed);
		options.evolution.mutationRate = static_cast<double>(state.optimizerMutationRate);
		options.evolution.randomImmigrationRate = static_cast<double>(state.optimizerRandomFraction);
		options.evaluator = state.evaluator == 1 ? "cuda" : "cpu";
		options.cuda.device = state.cudaDevice;
		if (state.cudaBuilder == 1)
			options.cuda.builder = "kdtree";
		else if (state.cudaBuilder == 2)
			options.cuda.builder = "bih";
		else if (state.cudaBuilder == 3)
			options.cuda.builder = "octree";
		else if (state.cudaBuilder == 4)
			options.cuda.builder = "karras_octree";
		else if (state.cudaBuilder == 5)
			options.cuda.builder = "quadtree";
		else if (state.cudaBuilder == 6)
			options.cuda.builder = "regular_grid";
		else if (state.cudaBuilder == 7)
			options.cuda.builder = "hgrid";
		else if (state.cudaBuilder == 8)
			options.cuda.builder = "mixed";
		else
			options.cuda.builder = "lbvh";
		options.cuda.queryBatchSize = static_cast<size_t>(state.cudaQueryBatch);
		options.cuda.memoryBudgetMb = static_cast<size_t>(state.cudaMemoryBudgetMb);
		return std::nullopt;
	}

	void startRun(RunSession& session, const GuiState& state)
	{
		if (session.running)
			return;

		if (session.worker.joinable())
			session.worker.join();

		Experiments::SchemaSearchOptions options;
		const std::optional<std::string> validation = buildSearchOptions(state, options);

		{
			std::lock_guard<std::mutex> lock(session.mutex);
			session.error.clear();
			session.log = "Starting optimization...\n";
			session.bestResults.clear();
			session.liveRanking.clear();
			session.evaluatedCandidates = 0;
			session.exitCode = 0;
			if (validation.has_value())
			{
				session.status = "Ready";
				session.error = validation.value();
				return;
			}
			session.status = "Running";
		}

		options.progressCallback = [&session](const Experiments::SchemaSearchRecord& record) {
			updateLiveRanking(session, record);
		};

		const std::string bestCsvPath = options.bestCsvPath;
		session.running = true;
		session.finished = false;
		session.worker = std::thread([&session, options, bestCsvPath]() {
			int code = 1;
			std::string errorText;
			std::vector<BestResult> bestResults;

			try
			{
				ScopedStreamCapture capture(session);
				code = Experiments::runSchemaSearch(options);
				bestResults = loadBestResults(bestCsvPath);
			}
			catch (const std::exception& exception)
			{
				errorText = exception.what();
			}
			catch (...)
			{
				errorText = "Unknown optimization error.";
			}

			{
				std::lock_guard<std::mutex> lock(session.mutex);
				session.exitCode = code;
				session.error = std::move(errorText);
				session.bestResults = std::move(bestResults);
				session.status = session.error.empty() && code == 0 ? "Complete" : "Failed";
			}
			session.running = false;
			session.finished = true;
		});
	}

	void drawDatasetPanel(GuiState& state)
	{
		drawSectionTitle("Dataset");
		drawPathInput("Point cloud", state.inputPath, "Point cloud to optimize against. With synthetic datasets disabled, the selected best schema is overfit to this one cloud and workload.");
		ImGui::Checkbox("Use binary cache", &state.useBinaryCache);
		drawHelpMarker("Loads and writes the .mdspc cache beside the source point cloud. This speeds repeated runs but does not change the measured query workload.");
		ImGui::SameLine();
		ImGui::Checkbox("Rebuild cache", &state.rebuildBinaryCache);
		drawHelpMarker("Forces a fresh read from the source point cloud and replaces the .mdspc cache. Useful after the input file changes or if the cache looks stale.");
		ImGui::Checkbox("Include synthetic datasets", &state.includeSynthetic);
		drawHelpMarker("Adds built-in synthetic point clouds to the search. Keep this off when you want the optimizer to specialize to your current point cloud only.");
		ImGui::InputInt("Synthetic scale", &state.syntheticScale);
		drawHelpMarker("Point count scale for each synthetic dataset. Larger values make synthetic validation more realistic and slower.");

		drawSectionTitle("Workload");
		if (!state.workloads.empty())
		{
			const char* preview = state.workloads[static_cast<size_t>(state.selectedWorkload)].label.c_str();
			if (ImGui::BeginCombo("Profile", preview))
			{
				for (size_t i = 0; i < state.workloads.size(); ++i)
				{
					const bool selected = static_cast<int>(i) == state.selectedWorkload;
					if (ImGui::Selectable(state.workloads[i].label.c_str(), selected))
						state.selectedWorkload = static_cast<int>(i);
					if (selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			drawHelpMarker("Defines the query mix and query volume/radius scales used during measured search. This is a major part of what the selected schema is optimized for.");
		}
		else
		{
			ImGui::TextUnformatted("No workload profiles found.");
		}
		ImGui::InputInt("Queries", &state.queryCount);
		drawHelpMarker("Number of generated measured queries for the workload. More queries reduce noise but increase optimization time.");
		ImGui::InputInt("KNN k", &state.knnK);
		drawHelpMarker("Neighbor count for KNN queries. Larger k usually increases KNN cost and can favor different structures.");
		ImGui::InputInt("Query seed", &state.querySeed);
		drawHelpMarker("Random seed for generated query centers and boxes. Keep fixed for comparable runs; change it to test robustness.");
	}

	void drawSchemaPanel(GuiState& state)
	{
		drawSectionTitle("Fixed Schemas");
		if (ImGui::Button("Select all"))
		{
			for (SchemaEntry& entry : state.schemas)
				entry.selected = true;
		}
		ImGui::SameLine();
		if (ImGui::Button("Clear"))
		{
			for (SchemaEntry& entry : state.schemas)
				entry.selected = false;
		}
		ImGui::SameLine();
		ImGui::Checkbox("Generated only", &state.generatedOnly);
		drawHelpMarker("Ignores checked fixed schemas and searches only generated candidates. Turn this off to compare generated candidates against known baselines.");

		if (ImGui::BeginTable("schemas", 3, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
		{
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 32.0f);
			ImGui::TableSetupColumn("Schema", ImGuiTableColumnFlags_WidthStretch, 0.32f);
			ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch, 0.68f);
			ImGui::TableHeadersRow();
			for (size_t i = 0; i < state.schemas.size(); ++i)
			{
				SchemaEntry& entry = state.schemas[i];
				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::PushID(static_cast<int>(i));
				ImGui::BeginDisabled(state.generatedOnly);
				ImGui::Checkbox("##selected", &entry.selected);
				ImGui::EndDisabled();
				ImGui::PopID();
				ImGui::TableSetColumnIndex(1);
				ImGui::TextUnformatted(entry.label.c_str());
				ImGui::TableSetColumnIndex(2);
				ImGui::TextUnformatted(entry.path.c_str());
			}
			ImGui::EndTable();
		}

		drawSectionTitle("Generated Search");
		ImGui::Checkbox("Generate schemas", &state.generateSchemas);
		drawHelpMarker("Samples new multi-DS JSON candidates from the configured bounds, then benchmarks them. This expands beyond the fixed schema list.");
		ImGui::SameLine();
		ImGui::Checkbox("Conditional blocks", &state.generatedConditional);
		drawHelpMarker("Allows later schema blocks to activate only for local node conditions such as point count, density, or height ratio. This is the current branch-adaptive multi-DS mechanism.");
		ImGui::InputInt("Generated count", &state.generatedCount);
		drawHelpMarker("Number of candidate JSON schemas to sample. More candidates explores more of the space and takes longer.");
		ImGui::InputInt("Benchmark top-k", &state.benchmarkTopK);
		drawHelpMarker("If a rank model is enabled, benchmark only the model's top-k candidates. 0 means benchmark every candidate and rely only on measured results.");
		ImGui::InputInt("Max blocks", &state.generatedMaxBlocks);
		drawHelpMarker("Maximum number of nested structure blocks per generated schema, for example quadtree then octree then kdtree.");
		ImGui::InputInt("Max depth", &state.generatedMaxDepth);
		drawHelpMarker("Maximum total tree depth across generated blocks. Larger depth can improve pruning but increases build cost and memory risk.");
		ImGui::InputInt("Min leaf", &state.generatedMinLeaf);
		drawHelpMarker("Smallest generated leaf capacity. Lower values produce deeper/finer trees and usually test fewer points per leaf.");
		ImGui::InputInt("Max leaf", &state.generatedMaxLeaf);
		drawHelpMarker("Largest generated leaf capacity. Higher values produce coarser leaves and can reduce memory/build cost while increasing per-leaf point tests.");
		ImGui::InputInt("Generated seed", &state.generatedSeed);
		drawHelpMarker("Random seed for schema sampling. Keep fixed for repeatability; change it to explore a different batch of candidates.");
		ImGui::SliderFloat("Condition probability", &state.generatedConditionProbability, 0.0f, 1.0f, "%.2f");
		drawHelpMarker("Probability that a generated block gets a local activation condition. Higher values make more branch-adaptive schemas.");
		drawPathInput("Generated dir", state.generatedSchemaDir, "Directory where generated schema JSON files are written. The best result table points back to one of these files when a generated candidate wins.");

		drawSectionTitle("Iterative Optimizer");
		ImGui::Checkbox("Optimize iteratively", &state.optimizeSchemas);
		drawHelpMarker("Runs evolutionary mutation search after the initial fixed/generated population. Selection is driven by measured C++ benchmark scores, not gradients.");
		ImGui::InputInt("Generations", &state.optimizerGenerations);
		drawHelpMarker("Number of mutation rounds after the initial population is measured. 0 evaluates only the initial population.");
		ImGui::InputInt("Population", &state.optimizerPopulation);
		drawHelpMarker("Number of new candidates evaluated per generation. Larger populations explore more schemas and take longer.");
		ImGui::InputInt("Elites", &state.optimizerElites);
		drawHelpMarker("Best measured candidates used as parents for the next generation. Too few can get stuck; too many makes search less focused.");
		ImGui::InputInt("Optimizer seed", &state.optimizerSeed);
		drawHelpMarker("Seed for parent choice and mutations. Keep fixed for repeatability; change it for a different search trajectory.");
		ImGui::SliderFloat("Mutation rate", &state.optimizerMutationRate, 0.0f, 1.0f, "%.2f");
		drawHelpMarker("Probability of applying extra edits to a child schema. Higher values make larger jumps in topology/depth/leaf/condition space.");
		ImGui::SliderFloat("Random fraction", &state.optimizerRandomFraction, 0.0f, 1.0f, "%.2f");
		drawHelpMarker("Fraction of each generation filled with fresh random candidates instead of mutations. This preserves exploration.");
	}

	void drawScoringPanel(GuiState& state)
	{
		drawSectionTitle("Evaluator");
		const char* evaluators[] = { "CPU", "CUDA" };
		ImGui::Combo("Backend", &state.evaluator, evaluators, IM_ARRAYSIZE(evaluators));
		drawHelpMarker("Chooses where measured schema fitness runs. CUDA builds the selected GPU structure and measures range/count/radius queries there.");
		if (state.evaluator == 1)
		{
			std::string cudaError;
			const bool cudaAvailable = PointGpu::LBVH::isAvailable(&cudaError);
			if (cudaAvailable)
			{
				ImGui::Text("CUDA devices: %d", PointGpu::LBVH::deviceCount());
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.38f, 1.0f));
				ImGui::TextWrapped("CUDA unavailable: %s", cudaError.c_str());
				ImGui::PopStyleColor();
			}

			ImGui::InputInt("CUDA device", &state.cudaDevice);
			drawHelpMarker("GPU id passed to cudaSetDevice. Use 0 unless you have several CUDA GPUs.");
			const char* builders[] = { "LBVH", "KDTree", "BIH", "Octree", "KarrasOctree", "QuadTree", "RegularGrid", "HGrid", "Mixed" };
			ImGui::Combo("Structure", &state.cudaBuilder, builders, IM_ARRAYSIZE(builders));
			drawHelpMarker("LBVH, KDTree, BIH, Octree, KarrasOctree, QuadTree, RegularGrid, HGrid, and MixedTree schemas are implemented. KarrasOctree uses Morton sorting and prefix child ranges; BIH is a binary interval hierarchy with tight child bounds; standalone HGrid builds several RegularGrid levels and chooses one per query; Mixed follows the schema's per-depth structure schedule, including RegularGrid and HGrid grid split levels.");
			ImGui::InputInt("Query batch", &state.cudaQueryBatch);
			drawHelpMarker("Number of CUDA queries uploaded/launched per batch. 0 runs the whole generated workload as one batch.");
			ImGui::InputInt("Memory budget MB", &state.cudaMemoryBudgetMb);
			drawHelpMarker("Optional guardrail that rejects CUDA builds whose point buffers, sorted arrays, nodes, and sort scratch exceed this budget.");
		}

		drawSectionTitle("Surrogate");
		ImGui::Checkbox("Use rank model", &state.useRankModel);
		drawHelpMarker("Uses the exported JSON/ONNX selector only to rank/prune candidates before benchmarking. The final best schema still comes from measured C++ timings.");
		drawPathInput("Rank model", state.rankModelPath, "Path to a selector wrapper JSON, usually models/schema_selector.json or models/schema_selector_onnx.json.");

		drawSectionTitle("Score Weights");
		ImGui::InputFloat("Build", &state.scoreBuildWeight, 0.001f, 0.01f, "%.4f");
		drawHelpMarker("Adds build time into the score. Keep at 0 if you only care about saved/reused structures and query speed.");
		ImGui::InputFloat("Memory", &state.scoreMemoryWeight, 0.001f, 0.01f, "%.4f");
		drawHelpMarker("Adds a memory penalty. Increase this if two schemas have similar query time but one is much larger.");
		ImGui::InputFloat("Imbalance", &state.scoreImbalanceWeight, 0.001f, 0.01f, "%.4f");
		drawHelpMarker("Adds a penalty for uneven leaf occupancy. Increase this if winners have pathological leaves or unstable query behavior.");

		drawSectionTitle("Outputs");
		drawPathInput("CSV", state.csvPath, "Full measured result table, with one row per dataset/workload/schema candidate.");
		drawPathInput("Best CSV", state.bestCsvPath, "Compact winner table. This is what the GUI reads back to populate Best Results.");
	}

	bool isCudaResult(const std::string& backend)
	{
		return backend == "cuda" || backend == "gpu";
	}

	void drawResultsTable(const std::vector<BestResult>& results)
	{
		if (results.empty())
		{
			ImGui::TextUnformatted("No completed results yet.");
			return;
		}

		if (ImGui::BeginTable("best-results", 11, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp))
		{
			ImGui::TableSetupColumn("Dataset");
			ImGui::TableSetupColumn("Workload");
			ImGui::TableSetupColumn("Best schema");
			ImGui::TableSetupColumn("Backend", ImGuiTableColumnFlags_WidthFixed, 76.0f);
			ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, 78.0f);
			ImGui::TableSetupColumn("Avg ms", ImGuiTableColumnFlags_WidthFixed, 78.0f);
			ImGui::TableSetupColumn("Build ms", ImGuiTableColumnFlags_WidthFixed, 84.0f);
			ImGui::TableSetupColumn("GPU build", ImGuiTableColumnFlags_WidthFixed, 84.0f);
			ImGui::TableSetupColumn("GPU query", ImGuiTableColumnFlags_WidthFixed, 84.0f);
			ImGui::TableSetupColumn("Memory MB", ImGuiTableColumnFlags_WidthFixed, 92.0f);
			ImGui::TableSetupColumn("Candidates", ImGuiTableColumnFlags_WidthFixed, 92.0f);
			ImGui::TableHeadersRow();

			for (const BestResult& result : results)
			{
				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::TextUnformatted(result.dataset.c_str());
				ImGui::TableSetColumnIndex(1);
				ImGui::TextUnformatted(result.workload.c_str());
				ImGui::TableSetColumnIndex(2);
				ImGui::TextUnformatted(result.schemaName.c_str());
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("%s", result.schemaPath.c_str());
				ImGui::TableSetColumnIndex(3);
				ImGui::TextUnformatted(result.backend.empty() ? "cpu" : result.backend.c_str());
				ImGui::TableSetColumnIndex(4);
				ImGui::Text("%.3f", result.score);
				ImGui::TableSetColumnIndex(5);
				ImGui::Text("%.3f", result.averageLatencyMs);
				ImGui::TableSetColumnIndex(6);
				ImGui::Text("%.1f", result.buildTimeMs);
				ImGui::TableSetColumnIndex(7);
				if (isCudaResult(result.backend))
					ImGui::Text("%.1f", result.gpuBuildMs);
				else
					ImGui::TextUnformatted("-");
				ImGui::TableSetColumnIndex(8);
				if (isCudaResult(result.backend))
					ImGui::Text("%.1f", result.gpuQueryMs);
				else
					ImGui::TextUnformatted("-");
				ImGui::TableSetColumnIndex(9);
				ImGui::Text("%.1f", static_cast<double>(result.memoryBytes) / (1024.0 * 1024.0));
				ImGui::TableSetColumnIndex(10);
				ImGui::Text("%zu", result.candidates);
			}

			ImGui::EndTable();
		}
	}

	void drawLiveRankingTable(const std::vector<LiveRankingEntry>& ranking, size_t evaluatedCandidates, int topN)
	{
		ImGui::Text("Measured candidates: %zu", evaluatedCandidates);
		if (ranking.empty())
		{
			ImGui::TextUnformatted("Live ranking will appear after the first candidate finishes.");
			return;
		}

		if (ImGui::BeginTable("live-ranking", 11, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp))
		{
			ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 34.0f);
			ImGui::TableSetupColumn("Schema");
			ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, 78.0f);
			ImGui::TableSetupColumn("Avg", ImGuiTableColumnFlags_WidthFixed, 66.0f);
			ImGui::TableSetupColumn("P95", ImGuiTableColumnFlags_WidthFixed, 66.0f);
			ImGui::TableSetupColumn("Tested", ImGuiTableColumnFlags_WidthFixed, 78.0f);
			ImGui::TableSetupColumn("Visited", ImGuiTableColumnFlags_WidthFixed, 76.0f);
			ImGui::TableSetupColumn("Build", ImGuiTableColumnFlags_WidthFixed, 72.0f);
			ImGui::TableSetupColumn("GPU build", ImGuiTableColumnFlags_WidthFixed, 84.0f);
			ImGui::TableSetupColumn("GPU query", ImGuiTableColumnFlags_WidthFixed, 84.0f);
			ImGui::TableSetupColumn("Backend", ImGuiTableColumnFlags_WidthFixed, 88.0f);
			ImGui::TableHeadersRow();

			const size_t rowCount = std::min<size_t>(ranking.size(), static_cast<size_t>(std::max(1, topN)));
			for (size_t i = 0; i < rowCount; ++i)
			{
				const LiveRankingEntry& entry = ranking[i];
				ImGui::TableNextRow();
				const bool highlight = i < 3;
				if (highlight)
				{
					const ImU32 color = i == 0
						? IM_COL32(68, 122, 105, 72)
						: i == 1
							? IM_COL32(84, 96, 122, 54)
							: IM_COL32(122, 104, 72, 44);
					ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, color);
				}

				ImGui::TableSetColumnIndex(0);
				ImGui::Text("%zu", i + 1);
				ImGui::TableSetColumnIndex(1);
				ImGui::TextUnformatted(entry.schemaName.c_str());
				if (ImGui::IsItemHovered())
				{
					ImGui::SetTooltip(
						"%s\n%s / %s\norder #%zu",
						entry.schemaPath.c_str(),
						entry.dataset.c_str(),
						entry.workload.c_str(),
						entry.order);
				}
				ImGui::TableSetColumnIndex(2);
				ImGui::Text("%.4f", entry.score);
				ImGui::TableSetColumnIndex(3);
				ImGui::Text("%.3f", entry.averageLatencyMs);
				ImGui::TableSetColumnIndex(4);
				ImGui::Text("%.3f", entry.p95LatencyMs);
				ImGui::TableSetColumnIndex(5);
				ImGui::Text("%.0f", entry.averageTestedPoints);
				ImGui::TableSetColumnIndex(6);
				ImGui::Text("%.1f", entry.averageVisitedNodes);
				ImGui::TableSetColumnIndex(7);
				ImGui::Text("%.1f", entry.buildTimeMs);
				ImGui::TableSetColumnIndex(8);
				if (isCudaResult(entry.backend))
					ImGui::Text("%.1f", entry.gpuBuildMs);
				else
					ImGui::TextUnformatted("-");
				ImGui::TableSetColumnIndex(9);
				if (isCudaResult(entry.backend))
					ImGui::Text("%.1f", entry.gpuQueryMs);
				else
					ImGui::TextUnformatted("-");
				ImGui::TableSetColumnIndex(10);
				if (!entry.cudaBuilder.empty())
					ImGui::Text("%s/%s", entry.backend.c_str(), entry.cudaBuilder.c_str());
				else
					ImGui::TextUnformatted(entry.backend.empty() ? "cpu" : entry.backend.c_str());
			}

			ImGui::EndTable();
		}
	}

	void drawRunPanel(GuiState& state, RunSession& session)
	{
		std::string status;
		std::string error;
		std::string log;
		std::vector<BestResult> results;
		std::vector<LiveRankingEntry> liveRanking;
		size_t evaluatedCandidates = 0;
		{
			std::lock_guard<std::mutex> lock(session.mutex);
			status = session.status;
			error = session.error;
			log = session.log;
			results = session.bestResults;
			liveRanking = session.liveRanking;
			evaluatedCandidates = session.evaluatedCandidates;
		}

		drawSectionTitle("Run");
		const bool canStart = !session.running;
		ImGui::BeginDisabled(!canStart);
		if (ImGui::Button("Start optimization", ImVec2(180.0f, 34.0f)))
			startRun(session, state);
		ImGui::EndDisabled();
		ImGui::SameLine();
		ImGui::Text("Status: %s", status.c_str());
		if (session.running)
			ImGui::ProgressBar(-1.0f * static_cast<float>(ImGui::GetTime()), ImVec2(-1.0f, 0.0f));

		if (!error.empty())
		{
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.38f, 1.0f));
			ImGui::TextWrapped("%s", error.c_str());
			ImGui::PopStyleColor();
		}

		drawSectionTitle("Live Ranking");
		ImGui::SetNextItemWidth(96.0f);
		ImGui::InputInt("Top N", &state.liveRankingTopN);
		drawHelpMarker("Number of live leaderboard rows to display. The run still measures every candidate and writes every row to CSV.");
		drawLiveRankingTable(liveRanking, evaluatedCandidates, state.liveRankingTopN);

		drawSectionTitle("Best Results");
		drawResultsTable(results);

		drawSectionTitle("Log");
		ImGui::BeginChild("log", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_HorizontalScrollbar);
		if (log.empty())
			ImGui::TextUnformatted("The run log will appear here.");
		else
			ImGui::TextUnformatted(log.c_str());
		ImGui::EndChild();
	}

	void drawInterface(GuiState& state, RunSession& session)
	{
		clampState(state);

		const ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(viewport->WorkPos);
		ImGui::SetNextWindowSize(viewport->WorkSize);

		const ImGuiWindowFlags flags =
			ImGuiWindowFlags_NoDecoration |
			ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoBringToFrontOnFocus;

		ImGui::Begin("MultiDataStructure Optimizer", nullptr, flags);
		ImGui::TextUnformatted("MultiDataStructure Optimizer");
		ImGui::SameLine();
		ImGui::TextDisabled("measured schema search");
		ImGui::Separator();

		const float leftWidth = std::max(360.0f, ImGui::GetContentRegionAvail().x * 0.36f);
		ImGui::BeginChild("left-panel", ImVec2(leftWidth, 0.0f), true);
		if (ImGui::BeginTabBar("configuration-tabs"))
		{
			if (ImGui::BeginTabItem("Dataset"))
			{
				drawDatasetPanel(state);
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Schemas"))
			{
				drawSchemaPanel(state);
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Scoring"))
			{
				drawScoringPanel(state);
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}
		ImGui::EndChild();

		ImGui::SameLine();
		ImGui::BeginChild("right-panel", ImVec2(0.0f, 0.0f), true);
		drawRunPanel(state, session);
		ImGui::EndChild();
		ImGui::End();
	}
}

int OptimizerGui::run()
{
	if (!glfwInit())
		throw std::runtime_error("Unable to initialize GLFW.");

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(1440, 900, "MultiDataStructure Optimizer", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		throw std::runtime_error("Unable to create optimizer window.");
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	glewExperimental = GL_TRUE;
	const GLenum glewStatus = glewInit();
	if (glewStatus != GLEW_OK)
	{
		glfwDestroyWindow(window);
		glfwTerminate();
		throw std::runtime_error("Unable to initialize GLEW.");
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.IniFilename = nullptr;
	applyTheme();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	GuiState state;
	initializeState(state);
	RunSession session;

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		if (glfwWindowShouldClose(window) && session.running)
		{
			glfwSetWindowShouldClose(window, GLFW_FALSE);
			std::lock_guard<std::mutex> lock(session.mutex);
			session.status = "Running; close after completion";
		}

		if (session.finished.exchange(false) && session.worker.joinable())
			session.worker.join();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		drawInterface(state, session);
		ImGui::Render();

		int displayWidth = 0;
		int displayHeight = 0;
		glfwGetFramebufferSize(window, &displayWidth, &displayHeight);
		glViewport(0, 0, displayWidth, displayHeight);
		glClearColor(0.055f, 0.064f, 0.075f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(window);
	}

	if (session.worker.joinable())
		session.worker.join();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}

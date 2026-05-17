#include "OptimizerGui.h"

#include "../core/Config.h"
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

	struct SchemaFileViewer
	{
		bool open = false;
		bool wrap = false;
		std::string title = "Schema JSON";
		std::string path;
		std::string content;
		std::string error;
	};

	struct PreviewBox
	{
		glm::vec3 min = glm::vec3(0.0f);
		glm::vec3 max = glm::vec3(0.0f);
		std::string typeName;
		size_t depth = 0;
	};

	struct PreviewPhase
	{
		SchemaLevelConfig level;
		std::string label;
		size_t startDepth = 0;
		size_t endDepth = 0;
	};

	struct StructurePreview
	{
		bool open = false;
		bool needsRebuild = false;
		bool truncated = false;
		bool showFullSchedule = false;
		std::string title = "Structure Preview";
		std::string path;
		std::string schemaName;
		std::string error;
		std::vector<PreviewBox> boxes;
		std::vector<PreviewPhase> phases;
		int phaseIndex = 0;
		int maxDepth = 2;
		int maxBoxes = 2048;
		float yaw = 0.68f;
		float pitch = 0.42f;
		float zoom = 3.2f;
	};

	struct GuiState
	{
		std::array<char, TextBufferSize> inputPath{};
		std::array<char, TextBufferSize> rankModelPath{};
		std::array<char, TextBufferSize> csvPath{};
		std::array<char, TextBufferSize> bestCsvPath{};
		std::array<char, TextBufferSize> paretoCsvPath{};
		std::array<char, TextBufferSize> generatedSchemaDir{};
		std::array<char, TextBufferSize> autoConditionSchemaDir{};
		std::array<char, TextBufferSize> selectorOutputPath{};

		std::vector<SchemaEntry> schemas;
		std::vector<WorkloadEntry> workloads;
		int selectedWorkload = 0;

		bool autoConditions = true;
		bool includeSynthetic = false;
		bool useBinaryCache = true;
		bool rebuildBinaryCache = false;
		bool generateSchemas = true;
		bool generatedOnly = true;
		bool generatedConditional = true;
		bool useRankModel = false;
		// Default-on: when the user disables auto-conditions, the GA path is the publication
		// pipeline and Phase A+B1 (rungs + threshold refinement) ride on top of it.
		bool optimizeSchemas = true;
		int evaluator = 0;
		int cudaDevice = 0;
		int cudaBuilder = 0;
		int cudaQueryBatch = 0;
		int cudaMemoryBudgetMb = 0;
		int liveRankingTopN = 10;
		bool advancedEvaluatorOpen = false;
		bool advancedScoringOpen = false;

		int queryCount = 64;
		int knnK = 16;
		int querySeed = 1337;
		int syntheticScale = 512;
		int generatedCount = 256;
		int benchmarkTopK = 32;
		int generatedMaxBlocks = 3;
		int generatedMaxDepth = 12;
		int generatedMinLeaf = 32;
		int generatedMaxLeaf = 32768;
		int generatedSeed = 1337;
		int optimizerGenerations = 3;
		int optimizerPopulation = 64;
		int optimizerElites = 6;
		int optimizerSeed = 1337;
		int conditionProxyCandidates = 256;
		int conditionProxyPoints = 262144;
		int conditionProxyQueries = 8;
		int conditionFinalTopK = 16;
		int conditionConfirmTopK = 4;
		float generatedConditionProbability = 0.5f;
		float optimizerMutationRate = 0.65f;
		float optimizerRandomFraction = 0.20f;
		float scoreBuildWeight = 0.0f;
		float scoreMemoryWeight = 0.0f;
		float scoreImbalanceWeight = 0.0f;

		// Score cache + parallel dispatch defaults exposed via the optimizer panel.
		bool scoreCacheEnabled = true;
		std::array<char, 512> scoreCachePath{};
		bool rebuildScoreCache = false;
		int parallelDispatch = 1;
		bool includeBaselineSchemas = true;

		// Multi-fidelity rung schedule (Phase A successive halving). When enabled, each
		// optimizer batch flows through a cheap visit-proxy stage, a mid-fidelity latency stage,
		// and a confirmation stage at the full workload. Default-on so the smarter-search
		// pipeline is the GUI's recommended starting point.
		bool useRungSchedule = true;
		int rungProxyQueries = 4;
		int rungProxyAdvance = 32;
		float rungProxyAlpha = 0.1f;
		int rungFullQueries = 16;
		int rungFullAdvance = 8;
		int rungConfirmQueries = 64;
		std::array<char, TextBufferSize> rungSurrogatePath{};
		int rungSurrogatePool = 0;
		int rungSurrogateTop = 0;

		// Threshold refinement (Phase B1). After the GA finishes, the top-K archive entries with
		// conditional levels get their numeric thresholds tuned by a (1+lambda)-ES using the
		// visit-proxy score; survivors that improved are re-measured at full fidelity. Default-on
		// so the GA path ships its best Phase B1 settings out of the box.
		bool refineThresholds = true;
		int refineThresholdsTopK = 4;
		int refineThresholdsEvals = 60;
		float refineThresholdsSigma = 0.3f;
		int refineThresholdsSeed = 1337;

		SchemaFileViewer fileViewer;
		StructurePreview structurePreview;
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

	std::string loadTextFile(const std::string& configuredPath, std::string& error)
	{
		error.clear();
		const std::string path = resolvePath(configuredPath);
		std::ifstream input(path, std::ios::binary);
		if (!input.is_open())
		{
			error = "Unable to open file: " + configuredPath;
			return {};
		}

		std::ostringstream buffer;
		buffer << input.rdbuf();
		std::string content = buffer.str();
		constexpr size_t MaxPreviewBytes = 4 * 1024 * 1024;
		if (content.size() > MaxPreviewBytes)
		{
			content.resize(MaxPreviewBytes);
			content += "\n\n... truncated ...";
		}
		return content;
	}

	std::string normalizedStructureName(std::string value)
	{
		value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
			return std::isspace(c) || c == '_' || c == '-';
		}), value.end());
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return value;
	}

	bool isStructureName(const std::string& normalized, std::initializer_list<const char*> names)
	{
		for (const char* name : names)
		{
			if (normalized == name)
				return true;
		}
		return false;
	}

	ImVec4 colorForStructureName(const std::string& typeName, float alpha = 1.0f)
	{
		const std::string normalized = normalizedStructureName(typeName);
		if (isStructureName(normalized, { "quadtree", "quadtreenode", "qt" }))
			return ImVec4(0.38f, 0.86f, 0.58f, alpha);
		if (isStructureName(normalized, { "kdtree", "kdtreenode", "kd" }))
			return ImVec4(0.95f, 0.78f, 0.32f, alpha);
		if (isStructureName(normalized, { "bih", "binaryintervalhierarchy", "intervalhierarchy" }))
			return ImVec4(1.00f, 0.48f, 0.55f, alpha);
		if (isStructureName(normalized, { "karrasoctree", "mortonoctree", "octreekarras", "octreemorton" }))
			return ImVec4(0.42f, 0.72f, 1.00f, alpha);
		if (isStructureName(normalized, { "octree", "octreenode", "ot" }))
			return ImVec4(0.35f, 0.88f, 0.92f, alpha);
		if (isStructureName(normalized, { "regulargrid", "uniformgrid", "grid", "grid3d" }))
			return ImVec4(1.00f, 0.58f, 0.28f, alpha);
		if (isStructureName(normalized, { "hgrid", "hierarchicalgrid", "hierarchicalgrid3d" }))
			return ImVec4(0.78f, 0.64f, 1.00f, alpha);
		if (isStructureName(normalized, { "bvh", "bvhnode" }))
			return ImVec4(0.55f, 0.70f, 1.00f, alpha);
		if (isStructureName(normalized, { "lbvh", "linearbvh" }))
			return ImVec4(0.66f, 0.82f, 1.00f, alpha);
		return ImVec4(0.78f, 0.82f, 0.88f, alpha);
	}

	ImU32 packedColorForStructureName(const std::string& typeName, float alpha = 1.0f)
	{
		return ImGui::ColorConvertFloat4ToU32(colorForStructureName(typeName, alpha));
	}

	int longestAxis(const glm::vec3& extent)
	{
		if (extent.x >= extent.y && extent.x >= extent.z)
			return 0;
		if (extent.y >= extent.z)
			return 1;
		return 2;
	}

	std::string previewPhaseLabel(const SchemaLevelConfig& level, size_t startDepth, size_t endDepth)
	{
		std::ostringstream output;
		output << level.typeName << " d" << startDepth << "-d" << (endDepth == 0 ? 0 : endDepth - 1);
		if (!level.condition.empty())
			output << " conditional";
		return output.str();
	}

	std::vector<PreviewPhase> previewPhasesForSchema(const SchemaConfig& schema)
	{
		std::vector<PreviewPhase> phases;
		const size_t schemaMaxDepth = schema.buildPolicy.maxDepth > 0
			? std::min(schema.buildPolicy.maxDepth, schema.totalLevels())
			: schema.totalLevels();
		size_t startDepth = 0;
		for (const SchemaLevelConfig& level : schema.levels)
		{
			const size_t endDepth = startDepth + level.numLevels;
			if (startDepth >= schemaMaxDepth)
				break;

			PreviewPhase phase;
			phase.level = level;
			phase.startDepth = startDepth;
			phase.endDepth = std::min(endDepth, schemaMaxDepth);
			phase.label = previewPhaseLabel(level, phase.startDepth, phase.endDepth);
			phases.push_back(std::move(phase));
			startDepth = endDepth;
		}
		return phases;
	}

	void pushGridPreviewChildren(
		const PreviewBox& parent,
		size_t depth,
		const std::string& typeName,
		int cellsX,
		int cellsY,
		int cellsZ,
		std::vector<PreviewBox>& children)
	{
		const glm::vec3 extent = parent.max - parent.min;
		for (int z = 0; z < cellsZ; ++z)
		{
			for (int y = 0; y < cellsY; ++y)
			{
				for (int x = 0; x < cellsX; ++x)
				{
					PreviewBox child;
					child.typeName = typeName;
					child.depth = depth;
					child.min = glm::vec3(
						parent.min.x + extent.x * (static_cast<float>(x) / static_cast<float>(cellsX)),
						parent.min.y + extent.y * (static_cast<float>(y) / static_cast<float>(cellsY)),
						parent.min.z + extent.z * (static_cast<float>(z) / static_cast<float>(cellsZ)));
					child.max = glm::vec3(
						parent.min.x + extent.x * (static_cast<float>(x + 1) / static_cast<float>(cellsX)),
						parent.min.y + extent.y * (static_cast<float>(y + 1) / static_cast<float>(cellsY)),
						parent.min.z + extent.z * (static_cast<float>(z + 1) / static_cast<float>(cellsZ)));
					children.push_back(child);
				}
			}
		}
	}

	void pushGridPreviewChildren(
		const PreviewBox& parent,
		size_t depth,
		const std::string& typeName,
		const glm::uvec3& cells,
		std::vector<PreviewBox>& children)
	{
		pushGridPreviewChildren(
			parent,
			depth,
			typeName,
			static_cast<int>(cells.x),
			static_cast<int>(cells.y),
			static_cast<int>(cells.z),
			children);
	}

	std::vector<PreviewBox> previewChildrenForLevel(const PreviewBox& parent, const SchemaLevelConfig& level, size_t depth)
	{
		std::vector<PreviewBox> children;
		const std::string normalized = normalizedStructureName(level.typeName);
		if (isStructureName(normalized, { "quadtree", "quadtreenode", "qt" }))
		{
			glm::uvec3 cells(2, 2, 2);
			cells[longestAxis(parent.max - parent.min)] = 1;
			pushGridPreviewChildren(parent, depth, level.typeName, cells, children);
		}
		else if (isStructureName(normalized, { "octree", "octreenode", "ot", "karrasoctree", "mortonoctree", "octreekarras", "octreemorton" }))
		{
			pushGridPreviewChildren(parent, depth, level.typeName, 2, 2, 2, children);
		}
		else if (isStructureName(normalized, { "regulargrid", "uniformgrid", "grid", "grid3d", "hgrid", "hierarchicalgrid", "hierarchicalgrid3d" }))
		{
			pushGridPreviewChildren(parent, depth, level.typeName, 3, 3, 3, children);
		}
		else
		{
			const glm::vec3 extent = parent.max - parent.min;
			const int axis = longestAxis(extent);
			const float splitValue = parent.min[axis] + extent[axis] * 0.5f;

			PreviewBox left = parent;
			PreviewBox right = parent;
			left.typeName = level.typeName;
			right.typeName = level.typeName;
			left.depth = depth;
			right.depth = depth;
			left.max[axis] = splitValue;
			right.min[axis] = splitValue;
			children.push_back(left);
			children.push_back(right);
		}
		return children;
	}

	void rebuildStructurePreview(StructurePreview& preview)
	{
		preview.error.clear();
		preview.boxes.clear();
		preview.phases.clear();
		preview.truncated = false;
		preview.needsRebuild = false;
		preview.maxDepth = std::clamp(preview.maxDepth, 1, 9);
		preview.maxBoxes = std::clamp(preview.maxBoxes, 64, 20000);

		try
		{
			const SchemaConfig schema = Config::loadSchemaConfig(preview.path);
			preview.schemaName = schema.name.empty() ? schemaLabelFromPath(preview.path) : schema.name;
			preview.phases = previewPhasesForSchema(schema);
			if (preview.phases.empty())
				throw std::runtime_error("Schema has no previewable levels");
			preview.phaseIndex = std::clamp(preview.phaseIndex, 0, static_cast<int>(preview.phases.size() - 1));
			const size_t schemaMaxDepth = schema.buildPolicy.maxDepth > 0
				? std::min(schema.buildPolicy.maxDepth, schema.totalLevels())
				: schema.totalLevels();

			PreviewBox root;
			root.min = glm::vec3(-1.0f, -1.0f, -1.0f);
			root.max = glm::vec3(1.0f, 1.0f, 1.0f);
			root.typeName = preview.showFullSchedule ? "Bounds" : preview.phases[preview.phaseIndex].level.typeName;
			root.depth = 0;
			preview.boxes.push_back(root);

			std::vector<PreviewBox> frontier = { root };
			if (preview.showFullSchedule)
			{
				const size_t targetDepth = std::min<size_t>(schemaMaxDepth, static_cast<size_t>(preview.maxDepth));
				for (size_t depth = 0; depth < targetDepth && !frontier.empty(); ++depth)
				{
					const SchemaLevelConfig& level = schema.levelForDepth(depth);
					std::vector<PreviewBox> next;
					for (const PreviewBox& node : frontier)
					{
						std::vector<PreviewBox> children = previewChildrenForLevel(node, level, depth + 1);
						for (PreviewBox& child : children)
						{
							if (preview.boxes.size() >= static_cast<size_t>(preview.maxBoxes))
							{
								preview.truncated = true;
								break;
							}
							next.push_back(child);
							preview.boxes.push_back(std::move(child));
						}
						if (preview.truncated)
							break;
					}
					frontier = std::move(next);
					if (preview.truncated)
						break;
				}
			}
			else
			{
				const PreviewPhase& phase = preview.phases[preview.phaseIndex];
				const size_t phaseLevels = std::max<size_t>(1, phase.endDepth - phase.startDepth);
				const size_t targetDepth = std::min<size_t>(phaseLevels, static_cast<size_t>(preview.maxDepth));
				for (size_t depth = 0; depth < targetDepth && !frontier.empty(); ++depth)
				{
					std::vector<PreviewBox> next;
					for (const PreviewBox& node : frontier)
					{
						std::vector<PreviewBox> children = previewChildrenForLevel(node, phase.level, depth + 1);
						for (PreviewBox& child : children)
						{
							if (preview.boxes.size() >= static_cast<size_t>(preview.maxBoxes))
							{
								preview.truncated = true;
								break;
							}
							next.push_back(child);
							preview.boxes.push_back(std::move(child));
						}
						if (preview.truncated)
							break;
					}
					frontier = std::move(next);
					if (preview.truncated)
						break;
				}
			}
		}
		catch (const std::exception& exception)
		{
			preview.error = exception.what();
		}
	}

	void openSchemaFile(GuiState& state, const std::string& path)
	{
		state.fileViewer.open = true;
		state.fileViewer.path = resolvePath(path);
		state.fileViewer.title = schemaLabelFromPath(path);
		state.fileViewer.content = loadTextFile(path, state.fileViewer.error);
	}

	void openStructurePreview(GuiState& state, const std::string& path)
	{
		const std::string resolved = resolvePath(path);
		if (state.structurePreview.path != resolved)
		{
			state.structurePreview.phaseIndex = 0;
			state.structurePreview.maxDepth = 2;
			state.structurePreview.showFullSchedule = false;
		}
		state.structurePreview.open = true;
		state.structurePreview.path = resolved;
		state.structurePreview.title = schemaLabelFromPath(path);
		state.structurePreview.needsRebuild = true;
		rebuildStructurePreview(state.structurePreview);
	}

	void drawSchemaActions(GuiState& state, const std::string& path)
	{
		const bool hasPath = !path.empty();
		ImGui::BeginDisabled(!hasPath);
		if (ImGui::SmallButton("JSON"))
			openSchemaFile(state, path);
		ImGui::SameLine();
		if (ImGui::SmallButton("Boxes"))
			openStructurePreview(state, path);
		ImGui::EndDisabled();
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

	int workloadIndexForToken(const GuiState& state, const std::string& token)
	{
		for (size_t i = 0; i < state.workloads.size(); ++i)
		{
			const std::string haystack = state.workloads[i].path + "|" + state.workloads[i].label;
			if (haystack.find(token) != std::string::npos)
				return static_cast<int>(i);
		}
		return -1;
	}

	void applyPublicationDefaults(GuiState& state)
	{
		state.autoConditions = true;
		state.includeSynthetic = false;
		state.useBinaryCache = true;
		state.rebuildBinaryCache = false;
		state.generateSchemas = true;
		state.generatedOnly = true;
		state.generatedConditional = true;
		state.useRankModel = false;
		// Reset defaults match the struct defaults: GA + rung schedule + threshold refinement on
		// so the optimizer panel reflects the recommended pipeline whenever the user resets state.
		state.optimizeSchemas = true;
		state.useRungSchedule = true;
		state.refineThresholds = true;
		state.evaluator = 1;
		state.cudaDevice = 0;
		state.cudaBuilder = 8;
		state.cudaQueryBatch = 0;
		state.cudaMemoryBudgetMb = 0;
		state.queryCount = 64;
		state.knnK = 16;
		state.querySeed = 1337;
		state.generatedCount = 256;
		state.benchmarkTopK = 32;
		state.generatedMaxBlocks = 3;
		state.generatedMaxDepth = 12;
		state.generatedMinLeaf = 32;
		state.generatedMaxLeaf = 32768;
		state.generatedSeed = 1337;
		state.generatedConditionProbability = 0.75f;
		state.conditionProxyCandidates = 256;
		state.conditionProxyPoints = 262144;
		state.conditionProxyQueries = 8;
		state.conditionFinalTopK = 16;
		state.conditionConfirmTopK = 4;
		state.scoreBuildWeight = 0.0f;
		state.scoreMemoryWeight = 0.0f;
		state.scoreImbalanceWeight = 0.0f;
		const int volumeIndex = workloadIndexForToken(state, "volume_small_medium");
		if (volumeIndex >= 0)
			state.selectedWorkload = volumeIndex;
	}

	void initializeState(GuiState& state)
	{
		setText(state.inputPath, "C:/Datasets/points/Alhambra_100M.las");
		const std::string rankModel = resolvePath("models/schema_selector.json");
		setText(state.rankModelPath, rankModel);
		setText(state.csvPath, projectPath("results/gui_schema_search.csv"));
		setText(state.bestCsvPath, projectPath("results/gui_schema_search_best.csv"));
		setText(state.paretoCsvPath, projectPath("results/gui_schema_search_pareto.csv"));
		setText(state.generatedSchemaDir, projectPath("results/generated_schemas"));
		setText(state.autoConditionSchemaDir, projectPath("results/auto_conditions"));
		setText(state.selectorOutputPath, projectPath("models/local_schema_selector.json"));
		setText(state.scoreCachePath, projectPath("results/gui_score_cache.jsonl"));
		state.schemas = discoverSchemas();
		state.workloads = discoverWorkloads();
		applyPublicationDefaults(state);
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
		state.conditionProxyCandidates = std::max(1, state.conditionProxyCandidates);
		state.conditionProxyPoints = std::max(1, state.conditionProxyPoints);
		state.conditionProxyQueries = std::max(1, state.conditionProxyQueries);
		state.conditionFinalTopK = std::max(1, state.conditionFinalTopK);
		state.conditionConfirmTopK = std::max(1, std::min(state.conditionConfirmTopK, state.conditionFinalTopK));
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
		options.paretoCsvPath = textValue(state.paretoCsvPath);

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

		options.autoConditions.enabled = state.autoConditions;
		options.autoConditions.proxyCandidateCount = static_cast<size_t>(state.conditionProxyCandidates);
		options.autoConditions.proxyPointCap = static_cast<size_t>(state.conditionProxyPoints);
		options.autoConditions.proxyQueryCount = static_cast<size_t>(state.conditionProxyQueries);
		options.autoConditions.finalTopK = static_cast<size_t>(state.conditionFinalTopK);
		options.autoConditions.confirmationTopK = static_cast<size_t>(state.conditionConfirmTopK);
		options.autoConditions.outputDirectory = textValue(state.autoConditionSchemaDir);
		options.autoConditions.selectorOutputPath = textValue(state.selectorOutputPath);
		if (options.autoConditions.enabled)
		{
			options.generation.conditionalLevels = true;
			options.generation.conditionalProbability = std::max(options.generation.conditionalProbability, 0.75);
			if (!state.generateSchemas && options.schemaPaths.empty())
				return "Auto-condition tuning needs generated schemas or at least one selected fixed schema.";
		}

		options.benchmarkTopK = static_cast<size_t>(state.benchmarkTopK);
		if (state.useRankModel)
			options.rankModelPath = resolvePath(textValue(state.rankModelPath));

		options.weights.lambdaBuild = state.scoreBuildWeight;
		options.weights.lambdaMemory = state.scoreMemoryWeight;
		options.weights.lambdaImbalance = state.scoreImbalanceWeight;
		if (state.scoreCacheEnabled)
		{
			options.scoreCachePath = textValue(state.scoreCachePath);
			options.rebuildScoreCache = state.rebuildScoreCache;
		}
		else
		{
			options.scoreCachePath.clear();
		}
		options.parallelDispatch = std::max(1, state.parallelDispatch);
		options.includeBaselineSchemas = state.includeBaselineSchemas;
		options.evolution.enabled = state.optimizeSchemas;
		options.evolution.generations = static_cast<size_t>(state.optimizerGenerations);
		options.evolution.populationSize = static_cast<size_t>(state.optimizerPopulation);
		options.evolution.eliteCount = static_cast<size_t>(state.optimizerElites);
		options.evolution.seed = static_cast<uint32_t>(state.optimizerSeed);
		options.evolution.mutationRate = static_cast<double>(state.optimizerMutationRate);
		options.evolution.randomImmigrationRate = static_cast<double>(state.optimizerRandomFraction);

		if (state.optimizeSchemas && state.useRungSchedule)
		{
			Experiments::RungSpec proxy;
			proxy.name = "proxy";
			proxy.queryCountOverride = static_cast<size_t>(std::max(1, state.rungProxyQueries));
			proxy.useVisitProxy = true;
			proxy.visitProxyAlpha = static_cast<double>(state.rungProxyAlpha);
			proxy.advanceTopK = static_cast<size_t>(std::max(1, state.rungProxyAdvance));

			Experiments::RungSpec full;
			full.name = "full";
			full.queryCountOverride = static_cast<size_t>(std::max(1, state.rungFullQueries));
			full.useVisitProxy = false;
			full.advanceTopK = static_cast<size_t>(std::max(1, state.rungFullAdvance));

			Experiments::RungSpec confirm;
			confirm.name = "confirm";
			confirm.queryCountOverride = static_cast<size_t>(std::max(1, state.rungConfirmQueries));
			confirm.useVisitProxy = false;
			confirm.advanceTopK = 0;

			options.evolution.rungSchedule.rungs = { proxy, full, confirm };
			options.evolution.rungSchedule.surrogateModelPath = resolvePath(textValue(state.rungSurrogatePath));
			options.evolution.rungSchedule.surrogateCandidatePool = static_cast<size_t>(std::max(0, state.rungSurrogatePool));
			options.evolution.rungSchedule.surrogateProposalsPerStep = static_cast<size_t>(std::max(0, state.rungSurrogateTop));
		}
		else
		{
			options.evolution.rungSchedule = Experiments::RungSchedule{};
		}

		if (state.optimizeSchemas && state.refineThresholds)
		{
			options.evolution.refineThresholds = true;
			options.evolution.refineThresholdsTopK = static_cast<size_t>(std::max(1, state.refineThresholdsTopK));
			options.evolution.refineThresholdsEvaluations = static_cast<size_t>(std::max(1, state.refineThresholdsEvals));
			options.evolution.refineThresholdsSigma0 = static_cast<double>(state.refineThresholdsSigma);
			options.evolution.refineThresholdsSeed = static_cast<uint32_t>(state.refineThresholdsSeed);
		}
		else
		{
			options.evolution.refineThresholds = false;
		}
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

	void drawPublicationPanel(GuiState& state)
	{
		drawSectionTitle("Publication Path");
		if (ImGui::Button("Apply publication defaults"))
			applyPublicationDefaults(state);
		drawHelpMarker("CUDA/Mixed, one real cloud, volume workload, generated conditional schemas, staged auto-condition tuning, and query-only score.");

		if (ImGui::Checkbox("Per-cloud auto conditions", &state.autoConditions) && state.autoConditions)
		{
			state.generatedConditional = true;
			state.useRankModel = false;
		}
		drawHelpMarker("Estimates cheap point-cloud/node domains, tunes numeric condition thresholds, and writes a reusable measured selector.");
		ImGui::SameLine();
		bool realCloudOnly = !state.includeSynthetic;
		if (ImGui::Checkbox("Real cloud only", &realCloudOnly))
			state.includeSynthetic = !realCloudOnly;
		drawHelpMarker("Keeps the run specialized to the selected cloud instead of mixing in synthetic validation clouds.");

		bool cudaMixed = state.evaluator == 1 && state.cudaBuilder == 8 && state.cudaDevice == 0;
		if (ImGui::Checkbox("CUDA Mixed device 0", &cudaMixed))
		{
			if (cudaMixed)
			{
				state.evaluator = 1;
				state.cudaDevice = 0;
				state.cudaBuilder = 8;
			}
			else
			{
				state.evaluator = 0;
			}
		}
		drawHelpMarker("Uses the schema-aware mixed CUDA builder on device 0. Schema-search still falls back to CPU if CUDA is unavailable.");
		ImGui::SameLine();
		const int volumeIndex = workloadIndexForToken(state, "volume_small_medium");
		bool volumeWorkload = volumeIndex >= 0 && state.selectedWorkload == volumeIndex;
		ImGui::BeginDisabled(volumeIndex < 0);
		if (ImGui::Checkbox("Volume workload", &volumeWorkload) && volumeWorkload)
			state.selectedWorkload = volumeIndex;
		ImGui::EndDisabled();
		drawHelpMarker("Uses the small/medium 3D volume workload, avoiding KNN in the default tuning path.");

		if (ImGui::Checkbox("Generated-only candidates", &state.generatedOnly))
			state.generateSchemas = state.generateSchemas || state.generatedOnly;
		drawHelpMarker("Focuses the measured search on generated schema variants instead of fixed baselines.");
		ImGui::SameLine();
		if (ImGui::Checkbox("Conditional generated blocks", &state.generatedConditional) && state.autoConditions)
			state.generatedConditional = true;
		drawHelpMarker("Keeps local node predicates in the generated schema space.");

		if (state.autoConditions)
		{
			ImGui::Text("Budget: %d proxy candidates, %d proxy queries, top %d -> %d",
				state.conditionProxyCandidates,
				state.conditionProxyQueries,
				state.conditionFinalTopK,
				state.conditionConfirmTopK);
		}
		else
		{
			ImGui::Text("Budget: %d generated candidates, top-k %d",
				state.generatedCount,
				state.benchmarkTopK);
		}
	}

	void drawDatasetPanel(GuiState& state)
	{
		drawSectionTitle("Point Cloud");
		drawPathInput("Point cloud", state.inputPath, "Point cloud to optimize against. With synthetic datasets disabled, the selected best schema is overfit to this one cloud and workload.");
		ImGui::Checkbox("Use binary cache", &state.useBinaryCache);
		drawHelpMarker("Loads and writes the .mdspc cache beside the source point cloud. This speeds repeated runs but does not change the measured query workload.");
		ImGui::SameLine();
		ImGui::Checkbox("Rebuild cache", &state.rebuildBinaryCache);
		drawHelpMarker("Forces a fresh read from the source point cloud and replaces the .mdspc cache. Useful after the input file changes or if the cache looks stale.");
		ImGui::Checkbox("Include synthetic datasets", &state.includeSynthetic);
		drawHelpMarker("Adds built-in synthetic point clouds to the search. Keep this off when you want the optimizer to specialize to your current point cloud only.");
		if (state.includeSynthetic)
		{
			ImGui::InputInt("Synthetic scale", &state.syntheticScale);
			drawHelpMarker("Point count scale for each synthetic dataset. Larger values make synthetic validation more realistic and slower.");
		}

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
		const bool knnWorkload = !state.workloads.empty() &&
			state.workloads[static_cast<size_t>(state.selectedWorkload)].path.find("knn") != std::string::npos;
		if (knnWorkload)
		{
			ImGui::InputInt("KNN k", &state.knnK);
			drawHelpMarker("Neighbor count for KNN queries. Larger k usually increases KNN cost and can favor different structures.");
		}
		ImGui::InputInt("Query seed", &state.querySeed);
		drawHelpMarker("Random seed for generated query centers and boxes. Keep fixed for comparable runs; change it to test robustness.");
	}

	void drawSchemaPanel(GuiState& state)
	{
		drawSectionTitle("Search Space");
		ImGui::Checkbox("Generate schemas", &state.generateSchemas);
		drawHelpMarker("Samples new multi-DS JSON candidates from the configured bounds, then benchmarks them. This expands beyond the fixed schema list.");
		ImGui::SameLine();
		if (ImGui::Checkbox("Conditional blocks", &state.generatedConditional) && state.autoConditions)
			state.generatedConditional = true;
		drawHelpMarker("Allows later schema blocks to activate only for local node conditions such as point count, density, or height ratio. This is the current branch-adaptive multi-DS mechanism.");
		ImGui::SameLine();
		ImGui::Checkbox("Generated only", &state.generatedOnly);
		drawHelpMarker("Ignores checked fixed schemas and searches only generated candidates. Turn this off to compare generated candidates against known baselines.");

		if (state.autoConditions)
		{
			drawSectionTitle("Auto-Condition Budget");
			ImGui::InputInt("Proxy candidates", &state.conditionProxyCandidates);
			drawHelpMarker("Number of domain-aware candidate schemas screened on the downsampled proxy cloud.");
			ImGui::InputInt("Proxy points", &state.conditionProxyPoints);
			drawHelpMarker("Point cap for the proxy stage. The full cloud is still used for shortlist and confirmation stages.");
			ImGui::InputInt("Proxy queries", &state.conditionProxyQueries);
			drawHelpMarker("Prepared workload query count for the first cheap stage.");
			ImGui::InputInt("Shortlist top-k", &state.conditionFinalTopK);
			drawHelpMarker("Candidates promoted from proxy screening to a full-cloud short run.");
			ImGui::InputInt("Confirm top-k", &state.conditionConfirmTopK);
			drawHelpMarker("Candidates promoted from the short full-cloud run to final confirmation with the requested query count.");
			drawPathInput("Tuned schemas", state.autoConditionSchemaDir, "Directory for the final numeric tuned schema JSON files.");
			drawPathInput("Selector", state.selectorOutputPath, "Measured selector artifact used by later --schema auto runs.");
		}
		else
		{
			ImGui::InputInt("Generated count", &state.generatedCount);
			drawHelpMarker("Number of candidate JSON schemas to sample. More candidates explores more of the space and takes longer.");
			ImGui::InputInt("Benchmark top-k", &state.benchmarkTopK);
			drawHelpMarker("If a rank model is enabled, benchmark only the model's top-k candidates. 0 means benchmark every candidate and rely only on measured results.");
		}

		drawPathInput("Generated dir", state.generatedSchemaDir, "Directory where generated schema JSON files are written. The best result table points back to one of these files when a generated candidate wins.");

		if (ImGui::CollapsingHeader("Baseline schemas"))
		{
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

			if (ImGui::BeginTable("schemas", 4, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
			{
				ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 32.0f);
				ImGui::TableSetupColumn("Schema", ImGuiTableColumnFlags_WidthStretch, 0.32f);
				ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch, 0.58f);
				ImGui::TableSetupColumn("View", ImGuiTableColumnFlags_WidthFixed, 116.0f);
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
					ImGui::TableSetColumnIndex(1);
					ImGui::TextUnformatted(entry.label.c_str());
					ImGui::TableSetColumnIndex(2);
					ImGui::TextUnformatted(entry.path.c_str());
					ImGui::TableSetColumnIndex(3);
					drawSchemaActions(state, entry.path);
					ImGui::PopID();
				}
				ImGui::EndTable();
			}
		}

		if (ImGui::CollapsingHeader("Advanced generated bounds"))
		{
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
		}

		if (ImGui::CollapsingHeader("Evolutionary optimizer"))
		{
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

			ImGui::Separator();
			ImGui::Checkbox("Multi-fidelity rungs", &state.useRungSchedule);
			drawHelpMarker("Successive halving: each batch is first scored cheaply with the visit-count surrogate, then the top survivors are re-measured with wall-clock latency at increasing query counts. Lets the optimizer touch more candidates for the same wall-clock.");
			ImGui::BeginDisabled(!state.useRungSchedule);
			ImGui::InputInt("R0 proxy queries", &state.rungProxyQueries);
			drawHelpMarker("Query count for the cheap proxy rung. 4 is plenty since the proxy uses deterministic visit/test counters, not noisy timings.");
			ImGui::InputInt("R0 proxy advance top-K", &state.rungProxyAdvance);
			drawHelpMarker("Number of candidates promoted from the proxy rung to the latency rung. Typically a quarter of the population.");
			ImGui::SliderFloat("R0 proxy alpha", &state.rungProxyAlpha, 0.0f, 1.0f, "%.2f");
			drawHelpMarker("Weight applied to averageTestedPoints in the proxy score (visitedNodes + alpha * testedPoints). Raise it to penalise schemas that touch many points per visited node.");
			ImGui::InputInt("R1 full queries", &state.rungFullQueries);
			drawHelpMarker("Query count for the mid-fidelity latency rung. 16 keeps measurement cheap while still discriminating real winners.");
			ImGui::InputInt("R1 full advance top-K", &state.rungFullAdvance);
			drawHelpMarker("Number of candidates promoted from the latency rung to the confirmation rung. 4-8 is a normal publication setting.");
			ImGui::InputInt("R2 confirm queries", &state.rungConfirmQueries);
			drawHelpMarker("Query count for the confirmation rung. The optimizer only writes records from this final rung to the CSV.");
			drawPathInput("Surrogate model", state.rungSurrogatePath, "Optional. Path to an exported linear/ONNX selector JSON. When set, between rungs the surrogate proposes top-K candidates from a random pool, injected as extra immigrants.");
			ImGui::InputInt("Surrogate pool", &state.rungSurrogatePool);
			drawHelpMarker("Number of fresh genomes sampled per generation and scored by the surrogate. 0 disables the acquisition step.");
			ImGui::InputInt("Surrogate top-K", &state.rungSurrogateTop);
			drawHelpMarker("Number of surrogate-ranked candidates injected as additional immigrants each generation. Cheap to raise; the proxy rung filters them anyway.");
			ImGui::EndDisabled();

			ImGui::Separator();
			ImGui::Checkbox("Refine thresholds (CMA-style ES)", &state.refineThresholds);
			drawHelpMarker("After the GA finishes, the top-K archive entries with conditional levels get their numeric thresholds tuned by a (1+lambda)-ES under the visit-proxy. Survivors that improved are re-measured at full fidelity.");
			ImGui::BeginDisabled(!state.refineThresholds);
			ImGui::InputInt("Refine top-K", &state.refineThresholdsTopK);
			drawHelpMarker("Number of GA archive survivors that get their thresholds refined. 4 is a normal publication setting.");
			ImGui::InputInt("Refine evals/cand", &state.refineThresholdsEvals);
			drawHelpMarker("Total evaluation budget per refined candidate. 40-80 is the regime where the (1+lambda)-ES converges without overspending.");
			ImGui::SliderFloat("Refine sigma0", &state.refineThresholdsSigma, 0.01f, 0.8f, "%.2f");
			drawHelpMarker("Initial step size in normalised [0,1] threshold space. 0.3 is a good default; larger values explore wider, smaller values polish.");
			ImGui::InputInt("Refine seed", &state.refineThresholdsSeed);
			drawHelpMarker("RNG seed for the refiner. Different seeds produce different threshold trajectories.");
			ImGui::EndDisabled();
		}
	}

	void drawScoringPanel(GuiState& state)
	{
		drawSectionTitle("Evaluator");
		const char* evaluators[] = { "CPU", "CUDA" };
		ImGui::Combo("Backend", &state.evaluator, evaluators, IM_ARRAYSIZE(evaluators));
		drawHelpMarker("Chooses where measured schema fitness runs. CUDA builds the selected GPU structure and measures range/count/radius/KNN queries there.");
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

			const char* builders[] = { "LBVH", "KDTree", "BIH", "Octree", "KarrasOctree", "QuadTree", "RegularGrid", "HGrid", "Mixed" };
			ImGui::Text("CUDA: device %d, %s", state.cudaDevice, builders[state.cudaBuilder]);
			if (ImGui::CollapsingHeader("Advanced CUDA"))
			{
				ImGui::InputInt("CUDA device", &state.cudaDevice);
				drawHelpMarker("GPU id passed to cudaSetDevice. Use 0 unless you have several CUDA GPUs.");
				ImGui::Combo("Structure", &state.cudaBuilder, builders, IM_ARRAYSIZE(builders));
				drawHelpMarker("LBVH, KDTree, BIH, Octree, KarrasOctree, QuadTree, RegularGrid, HGrid, and MixedTree schemas are implemented. KarrasOctree uses Morton sorting and prefix child ranges; BIH is a binary interval hierarchy with tight child bounds; standalone HGrid builds several RegularGrid levels and chooses one per query; Mixed follows the schema's per-depth structure schedule, including RegularGrid and HGrid grid split levels.");
				ImGui::InputInt("Query batch", &state.cudaQueryBatch);
				drawHelpMarker("Number of CUDA queries uploaded/launched per batch. 0 runs the whole generated workload as one batch.");
				ImGui::InputInt("Memory budget MB", &state.cudaMemoryBudgetMb);
				drawHelpMarker("Optional guardrail that rejects CUDA builds whose point buffers, sorted arrays, nodes, and sort scratch exceed this budget.");
			}
		}

		drawSectionTitle("Surrogate");
		if (state.autoConditions)
			ImGui::TextDisabled("Bypassed while per-cloud auto-condition tuning is enabled.");
		ImGui::BeginDisabled(state.autoConditions);
		ImGui::Checkbox("Use rank model", &state.useRankModel);
		drawHelpMarker("Uses the exported JSON/ONNX selector only to rank/prune candidates before benchmarking. The final best schema still comes from measured C++ timings.");
		drawPathInput("Rank model", state.rankModelPath, "Path to a selector wrapper JSON, usually models/schema_selector.json or models/schema_selector_onnx.json.");
		ImGui::EndDisabled();

		drawSectionTitle("Score Weights");
		ImGui::Text("Build %.4f, memory %.4f, imbalance %.4f",
			state.scoreBuildWeight,
			state.scoreMemoryWeight,
			state.scoreImbalanceWeight);
		if (ImGui::CollapsingHeader("Advanced scoring"))
		{
			ImGui::InputFloat("Build", &state.scoreBuildWeight, 0.001f, 0.01f, "%.4f");
			drawHelpMarker("Adds build time into the score. Keep at 0 if you only care about saved/reused structures and query speed.");
			ImGui::InputFloat("Memory", &state.scoreMemoryWeight, 0.001f, 0.01f, "%.4f");
			drawHelpMarker("Adds a memory penalty. Increase this if two schemas have similar query time but one is much larger.");
			ImGui::InputFloat("Imbalance", &state.scoreImbalanceWeight, 0.001f, 0.01f, "%.4f");
			drawHelpMarker("Adds a penalty for uneven leaf occupancy. Increase this if winners have pathological leaves or unstable query behavior.");
		}

		drawSectionTitle("Outputs");
		drawPathInput("CSV", state.csvPath, "Full measured result table, with one row per dataset/workload/schema candidate.");
		drawPathInput("Best CSV", state.bestCsvPath, "Compact winner table. This is what the GUI reads back to populate Best Results.");
		drawPathInput("Pareto CSV", state.paretoCsvPath, "Non-dominated front over (avg latency, build time, memory, imbalance) per (dataset, workload). Empty path disables.");

		drawSectionTitle("Speed-ups");
		ImGui::TextWrapped(
			"Auto-conditions runs three stages: proxy (256 candidates on a tiny downsampled cloud, "
			"ranked by deterministic visit-counts), shortlist (top 16 rebuilt on the full cloud with "
			"a short query set), confirmation (top 4 with the full query set). The score cache "
			"survives all stages and across runs, so a re-run after a crash or interruption resumes "
			"where it left off.");
		ImGui::Checkbox("Persistent score cache", &state.scoreCacheEnabled);
		drawHelpMarker("Memoises (schema, cloud, workload) -> measured score. If you abort a run mid-shortlist, the next run skips everything already measured. Designed for safe restarts on stalled candidates like deep hgrid configs.");
		if (state.scoreCacheEnabled)
		{
			drawPathInput("Cache JSONL", state.scoreCachePath, "Append-only JSONL file. Delete it (or tick rebuild) to invalidate.");
			ImGui::Checkbox("Rebuild cache this run", &state.rebuildScoreCache);
			drawHelpMarker("Deletes the cache file before this run so every candidate is measured fresh.");
		}
		ImGui::SliderInt("Parallel CPU workers", &state.parallelDispatch, 1, 32);
		drawHelpMarker("Honored only when the evaluator is CPU. Parallelises the evolutionary GA loop AND the auto-condition proxy/shortlist/confirmation loops. The CUDA path stays serial: GPU state and the per-builder build cache are not thread-safe.");
		if (state.evaluator == 1 && state.parallelDispatch > 1)
			ImGui::TextDisabled("(currently using CUDA: workers are ignored)");

		ImGui::Checkbox("Include baseline data structures", &state.includeBaselineSchemas);
		drawHelpMarker("Adds canonical single-block schemas (pure QuadTree, Octree, KDTree, BVH on CPU; also LBVH, KarrasOctree, RegularGrid, HGrid, BIH on CUDA) as controls. Baselines are force-promoted through the proxy/shortlist/confirmation stages so you can always see how the naive structures perform on the full cloud, even if their proxy rank is poor.");
	}

	bool isCudaResult(const std::string& backend)
	{
		return backend == "cuda" || backend == "gpu";
	}

	void drawResultsTable(const std::vector<BestResult>& results, GuiState& state)
	{
		if (results.empty())
		{
			ImGui::TextUnformatted("No completed results yet.");
			return;
		}

		if (ImGui::BeginTable("best-results", 12, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp))
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
			ImGui::TableSetupColumn("View", ImGuiTableColumnFlags_WidthFixed, 116.0f);
			ImGui::TableHeadersRow();

			for (size_t i = 0; i < results.size(); ++i)
			{
				const BestResult& result = results[i];
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
				ImGui::TableSetColumnIndex(11);
				ImGui::PushID(static_cast<int>(i));
				drawSchemaActions(state, result.schemaPath);
				ImGui::PopID();
			}

			ImGui::EndTable();
		}
	}

	void drawLiveRankingTable(const std::vector<LiveRankingEntry>& ranking, size_t evaluatedCandidates, int topN, GuiState& state)
	{
		ImGui::Text("Measured candidates: %zu", evaluatedCandidates);
		if (ranking.empty())
		{
			ImGui::TextUnformatted("Live ranking will appear after the first candidate finishes.");
			return;
		}

		if (ImGui::BeginTable("live-ranking", 12, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp))
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
			ImGui::TableSetupColumn("View", ImGuiTableColumnFlags_WidthFixed, 116.0f);
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
				ImGui::TableSetColumnIndex(11);
				ImGui::PushID(static_cast<int>(i));
				drawSchemaActions(state, entry.schemaPath);
				ImGui::PopID();
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
		if (ImGui::Button("Start tuning", ImVec2(180.0f, 34.0f)))
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
		drawLiveRankingTable(liveRanking, evaluatedCandidates, state.liveRankingTopN, state);

		drawSectionTitle("Best Results");
		drawResultsTable(results, state);

		drawSectionTitle("Log");
		ImGui::BeginChild("log", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_HorizontalScrollbar);
		if (log.empty())
			ImGui::TextUnformatted("The run log will appear here.");
		else
			ImGui::TextUnformatted(log.c_str());
		ImGui::EndChild();
	}

	ImVec2 projectPreviewPoint(const glm::vec3& point, const ImVec2& origin, const ImVec2& size, const StructurePreview& preview)
	{
		const float yawCos = std::cos(preview.yaw);
		const float yawSin = std::sin(preview.yaw);
		const float pitchCos = std::cos(preview.pitch);
		const float pitchSin = std::sin(preview.pitch);

		const float x0 = yawCos * point.x + yawSin * point.z;
		const float z0 = -yawSin * point.x + yawCos * point.z;
		const float y1 = pitchCos * point.y - pitchSin * z0;
		const float z1 = pitchSin * point.y + pitchCos * z0;
		const float perspective = preview.zoom / std::max(0.25f, preview.zoom + z1);
		const float scale = std::min(size.x, size.y) * 0.34f * perspective;
		return ImVec2(origin.x + size.x * 0.5f + x0 * scale, origin.y + size.y * 0.52f - y1 * scale);
	}

	void drawPreviewBox(ImDrawList* drawList, const PreviewBox& box, const ImVec2& origin, const ImVec2& size, const StructurePreview& preview)
	{
		const glm::vec3 corners[8] = {
			{ box.min.x, box.min.y, box.min.z },
			{ box.max.x, box.min.y, box.min.z },
			{ box.max.x, box.max.y, box.min.z },
			{ box.min.x, box.max.y, box.min.z },
			{ box.min.x, box.min.y, box.max.z },
			{ box.max.x, box.min.y, box.max.z },
			{ box.max.x, box.max.y, box.max.z },
			{ box.min.x, box.max.y, box.max.z },
		};
		constexpr int edges[12][2] = {
			{ 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
			{ 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
			{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
		};

		const bool rootBounds = box.depth == 0 && preview.showFullSchedule;
		const float alpha = rootBounds ? 0.30f : std::clamp(0.92f - static_cast<float>(box.depth) * 0.06f, 0.35f, 0.92f);
		const float thickness = box.depth == 0 ? 1.6f : 1.0f;
		const ImU32 color = rootBounds
			? IM_COL32(174, 184, 196, 120)
			: packedColorForStructureName(box.typeName, alpha);

		for (const auto& edge : edges)
		{
			const ImVec2 a = projectPreviewPoint(corners[edge[0]], origin, size, preview);
			const ImVec2 b = projectPreviewPoint(corners[edge[1]], origin, size, preview);
			drawList->AddLine(a, b, color, thickness);
		}
	}

	void drawPreviewLegend()
	{
		const struct
		{
			const char* label;
			const char* typeName;
		} entries[] = {
			{ "QuadTree", "QuadTree" },
			{ "KDTree", "KDTree" },
			{ "BIH", "BIH" },
			{ "Octree", "Octree" },
			{ "KarrasOctree", "KarrasOctree" },
			{ "BVH", "BVH" },
			{ "LBVH", "LBVH" },
			{ "RegularGrid", "RegularGrid" },
			{ "HGrid", "HGrid" },
		};

		for (const auto& entry : entries)
		{
			const ImVec4 color = colorForStructureName(entry.typeName);
			ImGui::TextColored(color, "%s", entry.label);
			ImGui::SameLine();
		}
		ImGui::NewLine();
	}

	void drawSchemaFileViewerWindow(GuiState& state)
	{
		SchemaFileViewer& viewer = state.fileViewer;
		if (!viewer.open)
			return;

		const ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 72.0f, viewport->WorkPos.y + 72.0f), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(720.0f, 560.0f), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Schema JSON", &viewer.open, ImGuiWindowFlags_HorizontalScrollbar))
		{
			ImGui::TextUnformatted(viewer.title.c_str());
			if (!viewer.path.empty())
			{
				ImGui::TextDisabled("%s", viewer.path.c_str());
				if (ImGui::Button("Reload"))
					viewer.content = loadTextFile(viewer.path, viewer.error);
				ImGui::SameLine();
				if (ImGui::Button("Preview boxes"))
					openStructurePreview(state, viewer.path);
				ImGui::SameLine();
				ImGui::Checkbox("Wrap", &viewer.wrap);
			}
			if (!viewer.error.empty())
			{
				ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.38f, 1.0f));
				ImGui::TextWrapped("%s", viewer.error.c_str());
				ImGui::PopStyleColor();
			}

			ImGui::Separator();
			ImGui::BeginChild("schema-json-content", ImVec2(0.0f, 0.0f), true, viewer.wrap ? 0 : ImGuiWindowFlags_HorizontalScrollbar);
			if (viewer.wrap)
				ImGui::TextWrapped("%s", viewer.content.c_str());
			else
				ImGui::TextUnformatted(viewer.content.c_str());
			ImGui::EndChild();
		}
		ImGui::End();
	}

	void drawStructurePreviewWindow(GuiState& state, bool optimizerRunning)
	{
		StructurePreview& preview = state.structurePreview;
		if (!preview.open)
			return;

		if (preview.needsRebuild)
			rebuildStructurePreview(preview);

		const ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 620.0f, viewport->WorkPos.y + 96.0f), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(760.0f, 560.0f), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Structure Preview", &preview.open))
		{
			ImGui::TextUnformatted(preview.title.c_str());
			if (!preview.schemaName.empty())
			{
				ImGui::SameLine();
				ImGui::TextDisabled("%s", preview.schemaName.c_str());
			}
			if (!preview.path.empty())
				ImGui::TextDisabled("%s", preview.path.c_str());

			if (ImGui::Checkbox("Full schedule", &preview.showFullSchedule))
				preview.needsRebuild = true;
			ImGui::SameLine();
			drawHelpMarker("Focused mode shows one schema block at a time. Full schedule shows the old cumulative expansion and is mainly useful for simple fixed schemas.");

			if (!preview.showFullSchedule && !preview.phases.empty())
			{
				ImGui::SetNextItemWidth(260.0f);
				const int phaseCount = static_cast<int>(preview.phases.size());
				preview.phaseIndex = std::clamp(preview.phaseIndex, 0, phaseCount - 1);
				const char* currentPhase = preview.phases[preview.phaseIndex].label.c_str();
				if (ImGui::BeginCombo("Block", currentPhase))
				{
					for (int i = 0; i < phaseCount; ++i)
					{
						const bool selected = i == preview.phaseIndex;
						if (ImGui::Selectable(preview.phases[i].label.c_str(), selected))
						{
							preview.phaseIndex = i;
							preview.needsRebuild = true;
						}
						if (selected)
							ImGui::SetItemDefaultFocus();
					}
					ImGui::EndCombo();
				}
				ImGui::SameLine();
			}

			ImGui::SetNextItemWidth(92.0f);
			if (ImGui::InputInt(preview.showFullSchedule ? "Global depth" : "Local levels", &preview.maxDepth))
				preview.needsRebuild = true;
			ImGui::SameLine();
			ImGui::SetNextItemWidth(110.0f);
			if (ImGui::InputInt("Box cap", &preview.maxBoxes))
				preview.needsRebuild = true;
			ImGui::SameLine();
			if (ImGui::Button("Reload"))
				preview.needsRebuild = true;
			ImGui::SameLine();
			drawHelpMarker("This preview renders normalized boxes only. It does not load or rebuild the point cloud, so it stays outside the optimizer worker.");

			if (optimizerRunning)
			{
				ImGui::SameLine();
				ImGui::TextDisabled("optimizer running");
			}

			if (!preview.error.empty())
			{
				ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.38f, 1.0f));
				ImGui::TextWrapped("%s", preview.error.c_str());
				ImGui::PopStyleColor();
			}

			if (!preview.showFullSchedule && !preview.phases.empty())
			{
				const PreviewPhase& phase = preview.phases[preview.phaseIndex];
				ImGui::Text("Focused block: %s, schema depths %zu-%zu",
					phase.level.typeName.c_str(),
					phase.startDepth,
					phase.endDepth == 0 ? 0 : phase.endDepth - 1);
			}
			ImGui::Text("Boxes: %zu%s", preview.boxes.size(), preview.truncated ? " (capped)" : "");
			drawPreviewLegend();

			const ImVec2 canvasSize = ImVec2(
				std::max(320.0f, ImGui::GetContentRegionAvail().x),
				std::max(280.0f, ImGui::GetContentRegionAvail().y));
			const ImVec2 canvasOrigin = ImGui::GetCursorScreenPos();
			ImGui::InvisibleButton("structure-preview-canvas", canvasSize, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool hovered = ImGui::IsItemHovered();
			const bool active = ImGui::IsItemActive();
			ImGuiIO& io = ImGui::GetIO();
			if (active && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
			{
				preview.yaw += io.MouseDelta.x * 0.008f;
				preview.pitch = std::clamp(preview.pitch + io.MouseDelta.y * 0.008f, -1.25f, 1.25f);
			}
			if (hovered && io.MouseWheel != 0.0f)
				preview.zoom = std::clamp(preview.zoom - io.MouseWheel * 0.18f, 1.4f, 8.0f);

			ImDrawList* drawList = ImGui::GetWindowDrawList();
			drawList->AddRectFilled(canvasOrigin, ImVec2(canvasOrigin.x + canvasSize.x, canvasOrigin.y + canvasSize.y), IM_COL32(12, 15, 18, 255));
			drawList->AddRect(canvasOrigin, ImVec2(canvasOrigin.x + canvasSize.x, canvasOrigin.y + canvasSize.y), IM_COL32(55, 64, 75, 255));
			drawList->PushClipRect(canvasOrigin, ImVec2(canvasOrigin.x + canvasSize.x, canvasOrigin.y + canvasSize.y), true);
			for (const PreviewBox& box : preview.boxes)
				drawPreviewBox(drawList, box, canvasOrigin, canvasSize, preview);
			drawList->PopClipRect();
		}
		ImGui::End();
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
		drawPublicationPanel(state);
		if (ImGui::BeginTabBar("configuration-tabs"))
		{
			if (ImGui::BeginTabItem("Inputs"))
			{
				drawDatasetPanel(state);
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Search"))
			{
				drawSchemaPanel(state);
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Advanced"))
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

		drawSchemaFileViewerWindow(state);
		drawStructurePreviewWindow(state, session.running);
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

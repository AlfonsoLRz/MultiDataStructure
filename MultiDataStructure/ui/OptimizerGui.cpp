#include "OptimizerGui.h"

#include "../core/Config.h"
#include "../experiments/SchemaSearch.h"
#include "../workloads/points/LBVH.h"

static constexpr size_t TextBufferSize = 512;

struct SchemaEntry
{
	std::string _label;
	std::string _path;
	bool _selected = true;
};

struct WorkloadEntry
{
	std::string _label;
	std::string _path;
};

struct BestResult
{
	std::string _dataset;
	std::string _workload;
	std::string _schemaName;
	std::string _schemaPath;
	double _score = 0.0;
	double _averageLatencyMs = 0.0;
	double _buildTimeMs = 0.0;
	double _gpuBuildMs = 0.0;
	double _gpuQueryMs = 0.0;
	uint64_t _memoryBytes = 0;
	size_t _candidates = 0;
	std::string _backend;
	std::string _scoreMode = "unknown";
	std::string _scoreStage = "unknown";
	bool _scoreIsFinalLatency = false;
	size_t _effectiveQueries = 0;
};

struct LiveRankingEntry
{
	size_t _order = 0;
	std::string _dataset;
	std::string _workload;
	std::string _schemaName;
	std::string _schemaPath;
	std::string _backend;
	std::string _cudaBuilder;
	double _score = 0.0;
	double _averageLatencyMs = 0.0;
	double _p95LatencyMs = 0.0;
	double _buildTimeMs = 0.0;
	double _gpuBuildMs = 0.0;
	double _gpuQueryMs = 0.0;
	double _averageVisitedNodes = 0.0;
	double _averageTestedPoints = 0.0;
	uint64_t _memoryBytes = 0;
	std::string _scoreMode = "latency";
	std::string _scoreStage = "final";
	bool _scoreIsFinalLatency = true;
	size_t _effectiveQueries = 0;
};

struct SchemaFileViewer
{
	bool _open = false;
	bool _wrap = false;
	std::string _title = "Schema JSON";
	std::string _path;
	std::string _content;
	std::string _error;
};

struct PreviewBox
{
	glm::vec3 min = glm::vec3(0.0f);
	glm::vec3 max = glm::vec3(0.0f);
	std::string _typeName;
	size_t _depth = 0;
};

struct PreviewPhase
{
	SchemaLevelConfig _level;
	std::string _label;
	size_t _startDepth = 0;
	size_t _endDepth = 0;
};

struct StructurePreview
{
	bool _open = false;
	bool _needsRebuild = false;
	bool _truncated = false;
	bool _showFullSchedule = false;
	std::string _title = "Structure Preview";
	std::string _path;
	std::string _schemaName;
	std::string _error;
	std::vector<PreviewBox> _boxes;
	std::vector<PreviewPhase> _phases;
	int _phaseIndex = 0;
	int _maxDepth = 2;
	int _maxBoxes = 2048;
	float _yaw = 0.68f;
	float _pitch = 0.42f;
	float _zoom = 3.2f;
};

struct GuiState
{
	std::array<char, TextBufferSize> _inputPath{};
	std::array<char, TextBufferSize> _rankModelPath{};
	std::array<char, TextBufferSize> _csvPath{};
	std::array<char, TextBufferSize> _bestCsvPath{};
	std::array<char, TextBufferSize> _paretoCsvPath{};
	std::array<char, TextBufferSize> _explainReportPath{};
	std::array<char, TextBufferSize> _queryTracePath{};
	std::array<char, TextBufferSize> _generatedSchemaDir{};
	std::array<char, TextBufferSize> _autoConditionSchemaDir{};
	std::array<char, TextBufferSize> _selectorOutputPath{};

	std::vector<SchemaEntry> _schemas;
	std::vector<WorkloadEntry> _workloads;
	int _selectedWorkload = 0;

	bool _autoConditions = true;
	bool _includeSynthetic = false;
	bool _useBinaryCache = true;
	bool _rebuildBinaryCache = false;
	bool _generateSchemas = true;
	bool _generatedOnly = true;
	bool _generatedConditional = true;
	bool _generatedAdaptiveLeafCapacity = false;
	bool _queryMinimalPrimitives = true;
	bool _useRankModel = false;
	// GA path with rungs + threshold refinement; default-on.
	bool _optimizeSchemas = true;
	int _evaluator = 0;
	int _cudaDevice = 0;
	int _cudaBuilder = 0;
	int _cudaKnnBackend = 0;
	int _cudaQueryBatch = 0;
	int _cudaMemoryBudgetMb = 0;
	int _liveRankingTopN = 10;
	bool _advancedEvaluatorOpen = false;
	bool _advancedScoringOpen = false;

	int _queryCount = 64;
	int _knnK = 16;
	int _querySeed = 1337;
	int _syntheticScale = 512;
	int _generatedCount = 256;
	int _benchmarkTopK = 32;
	int _generatedMinBlocks = 2;
	int _generatedMaxBlocks = 3;
	int _generatedMaxDepth = 12;
	int _generatedMinLeaf = 32;
	int _generatedMaxLeaf = 32768;
	int _generatedSeed = 1337;
	int _optimizerGenerations = 3;
	int _optimizerPopulation = 64;
	int _optimizerElites = 6;
	int _optimizerSeed = 1337;
	int _conditionProxyCandidates = 256;
	int _conditionProxyPoints = 262144;
	int _conditionProxyQueries = 8;
	int _conditionFinalTopK = 16;
	int _conditionConfirmTopK = 4;
	float _generatedConditionProbability = 0.5f;
	float _generatedAdaptiveLeafProbability = 0.25f;
	float _optimizerMutationRate = 0.65f;
	float _optimizerRandomFraction = 0.20f;
	float _scoreBuildWeight = 0.0f;
	float _scoreMemoryWeight = 0.0f;
	float _scoreImbalanceWeight = 0.0f;

	// Score cache and parallel dispatch defaults.
	bool _scoreCacheEnabled = true;
	std::array<char, 512> _scoreCachePath{};
	bool _rebuildScoreCache = false;
	int _parallelDispatch = 1;
	bool _includeBaselineSchemas = true;

	// Multi-fidelity rung schedule: visit-proxy, latency, then full-workload confirmation; default-on.
	bool _useRungSchedule = true;
	int _rungProxyQueries = 4;
	int _rungProxyAdvance = 32;
	float _rungProxyAlpha = 0.1f;
	int _rungFullQueries = 16;
	int _rungFullAdvance = 8;
	int _rungConfirmQueries = 64;
	std::array<char, TextBufferSize> rungSurrogatePath{};
	int _rungSurrogatePool = 0;
	int _rungSurrogateTop = 0;

	// Tune top-K conditional thresholds with a (1+lambda)-ES after the GA, then re-measure improvers; default-on.
	bool _refineThresholds = true;
	int _refineThresholdsTopK = 4;
	int _refineThresholdsEvals = 60;
	float _refineThresholdsSigma = 0.3f;
	int _refineThresholdsSeed = 1337;

	// Re-measure top-K per (dataset, workload) over N seeds for a mean + 95% bootstrap CI; default-on.
	bool _confirmSeeds = true;
	int _confirmSeedsCount = 5;
	int _confirmTopK = 4;

	// Diversity controls: crossover plus NSGA-II elite ranking; default-on.
	float _optimizerCrossoverRate = 0.4f;
	bool _optimizerUseNsga2 = true;
	bool _repairMutations = true;
	int _repairTopK = 4;
	int _repairPerCandidate = 2;

	SchemaFileViewer _fileViewer;
	StructurePreview _structurePreview;
};

struct RunSession
{
	std::thread _worker;
	std::atomic<bool> _running = false;
	std::atomic<bool> _finished = false;
	std::mutex _mutex;
	std::string _status = "Idle";
	std::string _log;
	std::string _error;
	std::vector<BestResult> _bestResults;
	std::vector<LiveRankingEntry> _liveRanking;
	size_t _evaluatedCandidates = 0;
	int _exitCode = 0;
};

template <size_t N>
static void setText(std::array<char, N>& buffer, const std::string& text)
{
	buffer.fill('\0');
	const size_t count = std::min(text.size(), N - 1);
	std::memcpy(buffer.data(), text.data(), count);
}

template <size_t N>
static std::string textValue(const std::array<char, N>& buffer)
{
	return std::string(buffer.data());
}

static bool filesystemExists(const std::filesystem::path& value)
{
	if (value.empty())
		return false;

	std::error_code error;
	return std::filesystem::exists(value, error);
}

static void addAncestorSearchRoots(std::vector<std::filesystem::path>& roots, std::filesystem::path start)
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

static std::filesystem::path executablePath()
{
#ifdef _WIN32
	char* programPath = nullptr;
	if (_get_pgmptr(&programPath) == 0 && programPath)
		return std::filesystem::path(programPath);
#endif
	return {};
}

static std::vector<std::filesystem::path> searchRoots()
{
	std::vector<std::filesystem::path> roots;
	addAncestorSearchRoots(roots, std::filesystem::current_path());
	addAncestorSearchRoots(roots, executablePath());
	return roots;
}

static std::filesystem::path projectRoot()
{
	for (const std::filesystem::path& root : searchRoots())
	{
		if (filesystemExists(root / "configs" / "schemas") && filesystemExists(root / "MultiDataStructure.sln"))
			return root;
	}

	return {};
}

static std::string projectPath(const std::string& relativePath)
{
	const std::filesystem::path root = projectRoot();
	if (root.empty())
		return relativePath;
	return (root / relativePath).lexically_normal().string();
}

static std::string resolvePath(const std::string& configuredPath)
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

static std::string schemaLabelFromPath(const std::string& path)
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

static std::vector<std::string> splitCsvLine(const std::string& line)
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

static size_t columnIndex(const std::vector<std::string>& header, const std::string& name)
{
	const auto found = std::find(header.begin(), header.end(), name);
	if (found == header.end())
		return std::numeric_limits<size_t>::max();
	return static_cast<size_t>(std::distance(header.begin(), found));
}

static std::string csvValue(const std::vector<std::string>& row, size_t index)
{
	if (index == std::numeric_limits<size_t>::max() || index >= row.size())
		return {};
	return row[index];
}

static double parseDouble(const std::string& value)
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

static uint64_t parseUint64(const std::string& value)
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

static std::string loadTextFile(const std::string& configuredPath, std::string& error)
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

static std::string normalizedStructureName(std::string value)
{
	value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
		return std::isspace(c) || c == '_' || c == '-';
	}), value.end());
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	});
	return value;
}

static bool isStructureName(const std::string& normalized, std::initializer_list<const char*> names)
{
	for (const char* name : names)
	{
		if (normalized == name)
			return true;
	}
	return false;
}

static ImVec4 colorForStructureName(const std::string& typeName, float alpha = 1.0f)
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

static ImU32 packedColorForStructureName(const std::string& typeName, float alpha = 1.0f)
{
	return ImGui::ColorConvertFloat4ToU32(colorForStructureName(typeName, alpha));
}

static int longestAxis(const glm::vec3& extent)
{
	if (extent.x >= extent.y && extent.x >= extent.z)
		return 0;
	if (extent.y >= extent.z)
		return 1;
	return 2;
}

static std::string previewPhaseLabel(const SchemaLevelConfig& level, size_t startDepth, size_t endDepth)
{
	std::ostringstream output;
	output << level._typeName << " d" << startDepth << "-d" << (endDepth == 0 ? 0 : endDepth - 1);
	if (!level._condition.empty())
		output << " conditional";
	return output.str();
}

static std::vector<PreviewPhase> previewPhasesForSchema(const SchemaConfig& schema)
{
	std::vector<PreviewPhase> phases;
	const size_t schemaMaxDepth = schema._buildPolicy._maxDepth > 0
		? std::min(schema._buildPolicy._maxDepth, schema.totalLevels())
		: schema.totalLevels();
	size_t startDepth = 0;
	for (const SchemaLevelConfig& level : schema._levels)
	{
		const size_t endDepth = startDepth + level._numLevels;
		if (startDepth >= schemaMaxDepth)
			break;

		PreviewPhase phase;
		phase._level = level;
		phase._startDepth = startDepth;
		phase._endDepth = std::min(endDepth, schemaMaxDepth);
		phase._label = previewPhaseLabel(level, phase._startDepth, phase._endDepth);
		phases.push_back(std::move(phase));
		startDepth = endDepth;
	}
	return phases;
}

static void pushGridPreviewChildren(
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
				child._typeName = typeName;
				child._depth = depth;
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

static void pushGridPreviewChildren(
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

static std::vector<PreviewBox> previewChildrenForLevel(const PreviewBox& parent, const SchemaLevelConfig& level, size_t depth)
{
	std::vector<PreviewBox> children;
	const std::string normalized = normalizedStructureName(level._typeName);
	if (isStructureName(normalized, { "quadtree", "quadtreenode", "qt" }))
	{
		glm::uvec3 cells(2, 2, 2);
		cells[longestAxis(parent.max - parent.min)] = 1;
		pushGridPreviewChildren(parent, depth, level._typeName, cells, children);
	}
	else if (isStructureName(normalized, { "octree", "octreenode", "ot", "karrasoctree", "mortonoctree", "octreekarras", "octreemorton" }))
	{
		pushGridPreviewChildren(parent, depth, level._typeName, 2, 2, 2, children);
	}
	else if (isStructureName(normalized, { "regulargrid", "uniformgrid", "grid", "grid3d", "hgrid", "hierarchicalgrid", "hierarchicalgrid3d" }))
	{
		pushGridPreviewChildren(parent, depth, level._typeName, 3, 3, 3, children);
	}
	else
	{
		const glm::vec3 extent = parent.max - parent.min;
		const int axis = longestAxis(extent);
		const float splitValue = parent.min[axis] + extent[axis] * 0.5f;

		PreviewBox left = parent;
		PreviewBox right = parent;
		left._typeName = level._typeName;
		right._typeName = level._typeName;
		left._depth = depth;
		right._depth = depth;
		left.max[axis] = splitValue;
		right.min[axis] = splitValue;
		children.push_back(left);
		children.push_back(right);
	}
	return children;
}

static void rebuildStructurePreview(StructurePreview& preview)
{
	preview._error.clear();
	preview._boxes.clear();
	preview._phases.clear();
	preview._truncated = false;
	preview._needsRebuild = false;
	preview._maxDepth = std::clamp(preview._maxDepth, 1, 9);
	preview._maxBoxes = std::clamp(preview._maxBoxes, 64, 20000);

	try
	{
		const SchemaConfig schema = Config::loadSchemaConfig(preview._path);
		preview._schemaName = schema._name.empty() ? schemaLabelFromPath(preview._path) : schema._name;
		preview._phases = previewPhasesForSchema(schema);
		if (preview._phases.empty())
			throw std::runtime_error("Schema has no previewable levels");
		preview._phaseIndex = std::clamp(preview._phaseIndex, 0, static_cast<int>(preview._phases.size() - 1));
		const size_t schemaMaxDepth = schema._buildPolicy._maxDepth > 0
			? std::min(schema._buildPolicy._maxDepth, schema.totalLevels())
			: schema.totalLevels();

		PreviewBox root;
		root.min = glm::vec3(-1.0f, -1.0f, -1.0f);
		root.max = glm::vec3(1.0f, 1.0f, 1.0f);
		root._typeName = preview._showFullSchedule ? "Bounds" : preview._phases[preview._phaseIndex]._level._typeName;
		root._depth = 0;
		preview._boxes.push_back(root);

		std::vector<PreviewBox> frontier = { root };
		if (preview._showFullSchedule)
		{
			const size_t targetDepth = std::min<size_t>(schemaMaxDepth, static_cast<size_t>(preview._maxDepth));
			for (size_t depth = 0; depth < targetDepth && !frontier.empty(); ++depth)
			{
				const SchemaLevelConfig& level = schema.levelForDepth(depth);
				std::vector<PreviewBox> next;
				for (const PreviewBox& node : frontier)
				{
					std::vector<PreviewBox> children = previewChildrenForLevel(node, level, depth + 1);
					for (PreviewBox& child : children)
					{
						if (preview._boxes.size() >= static_cast<size_t>(preview._maxBoxes))
						{
							preview._truncated = true;
							break;
						}
						next.push_back(child);
						preview._boxes.push_back(std::move(child));
					}
					if (preview._truncated)
						break;
				}
				frontier = std::move(next);
				if (preview._truncated)
					break;
			}
		}
		else
		{
			const PreviewPhase& phase = preview._phases[preview._phaseIndex];
			const size_t phaseLevels = std::max<size_t>(1, phase._endDepth - phase._startDepth);
			const size_t targetDepth = std::min<size_t>(phaseLevels, static_cast<size_t>(preview._maxDepth));
			for (size_t depth = 0; depth < targetDepth && !frontier.empty(); ++depth)
			{
				std::vector<PreviewBox> next;
				for (const PreviewBox& node : frontier)
				{
					std::vector<PreviewBox> children = previewChildrenForLevel(node, phase._level, depth + 1);
					for (PreviewBox& child : children)
					{
						if (preview._boxes.size() >= static_cast<size_t>(preview._maxBoxes))
						{
							preview._truncated = true;
							break;
						}
						next.push_back(child);
						preview._boxes.push_back(std::move(child));
					}
					if (preview._truncated)
						break;
				}
				frontier = std::move(next);
				if (preview._truncated)
					break;
			}
		}
	}
	catch (const std::exception& exception)
	{
		preview._error = exception.what();
	}
}

static void openSchemaFile(GuiState& state, const std::string& path)
{
	state._fileViewer._open = true;
	state._fileViewer._path = resolvePath(path);
	state._fileViewer._title = schemaLabelFromPath(path);
	state._fileViewer._content = loadTextFile(path, state._fileViewer._error);
}

static void openStructurePreview(GuiState& state, const std::string& path)
{
	const std::string resolved = resolvePath(path);
	if (state._structurePreview._path != resolved)
	{
		state._structurePreview._phaseIndex = 0;
		state._structurePreview._maxDepth = 2;
		state._structurePreview._showFullSchedule = false;
	}
	state._structurePreview._open = true;
	state._structurePreview._path = resolved;
	state._structurePreview._title = schemaLabelFromPath(path);
	state._structurePreview._needsRebuild = true;
	rebuildStructurePreview(state._structurePreview);
}

static void drawSchemaActions(GuiState& state, const std::string& path)
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

static std::vector<BestResult> loadBestResults(const std::string& bestCsvPath)
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
	const size_t scoreModeIndex = columnIndex(header, "score_mode");
	const size_t scoreStageIndex = columnIndex(header, "score_stage");
	const size_t finalLatencyIndex = columnIndex(header, "score_is_final_latency");
	const size_t effectiveQueriesIndex = columnIndex(header, "effective_queries");

	std::string rowLine;
	while (std::getline(input, rowLine))
	{
		if (rowLine.empty())
			continue;

		const std::vector<std::string> row = splitCsvLine(rowLine);
		BestResult result;
		result._dataset = csvValue(row, datasetIndex);
		result._workload = csvValue(row, workloadIndex);
		result._schemaName = csvValue(row, schemaNameIndex);
		result._schemaPath = csvValue(row, schemaPathIndex);
		result._score = parseDouble(csvValue(row, scoreIndex));
		result._averageLatencyMs = parseDouble(csvValue(row, latencyIndex));
		result._buildTimeMs = parseDouble(csvValue(row, buildIndex));
		result._gpuBuildMs = parseDouble(csvValue(row, gpuBuildIndex));
		result._gpuQueryMs = parseDouble(csvValue(row, gpuQueryIndex));
		result._memoryBytes = parseUint64(csvValue(row, memoryIndex));
		result._candidates = static_cast<size_t>(parseUint64(csvValue(row, candidateIndex)));
		result._backend = csvValue(row, backendIndex);
		result._scoreMode = csvValue(row, scoreModeIndex);
		if (result._scoreMode.empty())
			result._scoreMode = "unknown";
		result._scoreStage = csvValue(row, scoreStageIndex);
		if (result._scoreStage.empty())
			result._scoreStage = "unknown";
		result._scoreIsFinalLatency = parseUint64(csvValue(row, finalLatencyIndex)) != 0;
		result._effectiveQueries = static_cast<size_t>(parseUint64(csvValue(row, effectiveQueriesIndex)));
		results.push_back(std::move(result));
	}

	return results;
}

static LiveRankingEntry makeLiveRankingEntry(const Experiments::SchemaSearchRecord& record, size_t order)
{
	LiveRankingEntry entry;
	entry._order = order;
	entry._dataset = record._datasetName;
	entry._workload = record._workloadName;
	entry._schemaName = record._schemaName;
	entry._schemaPath = record._schemaPath;
	entry._backend = record._backend;
	entry._cudaBuilder = record._cudaBuilder;
	entry._score = record._score;
	entry._averageLatencyMs = record._queryMetrics._averageLatencyMs;
	entry._p95LatencyMs = record._queryMetrics._p95LatencyMs;
	entry._buildTimeMs = record._buildMetrics._buildTimeMs;
	entry._gpuBuildMs = record._gpuBuildMs;
	entry._gpuQueryMs = record._gpuQueryMs;
	entry._averageVisitedNodes = record._queryMetrics._averageVisitedNodes;
	entry._averageTestedPoints = record._queryMetrics._averageTestedPoints;
	entry._memoryBytes = static_cast<uint64_t>(record._buildMetrics._memoryEstimateBytes);
	entry._scoreMode = record._scoreMode;
	entry._scoreStage = record._scoreStage;
	entry._scoreIsFinalLatency = record._scoreIsFinalLatency;
	entry._effectiveQueries = record._queryMetrics._totalQueries;
	return entry;
}

static bool sameLiveCandidate(const LiveRankingEntry& entry, const Experiments::SchemaSearchRecord& record)
{
	return entry._dataset == record._datasetName &&
		entry._workload == record._workloadName &&
		entry._schemaName == record._schemaName &&
		entry._schemaPath == record._schemaPath;
}

static void sortAndTrimLiveRanking(std::vector<LiveRankingEntry>& ranking)
{
	std::sort(ranking.begin(), ranking.end(), [](const LiveRankingEntry& left, const LiveRankingEntry& right) {
		if (left._score != right._score)
			return left._score < right._score;
		return left._order < right._order;
	});

	constexpr size_t MaxLiveRankingRows = 128;
	if (ranking.size() > MaxLiveRankingRows)
		ranking.resize(MaxLiveRankingRows);
}

static void updateLiveRanking(RunSession& session, const Experiments::SchemaSearchRecord& record)
{
	std::lock_guard<std::mutex> lock(session._mutex);
	const size_t order = ++session._evaluatedCandidates;
	const auto existing = std::find_if(session._liveRanking.begin(), session._liveRanking.end(), [&record](const LiveRankingEntry& entry) {
		return sameLiveCandidate(entry, record);
	});

	if (existing == session._liveRanking.end())
	{
		session._liveRanking.push_back(makeLiveRankingEntry(record, order));
	}
	else if (record._score < existing->_score)
	{
		*existing = makeLiveRankingEntry(record, order);
	}

	sortAndTrimLiveRanking(session._liveRanking);
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
		std::lock_guard<std::mutex> lock(_session._mutex);
		_session._log.push_back(c);
		return value;
	}

	std::streamsize xsputn(const char* text, std::streamsize count) override
	{
		std::lock_guard<std::mutex> lock(_session._mutex);
		_session._log.append(text, static_cast<size_t>(count));
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

static std::vector<SchemaEntry> discoverSchemas()
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
		entry._label = schemaLabelFromPath(path);
		entry._path = resolved;
		entry._selected = path.find("adaptive") == std::string::npos;
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
				return entry._path == path || std::filesystem::path(entry._path).lexically_normal() == std::filesystem::path(path).lexically_normal();
			});
			if (existing != entries.end())
				continue;

			SchemaEntry entry;
			entry._label = schemaLabelFromPath(path);
			entry._path = path;
			entry._selected = false;
			entries.push_back(std::move(entry));
		}
	}

	return entries;
}

static std::vector<WorkloadEntry> discoverWorkloads()
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
		entry._label = schemaLabelFromPath(path);
		entry._path = resolved;
		entries.push_back(std::move(entry));
	}

	return entries;
}

static int workloadIndexForToken(const GuiState& state, const std::string& token)
{
	for (size_t i = 0; i < state._workloads.size(); ++i)
	{
		const std::string haystack = state._workloads[i]._path + "|" + state._workloads[i]._label;
		if (haystack.find(token) != std::string::npos)
			return static_cast<int>(i);
	}
	return -1;
}

static void applyPublicationDefaults(GuiState& state)
{
	state._autoConditions = true;
	state._includeSynthetic = false;
	state._useBinaryCache = true;
	state._rebuildBinaryCache = false;
	state._generateSchemas = true;
	state._generatedOnly = true;
	state._generatedConditional = true;
	state._generatedAdaptiveLeafCapacity = false;
	state._useRankModel = false;
	// Reset to struct defaults: GA, rung schedule, and threshold refinement on.
	state._optimizeSchemas = true;
	state._useRungSchedule = true;
	state._refineThresholds = true;
	state._repairMutations = true;
	state._repairTopK = 4;
	state._repairPerCandidate = 2;
	state._evaluator = 1;
	state._cudaDevice = 0;
	state._cudaBuilder = 8;
	state._cudaKnnBackend = 0;
	state._cudaQueryBatch = 0;
	state._cudaMemoryBudgetMb = 0;
	state._queryCount = 64;
	state._knnK = 16;
	state._querySeed = 1337;
	state._generatedCount = 256;
	state._benchmarkTopK = 32;
	state._generatedMinBlocks = 2;
	state._generatedMaxBlocks = 3;
	state._generatedMaxDepth = 12;
	state._generatedMinLeaf = 32;
	state._generatedMaxLeaf = 32768;
	state._generatedSeed = 1337;
	state._generatedConditionProbability = 0.75f;
	state._generatedAdaptiveLeafProbability = 0.25f;
	state._conditionProxyCandidates = 256;
	state._conditionProxyPoints = 262144;
	state._conditionProxyQueries = 8;
	state._conditionFinalTopK = 16;
	state._conditionConfirmTopK = 4;
	state._scoreBuildWeight = 0.0f;
	state._scoreMemoryWeight = 0.0f;
	state._scoreImbalanceWeight = 0.0f;
	const int volumeIndex = workloadIndexForToken(state, "volume_small_medium");
	if (volumeIndex >= 0)
		state._selectedWorkload = volumeIndex;
}

static void initializeState(GuiState& state)
{
	setText(state._inputPath, "C:/Datasets/points/Alhambra_100M.las");
	const std::string rankModel = resolvePath("models/schema_selector.json");
	setText(state._rankModelPath, rankModel);
	setText(state._csvPath, projectPath("results/gui_schema_search.csv"));
	setText(state._bestCsvPath, projectPath("results/gui_schema_search_best.csv"));
	setText(state._paretoCsvPath, projectPath("results/gui_schema_search_pareto.csv"));
	setText(state._explainReportPath, projectPath("results/gui_schema_explain.md"));
	setText(state._queryTracePath, "");
	setText(state._generatedSchemaDir, projectPath("results/generated_schemas"));
	setText(state._autoConditionSchemaDir, projectPath("results/auto_conditions"));
	setText(state._selectorOutputPath, projectPath("models/local_schema_selector.json"));
	setText(state._scoreCachePath, projectPath("results/gui_score_cache.jsonl"));
	state._schemas = discoverSchemas();
	state._workloads = discoverWorkloads();
	applyPublicationDefaults(state);
}

static void applyTheme()
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

static void drawSectionTitle(const char* label)
{
	ImGui::Spacing();
	ImGui::TextUnformatted(label);
	ImGui::Separator();
}

static void drawHelpMarker(const char* description)
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

static void drawPathInput(const char* label, std::array<char, TextBufferSize>& value, const char* help = nullptr)
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

static void clampState(GuiState& state)
{
	state._queryCount = std::max(1, state._queryCount);
	state._knnK = std::max(1, state._knnK);
	state._syntheticScale = std::max(1, state._syntheticScale);
	state._generatedCount = std::max(0, state._generatedCount);
	state._benchmarkTopK = std::max(0, state._benchmarkTopK);
	state._generatedMinBlocks = std::max(1, state._generatedMinBlocks);
	state._generatedMaxBlocks = std::max(state._generatedMinBlocks, state._generatedMaxBlocks);
	state._generatedMaxDepth = std::max(state._generatedMinBlocks, state._generatedMaxDepth);
	state._generatedMinLeaf = std::max(1, state._generatedMinLeaf);
	state._generatedMaxLeaf = std::max(state._generatedMinLeaf, state._generatedMaxLeaf);
	state._optimizerGenerations = std::max(0, state._optimizerGenerations);
	state._optimizerPopulation = std::max(1, state._optimizerPopulation);
	state._optimizerElites = std::max(1, state._optimizerElites);
	state._evaluator = std::clamp(state._evaluator, 0, 1);
	state._cudaDevice = std::max(0, state._cudaDevice);
	state._cudaBuilder = std::clamp(state._cudaBuilder, 0, 8);
	state._cudaKnnBackend = std::clamp(state._cudaKnnBackend, 0, 2);
	state._cudaQueryBatch = std::max(0, state._cudaQueryBatch);
	state._cudaMemoryBudgetMb = std::max(0, state._cudaMemoryBudgetMb);
	state._liveRankingTopN = std::clamp(state._liveRankingTopN, 1, 100);
	state._conditionProxyCandidates = std::max(1, state._conditionProxyCandidates);
	state._conditionProxyPoints = std::max(1, state._conditionProxyPoints);
	state._conditionProxyQueries = std::max(1, state._conditionProxyQueries);
	state._conditionFinalTopK = std::max(1, state._conditionFinalTopK);
	state._conditionConfirmTopK = std::max(1, std::min(state._conditionConfirmTopK, state._conditionFinalTopK));
	state._generatedConditionProbability = std::clamp(state._generatedConditionProbability, 0.0f, 1.0f);
	state._generatedAdaptiveLeafProbability = std::clamp(state._generatedAdaptiveLeafProbability, 0.0f, 1.0f);
	state._optimizerMutationRate = std::clamp(state._optimizerMutationRate, 0.0f, 1.0f);
	state._optimizerRandomFraction = std::clamp(state._optimizerRandomFraction, 0.0f, 1.0f);
	state._repairTopK = std::max(1, state._repairTopK);
	state._repairPerCandidate = std::max(1, state._repairPerCandidate);
	state._scoreBuildWeight = std::max(0.0f, state._scoreBuildWeight);
	state._scoreMemoryWeight = std::max(0.0f, state._scoreMemoryWeight);
	state._scoreImbalanceWeight = std::max(0.0f, state._scoreImbalanceWeight);
	if (state._selectedWorkload < 0 || state._selectedWorkload >= static_cast<int>(state._workloads.size()))
		state._selectedWorkload = 0;
}

static std::optional<std::string> buildSearchOptions(const GuiState& state, Experiments::SchemaSearchOptions& options)
{
	options = Experiments::SchemaSearchOptions();
	options._pauseAtEnd = false;
	options._useBinaryCache = state._useBinaryCache;
	options._rebuildBinaryCache = state._rebuildBinaryCache;
	options._includeSyntheticDatasets = state._includeSynthetic;
	options._syntheticScale = static_cast<size_t>(state._syntheticScale);
	options._queryCountOverride = static_cast<size_t>(state._queryCount);
	options._knnKOverride = static_cast<size_t>(state._knnK);
	options._querySeed = static_cast<uint32_t>(state._querySeed);
	options._querySeedOverride = true;
	options._csvPath = textValue(state._csvPath);
	options._bestCsvPath = textValue(state._bestCsvPath);
	options._paretoCsvPath = textValue(state._paretoCsvPath);
	options._explainReportPath = textValue(state._explainReportPath);
	options._queryTracePath = textValue(state._queryTracePath);

	const std::string inputPath = resolvePath(textValue(state._inputPath));
	if (!inputPath.empty())
		options._inputPaths.push_back(inputPath);

	if (options._inputPaths.empty() && !options._includeSyntheticDatasets)
		return "Select a point cloud input or enable synthetic datasets.";

	if (state._workloads.empty())
		return "No workload profiles were found under configs/workloads.";

	options._workloadPaths.push_back(state._workloads[static_cast<size_t>(state._selectedWorkload)]._path);

	if (!state._generatedOnly)
	{
		for (const SchemaEntry& entry : state._schemas)
		{
			if (entry._selected)
				options._schemaPaths.push_back(entry._path);
		}
	}
	options._includeConfiguredSchemas = !state._generatedOnly;

	if (state._generateSchemas)
	{
		options._generation._count = static_cast<size_t>(state._generatedCount);
		options._generation._minBlocks = static_cast<size_t>(state._generatedMinBlocks);
		options._generation._maxBlocks = static_cast<size_t>(state._generatedMaxBlocks);
		options._generation._maxDepth = static_cast<size_t>(state._generatedMaxDepth);
		options._generation._minLeafCapacity = static_cast<size_t>(state._generatedMinLeaf);
		options._generation._maxLeafCapacity = static_cast<size_t>(state._generatedMaxLeaf);
		options._generation._conditionalLevels = state._generatedConditional;
		options._generation._conditionalProbability = static_cast<double>(state._generatedConditionProbability);
		options._generation._adaptiveLeafCapacity = state._generatedAdaptiveLeafCapacity && state._evaluator == 0;
		options._generation._adaptiveLeafProbability = static_cast<double>(state._generatedAdaptiveLeafProbability);
		options._generation._seed = static_cast<uint32_t>(state._generatedSeed);
		options._generation._outputDirectory = textValue(state._generatedSchemaDir);
		options._generation._primitiveProfile = state._queryMinimalPrimitives
			? std::string("query_minimal_cpu")
			: (state._evaluator == 1 ? std::string("cuda_query_full") : std::string("all"));
	}

	if (state._generatedOnly && options._generation._count == 0)
		return "Generated-only search needs a generated schema count greater than zero.";

	if (!state._generatedOnly && options._schemaPaths.empty() && options._generation._count == 0)
		return "Select at least one fixed schema or enable generated schemas.";

	options._autoConditions._enabled = state._autoConditions;
	options._autoConditions._proxyCandidateCount = static_cast<size_t>(state._conditionProxyCandidates);
	options._autoConditions._proxyPointCap = static_cast<size_t>(state._conditionProxyPoints);
	options._autoConditions._proxyQueryCount = static_cast<size_t>(state._conditionProxyQueries);
	options._autoConditions._finalTopK = static_cast<size_t>(state._conditionFinalTopK);
	options._autoConditions._confirmationTopK = static_cast<size_t>(state._conditionConfirmTopK);
	options._autoConditions._outputDirectory = textValue(state._autoConditionSchemaDir);
	options._autoConditions._selectorOutputPath = textValue(state._selectorOutputPath);
	if (options._autoConditions._enabled)
	{
		options._generation._conditionalLevels = true;
		options._generation._conditionalProbability = std::max(options._generation._conditionalProbability, 0.75);
		if (!state._generateSchemas && options._schemaPaths.empty())
			return "Auto-condition tuning needs generated schemas or at least one selected fixed schema.";
	}

	options._benchmarkTopK = static_cast<size_t>(state._benchmarkTopK);
	if (state._useRankModel)
		options._rankModelPath = resolvePath(textValue(state._rankModelPath));

	options._weights._lambdaBuild = state._scoreBuildWeight;
	options._weights._lambdaMemory = state._scoreMemoryWeight;
	options._weights._lambdaImbalance = state._scoreImbalanceWeight;
	options._scoreWeightsOverride =
		options._weights._lambdaBuild > 0.0 ||
		options._weights._lambdaMemory > 0.0 ||
		options._weights._lambdaImbalance > 0.0;
	if (state._scoreCacheEnabled)
	{
		options._scoreCachePath = textValue(state._scoreCachePath);
		options._rebuildScoreCache = state._rebuildScoreCache;
	}
	else
	{
		options._scoreCachePath.clear();
	}
	options._parallelDispatch = std::max(1, state._parallelDispatch);
	options._includeBaselineSchemas = state._includeBaselineSchemas;
	options._evolution._enabled = state._optimizeSchemas;
	options._evolution._generations = static_cast<size_t>(state._optimizerGenerations);
	options._evolution._populationSize = static_cast<size_t>(state._optimizerPopulation);
	options._evolution._eliteCount = static_cast<size_t>(state._optimizerElites);
	options._evolution._seed = static_cast<uint32_t>(state._optimizerSeed);
	options._evolution._mutationRate = static_cast<double>(state._optimizerMutationRate);
	options._evolution._randomImmigrationRate = static_cast<double>(state._optimizerRandomFraction);
	options._evolution._crossoverRate = static_cast<double>(state._optimizerCrossoverRate);
	options._evolution._useNsga2Ranking = state._optimizerUseNsga2;
	options._evolution._repairMutations = state._optimizeSchemas && state._repairMutations;
	options._evolution._repairTopK = static_cast<size_t>(std::max(1, state._repairTopK));
	options._evolution._repairPerCandidate = static_cast<size_t>(std::max(1, state._repairPerCandidate));

	if (state._optimizeSchemas && state._useRungSchedule)
	{
		Experiments::RungSpec proxy;
		proxy._name = "proxy";
		proxy._queryCountOverride = static_cast<size_t>(std::max(1, state._rungProxyQueries));
		proxy._useVisitProxy = true;
		proxy._visitProxyAlpha = static_cast<double>(state._rungProxyAlpha);
		proxy._advanceTopK = static_cast<size_t>(std::max(1, state._rungProxyAdvance));

		Experiments::RungSpec full;
		full._name = "full";
		full._queryCountOverride = static_cast<size_t>(std::max(1, state._rungFullQueries));
		full._useVisitProxy = false;
		full._advanceTopK = static_cast<size_t>(std::max(1, state._rungFullAdvance));

		Experiments::RungSpec confirm;
		confirm._name = "confirm";
		confirm._queryCountOverride = static_cast<size_t>(std::max(1, state._rungConfirmQueries));
		confirm._useVisitProxy = false;
		confirm._advanceTopK = 0;

		options._evolution._rungSchedule._rungs = { proxy, full, confirm };
		options._evolution._rungSchedule._surrogateModelPath = resolvePath(textValue(state.rungSurrogatePath));
		options._evolution._rungSchedule._surrogateCandidatePool = static_cast<size_t>(std::max(0, state._rungSurrogatePool));
		options._evolution._rungSchedule._surrogateProposalsPerStep = static_cast<size_t>(std::max(0, state._rungSurrogateTop));
	}
	else
	{
		options._evolution._rungSchedule = Experiments::RungSchedule{};
	}

	if (state._optimizeSchemas && state._refineThresholds)
	{
		options._evolution._refineThresholds = true;
		options._evolution._refineThresholdsTopK = static_cast<size_t>(std::max(1, state._refineThresholdsTopK));
		options._evolution._refineThresholdsEvaluations = static_cast<size_t>(std::max(1, state._refineThresholdsEvals));
		options._evolution._refineThresholdsSigma0 = static_cast<double>(state._refineThresholdsSigma);
		options._evolution._refineThresholdsSeed = static_cast<uint32_t>(state._refineThresholdsSeed);
	}
	else
	{
		options._evolution._refineThresholds = false;
	}

	if (state._confirmSeeds && state._confirmSeedsCount >= 2)
	{
		options._confirmSeeds = static_cast<size_t>(state._confirmSeedsCount);
		options._confirmTopK = static_cast<size_t>(std::max(1, state._confirmTopK));
	}
	else
	{
		options._confirmSeeds = 0;
	}
	options._evaluator = state._evaluator == 1 ? "cuda" : "cpu";
	options._cuda._device = state._cudaDevice;
	if (state._cudaBuilder == 1)
		options._cuda._builder = "kdtree";
	else if (state._cudaBuilder == 2)
		options._cuda._builder = "bih";
	else if (state._cudaBuilder == 3)
		options._cuda._builder = "octree";
	else if (state._cudaBuilder == 4)
		options._cuda._builder = "karras_octree";
	else if (state._cudaBuilder == 5)
		options._cuda._builder = "quadtree";
	else if (state._cudaBuilder == 6)
		options._cuda._builder = "regular_grid";
	else if (state._cudaBuilder == 7)
		options._cuda._builder = "hgrid";
	else if (state._cudaBuilder == 8)
		options._cuda._builder = "mixed";
	else
		options._cuda._builder = "lbvh";
	static const char* knnBackends[] = { "auto", "gpu_tree_knn", "gpu_bruteforce_knn" };
	options._cuda._knnBackend = knnBackends[state._cudaKnnBackend];
	options._cuda._queryBatchSize = static_cast<size_t>(state._cudaQueryBatch);
	options._cuda._memoryBudgetMb = static_cast<size_t>(state._cudaMemoryBudgetMb);
	return std::nullopt;
}

static void startRun(RunSession& session, const GuiState& state)
{
	if (session._running)
		return;

	if (session._worker.joinable())
		session._worker.join();

	Experiments::SchemaSearchOptions options;
	const std::optional<std::string> validation = buildSearchOptions(state, options);

	{
		std::lock_guard<std::mutex> lock(session._mutex);
		session._error.clear();
		session._log = "Starting optimization...\n";
		session._bestResults.clear();
		session._liveRanking.clear();
		session._evaluatedCandidates = 0;
		session._exitCode = 0;
		if (validation.has_value())
		{
			session._status = "Ready";
			session._error = validation.value();
			return;
		}
		session._status = "Running";
	}

	options.progressCallback = [&session](const Experiments::SchemaSearchRecord& record) {
		updateLiveRanking(session, record);
	};

	const std::string bestCsvPath = options._bestCsvPath;
	session._running = true;
	session._finished = false;
	session._worker = std::thread([&session, options, bestCsvPath]() {
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
			std::lock_guard<std::mutex> lock(session._mutex);
			session._exitCode = code;
			session._error = std::move(errorText);
			session._bestResults = std::move(bestResults);
			session._status = session._error.empty() && code == 0 ? "Complete" : "Failed";
		}
		session._running = false;
		session._finished = true;
	});
}

static void drawPublicationPanel(GuiState& state)
{
	drawSectionTitle("Quick start");
	ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.62f, 0.68f, 0.78f, 1.0f));
	ImGui::TextWrapped("Apply the recommended setup, choose your point cloud under \"1. Inputs\", then press \"Start tuning\" on the right.");
	ImGui::PopStyleColor();

	if (ImGui::Button("Apply recommended setup", ImVec2(-1.0f, 30.0f)))
		applyPublicationDefaults(state);
	drawHelpMarker("CUDA/Mixed, one real cloud, volume workload, generated conditional schemas, staged auto-condition tuning, and query-only score.");

	// Budget summary stays visible so the recommended run's cost is clear without expanding anything.
	if (state._autoConditions)
	{
		ImGui::Text("Budget: %d proxy candidates, %d proxy queries, top %d -> %d",
			state._conditionProxyCandidates,
			state._conditionProxyQueries,
			state._conditionFinalTopK,
			state._conditionConfirmTopK);
	}
	else
	{
		ImGui::Text("Budget: %d generated candidates, top-k %d",
			state._generatedCount,
			state._benchmarkTopK);
	}

	// Toggles tucked away; open only to deviate from the recommended setup.
	if (ImGui::CollapsingHeader("Fine-tune recommended setup"))
	{
		if (ImGui::Checkbox("Per-cloud auto conditions", &state._autoConditions) && state._autoConditions)
		{
			state._generatedConditional = true;
			state._useRankModel = false;
		}
		drawHelpMarker("Estimates cheap point-cloud/node domains, tunes numeric condition thresholds, and writes a reusable measured selector.");
		ImGui::SameLine();
		bool realCloudOnly = !state._includeSynthetic;
		if (ImGui::Checkbox("Real cloud only", &realCloudOnly))
			state._includeSynthetic = !realCloudOnly;
		drawHelpMarker("Keeps the run specialized to the selected cloud instead of mixing in synthetic validation clouds.");

		bool cudaMixed = state._evaluator == 1 && state._cudaBuilder == 8 && state._cudaDevice == 0;
		if (ImGui::Checkbox("CUDA Mixed device 0", &cudaMixed))
		{
			if (cudaMixed)
			{
				state._evaluator = 1;
				state._cudaDevice = 0;
				state._cudaBuilder = 8;
			}
			else
			{
				state._evaluator = 0;
			}
		}
		drawHelpMarker("Uses the schema-aware mixed CUDA builder on device 0. Schema-search still falls back to CPU if CUDA is unavailable.");
		ImGui::SameLine();
		const int volumeIndex = workloadIndexForToken(state, "volume_small_medium");
		bool volumeWorkload = volumeIndex >= 0 && state._selectedWorkload == volumeIndex;
		ImGui::BeginDisabled(volumeIndex < 0);
		if (ImGui::Checkbox("Volume workload", &volumeWorkload) && volumeWorkload)
			state._selectedWorkload = volumeIndex;
		ImGui::EndDisabled();
		drawHelpMarker("Uses the small/medium 3D volume workload, avoiding KNN in the default tuning path.");

		if (ImGui::Checkbox("Generated-only candidates", &state._generatedOnly))
			state._generateSchemas = state._generateSchemas || state._generatedOnly;
		drawHelpMarker("Focuses the measured search on generated schema variants instead of fixed baselines.");
		ImGui::SameLine();
		if (ImGui::Checkbox("Conditional generated blocks", &state._generatedConditional) && state._autoConditions)
			state._generatedConditional = true;
		drawHelpMarker("Keeps local node predicates in the generated schema space.");
	}
}

static void drawDatasetPanel(GuiState& state)
{
	drawSectionTitle("Point Cloud");
	drawPathInput("Point cloud", state._inputPath, "Point cloud to optimize against. With synthetic datasets disabled, the selected best schema is overfit to this one cloud and workload.");
	ImGui::Checkbox("Use binary cache", &state._useBinaryCache);
	drawHelpMarker("Loads and writes the .mdspc cache beside the source point cloud. This speeds repeated runs but does not change the measured query workload.");
	ImGui::SameLine();
	ImGui::Checkbox("Rebuild cache", &state._rebuildBinaryCache);
	drawHelpMarker("Forces a fresh read from the source point cloud and replaces the .mdspc cache. Useful after the input file changes or if the cache looks stale.");
	ImGui::Checkbox("Include synthetic datasets", &state._includeSynthetic);
	drawHelpMarker("Adds built-in synthetic point clouds to the search. Keep this off when you want the optimizer to specialize to your current point cloud only.");
	if (state._includeSynthetic)
	{
		ImGui::InputInt("Synthetic scale", &state._syntheticScale);
		drawHelpMarker("Point count scale for each synthetic dataset. Larger values make synthetic validation more realistic and slower.");
	}

	drawSectionTitle("Workload");
	if (!state._workloads.empty())
	{
		const char* preview = state._workloads[static_cast<size_t>(state._selectedWorkload)]._label.c_str();
		if (ImGui::BeginCombo("Profile", preview))
		{
			for (size_t i = 0; i < state._workloads.size(); ++i)
			{
				const bool selected = static_cast<int>(i) == state._selectedWorkload;
				if (ImGui::Selectable(state._workloads[i]._label.c_str(), selected))
					state._selectedWorkload = static_cast<int>(i);
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
	ImGui::InputInt("Queries", &state._queryCount);
	drawHelpMarker("Number of generated measured queries for the workload. More queries reduce noise but increase optimization time.");
	const bool knnWorkload = !state._workloads.empty() &&
		state._workloads[static_cast<size_t>(state._selectedWorkload)]._path.find("knn") != std::string::npos;
	if (knnWorkload)
	{
		ImGui::InputInt("KNN k", &state._knnK);
		drawHelpMarker("Neighbor count for KNN queries. Larger k usually increases KNN cost and can favor different structures.");
	}
	ImGui::InputInt("Query seed", &state._querySeed);
	drawHelpMarker("Random seed for generated query centers and boxes. Keep fixed for comparable runs; change it to test robustness.");
}

static void drawSchemaPanel(GuiState& state)
{
	drawSectionTitle("Search Space");
	ImGui::Checkbox("Generate schemas", &state._generateSchemas);
	drawHelpMarker("Samples new multi-DS JSON candidates from the configured bounds, then benchmarks them. This expands beyond the fixed schema list.");
	ImGui::SameLine();
	if (ImGui::Checkbox("Conditional blocks", &state._generatedConditional) && state._autoConditions)
		state._generatedConditional = true;
	drawHelpMarker("Allows later schema blocks to activate only for local node conditions such as point count, density, or height ratio. This is the current branch-adaptive multi-DS mechanism.");
	ImGui::SameLine();
	const bool cpuAdaptiveLeafSupported = state._evaluator == 0;
	ImGui::BeginDisabled(!cpuAdaptiveLeafSupported);
	ImGui::Checkbox("Adaptive leaf capacity", &state._generatedAdaptiveLeafCapacity);
	ImGui::EndDisabled();
	drawHelpMarker("CPU-only generated rule that scales leaf capacity per node using density, height-ratio, anisotropy, and query-mix factors. CUDA runs leave this off until GPU support exists.");
	ImGui::SameLine();
	ImGui::Checkbox("Generated only", &state._generatedOnly);
	drawHelpMarker("Ignores checked fixed schemas and searches only generated candidates. Turn this off to compare generated candidates against known baselines.");
	ImGui::Checkbox("Query-minimal primitives", &state._queryMinimalPrimitives);
	drawHelpMarker("Default publication search space for CPU discovery: generated schemas use QuadTree, Octree, KDTree, and BVH only. CUDA-only aliases such as KarrasOctree, LBVH, BIH, RegularGrid, and HGrid stay available for explicit CUDA/full runs.");

	if (state._autoConditions)
	{
		drawSectionTitle("Auto-Condition Budget");
		ImGui::InputInt("Proxy candidates", &state._conditionProxyCandidates);
		drawHelpMarker("Number of domain-aware candidate schemas screened on the downsampled proxy cloud.");
		ImGui::InputInt("Proxy points", &state._conditionProxyPoints);
		drawHelpMarker("Point cap for the proxy stage. The full cloud is still used for shortlist and confirmation stages.");
		ImGui::InputInt("Proxy queries", &state._conditionProxyQueries);
		drawHelpMarker("Prepared workload query count for the first cheap stage.");
		ImGui::InputInt("Shortlist top-k", &state._conditionFinalTopK);
		drawHelpMarker("Candidates promoted from proxy screening to a full-cloud short run.");
		ImGui::InputInt("Confirm top-k", &state._conditionConfirmTopK);
		drawHelpMarker("Candidates promoted from the short full-cloud run to final confirmation with the requested query count.");
		drawPathInput("Tuned schemas", state._autoConditionSchemaDir, "Directory for the final numeric tuned schema JSON files.");
		drawPathInput("Selector", state._selectorOutputPath, "Measured selector artifact used by later --schema auto runs.");
	}
	else
	{
		ImGui::InputInt("Generated count", &state._generatedCount);
		drawHelpMarker("Number of candidate JSON schemas to sample. More candidates explores more of the space and takes longer.");
		ImGui::InputInt("Benchmark top-k", &state._benchmarkTopK);
		drawHelpMarker("If a rank model is enabled, benchmark only the model's top-k candidates. 0 means benchmark every candidate and rely only on measured results.");
	}

	drawPathInput("Generated dir", state._generatedSchemaDir, "Directory where generated schema JSON files are written. The best result table points back to one of these files when a generated candidate wins.");

	if (ImGui::CollapsingHeader("Baseline schemas"))
	{
		if (ImGui::Button("Select all"))
		{
			for (SchemaEntry& entry : state._schemas)
				entry._selected = true;
		}
		ImGui::SameLine();
		if (ImGui::Button("Clear"))
		{
			for (SchemaEntry& entry : state._schemas)
				entry._selected = false;
		}

		if (ImGui::BeginTable("schemas", 4, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
		{
			ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 32.0f);
			ImGui::TableSetupColumn("Schema", ImGuiTableColumnFlags_WidthStretch, 0.32f);
			ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch, 0.58f);
			ImGui::TableSetupColumn("View", ImGuiTableColumnFlags_WidthFixed, 116.0f);
			ImGui::TableHeadersRow();
			for (size_t i = 0; i < state._schemas.size(); ++i)
			{
				SchemaEntry& entry = state._schemas[i];
				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::PushID(static_cast<int>(i));
				ImGui::BeginDisabled(state._generatedOnly);
				ImGui::Checkbox("##selected", &entry._selected);
				ImGui::EndDisabled();
				ImGui::TableSetColumnIndex(1);
				ImGui::TextUnformatted(entry._label.c_str());
				ImGui::TableSetColumnIndex(2);
				ImGui::TextUnformatted(entry._path.c_str());
				ImGui::TableSetColumnIndex(3);
				drawSchemaActions(state, entry._path);
				ImGui::PopID();
			}
			ImGui::EndTable();
		}
	}

	if (ImGui::CollapsingHeader("Advanced generated bounds"))
	{
		ImGui::InputInt("Min blocks", &state._generatedMinBlocks);
		drawHelpMarker("Minimum number of structure blocks in generated schemas. Keep this at 2 for nested/mixed discovery; pure baselines are still included separately as controls.");
		ImGui::InputInt("Max blocks", &state._generatedMaxBlocks);
		drawHelpMarker("Maximum number of nested structure blocks per generated schema, for example quadtree then octree then kdtree.");
		ImGui::InputInt("Max depth", &state._generatedMaxDepth);
		drawHelpMarker("Maximum total tree depth across generated blocks. Larger depth can improve pruning but increases build cost and memory risk.");
		ImGui::InputInt("Min leaf", &state._generatedMinLeaf);
		drawHelpMarker("Smallest generated leaf capacity. Lower values produce deeper/finer trees and usually test fewer points per leaf.");
		ImGui::InputInt("Max leaf", &state._generatedMaxLeaf);
		drawHelpMarker("Largest generated leaf capacity. Higher values produce coarser leaves and can reduce memory/build cost while increasing per-leaf point tests.");
		ImGui::InputInt("Generated seed", &state._generatedSeed);
		drawHelpMarker("Random seed for schema sampling. Keep fixed for repeatability; change it to explore a different batch of candidates.");
		ImGui::SliderFloat("Condition probability", &state._generatedConditionProbability, 0.0f, 1.0f, "%.2f");
		drawHelpMarker("Probability that a generated block gets a local activation condition. Higher values make more branch-adaptive schemas.");
		ImGui::BeginDisabled(!cpuAdaptiveLeafSupported);
		ImGui::SliderFloat("Adaptive leaf probability", &state._generatedAdaptiveLeafProbability, 0.0f, 1.0f, "%.2f");
		ImGui::EndDisabled();
		drawHelpMarker("Probability that a generated level gets per-node adaptive leaf capacity. Applies only to CPU schema discovery.");
		bool allPrimitiveVariants = !state._queryMinimalPrimitives;
		if (ImGui::Checkbox("All CUDA primitive variants", &allPrimitiveVariants))
			state._queryMinimalPrimitives = !allPrimitiveVariants;
		drawHelpMarker("Expands generated schemas to include CUDA-native aliases and CPU-compatible variants. Use this for CUDA confirmation sweeps or exhaustive research, not for the default CPU discovery pass.");
	}

	if (ImGui::CollapsingHeader("Evolutionary optimizer"))
	{
		ImGui::Checkbox("Optimize iteratively", &state._optimizeSchemas);
		drawHelpMarker("Runs evolutionary mutation search after the initial fixed/generated population. Selection is driven by measured C++ benchmark scores, not gradients.");
		ImGui::InputInt("Generations", &state._optimizerGenerations);
		drawHelpMarker("Number of mutation rounds after the initial population is measured. 0 evaluates only the initial population.");
		ImGui::InputInt("Population", &state._optimizerPopulation);
		drawHelpMarker("Number of new candidates evaluated per generation. Larger populations explore more schemas and take longer.");
		ImGui::InputInt("Elites", &state._optimizerElites);
		drawHelpMarker("Best measured candidates used as parents for the next generation. Too few can get stuck; too many makes search less focused.");
		ImGui::InputInt("Optimizer seed", &state._optimizerSeed);
		drawHelpMarker("Seed for parent choice and mutations. Keep fixed for repeatability; change it for a different search trajectory.");
		ImGui::SliderFloat("Mutation rate", &state._optimizerMutationRate, 0.0f, 1.0f, "%.2f");
		drawHelpMarker("Probability of applying extra edits to a child schema. Higher values make larger jumps in topology/depth/leaf/condition space.");
		ImGui::SliderFloat("Random fraction", &state._optimizerRandomFraction, 0.0f, 1.0f, "%.2f");
		drawHelpMarker("Fraction of each generation filled with fresh random candidates instead of mutations. This preserves exploration.");
		ImGui::SliderFloat("Crossover rate", &state._optimizerCrossoverRate, 0.0f, 1.0f, "%.2f");
		drawHelpMarker("Probability that a child is built by splicing two elite parents' level lists instead of mutating one. Reaches topology combinations neither parent had; 0 disables crossover (pre-B4 behavior).");
		ImGui::Checkbox("NSGA-II elite ranking", &state._optimizerUseNsga2);
		drawHelpMarker("Rank elites by non-dominated-sort + crowding distance over (latency, build, memory, imbalance). Preserves diversity across the Pareto front instead of collapsing it to a scalar score winner.");
		ImGui::Checkbox("Repair mutations", &state._repairMutations);
		drawHelpMarker("Creates targeted children from measured bottlenecks: high leaf occupancy tightens leaves/depth, high tested-points adds local indexing, high visited-nodes coarsens the root, and high full-containment prefers coarse grid/quadtree shapes.");
		ImGui::BeginDisabled(!state._repairMutations);
		ImGui::InputInt("Repair top-K", &state._repairTopK);
		drawHelpMarker("Number of measured archive parents to diagnose each generation.");
		ImGui::InputInt("Repairs/parent", &state._repairPerCandidate);
		drawHelpMarker("Maximum targeted repair children created from each diagnosed parent before normal mutation fills the remaining population.");
		ImGui::EndDisabled();

		ImGui::Separator();
		ImGui::Checkbox("Multi-fidelity rungs", &state._useRungSchedule);
		drawHelpMarker("Successive halving: each batch is first scored cheaply with the visit-count surrogate, then the top survivors are re-measured with wall-clock latency at increasing query counts. Lets the optimizer touch more candidates for the same wall-clock.");
		ImGui::BeginDisabled(!state._useRungSchedule);
		ImGui::InputInt("R0 proxy queries", &state._rungProxyQueries);
		drawHelpMarker("Query count for the cheap proxy rung. 4 is plenty since the proxy uses deterministic visit/test counters, not noisy timings.");
		ImGui::InputInt("R0 proxy advance top-K", &state._rungProxyAdvance);
		drawHelpMarker("Number of candidates promoted from the proxy rung to the latency rung. Typically a quarter of the population.");
		ImGui::SliderFloat("R0 proxy alpha", &state._rungProxyAlpha, 0.0f, 1.0f, "%.2f");
		drawHelpMarker("Weight applied to averageTestedPoints in the proxy score (visitedNodes + alpha * testedPoints). Raise it to penalise schemas that touch many points per visited node.");
		ImGui::InputInt("R1 full queries", &state._rungFullQueries);
		drawHelpMarker("Query count for the mid-fidelity latency rung. 16 keeps measurement cheap while still discriminating real winners.");
		ImGui::InputInt("R1 full advance top-K", &state._rungFullAdvance);
		drawHelpMarker("Number of candidates promoted from the latency rung to the confirmation rung. 4-8 is a normal publication setting.");
		ImGui::InputInt("R2 confirm queries", &state._rungConfirmQueries);
		drawHelpMarker("Query count for the confirmation rung. The optimizer only writes records from this final rung to the CSV.");
		drawPathInput("Surrogate model", state.rungSurrogatePath, "Optional. Path to an exported linear/ONNX selector JSON. When set, between rungs the surrogate proposes top-K candidates from a random pool, injected as extra immigrants.");
		ImGui::InputInt("Surrogate pool", &state._rungSurrogatePool);
		drawHelpMarker("Number of fresh genomes sampled per generation and scored by the surrogate. 0 disables the acquisition step.");
		ImGui::InputInt("Surrogate top-K", &state._rungSurrogateTop);
		drawHelpMarker("Number of surrogate-ranked candidates injected as additional immigrants each generation. Cheap to raise; the proxy rung filters them anyway.");
		ImGui::EndDisabled();

		ImGui::Separator();
		ImGui::Checkbox("Refine thresholds (CMA-style ES)", &state._refineThresholds);
		drawHelpMarker("After the GA finishes, the top-K archive entries with conditional levels get their numeric thresholds tuned by a (1+lambda)-ES under the visit-proxy. Survivors that improved are re-measured at full fidelity.");
		ImGui::BeginDisabled(!state._refineThresholds);
		ImGui::InputInt("Refine top-K", &state._refineThresholdsTopK);
		drawHelpMarker("Number of GA archive survivors that get their thresholds refined. 4 is a normal publication setting.");
		ImGui::InputInt("Refine evals/cand", &state._refineThresholdsEvals);
		drawHelpMarker("Total evaluation budget per refined candidate. 40-80 is the regime where the (1+lambda)-ES converges without overspending.");
		ImGui::SliderFloat("Refine sigma0", &state._refineThresholdsSigma, 0.01f, 0.8f, "%.2f");
		drawHelpMarker("Initial step size in normalised [0,1] threshold space. 0.3 is a good default; larger values explore wider, smaller values polish.");
		ImGui::InputInt("Refine seed", &state._refineThresholdsSeed);
		drawHelpMarker("RNG seed for the refiner. Different seeds produce different threshold trajectories.");
		ImGui::EndDisabled();

		ImGui::Separator();
		ImGui::Checkbox("Multi-seed confirmation", &state._confirmSeeds);
		drawHelpMarker("After the main run, the top-K per (dataset, workload) is re-measured with N distinct query seeds. Records pick up a seed-averaged mean and 95% bootstrap CI on avg latency, p95 latency, and GPU build time; the Pareto step uses the mean so noise can't fake a win.");
		ImGui::BeginDisabled(!state._confirmSeeds);
		ImGui::InputInt("Confirm seeds", &state._confirmSeedsCount);
		drawHelpMarker("Number of distinct query seeds to draw. 5 is the publication default; lower it for quick iteration, raise it for tighter CIs.");
		ImGui::InputInt("Confirm top-K", &state._confirmTopK);
		drawHelpMarker("Number of top-by-score candidates per (dataset, workload) to re-measure. 4 keeps the extra cost bounded while still covering the Pareto front entries in most setups.");
		ImGui::EndDisabled();
	}
}

static void drawScoringPanel(GuiState& state)
{
	drawSectionTitle("Evaluator");
	const char* evaluators[] = { "CPU", "CUDA" };
	ImGui::Combo("Backend", &state._evaluator, evaluators, IM_ARRAYSIZE(evaluators));
	drawHelpMarker("Chooses where measured schema fitness runs. CUDA builds the selected GPU structure and measures range/count/radius/KNN queries there.");
	if (state._evaluator == 1)
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
		ImGui::Text("CUDA: device %d, %s", state._cudaDevice, builders[state._cudaBuilder]);
		if (ImGui::CollapsingHeader("Advanced CUDA"))
		{
			ImGui::InputInt("CUDA device", &state._cudaDevice);
			drawHelpMarker("GPU id passed to cudaSetDevice. Use 0 unless you have several CUDA GPUs.");
			ImGui::Combo("Structure", &state._cudaBuilder, builders, IM_ARRAYSIZE(builders));
			drawHelpMarker("LBVH, KDTree, BIH, Octree, KarrasOctree, QuadTree, RegularGrid, HGrid, and MixedTree schemas are implemented. KarrasOctree uses Morton sorting and prefix child ranges; BIH is a binary interval hierarchy with tight child bounds; standalone HGrid builds several RegularGrid levels and chooses one per query; Mixed follows the schema's per-depth structure schedule, including RegularGrid and HGrid grid split levels.");
			const char* knnBackends[] = { "Auto", "Tree KNN", "Bruteforce scan" };
			ImGui::Combo("KNN backend", &state._cudaKnnBackend, knnBackends, IM_ARRAYSIZE(knnBackends));
			drawHelpMarker("Auto uses exact KDTree/BIH tree KNN for k<=16 and brute-force GPU scan elsewhere. Tree KNN returns hit IDs for diagnostics and tests.");
			ImGui::InputInt("Query batch", &state._cudaQueryBatch);
			drawHelpMarker("Number of CUDA queries uploaded/launched per batch. 0 runs the whole generated workload as one batch.");
			ImGui::InputInt("Memory budget MB", &state._cudaMemoryBudgetMb);
			drawHelpMarker("Optional guardrail that rejects CUDA builds whose point buffers, sorted arrays, nodes, and sort scratch exceed this budget.");
		}
	}

	drawSectionTitle("Surrogate");
	if (state._autoConditions)
		ImGui::TextDisabled("Bypassed while per-cloud auto-condition tuning is enabled.");
	ImGui::BeginDisabled(state._autoConditions);
	ImGui::Checkbox("Use rank model", &state._useRankModel);
	drawHelpMarker("Uses the exported JSON/ONNX selector only to rank/prune candidates before benchmarking. The final best schema still comes from measured C++ timings.");
	drawPathInput("Rank model", state._rankModelPath, "Path to a selector wrapper JSON, usually models/schema_selector.json or models/schema_selector_onnx.json.");
	ImGui::EndDisabled();

	drawSectionTitle("Score Weights");
	if (state._optimizeSchemas && state._useRungSchedule)
		ImGui::TextDisabled("GA proxy rungs only promote candidates; CSV/best tables use the final latency rung.");
	else if (state._scoreBuildWeight == 0.0f && state._scoreMemoryWeight == 0.0f && state._scoreImbalanceWeight == 0.0f)
		ImGui::TextDisabled("Score is query latency only.");
	ImGui::Text("Build %.4f, memory %.4f, imbalance %.4f",
		state._scoreBuildWeight,
		state._scoreMemoryWeight,
		state._scoreImbalanceWeight);
	if (ImGui::CollapsingHeader("Advanced scoring"))
	{
		ImGui::InputFloat("Build", &state._scoreBuildWeight, 0.001f, 0.01f, "%.4f");
		drawHelpMarker("Adds build time into the score. Keep at 0 if you only care about saved/reused structures and query speed.");
		ImGui::InputFloat("Memory", &state._scoreMemoryWeight, 0.001f, 0.01f, "%.4f");
		drawHelpMarker("Adds a memory penalty. Increase this if two schemas have similar query time but one is much larger.");
		ImGui::InputFloat("Imbalance", &state._scoreImbalanceWeight, 0.001f, 0.01f, "%.4f");
		drawHelpMarker("Adds a penalty for uneven leaf occupancy. Increase this if winners have pathological leaves or unstable query behavior.");
	}

	drawSectionTitle("Outputs");
	drawPathInput("CSV", state._csvPath, "Full measured result table, with one row per dataset/workload/schema candidate.");
	drawPathInput("Best CSV", state._bestCsvPath, "Compact winner table. This is what the GUI reads back to populate Best Results.");
	drawPathInput("Pareto CSV", state._paretoCsvPath, "Non-dominated front over (avg latency, build time, memory, imbalance) per (dataset, workload). Empty path disables.");
	drawPathInput("Explain report", state._explainReportPath, "Markdown explanation report with schema chain, active-structure fractions, query-family behavior against baselines, and repair-style diagnosis. Empty path disables.");
	drawPathInput("Query trace", state._queryTracePath, "Optional per-query CSV trace. Leave empty for normal runs; enable for noisy/outlier audits. Score-cache hits are bypassed while tracing.");

	drawSectionTitle("Speed-ups");
	ImGui::TextWrapped(
		"Auto-conditions runs three stages: proxy (256 candidates on a tiny downsampled cloud, "
		"ranked by deterministic visit-counts), shortlist (top 16 rebuilt on the full cloud with "
		"a short query set), confirmation (top 4 with the full query set). The score cache "
		"survives all stages and across runs, so a re-run after a crash or interruption resumes "
		"where it left off.");
	ImGui::Checkbox("Persistent score cache", &state._scoreCacheEnabled);
	drawHelpMarker("Memoises (schema, cloud, workload) -> measured score. If you abort a run mid-shortlist, the next run skips everything already measured. Designed for safe restarts on stalled candidates like deep hgrid configs.");
	if (state._scoreCacheEnabled)
	{
		drawPathInput("Cache JSONL", state._scoreCachePath, "Append-only JSONL file. Delete it (or tick rebuild) to invalidate.");
		ImGui::Checkbox("Rebuild cache this run", &state._rebuildScoreCache);
		drawHelpMarker("Deletes the cache file before this run so every candidate is measured fresh.");
	}
	ImGui::SliderInt("Parallel CPU workers", &state._parallelDispatch, 1, 32);
	drawHelpMarker("Honored only when the evaluator is CPU. Parallelises the evolutionary GA loop AND the auto-condition proxy/shortlist/confirmation loops. The CUDA path stays serial: GPU state and the per-builder build cache are not thread-safe.");
	if (state._evaluator == 1 && state._parallelDispatch > 1)
		ImGui::TextDisabled("(currently using CUDA: workers are ignored)");

	ImGui::Checkbox("Include baseline data structures", &state._includeBaselineSchemas);
	drawHelpMarker("Adds canonical single-block controls. CPU discovery uses only distinct query families (QuadTree, Octree, KDTree, BVH); CUDA/full runs also include LBVH, KarrasOctree, RegularGrid, HGrid, and BIH.");
}

static bool isCudaResult(const std::string& backend)
{
	return backend == "cuda" || backend == "gpu";
}

static void drawResultsTable(const std::vector<BestResult>& results, GuiState& state)
{
	if (results.empty())
	{
		ImGui::TextUnformatted("No completed results yet.");
		return;
	}

	if (ImGui::BeginTable("best-results", 13, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp))
	{
		ImGui::TableSetupColumn("Dataset");
		ImGui::TableSetupColumn("Workload");
		ImGui::TableSetupColumn("Best schema");
		ImGui::TableSetupColumn("Backend", ImGuiTableColumnFlags_WidthFixed, 76.0f);
		ImGui::TableSetupColumn("Mode", ImGuiTableColumnFlags_WidthFixed, 92.0f);
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
			ImGui::TextUnformatted(result._dataset.c_str());
			ImGui::TableSetColumnIndex(1);
			ImGui::TextUnformatted(result._workload.c_str());
			ImGui::TableSetColumnIndex(2);
			ImGui::TextUnformatted(result._schemaName.c_str());
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", result._schemaPath.c_str());
			ImGui::TableSetColumnIndex(3);
			ImGui::TextUnformatted(result._backend.empty() ? "cpu" : result._backend.c_str());
			ImGui::TableSetColumnIndex(4);
			ImGui::Text("%s/%s", result._scoreMode.c_str(), result._scoreStage.c_str());
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("%s score, %zu effective queries%s",
					result._scoreMode.c_str(),
					result._effectiveQueries,
					result._scoreIsFinalLatency ? ", final latency" : ", not final latency");
			ImGui::TableSetColumnIndex(5);
			ImGui::Text("%.3f", result._score);
			ImGui::TableSetColumnIndex(6);
			ImGui::Text("%.3f", result._averageLatencyMs);
			ImGui::TableSetColumnIndex(7);
			ImGui::Text("%.1f", result._buildTimeMs);
			ImGui::TableSetColumnIndex(8);
			if (isCudaResult(result._backend))
				ImGui::Text("%.1f", result._gpuBuildMs);
			else
				ImGui::TextUnformatted("-");
			ImGui::TableSetColumnIndex(9);
			if (isCudaResult(result._backend))
				ImGui::Text("%.1f", result._gpuQueryMs);
			else
				ImGui::TextUnformatted("-");
			ImGui::TableSetColumnIndex(10);
			ImGui::Text("%.1f", static_cast<double>(result._memoryBytes) / (1024.0 * 1024.0));
			ImGui::TableSetColumnIndex(11);
			ImGui::Text("%zu", result._candidates);
			ImGui::TableSetColumnIndex(12);
			ImGui::PushID(static_cast<int>(i));
			drawSchemaActions(state, result._schemaPath);
			ImGui::PopID();
		}

		ImGui::EndTable();
	}
}

// Lineage classification parsed from the generator's schema naming convention.
enum class LineageOperator
{
	Baseline = 0,
	Generated = 1,
	Mutation = 2,
	Crossover = 3,
	Refined = 4,
	Other = 5,
};

struct LineageInfo
{
	LineageOperator _op = LineageOperator::Other;
	int _generation = 0;   // -1 = not part of a generation (e.g. refiner inner loop)
};

static const char* lineageOperatorLabel(LineageOperator op)
{
	switch (op)
	{
	case LineageOperator::Baseline: return "baseline";
	case LineageOperator::Generated: return "generated";
	case LineageOperator::Mutation: return "mutation";
	case LineageOperator::Crossover: return "crossover";
	case LineageOperator::Refined: return "refined";
	default: return "other";
	}
}

static ImU32 lineageOperatorColor(LineageOperator op)
{
	switch (op)
	{
	case LineageOperator::Baseline:  return IM_COL32(150, 150, 150, 255);
	case LineageOperator::Generated: return IM_COL32(120, 160, 200, 255);
	case LineageOperator::Mutation:  return IM_COL32(120, 200, 130, 255);
	case LineageOperator::Crossover: return IM_COL32(200, 140, 200, 255);
	case LineageOperator::Refined:   return IM_COL32(220, 180, 100, 255);
	default: return IM_COL32(180, 180, 180, 255);
	}
}

static bool schemaNameStartsWith(const std::string& schemaName, const char* prefix)
{
	const size_t len = std::strlen(prefix);
	return schemaName.size() >= len && schemaName.compare(0, len, prefix) == 0;
}

// Extract the generation index from a `..._g<N>_...` pattern. Returns -1 if absent.
static int extractGenerationIndex(const std::string& schemaName, const char* prefix)
{
	const size_t prefixLen = std::strlen(prefix);
	if (schemaName.size() <= prefixLen)
		return -1;
	size_t cursor = prefixLen;
	int value = 0;
	bool any = false;
	while (cursor < schemaName.size() && std::isdigit(static_cast<unsigned char>(schemaName[cursor])))
	{
		value = value * 10 + (schemaName[cursor] - '0');
		++cursor;
		any = true;
	}
	return any ? value : -1;
}

static LineageInfo parseLineage(const std::string& schemaName)
{
	LineageInfo info;
	if (schemaNameStartsWith(schemaName, "evolved_g"))
	{
		info._op = LineageOperator::Mutation;
		info._generation = std::max(1, extractGenerationIndex(schemaName, "evolved_g"));
	}
	else if (schemaNameStartsWith(schemaName, "xover_g"))
	{
		info._op = LineageOperator::Crossover;
		info._generation = std::max(1, extractGenerationIndex(schemaName, "xover_g"));
	}
	else if (schemaNameStartsWith(schemaName, "refined_g"))
	{
		info._op = LineageOperator::Refined;
		info._generation = -1;   // threshold-refiner runs post-GA
	}
	else if (schemaNameStartsWith(schemaName, "generated_"))
	{
		info._op = LineageOperator::Generated;
		info._generation = 0;
	}
	else if (schemaName.find("_default") != std::string::npos)
	{
		info._op = LineageOperator::Baseline;
		info._generation = 0;
	}
	else
	{
		info._op = LineageOperator::Other;
		info._generation = 0;
	}
	return info;
}

// Per-generation rollup of the live ranking: operator counts and best candidate.
struct GenerationBucket
{
	int _generation = -1;
	size_t _total = 0;
	std::array<size_t, 6> operatorCounts{};   // indexed by LineageOperator
	const LiveRankingEntry* _best = nullptr;
	LineageOperator _bestOp = LineageOperator::Other;
};

static std::vector<GenerationBucket> bucketByGeneration(const std::vector<LiveRankingEntry>& ranking)
{
	std::map<int, GenerationBucket> buckets;
	for (const LiveRankingEntry& entry : ranking)
	{
		const LineageInfo info = parseLineage(entry._schemaName);
		GenerationBucket& bucket = buckets[info._generation];
		bucket._generation = info._generation;
		++bucket._total;
		++bucket.operatorCounts[static_cast<size_t>(info._op)];
		if (!bucket._best || entry._score < bucket._best->_score)
		{
			bucket._best = &entry;
			bucket._bestOp = info._op;
		}
	}
	std::vector<GenerationBucket> out;
	out.reserve(buckets.size());
	for (auto& [gen, bucket] : buckets)
		out.push_back(std::move(bucket));
	std::sort(out.begin(), out.end(), [](const GenerationBucket& a, const GenerationBucket& b) {
		// Refiner candidates (-1) go last; otherwise ascending generation index.
		if (a._generation == -1 && b._generation != -1) return false;
		if (b._generation == -1 && a._generation != -1) return true;
		return a._generation < b._generation;
	});
	return out;
}

// GA-progress panel: one row per generation with operator breakdown and best schema, parsed from schema names.
static void drawGenerationSummary(const std::vector<LiveRankingEntry>& ranking)
{
	if (ranking.empty())
	{
		ImGui::TextUnformatted("Generation summary will appear after the first candidate finishes.");
		return;
	}
	const std::vector<GenerationBucket> buckets = bucketByGeneration(ranking);
	if (buckets.empty())
		return;

	// Track global-best across all generations so we can mark the row with a star.
	const LiveRankingEntry* globalBest = nullptr;
	for (const GenerationBucket& bucket : buckets)
	{
		if (bucket._best && (!globalBest || bucket._best->_score < globalBest->_score))
			globalBest = bucket._best;
	}

	if (ImGui::BeginTable("ga-generation-summary", 6,
		ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp))
	{
		ImGui::TableSetupColumn("Gen", ImGuiTableColumnFlags_WidthFixed, 48.0f);
		ImGui::TableSetupColumn("Population", ImGuiTableColumnFlags_WidthFixed, 88.0f);
		ImGui::TableSetupColumn("Mix");
		ImGui::TableSetupColumn("Gen best", ImGuiTableColumnFlags_WidthFixed, 96.0f);
		ImGui::TableSetupColumn("Best so far", ImGuiTableColumnFlags_WidthFixed, 96.0f);
		ImGui::TableSetupColumn("Best-of-gen schema");
		ImGui::TableHeadersRow();

		// Cumulative best so far; excludes the post-GA refiner bucket from the running min.
		double cumulativeBest = std::numeric_limits<double>::infinity();

		for (const GenerationBucket& bucket : buckets)
		{
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			if (bucket._generation == -1)
				ImGui::TextUnformatted("refine");
			else if (bucket._generation == 0)
				ImGui::TextUnformatted("init");
			else
				ImGui::Text("g%d", bucket._generation);

			ImGui::TableNextColumn();
			ImGui::Text("%zu", bucket._total);

			ImGui::TableNextColumn();
			// Inline operator counts colored by LineageOperator, only non-zero ones.
			bool firstChip = true;
			for (size_t opIdx = 0; opIdx < bucket.operatorCounts.size(); ++opIdx)
			{
				const size_t count = bucket.operatorCounts[opIdx];
				if (count == 0)
					continue;
				if (!firstChip)
					ImGui::SameLine();
				firstChip = false;
				const ImU32 color = lineageOperatorColor(static_cast<LineageOperator>(opIdx));
				ImGui::PushStyleColor(ImGuiCol_Text, color);
				ImGui::Text("%s %zu",
					lineageOperatorLabel(static_cast<LineageOperator>(opIdx)),
					count);
				ImGui::PopStyleColor();
			}

			// Gen best: this generation's children; noisy and not expected to decrease monotonically.
			ImGui::TableNextColumn();
			if (bucket._best)
			{
				const bool isGlobal = (bucket._best == globalBest);
				if (isGlobal)
				{
					ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 220, 120, 255));
					ImGui::Text("* %.4f", bucket._best->_score);
					ImGui::PopStyleColor();
				}
				else
				{
					ImGui::Text("%.4f", bucket._best->_score);
				}
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("Best score among candidates first produced in this generation.\nNoisy — most children score worse than their elite parents; the optimizer improves when occasional children beat the elite.");
			}
			else
			{
				ImGui::TextUnformatted("--");
			}

			// Best so far: cumulative running min excluding the refiner bucket; monotonically non-increasing.
			ImGui::TableNextColumn();
			if (bucket._generation != -1 && bucket._best)
			{
				cumulativeBest = std::min(cumulativeBest, bucket._best->_score);
				ImGui::Text("%.4f", cumulativeBest);
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("Running min across this and all prior generations.\nThis is the metric that should be monotonically non-increasing.");
			}
			else if (bucket._generation == -1 && bucket._best)
			{
				// Show the refiner's own best, separated from the cumulative GA trace.
				const double refinedDelta = cumulativeBest - bucket._best->_score;
				ImGui::Text("%.4f", bucket._best->_score);
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("Threshold refiner's best (post-GA).\nDelta vs. cumulative GA best: %+.4f",
						-refinedDelta);
			}
			else
			{
				ImGui::TextUnformatted("--");
			}

			ImGui::TableNextColumn();
			if (bucket._best)
			{
				ImGui::PushStyleColor(ImGuiCol_Text, lineageOperatorColor(bucket._bestOp));
				ImGui::TextUnformatted(bucket._best->_schemaName.c_str());
				ImGui::PopStyleColor();
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("Operator: %s\nLatency: %.4f ms\nBuild: %.4f ms",
						lineageOperatorLabel(bucket._bestOp),
						bucket._best->_averageLatencyMs,
						bucket._best->_buildTimeMs);
			}
		}
		ImGui::EndTable();
	}

	// Cumulative best progress trace; if it flattens across generations the GA is stuck.
	if (buckets.size() >= 2)
	{
		ImGui::Spacing();
		ImGui::TextDisabled("Cumulative best (monotone — what to actually watch):");
		std::string trace;
		double running = std::numeric_limits<double>::infinity();
		for (size_t i = 0; i < buckets.size(); ++i)
		{
			if (buckets[i]._generation == -1 || !buckets[i]._best)
				continue;
			running = std::min(running, buckets[i]._best->_score);
			if (!trace.empty())
				trace += " -> ";
			char prefix[16];
			if (buckets[i]._generation == 0)
				std::snprintf(prefix, sizeof(prefix), "init");
			else
				std::snprintf(prefix, sizeof(prefix), "g%d", buckets[i]._generation);
			char score[32];
			std::snprintf(score, sizeof(score), "%.4f", running);
			trace += std::string(prefix) + ":" + score;
		}
		// Append refiner bucket separately so its delta is visible without polluting the trace.
		for (const GenerationBucket& bucket : buckets)
		{
			if (bucket._generation == -1 && bucket._best)
			{
				char score[32];
				std::snprintf(score, sizeof(score), "%.4f", bucket._best->_score);
				trace += "  +  ref:" + std::string(score);
			}
		}
		ImGui::TextWrapped("%s", trace.c_str());
	}
}

static void drawLiveRankingTable(const std::vector<LiveRankingEntry>& ranking, size_t evaluatedCandidates, int topN, GuiState& state)
{
	ImGui::Text("Measured candidates: %zu", evaluatedCandidates);
	if (ranking.empty())
	{
		ImGui::TextUnformatted("Live ranking will appear after the first candidate finishes.");
		return;
	}

	if (ImGui::BeginTable("live-ranking", 13, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp))
	{
		ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 34.0f);
		ImGui::TableSetupColumn("Schema");
		ImGui::TableSetupColumn("Mode", ImGuiTableColumnFlags_WidthFixed, 92.0f);
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
			ImGui::TextUnformatted(entry._schemaName.c_str());
			if (ImGui::IsItemHovered())
			{
				ImGui::SetTooltip(
					"%s\n%s / %s\norder #%zu",
					entry._schemaPath.c_str(),
					entry._dataset.c_str(),
					entry._workload.c_str(),
					entry._order);
			}
			ImGui::TableSetColumnIndex(2);
			ImGui::Text("%s/%s", entry._scoreMode.c_str(), entry._scoreStage.c_str());
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("%s score, %zu effective queries%s",
					entry._scoreMode.c_str(),
					entry._effectiveQueries,
					entry._scoreIsFinalLatency ? ", final latency" : ", not final latency");
			ImGui::TableSetColumnIndex(3);
			ImGui::Text("%.4f", entry._score);
			ImGui::TableSetColumnIndex(4);
			ImGui::Text("%.3f", entry._averageLatencyMs);
			ImGui::TableSetColumnIndex(5);
			ImGui::Text("%.3f", entry._p95LatencyMs);
			ImGui::TableSetColumnIndex(6);
			ImGui::Text("%.0f", entry._averageTestedPoints);
			ImGui::TableSetColumnIndex(7);
			ImGui::Text("%.1f", entry._averageVisitedNodes);
			ImGui::TableSetColumnIndex(8);
			ImGui::Text("%.1f", entry._buildTimeMs);
			ImGui::TableSetColumnIndex(9);
			if (isCudaResult(entry._backend))
				ImGui::Text("%.1f", entry._gpuBuildMs);
			else
				ImGui::TextUnformatted("-");
			ImGui::TableSetColumnIndex(10);
			if (isCudaResult(entry._backend))
				ImGui::Text("%.1f", entry._gpuQueryMs);
			else
				ImGui::TextUnformatted("-");
			ImGui::TableSetColumnIndex(11);
			if (!entry._cudaBuilder.empty())
				ImGui::Text("%s/%s", entry._backend.c_str(), entry._cudaBuilder.c_str());
			else
				ImGui::TextUnformatted(entry._backend.empty() ? "cpu" : entry._backend.c_str());
			ImGui::TableSetColumnIndex(12);
			ImGui::PushID(static_cast<int>(i));
			drawSchemaActions(state, entry._schemaPath);
			ImGui::PopID();
		}

		ImGui::EndTable();
	}
}

static void drawRunPanel(GuiState& state, RunSession& session)
{
	std::string status;
	std::string error;
	std::string log;
	std::vector<BestResult> results;
	std::vector<LiveRankingEntry> liveRanking;
	size_t evaluatedCandidates = 0;
	{
		std::lock_guard<std::mutex> lock(session._mutex);
		status = session._status;
		error = session._error;
		log = session._log;
		results = session._bestResults;
		liveRanking = session._liveRanking;
		evaluatedCandidates = session._evaluatedCandidates;
	}

	drawSectionTitle("4. Run");
	const bool canStart = !session._running;
	ImGui::BeginDisabled(!canStart);
	if (ImGui::Button("Start tuning", ImVec2(180.0f, 34.0f)))
		startRun(session, state);
	ImGui::EndDisabled();
	ImGui::SameLine();
	ImGui::Text("Status: %s", status.c_str());
	if (session._running)
		ImGui::ProgressBar(-1.0f * static_cast<float>(ImGui::GetTime()), ImVec2(-1.0f, 0.0f));

	if (!error.empty())
	{
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.38f, 1.0f));
		ImGui::TextWrapped("%s", error.c_str());
		ImGui::PopStyleColor();
	}

	drawSectionTitle("Live Ranking");
	ImGui::SetNextItemWidth(96.0f);
	ImGui::InputInt("Top N", &state._liveRankingTopN);
	drawHelpMarker("Number of live leaderboard rows to display. The run still measures every candidate and writes every row to CSV.");
	drawLiveRankingTable(liveRanking, evaluatedCandidates, state._liveRankingTopN, state);

	drawSectionTitle("Generations");
	drawHelpMarker("Per-generation rollup parsed from schema names. Shows population size, the mix of operators (baseline/generated/mutation/crossover/refined) that contributed, and the best-of-generation candidate. The global-best row is marked with *. Works on any optimizer run (GA, auto-conditions, or both) since the data comes from live ranking entries.");
	drawGenerationSummary(liveRanking);

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

static ImVec2 projectPreviewPoint(const glm::vec3& point, const ImVec2& origin, const ImVec2& size, const StructurePreview& preview)
{
	const float yawCos = std::cos(preview._yaw);
	const float yawSin = std::sin(preview._yaw);
	const float pitchCos = std::cos(preview._pitch);
	const float pitchSin = std::sin(preview._pitch);

	const float x0 = yawCos * point.x + yawSin * point.z;
	const float z0 = -yawSin * point.x + yawCos * point.z;
	const float y1 = pitchCos * point.y - pitchSin * z0;
	const float z1 = pitchSin * point.y + pitchCos * z0;
	const float perspective = preview._zoom / std::max(0.25f, preview._zoom + z1);
	const float scale = std::min(size.x, size.y) * 0.34f * perspective;
	return ImVec2(origin.x + size.x * 0.5f + x0 * scale, origin.y + size.y * 0.52f - y1 * scale);
}

static void drawPreviewBox(ImDrawList* drawList, const PreviewBox& box, const ImVec2& origin, const ImVec2& size, const StructurePreview& preview)
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

	const bool rootBounds = box._depth == 0 && preview._showFullSchedule;
	const float alpha = rootBounds ? 0.30f : std::clamp(0.92f - static_cast<float>(box._depth) * 0.06f, 0.35f, 0.92f);
	const float thickness = box._depth == 0 ? 1.6f : 1.0f;
	const ImU32 color = rootBounds
		? IM_COL32(174, 184, 196, 120)
		: packedColorForStructureName(box._typeName, alpha);

	for (const auto& edge : edges)
	{
		const ImVec2 a = projectPreviewPoint(corners[edge[0]], origin, size, preview);
		const ImVec2 b = projectPreviewPoint(corners[edge[1]], origin, size, preview);
		drawList->AddLine(a, b, color, thickness);
	}
}

static void drawPreviewLegend()
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

static void drawSchemaFileViewerWindow(GuiState& state)
{
	SchemaFileViewer& viewer = state._fileViewer;
	if (!viewer._open)
		return;

	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 72.0f, viewport->WorkPos.y + 72.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(720.0f, 560.0f), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Schema JSON", &viewer._open, ImGuiWindowFlags_HorizontalScrollbar))
	{
		ImGui::TextUnformatted(viewer._title.c_str());
		if (!viewer._path.empty())
		{
			ImGui::TextDisabled("%s", viewer._path.c_str());
			if (ImGui::Button("Reload"))
				viewer._content = loadTextFile(viewer._path, viewer._error);
			ImGui::SameLine();
			if (ImGui::Button("Preview boxes"))
				openStructurePreview(state, viewer._path);
			ImGui::SameLine();
			ImGui::Checkbox("Wrap", &viewer._wrap);
		}
		if (!viewer._error.empty())
		{
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.38f, 1.0f));
			ImGui::TextWrapped("%s", viewer._error.c_str());
			ImGui::PopStyleColor();
		}

		ImGui::Separator();
		ImGui::BeginChild("schema-json-content", ImVec2(0.0f, 0.0f), true, viewer._wrap ? 0 : ImGuiWindowFlags_HorizontalScrollbar);
		if (viewer._wrap)
			ImGui::TextWrapped("%s", viewer._content.c_str());
		else
			ImGui::TextUnformatted(viewer._content.c_str());
		ImGui::EndChild();
	}
	ImGui::End();
}

static void drawStructurePreviewWindow(GuiState& state, bool optimizerRunning)
{
	StructurePreview& preview = state._structurePreview;
	if (!preview._open)
		return;

	if (preview._needsRebuild)
		rebuildStructurePreview(preview);

	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 620.0f, viewport->WorkPos.y + 96.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(760.0f, 560.0f), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Structure Preview", &preview._open))
	{
		ImGui::TextUnformatted(preview._title.c_str());
		if (!preview._schemaName.empty())
		{
			ImGui::SameLine();
			ImGui::TextDisabled("%s", preview._schemaName.c_str());
		}
		if (!preview._path.empty())
			ImGui::TextDisabled("%s", preview._path.c_str());

		if (ImGui::Checkbox("Full schedule", &preview._showFullSchedule))
			preview._needsRebuild = true;
		ImGui::SameLine();
		drawHelpMarker("Focused mode shows one schema block at a time. Full schedule shows the old cumulative expansion and is mainly useful for simple fixed schemas.");

		if (!preview._showFullSchedule && !preview._phases.empty())
		{
			ImGui::SetNextItemWidth(260.0f);
			const int phaseCount = static_cast<int>(preview._phases.size());
			preview._phaseIndex = std::clamp(preview._phaseIndex, 0, phaseCount - 1);
			const char* currentPhase = preview._phases[preview._phaseIndex]._label.c_str();
			if (ImGui::BeginCombo("Block", currentPhase))
			{
				for (int i = 0; i < phaseCount; ++i)
				{
					const bool selected = i == preview._phaseIndex;
					if (ImGui::Selectable(preview._phases[i]._label.c_str(), selected))
					{
						preview._phaseIndex = i;
						preview._needsRebuild = true;
					}
					if (selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			ImGui::SameLine();
		}

		ImGui::SetNextItemWidth(92.0f);
		if (ImGui::InputInt(preview._showFullSchedule ? "Global depth" : "Local levels", &preview._maxDepth))
			preview._needsRebuild = true;
		ImGui::SameLine();
		ImGui::SetNextItemWidth(110.0f);
		if (ImGui::InputInt("Box cap", &preview._maxBoxes))
			preview._needsRebuild = true;
		ImGui::SameLine();
		if (ImGui::Button("Reload"))
			preview._needsRebuild = true;
		ImGui::SameLine();
		drawHelpMarker("This preview renders normalized boxes only. It does not load or rebuild the point cloud, so it stays outside the optimizer worker.");

		if (optimizerRunning)
		{
			ImGui::SameLine();
			ImGui::TextDisabled("optimizer running");
		}

		if (!preview._error.empty())
		{
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.38f, 1.0f));
			ImGui::TextWrapped("%s", preview._error.c_str());
			ImGui::PopStyleColor();
		}

		if (!preview._showFullSchedule && !preview._phases.empty())
		{
			const PreviewPhase& phase = preview._phases[preview._phaseIndex];
			ImGui::Text("Focused block: %s, schema depths %zu-%zu",
				phase._level._typeName.c_str(),
				phase._startDepth,
				phase._endDepth == 0 ? 0 : phase._endDepth - 1);
		}
		ImGui::Text("Boxes: %zu%s", preview._boxes.size(), preview._truncated ? " (capped)" : "");
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
			preview._yaw += io.MouseDelta.x * 0.008f;
			preview._pitch = std::clamp(preview._pitch + io.MouseDelta.y * 0.008f, -1.25f, 1.25f);
		}
		if (hovered && io.MouseWheel != 0.0f)
			preview._zoom = std::clamp(preview._zoom - io.MouseWheel * 0.18f, 1.4f, 8.0f);

		ImDrawList* drawList = ImGui::GetWindowDrawList();
		drawList->AddRectFilled(canvasOrigin, ImVec2(canvasOrigin.x + canvasSize.x, canvasOrigin.y + canvasSize.y), IM_COL32(12, 15, 18, 255));
		drawList->AddRect(canvasOrigin, ImVec2(canvasOrigin.x + canvasSize.x, canvasOrigin.y + canvasSize.y), IM_COL32(55, 64, 75, 255));
		drawList->PushClipRect(canvasOrigin, ImVec2(canvasOrigin.x + canvasSize.x, canvasOrigin.y + canvasSize.y), true);
		for (const PreviewBox& box : preview._boxes)
			drawPreviewBox(drawList, box, canvasOrigin, canvasSize, preview);
		drawList->PopClipRect();
	}
	ImGui::End();
}

static void drawInterface(GuiState& state, RunSession& session)
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
	ImGui::SameLine();
	ImGui::TextDisabled("|  steps 1-3 configure on the left, step 4 runs on the right");
	ImGui::Separator();

	const float leftWidth = std::max(360.0f, ImGui::GetContentRegionAvail().x * 0.36f);
	ImGui::BeginChild("left-panel", ImVec2(leftWidth, 0.0f), true);
	drawPublicationPanel(state);
	if (ImGui::BeginTabBar("configuration-tabs"))
	{
		if (ImGui::BeginTabItem("1. Inputs"))
		{
			drawDatasetPanel(state);
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("2. Search"))
		{
			drawSchemaPanel(state);
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("3. Advanced"))
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
	drawStructurePreviewWindow(state, session._running);
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
		if (glfwWindowShouldClose(window) && session._running)
		{
			glfwSetWindowShouldClose(window, GLFW_FALSE);
			std::lock_guard<std::mutex> lock(session._mutex);
			session._status = "Running; close after completion";
		}

		if (session._finished.exchange(false) && session._worker.joinable())
			session._worker.join();

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

	if (session._worker.joinable())
		session._worker.join();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}

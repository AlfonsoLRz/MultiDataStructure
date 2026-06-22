#include "stdafx.h"
#include "AppConfig.h"

// Parses a rung spec `name:queries:scoreMode:advance` (queries=0 uses workload default, advance=0 promotes all).
static bool parseRungSpec(const std::string& token, Experiments::RungSpec& outRung)
{
	std::vector<std::string> parts;
	size_t start = 0;
	while (start <= token.size())
	{
		const size_t end = token.find(':', start);
		parts.push_back(token.substr(start, end == std::string::npos ? std::string::npos : end - start));
		if (end == std::string::npos)
			break;
		start = end + 1;
	}
	if (parts.size() < 3)
		return false;

	outRung._name = parts[0];
	try { outRung._queryCountOverride = static_cast<size_t>(std::stoull(parts[1])); }
	catch (...) { return false; }

	const std::string& mode = parts[2];
	if (mode == "visit" || mode == "visit-proxy" || mode == "proxy")
	{
		outRung._useVisitProxy = true;
		if (outRung._visitProxyAlpha <= 0.0)
			outRung._visitProxyAlpha = 0.1;
	}
	else if (mode == "latency")
	{
		outRung._useVisitProxy = false;
	}
	else
	{
		return false;
	}

	if (parts.size() >= 4 && !parts[3].empty())
	{
		try { outRung._advanceTopK = static_cast<size_t>(std::stoull(parts[3])); }
		catch (...) { return false; }
	}
	return true;
}

// Parses `--rungs` argument: comma-separated rung specs.
static bool parseRungSchedule(const std::string& argument, std::vector<Experiments::RungSpec>& outRungs)
{
	outRungs.clear();
	if (argument.empty())
		return true;

	size_t start = 0;
	while (start <= argument.size())
	{
		const size_t end = argument.find(',', start);
		const std::string token = argument.substr(start, end == std::string::npos ? std::string::npos : end - start);
		if (!token.empty())
		{
			Experiments::RungSpec rung;
			if (!parseRungSpec(token, rung))
				return false;
			outRungs.push_back(std::move(rung));
		}
		if (end == std::string::npos)
			break;
		start = end + 1;
	}
	return true;
}

static bool pathExists(const std::filesystem::path& path)
{
	std::error_code error;
	return std::filesystem::exists(path, error);
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
		const auto alreadyAdded = std::find(roots.begin(), roots.end(), normalized);
		if (alreadyAdded == roots.end())
			roots.push_back(normalized);

		if (!start.has_parent_path() || start == start.parent_path())
			break;

		start = start.parent_path();
	}
}

static std::string trimCopy(const std::string& value)
{
	const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); });
	const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) { return std::isspace(c); }).base();
	if (first >= last)
		return {};
	return std::string(first, last);
}

static std::vector<std::string> splitPathList(const std::string& value)
{
	std::vector<std::string> paths;
	std::string current;
	for (const char c : value)
	{
		if (c == ';' || c == ',')
		{
			std::string path = trimCopy(current);
			if (!path.empty())
				paths.push_back(std::move(path));
			current.clear();
			continue;
		}

		current.push_back(c);
	}

	std::string path = trimCopy(current);
	if (!path.empty())
		paths.push_back(std::move(path));

	return paths;
}

AppConfig::AppConfig()
	: _pointOptions(defaultPointOptions()),
	  _schemaSearchOptions(defaultSchemaSearchOptions())
{
}

PointBenchmark::Options AppConfig::defaultPointOptions()
{
	PointBenchmark::Options options;
	options._inputPath = AppDefaults::POINT_INPUT_PATH;
	options._schemaPath = AppDefaults::POINT_SCHEMA_PATH;
	options._schemaPaths = splitPathList(AppDefaults::POINT_SCHEMA_PATHS);
	options._outputPath = AppDefaults::POINT_OUTPUT_PATH;
	options._csvPath = AppDefaults::POINT_CSV_OUTPUT_PATH;
	options._modelPath = AppDefaults::POINT_MODEL_PATH;
	options._workloadProfilePath = AppDefaults::POINT_WORKLOAD_PROFILE_PATH;
	options._useBinaryCache = AppDefaults::POINT_USE_BINARY_CACHE;
	options._rebuildBinaryCache = AppDefaults::POINT_REBUILD_BINARY_CACHE;
	options._pauseAtEnd = AppDefaults::PAUSE_AT_END;
	options._queryCount = AppDefaults::POINT_QUERY_COUNT;
	options._queryK = AppDefaults::POINT_QUERY_K;
	options._querySeed = AppDefaults::POINT_QUERY_SEED;
	options._enableLeafMicroIndexes = AppDefaults::ENABLE_LEAF_MICRO_INDEXES;
	options._leafMicroIndexThreshold = AppDefaults::LEAF_MICRO_INDEX_THRESHOLD;
	return options;
}

Experiments::SchemaSearchOptions AppConfig::defaultSchemaSearchOptions()
{
	Experiments::SchemaSearchOptions options;
	options._schemaPaths = splitPathList(AppDefaults::SCHEMA_SEARCH_SCHEMA_PATHS);
	options._workloadPaths = splitPathList(AppDefaults::SCHEMA_SEARCH_WORKLOAD_PATHS);
	options._csvPath = AppDefaults::SCHEMA_SEARCH_CSV_PATH;
	options._bestCsvPath = AppDefaults::SCHEMA_SEARCH_BEST_CSV_PATH;
	options._paretoCsvPath = AppDefaults::SCHEMA_SEARCH_PARETO_CSV_PATH;
	options._explainReportPath = AppDefaults::SCHEMA_SEARCH_EXPLAIN_REPORT_PATH;
	options._syntheticScale = AppDefaults::SCHEMA_SEARCH_SYNTHETIC_SCALE;
	options._queryCountOverride = AppDefaults::SCHEMA_SEARCH_QUERY_COUNT;
	options._querySeed = AppDefaults::POINT_QUERY_SEED;
	options._rankModelPath = AppDefaults::SCHEMA_SEARCH_RANK_MODEL_PATH;
	options._generation._count = AppDefaults::SCHEMA_SEARCH_GENERATED_COUNT;
	options._benchmarkTopK = AppDefaults::SCHEMA_SEARCH_BENCHMARK_TOP_K;
	options._evaluator = AppDefaults::SCHEMA_SEARCH_EVALUATOR;
	options._cuda._device = AppDefaults::SCHEMA_SEARCH_CUDA_DEVICE;
	options._cuda._builder = AppDefaults::SCHEMA_SEARCH_CUDA_BUILDER;
	options._autoConditions._proxyCandidateCount = AppDefaults::AUTO_CONDITION_PROXY_CANDIDATES;
	options._autoConditions._proxyPointCap = AppDefaults::AUTO_CONDITION_PROXY_POINTS;
	options._autoConditions._proxyQueryCount = AppDefaults::AUTO_CONDITION_PROXY_QUERIES;
	options._autoConditions._finalTopK = AppDefaults::AUTO_CONDITION_FINAL_TOP_K;
	options._autoConditions._confirmationTopK = AppDefaults::AUTO_CONDITION_CONFIRM_TOP_K;
	options._enableLeafMicroIndexes = AppDefaults::ENABLE_LEAF_MICRO_INDEXES;
	options._leafMicroIndexThreshold = AppDefaults::LEAF_MICRO_INDEX_THRESHOLD;
	options._pauseAtEnd = AppDefaults::PAUSE_AT_END;
	return options;
}

AppConfig AppConfig::parse(int argc, char* argv[])
{
	AppConfig config;
	bool modeWasSpecified = false;

	for (int i = 1; i < argc; ++i)
	{
		const std::string arg = argv[i];
		if (arg == "--help" || arg == "-h")
		{
			config._showHelp = true;
		}
		else if (arg == "--run-tests")
		{
			config._runTests = true;
			config._pointOptions._pauseAtEnd = false;
			config._schemaSearchOptions._pauseAtEnd = false;
		}
		else if (arg == "--gui")
		{
			config._mode = "gui";
			modeWasSpecified = true;
		}
		else if (arg == "--no-pause")
		{
			config._pointOptions._pauseAtEnd = false;
			config._schemaSearchOptions._pauseAtEnd = false;
		}
		else if (arg == "--no-cache")
		{
			config._pointOptions._useBinaryCache = false;
			config._schemaSearchOptions._useBinaryCache = false;
		}
		else if (arg == "--rebuild-cache")
		{
			config._pointOptions._useBinaryCache = true;
			config._pointOptions._rebuildBinaryCache = true;
			config._schemaSearchOptions._useBinaryCache = true;
			config._schemaSearchOptions._rebuildBinaryCache = true;
		}
		else if (arg == "--mode" && i + 1 < argc)
		{
			config._mode = argv[++i];
			modeWasSpecified = true;
		}
		else if (arg == "--input" && i + 1 < argc)
		{
			config._pointOptions._inputPath = argv[++i];
			config._schemaSearchOptions._inputPaths.clear();
			config._schemaSearchOptions._inputPaths.push_back(config._pointOptions._inputPath);
			if (!modeWasSpecified)
				config._mode = "points";
		}
		else if (arg == "--schema" && i + 1 < argc)
		{
			config._pointOptions._schemaPath = argv[++i];
			config._pointOptions._schemaPaths.clear();
			config._schemaSearchOptions._schemaPaths.clear();
			config._schemaSearchOptions._schemaPaths.push_back(config._pointOptions._schemaPath);
		}
		else if (arg == "--schemas" && i + 1 < argc)
		{
			config._pointOptions._schemaPaths = splitPathList(argv[++i]);
			config._schemaSearchOptions._schemaPaths = config._pointOptions._schemaPaths;
		}
		else if (arg == "--workloads" && i + 1 < argc)
		{
			config._schemaSearchOptions._workloadPaths = splitPathList(argv[++i]);
		}
		else if (arg == "--model" && i + 1 < argc)
		{
			config._pointOptions._modelPath = argv[++i];
			config._schemaSearchOptions._rankModelPath = config._pointOptions._modelPath;
		}
		else if (arg == "--rank-model" && i + 1 < argc)
		{
			config._schemaSearchOptions._rankModelPath = argv[++i];
		}
		else if (arg == "--workload-profile" && i + 1 < argc)
		{
			config._pointOptions._workloadProfilePath = argv[++i];
		}
		else if (arg == "--no-synthetic")
		{
			config._schemaSearchOptions._includeSyntheticDatasets = false;
		}
		else if (arg == "--synthetic-scale" && i + 1 < argc)
		{
			config._schemaSearchOptions._syntheticScale = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generate-schemas" && i + 1 < argc)
		{
			config._schemaSearchOptions._generation._count = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-only")
		{
			config._schemaSearchOptions._includeConfiguredSchemas = false;
		}
		else if (arg == "--deep-nested-search")
		{
			config._schemaSearchOptions._deepNestedSearch = true;
			config._schemaSearchOptions._evaluator = "cpu";
			config._schemaSearchOptions._rankModelPath.clear();
			config._schemaSearchOptions._includeConfiguredSchemas = false;
			config._schemaSearchOptions._includeBaselineSchemas = true;
			config._schemaSearchOptions._autoConditions._enabled = true;
			config._schemaSearchOptions._autoConditions._proxyCandidateCount = 2048;
			config._schemaSearchOptions._autoConditions._proxyPointCap = 262144;
			config._schemaSearchOptions._autoConditions._proxyQueryCount = 8;
			config._schemaSearchOptions._autoConditions._finalTopK = 128;
			config._schemaSearchOptions._autoConditions._confirmationTopK = 24;
			config._schemaSearchOptions._generation._count = 2048;
			config._schemaSearchOptions._generation._minBlocks = 2;
			config._schemaSearchOptions._generation._maxBlocks = std::max<size_t>(3, config._schemaSearchOptions._generation._maxBlocks);
			config._schemaSearchOptions._generation._conditionalLevels = true;
			config._schemaSearchOptions._generation._conditionalProbability = std::max(0.75, config._schemaSearchOptions._generation._conditionalProbability);
			config._schemaSearchOptions._queryCountOverride = 256;
			if (config._schemaSearchOptions._scoreCachePath.empty())
				config._schemaSearchOptions._scoreCachePath = "results/deep_nested_score_cache.jsonl";
			config._schemaSearchOptions._cuda._device = 0;
			config._schemaSearchOptions._cuda._builder = "mixed";
		}
		else if (arg == "--benchmark-top" && i + 1 < argc)
		{
			config._schemaSearchOptions._benchmarkTopK = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-min-blocks" && i + 1 < argc)
		{
			config._schemaSearchOptions._generation._minBlocks = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-max-blocks" && i + 1 < argc)
		{
			config._schemaSearchOptions._generation._maxBlocks = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-max-depth" && i + 1 < argc)
		{
			config._schemaSearchOptions._generation._maxDepth = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-min-leaf" && i + 1 < argc)
		{
			config._schemaSearchOptions._generation._minLeafCapacity = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-max-leaf" && i + 1 < argc)
		{
			config._schemaSearchOptions._generation._maxLeafCapacity = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-conditional")
		{
			config._schemaSearchOptions._generation._conditionalLevels = true;
		}
		else if (arg == "--generated-condition-probability" && i + 1 < argc)
		{
			config._schemaSearchOptions._generation._conditionalProbability = std::stod(argv[++i]);
		}
		else if (arg == "--generated-adaptive-leaf-capacity")
		{
			config._schemaSearchOptions._generation._adaptiveLeafCapacity = true;
		}
		else if (arg == "--generated-adaptive-leaf-probability" && i + 1 < argc)
		{
			config._schemaSearchOptions._generation._adaptiveLeafProbability = std::stod(argv[++i]);
		}
		else if (arg == "--generated-seed" && i + 1 < argc)
		{
			config._schemaSearchOptions._generation._seed = static_cast<uint32_t>(std::stoul(argv[++i]));
		}
		else if (arg == "--generated-schema-dir" && i + 1 < argc)
		{
			config._schemaSearchOptions._generation._outputDirectory = argv[++i];
		}
		else if (arg == "--primitive-profile" && i + 1 < argc)
		{
			config._schemaSearchOptions._generation._primitiveProfile = argv[++i];
		}
		else if (arg == "--query-minimal-primitives")
		{
			config._schemaSearchOptions._generation._primitiveProfile = "query_minimal_cpu";
		}
		else if (arg == "--cuda-query-full-primitives")
		{
			config._schemaSearchOptions._generation._primitiveProfile = "cuda_query_full";
		}
		else if (arg == "--all-primitives")
		{
			config._schemaSearchOptions._generation._primitiveProfile = "all";
		}
		else if (arg == "--auto-conditions")
		{
			config._schemaSearchOptions._autoConditions._enabled = true;
			config._schemaSearchOptions._generation._conditionalLevels = true;
		}
		else if (arg == "--condition-proxy-candidates" && i + 1 < argc)
		{
			config._schemaSearchOptions._autoConditions._proxyCandidateCount = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--condition-proxy-points" && i + 1 < argc)
		{
			config._schemaSearchOptions._autoConditions._proxyPointCap = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--condition-proxy-queries" && i + 1 < argc)
		{
			config._schemaSearchOptions._autoConditions._proxyQueryCount = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--condition-final-top" && i + 1 < argc)
		{
			config._schemaSearchOptions._autoConditions._finalTopK = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--condition-confirm-top" && i + 1 < argc)
		{
			config._schemaSearchOptions._autoConditions._confirmationTopK = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--condition-output-dir" && i + 1 < argc)
		{
			config._schemaSearchOptions._autoConditions._outputDirectory = argv[++i];
		}
		else if (arg == "--condition-selector-output" && i + 1 < argc)
		{
			config._schemaSearchOptions._autoConditions._selectorOutputPath = argv[++i];
		}
		else if (arg == "--optimize-schemas")
		{
			config._schemaSearchOptions._evolution._enabled = true;
		}
		else if (arg == "--optimizer-generations" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._generations = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--optimizer-population" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._populationSize = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--optimizer-elites" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._eliteCount = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--optimizer-mutation-rate" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._mutationRate = std::stod(argv[++i]);
		}
		else if (arg == "--optimizer-random-fraction" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._randomImmigrationRate = std::stod(argv[++i]);
		}
		else if (arg == "--optimizer-seed" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._seed = static_cast<uint32_t>(std::stoul(argv[++i]));
		}
		else if (arg == "--optimizer-crossover-rate" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._crossoverRate = std::stod(argv[++i]);
		}
		else if (arg == "--optimizer-nsga2")
		{
			config._schemaSearchOptions._evolution._useNsga2Ranking = true;
		}
		else if (arg == "--no-optimizer-nsga2")
		{
			config._schemaSearchOptions._evolution._useNsga2Ranking = false;
		}
		else if (arg == "--repair-mutations")
		{
			config._schemaSearchOptions._evolution._repairMutations = true;
		}
		else if (arg == "--no-repair-mutations")
		{
			config._schemaSearchOptions._evolution._repairMutations = false;
		}
		else if (arg == "--repair-top" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._repairTopK = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--repair-per-candidate" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._repairPerCandidate = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--rungs" && i + 1 < argc)
		{
			const std::string spec = argv[++i];
			if (!parseRungSchedule(spec, config._schemaSearchOptions._evolution._rungSchedule._rungs))
				throw std::runtime_error("--rungs expects 'name:queries:visit|latency:advance,...' (e.g. proxy:4:visit:32,full:16:latency:8,confirm:64:latency:0)");
		}
		else if (arg == "--rung-surrogate" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._rungSchedule._surrogateModelPath = argv[++i];
		}
		else if (arg == "--rung-surrogate-pool" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._rungSchedule._surrogateCandidatePool = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--rung-surrogate-top" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._rungSchedule._surrogateProposalsPerStep = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--refine-thresholds")
		{
			config._schemaSearchOptions._evolution._refineThresholds = true;
		}
		else if (arg == "--refine-thresholds-top" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._refineThresholdsTopK = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--refine-thresholds-evals" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._refineThresholdsEvaluations = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--refine-thresholds-sigma" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._refineThresholdsSigma0 = std::stod(argv[++i]);
		}
		else if (arg == "--refine-thresholds-seed" && i + 1 < argc)
		{
			config._schemaSearchOptions._evolution._refineThresholdsSeed = static_cast<uint32_t>(std::stoul(argv[++i]));
		}
		else if (arg == "--evaluator" && i + 1 < argc)
		{
			config._schemaSearchOptions._evaluator = argv[++i];
		}
		else if (arg == "--cuda-device" && i + 1 < argc)
		{
			config._schemaSearchOptions._cuda._device = std::stoi(argv[++i]);
		}
		else if (arg == "--cuda-builder" && i + 1 < argc)
		{
			config._schemaSearchOptions._cuda._builder = argv[++i];
		}
		else if (arg == "--cuda-query-batch" && i + 1 < argc)
		{
			config._schemaSearchOptions._cuda._queryBatchSize = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--cuda-memory-budget-mb" && i + 1 < argc)
		{
			config._schemaSearchOptions._cuda._memoryBudgetMb = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--cuda-knn-backend" && i + 1 < argc)
		{
			config._schemaSearchOptions._cuda._knnBackend = argv[++i];
		}
		else if (arg == "--score-build-weight" && i + 1 < argc)
		{
			config._schemaSearchOptions._weights._lambdaBuild = std::stod(argv[++i]);
			config._schemaSearchOptions._scoreWeightsOverride = true;
		}
		else if (arg == "--score-memory-weight" && i + 1 < argc)
		{
			config._schemaSearchOptions._weights._lambdaMemory = std::stod(argv[++i]);
			config._schemaSearchOptions._scoreWeightsOverride = true;
		}
		else if (arg == "--score-imbalance-weight" && i + 1 < argc)
		{
			config._schemaSearchOptions._weights._lambdaImbalance = std::stod(argv[++i]);
			config._schemaSearchOptions._scoreWeightsOverride = true;
		}
		else if (arg == "--output" && i + 1 < argc)
		{
			config._pointOptions._outputPath = argv[++i];
		}
		else if (arg == "--csv" && i + 1 < argc)
		{
			config._pointOptions._csvPath = argv[++i];
			config._schemaSearchOptions._csvPath = config._pointOptions._csvPath;
		}
		else if (arg == "--query-trace" && i + 1 < argc)
		{
			config._pointOptions._queryTracePath = argv[++i];
			config._schemaSearchOptions._queryTracePath = config._pointOptions._queryTracePath;
		}
		else if (arg == "--no-csv")
		{
			config._pointOptions._csvPath.clear();
			config._schemaSearchOptions._csvPath.clear();
			config._schemaSearchOptions._bestCsvPath.clear();
			config._schemaSearchOptions._paretoCsvPath.clear();
		}
		else if (arg == "--best-csv" && i + 1 < argc)
		{
			config._schemaSearchOptions._bestCsvPath = argv[++i];
		}
		else if (arg == "--pareto-csv" && i + 1 < argc)
		{
			config._schemaSearchOptions._paretoCsvPath = argv[++i];
		}
		else if (arg == "--explain-report" && i + 1 < argc)
		{
			config._schemaSearchOptions._explainReportPath = argv[++i];
		}
		else if (arg == "--confirm-seeds" && i + 1 < argc)
		{
			config._schemaSearchOptions._confirmSeeds = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--confirm-top" && i + 1 < argc)
		{
			config._schemaSearchOptions._confirmTopK = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--measure-repeats" && i + 1 < argc)
		{
			config._schemaSearchOptions._measurementRepeats = std::max<size_t>(1, static_cast<size_t>(std::stoull(argv[++i])));
		}
		else if (arg == "--proxy-correlation-csv" && i + 1 < argc)
		{
			config._schemaSearchOptions._proxyCorrelationCsvPath = argv[++i];
		}
		else if (arg == "--verify-parity")
		{
			config._schemaSearchOptions._verifyParity = true;
		}
		else if (arg == "--estimate-prefilter")
		{
			config._schemaSearchOptions._estimatePrefilter = true;
		}
		else if (arg == "--queries" && i + 1 < argc)
		{
			config._pointOptions._queryCount = static_cast<size_t>(std::stoull(argv[++i]));
			config._schemaSearchOptions._queryCountOverride = config._pointOptions._queryCount;
		}
		else if (arg == "--knn-k" && i + 1 < argc)
		{
			config._pointOptions._queryK = static_cast<size_t>(std::stoull(argv[++i]));
			config._schemaSearchOptions._knnKOverride = config._pointOptions._queryK;
		}
		else if (arg == "--leaf-micro-indexes")
		{
			config._pointOptions._enableLeafMicroIndexes = true;
			config._schemaSearchOptions._enableLeafMicroIndexes = true;
		}
		else if (arg == "--leaf-micro-threshold" && i + 1 < argc)
		{
			config._pointOptions._leafMicroIndexThreshold = static_cast<size_t>(std::stoull(argv[++i]));
			config._schemaSearchOptions._leafMicroIndexThreshold = config._pointOptions._leafMicroIndexThreshold;
		}
		else if (arg == "--query-seed" && i + 1 < argc)
		{
			config._pointOptions._querySeed = static_cast<uint32_t>(std::stoul(argv[++i]));
			config._schemaSearchOptions._querySeed = config._pointOptions._querySeed;
			config._schemaSearchOptions._querySeedOverride = true;
		}
		else if (arg == "--score-cache" && i + 1 < argc)
		{
			config._schemaSearchOptions._scoreCachePath = argv[++i];
		}
		else if (arg == "--score-visit-proxy")
		{
			config._schemaSearchOptions._weights._useVisitProxy = true;
			config._schemaSearchOptions._scoreWeightsOverride = true;
		}
		else if (arg == "--score-visit-alpha" && i + 1 < argc)
		{
			config._schemaSearchOptions._weights._visitProxyAlpha = std::stod(argv[++i]);
			config._schemaSearchOptions._scoreWeightsOverride = true;
		}
		else if (arg == "--parallel" && i + 1 < argc)
		{
			config._schemaSearchOptions._parallelDispatch = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--no-baselines")
		{
			config._schemaSearchOptions._includeBaselineSchemas = false;
		}
		else if (arg == "--include-baselines")
		{
			config._schemaSearchOptions._includeBaselineSchemas = true;
		}
		else if (arg == "--no-score-cache")
		{
			config._schemaSearchOptions._scoreCachePath.clear();
		}
		else if (arg == "--rebuild-score-cache")
		{
			config._schemaSearchOptions._rebuildScoreCache = true;
		}
		else
		{
			throw std::invalid_argument("Unknown or incomplete argument: " + arg);
		}
	}

	const std::string executablePath = argc > 0 ? argv[0] : std::string();
	config._pointOptions._schemaPath = resolveExistingPath(config._pointOptions._schemaPath, executablePath);
	config._pointOptions._modelPath = resolveExistingPath(config._pointOptions._modelPath, executablePath);
	config._schemaSearchOptions._rankModelPath = resolveExistingPath(config._schemaSearchOptions._rankModelPath, executablePath);
	config._pointOptions._workloadProfilePath = resolveExistingPath(config._pointOptions._workloadProfilePath, executablePath);
	for (std::string& schemaPath : config._pointOptions._schemaPaths)
		schemaPath = resolveExistingPath(schemaPath, executablePath);
	for (std::string& schemaPath : config._schemaSearchOptions._schemaPaths)
		schemaPath = resolveExistingPath(schemaPath, executablePath);
	for (std::string& workloadPath : config._schemaSearchOptions._workloadPaths)
		workloadPath = resolveExistingPath(workloadPath, executablePath);

	return config;
}

void AppConfig::printHelp(std::ostream& output)
{
	output
		<< "MultiDataStructure point-cloud benchmark\n"
		<< "  Editable defaults live in MultiDataStructure/AppConfig.h\n"
		<< "  --mode gui|points|schema-search|evaluate-one|tests\n"
		<< "  --mode evaluate-one          One-shot: build/query a single (cloud, schema, workload) and emit JSON to stdout\n"
		<< "  --gui                       Open the ImGui optimizer interface\n"
		<< "  --input <path>              Point cloud path (.las, .ply, .xyz, .csv)\n"
		<< "  --schema <path>             Spatial schema JSON\n"
		<< "  --schema auto               Select a schema with a measured, JSON, or ONNX score ranker\n"
		<< "  --schemas <a;b;c>           Run the same point benchmark across schemas\n"
		<< "  --model <path>              Runtime selector JSON for --schema auto\n"
		<< "  --rank-model <path>         Surrogate selector JSON/ONNX model wrapper for generated schema search\n"
		<< "  --workload-profile <path>   Workload profile JSON for --schema auto\n"
		<< "  --workloads <a;b;c>         Workload profile JSONs for schema-search mode\n"
		<< "  --generate-schemas <count>  Generate nested schema candidates from bounded intervals\n"
		<< "  --generated-only            Search generated candidates without the configured static schema list\n"
		<< "  --deep-nested-search        CPU-first staged search for nested schemas against injected single-DS baselines\n"
		<< "  --benchmark-top <count>     Benchmark only top-k generated/static schemas after surrogate ranking\n"
		<< "  --generated-min-blocks <n>  Min nested blocks for generated schemas; deep search sets 2\n"
		<< "  --generated-max-blocks <n>  Max nested blocks for generated schemas\n"
		<< "  --generated-max-depth <n>   Max total generated schema depth\n"
		<< "  --generated-min-leaf <n>    Min generated leaf capacity\n"
		<< "  --generated-max-leaf <n>    Max generated leaf capacity\n"
		<< "  --generated-conditional     Add local node predicates to sampled nested blocks\n"
		<< "  --generated-condition-probability <v> Probability for generated conditional blocks\n"
		<< "  --generated-adaptive-leaf-capacity Add per-node adaptive leaf-capacity rules to generated CPU schemas\n"
		<< "  --generated-adaptive-leaf-probability <v> Probability for generated adaptive leaf-capacity levels\n"
		<< "  --generated-seed <seed>     Seed for generated schema search space sampling\n"
		<< "  --generated-schema-dir <p>  Directory for generated schema JSON files\n"
		<< "  --primitive-profile <auto|query_minimal_cpu|cuda_query_full|all> Generated primitive family profile\n"
		<< "  --query-minimal-primitives  CPU query-first generation: QuadTree/Octree/KDTree/BVH only\n"
		<< "  --cuda-query-full-primitives Include CUDA-native generated variants such as KarrasOctree/LBVH/BIH/Grid\n"
		<< "  --all-primitives            Exhaustive generated primitive aliases, including CPU-equivalent variants\n"
		<< "  --auto-conditions           Tune generated conditional schema thresholds for the current cloud/workload\n"
		<< "  --condition-proxy-candidates <n> Candidates for auto-condition proxy stage; default 256\n"
		<< "  --condition-proxy-points <n> Downsample cap for auto-condition proxy stage; default 262144\n"
		<< "  --condition-proxy-queries <n> Queries for auto-condition proxy stage; default 8\n"
		<< "  --condition-final-top <n>   Full-cloud candidates after proxy pruning; default 16\n"
		<< "  --condition-confirm-top <n> Confirmation candidates after short full run; default 4\n"
		<< "  --condition-output-dir <dir> Directory for tuned schema JSON artifacts\n"
		<< "  --condition-selector-output <path> Measured selector JSON path for tuned schemas\n"
		<< "  --optimize-schemas          Run evolutionary mutation search after the initial candidate set\n"
		<< "  --optimizer-generations <n> Evolution generations, default 3\n"
		<< "  --optimizer-population <n>  Mutated/random candidates per generation, default 64\n"
		<< "  --optimizer-elites <n>      Best measured candidates kept as parents, default 6\n"
		<< "  --optimizer-mutation-rate <v> Extra mutation probability per child, default 0.65\n"
		<< "  --optimizer-random-fraction <v> Fraction of each generation sampled randomly, default 0.20\n"
		<< "  --optimizer-crossover-rate <v> Probability that a child is built by splicing two elites, default 0.40\n"
		<< "  --optimizer-nsga2 / --no-optimizer-nsga2  Toggle NSGA-II non-dominated + crowding-distance elite ranking\n"
		<< "  --repair-mutations        Add diagnostic-guided repair children from measured GA archive entries\n"
		<< "  --repair-top <n>          Number of measured archive parents to repair per generation, default 4\n"
		<< "  --repair-per-candidate <n> Targeted repair children per parent, default 2\n"
		<< "  --rungs <schedule>          Multi-fidelity rung schedule applied inside the optimizer.\n"
		<< "                              Format: name:queries:visit|latency:advance,... e.g.\n"
		<< "                              proxy:4:visit:32,full:16:latency:8,confirm:64:latency:0\n"
		<< "  --rung-surrogate <path>     Exported selector JSON used to propose candidates each generation\n"
		<< "  --rung-surrogate-pool <n>   Pool size sampled and scored by the surrogate per generation\n"
		<< "  --rung-surrogate-top <n>    Top-K surrogate predictions injected as immigrants per generation\n"
		<< "  --refine-thresholds         Run a continuous-parameter (1+lambda)-ES on the top-K archive survivors\n"
		<< "                              with conditional levels. Visit-proxy scoring inside the inner loop.\n"
		<< "  --refine-thresholds-top <n> Number of archive entries to refine, default 4\n"
		<< "  --refine-thresholds-evals <n> Evaluation budget per candidate, default 60\n"
		<< "  --refine-thresholds-sigma <v> Initial step size in normalised [0,1] threshold space, default 0.3\n"
		<< "  --refine-thresholds-seed <s>  RNG seed for the refinement step, default 1337\n"
		<< "  --optimizer-seed <seed>     Seed for optimizer parent choice and mutation\n"
		<< "  --evaluator cpu|cuda        Benchmark backend for schema-search mode; default cuda with CPU fallback\n"
		<< "  --cuda-device <id>          CUDA device id for --evaluator cuda; default 0\n"
		<< "  --cuda-builder lbvh|kdtree|bih|octree|karras_octree|quadtree|regular_grid|hgrid|mixed CUDA builder; LBVH, KDTree, BIH, Octree, KarrasOctree, QuadTree, RegularGrid, HGrid, and MixedTree schemas are implemented\n"
		<< "  --cuda-query-batch <n>      Query batch size for CUDA evaluator; 0 uses all queries\n"
		<< "  --cuda-memory-budget-mb <n> Reject CUDA builds estimated above this memory budget\n"
		<< "  --cuda-knn-backend <auto|gpu_tree_knn|gpu_bruteforce_knn> CUDA KNN backend; KDTree/BIH auto uses tree KNN for k<=16\n"
		<< "  --score-build-weight <v>    Build-time score weight, default 0\n"
		<< "  --score-memory-weight <v>   Memory score weight, default 0\n"
		<< "  --score-imbalance-weight <v> Leaf-imbalance score weight, default 0\n"
		<< "                              These override optional workload JSON scoreWeights.\n"
		<< "  --score-visit-proxy         Score by visited-nodes + alpha*tested-points (deterministic, GPU-free)\n"
		<< "  --score-visit-alpha <v>     Weight for tested-points in visit-proxy score, default 0.1\n"
		<< "  --parallel <n>              Number of concurrent CPU candidate workers in evolutionary mode (default 1; CUDA path stays serial)\n"
		<< "  --no-baselines              Skip canonical single-block control schemas (CPU: quadtree/octree/kdtree/bvh; CUDA also includes GPU-native variants)\n"
		<< "  --include-baselines         Force baseline schemas to be evaluated alongside generated candidates (default on)\n"
		<< "  --output <path>             Write benchmark results as JSON\n"
		<< "  --csv <path>                Write benchmark/search summary rows as CSV\n"
		<< "  --query-trace <path>        Write optional per-query CSV traces for point/schema runs\n"
		<< "  --best-csv <path>           Write best schema rows for schema-search mode\n"
		<< "  --pareto-csv <path>         Write Pareto front (non-dominated rows over latency/build/memory/imbalance)\n"
		<< "  --explain-report <path>     Write Markdown schema explanations with active structures, query behavior, and diagnoses\n"
		<< "  --confirm-seeds <n>         After the main run, re-measure the top-K with N distinct query seeds and\n"
		<< "                              record mean + 95% bootstrap CI on (avg latency, p95 latency, GPU build).\n"
		<< "                              N < 2 disables. The Pareto step uses the seed-averaged mean.\n"
		<< "  --confirm-top <k>           Number of top candidates per (dataset, workload) to re-measure, default 4\n"
		<< "  --measure-repeats <n>       Re-time each candidate's query batch N times (same queries) and record\n"
		<< "                              latency mean/stddev/CV/95% CI on queryMetrics. 1 disables (default).\n"
		<< "  --proxy-correlation-csv <p> Write per-(dataset,workload) Spearman correlation between the visit\n"
		<< "                              proxy and measured latency. The summary is always printed to stdout.\n"
		<< "  --verify-parity             Build canonical schemas on CPU and GPU and compare per-query returned\n"
		<< "                              counts (results/parity_report.csv). Requires CUDA; skipped otherwise.\n"
		<< "  --estimate-prefilter        With --benchmark-top and no rank model, keep the K cheapest candidates\n"
		<< "                              by the zero-build cost estimate instead of an arbitrary first-K prefix.\n"
		<< "  --no-csv                    Disable CSV summary output\n"
		<< "  --queries <count>           Run generated query profile; 0 disables it\n"
		<< "  --knn-k <count>             Neighbor count for generated KNN queries\n"
		<< "  --leaf-micro-indexes        Enable CPU heavy-leaf micro-indexes for point/schema benchmarks\n"
		<< "  --leaf-micro-threshold <n>  Build micro-indexes for leaves with more than n points; default 512\n"
		<< "  --query-seed <seed>         Seed for generated query profile\n"
		<< "  --synthetic-scale <count>   Synthetic point count scale for schema-search mode\n"
		<< "  --no-synthetic              Use only --input datasets in schema-search mode\n"
		<< "  --no-cache                  Read the source point cloud without using/writing .mdspc\n"
		<< "  --rebuild-cache             Read the source point cloud and replace the .mdspc cache\n"
		<< "  --score-cache <path>        Persistent JSONL cache of (schema,dataset,workload)->_score; reuses prior runs\n"
		<< "  --no-score-cache            Disable score cache lookup/write for this run\n"
		<< "  --rebuild-score-cache       Delete the score cache file before this run\n"
		<< "  --run-tests                 Run smoke tests\n"
		<< "  --no-pause                  Skip the final console pause\n";
}

bool AppConfig::wantsTests() const
{
	return _runTests || _mode == "tests";
}

std::string AppConfig::resolveExistingPath(const std::string& configuredPath, const std::string& executablePath)
{
	if (configuredPath.empty())
		return configuredPath;

	const std::filesystem::path path(configuredPath);
	if (pathExists(path))
		return path.lexically_normal().string();

	if (path.is_absolute())
		return configuredPath;

	std::vector<std::filesystem::path> roots;
	addAncestorSearchRoots(roots, std::filesystem::current_path());
	addAncestorSearchRoots(roots, executablePath);

	for (const std::filesystem::path& root : roots)
	{
		const std::filesystem::path candidate = (root / path).lexically_normal();
		if (pathExists(candidate))
			return candidate.string();
	}

	return configuredPath;
}

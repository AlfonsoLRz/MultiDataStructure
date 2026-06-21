#include "stdafx.h"
#include "AppConfig.h"

namespace
{
	// Parses a single rung spec of the form `name:queries:scoreMode:advance`. queries=0 means
	// "use workload default"; scoreMode is `visit` or `latency`; advance=0 means "promote all
	// candidates" (only meaningful for the last rung).
	bool parseRungSpec(const std::string& token, Experiments::RungSpec& outRung)
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

		outRung.name = parts[0];
		try { outRung.queryCountOverride = static_cast<size_t>(std::stoull(parts[1])); }
		catch (...) { return false; }

		const std::string& mode = parts[2];
		if (mode == "visit" || mode == "visit-proxy" || mode == "proxy")
		{
			outRung.useVisitProxy = true;
			if (outRung.visitProxyAlpha <= 0.0)
				outRung.visitProxyAlpha = 0.1;
		}
		else if (mode == "latency")
		{
			outRung.useVisitProxy = false;
		}
		else
		{
			return false;
		}

		if (parts.size() >= 4 && !parts[3].empty())
		{
			try { outRung.advanceTopK = static_cast<size_t>(std::stoull(parts[3])); }
			catch (...) { return false; }
		}
		return true;
	}

	// Parses `--rungs` argument: comma-separated rung specs.
	bool parseRungSchedule(const std::string& argument, std::vector<Experiments::RungSpec>& outRungs)
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

	bool pathExists(const std::filesystem::path& path)
	{
		std::error_code error;
		return std::filesystem::exists(path, error);
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
			const auto alreadyAdded = std::find(roots.begin(), roots.end(), normalized);
			if (alreadyAdded == roots.end())
				roots.push_back(normalized);

			if (!start.has_parent_path() || start == start.parent_path())
				break;

			start = start.parent_path();
		}
	}

	std::string trimCopy(const std::string& value)
	{
		const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); });
		const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) { return std::isspace(c); }).base();
		if (first >= last)
			return {};
		return std::string(first, last);
	}

	std::vector<std::string> splitPathList(const std::string& value)
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
}

AppConfig::AppConfig()
	: pointOptions(defaultPointOptions()),
	  schemaSearchOptions(defaultSchemaSearchOptions())
{
}

PointBenchmark::Options AppConfig::defaultPointOptions()
{
	PointBenchmark::Options options;
	options.inputPath = AppDefaults::POINT_INPUT_PATH;
	options.schemaPath = AppDefaults::POINT_SCHEMA_PATH;
	options.schemaPaths = splitPathList(AppDefaults::POINT_SCHEMA_PATHS);
	options.outputPath = AppDefaults::POINT_OUTPUT_PATH;
	options.csvPath = AppDefaults::POINT_CSV_OUTPUT_PATH;
	options.modelPath = AppDefaults::POINT_MODEL_PATH;
	options.workloadProfilePath = AppDefaults::POINT_WORKLOAD_PROFILE_PATH;
	options.useBinaryCache = AppDefaults::POINT_USE_BINARY_CACHE;
	options.rebuildBinaryCache = AppDefaults::POINT_REBUILD_BINARY_CACHE;
	options.pauseAtEnd = AppDefaults::PAUSE_AT_END;
	options.queryCount = AppDefaults::POINT_QUERY_COUNT;
	options.queryK = AppDefaults::POINT_QUERY_K;
	options.querySeed = AppDefaults::POINT_QUERY_SEED;
	options.enableLeafMicroIndexes = AppDefaults::ENABLE_LEAF_MICRO_INDEXES;
	options.leafMicroIndexThreshold = AppDefaults::LEAF_MICRO_INDEX_THRESHOLD;
	return options;
}

Experiments::SchemaSearchOptions AppConfig::defaultSchemaSearchOptions()
{
	Experiments::SchemaSearchOptions options;
	options.schemaPaths = splitPathList(AppDefaults::SCHEMA_SEARCH_SCHEMA_PATHS);
	options.workloadPaths = splitPathList(AppDefaults::SCHEMA_SEARCH_WORKLOAD_PATHS);
	options.csvPath = AppDefaults::SCHEMA_SEARCH_CSV_PATH;
	options.bestCsvPath = AppDefaults::SCHEMA_SEARCH_BEST_CSV_PATH;
	options.paretoCsvPath = AppDefaults::SCHEMA_SEARCH_PARETO_CSV_PATH;
	options.explainReportPath = AppDefaults::SCHEMA_SEARCH_EXPLAIN_REPORT_PATH;
	options.syntheticScale = AppDefaults::SCHEMA_SEARCH_SYNTHETIC_SCALE;
	options.queryCountOverride = AppDefaults::SCHEMA_SEARCH_QUERY_COUNT;
	options.querySeed = AppDefaults::POINT_QUERY_SEED;
	options.rankModelPath = AppDefaults::SCHEMA_SEARCH_RANK_MODEL_PATH;
	options.generation.count = AppDefaults::SCHEMA_SEARCH_GENERATED_COUNT;
	options.benchmarkTopK = AppDefaults::SCHEMA_SEARCH_BENCHMARK_TOP_K;
	options.evaluator = AppDefaults::SCHEMA_SEARCH_EVALUATOR;
	options.cuda.device = AppDefaults::SCHEMA_SEARCH_CUDA_DEVICE;
	options.cuda.builder = AppDefaults::SCHEMA_SEARCH_CUDA_BUILDER;
	options.autoConditions.proxyCandidateCount = AppDefaults::AUTO_CONDITION_PROXY_CANDIDATES;
	options.autoConditions.proxyPointCap = AppDefaults::AUTO_CONDITION_PROXY_POINTS;
	options.autoConditions.proxyQueryCount = AppDefaults::AUTO_CONDITION_PROXY_QUERIES;
	options.autoConditions.finalTopK = AppDefaults::AUTO_CONDITION_FINAL_TOP_K;
	options.autoConditions.confirmationTopK = AppDefaults::AUTO_CONDITION_CONFIRM_TOP_K;
	options.enableLeafMicroIndexes = AppDefaults::ENABLE_LEAF_MICRO_INDEXES;
	options.leafMicroIndexThreshold = AppDefaults::LEAF_MICRO_INDEX_THRESHOLD;
	options.pauseAtEnd = AppDefaults::PAUSE_AT_END;
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
			config.showHelp = true;
		}
		else if (arg == "--run-tests")
		{
			config.runTests = true;
			config.pointOptions.pauseAtEnd = false;
			config.schemaSearchOptions.pauseAtEnd = false;
		}
		else if (arg == "--gui")
		{
			config.mode = "gui";
			modeWasSpecified = true;
		}
		else if (arg == "--no-pause")
		{
			config.pointOptions.pauseAtEnd = false;
			config.schemaSearchOptions.pauseAtEnd = false;
		}
		else if (arg == "--no-cache")
		{
			config.pointOptions.useBinaryCache = false;
			config.schemaSearchOptions.useBinaryCache = false;
		}
		else if (arg == "--rebuild-cache")
		{
			config.pointOptions.useBinaryCache = true;
			config.pointOptions.rebuildBinaryCache = true;
			config.schemaSearchOptions.useBinaryCache = true;
			config.schemaSearchOptions.rebuildBinaryCache = true;
		}
		else if (arg == "--mode" && i + 1 < argc)
		{
			config.mode = argv[++i];
			modeWasSpecified = true;
		}
		else if (arg == "--input" && i + 1 < argc)
		{
			config.pointOptions.inputPath = argv[++i];
			config.schemaSearchOptions.inputPaths.clear();
			config.schemaSearchOptions.inputPaths.push_back(config.pointOptions.inputPath);
			if (!modeWasSpecified)
				config.mode = "points";
		}
		else if (arg == "--schema" && i + 1 < argc)
		{
			config.pointOptions.schemaPath = argv[++i];
			config.pointOptions.schemaPaths.clear();
			config.schemaSearchOptions.schemaPaths.clear();
			config.schemaSearchOptions.schemaPaths.push_back(config.pointOptions.schemaPath);
		}
		else if (arg == "--schemas" && i + 1 < argc)
		{
			config.pointOptions.schemaPaths = splitPathList(argv[++i]);
			config.schemaSearchOptions.schemaPaths = config.pointOptions.schemaPaths;
		}
		else if (arg == "--workloads" && i + 1 < argc)
		{
			config.schemaSearchOptions.workloadPaths = splitPathList(argv[++i]);
		}
		else if (arg == "--model" && i + 1 < argc)
		{
			config.pointOptions.modelPath = argv[++i];
			config.schemaSearchOptions.rankModelPath = config.pointOptions.modelPath;
		}
		else if (arg == "--rank-model" && i + 1 < argc)
		{
			config.schemaSearchOptions.rankModelPath = argv[++i];
		}
		else if (arg == "--workload-profile" && i + 1 < argc)
		{
			config.pointOptions.workloadProfilePath = argv[++i];
		}
		else if (arg == "--no-synthetic")
		{
			config.schemaSearchOptions.includeSyntheticDatasets = false;
		}
		else if (arg == "--synthetic-scale" && i + 1 < argc)
		{
			config.schemaSearchOptions.syntheticScale = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generate-schemas" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.count = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-only")
		{
			config.schemaSearchOptions.includeConfiguredSchemas = false;
		}
		else if (arg == "--deep-nested-search")
		{
			config.schemaSearchOptions.deepNestedSearch = true;
			config.schemaSearchOptions.evaluator = "cpu";
			config.schemaSearchOptions.rankModelPath.clear();
			config.schemaSearchOptions.includeConfiguredSchemas = false;
			config.schemaSearchOptions.includeBaselineSchemas = true;
			config.schemaSearchOptions.autoConditions.enabled = true;
			config.schemaSearchOptions.autoConditions.proxyCandidateCount = 2048;
			config.schemaSearchOptions.autoConditions.proxyPointCap = 262144;
			config.schemaSearchOptions.autoConditions.proxyQueryCount = 8;
			config.schemaSearchOptions.autoConditions.finalTopK = 128;
			config.schemaSearchOptions.autoConditions.confirmationTopK = 24;
			config.schemaSearchOptions.generation.count = 2048;
			config.schemaSearchOptions.generation.minBlocks = 2;
			config.schemaSearchOptions.generation.maxBlocks = std::max<size_t>(3, config.schemaSearchOptions.generation.maxBlocks);
			config.schemaSearchOptions.generation.conditionalLevels = true;
			config.schemaSearchOptions.generation.conditionalProbability = std::max(0.75, config.schemaSearchOptions.generation.conditionalProbability);
			config.schemaSearchOptions.queryCountOverride = 256;
			if (config.schemaSearchOptions.scoreCachePath.empty())
				config.schemaSearchOptions.scoreCachePath = "results/deep_nested_score_cache.jsonl";
			config.schemaSearchOptions.cuda.device = 0;
			config.schemaSearchOptions.cuda.builder = "mixed";
		}
		else if (arg == "--benchmark-top" && i + 1 < argc)
		{
			config.schemaSearchOptions.benchmarkTopK = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-min-blocks" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.minBlocks = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-max-blocks" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.maxBlocks = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-max-depth" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.maxDepth = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-min-leaf" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.minLeafCapacity = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-max-leaf" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.maxLeafCapacity = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--generated-conditional")
		{
			config.schemaSearchOptions.generation.conditionalLevels = true;
		}
		else if (arg == "--generated-condition-probability" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.conditionalProbability = std::stod(argv[++i]);
		}
		else if (arg == "--generated-adaptive-leaf-capacity")
		{
			config.schemaSearchOptions.generation.adaptiveLeafCapacity = true;
		}
		else if (arg == "--generated-adaptive-leaf-probability" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.adaptiveLeafProbability = std::stod(argv[++i]);
		}
		else if (arg == "--generated-seed" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
		}
		else if (arg == "--generated-schema-dir" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.outputDirectory = argv[++i];
		}
		else if (arg == "--primitive-profile" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.primitiveProfile = argv[++i];
		}
		else if (arg == "--query-minimal-primitives")
		{
			config.schemaSearchOptions.generation.primitiveProfile = "query_minimal_cpu";
		}
		else if (arg == "--cuda-query-full-primitives")
		{
			config.schemaSearchOptions.generation.primitiveProfile = "cuda_query_full";
		}
		else if (arg == "--all-primitives")
		{
			config.schemaSearchOptions.generation.primitiveProfile = "all";
		}
		else if (arg == "--auto-conditions")
		{
			config.schemaSearchOptions.autoConditions.enabled = true;
			config.schemaSearchOptions.generation.conditionalLevels = true;
		}
		else if (arg == "--condition-proxy-candidates" && i + 1 < argc)
		{
			config.schemaSearchOptions.autoConditions.proxyCandidateCount = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--condition-proxy-points" && i + 1 < argc)
		{
			config.schemaSearchOptions.autoConditions.proxyPointCap = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--condition-proxy-queries" && i + 1 < argc)
		{
			config.schemaSearchOptions.autoConditions.proxyQueryCount = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--condition-final-top" && i + 1 < argc)
		{
			config.schemaSearchOptions.autoConditions.finalTopK = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--condition-confirm-top" && i + 1 < argc)
		{
			config.schemaSearchOptions.autoConditions.confirmationTopK = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--condition-output-dir" && i + 1 < argc)
		{
			config.schemaSearchOptions.autoConditions.outputDirectory = argv[++i];
		}
		else if (arg == "--condition-selector-output" && i + 1 < argc)
		{
			config.schemaSearchOptions.autoConditions.selectorOutputPath = argv[++i];
		}
		else if (arg == "--optimize-schemas")
		{
			config.schemaSearchOptions.evolution.enabled = true;
		}
		else if (arg == "--optimizer-generations" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.generations = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--optimizer-population" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.populationSize = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--optimizer-elites" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.eliteCount = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--optimizer-mutation-rate" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.mutationRate = std::stod(argv[++i]);
		}
		else if (arg == "--optimizer-random-fraction" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.randomImmigrationRate = std::stod(argv[++i]);
		}
		else if (arg == "--optimizer-seed" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
		}
		else if (arg == "--optimizer-crossover-rate" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.crossoverRate = std::stod(argv[++i]);
		}
		else if (arg == "--optimizer-nsga2")
		{
			config.schemaSearchOptions.evolution.useNsga2Ranking = true;
		}
		else if (arg == "--no-optimizer-nsga2")
		{
			config.schemaSearchOptions.evolution.useNsga2Ranking = false;
		}
		else if (arg == "--repair-mutations")
		{
			config.schemaSearchOptions.evolution.repairMutations = true;
		}
		else if (arg == "--no-repair-mutations")
		{
			config.schemaSearchOptions.evolution.repairMutations = false;
		}
		else if (arg == "--repair-top" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.repairTopK = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--repair-per-candidate" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.repairPerCandidate = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--rungs" && i + 1 < argc)
		{
			const std::string spec = argv[++i];
			if (!parseRungSchedule(spec, config.schemaSearchOptions.evolution.rungSchedule.rungs))
				throw std::runtime_error("--rungs expects 'name:queries:visit|latency:advance,...' (e.g. proxy:4:visit:32,full:16:latency:8,confirm:64:latency:0)");
		}
		else if (arg == "--rung-surrogate" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.rungSchedule.surrogateModelPath = argv[++i];
		}
		else if (arg == "--rung-surrogate-pool" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.rungSchedule.surrogateCandidatePool = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--rung-surrogate-top" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.rungSchedule.surrogateProposalsPerStep = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--refine-thresholds")
		{
			config.schemaSearchOptions.evolution.refineThresholds = true;
		}
		else if (arg == "--refine-thresholds-top" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.refineThresholdsTopK = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--refine-thresholds-evals" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.refineThresholdsEvaluations = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--refine-thresholds-sigma" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.refineThresholdsSigma0 = std::stod(argv[++i]);
		}
		else if (arg == "--refine-thresholds-seed" && i + 1 < argc)
		{
			config.schemaSearchOptions.evolution.refineThresholdsSeed = static_cast<uint32_t>(std::stoul(argv[++i]));
		}
		else if (arg == "--evaluator" && i + 1 < argc)
		{
			config.schemaSearchOptions.evaluator = argv[++i];
		}
		else if (arg == "--cuda-device" && i + 1 < argc)
		{
			config.schemaSearchOptions.cuda.device = std::stoi(argv[++i]);
		}
		else if (arg == "--cuda-builder" && i + 1 < argc)
		{
			config.schemaSearchOptions.cuda.builder = argv[++i];
		}
		else if (arg == "--cuda-query-batch" && i + 1 < argc)
		{
			config.schemaSearchOptions.cuda.queryBatchSize = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--cuda-memory-budget-mb" && i + 1 < argc)
		{
			config.schemaSearchOptions.cuda.memoryBudgetMb = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--cuda-knn-backend" && i + 1 < argc)
		{
			config.schemaSearchOptions.cuda.knnBackend = argv[++i];
		}
		else if (arg == "--score-build-weight" && i + 1 < argc)
		{
			config.schemaSearchOptions.weights.lambdaBuild = std::stod(argv[++i]);
			config.schemaSearchOptions.scoreWeightsOverride = true;
		}
		else if (arg == "--score-memory-weight" && i + 1 < argc)
		{
			config.schemaSearchOptions.weights.lambdaMemory = std::stod(argv[++i]);
			config.schemaSearchOptions.scoreWeightsOverride = true;
		}
		else if (arg == "--score-imbalance-weight" && i + 1 < argc)
		{
			config.schemaSearchOptions.weights.lambdaImbalance = std::stod(argv[++i]);
			config.schemaSearchOptions.scoreWeightsOverride = true;
		}
		else if (arg == "--output" && i + 1 < argc)
		{
			config.pointOptions.outputPath = argv[++i];
		}
		else if (arg == "--csv" && i + 1 < argc)
		{
			config.pointOptions.csvPath = argv[++i];
			config.schemaSearchOptions.csvPath = config.pointOptions.csvPath;
		}
		else if (arg == "--query-trace" && i + 1 < argc)
		{
			config.pointOptions.queryTracePath = argv[++i];
			config.schemaSearchOptions.queryTracePath = config.pointOptions.queryTracePath;
		}
		else if (arg == "--no-csv")
		{
			config.pointOptions.csvPath.clear();
			config.schemaSearchOptions.csvPath.clear();
			config.schemaSearchOptions.bestCsvPath.clear();
			config.schemaSearchOptions.paretoCsvPath.clear();
		}
		else if (arg == "--best-csv" && i + 1 < argc)
		{
			config.schemaSearchOptions.bestCsvPath = argv[++i];
		}
		else if (arg == "--pareto-csv" && i + 1 < argc)
		{
			config.schemaSearchOptions.paretoCsvPath = argv[++i];
		}
		else if (arg == "--explain-report" && i + 1 < argc)
		{
			config.schemaSearchOptions.explainReportPath = argv[++i];
		}
		else if (arg == "--confirm-seeds" && i + 1 < argc)
		{
			config.schemaSearchOptions.confirmSeeds = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--confirm-top" && i + 1 < argc)
		{
			config.schemaSearchOptions.confirmTopK = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--measure-repeats" && i + 1 < argc)
		{
			config.schemaSearchOptions.measurementRepeats = std::max<size_t>(1, static_cast<size_t>(std::stoull(argv[++i])));
		}
		else if (arg == "--queries" && i + 1 < argc)
		{
			config.pointOptions.queryCount = static_cast<size_t>(std::stoull(argv[++i]));
			config.schemaSearchOptions.queryCountOverride = config.pointOptions.queryCount;
		}
		else if (arg == "--knn-k" && i + 1 < argc)
		{
			config.pointOptions.queryK = static_cast<size_t>(std::stoull(argv[++i]));
			config.schemaSearchOptions.knnKOverride = config.pointOptions.queryK;
		}
		else if (arg == "--leaf-micro-indexes")
		{
			config.pointOptions.enableLeafMicroIndexes = true;
			config.schemaSearchOptions.enableLeafMicroIndexes = true;
		}
		else if (arg == "--leaf-micro-threshold" && i + 1 < argc)
		{
			config.pointOptions.leafMicroIndexThreshold = static_cast<size_t>(std::stoull(argv[++i]));
			config.schemaSearchOptions.leafMicroIndexThreshold = config.pointOptions.leafMicroIndexThreshold;
		}
		else if (arg == "--query-seed" && i + 1 < argc)
		{
			config.pointOptions.querySeed = static_cast<uint32_t>(std::stoul(argv[++i]));
			config.schemaSearchOptions.querySeed = config.pointOptions.querySeed;
			config.schemaSearchOptions.querySeedOverride = true;
		}
		else if (arg == "--score-cache" && i + 1 < argc)
		{
			config.schemaSearchOptions.scoreCachePath = argv[++i];
		}
		else if (arg == "--score-visit-proxy")
		{
			config.schemaSearchOptions.weights.useVisitProxy = true;
			config.schemaSearchOptions.scoreWeightsOverride = true;
		}
		else if (arg == "--score-visit-alpha" && i + 1 < argc)
		{
			config.schemaSearchOptions.weights.visitProxyAlpha = std::stod(argv[++i]);
			config.schemaSearchOptions.scoreWeightsOverride = true;
		}
		else if (arg == "--parallel" && i + 1 < argc)
		{
			config.schemaSearchOptions.parallelDispatch = static_cast<size_t>(std::stoull(argv[++i]));
		}
		else if (arg == "--no-baselines")
		{
			config.schemaSearchOptions.includeBaselineSchemas = false;
		}
		else if (arg == "--include-baselines")
		{
			config.schemaSearchOptions.includeBaselineSchemas = true;
		}
		else if (arg == "--no-score-cache")
		{
			config.schemaSearchOptions.scoreCachePath.clear();
		}
		else if (arg == "--rebuild-score-cache")
		{
			config.schemaSearchOptions.rebuildScoreCache = true;
		}
		else
		{
			throw std::invalid_argument("Unknown or incomplete argument: " + arg);
		}
	}

	const std::string executablePath = argc > 0 ? argv[0] : std::string();
	config.pointOptions.schemaPath = resolveExistingPath(config.pointOptions.schemaPath, executablePath);
	config.pointOptions.modelPath = resolveExistingPath(config.pointOptions.modelPath, executablePath);
	config.schemaSearchOptions.rankModelPath = resolveExistingPath(config.schemaSearchOptions.rankModelPath, executablePath);
	config.pointOptions.workloadProfilePath = resolveExistingPath(config.pointOptions.workloadProfilePath, executablePath);
	for (std::string& schemaPath : config.pointOptions.schemaPaths)
		schemaPath = resolveExistingPath(schemaPath, executablePath);
	for (std::string& schemaPath : config.schemaSearchOptions.schemaPaths)
		schemaPath = resolveExistingPath(schemaPath, executablePath);
	for (std::string& workloadPath : config.schemaSearchOptions.workloadPaths)
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
		<< "  --score-cache <path>        Persistent JSONL cache of (schema,dataset,workload)->score; reuses prior runs\n"
		<< "  --no-score-cache            Disable score cache lookup/write for this run\n"
		<< "  --rebuild-score-cache       Delete the score cache file before this run\n"
		<< "  --run-tests                 Run smoke tests\n"
		<< "  --no-pause                  Skip the final console pause\n";
}

bool AppConfig::wantsTests() const
{
	return runTests || mode == "tests";
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

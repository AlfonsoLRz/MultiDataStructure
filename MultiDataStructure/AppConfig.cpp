#include "stdafx.h"
#include "AppConfig.h"

namespace
{
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
	return options;
}

Experiments::SchemaSearchOptions AppConfig::defaultSchemaSearchOptions()
{
	Experiments::SchemaSearchOptions options;
	options.schemaPaths = splitPathList(AppDefaults::SCHEMA_SEARCH_SCHEMA_PATHS);
	options.workloadPaths = splitPathList(AppDefaults::SCHEMA_SEARCH_WORKLOAD_PATHS);
	options.csvPath = AppDefaults::SCHEMA_SEARCH_CSV_PATH;
	options.bestCsvPath = AppDefaults::SCHEMA_SEARCH_BEST_CSV_PATH;
	options.syntheticScale = AppDefaults::SCHEMA_SEARCH_SYNTHETIC_SCALE;
	options.querySeed = AppDefaults::POINT_QUERY_SEED;
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
		else if (arg == "--benchmark-top" && i + 1 < argc)
		{
			config.schemaSearchOptions.benchmarkTopK = static_cast<size_t>(std::stoull(argv[++i]));
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
		else if (arg == "--generated-seed" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
		}
		else if (arg == "--generated-schema-dir" && i + 1 < argc)
		{
			config.schemaSearchOptions.generation.outputDirectory = argv[++i];
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
		else if (arg == "--score-build-weight" && i + 1 < argc)
		{
			config.schemaSearchOptions.weights.lambdaBuild = std::stod(argv[++i]);
		}
		else if (arg == "--score-memory-weight" && i + 1 < argc)
		{
			config.schemaSearchOptions.weights.lambdaMemory = std::stod(argv[++i]);
		}
		else if (arg == "--score-imbalance-weight" && i + 1 < argc)
		{
			config.schemaSearchOptions.weights.lambdaImbalance = std::stod(argv[++i]);
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
		else if (arg == "--no-csv")
		{
			config.pointOptions.csvPath.clear();
			config.schemaSearchOptions.csvPath.clear();
			config.schemaSearchOptions.bestCsvPath.clear();
		}
		else if (arg == "--best-csv" && i + 1 < argc)
		{
			config.schemaSearchOptions.bestCsvPath = argv[++i];
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
		else if (arg == "--query-seed" && i + 1 < argc)
		{
			config.pointOptions.querySeed = static_cast<uint32_t>(std::stoul(argv[++i]));
			config.schemaSearchOptions.querySeed = config.pointOptions.querySeed;
			config.schemaSearchOptions.querySeedOverride = true;
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
		<< "  --mode gui|points|schema-search|tests\n"
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
		<< "  --benchmark-top <count>     Benchmark only top-k generated/static schemas after surrogate ranking\n"
		<< "  --generated-max-blocks <n>  Max nested blocks for generated schemas\n"
		<< "  --generated-max-depth <n>   Max total generated schema depth\n"
		<< "  --generated-min-leaf <n>    Min generated leaf capacity\n"
		<< "  --generated-max-leaf <n>    Max generated leaf capacity\n"
		<< "  --generated-conditional     Add local node predicates to sampled nested blocks\n"
		<< "  --generated-condition-probability <v> Probability for generated conditional blocks\n"
		<< "  --generated-seed <seed>     Seed for generated schema search space sampling\n"
		<< "  --generated-schema-dir <p>  Directory for generated schema JSON files\n"
		<< "  --optimize-schemas          Run evolutionary mutation search after the initial candidate set\n"
		<< "  --optimizer-generations <n> Evolution generations, default 3\n"
		<< "  --optimizer-population <n>  Mutated/random candidates per generation, default 64\n"
		<< "  --optimizer-elites <n>      Best measured candidates kept as parents, default 6\n"
		<< "  --optimizer-mutation-rate <v> Extra mutation probability per child, default 0.65\n"
		<< "  --optimizer-random-fraction <v> Fraction of each generation sampled randomly, default 0.20\n"
		<< "  --optimizer-seed <seed>     Seed for optimizer parent choice and mutation\n"
		<< "  --score-build-weight <v>    Build-time score weight, default 0\n"
		<< "  --score-memory-weight <v>   Memory score weight, default 0.01\n"
		<< "  --score-imbalance-weight <v> Leaf-imbalance score weight, default 0.01\n"
		<< "  --output <path>             Write benchmark results as JSON\n"
		<< "  --csv <path>                Write benchmark/search summary rows as CSV\n"
		<< "  --best-csv <path>           Write best schema rows for schema-search mode\n"
		<< "  --no-csv                    Disable CSV summary output\n"
		<< "  --queries <count>           Run generated query profile; 0 disables it\n"
		<< "  --knn-k <count>             Neighbor count for generated KNN queries\n"
		<< "  --query-seed <seed>         Seed for generated query profile\n"
		<< "  --synthetic-scale <count>   Synthetic point count scale for schema-search mode\n"
		<< "  --no-synthetic              Use only --input datasets in schema-search mode\n"
		<< "  --no-cache                  Read the source point cloud without using/writing .mdspc\n"
		<< "  --rebuild-cache             Read the source point cloud and replace the .mdspc cache\n"
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

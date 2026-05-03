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
}

AppConfig::AppConfig()
	: pointOptions(defaultPointOptions())
{
}

PointBenchmark::Options AppConfig::defaultPointOptions()
{
	PointBenchmark::Options options;
	options.inputPath = AppDefaults::POINT_INPUT_PATH;
	options.schemaPath = AppDefaults::POINT_SCHEMA_PATH;
	options.outputPath = AppDefaults::POINT_OUTPUT_PATH;
	options.useBinaryCache = AppDefaults::POINT_USE_BINARY_CACHE;
	options.rebuildBinaryCache = AppDefaults::POINT_REBUILD_BINARY_CACHE;
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
		}
		else if (arg == "--no-pause")
		{
			config.pointOptions.pauseAtEnd = false;
		}
		else if (arg == "--no-cache")
		{
			config.pointOptions.useBinaryCache = false;
		}
		else if (arg == "--rebuild-cache")
		{
			config.pointOptions.useBinaryCache = true;
			config.pointOptions.rebuildBinaryCache = true;
		}
		else if (arg == "--mode" && i + 1 < argc)
		{
			config.mode = argv[++i];
			modeWasSpecified = true;
		}
		else if (arg == "--input" && i + 1 < argc)
		{
			config.pointOptions.inputPath = argv[++i];
			if (!modeWasSpecified)
				config.mode = "points";
		}
		else if (arg == "--schema" && i + 1 < argc)
		{
			config.pointOptions.schemaPath = argv[++i];
		}
		else if (arg == "--output" && i + 1 < argc)
		{
			config.pointOptions.outputPath = argv[++i];
		}
		else
		{
			throw std::invalid_argument("Unknown or incomplete argument: " + arg);
		}
	}

	const std::string executablePath = argc > 0 ? argv[0] : std::string();
	config.pointOptions.schemaPath = resolveExistingPath(config.pointOptions.schemaPath, executablePath);

	return config;
}

void AppConfig::printHelp(std::ostream& output)
{
	output
		<< "MultiDataStructure point-cloud benchmark\n"
		<< "  Editable defaults live in MultiDataStructure/AppConfig.h\n"
		<< "  --mode points|tests         Override AppDefaults::DEFAULT_MODE\n"
		<< "  --input <path>              Point cloud path (.las, .ply, .xyz, .csv)\n"
		<< "  --schema <path>             Spatial schema JSON\n"
		<< "  --output <path>             Write benchmark results as JSON\n"
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

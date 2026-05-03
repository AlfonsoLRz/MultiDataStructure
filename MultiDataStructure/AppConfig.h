#pragma once

#include "stdafx.h"
#include "workloads/points/PointBenchmark.h"

namespace AppDefaults
{
	// Supported values: "points" or "tests".
	inline constexpr const char* DEFAULT_MODE = "points";
	inline constexpr const char* POINT_INPUT_PATH = "C:/Datasets/points/Alhambra_100M.las";
	inline constexpr const char* POINT_SCHEMA_PATH = "configs/schemas/octree.json";
	inline constexpr const char* POINT_OUTPUT_PATH = "results/points_default.json";
	inline constexpr bool POINT_USE_BINARY_CACHE = true;
	inline constexpr bool POINT_REBUILD_BINARY_CACHE = false;
	inline constexpr bool RUN_TESTS = false;
	inline constexpr bool PAUSE_AT_END = false;
}

class AppConfig
{
public:
	std::string mode = AppDefaults::DEFAULT_MODE;
	bool runTests = AppDefaults::RUN_TESTS;
	bool showHelp = false;
	PointBenchmark::Options pointOptions;

	AppConfig();

	static AppConfig parse(int argc, char* argv[]);
	static void printHelp(std::ostream& output);
	bool wantsTests() const;

private:
	static PointBenchmark::Options defaultPointOptions();
	static std::string resolveExistingPath(const std::string& configuredPath, const std::string& executablePath);
};

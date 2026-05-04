#pragma once

#include "stdafx.h"
#include "experiments/SchemaSearch.h"
#include "workloads/points/PointBenchmark.h"

namespace AppDefaults
{
	// Supported values: "gui", "points", "schema-search", or "tests".
	inline constexpr const char* DEFAULT_MODE = "gui";
	inline constexpr const char* POINT_INPUT_PATH = "C:/Datasets/points/Alhambra_100M.las";
	//inline constexpr const char* POINT_SCHEMA_PATH = "configs/schemas/octree.json";
	inline constexpr const char* POINT_SCHEMA_PATH = "auto";
	inline constexpr const char* POINT_SCHEMA_PATHS = "";
	inline constexpr const char* POINT_OUTPUT_PATH = "results/points_default.json";
	inline constexpr const char* POINT_CSV_OUTPUT_PATH = "results/points_summary.csv";
	inline constexpr const char* POINT_MODEL_PATH = "models/schema_selector.json";
	inline constexpr const char* POINT_WORKLOAD_PROFILE_PATH = "configs/workloads/mixed.json";
	inline constexpr bool POINT_USE_BINARY_CACHE = true;
	inline constexpr bool POINT_REBUILD_BINARY_CACHE = false;
	inline constexpr size_t POINT_QUERY_COUNT = 16;
	inline constexpr size_t POINT_QUERY_K = 8;
	inline constexpr uint32_t POINT_QUERY_SEED = 1337;
	inline constexpr const char* SCHEMA_SEARCH_SCHEMA_PATHS = "configs/schemas/quadtree.json;configs/schemas/octree.json;configs/schemas/kdtree.json;configs/schemas/quadtree_octree.json;configs/schemas/octree_kdtree.json;configs/schemas/urban_hybrid.json";
	inline constexpr const char* SCHEMA_SEARCH_WORKLOAD_PATHS = "configs/workloads/range_heavy.json;configs/workloads/knn_heavy.json;configs/workloads/mixed.json";
	inline constexpr const char* SCHEMA_SEARCH_CSV_PATH = "results/schema_search.csv";
	inline constexpr const char* SCHEMA_SEARCH_BEST_CSV_PATH = "results/schema_search_best.csv";
	inline constexpr size_t SCHEMA_SEARCH_SYNTHETIC_SCALE = 512;
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
	Experiments::SchemaSearchOptions schemaSearchOptions;

	AppConfig();

	static AppConfig parse(int argc, char* argv[]);
	static void printHelp(std::ostream& output);
	bool wantsTests() const;

private:
	static PointBenchmark::Options defaultPointOptions();
	static Experiments::SchemaSearchOptions defaultSchemaSearchOptions();
	static std::string resolveExistingPath(const std::string& configuredPath, const std::string& executablePath);
};

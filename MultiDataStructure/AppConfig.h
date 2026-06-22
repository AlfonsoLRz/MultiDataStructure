#pragma once

#include "stdafx.h"
#include "experiments/SchemaSearch.h"
#include "workloads/points/PointBenchmark.h"

namespace AppDefaults
{
	// Supported values: "gui", "points", "schema-search", or "tests".
	inline constexpr const char* DEFAULT_MODE = "gui";
	inline constexpr const char* POINT_INPUT_PATH = "C:/Datasets/points/Alhambra_100M.las";
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
	inline constexpr bool ENABLE_LEAF_MICRO_INDEXES = false;
	inline constexpr size_t LEAF_MICRO_INDEX_THRESHOLD = 512;
	inline constexpr const char* SCHEMA_SEARCH_SCHEMA_PATHS = "configs/schemas/quadtree.json;configs/schemas/octree.json;configs/schemas/kdtree.json;configs/schemas/bvh.json;configs/schemas/quadtree_octree.json;configs/schemas/octree_kdtree.json;configs/schemas/urban_hybrid.json";
	inline constexpr const char* SCHEMA_SEARCH_WORKLOAD_PATHS = "configs/workloads/volume_small_medium.json";
	inline constexpr const char* SCHEMA_SEARCH_CSV_PATH = "results/schema_search.csv";
	inline constexpr const char* SCHEMA_SEARCH_BEST_CSV_PATH = "results/schema_search_best.csv";
	inline constexpr const char* SCHEMA_SEARCH_PARETO_CSV_PATH = "results/schema_search_pareto.csv";
	inline constexpr const char* SCHEMA_SEARCH_EXPLAIN_REPORT_PATH = "";
	inline constexpr size_t SCHEMA_SEARCH_SYNTHETIC_SCALE = 512;
	inline constexpr size_t SCHEMA_SEARCH_QUERY_COUNT = 64;
	inline constexpr const char* SCHEMA_SEARCH_RANK_MODEL_PATH = "models/schema_selector.json";
	inline constexpr size_t SCHEMA_SEARCH_GENERATED_COUNT = 256;
	inline constexpr size_t SCHEMA_SEARCH_BENCHMARK_TOP_K = 32;
	inline constexpr const char* SCHEMA_SEARCH_EVALUATOR = "cuda";
	inline constexpr int SCHEMA_SEARCH_CUDA_DEVICE = 0;
	inline constexpr const char* SCHEMA_SEARCH_CUDA_BUILDER = "mixed";
	inline constexpr size_t AUTO_CONDITION_PROXY_CANDIDATES = 256;
	inline constexpr size_t AUTO_CONDITION_PROXY_POINTS = 262144;
	inline constexpr size_t AUTO_CONDITION_PROXY_QUERIES = 8;
	inline constexpr size_t AUTO_CONDITION_FINAL_TOP_K = 16;
	inline constexpr size_t AUTO_CONDITION_CONFIRM_TOP_K = 4;
	inline constexpr bool RUN_TESTS = false;
	inline constexpr bool PAUSE_AT_END = false;
}

class AppConfig
{
public:
	std::string	_mode = AppDefaults::DEFAULT_MODE;
	bool	_runTests = AppDefaults::RUN_TESTS;
	bool	_showHelp = false;
	PointBenchmark::Options	_pointOptions;
	Experiments::SchemaSearchOptions	_schemaSearchOptions;

	AppConfig();

	static AppConfig parse(int argc, char* argv[]);
	static void printHelp(std::ostream& output);
	bool wantsTests() const;

private:
	static PointBenchmark::Options defaultPointOptions();
	static Experiments::SchemaSearchOptions defaultSchemaSearchOptions();
	static std::string resolveExistingPath(const std::string& configuredPath, const std::string& executablePath);
};

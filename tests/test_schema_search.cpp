#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/experiments/SchemaSearch.h"
#include "../MultiDataStructure/workloads/points/MixedTree.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		bool nearlyEqual(double left, double right, double epsilon = 0.0001)
		{
			return std::abs(left - right) <= epsilon;
		}
	}

	void runSchemaSearchTests()
	{
		const char* workloadJson = R"json(
		{
		  "name": "focused_knn",
		  "queries": {
		    "aabb_range": 0.2,
		    "radius": 0.1,
		    "knn": 0.7
		  },
		  "knnK": 12,
		  "numQueries": 42,
		  "querySeed": 99,
		  "queryScales": {
		    "aabb_range": {
		      "min": 0.02,
		      "max": 0.25
		    },
		    "radius": {
		      "min": 0.03,
		      "max": 0.12
		    }
		  }
		}
		)json";

		const Experiments::WorkloadProfile profile = Experiments::parseWorkloadProfile(workloadJson, "focused_knn");
		expect(profile.name == "focused_knn", "schema search parses workload name");
		expect(nearlyEqual(profile.rangeWeight, 0.2), "schema search parses range workload weight");
		expect(nearlyEqual(profile.radiusWeight, 0.1), "schema search parses radius workload weight");
		expect(nearlyEqual(profile.knnWeight, 0.7), "schema search parses knn workload weight");
		expect(profile.knnK == 12, "schema search parses workload knn k");
		expect(profile.numQueries == 42, "schema search parses workload query count");
		expect(profile.querySeed == 99, "schema search parses workload query seed");
		expect(nearlyEqual(profile.rangeScaleMin, 0.02), "schema search parses range scale min");
		expect(nearlyEqual(profile.rangeScaleMax, 0.25), "schema search parses range scale max");
		expect(nearlyEqual(profile.radiusScaleMin, 0.03), "schema search parses radius scale min");
		expect(nearlyEqual(profile.radiusScaleMax, 0.12), "schema search parses radius scale max");

		Experiments::BuildMetrics buildMetrics;
		buildMetrics.buildTimeMs = 10.0;
		buildMetrics.memoryEstimateBytes = 2 * 1024 * 1024;
		buildMetrics.averageLeafOccupancy = 4.0;
		buildMetrics.maxLeafOccupancy = 12;

		Experiments::QueryMetrics queryMetrics;
		queryMetrics.averageLatencyMs = 2.0;

		double memoryMb = 0.0;
		double imbalancePenalty = 0.0;
		Experiments::ScoreWeights scoreWeights;
		scoreWeights.lambdaBuild = 0.001;
		scoreWeights.lambdaMemory = 0.01;
		scoreWeights.lambdaImbalance = 0.01;
		const double score = Experiments::computeSchemaSearchScore(
			buildMetrics,
			queryMetrics,
			scoreWeights,
			memoryMb,
			imbalancePenalty);

		expect(nearlyEqual(memoryMb, 2.0), "schema search score computes memory MB");
		expect(nearlyEqual(imbalancePenalty, 3.0), "schema search score computes imbalance penalty");
		expect(nearlyEqual(score, 2.06), "schema search score combines latency, build, memory, and imbalance");

		double defaultMemoryMb = 0.0;
		double defaultImbalancePenalty = 0.0;
		const double defaultScore = Experiments::computeSchemaSearchScore(
			buildMetrics,
			queryMetrics,
			Experiments::ScoreWeights{},
			defaultMemoryMb,
			defaultImbalancePenalty);
		expect(nearlyEqual(defaultScore, 2.0), "schema search default score is query-only");

		std::vector<Experiments::SchemaSearchRecord> records(3);
		records[0].datasetName = "flat";
		records[0].workloadName = "mixed";
		records[0].schemaName = "slow";
		records[0].score = 3.0;
		records[1].datasetName = "flat";
		records[1].workloadName = "mixed";
		records[1].schemaName = "fast";
		records[1].score = 1.0;
		records[2].datasetName = "facade";
		records[2].workloadName = "mixed";
		records[2].schemaName = "only";
		records[2].score = 2.0;

		const std::vector<Experiments::SchemaSearchRecord> best = Experiments::selectBestRecords(records);
		expect(best.size() == 2, "schema search picks one best record per dataset/workload");
		expect(best[0].schemaName == "fast", "schema search keeps lowest score as best schema");
		expect(best[1].schemaName == "only", "schema search keeps independent dataset/workload groups");

		const Experiments::EvaluatorResolution cudaResolution = Experiments::resolveSchemaSearchEvaluator("cuda", true);
		expect(cudaResolution.usingCuda, "schema search resolver keeps available CUDA");
		expect(cudaResolution.evaluator == "cuda", "schema search resolver returns cuda evaluator");

		const Experiments::EvaluatorResolution fallbackResolution = Experiments::resolveSchemaSearchEvaluator("cuda", false, "missing device");
		expect(!fallbackResolution.usingCuda, "schema search resolver disables unavailable CUDA");
		expect(fallbackResolution.fellBackToCpu, "schema search resolver marks CPU fallback");
		expect(fallbackResolution.evaluator == "cpu", "schema search resolver falls back to CPU");
		expect(fallbackResolution.warning.find("missing device") != std::string::npos, "schema search resolver includes CUDA error in warning");

		Experiments::SchemaGenerationOptions generation;
		generation.count = 12;
		generation.maxBlocks = 3;
		generation.maxDepth = 8;
		generation.minLeafCapacity = 32;
		generation.maxLeafCapacity = 512;
		generation.seed = 11;
		generation.outputDirectory.clear();

		const std::vector<Experiments::SchemaCandidate> generated = Experiments::generateSchemaCandidates(generation);
		expect(generated.size() == generation.count, "schema generator creates requested candidate count");
		for (const Experiments::SchemaCandidate& candidate : generated)
		{
			expect(candidate.generated, "schema generator marks generated candidates");
			expect(candidate.config.totalLevels() <= generation.maxDepth, "schema generator respects max depth");
			expect(!candidate.config.levels.empty(), "schema generator creates non-empty level schedules");
			expect(candidate.path.rfind("generated:", 0) == 0, "schema generator uses generated pseudo path");
		}

		const std::filesystem::path tempRoot = std::filesystem::temp_directory_path() / "mdspc_schema_search_tests";
		std::filesystem::create_directories(tempRoot);
		const std::filesystem::path csvPath = tempRoot / "prepared_queries.csv";
		const std::filesystem::path bestCsvPath = tempRoot / "prepared_queries_best.csv";

		Experiments::SchemaSearchOptions options;
		options.schemaPaths = { "configs/schemas/quadtree.json" };
		options.workloadPaths = { "configs/workloads/mixed.json" };
		options.csvPath = csvPath.string();
		options.bestCsvPath = bestCsvPath.string();
		options.syntheticScale = 64;
		options.queryCountOverride = 9;
		options.evaluator = "cpu";
		options.pauseAtEnd = false;
		const int exitCode = Experiments::runSchemaSearch(options);
		expect(exitCode == 0, "schema search smoke run succeeds");

		std::ifstream csv(csvPath);
		expect(csv.is_open(), "schema search writes prepared-query CSV");
		std::string header;
		std::getline(csv, header);
		std::string row;
		std::getline(csv, row);
		expect(!row.empty(), "schema search prepared-query CSV has at least one row");
		auto splitCsv = [](const std::string& line) {
			std::vector<std::string> values;
			std::string current;
			for (const char c : line)
			{
				if (c == ',')
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
		};
		const std::vector<std::string> columns = splitCsv(header);
		const std::vector<std::string> values = splitCsv(row);
		auto columnIndex = [&columns](const std::string& name) {
			const auto found = std::find(columns.begin(), columns.end(), name);
			return found == columns.end()
				? std::numeric_limits<size_t>::max()
				: static_cast<size_t>(std::distance(columns.begin(), found));
		};
		const size_t totalQueriesColumn = columnIndex("total_queries");
		const size_t rangeQueriesColumn = columnIndex("range_queries");
		const size_t radiusQueriesColumn = columnIndex("radius_queries");
		const size_t knnQueriesColumn = columnIndex("knn_queries");
		expect(totalQueriesColumn < values.size() && rangeQueriesColumn < values.size() &&
			radiusQueriesColumn < values.size() && knnQueriesColumn < values.size(),
			"schema search prepared-query CSV includes query count columns");
		const size_t totalQueries = static_cast<size_t>(std::stoull(values[totalQueriesColumn]));
		const size_t rangeQueries = static_cast<size_t>(std::stoull(values[rangeQueriesColumn]));
		const size_t radiusQueries = static_cast<size_t>(std::stoull(values[radiusQueriesColumn]));
		const size_t knnQueries = static_cast<size_t>(std::stoull(values[knnQueriesColumn]));
		expect(totalQueries == 9, "schema search prepared workload keeps query override count");
		expect(rangeQueries + radiusQueries + knnQueries == totalQueries, "schema search prepared CPU query counts match total");

		std::string cudaError;
		if (PointGpu::MixedTree::isAvailable(&cudaError))
		{
			Experiments::SchemaSearchOptions cudaOptions;
			cudaOptions.schemaPaths = { "configs/schemas/gpu_mixed_all.json" };
			cudaOptions.workloadPaths = { "configs/workloads/volume_small_medium.json" };
			cudaOptions.csvPath.clear();
			cudaOptions.bestCsvPath.clear();
			cudaOptions.syntheticScale = 64;
			cudaOptions.queryCountOverride = 4;
			cudaOptions.evaluator = "cuda";
			cudaOptions.cuda.device = 0;
			cudaOptions.cuda.builder = "mixed";
			cudaOptions.pauseAtEnd = false;
			expect(Experiments::runSchemaSearch(cudaOptions) == 0, "schema search CUDA mixed smoke run succeeds");

			const std::filesystem::path knnWorkloadPath = tempRoot / "cuda_knn_workload.json";
			{
				std::ofstream workload(knnWorkloadPath);
				workload << R"json({
  "name": "cuda_knn_only",
  "queries": {
    "aabb_range": 0.0,
    "radius": 0.0,
    "knn": 1.0
  },
  "knnK": 7,
  "numQueries": 5
})json";
			}

			const std::filesystem::path cudaKnnCsvPath = tempRoot / "cuda_knn_queries.csv";
			Experiments::SchemaSearchOptions cudaKnnOptions;
			cudaKnnOptions.schemaPaths = { "configs/schemas/gpu_mixed_all.json" };
			cudaKnnOptions.workloadPaths = { knnWorkloadPath.string() };
			cudaKnnOptions.csvPath = cudaKnnCsvPath.string();
			cudaKnnOptions.bestCsvPath.clear();
			cudaKnnOptions.syntheticScale = 64;
			cudaKnnOptions.queryCountOverride = 5;
			cudaKnnOptions.evaluator = "cuda";
			cudaKnnOptions.cuda.device = 0;
			cudaKnnOptions.cuda.builder = "mixed";
			cudaKnnOptions.pauseAtEnd = false;
			expect(Experiments::runSchemaSearch(cudaKnnOptions) == 0, "schema search CUDA KNN workload succeeds");

			std::ifstream cudaKnnCsv(cudaKnnCsvPath);
			expect(cudaKnnCsv.is_open(), "schema search writes CUDA KNN CSV");
			std::string cudaHeader;
			std::string cudaRow;
			std::getline(cudaKnnCsv, cudaHeader);
			std::getline(cudaKnnCsv, cudaRow);
			const std::vector<std::string> cudaColumns = splitCsv(cudaHeader);
			const std::vector<std::string> cudaValues = splitCsv(cudaRow);
			auto cudaColumnIndex = [&cudaColumns](const std::string& name) {
				const auto found = std::find(cudaColumns.begin(), cudaColumns.end(), name);
				return found == cudaColumns.end()
					? std::numeric_limits<size_t>::max()
					: static_cast<size_t>(std::distance(cudaColumns.begin(), found));
			};
			const size_t cudaTotalQueriesColumn = cudaColumnIndex("total_queries");
			const size_t cudaKnnQueriesColumn = cudaColumnIndex("knn_queries");
			expect(cudaTotalQueriesColumn < cudaValues.size() && cudaKnnQueriesColumn < cudaValues.size(),
				"schema search CUDA KNN CSV includes query count columns");
			expect(static_cast<size_t>(std::stoull(cudaValues[cudaTotalQueriesColumn])) == 5,
				"schema search CUDA KNN workload keeps query override count");
			expect(static_cast<size_t>(std::stoull(cudaValues[cudaKnnQueriesColumn])) == 5,
				"schema search CUDA KNN workload measures KNN on GPU");
		}
		else
		{
			std::cout << "Schema-search CUDA smoke skipped: " << cudaError << '\n';
		}
	}
}

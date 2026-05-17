#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/experiments/EvaluationCache.h"
#include "../MultiDataStructure/experiments/SchemaSearch.h"
#include "../MultiDataStructure/experiments/ThresholdRefiner.h"
#include "../MultiDataStructure/workloads/points/MixedTree.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

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

		// Visit-count surrogate substitutes deterministic counters for noisy latency.
		Experiments::QueryMetrics visitMetrics;
		visitMetrics.averageLatencyMs = 99.0;        // ignored when surrogate is on
		visitMetrics.averageVisitedNodes = 25.0;
		visitMetrics.averageTestedPoints = 100.0;
		Experiments::ScoreWeights visitWeights;
		visitWeights.useVisitProxy = true;
		visitWeights.visitProxyAlpha = 0.2;
		double visitMemoryMb = 0.0;
		double visitImbalance = 0.0;
		const double visitScore = Experiments::computeSchemaSearchScore(
			Experiments::BuildMetrics{},
			visitMetrics,
			visitWeights,
			visitMemoryMb,
			visitImbalance);
		expect(nearlyEqual(visitScore, 45.0), "visit-proxy score combines visited nodes and tested points");

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

		Experiments::SchemaGenerationOptions nestedGeneration = generation;
		nestedGeneration.count = 8;
		nestedGeneration.minBlocks = 2;
		nestedGeneration.maxBlocks = 2;
		const std::vector<Experiments::SchemaCandidate> nestedGenerated = Experiments::generateSchemaCandidates(nestedGeneration);
		expect(nestedGenerated.size() == nestedGeneration.count, "schema generator creates requested minimum-block candidates");
		for (const Experiments::SchemaCandidate& candidate : nestedGenerated)
			expect(candidate.config.levels.size() >= 2, "schema generator respects generated min blocks");

		const PointCloud flatCloud = SyntheticPointClouds::generateFlatTerrain(128, 80.0f, 80.0f, 0.02f, 7);
		const PointCloud tallCloud = SyntheticPointClouds::generateFacade(128, 80.0f, 40.0f, 0.05f, 8);
		const PointCloud sparseDenseCloud = SyntheticPointClouds::generateSparseDenseMixture(96, 96, 9);
		const Experiments::ConditionDomain flatDomainA = Experiments::estimateConditionDomain(flatCloud, 64);
		const Experiments::ConditionDomain flatDomainB = Experiments::estimateConditionDomain(flatCloud, 64);
		const Experiments::ConditionDomain tallDomain = Experiments::estimateConditionDomain(tallCloud, 64);
		const Experiments::ConditionDomain sparseDenseDomain = Experiments::estimateConditionDomain(sparseDenseCloud, 64);
		expect(flatDomainA.estimatedFromCloud, "condition domain records cloud estimate");
		expect(flatDomainA.samplePoints == 64, "condition domain respects sample cap");
		expect(flatDomainA.pointThresholds == flatDomainB.pointThresholds, "condition domain point thresholds are deterministic");
		expect(flatDomainA.heightRatioThresholds == flatDomainB.heightRatioThresholds, "condition domain height thresholds are deterministic");
		expect(!flatDomainA.pointThresholds.empty(), "condition domain estimates point thresholds");
		expect(!flatDomainA.heightRatioThresholds.empty(), "condition domain estimates height thresholds");
		expect(!tallDomain.heightRatioThresholds.empty(), "condition domain estimates tall-cloud height thresholds");
		expect(!sparseDenseDomain.densityThresholds.empty(), "condition domain estimates density thresholds");

		Experiments::SchemaGenerationOptions conditionalGeneration;
		conditionalGeneration.count = 10;
		conditionalGeneration.maxBlocks = 3;
		conditionalGeneration.maxDepth = 6;
		conditionalGeneration.minLeafCapacity = 16;
		conditionalGeneration.maxLeafCapacity = 256;
		conditionalGeneration.conditionalLevels = true;
		conditionalGeneration.conditionalProbability = 1.0;
		conditionalGeneration.seed = 12;
		conditionalGeneration.outputDirectory.clear();
		const std::vector<Experiments::SchemaCandidate> conditionalGenerated =
			Experiments::generateSchemaCandidates(conditionalGeneration, &flatDomainA);
		const std::vector<Experiments::SchemaCandidate> conditionalGeneratedAgain =
			Experiments::generateSchemaCandidates(conditionalGeneration, &flatDomainA);
		expect(conditionalGenerated.size() == conditionalGeneration.count, "domain-aware schema generator creates requested candidates");
		expect(conditionalGeneratedAgain.size() == conditionalGenerated.size(), "domain-aware schema generator is deterministic for fixed seed");
		bool sawNumericCondition = false;
		for (size_t candidateIndex = 0; candidateIndex < conditionalGenerated.size(); ++candidateIndex)
		{
			const Experiments::SchemaCandidate& candidate = conditionalGenerated[candidateIndex];
			expect(candidate.name == conditionalGeneratedAgain[candidateIndex].name,
				"domain-aware schema generator repeats candidate order for fixed seed");
			for (size_t levelIndex = 1; levelIndex < candidate.config.levels.size(); ++levelIndex)
			{
				if (!candidate.config.levels[levelIndex].condition.empty())
				{
					sawNumericCondition = true;
					expect(candidate.config.levels[levelIndex].condition.minPoints.has_value(),
						"domain-aware generated condition includes numeric minPoints");
				}
			}
		}
		expect(sawNumericCondition, "domain-aware schema generator emits conditional levels");

		const std::filesystem::path tempRoot = std::filesystem::temp_directory_path() / "mdspc_schema_search_tests";
		std::filesystem::create_directories(tempRoot);
		const std::filesystem::path csvPath = tempRoot / "prepared_queries.csv";
		const std::filesystem::path bestCsvPath = tempRoot / "prepared_queries_best.csv";

		Experiments::SchemaSearchOptions options;
		options.schemaPaths = { "configs/schemas/quadtree.json" };
		options.workloadPaths = { "configs/workloads/mixed.json" };
		options.includeBaselineSchemas = false;
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
		const size_t nestedFractionColumn = columnIndex("nested_active_fraction");
		const size_t baselineSchemaColumn = columnIndex("best_baseline_schema");
		expect(totalQueriesColumn < values.size() && rangeQueriesColumn < values.size() &&
			radiusQueriesColumn < values.size() && knnQueriesColumn < values.size(),
			"schema search prepared-query CSV includes query count columns");
		expect(nestedFractionColumn < values.size() && baselineSchemaColumn < values.size(),
			"schema search CSV includes nested accounting and baseline-normalized columns");
		const size_t totalQueries = static_cast<size_t>(std::stoull(values[totalQueriesColumn]));
		const size_t rangeQueries = static_cast<size_t>(std::stoull(values[rangeQueriesColumn]));
		const size_t radiusQueries = static_cast<size_t>(std::stoull(values[radiusQueriesColumn]));
		const size_t knnQueries = static_cast<size_t>(std::stoull(values[knnQueriesColumn]));
		expect(totalQueries == 9, "schema search prepared workload keeps query override count");
		expect(rangeQueries + radiusQueries + knnQueries == totalQueries, "schema search prepared CPU query counts match total");

		const std::filesystem::path autoCsvPath = tempRoot / "auto_conditions.csv";
		const std::filesystem::path autoBestCsvPath = tempRoot / "auto_conditions_best.csv";
		Experiments::SchemaSearchOptions autoOptions;
		autoOptions.schemaPaths = { "configs/schemas/quadtree.json" };
		autoOptions.workloadPaths = { "configs/workloads/volume_small_medium.json" };
		autoOptions.includeBaselineSchemas = false;
		autoOptions.csvPath = autoCsvPath.string();
		autoOptions.bestCsvPath = autoBestCsvPath.string();
		autoOptions.syntheticScale = 64;
		autoOptions.queryCountOverride = 6;
		autoOptions.evaluator = "cpu";
		autoOptions.pauseAtEnd = false;
		autoOptions.rankModelPath.clear();
		autoOptions.autoConditions.enabled = true;
		autoOptions.autoConditions.proxyCandidateCount = 5;
		autoOptions.autoConditions.proxyPointCap = 64;
		autoOptions.autoConditions.proxyQueryCount = 2;
		autoOptions.autoConditions.finalTopK = 3;
		autoOptions.autoConditions.confirmationTopK = 2;
		autoOptions.autoConditions.outputDirectory = (tempRoot / "auto_condition_schemas").string();
		autoOptions.autoConditions.selectorOutputPath = (tempRoot / "auto_condition_selector.json").string();
		autoOptions.generation.outputDirectory = (tempRoot / "generated_auto_conditions").string();
		expect(Experiments::runSchemaSearch(autoOptions) == 0, "schema search auto-condition staged run succeeds");

		std::ifstream autoCsv(autoCsvPath);
		expect(autoCsv.is_open(), "schema search writes auto-condition CSV");
		std::string autoHeader;
		std::getline(autoCsv, autoHeader);
		std::vector<std::string> autoRows;
		while (std::getline(autoCsv, row))
		{
			if (!row.empty())
				autoRows.push_back(row);
		}
		expect(autoRows.size() == autoOptions.autoConditions.confirmationTopK * 3,
			"auto-condition CSV only contains confirmation-stage rows");
		expect(autoHeader.find("conditional_levels") != std::string::npos &&
			autoHeader.find("condition_fields") != std::string::npos &&
			autoHeader.find("condition_summary") != std::string::npos,
			"auto-condition CSV appends condition summary columns");
		expect(std::filesystem::exists(autoOptions.autoConditions.selectorOutputPath),
			"auto-condition tuning writes measured selector artifact");

		std::string cudaError;
		if (PointGpu::MixedTree::isAvailable(&cudaError))
		{
			Experiments::SchemaSearchOptions cudaOptions;
			cudaOptions.schemaPaths = { "configs/schemas/gpu_mixed_all.json" };
			cudaOptions.workloadPaths = { "configs/workloads/volume_small_medium.json" };
			cudaOptions.includeBaselineSchemas = false;
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
			cudaKnnOptions.includeBaselineSchemas = false;
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

		// Persistent score cache round-trip: write a record, reopen, read it back.
		{
			const std::filesystem::path cacheRoot = std::filesystem::temp_directory_path() / "mdspc_evaluation_cache_test";
			std::filesystem::create_directories(cacheRoot);
			const std::filesystem::path cachePath = cacheRoot / "cache.jsonl";
			std::error_code rmError;
			std::filesystem::remove(cachePath, rmError);

			Experiments::WorkloadProfile workload;
			workload.name = "mixed";
			workload.rangeWeight = 0.4;
			workload.radiusWeight = 0.3;
			workload.knnWeight = 0.3;
			workload.numQueries = 16;
			workload.knnK = 8;
			workload.querySeed = 42;
			workload.rangeScaleMin = 0.01;
			workload.rangeScaleMax = 0.05;
			workload.radiusScaleMin = 0.01;
			workload.radiusScaleMax = 0.04;

			Experiments::ScoreWeights weights;
			const glm::vec3 bboxMin(0.0f, 0.0f, 0.0f);
			const glm::vec3 bboxMax(1.0f, 1.0f, 1.0f);
			const Experiments::EvaluationCacheKey key = Experiments::makeEvaluationCacheKey(
				"oct4_l64_kd3_l32", "synthetic_flat", 1234, bboxMin, bboxMax, workload, "cpu", "", weights);

			Experiments::SchemaSearchRecord record;
			record.datasetName = "synthetic_flat";
			record.workloadName = workload.name;
			record.schemaName = "oct4_l64_kd3_l32";
			record.score = 1.75;
			record.buildMetrics.buildTimeMs = 42.0;
			record.buildMetrics.numNodes = 17;
			record.queryMetrics.averageLatencyMs = 0.5;
			record.queryMetrics.averageVisitedNodes = 12.5;
			record.queryMetrics.totalQueries = 16;
			record.backend = "cpu";
			record.isBaseline = true;
			record.activeStructureTypes = 2;
			record.nestedActiveFraction = 0.25;
			record.activeStructureSummary = "ot:nodes=10|points=900;kd:nodes=4|points=100";

			{
				Experiments::EvaluationCache cache;
				expect(cache.open(cachePath.string(), false), "evaluation cache opens for write");
				expect(cache.entryCount() == 0, "evaluation cache starts empty on fresh file");
				cache.put(key, record);
				expect(cache.entryCount() == 1, "evaluation cache reports one entry after put");
			}

			{
				Experiments::EvaluationCache cache;
				expect(cache.open(cachePath.string(), false), "evaluation cache reopens existing file");
				expect(cache.entryCount() == 1, "evaluation cache loads prior entry from disk");

				Experiments::SchemaSearchRecord roundTrip;
				roundTrip.datasetName = "caller-set name";
				roundTrip.schemaName = "caller-set schema";
				roundTrip.isBaseline = true;
				roundTrip.weights.lambdaMemory = 0.5;
				roundTrip.pointFeatures.numPoints = 999;
				expect(cache.tryGet(key, roundTrip), "evaluation cache hit on identical key");
				expect(nearlyEqual(roundTrip.score, 1.75), "cache restores score");
				expect(nearlyEqual(roundTrip.buildMetrics.buildTimeMs, 42.0), "cache restores build metrics");
				expect(roundTrip.buildMetrics.numNodes == 17, "cache restores build node count");
				expect(nearlyEqual(roundTrip.queryMetrics.averageVisitedNodes, 12.5), "cache restores visit counts");
				expect(roundTrip.backend == "cpu", "cache restores backend");
				expect(roundTrip.isBaseline, "cache preserves caller baseline marker");
				expect(roundTrip.activeStructureTypes == 2, "cache restores active structure type count");
				expect(nearlyEqual(roundTrip.nestedActiveFraction, 0.25), "cache restores nested active fraction");
				expect(roundTrip.activeStructureSummary.find("kd:nodes") != std::string::npos,
					"cache restores active structure summary");
				expect(roundTrip.datasetName == "caller-set name", "cache preserves caller dataset name");
				expect(roundTrip.schemaName == "caller-set schema", "cache preserves caller schema name");
				expect(nearlyEqual(roundTrip.weights.lambdaMemory, 0.5), "cache preserves caller weights");
				expect(roundTrip.pointFeatures.numPoints == 999, "cache preserves caller point features");
				expect(cache.hitCount() == 1, "cache counts hit");

				Experiments::WorkloadProfile differentWorkload = workload;
				differentWorkload.querySeed = 7;
				const Experiments::EvaluationCacheKey missKey = Experiments::makeEvaluationCacheKey(
					"oct4_l64_kd3_l32", "synthetic_flat", 1234, bboxMin, bboxMax, differentWorkload, "cpu", "", weights);
				Experiments::SchemaSearchRecord missRecord;
				expect(!cache.tryGet(missKey, missRecord), "cache miss when workload key changes");
				expect(cache.missCount() == 1, "cache counts miss");
			}

			std::filesystem::remove_all(cacheRoot, rmError);
		}

		// Baseline injection: when includeBaselineSchemas is on (the default), the canonical
		// single-block schemas appear in the measured records even when the user asked for
		// generated-only mode.
		{
			const std::filesystem::path baselineRoot = std::filesystem::temp_directory_path() / "mdspc_baseline_test";
			std::filesystem::create_directories(baselineRoot);
			const std::filesystem::path baselineCsv = baselineRoot / "baseline_search.csv";

			Experiments::SchemaSearchOptions baselineOptions;
			baselineOptions.workloadPaths = { "configs/workloads/volume_small_medium.json" };
			baselineOptions.csvPath = baselineCsv.string();
			baselineOptions.bestCsvPath.clear();
			baselineOptions.syntheticScale = 64;
			baselineOptions.queryCountOverride = 4;
			baselineOptions.evaluator = "cpu";
			baselineOptions.pauseAtEnd = false;
			baselineOptions.includeConfiguredSchemas = false;
			baselineOptions.generation.count = 4;
			baselineOptions.generation.outputDirectory = (baselineRoot / "generated").string();
			baselineOptions.includeBaselineSchemas = true;
			expect(Experiments::runSchemaSearch(baselineOptions) == 0, "schema search runs with baselines + generated");

			std::ifstream baselineCsvStream(baselineCsv);
			std::string baselineHeader;
			std::getline(baselineCsvStream, baselineHeader);
			bool sawQuadtreeBaseline = false;
			bool sawOctreeBaseline = false;
			bool sawKdtreeBaseline = false;
			bool sawBvhBaseline = false;
			bool sawLbvhBaseline = false;
			bool sawKarrasBaseline = false;
			bool sawRegularGridBaseline = false;
			bool sawHgridBaseline = false;
			bool sawBihBaseline = false;
			std::string baselineRow;
			while (std::getline(baselineCsvStream, baselineRow))
			{
				if (baselineRow.find("quadtree_default") != std::string::npos) sawQuadtreeBaseline = true;
				if (baselineRow.find("octree_default") != std::string::npos) sawOctreeBaseline = true;
				if (baselineRow.find("kdtree_default") != std::string::npos) sawKdtreeBaseline = true;
				if (baselineRow.find("bvh_default") != std::string::npos) sawBvhBaseline = true;
				if (baselineRow.find("lbvh_default") != std::string::npos) sawLbvhBaseline = true;
				if (baselineRow.find("karras_octree_default") != std::string::npos) sawKarrasBaseline = true;
				if (baselineRow.find("regular_grid_default") != std::string::npos) sawRegularGridBaseline = true;
				if (baselineRow.find("hgrid_default") != std::string::npos) sawHgridBaseline = true;
				if (baselineRow.find("bih_default") != std::string::npos) sawBihBaseline = true;
			}
			expect(sawQuadtreeBaseline, "baseline injection measures pure QuadTree as a control");
			expect(sawOctreeBaseline, "baseline injection measures pure Octree as a control");
			expect(sawKdtreeBaseline, "baseline injection measures pure KDTree as a control");
			expect(sawBvhBaseline, "baseline injection measures pure BVH as a control");
			expect(sawLbvhBaseline, "baseline injection measures pure LBVH as a control");
			expect(sawKarrasBaseline, "baseline injection measures pure KarrasOctree as a control");
			expect(sawRegularGridBaseline, "baseline injection measures pure RegularGrid as a control");
			expect(sawHgridBaseline, "baseline injection measures pure HGrid as a control");
			expect(sawBihBaseline, "baseline injection measures pure BIH as a control");

			std::error_code rmError;
			std::filesystem::remove_all(baselineRoot, rmError);
		}

		// Multi-fidelity cache dedup: the visit-proxy rung and the latency rung evaluate the same
		// (schema, dataset, workload) tuple but compute different scores from different counters.
		// Cache fingerprints must distinguish them so a cheap proxy hit cannot satisfy a latency
		// query (and vice versa).
		{
			Experiments::WorkloadProfile workload;
			workload.name = "rung_workload";
			workload.numQueries = 8;
			workload.knnK = 4;
			workload.querySeed = 99;
			workload.rangeWeight = 0.5;
			workload.radiusWeight = 0.3;
			workload.knnWeight = 0.2;
			workload.rangeScaleMin = 0.01;
			workload.rangeScaleMax = 0.05;
			workload.radiusScaleMin = 0.01;
			workload.radiusScaleMax = 0.04;

			const glm::vec3 bboxMin(-1.0f, -1.0f, -1.0f);
			const glm::vec3 bboxMax(1.0f, 1.0f, 1.0f);

			Experiments::ScoreWeights latencyWeights;
			Experiments::ScoreWeights proxyWeights;
			proxyWeights.useVisitProxy = true;
			proxyWeights.visitProxyAlpha = 0.1;

			const Experiments::EvaluationCacheKey latencyKey = Experiments::makeEvaluationCacheKey(
				"sig", "rung_ds", 2048, bboxMin, bboxMax, workload, "cpu", "", latencyWeights);
			const Experiments::EvaluationCacheKey proxyKey = Experiments::makeEvaluationCacheKey(
				"sig", "rung_ds", 2048, bboxMin, bboxMax, workload, "cpu", "", proxyWeights);
			expect(latencyKey.evaluatorFingerprint != proxyKey.evaluatorFingerprint,
				"cache fingerprint distinguishes visit-proxy from latency for same workload");

			Experiments::ScoreWeights proxyWeightsOtherAlpha = proxyWeights;
			proxyWeightsOtherAlpha.visitProxyAlpha = 0.5;
			const Experiments::EvaluationCacheKey otherAlphaKey = Experiments::makeEvaluationCacheKey(
				"sig", "rung_ds", 2048, bboxMin, bboxMax, workload, "cpu", "", proxyWeightsOtherAlpha);
			expect(otherAlphaKey.evaluatorFingerprint != proxyKey.evaluatorFingerprint,
				"cache fingerprint distinguishes visit-proxy alpha values");

			// And query count overrides (rung A: 4 queries; rung B: 64 queries) must produce
			// distinct cache entries even at the same fidelity mode.
			Experiments::WorkloadProfile longWorkload = workload;
			longWorkload.numQueries = 64;
			const Experiments::EvaluationCacheKey shortKey = Experiments::makeEvaluationCacheKey(
				"sig", "rung_ds", 2048, bboxMin, bboxMax, workload, "cpu", "", latencyWeights);
			const Experiments::EvaluationCacheKey longKey = Experiments::makeEvaluationCacheKey(
				"sig", "rung_ds", 2048, bboxMin, bboxMax, longWorkload, "cpu", "", latencyWeights);
			expect(shortKey.workloadFingerprint != longKey.workloadFingerprint,
				"cache fingerprint distinguishes rungs by effective query count");
		}

		// RungSchedule plumbs through SchemaSearchOptions. An empty schedule means the GA stays
		// on its flat path; a non-empty schedule must survive struct copy/move.
		{
			Experiments::SchemaSearchOptions options;
			expect(options.evolution.rungSchedule.rungs.empty(),
				"default rung schedule is empty (flat GA)");

			Experiments::RungSpec proxy;
			proxy.name = "proxy";
			proxy.queryCountOverride = 4;
			proxy.useVisitProxy = true;
			proxy.visitProxyAlpha = 0.2;
			proxy.advanceTopK = 32;

			Experiments::RungSpec confirm;
			confirm.name = "confirm";
			confirm.queryCountOverride = 64;
			confirm.useVisitProxy = false;
			confirm.advanceTopK = 0;

			options.evolution.rungSchedule.rungs = { proxy, confirm };
			options.evolution.rungSchedule.surrogateProposalsPerStep = 4;
			options.evolution.rungSchedule.surrogateCandidatePool = 32;

			Experiments::SchemaSearchOptions copied = options;
			expect(copied.evolution.rungSchedule.rungs.size() == 2,
				"rung schedule survives SchemaSearchOptions copy");
			expect(copied.evolution.rungSchedule.rungs[0].useVisitProxy,
				"first rung is the visit-proxy stage");
			expect(!copied.evolution.rungSchedule.rungs[1].useVisitProxy,
				"second rung is the latency confirmation stage");
			expect(copied.evolution.rungSchedule.rungs[0].advanceTopK == 32,
				"rung schedule survivor count round-trips");
			expect(copied.evolution.rungSchedule.surrogateProposalsPerStep == 4,
				"surrogate proposal count round-trips");
		}

		// Phase B1: ThresholdRefiner. Builds a candidate whose level has a numeric minPoints
		// threshold, sets up a synthetic ConditionDomain bounding it, and runs the refiner with
		// a synthetic score function that has a known minimum. Verifies that:
		//   (a) collectRefinementDimensions discovers exactly the dims the candidate set
		//   (b) the (1+lambda)-ES drives the score below the initial value
		//   (c) the final candidate has minPoints inside the searched range
		{
			Experiments::SchemaCandidate seed;
			seed.config.name = "ot_with_conditional_kd";
			seed.config.levels.resize(2);
			seed.config.levels[0].type = MultiDataStructure::DataStructureLevel::OctreeNode;
			seed.config.levels[0].typeName = "Octree";
			seed.config.levels[0].numLevels = 4;
			seed.config.levels[0].leafCapacity = 64;
			seed.config.levels[1].type = MultiDataStructure::DataStructureLevel::KDTreeNode;
			seed.config.levels[1].typeName = "KDTree";
			seed.config.levels[1].numLevels = 3;
			seed.config.levels[1].leafCapacity = 32;
			seed.config.levels[1].condition.minPoints = 4096;     // starting point — refiner should move it
			seed.config.levels[1].condition.minHeightRatio = 0.3; // second active dim
			seed.path = "test:seed";

			Experiments::ConditionDomain domain;
			domain.pointThresholds = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };
			domain.heightRatioThresholds = { 0.05, 0.1, 0.2, 0.4, 0.6, 0.8 };
			domain.estimatedFromCloud = true;
			domain.samplePoints = 64;

			const std::vector<Experiments::RefinementDimension> dims =
				Experiments::collectRefinementDimensions(seed, domain);
			expect(dims.size() == 2,
				"refiner discovers exactly the two active condition dimensions");
			bool foundPoints = false;
			bool foundHeight = false;
			for (const Experiments::RefinementDimension& dim : dims)
			{
				if (dim.field == "minPoints")
				{
					foundPoints = true;
					expect(dim.integerValued && dim.logScale,
						"minPoints uses integer + log-scale encoding");
					expect(nearlyEqual(dim.lo, 256.0) && nearlyEqual(dim.hi, 32768.0),
						"minPoints bounds come from the supplied domain");
				}
				else if (dim.field == "minHeightRatio")
				{
					foundHeight = true;
					expect(!dim.integerValued && !dim.logScale,
						"minHeightRatio uses continuous linear encoding");
					expect(dim.lo < dim.hi && dim.lo >= 0.05 && dim.hi <= 0.8,
						"minHeightRatio bounds come from the supplied domain");
				}
			}
			expect(foundPoints && foundHeight,
				"refiner reports both minPoints and minHeightRatio as active dims");

			// Synthetic objective: minimised when minPoints ~ 1024 and minHeightRatio ~ 0.4.
			// Quadratic bowl in encoded threshold space; the refiner must drive the score down.
			auto syntheticScore = [&](const Experiments::SchemaCandidate& candidate) {
				if (candidate.config.levels.size() < 2)
					return 1.0e9;
				const SchemaLevelCondition& c = candidate.config.levels[1].condition;
				const double minP = c.minPoints.has_value() ? static_cast<double>(c.minPoints.value()) : 4096.0;
				const double minH = c.minHeightRatio.value_or(0.3);
				const double pointError = std::log2(std::max(minP, 1.0)) - std::log2(1024.0);
				const double heightError = minH - 0.4;
				return pointError * pointError + 12.0 * heightError * heightError;
			};

			Experiments::ThresholdRefinementOptions refineOptions;
			refineOptions.enabled = true;
			refineOptions.maxEvaluations = 50;
			refineOptions.populationLambda = 5;
			refineOptions.sigma0 = 0.4;
			refineOptions.seed = 4242;
			refineOptions.outputDirectory.clear();

			const Experiments::ThresholdRefinementResult result =
				Experiments::refineSchemaThresholds(seed, domain, refineOptions, syntheticScore);
			expect(result.dimensions == 2, "refinement result reports 2 active dimensions");
			expect(result.evaluationsUsed >= 6,
				"refinement consumes at least one generation worth of evaluations");
			expect(result.refinedScore < result.initialScore,
				"refinement drives the synthetic objective below its starting value");
			expect(result.refinedCandidate.config.levels.size() == 2,
				"refined candidate preserves discrete topology");
			expect(result.refinedCandidate.config.levels[1].condition.minPoints.has_value(),
				"refined candidate still carries the minPoints threshold");
			const size_t refinedMinPoints = result.refinedCandidate.config.levels[1].condition.minPoints.value();
			expect(refinedMinPoints >= 256 && refinedMinPoints <= 32768,
				"refined minPoints stays inside the supplied domain bounds");
			expect(result.refinedCandidate.config.levels[1].condition.minHeightRatio.has_value(),
				"refined candidate still carries the minHeightRatio threshold");

			// Topology must be frozen: types, level counts, and leaf capacities are untouched.
			expect(result.refinedCandidate.config.levels[0].typeName == seed.config.levels[0].typeName,
				"refinement does not change level types");
			expect(result.refinedCandidate.config.levels[0].leafCapacity == seed.config.levels[0].leafCapacity,
				"refinement does not change leaf capacity");
			expect(result.refinedCandidate.config.levels[1].numLevels == seed.config.levels[1].numLevels,
				"refinement does not change level counts");
		}

		// Phase C1: Pareto front. Build a synthetic record table with a known structure and
		// assert selectParetoRecords returns exactly the non-dominated set, ranked by latency.
		{
			auto makeRecord = [](const std::string& dataset, const std::string& workload,
				const std::string& schemaName, double latencyMs, double buildMs,
				size_t memoryBytes, double avgOccupancy, size_t maxOccupancy) {
				Experiments::SchemaSearchRecord record;
				record.datasetName = dataset;
				record.workloadName = workload;
				record.schemaName = schemaName;
				record.queryMetrics.averageLatencyMs = latencyMs;
				record.buildMetrics.buildTimeMs = buildMs;
				record.buildMetrics.memoryEstimateBytes = memoryBytes;
				record.buildMetrics.averageLeafOccupancy = avgOccupancy;
				record.buildMetrics.maxLeafOccupancy = maxOccupancy;
				record.score = latencyMs;
				return record;
			};

			// Group A: latency-tradeoff front, three should survive.
			//   fast_big       0.2 ms / 50 ms build / 200 MB / imbalance 2.0   (cheapest latency)
			//   balanced       0.5 ms / 20 ms build / 100 MB / imbalance 1.5   (balanced)
			//   tiny           1.0 ms /  5 ms build /  20 MB / imbalance 1.2   (cheapest build/mem)
			//   dominated      0.6 ms / 25 ms build / 110 MB / imbalance 1.6   (worse than balanced on all)
			//   matches_balanced 0.5 / 20 / 100 / 1.5  (identical to balanced; co-front, both kept)
			std::vector<Experiments::SchemaSearchRecord> raw;
			raw.push_back(makeRecord("ds_a", "wl_a", "fast_big",         0.2, 50.0, 200ull * 1024 * 1024, 50.0, 100));
			raw.push_back(makeRecord("ds_a", "wl_a", "balanced",         0.5, 20.0, 100ull * 1024 * 1024, 40.0, 60));
			raw.push_back(makeRecord("ds_a", "wl_a", "tiny",             1.0,  5.0,  20ull * 1024 * 1024, 50.0, 60));
			raw.push_back(makeRecord("ds_a", "wl_a", "dominated",        0.6, 25.0, 110ull * 1024 * 1024, 40.0, 64));
			raw.push_back(makeRecord("ds_a", "wl_a", "matches_balanced", 0.5, 20.0, 100ull * 1024 * 1024, 40.0, 60));

			// Group B: single non-dominated entry (only one candidate measured).
			raw.push_back(makeRecord("ds_b", "wl_b", "lone",             0.3, 10.0, 50ull * 1024 * 1024, 30.0, 45));

			const std::vector<Experiments::SchemaSearchRecord> front = Experiments::selectParetoRecords(raw);

			size_t groupA = 0;
			size_t groupB = 0;
			bool sawDominated = false;
			bool sawTiny = false;
			bool sawFast = false;
			bool sawBalanced = false;
			bool sawCofront = false;
			int previousLatencyRank = -1;
			double previousLatency = -1.0;
			for (const Experiments::SchemaSearchRecord& entry : front)
			{
				if (entry.datasetName == "ds_a")
				{
					++groupA;
					expect(entry.paretoRank >= 0, "front entry carries non-negative paretoRank");
					if (previousLatencyRank == -1 || entry.paretoRank == 0)
					{
						previousLatency = entry.queryMetrics.averageLatencyMs;
						previousLatencyRank = entry.paretoRank;
					}
					else
					{
						expect(entry.queryMetrics.averageLatencyMs >= previousLatency,
							"pareto front sorted by ascending latency within group");
						previousLatency = entry.queryMetrics.averageLatencyMs;
					}
					if (entry.schemaName == "dominated")
						sawDominated = true;
					if (entry.schemaName == "tiny")
						sawTiny = true;
					if (entry.schemaName == "fast_big")
						sawFast = true;
					if (entry.schemaName == "balanced")
						sawBalanced = true;
					if (entry.schemaName == "matches_balanced")
						sawCofront = true;
				}
				else if (entry.datasetName == "ds_b")
				{
					++groupB;
					expect(entry.paretoRank == 0, "single-entry group ranks the lone candidate at 0");
				}
			}

			expect(!sawDominated, "dominated candidate is filtered out of the Pareto front");
			expect(sawFast && sawTiny && sawBalanced, "non-dominated trio survives the front");
			expect(sawCofront, "candidate with identical metrics to the balanced one is co-front (no strict domination)");
			expect(groupA == 4, "ds_a front contains 4 non-dominated rows (fast_big, balanced, matches_balanced, tiny)");
			expect(groupB == 1, "ds_b front contains the single measured candidate");
			expect(front.front().paretoRank == 0, "first row of returned front is rank 0");
		}

		// Phase C2: bootstrap mean + 95% CI helper.
		{
			// Deterministic samples: mean is exactly 5.0, CI should bracket it tightly.
			const std::vector<double> samples = { 4.0, 4.5, 5.0, 5.5, 6.0 };
			const auto [mean, ciLow, ciHigh] = Experiments::bootstrapMeanCI(samples, 2000, 0xC0FFEE);
			expect(nearlyEqual(mean, 5.0), "bootstrap reports the sample mean");
			expect(ciLow <= mean && mean <= ciHigh,
				"95% CI brackets the point mean");
			expect(ciLow >= 4.0 && ciHigh <= 6.0,
				"95% CI stays within the sample range");
			expect((ciHigh - ciLow) < 2.0,
				"95% CI is narrower than the sample range");

			// Empty input collapses cleanly.
			const auto [emptyMean, emptyLo, emptyHi] = Experiments::bootstrapMeanCI({}, 100, 1);
			expect(nearlyEqual(emptyMean, 0.0) && nearlyEqual(emptyLo, 0.0) && nearlyEqual(emptyHi, 0.0),
				"empty sample produces zeroed mean/CI");

			// Single-element input collapses CI to the point value.
			const auto [oneMean, oneLo, oneHi] = Experiments::bootstrapMeanCI({ 7.25 }, 100, 1);
			expect(nearlyEqual(oneMean, 7.25) && nearlyEqual(oneLo, 7.25) && nearlyEqual(oneHi, 7.25),
				"single-element CI collapses to the point value");

			// Deterministic for fixed seed: repeating the call with the same seed yields the same CI.
			const auto [_, ciLowA, ciHighA] = Experiments::bootstrapMeanCI(samples, 500, 99);
			const auto [__, ciLowB, ciHighB] = Experiments::bootstrapMeanCI(samples, 500, 99);
			expect(nearlyEqual(ciLowA, ciLowB) && nearlyEqual(ciHighA, ciHighB),
				"bootstrap CI is reproducible for a fixed seed");
		}

		// Phase C2: Pareto dominance uses latencyMean when the record came out of multi-seed
		// confirmation. A noisy record with a lucky low single-seed latency must not Pareto-dominate
		// a stable record whose mean is genuinely better.
		{
			auto makeRecord = [](const std::string& schemaName, double latencyMs, double buildMs,
				size_t memoryBytes, double avgOccupancy, size_t maxOccupancy,
				size_t confirmSeeds, double latencyMean) {
				Experiments::SchemaSearchRecord record;
				record.datasetName = "ds";
				record.workloadName = "wl";
				record.schemaName = schemaName;
				record.queryMetrics.averageLatencyMs = latencyMs;
				record.buildMetrics.buildTimeMs = buildMs;
				record.buildMetrics.memoryEstimateBytes = memoryBytes;
				record.buildMetrics.averageLeafOccupancy = avgOccupancy;
				record.buildMetrics.maxLeafOccupancy = maxOccupancy;
				record.score = latencyMs;
				record.confirmSeedsUsed = confirmSeeds;
				record.latencyMean = latencyMean;
				return record;
			};

			std::vector<Experiments::SchemaSearchRecord> raw;
			// A: noisy single-seed point estimate is 0.1 ms but the confirmed mean across 5 seeds is 1.0 ms.
			raw.push_back(makeRecord("noisy_lucky", 0.1, 10.0, 50ull * 1024 * 1024, 40.0, 60, 5, 1.0));
			// B: confirmed mean of 0.5 ms — better than A's confirmed mean, but worse than A's
			// single-seed point estimate.
			raw.push_back(makeRecord("stable",       0.6, 10.0, 50ull * 1024 * 1024, 40.0, 60, 5, 0.5));

			const std::vector<Experiments::SchemaSearchRecord> front = Experiments::selectParetoRecords(raw);
			bool sawNoisy = false;
			bool sawStable = false;
			for (const auto& entry : front)
			{
				if (entry.schemaName == "noisy_lucky") sawNoisy = true;
				if (entry.schemaName == "stable")      sawStable = true;
			}
			expect(sawStable, "stable candidate (best confirmed mean) survives the Pareto front");
			expect(!sawNoisy, "noisy single-seed lucky candidate is dominated by the confirmed mean");
		}

		// Candidates with no conditional levels short-circuit cleanly — no evaluations, score
		// equals initial, candidate returned unchanged.
		{
			Experiments::SchemaCandidate flat;
			flat.config.name = "octree_flat";
			flat.config.levels.resize(1);
			flat.config.levels[0].type = MultiDataStructure::DataStructureLevel::OctreeNode;
			flat.config.levels[0].typeName = "Octree";
			flat.config.levels[0].numLevels = 6;
			flat.config.levels[0].leafCapacity = 1024;

			Experiments::ConditionDomain emptyDomain;
			Experiments::ThresholdRefinementOptions refineOptions;
			refineOptions.enabled = true;
			refineOptions.maxEvaluations = 20;
			refineOptions.outputDirectory.clear();

			size_t evalCount = 0;
			auto trackedScore = [&](const Experiments::SchemaCandidate&) {
				++evalCount;
				return 1.0;
			};

			const Experiments::ThresholdRefinementResult result =
				Experiments::refineSchemaThresholds(flat, emptyDomain, refineOptions, trackedScore);
			expect(result.dimensions == 0, "no active dims when no conditional levels exist");
			expect(evalCount <= 1, "refiner does not waste evaluations on a flat candidate");
			expect(result.refinedCandidate.config.name == flat.config.name,
				"flat candidate is returned unchanged");
		}
	}
}

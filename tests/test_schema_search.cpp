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
		  "stratifyQueries": true,
		  "queryScales": {
		    "aabb_range": {
		      "min": 0.02,
		      "max": 0.25
		    },
		    "radius": {
		      "min": 0.03,
		      "max": 0.12
		    }
		  },
		  "scoreWeights": {
		    "latency": 1.0,
		    "buildTime": 0.25,
		    "memory": 0.5,
		    "imbalance": 0.75
		  }
		}
		)json";

		const Experiments::WorkloadProfile profile = Experiments::parseWorkloadProfile(workloadJson, "focused_knn");
		expect(profile._name == "focused_knn", "schema search parses workload name");
		expect(nearlyEqual(profile._rangeWeight, 0.2), "schema search parses range workload weight");
		expect(nearlyEqual(profile._radiusWeight, 0.1), "schema search parses radius workload weight");
		expect(nearlyEqual(profile._knnWeight, 0.7), "schema search parses knn workload weight");
		expect(profile._knnK == 12, "schema search parses workload knn k");
		expect(profile._numQueries == 42, "schema search parses workload query count");
		expect(profile._querySeed == 99, "schema search parses workload query seed");
		expect(profile._stratifyQueries, "schema search parses query stratification flag");
		expect(nearlyEqual(profile._rangeScaleMin, 0.02), "schema search parses range scale min");
		expect(nearlyEqual(profile._rangeScaleMax, 0.25), "schema search parses range scale max");
		expect(nearlyEqual(profile._radiusScaleMin, 0.03), "schema search parses radius scale min");
		expect(nearlyEqual(profile._radiusScaleMax, 0.12), "schema search parses radius scale max");
		expect(profile._hasScoreWeights, "schema search parses workload-local score weights");
		expect(nearlyEqual(profile._scoreWeights._lambdaLatency, 1.0), "schema search parses workload latency score weight");
		expect(nearlyEqual(profile._scoreWeights._lambdaBuild, 0.25), "schema search parses workload build score weight");
		expect(nearlyEqual(profile._scoreWeights._lambdaMemory, 0.5), "schema search parses workload memory score weight");
		expect(nearlyEqual(profile._scoreWeights._lambdaImbalance, 0.75), "schema search parses workload imbalance score weight");

		const Experiments::WorkloadProfile proxyProfile = Experiments::parseWorkloadProfile(R"json(
		{
		  "name": "proxy_weighted",
		  "scoreWeights": {
		    "visitedNodes": 1.0,
		    "testedPoints": 0.25
		  }
		}
		)json", "proxy_weighted");
		expect(proxyProfile._hasScoreWeights && proxyProfile._scoreWeights._useVisitProxy,
			"schema search workload score weights can request visit-proxy scoring");
		expect(nearlyEqual(proxyProfile._scoreWeights._visitProxyAlpha, 0.25),
			"schema search parses tested-points proxy weight as visit-proxy alpha");

		Experiments::BuildMetrics buildMetrics;
		buildMetrics._buildTimeMs = 10.0;
		buildMetrics._memoryEstimateBytes = 2 * 1024 * 1024;
		buildMetrics._averageLeafOccupancy = 4.0;
		buildMetrics._maxLeafOccupancy = 12;

		Experiments::QueryMetrics queryMetrics;
		queryMetrics._averageLatencyMs = 2.0;

		double memoryMb = 0.0;
		double imbalancePenalty = 0.0;
		Experiments::ScoreWeights scoreWeights;
		scoreWeights._lambdaBuild = 0.001;
		scoreWeights._lambdaMemory = 0.01;
		scoreWeights._lambdaImbalance = 0.01;
		const double score = Experiments::computeSchemaSearchScore(
			buildMetrics,
			queryMetrics,
			scoreWeights,
			memoryMb,
			imbalancePenalty);

		expect(nearlyEqual(memoryMb, 2.0), "schema search score computes memory MB");
		expect(nearlyEqual(imbalancePenalty, 3.0), "schema search score computes imbalance penalty");
		expect(nearlyEqual(score, 2.06), "schema search score combines latency, build, memory, and imbalance");

		Experiments::ScoreWeights latencyScaleWeights;
		latencyScaleWeights._lambdaLatency = 0.5;
		double scaledMemoryMb = 0.0;
		double scaledImbalancePenalty = 0.0;
		const double scaledScore = Experiments::computeSchemaSearchScore(
			buildMetrics,
			queryMetrics,
			latencyScaleWeights,
			scaledMemoryMb,
			scaledImbalancePenalty);
		expect(nearlyEqual(scaledScore, 1.0), "schema search score applies latency weight from workload JSON");

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
		visitMetrics._averageLatencyMs = 99.0;        // ignored when surrogate is on
		visitMetrics._averageVisitedNodes = 25.0;
		visitMetrics._averageTestedPoints = 100.0;
		Experiments::ScoreWeights visitWeights;
		visitWeights._useVisitProxy = true;
		visitWeights._visitProxyAlpha = 0.2;
		double visitMemoryMb = 0.0;
		double visitImbalance = 0.0;
		const double visitScore = Experiments::computeSchemaSearchScore(
			Experiments::BuildMetrics{},
			visitMetrics,
			visitWeights,
			visitMemoryMb,
			visitImbalance);
		expect(nearlyEqual(visitScore, 45.0), "visit-proxy score combines visited nodes and tested points");

		// Diagnostic-guided repair mutations emit targeted schema edits from measured failure modes.
		{
			Experiments::SchemaCandidate parent;
			parent._config._name = "repair_parent";
			parent._config._levels.resize(1);
			parent._config._levels[0]._type = MultiDataStructure::DataStructureLevel::OctreeNode;
			parent._config._levels[0]._typeName = "Octree";
			parent._config._levels[0]._numLevels = 4;
			parent._config._levels[0]._leafCapacity = 1024;
			parent._config._levels[0]._minPrimitivesToSplit = 256;
			parent._config._buildPolicy._maxDepth = 8;
			parent._config._buildPolicy._leafCapacity = 1024;
			parent._config._buildPolicy._minPrimitivesToSplit = 256;

			Experiments::SchemaSearchRecord measured;
			measured._numPoints = 10000;
			measured._knnWeight = 0.7;
			measured._buildMetrics._averageLeafOccupancy = 32.0;
			measured._buildMetrics._maxLeafOccupancy = 4096;
			measured._buildMetrics._numNodes = 20;
			measured._buildMetrics._numLeaves = 10;
			measured._buildMetrics._maxDepth = 4;
			measured._queryMetrics._averageVisitedNodes = 4.0;
			measured._queryMetrics._averageTestedPoints = 512.0;

			Experiments::SchemaGenerationOptions repairOptions;
			repairOptions._maxDepth = 8;
			repairOptions._maxBlocks = 3;
			repairOptions._minLeafCapacity = 32;
			repairOptions._maxLeafCapacity = 4096;
			repairOptions._outputDirectory.clear();
			repairOptions._primitiveProfile = "query_minimal_cpu";

			Experiments::ConditionDomain domain;
			domain._pointThresholds = { 64, 128, 256, 512, 1024 };

			const Experiments::SchemaRepairDiagnostics diagnostics =
				Experiments::diagnoseSchemaRepair(parent, { measured });
			expect(diagnostics._highLeafOccupancy, "repair diagnosis detects high leaf occupancy");
			expect(diagnostics._testedPointDominated, "repair diagnosis detects tested-point-heavy queries");

			const std::vector<Experiments::SchemaCandidate> repaired =
				Experiments::generateSchemaRepairCandidates(parent, { measured }, repairOptions, &domain, 2, 1234, "");
			expect(repaired.size() == 2, "repair generator emits bounded targeted candidates");
			expect(repaired[0]._config._levels[0]._leafCapacity < parent._config._levels[0]._leafCapacity,
				"high-occupancy repair reduces leaf capacity");
			expect(repaired[0]._config._levels[0]._numLevels > parent._config._levels[0]._numLevels,
				"high-occupancy repair increases depth when budget allows");
			expect(repaired[1]._config._levels.size() > parent._config._levels.size(),
				"tested-point repair appends a local micro-index block");
			expect(!repaired[1]._config._levels.back()._condition.empty(),
				"micro-index repair gates the added block with measured point thresholds");
		}

		{
			Experiments::SchemaCandidate parent;
			parent._config._name = "visited_parent";
			parent._config._levels.resize(1);
			parent._config._levels[0]._type = MultiDataStructure::DataStructureLevel::KDTreeNode;
			parent._config._levels[0]._typeName = "KDTree";
			parent._config._levels[0]._numLevels = 6;
			parent._config._levels[0]._leafCapacity = 128;
			parent._config._levels[0]._minPrimitivesToSplit = 32;

			Experiments::SchemaSearchRecord measured;
			measured._numPoints = 20000;
			measured._pointFeatures._flatnessScore = 1.0;
			measured._buildMetrics._averageLeafOccupancy = 64.0;
			measured._buildMetrics._maxLeafOccupancy = 128;
			measured._buildMetrics._numNodes = 4096;
			measured._buildMetrics._numLeaves = 2048;
			measured._buildMetrics._maxDepth = 6;
			measured._queryMetrics._averageVisitedNodes = 256.0;
			measured._queryMetrics._averageTestedPoints = 256.0;

			Experiments::SchemaGenerationOptions repairOptions;
			repairOptions._maxDepth = 8;
			repairOptions._maxBlocks = 2;
			repairOptions._minLeafCapacity = 32;
			repairOptions._maxLeafCapacity = 4096;
			repairOptions._outputDirectory.clear();
			repairOptions._primitiveProfile = "query_minimal_cpu";

			const Experiments::SchemaRepairDiagnostics diagnostics =
				Experiments::diagnoseSchemaRepair(parent, { measured });
			expect(diagnostics._visitedNodeDominated, "repair diagnosis detects visited-node-heavy traversal");

			const std::vector<Experiments::SchemaCandidate> repaired =
				Experiments::generateSchemaRepairCandidates(parent, { measured }, repairOptions, nullptr, 1, 99, "");
			expect(repaired.size() == 1, "visited-node repair emits one targeted candidate");
			expect(repaired[0]._config._levels[0]._type == MultiDataStructure::DataStructureLevel::QuadTreeNode,
				"visited-node repair switches flat root to quadtree");
			expect(repaired[0]._config._levels[0]._leafCapacity > parent._config._levels[0]._leafCapacity,
				"visited-node repair coarsens root leaves");
		}

		std::vector<Experiments::SchemaSearchRecord> records(3);
		records[0]._datasetName = "flat";
		records[0]._workloadName = "mixed";
		records[0]._schemaName = "slow";
		records[0]._score = 3.0;
		records[1]._datasetName = "flat";
		records[1]._workloadName = "mixed";
		records[1]._schemaName = "fast";
		records[1]._score = 1.0;
		records[2]._datasetName = "facade";
		records[2]._workloadName = "mixed";
		records[2]._schemaName = "only";
		records[2]._score = 2.0;

		const std::vector<Experiments::SchemaSearchRecord> best = Experiments::selectBestRecords(records);
		expect(best.size() == 2, "schema search picks one best record per dataset/workload");
		expect(best[0]._schemaName == "fast", "schema search keeps lowest score as best schema");
		expect(best[1]._schemaName == "only", "schema search keeps independent dataset/workload groups");

		records[0]._schemaName = "single_seed_lucky";
		records[0]._score = 1.0;
		records[0]._queryMetrics._averageLatencyMs = 1.0;
		records[0]._confirmSeedsUsed = 5;
		records[0]._latencyMean = 4.0;
		records[1]._schemaName = "confirmed_fast";
		records[1]._score = 2.0;
		records[1]._queryMetrics._averageLatencyMs = 2.0;
		records[1]._confirmSeedsUsed = 5;
		records[1]._latencyMean = 1.5;
		const std::vector<Experiments::SchemaSearchRecord> confirmedBest = Experiments::selectBestRecords(records);
		expect(confirmedBest[0]._schemaName == "confirmed_fast",
			"schema search best selection prefers multi-seed confirmed latency over lucky single-seed score");

		const Experiments::EvaluatorResolution cudaResolution = Experiments::resolveSchemaSearchEvaluator("cuda", true);
		expect(cudaResolution._usingCuda, "schema search resolver keeps available CUDA");
		expect(cudaResolution._evaluator == "cuda", "schema search resolver returns cuda evaluator");

		const Experiments::EvaluatorResolution fallbackResolution = Experiments::resolveSchemaSearchEvaluator("cuda", false, "missing device");
		expect(!fallbackResolution._usingCuda, "schema search resolver disables unavailable CUDA");
		expect(fallbackResolution._fellBackToCpu, "schema search resolver marks CPU fallback");
		expect(fallbackResolution._evaluator == "cpu", "schema search resolver falls back to CPU");
		expect(fallbackResolution._warning.find("missing device") != std::string::npos, "schema search resolver includes CUDA error in warning");

		Experiments::SchemaGenerationOptions generation;
		generation._count = 12;
		generation._maxBlocks = 3;
		generation._maxDepth = 8;
		generation._minLeafCapacity = 32;
		generation._maxLeafCapacity = 512;
		generation._seed = 11;
		generation._outputDirectory.clear();

		const std::vector<Experiments::SchemaCandidate> generated = Experiments::generateSchemaCandidates(generation);
		expect(generated.size() == generation._count, "schema generator creates requested candidate count");
		for (const Experiments::SchemaCandidate& candidate : generated)
		{
			expect(candidate._generated, "schema generator marks generated candidates");
			expect(candidate._config.totalLevels() <= generation._maxDepth, "schema generator respects max depth");
			expect(!candidate._config._levels.empty(), "schema generator creates non-empty level schedules");
			expect(candidate._path.rfind("generated:", 0) == 0, "schema generator uses generated pseudo path");
		}

		Experiments::SchemaGenerationOptions nestedGeneration = generation;
		nestedGeneration._count = 8;
		nestedGeneration._minBlocks = 2;
		nestedGeneration._maxBlocks = 2;
		const std::vector<Experiments::SchemaCandidate> nestedGenerated = Experiments::generateSchemaCandidates(nestedGeneration);
		expect(nestedGenerated.size() == nestedGeneration._count, "schema generator creates requested minimum-block candidates");
		for (const Experiments::SchemaCandidate& candidate : nestedGenerated)
			expect(candidate._config._levels.size() >= 2, "schema generator respects generated min blocks");

		auto isCpuDuplicateName = [](const std::string& typeName) {
			return typeName == "BIH" ||
				typeName == "KarrasOctree" ||
				typeName == "RegularGrid" ||
				typeName == "HGrid" ||
				typeName == "LBVH";
		};
		Experiments::SchemaGenerationOptions minimalGeneration = generation;
		minimalGeneration._count = 48;
		minimalGeneration._seed = 33;
		minimalGeneration._outputDirectory.clear();
		minimalGeneration._primitiveProfile = "query_minimal_cpu";
		const std::vector<Experiments::SchemaCandidate> minimalGenerated =
			Experiments::generateSchemaCandidates(minimalGeneration);
		for (const Experiments::SchemaCandidate& candidate : minimalGenerated)
		{
			for (const SchemaLevelConfig& level : candidate._config._levels)
				expect(!isCpuDuplicateName(level._typeName), "query-minimal primitive profile excludes CPU-equivalent aliases");
		}

		Experiments::SchemaGenerationOptions fullGeneration = minimalGeneration;
		fullGeneration._count = 96;
		fullGeneration._seed = 44;
		fullGeneration._primitiveProfile = "cuda_query_full";
		const std::vector<Experiments::SchemaCandidate> fullGenerated =
			Experiments::generateSchemaCandidates(fullGeneration);
		bool sawCudaVariant = false;
		for (const Experiments::SchemaCandidate& candidate : fullGenerated)
		{
			for (const SchemaLevelConfig& level : candidate._config._levels)
				sawCudaVariant = sawCudaVariant || isCpuDuplicateName(level._typeName);
		}
		expect(sawCudaVariant, "cuda-full primitive profile includes CUDA-native aliases");
		expect(Experiments::resolvePrimitiveProfile("auto", false) == "query_minimal_cpu",
			"primitive profile auto resolves to query-minimal on CPU");
		expect(Experiments::resolvePrimitiveProfile("auto", true) == "cuda_query_full",
			"primitive profile auto resolves to CUDA-full on CUDA");

		const PointCloud flatCloud = SyntheticPointClouds::generateFlatTerrain(128, 80.0f, 80.0f, 0.02f, 7);
		const PointCloud tallCloud = SyntheticPointClouds::generateFacade(128, 80.0f, 40.0f, 0.05f, 8);
		const PointCloud sparseDenseCloud = SyntheticPointClouds::generateSparseDenseMixture(96, 96, 9);
		const Experiments::ConditionDomain flatDomainA = Experiments::estimateConditionDomain(flatCloud, 64);
		const Experiments::ConditionDomain flatDomainB = Experiments::estimateConditionDomain(flatCloud, 64);
		const Experiments::ConditionDomain tallDomain = Experiments::estimateConditionDomain(tallCloud, 64);
		const Experiments::ConditionDomain sparseDenseDomain = Experiments::estimateConditionDomain(sparseDenseCloud, 64);
		expect(flatDomainA._estimatedFromCloud, "condition domain records cloud estimate");
		expect(flatDomainA._samplePoints == 64, "condition domain respects sample cap");
		expect(flatDomainA._pointThresholds == flatDomainB._pointThresholds, "condition domain point thresholds are deterministic");
		expect(flatDomainA._heightRatioThresholds == flatDomainB._heightRatioThresholds, "condition domain height thresholds are deterministic");
		expect(!flatDomainA._pointThresholds.empty(), "condition domain estimates point thresholds");
		expect(!flatDomainA._heightRatioThresholds.empty(), "condition domain estimates height thresholds");
		expect(!tallDomain._heightRatioThresholds.empty(), "condition domain estimates tall-cloud height thresholds");
		expect(!sparseDenseDomain._densityThresholds.empty(), "condition domain estimates density thresholds");

		Experiments::SchemaGenerationOptions conditionalGeneration;
		conditionalGeneration._count = 10;
		conditionalGeneration._maxBlocks = 3;
		conditionalGeneration._maxDepth = 6;
		conditionalGeneration._minLeafCapacity = 16;
		conditionalGeneration._maxLeafCapacity = 256;
		conditionalGeneration._conditionalLevels = true;
		conditionalGeneration._conditionalProbability = 1.0;
		conditionalGeneration._seed = 12;
		conditionalGeneration._outputDirectory.clear();
		const std::vector<Experiments::SchemaCandidate> conditionalGenerated =
			Experiments::generateSchemaCandidates(conditionalGeneration, &flatDomainA);
		const std::vector<Experiments::SchemaCandidate> conditionalGeneratedAgain =
			Experiments::generateSchemaCandidates(conditionalGeneration, &flatDomainA);
		expect(conditionalGenerated.size() == conditionalGeneration._count, "domain-aware schema generator creates requested candidates");
		expect(conditionalGeneratedAgain.size() == conditionalGenerated.size(), "domain-aware schema generator is deterministic for fixed seed");
		bool sawNumericCondition = false;
		for (size_t candidateIndex = 0; candidateIndex < conditionalGenerated.size(); ++candidateIndex)
		{
			const Experiments::SchemaCandidate& candidate = conditionalGenerated[candidateIndex];
			expect(candidate._name == conditionalGeneratedAgain[candidateIndex]._name,
				"domain-aware schema generator repeats candidate order for fixed seed");
			for (size_t levelIndex = 1; levelIndex < candidate._config._levels.size(); ++levelIndex)
			{
				if (!candidate._config._levels[levelIndex]._condition.empty())
				{
					sawNumericCondition = true;
					expect(candidate._config._levels[levelIndex]._condition._minPoints.has_value(),
						"domain-aware generated condition includes numeric minPoints");
				}
			}
		}
		expect(sawNumericCondition, "domain-aware schema generator emits conditional levels");

		Experiments::SchemaGenerationOptions adaptiveGeneration = conditionalGeneration;
		adaptiveGeneration._count = 12;
		adaptiveGeneration._seed = 18;
		adaptiveGeneration._adaptiveLeafCapacity = true;
		adaptiveGeneration._adaptiveLeafProbability = 1.0;
		const std::vector<Experiments::SchemaCandidate> adaptiveGenerated =
			Experiments::generateSchemaCandidates(adaptiveGeneration, &flatDomainA);
		expect(adaptiveGenerated.size() == adaptiveGeneration._count,
			"adaptive leaf-capacity generator creates requested candidates");
		bool sawAdaptiveLeafCapacity = false;
		for (const Experiments::SchemaCandidate& candidate : adaptiveGenerated)
		{
			for (const SchemaLevelConfig& level : candidate._config._levels)
			{
				if (level._adaptiveLeafCapacity._enabled)
				{
					sawAdaptiveLeafCapacity = true;
					expect(level._adaptiveLeafCapacity._minCapacity > 0,
						"adaptive leaf-capacity generator writes numeric minimum capacity");
					expect(level._adaptiveLeafCapacity._maxCapacity >= level._adaptiveLeafCapacity._minCapacity,
						"adaptive leaf-capacity generator writes valid capacity bounds");
				}
			}
		}
		expect(sawAdaptiveLeafCapacity, "adaptive leaf-capacity generator emits adaptive levels");

		const std::filesystem::path tempRoot = std::filesystem::temp_directory_path() / "mdspc_schema_search_tests";
		std::filesystem::create_directories(tempRoot);
		const std::filesystem::path csvPath = tempRoot / "prepared_queries.csv";
		const std::filesystem::path bestCsvPath = tempRoot / "prepared_queries_best.csv";
		const std::filesystem::path explainPath = tempRoot / "prepared_queries_explain.md";

		Experiments::SchemaSearchOptions options;
		options._schemaPaths = { "configs/schemas/quadtree.json" };
		options._workloadPaths = { "configs/workloads/mixed.json" };
		options._includeBaselineSchemas = false;
		options._csvPath = csvPath.string();
		options._bestCsvPath = bestCsvPath.string();
		options._explainReportPath = explainPath.string();
		options._syntheticScale = 64;
		options._queryCountOverride = 9;
		options._evaluator = "cpu";
		options._pauseAtEnd = false;
		const int exitCode = Experiments::runSchemaSearch(options);
		expect(exitCode == 0, "schema search smoke run succeeds");
		std::ifstream explain(explainPath);
		expect(explain.is_open(), "schema search writes schema explain report");
		const std::string explainText((std::istreambuf_iterator<char>(explain)), std::istreambuf_iterator<char>());
		expect(explainText.find("Schema Explain Report") != std::string::npos,
			"schema explain report has markdown title");
		expect(explainText.find("Active structures") != std::string::npos,
			"schema explain report includes active structure section");
		expect(explainText.find("Query behavior") != std::string::npos,
			"schema explain report includes query behavior section");
		expect(explainText.find("Diagnosis") != std::string::npos,
			"schema explain report includes diagnosis section");

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
		const size_t queryStrataColumn = columnIndex("query_strata_summary");
		const size_t leafP90Column = columnIndex("leaf_occupancy_p90");
		const size_t emptyChildColumn = columnIndex("empty_child_ratio");
		const size_t tightVolumeColumn = columnIndex("mean_tight_bounds_volume_ratio");
		const size_t nestedFractionColumn = columnIndex("nested_active_fraction");
		const size_t baselineSchemaColumn = columnIndex("best_baseline_schema");
		const size_t scoreObjectiveColumn = columnIndex("score_objective");
		const size_t scoreModeColumn = columnIndex("score_mode");
		const size_t scoreStageColumn = columnIndex("score_stage");
		const size_t finalLatencyColumn = columnIndex("score_is_final_latency");
		const size_t effectiveQueriesColumn = columnIndex("effective_queries");
		const size_t visitProxyColumn = columnIndex("score_uses_visit_proxy");
		const size_t gpuSupportColumn = columnIndex("gpu_support_status");
		expect(totalQueriesColumn < values.size() && rangeQueriesColumn < values.size() &&
			radiusQueriesColumn < values.size() && knnQueriesColumn < values.size(),
			"schema search prepared-query CSV includes query count columns");
		expect(queryStrataColumn < values.size(),
			"schema search CSV includes query stratum summary column");
		expect(leafP90Column < values.size() && emptyChildColumn < values.size() && tightVolumeColumn < values.size(),
			"schema search CSV includes tree health columns");
		expect(nestedFractionColumn < values.size() && baselineSchemaColumn < values.size(),
			"schema search CSV includes nested accounting and baseline-normalized columns");
		expect(scoreObjectiveColumn < values.size() && scoreModeColumn < values.size() && scoreStageColumn < values.size() &&
			finalLatencyColumn < values.size() && effectiveQueriesColumn < values.size() &&
			visitProxyColumn < values.size() && gpuSupportColumn < values.size(),
			"schema search CSV includes score provenance columns");
		const size_t totalQueries = static_cast<size_t>(std::stoull(values[totalQueriesColumn]));
		const size_t rangeQueries = static_cast<size_t>(std::stoull(values[rangeQueriesColumn]));
		const size_t radiusQueries = static_cast<size_t>(std::stoull(values[radiusQueriesColumn]));
		const size_t knnQueries = static_cast<size_t>(std::stoull(values[knnQueriesColumn]));
		expect(totalQueries == 9, "schema search prepared workload keeps query override count");
		expect(rangeQueries + radiusQueries + knnQueries == totalQueries, "schema search prepared CPU query counts match total");
		expect(!values[queryStrataColumn].empty(), "schema search records per-stratum query metrics");
		expect(std::stod(values[leafP90Column]) >= 0.0, "schema search writes leaf occupancy health metric");
		expect(values[scoreObjectiveColumn] == "latency", "schema search direct CSV marks latency score objective");
		expect(values[scoreModeColumn] == "latency", "schema search direct CSV marks latency score mode");
		expect(values[scoreStageColumn] == "final", "schema search direct CSV marks final score stage");
		expect(values[finalLatencyColumn] == "1", "schema search direct CSV marks final latency score");
		expect(static_cast<size_t>(std::stoull(values[effectiveQueriesColumn])) == totalQueries,
			"schema search CSV effective query count matches measured total");
		expect(values[visitProxyColumn] == "0", "schema search direct CSV marks visit-proxy disabled");
		expect(values[gpuSupportColumn] == "full", "schema search direct CSV marks full GPU support status for CPU rows");

		const std::filesystem::path balancedCsvPath = tempRoot / "balanced_objective.csv";
		Experiments::SchemaSearchOptions balancedOptions = options;
		balancedOptions._csvPath = balancedCsvPath.string();
		balancedOptions._bestCsvPath.clear();
		balancedOptions._paretoCsvPath.clear();
		balancedOptions._explainReportPath.clear();
		balancedOptions._scoreObjective = "balanced";
		balancedOptions._queryCountOverride = 3;
		expect(Experiments::runSchemaSearch(balancedOptions) == 0, "schema search balanced objective run succeeds");
		std::ifstream balancedCsv(balancedCsvPath);
		expect(balancedCsv.is_open(), "schema search writes balanced objective CSV");
		std::string balancedHeader;
		std::string balancedRow;
		std::getline(balancedCsv, balancedHeader);
		std::getline(balancedCsv, balancedRow);
		const std::vector<std::string> balancedColumns = splitCsv(balancedHeader);
		const std::vector<std::string> balancedValues = splitCsv(balancedRow);
		auto balancedColumnIndex = [&balancedColumns](const std::string& name) {
			const auto found = std::find(balancedColumns.begin(), balancedColumns.end(), name);
			return found == balancedColumns.end()
				? std::numeric_limits<size_t>::max()
				: static_cast<size_t>(std::distance(balancedColumns.begin(), found));
		};
		const size_t balancedObjectiveColumn = balancedColumnIndex("score_objective");
		const size_t balancedModeColumn = balancedColumnIndex("score_mode");
		const size_t balancedFinalLatencyColumn = balancedColumnIndex("score_is_final_latency");
		const size_t balancedBuildColumn = balancedColumnIndex("lambda_build");
		const size_t balancedMemoryColumn = balancedColumnIndex("lambda_memory");
		const size_t balancedImbalanceColumn = balancedColumnIndex("lambda_imbalance");
		expect(balancedObjectiveColumn < balancedValues.size() && balancedValues[balancedObjectiveColumn] == "balanced",
			"schema search balanced objective records objective name");
		expect(balancedModeColumn < balancedValues.size() && balancedValues[balancedModeColumn] == "weighted_latency",
			"schema search balanced objective records weighted score mode");
		expect(balancedFinalLatencyColumn < balancedValues.size() && balancedValues[balancedFinalLatencyColumn] == "0",
			"schema search balanced objective is not marked final-latency-only");
		expect(balancedBuildColumn < balancedValues.size() && nearlyEqual(static_cast<float>(std::stod(balancedValues[balancedBuildColumn])), 0.001f),
			"schema search balanced objective sets build weight");
		expect(balancedMemoryColumn < balancedValues.size() && nearlyEqual(static_cast<float>(std::stod(balancedValues[balancedMemoryColumn])), 0.01f),
			"schema search balanced objective sets memory weight");
		expect(balancedImbalanceColumn < balancedValues.size() && nearlyEqual(static_cast<float>(std::stod(balancedValues[balancedImbalanceColumn])), 0.01f),
			"schema search balanced objective sets imbalance weight");

		const std::filesystem::path autoCsvPath = tempRoot / "auto_conditions.csv";
		const std::filesystem::path autoBestCsvPath = tempRoot / "auto_conditions_best.csv";
		Experiments::SchemaSearchOptions autoOptions;
		autoOptions._schemaPaths = { "configs/schemas/quadtree.json" };
		autoOptions._workloadPaths = { "configs/workloads/volume_small_medium.json" };
		autoOptions._includeBaselineSchemas = false;
		autoOptions._csvPath = autoCsvPath.string();
		autoOptions._bestCsvPath = autoBestCsvPath.string();
		autoOptions._syntheticScale = 64;
		autoOptions._queryCountOverride = 6;
		autoOptions._evaluator = "cpu";
		autoOptions._pauseAtEnd = false;
		autoOptions._rankModelPath.clear();
		autoOptions._autoConditions._enabled = true;
		autoOptions._autoConditions._proxyCandidateCount = 5;
		autoOptions._autoConditions._proxyPointCap = 64;
		autoOptions._autoConditions._proxyQueryCount = 2;
		autoOptions._autoConditions._finalTopK = 3;
		autoOptions._autoConditions._confirmationTopK = 2;
		autoOptions._autoConditions._outputDirectory = (tempRoot / "auto_condition_schemas").string();
		autoOptions._autoConditions._selectorOutputPath = (tempRoot / "auto_condition_selector.json").string();
		autoOptions._generation._outputDirectory = (tempRoot / "generated_auto_conditions").string();
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
		expect(autoRows.size() == autoOptions._autoConditions._confirmationTopK * 3,
			"auto-condition CSV only contains confirmation-stage rows");
		expect(autoHeader.find("conditional_levels") != std::string::npos &&
			autoHeader.find("condition_fields") != std::string::npos &&
			autoHeader.find("condition_summary") != std::string::npos,
			"auto-condition CSV appends condition summary columns");
		{
			const std::vector<std::string> autoColumns = splitCsv(autoHeader);
			const std::vector<std::string> firstAutoValues = splitCsv(autoRows.front());
			const auto foundStage = std::find(autoColumns.begin(), autoColumns.end(), "score_stage");
			const auto foundMode = std::find(autoColumns.begin(), autoColumns.end(), "score_mode");
			expect(foundStage != autoColumns.end() && foundMode != autoColumns.end(),
				"auto-condition CSV includes score provenance");
			const size_t stageIndex = static_cast<size_t>(std::distance(autoColumns.begin(), foundStage));
			const size_t modeIndex = static_cast<size_t>(std::distance(autoColumns.begin(), foundMode));
			expect(stageIndex < firstAutoValues.size() && firstAutoValues[stageIndex] == "confirmation",
				"auto-condition CSV writes only confirmation-stage rows");
			expect(modeIndex < firstAutoValues.size() && firstAutoValues[modeIndex] == "latency",
				"auto-condition confirmation rows use latency score mode");
		}
		expect(std::filesystem::exists(autoOptions._autoConditions._selectorOutputPath),
			"auto-condition tuning writes measured selector artifact");

		std::string cudaError;
		if (PointGpu::MixedTree::isAvailable(&cudaError))
		{
			Experiments::SchemaSearchOptions cudaOptions;
			cudaOptions._schemaPaths = { "configs/schemas/gpu_mixed_all.json" };
			cudaOptions._workloadPaths = { "configs/workloads/volume_small_medium.json" };
			cudaOptions._includeBaselineSchemas = false;
			cudaOptions._csvPath.clear();
			cudaOptions._bestCsvPath.clear();
			cudaOptions._syntheticScale = 64;
			cudaOptions._queryCountOverride = 4;
			cudaOptions._evaluator = "cuda";
			cudaOptions._cuda._device = 0;
			cudaOptions._cuda._builder = "mixed";
			cudaOptions._pauseAtEnd = false;
			expect(Experiments::runSchemaSearch(cudaOptions) == 0, "schema search CUDA mixed smoke run succeeds");

			const std::filesystem::path unsupportedPolicyCsvPath = tempRoot / "cuda_unsupported_policy.csv";
			Experiments::SchemaSearchOptions unsupportedPolicyOptions = cudaOptions;
			unsupportedPolicyOptions._schemaPaths = { "configs/schemas/kdtree.json" };
			unsupportedPolicyOptions._csvPath = unsupportedPolicyCsvPath.string();
			unsupportedPolicyOptions._queryCountOverride = 2;
			expect(Experiments::runSchemaSearch(unsupportedPolicyOptions) == 0,
				"schema search CUDA unsupported split policy falls back cleanly");
			std::ifstream unsupportedPolicyCsv(unsupportedPolicyCsvPath);
			expect(unsupportedPolicyCsv.is_open(), "schema search writes unsupported policy CSV");
			std::string unsupportedPolicyHeader;
			std::string unsupportedPolicyRow;
			std::getline(unsupportedPolicyCsv, unsupportedPolicyHeader);
			std::getline(unsupportedPolicyCsv, unsupportedPolicyRow);
			const std::vector<std::string> unsupportedPolicyColumns = splitCsv(unsupportedPolicyHeader);
			const std::vector<std::string> unsupportedPolicyValues = splitCsv(unsupportedPolicyRow);
			auto unsupportedPolicyColumnIndex = [&unsupportedPolicyColumns](const std::string& name) {
				const auto found = std::find(unsupportedPolicyColumns.begin(), unsupportedPolicyColumns.end(), name);
				return found == unsupportedPolicyColumns.end()
					? std::numeric_limits<size_t>::max()
					: static_cast<size_t>(std::distance(unsupportedPolicyColumns.begin(), found));
			};
			const size_t unsupportedBackendColumn = unsupportedPolicyColumnIndex("backend");
			const size_t unsupportedStatusColumn = unsupportedPolicyColumnIndex("gpu_support_status");
			expect(unsupportedBackendColumn < unsupportedPolicyValues.size() && unsupportedPolicyValues[unsupportedBackendColumn] == "cpu",
				"schema search CUDA unsupported split policy records CPU fallback backend");
			expect(unsupportedStatusColumn < unsupportedPolicyValues.size() && unsupportedPolicyValues[unsupportedStatusColumn] == "unsupported_policy",
				"schema search CUDA unsupported split policy records support status");

			const char* entropyConditionJson = R"json(
			{
			  "name": "cuda_entropy_rejected",
			  "levels": [
			    { "type": "QuadTree", "numLevels": 1, "leafCapacity": 8, "minPointsToSplit": 2 },
			    {
			      "type": "Octree",
			      "numLevels": 1,
			      "leafCapacity": 8,
			      "minPointsToSplit": 2,
			      "condition": {
			        "minOccupancyEntropy": 0.2
			      }
			    }
			  ],
			  "buildPolicy": {
			    "maxDepth": 2,
			    "leafCapacity": 8,
			    "minPointsToSplit": 2,
			    "collapseSingleChild": false,
			    "removeEmptyNodes": true,
			    "allowOverlapDuplication": false
			  }
			}
			)json";
			bool rejectedEntropyCondition = false;
			try
			{
				PointGpu::MixedTree mixedTree;
				PointGpu::Options mixedOptions;
				mixedOptions._device = 0;
				mixedOptions._builder = "mixed";
				const PointCloud entropyCloud = SyntheticPointClouds::generateUrbanMixed(64, 64, 4);
				mixedTree.build(
					entropyCloud,
					Config::parseSchemaConfig(entropyConditionJson, "cuda_entropy_rejected"),
					mixedOptions);
			}
			catch (const std::runtime_error& exception)
			{
				rejectedEntropyCondition =
					std::string(exception.what()).find("occupancy-entropy") != std::string::npos;
			}
			expect(rejectedEntropyCondition,
				"CUDA MixedTree rejects occupancy-entropy conditions instead of silently ignoring them");

			const char* adaptiveLeafJson = R"json(
			{
			  "name": "cuda_adaptive_leaf_rejected",
			  "levels": [
			    {
			      "type": "Octree",
			      "numLevels": 2,
			      "leafCapacity": 8,
			      "minPointsToSplit": 2,
			      "adaptiveLeafCapacity": {
			        "enabled": true,
			        "minCapacity": 2,
			        "maxCapacity": 32,
			        "densityWeight": 1.0
			      }
			    }
			  ],
			  "buildPolicy": {
			    "maxDepth": 2,
			    "leafCapacity": 8,
			    "minPointsToSplit": 2,
			    "collapseSingleChild": false,
			    "removeEmptyNodes": true,
			    "allowOverlapDuplication": false
			  }
			}
			)json";
			bool rejectedAdaptiveLeafCapacity = false;
			try
			{
				PointGpu::MixedTree mixedTree;
				PointGpu::Options mixedOptions;
				mixedOptions._device = 0;
				mixedOptions._builder = "mixed";
				const PointCloud adaptiveCloud = SyntheticPointClouds::generateUrbanMixed(64, 64, 5);
				mixedTree.build(
					adaptiveCloud,
					Config::parseSchemaConfig(adaptiveLeafJson, "cuda_adaptive_leaf_rejected"),
					mixedOptions);
			}
			catch (const std::runtime_error& exception)
			{
				rejectedAdaptiveLeafCapacity =
					std::string(exception.what()).find("adaptiveLeafCapacity") != std::string::npos;
			}
			expect(rejectedAdaptiveLeafCapacity,
				"CUDA MixedTree rejects adaptive leaf-capacity schemas instead of silently ignoring them");

			const char* conditionalSkipJson = R"json(
			{
			  "name": "cuda_conditional_skip",
			  "levels": [
			    { "type": "QuadTree", "numLevels": 1, "leafCapacity": 1, "minPointsToSplit": 2 },
			    {
			      "type": "Octree",
			      "numLevels": 2,
			      "leafCapacity": 1,
			      "minPointsToSplit": 2,
			      "condition": { "minHeightRatio": 999.0 }
			    },
			    { "type": "KDTree", "numLevels": 2, "leafCapacity": 1, "minPointsToSplit": 2, "axisPolicy": "center_longest_axis" }
			  ],
			  "buildPolicy": {
			    "maxDepth": 5,
			    "leafCapacity": 1,
			    "minPointsToSplit": 2,
			    "collapseSingleChild": false,
			    "removeEmptyNodes": true,
			    "allowOverlapDuplication": false
			  }
			}
			)json";
			PointGpu::MixedTree conditionalMixedTree;
			PointGpu::Options conditionalMixedOptions;
			conditionalMixedOptions._device = 0;
			conditionalMixedOptions._builder = "mixed";
			const PointCloud conditionalCloud = SyntheticPointClouds::generateUrbanMixed(64, 64, 4);
			const PointGpu::BuildResult conditionalBuild = conditionalMixedTree.build(
				conditionalCloud,
				Config::parseSchemaConfig(conditionalSkipJson, "cuda_conditional_skip"),
				conditionalMixedOptions);
			expect(conditionalBuild._activeStructureSummary.find("KDTree") != std::string::npos,
				"CUDA MixedTree skips failed conditional Octree block and enters KDTree block");
			expect(conditionalBuild._activeStructureSummary.find("Octree") == std::string::npos,
				"CUDA MixedTree does not count skipped Octree block as active");

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
			cudaKnnOptions._schemaPaths = { "configs/schemas/gpu_mixed_all.json" };
			cudaKnnOptions._workloadPaths = { knnWorkloadPath.string() };
			cudaKnnOptions._includeBaselineSchemas = false;
			cudaKnnOptions._csvPath = cudaKnnCsvPath.string();
			cudaKnnOptions._bestCsvPath.clear();
			cudaKnnOptions._syntheticScale = 64;
			cudaKnnOptions._queryCountOverride = 5;
			cudaKnnOptions._evaluator = "cuda";
			cudaKnnOptions._cuda._device = 0;
			cudaKnnOptions._cuda._builder = "mixed";
			cudaKnnOptions._pauseAtEnd = false;
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
			const size_t cudaKnnBackendColumn = cudaColumnIndex("knn_backend");
			expect(cudaTotalQueriesColumn < cudaValues.size() && cudaKnnQueriesColumn < cudaValues.size(),
				"schema search CUDA KNN CSV includes query count columns");
			expect(static_cast<size_t>(std::stoull(cudaValues[cudaTotalQueriesColumn])) == 5,
				"schema search CUDA KNN workload keeps query override count");
			expect(static_cast<size_t>(std::stoull(cudaValues[cudaKnnQueriesColumn])) == 5,
				"schema search CUDA KNN workload measures KNN on GPU");
			expect(cudaKnnBackendColumn < cudaValues.size() && cudaValues[cudaKnnBackendColumn] == "bruteforce_gpu_scan",
				"schema search CUDA KNN CSV labels brute-force GPU scan backend");
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
			workload._name = "mixed";
			workload._rangeWeight = 0.4;
			workload._radiusWeight = 0.3;
			workload._knnWeight = 0.3;
			workload._numQueries = 16;
			workload._knnK = 8;
			workload._querySeed = 42;
			workload._rangeScaleMin = 0.01;
			workload._rangeScaleMax = 0.05;
			workload._radiusScaleMin = 0.01;
			workload._radiusScaleMax = 0.04;

			Experiments::ScoreWeights weights;
			const glm::vec3 bboxMin(0.0f, 0.0f, 0.0f);
			const glm::vec3 bboxMax(1.0f, 1.0f, 1.0f);
			const Experiments::EvaluationCacheKey key = Experiments::makeEvaluationCacheKey(
				"oct4_l64_kd3_l32", "synthetic_flat", 1234, bboxMin, bboxMax, workload, "cpu", "", weights);

			Experiments::SchemaSearchRecord record;
			record._datasetName = "synthetic_flat";
			record._workloadName = workload._name;
			record._schemaName = "oct4_l64_kd3_l32";
			record._score = 1.75;
			record._buildMetrics._buildTimeMs = 42.0;
			record._buildMetrics._numNodes = 17;
			record._queryMetrics._averageLatencyMs = 0.5;
			record._queryMetrics._averageVisitedNodes = 12.5;
			record._queryMetrics._totalQueries = 16;
			record._rangeMetrics._totalQueries = 6;
			record._rangeMetrics._averageVisitedNodes = 8.0;
			record._knnMetrics._totalQueries = 4;
			record._knnMetrics._averageTestedPoints = 64.0;
			record._backend = "cpu";
			record._scoreMode = "latency";
			record._scoreStage = "final";
			record._scoreIsFinalLatency = true;
			record._isBaseline = true;
			record._activeStructureTypes = 2;
			record._nestedActiveFraction = 0.25;
			record._activeStructureSummary = "ot:nodes=10|points=900;kd:nodes=4|points=100";

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
				roundTrip._datasetName = "caller-set name";
				roundTrip._schemaName = "caller-set schema";
				roundTrip._isBaseline = true;
				roundTrip._weights._lambdaMemory = 0.5;
				roundTrip._scoreMode = "latency";
				roundTrip._scoreStage = "confirm";
				roundTrip._scoreIsFinalLatency = true;
				roundTrip._pointFeatures._numPoints = 999;
				expect(cache.tryGet(key, roundTrip), "evaluation cache hit on identical key");
				expect(nearlyEqual(roundTrip._score, 1.75), "cache restores score");
				expect(nearlyEqual(roundTrip._buildMetrics._buildTimeMs, 42.0), "cache restores build metrics");
				expect(roundTrip._buildMetrics._numNodes == 17, "cache restores build node count");
				expect(nearlyEqual(roundTrip._queryMetrics._averageVisitedNodes, 12.5), "cache restores visit counts");
				expect(roundTrip._rangeMetrics._totalQueries == 6, "cache restores range-family query count");
				expect(nearlyEqual(roundTrip._rangeMetrics._averageVisitedNodes, 8.0), "cache restores range-family metrics");
				expect(roundTrip._knnMetrics._totalQueries == 4, "cache restores knn-family query count");
				expect(nearlyEqual(roundTrip._knnMetrics._averageTestedPoints, 64.0), "cache restores knn-family metrics");
				expect(roundTrip._backend == "cpu", "cache restores backend");
				expect(roundTrip._isBaseline, "cache preserves caller baseline marker");
				expect(roundTrip._scoreMode == "latency", "cache preserves caller score mode");
				expect(roundTrip._scoreStage == "confirm", "cache preserves caller score stage");
				expect(roundTrip._scoreIsFinalLatency, "cache preserves caller final-latency marker");
				expect(roundTrip._activeStructureTypes == 2, "cache restores active structure type count");
				expect(nearlyEqual(roundTrip._nestedActiveFraction, 0.25), "cache restores nested active fraction");
				expect(roundTrip._activeStructureSummary.find("kd:nodes") != std::string::npos,
					"cache restores active structure summary");
				expect(roundTrip._datasetName == "caller-set name", "cache preserves caller dataset name");
				expect(roundTrip._schemaName == "caller-set schema", "cache preserves caller schema name");
				expect(nearlyEqual(roundTrip._weights._lambdaMemory, 0.5), "cache preserves caller weights");
				expect(roundTrip._pointFeatures._numPoints == 999, "cache preserves caller point features");
				expect(cache.hitCount() == 1, "cache counts hit");

				Experiments::WorkloadProfile differentWorkload = workload;
				differentWorkload._querySeed = 7;
				const Experiments::EvaluationCacheKey missKey = Experiments::makeEvaluationCacheKey(
					"oct4_l64_kd3_l32", "synthetic_flat", 1234, bboxMin, bboxMax, differentWorkload, "cpu", "", weights);
				Experiments::SchemaSearchRecord missRecord;
				expect(!cache.tryGet(missKey, missRecord), "cache miss when workload key changes");
				expect(cache.missCount() == 1, "cache counts miss");
			}

			std::filesystem::remove_all(cacheRoot, rmError);
		}

		// Baseline injection: canonical single-block schemas appear in records even in generated-only mode.
		{
			const std::filesystem::path baselineRoot = std::filesystem::temp_directory_path() / "mdspc_baseline_test";
			std::filesystem::create_directories(baselineRoot);
			const std::filesystem::path baselineCsv = baselineRoot / "baseline_search.csv";

			Experiments::SchemaSearchOptions baselineOptions;
			baselineOptions._workloadPaths = { "configs/workloads/volume_small_medium.json" };
			baselineOptions._csvPath = baselineCsv.string();
			baselineOptions._bestCsvPath.clear();
			baselineOptions._syntheticScale = 64;
			baselineOptions._queryCountOverride = 4;
			baselineOptions._evaluator = "cpu";
			baselineOptions._pauseAtEnd = false;
			baselineOptions._includeConfiguredSchemas = false;
			baselineOptions._generation._count = 4;
			baselineOptions._generation._outputDirectory = (baselineRoot / "generated").string();
			baselineOptions._includeBaselineSchemas = true;
			expect(Experiments::runSchemaSearch(baselineOptions) == 0, "schema search runs with baselines + generated");

			std::ifstream baselineCsvStream(baselineCsv);
			std::string baselineHeader;
			std::getline(baselineCsvStream, baselineHeader);
			bool sawQuadtreeBaseline = false;
			bool sawOctreeBaseline = false;
			bool sawKdtreeBaseline = false;
			bool sawBvhBaseline = false;
			bool sawCpuDuplicateBaseline = false;
			std::string baselineRow;
			while (std::getline(baselineCsvStream, baselineRow))
			{
				if (baselineRow.find("quadtree_default") != std::string::npos) sawQuadtreeBaseline = true;
				if (baselineRow.find("octree_default") != std::string::npos) sawOctreeBaseline = true;
				if (baselineRow.find("kdtree_default") != std::string::npos) sawKdtreeBaseline = true;
				if (baselineRow.find("bvh_default") != std::string::npos) sawBvhBaseline = true;
				if (baselineRow.find("lbvh_default") != std::string::npos ||
					baselineRow.find("karras_octree_default") != std::string::npos ||
					baselineRow.find("regular_grid_default") != std::string::npos ||
					baselineRow.find("hgrid_default") != std::string::npos ||
					baselineRow.find("bih_default") != std::string::npos)
				{
					sawCpuDuplicateBaseline = true;
				}
			}
			expect(sawQuadtreeBaseline, "baseline injection measures pure QuadTree as a control");
			expect(sawOctreeBaseline, "baseline injection measures pure Octree as a control");
			expect(sawKdtreeBaseline, "baseline injection measures pure KDTree as a control");
			expect(sawBvhBaseline, "baseline injection measures pure BVH as a control");
			expect(!sawCpuDuplicateBaseline, "CPU baseline injection omits CPU-equivalent GPU aliases");

			std::error_code rmError;
			std::filesystem::remove_all(baselineRoot, rmError);
		}

		// Multi-fidelity cache dedup: fingerprints must distinguish visit-proxy and latency rungs for the same tuple.
		{
			Experiments::WorkloadProfile workload;
			workload._name = "rung_workload";
			workload._numQueries = 8;
			workload._knnK = 4;
			workload._querySeed = 99;
			workload._rangeWeight = 0.5;
			workload._radiusWeight = 0.3;
			workload._knnWeight = 0.2;
			workload._rangeScaleMin = 0.01;
			workload._rangeScaleMax = 0.05;
			workload._radiusScaleMin = 0.01;
			workload._radiusScaleMax = 0.04;

			const glm::vec3 bboxMin(-1.0f, -1.0f, -1.0f);
			const glm::vec3 bboxMax(1.0f, 1.0f, 1.0f);

			Experiments::ScoreWeights latencyWeights;
			Experiments::ScoreWeights proxyWeights;
			proxyWeights._useVisitProxy = true;
			proxyWeights._visitProxyAlpha = 0.1;

			const Experiments::EvaluationCacheKey latencyKey = Experiments::makeEvaluationCacheKey(
				"sig", "rung_ds", 2048, bboxMin, bboxMax, workload, "cpu", "", latencyWeights);
			const Experiments::EvaluationCacheKey proxyKey = Experiments::makeEvaluationCacheKey(
				"sig", "rung_ds", 2048, bboxMin, bboxMax, workload, "cpu", "", proxyWeights);
			expect(latencyKey._evaluatorFingerprint != proxyKey._evaluatorFingerprint,
				"cache fingerprint distinguishes visit-proxy from latency for same workload");

			Experiments::ScoreWeights proxyWeightsOtherAlpha = proxyWeights;
			proxyWeightsOtherAlpha._visitProxyAlpha = 0.5;
			const Experiments::EvaluationCacheKey otherAlphaKey = Experiments::makeEvaluationCacheKey(
				"sig", "rung_ds", 2048, bboxMin, bboxMax, workload, "cpu", "", proxyWeightsOtherAlpha);
			expect(otherAlphaKey._evaluatorFingerprint != proxyKey._evaluatorFingerprint,
				"cache fingerprint distinguishes visit-proxy alpha values");

			// Query count overrides must produce distinct cache entries at the same fidelity mode.
			Experiments::WorkloadProfile longWorkload = workload;
			longWorkload._numQueries = 64;
			const Experiments::EvaluationCacheKey shortKey = Experiments::makeEvaluationCacheKey(
				"sig", "rung_ds", 2048, bboxMin, bboxMax, workload, "cpu", "", latencyWeights);
			const Experiments::EvaluationCacheKey longKey = Experiments::makeEvaluationCacheKey(
				"sig", "rung_ds", 2048, bboxMin, bboxMax, longWorkload, "cpu", "", latencyWeights);
			expect(shortKey._workloadFingerprint != longKey._workloadFingerprint,
				"cache fingerprint distinguishes rungs by effective query count");
		}

		// RungSchedule plumbs through SchemaSearchOptions and survives struct copy/move.
		{
			Experiments::SchemaSearchOptions options;
			expect(options._evolution._rungSchedule._rungs.empty(),
				"default rung schedule is empty (flat GA)");

			Experiments::RungSpec proxy;
			proxy._name = "proxy";
			proxy._queryCountOverride = 4;
			proxy._useVisitProxy = true;
			proxy._visitProxyAlpha = 0.2;
			proxy._advanceTopK = 32;

			Experiments::RungSpec confirm;
			confirm._name = "confirm";
			confirm._queryCountOverride = 64;
			confirm._useVisitProxy = false;
			confirm._advanceTopK = 0;

			options._evolution._rungSchedule._rungs = { proxy, confirm };
			options._evolution._rungSchedule._surrogateProposalsPerStep = 4;
			options._evolution._rungSchedule._surrogateCandidatePool = 32;

			Experiments::SchemaSearchOptions copied = options;
			expect(copied._evolution._rungSchedule._rungs.size() == 2,
				"rung schedule survives SchemaSearchOptions copy");
			expect(copied._evolution._rungSchedule._rungs[0]._useVisitProxy,
				"first rung is the visit-proxy stage");
			expect(!copied._evolution._rungSchedule._rungs[1]._useVisitProxy,
				"second rung is the latency confirmation stage");
			expect(copied._evolution._rungSchedule._rungs[0]._advanceTopK == 32,
				"rung schedule survivor count round-trips");
			expect(copied._evolution._rungSchedule._surrogateProposalsPerStep == 4,
				"surrogate proposal count round-trips");
		}

		// ThresholdRefiner: dims are discovered, the (1+lambda)-ES lowers the score, and minPoints stays in range.
		{
			Experiments::SchemaCandidate seed;
			seed._config._name = "ot_with_conditional_kd";
			seed._config._levels.resize(2);
			seed._config._levels[0]._type = MultiDataStructure::DataStructureLevel::OctreeNode;
			seed._config._levels[0]._typeName = "Octree";
			seed._config._levels[0]._numLevels = 4;
			seed._config._levels[0]._leafCapacity = 64;
			seed._config._levels[1]._type = MultiDataStructure::DataStructureLevel::KDTreeNode;
			seed._config._levels[1]._typeName = "KDTree";
			seed._config._levels[1]._numLevels = 3;
			seed._config._levels[1]._leafCapacity = 32;
			seed._config._levels[1]._condition._minPoints = 4096;     // starting point — refiner should move it
			seed._config._levels[1]._condition._minHeightRatio = 0.3; // second active dim
			seed._path = "test:seed";

			Experiments::ConditionDomain domain;
			domain._pointThresholds = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };
			domain._heightRatioThresholds = { 0.05, 0.1, 0.2, 0.4, 0.6, 0.8 };
			domain._estimatedFromCloud = true;
			domain._samplePoints = 64;

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
					expect(nearlyEqual(dim._lo, 256.0) && nearlyEqual(dim._hi, 32768.0),
						"minPoints bounds come from the supplied domain");
				}
				else if (dim.field == "minHeightRatio")
				{
					foundHeight = true;
					expect(!dim.integerValued && !dim.logScale,
						"minHeightRatio uses continuous linear encoding");
					expect(dim._lo < dim._hi && dim._lo >= 0.05 && dim._hi <= 0.8,
						"minHeightRatio bounds come from the supplied domain");
				}
			}
			expect(foundPoints && foundHeight,
				"refiner reports both minPoints and minHeightRatio as active dims");

			// Synthetic quadratic bowl minimised at minPoints ~ 1024 and minHeightRatio ~ 0.4.
			auto syntheticScore = [&](const Experiments::SchemaCandidate& candidate) {
				if (candidate._config._levels.size() < 2)
					return 1.0e9;
				const SchemaLevelCondition& c = candidate._config._levels[1]._condition;
				const double minP = c._minPoints.has_value() ? static_cast<double>(c._minPoints.value()) : 4096.0;
				const double minH = c._minHeightRatio.value_or(0.3);
				const double pointError = std::log2(std::max(minP, 1.0)) - std::log2(1024.0);
				const double heightError = minH - 0.4;
				return pointError * pointError + 12.0 * heightError * heightError;
			};

			Experiments::ThresholdRefinementOptions refineOptions;
			refineOptions._enabled = true;
			refineOptions._maxEvaluations = 50;
			refineOptions._populationLambda = 5;
			refineOptions._sigma0 = 0.4;
			refineOptions._seed = 4242;
			refineOptions._outputDirectory.clear();

			const Experiments::ThresholdRefinementResult result =
				Experiments::refineSchemaThresholds(seed, domain, refineOptions, syntheticScore);
			expect(result._dimensions == 2, "refinement result reports 2 active dimensions");
			expect(result._evaluationsUsed >= 6,
				"refinement consumes at least one generation worth of evaluations");
			expect(result._refinedScore < result._initialScore,
				"refinement drives the synthetic objective below its starting value");
			expect(result._refinedCandidate._config._levels.size() == 2,
				"refined candidate preserves discrete topology");
			expect(result._refinedCandidate._config._levels[1]._condition._minPoints.has_value(),
				"refined candidate still carries the minPoints threshold");
			const size_t refinedMinPoints = result._refinedCandidate._config._levels[1]._condition._minPoints.value();
			expect(refinedMinPoints >= 256 && refinedMinPoints <= 32768,
				"refined minPoints stays inside the supplied domain bounds");
			expect(result._refinedCandidate._config._levels[1]._condition._minHeightRatio.has_value(),
				"refined candidate still carries the minHeightRatio threshold");

			// Topology must be frozen: types, level counts, and leaf capacities are untouched.
			expect(result._refinedCandidate._config._levels[0]._typeName == seed._config._levels[0]._typeName,
				"refinement does not change level types");
			expect(result._refinedCandidate._config._levels[0]._leafCapacity == seed._config._levels[0]._leafCapacity,
				"refinement does not change leaf capacity");
			expect(result._refinedCandidate._config._levels[1]._numLevels == seed._config._levels[1]._numLevels,
				"refinement does not change level counts");
		}

		// Pareto front: selectParetoRecords returns exactly the non-dominated set, ranked by latency.
		{
			auto makeRecord = [](const std::string& dataset, const std::string& workload,
				const std::string& schemaName, double latencyMs, double buildMs,
				size_t memoryBytes, double avgOccupancy, size_t maxOccupancy) {
				Experiments::SchemaSearchRecord record;
				record._datasetName = dataset;
				record._workloadName = workload;
				record._schemaName = schemaName;
				record._queryMetrics._averageLatencyMs = latencyMs;
				record._buildMetrics._buildTimeMs = buildMs;
				record._buildMetrics._memoryEstimateBytes = memoryBytes;
				record._buildMetrics._averageLeafOccupancy = avgOccupancy;
				record._buildMetrics._maxLeafOccupancy = maxOccupancy;
				record._score = latencyMs;
				return record;
			};

			// Group A: latency-tradeoff front where fast_big, balanced, tiny, and matches_balanced survive but dominated does not.
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
				if (entry._datasetName == "ds_a")
				{
					++groupA;
					expect(entry._paretoRank >= 0, "front entry carries non-negative paretoRank");
					if (previousLatencyRank == -1 || entry._paretoRank == 0)
					{
						previousLatency = entry._queryMetrics._averageLatencyMs;
						previousLatencyRank = entry._paretoRank;
					}
					else
					{
						expect(entry._queryMetrics._averageLatencyMs >= previousLatency,
							"pareto front sorted by ascending latency within group");
						previousLatency = entry._queryMetrics._averageLatencyMs;
					}
					if (entry._schemaName == "dominated")
						sawDominated = true;
					if (entry._schemaName == "tiny")
						sawTiny = true;
					if (entry._schemaName == "fast_big")
						sawFast = true;
					if (entry._schemaName == "balanced")
						sawBalanced = true;
					if (entry._schemaName == "matches_balanced")
						sawCofront = true;
				}
				else if (entry._datasetName == "ds_b")
				{
					++groupB;
					expect(entry._paretoRank == 0, "single-entry group ranks the lone candidate at 0");
				}
			}

			expect(!sawDominated, "dominated candidate is filtered out of the Pareto front");
			expect(sawFast && sawTiny && sawBalanced, "non-dominated trio survives the front");
			expect(sawCofront, "candidate with identical metrics to the balanced one is co-front (no strict domination)");
			expect(groupA == 4, "ds_a front contains 4 non-dominated rows (fast_big, balanced, matches_balanced, tiny)");
			expect(groupB == 1, "ds_b front contains the single measured candidate");
			expect(front.front()._paretoRank == 0, "first row of returned front is rank 0");
		}

		// Bootstrap mean + 95% CI helper.
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

		// Pareto dominance uses latencyMean after multi-seed confirmation, so a lucky single-seed record cannot dominate a better mean.
		{
			auto makeRecord = [](const std::string& schemaName, double latencyMs, double buildMs,
				size_t memoryBytes, double avgOccupancy, size_t maxOccupancy,
				size_t confirmSeeds, double latencyMean) {
				Experiments::SchemaSearchRecord record;
				record._datasetName = "ds";
				record._workloadName = "wl";
				record._schemaName = schemaName;
				record._queryMetrics._averageLatencyMs = latencyMs;
				record._buildMetrics._buildTimeMs = buildMs;
				record._buildMetrics._memoryEstimateBytes = memoryBytes;
				record._buildMetrics._averageLeafOccupancy = avgOccupancy;
				record._buildMetrics._maxLeafOccupancy = maxOccupancy;
				record._score = latencyMs;
				record._confirmSeedsUsed = confirmSeeds;
				record._latencyMean = latencyMean;
				return record;
			};

			std::vector<Experiments::SchemaSearchRecord> raw;
			// A: noisy single-seed point estimate is 0.1 ms but the confirmed mean across 5 seeds is 1.0 ms.
			raw.push_back(makeRecord("noisy_lucky", 0.1, 10.0, 50ull * 1024 * 1024, 40.0, 60, 5, 1.0));
			// B: confirmed mean of 0.5 ms beats A's confirmed mean but not A's single-seed estimate.
			raw.push_back(makeRecord("stable",       0.6, 10.0, 50ull * 1024 * 1024, 40.0, 60, 5, 0.5));

			const std::vector<Experiments::SchemaSearchRecord> front = Experiments::selectParetoRecords(raw);
			bool sawNoisy = false;
			bool sawStable = false;
			for (const auto& entry : front)
			{
				if (entry._schemaName == "noisy_lucky") sawNoisy = true;
				if (entry._schemaName == "stable")      sawStable = true;
			}
			expect(sawStable, "stable candidate (best confirmed mean) survives the Pareto front");
			expect(!sawNoisy, "noisy single-seed lucky candidate is dominated by the confirmed mean");
		}

		// Candidates with no conditional levels short-circuit: no evaluations, candidate unchanged.
		{
			Experiments::SchemaCandidate flat;
			flat._config._name = "octree_flat";
			flat._config._levels.resize(1);
			flat._config._levels[0]._type = MultiDataStructure::DataStructureLevel::OctreeNode;
			flat._config._levels[0]._typeName = "Octree";
			flat._config._levels[0]._numLevels = 6;
			flat._config._levels[0]._leafCapacity = 1024;

			Experiments::ConditionDomain emptyDomain;
			Experiments::ThresholdRefinementOptions refineOptions;
			refineOptions._enabled = true;
			refineOptions._maxEvaluations = 20;
			refineOptions._outputDirectory.clear();

			size_t evalCount = 0;
			auto trackedScore = [&](const Experiments::SchemaCandidate&) {
				++evalCount;
				return 1.0;
			};

			const Experiments::ThresholdRefinementResult result =
				Experiments::refineSchemaThresholds(flat, emptyDomain, refineOptions, trackedScore);
			expect(result._dimensions == 0, "no active dims when no conditional levels exist");
			expect(evalCount <= 1, "refiner does not waste evaluations on a flat candidate");
			expect(result._refinedCandidate._config._name == flat._config._name,
				"flat candidate is returned unchanged");
		}

		// Measurement-reliability: confidentlyBetter and annotateRankingConfidence with default ScoreWeights (score equals average latency).
		{
			auto makeRecord = [](const std::string& schema, double avgLatency, size_t repeats,
				double ciLow, double ciHigh) {
				Experiments::SchemaSearchRecord r;
				r._datasetName = "ds";
				r._workloadName = "wl";
				r._schemaName = schema;
				r._queryMetrics._averageLatencyMs = avgLatency;
				r._queryMetrics._latencyMeanMs = avgLatency;
				r._queryMetrics._measurementRepeats = repeats;
				r._queryMetrics._latencyCiLowMs = ciLow;
				r._queryMetrics._latencyCiHighMs = ciHigh;
				r._score = avgLatency;
				return r;
			};

			const Experiments::SchemaSearchRecord fast = makeRecord("fast", 1.0, 5, 0.9, 1.1);
			const Experiments::SchemaSearchRecord slow = makeRecord("slow", 2.0, 5, 1.8, 2.2);
			expect(Experiments::confidentlyBetter(fast, slow), "fast confidently beats slow (disjoint repeat CIs)");
			expect(!Experiments::confidentlyBetter(slow, fast), "slow does not confidently beat fast");

			const Experiments::SchemaSearchRecord overlapA = makeRecord("a", 1.0, 5, 0.5, 1.5);
			const Experiments::SchemaSearchRecord overlapB = makeRecord("b", 1.2, 5, 0.7, 1.7);
			expect(!Experiments::confidentlyBetter(overlapA, overlapB), "overlapping repeat CIs are not confidently separable");

			const Experiments::SchemaSearchRecord oneShotFast = makeRecord("s1", 1.0, 1, 1.0, 1.0);
			const Experiments::SchemaSearchRecord oneShotSlow = makeRecord("s2", 5.0, 1, 5.0, 5.0);
			expect(!Experiments::confidentlyBetter(oneShotFast, oneShotSlow), "single-shot measurement yields no confidence");

			std::vector<Experiments::SchemaSearchRecord> separable = { slow, fast };
			Experiments::annotateRankingConfidence(separable);
			expect(separable[0]._rankingConfident && separable[1]._rankingConfident,
				"group with separable winner is flagged confident on all rows");

			std::vector<Experiments::SchemaSearchRecord> overlapping = { overlapA, overlapB };
			Experiments::annotateRankingConfidence(overlapping);
			expect(!overlapping[0]._rankingConfident && !overlapping[1]._rankingConfident,
				"group with overlapping winner is not flagged confident");

			std::vector<Experiments::SchemaSearchRecord> lone = { fast };
			Experiments::annotateRankingConfidence(lone);
			expect(lone[0]._rankingConfident, "single-candidate group is trivially confident");
		}

		// Spearman rank correlation used by the proxy/latency validation report.
		{
			expect(nearlyEqual(Experiments::spearmanRankCorrelation({ 1.0, 2.0, 3.0, 4.0 }, { 10.0, 20.0, 30.0, 40.0 }), 1.0),
				"spearman perfect positive");
			expect(nearlyEqual(Experiments::spearmanRankCorrelation({ 1.0, 2.0, 3.0, 4.0 }, { 40.0, 30.0, 20.0, 10.0 }), -1.0),
				"spearman perfect negative");
			expect(nearlyEqual(Experiments::spearmanRankCorrelation({ 1.0, 2.0, 3.0, 4.0 }, { 1.0, 4.0, 9.0, 100.0 }), 1.0),
				"spearman is monotonic, not linear");
			expect(nearlyEqual(Experiments::spearmanRankCorrelation({ 1.0 }, { 2.0 }), 0.0),
				"spearman returns 0 for fewer than two points");
			expect(nearlyEqual(Experiments::spearmanRankCorrelation({ 5.0, 5.0, 5.0 }, { 1.0, 2.0, 3.0 }), 0.0),
				"spearman returns 0 for a zero-variance series");
		}

		// Pareto knee selection: among a latency/build tradeoff front, the balanced entry is the knee.
		{
			auto mk = [](const std::string& name, double latency, double build) {
				Experiments::SchemaSearchRecord r;
				r._datasetName = "ds";
				r._workloadName = "wl";
				r._schemaName = name;
				r._queryMetrics._averageLatencyMs = latency;
				r._buildMetrics._buildTimeMs = build;
				r._buildMetrics._memoryEstimateBytes = 1000; // equal memory + imbalance across all
				return r;
			};

			// fast: low latency, slow build; slow: high latency, fast build; balanced: middle of both.
			std::vector<Experiments::SchemaSearchRecord> recs = {
				mk("fast", 1.0, 10.0), mk("balanced", 2.0, 2.0), mk("slow", 10.0, 1.0)
			};
			const std::vector<Experiments::SchemaSearchRecord> frontRecords = Experiments::selectParetoRecords(recs);
			expect(frontRecords.size() == 3, "all three tradeoff candidates are non-dominated");

			std::string knee;
			std::string fastest;
			for (const Experiments::SchemaSearchRecord& r : frontRecords)
			{
				if (r._paretoKnee)
					knee = r._schemaName;
				if (r._paretoRank == 0)
					fastest = r._schemaName;
			}
			expect(knee == "balanced", "pareto knee is the balanced compromise");
			expect(fastest == "fast", "pareto rank 0 is the lowest-latency entry");
		}

		// Zero-build query-cost estimate: coarser leaves cost more despite fewer nodes; a structureless schema estimates a full scan.
		{
			Experiments::PointCloudFeatures features;
			features._numPoints = 1000000;
			Experiments::WorkloadFeatures workload;
			workload._queryScaleMean = 0.1;
			workload._wRange = 1.0;

			auto octreeWithCapacity = [](size_t cap) {
				SchemaConfig schema;
				schema._name = "octree_cap";
				SchemaLevelConfig level;
				level._primitiveKind = SchemaPrimitiveKind::Octree;
				level._typeName = "Octree";
				level._numLevels = 10;
				level._leafCapacity = cap;
				schema._levels.push_back(level);
				schema._buildPolicy._leafCapacity = cap;
				return schema;
			};

			const double fine = Experiments::estimateSchemaQueryCost(octreeWithCapacity(16), features, workload, 0.1);
			const double coarse = Experiments::estimateSchemaQueryCost(octreeWithCapacity(4096), features, workload, 0.1);
			expect(fine > 0.0 && coarse > 0.0, "estimated query cost is positive");
			expect(coarse > fine, "coarser leaves test more points per query (higher estimated cost)");

			SchemaConfig empty;
			empty._name = "empty";
			expect(nearlyEqual(Experiments::estimateSchemaQueryCost(empty, features, workload, 0.1), 1000000.0),
				"structureless schema estimates a full scan");
		}
	}
}

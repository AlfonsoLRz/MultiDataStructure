#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/experiments/SchemaSearch.h"

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
	}
}

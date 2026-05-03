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
		  "querySeed": 99
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

		Experiments::BuildMetrics buildMetrics;
		buildMetrics.buildTimeMs = 10.0;
		buildMetrics.memoryEstimateBytes = 2 * 1024 * 1024;
		buildMetrics.averageLeafOccupancy = 4.0;
		buildMetrics.maxLeafOccupancy = 12;

		Experiments::QueryMetrics queryMetrics;
		queryMetrics.averageLatencyMs = 2.0;

		double memoryMb = 0.0;
		double imbalancePenalty = 0.0;
		const double score = Experiments::computeSchemaSearchScore(
			buildMetrics,
			queryMetrics,
			Experiments::ScoreWeights{},
			memoryMb,
			imbalancePenalty);

		expect(nearlyEqual(memoryMb, 2.0), "schema search score computes memory MB");
		expect(nearlyEqual(imbalancePenalty, 3.0), "schema search score computes imbalance penalty");
		expect(nearlyEqual(score, 2.06), "schema search score combines latency, build, memory, and imbalance");

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
	}
}

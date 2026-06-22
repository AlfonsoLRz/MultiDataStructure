#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/experiments/Metrics.h"
#include "../MultiDataStructure/workloads/points/PointSpatialIndex.h"
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

		SchemaConfig makeMetricsSchema()
		{
			SchemaLevelConfig level;
			level._type = MultiDataStructure::DataStructureLevel::OctreeNode;
			level._typeName = "Octree";
			level._numLevels = 4;
			level._leafCapacity = 8;
			level._minPrimitivesToSplit = 4;

			SchemaConfig schema;
			schema._name = "metrics_test";
			schema._levels.push_back(level);
			schema._buildPolicy._maxDepth = 4;
			schema._buildPolicy._leafCapacity = 8;
			schema._buildPolicy._minPrimitivesToSplit = 4;
			schema._buildPolicy._collapseSingleChild = false;
			schema._buildPolicy._removeEmptyNodes = true;
			schema._buildPolicy._allowOverlapDuplication = false;
			return schema;
		}
	}

	void runMetricsTests()
	{
		std::vector<PointSpatialIndex::QueryStats> querySamples(4);
		querySamples[0]._elapsedMs = 1.0;
		querySamples[0]._visitedNodes = 2;
		querySamples[0]._testedPoints = 10;
		querySamples[0]._returnedPoints = 1;
		querySamples[1]._elapsedMs = 3.0;
		querySamples[1]._visitedNodes = 4;
		querySamples[1]._testedPoints = 20;
		querySamples[1]._returnedPoints = 2;
		querySamples[2]._elapsedMs = 2.0;
		querySamples[2]._visitedNodes = 6;
		querySamples[2]._testedPoints = 30;
		querySamples[2]._returnedPoints = 3;
		querySamples[3]._elapsedMs = 100.0;
		querySamples[3]._visitedNodes = 8;
		querySamples[3]._testedPoints = 40;
		querySamples[3]._returnedPoints = 4;

		const Experiments::QueryMetrics queryMetrics = Experiments::summarizeQueryStats(querySamples);
		expect(queryMetrics._totalQueries == 4, "query metrics count samples");
		expect(nearlyEqual(queryMetrics._averageLatencyMs, 26.5), "query metrics average latency");
		expect(nearlyEqual(queryMetrics._medianLatencyMs, 2.0), "query metrics median latency");
		expect(nearlyEqual(queryMetrics._p95LatencyMs, 100.0), "query metrics p95 latency");
		expect(nearlyEqual(queryMetrics._averageVisitedNodes, 5.0), "query metrics average visited nodes");
		expect(queryMetrics._totalTestedPoints == 100, "query metrics total tested points");
		expect(queryMetrics._totalReturnedPoints == 10, "query metrics total returned points");
		expect(queryMetrics._throughputQueriesPerSecond > 0.0, "query metrics throughput");

		// Single-batch summaries degenerate the measurement-reliability fields to the point estimate.
		expect(queryMetrics._measurementRepeats == 1, "query metrics single-batch repeat count");
		expect(nearlyEqual(queryMetrics._latencyMeanMs, queryMetrics._averageLatencyMs), "query metrics latency mean equals average");
		expect(nearlyEqual(queryMetrics._latencyStdDevMs, 0.0), "query metrics single-batch stddev is zero");
		expect(nearlyEqual(queryMetrics._latencyCoeffVar, 0.0), "query metrics single-batch cv is zero");
		expect(nearlyEqual(queryMetrics._latencyCiLowMs, queryMetrics._averageLatencyMs), "query metrics ci low equals average");
		expect(nearlyEqual(queryMetrics._latencyCiHighMs, queryMetrics._averageLatencyMs), "query metrics ci high equals average");

		const PointCloud cloud = SyntheticPointClouds::generateSparseDenseMixture(32, 64, 7);
		PointSpatialIndex index;
		index.build(cloud, makeMetricsSchema());

		const PointSpatialIndex::Stats stats = index.stats();
		const Experiments::BuildMetrics buildMetrics = Experiments::collectBuildMetrics(stats, index.root(), 12.5);
		expect(nearlyEqual(buildMetrics._buildTimeMs, 12.5), "build metrics preserve build time");
		expect(buildMetrics._numNodes == stats._numNodes, "build metrics preserve node count");
		expect(buildMetrics._numLeaves == stats._numLeaves, "build metrics preserve leaf count");
		expect(buildMetrics._indexedPoints == cloud.size(), "build metrics preserve indexed points");
		expect(buildMetrics._averageLeafOccupancy > 0.0, "build metrics compute average leaf occupancy");
		expect(buildMetrics._maxLeafOccupancy > 0, "build metrics compute max leaf occupancy");
		expect(buildMetrics._leafOccupancyP50 > 0.0, "build metrics compute leaf occupancy median");
		expect(buildMetrics._leafOccupancyP90 >= buildMetrics._leafOccupancyP50, "build metrics compute ordered leaf occupancy quantiles");
		expect(buildMetrics._averageDepth > 0.0, "build metrics compute average depth");
		expect(buildMetrics._averageFanout > 0.0, "build metrics compute average fanout");
		expect(buildMetrics._maxFanout > 0, "build metrics compute max fanout");
		expect(buildMetrics._emptyChildRatio >= 0.0 && buildMetrics._emptyChildRatio <= 1.0, "build metrics compute bounded empty-child ratio");
		expect(buildMetrics._meanTightBoundsVolumeRatio >= 0.0 && buildMetrics._meanTightBoundsVolumeRatio <= 1.0,
			"build metrics compute bounded tight-bounds volume ratio");
		expect(!buildMetrics._nodeFanoutSummary.empty(), "build metrics report fanout distribution");
		expect(buildMetrics._memoryEstimateBytes > 0, "build metrics compute memory estimate");
	}
}

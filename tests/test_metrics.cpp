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
			level.type = MultiDataStructure::DataStructureLevel::OctreeNode;
			level.typeName = "Octree";
			level.numLevels = 4;
			level.leafCapacity = 8;
			level.minPrimitivesToSplit = 4;

			SchemaConfig schema;
			schema.name = "metrics_test";
			schema.levels.push_back(level);
			schema.buildPolicy.maxDepth = 4;
			schema.buildPolicy.leafCapacity = 8;
			schema.buildPolicy.minPrimitivesToSplit = 4;
			schema.buildPolicy.collapseSingleChild = false;
			schema.buildPolicy.removeEmptyNodes = true;
			schema.buildPolicy.allowOverlapDuplication = false;
			return schema;
		}
	}

	void runMetricsTests()
	{
		std::vector<PointSpatialIndex::QueryStats> querySamples(4);
		querySamples[0].elapsedMs = 1.0;
		querySamples[0].visitedNodes = 2;
		querySamples[0].testedPoints = 10;
		querySamples[0].returnedPoints = 1;
		querySamples[1].elapsedMs = 3.0;
		querySamples[1].visitedNodes = 4;
		querySamples[1].testedPoints = 20;
		querySamples[1].returnedPoints = 2;
		querySamples[2].elapsedMs = 2.0;
		querySamples[2].visitedNodes = 6;
		querySamples[2].testedPoints = 30;
		querySamples[2].returnedPoints = 3;
		querySamples[3].elapsedMs = 100.0;
		querySamples[3].visitedNodes = 8;
		querySamples[3].testedPoints = 40;
		querySamples[3].returnedPoints = 4;

		const Experiments::QueryMetrics queryMetrics = Experiments::summarizeQueryStats(querySamples);
		expect(queryMetrics.totalQueries == 4, "query metrics count samples");
		expect(nearlyEqual(queryMetrics.averageLatencyMs, 26.5), "query metrics average latency");
		expect(nearlyEqual(queryMetrics.medianLatencyMs, 2.0), "query metrics median latency");
		expect(nearlyEqual(queryMetrics.p95LatencyMs, 100.0), "query metrics p95 latency");
		expect(nearlyEqual(queryMetrics.averageVisitedNodes, 5.0), "query metrics average visited nodes");
		expect(queryMetrics.totalTestedPoints == 100, "query metrics total tested points");
		expect(queryMetrics.totalReturnedPoints == 10, "query metrics total returned points");
		expect(queryMetrics.throughputQueriesPerSecond > 0.0, "query metrics throughput");

		// Single-batch summaries degenerate the measurement-reliability fields to the point estimate.
		expect(queryMetrics.measurementRepeats == 1, "query metrics single-batch repeat count");
		expect(nearlyEqual(queryMetrics.latencyMeanMs, queryMetrics.averageLatencyMs), "query metrics latency mean equals average");
		expect(nearlyEqual(queryMetrics.latencyStdDevMs, 0.0), "query metrics single-batch stddev is zero");
		expect(nearlyEqual(queryMetrics.latencyCoeffVar, 0.0), "query metrics single-batch cv is zero");
		expect(nearlyEqual(queryMetrics.latencyCiLowMs, queryMetrics.averageLatencyMs), "query metrics ci low equals average");
		expect(nearlyEqual(queryMetrics.latencyCiHighMs, queryMetrics.averageLatencyMs), "query metrics ci high equals average");

		const PointCloud cloud = SyntheticPointClouds::generateSparseDenseMixture(32, 64, 7);
		PointSpatialIndex index;
		index.build(cloud, makeMetricsSchema());

		const PointSpatialIndex::Stats stats = index.stats();
		const Experiments::BuildMetrics buildMetrics = Experiments::collectBuildMetrics(stats, index.root(), 12.5);
		expect(nearlyEqual(buildMetrics.buildTimeMs, 12.5), "build metrics preserve build time");
		expect(buildMetrics.numNodes == stats.numNodes, "build metrics preserve node count");
		expect(buildMetrics.numLeaves == stats.numLeaves, "build metrics preserve leaf count");
		expect(buildMetrics.indexedPoints == cloud.size(), "build metrics preserve indexed points");
		expect(buildMetrics.averageLeafOccupancy > 0.0, "build metrics compute average leaf occupancy");
		expect(buildMetrics.maxLeafOccupancy > 0, "build metrics compute max leaf occupancy");
		expect(buildMetrics.leafOccupancyP50 > 0.0, "build metrics compute leaf occupancy median");
		expect(buildMetrics.leafOccupancyP90 >= buildMetrics.leafOccupancyP50, "build metrics compute ordered leaf occupancy quantiles");
		expect(buildMetrics.averageDepth > 0.0, "build metrics compute average depth");
		expect(buildMetrics.averageFanout > 0.0, "build metrics compute average fanout");
		expect(buildMetrics.maxFanout > 0, "build metrics compute max fanout");
		expect(buildMetrics.emptyChildRatio >= 0.0 && buildMetrics.emptyChildRatio <= 1.0, "build metrics compute bounded empty-child ratio");
		expect(buildMetrics.meanTightBoundsVolumeRatio >= 0.0 && buildMetrics.meanTightBoundsVolumeRatio <= 1.0,
			"build metrics compute bounded tight-bounds volume ratio");
		expect(!buildMetrics.nodeFanoutSummary.empty(), "build metrics report fanout distribution");
		expect(buildMetrics.memoryEstimateBytes > 0, "build metrics compute memory estimate");
	}
}

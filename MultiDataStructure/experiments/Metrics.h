#pragma once

#include "../stdafx.h"
#include "../workloads/points/PointSpatialIndex.h"

namespace Experiments
{
	struct BuildMetrics
	{
		double buildTimeMs = 0.0;
		size_t numNodes = 0;
		size_t numLeaves = 0;
		size_t indexedPoints = 0;
		size_t maxDepth = 0;
		double averageLeafOccupancy = 0.0;
		size_t maxLeafOccupancy = 0;
		double leafOccupancyP50 = 0.0;
		double leafOccupancyP90 = 0.0;
		double leafOccupancyP99 = 0.0;
		double averageDepth = 0.0;
		double averageFanout = 0.0;
		size_t maxFanout = 0;
		double emptyChildRatio = 0.0;
		size_t singleChildNodeCount = 0;
		double meanTightBoundsVolumeRatio = 0.0;
		size_t microIndexedLeaves = 0;
		size_t microIndexedPoints = 0;
		std::string nodeFanoutSummary;
		size_t memoryEstimateBytes = 0;
	};

	struct QueryMetrics
	{
		size_t totalQueries = 0;
		double totalLatencyMs = 0.0;
		double averageLatencyMs = 0.0;
		double medianLatencyMs = 0.0;
		double p95LatencyMs = 0.0;
		double throughputQueriesPerSecond = 0.0;
		double averageVisitedNodes = 0.0;
		double averageTestedPoints = 0.0;
		double averageReturnedPoints = 0.0;
		double averageFullyContainedNodes = 0.0;
		size_t totalVisitedNodes = 0;
		size_t totalTestedPoints = 0;
		size_t totalReturnedPoints = 0;
		size_t totalFullyContainedNodes = 0;
		// Run-to-run timing noise over measurementRepeats re-timed batches; point estimate when 1.
		size_t measurementRepeats = 1;
		double latencyMeanMs = 0.0;
		double latencyStdDevMs = 0.0;
		double latencyCoeffVar = 0.0;
		double latencyCiLowMs = 0.0;
		double latencyCiHighMs = 0.0;
	};

	BuildMetrics collectBuildMetrics(const PointSpatialIndex::Stats& stats, const PointSpatialIndex::Node* root, double buildTimeMs);
	BuildMetrics collectBuildMetrics(const PointSpatialIndex::Stats& stats, const PointSpatialIndex::Node* root, double buildTimeMs, const SchemaConfig& schema);
	QueryMetrics summarizeQueryStats(const std::vector<PointSpatialIndex::QueryStats>& samples);
}

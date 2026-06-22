#pragma once

#include "../stdafx.h"
#include "../workloads/points/PointSpatialIndex.h"

namespace Experiments
{
	struct BuildMetrics
	{
		double		_buildTimeMs = 0.0;
		size_t		_numNodes = 0;
		size_t		_numLeaves = 0;
		size_t		_indexedPoints = 0;
		size_t		_maxDepth = 0;
		double		_averageLeafOccupancy = 0.0;
		size_t		_maxLeafOccupancy = 0;
		double		_leafOccupancyP50 = 0.0;
		double		_leafOccupancyP90 = 0.0;
		double		_leafOccupancyP99 = 0.0;
		double		_averageDepth = 0.0;
		double		_averageFanout = 0.0;
		size_t		_maxFanout = 0;
		double		_emptyChildRatio = 0.0;
		size_t		_singleChildNodeCount = 0;
		double		_meanTightBoundsVolumeRatio = 0.0;
		size_t		_microIndexedLeaves = 0;
		size_t		_microIndexedPoints = 0;
		std::string	_nodeFanoutSummary;
		size_t		_memoryEstimateBytes = 0;
	};

	struct QueryMetrics
	{
		size_t	_totalQueries = 0;
		double	_totalLatencyMs = 0.0;
		double	_averageLatencyMs = 0.0;
		double	_medianLatencyMs = 0.0;
		double	_p95LatencyMs = 0.0;
		double	_throughputQueriesPerSecond = 0.0;
		double	_averageVisitedNodes = 0.0;
		double	_averageTestedPoints = 0.0;
		double	_averageReturnedPoints = 0.0;
		double	_averageFullyContainedNodes = 0.0;
		size_t	_totalVisitedNodes = 0;
		size_t	_totalTestedPoints = 0;
		size_t	_totalReturnedPoints = 0;
		size_t	_totalFullyContainedNodes = 0;
		// Run-to-run timing noise over measurementRepeats re-timed batches; point estimate when 1.
		size_t	_measurementRepeats = 1;
		double	_latencyMeanMs = 0.0;
		double	_latencyStdDevMs = 0.0;
		double	_latencyCoeffVar = 0.0;
		double	_latencyCiLowMs = 0.0;
		double	_latencyCiHighMs = 0.0;
	};

	BuildMetrics collectBuildMetrics(const PointSpatialIndex::Stats& stats, const PointSpatialIndex::Node* root, double buildTimeMs);
	BuildMetrics collectBuildMetrics(const PointSpatialIndex::Stats& stats, const PointSpatialIndex::Node* root, double buildTimeMs, const SchemaConfig& schema);
	QueryMetrics summarizeQueryStats(const std::vector<PointSpatialIndex::QueryStats>& samples);
}

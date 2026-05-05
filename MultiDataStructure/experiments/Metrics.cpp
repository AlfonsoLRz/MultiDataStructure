#include "../stdafx.h"
#include "Metrics.h"

namespace
{
	void collectLeafOccupancies(const PointSpatialIndex::Node* node, std::vector<size_t>& occupancies)
	{
		if (!node)
			return;

		if (node->isLeaf())
		{
			occupancies.push_back(node->pointIndices.size());
			return;
		}

		for (const std::unique_ptr<PointSpatialIndex::Node>& child : node->children)
			collectLeafOccupancies(child.get(), occupancies);
	}

	double percentile(std::vector<double> values, double p)
	{
		if (values.empty())
			return 0.0;

		std::sort(values.begin(), values.end());
		const double clamped = std::clamp(p, 0.0, 1.0);
		const size_t index = static_cast<size_t>(std::ceil(clamped * static_cast<double>(values.size())) - 1.0);
		return values[std::min(index, values.size() - 1)];
	}
}

Experiments::BuildMetrics Experiments::collectBuildMetrics(const PointSpatialIndex::Stats& stats, const PointSpatialIndex::Node* root, double buildTimeMs)
{
	BuildMetrics metrics;
	metrics.buildTimeMs = buildTimeMs;
	metrics.numNodes = stats.numNodes;
	metrics.numLeaves = stats.numLeaves;
	metrics.indexedPoints = stats.numPoints;
	metrics.maxDepth = stats.maxDepth;

	std::vector<size_t> occupancies;
	collectLeafOccupancies(root, occupancies);
	if (!occupancies.empty())
	{
		const size_t totalOccupancy = std::accumulate(occupancies.begin(), occupancies.end(), size_t(0));
		metrics.averageLeafOccupancy = static_cast<double>(totalOccupancy) / static_cast<double>(occupancies.size());
		metrics.maxLeafOccupancy = *std::max_element(occupancies.begin(), occupancies.end());
	}

	metrics.memoryEstimateBytes =
		stats.numNodes * sizeof(PointSpatialIndex::Node) +
		stats.numPoints * sizeof(uint32_t);
	return metrics;
}

Experiments::QueryMetrics Experiments::summarizeQueryStats(const std::vector<PointSpatialIndex::QueryStats>& samples)
{
	QueryMetrics metrics;
	metrics.totalQueries = samples.size();
	if (samples.empty())
		return metrics;

	std::vector<double> latencies;
	latencies.reserve(samples.size());
	for (const PointSpatialIndex::QueryStats& sample : samples)
	{
		latencies.push_back(sample.elapsedMs);
		metrics.totalLatencyMs += sample.elapsedMs;
		metrics.totalVisitedNodes += sample.visitedNodes;
		metrics.totalTestedPoints += sample.testedPoints;
		metrics.totalReturnedPoints += sample.returnedPoints;
	}

	const double queryCount = static_cast<double>(metrics.totalQueries);
	metrics.averageLatencyMs = metrics.totalLatencyMs / queryCount;
	metrics.medianLatencyMs = percentile(latencies, 0.50);
	metrics.p95LatencyMs = percentile(latencies, 0.95);
	metrics.throughputQueriesPerSecond = metrics.totalLatencyMs > 0.0
		? queryCount / (metrics.totalLatencyMs / 1000.0)
		: 0.0;
	metrics.averageVisitedNodes = static_cast<double>(metrics.totalVisitedNodes) / queryCount;
	metrics.averageTestedPoints = static_cast<double>(metrics.totalTestedPoints) / queryCount;
	metrics.averageReturnedPoints = static_cast<double>(metrics.totalReturnedPoints) / queryCount;
	return metrics;
}

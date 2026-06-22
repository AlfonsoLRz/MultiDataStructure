#include "../stdafx.h"
#include "Metrics.h"

static double boundsVolume(const AABB& bounds)
{
	const glm::vec3 size = bounds.size();
	if (size.x <= 0.0f || size.y <= 0.0f || size.z <= 0.0f)
		return 0.0;
	return static_cast<double>(size.x) * static_cast<double>(size.y) * static_cast<double>(size.z);
}

static const SchemaLevelConfig* schemaLevelForNode(const SchemaConfig* schema, const PointSpatialIndex::Node& node)
{
	if (!schema || schema->_levels.empty() || schema->totalLevels() == 0)
		return nullptr;

	const size_t schemaDepth = std::min(node._schemaDepth, schema->totalLevels() - 1);
	return &schema->levelForDepth(schemaDepth);
}

static size_t expectedFanoutForNode(const PointSpatialIndex::Node& node, const SchemaConfig* schema)
{
	if (const SchemaLevelConfig* level = schemaLevelForNode(schema, node))
	{
		const SchemaPrimitiveKind kind = level->_primitiveKind;
		switch (kind)
		{
		case SchemaPrimitiveKind::QuadTree:
			return 4;
		case SchemaPrimitiveKind::RegularGrid:
			return 27;
		case SchemaPrimitiveKind::HGrid:
			return 64;
		case SchemaPrimitiveKind::KDTree:
		case SchemaPrimitiveKind::BVH:
		case SchemaPrimitiveKind::BIH:
		case SchemaPrimitiveKind::LBVH:
			return 2;
		case SchemaPrimitiveKind::Octree:
		case SchemaPrimitiveKind::KarrasOctree:
		case SchemaPrimitiveKind::Mixed:
		default:
			return 8;
		}
	}

	switch (node._type)
	{
	case MultiDataStructure::DataStructureLevel::QuadTreeNode:
		return 4;
	case MultiDataStructure::DataStructureLevel::KDTreeNode:
	case MultiDataStructure::DataStructureLevel::BvhNode:
		return 2;
	case MultiDataStructure::DataStructureLevel::OctreeNode:
	default:
		return 8;
	}
}

struct TreeHealthAccumulator
{
	std::vector<size_t>			_occupancies;
	std::map<size_t, size_t>	_fanoutCounts;
	size_t						_totalDepth = 0;
	size_t						_internalNodes = 0;
	size_t						_totalExpectedChildren = 0;
	size_t						_totalPresentChildren = 0;
	size_t						_singleChildNodeCount = 0;
	double						_tightVolumeRatioSum = 0.0;
	size_t						_tightVolumeRatioSamples = 0;
	size_t						_microIndexedLeaves = 0;
	size_t						_microIndexedPoints = 0;
};

static void collectTreeHealth(const PointSpatialIndex::Node* node, const SchemaConfig* schema, TreeHealthAccumulator& accumulator)
{
	if (!node)
		return;

	accumulator._totalDepth += node->_depth;

	const double nodeVolume = boundsVolume(node->_bounds);
	if (nodeVolume > 0.0 && node->_subtreePointCount > 0)
	{
		const double tightVolume = boundsVolume(node->_tightBounds);
		accumulator._tightVolumeRatioSum += std::clamp(tightVolume / nodeVolume, 0.0, 1.0);
		++accumulator._tightVolumeRatioSamples;
	}

	if (node->isLeaf())
	{
		accumulator._occupancies.push_back(node->_pointCount);
		if (node->_microIndex)
		{
			++accumulator._microIndexedLeaves;
			accumulator._microIndexedPoints += node->_pointCount;
		}
		return;
	}

	const size_t fanout = node->_children.size();
	++accumulator._fanoutCounts[fanout];
	++accumulator._internalNodes;
	if (fanout == 1)
		++accumulator._singleChildNodeCount;

	const size_t expected = std::max(expectedFanoutForNode(*node, schema), fanout);
	accumulator._totalExpectedChildren += expected;
	accumulator._totalPresentChildren += fanout;

	for (const std::unique_ptr<PointSpatialIndex::Node>& child : node->_children)
		collectTreeHealth(child.get(), schema, accumulator);
}

static double percentile(std::vector<double> values, double p)
{
	if (values.empty())
		return 0.0;

	std::sort(values.begin(), values.end());
	const double clamped = std::clamp(p, 0.0, 1.0);
	const size_t index = static_cast<size_t>(std::ceil(clamped * static_cast<double>(values.size())) - 1.0);
	return values[std::min(index, values.size() - 1)];
}

static std::string formatFanoutSummary(const std::map<size_t, size_t>& fanoutCounts)
{
	std::ostringstream output;
	bool first = true;
	for (const auto& [fanout, count] : fanoutCounts)
	{
		if (!first)
			output << ';';
		first = false;
		output << fanout << ':' << count;
	}
	return output.str();
}

static Experiments::BuildMetrics collectBuildMetricsImpl(const PointSpatialIndex::Stats& stats, const PointSpatialIndex::Node* root, double buildTimeMs, const SchemaConfig* schema)
{
	Experiments::BuildMetrics metrics;
	metrics._buildTimeMs = buildTimeMs;
	metrics._numNodes = stats._numNodes;
	metrics._numLeaves = stats._numLeaves;
	metrics._indexedPoints = stats._numPoints;
	metrics._maxDepth = stats._maxDepth;

	TreeHealthAccumulator health;
	collectTreeHealth(root, schema, health);
	if (!health._occupancies.empty())
	{
		const size_t totalOccupancy = std::accumulate(health._occupancies.begin(), health._occupancies.end(), size_t(0));
		metrics._averageLeafOccupancy = static_cast<double>(totalOccupancy) / static_cast<double>(health._occupancies.size());
		metrics._maxLeafOccupancy = *std::max_element(health._occupancies.begin(), health._occupancies.end());

		std::vector<double> occupancyValues;
		occupancyValues.reserve(health._occupancies.size());
		for (const size_t occupancy : health._occupancies)
			occupancyValues.push_back(static_cast<double>(occupancy));
		metrics._leafOccupancyP50 = percentile(occupancyValues, 0.50);
		metrics._leafOccupancyP90 = percentile(occupancyValues, 0.90);
		metrics._leafOccupancyP99 = percentile(occupancyValues, 0.99);
	}

	metrics._averageDepth = stats._numNodes > 0
		? static_cast<double>(health._totalDepth) / static_cast<double>(stats._numNodes)
		: 0.0;
	metrics._averageFanout = health._internalNodes > 0
		? static_cast<double>(health._totalPresentChildren) / static_cast<double>(health._internalNodes)
		: 0.0;
	metrics._maxFanout = health._fanoutCounts.empty() ? 0 : health._fanoutCounts.rbegin()->first;
	metrics._emptyChildRatio = health._totalExpectedChildren > 0
		? static_cast<double>(health._totalExpectedChildren - health._totalPresentChildren) /
			static_cast<double>(health._totalExpectedChildren)
		: 0.0;
	metrics._singleChildNodeCount = health._singleChildNodeCount;
	metrics._meanTightBoundsVolumeRatio = health._tightVolumeRatioSamples > 0
		? health._tightVolumeRatioSum / static_cast<double>(health._tightVolumeRatioSamples)
		: 0.0;
	metrics._microIndexedLeaves = health._microIndexedLeaves;
	metrics._microIndexedPoints = health._microIndexedPoints;
	metrics._nodeFanoutSummary = formatFanoutSummary(health._fanoutCounts);
	metrics._memoryEstimateBytes =
		stats._numNodes * sizeof(PointSpatialIndex::Node) +
		stats._numPoints * sizeof(uint32_t);
	if (root)
	{
		metrics._memoryEstimateBytes += stats._numPoints * 3 * sizeof(float);
		metrics._memoryEstimateBytes += metrics._microIndexedPoints * 2 * sizeof(uint32_t);
		metrics._memoryEstimateBytes += metrics._microIndexedLeaves * 64 * sizeof(uint32_t);
	}
	return metrics;
}

Experiments::BuildMetrics Experiments::collectBuildMetrics(const PointSpatialIndex::Stats& stats, const PointSpatialIndex::Node* root, double buildTimeMs)
{
	return collectBuildMetricsImpl(stats, root, buildTimeMs, nullptr);
}

Experiments::BuildMetrics Experiments::collectBuildMetrics(const PointSpatialIndex::Stats& stats, const PointSpatialIndex::Node* root, double buildTimeMs, const SchemaConfig& schema)
{
	return collectBuildMetricsImpl(stats, root, buildTimeMs, &schema);
}

Experiments::QueryMetrics Experiments::summarizeQueryStats(const std::vector<PointSpatialIndex::QueryStats>& samples)
{
	QueryMetrics metrics;
	metrics._totalQueries = samples.size();
	if (samples.empty())
		return metrics;

	std::vector<double> latencies;
	latencies.reserve(samples.size());
	for (const PointSpatialIndex::QueryStats& sample : samples)
	{
		latencies.push_back(sample._elapsedMs);
		metrics._totalLatencyMs += sample._elapsedMs;
		metrics._totalVisitedNodes += sample._visitedNodes;
		metrics._totalTestedPoints += sample._testedPoints;
		metrics._totalReturnedPoints += sample._returnedPoints;
		metrics._totalFullyContainedNodes += sample._fullyContainedNodes;
	}

	const double queryCount = static_cast<double>(metrics._totalQueries);
	metrics._averageLatencyMs = metrics._totalLatencyMs / queryCount;
	metrics._medianLatencyMs = percentile(latencies, 0.50);
	metrics._p95LatencyMs = percentile(latencies, 0.95);
	metrics._throughputQueriesPerSecond = metrics._totalLatencyMs > 0.0
		? queryCount / (metrics._totalLatencyMs / 1000.0)
		: 0.0;
	metrics._averageVisitedNodes = static_cast<double>(metrics._totalVisitedNodes) / queryCount;
	metrics._averageTestedPoints = static_cast<double>(metrics._totalTestedPoints) / queryCount;
	metrics._averageReturnedPoints = static_cast<double>(metrics._totalReturnedPoints) / queryCount;
	metrics._averageFullyContainedNodes = static_cast<double>(metrics._totalFullyContainedNodes) / queryCount;
	// Single batch: reliability summary degenerates to the point estimate; multi-repeat callers overwrite via the schema-search profilers.
	metrics._measurementRepeats = 1;
	metrics._latencyMeanMs = metrics._averageLatencyMs;
	metrics._latencyStdDevMs = 0.0;
	metrics._latencyCoeffVar = 0.0;
	metrics._latencyCiLowMs = metrics._averageLatencyMs;
	metrics._latencyCiHighMs = metrics._averageLatencyMs;
	return metrics;
}

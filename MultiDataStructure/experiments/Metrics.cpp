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
	if (!schema || schema->levels.empty() || schema->totalLevels() == 0)
		return nullptr;

	const size_t schemaDepth = std::min(node.schemaDepth, schema->totalLevels() - 1);
	return &schema->levelForDepth(schemaDepth);
}

static size_t expectedFanoutForNode(const PointSpatialIndex::Node& node, const SchemaConfig* schema)
{
	if (const SchemaLevelConfig* level = schemaLevelForNode(schema, node))
	{
		const SchemaPrimitiveKind kind = level->primitiveKind;
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

	switch (node.type)
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
	std::vector<size_t> occupancies;
	std::map<size_t, size_t> fanoutCounts;
	size_t totalDepth = 0;
	size_t internalNodes = 0;
	size_t totalExpectedChildren = 0;
	size_t totalPresentChildren = 0;
	size_t singleChildNodeCount = 0;
	double tightVolumeRatioSum = 0.0;
	size_t tightVolumeRatioSamples = 0;
	size_t microIndexedLeaves = 0;
	size_t microIndexedPoints = 0;
};

static void collectTreeHealth(const PointSpatialIndex::Node* node, const SchemaConfig* schema, TreeHealthAccumulator& accumulator)
{
	if (!node)
		return;

	accumulator.totalDepth += node->depth;

	const double nodeVolume = boundsVolume(node->bounds);
	if (nodeVolume > 0.0 && node->subtreePointCount > 0)
	{
		const double tightVolume = boundsVolume(node->tightBounds);
		accumulator.tightVolumeRatioSum += std::clamp(tightVolume / nodeVolume, 0.0, 1.0);
		++accumulator.tightVolumeRatioSamples;
	}

	if (node->isLeaf())
	{
		accumulator.occupancies.push_back(node->pointCount);
		if (node->microIndex)
		{
			++accumulator.microIndexedLeaves;
			accumulator.microIndexedPoints += node->pointCount;
		}
		return;
	}

	const size_t fanout = node->children.size();
	++accumulator.fanoutCounts[fanout];
	++accumulator.internalNodes;
	if (fanout == 1)
		++accumulator.singleChildNodeCount;

	const size_t expected = std::max(expectedFanoutForNode(*node, schema), fanout);
	accumulator.totalExpectedChildren += expected;
	accumulator.totalPresentChildren += fanout;

	for (const std::unique_ptr<PointSpatialIndex::Node>& child : node->children)
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
	metrics.buildTimeMs = buildTimeMs;
	metrics.numNodes = stats.numNodes;
	metrics.numLeaves = stats.numLeaves;
	metrics.indexedPoints = stats.numPoints;
	metrics.maxDepth = stats.maxDepth;

	TreeHealthAccumulator health;
	collectTreeHealth(root, schema, health);
	if (!health.occupancies.empty())
	{
		const size_t totalOccupancy = std::accumulate(health.occupancies.begin(), health.occupancies.end(), size_t(0));
		metrics.averageLeafOccupancy = static_cast<double>(totalOccupancy) / static_cast<double>(health.occupancies.size());
		metrics.maxLeafOccupancy = *std::max_element(health.occupancies.begin(), health.occupancies.end());

		std::vector<double> occupancyValues;
		occupancyValues.reserve(health.occupancies.size());
		for (const size_t occupancy : health.occupancies)
			occupancyValues.push_back(static_cast<double>(occupancy));
		metrics.leafOccupancyP50 = percentile(occupancyValues, 0.50);
		metrics.leafOccupancyP90 = percentile(occupancyValues, 0.90);
		metrics.leafOccupancyP99 = percentile(occupancyValues, 0.99);
	}

	metrics.averageDepth = stats.numNodes > 0
		? static_cast<double>(health.totalDepth) / static_cast<double>(stats.numNodes)
		: 0.0;
	metrics.averageFanout = health.internalNodes > 0
		? static_cast<double>(health.totalPresentChildren) / static_cast<double>(health.internalNodes)
		: 0.0;
	metrics.maxFanout = health.fanoutCounts.empty() ? 0 : health.fanoutCounts.rbegin()->first;
	metrics.emptyChildRatio = health.totalExpectedChildren > 0
		? static_cast<double>(health.totalExpectedChildren - health.totalPresentChildren) /
			static_cast<double>(health.totalExpectedChildren)
		: 0.0;
	metrics.singleChildNodeCount = health.singleChildNodeCount;
	metrics.meanTightBoundsVolumeRatio = health.tightVolumeRatioSamples > 0
		? health.tightVolumeRatioSum / static_cast<double>(health.tightVolumeRatioSamples)
		: 0.0;
	metrics.microIndexedLeaves = health.microIndexedLeaves;
	metrics.microIndexedPoints = health.microIndexedPoints;
	metrics.nodeFanoutSummary = formatFanoutSummary(health.fanoutCounts);
	metrics.memoryEstimateBytes =
		stats.numNodes * sizeof(PointSpatialIndex::Node) +
		stats.numPoints * sizeof(uint32_t);
	if (root)
	{
		metrics.memoryEstimateBytes += stats.numPoints * 3 * sizeof(float);
		metrics.memoryEstimateBytes += metrics.microIndexedPoints * 2 * sizeof(uint32_t);
		metrics.memoryEstimateBytes += metrics.microIndexedLeaves * 64 * sizeof(uint32_t);
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
		metrics.totalFullyContainedNodes += sample.fullyContainedNodes;
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
	metrics.averageFullyContainedNodes = static_cast<double>(metrics.totalFullyContainedNodes) / queryCount;
	// Single batch: reliability summary degenerates to the point estimate; multi-repeat callers overwrite via the schema-search profilers.
	metrics.measurementRepeats = 1;
	metrics.latencyMeanMs = metrics.averageLatencyMs;
	metrics.latencyStdDevMs = 0.0;
	metrics.latencyCoeffVar = 0.0;
	metrics.latencyCiLowMs = metrics.averageLatencyMs;
	metrics.latencyCiHighMs = metrics.averageLatencyMs;
	return metrics;
}

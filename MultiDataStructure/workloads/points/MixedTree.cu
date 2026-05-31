#include "../../stdafx.h"
#include "MixedTree.h"

#include "../../CudaHelper.h"

namespace
{
	using PointGpu::DevicePoint;
	using PointGpu::DeviceQuery;
	using PointGpu::DeviceQuerySample;
	using PointGpu::LinearMixedTreeNode;

	constexpr int ThreadsPerBlock = 256;
	constexpr int QueryStackSize = 256;
	constexpr size_t MaxSupportedDepth = 12;
	constexpr uint32_t MaxChildCount = 27;

	enum SplitType : int
	{
		SplitQuadTree = MultiDataStructure::DataStructureLevel::QuadTreeNode,
		SplitKDTree = MultiDataStructure::DataStructureLevel::KDTreeNode,
		SplitOctree = MultiDataStructure::DataStructureLevel::OctreeNode,
		SplitBVH = MultiDataStructure::DataStructureLevel::BvhNode,
		SplitBIH = 100,
		SplitKarrasOctree = 101,
		SplitLBVH = 102,
		SplitRegularGrid = 103,
		SplitHGrid = 104,
	};

	enum ConditionFlag : uint32_t
	{
		HasMinPoints = 1u << 0,
		HasMaxPoints = 1u << 1,
		HasMinDensity = 1u << 2,
		HasMaxDensity = 1u << 3,
		HasMinHeightRatio = 1u << 4,
		HasMaxHeightRatio = 1u << 5,
		HasMinExtentX = 1u << 6,
		HasMaxExtentX = 1u << 7,
		HasMinExtentY = 1u << 8,
		HasMaxExtentY = 1u << 9,
		HasMinExtentZ = 1u << 10,
		HasMaxExtentZ = 1u << 11,
	};

	struct DeviceLevelCondition
	{
		uint32_t flags = 0;
		uint32_t minPoints = 0;
		uint32_t maxPoints = 0;
		float minDensity = 0.0f;
		float maxDensity = 0.0f;
		float minHeightRatio = 0.0f;
		float maxHeightRatio = 0.0f;
		float minExtentX = 0.0f;
		float maxExtentX = 0.0f;
		float minExtentY = 0.0f;
		float maxExtentY = 0.0f;
		float minExtentZ = 0.0f;
		float maxExtentZ = 0.0f;
	};

	struct HostActiveTypeAccumulator
	{
		size_t nodes = 0;
		size_t leafPoints = 0;
	};

	size_t divUp(size_t value, size_t divisor)
	{
		return (value + divisor - 1) / divisor;
	}

	size_t leafCapacityForSchema(const SchemaConfig& schema)
	{
		size_t leafCapacity = schema.buildPolicy.leafCapacity;
		if (!schema.levels.empty() && schema.levels.front().leafCapacity > 0)
			leafCapacity = schema.levels.front().leafCapacity;

		return std::max<size_t>(1, leafCapacity);
	}

	size_t minSplitForSchema(const SchemaConfig& schema)
	{
		size_t minSplit = schema.buildPolicy.minPrimitivesToSplit;
		if (!schema.levels.empty() && schema.levels.front().minPrimitivesToSplit > 0)
			minSplit = schema.levels.front().minPrimitivesToSplit;

		return std::max<size_t>(2, minSplit);
	}

	size_t maxDepthForSchema(const SchemaConfig& schema)
	{
		size_t maxDepth = schema.buildPolicy.maxDepth;
		if (maxDepth == 0)
			maxDepth = schema.totalLevels();
		if (maxDepth == 0)
			maxDepth = 8;
		if (maxDepth > MaxSupportedDepth)
			throw std::runtime_error("MixedTree currently supports maxDepth <= 12.");
		return maxDepth;
	}

	std::string normalizedTypeName(std::string value)
	{
		value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
			return std::isspace(c) || c == '_' || c == '-';
		}), value.end());
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return value;
	}

	int splitTypeForLevel(const SchemaLevelConfig& level)
	{
		const std::string typeName = normalizedTypeName(level.typeName);
		if (typeName == "bih" || typeName == "binaryintervalhierarchy" || typeName == "intervalhierarchy")
			return SplitBIH;
		if (typeName == "karrasoctree" || typeName == "mortonoctree" || typeName == "octreekarras" || typeName == "octreemorton")
			return SplitKarrasOctree;
		if (typeName == "lbvh" || typeName == "linearbvh")
			return SplitLBVH;
		if (typeName == "regulargrid" || typeName == "uniformgrid" || typeName == "grid" || typeName == "grid3d")
			return SplitRegularGrid;
		if (typeName == "hgrid" || typeName == "hierarchicalgrid" || typeName == "hierarchicalgrid3d")
			return SplitHGrid;
		return static_cast<int>(level.type);
	}

	std::string activeTypeNameForDepth(const SchemaConfig& schema, size_t depth)
	{
		if (schema.levels.empty())
			return "unknown";

		const SchemaLevelConfig& level = schema.levelForDepth(depth);
		return level.typeName.empty()
			? Config::dataStructureLevelName(level.type)
			: level.typeName;
	}

	void fillActiveStructureStats(
		PointGpu::BuildResult& result,
		const SchemaConfig& schema,
		const std::vector<LinearMixedTreeNode>& hostNodes)
	{
		if (schema.levels.empty() || hostNodes.empty())
			return;

		std::map<std::string, HostActiveTypeAccumulator> byType;
		size_t totalNodes = 0;
		size_t totalLeafPoints = 0;
		for (const LinearMixedTreeNode& node : hostNodes)
		{
			if (node.pointCount == 0)
				continue;

			const std::string typeName = activeTypeNameForDepth(schema, node.depth);
			HostActiveTypeAccumulator& accumulator = byType[typeName];
			++accumulator.nodes;
			++totalNodes;
			if (node.childBase < 0)
			{
				accumulator.leafPoints += node.pointCount;
				totalLeafPoints += node.pointCount;
			}
		}

		result.activeStructureTypes = byType.size();
		if (byType.empty())
			return;

		const std::string primaryType = activeTypeNameForDepth(schema, 0);
		size_t nonPrimaryNodes = 0;
		size_t nonPrimaryLeafPoints = 0;
		std::ostringstream summary;
		bool first = true;
		for (const auto& [typeName, accumulator] : byType)
		{
			if (!first)
				summary << ';';
			first = false;
			summary << typeName << ":nodes=" << accumulator.nodes << "|points=" << accumulator.leafPoints;

			if (typeName != primaryType)
			{
				nonPrimaryNodes += accumulator.nodes;
				nonPrimaryLeafPoints += accumulator.leafPoints;
			}
		}

		const double pointFraction = totalLeafPoints > 0
			? static_cast<double>(nonPrimaryLeafPoints) / static_cast<double>(totalLeafPoints)
			: 0.0;
		const double nodeFraction = totalNodes > 0
			? static_cast<double>(nonPrimaryNodes) / static_cast<double>(totalNodes)
			: 0.0;
		result.nestedActiveFraction = std::max(pointFraction, nodeFraction);
		result.activeStructureSummary = summary.str();
	}

	size_t hostChildSlotCountForSplitType(int splitType)
	{
		if (splitType == SplitRegularGrid || splitType == SplitHGrid)
			return 27;
		if (splitType == SplitOctree || splitType == SplitKarrasOctree)
			return 8;
		if (splitType == SplitQuadTree)
			return 4;
		return 2;
	}

	size_t maxChildSlotsForSchema(const std::vector<int>& splitTypes)
	{
		size_t result = 2;
		for (int splitType : splitTypes)
			result = std::max(result, hostChildSlotCountForSplitType(splitType));
		return std::min<size_t>(MaxChildCount, result);
	}

	std::vector<int> splitTypesForSchema(const SchemaConfig& schema, size_t maxDepth)
	{
		if (schema.levels.empty())
			throw std::runtime_error("MixedTree requires at least one schema level.");

		std::vector<int> splitTypes;
		splitTypes.reserve(maxDepth);
		for (size_t depth = 0; depth < maxDepth; ++depth)
		{
			const SchemaLevelConfig& level = schema.levelForDepth(depth);
			splitTypes.push_back(splitTypeForLevel(level));
		}
		return splitTypes;
	}

	DeviceLevelCondition makeDeviceCondition(const SchemaLevelCondition& condition)
	{
		DeviceLevelCondition result;
		if (condition.minPoints)
		{
			result.flags |= HasMinPoints;
			result.minPoints = static_cast<uint32_t>(std::min<size_t>(condition.minPoints.value(), std::numeric_limits<uint32_t>::max()));
		}
		if (condition.maxPoints)
		{
			result.flags |= HasMaxPoints;
			result.maxPoints = static_cast<uint32_t>(std::min<size_t>(condition.maxPoints.value(), std::numeric_limits<uint32_t>::max()));
		}
		if (condition.minDensity)
		{
			result.flags |= HasMinDensity;
			result.minDensity = static_cast<float>(condition.minDensity.value());
		}
		if (condition.maxDensity)
		{
			result.flags |= HasMaxDensity;
			result.maxDensity = static_cast<float>(condition.maxDensity.value());
		}
		if (condition.minHeightRatio)
		{
			result.flags |= HasMinHeightRatio;
			result.minHeightRatio = static_cast<float>(condition.minHeightRatio.value());
		}
		if (condition.maxHeightRatio)
		{
			result.flags |= HasMaxHeightRatio;
			result.maxHeightRatio = static_cast<float>(condition.maxHeightRatio.value());
		}
		if (condition.minExtentX)
		{
			result.flags |= HasMinExtentX;
			result.minExtentX = static_cast<float>(condition.minExtentX.value());
		}
		if (condition.maxExtentX)
		{
			result.flags |= HasMaxExtentX;
			result.maxExtentX = static_cast<float>(condition.maxExtentX.value());
		}
		if (condition.minExtentY)
		{
			result.flags |= HasMinExtentY;
			result.minExtentY = static_cast<float>(condition.minExtentY.value());
		}
		if (condition.maxExtentY)
		{
			result.flags |= HasMaxExtentY;
			result.maxExtentY = static_cast<float>(condition.maxExtentY.value());
		}
		if (condition.minExtentZ)
		{
			result.flags |= HasMinExtentZ;
			result.minExtentZ = static_cast<float>(condition.minExtentZ.value());
		}
		if (condition.maxExtentZ)
		{
			result.flags |= HasMaxExtentZ;
			result.maxExtentZ = static_cast<float>(condition.maxExtentZ.value());
		}
		return result;
	}

	std::vector<DeviceLevelCondition> conditionsForSchema(const SchemaConfig& schema, size_t maxDepth)
	{
		std::vector<DeviceLevelCondition> values;
		values.reserve(maxDepth);
		for (size_t depth = 0; depth < maxDepth; ++depth)
			values.push_back(makeDeviceCondition(schema.levelForDepth(depth).condition));
		return values;
	}

	std::vector<uint32_t> leafCapacitiesForSchema(const SchemaConfig& schema, size_t maxDepth)
	{
		std::vector<uint32_t> values;
		values.reserve(maxDepth);
		for (size_t depth = 0; depth < maxDepth; ++depth)
		{
			const SchemaLevelConfig& level = schema.levelForDepth(depth);
			const size_t leafCapacity = level.leafCapacity > 0 ? level.leafCapacity : schema.buildPolicy.leafCapacity;
			values.push_back(static_cast<uint32_t>(std::max<size_t>(1, leafCapacity)));
		}
		return values;
	}

	std::vector<uint32_t> minSplitsForSchema(const SchemaConfig& schema, size_t maxDepth)
	{
		std::vector<uint32_t> values;
		values.reserve(maxDepth);
		for (size_t depth = 0; depth < maxDepth; ++depth)
		{
			const SchemaLevelConfig& level = schema.levelForDepth(depth);
			const size_t minSplit = level.minPrimitivesToSplit > 0 ? level.minPrimitivesToSplit : schema.buildPolicy.minPrimitivesToSplit;
			values.push_back(static_cast<uint32_t>(std::max<size_t>(2, minSplit)));
		}
		return values;
	}

	size_t fullNodeCapacityForDepth(size_t maxDepth, size_t maxChildSlots)
	{
		size_t capacity = 1;
		size_t levelWidth = 1;
		for (size_t depth = 0; depth < maxDepth; ++depth)
		{
			if (levelWidth > std::numeric_limits<size_t>::max() / maxChildSlots)
				return std::numeric_limits<size_t>::max();
			levelWidth *= maxChildSlots;
			if (capacity > std::numeric_limits<size_t>::max() - levelWidth)
				return std::numeric_limits<size_t>::max();
			capacity += levelWidth;
		}
		return capacity;
	}

	size_t estimateNodeCapacity(size_t pointCount, size_t leafCapacity, size_t maxDepth, size_t maxChildSlots)
	{
		maxChildSlots = std::max<size_t>(2, std::min<size_t>(MaxChildCount, maxChildSlots));
		const size_t fullCapacity = fullNodeCapacityForDepth(maxDepth, maxChildSlots);
		const size_t targetLeaves = std::max<size_t>(1, divUp(pointCount, std::max<size_t>(1, leafCapacity)));
		const size_t leafBound = std::min(
			std::max<size_t>(1, pointCount),
			std::max(targetLeaves * 4, targetLeaves + maxChildSlots));
		const size_t estimatedCapacity = leafBound > (std::numeric_limits<size_t>::max() - 1) / maxChildSlots
			? std::numeric_limits<size_t>::max()
			: 1 + leafBound * maxChildSlots;
		const size_t capacity = std::min(fullCapacity, estimatedCapacity);
		if (capacity > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
			throw std::runtime_error("MixedTree currently supports up to 2^32 - 1 allocated nodes.");
		return std::max<size_t>(static_cast<size_t>(maxChildSlots + 1), capacity);
	}

	size_t estimateLevelScratchNodeCapacity(size_t pointCount, size_t leafCapacity, size_t nodeCapacity, size_t maxChildSlots)
	{
		maxChildSlots = std::max<size_t>(2, std::min<size_t>(MaxChildCount, maxChildSlots));
		const size_t targetLeaves = std::max<size_t>(1, divUp(pointCount, std::max<size_t>(1, leafCapacity)));
		const size_t estimatedFrontier = targetLeaves > std::numeric_limits<size_t>::max() / maxChildSlots
			? std::numeric_limits<size_t>::max()
			: targetLeaves * maxChildSlots;
		return std::max<size_t>(maxChildSlots + 1, std::min(nodeCapacity, std::min(pointCount, estimatedFrontier)));
	}

	DeviceQuery makeDeviceQuery(const PointGpu::Query& query)
	{
		DeviceQuery result{};
		result.type = static_cast<int>(query.type);
		const glm::vec3 min = query.bounds.min();
		const glm::vec3 max = query.bounds.max();
		result.minX = min.x;
		result.minY = min.y;
		result.minZ = min.z;
		result.maxX = max.x;
		result.maxY = max.y;
		result.maxZ = max.z;
		result.centerX = query.center.x;
		result.centerY = query.center.y;
		result.centerZ = query.center.z;
		result.radius = query.radius;
		result.knnK = static_cast<uint32_t>(std::min<size_t>(query.k, std::numeric_limits<uint32_t>::max()));
		return result;
	}

	Experiments::QueryMetrics summarizeGpuSamples(const std::vector<PointGpu::QuerySample>& samples)
	{
		std::vector<PointSpatialIndex::QueryStats> cpuSamples;
		cpuSamples.reserve(samples.size());
		for (const PointGpu::QuerySample& sample : samples)
		{
			PointSpatialIndex::QueryStats stats;
			stats.visitedNodes = sample.visitedNodes;
			stats.testedPoints = sample.testedPoints;
			stats.returnedPoints = sample.returnedPoints;
			stats.elapsedMs = sample.elapsedMs;
			cpuSamples.push_back(stats);
		}
		return Experiments::summarizeQueryStats(cpuSamples);
	}

	__device__ uint32_t activeGridDimension(float extent, uint32_t targetDimension)
	{
		return extent > 1.0e-7f ? targetDimension : 1u;
	}

	__device__ uint32_t regularGridDimensionForAxis(const LinearMixedTreeNode& node, int axis)
	{
		if (axis == 0)
			return activeGridDimension(node.maxX - node.minX, 3u);
		if (axis == 1)
			return activeGridDimension(node.maxY - node.minY, 3u);
		return activeGridDimension(node.maxZ - node.minZ, 3u);
	}

	__device__ uint32_t hgridDimensionForAxis(const LinearMixedTreeNode& node, int axis)
	{
		const uint32_t target = (node.depth & 1u) == 0 ? 2u : 3u;
		if (axis == 0)
			return activeGridDimension(node.maxX - node.minX, target);
		if (axis == 1)
			return activeGridDimension(node.maxY - node.minY, target);
		return activeGridDimension(node.maxZ - node.minZ, target);
	}

	__device__ void gridDimensionsForNode(const LinearMixedTreeNode& node, int splitType, uint32_t& dimX, uint32_t& dimY, uint32_t& dimZ)
	{
		if (splitType == SplitHGrid)
		{
			dimX = hgridDimensionForAxis(node, 0);
			dimY = hgridDimensionForAxis(node, 1);
			dimZ = hgridDimensionForAxis(node, 2);
			return;
		}

		dimX = regularGridDimensionForAxis(node, 0);
		dimY = regularGridDimensionForAxis(node, 1);
		dimZ = regularGridDimensionForAxis(node, 2);
	}

	__device__ uint32_t childCountForNode(const LinearMixedTreeNode& node, int splitType)
	{
		if (splitType == SplitOctree || splitType == SplitKarrasOctree)
			return 8;
		if (splitType == SplitQuadTree)
			return 4;
		if (splitType == SplitRegularGrid || splitType == SplitHGrid)
		{
			uint32_t dimX = 1;
			uint32_t dimY = 1;
			uint32_t dimZ = 1;
			gridDimensionsForNode(node, splitType, dimX, dimY, dimZ);
			return dimX * dimY * dimZ;
		}
		return 2;
	}

	__device__ bool matchesCondition(const LinearMixedTreeNode& node, const DeviceLevelCondition& condition)
	{
		if (condition.flags == 0)
			return true;

		if ((condition.flags & HasMinPoints) && node.pointCount < condition.minPoints)
			return false;
		if ((condition.flags & HasMaxPoints) && node.pointCount > condition.maxPoints)
			return false;

		const float extentX = fmaxf(node.maxX - node.minX, 0.0f);
		const float extentY = fmaxf(node.maxY - node.minY, 0.0f);
		const float extentZ = fmaxf(node.maxZ - node.minZ, 0.0f);
		const float horizontalExtent = fmaxf(fmaxf(extentX, extentY), 1.0e-9f);
		const float heightRatio = extentZ / horizontalExtent;
		if ((condition.flags & HasMinHeightRatio) && heightRatio < condition.minHeightRatio)
			return false;
		if ((condition.flags & HasMaxHeightRatio) && heightRatio > condition.maxHeightRatio)
			return false;

		const float volume = extentX * extentY * extentZ;
		const float density = volume > 1.0e-9f ? static_cast<float>(node.pointCount) / volume : 0.0f;
		if ((condition.flags & HasMinDensity) && density < condition.minDensity)
			return false;
		if ((condition.flags & HasMaxDensity) && density > condition.maxDensity)
			return false;

		if ((condition.flags & HasMinExtentX) && extentX < condition.minExtentX)
			return false;
		if ((condition.flags & HasMaxExtentX) && extentX > condition.maxExtentX)
			return false;
		if ((condition.flags & HasMinExtentY) && extentY < condition.minExtentY)
			return false;
		if ((condition.flags & HasMaxExtentY) && extentY > condition.maxExtentY)
			return false;
		if ((condition.flags & HasMinExtentZ) && extentZ < condition.minExtentZ)
			return false;
		if ((condition.flags & HasMaxExtentZ) && extentZ > condition.maxExtentZ)
			return false;

		return true;
	}

	__device__ int longestAxisForNode(const LinearMixedTreeNode& node)
	{
		const float extentX = node.maxX - node.minX;
		const float extentY = node.maxY - node.minY;
		const float extentZ = node.maxZ - node.minZ;
		if (extentX >= extentY && extentX >= extentZ)
			return 0;
		if (extentY >= extentZ)
			return 1;
		return 2;
	}

	__device__ float coordinateForAxis(const DevicePoint& point, int axis)
	{
		if (axis == 0)
			return point.x;
		if (axis == 1)
			return point.y;
		return point.z;
	}

	__device__ float midpointForAxis(const LinearMixedTreeNode& node, int axis)
	{
		if (axis == 0)
			return (node.minX + node.maxX) * 0.5f;
		if (axis == 1)
			return (node.minY + node.maxY) * 0.5f;
		return (node.minZ + node.maxZ) * 0.5f;
	}

	__device__ uint32_t clampGridCoordinate(float value, float minValue, float maxValue, uint32_t dimension)
	{
		if (dimension <= 1 || maxValue <= minValue)
			return 0;

		const float normalized = (value - minValue) / (maxValue - minValue);
		const int coordinate = static_cast<int>(floorf(normalized * static_cast<float>(dimension)));
		return static_cast<uint32_t>(min(max(coordinate, 0), static_cast<int>(dimension) - 1));
	}

	__device__ uint32_t childForPoint(const DevicePoint& point, const LinearMixedTreeNode& node, int splitType)
	{
		const float midX = (node.minX + node.maxX) * 0.5f;
		const float midY = (node.minY + node.maxY) * 0.5f;
		const float midZ = (node.minZ + node.maxZ) * 0.5f;

		if (splitType == SplitQuadTree)
		{
			return (point.x > midX ? 1u : 0u) |
				(point.y > midY ? 2u : 0u);
		}

		if (splitType == SplitKDTree || splitType == SplitBVH || splitType == SplitBIH || splitType == SplitLBVH)
		{
			const int axis = longestAxisForNode(node);
			return coordinateForAxis(point, axis) > midpointForAxis(node, axis) ? 1u : 0u;
		}

		if (splitType == SplitRegularGrid || splitType == SplitHGrid)
		{
			uint32_t dimX = 1;
			uint32_t dimY = 1;
			uint32_t dimZ = 1;
			gridDimensionsForNode(node, splitType, dimX, dimY, dimZ);
			const uint32_t x = clampGridCoordinate(point.x, node.minX, node.maxX, dimX);
			const uint32_t y = clampGridCoordinate(point.y, node.minY, node.maxY, dimY);
			const uint32_t z = clampGridCoordinate(point.z, node.minZ, node.maxZ, dimZ);
			return (z * dimY + y) * dimX + x;
		}

		return (point.x > midX ? 1u : 0u) |
			(point.y > midY ? 2u : 0u) |
			(point.z > midZ ? 4u : 0u);
	}

	__device__ LinearMixedTreeNode makeChildNode(
		const LinearMixedTreeNode& parent,
		uint32_t child,
		uint32_t offset,
		uint32_t count,
		int parentIndex,
		int splitType)
	{
		const float midX = (parent.minX + parent.maxX) * 0.5f;
		const float midY = (parent.minY + parent.maxY) * 0.5f;
		const float midZ = (parent.minZ + parent.maxZ) * 0.5f;

		LinearMixedTreeNode node{};
		if (splitType == SplitQuadTree)
		{
			node.minX = (child & 1u) ? midX : parent.minX;
			node.maxX = (child & 1u) ? parent.maxX : midX;
			node.minY = (child & 2u) ? midY : parent.minY;
			node.maxY = (child & 2u) ? parent.maxY : midY;
			node.minZ = parent.minZ;
			node.maxZ = parent.maxZ;
		}
		else if (splitType == SplitKDTree || splitType == SplitBVH || splitType == SplitBIH || splitType == SplitLBVH)
		{
			node.minX = parent.minX;
			node.minY = parent.minY;
			node.minZ = parent.minZ;
			node.maxX = parent.maxX;
			node.maxY = parent.maxY;
			node.maxZ = parent.maxZ;
			const int axis = longestAxisForNode(parent);
			const float plane = midpointForAxis(parent, axis);
			if (axis == 0)
			{
				if (child == 0)
					node.maxX = plane;
				else
					node.minX = plane;
			}
			else if (axis == 1)
			{
				if (child == 0)
					node.maxY = plane;
				else
					node.minY = plane;
			}
			else
			{
				if (child == 0)
					node.maxZ = plane;
				else
					node.minZ = plane;
			}
		}
		else if (splitType == SplitRegularGrid || splitType == SplitHGrid)
		{
			uint32_t dimX = 1;
			uint32_t dimY = 1;
			uint32_t dimZ = 1;
			gridDimensionsForNode(parent, splitType, dimX, dimY, dimZ);
			const uint32_t cellX = child % dimX;
			const uint32_t cellY = (child / dimX) % dimY;
			const uint32_t cellZ = child / (dimX * dimY);
			const float cellSizeX = (parent.maxX - parent.minX) / static_cast<float>(dimX);
			const float cellSizeY = (parent.maxY - parent.minY) / static_cast<float>(dimY);
			const float cellSizeZ = (parent.maxZ - parent.minZ) / static_cast<float>(dimZ);
			node.minX = parent.minX + cellSizeX * static_cast<float>(cellX);
			node.maxX = cellX + 1 == dimX ? parent.maxX : parent.minX + cellSizeX * static_cast<float>(cellX + 1);
			node.minY = parent.minY + cellSizeY * static_cast<float>(cellY);
			node.maxY = cellY + 1 == dimY ? parent.maxY : parent.minY + cellSizeY * static_cast<float>(cellY + 1);
			node.minZ = parent.minZ + cellSizeZ * static_cast<float>(cellZ);
			node.maxZ = cellZ + 1 == dimZ ? parent.maxZ : parent.minZ + cellSizeZ * static_cast<float>(cellZ + 1);
		}
		else
		{
			node.minX = (child & 1u) ? midX : parent.minX;
			node.maxX = (child & 1u) ? parent.maxX : midX;
			node.minY = (child & 2u) ? midY : parent.minY;
			node.maxY = (child & 2u) ? parent.maxY : midY;
			node.minZ = (child & 4u) ? midZ : parent.minZ;
			node.maxZ = (child & 4u) ? parent.maxZ : midZ;
		}
		node.parent = parentIndex;
		node.childBase = -1;
		node.childMask = 0;
		node.pointOffset = offset;
		node.pointCount = count;
		node.flags = count > 0 ? 1u : 0u;
		node.depth = parent.depth + 1;
		return node;
	}

	__device__ bool rangeIntersectsNode(const LinearMixedTreeNode& node, const DeviceQuery& query)
	{
		return node.minX <= query.maxX && node.maxX >= query.minX &&
			node.minY <= query.maxY && node.maxY >= query.minY &&
			node.minZ <= query.maxZ && node.maxZ >= query.minZ;
	}

	__device__ bool pointInsideRange(const DevicePoint& point, const DeviceQuery& query)
	{
		return point.x >= query.minX && point.x <= query.maxX &&
			point.y >= query.minY && point.y <= query.maxY &&
			point.z >= query.minZ && point.z <= query.maxZ;
	}

	__device__ float distanceSquaredToNode(const LinearMixedTreeNode& node, const DeviceQuery& query)
	{
		const float x = fminf(fmaxf(query.centerX, node.minX), node.maxX);
		const float y = fminf(fmaxf(query.centerY, node.minY), node.maxY);
		const float z = fminf(fmaxf(query.centerZ, node.minZ), node.maxZ);
		const float dx = query.centerX - x;
		const float dy = query.centerY - y;
		const float dz = query.centerZ - z;
		return dx * dx + dy * dy + dz * dz;
	}

	__device__ bool pointInsideRadius(const DevicePoint& point, const DeviceQuery& query)
	{
		const float dx = point.x - query.centerX;
		const float dy = point.y - query.centerY;
		const float dz = point.z - query.centerZ;
		return dx * dx + dy * dy + dz * dz <= query.radius * query.radius;
	}

	__global__ void initializeIndicesKernel(uint32_t* indices, size_t pointCount)
	{
		const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (index >= pointCount)
			return;

		indices[index] = static_cast<uint32_t>(index);
	}

	__global__ void initializeRootKernel(
		LinearMixedTreeNode* nodes,
		uint32_t* nodeCounter,
		size_t pointCount,
		float minX,
		float minY,
		float minZ,
		float maxX,
		float maxY,
		float maxZ)
	{
		LinearMixedTreeNode root{};
		root.minX = minX;
		root.minY = minY;
		root.minZ = minZ;
		root.maxX = maxX;
		root.maxY = maxY;
		root.maxZ = maxZ;
		root.parent = -1;
		root.childBase = -1;
		root.childMask = 0;
		root.pointOffset = 0;
		root.pointCount = static_cast<uint32_t>(pointCount);
		root.flags = 1;
		root.depth = 0;
		nodes[0] = root;
		*nodeCounter = 1;
	}

	__global__ void countChildBucketsKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		LinearMixedTreeNode* nodes,
		size_t levelStart,
		size_t levelCount,
		uint32_t maxDepth,
		const int* splitTypes,
		const uint32_t* leafCapacities,
		const uint32_t* minSplits,
		const DeviceLevelCondition* conditions,
		uint32_t childSlotStride,
		uint32_t* childCounts)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= levelCount)
			return;

		const size_t nodeIndex = levelStart + localNode;
		LinearMixedTreeNode node = nodes[nodeIndex];
		const uint32_t depth = node.depth < maxDepth ? node.depth : maxDepth - 1;
		const int splitType = splitTypes[depth];
		const uint32_t leafCapacity = leafCapacities[depth];
		const uint32_t minSplit = minSplits[depth];
		const uint32_t childSlots = childCountForNode(node, splitType);
		if (node.pointCount == 0 ||
			node.pointCount <= leafCapacity ||
			node.pointCount < minSplit ||
			node.depth >= maxDepth ||
			!matchesCondition(node, conditions[depth]))
		{
			if (threadIdx.x == 0)
			{
				nodes[nodeIndex].childBase = -1;
				nodes[nodeIndex].childMask = 0;
				nodes[nodeIndex].flags = node.pointCount > 0 ? 1u : 0u;
			}
			return;
		}

		uint32_t localCounts[MaxChildCount] = {};
		for (uint32_t offset = threadIdx.x; offset < node.pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node.pointOffset + offset];
			const uint32_t child = childForPoint(points[pointIndex], node, splitType);
			++localCounts[child];
		}

		for (uint32_t child = 0; child < childSlots; ++child)
		{
			if (localCounts[child] > 0)
				atomicAdd(&childCounts[localNode * childSlotStride + child], localCounts[child]);
		}
	}

	__global__ void prepareChildrenKernel(
		LinearMixedTreeNode* nodes,
		size_t levelStart,
		size_t levelCount,
		size_t nodeCapacity,
		uint32_t maxDepth,
		const int* splitTypes,
		const uint32_t* childCounts,
		uint32_t childSlotStride,
		uint32_t* writeCursors,
		uint32_t* nodeCounter,
		uint32_t* overflowFlag)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (localNode >= levelCount)
			return;

		const size_t nodeIndex = levelStart + localNode;
		LinearMixedTreeNode node = nodes[nodeIndex];
		if (node.pointCount == 0)
			return;

		const uint32_t depth = node.depth < maxDepth ? node.depth : maxDepth - 1;
		const int splitType = splitTypes[depth];
		const uint32_t childSlots = childCountForNode(node, splitType);
		uint32_t nonEmptyChildren = 0;
		uint32_t childMask = 0;
		for (uint32_t child = 0; child < childSlots; ++child)
		{
			if (childCounts[localNode * childSlotStride + child] > 0)
			{
				++nonEmptyChildren;
				childMask |= (1u << child);
			}
		}

		if (nonEmptyChildren <= 1)
		{
			nodes[nodeIndex].childBase = -1;
			nodes[nodeIndex].childMask = 0;
			nodes[nodeIndex].flags = 1;
			return;
		}

		const uint32_t childBase = atomicAdd(nodeCounter, childSlots);
		if (static_cast<size_t>(childBase) + childSlots > nodeCapacity)
		{
			*overflowFlag = 1;
			nodes[nodeIndex].childBase = -1;
			nodes[nodeIndex].childMask = 0;
			nodes[nodeIndex].flags = 1;
			return;
		}

		nodes[nodeIndex].childBase = static_cast<int>(childBase);
		nodes[nodeIndex].childMask = childMask;
		nodes[nodeIndex].flags = 0;

		uint32_t runningOffset = node.pointOffset;
		for (uint32_t child = 0; child < childSlots; ++child)
		{
			const uint32_t count = childCounts[localNode * childSlotStride + child];
			nodes[childBase + child] = makeChildNode(node, child, runningOffset, count, static_cast<int>(nodeIndex), splitType);
			writeCursors[localNode * childSlotStride + child] = runningOffset;
			runningOffset += count;
		}
	}

	__global__ void partitionIndicesKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		uint32_t* outputIndices,
		const LinearMixedTreeNode* nodes,
		size_t levelStart,
		size_t levelCount,
		uint32_t maxDepth,
		const int* splitTypes,
		uint32_t childSlotStride,
		uint32_t* writeCursors)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= levelCount)
			return;

		const size_t nodeIndex = levelStart + localNode;
		const LinearMixedTreeNode node = nodes[nodeIndex];
		if (node.pointCount == 0 || node.childBase < 0)
			return;

		const uint32_t depth = node.depth < maxDepth ? node.depth : maxDepth - 1;
		const int splitType = splitTypes[depth];
		for (uint32_t offset = threadIdx.x; offset < node.pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node.pointOffset + offset];
			const uint32_t child = childForPoint(points[pointIndex], node, splitType);
			const uint32_t writeOffset = atomicAdd(&writeCursors[localNode * childSlotStride + child], 1u);
			outputIndices[writeOffset] = pointIndex;
		}
	}

	__global__ void refitNodeBoundsKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		LinearMixedTreeNode* nodes,
		size_t nodeStart,
		size_t nodeCount)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= nodeCount)
			return;

		const size_t nodeIndex = nodeStart + localNode;
		const LinearMixedTreeNode node = nodes[nodeIndex];
		if (node.pointCount == 0)
			return;

		float minX = 3.402823466e+38F;
		float minY = 3.402823466e+38F;
		float minZ = 3.402823466e+38F;
		float maxX = -3.402823466e+38F;
		float maxY = -3.402823466e+38F;
		float maxZ = -3.402823466e+38F;

		for (uint32_t offset = threadIdx.x; offset < node.pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node.pointOffset + offset];
			const DevicePoint point = points[pointIndex];
			minX = fminf(minX, point.x);
			minY = fminf(minY, point.y);
			minZ = fminf(minZ, point.z);
			maxX = fmaxf(maxX, point.x);
			maxY = fmaxf(maxY, point.y);
			maxZ = fmaxf(maxZ, point.z);
		}

		__shared__ float sharedMinX[ThreadsPerBlock];
		__shared__ float sharedMinY[ThreadsPerBlock];
		__shared__ float sharedMinZ[ThreadsPerBlock];
		__shared__ float sharedMaxX[ThreadsPerBlock];
		__shared__ float sharedMaxY[ThreadsPerBlock];
		__shared__ float sharedMaxZ[ThreadsPerBlock];
		sharedMinX[threadIdx.x] = minX;
		sharedMinY[threadIdx.x] = minY;
		sharedMinZ[threadIdx.x] = minZ;
		sharedMaxX[threadIdx.x] = maxX;
		sharedMaxY[threadIdx.x] = maxY;
		sharedMaxZ[threadIdx.x] = maxZ;
		__syncthreads();

		for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
		{
			if (threadIdx.x < stride)
			{
				sharedMinX[threadIdx.x] = fminf(sharedMinX[threadIdx.x], sharedMinX[threadIdx.x + stride]);
				sharedMinY[threadIdx.x] = fminf(sharedMinY[threadIdx.x], sharedMinY[threadIdx.x + stride]);
				sharedMinZ[threadIdx.x] = fminf(sharedMinZ[threadIdx.x], sharedMinZ[threadIdx.x + stride]);
				sharedMaxX[threadIdx.x] = fmaxf(sharedMaxX[threadIdx.x], sharedMaxX[threadIdx.x + stride]);
				sharedMaxY[threadIdx.x] = fmaxf(sharedMaxY[threadIdx.x], sharedMaxY[threadIdx.x + stride]);
				sharedMaxZ[threadIdx.x] = fmaxf(sharedMaxZ[threadIdx.x], sharedMaxZ[threadIdx.x + stride]);
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			nodes[nodeIndex].minX = sharedMinX[0];
			nodes[nodeIndex].minY = sharedMinY[0];
			nodes[nodeIndex].minZ = sharedMinZ[0];
			nodes[nodeIndex].maxX = sharedMaxX[0];
			nodes[nodeIndex].maxY = sharedMaxY[0];
			nodes[nodeIndex].maxZ = sharedMaxZ[0];
		}
	}

	__global__ void queryKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		const LinearMixedTreeNode* nodes,
		size_t pointCount,
		const DeviceQuery* queries,
		size_t queryCount,
		float clockRateKHz,
		DeviceQuerySample* samples)
	{
		const size_t queryIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (queryIndex >= queryCount)
			return;

		const DeviceQuery query = queries[queryIndex];
		if (query.type == static_cast<int>(PointGpu::QueryType::Knn))
		{
			samples[queryIndex] = DeviceQuerySample{};
			return;
		}

		const unsigned long long begin = clock64();
		unsigned long long visited = 0;
		unsigned long long tested = 0;
		unsigned long long returned = 0;

		int stack[QueryStackSize];
		int stackSize = 0;
		stack[stackSize++] = 0;

		while (stackSize > 0)
		{
			const int nodeIndex = stack[--stackSize];
			const LinearMixedTreeNode node = nodes[nodeIndex];
			if (node.pointCount == 0)
				continue;

			bool intersects = false;
			if (query.type == static_cast<int>(PointGpu::QueryType::Radius))
				intersects = distanceSquaredToNode(node, query) <= query.radius * query.radius;
			else
				intersects = rangeIntersectsNode(node, query);

			if (!intersects)
				continue;

			++visited;
			if (node.childBase < 0)
			{
				for (uint32_t i = 0; i < node.pointCount; ++i)
				{
					++tested;
					const DevicePoint point = points[indices[node.pointOffset + i]];
					if (query.type == static_cast<int>(PointGpu::QueryType::Radius))
					{
						if (pointInsideRadius(point, query))
							++returned;
					}
					else if (pointInsideRange(point, query))
					{
						++returned;
					}
				}
				continue;
			}

			for (uint32_t child = 0; child < MaxChildCount; ++child)
			{
				if ((node.childMask & (1u << child)) == 0)
					continue;
				if (stackSize < QueryStackSize)
					stack[stackSize++] = node.childBase + static_cast<int>(child);
			}
		}

		const unsigned long long end = clock64();
		DeviceQuerySample sample{};
		sample.visitedNodes = visited;
		sample.testedPoints = tested;
		sample.returnedPoints = returned;
		sample.elapsedMs = clockRateKHz > 0.0f ? static_cast<float>(end - begin) / clockRateKHz : 0.0f;
		samples[queryIndex] = sample;
	}
}

struct PointGpu::MixedTree::DeviceState
{
	DevicePoint* points = nullptr;
	uint32_t* indices = nullptr;
	uint32_t* tempIndices = nullptr;
	LinearMixedTreeNode* nodes = nullptr;
	uint32_t* childCounts = nullptr;
	uint32_t* writeCursors = nullptr;
	int* splitTypes = nullptr;
	uint32_t* leafCapacities = nullptr;
	uint32_t* minSplits = nullptr;
	DeviceLevelCondition* conditions = nullptr;
	uint32_t* nodeCounter = nullptr;
	uint32_t* overflowFlag = nullptr;
	DeviceQuery* queryBuffer = nullptr;
	DeviceQuerySample* sampleBuffer = nullptr;
	size_t pointCount = 0;
	size_t nodeCapacity = 0;
	size_t allocatedNodes = 0;
	size_t actualNodes = 0;
	size_t actualLeaves = 0;
	size_t leafCapacity = 1;
	size_t minSplit = 2;
	size_t maxDepth = 0;
	size_t childSlotStride = 2;
	size_t levelScratchNodeCapacity = 0;
	size_t queryCapacity = 0;
	size_t baseMemoryBytes = 0;
	size_t memoryBytes = 0;
	int device = 0;
	const PointCloud* cloud = nullptr;
	bool pointsReady = false;
};

namespace
{
	void checkMemoryBudget(size_t bytes, size_t budgetMb)
	{
		if (budgetMb == 0)
			return;

		const size_t budgetBytes = budgetMb * 1024ull * 1024ull;
		if (bytes > budgetBytes)
		{
			std::ostringstream message;
			message << "MixedTree needs about " << (static_cast<double>(bytes) / (1024.0 * 1024.0))
				<< " MB, above --cuda-memory-budget-mb " << budgetMb;
			throw std::runtime_error(message.str());
		}
	}

	float deviceClockRateKHz(int device)
	{
		cudaDeviceProp properties{};
		CudaHelper::checkError(cudaGetDeviceProperties(&properties, device));
		return static_cast<float>(properties.clockRate);
	}

	template <typename State>
	void releaseQueryBuffers(State& state)
	{
		cudaFree(state.queryBuffer);
		cudaFree(state.sampleBuffer);
		state.queryBuffer = nullptr;
		state.sampleBuffer = nullptr;
		state.queryCapacity = 0;
	}

	template <typename State>
	void ensureQueryBuffers(State& state, size_t capacity)
	{
		if (state.queryCapacity >= capacity && state.queryBuffer && state.sampleBuffer)
			return;

		releaseQueryBuffers(state);
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&state.queryBuffer), sizeof(DeviceQuery) * capacity));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&state.sampleBuffer), sizeof(DeviceQuerySample) * capacity));
		state.queryCapacity = capacity;
	}
}

PointGpu::MixedTree::MixedTree()
	: _state(std::make_unique<DeviceState>())
{
}

PointGpu::MixedTree::~MixedTree()
{
	if (_state)
		release();
}

void PointGpu::MixedTree::release()
{
	releaseTree();
	cudaFree(_state->points);
	cudaFree(_state->indices);
	cudaFree(_state->tempIndices);
	releaseQueryBuffers(*_state);
	*_state = DeviceState();
}

void PointGpu::MixedTree::releaseTree()
{
	cudaFree(_state->nodes);
	cudaFree(_state->childCounts);
	cudaFree(_state->writeCursors);
	cudaFree(_state->splitTypes);
	cudaFree(_state->leafCapacities);
	cudaFree(_state->minSplits);
	cudaFree(_state->conditions);
	cudaFree(_state->nodeCounter);
	cudaFree(_state->overflowFlag);
	_state->nodes = nullptr;
	_state->childCounts = nullptr;
	_state->writeCursors = nullptr;
	_state->splitTypes = nullptr;
	_state->leafCapacities = nullptr;
	_state->minSplits = nullptr;
	_state->conditions = nullptr;
	_state->nodeCounter = nullptr;
	_state->overflowFlag = nullptr;
	_state->nodeCapacity = 0;
	_state->allocatedNodes = 0;
	_state->actualNodes = 0;
	_state->actualLeaves = 0;
	_state->childSlotStride = 2;
	_state->levelScratchNodeCapacity = 0;
	_state->memoryBytes = _state->baseMemoryBytes;
}

bool PointGpu::MixedTree::isAvailable(std::string* error)
{
	int count = 0;
	const cudaError_t result = cudaGetDeviceCount(&count);
	if (result != cudaSuccess)
	{
		if (error)
			*error = cudaGetErrorString(result);
		return false;
	}

	if (count <= 0)
	{
		if (error)
			*error = "No CUDA devices were found.";
		return false;
	}

	if (error)
		error->clear();
	return true;
}

int PointGpu::MixedTree::deviceCount()
{
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess)
		return 0;
	return count;
}

PointGpu::BuildResult PointGpu::MixedTree::build(const PointCloud& cloud, const SchemaConfig& schema, const Options& options)
{
	const std::string builder = options.builder.empty() ? "mixed" : options.builder;
	if (builder != "mixed" && builder != "hybrid")
		throw std::runtime_error("MixedTree evaluator supports --cuda-builder mixed or hybrid.");
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("MixedTree currently supports up to 2^32 - 1 points.");

	std::string availabilityError;
	if (!isAvailable(&availabilityError))
		throw std::runtime_error("CUDA point evaluator is unavailable: " + availabilityError);

	const int count = deviceCount();
	const int device = options.device >= 0 ? std::min(options.device, count - 1) : 0;
	CudaHelper::checkError(cudaSetDevice(device));

	const bool canReusePoints =
		_state->pointsReady &&
		_state->cloud == &cloud &&
		_state->pointCount == cloud.size() &&
		_state->device == device;

	if (!canReusePoints)
	{
		if (_state->points || _state->nodes)
		{
			CudaHelper::checkError(cudaSetDevice(_state->device));
			release();
			CudaHelper::checkError(cudaSetDevice(device));
		}
		else
		{
			release();
		}
	}
	else
	{
		releaseTree();
	}

	_state->pointCount = cloud.size();
	_state->leafCapacity = leafCapacityForSchema(schema);
	_state->minSplit = minSplitForSchema(schema);
	_state->maxDepth = maxDepthForSchema(schema);
	_state->device = device;
	_state->cloud = &cloud;

	BuildResult result;
	result.device = device;
	result.builder = "mixed";

	if (cloud.empty())
		return result;

	const std::vector<int> hostSplitTypes = splitTypesForSchema(schema, _state->maxDepth);
	const std::vector<uint32_t> hostLeafCapacities = leafCapacitiesForSchema(schema, _state->maxDepth);
	const std::vector<uint32_t> hostMinSplits = minSplitsForSchema(schema, _state->maxDepth);
	const std::vector<DeviceLevelCondition> hostConditions = conditionsForSchema(schema, _state->maxDepth);
	_state->childSlotStride = maxChildSlotsForSchema(hostSplitTypes);
	_state->leafCapacity = *std::min_element(hostLeafCapacities.begin(), hostLeafCapacities.end());
	const glm::vec3 boundsMin = cloud.bounds().min();
	const glm::vec3 boundsMax = cloud.bounds().max();

	if (!canReusePoints)
	{
		cudaEvent_t uploadBegin = nullptr;
		cudaEvent_t uploadEnd = nullptr;
		CudaHelper::startTimer(uploadBegin, uploadEnd);
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->points), sizeof(DevicePoint) * _state->pointCount));
		CudaHelper::checkError(cudaMemcpy(_state->points, cloud.points().data(), sizeof(DevicePoint) * _state->pointCount, cudaMemcpyHostToDevice));
		result.uploadTimeMs = CudaHelper::stopTimer(uploadBegin, uploadEnd);
		cudaEventDestroy(uploadBegin);
		cudaEventDestroy(uploadEnd);

		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->indices), sizeof(uint32_t) * _state->pointCount));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->tempIndices), sizeof(uint32_t) * _state->pointCount));
		_state->baseMemoryBytes =
			sizeof(DevicePoint) * _state->pointCount +
			sizeof(uint32_t) * _state->pointCount * 2;
		_state->pointsReady = true;
	}

	_state->nodeCapacity = estimateNodeCapacity(_state->pointCount, _state->leafCapacity, _state->maxDepth, _state->childSlotStride);
	_state->levelScratchNodeCapacity = estimateLevelScratchNodeCapacity(
		_state->pointCount,
		_state->leafCapacity,
		_state->nodeCapacity,
		_state->childSlotStride);
	_state->memoryBytes =
		_state->baseMemoryBytes +
		sizeof(LinearMixedTreeNode) * _state->nodeCapacity +
		sizeof(uint32_t) * _state->levelScratchNodeCapacity * _state->childSlotStride * 2 +
		sizeof(int) * _state->maxDepth +
		sizeof(uint32_t) * _state->maxDepth * 2 +
		sizeof(DeviceLevelCondition) * _state->maxDepth +
		sizeof(uint32_t) * 2;
	checkMemoryBudget(_state->memoryBytes, options.memoryBudgetMb);

	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->nodes), sizeof(LinearMixedTreeNode) * _state->nodeCapacity));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->childCounts), sizeof(uint32_t) * _state->levelScratchNodeCapacity * _state->childSlotStride));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->writeCursors), sizeof(uint32_t) * _state->levelScratchNodeCapacity * _state->childSlotStride));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->splitTypes), sizeof(int) * _state->maxDepth));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->leafCapacities), sizeof(uint32_t) * _state->maxDepth));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->minSplits), sizeof(uint32_t) * _state->maxDepth));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->conditions), sizeof(DeviceLevelCondition) * _state->maxDepth));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->nodeCounter), sizeof(uint32_t)));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->overflowFlag), sizeof(uint32_t)));
	CudaHelper::checkError(cudaMemcpy(_state->splitTypes, hostSplitTypes.data(), sizeof(int) * _state->maxDepth, cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaMemcpy(_state->leafCapacities, hostLeafCapacities.data(), sizeof(uint32_t) * _state->maxDepth, cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaMemcpy(_state->minSplits, hostMinSplits.data(), sizeof(uint32_t) * _state->maxDepth, cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaMemcpy(_state->conditions, hostConditions.data(), sizeof(DeviceLevelCondition) * _state->maxDepth, cudaMemcpyHostToDevice));

	cudaEvent_t buildBegin = nullptr;
	cudaEvent_t buildEnd = nullptr;
	CudaHelper::startTimer(buildBegin, buildEnd);

	const dim3 pointBlocks(static_cast<unsigned int>(divUp(_state->pointCount, ThreadsPerBlock)));
	initializeIndicesKernel<<<pointBlocks, ThreadsPerBlock>>>(_state->indices, _state->pointCount);
	CudaHelper::synchronize("initializeMixedTreeIndicesKernel");

	initializeRootKernel<<<1, 1>>>(
		_state->nodes,
		_state->nodeCounter,
		_state->pointCount,
		boundsMin.x,
		boundsMin.y,
		boundsMin.z,
		boundsMax.x,
		boundsMax.y,
		boundsMax.z);
	CudaHelper::synchronize("initializeMixedTreeRootKernel");
	CudaHelper::checkError(cudaMemset(_state->overflowFlag, 0, sizeof(uint32_t)));

	size_t levelStart = 0;
	size_t levelCount = 1;
	uint32_t currentNodeCount = 1;
	for (size_t depth = 0; depth < _state->maxDepth && levelCount > 0; ++depth)
	{
		if (levelCount > _state->levelScratchNodeCapacity)
			throw std::runtime_error("MixedTree exceeded its level scratch budget. Increase leaf capacity or reduce max depth.");

		CudaHelper::checkError(cudaMemset(
			_state->childCounts,
			0,
			sizeof(uint32_t) * levelCount * _state->childSlotStride));
		const dim3 prepareBlocks(static_cast<unsigned int>(divUp(levelCount, ThreadsPerBlock)));

		countChildBucketsKernel<<<static_cast<unsigned int>(levelCount), ThreadsPerBlock>>>(
			_state->points,
			_state->indices,
			_state->nodes,
			levelStart,
			levelCount,
			static_cast<uint32_t>(_state->maxDepth),
			_state->splitTypes,
			_state->leafCapacities,
			_state->minSplits,
			_state->conditions,
			static_cast<uint32_t>(_state->childSlotStride),
			_state->childCounts);
		CudaHelper::synchronize("countMixedTreeChildBucketsKernel");

		const uint32_t previousNodeCount = currentNodeCount;

		prepareChildrenKernel<<<prepareBlocks, ThreadsPerBlock>>>(
			_state->nodes,
			levelStart,
			levelCount,
			_state->nodeCapacity,
			static_cast<uint32_t>(_state->maxDepth),
			_state->splitTypes,
			_state->childCounts,
			static_cast<uint32_t>(_state->childSlotStride),
			_state->writeCursors,
			_state->nodeCounter,
			_state->overflowFlag);
		CudaHelper::synchronize("prepareMixedTreeChildrenKernel");

		uint32_t overflow = 0;
		CudaHelper::checkError(cudaMemcpy(&overflow, _state->overflowFlag, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		if (overflow != 0)
			throw std::runtime_error("MixedTree exceeded its allocated node budget. Increase leaf capacity or reduce max depth.");

		CudaHelper::checkError(cudaMemcpy(
			_state->tempIndices,
			_state->indices,
			sizeof(uint32_t) * _state->pointCount,
			cudaMemcpyDeviceToDevice));

		partitionIndicesKernel<<<static_cast<unsigned int>(levelCount), ThreadsPerBlock>>>(
			_state->points,
			_state->indices,
			_state->tempIndices,
			_state->nodes,
			levelStart,
			levelCount,
			static_cast<uint32_t>(_state->maxDepth),
			_state->splitTypes,
			static_cast<uint32_t>(_state->childSlotStride),
			_state->writeCursors);
		CudaHelper::synchronize("partitionMixedTreeIndicesKernel");
		std::swap(_state->indices, _state->tempIndices);

		uint32_t nextNodeCount = 0;
		CudaHelper::checkError(cudaMemcpy(&nextNodeCount, _state->nodeCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		currentNodeCount = nextNodeCount;
		const int levelSplitType = hostSplitTypes[std::min(depth, hostSplitTypes.size() - 1)];
		if ((levelSplitType == SplitBIH ||
			levelSplitType == SplitKarrasOctree ||
			levelSplitType == SplitLBVH ||
			levelSplitType == SplitRegularGrid ||
			levelSplitType == SplitHGrid) &&
			nextNodeCount > previousNodeCount)
		{
			refitNodeBoundsKernel<<<static_cast<unsigned int>(nextNodeCount - previousNodeCount), ThreadsPerBlock>>>(
				_state->points,
				_state->indices,
				_state->nodes,
				previousNodeCount,
				static_cast<size_t>(nextNodeCount - previousNodeCount));
			CudaHelper::synchronize("refitMixedTreeNodeBoundsKernel");
		}
		levelStart = previousNodeCount;
		levelCount = nextNodeCount > previousNodeCount
			? static_cast<size_t>(nextNodeCount - previousNodeCount)
			: 0;
		_state->allocatedNodes = nextNodeCount;
	}

	result.gpuBuildTimeMs = CudaHelper::stopTimer(buildBegin, buildEnd);
	cudaEventDestroy(buildBegin);
	cudaEventDestroy(buildEnd);

	if (_state->allocatedNodes == 0)
	{
		uint32_t allocated = 0;
		CudaHelper::checkError(cudaMemcpy(&allocated, _state->nodeCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		_state->allocatedNodes = allocated;
	}

	std::vector<LinearMixedTreeNode> hostNodes(_state->allocatedNodes);
	CudaHelper::checkError(cudaMemcpy(hostNodes.data(), _state->nodes, sizeof(LinearMixedTreeNode) * _state->allocatedNodes, cudaMemcpyDeviceToHost));

	size_t maxLeafOccupancy = 0;
	size_t indexedPoints = 0;
	size_t deepestNode = 0;
	for (const LinearMixedTreeNode& node : hostNodes)
	{
		if (node.pointCount == 0)
			continue;

		++_state->actualNodes;
		deepestNode = std::max<size_t>(deepestNode, node.depth);
		if (node.childBase < 0)
		{
			++_state->actualLeaves;
			indexedPoints += node.pointCount;
			maxLeafOccupancy = std::max<size_t>(maxLeafOccupancy, node.pointCount);
		}
	}

	result.gpuMemoryBytes = _state->memoryBytes;
	result.metrics.buildTimeMs = result.gpuBuildTimeMs;
	result.metrics.numNodes = _state->actualNodes;
	result.metrics.numLeaves = _state->actualLeaves;
	result.metrics.indexedPoints = indexedPoints;
	result.metrics.maxDepth = deepestNode;
	result.metrics.averageLeafOccupancy = _state->actualLeaves > 0
		? static_cast<double>(indexedPoints) / static_cast<double>(_state->actualLeaves)
		: 0.0;
	result.metrics.maxLeafOccupancy = maxLeafOccupancy;
	result.metrics.memoryEstimateBytes = _state->memoryBytes;
	fillActiveStructureStats(result, schema, hostNodes);
	return result;
}

PointGpu::QueryResult PointGpu::MixedTree::query(const std::vector<Query>& queries, const Options& options) const
{
	if (!_state || _state->actualNodes == 0 || queries.empty())
		return {};

	CudaHelper::checkError(cudaSetDevice(_state->device));

	QueryResult result;
	result.samples.reserve(queries.size());
	for (const Query& query : queries)
	{
		if (query.type == QueryType::Radius)
			++result.radiusQueries;
		else if (query.type == QueryType::CountRange)
			++result.countRangeQueries;
		else if (query.type == QueryType::Knn)
			++result.knnQueries;
		else
			++result.rangeQueries;
	}

	const size_t batchSize = options.queryBatchSize > 0
		? std::max<size_t>(1, options.queryBatchSize)
		: queries.size();
	const float clockRate = deviceClockRateKHz(_state->device);

	ensureQueryBuffers(*_state, batchSize);

	cudaEvent_t queryBegin = nullptr;
	cudaEvent_t queryEnd = nullptr;
	CudaHelper::startTimer(queryBegin, queryEnd);

	std::vector<DeviceQuery> hostQueries;
	std::vector<DeviceQuerySample> hostSamples;
	for (size_t offset = 0; offset < queries.size(); offset += batchSize)
	{
		const size_t currentBatch = std::min(batchSize, queries.size() - offset);
		hostQueries.clear();
		hostQueries.reserve(currentBatch);
		for (size_t i = 0; i < currentBatch; ++i)
			hostQueries.push_back(makeDeviceQuery(queries[offset + i]));
		const bool batchHasKnn = std::any_of(hostQueries.begin(), hostQueries.end(), [](const DeviceQuery& query) {
			return query.type == static_cast<int>(PointGpu::QueryType::Knn);
		});

		CudaHelper::checkError(cudaMemcpy(_state->queryBuffer, hostQueries.data(), sizeof(DeviceQuery) * currentBatch, cudaMemcpyHostToDevice));
		const dim3 queryBlocks(static_cast<unsigned int>(divUp(currentBatch, ThreadsPerBlock)));
		queryKernel<<<queryBlocks, ThreadsPerBlock>>>(
			_state->points,
			_state->indices,
			_state->nodes,
			_state->pointCount,
			_state->queryBuffer,
			currentBatch,
			clockRate,
			_state->sampleBuffer);
		CudaHelper::synchronize("MixedTreeQueryKernel");
		if (batchHasKnn)
		{
			PointGpu::bruteForceKnnKernel<<<static_cast<unsigned int>(currentBatch), ThreadsPerBlock, sizeof(float) * ThreadsPerBlock>>>(
				_state->points,
				_state->pointCount,
				_state->queryBuffer,
				currentBatch,
				clockRate,
				_state->sampleBuffer);
			CudaHelper::synchronize("MixedTreeKnnQueryKernel");
		}

		hostSamples.resize(currentBatch);
		CudaHelper::checkError(cudaMemcpy(hostSamples.data(), _state->sampleBuffer, sizeof(DeviceQuerySample) * currentBatch, cudaMemcpyDeviceToHost));
		for (const DeviceQuerySample& sample : hostSamples)
		{
			QuerySample converted;
			converted.visitedNodes = static_cast<size_t>(sample.visitedNodes);
			converted.testedPoints = static_cast<size_t>(sample.testedPoints);
			converted.returnedPoints = static_cast<size_t>(sample.returnedPoints);
			converted.elapsedMs = sample.elapsedMs;
			result.samples.push_back(converted);
		}
	}

	result.gpuQueryTimeMs = CudaHelper::stopTimer(queryBegin, queryEnd);
	cudaEventDestroy(queryBegin);
	cudaEventDestroy(queryEnd);

	result.metrics = summarizeGpuSamples(result.samples);
	return result;
}

bool PointGpu::MixedTree::built() const
{
	return _state && _state->actualNodes > 0;
}

size_t PointGpu::MixedTree::pointCount() const
{
	return _state ? _state->pointCount : 0;
}

size_t PointGpu::MixedTree::nodeCount() const
{
	return _state ? _state->actualNodes : 0;
}

size_t PointGpu::MixedTree::leafCount() const
{
	return _state ? _state->actualLeaves : 0;
}

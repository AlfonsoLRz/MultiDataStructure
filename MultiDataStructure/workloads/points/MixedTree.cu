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
		SplitQuadTreeXZ = 105,
		SplitQuadTreeYZ = 106,
		SplitQuadTreeIgnoreShortest = 107,
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
		HasMinAnisotropy = 1u << 12,
		HasMaxAnisotropy = 1u << 13,
	};

	struct DeviceLevelCondition
	{
		uint32_t	_flags = 0;
		uint32_t	_minPoints = 0;
		uint32_t	_maxPoints = 0;
		float		_minDensity = 0.0f;
		float		_maxDensity = 0.0f;
		float		_minHeightRatio = 0.0f;
		float		_maxHeightRatio = 0.0f;
		float		_minExtentX = 0.0f;
		float		_maxExtentX = 0.0f;
		float		_minExtentY = 0.0f;
		float		_maxExtentY = 0.0f;
		float		_minExtentZ = 0.0f;
		float		_maxExtentZ = 0.0f;
		float		_minAnisotropy = 0.0f;
		float		_maxAnisotropy = 0.0f;
	};

	struct HostActiveTypeAccumulator
	{
		size_t	_nodes = 0;
		size_t	_leafPoints = 0;
	};

	size_t divUp(size_t value, size_t divisor)
	{
		return (value + divisor - 1) / divisor;
	}

	size_t leafCapacityForSchema(const SchemaConfig& schema)
	{
		size_t leafCapacity = schema._buildPolicy._leafCapacity;
		if (!schema._levels.empty() && schema._levels.front()._leafCapacity > 0)
			leafCapacity = schema._levels.front()._leafCapacity;

		return std::max<size_t>(1, leafCapacity);
	}

	size_t minSplitForSchema(const SchemaConfig& schema)
	{
		size_t minSplit = schema._buildPolicy._minPrimitivesToSplit;
		if (!schema._levels.empty() && schema._levels.front()._minPrimitivesToSplit > 0)
			minSplit = schema._levels.front()._minPrimitivesToSplit;

		return std::max<size_t>(2, minSplit);
	}

	size_t maxDepthForSchema(const SchemaConfig& schema)
	{
		size_t maxDepth = schema._buildPolicy._maxDepth;
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
		const std::string typeName = normalizedTypeName(level._typeName);
		if (typeName == "quadtree" || typeName == "quadtreenode")
		{
			const std::string policy = normalizedTypeName(level._axisPolicy.empty() ? std::string("xy") : level._axisPolicy);
			if (policy == "xz" || policy == "ignorey" || policy == "y")
				return SplitQuadTreeXZ;
			if (policy == "yz" || policy == "ignorex" || policy == "x")
				return SplitQuadTreeYZ;
			if (policy == "ignoreshortest" || policy == "shortest")
				return SplitQuadTreeIgnoreShortest;
			return SplitQuadTree;
		}
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
		return static_cast<int>(level._type);
	}

	bool hasUnsupportedMixedCondition(const SchemaLevelCondition& condition)
	{
		return condition._minOccupancyEntropy.has_value() ||
			condition._maxOccupancyEntropy.has_value();
	}

	void validateMixedTreeConditions(const SchemaConfig& schema)
	{
		for (const SchemaLevelConfig& level : schema._levels)
		{
			if (level._adaptiveLeafCapacity._enabled)
				throw std::runtime_error(
					"CUDA MixedTree does not currently support adaptiveLeafCapacity; "
					"remove the adaptive leaf-capacity rule or use the CPU evaluator.");
			if (hasUnsupportedMixedCondition(level._condition))
				throw std::runtime_error(
					"CUDA MixedTree does not currently support occupancy-entropy conditions; "
					"remove minOccupancyEntropy/maxOccupancyEntropy or use the CPU evaluator.");
		}
	}

	std::string activeTypeNameForDepth(const SchemaConfig& schema, size_t depth)
	{
		if (schema._levels.empty())
			return "unknown";

		const SchemaLevelConfig& level = schema.levelForDepth(depth);
		return level._typeName.empty()
			? Config::dataStructureLevelName(level._type)
			: level._typeName;
	}

	void fillActiveStructureStats(
		PointGpu::BuildResult& result,
		const SchemaConfig& schema,
		const std::vector<LinearMixedTreeNode>& hostNodes)
	{
		if (schema._levels.empty() || hostNodes.empty())
			return;

		std::map<std::string, HostActiveTypeAccumulator> byType;
		size_t totalNodes = 0;
		size_t totalLeafPoints = 0;
		for (const LinearMixedTreeNode& node : hostNodes)
		{
			if (node._pointCount == 0)
				continue;

			const std::string typeName = activeTypeNameForDepth(schema, node._schemaDepth);
			HostActiveTypeAccumulator& accumulator = byType[typeName];
			++accumulator._nodes;
			++totalNodes;
			if (node._childBase < 0)
			{
				accumulator._leafPoints += node._pointCount;
				totalLeafPoints += node._pointCount;
			}
		}

		result._activeStructureTypes = byType.size();
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
			summary << typeName << ":nodes=" << accumulator._nodes << "|points=" << accumulator._leafPoints;

			if (typeName != primaryType)
			{
				nonPrimaryNodes += accumulator._nodes;
				nonPrimaryLeafPoints += accumulator._leafPoints;
			}
		}

		const double pointFraction = totalLeafPoints > 0
			? static_cast<double>(nonPrimaryLeafPoints) / static_cast<double>(totalLeafPoints)
			: 0.0;
		const double nodeFraction = totalNodes > 0
			? static_cast<double>(nonPrimaryNodes) / static_cast<double>(totalNodes)
			: 0.0;
		result._nestedActiveFraction = std::max(pointFraction, nodeFraction);
		result._activeStructureSummary = summary.str();
	}

	size_t hostChildSlotCountForSplitType(int splitType)
	{
		if (splitType == SplitRegularGrid || splitType == SplitHGrid)
			return 27;
		if (splitType == SplitOctree || splitType == SplitKarrasOctree)
			return 8;
		if (splitType == SplitQuadTree ||
			splitType == SplitQuadTreeXZ ||
			splitType == SplitQuadTreeYZ ||
			splitType == SplitQuadTreeIgnoreShortest)
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
		if (schema._levels.empty())
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
		if (condition._minPoints)
		{
			result._flags |= HasMinPoints;
			result._minPoints = static_cast<uint32_t>(std::min<size_t>(condition._minPoints.value(), std::numeric_limits<uint32_t>::max()));
		}
		if (condition._maxPoints)
		{
			result._flags |= HasMaxPoints;
			result._maxPoints = static_cast<uint32_t>(std::min<size_t>(condition._maxPoints.value(), std::numeric_limits<uint32_t>::max()));
		}
		if (condition._minDensity)
		{
			result._flags |= HasMinDensity;
			result._minDensity = static_cast<float>(condition._minDensity.value());
		}
		if (condition._maxDensity)
		{
			result._flags |= HasMaxDensity;
			result._maxDensity = static_cast<float>(condition._maxDensity.value());
		}
		if (condition._minHeightRatio)
		{
			result._flags |= HasMinHeightRatio;
			result._minHeightRatio = static_cast<float>(condition._minHeightRatio.value());
		}
		if (condition._maxHeightRatio)
		{
			result._flags |= HasMaxHeightRatio;
			result._maxHeightRatio = static_cast<float>(condition._maxHeightRatio.value());
		}
		if (condition._minExtentX)
		{
			result._flags |= HasMinExtentX;
			result._minExtentX = static_cast<float>(condition._minExtentX.value());
		}
		if (condition._maxExtentX)
		{
			result._flags |= HasMaxExtentX;
			result._maxExtentX = static_cast<float>(condition._maxExtentX.value());
		}
		if (condition._minExtentY)
		{
			result._flags |= HasMinExtentY;
			result._minExtentY = static_cast<float>(condition._minExtentY.value());
		}
		if (condition._maxExtentY)
		{
			result._flags |= HasMaxExtentY;
			result._maxExtentY = static_cast<float>(condition._maxExtentY.value());
		}
		if (condition._minExtentZ)
		{
			result._flags |= HasMinExtentZ;
			result._minExtentZ = static_cast<float>(condition._minExtentZ.value());
		}
		if (condition._maxExtentZ)
		{
			result._flags |= HasMaxExtentZ;
			result._maxExtentZ = static_cast<float>(condition._maxExtentZ.value());
		}
		if (condition._minAnisotropy)
		{
			result._flags |= HasMinAnisotropy;
			result._minAnisotropy = static_cast<float>(condition._minAnisotropy.value());
		}
		if (condition._maxAnisotropy)
		{
			result._flags |= HasMaxAnisotropy;
			result._maxAnisotropy = static_cast<float>(condition._maxAnisotropy.value());
		}
		return result;
	}

	std::vector<DeviceLevelCondition> conditionsForSchema(const SchemaConfig& schema, size_t maxDepth)
	{
		std::vector<DeviceLevelCondition> values;
		values.reserve(maxDepth);
		for (size_t depth = 0; depth < maxDepth; ++depth)
			values.push_back(makeDeviceCondition(schema.levelForDepth(depth)._condition));
		return values;
	}

	std::vector<uint32_t> schemaBlockEndDepthsForSchema(const SchemaConfig& schema, size_t maxDepth)
	{
		const size_t schemaDepthLimit = std::min(maxDepth, std::max<size_t>(1, schema.totalLevels()));
		std::vector<uint32_t> values;
		values.reserve(maxDepth);
		for (size_t depth = 0; depth < maxDepth; ++depth)
		{
			size_t cumulative = 0;
			for (const SchemaLevelConfig& level : schema._levels)
			{
				cumulative += level._numLevels;
				if (depth < cumulative)
					break;
			}
			if (cumulative == 0)
				cumulative = schemaDepthLimit;
			values.push_back(static_cast<uint32_t>(std::min(cumulative, schemaDepthLimit)));
		}
		return values;
	}

	std::vector<uint32_t> leafCapacitiesForSchema(const SchemaConfig& schema, size_t maxDepth)
	{
		std::vector<uint32_t> values;
		values.reserve(maxDepth);
		for (size_t depth = 0; depth < maxDepth; ++depth)
		{
			const SchemaLevelConfig& level = schema.levelForDepth(depth);
			const size_t leafCapacity = level._leafCapacity > 0 ? level._leafCapacity : schema._buildPolicy._leafCapacity;
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
			const size_t minSplit = level._minPrimitivesToSplit > 0 ? level._minPrimitivesToSplit : schema._buildPolicy._minPrimitivesToSplit;
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
		result._type = static_cast<int>(query._type);
		const glm::vec3 min = query._bounds.min();
		const glm::vec3 max = query._bounds.max();
		result._minX = min.x;
		result._minY = min.y;
		result._minZ = min.z;
		result._maxX = max.x;
		result._maxY = max.y;
		result._maxZ = max.z;
		result._centerX = query.center.x;
		result._centerY = query.center.y;
		result._centerZ = query.center.z;
		result._radius = query._radius;
		result._knnK = static_cast<uint32_t>(std::min<size_t>(query._k, std::numeric_limits<uint32_t>::max()));
		return result;
	}

	Experiments::QueryMetrics summarizeGpuSamples(const std::vector<PointGpu::QuerySample>& samples)
	{
		std::vector<PointSpatialIndex::QueryStats> cpuSamples;
		cpuSamples.reserve(samples.size());
		for (const PointGpu::QuerySample& sample : samples)
		{
			PointSpatialIndex::QueryStats stats;
			stats._visitedNodes = sample._visitedNodes;
			stats._testedPoints = sample._testedPoints;
			stats._returnedPoints = sample._returnedPoints;
			stats._elapsedMs = sample._elapsedMs;
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
			return activeGridDimension(node._maxX - node._minX, 3u);
		if (axis == 1)
			return activeGridDimension(node._maxY - node._minY, 3u);
		return activeGridDimension(node._maxZ - node._minZ, 3u);
	}

	__device__ uint32_t hgridDimensionForAxis(const LinearMixedTreeNode& node, int axis)
	{
		const uint32_t target = (node._depth & 1u) == 0 ? 2u : 3u;
		if (axis == 0)
			return activeGridDimension(node._maxX - node._minX, target);
		if (axis == 1)
			return activeGridDimension(node._maxY - node._minY, target);
		return activeGridDimension(node._maxZ - node._minZ, target);
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
		if (splitType == SplitQuadTree ||
			splitType == SplitQuadTreeXZ ||
			splitType == SplitQuadTreeYZ ||
			splitType == SplitQuadTreeIgnoreShortest)
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

	__device__ int ignoredAxisForQuadTree(const LinearMixedTreeNode& node, int splitType)
	{
		if (splitType == SplitQuadTreeYZ)
			return 0;
		if (splitType == SplitQuadTreeXZ)
			return 1;
		if (splitType == SplitQuadTreeIgnoreShortest)
		{
			const float extentX = node._maxX - node._minX;
			const float extentY = node._maxY - node._minY;
			const float extentZ = node._maxZ - node._minZ;
			if (extentX <= extentY && extentX <= extentZ)
				return 0;
			if (extentY <= extentZ)
				return 1;
		}
		return 2;
	}

	__device__ void quadTreeActiveAxes(const LinearMixedTreeNode& node, int splitType, int& firstAxis, int& secondAxis)
	{
		const int ignoredAxis = ignoredAxisForQuadTree(node, splitType);
		firstAxis = ignoredAxis == 0 ? 1 : 0;
		secondAxis = ignoredAxis == 2 ? 1 : 2;
	}

	__device__ bool matchesCondition(const LinearMixedTreeNode& node, const DeviceLevelCondition& condition)
	{
		if (condition._flags == 0)
			return true;

		if ((condition._flags & HasMinPoints) && node._pointCount < condition._minPoints)
			return false;
		if ((condition._flags & HasMaxPoints) && node._pointCount > condition._maxPoints)
			return false;

		const float extentX = fmaxf(node._maxX - node._minX, 0.0f);
		const float extentY = fmaxf(node._maxY - node._minY, 0.0f);
		const float extentZ = fmaxf(node._maxZ - node._minZ, 0.0f);
		const float horizontalExtent = fmaxf(fmaxf(extentX, extentY), 1.0e-9f);
		const float heightRatio = extentZ / horizontalExtent;
		if ((condition._flags & HasMinHeightRatio) && heightRatio < condition._minHeightRatio)
			return false;
		if ((condition._flags & HasMaxHeightRatio) && heightRatio > condition._maxHeightRatio)
			return false;

		const float volume = extentX * extentY * extentZ;
		const float density = volume > 1.0e-9f ? static_cast<float>(node._pointCount) / volume : 0.0f;
		if ((condition._flags & HasMinDensity) && density < condition._minDensity)
			return false;
		if ((condition._flags & HasMaxDensity) && density > condition._maxDensity)
			return false;

		if ((condition._flags & HasMinExtentX) && extentX < condition._minExtentX)
			return false;
		if ((condition._flags & HasMaxExtentX) && extentX > condition._maxExtentX)
			return false;
		if ((condition._flags & HasMinExtentY) && extentY < condition._minExtentY)
			return false;
		if ((condition._flags & HasMaxExtentY) && extentY > condition._maxExtentY)
			return false;
		if ((condition._flags & HasMinExtentZ) && extentZ < condition._minExtentZ)
			return false;
		if ((condition._flags & HasMaxExtentZ) && extentZ > condition._maxExtentZ)
			return false;

		if (condition._flags & (HasMinAnisotropy | HasMaxAnisotropy))
		{
			const float longExtent = fmaxf(fmaxf(extentX, extentY), extentZ);
			const float shortExtent = fminf(fminf(extentX, extentY), extentZ);
			if (longExtent > 1.0e-9f)
			{
				const float anisotropy = 1.0f - (shortExtent / longExtent);
				if ((condition._flags & HasMinAnisotropy) && anisotropy < condition._minAnisotropy)
					return false;
				if ((condition._flags & HasMaxAnisotropy) && anisotropy > condition._maxAnisotropy)
					return false;
			}
		}

		return true;
	}

	__device__ uint32_t activeSchemaDepthForNode(
		const LinearMixedTreeNode& node,
		uint32_t schemaDepthLimit,
		const DeviceLevelCondition* conditions,
		const uint32_t* schemaBlockEndDepths)
	{
		uint32_t schemaDepth = node._schemaDepth;
		while (schemaDepth < schemaDepthLimit)
		{
			if (matchesCondition(node, conditions[schemaDepth]))
				return schemaDepth;

			const uint32_t nextDepth = schemaBlockEndDepths[schemaDepth];
			schemaDepth = nextDepth > schemaDepth ? nextDepth : schemaDepth + 1;
		}

		return schemaDepthLimit;
	}

	__device__ int longestAxisForNode(const LinearMixedTreeNode& node)
	{
		const float extentX = node._maxX - node._minX;
		const float extentY = node._maxY - node._minY;
		const float extentZ = node._maxZ - node._minZ;
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
			return (node._minX + node._maxX) * 0.5f;
		if (axis == 1)
			return (node._minY + node._maxY) * 0.5f;
		return (node._minZ + node._maxZ) * 0.5f;
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
		const float midX = (node._minX + node._maxX) * 0.5f;
		const float midY = (node._minY + node._maxY) * 0.5f;
		const float midZ = (node._minZ + node._maxZ) * 0.5f;

		if (splitType == SplitQuadTree ||
			splitType == SplitQuadTreeXZ ||
			splitType == SplitQuadTreeYZ ||
			splitType == SplitQuadTreeIgnoreShortest)
		{
			int firstAxis = 0;
			int secondAxis = 1;
			quadTreeActiveAxes(node, splitType, firstAxis, secondAxis);
			return (coordinateForAxis(point, firstAxis) > midpointForAxis(node, firstAxis) ? 1u : 0u) |
				(coordinateForAxis(point, secondAxis) > midpointForAxis(node, secondAxis) ? 2u : 0u);
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
			const uint32_t x = clampGridCoordinate(point.x, node._minX, node._maxX, dimX);
			const uint32_t y = clampGridCoordinate(point.y, node._minY, node._maxY, dimY);
			const uint32_t z = clampGridCoordinate(point.z, node._minZ, node._maxZ, dimZ);
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
		int splitType,
		uint32_t childSchemaDepth)
	{
		const float midX = (parent._minX + parent._maxX) * 0.5f;
		const float midY = (parent._minY + parent._maxY) * 0.5f;
		const float midZ = (parent._minZ + parent._maxZ) * 0.5f;

		LinearMixedTreeNode node{};
		if (splitType == SplitQuadTree ||
			splitType == SplitQuadTreeXZ ||
			splitType == SplitQuadTreeYZ ||
			splitType == SplitQuadTreeIgnoreShortest)
		{
			node._minX = parent._minX;
			node._maxX = parent._maxX;
			node._minY = parent._minY;
			node._maxY = parent._maxY;
			node._minZ = parent._minZ;
			node._maxZ = parent._maxZ;
			int firstAxis = 0;
			int secondAxis = 1;
			quadTreeActiveAxes(parent, splitType, firstAxis, secondAxis);
			const float firstPlane = midpointForAxis(parent, firstAxis);
			const float secondPlane = midpointForAxis(parent, secondAxis);
			if (firstAxis == 0)
			{
				if (child & 1u) node._minX = firstPlane; else node._maxX = firstPlane;
			}
			else if (firstAxis == 1)
			{
				if (child & 1u) node._minY = firstPlane; else node._maxY = firstPlane;
			}
			else
			{
				if (child & 1u) node._minZ = firstPlane; else node._maxZ = firstPlane;
			}
			if (secondAxis == 0)
			{
				if (child & 2u) node._minX = secondPlane; else node._maxX = secondPlane;
			}
			else if (secondAxis == 1)
			{
				if (child & 2u) node._minY = secondPlane; else node._maxY = secondPlane;
			}
			else
			{
				if (child & 2u) node._minZ = secondPlane; else node._maxZ = secondPlane;
			}
		}
		else if (splitType == SplitKDTree || splitType == SplitBVH || splitType == SplitBIH || splitType == SplitLBVH)
		{
			node._minX = parent._minX;
			node._minY = parent._minY;
			node._minZ = parent._minZ;
			node._maxX = parent._maxX;
			node._maxY = parent._maxY;
			node._maxZ = parent._maxZ;
			const int axis = longestAxisForNode(parent);
			const float plane = midpointForAxis(parent, axis);
			if (axis == 0)
			{
				if (child == 0)
					node._maxX = plane;
				else
					node._minX = plane;
			}
			else if (axis == 1)
			{
				if (child == 0)
					node._maxY = plane;
				else
					node._minY = plane;
			}
			else
			{
				if (child == 0)
					node._maxZ = plane;
				else
					node._minZ = plane;
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
			const float cellSizeX = (parent._maxX - parent._minX) / static_cast<float>(dimX);
			const float cellSizeY = (parent._maxY - parent._minY) / static_cast<float>(dimY);
			const float cellSizeZ = (parent._maxZ - parent._minZ) / static_cast<float>(dimZ);
			node._minX = parent._minX + cellSizeX * static_cast<float>(cellX);
			node._maxX = cellX + 1 == dimX ? parent._maxX : parent._minX + cellSizeX * static_cast<float>(cellX + 1);
			node._minY = parent._minY + cellSizeY * static_cast<float>(cellY);
			node._maxY = cellY + 1 == dimY ? parent._maxY : parent._minY + cellSizeY * static_cast<float>(cellY + 1);
			node._minZ = parent._minZ + cellSizeZ * static_cast<float>(cellZ);
			node._maxZ = cellZ + 1 == dimZ ? parent._maxZ : parent._minZ + cellSizeZ * static_cast<float>(cellZ + 1);
		}
		else
		{
			node._minX = (child & 1u) ? midX : parent._minX;
			node._maxX = (child & 1u) ? parent._maxX : midX;
			node._minY = (child & 2u) ? midY : parent._minY;
			node._maxY = (child & 2u) ? parent._maxY : midY;
			node._minZ = (child & 4u) ? midZ : parent._minZ;
			node._maxZ = (child & 4u) ? parent._maxZ : midZ;
		}
		node._parent = parentIndex;
		node._childBase = -1;
		node._childMask = 0;
		node._pointOffset = offset;
		node._pointCount = count;
		node._flags = count > 0 ? 1u : 0u;
		node._depth = parent._depth + 1;
		node._schemaDepth = childSchemaDepth;
		return node;
	}

	__device__ bool rangeIntersectsNode(const LinearMixedTreeNode& node, const DeviceQuery& query)
	{
		return node._minX <= query._maxX && node._maxX >= query._minX &&
			node._minY <= query._maxY && node._maxY >= query._minY &&
			node._minZ <= query._maxZ && node._maxZ >= query._minZ;
	}

	__device__ bool pointInsideRange(const DevicePoint& point, const DeviceQuery& query)
	{
		return point.x >= query._minX && point.x <= query._maxX &&
			point.y >= query._minY && point.y <= query._maxY &&
			point.z >= query._minZ && point.z <= query._maxZ;
	}

	__device__ float distanceSquaredToNode(const LinearMixedTreeNode& node, const DeviceQuery& query)
	{
		const float x = fminf(fmaxf(query._centerX, node._minX), node._maxX);
		const float y = fminf(fmaxf(query._centerY, node._minY), node._maxY);
		const float z = fminf(fmaxf(query._centerZ, node._minZ), node._maxZ);
		const float dx = query._centerX - x;
		const float dy = query._centerY - y;
		const float dz = query._centerZ - z;
		return dx * dx + dy * dy + dz * dz;
	}

	__device__ bool pointInsideRadius(const DevicePoint& point, const DeviceQuery& query)
	{
		const float dx = point.x - query._centerX;
		const float dy = point.y - query._centerY;
		const float dz = point.z - query._centerZ;
		return dx * dx + dy * dy + dz * dz <= query._radius * query._radius;
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
		root._minX = minX;
		root._minY = minY;
		root._minZ = minZ;
		root._maxX = maxX;
		root._maxY = maxY;
		root._maxZ = maxZ;
		root._parent = -1;
		root._childBase = -1;
		root._childMask = 0;
		root._pointOffset = 0;
		root._pointCount = static_cast<uint32_t>(pointCount);
		root._flags = 1;
		root._depth = 0;
		root._schemaDepth = 0;
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
		const uint32_t* schemaBlockEndDepths,
		uint32_t schemaDepthLimit,
		uint32_t childSlotStride,
		uint32_t* childCounts)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= levelCount)
			return;

		const size_t nodeIndex = levelStart + localNode;
		LinearMixedTreeNode node = nodes[nodeIndex];
		const uint32_t schemaDepth = activeSchemaDepthForNode(node, schemaDepthLimit, conditions, schemaBlockEndDepths);
		const bool hasActiveSchemaLevel = schemaDepth < schemaDepthLimit;
		const int splitType = hasActiveSchemaLevel ? splitTypes[schemaDepth] : SplitOctree;
		const uint32_t leafCapacity = hasActiveSchemaLevel ? leafCapacities[schemaDepth] : UINT_MAX;
		const uint32_t minSplit = hasActiveSchemaLevel ? minSplits[schemaDepth] : UINT_MAX;
		const uint32_t childSlots = childCountForNode(node, splitType);
		if (node._pointCount == 0 ||
			node._pointCount <= leafCapacity ||
			node._pointCount < minSplit ||
			node._depth >= maxDepth ||
			!hasActiveSchemaLevel)
		{
			if (threadIdx.x == 0)
			{
				nodes[nodeIndex]._childBase = -1;
				nodes[nodeIndex]._childMask = 0;
				nodes[nodeIndex]._flags = node._pointCount > 0 ? 1u : 0u;
			}
			return;
		}

		if (threadIdx.x == 0)
			nodes[nodeIndex]._schemaDepth = schemaDepth;

		uint32_t localCounts[MaxChildCount] = {};
		for (uint32_t offset = threadIdx.x; offset < node._pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node._pointOffset + offset];
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
		uint32_t schemaDepthLimit,
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
		if (node._pointCount == 0)
			return;

		const uint32_t schemaDepth = node._schemaDepth < schemaDepthLimit ? node._schemaDepth : schemaDepthLimit - 1;
		const int splitType = splitTypes[schemaDepth];
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
			nodes[nodeIndex]._childBase = -1;
			nodes[nodeIndex]._childMask = 0;
			nodes[nodeIndex]._flags = 1;
			return;
		}

		const uint32_t childBase = atomicAdd(nodeCounter, childSlots);
		if (static_cast<size_t>(childBase) + childSlots > nodeCapacity)
		{
			*overflowFlag = 1;
			nodes[nodeIndex]._childBase = -1;
			nodes[nodeIndex]._childMask = 0;
			nodes[nodeIndex]._flags = 1;
			return;
		}

		nodes[nodeIndex]._childBase = static_cast<int>(childBase);
		nodes[nodeIndex]._childMask = childMask;
		nodes[nodeIndex]._flags = 0;

		uint32_t runningOffset = node._pointOffset;
		for (uint32_t child = 0; child < childSlots; ++child)
		{
			const uint32_t count = childCounts[localNode * childSlotStride + child];
			nodes[childBase + child] = makeChildNode(
				node,
				child,
				runningOffset,
				count,
				static_cast<int>(nodeIndex),
				splitType,
				node._schemaDepth + 1u < schemaDepthLimit ? node._schemaDepth + 1u : schemaDepthLimit);
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
		uint32_t schemaDepthLimit,
		uint32_t childSlotStride,
		uint32_t* writeCursors)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= levelCount)
			return;

		const size_t nodeIndex = levelStart + localNode;
		const LinearMixedTreeNode node = nodes[nodeIndex];
		if (node._pointCount == 0 || node._childBase < 0)
			return;

		const uint32_t schemaDepth = node._schemaDepth < schemaDepthLimit ? node._schemaDepth : schemaDepthLimit - 1;
		const int splitType = splitTypes[schemaDepth];
		for (uint32_t offset = threadIdx.x; offset < node._pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node._pointOffset + offset];
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
		size_t nodeCount,
		const int* splitTypes,
		uint32_t schemaDepthLimit)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= nodeCount)
			return;

		const size_t nodeIndex = nodeStart + localNode;
		const LinearMixedTreeNode node = nodes[nodeIndex];
		if (node._pointCount == 0)
			return;

		const uint32_t parentSchemaDepth = node._schemaDepth > 0 ? node._schemaDepth - 1u : 0u;
		if (parentSchemaDepth >= schemaDepthLimit)
			return;

		const int parentSplitType = splitTypes[parentSchemaDepth];
		if (parentSplitType != SplitBIH &&
			parentSplitType != SplitKarrasOctree &&
			parentSplitType != SplitLBVH &&
			parentSplitType != SplitRegularGrid &&
			parentSplitType != SplitHGrid)
			return;

		float minX = 3.402823466e+38F;
		float minY = 3.402823466e+38F;
		float minZ = 3.402823466e+38F;
		float maxX = -3.402823466e+38F;
		float maxY = -3.402823466e+38F;
		float maxZ = -3.402823466e+38F;

		for (uint32_t offset = threadIdx.x; offset < node._pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node._pointOffset + offset];
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
			nodes[nodeIndex]._minX = sharedMinX[0];
			nodes[nodeIndex]._minY = sharedMinY[0];
			nodes[nodeIndex]._minZ = sharedMinZ[0];
			nodes[nodeIndex]._maxX = sharedMaxX[0];
			nodes[nodeIndex]._maxY = sharedMaxY[0];
			nodes[nodeIndex]._maxZ = sharedMaxZ[0];
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
		if (query._type == static_cast<int>(PointGpu::QueryType::Knn))
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
			if (node._pointCount == 0)
				continue;

			bool intersects = false;
			if (query._type == static_cast<int>(PointGpu::QueryType::Radius))
				intersects = distanceSquaredToNode(node, query) <= query._radius * query._radius;
			else
				intersects = rangeIntersectsNode(node, query);

			if (!intersects)
				continue;

			++visited;
			if (node._childBase < 0)
			{
				for (uint32_t i = 0; i < node._pointCount; ++i)
				{
					++tested;
					const DevicePoint point = points[indices[node._pointOffset + i]];
					if (query._type == static_cast<int>(PointGpu::QueryType::Radius))
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
				if ((node._childMask & (1u << child)) == 0)
					continue;
				if (stackSize < QueryStackSize)
					stack[stackSize++] = node._childBase + static_cast<int>(child);
			}
		}

		const unsigned long long end = clock64();
		DeviceQuerySample sample{};
		sample._visitedNodes = visited;
		sample._testedPoints = tested;
		sample._returnedPoints = returned;
		sample._elapsedMs = clockRateKHz > 0.0f ? static_cast<float>(end - begin) / clockRateKHz : 0.0f;
		samples[queryIndex] = sample;
	}

	__global__ void cooperativeQueryKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		const LinearMixedTreeNode* nodes,
		const DeviceQuery* queries,
		size_t queryCount,
		float clockRateKHz,
		DeviceQuerySample* samples)
	{
		const size_t queryIndex = static_cast<size_t>(blockIdx.x);
		if (queryIndex >= queryCount)
			return;

		const DeviceQuery query = queries[queryIndex];
		if (query._type == static_cast<int>(PointGpu::QueryType::Knn))
		{
			if (threadIdx.x == 0)
				samples[queryIndex] = DeviceQuerySample{};
			return;
		}

		const unsigned long long begin = clock64();
		__shared__ int stack[QueryStackSize];
		__shared__ int stackSize;
		__shared__ unsigned long long visited;
		__shared__ unsigned long long tested;
		__shared__ unsigned long long returned;
		__shared__ unsigned long long sharedCounts[ThreadsPerBlock];

		if (threadIdx.x == 0)
		{
			stackSize = 0;
			visited = 0;
			tested = 0;
			returned = 0;
			stack[stackSize++] = 0;
		}
		__syncthreads();

		while (true)
		{
			__shared__ int nodeIndex;
			if (threadIdx.x == 0)
				nodeIndex = stackSize > 0 ? stack[--stackSize] : -1;
			__syncthreads();

			if (nodeIndex < 0)
				break;

			const LinearMixedTreeNode node = nodes[nodeIndex];
			bool intersects = node._pointCount > 0;
			if (intersects)
			{
				if (query._type == static_cast<int>(PointGpu::QueryType::Radius))
					intersects = distanceSquaredToNode(node, query) <= query._radius * query._radius;
				else
					intersects = rangeIntersectsNode(node, query);
			}

			if (intersects && threadIdx.x == 0)
				++visited;
			__syncthreads();

			if (!intersects)
				continue;

			if (node._childBase < 0)
			{
				unsigned long long localReturned = 0;
				for (uint32_t i = threadIdx.x; i < node._pointCount; i += blockDim.x)
				{
					const DevicePoint point = points[indices[node._pointOffset + i]];
					if (query._type == static_cast<int>(PointGpu::QueryType::Radius))
					{
						if (pointInsideRadius(point, query))
							++localReturned;
					}
					else if (pointInsideRange(point, query))
					{
						++localReturned;
					}
				}

				sharedCounts[threadIdx.x] = localReturned;
				__syncthreads();
				for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
				{
					if (threadIdx.x < stride)
						sharedCounts[threadIdx.x] += sharedCounts[threadIdx.x + stride];
					__syncthreads();
				}

				if (threadIdx.x == 0)
				{
					tested += node._pointCount;
					returned += sharedCounts[0];
				}
				__syncthreads();
				continue;
			}

			if (threadIdx.x == 0)
			{
				for (uint32_t child = 0; child < MaxChildCount; ++child)
				{
					if ((node._childMask & (1u << child)) == 0)
						continue;
					if (stackSize < QueryStackSize)
						stack[stackSize++] = node._childBase + static_cast<int>(child);
				}
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			const unsigned long long end = clock64();
			DeviceQuerySample sample{};
			sample._visitedNodes = visited;
			sample._testedPoints = tested;
			sample._returnedPoints = returned;
			sample._elapsedMs = clockRateKHz > 0.0f ? static_cast<float>(end - begin) / clockRateKHz : 0.0f;
			samples[queryIndex] = sample;
		}
	}
}

struct PointGpu::MixedTree::DeviceState
{
	DevicePoint*			_points = nullptr;
	uint32_t*				_indices = nullptr;
	uint32_t*				_tempIndices = nullptr;
	LinearMixedTreeNode*	_nodes = nullptr;
	uint32_t*				_childCounts = nullptr;
	uint32_t*				_writeCursors = nullptr;
	int*					_splitTypes = nullptr;
	uint32_t*				_leafCapacities = nullptr;
	uint32_t*				_minSplits = nullptr;
	DeviceLevelCondition*	_conditions = nullptr;
	uint32_t*				_schemaBlockEndDepths = nullptr;
	uint32_t*				_nodeCounter = nullptr;
	uint32_t*				_overflowFlag = nullptr;
	DeviceQuery*			_queryBuffer = nullptr;
	DeviceQuerySample*		_sampleBuffer = nullptr;
	size_t					_pointCount = 0;
	size_t					_nodeCapacity = 0;
	size_t					_allocatedNodes = 0;
	size_t					_actualNodes = 0;
	size_t					_actualLeaves = 0;
	size_t					_leafCapacity = 1;
	size_t					_minSplit = 2;
	size_t					_maxDepth = 0;
	size_t					_schemaDepthLimit = 0;
	size_t					_childSlotStride = 2;
	size_t					_levelScratchNodeCapacity = 0;
	size_t					_queryCapacity = 0;
	size_t					_baseMemoryBytes = 0;
	size_t					_memoryBytes = 0;
	int						_device = 0;
	const PointCloud*		_cloud = nullptr;
	bool					_pointsReady = false;
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
		cudaFree(state._queryBuffer);
		cudaFree(state._sampleBuffer);
		state._queryBuffer = nullptr;
		state._sampleBuffer = nullptr;
		state._queryCapacity = 0;
	}

	template <typename State>
	void ensureQueryBuffers(State& state, size_t capacity)
	{
		if (state._queryCapacity >= capacity && state._queryBuffer && state._sampleBuffer)
			return;

		releaseQueryBuffers(state);
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&state._queryBuffer), sizeof(DeviceQuery) * capacity));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&state._sampleBuffer), sizeof(DeviceQuerySample) * capacity));
		state._queryCapacity = capacity;
	}

	double safeVolume(const glm::vec3& extent)
	{
		return static_cast<double>(std::max(extent.x, 0.0f)) *
			static_cast<double>(std::max(extent.y, 0.0f)) *
			static_cast<double>(std::max(extent.z, 0.0f));
	}

	bool shouldUseCooperativeQueryKernel(const std::vector<DeviceQuery>& queries, const PointCloud* cloud)
	{
		if (!cloud || cloud->empty())
			return false;

		const double cloudVolume = std::max(safeVolume(cloud->bounds().size()), 1.0e-12);
		for (const DeviceQuery& query : queries)
		{
			if (query._type == static_cast<int>(PointGpu::QueryType::Knn))
				continue;

			if (query._type == static_cast<int>(PointGpu::QueryType::Radius))
			{
				const double radius = std::max(0.0f, query._radius);
				const double radiusVolume = (4.0 / 3.0) * 3.14159265358979323846 * radius * radius * radius;
				if (radiusVolume / cloudVolume >= 0.005)
					return true;
				continue;
			}

			const glm::vec3 queryExtent(
				std::max(0.0f, query._maxX - query._minX),
				std::max(0.0f, query._maxY - query._minY),
				std::max(0.0f, query._maxZ - query._minZ));
			if (safeVolume(queryExtent) / cloudVolume >= 0.005)
				return true;
		}

		return false;
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
	cudaFree(_state->_points);
	cudaFree(_state->_indices);
	cudaFree(_state->_tempIndices);
	releaseQueryBuffers(*_state);
	*_state = DeviceState();
}

void PointGpu::MixedTree::releaseTree()
{
	cudaFree(_state->_nodes);
	cudaFree(_state->_childCounts);
	cudaFree(_state->_writeCursors);
	cudaFree(_state->_splitTypes);
	cudaFree(_state->_leafCapacities);
	cudaFree(_state->_minSplits);
	cudaFree(_state->_conditions);
	cudaFree(_state->_schemaBlockEndDepths);
	cudaFree(_state->_nodeCounter);
	cudaFree(_state->_overflowFlag);
	_state->_nodes = nullptr;
	_state->_childCounts = nullptr;
	_state->_writeCursors = nullptr;
	_state->_splitTypes = nullptr;
	_state->_leafCapacities = nullptr;
	_state->_minSplits = nullptr;
	_state->_conditions = nullptr;
	_state->_schemaBlockEndDepths = nullptr;
	_state->_nodeCounter = nullptr;
	_state->_overflowFlag = nullptr;
	_state->_nodeCapacity = 0;
	_state->_allocatedNodes = 0;
	_state->_actualNodes = 0;
	_state->_actualLeaves = 0;
	_state->_schemaDepthLimit = 0;
	_state->_childSlotStride = 2;
	_state->_levelScratchNodeCapacity = 0;
	_state->_memoryBytes = _state->_baseMemoryBytes;
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
	const std::string builder = options._builder.empty() ? "mixed" : options._builder;
	if (builder != "mixed" && builder != "hybrid")
		throw std::runtime_error("MixedTree evaluator supports --cuda-builder mixed or hybrid.");
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("MixedTree currently supports up to 2^32 - 1 points.");
	validateMixedTreeConditions(schema);

	std::string availabilityError;
	if (!isAvailable(&availabilityError))
		throw std::runtime_error("CUDA point evaluator is unavailable: " + availabilityError);

	const int count = deviceCount();
	const int device = options._device >= 0 ? std::min(options._device, count - 1) : 0;
	CudaHelper::checkError(cudaSetDevice(device));

	const bool canReusePoints =
		_state->_pointsReady &&
		_state->_cloud == &cloud &&
		_state->_pointCount == cloud.size() &&
		_state->_device == device;

	if (!canReusePoints)
	{
		if (_state->_points || _state->_nodes)
		{
			CudaHelper::checkError(cudaSetDevice(_state->_device));
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

	_state->_pointCount = cloud.size();
	_state->_leafCapacity = leafCapacityForSchema(schema);
	_state->_minSplit = minSplitForSchema(schema);
	_state->_maxDepth = maxDepthForSchema(schema);
	_state->_schemaDepthLimit = std::min(_state->_maxDepth, std::max<size_t>(1, schema.totalLevels()));
	_state->_device = device;
	_state->_cloud = &cloud;

	BuildResult result;
	result._device = device;
	result._builder = "mixed";

	if (cloud.empty())
		return result;

	const std::vector<int> hostSplitTypes = splitTypesForSchema(schema, _state->_maxDepth);
	const std::vector<uint32_t> hostLeafCapacities = leafCapacitiesForSchema(schema, _state->_maxDepth);
	const std::vector<uint32_t> hostMinSplits = minSplitsForSchema(schema, _state->_maxDepth);
	const std::vector<DeviceLevelCondition> hostConditions = conditionsForSchema(schema, _state->_maxDepth);
	const std::vector<uint32_t> hostSchemaBlockEndDepths = schemaBlockEndDepthsForSchema(schema, _state->_maxDepth);
	_state->_childSlotStride = maxChildSlotsForSchema(hostSplitTypes);
	_state->_leafCapacity = *std::min_element(hostLeafCapacities.begin(), hostLeafCapacities.end());
	const glm::vec3 boundsMin = cloud.bounds().min();
	const glm::vec3 boundsMax = cloud.bounds().max();

	if (!canReusePoints)
	{
		cudaEvent_t uploadBegin = nullptr;
		cudaEvent_t uploadEnd = nullptr;
		CudaHelper::startTimer(uploadBegin, uploadEnd);
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_points), sizeof(DevicePoint) * _state->_pointCount));
		CudaHelper::checkError(cudaMemcpy(_state->_points, cloud.points().data(), sizeof(DevicePoint) * _state->_pointCount, cudaMemcpyHostToDevice));
		result._uploadTimeMs = CudaHelper::stopTimer(uploadBegin, uploadEnd);
		cudaEventDestroy(uploadBegin);
		cudaEventDestroy(uploadEnd);

		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_indices), sizeof(uint32_t) * _state->_pointCount));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_tempIndices), sizeof(uint32_t) * _state->_pointCount));
		_state->_baseMemoryBytes =
			sizeof(DevicePoint) * _state->_pointCount +
			sizeof(uint32_t) * _state->_pointCount * 2;
		_state->_pointsReady = true;
	}

	_state->_nodeCapacity = estimateNodeCapacity(_state->_pointCount, _state->_leafCapacity, _state->_maxDepth, _state->_childSlotStride);
	_state->_levelScratchNodeCapacity = estimateLevelScratchNodeCapacity(
		_state->_pointCount,
		_state->_leafCapacity,
		_state->_nodeCapacity,
		_state->_childSlotStride);
	_state->_memoryBytes =
		_state->_baseMemoryBytes +
		sizeof(LinearMixedTreeNode) * _state->_nodeCapacity +
		sizeof(uint32_t) * _state->_levelScratchNodeCapacity * _state->_childSlotStride * 2 +
		sizeof(int) * _state->_maxDepth +
		sizeof(uint32_t) * _state->_maxDepth * 3 +
		sizeof(DeviceLevelCondition) * _state->_maxDepth +
		sizeof(uint32_t) * 2;
	checkMemoryBudget(_state->_memoryBytes, options._memoryBudgetMb);

	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_nodes), sizeof(LinearMixedTreeNode) * _state->_nodeCapacity));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_childCounts), sizeof(uint32_t) * _state->_levelScratchNodeCapacity * _state->_childSlotStride));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_writeCursors), sizeof(uint32_t) * _state->_levelScratchNodeCapacity * _state->_childSlotStride));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_splitTypes), sizeof(int) * _state->_maxDepth));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_leafCapacities), sizeof(uint32_t) * _state->_maxDepth));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_minSplits), sizeof(uint32_t) * _state->_maxDepth));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_conditions), sizeof(DeviceLevelCondition) * _state->_maxDepth));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_schemaBlockEndDepths), sizeof(uint32_t) * _state->_maxDepth));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_nodeCounter), sizeof(uint32_t)));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_overflowFlag), sizeof(uint32_t)));
	CudaHelper::checkError(cudaMemcpy(_state->_splitTypes, hostSplitTypes.data(), sizeof(int) * _state->_maxDepth, cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaMemcpy(_state->_leafCapacities, hostLeafCapacities.data(), sizeof(uint32_t) * _state->_maxDepth, cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaMemcpy(_state->_minSplits, hostMinSplits.data(), sizeof(uint32_t) * _state->_maxDepth, cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaMemcpy(_state->_conditions, hostConditions.data(), sizeof(DeviceLevelCondition) * _state->_maxDepth, cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaMemcpy(_state->_schemaBlockEndDepths, hostSchemaBlockEndDepths.data(), sizeof(uint32_t) * _state->_maxDepth, cudaMemcpyHostToDevice));

	cudaEvent_t buildBegin = nullptr;
	cudaEvent_t buildEnd = nullptr;
	CudaHelper::startTimer(buildBegin, buildEnd);

	const dim3 pointBlocks(static_cast<unsigned int>(divUp(_state->_pointCount, ThreadsPerBlock)));
	initializeIndicesKernel<<<pointBlocks, ThreadsPerBlock>>>(_state->_indices, _state->_pointCount);
	CudaHelper::synchronize("initializeMixedTreeIndicesKernel");

	initializeRootKernel<<<1, 1>>>(
		_state->_nodes,
		_state->_nodeCounter,
		_state->_pointCount,
		boundsMin.x,
		boundsMin.y,
		boundsMin.z,
		boundsMax.x,
		boundsMax.y,
		boundsMax.z);
	CudaHelper::synchronize("initializeMixedTreeRootKernel");
	CudaHelper::checkError(cudaMemset(_state->_overflowFlag, 0, sizeof(uint32_t)));

	size_t levelStart = 0;
	size_t levelCount = 1;
	uint32_t currentNodeCount = 1;
	for (size_t depth = 0; depth < _state->_maxDepth && levelCount > 0; ++depth)
	{
		if (levelCount > _state->_levelScratchNodeCapacity)
			throw std::runtime_error("MixedTree exceeded its level scratch budget. Increase leaf capacity or reduce max depth.");

		CudaHelper::checkError(cudaMemset(
			_state->_childCounts,
			0,
			sizeof(uint32_t) * levelCount * _state->_childSlotStride));
		const dim3 prepareBlocks(static_cast<unsigned int>(divUp(levelCount, ThreadsPerBlock)));

		countChildBucketsKernel<<<static_cast<unsigned int>(levelCount), ThreadsPerBlock>>>(
			_state->_points,
			_state->_indices,
			_state->_nodes,
			levelStart,
			levelCount,
			static_cast<uint32_t>(_state->_maxDepth),
			_state->_splitTypes,
			_state->_leafCapacities,
			_state->_minSplits,
			_state->_conditions,
			_state->_schemaBlockEndDepths,
			static_cast<uint32_t>(_state->_schemaDepthLimit),
			static_cast<uint32_t>(_state->_childSlotStride),
			_state->_childCounts);
		CudaHelper::synchronize("countMixedTreeChildBucketsKernel");

		const uint32_t previousNodeCount = currentNodeCount;

		prepareChildrenKernel<<<prepareBlocks, ThreadsPerBlock>>>(
			_state->_nodes,
			levelStart,
			levelCount,
			_state->_nodeCapacity,
			static_cast<uint32_t>(_state->_maxDepth),
			_state->_splitTypes,
			static_cast<uint32_t>(_state->_schemaDepthLimit),
			_state->_childCounts,
			static_cast<uint32_t>(_state->_childSlotStride),
			_state->_writeCursors,
			_state->_nodeCounter,
			_state->_overflowFlag);
		CudaHelper::synchronize("prepareMixedTreeChildrenKernel");

		uint32_t overflow = 0;
		CudaHelper::checkError(cudaMemcpy(&overflow, _state->_overflowFlag, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		if (overflow != 0)
			throw std::runtime_error("MixedTree exceeded its allocated node budget. Increase leaf capacity or reduce max depth.");

		CudaHelper::checkError(cudaMemcpy(
			_state->_tempIndices,
			_state->_indices,
			sizeof(uint32_t) * _state->_pointCount,
			cudaMemcpyDeviceToDevice));

		partitionIndicesKernel<<<static_cast<unsigned int>(levelCount), ThreadsPerBlock>>>(
			_state->_points,
			_state->_indices,
			_state->_tempIndices,
			_state->_nodes,
			levelStart,
			levelCount,
			static_cast<uint32_t>(_state->_maxDepth),
			_state->_splitTypes,
			static_cast<uint32_t>(_state->_schemaDepthLimit),
			static_cast<uint32_t>(_state->_childSlotStride),
			_state->_writeCursors);
		CudaHelper::synchronize("partitionMixedTreeIndicesKernel");
		std::swap(_state->_indices, _state->_tempIndices);

		uint32_t nextNodeCount = 0;
		CudaHelper::checkError(cudaMemcpy(&nextNodeCount, _state->_nodeCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		currentNodeCount = nextNodeCount;
		if (nextNodeCount > previousNodeCount)
		{
			refitNodeBoundsKernel<<<static_cast<unsigned int>(nextNodeCount - previousNodeCount), ThreadsPerBlock>>>(
				_state->_points,
				_state->_indices,
				_state->_nodes,
				previousNodeCount,
				static_cast<size_t>(nextNodeCount - previousNodeCount),
				_state->_splitTypes,
				static_cast<uint32_t>(_state->_schemaDepthLimit));
			CudaHelper::synchronize("refitMixedTreeNodeBoundsKernel");
		}
		levelStart = previousNodeCount;
		levelCount = nextNodeCount > previousNodeCount
			? static_cast<size_t>(nextNodeCount - previousNodeCount)
			: 0;
		_state->_allocatedNodes = nextNodeCount;
	}

	result._gpuBuildTimeMs = CudaHelper::stopTimer(buildBegin, buildEnd);
	cudaEventDestroy(buildBegin);
	cudaEventDestroy(buildEnd);

	if (_state->_allocatedNodes == 0)
	{
		uint32_t allocated = 0;
		CudaHelper::checkError(cudaMemcpy(&allocated, _state->_nodeCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		_state->_allocatedNodes = allocated;
	}

	std::vector<LinearMixedTreeNode> hostNodes(_state->_allocatedNodes);
	CudaHelper::checkError(cudaMemcpy(hostNodes.data(), _state->_nodes, sizeof(LinearMixedTreeNode) * _state->_allocatedNodes, cudaMemcpyDeviceToHost));

	size_t maxLeafOccupancy = 0;
	size_t indexedPoints = 0;
	size_t deepestNode = 0;
	for (const LinearMixedTreeNode& node : hostNodes)
	{
		if (node._pointCount == 0)
			continue;

		++_state->_actualNodes;
		deepestNode = std::max<size_t>(deepestNode, node._depth);
		if (node._childBase < 0)
		{
			++_state->_actualLeaves;
			indexedPoints += node._pointCount;
			maxLeafOccupancy = std::max<size_t>(maxLeafOccupancy, node._pointCount);
		}
	}

	result._gpuMemoryBytes = _state->_memoryBytes;
	result._metrics._buildTimeMs = result._gpuBuildTimeMs;
	result._metrics._numNodes = _state->_actualNodes;
	result._metrics._numLeaves = _state->_actualLeaves;
	result._metrics._indexedPoints = indexedPoints;
	result._metrics._maxDepth = deepestNode;
	result._metrics._averageLeafOccupancy = _state->_actualLeaves > 0
		? static_cast<double>(indexedPoints) / static_cast<double>(_state->_actualLeaves)
		: 0.0;
	result._metrics._maxLeafOccupancy = maxLeafOccupancy;
	result._metrics._memoryEstimateBytes = _state->_memoryBytes;
	fillActiveStructureStats(result, schema, hostNodes);
	return result;
}

PointGpu::QueryResult PointGpu::MixedTree::query(const std::vector<Query>& queries, const Options& options) const
{
	if (!_state || _state->_actualNodes == 0 || queries.empty())
		return {};

	CudaHelper::checkError(cudaSetDevice(_state->_device));

	QueryResult result;
	result._samples.reserve(queries.size());
	for (const Query& query : queries)
	{
		if (query._type == QueryType::Radius)
			++result._radiusQueries;
		else if (query._type == QueryType::CountRange)
			++result._countRangeQueries;
		else if (query._type == QueryType::Knn)
			++result._knnQueries;
		else
			++result._rangeQueries;
	}

	const size_t batchSize = options._queryBatchSize > 0
		? std::max<size_t>(1, options._queryBatchSize)
		: queries.size();
	const float clockRate = deviceClockRateKHz(_state->_device);

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
			return query._type == static_cast<int>(PointGpu::QueryType::Knn);
		});
		if (batchHasKnn)
			result._knnBackend = "gpu_bruteforce_knn";
		const bool useCooperativeQuery = shouldUseCooperativeQueryKernel(hostQueries, _state->_cloud);

		CudaHelper::checkError(cudaMemcpy(_state->_queryBuffer, hostQueries.data(), sizeof(DeviceQuery) * currentBatch, cudaMemcpyHostToDevice));
		if (useCooperativeQuery)
		{
			cooperativeQueryKernel<<<static_cast<unsigned int>(currentBatch), ThreadsPerBlock>>>(
				_state->_points,
				_state->_indices,
				_state->_nodes,
				_state->_queryBuffer,
				currentBatch,
				clockRate,
				_state->_sampleBuffer);
			CudaHelper::synchronize("MixedTreeCooperativeQueryKernel");
		}
		else
		{
			const dim3 queryBlocks(static_cast<unsigned int>(divUp(currentBatch, ThreadsPerBlock)));
			queryKernel<<<queryBlocks, ThreadsPerBlock>>>(
				_state->_points,
				_state->_indices,
				_state->_nodes,
				_state->_pointCount,
				_state->_queryBuffer,
				currentBatch,
				clockRate,
				_state->_sampleBuffer);
			CudaHelper::synchronize("MixedTreeQueryKernel");
		}
		if (batchHasKnn)
		{
			PointGpu::bruteForceKnnKernel<<<static_cast<unsigned int>(currentBatch), ThreadsPerBlock, sizeof(float) * ThreadsPerBlock>>>(
				_state->_points,
				_state->_pointCount,
				_state->_queryBuffer,
				currentBatch,
				clockRate,
				_state->_sampleBuffer);
			CudaHelper::synchronize("MixedTreeKnnQueryKernel");
		}

		hostSamples.resize(currentBatch);
		CudaHelper::checkError(cudaMemcpy(hostSamples.data(), _state->_sampleBuffer, sizeof(DeviceQuerySample) * currentBatch, cudaMemcpyDeviceToHost));
		for (const DeviceQuerySample& sample : hostSamples)
		{
			QuerySample converted;
			converted._visitedNodes = static_cast<size_t>(sample._visitedNodes);
			converted._testedPoints = static_cast<size_t>(sample._testedPoints);
			converted._returnedPoints = static_cast<size_t>(sample._returnedPoints);
			converted._elapsedMs = sample._elapsedMs;
			result._samples.push_back(converted);
		}
	}

	result._gpuQueryTimeMs = CudaHelper::stopTimer(queryBegin, queryEnd);
	cudaEventDestroy(queryBegin);
	cudaEventDestroy(queryEnd);

	result._metrics = summarizeGpuSamples(result._samples);
	return result;
}

bool PointGpu::MixedTree::built() const
{
	return _state && _state->_actualNodes > 0;
}

size_t PointGpu::MixedTree::pointCount() const
{
	return _state ? _state->_pointCount : 0;
}

size_t PointGpu::MixedTree::nodeCount() const
{
	return _state ? _state->_actualNodes : 0;
}

size_t PointGpu::MixedTree::leafCount() const
{
	return _state ? _state->_actualLeaves : 0;
}

#include "../../stdafx.h"
#include "Octree.h"

#include "../../CudaHelper.h"

#include <cub/cub.cuh>

namespace
{
	using PointGpu::DevicePoint;
	using PointGpu::DeviceQuery;
	using PointGpu::DeviceQuerySample;
	using PointGpu::LinearOctreeNode;

	constexpr int ThreadsPerBlock = 256;
	constexpr int QueryStackSize = 256;
	constexpr size_t MaxSupportedDepth = 12;

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
			throw std::runtime_error("Octree currently supports maxDepth <= 12.");
		return maxDepth;
	}

	size_t fullNodeCapacityForDepth(size_t maxDepth)
	{
		size_t capacity = 1;
		size_t levelWidth = 1;
		for (size_t depth = 0; depth < maxDepth; ++depth)
		{
			if (levelWidth > std::numeric_limits<size_t>::max() / 8)
				return std::numeric_limits<size_t>::max();
			levelWidth *= 8;
			if (capacity > std::numeric_limits<size_t>::max() - levelWidth)
				return std::numeric_limits<size_t>::max();
			capacity += levelWidth;
		}
		return capacity;
	}

	size_t estimateNodeCapacity(size_t pointCount, size_t leafCapacity, size_t maxDepth)
	{
		const size_t fullCapacity = fullNodeCapacityForDepth(maxDepth);
		const size_t targetLeaves = std::max<size_t>(1, divUp(pointCount, std::max<size_t>(1, leafCapacity)));
		const size_t estimatedInternal = std::max<size_t>(1, targetLeaves * 8);
		const size_t estimatedCapacity = estimatedInternal > (std::numeric_limits<size_t>::max() - 1) / 8
			? std::numeric_limits<size_t>::max()
			: 1 + estimatedInternal * 8;
		const size_t capacity = std::min(fullCapacity, estimatedCapacity);
		if (capacity > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
			throw std::runtime_error("Octree currently supports up to 2^32 - 1 allocated nodes.");
		return std::max<size_t>(9, capacity);
	}

	size_t estimateLevelScratchNodeCapacity(size_t pointCount, size_t leafCapacity, size_t nodeCapacity)
	{
		const size_t targetLeaves = std::max<size_t>(1, divUp(pointCount, std::max<size_t>(1, leafCapacity)));
		const size_t estimatedFrontier = targetLeaves > std::numeric_limits<size_t>::max() / 8
			? std::numeric_limits<size_t>::max()
			: targetLeaves * 8;
		return std::max<size_t>(9, std::min(nodeCapacity, std::min(pointCount, estimatedFrontier)));
	}

	bool isMidpointOctreeBuilder(const std::string& builder)
	{
		return builder == "octree" || builder == "ot";
	}

	bool isKarrasOctreeBuilder(const std::string& builder)
	{
		return builder == "karras_octree" || builder == "morton_octree" || builder == "octree_karras" || builder == "octree_morton";
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

	__device__ uint64_t expandBits21(uint32_t value)
	{
		uint64_t x = value & 0x1fffffu;
		x = (x | (x << 32)) & 0x1f00000000ffffull;
		x = (x | (x << 16)) & 0x1f0000ff0000ffull;
		x = (x | (x << 8)) & 0x100f00f00f00f00full;
		x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
		x = (x | (x << 2)) & 0x1249249249249249ull;
		return x;
	}

	__device__ uint64_t mortonCode21(const DevicePoint& point, float minX, float minY, float minZ, float extentX, float extentY, float extentZ)
	{
		const float nx = extentX > 0.0f ? fminf(fmaxf((point.x - minX) / extentX, 0.0f), 0.99999994f) : 0.0f;
		const float ny = extentY > 0.0f ? fminf(fmaxf((point.y - minY) / extentY, 0.0f), 0.99999994f) : 0.0f;
		const float nz = extentZ > 0.0f ? fminf(fmaxf((point.z - minZ) / extentZ, 0.0f), 0.99999994f) : 0.0f;
		const uint32_t ix = static_cast<uint32_t>(nx * 2097152.0f);
		const uint32_t iy = static_cast<uint32_t>(ny * 2097152.0f);
		const uint32_t iz = static_cast<uint32_t>(nz * 2097152.0f);
		return (expandBits21(ix) << 2) | (expandBits21(iy) << 1) | expandBits21(iz);
	}

	__device__ uint64_t prefixLowerKey(uint64_t prefix, uint32_t depth)
	{
		const uint32_t shift = 63u - depth * 3u;
		return prefix << shift;
	}

	__device__ uint64_t prefixUpperKey(uint64_t prefix, uint32_t depth)
	{
		const uint32_t shift = 63u - depth * 3u;
		return (prefix + 1ull) << shift;
	}

	__device__ uint32_t lowerBoundKey(const uint64_t* keys, uint32_t begin, uint32_t end, uint64_t target)
	{
		uint32_t first = begin;
		uint32_t count = end - begin;
		while (count > 0)
		{
			const uint32_t step = count / 2;
			const uint32_t candidate = first + step;
			if (keys[candidate] < target)
			{
				first = candidate + 1;
				count -= step + 1;
			}
			else
			{
				count = step;
			}
		}
		return first;
	}

	__device__ uint32_t octantForPoint(const DevicePoint& point, const LinearOctreeNode& node)
	{
		const float midX = (node.minX + node.maxX) * 0.5f;
		const float midY = (node.minY + node.maxY) * 0.5f;
		const float midZ = (node.minZ + node.maxZ) * 0.5f;
		return (point.x > midX ? 1u : 0u) |
			(point.y > midY ? 2u : 0u) |
			(point.z > midZ ? 4u : 0u);
	}

	__device__ LinearOctreeNode makeChildNode(const LinearOctreeNode& parent, uint32_t child, uint32_t offset, uint32_t count, int parentIndex)
	{
		const float midX = (parent.minX + parent.maxX) * 0.5f;
		const float midY = (parent.minY + parent.maxY) * 0.5f;
		const float midZ = (parent.minZ + parent.maxZ) * 0.5f;

		LinearOctreeNode node{};
		node.minX = (child & 1u) ? midX : parent.minX;
		node.maxX = (child & 1u) ? parent.maxX : midX;
		node.minY = (child & 2u) ? midY : parent.minY;
		node.maxY = (child & 2u) ? parent.maxY : midY;
		node.minZ = (child & 4u) ? midZ : parent.minZ;
		node.maxZ = (child & 4u) ? parent.maxZ : midZ;
		node.parent = parentIndex;
		node.childBase = -1;
		node.childMask = 0;
		node.pointOffset = offset;
		node.pointCount = count;
		node.flags = count > 0 ? 1u : 0u;
		node.depth = parent.depth + 1;
		return node;
	}

	__device__ bool rangeIntersectsNode(const LinearOctreeNode& node, const DeviceQuery& query)
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

	__device__ float distanceSquaredToNode(const LinearOctreeNode& node, const DeviceQuery& query)
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

	__global__ void initializeMortonKeysKernel(
		const DevicePoint* points,
		size_t pointCount,
		float minX,
		float minY,
		float minZ,
		float extentX,
		float extentY,
		float extentZ,
		uint64_t* keys,
		uint32_t* indices)
	{
		const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (index >= pointCount)
			return;

		keys[index] = mortonCode21(points[index], minX, minY, minZ, extentX, extentY, extentZ);
		indices[index] = static_cast<uint32_t>(index);
	}

	__global__ void initializeRootKernel(
		LinearOctreeNode* nodes,
		uint32_t* nodeCounter,
		size_t pointCount,
		float minX,
		float minY,
		float minZ,
		float maxX,
		float maxY,
		float maxZ)
	{
		LinearOctreeNode root{};
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

	__global__ void initializeMortonRootKernel(
		LinearOctreeNode* nodes,
		uint64_t* nodePrefixes,
		uint32_t* nodeCounter,
		size_t pointCount,
		float minX,
		float minY,
		float minZ,
		float maxX,
		float maxY,
		float maxZ)
	{
		LinearOctreeNode root{};
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
		nodePrefixes[0] = 0;
		*nodeCounter = 1;
	}

	__global__ void countChildBucketsKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		LinearOctreeNode* nodes,
		size_t levelStart,
		size_t levelCount,
		uint32_t leafCapacity,
		uint32_t minSplit,
		uint32_t maxDepth,
		uint32_t* childCounts)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= levelCount)
			return;

		const size_t nodeIndex = levelStart + localNode;
		LinearOctreeNode node = nodes[nodeIndex];
		if (node.pointCount == 0 || node.pointCount <= leafCapacity || node.pointCount < minSplit || node.depth >= maxDepth)
		{
			if (threadIdx.x == 0)
			{
				nodes[nodeIndex].childBase = -1;
				nodes[nodeIndex].childMask = 0;
				nodes[nodeIndex].flags = node.pointCount > 0 ? 1u : 0u;
			}
			return;
		}

		uint32_t localCounts[8] = {};
		for (uint32_t offset = threadIdx.x; offset < node.pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node.pointOffset + offset];
			const uint32_t child = octantForPoint(points[pointIndex], node);
			++localCounts[child];
		}

		for (uint32_t child = 0; child < 8; ++child)
		{
			if (localCounts[child] > 0)
				atomicAdd(&childCounts[localNode * 8 + child], localCounts[child]);
		}
	}

	__global__ void prepareChildrenKernel(
		LinearOctreeNode* nodes,
		size_t levelStart,
		size_t levelCount,
		size_t nodeCapacity,
		const uint32_t* childCounts,
		uint32_t* writeCursors,
		uint32_t* nodeCounter,
		uint32_t* overflowFlag)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (localNode >= levelCount)
			return;

		const size_t nodeIndex = levelStart + localNode;
		LinearOctreeNode node = nodes[nodeIndex];
		if (node.pointCount == 0)
			return;

		uint32_t nonEmptyChildren = 0;
		uint32_t childMask = 0;
		for (uint32_t child = 0; child < 8; ++child)
		{
			if (childCounts[localNode * 8 + child] > 0)
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

		const uint32_t childBase = atomicAdd(nodeCounter, 8u);
		if (static_cast<size_t>(childBase) + 8 > nodeCapacity)
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
		for (uint32_t child = 0; child < 8; ++child)
		{
			const uint32_t count = childCounts[localNode * 8 + child];
			nodes[childBase + child] = makeChildNode(node, child, runningOffset, count, static_cast<int>(nodeIndex));
			writeCursors[localNode * 8 + child] = runningOffset;
			runningOffset += count;
		}
	}

	__global__ void partitionIndicesKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		uint32_t* outputIndices,
		const LinearOctreeNode* nodes,
		size_t levelStart,
		size_t levelCount,
		uint32_t* writeCursors)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= levelCount)
			return;

		const size_t nodeIndex = levelStart + localNode;
		const LinearOctreeNode node = nodes[nodeIndex];
		if (node.pointCount == 0 || node.childBase < 0)
			return;

		for (uint32_t offset = threadIdx.x; offset < node.pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node.pointOffset + offset];
			const uint32_t child = octantForPoint(points[pointIndex], node);
			const uint32_t writeOffset = atomicAdd(&writeCursors[localNode * 8 + child], 1u);
			outputIndices[writeOffset] = pointIndex;
		}
	}

	__global__ void prepareMortonChildrenKernel(
		const uint64_t* sortedKeys,
		LinearOctreeNode* nodes,
		uint64_t* nodePrefixes,
		size_t levelStart,
		size_t levelCount,
		size_t nodeCapacity,
		uint32_t leafCapacity,
		uint32_t minSplit,
		uint32_t maxDepth,
		uint32_t* nodeCounter,
		uint32_t* overflowFlag)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (localNode >= levelCount)
			return;

		const size_t nodeIndex = levelStart + localNode;
		LinearOctreeNode node = nodes[nodeIndex];
		if (node.pointCount == 0)
			return;

		if (node.pointCount <= leafCapacity || node.pointCount < minSplit || node.depth >= maxDepth)
		{
			nodes[nodeIndex].childBase = -1;
			nodes[nodeIndex].childMask = 0;
			nodes[nodeIndex].flags = 1;
			return;
		}

		const uint32_t childDepth = node.depth + 1;
		const uint64_t parentPrefix = nodePrefixes[nodeIndex];
		const uint32_t begin = node.pointOffset;
		const uint32_t end = node.pointOffset + node.pointCount;

		uint32_t childBegins[8] = {};
		uint32_t childCounts[8] = {};
		uint32_t nonEmptyChildren = 0;
		uint32_t childMask = 0;

		for (uint32_t child = 0; child < 8; ++child)
		{
			const uint64_t childPrefix = (parentPrefix << 3) | child;
			const uint64_t lower = prefixLowerKey(childPrefix, childDepth);
			const uint64_t upper = prefixUpperKey(childPrefix, childDepth);
			const uint32_t childBegin = lowerBoundKey(sortedKeys, begin, end, lower);
			const uint32_t childEnd = lowerBoundKey(sortedKeys, childBegin, end, upper);
			const uint32_t childCount = childEnd - childBegin;
			childBegins[child] = childBegin;
			childCounts[child] = childCount;
			if (childCount > 0)
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

		const uint32_t childBase = atomicAdd(nodeCounter, 8u);
		if (static_cast<size_t>(childBase) + 8 > nodeCapacity)
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

		for (uint32_t child = 0; child < 8; ++child)
		{
			nodes[childBase + child] = makeChildNode(node, child, childBegins[child], childCounts[child], static_cast<int>(nodeIndex));
			nodePrefixes[childBase + child] = (parentPrefix << 3) | child;
		}
	}

	__global__ void refitOctreeNodeBoundsKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		LinearOctreeNode* nodes,
		size_t nodeStart,
		size_t nodeCount)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= nodeCount)
			return;

		const size_t nodeIndex = nodeStart + localNode;
		const LinearOctreeNode node = nodes[nodeIndex];
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
		const LinearOctreeNode* nodes,
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
			const LinearOctreeNode node = nodes[nodeIndex];
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

			for (uint32_t child = 0; child < 8; ++child)
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

struct PointGpu::Octree::DeviceState
{
	DevicePoint* points = nullptr;
	uint32_t* indices = nullptr;
	uint32_t* tempIndices = nullptr;
	uint64_t* keys = nullptr;
	uint64_t* sortedKeys = nullptr;
	LinearOctreeNode* nodes = nullptr;
	uint64_t* nodePrefixes = nullptr;
	uint32_t* childCounts = nullptr;
	uint32_t* writeCursors = nullptr;
	uint32_t* nodeCounter = nullptr;
	uint32_t* overflowFlag = nullptr;
	DeviceQuery* queryBuffer = nullptr;
	DeviceQuerySample* sampleBuffer = nullptr;
	void* sortTemporary = nullptr;
	size_t sortTemporaryBytes = 0;
	size_t queryCapacity = 0;
	size_t pointCount = 0;
	size_t nodeCapacity = 0;
	size_t allocatedNodes = 0;
	size_t actualNodes = 0;
	size_t actualLeaves = 0;
	size_t leafCapacity = 1;
	size_t minSplit = 2;
	size_t maxDepth = 0;
	size_t levelScratchNodeCapacity = 0;
	size_t baseMemoryBytes = 0;
	size_t memoryBytes = 0;
	int device = 0;
	const PointCloud* cloud = nullptr;
	bool pointsReady = false;
	bool sortedReady = false;
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
			message << "Octree needs about " << (static_cast<double>(bytes) / (1024.0 * 1024.0))
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

PointGpu::Octree::Octree()
	: _state(std::make_unique<DeviceState>())
{
}

PointGpu::Octree::~Octree()
{
	if (_state)
		release();
}

void PointGpu::Octree::release()
{
	releaseTree();
	cudaFree(_state->points);
	cudaFree(_state->indices);
	cudaFree(_state->tempIndices);
	cudaFree(_state->keys);
	cudaFree(_state->sortedKeys);
	cudaFree(_state->sortTemporary);
	releaseQueryBuffers(*_state);
	*_state = DeviceState();
}

void PointGpu::Octree::releaseTree()
{
	cudaFree(_state->nodes);
	cudaFree(_state->nodePrefixes);
	cudaFree(_state->childCounts);
	cudaFree(_state->writeCursors);
	cudaFree(_state->nodeCounter);
	cudaFree(_state->overflowFlag);
	_state->nodes = nullptr;
	_state->nodePrefixes = nullptr;
	_state->childCounts = nullptr;
	_state->writeCursors = nullptr;
	_state->nodeCounter = nullptr;
	_state->overflowFlag = nullptr;
	_state->nodeCapacity = 0;
	_state->allocatedNodes = 0;
	_state->actualNodes = 0;
	_state->actualLeaves = 0;
	_state->levelScratchNodeCapacity = 0;
	_state->memoryBytes = _state->baseMemoryBytes;
}

bool PointGpu::Octree::isAvailable(std::string* error)
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

int PointGpu::Octree::deviceCount()
{
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess)
		return 0;
	return count;
}

PointGpu::BuildResult PointGpu::Octree::build(const PointCloud& cloud, const SchemaConfig& schema, const Options& options)
{
	const std::string builder = options.builder.empty() ? "octree" : options.builder;
	const bool buildKarrasOctree = isKarrasOctreeBuilder(builder);
	if (!isMidpointOctreeBuilder(builder) && !buildKarrasOctree)
		throw std::runtime_error("Octree evaluator supports --cuda-builder octree, ot, karras_octree, morton_octree, octree_karras, or octree_morton.");
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("Octree currently supports up to 2^32 - 1 points.");

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
	const bool canReuseSortedPoints = buildKarrasOctree && canReusePoints && _state->sortedReady;

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

	if (canReusePoints && !buildKarrasOctree && _state->keys)
	{
		cudaFree(_state->keys);
		cudaFree(_state->sortedKeys);
		cudaFree(_state->sortTemporary);
		_state->keys = nullptr;
		_state->sortedKeys = nullptr;
		_state->sortTemporary = nullptr;
		_state->sortTemporaryBytes = 0;
		_state->sortedReady = false;
		_state->baseMemoryBytes =
			sizeof(DevicePoint) * _state->pointCount +
			sizeof(uint32_t) * _state->pointCount * 2;
	}

	_state->pointCount = cloud.size();
	_state->leafCapacity = leafCapacityForSchema(schema);
	_state->minSplit = minSplitForSchema(schema);
	_state->maxDepth = maxDepthForSchema(schema);
	_state->device = device;
	_state->cloud = &cloud;

	BuildResult result;
	result.device = device;
	result.builder = buildKarrasOctree ? "karras_octree" : "octree";

	if (cloud.empty())
		return result;

	const glm::vec3 boundsMin = cloud.bounds().min();
	const glm::vec3 boundsMax = cloud.bounds().max();
	const glm::vec3 extent = glm::max(cloud.coordinateRange(), glm::vec3(0.0f));

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

	if (buildKarrasOctree && !_state->keys)
	{
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->keys), sizeof(uint64_t) * _state->pointCount));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->sortedKeys), sizeof(uint64_t) * _state->pointCount));
		CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(
			nullptr,
			_state->sortTemporaryBytes,
			_state->keys,
			_state->sortedKeys,
			_state->indices,
			_state->tempIndices,
			static_cast<int>(_state->pointCount)));
		CudaHelper::checkError(cudaMalloc(&_state->sortTemporary, _state->sortTemporaryBytes));
		_state->baseMemoryBytes +=
			sizeof(uint64_t) * _state->pointCount * 2 +
			_state->sortTemporaryBytes;
		_state->sortedReady = false;
	}

	_state->nodeCapacity = estimateNodeCapacity(_state->pointCount, _state->leafCapacity, _state->maxDepth);
	_state->levelScratchNodeCapacity = estimateLevelScratchNodeCapacity(_state->pointCount, _state->leafCapacity, _state->nodeCapacity);
	if (buildKarrasOctree)
	{
		_state->memoryBytes =
			_state->baseMemoryBytes +
			sizeof(LinearOctreeNode) * _state->nodeCapacity +
			sizeof(uint64_t) * _state->nodeCapacity +
			sizeof(uint32_t) * 2;
	}
	else
	{
		_state->memoryBytes =
			_state->baseMemoryBytes +
			sizeof(LinearOctreeNode) * _state->nodeCapacity +
			sizeof(uint32_t) * _state->levelScratchNodeCapacity * 8 * 2 +
			sizeof(uint32_t) * 2;
	}
	checkMemoryBudget(_state->memoryBytes, options.memoryBudgetMb);

	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->nodes), sizeof(LinearOctreeNode) * _state->nodeCapacity));
	if (buildKarrasOctree)
	{
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->nodePrefixes), sizeof(uint64_t) * _state->nodeCapacity));
	}
	else
	{
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->childCounts), sizeof(uint32_t) * _state->levelScratchNodeCapacity * 8));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->writeCursors), sizeof(uint32_t) * _state->levelScratchNodeCapacity * 8));
	}
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->nodeCounter), sizeof(uint32_t)));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->overflowFlag), sizeof(uint32_t)));

	cudaEvent_t buildBegin = nullptr;
	cudaEvent_t buildEnd = nullptr;
	CudaHelper::startTimer(buildBegin, buildEnd);

	const dim3 pointBlocks(static_cast<unsigned int>(divUp(_state->pointCount, ThreadsPerBlock)));
	if (buildKarrasOctree)
	{
		if (!canReuseSortedPoints)
		{
			initializeMortonKeysKernel<<<pointBlocks, ThreadsPerBlock>>>(
				_state->points,
				_state->pointCount,
				boundsMin.x,
				boundsMin.y,
				boundsMin.z,
				extent.x,
				extent.y,
				extent.z,
				_state->keys,
				_state->indices);
			CudaHelper::synchronize("initializeOctreeMortonKeysKernel");

			CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(
				_state->sortTemporary,
				_state->sortTemporaryBytes,
				_state->keys,
				_state->sortedKeys,
				_state->indices,
				_state->tempIndices,
				static_cast<int>(_state->pointCount)));
			CudaHelper::synchronize("cub::DeviceRadixSort::SortPairs octree");
			std::swap(_state->keys, _state->sortedKeys);
			std::swap(_state->indices, _state->tempIndices);
			_state->sortedReady = true;
		}
	}
	else
	{
		initializeIndicesKernel<<<pointBlocks, ThreadsPerBlock>>>(_state->indices, _state->pointCount);
		CudaHelper::synchronize("initializeOctreeIndicesKernel");
	}

	if (buildKarrasOctree)
	{
		initializeMortonRootKernel<<<1, 1>>>(
			_state->nodes,
			_state->nodePrefixes,
			_state->nodeCounter,
			_state->pointCount,
			boundsMin.x,
			boundsMin.y,
			boundsMin.z,
			boundsMax.x,
			boundsMax.y,
			boundsMax.z);
		CudaHelper::synchronize("initializeKarrasOctreeRootKernel");
	}
	else
	{
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
		CudaHelper::synchronize("initializeOctreeRootKernel");
	}
	CudaHelper::checkError(cudaMemset(_state->overflowFlag, 0, sizeof(uint32_t)));

	size_t levelStart = 0;
	size_t levelCount = 1;
	uint32_t currentNodeCount = 1;
	for (size_t depth = 0; depth < _state->maxDepth && levelCount > 0; ++depth)
	{
		const dim3 prepareBlocks(static_cast<unsigned int>(divUp(levelCount, ThreadsPerBlock)));
		const uint32_t previousNodeCount = currentNodeCount;

		if (buildKarrasOctree)
		{
			prepareMortonChildrenKernel<<<prepareBlocks, ThreadsPerBlock>>>(
				_state->keys,
				_state->nodes,
				_state->nodePrefixes,
				levelStart,
				levelCount,
				_state->nodeCapacity,
				static_cast<uint32_t>(_state->leafCapacity),
				static_cast<uint32_t>(_state->minSplit),
				static_cast<uint32_t>(_state->maxDepth),
				_state->nodeCounter,
				_state->overflowFlag);
			CudaHelper::synchronize("prepareKarrasOctreeChildrenKernel");
		}
		else
		{
			if (levelCount > _state->levelScratchNodeCapacity)
				throw std::runtime_error("Octree exceeded its level scratch budget. Increase leaf capacity or reduce max depth.");

			CudaHelper::checkError(cudaMemset(
				_state->childCounts,
				0,
				sizeof(uint32_t) * levelCount * 8));
			countChildBucketsKernel<<<static_cast<unsigned int>(levelCount), ThreadsPerBlock>>>(
				_state->points,
				_state->indices,
				_state->nodes,
				levelStart,
				levelCount,
				static_cast<uint32_t>(_state->leafCapacity),
				static_cast<uint32_t>(_state->minSplit),
				static_cast<uint32_t>(_state->maxDepth),
				_state->childCounts);
			CudaHelper::synchronize("countOctreeChildBucketsKernel");

			prepareChildrenKernel<<<prepareBlocks, ThreadsPerBlock>>>(
				_state->nodes,
				levelStart,
				levelCount,
				_state->nodeCapacity,
				_state->childCounts,
				_state->writeCursors,
				_state->nodeCounter,
				_state->overflowFlag);
			CudaHelper::synchronize("prepareOctreeChildrenKernel");
		}

		uint32_t overflow = 0;
		CudaHelper::checkError(cudaMemcpy(&overflow, _state->overflowFlag, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		if (overflow != 0)
			throw std::runtime_error("Octree exceeded its allocated node budget. Increase leaf capacity or reduce max depth.");

		if (!buildKarrasOctree)
		{
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
				_state->writeCursors);
			CudaHelper::synchronize("partitionOctreeIndicesKernel");
			std::swap(_state->indices, _state->tempIndices);
		}

		uint32_t nextNodeCount = 0;
		CudaHelper::checkError(cudaMemcpy(&nextNodeCount, _state->nodeCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		currentNodeCount = nextNodeCount;
		if (buildKarrasOctree && nextNodeCount > previousNodeCount)
		{
			refitOctreeNodeBoundsKernel<<<static_cast<unsigned int>(nextNodeCount - previousNodeCount), ThreadsPerBlock>>>(
				_state->points,
				_state->indices,
				_state->nodes,
				previousNodeCount,
				static_cast<size_t>(nextNodeCount - previousNodeCount));
			CudaHelper::synchronize("refitKarrasOctreeNodeBoundsKernel");
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

	std::vector<LinearOctreeNode> hostNodes(_state->allocatedNodes);
	CudaHelper::checkError(cudaMemcpy(hostNodes.data(), _state->nodes, sizeof(LinearOctreeNode) * _state->allocatedNodes, cudaMemcpyDeviceToHost));

	size_t maxLeafOccupancy = 0;
	size_t indexedPoints = 0;
	size_t deepestNode = 0;
	for (const LinearOctreeNode& node : hostNodes)
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
	return result;
}

PointGpu::QueryResult PointGpu::Octree::query(const std::vector<Query>& queries, const Options& options) const
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
		CudaHelper::synchronize("octreeQueryKernel");
		if (batchHasKnn)
		{
			PointGpu::bruteForceKnnKernel<<<static_cast<unsigned int>(currentBatch), ThreadsPerBlock, sizeof(float) * ThreadsPerBlock>>>(
				_state->points,
				_state->pointCount,
				_state->queryBuffer,
				currentBatch,
				clockRate,
				_state->sampleBuffer);
			CudaHelper::synchronize("octreeKnnQueryKernel");
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

bool PointGpu::Octree::built() const
{
	return _state && _state->actualNodes > 0;
}

size_t PointGpu::Octree::pointCount() const
{
	return _state ? _state->pointCount : 0;
}

size_t PointGpu::Octree::nodeCount() const
{
	return _state ? _state->actualNodes : 0;
}

size_t PointGpu::Octree::leafCount() const
{
	return _state ? _state->actualLeaves : 0;
}

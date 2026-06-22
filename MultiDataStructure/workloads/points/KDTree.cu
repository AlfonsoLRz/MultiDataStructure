#include "../../stdafx.h"
#include "KDTree.h"

#include "../../CudaHelper.h"

namespace
{
	using PointGpu::DevicePoint;
	using PointGpu::DeviceQuery;
	using PointGpu::DeviceQuerySample;
	using PointGpu::LinearNode;

	constexpr int ThreadsPerBlock = 256;
	constexpr int QueryStackSize = 128;
	constexpr size_t MaxSupportedDepth = 22;

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
			maxDepth = 12;
		if (maxDepth > MaxSupportedDepth)
			throw std::runtime_error("KDTree currently supports maxDepth <= 22.");
		return maxDepth;
	}

	// Reads the first level's axisPolicy; "round_robin" else LongestExtent (also for blank/unknown).
	PointGpu::KdAxisPolicy axisPolicyFromSchema(const SchemaConfig& schema)
	{
		for (const SchemaLevelConfig& level : schema._levels)
		{
			if (level._axisPolicy.empty())
				continue;
			std::string normalized = level._axisPolicy;
			std::transform(normalized.begin(), normalized.end(), normalized.begin(),
				[](unsigned char c) { return std::tolower(c); });
			normalized.erase(std::remove(normalized.begin(), normalized.end(), '_'), normalized.end());
			normalized.erase(std::remove(normalized.begin(), normalized.end(), '-'), normalized.end());
			if (normalized == "roundrobin" || normalized == "rr")
				return PointGpu::KdAxisPolicy::RoundRobin;
			break;
		}
		return PointGpu::KdAxisPolicy::LongestExtent;
	}

	size_t nodeCapacityForDepth(size_t maxDepth)
	{
		size_t capacity = 1;
		for (size_t depth = 0; depth < maxDepth + 1; ++depth)
			capacity *= 2;
		return capacity - 1;
	}

	size_t nodeStartForDepth(size_t depth)
	{
		return (static_cast<size_t>(1) << depth) - 1;
	}

	size_t nodeCountForDepth(size_t depth)
	{
		return static_cast<size_t>(1) << depth;
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

	// Flag bit layout matches LinearNode in PointGpuTypes.h: bit0 leaf, bits1..2 axis, bit3 axis-stored, bits4..11 depth.
	constexpr uint32_t kLeafFlagBit = 0x1u;
	constexpr uint32_t kAxisMask = 0x6u;       // bits 1..2
	constexpr int kAxisShift = 1;
	constexpr uint32_t kAxisStoredBit = 0x8u;  // bit 3
	constexpr uint32_t kDepthMask = 0xFF0u;    // bits 4..11
	constexpr int kDepthShift = 4;

	__device__ __host__ uint32_t encodeNodeFlags(bool isLeaf, int axis, bool axisStored, uint32_t depth)
	{
		uint32_t flags = isLeaf ? kLeafFlagBit : 0u;
		if (axisStored)
		{
			flags |= kAxisStoredBit;
			flags |= (static_cast<uint32_t>(axis & 0x3) << kAxisShift);
		}
		flags |= (std::min(depth, 0xFFu) << kDepthShift);
		return flags;
	}

	__device__ int splitAxisForNode(const LinearNode& node)
	{
		if (node._flags & kAxisStoredBit)
			return static_cast<int>((node._flags & kAxisMask) >> kAxisShift);
		const float extentX = node._maxX - node._minX;
		const float extentY = node._maxY - node._minY;
		const float extentZ = node._maxZ - node._minZ;
		if (extentX >= extentY && extentX >= extentZ)
			return 0;
		if (extentY >= extentZ)
			return 1;
		return 2;
	}

	__device__ float splitPlaneForNode(const LinearNode& node, int axis)
	{
		if (axis == 0)
			return (node._minX + node._maxX) * 0.5f;
		if (axis == 1)
			return (node._minY + node._maxY) * 0.5f;
		return (node._minZ + node._maxZ) * 0.5f;
	}

	__device__ float coordinateForAxis(const DevicePoint& point, int axis)
	{
		if (axis == 0)
			return point.x;
		if (axis == 1)
			return point.y;
		return point.z;
	}

	__device__ bool pointGoesLeft(const DevicePoint& point, const LinearNode& node)
	{
		const int axis = splitAxisForNode(node);
		const float plane = splitPlaneForNode(node, axis);
		return coordinateForAxis(point, axis) <= plane;
	}

	bool isKDTreeBuilder(const std::string& builder)
	{
		return builder == "kdtree" || builder == "kd_tree" || builder == "kd";
	}

	bool isBIHBuilder(const std::string& builder)
	{
		return builder == "bih" || builder == "interval_hierarchy" || builder == "binary_interval_hierarchy";
	}

	__device__ bool rangeIntersectsNode(const LinearNode& node, const DeviceQuery& query)
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

	__device__ float distanceSquaredToNode(const LinearNode& node, const DeviceQuery& query)
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

	__global__ void initializeNodesKernel(LinearNode* nodes, size_t nodeCount)
	{
		const size_t nodeIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (nodeIndex >= nodeCount)
			return;

		LinearNode node{};
		node._minX = 3.402823466e+38F;
		node._minY = 3.402823466e+38F;
		node._minZ = 3.402823466e+38F;
		node._maxX = -3.402823466e+38F;
		node._maxY = -3.402823466e+38F;
		node._maxZ = -3.402823466e+38F;
		node._left = -1;
		node._right = -1;
		node._parent = -1;
		node._pointOffset = 0;
		node._pointCount = 0;
		node._flags = 0;
		nodes[nodeIndex] = node;
	}

	__global__ void initializeRootKernel(
		LinearNode* nodes,
		size_t pointCount,
		float minX,
		float minY,
		float minZ,
		float maxX,
		float maxY,
		float maxZ,
		int axisPolicy)
	{
		LinearNode root{};
		root._minX = minX;
		root._minY = minY;
		root._minZ = minZ;
		root._maxX = maxX;
		root._maxY = maxY;
		root._maxZ = maxZ;
		root._left = -1;
		root._right = -1;
		root._parent = -1;
		root._pointOffset = 0;
		root._pointCount = static_cast<uint32_t>(pointCount);
		// Root depth 0; round-robin root axis is X, rewritten by the split kernel when it splits.
		const bool storeAxis = axisPolicy == static_cast<int>(PointGpu::KdAxisPolicy::RoundRobin);
		root._flags = encodeNodeFlags(true /*temporarily leaf*/, 0, storeAxis, 0u);
		nodes[0] = root;
	}

	__global__ void countSplitsKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		LinearNode* nodes,
		size_t nodeStart,
		size_t nodeCount,
		uint32_t leafCapacity,
		uint32_t minSplit,
		uint32_t* leftCounts)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= nodeCount)
			return;

		const size_t nodeIndex = nodeStart + localNode;
		LinearNode node = nodes[nodeIndex];
		if (node._pointCount == 0 || node._pointCount <= leafCapacity || node._pointCount < minSplit)
		{
			if (threadIdx.x == 0)
			{
				nodes[nodeIndex]._left = -1;
				nodes[nodeIndex]._right = -1;
				nodes[nodeIndex]._flags = node._pointCount > 0 ? 1u : 0u;
			}
			return;
		}

		uint32_t localLeft = 0;
		for (uint32_t offset = threadIdx.x; offset < node._pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node._pointOffset + offset];
			if (pointGoesLeft(points[pointIndex], node))
				++localLeft;
		}

		if (localLeft > 0)
			atomicAdd(&leftCounts[nodeIndex], localLeft);
	}

	__global__ void prepareSplitNodesKernel(
		LinearNode* nodes,
		size_t nodeStart,
		size_t nodeCount,
		const uint32_t* leftCounts,
		uint32_t* writeCursors,
		int axisPolicy,
		uint32_t depth)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (localNode >= nodeCount)
			return;

		const size_t nodeIndex = nodeStart + localNode;
		LinearNode node = nodes[nodeIndex];
		if (node._pointCount == 0)
			return;

		const bool storeAxis = axisPolicy == static_cast<int>(PointGpu::KdAxisPolicy::RoundRobin);
		const uint32_t leftCount = leftCounts[nodeIndex];
		const uint32_t rightCount = node._pointCount - leftCount;
		if (leftCount == 0 || rightCount == 0)
		{
			nodes[nodeIndex]._left = -1;
			nodes[nodeIndex]._right = -1;
			// Mark leaf, preserve depth, drop the axis-stored bit (no split happened here).
			nodes[nodeIndex]._flags = encodeNodeFlags(true, 0, false, depth);
			return;
		}

		// Pick the split axis: round-robin uses depth%3; longest-extent leaves the bit clear.
		int axis = 0;
		if (axisPolicy == static_cast<int>(PointGpu::KdAxisPolicy::RoundRobin))
			axis = static_cast<int>(depth % 3u);
		else
			axis = splitAxisForNode(node);

		const float plane = splitPlaneForNode(node, axis);
		const int leftIndex = static_cast<int>(nodeIndex * 2 + 1);
		const int rightIndex = leftIndex + 1;
		nodes[nodeIndex]._left = leftIndex;
		nodes[nodeIndex]._right = rightIndex;
		nodes[nodeIndex]._flags = encodeNodeFlags(false, axis, storeAxis, depth);

		LinearNode left = node;
		left._left = -1;
		left._right = -1;
		left._parent = static_cast<int>(nodeIndex);
		left._pointOffset = node._pointOffset;
		left._pointCount = leftCount;
		left._flags = encodeNodeFlags(true, 0, false, depth + 1u);

		LinearNode right = node;
		right._left = -1;
		right._right = -1;
		right._parent = static_cast<int>(nodeIndex);
		right._pointOffset = node._pointOffset + leftCount;
		right._pointCount = rightCount;
		right._flags = encodeNodeFlags(true, 0, false, depth + 1u);

		if (axis == 0)
		{
			left._maxX = plane;
			right._minX = plane;
		}
		else if (axis == 1)
		{
			left._maxY = plane;
			right._minY = plane;
		}
		else
		{
			left._maxZ = plane;
			right._minZ = plane;
		}

		nodes[leftIndex] = left;
		nodes[rightIndex] = right;
		writeCursors[leftIndex] = left._pointOffset;
		writeCursors[rightIndex] = right._pointOffset;
	}

	__global__ void partitionIndicesKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		uint32_t* outputIndices,
		const LinearNode* nodes,
		size_t nodeStart,
		size_t nodeCount,
		uint32_t* writeCursors)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= nodeCount)
			return;

		const size_t nodeIndex = nodeStart + localNode;
		const LinearNode node = nodes[nodeIndex];
		if (node._pointCount == 0 || node._left < 0 || node._right < 0)
			return;

		for (uint32_t offset = threadIdx.x; offset < node._pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node._pointOffset + offset];
			const int childIndex = pointGoesLeft(points[pointIndex], node) ? node._left : node._right;
			const uint32_t writeOffset = atomicAdd(&writeCursors[childIndex], 1u);
			outputIndices[writeOffset] = pointIndex;
		}
	}

	__global__ void refitNodeBoundsKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		LinearNode* nodes,
		size_t nodeStart,
		size_t nodeCount)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= nodeCount)
			return;

		const size_t nodeIndex = nodeStart + localNode;
		const LinearNode node = nodes[nodeIndex];
		if (node._pointCount == 0)
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
		const LinearNode* nodes,
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
			const LinearNode node = nodes[nodeIndex];
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
			if (node._left < 0 || node._right < 0)
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

			if (stackSize + 2 <= QueryStackSize)
			{
				stack[stackSize++] = node._left;
				stack[stackSize++] = node._right;
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

	__global__ void treeKnnKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		const LinearNode* nodes,
		const DeviceQuery* queries,
		size_t queryCount,
		float clockRateKHz,
		DeviceQuerySample* samples,
		uint32_t* outputIndices,
		float* outputDistances)
	{
		const size_t queryIndex = static_cast<size_t>(blockIdx.x);
		if (queryIndex >= queryCount)
			return;

		const DeviceQuery query = queries[queryIndex];
		if (query._type != static_cast<int>(PointGpu::QueryType::Knn))
			return;

		const unsigned long long begin = clock64();
		const uint32_t requestedK = query._knnK;
		const uint32_t trackedK = requestedK < PointGpu::MaxTrackedKnnK ? requestedK : PointGpu::MaxTrackedKnnK;

		__shared__ int stack[QueryStackSize];
		__shared__ int stackSize;
		__shared__ float bestDistances[PointGpu::MaxTrackedKnnK];
		__shared__ uint32_t bestIndices[PointGpu::MaxTrackedKnnK];
		__shared__ uint32_t bestFound;
		__shared__ unsigned long long visited;
		__shared__ unsigned long long tested;
		__shared__ float candidateDistances[ThreadsPerBlock * PointGpu::MaxTrackedKnnK];
		__shared__ uint32_t candidateIndices[ThreadsPerBlock * PointGpu::MaxTrackedKnnK];
		__shared__ uint32_t candidateCounts[ThreadsPerBlock];

		if (threadIdx.x == 0)
		{
			stackSize = 0;
			bestFound = 0;
			visited = 0;
			tested = 0;
			for (uint32_t i = 0; i < PointGpu::MaxTrackedKnnK; ++i)
			{
				bestDistances[i] = 3.402823466e+38F;
				bestIndices[i] = UINT_MAX;
			}
			if (trackedK > 0)
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

			const LinearNode node = nodes[nodeIndex];
			bool processNode = node._pointCount > 0;
			if (processNode)
			{
				const float lowerBound = distanceSquaredToNode(node, query);
				processNode = lowerBound <= PointGpu::worstKnnDistance(bestDistances, bestFound, trackedK);
			}

			if (processNode && threadIdx.x == 0)
				++visited;
			__syncthreads();

			if (!processNode)
				continue;

			if (node._left < 0 || node._right < 0)
			{
				float localDistances[PointGpu::MaxTrackedKnnK];
				uint32_t localIndices[PointGpu::MaxTrackedKnnK];
				uint32_t localFound = 0;
				for (uint32_t i = 0; i < PointGpu::MaxTrackedKnnK; ++i)
				{
					localDistances[i] = 3.402823466e+38F;
					localIndices[i] = UINT_MAX;
				}

				for (uint32_t i = threadIdx.x; i < node._pointCount; i += blockDim.x)
				{
					const uint32_t pointIndex = indices[node._pointOffset + i];
					const float distanceSquared = PointGpu::pointDistanceSquared(points[pointIndex], query);
					PointGpu::insertKnnHit(localDistances, localIndices, localFound, trackedK, distanceSquared, pointIndex);
				}

				const uint32_t candidateBase = threadIdx.x * PointGpu::MaxTrackedKnnK;
				candidateCounts[threadIdx.x] = localFound;
				for (uint32_t i = 0; i < PointGpu::MaxTrackedKnnK; ++i)
				{
					candidateDistances[candidateBase + i] = localDistances[i];
					candidateIndices[candidateBase + i] = localIndices[i];
				}
				__syncthreads();

				if (threadIdx.x == 0)
				{
					tested += node._pointCount;
					for (uint32_t thread = 0; thread < blockDim.x; ++thread)
					{
						const uint32_t count = candidateCounts[thread];
						const uint32_t base = thread * PointGpu::MaxTrackedKnnK;
						for (uint32_t i = 0; i < count; ++i)
						{
							PointGpu::insertKnnHit(
								bestDistances,
								bestIndices,
								bestFound,
								trackedK,
								candidateDistances[base + i],
								candidateIndices[base + i]);
						}
					}
				}
				__syncthreads();
				continue;
			}

			if (threadIdx.x == 0)
			{
				const float leftDistance = node._left >= 0 ? distanceSquaredToNode(nodes[node._left], query) : 3.402823466e+38F;
				const float rightDistance = node._right >= 0 ? distanceSquaredToNode(nodes[node._right], query) : 3.402823466e+38F;
				const float worst = PointGpu::worstKnnDistance(bestDistances, bestFound, trackedK);
				const bool pushLeft = node._left >= 0 && leftDistance <= worst;
				const bool pushRight = node._right >= 0 && rightDistance <= worst;
				if (pushLeft && pushRight)
				{
					if (leftDistance <= rightDistance)
					{
						if (stackSize < QueryStackSize) stack[stackSize++] = node._right;
						if (stackSize < QueryStackSize) stack[stackSize++] = node._left;
					}
					else
					{
						if (stackSize < QueryStackSize) stack[stackSize++] = node._left;
						if (stackSize < QueryStackSize) stack[stackSize++] = node._right;
					}
				}
				else if (pushLeft)
				{
					if (stackSize < QueryStackSize) stack[stackSize++] = node._left;
				}
				else if (pushRight)
				{
					if (stackSize < QueryStackSize) stack[stackSize++] = node._right;
				}
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			PointGpu::sortKnnHits(bestDistances, bestIndices, bestFound);
			const size_t outputBase = queryIndex * PointGpu::MaxTrackedKnnK;
			for (uint32_t i = 0; i < PointGpu::MaxTrackedKnnK; ++i)
			{
				outputIndices[outputBase + i] = i < bestFound ? bestIndices[i] : UINT_MAX;
				outputDistances[outputBase + i] = i < bestFound ? bestDistances[i] : 3.402823466e+38F;
			}

			const unsigned long long end = clock64();
			DeviceQuerySample sample{};
			sample._visitedNodes = visited;
			sample._testedPoints = tested;
			sample._returnedPoints = bestFound;
			sample._elapsedMs = clockRateKHz > 0.0f ? static_cast<float>(end - begin) / clockRateKHz : 0.0f;
			samples[queryIndex] = sample;
		}
	}
}

struct PointGpu::KDTree::DeviceState
{
	DevicePoint*		_points = nullptr;
	uint32_t*			_indices = nullptr;
	uint32_t*			_tempIndices = nullptr;
	LinearNode*			_nodes = nullptr;
	uint32_t*			_leftCounts = nullptr;
	uint32_t*			_writeCursors = nullptr;
	DeviceQuery*		_queryBuffer = nullptr;
	DeviceQuerySample*	_sampleBuffer = nullptr;
	uint32_t*			_knnIndexBuffer = nullptr;
	float*				_knnDistanceBuffer = nullptr;
	size_t				_pointCount = 0;
	size_t				_nodeCapacity = 0;
	size_t				_actualNodes = 0;
	size_t				_actualLeaves = 0;
	size_t				_leafCapacity = 1;
	size_t				_minSplit = 2;
	size_t				_maxDepth = 0;
	size_t				_queryCapacity = 0;
	size_t				_knnCapacity = 0;
	size_t				_baseMemoryBytes = 0;
	size_t				_memoryBytes = 0;
	int					_device = 0;
	const PointCloud*	_cloud = nullptr;
	bool				_pointsReady = false;
};

namespace
{
	void checkMemoryBudget(size_t bytes, size_t budgetMb, const char* label)
	{
		if (budgetMb == 0)
			return;

		const size_t budgetBytes = budgetMb * 1024ull * 1024ull;
		if (bytes > budgetBytes)
		{
			std::ostringstream message;
			message << label << " needs about " << (static_cast<double>(bytes) / (1024.0 * 1024.0))
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
		cudaFree(state._knnIndexBuffer);
		cudaFree(state._knnDistanceBuffer);
		state._queryBuffer = nullptr;
		state._sampleBuffer = nullptr;
		state._knnIndexBuffer = nullptr;
		state._knnDistanceBuffer = nullptr;
		state._queryCapacity = 0;
		state._knnCapacity = 0;
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

	template <typename State>
	void ensureKnnBuffers(State& state, size_t capacity)
	{
		if (state._knnCapacity >= capacity && state._knnIndexBuffer && state._knnDistanceBuffer)
			return;

		cudaFree(state._knnIndexBuffer);
		cudaFree(state._knnDistanceBuffer);
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&state._knnIndexBuffer), sizeof(uint32_t) * capacity * PointGpu::MaxTrackedKnnK));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&state._knnDistanceBuffer), sizeof(float) * capacity * PointGpu::MaxTrackedKnnK));
		state._knnCapacity = capacity;
	}

	std::string normalizeKnnBackend(std::string backend)
	{
		std::transform(backend.begin(), backend.end(), backend.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		std::replace(backend.begin(), backend.end(), '-', '_');
		return backend;
	}

	bool requestsBruteForceKnn(const PointGpu::Options& options)
	{
		const std::string backend = normalizeKnnBackend(options._knnBackend);
		return backend == "bruteforce" ||
			backend == "brute_force" ||
			backend == "gpu_bruteforce_knn" ||
			backend == "bruteforce_gpu_scan";
	}

	bool canUseTreeKnn(const std::vector<DeviceQuery>& queries, const PointGpu::Options& options)
	{
		if (requestsBruteForceKnn(options))
			return false;

		const std::string backend = normalizeKnnBackend(options._knnBackend);
		if (!(backend.empty() || backend == "auto" || backend == "tree" || backend == "gpu_tree_knn"))
			return false;

		for (const DeviceQuery& query : queries)
		{
			if (query._type == static_cast<int>(PointGpu::QueryType::Knn) &&
				query._knnK > PointGpu::MaxTrackedKnnK)
			{
				return false;
			}
		}
		return true;
	}
}

PointGpu::KDTree::KDTree()
	: _state(std::make_unique<DeviceState>())
{
}

PointGpu::KDTree::~KDTree()
{
	if (_state)
		release();
}

void PointGpu::KDTree::release()
{
	releaseTree();
	cudaFree(_state->_points);
	cudaFree(_state->_indices);
	cudaFree(_state->_tempIndices);
	releaseQueryBuffers(*_state);
	*_state = DeviceState();
}

void PointGpu::KDTree::releaseTree()
{
	cudaFree(_state->_nodes);
	cudaFree(_state->_leftCounts);
	cudaFree(_state->_writeCursors);
	_state->_nodes = nullptr;
	_state->_leftCounts = nullptr;
	_state->_writeCursors = nullptr;
	_state->_nodeCapacity = 0;
	_state->_actualNodes = 0;
	_state->_actualLeaves = 0;
	_state->_memoryBytes = _state->_baseMemoryBytes;
}

bool PointGpu::KDTree::isAvailable(std::string* error)
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

int PointGpu::KDTree::deviceCount()
{
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess)
		return 0;
	return count;
}

PointGpu::BuildResult PointGpu::KDTree::build(const PointCloud& cloud, const SchemaConfig& schema, const Options& options)
{
	const std::string builder = options._builder.empty() ? "kdtree" : options._builder;
	const bool buildBIH = isBIHBuilder(builder);
	if (!isKDTreeBuilder(builder) && !buildBIH)
		throw std::runtime_error("KDTree evaluator supports --cuda-builder kdtree, kd_tree, kd, bih, interval_hierarchy, or binary_interval_hierarchy.");
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("KDTree/BIH currently supports up to 2^32 - 1 points.");

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
	_state->_device = device;
	_state->_cloud = &cloud;

	BuildResult result;
	result._device = device;
	result._builder = buildBIH ? "bih" : "kdtree";

	if (cloud.empty())
		return result;

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

	_state->_nodeCapacity = nodeCapacityForDepth(_state->_maxDepth);
	_state->_memoryBytes =
		_state->_baseMemoryBytes +
		sizeof(LinearNode) * _state->_nodeCapacity +
		sizeof(uint32_t) * _state->_nodeCapacity * 2;
	checkMemoryBudget(_state->_memoryBytes, options._memoryBudgetMb, buildBIH ? "BIH" : "KDTree");

	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_nodes), sizeof(LinearNode) * _state->_nodeCapacity));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_leftCounts), sizeof(uint32_t) * _state->_nodeCapacity));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_writeCursors), sizeof(uint32_t) * _state->_nodeCapacity));

	cudaEvent_t buildBegin = nullptr;
	cudaEvent_t buildEnd = nullptr;
	CudaHelper::startTimer(buildBegin, buildEnd);

	const dim3 pointBlocks(static_cast<unsigned int>(divUp(_state->_pointCount, ThreadsPerBlock)));
	initializeIndicesKernel<<<pointBlocks, ThreadsPerBlock>>>(_state->_indices, _state->_pointCount);
	CudaHelper::synchronize("initializeKdIndicesKernel");

	const dim3 nodeBlocks(static_cast<unsigned int>(divUp(_state->_nodeCapacity, ThreadsPerBlock)));
	initializeNodesKernel<<<nodeBlocks, ThreadsPerBlock>>>(_state->_nodes, _state->_nodeCapacity);
	CudaHelper::synchronize("initializeKdNodesKernel");

	// Schema per-level axisPolicy wins over the options default; falls back to LongestExtent.
	const PointGpu::KdAxisPolicy resolvedPolicy = axisPolicyFromSchema(schema);
	const int axisPolicy = static_cast<int>(resolvedPolicy);
	initializeRootKernel<<<1, 1>>>(
		_state->_nodes,
		_state->_pointCount,
		boundsMin.x,
		boundsMin.y,
		boundsMin.z,
		boundsMax.x,
		boundsMax.y,
		boundsMax.z,
		axisPolicy);
	CudaHelper::synchronize("initializeKdRootKernel");

	for (size_t depth = 0; depth < _state->_maxDepth; ++depth)
	{
		const size_t nodeStart = nodeStartForDepth(depth);
		const size_t levelNodeCount = nodeCountForDepth(depth);
		CudaHelper::checkError(cudaMemset(_state->_leftCounts + nodeStart, 0, sizeof(uint32_t) * levelNodeCount));

		countSplitsKernel<<<static_cast<unsigned int>(levelNodeCount), ThreadsPerBlock>>>(
			_state->_points,
			_state->_indices,
			_state->_nodes,
			nodeStart,
			levelNodeCount,
			static_cast<uint32_t>(_state->_leafCapacity),
			static_cast<uint32_t>(_state->_minSplit),
			_state->_leftCounts);
		CudaHelper::synchronize("countKdSplitsKernel");

		const dim3 levelBlocks(static_cast<unsigned int>(divUp(levelNodeCount, ThreadsPerBlock)));
		prepareSplitNodesKernel<<<levelBlocks, ThreadsPerBlock>>>(
			_state->_nodes,
			nodeStart,
			levelNodeCount,
			_state->_leftCounts,
			_state->_writeCursors,
			axisPolicy,
			static_cast<uint32_t>(depth));
		CudaHelper::synchronize("prepareKdSplitNodesKernel");

		CudaHelper::checkError(cudaMemcpy(
			_state->_tempIndices,
			_state->_indices,
			sizeof(uint32_t) * _state->_pointCount,
			cudaMemcpyDeviceToDevice));

		partitionIndicesKernel<<<static_cast<unsigned int>(levelNodeCount), ThreadsPerBlock>>>(
			_state->_points,
			_state->_indices,
			_state->_tempIndices,
			_state->_nodes,
			nodeStart,
			levelNodeCount,
			_state->_writeCursors);
		CudaHelper::synchronize("partitionKdIndicesKernel");

		std::swap(_state->_indices, _state->_tempIndices);

		if (buildBIH)
		{
			const size_t childStart = nodeStartForDepth(depth + 1);
			const size_t childLevelNodeCount = nodeCountForDepth(depth + 1);
			refitNodeBoundsKernel<<<static_cast<unsigned int>(childLevelNodeCount), ThreadsPerBlock>>>(
				_state->_points,
				_state->_indices,
				_state->_nodes,
				childStart,
				childLevelNodeCount);
			CudaHelper::synchronize("refitBihNodeBoundsKernel");
		}
	}

	result._gpuBuildTimeMs = CudaHelper::stopTimer(buildBegin, buildEnd);
	cudaEventDestroy(buildBegin);
	cudaEventDestroy(buildEnd);

	std::vector<LinearNode> hostNodes(_state->_nodeCapacity);
	CudaHelper::checkError(cudaMemcpy(hostNodes.data(), _state->_nodes, sizeof(LinearNode) * _state->_nodeCapacity, cudaMemcpyDeviceToHost));

	size_t maxLeafOccupancy = 0;
	size_t indexedPoints = 0;
	size_t deepestNode = 0;
	for (size_t nodeIndex = 0; nodeIndex < hostNodes.size(); ++nodeIndex)
	{
		const LinearNode& node = hostNodes[nodeIndex];
		if (node._pointCount == 0)
			continue;

		++_state->_actualNodes;
		const size_t depth = static_cast<size_t>(std::floor(std::log2(static_cast<double>(nodeIndex + 1))));
		deepestNode = std::max(deepestNode, depth);
		if (node._left < 0 || node._right < 0)
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
	return result;
}

PointGpu::QueryResult PointGpu::KDTree::query(const std::vector<Query>& queries, const Options& options) const
{
	if (!_state || _state->_actualNodes == 0 || queries.empty())
		return {};

	CudaHelper::checkError(cudaSetDevice(_state->_device));

	QueryResult result;
	result._samples.reserve(queries.size());
	result._knnPointIndices.resize(queries.size());
	result._knnDistancesSquared.resize(queries.size());
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
		const bool useTreeKnn = batchHasKnn && canUseTreeKnn(hostQueries, options);
		if (batchHasKnn)
			result._knnBackend = useTreeKnn ? "gpu_tree_knn" : "gpu_bruteforce_knn";

		CudaHelper::checkError(cudaMemcpy(_state->_queryBuffer, hostQueries.data(), sizeof(DeviceQuery) * currentBatch, cudaMemcpyHostToDevice));
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
		CudaHelper::synchronize("kdQueryKernel");
		if (batchHasKnn)
		{
			if (useTreeKnn)
			{
				ensureKnnBuffers(*_state, currentBatch);
				treeKnnKernel<<<static_cast<unsigned int>(currentBatch), ThreadsPerBlock>>>(
					_state->_points,
					_state->_indices,
					_state->_nodes,
					_state->_queryBuffer,
					currentBatch,
					clockRate,
					_state->_sampleBuffer,
					_state->_knnIndexBuffer,
					_state->_knnDistanceBuffer);
				CudaHelper::synchronize("kdTreeKnnQueryKernel");
			}
			else
			{
				PointGpu::bruteForceKnnKernel<<<static_cast<unsigned int>(currentBatch), ThreadsPerBlock, sizeof(float) * ThreadsPerBlock>>>(
					_state->_points,
					_state->_pointCount,
					_state->_queryBuffer,
					currentBatch,
					clockRate,
					_state->_sampleBuffer);
				CudaHelper::synchronize("kdBruteForceKnnQueryKernel");
			}
		}

		hostSamples.resize(currentBatch);
		CudaHelper::checkError(cudaMemcpy(hostSamples.data(), _state->_sampleBuffer, sizeof(DeviceQuerySample) * currentBatch, cudaMemcpyDeviceToHost));
		std::vector<uint32_t> hostKnnIndices;
		std::vector<float> hostKnnDistances;
		if (batchHasKnn && useTreeKnn)
		{
			hostKnnIndices.resize(currentBatch * PointGpu::MaxTrackedKnnK);
			hostKnnDistances.resize(currentBatch * PointGpu::MaxTrackedKnnK);
			CudaHelper::checkError(cudaMemcpy(
				hostKnnIndices.data(),
				_state->_knnIndexBuffer,
				sizeof(uint32_t) * hostKnnIndices.size(),
				cudaMemcpyDeviceToHost));
			CudaHelper::checkError(cudaMemcpy(
				hostKnnDistances.data(),
				_state->_knnDistanceBuffer,
				sizeof(float) * hostKnnDistances.size(),
				cudaMemcpyDeviceToHost));
		}

		for (size_t sampleIndex = 0; sampleIndex < hostSamples.size(); ++sampleIndex)
		{
			const DeviceQuerySample& sample = hostSamples[sampleIndex];
			QuerySample converted;
			converted._visitedNodes = static_cast<size_t>(sample._visitedNodes);
			converted._testedPoints = static_cast<size_t>(sample._testedPoints);
			converted._returnedPoints = static_cast<size_t>(sample._returnedPoints);
			converted._elapsedMs = sample._elapsedMs;
			result._samples.push_back(converted);

			const size_t globalQueryIndex = offset + sampleIndex;
			if (useTreeKnn && hostQueries[sampleIndex]._type == static_cast<int>(PointGpu::QueryType::Knn))
			{
				const size_t outputBase = sampleIndex * PointGpu::MaxTrackedKnnK;
				const size_t outputCount = std::min<size_t>(converted._returnedPoints, PointGpu::MaxTrackedKnnK);
				result._knnPointIndices[globalQueryIndex].reserve(outputCount);
				result._knnDistancesSquared[globalQueryIndex].reserve(outputCount);
				for (size_t i = 0; i < outputCount; ++i)
				{
					result._knnPointIndices[globalQueryIndex].push_back(hostKnnIndices[outputBase + i]);
					result._knnDistancesSquared[globalQueryIndex].push_back(hostKnnDistances[outputBase + i]);
				}
			}
		}
	}

	result._gpuQueryTimeMs = CudaHelper::stopTimer(queryBegin, queryEnd);
	cudaEventDestroy(queryBegin);
	cudaEventDestroy(queryEnd);

	result._metrics = summarizeGpuSamples(result._samples);
	return result;
}

bool PointGpu::KDTree::built() const
{
	return _state && _state->_actualNodes > 0;
}

size_t PointGpu::KDTree::pointCount() const
{
	return _state ? _state->_pointCount : 0;
}

size_t PointGpu::KDTree::nodeCount() const
{
	return _state ? _state->_actualNodes : 0;
}

size_t PointGpu::KDTree::leafCount() const
{
	return _state ? _state->_actualLeaves : 0;
}

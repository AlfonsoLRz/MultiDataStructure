#include "../../stdafx.h"
#include "QuadTree.h"

#include "../../CudaHelper.h"

namespace
{
	using PointGpu::DevicePoint;
	using PointGpu::DeviceQuery;
	using PointGpu::DeviceQuerySample;
	using PointGpu::LinearQuadTreeNode;

	constexpr int ThreadsPerBlock = 256;
	constexpr int QueryStackSize = 256;
	constexpr size_t MaxSupportedDepth = 12;
	constexpr uint32_t ChildCount = 4;

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
			throw std::runtime_error("QuadTree currently supports maxDepth <= 12.");
		return maxDepth;
	}

	size_t fullNodeCapacityForDepth(size_t maxDepth)
	{
		size_t capacity = 1;
		size_t levelWidth = 1;
		for (size_t depth = 0; depth < maxDepth; ++depth)
		{
			if (levelWidth > std::numeric_limits<size_t>::max() / ChildCount)
				return std::numeric_limits<size_t>::max();
			levelWidth *= ChildCount;
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
		const size_t estimatedInternal = std::max<size_t>(1, targetLeaves * ChildCount);
		const size_t estimatedCapacity = estimatedInternal > (std::numeric_limits<size_t>::max() - 1) / ChildCount
			? std::numeric_limits<size_t>::max()
			: 1 + estimatedInternal * ChildCount;
		const size_t capacity = std::min(fullCapacity, estimatedCapacity);
		if (capacity > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
			throw std::runtime_error("QuadTree currently supports up to 2^32 - 1 allocated nodes.");
		return std::max<size_t>(static_cast<size_t>(ChildCount + 1), capacity);
	}

	size_t estimateLevelScratchNodeCapacity(size_t pointCount, size_t leafCapacity, size_t nodeCapacity)
	{
		const size_t targetLeaves = std::max<size_t>(1, divUp(pointCount, std::max<size_t>(1, leafCapacity)));
		const size_t estimatedFrontier = targetLeaves > std::numeric_limits<size_t>::max() / ChildCount
			? std::numeric_limits<size_t>::max()
			: targetLeaves * ChildCount;
		return std::max<size_t>(static_cast<size_t>(ChildCount + 1), std::min(nodeCapacity, std::min(pointCount, estimatedFrontier)));
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

	__device__ uint32_t quadrantForPoint(const DevicePoint& point, const LinearQuadTreeNode& node)
	{
		const float midX = (node._minX + node._maxX) * 0.5f;
		const float midY = (node._minY + node._maxY) * 0.5f;
		return (point.x > midX ? 1u : 0u) |
			(point.y > midY ? 2u : 0u);
	}

	__device__ LinearQuadTreeNode makeChildNode(const LinearQuadTreeNode& parent, uint32_t child, uint32_t offset, uint32_t count, int parentIndex)
	{
		const float midX = (parent._minX + parent._maxX) * 0.5f;
		const float midY = (parent._minY + parent._maxY) * 0.5f;

		LinearQuadTreeNode node{};
		node._minX = (child & 1u) ? midX : parent._minX;
		node._maxX = (child & 1u) ? parent._maxX : midX;
		node._minY = (child & 2u) ? midY : parent._minY;
		node._maxY = (child & 2u) ? parent._maxY : midY;
		node._minZ = parent._minZ;
		node._maxZ = parent._maxZ;
		node._parent = parentIndex;
		node._childBase = -1;
		node._childMask = 0;
		node._pointOffset = offset;
		node._pointCount = count;
		node._flags = count > 0 ? 1u : 0u;
		node._depth = parent._depth + 1;
		return node;
	}

	__device__ bool rangeIntersectsNode(const LinearQuadTreeNode& node, const DeviceQuery& query)
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

	__device__ float distanceSquaredToNode(const LinearQuadTreeNode& node, const DeviceQuery& query)
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
		LinearQuadTreeNode* nodes,
		uint32_t* nodeCounter,
		size_t pointCount,
		float minX,
		float minY,
		float minZ,
		float maxX,
		float maxY,
		float maxZ)
	{
		LinearQuadTreeNode root{};
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
		nodes[0] = root;
		*nodeCounter = 1;
	}

	__global__ void countChildBucketsKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		LinearQuadTreeNode* nodes,
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
		LinearQuadTreeNode node = nodes[nodeIndex];
		if (node._pointCount == 0 || node._pointCount <= leafCapacity || node._pointCount < minSplit || node._depth >= maxDepth)
		{
			if (threadIdx.x == 0)
			{
				nodes[nodeIndex]._childBase = -1;
				nodes[nodeIndex]._childMask = 0;
				nodes[nodeIndex]._flags = node._pointCount > 0 ? 1u : 0u;
			}
			return;
		}

		uint32_t localCounts[ChildCount] = {};
		for (uint32_t offset = threadIdx.x; offset < node._pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node._pointOffset + offset];
			const uint32_t child = quadrantForPoint(points[pointIndex], node);
			++localCounts[child];
		}

		for (uint32_t child = 0; child < ChildCount; ++child)
		{
			if (localCounts[child] > 0)
				atomicAdd(&childCounts[localNode * ChildCount + child], localCounts[child]);
		}
	}

	__global__ void prepareChildrenKernel(
		LinearQuadTreeNode* nodes,
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
		LinearQuadTreeNode node = nodes[nodeIndex];
		if (node._pointCount == 0)
			return;

		uint32_t nonEmptyChildren = 0;
		uint32_t childMask = 0;
		for (uint32_t child = 0; child < ChildCount; ++child)
		{
			if (childCounts[localNode * ChildCount + child] > 0)
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

		const uint32_t childBase = atomicAdd(nodeCounter, ChildCount);
		if (static_cast<size_t>(childBase) + ChildCount > nodeCapacity)
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
		for (uint32_t child = 0; child < ChildCount; ++child)
		{
			const uint32_t count = childCounts[localNode * ChildCount + child];
			nodes[childBase + child] = makeChildNode(node, child, runningOffset, count, static_cast<int>(nodeIndex));
			writeCursors[localNode * ChildCount + child] = runningOffset;
			runningOffset += count;
		}
	}

	__global__ void partitionIndicesKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		uint32_t* outputIndices,
		const LinearQuadTreeNode* nodes,
		size_t levelStart,
		size_t levelCount,
		uint32_t* writeCursors)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x);
		if (localNode >= levelCount)
			return;

		const size_t nodeIndex = levelStart + localNode;
		const LinearQuadTreeNode node = nodes[nodeIndex];
		if (node._pointCount == 0 || node._childBase < 0)
			return;

		for (uint32_t offset = threadIdx.x; offset < node._pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node._pointOffset + offset];
			const uint32_t child = quadrantForPoint(points[pointIndex], node);
			const uint32_t writeOffset = atomicAdd(&writeCursors[localNode * ChildCount + child], 1u);
			outputIndices[writeOffset] = pointIndex;
		}
	}

	__global__ void queryKernel(
		const DevicePoint* points,
		const uint32_t* indices,
		const LinearQuadTreeNode* nodes,
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
			const LinearQuadTreeNode node = nodes[nodeIndex];
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

			for (uint32_t child = 0; child < ChildCount; ++child)
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
}

struct PointGpu::QuadTree::DeviceState
{
	DevicePoint*		_points = nullptr;
	uint32_t*			_indices = nullptr;
	uint32_t*			_tempIndices = nullptr;
	LinearQuadTreeNode*	_nodes = nullptr;
	uint32_t*			_childCounts = nullptr;
	uint32_t*			_writeCursors = nullptr;
	uint32_t*			_nodeCounter = nullptr;
	uint32_t*			_overflowFlag = nullptr;
	DeviceQuery*		_queryBuffer = nullptr;
	DeviceQuerySample*	_sampleBuffer = nullptr;
	size_t				_pointCount = 0;
	size_t				_nodeCapacity = 0;
	size_t				_allocatedNodes = 0;
	size_t				_actualNodes = 0;
	size_t				_actualLeaves = 0;
	size_t				_leafCapacity = 1;
	size_t				_minSplit = 2;
	size_t				_maxDepth = 0;
	size_t				_levelScratchNodeCapacity = 0;
	size_t				_queryCapacity = 0;
	size_t				_baseMemoryBytes = 0;
	size_t				_memoryBytes = 0;
	int					_device = 0;
	const PointCloud*	_cloud = nullptr;
	bool				_pointsReady = false;
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
			message << "QuadTree needs about " << (static_cast<double>(bytes) / (1024.0 * 1024.0))
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
}

PointGpu::QuadTree::QuadTree()
	: _state(std::make_unique<DeviceState>())
{
}

PointGpu::QuadTree::~QuadTree()
{
	if (_state)
		release();
}

void PointGpu::QuadTree::release()
{
	releaseTree();
	cudaFree(_state->_points);
	cudaFree(_state->_indices);
	cudaFree(_state->_tempIndices);
	releaseQueryBuffers(*_state);
	*_state = DeviceState();
}

void PointGpu::QuadTree::releaseTree()
{
	cudaFree(_state->_nodes);
	cudaFree(_state->_childCounts);
	cudaFree(_state->_writeCursors);
	cudaFree(_state->_nodeCounter);
	cudaFree(_state->_overflowFlag);
	_state->_nodes = nullptr;
	_state->_childCounts = nullptr;
	_state->_writeCursors = nullptr;
	_state->_nodeCounter = nullptr;
	_state->_overflowFlag = nullptr;
	_state->_nodeCapacity = 0;
	_state->_allocatedNodes = 0;
	_state->_actualNodes = 0;
	_state->_actualLeaves = 0;
	_state->_levelScratchNodeCapacity = 0;
	_state->_memoryBytes = _state->_baseMemoryBytes;
}

bool PointGpu::QuadTree::isAvailable(std::string* error)
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

int PointGpu::QuadTree::deviceCount()
{
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess)
		return 0;
	return count;
}

PointGpu::BuildResult PointGpu::QuadTree::build(const PointCloud& cloud, const SchemaConfig& schema, const Options& options)
{
	const std::string builder = options._builder.empty() ? "quadtree" : options._builder;
	if (builder != "quadtree" && builder != "quad_tree" && builder != "qt")
		throw std::runtime_error("QuadTree evaluator supports --cuda-builder quadtree, quad_tree, or qt.");
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("QuadTree currently supports up to 2^32 - 1 points.");

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
	result._builder = "quadtree";

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

	_state->_nodeCapacity = estimateNodeCapacity(_state->_pointCount, _state->_leafCapacity, _state->_maxDepth);
	_state->_levelScratchNodeCapacity = estimateLevelScratchNodeCapacity(_state->_pointCount, _state->_leafCapacity, _state->_nodeCapacity);
	_state->_memoryBytes =
		_state->_baseMemoryBytes +
		sizeof(LinearQuadTreeNode) * _state->_nodeCapacity +
		sizeof(uint32_t) * _state->_levelScratchNodeCapacity * ChildCount * 2 +
		sizeof(uint32_t) * 2;
	checkMemoryBudget(_state->_memoryBytes, options._memoryBudgetMb);

	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_nodes), sizeof(LinearQuadTreeNode) * _state->_nodeCapacity));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_childCounts), sizeof(uint32_t) * _state->_levelScratchNodeCapacity * ChildCount));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_writeCursors), sizeof(uint32_t) * _state->_levelScratchNodeCapacity * ChildCount));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_nodeCounter), sizeof(uint32_t)));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_overflowFlag), sizeof(uint32_t)));

	cudaEvent_t buildBegin = nullptr;
	cudaEvent_t buildEnd = nullptr;
	CudaHelper::startTimer(buildBegin, buildEnd);

	const dim3 pointBlocks(static_cast<unsigned int>(divUp(_state->_pointCount, ThreadsPerBlock)));
	initializeIndicesKernel<<<pointBlocks, ThreadsPerBlock>>>(_state->_indices, _state->_pointCount);
	CudaHelper::synchronize("initializeQuadTreeIndicesKernel");

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
	CudaHelper::synchronize("initializeQuadTreeRootKernel");
	CudaHelper::checkError(cudaMemset(_state->_overflowFlag, 0, sizeof(uint32_t)));

	size_t levelStart = 0;
	size_t levelCount = 1;
	uint32_t currentNodeCount = 1;
	for (size_t depth = 0; depth < _state->_maxDepth && levelCount > 0; ++depth)
	{
		if (levelCount > _state->_levelScratchNodeCapacity)
			throw std::runtime_error("QuadTree exceeded its level scratch budget. Increase leaf capacity or reduce max depth.");

		CudaHelper::checkError(cudaMemset(
			_state->_childCounts,
			0,
			sizeof(uint32_t) * levelCount * ChildCount));
		const dim3 prepareBlocks(static_cast<unsigned int>(divUp(levelCount, ThreadsPerBlock)));

		countChildBucketsKernel<<<static_cast<unsigned int>(levelCount), ThreadsPerBlock>>>(
			_state->_points,
			_state->_indices,
			_state->_nodes,
			levelStart,
			levelCount,
			static_cast<uint32_t>(_state->_leafCapacity),
			static_cast<uint32_t>(_state->_minSplit),
			static_cast<uint32_t>(_state->_maxDepth),
			_state->_childCounts);
		CudaHelper::synchronize("countQuadTreeChildBucketsKernel");

		const uint32_t previousNodeCount = currentNodeCount;

		prepareChildrenKernel<<<prepareBlocks, ThreadsPerBlock>>>(
			_state->_nodes,
			levelStart,
			levelCount,
			_state->_nodeCapacity,
			_state->_childCounts,
			_state->_writeCursors,
			_state->_nodeCounter,
			_state->_overflowFlag);
		CudaHelper::synchronize("prepareQuadTreeChildrenKernel");

		uint32_t overflow = 0;
		CudaHelper::checkError(cudaMemcpy(&overflow, _state->_overflowFlag, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		if (overflow != 0)
			throw std::runtime_error("QuadTree exceeded its allocated node budget. Increase leaf capacity or reduce max depth.");

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
			_state->_writeCursors);
		CudaHelper::synchronize("partitionQuadTreeIndicesKernel");
		std::swap(_state->_indices, _state->_tempIndices);

		uint32_t nextNodeCount = 0;
		CudaHelper::checkError(cudaMemcpy(&nextNodeCount, _state->_nodeCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		currentNodeCount = nextNodeCount;
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

	std::vector<LinearQuadTreeNode> hostNodes(_state->_allocatedNodes);
	CudaHelper::checkError(cudaMemcpy(hostNodes.data(), _state->_nodes, sizeof(LinearQuadTreeNode) * _state->_allocatedNodes, cudaMemcpyDeviceToHost));

	size_t maxLeafOccupancy = 0;
	size_t indexedPoints = 0;
	size_t deepestNode = 0;
	for (const LinearQuadTreeNode& node : hostNodes)
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
	return result;
}

PointGpu::QueryResult PointGpu::QuadTree::query(const std::vector<Query>& queries, const Options& options) const
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
		CudaHelper::synchronize("QuadTreeQueryKernel");
		if (batchHasKnn)
		{
			PointGpu::bruteForceKnnKernel<<<static_cast<unsigned int>(currentBatch), ThreadsPerBlock, sizeof(float) * ThreadsPerBlock>>>(
				_state->_points,
				_state->_pointCount,
				_state->_queryBuffer,
				currentBatch,
				clockRate,
				_state->_sampleBuffer);
			CudaHelper::synchronize("QuadTreeKnnQueryKernel");
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

bool PointGpu::QuadTree::built() const
{
	return _state && _state->_actualNodes > 0;
}

size_t PointGpu::QuadTree::pointCount() const
{
	return _state ? _state->_pointCount : 0;
}

size_t PointGpu::QuadTree::nodeCount() const
{
	return _state ? _state->_actualNodes : 0;
}

size_t PointGpu::QuadTree::leafCount() const
{
	return _state ? _state->_actualLeaves : 0;
}

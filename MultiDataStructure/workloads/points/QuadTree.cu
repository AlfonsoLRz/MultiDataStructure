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

	__device__ uint32_t quadrantForPoint(const DevicePoint& point, const LinearQuadTreeNode& node)
	{
		const float midX = (node.minX + node.maxX) * 0.5f;
		const float midY = (node.minY + node.maxY) * 0.5f;
		return (point.x > midX ? 1u : 0u) |
			(point.y > midY ? 2u : 0u);
	}

	__device__ LinearQuadTreeNode makeChildNode(const LinearQuadTreeNode& parent, uint32_t child, uint32_t offset, uint32_t count, int parentIndex)
	{
		const float midX = (parent.minX + parent.maxX) * 0.5f;
		const float midY = (parent.minY + parent.maxY) * 0.5f;

		LinearQuadTreeNode node{};
		node.minX = (child & 1u) ? midX : parent.minX;
		node.maxX = (child & 1u) ? parent.maxX : midX;
		node.minY = (child & 2u) ? midY : parent.minY;
		node.maxY = (child & 2u) ? parent.maxY : midY;
		node.minZ = parent.minZ;
		node.maxZ = parent.maxZ;
		node.parent = parentIndex;
		node.childBase = -1;
		node.childMask = 0;
		node.pointOffset = offset;
		node.pointCount = count;
		node.flags = count > 0 ? 1u : 0u;
		node.depth = parent.depth + 1;
		return node;
	}

	__device__ bool rangeIntersectsNode(const LinearQuadTreeNode& node, const DeviceQuery& query)
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

	__device__ float distanceSquaredToNode(const LinearQuadTreeNode& node, const DeviceQuery& query)
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

		uint32_t localCounts[ChildCount] = {};
		for (uint32_t offset = threadIdx.x; offset < node.pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node.pointOffset + offset];
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
		if (node.pointCount == 0)
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
			nodes[nodeIndex].childBase = -1;
			nodes[nodeIndex].childMask = 0;
			nodes[nodeIndex].flags = 1;
			return;
		}

		const uint32_t childBase = atomicAdd(nodeCounter, ChildCount);
		if (static_cast<size_t>(childBase) + ChildCount > nodeCapacity)
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
		if (node.pointCount == 0 || node.childBase < 0)
			return;

		for (uint32_t offset = threadIdx.x; offset < node.pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node.pointOffset + offset];
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
			const LinearQuadTreeNode node = nodes[nodeIndex];
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

			for (uint32_t child = 0; child < ChildCount; ++child)
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

struct PointGpu::QuadTree::DeviceState
{
	DevicePoint* points = nullptr;
	uint32_t* indices = nullptr;
	uint32_t* tempIndices = nullptr;
	LinearQuadTreeNode* nodes = nullptr;
	uint32_t* childCounts = nullptr;
	uint32_t* writeCursors = nullptr;
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
	cudaFree(_state->points);
	cudaFree(_state->indices);
	cudaFree(_state->tempIndices);
	releaseQueryBuffers(*_state);
	*_state = DeviceState();
}

void PointGpu::QuadTree::releaseTree()
{
	cudaFree(_state->nodes);
	cudaFree(_state->childCounts);
	cudaFree(_state->writeCursors);
	cudaFree(_state->nodeCounter);
	cudaFree(_state->overflowFlag);
	_state->nodes = nullptr;
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
	const std::string builder = options.builder.empty() ? "quadtree" : options.builder;
	if (builder != "quadtree" && builder != "quad_tree" && builder != "qt")
		throw std::runtime_error("QuadTree evaluator supports --cuda-builder quadtree, quad_tree, or qt.");
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("QuadTree currently supports up to 2^32 - 1 points.");

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
	result.builder = "quadtree";

	if (cloud.empty())
		return result;

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

	_state->nodeCapacity = estimateNodeCapacity(_state->pointCount, _state->leafCapacity, _state->maxDepth);
	_state->levelScratchNodeCapacity = estimateLevelScratchNodeCapacity(_state->pointCount, _state->leafCapacity, _state->nodeCapacity);
	_state->memoryBytes =
		_state->baseMemoryBytes +
		sizeof(LinearQuadTreeNode) * _state->nodeCapacity +
		sizeof(uint32_t) * _state->levelScratchNodeCapacity * ChildCount * 2 +
		sizeof(uint32_t) * 2;
	checkMemoryBudget(_state->memoryBytes, options.memoryBudgetMb);

	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->nodes), sizeof(LinearQuadTreeNode) * _state->nodeCapacity));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->childCounts), sizeof(uint32_t) * _state->levelScratchNodeCapacity * ChildCount));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->writeCursors), sizeof(uint32_t) * _state->levelScratchNodeCapacity * ChildCount));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->nodeCounter), sizeof(uint32_t)));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->overflowFlag), sizeof(uint32_t)));

	cudaEvent_t buildBegin = nullptr;
	cudaEvent_t buildEnd = nullptr;
	CudaHelper::startTimer(buildBegin, buildEnd);

	const dim3 pointBlocks(static_cast<unsigned int>(divUp(_state->pointCount, ThreadsPerBlock)));
	initializeIndicesKernel<<<pointBlocks, ThreadsPerBlock>>>(_state->indices, _state->pointCount);
	CudaHelper::synchronize("initializeQuadTreeIndicesKernel");

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
	CudaHelper::synchronize("initializeQuadTreeRootKernel");
	CudaHelper::checkError(cudaMemset(_state->overflowFlag, 0, sizeof(uint32_t)));

	size_t levelStart = 0;
	size_t levelCount = 1;
	uint32_t currentNodeCount = 1;
	for (size_t depth = 0; depth < _state->maxDepth && levelCount > 0; ++depth)
	{
		if (levelCount > _state->levelScratchNodeCapacity)
			throw std::runtime_error("QuadTree exceeded its level scratch budget. Increase leaf capacity or reduce max depth.");

		CudaHelper::checkError(cudaMemset(
			_state->childCounts,
			0,
			sizeof(uint32_t) * levelCount * ChildCount));
		const dim3 prepareBlocks(static_cast<unsigned int>(divUp(levelCount, ThreadsPerBlock)));

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
		CudaHelper::synchronize("countQuadTreeChildBucketsKernel");

		const uint32_t previousNodeCount = currentNodeCount;

		prepareChildrenKernel<<<prepareBlocks, ThreadsPerBlock>>>(
			_state->nodes,
			levelStart,
			levelCount,
			_state->nodeCapacity,
			_state->childCounts,
			_state->writeCursors,
			_state->nodeCounter,
			_state->overflowFlag);
		CudaHelper::synchronize("prepareQuadTreeChildrenKernel");

		uint32_t overflow = 0;
		CudaHelper::checkError(cudaMemcpy(&overflow, _state->overflowFlag, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		if (overflow != 0)
			throw std::runtime_error("QuadTree exceeded its allocated node budget. Increase leaf capacity or reduce max depth.");

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
		CudaHelper::synchronize("partitionQuadTreeIndicesKernel");
		std::swap(_state->indices, _state->tempIndices);

		uint32_t nextNodeCount = 0;
		CudaHelper::checkError(cudaMemcpy(&nextNodeCount, _state->nodeCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		currentNodeCount = nextNodeCount;
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

	std::vector<LinearQuadTreeNode> hostNodes(_state->allocatedNodes);
	CudaHelper::checkError(cudaMemcpy(hostNodes.data(), _state->nodes, sizeof(LinearQuadTreeNode) * _state->allocatedNodes, cudaMemcpyDeviceToHost));

	size_t maxLeafOccupancy = 0;
	size_t indexedPoints = 0;
	size_t deepestNode = 0;
	for (const LinearQuadTreeNode& node : hostNodes)
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

PointGpu::QueryResult PointGpu::QuadTree::query(const std::vector<Query>& queries, const Options& options) const
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
		CudaHelper::synchronize("QuadTreeQueryKernel");
		if (batchHasKnn)
		{
			PointGpu::bruteForceKnnKernel<<<static_cast<unsigned int>(currentBatch), ThreadsPerBlock, sizeof(float) * ThreadsPerBlock>>>(
				_state->points,
				_state->pointCount,
				_state->queryBuffer,
				currentBatch,
				clockRate,
				_state->sampleBuffer);
			CudaHelper::synchronize("QuadTreeKnnQueryKernel");
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

bool PointGpu::QuadTree::built() const
{
	return _state && _state->actualNodes > 0;
}

size_t PointGpu::QuadTree::pointCount() const
{
	return _state ? _state->pointCount : 0;
}

size_t PointGpu::QuadTree::nodeCount() const
{
	return _state ? _state->actualNodes : 0;
}

size_t PointGpu::QuadTree::leafCount() const
{
	return _state ? _state->actualLeaves : 0;
}

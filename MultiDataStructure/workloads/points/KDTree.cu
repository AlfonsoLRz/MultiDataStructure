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
			maxDepth = 12;
		if (maxDepth > MaxSupportedDepth)
			throw std::runtime_error("KDTree currently supports maxDepth <= 22.");
		return maxDepth;
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

	__device__ int splitAxisForNode(const LinearNode& node)
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

	__device__ float splitPlaneForNode(const LinearNode& node, int axis)
	{
		if (axis == 0)
			return (node.minX + node.maxX) * 0.5f;
		if (axis == 1)
			return (node.minY + node.maxY) * 0.5f;
		return (node.minZ + node.maxZ) * 0.5f;
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

	__device__ float distanceSquaredToNode(const LinearNode& node, const DeviceQuery& query)
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

	__global__ void initializeNodesKernel(LinearNode* nodes, size_t nodeCount)
	{
		const size_t nodeIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (nodeIndex >= nodeCount)
			return;

		LinearNode node{};
		node.minX = 3.402823466e+38F;
		node.minY = 3.402823466e+38F;
		node.minZ = 3.402823466e+38F;
		node.maxX = -3.402823466e+38F;
		node.maxY = -3.402823466e+38F;
		node.maxZ = -3.402823466e+38F;
		node.left = -1;
		node.right = -1;
		node.parent = -1;
		node.pointOffset = 0;
		node.pointCount = 0;
		node.flags = 0;
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
		float maxZ)
	{
		LinearNode root{};
		root.minX = minX;
		root.minY = minY;
		root.minZ = minZ;
		root.maxX = maxX;
		root.maxY = maxY;
		root.maxZ = maxZ;
		root.left = -1;
		root.right = -1;
		root.parent = -1;
		root.pointOffset = 0;
		root.pointCount = static_cast<uint32_t>(pointCount);
		root.flags = 1;
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
		if (node.pointCount == 0 || node.pointCount <= leafCapacity || node.pointCount < minSplit)
		{
			if (threadIdx.x == 0)
			{
				nodes[nodeIndex].left = -1;
				nodes[nodeIndex].right = -1;
				nodes[nodeIndex].flags = node.pointCount > 0 ? 1u : 0u;
			}
			return;
		}

		uint32_t localLeft = 0;
		for (uint32_t offset = threadIdx.x; offset < node.pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node.pointOffset + offset];
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
		uint32_t* writeCursors)
	{
		const size_t localNode = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (localNode >= nodeCount)
			return;

		const size_t nodeIndex = nodeStart + localNode;
		LinearNode node = nodes[nodeIndex];
		if (node.pointCount == 0)
			return;

		const uint32_t leftCount = leftCounts[nodeIndex];
		const uint32_t rightCount = node.pointCount - leftCount;
		if (leftCount == 0 || rightCount == 0)
		{
			nodes[nodeIndex].left = -1;
			nodes[nodeIndex].right = -1;
			nodes[nodeIndex].flags = 1;
			return;
		}

		const int axis = splitAxisForNode(node);
		const float plane = splitPlaneForNode(node, axis);
		const int leftIndex = static_cast<int>(nodeIndex * 2 + 1);
		const int rightIndex = leftIndex + 1;
		nodes[nodeIndex].left = leftIndex;
		nodes[nodeIndex].right = rightIndex;
		nodes[nodeIndex].flags = 0;

		LinearNode left = node;
		left.left = -1;
		left.right = -1;
		left.parent = static_cast<int>(nodeIndex);
		left.pointOffset = node.pointOffset;
		left.pointCount = leftCount;
		left.flags = 1;

		LinearNode right = node;
		right.left = -1;
		right.right = -1;
		right.parent = static_cast<int>(nodeIndex);
		right.pointOffset = node.pointOffset + leftCount;
		right.pointCount = rightCount;
		right.flags = 1;

		if (axis == 0)
		{
			left.maxX = plane;
			right.minX = plane;
		}
		else if (axis == 1)
		{
			left.maxY = plane;
			right.minY = plane;
		}
		else
		{
			left.maxZ = plane;
			right.minZ = plane;
		}

		nodes[leftIndex] = left;
		nodes[rightIndex] = right;
		writeCursors[leftIndex] = left.pointOffset;
		writeCursors[rightIndex] = right.pointOffset;
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
		if (node.pointCount == 0 || node.left < 0 || node.right < 0)
			return;

		for (uint32_t offset = threadIdx.x; offset < node.pointCount; offset += blockDim.x)
		{
			const uint32_t pointIndex = indices[node.pointOffset + offset];
			const int childIndex = pointGoesLeft(points[pointIndex], node) ? node.left : node.right;
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
		const LinearNode* nodes,
		const DeviceQuery* queries,
		size_t queryCount,
		float clockRateKHz,
		DeviceQuerySample* samples)
	{
		const size_t queryIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (queryIndex >= queryCount)
			return;

		const unsigned long long begin = clock64();
		const DeviceQuery query = queries[queryIndex];
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
			if (node.left < 0 || node.right < 0)
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

			if (stackSize + 2 <= QueryStackSize)
			{
				stack[stackSize++] = node.left;
				stack[stackSize++] = node.right;
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

struct PointGpu::KDTree::DeviceState
{
	DevicePoint* points = nullptr;
	uint32_t* indices = nullptr;
	uint32_t* tempIndices = nullptr;
	LinearNode* nodes = nullptr;
	uint32_t* leftCounts = nullptr;
	uint32_t* writeCursors = nullptr;
	size_t pointCount = 0;
	size_t nodeCapacity = 0;
	size_t actualNodes = 0;
	size_t actualLeaves = 0;
	size_t leafCapacity = 1;
	size_t minSplit = 2;
	size_t maxDepth = 0;
	size_t baseMemoryBytes = 0;
	size_t memoryBytes = 0;
	int device = 0;
	const PointCloud* cloud = nullptr;
	bool pointsReady = false;
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
	cudaFree(_state->points);
	cudaFree(_state->indices);
	cudaFree(_state->tempIndices);
	*_state = DeviceState();
}

void PointGpu::KDTree::releaseTree()
{
	cudaFree(_state->nodes);
	cudaFree(_state->leftCounts);
	cudaFree(_state->writeCursors);
	_state->nodes = nullptr;
	_state->leftCounts = nullptr;
	_state->writeCursors = nullptr;
	_state->nodeCapacity = 0;
	_state->actualNodes = 0;
	_state->actualLeaves = 0;
	_state->memoryBytes = _state->baseMemoryBytes;
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
	const std::string builder = options.builder.empty() ? "kdtree" : options.builder;
	const bool buildBIH = isBIHBuilder(builder);
	if (!isKDTreeBuilder(builder) && !buildBIH)
		throw std::runtime_error("KDTree evaluator supports --cuda-builder kdtree, kd_tree, kd, bih, interval_hierarchy, or binary_interval_hierarchy.");
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("KDTree/BIH currently supports up to 2^32 - 1 points.");

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
	result.builder = buildBIH ? "bih" : "kdtree";

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

	_state->nodeCapacity = nodeCapacityForDepth(_state->maxDepth);
	_state->memoryBytes =
		_state->baseMemoryBytes +
		sizeof(LinearNode) * _state->nodeCapacity +
		sizeof(uint32_t) * _state->nodeCapacity * 2;
	checkMemoryBudget(_state->memoryBytes, options.memoryBudgetMb, buildBIH ? "BIH" : "KDTree");

	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->nodes), sizeof(LinearNode) * _state->nodeCapacity));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->leftCounts), sizeof(uint32_t) * _state->nodeCapacity));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->writeCursors), sizeof(uint32_t) * _state->nodeCapacity));

	cudaEvent_t buildBegin = nullptr;
	cudaEvent_t buildEnd = nullptr;
	CudaHelper::startTimer(buildBegin, buildEnd);

	const dim3 pointBlocks(static_cast<unsigned int>(divUp(_state->pointCount, ThreadsPerBlock)));
	initializeIndicesKernel<<<pointBlocks, ThreadsPerBlock>>>(_state->indices, _state->pointCount);
	CudaHelper::synchronize("initializeKdIndicesKernel");

	const dim3 nodeBlocks(static_cast<unsigned int>(divUp(_state->nodeCapacity, ThreadsPerBlock)));
	initializeNodesKernel<<<nodeBlocks, ThreadsPerBlock>>>(_state->nodes, _state->nodeCapacity);
	CudaHelper::synchronize("initializeKdNodesKernel");

	initializeRootKernel<<<1, 1>>>(
		_state->nodes,
		_state->pointCount,
		boundsMin.x,
		boundsMin.y,
		boundsMin.z,
		boundsMax.x,
		boundsMax.y,
		boundsMax.z);
	CudaHelper::synchronize("initializeKdRootKernel");

	for (size_t depth = 0; depth < _state->maxDepth; ++depth)
	{
		const size_t nodeStart = nodeStartForDepth(depth);
		const size_t levelNodeCount = nodeCountForDepth(depth);
		CudaHelper::checkError(cudaMemset(_state->leftCounts + nodeStart, 0, sizeof(uint32_t) * levelNodeCount));

		countSplitsKernel<<<static_cast<unsigned int>(levelNodeCount), ThreadsPerBlock>>>(
			_state->points,
			_state->indices,
			_state->nodes,
			nodeStart,
			levelNodeCount,
			static_cast<uint32_t>(_state->leafCapacity),
			static_cast<uint32_t>(_state->minSplit),
			_state->leftCounts);
		CudaHelper::synchronize("countKdSplitsKernel");

		const dim3 levelBlocks(static_cast<unsigned int>(divUp(levelNodeCount, ThreadsPerBlock)));
		prepareSplitNodesKernel<<<levelBlocks, ThreadsPerBlock>>>(
			_state->nodes,
			nodeStart,
			levelNodeCount,
			_state->leftCounts,
			_state->writeCursors);
		CudaHelper::synchronize("prepareKdSplitNodesKernel");

		CudaHelper::checkError(cudaMemcpy(
			_state->tempIndices,
			_state->indices,
			sizeof(uint32_t) * _state->pointCount,
			cudaMemcpyDeviceToDevice));

		partitionIndicesKernel<<<static_cast<unsigned int>(levelNodeCount), ThreadsPerBlock>>>(
			_state->points,
			_state->indices,
			_state->tempIndices,
			_state->nodes,
			nodeStart,
			levelNodeCount,
			_state->writeCursors);
		CudaHelper::synchronize("partitionKdIndicesKernel");

		std::swap(_state->indices, _state->tempIndices);

		if (buildBIH)
		{
			const size_t childStart = nodeStartForDepth(depth + 1);
			const size_t childLevelNodeCount = nodeCountForDepth(depth + 1);
			refitNodeBoundsKernel<<<static_cast<unsigned int>(childLevelNodeCount), ThreadsPerBlock>>>(
				_state->points,
				_state->indices,
				_state->nodes,
				childStart,
				childLevelNodeCount);
			CudaHelper::synchronize("refitBihNodeBoundsKernel");
		}
	}

	result.gpuBuildTimeMs = CudaHelper::stopTimer(buildBegin, buildEnd);
	cudaEventDestroy(buildBegin);
	cudaEventDestroy(buildEnd);

	std::vector<LinearNode> hostNodes(_state->nodeCapacity);
	CudaHelper::checkError(cudaMemcpy(hostNodes.data(), _state->nodes, sizeof(LinearNode) * _state->nodeCapacity, cudaMemcpyDeviceToHost));

	size_t maxLeafOccupancy = 0;
	size_t indexedPoints = 0;
	size_t deepestNode = 0;
	for (size_t nodeIndex = 0; nodeIndex < hostNodes.size(); ++nodeIndex)
	{
		const LinearNode& node = hostNodes[nodeIndex];
		if (node.pointCount == 0)
			continue;

		++_state->actualNodes;
		const size_t depth = static_cast<size_t>(std::floor(std::log2(static_cast<double>(nodeIndex + 1))));
		deepestNode = std::max(deepestNode, depth);
		if (node.left < 0 || node.right < 0)
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

PointGpu::QueryResult PointGpu::KDTree::query(const std::vector<Query>& queries, const Options& options) const
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
		else
			++result.rangeQueries;
	}

	const size_t batchSize = options.queryBatchSize > 0
		? std::max<size_t>(1, options.queryBatchSize)
		: queries.size();
	const float clockRate = deviceClockRateKHz(_state->device);

	DeviceQuery* deviceQueries = nullptr;
	DeviceQuerySample* deviceSamples = nullptr;
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&deviceQueries), sizeof(DeviceQuery) * batchSize));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&deviceSamples), sizeof(DeviceQuerySample) * batchSize));

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

		CudaHelper::checkError(cudaMemcpy(deviceQueries, hostQueries.data(), sizeof(DeviceQuery) * currentBatch, cudaMemcpyHostToDevice));
		const dim3 queryBlocks(static_cast<unsigned int>(divUp(currentBatch, ThreadsPerBlock)));
		queryKernel<<<queryBlocks, ThreadsPerBlock>>>(
			_state->points,
			_state->indices,
			_state->nodes,
			deviceQueries,
			currentBatch,
			clockRate,
			deviceSamples);
		CudaHelper::synchronize("kdQueryKernel");

		hostSamples.resize(currentBatch);
		CudaHelper::checkError(cudaMemcpy(hostSamples.data(), deviceSamples, sizeof(DeviceQuerySample) * currentBatch, cudaMemcpyDeviceToHost));
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
	cudaFree(deviceQueries);
	cudaFree(deviceSamples);

	result.metrics = summarizeGpuSamples(result.samples);
	return result;
}

bool PointGpu::KDTree::built() const
{
	return _state && _state->actualNodes > 0;
}

size_t PointGpu::KDTree::pointCount() const
{
	return _state ? _state->pointCount : 0;
}

size_t PointGpu::KDTree::nodeCount() const
{
	return _state ? _state->actualNodes : 0;
}

size_t PointGpu::KDTree::leafCount() const
{
	return _state ? _state->actualLeaves : 0;
}

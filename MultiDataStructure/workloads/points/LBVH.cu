#include "../../stdafx.h"
#include "LBVH.h"

#include "../../CudaHelper.h"

#include <cub/cub.cuh>

namespace
{
	using PointGpu::DevicePoint;
	using PointGpu::DeviceQuery;
	using PointGpu::DeviceQuerySample;
	using PointGpu::LinearNode;

	constexpr int ThreadsPerBlock = 256;
	constexpr int QueryStackSize = 128;

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

	size_t lbvhMaxDepth(size_t leafCount)
	{
		size_t depth = 0;
		size_t width = 1;
		while (width < leafCount)
		{
			width <<= 1;
			++depth;
		}
		return depth;
	}

	std::vector<DevicePoint> copyPoints(const PointCloud& cloud)
	{
		std::vector<DevicePoint> points;
		points.reserve(cloud.size());
		for (const PointPrimitive& point : cloud.points())
			points.push_back(DevicePoint{ point.position.x, point.position.y, point.position.z, 0.0f });
		return points;
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

	__device__ int commonPrefix(const uint64_t* keys, int count, int left, int right)
	{
		if (right < 0 || right >= count)
			return -1;

		const uint64_t leftKey = keys[left];
		const uint64_t rightKey = keys[right];
		if (leftKey == rightKey)
			return 64 + __clzll(static_cast<unsigned long long>(left ^ right));

		return __clzll(static_cast<unsigned long long>(leftKey ^ rightKey));
	}

	__device__ int findSplit(const uint64_t* keys, int count, int first, int last)
	{
		const int common = commonPrefix(keys, count, first, last);
		int split = first;
		int step = last - first;

		do
		{
			step = (step + 1) >> 1;
			const int candidate = split + step;
			if (candidate < last)
			{
				const int prefix = commonPrefix(keys, count, first, candidate);
				if (prefix > common)
					split = candidate;
			}
		} while (step > 1);

		return split;
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

	__global__ void initializePointKeysKernel(
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

	__global__ void initializeLeafKeysKernel(
		const uint64_t* sortedKeys,
		size_t pointCount,
		size_t leafCapacity,
		size_t leafCount,
		uint64_t* leafKeys)
	{
		const size_t leafIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (leafIndex >= leafCount)
			return;

		const size_t pointOffset = min(leafIndex * leafCapacity, pointCount - 1);
		leafKeys[leafIndex] = sortedKeys[pointOffset];
	}

	__global__ void initializeNodesKernel(LinearNode* nodes, int nodeCount)
	{
		const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;
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
		nodes[nodeIndex] = node;
	}

	__global__ void initializeLeavesKernel(
		LinearNode* nodes,
		size_t pointCount,
		size_t leafCapacity,
		int leafCount)
	{
		const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x;
		if (leafIndex >= leafCount)
			return;

		const int internalCount = max(leafCount - 1, 0);
		const int nodeIndex = internalCount + leafIndex;
		const size_t pointOffset = static_cast<size_t>(leafIndex) * leafCapacity;
		const size_t remaining = pointOffset < pointCount ? pointCount - pointOffset : 0;
		nodes[nodeIndex].pointOffset = static_cast<uint32_t>(pointOffset);
		nodes[nodeIndex].pointCount = static_cast<uint32_t>(min(leafCapacity, remaining));
		nodes[nodeIndex].flags = 1;
	}

	__global__ void buildRadixTreeKernel(const uint64_t* leafKeys, int leafCount, LinearNode* nodes)
	{
		const int internalIndex = blockDim.x * blockIdx.x + threadIdx.x;
		const int internalCount = leafCount - 1;
		if (internalIndex >= internalCount)
			return;

		const int nextPrefix = commonPrefix(leafKeys, leafCount, internalIndex, internalIndex + 1);
		const int previousPrefix = commonPrefix(leafKeys, leafCount, internalIndex, internalIndex - 1);
		const int direction = nextPrefix >= previousPrefix ? 1 : -1;
		const int minPrefix = commonPrefix(leafKeys, leafCount, internalIndex, internalIndex - direction);

		int maxLength = 2;
		while (true)
		{
			const int candidate = internalIndex + maxLength * direction;
			if (candidate < 0 || candidate >= leafCount)
				break;
			if (commonPrefix(leafKeys, leafCount, internalIndex, candidate) <= minPrefix)
				break;
			maxLength <<= 1;
		}

		int length = 0;
		for (int step = maxLength >> 1; step >= 1; step >>= 1)
		{
			const int candidateLength = length + step;
			const int candidate = internalIndex + candidateLength * direction;
			if (candidate >= 0 && candidate < leafCount &&
				commonPrefix(leafKeys, leafCount, internalIndex, candidate) > minPrefix)
			{
				length = candidateLength;
			}
		}

		const int other = internalIndex + length * direction;
		const int first = min(internalIndex, other);
		const int last = max(internalIndex, other);
		const int split = findSplit(leafKeys, leafCount, first, last);

		const int leftChild = split == first ? internalCount + split : split;
		const int rightChild = split + 1 == last ? internalCount + split + 1 : split + 1;

		nodes[internalIndex].left = leftChild;
		nodes[internalIndex].right = rightChild;
		nodes[internalIndex].flags = 0;
		nodes[leftChild].parent = internalIndex;
		nodes[rightChild].parent = internalIndex;
	}

	__global__ void computeLeafBoundsKernel(
		const DevicePoint* points,
		const uint32_t* sortedIndices,
		LinearNode* nodes,
		int leafCount)
	{
		const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x;
		if (leafIndex >= leafCount)
			return;

		const int internalCount = max(leafCount - 1, 0);
		const int nodeIndex = internalCount + leafIndex;
		LinearNode& node = nodes[nodeIndex];

		float minX = 3.402823466e+38F;
		float minY = 3.402823466e+38F;
		float minZ = 3.402823466e+38F;
		float maxX = -3.402823466e+38F;
		float maxY = -3.402823466e+38F;
		float maxZ = -3.402823466e+38F;

		for (uint32_t i = 0; i < node.pointCount; ++i)
		{
			const uint32_t pointIndex = sortedIndices[node.pointOffset + i];
			const DevicePoint point = points[pointIndex];
			minX = fminf(minX, point.x);
			minY = fminf(minY, point.y);
			minZ = fminf(minZ, point.z);
			maxX = fmaxf(maxX, point.x);
			maxY = fmaxf(maxY, point.y);
			maxZ = fmaxf(maxZ, point.z);
		}

		node.minX = minX;
		node.minY = minY;
		node.minZ = minZ;
		node.maxX = maxX;
		node.maxY = maxY;
		node.maxZ = maxZ;
	}

	__global__ void computeInternalBoundsKernel(LinearNode* nodes, int* counters, int leafCount)
	{
		const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x;
		if (leafIndex >= leafCount)
			return;

		const int internalCount = max(leafCount - 1, 0);
		int current = internalCount + leafIndex;

		while (true)
		{
			const int parent = nodes[current].parent;
			if (parent < 0)
				return;

			__threadfence();
			const int previous = atomicAdd(&counters[parent], 1);
			if (previous == 0)
				return;

			const int left = nodes[parent].left;
			const int right = nodes[parent].right;
			const LinearNode leftNode = nodes[left];
			const LinearNode rightNode = nodes[right];

			nodes[parent].minX = fminf(leftNode.minX, rightNode.minX);
			nodes[parent].minY = fminf(leftNode.minY, rightNode.minY);
			nodes[parent].minZ = fminf(leftNode.minZ, rightNode.minZ);
			nodes[parent].maxX = fmaxf(leftNode.maxX, rightNode.maxX);
			nodes[parent].maxY = fmaxf(leftNode.maxY, rightNode.maxY);
			nodes[parent].maxZ = fmaxf(leftNode.maxZ, rightNode.maxZ);
			current = parent;
		}
	}

	__global__ void queryKernel(
		const DevicePoint* points,
		const uint32_t* sortedIndices,
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
			++visited;

			const bool intersects = query.type == 2
				? distanceSquaredToNode(node, query) <= query.radius * query.radius
				: rangeIntersectsNode(node, query);
			if (!intersects)
				continue;

			if (node.flags == 1)
			{
				for (uint32_t i = 0; i < node.pointCount; ++i)
				{
					++tested;
					const uint32_t pointIndex = sortedIndices[node.pointOffset + i];
					const DevicePoint point = points[pointIndex];
					const bool inside = query.type == 2
						? pointInsideRadius(point, query)
						: pointInsideRange(point, query);
					if (inside)
						++returned;
				}
				continue;
			}

			if (node.left >= 0 && stackSize < QueryStackSize)
				stack[stackSize++] = node.left;
			if (node.right >= 0 && stackSize < QueryStackSize)
				stack[stackSize++] = node.right;
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

struct PointGpu::LBVH::DeviceState
{
	DevicePoint* points = nullptr;
	uint64_t* keys = nullptr;
	uint64_t* sortedKeys = nullptr;
	uint64_t* leafKeys = nullptr;
	uint32_t* indices = nullptr;
	uint32_t* sortedIndices = nullptr;
	LinearNode* nodes = nullptr;
	int* boundsCounters = nullptr;
	void* sortTemporary = nullptr;
	size_t sortTemporaryBytes = 0;
	size_t pointCount = 0;
	size_t leafCount = 0;
	size_t nodeCount = 0;
	size_t leafCapacity = 1;
	size_t baseMemoryBytes = 0;
	size_t memoryBytes = 0;
	int device = 0;
	std::string builder = "lbvh";
	const PointCloud* cloud = nullptr;
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
			message << "LBVH needs about " << (static_cast<double>(bytes) / (1024.0 * 1024.0))
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

PointGpu::LBVH::LBVH()
	: _state(std::make_unique<DeviceState>())
{
}

PointGpu::LBVH::~LBVH()
{
	if (_state)
		release();
}

void PointGpu::LBVH::release()
{
	releaseTree();
	cudaFree(_state->points);
	cudaFree(_state->keys);
	cudaFree(_state->sortedKeys);
	cudaFree(_state->indices);
	cudaFree(_state->sortedIndices);
	cudaFree(_state->sortTemporary);
	*_state = DeviceState();
}

void PointGpu::LBVH::releaseTree()
{
	cudaFree(_state->leafKeys);
	cudaFree(_state->nodes);
	cudaFree(_state->boundsCounters);
	_state->leafKeys = nullptr;
	_state->nodes = nullptr;
	_state->boundsCounters = nullptr;
	_state->leafCount = 0;
	_state->nodeCount = 0;
	_state->leafCapacity = 1;
	_state->memoryBytes = _state->baseMemoryBytes;
}

bool PointGpu::LBVH::isAvailable(std::string* error)
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

int PointGpu::LBVH::deviceCount()
{
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess)
		return 0;
	return count;
}

PointGpu::BuildResult PointGpu::LBVH::build(const PointCloud& cloud, const SchemaConfig& schema, const Options& options)
{
	if (options.builder != "lbvh")
		throw std::runtime_error("LBVH evaluator currently supports --cuda-builder lbvh only; mixed CUDA build is a later milestone.");
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("LBVH currently supports up to 2^32 - 1 points.");

	std::string availabilityError;
	if (!isAvailable(&availabilityError))
		throw std::runtime_error("CUDA point evaluator is unavailable: " + availabilityError);

	const int count = deviceCount();
	const int device = options.device >= 0 ? std::min(options.device, count - 1) : 0;
	CudaHelper::checkError(cudaSetDevice(device));

	const bool canReuseSortedPoints =
		_state->sortedReady &&
		_state->cloud == &cloud &&
		_state->pointCount == cloud.size() &&
		_state->device == device;

	if (!canReuseSortedPoints)
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
	_state->leafCount = cloud.empty() ? 0 : divUp(cloud.size(), _state->leafCapacity);
	_state->nodeCount = _state->leafCount == 0 ? 0 : 2 * _state->leafCount - 1;
	_state->device = device;
	_state->builder = options.builder;
	_state->cloud = &cloud;

	BuildResult result;
	result.device = device;
	result.builder = options.builder;

	if (cloud.empty())
		return result;

	const glm::vec3 boundsMin = cloud.bounds().min();
	const glm::vec3 extent = glm::max(cloud.coordinateRange(), glm::vec3(0.0f));

	if (!canReuseSortedPoints)
	{
		const std::vector<DevicePoint> hostPoints = copyPoints(cloud);

		cudaEvent_t uploadBegin = nullptr;
		cudaEvent_t uploadEnd = nullptr;
		CudaHelper::startTimer(uploadBegin, uploadEnd);
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->points), sizeof(DevicePoint) * _state->pointCount));
		CudaHelper::checkError(cudaMemcpy(_state->points, hostPoints.data(), sizeof(DevicePoint) * _state->pointCount, cudaMemcpyHostToDevice));
		result.uploadTimeMs = CudaHelper::stopTimer(uploadBegin, uploadEnd);
		cudaEventDestroy(uploadBegin);
		cudaEventDestroy(uploadEnd);

		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->keys), sizeof(uint64_t) * _state->pointCount));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->sortedKeys), sizeof(uint64_t) * _state->pointCount));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->indices), sizeof(uint32_t) * _state->pointCount));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->sortedIndices), sizeof(uint32_t) * _state->pointCount));

		CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(
			nullptr,
			_state->sortTemporaryBytes,
			_state->keys,
			_state->sortedKeys,
			_state->indices,
			_state->sortedIndices,
			static_cast<int>(_state->pointCount)));
		CudaHelper::checkError(cudaMalloc(&_state->sortTemporary, _state->sortTemporaryBytes));

		_state->baseMemoryBytes =
			sizeof(DevicePoint) * _state->pointCount +
			sizeof(uint64_t) * _state->pointCount * 2 +
			sizeof(uint32_t) * _state->pointCount * 2 +
			_state->sortTemporaryBytes;
	}

	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->leafKeys), sizeof(uint64_t) * _state->leafCount));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->nodes), sizeof(LinearNode) * _state->nodeCount));
	if (_state->leafCount > 1)
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->boundsCounters), sizeof(int) * (_state->leafCount - 1)));

	_state->memoryBytes =
		_state->baseMemoryBytes +
		sizeof(uint64_t) * _state->leafCount +
		sizeof(LinearNode) * _state->nodeCount +
		sizeof(int) * (_state->leafCount > 1 ? _state->leafCount - 1 : 0);
	checkMemoryBudget(_state->memoryBytes, options.memoryBudgetMb);

	cudaEvent_t buildBegin = nullptr;
	cudaEvent_t buildEnd = nullptr;
	CudaHelper::startTimer(buildBegin, buildEnd);

	if (!canReuseSortedPoints)
	{
		const dim3 pointBlocks(static_cast<unsigned int>(divUp(_state->pointCount, ThreadsPerBlock)));
		initializePointKeysKernel<<<pointBlocks, ThreadsPerBlock>>>(
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
		CudaHelper::synchronize("initializePointKeysKernel");

		CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(
			_state->sortTemporary,
			_state->sortTemporaryBytes,
			_state->keys,
			_state->sortedKeys,
			_state->indices,
			_state->sortedIndices,
			static_cast<int>(_state->pointCount)));
		CudaHelper::synchronize("cub::DeviceRadixSort::SortPairs");
		_state->sortedReady = true;
	}

	const dim3 nodeBlocks(static_cast<unsigned int>(divUp(_state->nodeCount, ThreadsPerBlock)));
	initializeNodesKernel<<<nodeBlocks, ThreadsPerBlock>>>(_state->nodes, static_cast<int>(_state->nodeCount));
	CudaHelper::synchronize("initializeNodesKernel");

	const dim3 leafBlocks(static_cast<unsigned int>(divUp(_state->leafCount, ThreadsPerBlock)));
	initializeLeafKeysKernel<<<leafBlocks, ThreadsPerBlock>>>(
		_state->sortedKeys,
		_state->pointCount,
		_state->leafCapacity,
		_state->leafCount,
		_state->leafKeys);
	CudaHelper::synchronize("initializeLeafKeysKernel");

	initializeLeavesKernel<<<leafBlocks, ThreadsPerBlock>>>(
		_state->nodes,
		_state->pointCount,
		_state->leafCapacity,
		static_cast<int>(_state->leafCount));
	CudaHelper::synchronize("initializeLeavesKernel");

	if (_state->leafCount > 1)
	{
		const dim3 internalBlocks(static_cast<unsigned int>(divUp(_state->leafCount - 1, ThreadsPerBlock)));
		buildRadixTreeKernel<<<internalBlocks, ThreadsPerBlock>>>(
			_state->leafKeys,
			static_cast<int>(_state->leafCount),
			_state->nodes);
		CudaHelper::synchronize("buildRadixTreeKernel");
	}

	computeLeafBoundsKernel<<<leafBlocks, ThreadsPerBlock>>>(
		_state->points,
		_state->sortedIndices,
		_state->nodes,
		static_cast<int>(_state->leafCount));
	CudaHelper::synchronize("computeLeafBoundsKernel");

	if (_state->leafCount > 1)
	{
		CudaHelper::checkError(cudaMemset(_state->boundsCounters, 0, sizeof(int) * (_state->leafCount - 1)));
		computeInternalBoundsKernel<<<leafBlocks, ThreadsPerBlock>>>(
			_state->nodes,
			_state->boundsCounters,
			static_cast<int>(_state->leafCount));
		CudaHelper::synchronize("computeInternalBoundsKernel");
	}

	result.gpuBuildTimeMs = CudaHelper::stopTimer(buildBegin, buildEnd);
	cudaEventDestroy(buildBegin);
	cudaEventDestroy(buildEnd);

	result.gpuMemoryBytes = _state->memoryBytes;
	result.metrics.buildTimeMs = result.gpuBuildTimeMs;
	result.metrics.numNodes = _state->nodeCount;
	result.metrics.numLeaves = _state->leafCount;
	result.metrics.indexedPoints = _state->pointCount;
	result.metrics.maxDepth = lbvhMaxDepth(_state->leafCount);
	result.metrics.averageLeafOccupancy = _state->leafCount > 0
		? static_cast<double>(_state->pointCount) / static_cast<double>(_state->leafCount)
		: 0.0;
	result.metrics.maxLeafOccupancy = std::min(_state->leafCapacity, _state->pointCount);
	result.metrics.memoryEstimateBytes = _state->memoryBytes;
	return result;
}

PointGpu::QueryResult PointGpu::LBVH::query(const std::vector<Query>& queries, const Options& options) const
{
	if (!_state || _state->nodeCount == 0 || queries.empty())
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
			_state->sortedIndices,
			_state->nodes,
			deviceQueries,
			currentBatch,
			clockRate,
			deviceSamples);
		CudaHelper::synchronize("queryKernel");

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

bool PointGpu::LBVH::built() const
{
	return _state && _state->nodeCount > 0;
}

size_t PointGpu::LBVH::pointCount() const
{
	return _state ? _state->pointCount : 0;
}

size_t PointGpu::LBVH::nodeCount() const
{
	return _state ? _state->nodeCount : 0;
}

size_t PointGpu::LBVH::leafCount() const
{
	return _state ? _state->leafCount : 0;
}

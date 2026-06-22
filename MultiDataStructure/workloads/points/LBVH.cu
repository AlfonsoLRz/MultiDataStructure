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
		size_t leafCapacity = schema._buildPolicy._leafCapacity;
		if (!schema._levels.empty() && schema._levels.front()._leafCapacity > 0)
			leafCapacity = schema._levels.front()._leafCapacity;

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
		node._minX = 3.402823466e+38F;
		node._minY = 3.402823466e+38F;
		node._minZ = 3.402823466e+38F;
		node._maxX = -3.402823466e+38F;
		node._maxY = -3.402823466e+38F;
		node._maxZ = -3.402823466e+38F;
		node._left = -1;
		node._right = -1;
		node._parent = -1;
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
		nodes[nodeIndex]._pointOffset = static_cast<uint32_t>(pointOffset);
		nodes[nodeIndex]._pointCount = static_cast<uint32_t>(min(leafCapacity, remaining));
		nodes[nodeIndex]._flags = 1;
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

		nodes[internalIndex]._left = leftChild;
		nodes[internalIndex]._right = rightChild;
		nodes[internalIndex]._flags = 0;
		nodes[leftChild]._parent = internalIndex;
		nodes[rightChild]._parent = internalIndex;
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

		for (uint32_t i = 0; i < node._pointCount; ++i)
		{
			const uint32_t pointIndex = sortedIndices[node._pointOffset + i];
			const DevicePoint point = points[pointIndex];
			minX = fminf(minX, point.x);
			minY = fminf(minY, point.y);
			minZ = fminf(minZ, point.z);
			maxX = fmaxf(maxX, point.x);
			maxY = fmaxf(maxY, point.y);
			maxZ = fmaxf(maxZ, point.z);
		}

		node._minX = minX;
		node._minY = minY;
		node._minZ = minZ;
		node._maxX = maxX;
		node._maxY = maxY;
		node._maxZ = maxZ;
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
			const int parent = nodes[current]._parent;
			if (parent < 0)
				return;

			__threadfence();
			const int previous = atomicAdd(&counters[parent], 1);
			if (previous == 0)
				return;

			const int left = nodes[parent]._left;
			const int right = nodes[parent]._right;
			const LinearNode leftNode = nodes[left];
			const LinearNode rightNode = nodes[right];

			nodes[parent]._minX = fminf(leftNode._minX, rightNode._minX);
			nodes[parent]._minY = fminf(leftNode._minY, rightNode._minY);
			nodes[parent]._minZ = fminf(leftNode._minZ, rightNode._minZ);
			nodes[parent]._maxX = fmaxf(leftNode._maxX, rightNode._maxX);
			nodes[parent]._maxY = fmaxf(leftNode._maxY, rightNode._maxY);
			nodes[parent]._maxZ = fmaxf(leftNode._maxZ, rightNode._maxZ);
			current = parent;
		}
	}

	__global__ void queryKernel(
		const DevicePoint* points,
		const uint32_t* sortedIndices,
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
			++visited;

			const bool intersects = query._type == 2
				? distanceSquaredToNode(node, query) <= query._radius * query._radius
				: rangeIntersectsNode(node, query);
			if (!intersects)
				continue;

			if (node._flags == 1)
			{
				for (uint32_t i = 0; i < node._pointCount; ++i)
				{
					++tested;
					const uint32_t pointIndex = sortedIndices[node._pointOffset + i];
					const DevicePoint point = points[pointIndex];
					const bool inside = query._type == 2
						? pointInsideRadius(point, query)
						: pointInsideRange(point, query);
					if (inside)
						++returned;
				}
				continue;
			}

			if (node._left >= 0 && stackSize < QueryStackSize)
				stack[stackSize++] = node._left;
			if (node._right >= 0 && stackSize < QueryStackSize)
				stack[stackSize++] = node._right;
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

struct PointGpu::LBVH::DeviceState
{
	DevicePoint*		_points = nullptr;
	uint64_t*			_keys = nullptr;
	uint64_t*			_sortedKeys = nullptr;
	uint64_t*			_leafKeys = nullptr;
	uint32_t*			_indices = nullptr;
	uint32_t*			_sortedIndices = nullptr;
	LinearNode*			_nodes = nullptr;
	int*				_boundsCounters = nullptr;
	DeviceQuery*		_queryBuffer = nullptr;
	DeviceQuerySample*	_sampleBuffer = nullptr;
	void*				_sortTemporary = nullptr;
	size_t				_sortTemporaryBytes = 0;
	size_t				_queryCapacity = 0;
	size_t				_pointCount = 0;
	size_t				_leafCount = 0;
	size_t				_nodeCount = 0;
	size_t				_leafCapacity = 1;
	size_t				_baseMemoryBytes = 0;
	size_t				_memoryBytes = 0;
	int					_device = 0;
	std::string			_builder = "lbvh";
	const PointCloud*	_cloud = nullptr;
	bool				_sortedReady = false;
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
	cudaFree(_state->_points);
	cudaFree(_state->_keys);
	cudaFree(_state->_sortedKeys);
	cudaFree(_state->_indices);
	cudaFree(_state->_sortedIndices);
	cudaFree(_state->_sortTemporary);
	releaseQueryBuffers(*_state);
	*_state = DeviceState();
}

void PointGpu::LBVH::releaseTree()
{
	cudaFree(_state->_leafKeys);
	cudaFree(_state->_nodes);
	cudaFree(_state->_boundsCounters);
	_state->_leafKeys = nullptr;
	_state->_nodes = nullptr;
	_state->_boundsCounters = nullptr;
	_state->_leafCount = 0;
	_state->_nodeCount = 0;
	_state->_leafCapacity = 1;
	_state->_memoryBytes = _state->_baseMemoryBytes;
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
	if (options._builder != "lbvh")
		throw std::runtime_error("LBVH evaluator currently supports --cuda-builder lbvh only; mixed CUDA build is a later milestone.");
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("LBVH currently supports up to 2^32 - 1 points.");

	std::string availabilityError;
	if (!isAvailable(&availabilityError))
		throw std::runtime_error("CUDA point evaluator is unavailable: " + availabilityError);

	const int count = deviceCount();
	const int device = options._device >= 0 ? std::min(options._device, count - 1) : 0;
	CudaHelper::checkError(cudaSetDevice(device));

	const bool canReuseSortedPoints =
		_state->_sortedReady &&
		_state->_cloud == &cloud &&
		_state->_pointCount == cloud.size() &&
		_state->_device == device;

	if (!canReuseSortedPoints)
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
	_state->_leafCount = cloud.empty() ? 0 : divUp(cloud.size(), _state->_leafCapacity);
	_state->_nodeCount = _state->_leafCount == 0 ? 0 : 2 * _state->_leafCount - 1;
	_state->_device = device;
	_state->_builder = options._builder;
	_state->_cloud = &cloud;

	BuildResult result;
	result._device = device;
	result._builder = options._builder;

	if (cloud.empty())
		return result;

	const glm::vec3 boundsMin = cloud.bounds().min();
	const glm::vec3 extent = glm::max(cloud.coordinateRange(), glm::vec3(0.0f));

	if (!canReuseSortedPoints)
	{
		cudaEvent_t uploadBegin = nullptr;
		cudaEvent_t uploadEnd = nullptr;
		CudaHelper::startTimer(uploadBegin, uploadEnd);
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_points), sizeof(DevicePoint) * _state->_pointCount));
		CudaHelper::checkError(cudaMemcpy(_state->_points, cloud.points().data(), sizeof(DevicePoint) * _state->_pointCount, cudaMemcpyHostToDevice));
		result._uploadTimeMs = CudaHelper::stopTimer(uploadBegin, uploadEnd);
		cudaEventDestroy(uploadBegin);
		cudaEventDestroy(uploadEnd);

		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_keys), sizeof(uint64_t) * _state->_pointCount));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_sortedKeys), sizeof(uint64_t) * _state->_pointCount));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_indices), sizeof(uint32_t) * _state->_pointCount));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_sortedIndices), sizeof(uint32_t) * _state->_pointCount));

		CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(
			nullptr,
			_state->_sortTemporaryBytes,
			_state->_keys,
			_state->_sortedKeys,
			_state->_indices,
			_state->_sortedIndices,
			static_cast<int>(_state->_pointCount)));
		CudaHelper::checkError(cudaMalloc(&_state->_sortTemporary, _state->_sortTemporaryBytes));

		_state->_baseMemoryBytes =
			sizeof(DevicePoint) * _state->_pointCount +
			sizeof(uint64_t) * _state->_pointCount * 2 +
			sizeof(uint32_t) * _state->_pointCount * 2 +
			_state->_sortTemporaryBytes;
	}

	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_leafKeys), sizeof(uint64_t) * _state->_leafCount));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_nodes), sizeof(LinearNode) * _state->_nodeCount));
	if (_state->_leafCount > 1)
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_boundsCounters), sizeof(int) * (_state->_leafCount - 1)));

	_state->_memoryBytes =
		_state->_baseMemoryBytes +
		sizeof(uint64_t) * _state->_leafCount +
		sizeof(LinearNode) * _state->_nodeCount +
		sizeof(int) * (_state->_leafCount > 1 ? _state->_leafCount - 1 : 0);
	checkMemoryBudget(_state->_memoryBytes, options._memoryBudgetMb);

	cudaEvent_t buildBegin = nullptr;
	cudaEvent_t buildEnd = nullptr;
	CudaHelper::startTimer(buildBegin, buildEnd);

	if (!canReuseSortedPoints)
	{
		const dim3 pointBlocks(static_cast<unsigned int>(divUp(_state->_pointCount, ThreadsPerBlock)));
		initializePointKeysKernel<<<pointBlocks, ThreadsPerBlock>>>(
			_state->_points,
			_state->_pointCount,
			boundsMin.x,
			boundsMin.y,
			boundsMin.z,
			extent.x,
			extent.y,
			extent.z,
			_state->_keys,
			_state->_indices);
		CudaHelper::synchronize("initializePointKeysKernel");

		CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(
			_state->_sortTemporary,
			_state->_sortTemporaryBytes,
			_state->_keys,
			_state->_sortedKeys,
			_state->_indices,
			_state->_sortedIndices,
			static_cast<int>(_state->_pointCount)));
		CudaHelper::synchronize("cub::DeviceRadixSort::SortPairs");
		_state->_sortedReady = true;
	}

	const dim3 nodeBlocks(static_cast<unsigned int>(divUp(_state->_nodeCount, ThreadsPerBlock)));
	initializeNodesKernel<<<nodeBlocks, ThreadsPerBlock>>>(_state->_nodes, static_cast<int>(_state->_nodeCount));
	CudaHelper::synchronize("initializeNodesKernel");

	const dim3 leafBlocks(static_cast<unsigned int>(divUp(_state->_leafCount, ThreadsPerBlock)));
	initializeLeafKeysKernel<<<leafBlocks, ThreadsPerBlock>>>(
		_state->_sortedKeys,
		_state->_pointCount,
		_state->_leafCapacity,
		_state->_leafCount,
		_state->_leafKeys);
	CudaHelper::synchronize("initializeLeafKeysKernel");

	initializeLeavesKernel<<<leafBlocks, ThreadsPerBlock>>>(
		_state->_nodes,
		_state->_pointCount,
		_state->_leafCapacity,
		static_cast<int>(_state->_leafCount));
	CudaHelper::synchronize("initializeLeavesKernel");

	if (_state->_leafCount > 1)
	{
		const dim3 internalBlocks(static_cast<unsigned int>(divUp(_state->_leafCount - 1, ThreadsPerBlock)));
		buildRadixTreeKernel<<<internalBlocks, ThreadsPerBlock>>>(
			_state->_leafKeys,
			static_cast<int>(_state->_leafCount),
			_state->_nodes);
		CudaHelper::synchronize("buildRadixTreeKernel");
	}

	computeLeafBoundsKernel<<<leafBlocks, ThreadsPerBlock>>>(
		_state->_points,
		_state->_sortedIndices,
		_state->_nodes,
		static_cast<int>(_state->_leafCount));
	CudaHelper::synchronize("computeLeafBoundsKernel");

	if (_state->_leafCount > 1)
	{
		CudaHelper::checkError(cudaMemset(_state->_boundsCounters, 0, sizeof(int) * (_state->_leafCount - 1)));
		computeInternalBoundsKernel<<<leafBlocks, ThreadsPerBlock>>>(
			_state->_nodes,
			_state->_boundsCounters,
			static_cast<int>(_state->_leafCount));
		CudaHelper::synchronize("computeInternalBoundsKernel");
	}

	result._gpuBuildTimeMs = CudaHelper::stopTimer(buildBegin, buildEnd);
	cudaEventDestroy(buildBegin);
	cudaEventDestroy(buildEnd);

	result._gpuMemoryBytes = _state->_memoryBytes;
	result._metrics._buildTimeMs = result._gpuBuildTimeMs;
	result._metrics._numNodes = _state->_nodeCount;
	result._metrics._numLeaves = _state->_leafCount;
	result._metrics._indexedPoints = _state->_pointCount;
	result._metrics._maxDepth = lbvhMaxDepth(_state->_leafCount);
	result._metrics._averageLeafOccupancy = _state->_leafCount > 0
		? static_cast<double>(_state->_pointCount) / static_cast<double>(_state->_leafCount)
		: 0.0;
	result._metrics._maxLeafOccupancy = std::min(_state->_leafCapacity, _state->_pointCount);
	result._metrics._memoryEstimateBytes = _state->_memoryBytes;
	return result;
}

PointGpu::QueryResult PointGpu::LBVH::query(const std::vector<Query>& queries, const Options& options) const
{
	if (!_state || _state->_nodeCount == 0 || queries.empty())
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
			_state->_sortedIndices,
			_state->_nodes,
			_state->_pointCount,
			_state->_queryBuffer,
			currentBatch,
			clockRate,
			_state->_sampleBuffer);
		CudaHelper::synchronize("queryKernel");
		if (batchHasKnn)
		{
			PointGpu::bruteForceKnnKernel<<<static_cast<unsigned int>(currentBatch), ThreadsPerBlock, sizeof(float) * ThreadsPerBlock>>>(
				_state->_points,
				_state->_pointCount,
				_state->_queryBuffer,
				currentBatch,
				clockRate,
				_state->_sampleBuffer);
			CudaHelper::synchronize("knnQueryKernel");
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

bool PointGpu::LBVH::built() const
{
	return _state && _state->_nodeCount > 0;
}

size_t PointGpu::LBVH::pointCount() const
{
	return _state ? _state->_pointCount : 0;
}

size_t PointGpu::LBVH::nodeCount() const
{
	return _state ? _state->_nodeCount : 0;
}

size_t PointGpu::LBVH::leafCount() const
{
	return _state ? _state->_leafCount : 0;
}

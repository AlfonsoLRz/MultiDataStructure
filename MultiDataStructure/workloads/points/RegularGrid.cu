#include "../../stdafx.h"
#include "RegularGrid.h"

#include "../../CudaHelper.h"

#include <cub/cub.cuh>

namespace
{
	using PointGpu::DevicePoint;
	using PointGpu::DeviceQuery;
	using PointGpu::DeviceQuerySample;

	constexpr int ThreadsPerBlock = 256;
	constexpr uint32_t InvalidCellRange = 0xffffffffu;

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

	struct GridShape
	{
		uint32_t x = 1;
		uint32_t y = 1;
		uint32_t z = 1;
		size_t cells = 1;
	};

	size_t checkedProduct(uint32_t x, uint32_t y, uint32_t z)
	{
		const uint64_t product = static_cast<uint64_t>(x) * static_cast<uint64_t>(y) * static_cast<uint64_t>(z);
		if (product > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
			throw std::runtime_error("RegularGrid currently supports up to 2^32 - 1 cells.");
		return static_cast<size_t>(product);
	}

	GridShape chooseGridShape(size_t pointCount, size_t leafCapacity, const glm::vec3& coordinateRange)
	{
		if (pointCount == 0)
			return {};

		const double targetCells = static_cast<double>(std::max<size_t>(1, divUp(pointCount, leafCapacity)));
		const glm::dvec3 range(
			std::max(0.0f, coordinateRange.x),
			std::max(0.0f, coordinateRange.y),
			std::max(0.0f, coordinateRange.z));

		std::array<bool, 3> active = {
			range.x > 0.000001,
			range.y > 0.000001,
			range.z > 0.000001,
		};
		int activeAxes = 0;
		double activeVolume = 1.0;
		for (int axis = 0; axis < 3; ++axis)
		{
			if (active[axis])
			{
				++activeAxes;
				activeVolume *= range[axis];
			}
		}

		GridShape shape;
		if (activeAxes == 0 || targetCells <= 1.0)
		{
			shape.cells = checkedProduct(shape.x, shape.y, shape.z);
			return shape;
		}

		const double density = std::pow(targetCells / std::max(activeVolume, 0.000001), 1.0 / static_cast<double>(activeAxes));
		uint32_t dimensions[3] = { 1, 1, 1 };
		for (int axis = 0; axis < 3; ++axis)
		{
			if (!active[axis])
				continue;

			const double raw = std::ceil(range[axis] * density);
			const double clamped = std::clamp(raw, 1.0, static_cast<double>(std::numeric_limits<uint32_t>::max()));
			dimensions[axis] = static_cast<uint32_t>(clamped);
		}

		shape.x = dimensions[0];
		shape.y = dimensions[1];
		shape.z = dimensions[2];
		shape.cells = checkedProduct(shape.x, shape.y, shape.z);
		return shape;
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

	__device__ uint32_t clampCellCoordinate(float value, float minValue, float extent, uint32_t dimension)
	{
		if (dimension <= 1 || extent <= 0.0f)
			return 0;

		const float normalized = (value - minValue) / extent;
		const int coordinate = static_cast<int>(floorf(normalized * static_cast<float>(dimension)));
		return static_cast<uint32_t>(min(max(coordinate, 0), static_cast<int>(dimension) - 1));
	}

	__device__ uint32_t gridCellIndex(
		const DevicePoint& point,
		float minX,
		float minY,
		float minZ,
		float extentX,
		float extentY,
		float extentZ,
		uint32_t dimX,
		uint32_t dimY,
		uint32_t dimZ)
	{
		const uint32_t x = clampCellCoordinate(point.x, minX, extentX, dimX);
		const uint32_t y = clampCellCoordinate(point.y, minY, extentY, dimY);
		const uint32_t z = clampCellCoordinate(point.z, minZ, extentZ, dimZ);
		return (z * dimY + y) * dimX + x;
	}

	__device__ bool pointInsideRange(const DevicePoint& point, const DeviceQuery& query)
	{
		return point.x >= query.minX && point.x <= query.maxX &&
			point.y >= query.minY && point.y <= query.maxY &&
			point.z >= query.minZ && point.z <= query.maxZ;
	}

	__device__ bool pointInsideRadius(const DevicePoint& point, const DeviceQuery& query)
	{
		const float dx = point.x - query.centerX;
		const float dy = point.y - query.centerY;
		const float dz = point.z - query.centerZ;
		return dx * dx + dy * dy + dz * dz <= query.radius * query.radius;
	}

	__device__ bool queryBoxIntersectsGrid(
		float minX,
		float minY,
		float minZ,
		float maxX,
		float maxY,
		float maxZ,
		float gridMinX,
		float gridMinY,
		float gridMinZ,
		float extentX,
		float extentY,
		float extentZ)
	{
		return maxX >= gridMinX && minX <= gridMinX + extentX &&
			maxY >= gridMinY && minY <= gridMinY + extentY &&
			maxZ >= gridMinZ && minZ <= gridMinZ + extentZ;
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
		uint32_t dimX,
		uint32_t dimY,
		uint32_t dimZ,
		uint32_t* keys,
		uint32_t* indices)
	{
		const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (index >= pointCount)
			return;

		keys[index] = gridCellIndex(points[index], minX, minY, minZ, extentX, extentY, extentZ, dimX, dimY, dimZ);
		indices[index] = static_cast<uint32_t>(index);
	}

	__global__ void initializeCellRangesKernel(uint32_t* starts, uint32_t* ends, size_t cellCount)
	{
		const size_t cellIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (cellIndex >= cellCount)
			return;

		starts[cellIndex] = InvalidCellRange;
		ends[cellIndex] = InvalidCellRange;
	}

	__global__ void buildCellRangesKernel(const uint32_t* sortedKeys, size_t pointCount, uint32_t* starts, uint32_t* ends)
	{
		const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (index >= pointCount)
			return;

		const uint32_t key = sortedKeys[index];
		if (index == 0 || sortedKeys[index - 1] != key)
			starts[key] = static_cast<uint32_t>(index);
		if (index + 1 == pointCount || sortedKeys[index + 1] != key)
			ends[key] = static_cast<uint32_t>(index + 1);
	}

	__global__ void queryKernel(
		const DevicePoint* points,
		const uint32_t* sortedIndices,
		size_t pointCount,
		const uint32_t* cellStarts,
		const uint32_t* cellEnds,
		uint32_t dimX,
		uint32_t dimY,
		uint32_t dimZ,
		float minX,
		float minY,
		float minZ,
		float extentX,
		float extentY,
		float extentZ,
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
		float queryMinX = query.minX;
		float queryMinY = query.minY;
		float queryMinZ = query.minZ;
		float queryMaxX = query.maxX;
		float queryMaxY = query.maxY;
		float queryMaxZ = query.maxZ;
		if (query.type == static_cast<int>(PointGpu::QueryType::Radius))
		{
			queryMinX = query.centerX - query.radius;
			queryMinY = query.centerY - query.radius;
			queryMinZ = query.centerZ - query.radius;
			queryMaxX = query.centerX + query.radius;
			queryMaxY = query.centerY + query.radius;
			queryMaxZ = query.centerZ + query.radius;
		}

		unsigned long long visited = 0;
		unsigned long long tested = 0;
		unsigned long long returned = 0;

		if (queryBoxIntersectsGrid(
			queryMinX,
			queryMinY,
			queryMinZ,
			queryMaxX,
			queryMaxY,
			queryMaxZ,
			minX,
			minY,
			minZ,
			extentX,
			extentY,
			extentZ))
		{
			const uint32_t x0 = clampCellCoordinate(queryMinX, minX, extentX, dimX);
			const uint32_t y0 = clampCellCoordinate(queryMinY, minY, extentY, dimY);
			const uint32_t z0 = clampCellCoordinate(queryMinZ, minZ, extentZ, dimZ);
			const uint32_t x1 = clampCellCoordinate(queryMaxX, minX, extentX, dimX);
			const uint32_t y1 = clampCellCoordinate(queryMaxY, minY, extentY, dimY);
			const uint32_t z1 = clampCellCoordinate(queryMaxZ, minZ, extentZ, dimZ);

			for (uint32_t z = z0; z <= z1; ++z)
			{
				for (uint32_t y = y0; y <= y1; ++y)
				{
					for (uint32_t x = x0; x <= x1; ++x)
					{
						++visited;
						const uint32_t cell = (z * dimY + y) * dimX + x;
						const uint32_t beginPoint = cellStarts[cell];
						const uint32_t endPoint = cellEnds[cell];
						if (beginPoint == InvalidCellRange || endPoint == InvalidCellRange)
							continue;

						for (uint32_t pointSlot = beginPoint; pointSlot < endPoint; ++pointSlot)
						{
							++tested;
							const DevicePoint point = points[sortedIndices[pointSlot]];
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
					}
				}
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

struct PointGpu::RegularGrid::DeviceState
{
	DevicePoint* points = nullptr;
	uint32_t* keys = nullptr;
	uint32_t* sortedKeys = nullptr;
	uint32_t* indices = nullptr;
	uint32_t* sortedIndices = nullptr;
	uint32_t* cellStarts = nullptr;
	uint32_t* cellEnds = nullptr;
	DeviceQuery* queryBuffer = nullptr;
	DeviceQuerySample* sampleBuffer = nullptr;
	void* sortTemporary = nullptr;
	size_t sortTemporaryBytes = 0;
	size_t queryCapacity = 0;
	size_t pointCount = 0;
	size_t cellCount = 0;
	size_t leafCapacity = 1;
	size_t baseMemoryBytes = 0;
	size_t memoryBytes = 0;
	uint32_t dimX = 1;
	uint32_t dimY = 1;
	uint32_t dimZ = 1;
	float minX = 0.0f;
	float minY = 0.0f;
	float minZ = 0.0f;
	float extentX = 0.0f;
	float extentY = 0.0f;
	float extentZ = 0.0f;
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
			message << "RegularGrid needs about " << (static_cast<double>(bytes) / (1024.0 * 1024.0))
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

PointGpu::RegularGrid::RegularGrid()
	: _state(std::make_unique<DeviceState>())
{
}

PointGpu::RegularGrid::~RegularGrid()
{
	if (_state)
		release();
}

void PointGpu::RegularGrid::release()
{
	releaseGrid();
	cudaFree(_state->points);
	cudaFree(_state->keys);
	cudaFree(_state->sortedKeys);
	cudaFree(_state->indices);
	cudaFree(_state->sortedIndices);
	cudaFree(_state->sortTemporary);
	releaseQueryBuffers(*_state);
	*_state = DeviceState();
}

void PointGpu::RegularGrid::releaseGrid()
{
	cudaFree(_state->cellStarts);
	cudaFree(_state->cellEnds);
	_state->cellStarts = nullptr;
	_state->cellEnds = nullptr;
	_state->cellCount = 0;
	_state->memoryBytes = _state->baseMemoryBytes;
}

bool PointGpu::RegularGrid::isAvailable(std::string* error)
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

int PointGpu::RegularGrid::deviceCount()
{
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess)
		return 0;
	return count;
}

PointGpu::BuildResult PointGpu::RegularGrid::build(const PointCloud& cloud, const SchemaConfig& schema, const Options& options)
{
	const std::string builder = options.builder.empty() ? "regular_grid" : options.builder;
	if (builder != "regular_grid" && builder != "regulargrid" && builder != "grid" && builder != "uniform_grid")
		throw std::runtime_error("RegularGrid evaluator supports --cuda-builder regular_grid, grid, or uniform_grid.");
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("RegularGrid currently supports up to 2^32 - 1 points.");

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
		if (_state->points || _state->cellStarts)
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
		releaseGrid();
	}

	_state->pointCount = cloud.size();
	_state->leafCapacity = leafCapacityForSchema(schema);
	_state->device = device;
	_state->cloud = &cloud;

	BuildResult result;
	result.device = device;
	result.builder = "regular_grid";

	if (cloud.empty())
		return result;

	const glm::vec3 boundsMin = cloud.bounds().min();
	const glm::vec3 extent = glm::max(cloud.coordinateRange(), glm::vec3(0.0f));
	const GridShape shape = chooseGridShape(cloud.size(), _state->leafCapacity, extent);
	_state->dimX = shape.x;
	_state->dimY = shape.y;
	_state->dimZ = shape.z;
	_state->cellCount = shape.cells;
	_state->minX = boundsMin.x;
	_state->minY = boundsMin.y;
	_state->minZ = boundsMin.z;
	_state->extentX = extent.x;
	_state->extentY = extent.y;
	_state->extentZ = extent.z;

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

		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->keys), sizeof(uint32_t) * _state->pointCount));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->sortedKeys), sizeof(uint32_t) * _state->pointCount));
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
			sizeof(uint32_t) * _state->pointCount * 4 +
			_state->sortTemporaryBytes;
		_state->pointsReady = true;
	}

	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->cellStarts), sizeof(uint32_t) * _state->cellCount));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->cellEnds), sizeof(uint32_t) * _state->cellCount));

	_state->memoryBytes =
		_state->baseMemoryBytes +
		sizeof(uint32_t) * _state->cellCount * 2;
	checkMemoryBudget(_state->memoryBytes, options.memoryBudgetMb);

	cudaEvent_t buildBegin = nullptr;
	cudaEvent_t buildEnd = nullptr;
	CudaHelper::startTimer(buildBegin, buildEnd);

	const dim3 pointBlocks(static_cast<unsigned int>(divUp(_state->pointCount, ThreadsPerBlock)));
	initializePointKeysKernel<<<pointBlocks, ThreadsPerBlock>>>(
		_state->points,
		_state->pointCount,
		_state->minX,
		_state->minY,
		_state->minZ,
		_state->extentX,
		_state->extentY,
		_state->extentZ,
		_state->dimX,
		_state->dimY,
		_state->dimZ,
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

	const dim3 cellBlocks(static_cast<unsigned int>(divUp(_state->cellCount, ThreadsPerBlock)));
	initializeCellRangesKernel<<<cellBlocks, ThreadsPerBlock>>>(_state->cellStarts, _state->cellEnds, _state->cellCount);
	CudaHelper::synchronize("initializeCellRangesKernel");

	buildCellRangesKernel<<<pointBlocks, ThreadsPerBlock>>>(_state->sortedKeys, _state->pointCount, _state->cellStarts, _state->cellEnds);
	CudaHelper::synchronize("buildCellRangesKernel");

	result.gpuBuildTimeMs = CudaHelper::stopTimer(buildBegin, buildEnd);
	cudaEventDestroy(buildBegin);
	cudaEventDestroy(buildEnd);

	std::vector<uint32_t> hostStarts(_state->cellCount);
	std::vector<uint32_t> hostEnds(_state->cellCount);
	CudaHelper::checkError(cudaMemcpy(hostStarts.data(), _state->cellStarts, sizeof(uint32_t) * _state->cellCount, cudaMemcpyDeviceToHost));
	CudaHelper::checkError(cudaMemcpy(hostEnds.data(), _state->cellEnds, sizeof(uint32_t) * _state->cellCount, cudaMemcpyDeviceToHost));

	size_t nonEmptyCells = 0;
	size_t maxOccupancy = 0;
	for (size_t cell = 0; cell < _state->cellCount; ++cell)
	{
		if (hostStarts[cell] == InvalidCellRange || hostEnds[cell] == InvalidCellRange)
			continue;

		++nonEmptyCells;
		maxOccupancy = std::max<size_t>(maxOccupancy, static_cast<size_t>(hostEnds[cell] - hostStarts[cell]));
	}

	result.gpuMemoryBytes = _state->memoryBytes;
	result.metrics.buildTimeMs = result.gpuBuildTimeMs;
	result.metrics.numNodes = _state->cellCount;
	result.metrics.numLeaves = nonEmptyCells;
	result.metrics.indexedPoints = _state->pointCount;
	result.metrics.maxDepth = 1;
	result.metrics.averageLeafOccupancy = nonEmptyCells > 0
		? static_cast<double>(_state->pointCount) / static_cast<double>(nonEmptyCells)
		: 0.0;
	result.metrics.maxLeafOccupancy = maxOccupancy;
	result.metrics.memoryEstimateBytes = _state->memoryBytes;
	return result;
}

PointGpu::QueryResult PointGpu::RegularGrid::query(const std::vector<Query>& queries, const Options& options) const
{
	if (!_state || _state->cellCount == 0 || queries.empty())
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
			_state->sortedIndices,
			_state->pointCount,
			_state->cellStarts,
			_state->cellEnds,
			_state->dimX,
			_state->dimY,
			_state->dimZ,
			_state->minX,
			_state->minY,
			_state->minZ,
			_state->extentX,
			_state->extentY,
			_state->extentZ,
			_state->queryBuffer,
			currentBatch,
			clockRate,
			_state->sampleBuffer);
		CudaHelper::synchronize("regularGridQueryKernel");
		if (batchHasKnn)
		{
			PointGpu::bruteForceKnnKernel<<<static_cast<unsigned int>(currentBatch), ThreadsPerBlock, sizeof(float) * ThreadsPerBlock>>>(
				_state->points,
				_state->pointCount,
				_state->queryBuffer,
				currentBatch,
				clockRate,
				_state->sampleBuffer);
			CudaHelper::synchronize("regularGridKnnQueryKernel");
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

bool PointGpu::RegularGrid::built() const
{
	return _state && _state->cellCount > 0;
}

size_t PointGpu::RegularGrid::pointCount() const
{
	return _state ? _state->pointCount : 0;
}

size_t PointGpu::RegularGrid::cellCount() const
{
	return _state ? _state->cellCount : 0;
}

glm::uvec3 PointGpu::RegularGrid::dimensions() const
{
	if (!_state)
		return glm::uvec3(0);
	return glm::uvec3(_state->dimX, _state->dimY, _state->dimZ);
}

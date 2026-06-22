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
		size_t leafCapacity = schema._buildPolicy._leafCapacity;
		if (!schema._levels.empty() && schema._levels.front()._leafCapacity > 0)
			leafCapacity = schema._levels.front()._leafCapacity;

		return std::max<size_t>(1, leafCapacity);
	}

	struct GridShape
	{
		uint32_t x = 1;
		uint32_t y = 1;
		uint32_t z = 1;
		size_t _cells = 1;
	};

	size_t checkedProduct(uint32_t x, uint32_t y, uint32_t z)
	{
		const uint64_t product = static_cast<uint64_t>(x) * static_cast<uint64_t>(y) * static_cast<uint64_t>(z);
		if (product > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
			throw std::runtime_error("RegularGrid currently supports up to 2^32 - 1 cells.");
		return static_cast<size_t>(product);
	}

	// Hard cap (applies even with unlimited budget); 256M cells is ~2 GB of cell tables alone.
	constexpr size_t MaxCellsPerLevel = 256ull * 1024 * 1024;

	GridShape chooseGridShape(size_t pointCount, size_t leafCapacity, const glm::vec3& coordinateRange)
	{
		if (pointCount == 0)
			return {};

		const double targetCells = static_cast<double>(std::max<size_t>(1, divUp(pointCount, leafCapacity)));
		if (targetCells > static_cast<double>(MaxCellsPerLevel))
		{
			throw std::runtime_error("RegularGrid target cell count ("
				+ std::to_string(static_cast<size_t>(targetCells))
				+ ") exceeds the per-level safety cap ("
				+ std::to_string(MaxCellsPerLevel)
				+ "). Increase leaf capacity or restrict the schema generator's min leaf bound.");
		}
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
			shape._cells = checkedProduct(shape.x, shape.y, shape.z);
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
		shape._cells = checkedProduct(shape.x, shape.y, shape.z);
		if (shape._cells > MaxCellsPerLevel)
		{
			throw std::runtime_error("RegularGrid post-rounding cell count ("
				+ std::to_string(shape._cells)
				+ ") exceeds the per-level safety cap ("
				+ std::to_string(MaxCellsPerLevel)
				+ ") after axis ceiling.");
		}
		return shape;
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
		return point.x >= query._minX && point.x <= query._maxX &&
			point.y >= query._minY && point.y <= query._maxY &&
			point.z >= query._minZ && point.z <= query._maxZ;
	}

	__device__ bool pointInsideRadius(const DevicePoint& point, const DeviceQuery& query)
	{
		const float dx = point.x - query._centerX;
		const float dy = point.y - query._centerY;
		const float dz = point.z - query._centerZ;
		return dx * dx + dy * dy + dz * dz <= query._radius * query._radius;
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
		if (query._type == static_cast<int>(PointGpu::QueryType::Knn))
		{
			samples[queryIndex] = DeviceQuerySample{};
			return;
		}

		const unsigned long long begin = clock64();
		float queryMinX = query._minX;
		float queryMinY = query._minY;
		float queryMinZ = query._minZ;
		float queryMaxX = query._maxX;
		float queryMaxY = query._maxY;
		float queryMaxZ = query._maxZ;
		if (query._type == static_cast<int>(PointGpu::QueryType::Radius))
		{
			queryMinX = query._centerX - query._radius;
			queryMinY = query._centerY - query._radius;
			queryMinZ = query._centerZ - query._radius;
			queryMaxX = query._centerX + query._radius;
			queryMaxY = query._centerY + query._radius;
			queryMaxZ = query._centerZ + query._radius;
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
					}
				}
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

struct PointGpu::RegularGrid::DeviceState
{
	DevicePoint* _points = nullptr;
	uint32_t* _keys = nullptr;
	uint32_t* _sortedKeys = nullptr;
	uint32_t* _indices = nullptr;
	uint32_t* _sortedIndices = nullptr;
	uint32_t* _cellStarts = nullptr;
	uint32_t* _cellEnds = nullptr;
	DeviceQuery* _queryBuffer = nullptr;
	DeviceQuerySample* _sampleBuffer = nullptr;
	void* _sortTemporary = nullptr;
	size_t _sortTemporaryBytes = 0;
	size_t _queryCapacity = 0;
	size_t _pointCount = 0;
	size_t _cellCount = 0;
	size_t _leafCapacity = 1;
	size_t _baseMemoryBytes = 0;
	size_t _memoryBytes = 0;
	uint32_t _dimX = 1;
	uint32_t _dimY = 1;
	uint32_t _dimZ = 1;
	float _minX = 0.0f;
	float _minY = 0.0f;
	float _minZ = 0.0f;
	float _extentX = 0.0f;
	float _extentY = 0.0f;
	float _extentZ = 0.0f;
	int _device = 0;
	const PointCloud* _cloud = nullptr;
	bool _pointsReady = false;
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
	cudaFree(_state->_points);
	cudaFree(_state->_keys);
	cudaFree(_state->_sortedKeys);
	cudaFree(_state->_indices);
	cudaFree(_state->_sortedIndices);
	cudaFree(_state->_sortTemporary);
	releaseQueryBuffers(*_state);
	*_state = DeviceState();
}

void PointGpu::RegularGrid::releaseGrid()
{
	cudaFree(_state->_cellStarts);
	cudaFree(_state->_cellEnds);
	_state->_cellStarts = nullptr;
	_state->_cellEnds = nullptr;
	_state->_cellCount = 0;
	_state->_memoryBytes = _state->_baseMemoryBytes;
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
	const std::string builder = options._builder.empty() ? "regular_grid" : options._builder;
	if (builder != "regular_grid" && builder != "regulargrid" && builder != "grid" && builder != "uniform_grid")
		throw std::runtime_error("RegularGrid evaluator supports --cuda-builder regular_grid, grid, or uniform_grid.");
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("RegularGrid currently supports up to 2^32 - 1 points.");

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
		if (_state->_points || _state->_cellStarts)
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
		releaseGrid();
	}

	_state->_pointCount = cloud.size();
	_state->_leafCapacity = leafCapacityForSchema(schema);
	_state->_device = device;
	_state->_cloud = &cloud;

	BuildResult result;
	result._device = device;
	result._builder = "regular_grid";

	if (cloud.empty())
		return result;

	const glm::vec3 boundsMin = cloud.bounds().min();
	const glm::vec3 extent = glm::max(cloud.coordinateRange(), glm::vec3(0.0f));
	const GridShape shape = chooseGridShape(cloud.size(), _state->_leafCapacity, extent);
	_state->_dimX = shape.x;
	_state->_dimY = shape.y;
	_state->_dimZ = shape.z;
	_state->_cellCount = shape._cells;
	_state->_minX = boundsMin.x;
	_state->_minY = boundsMin.y;
	_state->_minZ = boundsMin.z;
	_state->_extentX = extent.x;
	_state->_extentY = extent.y;
	_state->_extentZ = extent.z;

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

		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_keys), sizeof(uint32_t) * _state->_pointCount));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_sortedKeys), sizeof(uint32_t) * _state->_pointCount));
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
			sizeof(uint32_t) * _state->_pointCount * 4 +
			_state->_sortTemporaryBytes;
		_state->_pointsReady = true;
	}

	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_cellStarts), sizeof(uint32_t) * _state->_cellCount));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_state->_cellEnds), sizeof(uint32_t) * _state->_cellCount));

	_state->_memoryBytes =
		_state->_baseMemoryBytes +
		sizeof(uint32_t) * _state->_cellCount * 2;
	checkMemoryBudget(_state->_memoryBytes, options._memoryBudgetMb);

	cudaEvent_t buildBegin = nullptr;
	cudaEvent_t buildEnd = nullptr;
	CudaHelper::startTimer(buildBegin, buildEnd);

	const dim3 pointBlocks(static_cast<unsigned int>(divUp(_state->_pointCount, ThreadsPerBlock)));
	initializePointKeysKernel<<<pointBlocks, ThreadsPerBlock>>>(
		_state->_points,
		_state->_pointCount,
		_state->_minX,
		_state->_minY,
		_state->_minZ,
		_state->_extentX,
		_state->_extentY,
		_state->_extentZ,
		_state->_dimX,
		_state->_dimY,
		_state->_dimZ,
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

	const dim3 cellBlocks(static_cast<unsigned int>(divUp(_state->_cellCount, ThreadsPerBlock)));
	initializeCellRangesKernel<<<cellBlocks, ThreadsPerBlock>>>(_state->_cellStarts, _state->_cellEnds, _state->_cellCount);
	CudaHelper::synchronize("initializeCellRangesKernel");

	buildCellRangesKernel<<<pointBlocks, ThreadsPerBlock>>>(_state->_sortedKeys, _state->_pointCount, _state->_cellStarts, _state->_cellEnds);
	CudaHelper::synchronize("buildCellRangesKernel");

	result._gpuBuildTimeMs = CudaHelper::stopTimer(buildBegin, buildEnd);
	cudaEventDestroy(buildBegin);
	cudaEventDestroy(buildEnd);

	std::vector<uint32_t> hostStarts(_state->_cellCount);
	std::vector<uint32_t> hostEnds(_state->_cellCount);
	CudaHelper::checkError(cudaMemcpy(hostStarts.data(), _state->_cellStarts, sizeof(uint32_t) * _state->_cellCount, cudaMemcpyDeviceToHost));
	CudaHelper::checkError(cudaMemcpy(hostEnds.data(), _state->_cellEnds, sizeof(uint32_t) * _state->_cellCount, cudaMemcpyDeviceToHost));

	size_t nonEmptyCells = 0;
	size_t maxOccupancy = 0;
	for (size_t cell = 0; cell < _state->_cellCount; ++cell)
	{
		if (hostStarts[cell] == InvalidCellRange || hostEnds[cell] == InvalidCellRange)
			continue;

		++nonEmptyCells;
		maxOccupancy = std::max<size_t>(maxOccupancy, static_cast<size_t>(hostEnds[cell] - hostStarts[cell]));
	}

	result._gpuMemoryBytes = _state->_memoryBytes;
	result._metrics._buildTimeMs = result._gpuBuildTimeMs;
	result._metrics._numNodes = _state->_cellCount;
	result._metrics._numLeaves = nonEmptyCells;
	result._metrics._indexedPoints = _state->_pointCount;
	result._metrics._maxDepth = 1;
	result._metrics._averageLeafOccupancy = nonEmptyCells > 0
		? static_cast<double>(_state->_pointCount) / static_cast<double>(nonEmptyCells)
		: 0.0;
	result._metrics._maxLeafOccupancy = maxOccupancy;
	result._metrics._memoryEstimateBytes = _state->_memoryBytes;
	return result;
}

PointGpu::QueryResult PointGpu::RegularGrid::query(const std::vector<Query>& queries, const Options& options) const
{
	if (!_state || _state->_cellCount == 0 || queries.empty())
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
			_state->_pointCount,
			_state->_cellStarts,
			_state->_cellEnds,
			_state->_dimX,
			_state->_dimY,
			_state->_dimZ,
			_state->_minX,
			_state->_minY,
			_state->_minZ,
			_state->_extentX,
			_state->_extentY,
			_state->_extentZ,
			_state->_queryBuffer,
			currentBatch,
			clockRate,
			_state->_sampleBuffer);
		CudaHelper::synchronize("regularGridQueryKernel");
		if (batchHasKnn)
		{
			PointGpu::bruteForceKnnKernel<<<static_cast<unsigned int>(currentBatch), ThreadsPerBlock, sizeof(float) * ThreadsPerBlock>>>(
				_state->_points,
				_state->_pointCount,
				_state->_queryBuffer,
				currentBatch,
				clockRate,
				_state->_sampleBuffer);
			CudaHelper::synchronize("regularGridKnnQueryKernel");
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

bool PointGpu::RegularGrid::built() const
{
	return _state && _state->_cellCount > 0;
}

size_t PointGpu::RegularGrid::pointCount() const
{
	return _state ? _state->_pointCount : 0;
}

size_t PointGpu::RegularGrid::cellCount() const
{
	return _state ? _state->_cellCount : 0;
}

glm::uvec3 PointGpu::RegularGrid::dimensions() const
{
	if (!_state)
		return glm::uvec3(0);
	return glm::uvec3(_state->_dimX, _state->_dimY, _state->_dimZ);
}

#include "../../stdafx.h"
#include "HGrid.h"

#include "PointSpatialIndex.h"
#include "RegularGrid.h"

// Upper bound on cells per RegularGrid level: divUp(pointCount, leafCapacity) plus a 1.25x margin.
static size_t estimateCellsForLevel(size_t pointCount, size_t leafCapacity)
{
	if (pointCount == 0 || leafCapacity == 0)
		return 0;
	const size_t target = (pointCount + leafCapacity - 1) / leafCapacity;
	const size_t margin = target + target / 4;
	return std::max<size_t>(1, margin);
}

static size_t leafCapacityForSchema(const SchemaConfig& schema)
{
	size_t leafCapacity = schema._buildPolicy._leafCapacity;
	if (!schema._levels.empty() && schema._levels.front()._leafCapacity > 0)
		leafCapacity = schema._levels.front()._leafCapacity;

	return std::max<size_t>(1, leafCapacity);
}

// Max HGrid levels; memory pre-check guards the budget so this cap can stay high.
static constexpr size_t MaxHGridLevels = 12;

static size_t hgridLevelCountForSchema(const SchemaConfig& schema)
{
	size_t configuredDepth = schema._buildPolicy._maxDepth;
	if (configuredDepth == 0)
		configuredDepth = schema.totalLevels();
	if (configuredDepth == 0)
		configuredDepth = 3;

	const size_t clamped = std::clamp(configuredDepth, static_cast<size_t>(2), MaxHGridLevels);
	if (clamped != configuredDepth)
	{
		std::cerr << "    HGrid: requested " << configuredDepth
			<< " levels, clamped to " << clamped
			<< " (range [2, " << MaxHGridLevels << "])\n";
	}
	return clamped;
}

static size_t scaledLeafCapacity(size_t baseLeafCapacity, size_t level, size_t levelCount)
{
	const size_t remaining = levelCount - 1 - level;
	size_t factor = 1;
	for (size_t i = 0; i < remaining; ++i)
	{
		if (factor > std::numeric_limits<size_t>::max() / 4)
			return std::numeric_limits<size_t>::max();
		factor *= 4;
	}

	if (baseLeafCapacity > std::numeric_limits<size_t>::max() / factor)
		return std::numeric_limits<size_t>::max();
	return std::max<size_t>(1, baseLeafCapacity * factor);
}

static SchemaConfig schemaForLevel(const SchemaConfig& base, size_t leafCapacity)
{
	SchemaConfig schema = base;
	schema._buildPolicy._leafCapacity = leafCapacity;
	schema._buildPolicy._minPrimitivesToSplit = std::max<size_t>(2, leafCapacity / 4);
	if (schema._levels.empty())
	{
		SchemaLevelConfig level;
		level._type = MultiDataStructure::DataStructureLevel::BvhNode;
		level._typeName = "BVH";
		level._numLevels = 1;
		level._leafCapacity = leafCapacity;
		level._minPrimitivesToSplit = schema._buildPolicy._minPrimitivesToSplit;
		schema._levels.push_back(level);
	}
	else
	{
		schema._levels.front()._leafCapacity = leafCapacity;
		schema._levels.front()._minPrimitivesToSplit = schema._buildPolicy._minPrimitivesToSplit;
	}
	return schema;
}

static PointGpu::Options regularGridOptions(const PointGpu::Options& options)
{
	PointGpu::Options gridOptions = options;
	gridOptions._builder = "regular_grid";
	return gridOptions;
}

static bool intersectsGrid(
	float queryMinX,
	float queryMinY,
	float queryMinZ,
	float queryMaxX,
	float queryMaxY,
	float queryMaxZ,
	const AABB& bounds)
{
	const glm::vec3 min = bounds.min();
	const glm::vec3 max = bounds.max();
	return queryMaxX >= min.x && queryMinX <= max.x &&
		queryMaxY >= min.y && queryMinY <= max.y &&
		queryMaxZ >= min.z && queryMinZ <= max.z;
}

static uint32_t clampCellCoordinate(float value, float minValue, float extent, uint32_t dimension)
{
	if (dimension <= 1 || extent <= 0.0f)
		return 0;

	const float normalized = (value - minValue) / extent;
	const int coordinate = static_cast<int>(std::floor(normalized * static_cast<float>(dimension)));
	return static_cast<uint32_t>(std::clamp(coordinate, 0, static_cast<int>(dimension) - 1));
}

static size_t estimatedVisitedCells(const PointGpu::Query& query, const AABB& bounds, const glm::uvec3& dimensions)
{
	float queryMinX = query._bounds.min().x;
	float queryMinY = query._bounds.min().y;
	float queryMinZ = query._bounds.min().z;
	float queryMaxX = query._bounds.max().x;
	float queryMaxY = query._bounds.max().y;
	float queryMaxZ = query._bounds.max().z;
	if (query._type == PointGpu::QueryType::Radius)
	{
		queryMinX = query.center.x - query._radius;
		queryMinY = query.center.y - query._radius;
		queryMinZ = query.center.z - query._radius;
		queryMaxX = query.center.x + query._radius;
		queryMaxY = query.center.y + query._radius;
		queryMaxZ = query.center.z + query._radius;
	}

	if (!intersectsGrid(queryMinX, queryMinY, queryMinZ, queryMaxX, queryMaxY, queryMaxZ, bounds))
		return 0;

	const glm::vec3 min = bounds.min();
	const glm::vec3 extent = glm::max(bounds.size(), glm::vec3(0.0f));
	const uint32_t x0 = clampCellCoordinate(queryMinX, min.x, extent.x, dimensions.x);
	const uint32_t y0 = clampCellCoordinate(queryMinY, min.y, extent.y, dimensions.y);
	const uint32_t z0 = clampCellCoordinate(queryMinZ, min.z, extent.z, dimensions.z);
	const uint32_t x1 = clampCellCoordinate(queryMaxX, min.x, extent.x, dimensions.x);
	const uint32_t y1 = clampCellCoordinate(queryMaxY, min.y, extent.y, dimensions.y);
	const uint32_t z1 = clampCellCoordinate(queryMaxZ, min.z, extent.z, dimensions.z);
	return static_cast<size_t>(x1 - x0 + 1) *
		static_cast<size_t>(y1 - y0 + 1) *
		static_cast<size_t>(z1 - z0 + 1);
}

static Experiments::QueryMetrics summarizeGpuSamples(const std::vector<PointGpu::QuerySample>& samples)
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

struct PointGpu::HGrid::DeviceState
{
	struct Level
	{
		std::unique_ptr<RegularGrid>	_grid;
		size_t	_leafCapacity = 1;
		size_t	_cellCount = 0;
		glm::uvec3	_dimensions = glm::uvec3(1);
	};

	std::vector<Level>	_levels;
	size_t				_pointCount = 0;
	size_t				_totalCellCount = 0;
	size_t				_memoryBytes = 0;
	int					_device = 0;
	const PointCloud*	_cloud = nullptr;
	AABB				_bounds;
	bool				_ready = false;
};

PointGpu::HGrid::HGrid()
	: _state(std::make_unique<DeviceState>())
{
}

PointGpu::HGrid::~HGrid()
{
	release();
}

void PointGpu::HGrid::release()
{
	if (_state)
		*_state = DeviceState();
}

bool PointGpu::HGrid::isAvailable(std::string* error)
{
	return RegularGrid::isAvailable(error);
}

int PointGpu::HGrid::deviceCount()
{
	return RegularGrid::deviceCount();
}

PointGpu::BuildResult PointGpu::HGrid::build(const PointCloud& cloud, const SchemaConfig& schema, const Options& options)
{
	const std::string builder = options._builder.empty() ? "hgrid" : options._builder;
	if (builder != "hgrid" && builder != "hierarchical_grid" && builder != "hierarchicalgrid")
		throw std::runtime_error("HGrid evaluator supports --cuda-builder hgrid or hierarchical_grid.");

	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("HGrid currently supports up to 2^32 - 1 points.");

	std::string availabilityError;
	if (!isAvailable(&availabilityError))
		throw std::runtime_error("CUDA point evaluator is unavailable: " + availabilityError);

	const int count = deviceCount();
	const int device = options._device >= 0 ? std::min(options._device, count - 1) : 0;
	const size_t baseLeafCapacity = leafCapacityForSchema(schema);
	const size_t levelCount = hgridLevelCountForSchema(schema);
	const bool canReuseLevels =
		_state->_ready &&
		_state->_cloud == &cloud &&
		_state->_pointCount == cloud.size() &&
		_state->_device == device &&
		_state->_levels.size() == levelCount;
	std::vector<DeviceState::Level> previousLevels;
	if (canReuseLevels)
	{
		previousLevels = std::move(_state->_levels);
		_state->_levels.clear();
		_state->_totalCellCount = 0;
		_state->_memoryBytes = 0;
		_state->_ready = false;
	}
	else
	{
		release();
	}

	_state->_pointCount = cloud.size();
	_state->_cloud = &cloud;
	_state->_bounds = cloud.bounds();
	_state->_device = device;

	BuildResult result;
	result._device = _state->_device;
	result._builder = "hgrid";
	if (cloud.empty())
		return result;

	// Aggregate memory pre-check: per-level budget checks miss cases where the level sum overflows.
	if (options._memoryBudgetMb > 0)
	{
		// Per-point RegularGrid overhead (point + 4 uint32 keys/indices); scratch approximated as 1.5x.
		const size_t basePerPointBytes = sizeof(float) * 3 + sizeof(uint32_t) * 4;
		const size_t basePerLevelBytes = static_cast<size_t>(cloud.size()) *
			(basePerPointBytes + sizeof(uint32_t) * 6 / 4);
		size_t aggregateBytes = 0;
		for (size_t levelIndex = 0; levelIndex < levelCount; ++levelIndex)
		{
			const size_t levelLeafCap = scaledLeafCapacity(baseLeafCapacity, levelIndex, levelCount);
			const size_t estimatedCells = estimateCellsForLevel(cloud.size(), levelLeafCap);
			const size_t cellBytes = estimatedCells * sizeof(uint32_t) * 2;
			aggregateBytes += basePerLevelBytes + cellBytes;
		}
		const size_t budgetBytes = options._memoryBudgetMb * size_t(1024) * size_t(1024);
		if (aggregateBytes > budgetBytes)
		{
			throw std::runtime_error("HGrid aggregate memory (" + std::to_string(aggregateBytes / (1024 * 1024))
				+ " MB across " + std::to_string(levelCount) + " levels) exceeds the configured budget ("
				+ std::to_string(options._memoryBudgetMb) + " MB). Raise --cuda-memory-budget-mb or use larger leaf capacities.");
		}
	}

	_state->_levels.reserve(levelCount);

	PointGpu::Options gridOptions = regularGridOptions(options);
	gridOptions._device = _state->_device;

	size_t maxLeafOccupancy = 0;
	size_t totalLeaves = 0;
	double totalLeafOccupancyWeight = 0.0;
	for (size_t levelIndex = 0; levelIndex < levelCount; ++levelIndex)
	{
		DeviceState::Level level;
		level._leafCapacity = scaledLeafCapacity(baseLeafCapacity, levelIndex, levelCount);
		if (levelIndex < previousLevels.size() && previousLevels[levelIndex]._grid)
			level._grid = std::move(previousLevels[levelIndex]._grid);
		else
			level._grid = std::make_unique<RegularGrid>();

		const BuildResult levelBuild = level._grid->build(cloud, schemaForLevel(schema, level._leafCapacity), gridOptions);
		level._cellCount = level._grid->cellCount();
		level._dimensions = level._grid->dimensions();

		result._uploadTimeMs += levelBuild._uploadTimeMs;
		result._gpuBuildTimeMs += levelBuild._gpuBuildTimeMs;
		result._gpuMemoryBytes += levelBuild._gpuMemoryBytes;
		_state->_totalCellCount += levelBuild._metrics._numNodes;
		totalLeaves += levelBuild._metrics._numLeaves;
		totalLeafOccupancyWeight += levelBuild._metrics._averageLeafOccupancy * static_cast<double>(levelBuild._metrics._numLeaves);
		maxLeafOccupancy = std::max(maxLeafOccupancy, levelBuild._metrics._maxLeafOccupancy);

		_state->_levels.push_back(std::move(level));
	}

	_state->_memoryBytes = result._gpuMemoryBytes;
	_state->_ready = true;

	result._metrics._buildTimeMs = result._gpuBuildTimeMs;
	result._metrics._numNodes = _state->_totalCellCount;
	result._metrics._numLeaves = totalLeaves;
	result._metrics._indexedPoints = cloud.size();
	result._metrics._maxDepth = levelCount;
	result._metrics._averageLeafOccupancy = totalLeaves > 0
		? totalLeafOccupancyWeight / static_cast<double>(totalLeaves)
		: 0.0;
	result._metrics._maxLeafOccupancy = maxLeafOccupancy;
	result._metrics._memoryEstimateBytes = result._gpuMemoryBytes;
	return result;
}

PointGpu::QueryResult PointGpu::HGrid::query(const std::vector<Query>& queries, const Options& options) const
{
	if (!_state || !_state->_ready || _state->_levels.empty() || queries.empty())
		return {};

	QueryResult result;
	result._samples.resize(queries.size());

	std::vector<std::vector<Query>> queriesByLevel(_state->_levels.size());
	std::vector<std::vector<size_t>> indicesByLevel(_state->_levels.size());
	for (size_t queryIndex = 0; queryIndex < queries.size(); ++queryIndex)
	{
		const Query& query = queries[queryIndex];
		if (query._type == QueryType::Radius)
			++result._radiusQueries;
		else if (query._type == QueryType::CountRange)
			++result._countRangeQueries;
		else if (query._type == QueryType::Knn)
			++result._knnQueries;
		else
			++result._rangeQueries;

		size_t bestLevel = 0;
		double bestCost = std::numeric_limits<double>::infinity();
		for (size_t levelIndex = 0; levelIndex < _state->_levels.size(); ++levelIndex)
		{
			const DeviceState::Level& level = _state->_levels[levelIndex];
			const size_t visited = estimatedVisitedCells(query, _state->_bounds, level._dimensions);
			const double averageOccupancy = level._cellCount > 0
				? static_cast<double>(_state->_pointCount) / static_cast<double>(level._cellCount)
				: static_cast<double>(_state->_pointCount);
			const double cost = static_cast<double>(visited) * (1.0 + averageOccupancy);
			if (cost < bestCost)
			{
				bestCost = cost;
				bestLevel = levelIndex;
			}
		}

		queriesByLevel[bestLevel].push_back(query);
		indicesByLevel[bestLevel].push_back(queryIndex);
	}

	PointGpu::Options gridOptions = regularGridOptions(options);
	gridOptions._device = _state->_device;
	for (size_t levelIndex = 0; levelIndex < _state->_levels.size(); ++levelIndex)
	{
		if (queriesByLevel[levelIndex].empty())
			continue;

		const QueryResult levelResult = _state->_levels[levelIndex]._grid->query(queriesByLevel[levelIndex], gridOptions);
		result._gpuQueryTimeMs += levelResult._gpuQueryTimeMs;
		for (size_t sampleIndex = 0; sampleIndex < levelResult._samples.size(); ++sampleIndex)
			result._samples[indicesByLevel[levelIndex][sampleIndex]] = levelResult._samples[sampleIndex];
	}

	result._metrics = summarizeGpuSamples(result._samples);
	return result;
}

bool PointGpu::HGrid::built() const
{
	return _state && _state->_ready;
}

size_t PointGpu::HGrid::pointCount() const
{
	return _state ? _state->_pointCount : 0;
}

size_t PointGpu::HGrid::levelCount() const
{
	return _state ? _state->_levels.size() : 0;
}

size_t PointGpu::HGrid::cellCount() const
{
	return _state ? _state->_totalCellCount : 0;
}

glm::uvec3 PointGpu::HGrid::dimensions(size_t level) const
{
	if (!_state || level >= _state->_levels.size())
		return glm::uvec3(0);
	return _state->_levels[level]._dimensions;
}

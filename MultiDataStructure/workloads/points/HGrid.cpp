#include "../../stdafx.h"
#include "HGrid.h"

#include "PointSpatialIndex.h"
#include "RegularGrid.h"

// Upper bound on cells in a single RegularGrid level, derived the same way
// `chooseGridShape` derives `targetCells`: divUp(pointCount, leafCapacity). The actual cell
// count after `chooseGridShape` rounds up may exceed this by at most a small factor (one
// extra cell per active axis), so we add a 1.25x safety margin.
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
	size_t leafCapacity = schema.buildPolicy.leafCapacity;
	if (!schema.levels.empty() && schema.levels.front().leafCapacity > 0)
		leafCapacity = schema.levels.front().leafCapacity;

	return std::max<size_t>(1, leafCapacity);
}

// Was clamped to 4 historically to keep memory bounded on cards we couldn't measure. With
// the VRAM auto-detection + aggregate memory pre-check now in place, the build path throws
// early when a schema would exceed the budget, so the cap can be relaxed. 12 lets `hg6` /
// `hg9` / `hg12` schemas actually realize their requested level count (previously they
// silently truncated to 4 and the schema name lied about reality).
static constexpr size_t MaxHGridLevels = 12;

static size_t hgridLevelCountForSchema(const SchemaConfig& schema)
{
	size_t configuredDepth = schema.buildPolicy.maxDepth;
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
	schema.buildPolicy.leafCapacity = leafCapacity;
	schema.buildPolicy.minPrimitivesToSplit = std::max<size_t>(2, leafCapacity / 4);
	if (schema.levels.empty())
	{
		SchemaLevelConfig level;
		level.type = MultiDataStructure::DataStructureLevel::BvhNode;
		level.typeName = "BVH";
		level.numLevels = 1;
		level.leafCapacity = leafCapacity;
		level.minPrimitivesToSplit = schema.buildPolicy.minPrimitivesToSplit;
		schema.levels.push_back(level);
	}
	else
	{
		schema.levels.front().leafCapacity = leafCapacity;
		schema.levels.front().minPrimitivesToSplit = schema.buildPolicy.minPrimitivesToSplit;
	}
	return schema;
}

static PointGpu::Options regularGridOptions(const PointGpu::Options& options)
{
	PointGpu::Options gridOptions = options;
	gridOptions.builder = "regular_grid";
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
	float queryMinX = query.bounds.min().x;
	float queryMinY = query.bounds.min().y;
	float queryMinZ = query.bounds.min().z;
	float queryMaxX = query.bounds.max().x;
	float queryMaxY = query.bounds.max().y;
	float queryMaxZ = query.bounds.max().z;
	if (query.type == PointGpu::QueryType::Radius)
	{
		queryMinX = query.center.x - query.radius;
		queryMinY = query.center.y - query.radius;
		queryMinZ = query.center.z - query.radius;
		queryMaxX = query.center.x + query.radius;
		queryMaxY = query.center.y + query.radius;
		queryMaxZ = query.center.z + query.radius;
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
		stats.visitedNodes = sample.visitedNodes;
		stats.testedPoints = sample.testedPoints;
		stats.returnedPoints = sample.returnedPoints;
		stats.elapsedMs = sample.elapsedMs;
		cpuSamples.push_back(stats);
	}
	return Experiments::summarizeQueryStats(cpuSamples);
}

struct PointGpu::HGrid::DeviceState
{
	struct Level
	{
		std::unique_ptr<RegularGrid> grid;
		size_t leafCapacity = 1;
		size_t cellCount = 0;
		glm::uvec3 dimensions = glm::uvec3(1);
	};

	std::vector<Level> levels;
	size_t pointCount = 0;
	size_t totalCellCount = 0;
	size_t memoryBytes = 0;
	int device = 0;
	const PointCloud* cloud = nullptr;
	AABB bounds;
	bool ready = false;
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
	const std::string builder = options.builder.empty() ? "hgrid" : options.builder;
	if (builder != "hgrid" && builder != "hierarchical_grid" && builder != "hierarchicalgrid")
		throw std::runtime_error("HGrid evaluator supports --cuda-builder hgrid or hierarchical_grid.");

	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("HGrid currently supports up to 2^32 - 1 points.");

	std::string availabilityError;
	if (!isAvailable(&availabilityError))
		throw std::runtime_error("CUDA point evaluator is unavailable: " + availabilityError);

	const int count = deviceCount();
	const int device = options.device >= 0 ? std::min(options.device, count - 1) : 0;
	const size_t baseLeafCapacity = leafCapacityForSchema(schema);
	const size_t levelCount = hgridLevelCountForSchema(schema);
	const bool canReuseLevels =
		_state->ready &&
		_state->cloud == &cloud &&
		_state->pointCount == cloud.size() &&
		_state->device == device &&
		_state->levels.size() == levelCount;
	std::vector<DeviceState::Level> previousLevels;
	if (canReuseLevels)
	{
		previousLevels = std::move(_state->levels);
		_state->levels.clear();
		_state->totalCellCount = 0;
		_state->memoryBytes = 0;
		_state->ready = false;
	}
	else
	{
		release();
	}

	_state->pointCount = cloud.size();
	_state->cloud = &cloud;
	_state->bounds = cloud.bounds();
	_state->device = device;

	BuildResult result;
	result.device = _state->device;
	result.builder = "hgrid";
	if (cloud.empty())
		return result;

	// Aggregate memory pre-check. Each sub-level RegularGrid checks `options.memoryBudgetMb` on
	// its own, so configurations where each level fits the budget but the *sum* of all levels
	// exceeds it slip through and fill GPU memory one allocation at a time. Estimate the total
	// here so pathological cases throw immediately instead of taking minutes / hanging the GPU.
	if (options.memoryBudgetMb > 0)
	{
		// Per-point overhead in the RegularGrid base buffers (DevicePoint + 4 uint32 keys/indices
		// + radix-sort scratch buffer). The scratch is dataset-dependent so we approximate it
		// with 1.5x the point buffer size, which matches CUB's worst case for 100M-point sorts.
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
		const size_t budgetBytes = options.memoryBudgetMb * size_t(1024) * size_t(1024);
		if (aggregateBytes > budgetBytes)
		{
			throw std::runtime_error("HGrid aggregate memory (" + std::to_string(aggregateBytes / (1024 * 1024))
				+ " MB across " + std::to_string(levelCount) + " levels) exceeds the configured budget ("
				+ std::to_string(options.memoryBudgetMb) + " MB). Raise --cuda-memory-budget-mb or use larger leaf capacities.");
		}
	}

	_state->levels.reserve(levelCount);

	PointGpu::Options gridOptions = regularGridOptions(options);
	gridOptions.device = _state->device;

	size_t maxLeafOccupancy = 0;
	size_t totalLeaves = 0;
	double totalLeafOccupancyWeight = 0.0;
	for (size_t levelIndex = 0; levelIndex < levelCount; ++levelIndex)
	{
		DeviceState::Level level;
		level.leafCapacity = scaledLeafCapacity(baseLeafCapacity, levelIndex, levelCount);
		if (levelIndex < previousLevels.size() && previousLevels[levelIndex].grid)
			level.grid = std::move(previousLevels[levelIndex].grid);
		else
			level.grid = std::make_unique<RegularGrid>();

		const BuildResult levelBuild = level.grid->build(cloud, schemaForLevel(schema, level.leafCapacity), gridOptions);
		level.cellCount = level.grid->cellCount();
		level.dimensions = level.grid->dimensions();

		result.uploadTimeMs += levelBuild.uploadTimeMs;
		result.gpuBuildTimeMs += levelBuild.gpuBuildTimeMs;
		result.gpuMemoryBytes += levelBuild.gpuMemoryBytes;
		_state->totalCellCount += levelBuild.metrics.numNodes;
		totalLeaves += levelBuild.metrics.numLeaves;
		totalLeafOccupancyWeight += levelBuild.metrics.averageLeafOccupancy * static_cast<double>(levelBuild.metrics.numLeaves);
		maxLeafOccupancy = std::max(maxLeafOccupancy, levelBuild.metrics.maxLeafOccupancy);

		_state->levels.push_back(std::move(level));
	}

	_state->memoryBytes = result.gpuMemoryBytes;
	_state->ready = true;

	result.metrics.buildTimeMs = result.gpuBuildTimeMs;
	result.metrics.numNodes = _state->totalCellCount;
	result.metrics.numLeaves = totalLeaves;
	result.metrics.indexedPoints = cloud.size();
	result.metrics.maxDepth = levelCount;
	result.metrics.averageLeafOccupancy = totalLeaves > 0
		? totalLeafOccupancyWeight / static_cast<double>(totalLeaves)
		: 0.0;
	result.metrics.maxLeafOccupancy = maxLeafOccupancy;
	result.metrics.memoryEstimateBytes = result.gpuMemoryBytes;
	return result;
}

PointGpu::QueryResult PointGpu::HGrid::query(const std::vector<Query>& queries, const Options& options) const
{
	if (!_state || !_state->ready || _state->levels.empty() || queries.empty())
		return {};

	QueryResult result;
	result.samples.resize(queries.size());

	std::vector<std::vector<Query>> queriesByLevel(_state->levels.size());
	std::vector<std::vector<size_t>> indicesByLevel(_state->levels.size());
	for (size_t queryIndex = 0; queryIndex < queries.size(); ++queryIndex)
	{
		const Query& query = queries[queryIndex];
		if (query.type == QueryType::Radius)
			++result.radiusQueries;
		else if (query.type == QueryType::CountRange)
			++result.countRangeQueries;
		else if (query.type == QueryType::Knn)
			++result.knnQueries;
		else
			++result.rangeQueries;

		size_t bestLevel = 0;
		double bestCost = std::numeric_limits<double>::infinity();
		for (size_t levelIndex = 0; levelIndex < _state->levels.size(); ++levelIndex)
		{
			const DeviceState::Level& level = _state->levels[levelIndex];
			const size_t visited = estimatedVisitedCells(query, _state->bounds, level.dimensions);
			const double averageOccupancy = level.cellCount > 0
				? static_cast<double>(_state->pointCount) / static_cast<double>(level.cellCount)
				: static_cast<double>(_state->pointCount);
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
	gridOptions.device = _state->device;
	for (size_t levelIndex = 0; levelIndex < _state->levels.size(); ++levelIndex)
	{
		if (queriesByLevel[levelIndex].empty())
			continue;

		const QueryResult levelResult = _state->levels[levelIndex].grid->query(queriesByLevel[levelIndex], gridOptions);
		result.gpuQueryTimeMs += levelResult.gpuQueryTimeMs;
		for (size_t sampleIndex = 0; sampleIndex < levelResult.samples.size(); ++sampleIndex)
			result.samples[indicesByLevel[levelIndex][sampleIndex]] = levelResult.samples[sampleIndex];
	}

	result.metrics = summarizeGpuSamples(result.samples);
	return result;
}

bool PointGpu::HGrid::built() const
{
	return _state && _state->ready;
}

size_t PointGpu::HGrid::pointCount() const
{
	return _state ? _state->pointCount : 0;
}

size_t PointGpu::HGrid::levelCount() const
{
	return _state ? _state->levels.size() : 0;
}

size_t PointGpu::HGrid::cellCount() const
{
	return _state ? _state->totalCellCount : 0;
}

glm::uvec3 PointGpu::HGrid::dimensions(size_t level) const
{
	if (!_state || level >= _state->levels.size())
		return glm::uvec3(0);
	return _state->levels[level].dimensions;
}

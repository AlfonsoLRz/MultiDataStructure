#include "../stdafx.h"
#include "SchemaSearch.h"

#include "SchemaSelector.h"
#include "../workloads/points/BIH.h"
#include "../workloads/points/HGrid.h"
#include "../workloads/points/KDTree.h"
#include "../workloads/points/LBVH.h"
#include "../workloads/points/MixedTree.h"
#include "../workloads/points/Octree.h"
#include "../workloads/points/PointCloud.h"
#include "../workloads/points/PointSpatialIndex.h"
#include "../workloads/points/QuadTree.h"
#include "../workloads/points/RegularGrid.h"
#include "../workloads/points/SyntheticPointClouds.h"
#include "EvaluationCache.h"
#include "SurrogateAcquisition.h"
#include "ThresholdRefiner.h"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

#include <map>
#include <set>

static constexpr double EPSILON = 1e-9;

static const std::vector<std::string>& defaultSchemaPaths()
{
	static const std::vector<std::string> paths = {
		"configs/schemas/quadtree.json",
		"configs/schemas/octree.json",
		"configs/schemas/kdtree.json",
		"configs/schemas/bvh.json",
		"configs/schemas/quadtree_octree.json",
		"configs/schemas/octree_kdtree.json",
		"configs/schemas/urban_hybrid.json",
	};
	return paths;
}

static const std::vector<std::string>& defaultWorkloadPaths()
{
	static const std::vector<std::string> paths = {
		"configs/workloads/volume_small_medium.json",
	};
	return paths;
}

struct SearchDataset
{
	std::string _name;
	std::string _source;
	PointCloud _cloud;
};

struct WorkloadRun
{
	Experiments::QueryMetrics _metrics;
	Experiments::QueryMetrics _rangeMetrics;
	Experiments::QueryMetrics _countRangeMetrics;
	Experiments::QueryMetrics _radiusMetrics;
	Experiments::QueryMetrics _knnMetrics;
	std::vector<PointSpatialIndex::QueryStats> _samples;
	size_t _rangeQueries = 0;
	size_t _countRangeQueries = 0;
	size_t _radiusQueries = 0;
	size_t _knnQueries = 0;
	std::map<std::string, Experiments::QueryMetrics> _stratumMetrics;
	std::string _stratumSummary;
	double _gpuQueryMs = 0.0;
};

enum class PreparedQueryKind
{
	Range,
	Radius,
	Knn,
};

struct PreparedCpuQuery
{
	PreparedQueryKind _kind = PreparedQueryKind::Range;
	AABB _bounds;
	glm::vec3 center = glm::vec3(0.0f);
	float _radius = 0.0f;
	std::string _stratum;
};

struct PreparedWorkload
{
	std::vector<PreparedCpuQuery> _cpuQueries;
	std::vector<PointGpu::Query> _cudaQueries;
	std::vector<std::string> _cudaStrata;
	size_t _rangeQueries = 0;
	size_t _countRangeQueries = 0;
	size_t _radiusQueries = 0;
	size_t _knnQueries = 0;
};

static std::string csvEscape(const std::string& value);
static void createParentDirectory(const std::string& filename);

struct DatasetContext
{
	const SearchDataset* _dataset = nullptr;
	Experiments::PointCloudFeatures _features;
	std::vector<PreparedWorkload> _preparedWorkloads;
};

struct EvaluatedCandidate
{
	Experiments::SchemaCandidate _candidate;
	std::vector<Experiments::SchemaSearchRecord> _records;
	double aggregateScore = std::numeric_limits<double>::infinity();
	// NSGA-II tagging: paretoFront -1 = unranked, 0 = current front; crowdingDistance sums normalized objective gaps to same-front neighbors (larger = more isolated).
	int _paretoFront = -1;
	double _crowdingDistance = 0.0;
};

template <typename TIndex>
struct CudaCachedBuilder
{
	std::unique_ptr<TIndex> _index;
	std::string _lastSignature;
	PointGpu::BuildResult _lastResult;
	bool _hasResult = false;
};

struct CudaIndexCacheEntry
{
	CudaCachedBuilder<PointGpu::BIH> _bih;
	CudaCachedBuilder<PointGpu::HGrid> _hgrid;
	CudaCachedBuilder<PointGpu::KDTree> _kdTree;
	CudaCachedBuilder<PointGpu::LBVH> _lbvh;
	CudaCachedBuilder<PointGpu::MixedTree> _mixedTree;
	CudaCachedBuilder<PointGpu::Octree> _octree;
	CudaCachedBuilder<PointGpu::QuadTree> _quadTree;
	CudaCachedBuilder<PointGpu::RegularGrid> _regularGrid;
	size_t _buildHits = 0;
	size_t _buildMisses = 0;
};

using CudaIndexCache = std::unordered_map<const SearchDataset*, CudaIndexCacheEntry>;

template <typename TIndex>
static PointGpu::BuildResult cudaBuildOrReuse(
	CudaCachedBuilder<TIndex>& slot,
	const std::string& currentSignature,
	const PointCloud& cloud,
	const SchemaConfig& schemaConfig,
	const PointGpu::Options& cudaOptions,
	CudaIndexCacheEntry& parentEntry,
	TIndex*& outIndex)
{
	if (!slot._index)
		slot._index = std::make_unique<TIndex>();
	outIndex = slot._index.get();

	if (slot._hasResult && slot._lastSignature == currentSignature)
	{
		++parentEntry._buildHits;
		return slot._lastResult;
	}

	slot._lastResult = slot._index->build(cloud, schemaConfig, cudaOptions);
	slot._lastSignature = currentSignature;
	slot._hasResult = true;
	++parentEntry._buildMisses;
	return slot._lastResult;
}

static bool pathExists(const std::filesystem::path& path)
{
	std::error_code error;
	return std::filesystem::exists(path, error);
}

static std::filesystem::path resolveExistingPath(const std::string& filename)
{
	const std::filesystem::path configuredPath(filename);
	if (pathExists(configuredPath) || configuredPath.is_absolute())
		return configuredPath;

	std::error_code error;
	std::filesystem::path current = std::filesystem::absolute(std::filesystem::current_path(), error);
	if (error)
		return configuredPath;

	for (;;)
	{
		const std::filesystem::path candidate = (current / configuredPath).lexically_normal();
		if (pathExists(candidate))
			return candidate;

		if (!current.has_parent_path() || current == current.parent_path())
			break;

		current = current.parent_path();
	}

	return configuredPath;
}

static size_t asSize(const boost::json::object& object, const char* key, size_t fallback)
{
	if (const boost::json::value* value = object.if_contains(key))
	{
		if (value->is_int64())
			return static_cast<size_t>(value->as_int64());
		if (value->is_uint64())
			return static_cast<size_t>(value->as_uint64());
		if (value->is_double())
			return static_cast<size_t>(value->as_double());
	}

	return fallback;
}

static double asDouble(const boost::json::object& object, const char* key, double fallback)
{
	if (const boost::json::value* value = object.if_contains(key))
	{
		if (value->is_double())
			return value->as_double();
		if (value->is_int64())
			return static_cast<double>(value->as_int64());
		if (value->is_uint64())
			return static_cast<double>(value->as_uint64());
	}

	return fallback;
}

static bool asBool(const boost::json::object& object, const char* key, bool fallback)
{
	if (const boost::json::value* value = object.if_contains(key))
	{
		if (value->is_bool())
			return value->as_bool();
		if (value->is_int64())
			return value->as_int64() != 0;
		if (value->is_uint64())
			return value->as_uint64() != 0;
	}

	return fallback;
}

static std::string asString(const boost::json::object& object, const char* key, const std::string& fallback = {})
{
	if (const boost::json::value* value = object.if_contains(key))
	{
		if (value->is_string())
			return std::string(value->as_string().c_str());
	}

	return fallback;
}

static bool scoreWeightsAreDefault(const Experiments::ScoreWeights& weights)
{
	constexpr double epsilon = 1e-12;
	return std::abs(weights._lambdaLatency - 1.0) <= epsilon &&
		std::abs(weights._lambdaBuild) <= epsilon &&
		std::abs(weights._lambdaMemory) <= epsilon &&
		std::abs(weights._lambdaImbalance) <= epsilon &&
		!weights._useVisitProxy &&
		std::abs(weights._visitProxyAlpha - 0.1) <= epsilon;
}

static Experiments::ScoreWeights effectiveScoreWeights(
	const Experiments::WorkloadProfile& workload,
	const Experiments::SchemaSearchOptions& options)
{
	if (options._scoreWeightsOverride || !scoreWeightsAreDefault(options._weights))
		return options._weights;
	if (workload._hasScoreWeights)
		return workload._scoreWeights;
	return options._weights;
}

static double elapsedMilliseconds(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
{
	return std::chrono::duration<double, std::milli>(end - begin).count();
}

static float randomFloat(std::mt19937& rng, float minValue, float maxValue)
{
	if (minValue >= maxValue)
		return minValue;

	std::uniform_real_distribution<float> distribution(minValue, maxValue);
	return distribution(rng);
}

static glm::vec3 randomPointInBounds(std::mt19937& rng, const AABB& bounds)
{
	const glm::vec3 min = bounds.min();
	const glm::vec3 max = bounds.max();
	return glm::vec3(
		randomFloat(rng, min.x, max.x),
		randomFloat(rng, min.y, max.y),
		randomFloat(rng, min.z, max.z));
}

static void normalizeScaleRange(double& minScale, double& maxScale)
{
	minScale = std::max(0.0, minScale);
	maxScale = std::max(0.0, maxScale);
	if (maxScale < minScale)
		std::swap(minScale, maxScale);
}

static void parseScaleRange(const boost::json::object& object, const char* key, double& minScale, double& maxScale)
{
	const boost::json::value* value = object.if_contains(key);
	if (!value)
		return;
	if (!value->is_object())
		throw std::runtime_error(std::string("Workload query scale '") + key + "' must be an object");

	const boost::json::object& range = value->as_object();
	minScale = asDouble(range, "min", minScale);
	maxScale = asDouble(range, "max", maxScale);
}

static Experiments::ScoreWeights parseScoreWeightsObject(const boost::json::object& object)
{
	Experiments::ScoreWeights weights;
	weights._lambdaLatency = asDouble(object, "latency", weights._lambdaLatency);
	weights._lambdaLatency = asDouble(object, "lambdaLatency", weights._lambdaLatency);
	weights._lambdaBuild = asDouble(object, "buildTime", weights._lambdaBuild);
	weights._lambdaBuild = asDouble(object, "build", weights._lambdaBuild);
	weights._lambdaBuild = asDouble(object, "lambdaBuild", weights._lambdaBuild);
	weights._lambdaMemory = asDouble(object, "memory", weights._lambdaMemory);
	weights._lambdaMemory = asDouble(object, "lambdaMemory", weights._lambdaMemory);
	weights._lambdaImbalance = asDouble(object, "imbalance", weights._lambdaImbalance);
	weights._lambdaImbalance = asDouble(object, "lambdaImbalance", weights._lambdaImbalance);
	weights._useVisitProxy = asBool(object, "useVisitProxy", weights._useVisitProxy);
	weights._useVisitProxy = asBool(object, "visitProxy", weights._useVisitProxy);
	weights._visitProxyAlpha = asDouble(object, "visitProxyAlpha", weights._visitProxyAlpha);

	const double visitedNodes = asDouble(object, "visitedNodes", 0.0);
	const double testedPoints = asDouble(object, "testedPoints", 0.0);
	if (visitedNodes > 0.0 || testedPoints > 0.0)
	{
		weights._useVisitProxy = true;
		if (visitedNodes > 0.0)
			weights._visitProxyAlpha = std::max(0.0, testedPoints) / visitedNodes;
		else if (testedPoints > 0.0)
			weights._visitProxyAlpha = testedPoints;
	}

	weights._lambdaLatency = std::max(0.0, weights._lambdaLatency);
	weights._lambdaBuild = std::max(0.0, weights._lambdaBuild);
	weights._lambdaMemory = std::max(0.0, weights._lambdaMemory);
	weights._lambdaImbalance = std::max(0.0, weights._lambdaImbalance);
	weights._visitProxyAlpha = std::max(0.0, weights._visitProxyAlpha);
	return weights;
}

static AABB randomQueryBox(std::mt19937& rng, const PointCloud& cloud, const Experiments::WorkloadProfile& profile)
{
	const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.001f));
	const float scale = randomFloat(rng, static_cast<float>(profile._rangeScaleMin), static_cast<float>(profile._rangeScaleMax));
	const glm::vec3 halfExtent = glm::max(range * scale * 0.5f, glm::vec3(0.0005f));
	const glm::vec3 boundsMin = cloud.bounds().min();
	const glm::vec3 boundsMax = cloud.bounds().max();
	glm::vec3 center(0.0f);
	for (glm::uint axis = 0; axis < 3; ++axis)
	{
		const float minCenter = boundsMin[axis] + halfExtent[axis];
		const float maxCenter = boundsMax[axis] - halfExtent[axis];
		center[axis] = minCenter <= maxCenter
			? randomFloat(rng, minCenter, maxCenter)
			: (boundsMin[axis] + boundsMax[axis]) * 0.5f;
	}
	return AABB(center - halfExtent, center + halfExtent);
}

static float randomQueryRadius(std::mt19937& rng, const PointCloud& cloud, const Experiments::WorkloadProfile& profile)
{
	const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.001f));
	const float largestRange = std::max({ range.x, range.y, range.z, 1.0f });
	return largestRange * randomFloat(rng, static_cast<float>(profile._radiusScaleMin), static_cast<float>(profile._radiusScaleMax));
}

static AABB queryBoxAtScale(const glm::vec3& center, const PointCloud& cloud, double scale)
{
	const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.001f));
	const glm::vec3 halfExtent = glm::max(range * static_cast<float>(scale) * 0.5f, glm::vec3(0.0005f));
	return AABB(center - halfExtent, center + halfExtent);
}

static glm::vec3 sampledCloudPoint(std::mt19937& rng, const PointCloud& cloud)
{
	if (cloud.empty())
		return cloud.bounds().center();
	std::uniform_int_distribution<size_t> distribution(0, cloud.size() - 1);
	return cloud.points()[distribution(rng)].position;
}

static glm::vec3 boundaryPoint(std::mt19937& rng, const PointCloud& cloud)
{
	glm::vec3 point = randomPointInBounds(rng, cloud.bounds());
	const glm::vec3 minBound = cloud.bounds().min();
	const glm::vec3 maxBound = cloud.bounds().max();
	std::uniform_int_distribution<int> axisDistribution(0, 2);
	std::uniform_int_distribution<int> sideDistribution(0, 1);
	const int axis = axisDistribution(rng);
	point[axis] = sideDistribution(rng) == 0 ? minBound[axis] : maxBound[axis];
	return point;
}

static glm::vec3 outsidePoint(std::mt19937& rng, const PointCloud& cloud)
{
	const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(1.0f));
	glm::vec3 point = randomPointInBounds(rng, cloud.bounds());
	const glm::vec3 maxBound = cloud.bounds().max();
	std::uniform_int_distribution<int> axisDistribution(0, 2);
	const int axis = axisDistribution(rng);
	point[axis] = maxBound[axis] + range[axis] * 0.25f;
	return point;
}

static std::string lowerCopy(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	});
	return value;
}

static bool useCudaEvaluator(const Experiments::SchemaSearchOptions& options)
{
	const std::string evaluator = lowerCopy(options._evaluator);
	return evaluator == "cuda" || evaluator == "gpu";
}

enum class PrimitiveProfile
{
	QueryMinimalCpu,
	CudaQueryFull,
	All,
};

static PrimitiveProfile parsePrimitiveProfileInternal(const std::string& requestedProfile, bool cudaEvaluator)
{
	std::string profile = lowerCopy(requestedProfile);
	profile.erase(std::remove_if(profile.begin(), profile.end(), [](unsigned char c) {
		return c == '_' || c == '-' || c == ' ';
	}), profile.end());

	if (profile.empty() || profile == "auto")
		return cudaEvaluator ? PrimitiveProfile::CudaQueryFull : PrimitiveProfile::QueryMinimalCpu;
	if (profile == "queryminimalcpu" || profile == "cpuminimal" || profile == "minimal" || profile == "queryminimal")
		return PrimitiveProfile::QueryMinimalCpu;
	if (profile == "cudaqueryfull" || profile == "cudafull" || profile == "fullcuda")
		return PrimitiveProfile::CudaQueryFull;
	if (profile == "all" || profile == "exhaustive")
		return PrimitiveProfile::All;

	std::cerr << "Warning: unknown primitive profile '" << requestedProfile
		<< "'; using " << (cudaEvaluator ? "cuda_query_full" : "query_minimal_cpu") << '\n';
	return cudaEvaluator ? PrimitiveProfile::CudaQueryFull : PrimitiveProfile::QueryMinimalCpu;
}

static std::string primitiveProfileName(PrimitiveProfile profile)
{
	switch (profile)
	{
	case PrimitiveProfile::CudaQueryFull:
		return "cuda_query_full";
	case PrimitiveProfile::All:
		return "all";
	default:
		return "query_minimal_cpu";
	}
}

static bool queryMinimalPrimitiveProfile(const Experiments::SchemaGenerationOptions& options)
{
	return parsePrimitiveProfileInternal(options._primitiveProfile, false) == PrimitiveProfile::QueryMinimalCpu;
}

static std::string scoreModeForWeights(const Experiments::ScoreWeights& weights)
{
	if (weights._useVisitProxy)
		return "visit_proxy";
	return scoreWeightsAreDefault(weights) ? "latency" : "weighted_latency";
}

static bool isRegularGridBuilder(const std::string& builder)
{
	return builder == "regular_grid" || builder == "regulargrid" || builder == "grid" || builder == "uniform_grid";
}

static bool isHGridBuilder(const std::string& builder)
{
	return builder == "hgrid" || builder == "hierarchical_grid" || builder == "hierarchicalgrid";
}

static bool isBIHBuilder(const std::string& builder)
{
	return builder == "bih" || builder == "interval_hierarchy" || builder == "binary_interval_hierarchy";
}

static bool isKDTreeBuilder(const std::string& builder)
{
	return builder == "kdtree" || builder == "kd_tree" || builder == "kd";
}

static bool isOctreeBuilder(const std::string& builder)
{
	return builder == "octree" || builder == "ot" ||
		builder == "karras_octree" || builder == "morton_octree" ||
		builder == "octree_karras" || builder == "octree_morton";
}

static bool isKarrasOctreeBuilder(const std::string& builder)
{
	return builder == "karras_octree" || builder == "morton_octree" ||
		builder == "octree_karras" || builder == "octree_morton";
}

static bool isQuadTreeBuilder(const std::string& builder)
{
	return builder == "quadtree" || builder == "quad_tree" || builder == "qt";
}

static bool isMixedBuilder(const std::string& builder)
{
	return builder == "mixed" || builder == "hybrid";
}

static std::string canonicalCudaBuilder(std::string builder)
{
	builder = lowerCopy(builder);
	if (builder.empty())
		return "lbvh";
	if (isBIHBuilder(builder))
		return "bih";
	if (isHGridBuilder(builder))
		return "hgrid";
	if (isKDTreeBuilder(builder))
		return "kdtree";
	if (isOctreeBuilder(builder))
		return isKarrasOctreeBuilder(builder) ? "karras_octree" : "octree";
	if (isQuadTreeBuilder(builder))
		return "quadtree";
	if (isRegularGridBuilder(builder))
		return "regular_grid";
	if (isMixedBuilder(builder))
		return "mixed";
	return builder;
}

static std::string cudaBuilderDisplayName(const std::string& builder)
{
	const std::string canonical = canonicalCudaBuilder(builder);
	if (canonical == "bih")
		return "BIH";
	if (canonical == "hgrid")
		return "HGrid";
	if (canonical == "kdtree")
		return "KDTree";
	if (canonical == "lbvh")
		return "LBVH";
	if (canonical == "octree")
		return "Octree";
	if (canonical == "karras_octree")
		return "KarrasOctree";
	if (canonical == "quadtree")
		return "QuadTree";
	if (canonical == "regular_grid")
		return "RegularGrid";
	if (canonical == "mixed")
		return "Mixed";
	return builder;
}

static std::string cudaDeviceDescription(const PointGpu::Options& cudaOptions)
{
	int count = 0;
	const cudaError_t countResult = cudaGetDeviceCount(&count);
	if (countResult != cudaSuccess)
		return std::string("unavailable (") + cudaGetErrorString(countResult) + ")";
	if (count <= 0)
		return "unavailable (no CUDA devices)";

	int device = 0;
	if (cudaOptions._device >= 0)
	{
		device = std::min(cudaOptions._device, count - 1);
	}
	else
	{
		const cudaError_t currentResult = cudaGetDevice(&device);
		if (currentResult != cudaSuccess || device < 0 || device >= count)
			device = 0;
	}

	cudaDeviceProp properties{};
	const cudaError_t propertyResult = cudaGetDeviceProperties(&properties, device);
	if (propertyResult != cudaSuccess)
		return std::string("unavailable (") + cudaGetErrorString(propertyResult) + ")";

	std::ostringstream out;
	if (cudaOptions._device >= 0)
	{
		if (cudaOptions._device != device)
			out << cudaOptions._device << " -> ";
		out << device;
	}
	else
	{
		out << "default -> " << device;
	}
	size_t freeBytes = 0;
	size_t totalBytes = 0;
	const cudaError_t setResult = cudaSetDevice(device);
	(void)setResult;
	const cudaError_t memResult = cudaMemGetInfo(&freeBytes, &totalBytes);
	out << " (" << properties.name
		<< ", cc " << properties.major << '.' << properties.minor
		<< ", " << static_cast<size_t>(properties.totalGlobalMem / (1024 * 1024)) << " MB total";
	if (memResult == cudaSuccess && totalBytes > 0)
	{
		out << ", " << static_cast<size_t>(freeBytes / (1024 * 1024)) << " MB free";
	}
	out << ")";
	return out.str();
}

static int resolvedCudaDevice(const PointGpu::Options& cudaOptions)
{
	int count = 0;
	const cudaError_t countResult = cudaGetDeviceCount(&count);
	if (countResult != cudaSuccess)
		throw std::runtime_error(std::string("CUDA device query failed: ") + cudaGetErrorString(countResult));
	if (count <= 0)
		throw std::runtime_error("CUDA evaluator requested, but no CUDA devices are available.");

	if (cudaOptions._device >= 0)
		return std::min(cudaOptions._device, count - 1);

	int device = 0;
	const cudaError_t currentResult = cudaGetDevice(&device);
	if (currentResult != cudaSuccess || device < 0 || device >= count)
		return 0;
	return device;
}

static double warmUpCudaDevice(const PointGpu::Options& cudaOptions)
{
	const int device = resolvedCudaDevice(cudaOptions);
	const auto begin = std::chrono::steady_clock::now();
	const cudaError_t setResult = cudaSetDevice(device);
	if (setResult != cudaSuccess)
		throw std::runtime_error(std::string("CUDA device selection failed: ") + cudaGetErrorString(setResult));
	const cudaError_t warmupResult = cudaFree(nullptr);
	if (warmupResult != cudaSuccess)
		throw std::runtime_error(std::string("CUDA warm-up failed: ") + cudaGetErrorString(warmupResult));
	const auto end = std::chrono::steady_clock::now();
	return elapsedMilliseconds(begin, end);
}

// Total VRAM on the selected CUDA device, or 0 if unavailable; used for the header line and to auto-default memoryBudgetMb.
static size_t cudaTotalVramMb(int deviceHint)
{
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
		return 0;

	int device = 0;
	if (deviceHint >= 0)
		device = std::min(deviceHint, count - 1);
	else if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= count)
		device = 0;

	cudaDeviceProp properties{};
	if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
		return 0;
	return static_cast<size_t>(properties.totalGlobalMem / (1024 * 1024));
}

static PointGpu::Options cudaOptionsFrom(const Experiments::SchemaSearchOptions& options)
{
	PointGpu::Options cudaOptions;
	cudaOptions._device = options._cuda._device;
	cudaOptions._builder = canonicalCudaBuilder(options._cuda._builder);
	cudaOptions._queryBatchSize = options._cuda._queryBatchSize;
	cudaOptions._memoryBudgetMb = options._cuda._memoryBudgetMb;
	cudaOptions._knnBackend = options._cuda._knnBackend;
	// Auto-default the budget to 75% of total VRAM when the caller leaves it at 0, leaving headroom for the rest of the GPU stack.
	if (cudaOptions._memoryBudgetMb == 0)
	{
		const size_t totalMb = cudaTotalVramMb(cudaOptions._device);
		if (totalMb > 0)
			cudaOptions._memoryBudgetMb = (totalMb * 3) / 4;
	}
	return cudaOptions;
}

static std::string datasetNameFromPath(const std::string& inputPath)
{
	const std::filesystem::path path(inputPath);
	const std::string stem = path.stem().string();
	return stem.empty() ? "points" : stem;
}

static std::string normalizedLevelTypeName(std::string value)
{
	value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
		return std::isspace(c) || c == '_' || c == '-';
	}), value.end());
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	});
	return value;
}

static bool isBIHLevelName(const std::string& typeName)
{
	const std::string normalized = normalizedLevelTypeName(typeName);
	return normalized == "bih" || normalized == "binaryintervalhierarchy" || normalized == "intervalhierarchy";
}

static bool isKarrasOctreeLevelName(const std::string& typeName)
{
	const std::string normalized = normalizedLevelTypeName(typeName);
	return normalized == "karrasoctree" || normalized == "mortonoctree" || normalized == "octreekarras" || normalized == "octreemorton";
}

static bool isLBVHLevelName(const std::string& typeName)
{
	const std::string normalized = normalizedLevelTypeName(typeName);
	return normalized == "lbvh" || normalized == "linearbvh";
}

static bool isRegularGridLevelName(const std::string& typeName)
{
	const std::string normalized = normalizedLevelTypeName(typeName);
	return normalized == "regulargrid" || normalized == "uniformgrid" || normalized == "grid" || normalized == "grid3d";
}

static bool isHGridLevelName(const std::string& typeName)
{
	const std::string normalized = normalizedLevelTypeName(typeName);
	return normalized == "hgrid" || normalized == "hierarchicalgrid" || normalized == "hierarchicalgrid3d";
}

static std::string schemaTypeShortName(const SchemaLevelConfig& level)
{
	if (isBIHLevelName(level._typeName))
		return "bih";
	if (isKarrasOctreeLevelName(level._typeName))
		return "kot";
	if (isLBVHLevelName(level._typeName))
		return "lbvh";
	if (isRegularGridLevelName(level._typeName))
		return "rg";
	if (isHGridLevelName(level._typeName))
		return "hg";

	switch (level._type)
	{
	case MultiDataStructure::QuadTreeNode:
		return "qt";
	case MultiDataStructure::OctreeNode:
		return "ot";
	case MultiDataStructure::KDTreeNode:
		return "kd";
	case MultiDataStructure::BvhNode:
		return "bvh";
	default:
		return "x";
	}
}

static std::string levelJsonTypeName(const SchemaLevelConfig& level)
{
	return level._typeName.empty() ? Config::dataStructureLevelName(level._type) : level._typeName;
}

static std::string randomTypeNameForBase(
	MultiDataStructure::DataStructureLevel type,
	std::mt19937& rng,
	const Experiments::SchemaGenerationOptions& options)
{
	if (queryMinimalPrimitiveProfile(options))
		return Config::dataStructureLevelName(type);

	if (type == MultiDataStructure::KDTreeNode)
	{
		static const std::array<const char*, 2> names = { "KDTree", "BIH" };
		std::uniform_int_distribution<size_t> distribution(0, names.size() - 1);
		return names[distribution(rng)];
	}
	if (type == MultiDataStructure::OctreeNode)
	{
		static const std::array<const char*, 4> names = { "Octree", "KarrasOctree", "RegularGrid", "HGrid" };
		std::uniform_int_distribution<size_t> distribution(0, names.size() - 1);
		return names[distribution(rng)];
	}
	if (type == MultiDataStructure::BvhNode)
	{
		static const std::array<const char*, 2> names = { "BVH", "LBVH" };
		std::uniform_int_distribution<size_t> distribution(0, names.size() - 1);
		return names[distribution(rng)];
	}
	return Config::dataStructureLevelName(type);
}

static size_t clampPowerOfTwo(size_t value, size_t minValue, size_t maxValue)
{
	value = std::max<size_t>(1, value);
	size_t power = 1;
	while (power < value && power < (std::numeric_limits<size_t>::max() / 2))
		power *= 2;
	return std::clamp(power, minValue, maxValue);
}

static size_t randomPowerOfTwo(std::mt19937& rng, size_t minValue, size_t maxValue)
{
	minValue = std::max<size_t>(1, minValue);
	maxValue = std::max(minValue, maxValue);

	std::vector<size_t> values;
	for (size_t value = clampPowerOfTwo(minValue, minValue, maxValue); value <= maxValue; value *= 2)
	{
		values.push_back(value);
		if (value > maxValue / 2)
			break;
	}

	std::uniform_int_distribution<size_t> distribution(0, values.size() - 1);
	return values[distribution(rng)];
}

static MultiDataStructure::DataStructureLevel randomStructureType(
	std::mt19937& rng,
	std::optional<MultiDataStructure::DataStructureLevel> previous)
{
	static const std::array<MultiDataStructure::DataStructureLevel, 4> types = {
		MultiDataStructure::QuadTreeNode,
		MultiDataStructure::OctreeNode,
		MultiDataStructure::KDTreeNode,
		MultiDataStructure::BvhNode,
	};

	for (;;)
	{
		std::uniform_int_distribution<size_t> distribution(0, types.size() - 1);
		const MultiDataStructure::DataStructureLevel type = types[distribution(rng)];
		if (!previous.has_value() || type != previous.value())
			return type;
	}
}

template <typename T>
static void sortUniqueValues(std::vector<T>& values)
{
	std::sort(values.begin(), values.end());
	values.erase(std::unique(values.begin(), values.end()), values.end());
}

static void addUniqueSize(std::vector<size_t>& values, size_t value)
{
	if (value > 0)
		values.push_back(value);
}

static void addUniqueDouble(std::vector<double>& values, double value)
{
	if (std::isfinite(value) && value > 0.0)
		values.push_back(value);
}

template <typename T>
static T quantileValue(std::vector<T> values, double quantile)
{
	if (values.empty())
		return T{};

	sortUniqueValues(values);
	if (values.empty())
		return T{};

	const double clamped = std::clamp(quantile, 0.0, 1.0);
	const size_t index = static_cast<size_t>(std::round(clamped * static_cast<double>(values.size() - 1)));
	return values[std::min(index, values.size() - 1)];
}

static size_t nearestPowerOfTwoAtLeastOne(size_t value)
{
	value = std::max<size_t>(1, value);
	size_t power = 1;
	while (power < value && power < (std::numeric_limits<size_t>::max() / 2))
		power *= 2;
	const size_t lower = power > 1 ? power / 2 : power;
	return (value - lower) <= (power - value) ? lower : power;
}

static void addPointThresholdFamily(std::vector<size_t>& values, size_t rawValue)
{
	if (rawValue == 0)
		return;

	const size_t nearest = nearestPowerOfTwoAtLeastOne(rawValue);
	addUniqueSize(values, nearest);
	if (nearest > 1)
		addUniqueSize(values, nearest / 2);
	if (nearest <= std::numeric_limits<size_t>::max() / 2)
		addUniqueSize(values, nearest * 2);
}

static const std::vector<size_t>& fallbackPointThresholds()
{
	static const std::vector<size_t> values = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
	return values;
}

static const std::vector<double>& fallbackHeightRatioThresholds()
{
	static const std::vector<double> values = { 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00 };
	return values;
}

static const std::vector<double>& vectorOrFallback(const std::vector<double>& values, const std::vector<double>& fallback)
{
	return values.empty() ? fallback : values;
}

static const std::vector<size_t>& vectorOrFallback(const std::vector<size_t>& values, const std::vector<size_t>& fallback)
{
	return values.empty() ? fallback : values;
}

static size_t chooseSizeValue(
	std::mt19937& rng,
	const std::vector<size_t>& values,
	const std::vector<size_t>& fallback)
{
	const std::vector<size_t>& source = vectorOrFallback(values, fallback);
	std::uniform_int_distribution<size_t> distribution(0, source.size() - 1);
	return source[distribution(rng)];
}

static double chooseDoubleValue(
	std::mt19937& rng,
	const std::vector<double>& values,
	const std::vector<double>& fallback)
{
	const std::vector<double>& source = vectorOrFallback(values, fallback);
	std::uniform_int_distribution<size_t> distribution(0, source.size() - 1);
	return source[distribution(rng)];
}

static double chooseDoubleValue(std::mt19937& rng, const std::vector<double>& values)
{
	std::uniform_int_distribution<size_t> distribution(0, values.size() - 1);
	return values[distribution(rng)];
}

static AdaptiveLeafCapacityConfig randomAdaptiveLeafCapacity(
	std::mt19937& rng,
	size_t baseLeafCapacity)
{
	static const std::vector<double> weights = { 0.0, 0.5, 1.0, 1.5, 2.0 };
	static const std::vector<double> queryFactors = { 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 };

	AdaptiveLeafCapacityConfig config;
	config._enabled = true;
	config._minCapacity = std::max<size_t>(1, baseLeafCapacity / 4);
	config._maxCapacity = baseLeafCapacity > std::numeric_limits<size_t>::max() / 4
		? std::numeric_limits<size_t>::max()
		: std::max<size_t>(baseLeafCapacity, baseLeafCapacity * 4);
	config._densityWeight = chooseDoubleValue(rng, weights);
	config._anisotropyWeight = chooseDoubleValue(rng, weights);
	config._heightRatioWeight = chooseDoubleValue(rng, weights);
	config._queryMixFactor = chooseDoubleValue(rng, queryFactors);

	if (config._densityWeight == 0.0 &&
		config._anisotropyWeight == 0.0 &&
		config._heightRatioWeight == 0.0 &&
		config._queryMixFactor == 1.0)
	{
		config._densityWeight = 1.0;
	}

	return config;
}

static void normalizeAdaptiveLeafCapacity(SchemaLevelConfig& level)
{
	AdaptiveLeafCapacityConfig& adaptive = level._adaptiveLeafCapacity;
	if (!adaptive._enabled)
		return;

	const size_t baseCapacity = std::max<size_t>(1, level._leafCapacity);
	if (adaptive._minCapacity == 0)
		adaptive._minCapacity = std::max<size_t>(1, baseCapacity / 4);
	if (adaptive._maxCapacity == 0)
	{
		adaptive._maxCapacity = baseCapacity > std::numeric_limits<size_t>::max() / 4
			? std::numeric_limits<size_t>::max()
			: std::max<size_t>(baseCapacity, baseCapacity * 4);
	}
	if (adaptive._maxCapacity < adaptive._minCapacity)
		adaptive._maxCapacity = adaptive._minCapacity;

	adaptive._densityWeight = std::clamp(adaptive._densityWeight, -3.0, 3.0);
	adaptive._anisotropyWeight = std::clamp(adaptive._anisotropyWeight, -3.0, 3.0);
	adaptive._heightRatioWeight = std::clamp(adaptive._heightRatioWeight, -3.0, 3.0);
	if (!std::isfinite(adaptive._queryMixFactor) || adaptive._queryMixFactor <= 0.0)
		adaptive._queryMixFactor = 1.0;
	adaptive._queryMixFactor = std::clamp(adaptive._queryMixFactor, 0.25, 4.0);
}

static void maybeAssignAdaptiveLeafCapacity(
	SchemaLevelConfig& level,
	std::mt19937& rng,
	const Experiments::SchemaGenerationOptions& options)
{
	if (!options._adaptiveLeafCapacity)
		return;

	std::bernoulli_distribution adaptiveDistribution(std::clamp(options._adaptiveLeafProbability, 0.0, 1.0));
	if (adaptiveDistribution(rng))
		level._adaptiveLeafCapacity = randomAdaptiveLeafCapacity(rng, std::max<size_t>(1, level._leafCapacity));
}

template <typename T>
static T nearbyDomainValue(const std::vector<T>& values, T current, std::mt19937& rng)
{
	if (values.empty())
		return current;

	const auto lower = std::lower_bound(values.begin(), values.end(), current);
	size_t index = lower == values.end()
		? values.size() - 1
		: static_cast<size_t>(std::distance(values.begin(), lower));

	std::uniform_int_distribution<int> stepDistribution(-1, 1);
	const int step = stepDistribution(rng);
	if (step < 0 && index > 0)
		--index;
	else if (step > 0 && index + 1 < values.size())
		++index;

	return values[index];
}

static void appendOptionalSize(std::ostringstream& output, const char* jsonName, const std::optional<size_t>& value, bool& first)
{
	if (!value.has_value())
		return;

	output << (first ? "" : ",\n") << "        \"" << jsonName << "\": " << value.value();
	first = false;
}

static void appendOptionalDouble(std::ostringstream& output, const char* jsonName, const std::optional<double>& value, bool& first)
{
	if (!value.has_value())
		return;

	output << (first ? "" : ",\n") << "        \"" << jsonName << "\": " << value.value();
	first = false;
}

static void appendConditionSignature(std::ostringstream& output, const SchemaLevelCondition& condition)
{
	if (condition.empty())
		return;

	output << "c";
	if (condition._minPoints) output << "p" << condition._minPoints.value();
	if (condition._maxPoints) output << "P" << condition._maxPoints.value();
	if (condition._minDensity) output << "d" << static_cast<size_t>(condition._minDensity.value() * 1000.0);
	if (condition._maxDensity) output << "D" << static_cast<size_t>(condition._maxDensity.value() * 1000.0);
	if (condition._minHeightRatio) output << "h" << static_cast<size_t>(condition._minHeightRatio.value() * 1000.0);
	if (condition._maxHeightRatio) output << "H" << static_cast<size_t>(condition._maxHeightRatio.value() * 1000.0);
	if (condition._minExtentX) output << "x" << static_cast<size_t>(condition._minExtentX.value() * 1000.0);
	if (condition._maxExtentX) output << "X" << static_cast<size_t>(condition._maxExtentX.value() * 1000.0);
	if (condition._minExtentY) output << "y" << static_cast<size_t>(condition._minExtentY.value() * 1000.0);
	if (condition._maxExtentY) output << "Y" << static_cast<size_t>(condition._maxExtentY.value() * 1000.0);
	if (condition._minExtentZ) output << "z" << static_cast<size_t>(condition._minExtentZ.value() * 1000.0);
	if (condition._maxExtentZ) output << "Z" << static_cast<size_t>(condition._maxExtentZ.value() * 1000.0);
	if (condition._minAnisotropy) output << "a" << static_cast<size_t>(condition._minAnisotropy.value() * 1000.0);
	if (condition._maxAnisotropy) output << "A" << static_cast<size_t>(condition._maxAnisotropy.value() * 1000.0);
	if (condition._minOccupancyEntropy) output << "e" << static_cast<size_t>(condition._minOccupancyEntropy.value() * 1000.0);
	if (condition._maxOccupancyEntropy) output << "E" << static_cast<size_t>(condition._maxOccupancyEntropy.value() * 1000.0);
}

static size_t conditionFieldCount(const SchemaLevelCondition& condition)
{
	size_t count = 0;
	count += condition._minPoints.has_value() ? 1 : 0;
	count += condition._maxPoints.has_value() ? 1 : 0;
	count += condition._minDensity.has_value() ? 1 : 0;
	count += condition._maxDensity.has_value() ? 1 : 0;
	count += condition._minHeightRatio.has_value() ? 1 : 0;
	count += condition._maxHeightRatio.has_value() ? 1 : 0;
	count += condition._minExtentX.has_value() ? 1 : 0;
	count += condition._maxExtentX.has_value() ? 1 : 0;
	count += condition._minExtentY.has_value() ? 1 : 0;
	count += condition._maxExtentY.has_value() ? 1 : 0;
	count += condition._minExtentZ.has_value() ? 1 : 0;
	count += condition._maxExtentZ.has_value() ? 1 : 0;
	count += condition._minAnisotropy.has_value() ? 1 : 0;
	count += condition._maxAnisotropy.has_value() ? 1 : 0;
	count += condition._minOccupancyEntropy.has_value() ? 1 : 0;
	count += condition._maxOccupancyEntropy.has_value() ? 1 : 0;
	return count;
}

static size_t schemaConditionalLevels(const SchemaConfig& schema)
{
	size_t count = 0;
	for (const SchemaLevelConfig& level : schema._levels)
	{
		if (!level._condition.empty())
			++count;
	}
	return count;
}

static size_t schemaConditionFields(const SchemaConfig& schema)
{
	size_t count = 0;
	for (const SchemaLevelConfig& level : schema._levels)
		count += conditionFieldCount(level._condition);
	return count;
}

static bool schemaUsesAdaptiveLeafCapacity(const SchemaConfig& schema)
{
	return std::any_of(schema._levels.begin(), schema._levels.end(), [](const SchemaLevelConfig& level) {
		return level._adaptiveLeafCapacity._enabled;
	});
}

// Features the CUDA MixedTree cannot honor (adaptive leaf capacity, occupancy-entropy conditions); these schemas fall back to the CPU evaluator rather than being dropped.
static bool schemaUsesGpuUnsupportedFeature(const SchemaConfig& schema)
{
	if (schemaUsesAdaptiveLeafCapacity(schema))
		return true;
	return std::any_of(schema._levels.begin(), schema._levels.end(), [](const SchemaLevelConfig& level) {
		return level._condition._minOccupancyEntropy.has_value() || level._condition._maxOccupancyEntropy.has_value();
	});
}

// CUDA MixedTree has no occupancy-entropy gate; drop those thresholds so a CUDA search samples GPU-runnable schemas instead of silently falling back to the CPU.
static void restrictConditionDomainToGpuSafe(Experiments::ConditionDomain& domain)
{
	domain._occupancyEntropyThresholds.clear();
}

static std::string schemaConditionSummary(const SchemaConfig& schema)
{
	std::ostringstream output;
	bool first = true;
	for (const SchemaLevelConfig& level : schema._levels)
	{
		if (level._condition.empty())
			continue;

		if (!first)
			output << ';';
		first = false;
		output << schemaTypeShortName(level) << ':';
		appendConditionSignature(output, level._condition);
	}
	if (schema._buildPolicy._enableLeafMicroIndexes)
		output << "_mi" << schema._buildPolicy._leafMicroIndexThreshold;
	return output.str();
}

struct ActiveTypeAccumulator
{
	size_t _nodes = 0;
	size_t _leafPoints = 0;
};

struct ActiveStructureStats
{
	size_t _activeStructureTypes = 0;
	double _nestedActiveFraction = 0.0;
	std::string _summary;
};

static const SchemaLevelConfig* schemaLevelForNode(const SchemaConfig& schema, const PointSpatialIndex::Node& node)
{
	if (schema._levels.empty())
		return nullptr;
	const size_t totalLevels = std::max<size_t>(1, schema.totalLevels());
	const size_t schemaDepth = std::min(node._schemaDepth, totalLevels - 1);
	return &schema.levelForDepth(schemaDepth);
}

static std::string activeTypeNameForNode(const SchemaConfig& schema, const PointSpatialIndex::Node& node)
{
	const SchemaLevelConfig* level = schemaLevelForNode(schema, node);
	if (level)
		return schemaTypeShortName(*level);

	SchemaLevelConfig fallback;
	fallback._type = node._type;
	fallback._typeName = Config::dataStructureLevelName(node._type);
	return schemaTypeShortName(fallback);
}

static void collectActiveStructureStatsRecursive(
	const PointSpatialIndex::Node* node,
	const SchemaConfig& schema,
	std::map<std::string, ActiveTypeAccumulator>& byType,
	size_t& totalNodes,
	size_t& totalLeafPoints)
{
	if (!node)
		return;

	const std::string typeName = activeTypeNameForNode(schema, *node);
	ActiveTypeAccumulator& accumulator = byType[typeName];
	++accumulator._nodes;
	++totalNodes;
	if (node->isLeaf())
	{
		accumulator._leafPoints += node->_pointCount;
		totalLeafPoints += node->_pointCount;
	}

	for (const std::unique_ptr<PointSpatialIndex::Node>& child : node->_children)
		collectActiveStructureStatsRecursive(child.get(), schema, byType, totalNodes, totalLeafPoints);
}

static ActiveStructureStats collectActiveStructureStats(const PointSpatialIndex::Node* root, const SchemaConfig& schema)
{
	ActiveStructureStats stats;
	if (!root)
		return stats;

	std::map<std::string, ActiveTypeAccumulator> byType;
	size_t totalNodes = 0;
	size_t totalLeafPoints = 0;
	collectActiveStructureStatsRecursive(root, schema, byType, totalNodes, totalLeafPoints);

	stats._activeStructureTypes = byType.size();
	const std::string primaryType = schema._levels.empty() ? activeTypeNameForNode(schema, *root) : schemaTypeShortName(schema._levels.front());
	size_t nonPrimaryNodes = 0;
	size_t nonPrimaryLeafPoints = 0;

	std::ostringstream summary;
	bool first = true;
	for (const auto& [typeName, accumulator] : byType)
	{
		if (!first)
			summary << ';';
		first = false;
		summary << typeName << ":nodes=" << accumulator._nodes << "|points=" << accumulator._leafPoints;

		if (typeName != primaryType)
		{
			nonPrimaryNodes += accumulator._nodes;
			nonPrimaryLeafPoints += accumulator._leafPoints;
		}
	}
	stats._summary = summary.str();

	const double pointFraction = totalLeafPoints > 0
		? static_cast<double>(nonPrimaryLeafPoints) / static_cast<double>(totalLeafPoints)
		: 0.0;
	const double nodeFraction = totalNodes > 0
		? static_cast<double>(nonPrimaryNodes) / static_cast<double>(totalNodes)
		: 0.0;
	stats._nestedActiveFraction = std::max(pointFraction, nodeFraction);
	return stats;
}

static ActiveStructureStats staticSchemaStructureStats(const SchemaConfig& schema)
{
	ActiveStructureStats stats;
	std::set<std::string> types;
	for (const SchemaLevelConfig& level : schema._levels)
		types.insert(schemaTypeShortName(level));
	stats._activeStructureTypes = types.size();

	std::ostringstream summary;
	bool first = true;
	for (const std::string& typeName : types)
	{
		if (!first)
			summary << ';';
		first = false;
		summary << typeName << ":schema";
	}
	stats._summary = summary.str();
	return stats;
}

static SchemaLevelCondition randomLevelCondition(
	std::mt19937& rng,
	const SchemaLevelConfig& level,
	size_t minLeaf,
	size_t maxLeaf,
	const Experiments::ConditionDomain* domain)
{
	SchemaLevelCondition condition;
	if (domain && !domain->_pointThresholds.empty())
	{
		condition._minPoints = chooseSizeValue(rng, domain->_pointThresholds, fallbackPointThresholds());
	}
	else
	{
		const size_t maxPointThreshold = std::max<size_t>(minLeaf * 2, std::min<size_t>(maxLeaf * 8, 1 << 20));
		condition._minPoints = randomPowerOfTwo(rng, std::max<size_t>(minLeaf, 32), maxPointThreshold);
	}

	if (level._type == MultiDataStructure::OctreeNode)
	{
		if (isRegularGridLevelName(level._typeName) || isHGridLevelName(level._typeName))
		{
			if (domain && !domain->_densityThresholds.empty())
				condition._minDensity = chooseDoubleValue(rng, domain->_densityThresholds, {});
		}
		else
		{
			condition._minHeightRatio = chooseDoubleValue(
				rng,
				domain ? domain->_heightRatioThresholds : std::vector<double>{},
				fallbackHeightRatioThresholds());
		}
	}
	else if (level._type == MultiDataStructure::QuadTreeNode)
	{
		condition._maxHeightRatio = chooseDoubleValue(
			rng,
			domain ? domain->_heightRatioThresholds : std::vector<double>{},
			fallbackHeightRatioThresholds());
	}

	// Add an anisotropy gate with probability 0.5 so the optimizer can compare shape-gated branches against ungated ones.
	if (domain && !domain->_anisotropyThresholds.empty())
	{
		std::bernoulli_distribution coin(0.5);
		if (coin(rng))
		{
			std::bernoulli_distribution minOrMax(0.5);
			if (minOrMax(rng))
				condition._minAnisotropy = chooseDoubleValue(rng, domain->_anisotropyThresholds, {});
			else
				condition._maxAnisotropy = chooseDoubleValue(rng, domain->_anisotropyThresholds, {});
		}
	}

	// Same pattern for occupancy entropy: 0.5 chance to gate, then 0.5 between min/max.
	if (domain && !domain->_occupancyEntropyThresholds.empty())
	{
		std::bernoulli_distribution coin(0.5);
		if (coin(rng))
		{
			std::bernoulli_distribution minOrMax(0.5);
			if (minOrMax(rng))
				condition._minOccupancyEntropy = chooseDoubleValue(rng, domain->_occupancyEntropyThresholds, {});
			else
				condition._maxOccupancyEntropy = chooseDoubleValue(rng, domain->_occupancyEntropyThresholds, {});
		}
	}

	return condition;
}

static std::string schemaSignature(const SchemaConfig& schema)
{
	std::ostringstream output;
	for (size_t i = 0; i < schema._levels.size(); ++i)
	{
		const SchemaLevelConfig& level = schema._levels[i];
		if (i > 0)
			output << '_';
		output
			<< schemaTypeShortName(level)
			<< level._numLevels
			<< "l"
			<< level._leafCapacity;
		if (level._adaptiveLeafCapacity._enabled)
		{
			output
				<< "ac"
				<< level._adaptiveLeafCapacity._minCapacity
				<< "m"
				<< level._adaptiveLeafCapacity._maxCapacity
				<< "d"
				<< static_cast<int>(std::llround(level._adaptiveLeafCapacity._densityWeight * 100.0))
				<< "a"
				<< static_cast<int>(std::llround(level._adaptiveLeafCapacity._anisotropyWeight * 100.0))
				<< "h"
				<< static_cast<int>(std::llround(level._adaptiveLeafCapacity._heightRatioWeight * 100.0))
				<< "q"
				<< static_cast<int>(std::llround(level._adaptiveLeafCapacity._queryMixFactor * 100.0));
		}
		// Fold the axis policy into the signature so policy-only KDTree/BIH variants get distinct keys; default/empty policies emit no suffix to keep legacy signatures stable.
		if (!level._axisPolicy.empty() &&
			level._axisPolicy != "median_longest_axis" &&
			level._axisPolicy != "xy")
		{
			if (level._axisPolicy == "round_robin")
				output << "ap=rr";
			else if (level._axisPolicy == "center_longest_axis")
				output << "ap=cl";
			else if (level._axisPolicy == "xz")
				output << "ap=xz";
			else if (level._axisPolicy == "yz")
				output << "ap=yz";
			else if (level._axisPolicy == "ignore_shortest")
				output << "ap=is";
			else if (level._axisPolicy == "ignore_x")
				output << "ap=ix";
			else if (level._axisPolicy == "ignore_y")
				output << "ap=iy";
			else if (level._axisPolicy == "ignore_z")
				output << "ap=iz";
			else
				output << "ap=other";
		}
		appendConditionSignature(output, level._condition);
	}
	return output.str();
}

// Coarse key: just the level-type sequence, ignoring numeric details. Used by the diversity audit to count qualitatively distinct shapes.
static std::string schemaTopologyKey(const SchemaConfig& schema)
{
	std::ostringstream output;
	for (size_t i = 0; i < schema._levels.size(); ++i)
	{
		if (i > 0)
			output << ':';
		output << schemaTypeShortName(schema._levels[i]);
	}
	return output.str();
}

static std::string schemaConfigToJson(const SchemaConfig& schema)
{
	std::ostringstream output;
	output << "{\n";
	output << "  \"name\": \"" << schema._name << "\",\n";
	output << "  \"levels\": [\n";
	for (size_t i = 0; i < schema._levels.size(); ++i)
	{
		const SchemaLevelConfig& level = schema._levels[i];
		output << "    {\n";
		output << "      \"type\": \"" << levelJsonTypeName(level) << "\",\n";
		output << "      \"numLevels\": " << level._numLevels << ",\n";
		output << "      \"leafCapacity\": " << level._leafCapacity << ",\n";
		output << "      \"minPointsToSplit\": " << level._minPrimitivesToSplit;
		if (!level._axisPolicy.empty())
			output << ",\n      \"axisPolicy\": \"" << level._axisPolicy << "\"";
		if (level._adaptiveLeafCapacity._enabled)
		{
			output << ",\n      \"adaptiveLeafCapacity\": {\n";
			output << "        \"enabled\": true,\n";
			output << "        \"minCapacity\": " << level._adaptiveLeafCapacity._minCapacity << ",\n";
			output << "        \"maxCapacity\": " << level._adaptiveLeafCapacity._maxCapacity << ",\n";
			output << "        \"densityWeight\": " << level._adaptiveLeafCapacity._densityWeight << ",\n";
			output << "        \"anisotropyWeight\": " << level._adaptiveLeafCapacity._anisotropyWeight << ",\n";
			output << "        \"heightRatioWeight\": " << level._adaptiveLeafCapacity._heightRatioWeight << ",\n";
			output << "        \"queryMixFactor\": " << level._adaptiveLeafCapacity._queryMixFactor << "\n";
			output << "      }";
		}
		if (!level._condition.empty())
		{
			output << ",\n      \"condition\": {\n";
			bool first = true;
			appendOptionalSize(output, "minPoints", level._condition._minPoints, first);
			appendOptionalSize(output, "maxPoints", level._condition._maxPoints, first);
			appendOptionalDouble(output, "minDensity", level._condition._minDensity, first);
			appendOptionalDouble(output, "maxDensity", level._condition._maxDensity, first);
			appendOptionalDouble(output, "minHeightRatio", level._condition._minHeightRatio, first);
			appendOptionalDouble(output, "maxHeightRatio", level._condition._maxHeightRatio, first);
			appendOptionalDouble(output, "minExtentX", level._condition._minExtentX, first);
			appendOptionalDouble(output, "maxExtentX", level._condition._maxExtentX, first);
			appendOptionalDouble(output, "minExtentY", level._condition._minExtentY, first);
			appendOptionalDouble(output, "maxExtentY", level._condition._maxExtentY, first);
			appendOptionalDouble(output, "minExtentZ", level._condition._minExtentZ, first);
			appendOptionalDouble(output, "maxExtentZ", level._condition._maxExtentZ, first);
			appendOptionalDouble(output, "minAnisotropy", level._condition._minAnisotropy, first);
			appendOptionalDouble(output, "maxAnisotropy", level._condition._maxAnisotropy, first);
			appendOptionalDouble(output, "minOccupancyEntropy", level._condition._minOccupancyEntropy, first);
			appendOptionalDouble(output, "maxOccupancyEntropy", level._condition._maxOccupancyEntropy, first);
			output << "\n      }";
		}
		output << '\n';
		output << "    }" << (i + 1 < schema._levels.size() ? "," : "") << "\n";
	}
	output << "  ],\n";
	output << "  \"buildPolicy\": {\n";
	output << "    \"maxDepth\": " << schema._buildPolicy._maxDepth << ",\n";
	output << "    \"leafCapacity\": " << schema._buildPolicy._leafCapacity << ",\n";
	output << "    \"minPointsToSplit\": " << schema._buildPolicy._minPrimitivesToSplit << ",\n";
	output << "    \"collapseSingleChild\": " << (schema._buildPolicy._collapseSingleChild ? "true" : "false") << ",\n";
	output << "    \"removeEmptyNodes\": " << (schema._buildPolicy._removeEmptyNodes ? "true" : "false") << ",\n";
	output << "    \"allowOverlapDuplication\": " << (schema._buildPolicy._allowOverlapDuplication ? "true" : "false") << ",\n";
	output << "    \"enableLeafMicroIndexes\": " << (schema._buildPolicy._enableLeafMicroIndexes ? "true" : "false") << ",\n";
	output << "    \"leafMicroIndexThreshold\": " << schema._buildPolicy._leafMicroIndexThreshold << "\n";
	output << "  }\n";
	output << "}\n";
	return output.str();
}

static Experiments::SchemaCandidate materializeGeneratedSchema(
	SchemaConfig schema,
	const std::string& namePrefix,
	const std::string& outputDirectory)
{
	const std::string signature = schemaSignature(schema);
	schema._name = namePrefix + "_" + signature;

	Experiments::SchemaCandidate candidate;
	candidate._name = schema._name;
	candidate._config = schema;
	candidate._generated = true;

	if (!outputDirectory.empty())
	{
		const std::filesystem::path schemaPath = std::filesystem::path(outputDirectory) / (schema._name + ".json");
		if (schemaPath.has_parent_path())
			std::filesystem::create_directories(schemaPath.parent_path());

		std::ofstream output(schemaPath);
		if (!output.is_open())
			throw std::runtime_error("Unable to write generated schema: " + schemaPath.string());
		output << schemaConfigToJson(schema);
		candidate._path = schemaPath.string();
	}
	else
	{
		candidate._path = "generated:" + schema._name;
	}

	return candidate;
}

static SchemaLevelConfig randomLevelConfig(
	std::mt19937& rng,
	const Experiments::SchemaGenerationOptions& options,
	std::optional<MultiDataStructure::DataStructureLevel> previousType = std::nullopt)
{
	const size_t minLeaf = std::max<size_t>(1, std::min(options._minLeafCapacity, options._maxLeafCapacity));
	const size_t maxLeaf = std::max(minLeaf, options._maxLeafCapacity);

	SchemaLevelConfig level;
	level._type = randomStructureType(rng, previousType);
	level._typeName = randomTypeNameForBase(level._type, rng, options);
	level._numLevels = 1;
	level._leafCapacity = randomPowerOfTwo(rng, minLeaf, maxLeaf);
	level._minPrimitivesToSplit = std::max<size_t>(2, level._leafCapacity / 4);
	maybeAssignAdaptiveLeafCapacity(level, rng, options);
	if (level._type == MultiDataStructure::QuadTreeNode)
		level._axisPolicy = "xy";
	else if (level._type == MultiDataStructure::KDTreeNode)
		level._axisPolicy = std::bernoulli_distribution(0.5)(rng) ? "round_robin" : "median_longest_axis";
	return level;
}

// Coin-flips the axis policy for KDTree/BIH levels: round_robin alternates X/Y/Z by depth, median_longest_axis is the legacy extent-driven default.
static std::string sampleAxisPolicy(std::mt19937& rng, MultiDataStructure::DataStructureLevel type)
{
	if (type == MultiDataStructure::QuadTreeNode)
		return "xy";
	if (type != MultiDataStructure::KDTreeNode)
		return "";
	std::bernoulli_distribution coin(0.5);
	return coin(rng) ? "round_robin" : "median_longest_axis";
}

static void refreshLevelTypeName(SchemaLevelConfig& level, const Experiments::SchemaGenerationOptions& options)
{
	if (queryMinimalPrimitiveProfile(options))
	{
		level._typeName = Config::dataStructureLevelName(level._type);
		if (level._type == MultiDataStructure::QuadTreeNode)
		{
			if (level._axisPolicy.empty())
				level._axisPolicy = "xy";
		}
		else if (level._type != MultiDataStructure::KDTreeNode)
			level._axisPolicy = "";
		else if (level._axisPolicy.empty())
			level._axisPolicy = "median_longest_axis";
		return;
	}

	const bool compatibleBIH = level._type == MultiDataStructure::KDTreeNode && isBIHLevelName(level._typeName);
	const bool compatibleKarras = level._type == MultiDataStructure::OctreeNode && isKarrasOctreeLevelName(level._typeName);
	const bool compatibleLBVH = level._type == MultiDataStructure::BvhNode && isLBVHLevelName(level._typeName);
	const bool compatibleRegularGrid = level._type == MultiDataStructure::OctreeNode && isRegularGridLevelName(level._typeName);
	const bool compatibleHGrid = level._type == MultiDataStructure::OctreeNode && isHGridLevelName(level._typeName);
	if (!compatibleBIH && !compatibleKarras && !compatibleLBVH && !compatibleRegularGrid && !compatibleHGrid)
		level._typeName = Config::dataStructureLevelName(level._type);
	// Preserve a recognized axisPolicy, resetting only for non-KDTree-family types, so children inherit their parent's sampled policy.
	if (level._type == MultiDataStructure::QuadTreeNode)
	{
		if (level._axisPolicy.empty())
			level._axisPolicy = "xy";
	}
	else if (level._type != MultiDataStructure::KDTreeNode)
		level._axisPolicy = "";
	else if (level._axisPolicy.empty())
		level._axisPolicy = "median_longest_axis";
}

static void normalizeSchemaForGeneration(SchemaConfig& schema, const Experiments::SchemaGenerationOptions& options)
{
	const size_t maxDepth = std::max<size_t>(1, options._maxDepth);
	const size_t maxBlocks = std::max<size_t>(1, std::min(options._maxBlocks, maxDepth));
	const size_t minLeaf = std::max<size_t>(1, std::min(options._minLeafCapacity, options._maxLeafCapacity));
	const size_t maxLeaf = std::max(minLeaf, options._maxLeafCapacity);

	if (schema._levels.empty())
	{
		std::mt19937 fallbackRng(options._seed);
		schema._levels.push_back(randomLevelConfig(fallbackRng, options));
	}

	while (schema._levels.size() > maxBlocks)
		schema._levels.pop_back();

	for (size_t i = 0; i < schema._levels.size(); ++i)
	{
		SchemaLevelConfig& level = schema._levels[i];
		level._numLevels = std::max<size_t>(1, level._numLevels);
		level._leafCapacity = clampPowerOfTwo(level._leafCapacity, minLeaf, maxLeaf);
		level._minPrimitivesToSplit = std::clamp(level._minPrimitivesToSplit, static_cast<size_t>(2), std::max<size_t>(2, level._leafCapacity));
		refreshLevelTypeName(level, options);
		normalizeAdaptiveLeafCapacity(level);
		if (i == 0)
			level._condition = {};
	}

	while (schema.totalLevels() > maxDepth && !schema._levels.empty())
	{
		auto reducible = std::find_if(schema._levels.rbegin(), schema._levels.rend(), [](const SchemaLevelConfig& level) {
			return level._numLevels > 1;
		});
		if (reducible != schema._levels.rend())
		{
			--reducible->_numLevels;
			continue;
		}

		if (schema._levels.size() > 1)
			schema._levels.pop_back();
		else
			break;
	}

	if (schema._levels.empty())
	{
		std::mt19937 fallbackRng(options._seed);
		schema._levels.push_back(randomLevelConfig(fallbackRng, options));
	}

	schema._buildPolicy._maxDepth = std::min(maxDepth, schema.totalLevels());
	schema._buildPolicy._leafCapacity = schema._levels.front()._leafCapacity;
	schema._buildPolicy._minPrimitivesToSplit = std::max<size_t>(2, schema._buildPolicy._leafCapacity / 4);
	schema._buildPolicy._collapseSingleChild = true;
	schema._buildPolicy._removeEmptyNodes = true;
	schema._buildPolicy._allowOverlapDuplication = false;
}

static void mutateLevelCondition(
	SchemaLevelConfig& level,
	std::mt19937& rng,
	const Experiments::SchemaGenerationOptions& options,
	const Experiments::ConditionDomain* domain)
{
	const size_t minLeaf = std::max<size_t>(1, std::min(options._minLeafCapacity, options._maxLeafCapacity));
	const size_t maxLeaf = std::max(minLeaf, options._maxLeafCapacity);
	if (level._condition.empty())
	{
		level._condition = randomLevelCondition(rng, level, minLeaf, maxLeaf, domain);
		return;
	}

	std::uniform_int_distribution<int> editDistribution(0, 10);
	switch (editDistribution(rng))
	{
	case 0:
		level._condition = randomLevelCondition(rng, level, minLeaf, maxLeaf, domain);
		break;
	case 1:
		if (domain && !domain->_pointThresholds.empty())
			level._condition._minPoints = nearbyDomainValue(domain->_pointThresholds, level._condition._minPoints.value_or(domain->_pointThresholds.front()), rng);
		else
			level._condition._minPoints = randomPowerOfTwo(rng, minLeaf, std::max<size_t>(minLeaf * 2, std::min<size_t>(maxLeaf * 8, 1 << 20)));
		break;
	case 2:
	{
		const std::vector<double>& heightValues = domain && !domain->_heightRatioThresholds.empty()
			? domain->_heightRatioThresholds
			: fallbackHeightRatioThresholds();
		if (level._type == MultiDataStructure::QuadTreeNode)
			level._condition._maxHeightRatio = nearbyDomainValue(heightValues, level._condition._maxHeightRatio.value_or(heightValues.front()), rng);
		else
			level._condition._minHeightRatio = nearbyDomainValue(heightValues, level._condition._minHeightRatio.value_or(heightValues.front()), rng);
		break;
	}
	case 3:
		if (domain && !domain->_pointThresholds.empty())
			level._condition._maxPoints = nearbyDomainValue(domain->_pointThresholds, level._condition._maxPoints.value_or(domain->_pointThresholds.back()), rng);
		break;
	case 4:
		if (domain && !domain->_densityThresholds.empty())
		{
			if (isRegularGridLevelName(level._typeName) || isHGridLevelName(level._typeName))
				level._condition._minDensity = nearbyDomainValue(domain->_densityThresholds, level._condition._minDensity.value_or(domain->_densityThresholds.front()), rng);
			else
				level._condition._maxDensity = nearbyDomainValue(domain->_densityThresholds, level._condition._maxDensity.value_or(domain->_densityThresholds.back()), rng);
		}
		break;
	case 5:
		if (domain)
		{
			std::uniform_int_distribution<int> axisDistribution(0, 2);
			const int axis = axisDistribution(rng);
			if (axis == 0 && !domain->_extentXThresholds.empty())
				level._condition._minExtentX = nearbyDomainValue(domain->_extentXThresholds, level._condition._minExtentX.value_or(domain->_extentXThresholds.front()), rng);
			else if (axis == 1 && !domain->_extentYThresholds.empty())
				level._condition._minExtentY = nearbyDomainValue(domain->_extentYThresholds, level._condition._minExtentY.value_or(domain->_extentYThresholds.front()), rng);
			else if (axis == 2 && !domain->_extentZThresholds.empty())
				level._condition._minExtentZ = nearbyDomainValue(domain->_extentZThresholds, level._condition._minExtentZ.value_or(domain->_extentZThresholds.front()), rng);
		}
		break;
	case 6:
		// Toggle a min/max anisotropy threshold from the domain.
		if (domain && !domain->_anisotropyThresholds.empty())
		{
			std::bernoulli_distribution minOrMax(0.5);
			if (minOrMax(rng))
				level._condition._minAnisotropy = nearbyDomainValue(domain->_anisotropyThresholds,
					level._condition._minAnisotropy.value_or(domain->_anisotropyThresholds.front()), rng);
			else
				level._condition._maxAnisotropy = nearbyDomainValue(domain->_anisotropyThresholds,
					level._condition._maxAnisotropy.value_or(domain->_anisotropyThresholds.back()), rng);
		}
		break;
	case 7:
		// Clear the anisotropy gate while keeping other threshold fields intact.
		level._condition._minAnisotropy.reset();
		level._condition._maxAnisotropy.reset();
		break;
	case 8:
		// Toggle a min/max occupancy entropy threshold from the domain.
		if (domain && !domain->_occupancyEntropyThresholds.empty())
		{
			std::bernoulli_distribution minOrMax(0.5);
			if (minOrMax(rng))
				level._condition._minOccupancyEntropy = nearbyDomainValue(domain->_occupancyEntropyThresholds,
					level._condition._minOccupancyEntropy.value_or(domain->_occupancyEntropyThresholds.front()), rng);
			else
				level._condition._maxOccupancyEntropy = nearbyDomainValue(domain->_occupancyEntropyThresholds,
					level._condition._maxOccupancyEntropy.value_or(domain->_occupancyEntropyThresholds.back()), rng);
		}
		break;
	case 9:
		// Clear the entropy gate.
		level._condition._minOccupancyEntropy.reset();
		level._condition._maxOccupancyEntropy.reset();
		break;
	default:
		level._condition = {};
		break;
	}
}

static void mutateAdaptiveLeafCapacity(
	SchemaLevelConfig& level,
	std::mt19937& rng)
{
	if (!level._adaptiveLeafCapacity._enabled)
	{
		level._adaptiveLeafCapacity = randomAdaptiveLeafCapacity(rng, std::max<size_t>(1, level._leafCapacity));
		return;
	}

	AdaptiveLeafCapacityConfig& adaptive = level._adaptiveLeafCapacity;
	std::uniform_int_distribution<int> mutationDistribution(0, 5);
	switch (mutationDistribution(rng))
	{
	case 0:
		adaptive._minCapacity = clampPowerOfTwo(
			std::max<size_t>(1, adaptive._minCapacity / 2),
			size_t(1),
			std::max<size_t>(1, level._leafCapacity));
		break;
	case 1:
		adaptive._maxCapacity = clampPowerOfTwo(
			adaptive._maxCapacity >= std::numeric_limits<size_t>::max() / 2
				? adaptive._maxCapacity
				: adaptive._maxCapacity * 2,
			std::max<size_t>(adaptive._minCapacity, level._leafCapacity),
			std::numeric_limits<size_t>::max());
		break;
	case 2:
		adaptive._densityWeight = std::clamp(
			adaptive._densityWeight + (std::bernoulli_distribution(0.5)(rng) ? 0.5 : -0.5),
			-3.0,
			3.0);
		break;
	case 3:
		adaptive._anisotropyWeight = std::clamp(
			adaptive._anisotropyWeight + (std::bernoulli_distribution(0.5)(rng) ? 0.5 : -0.5),
			-3.0,
			3.0);
		break;
	case 4:
		adaptive._heightRatioWeight = std::clamp(
			adaptive._heightRatioWeight + (std::bernoulli_distribution(0.5)(rng) ? 0.5 : -0.5),
			-3.0,
			3.0);
		break;
	case 5:
		adaptive._queryMixFactor = std::clamp(
			adaptive._queryMixFactor * (std::bernoulli_distribution(0.5)(rng) ? 1.25 : 0.8),
			0.25,
			4.0);
		break;
	}

	normalizeAdaptiveLeafCapacity(level);
}

// Single-point crossover on the level list: first splitA levels of parent A spliced with parent B's tail from splitB, reaching topology combinations neither parent had.
static SchemaConfig crossoverSchemaConfigs(
	const SchemaConfig& parentA,
	const SchemaConfig& parentB,
	std::mt19937& rng,
	const Experiments::SchemaGenerationOptions& options)
{
	if (parentA._levels.empty()) return parentB;
	if (parentB._levels.empty()) return parentA;

	std::uniform_int_distribution<size_t> cutADist(0, parentA._levels.size());
	std::uniform_int_distribution<size_t> cutBDist(0, parentB._levels.size());
	size_t splitA = cutADist(rng);
	size_t splitB = cutBDist(rng);

	// Reject the degenerate "all of A or all of B" case, which just returns a parent untouched.
	if (splitA == 0 && splitB == 0)
		splitA = 1;
	else if (splitA == parentA._levels.size() && splitB == parentB._levels.size())
		splitB = std::max<size_t>(0, parentB._levels.size() - 1);

	SchemaConfig child;
	child._buildPolicy = parentA._buildPolicy;
	child._levels.reserve(splitA + (parentB._levels.size() - splitB));
	for (size_t i = 0; i < splitA; ++i)
		child._levels.push_back(parentA._levels[i]);
	for (size_t i = splitB; i < parentB._levels.size(); ++i)
		child._levels.push_back(parentB._levels[i]);

	// Reuse the mutation operator's normalization so depth caps, leaf-cap monotonicity, and root-condition stripping match the rest of the population.
	normalizeSchemaForGeneration(child, options);
	return child;
}

static SchemaConfig mutateSchemaConfig(
	const SchemaConfig& parent,
	std::mt19937& rng,
	const Experiments::SchemaGenerationOptions& options,
	const Experiments::EvolutionOptions& evolution,
	const Experiments::ConditionDomain* domain)
{
	SchemaConfig schema = parent;
	const size_t maxDepth = std::max<size_t>(1, options._maxDepth);
	const size_t maxBlocks = std::max<size_t>(1, std::min(options._maxBlocks, maxDepth));
	const size_t minLeaf = std::max<size_t>(1, std::min(options._minLeafCapacity, options._maxLeafCapacity));
	const size_t maxLeaf = std::max(minLeaf, options._maxLeafCapacity);

	std::bernoulli_distribution extraEdit(std::clamp(evolution._mutationRate, 0.0, 1.0));
	size_t edits = 1;
	while (edits < 5 && extraEdit(rng))
		++edits;

	for (size_t edit = 0; edit < edits; ++edit)
	{
		if (schema._levels.empty())
			schema._levels.push_back(randomLevelConfig(rng, options));

		if (domain && options._conditionalLevels && schema._levels.size() > 1)
		{
			std::bernoulli_distribution conditionEdit(0.5);
			if (conditionEdit(rng))
			{
				std::uniform_int_distribution<size_t> conditionalLevelDistribution(1, schema._levels.size() - 1);
				mutateLevelCondition(schema._levels[conditionalLevelDistribution(rng)], rng, options, domain);
				continue;
			}
		}

		std::uniform_int_distribution<size_t> levelDistribution(0, schema._levels.size() - 1);
		const size_t levelIndex = levelDistribution(rng);
		SchemaLevelConfig& level = schema._levels[levelIndex];
		const bool canMutateAdaptiveLeafCapacity =
			options._adaptiveLeafCapacity ||
			std::any_of(schema._levels.begin(), schema._levels.end(), [](const SchemaLevelConfig& candidateLevel) {
				return candidateLevel._adaptiveLeafCapacity._enabled;
			});

		std::uniform_int_distribution<int> mutationDistribution(0, canMutateAdaptiveLeafCapacity ? 9 : 8);
		switch (mutationDistribution(rng))
		{
		case 0:
			level._type = randomStructureType(rng, std::nullopt);
			level._typeName = randomTypeNameForBase(level._type, rng, options);
			level._axisPolicy = sampleAxisPolicy(rng, level._type);
			if (levelIndex == 0)
				level._condition = {};
			break;
		case 1:
		{
			std::bernoulli_distribution grow(0.5);
			if (grow(rng) && schema.totalLevels() < maxDepth)
				++level._numLevels;
			else if (level._numLevels > 1)
				--level._numLevels;
			break;
		}
		case 2:
		{
			std::bernoulli_distribution grow(0.5);
			level._leafCapacity = grow(rng)
				? clampPowerOfTwo(level._leafCapacity * 2, minLeaf, maxLeaf)
				: clampPowerOfTwo(std::max<size_t>(1, level._leafCapacity / 2), minLeaf, maxLeaf);
			level._minPrimitivesToSplit = std::max<size_t>(2, level._leafCapacity / 4);
			break;
		}
		case 3:
		{
			static const std::array<size_t, 4> divisors = { 2, 4, 8, 16 };
			std::uniform_int_distribution<size_t> divisorDistribution(0, divisors.size() - 1);
			level._minPrimitivesToSplit = std::max<size_t>(2, level._leafCapacity / divisors[divisorDistribution(rng)]);
			break;
		}
		case 4:
			if (options._conditionalLevels && levelIndex > 0)
				mutateLevelCondition(level, rng, options, domain);
			break;
		case 5:
			if (schema._levels.size() < maxBlocks && schema.totalLevels() < maxDepth)
			{
				const auto insertAt = schema._levels.begin() + static_cast<std::ptrdiff_t>(levelIndex + 1);
				schema._levels.insert(insertAt, randomLevelConfig(rng, options, level._type));
			}
			break;
		case 6:
			if (schema._levels.size() > 1)
				schema._levels.erase(schema._levels.begin() + static_cast<std::ptrdiff_t>(levelIndex));
			break;
		case 7:
			if (schema._levels.size() > 1)
			{
				std::uniform_int_distribution<size_t> swapDistribution(0, schema._levels.size() - 1);
				const size_t other = swapDistribution(rng);
				if (other != levelIndex)
					std::swap(schema._levels[levelIndex], schema._levels[other]);
			}
			break;
		case 8:
			// Axis-policy flip; meaningful only for KDTree/BIH levels, a no-op elsewhere.
			if (level._type == MultiDataStructure::KDTreeNode)
			{
				level._axisPolicy = (level._axisPolicy == "round_robin")
					? std::string("median_longest_axis")
					: std::string("round_robin");
			}
			break;
		case 9:
			mutateAdaptiveLeafCapacity(level, rng);
			break;
		}
	}

	normalizeSchemaForGeneration(schema, options);
	return schema;
}

struct RepairSchemaMutation
{
	SchemaConfig _schema;
	std::string _reason;
};

static void assignRepairPrimitive(
	SchemaLevelConfig& level,
	SchemaPrimitiveKind primitive,
	const Experiments::SchemaGenerationOptions& options)
{
	level._primitiveKind = primitive;
	level._typeName = Config::schemaPrimitiveKindName(primitive);
	level._cpuFallbackType = Config::cpuFallbackForPrimitiveKind(primitive);
	level._type = level._cpuFallbackType;

	if (primitive == SchemaPrimitiveKind::QuadTree)
		level._axisPolicy = "xy";
	else if (primitive == SchemaPrimitiveKind::KDTree || primitive == SchemaPrimitiveKind::BIH)
		level._axisPolicy = "round_robin";
	else
		level._axisPolicy.clear();

	refreshLevelTypeName(level, options);
}

static size_t repairTargetLeafLevel(const SchemaConfig& schema)
{
	if (schema._levels.empty())
		return 0;
	return schema._levels.size() - 1;
}

static void tightenLeafCapacity(
	SchemaLevelConfig& level,
	const Experiments::SchemaGenerationOptions& options)
{
	const size_t minLeaf = std::max<size_t>(1, std::min(options._minLeafCapacity, options._maxLeafCapacity));
	const size_t maxLeaf = std::max(minLeaf, options._maxLeafCapacity);
	level._leafCapacity = clampPowerOfTwo(std::max<size_t>(1, level._leafCapacity / 2), minLeaf, maxLeaf);
	level._minPrimitivesToSplit = std::max<size_t>(2, level._leafCapacity / 4);
}

static void loosenLeafCapacity(
	SchemaLevelConfig& level,
	const Experiments::SchemaGenerationOptions& options)
{
	const size_t minLeaf = std::max<size_t>(1, std::min(options._minLeafCapacity, options._maxLeafCapacity));
	const size_t maxLeaf = std::max(minLeaf, options._maxLeafCapacity);
	level._leafCapacity = clampPowerOfTwo(
		level._leafCapacity >= std::numeric_limits<size_t>::max() / 2
			? level._leafCapacity
			: level._leafCapacity * 2,
		minLeaf,
		maxLeaf);
	level._minPrimitivesToSplit = std::max<size_t>(2, level._leafCapacity / 4);
}

static void addPointGateFromDomain(
	SchemaLevelConfig& level,
	const Experiments::ConditionDomain* domain,
	size_t fallback)
{
	if (domain && !domain->_pointThresholds.empty())
	{
		auto it = std::lower_bound(domain->_pointThresholds.begin(), domain->_pointThresholds.end(), fallback);
		if (it == domain->_pointThresholds.end())
			it = std::prev(domain->_pointThresholds.end());
		level._condition._minPoints = *it;
		return;
	}
	level._condition._minPoints = std::max<size_t>(2, fallback);
}

static Experiments::SchemaRepairDiagnostics diagnoseRepairInternal(
	const Experiments::SchemaCandidate& candidate,
	const std::vector<Experiments::SchemaSearchRecord>& measuredRecords)
{
	Experiments::SchemaRepairDiagnostics diagnostics;
	if (measuredRecords.empty())
		return diagnostics;

	double leafRatioSum = 0.0;
	double testedPerVisitedSum = 0.0;
	double visitedSum = 0.0;
	double testedFractionSum = 0.0;
	double fullContainmentRatioSum = 0.0;
	double nodeLeafRatioSum = 0.0;
	size_t count = 0;

	for (const Experiments::SchemaSearchRecord& record : measuredRecords)
	{
		if (!std::isfinite(record._score))
			continue;

		const double averageLeaf = std::max(record._buildMetrics._averageLeafOccupancy, 1.0);
		const double leafRatio = static_cast<double>(record._buildMetrics._maxLeafOccupancy) / averageLeaf;
		const double visited = std::max(record._queryMetrics._averageVisitedNodes, 0.0);
		const double tested = std::max(record._queryMetrics._averageTestedPoints, 0.0);
		const double testedPerVisited = tested / std::max(visited, 1.0);
		const double testedFraction = record._numPoints > 0
			? tested / static_cast<double>(record._numPoints)
			: 0.0;
		const double fullContainmentRatio = record._queryMetrics._averageFullyContainedNodes / std::max(visited, 1.0);
		const double nodeLeafRatio = record._buildMetrics._numLeaves > 0
			? static_cast<double>(record._buildMetrics._numNodes) / static_cast<double>(record._buildMetrics._numLeaves)
			: 0.0;

		leafRatioSum += leafRatio;
		testedPerVisitedSum += testedPerVisited;
		visitedSum += visited;
		testedFractionSum += testedFraction;
		fullContainmentRatioSum += fullContainmentRatio;
		nodeLeafRatioSum += nodeLeafRatio;
		++count;
	}

	if (count == 0)
		return diagnostics;

	const double invCount = 1.0 / static_cast<double>(count);
	diagnostics._leafOccupancyRatio = leafRatioSum * invCount;
	diagnostics._testedPerVisited = testedPerVisitedSum * invCount;
	diagnostics._visitedPerQuery = visitedSum * invCount;
	diagnostics._testedPointFraction = testedFractionSum * invCount;
	diagnostics._fullContainmentRatio = fullContainmentRatioSum * invCount;

	const size_t deepestConfiguredLeaf = candidate._config._levels.empty()
		? 1
		: candidate._config._levels[repairTargetLeafLevel(candidate._config)]._leafCapacity;

	diagnostics._highLeafOccupancy =
		diagnostics._leafOccupancyRatio >= 3.0 ||
		(!measuredRecords.empty() &&
			measuredRecords.front()._buildMetrics._maxLeafOccupancy >= std::max<size_t>(deepestConfiguredLeaf * 2, 1024));
	diagnostics._testedPointDominated =
		diagnostics._testedPerVisited >= 32.0 ||
		diagnostics._testedPointFraction >= 0.20;
	diagnostics._visitedNodeDominated =
		diagnostics._visitedPerQuery >= 64.0 &&
		diagnostics._testedPerVisited <= 8.0;
	diagnostics._fullContainmentDominated =
		diagnostics._fullContainmentRatio >= 0.20;
	diagnostics._likelySingleChildChains =
		(nodeLeafRatioSum * invCount) >= 2.50 &&
		measuredRecords.front()._buildMetrics._maxDepth > candidate._config.totalLevels() / 2;

	if (diagnostics._highLeafOccupancy)
		diagnostics._bottleneck = "high_leaf_occupancy";
	else if (diagnostics._testedPointDominated)
		diagnostics._bottleneck = "tested_points_dominated";
	else if (diagnostics._visitedNodeDominated)
		diagnostics._bottleneck = "visited_nodes_dominated";
	else if (diagnostics._fullContainmentDominated)
		diagnostics._bottleneck = "full_containment_dominated";
	else if (diagnostics._likelySingleChildChains)
		diagnostics._bottleneck = "single_child_chains";

	return diagnostics;
}

static void addRepairMutation(
	std::vector<RepairSchemaMutation>& mutations,
	SchemaConfig schema,
	const std::string& reason,
	const Experiments::SchemaGenerationOptions& options,
	std::unordered_set<std::string>& seen)
{
	normalizeSchemaForGeneration(schema, options);
	const std::string signature = schemaSignature(schema);
	if (!seen.insert(signature).second)
		return;
	mutations.push_back({ std::move(schema), reason });
}

static std::vector<RepairSchemaMutation> generateRepairSchemaMutations(
	const Experiments::SchemaCandidate& candidate,
	const std::vector<Experiments::SchemaSearchRecord>& measuredRecords,
	const Experiments::SchemaGenerationOptions& options,
	const Experiments::ConditionDomain* domain,
	size_t maxMutations,
	uint32_t seed)
{
	std::vector<RepairSchemaMutation> mutations;
	if (maxMutations == 0 || candidate._config._levels.empty())
		return mutations;

	const Experiments::SchemaRepairDiagnostics diagnostics = diagnoseRepairInternal(candidate, measuredRecords);
	const size_t maxDepth = std::max<size_t>(1, options._maxDepth);
	const size_t maxBlocks = std::max<size_t>(1, std::min(options._maxBlocks, maxDepth));
	const bool canAppendBlock = candidate._config._levels.size() < maxBlocks && candidate._config.totalLevels() < maxDepth;
	std::unordered_set<std::string> seen;
	seen.insert(schemaSignature(candidate._config));
	std::mt19937 rng(seed);

	auto maybeStop = [&]() { return mutations.size() >= maxMutations; };

	if (diagnostics._highLeafOccupancy || diagnostics._testedPointDominated)
	{
		SchemaConfig repaired = candidate._config;
		SchemaLevelConfig& leafLevel = repaired._levels[repairTargetLeafLevel(repaired)];
		tightenLeafCapacity(leafLevel, options);
		if (repaired.totalLevels() < maxDepth)
			++leafLevel._numLevels;
		addRepairMutation(mutations, std::move(repaired), diagnostics._highLeafOccupancy
			? "high_leaf_occupancy"
			: "tested_points_leaf_tighten", options, seen);
		if (maybeStop()) return mutations;
	}

	if (diagnostics._testedPointDominated && canAppendBlock)
	{
		SchemaConfig repaired = candidate._config;
		SchemaLevelConfig micro = repaired._levels.back();
		assignRepairPrimitive(micro,
			measuredRecords.empty() || measuredRecords.front()._knnWeight < 0.5
				? SchemaPrimitiveKind::BVH
				: SchemaPrimitiveKind::KDTree,
			options);
		micro._numLevels = std::min<size_t>(3, maxDepth - repaired.totalLevels());
		micro._leafCapacity = std::max<size_t>(options._minLeafCapacity, repaired._levels.back()._leafCapacity / 2);
		micro._leafCapacity = clampPowerOfTwo(micro._leafCapacity, std::max<size_t>(1, options._minLeafCapacity), std::max<size_t>(1, options._maxLeafCapacity));
		micro._minPrimitivesToSplit = std::max<size_t>(2, micro._leafCapacity / 4);
		addPointGateFromDomain(micro, domain, std::max<size_t>(micro._leafCapacity * 2, 32));
		repaired._levels.push_back(micro);
		addRepairMutation(mutations, std::move(repaired), "tested_points_micro_index", options, seen);
		if (maybeStop()) return mutations;
	}

	if (diagnostics._visitedNodeDominated)
	{
		SchemaConfig repaired = candidate._config;
		SchemaLevelConfig& root = repaired._levels.front();
		if (measuredRecords.empty() || measuredRecords.front()._pointFeatures._flatnessScore >= measuredRecords.front()._pointFeatures._verticalityScore)
			assignRepairPrimitive(root, SchemaPrimitiveKind::QuadTree, options);
		else
			assignRepairPrimitive(root, SchemaPrimitiveKind::Octree, options);
		loosenLeafCapacity(root, options);
		if (root._numLevels > 1)
			--root._numLevels;
		addRepairMutation(mutations, std::move(repaired), "visited_nodes_coarsen_root", options, seen);
		if (maybeStop()) return mutations;
	}

	if (diagnostics._fullContainmentDominated)
	{
		SchemaConfig repaired = candidate._config;
		SchemaLevelConfig& root = repaired._levels.front();
		const bool allowGrid = !queryMinimalPrimitiveProfile(options);
		if (allowGrid)
			assignRepairPrimitive(root, SchemaPrimitiveKind::RegularGrid, options);
		else if (measuredRecords.empty() || measuredRecords.front()._pointFeatures._flatnessScore >= measuredRecords.front()._pointFeatures._verticalityScore)
			assignRepairPrimitive(root, SchemaPrimitiveKind::QuadTree, options);
		else
			assignRepairPrimitive(root, SchemaPrimitiveKind::Octree, options);
		loosenLeafCapacity(root, options);
		addRepairMutation(mutations, std::move(repaired), "full_containment_coarse_grid", options, seen);
		if (maybeStop()) return mutations;
	}

	if (diagnostics._likelySingleChildChains)
	{
		SchemaConfig repaired = candidate._config;
		auto kdLevel = std::find_if(repaired._levels.begin(), repaired._levels.end(), [](const SchemaLevelConfig& level) {
			return level._type == MultiDataStructure::KDTreeNode || isBIHLevelName(level._typeName);
		});
		if (kdLevel != repaired._levels.end())
		{
			kdLevel->_axisPolicy = kdLevel->_axisPolicy == "round_robin"
				? "median_longest_axis"
				: "round_robin";
		}
		else
		{
			std::uniform_int_distribution<int> policyDistribution(0, 2);
			static const std::array<const char*, 3> policies = { "xy", "ignore_shortest", "ignore_z" };
			repaired._levels.front()._axisPolicy = policies[policyDistribution(rng)];
		}
		addRepairMutation(mutations, std::move(repaired), "single_child_axis_policy", options, seen);
		if (maybeStop()) return mutations;
	}

	if (mutations.empty())
	{
		SchemaConfig repaired = candidate._config;
		SchemaLevelConfig& leafLevel = repaired._levels[repairTargetLeafLevel(repaired)];
		if (diagnostics._testedPerVisited > 8.0)
			tightenLeafCapacity(leafLevel, options);
		else
			loosenLeafCapacity(repaired._levels.front(), options);
		addRepairMutation(mutations, std::move(repaired), "balanced_leaf_probe", options, seen);
	}

	return mutations;
}

static std::vector<SearchDataset> makeSyntheticDatasets(size_t scale)
{
	const size_t n = std::max<size_t>(64, scale);
	std::vector<SearchDataset> datasets;
	datasets.reserve(3);

	SearchDataset flat;
	flat._name = "synthetic_flat_terrain";
	flat._source = "synthetic";
	flat._cloud = SyntheticPointClouds::generateFlatTerrain(n, 100.0f, 100.0f, 0.05f, 101);
	datasets.push_back(std::move(flat));

	SearchDataset facade;
	facade._name = "synthetic_facade";
	facade._source = "synthetic";
	facade._cloud = SyntheticPointClouds::generateFacade(n, 60.0f, 40.0f, 0.2f, 202);
	datasets.push_back(std::move(facade));

	SearchDataset urban;
	urban._name = "synthetic_urban_mixed";
	urban._source = "synthetic";
	urban._cloud = SyntheticPointClouds::generateUrbanMixed(n, n, 5, 303);
	datasets.push_back(std::move(urban));

	return datasets;
}

static std::vector<SearchDataset> loadDatasets(const Experiments::SchemaSearchOptions& options)
{
	std::vector<SearchDataset> datasets;

	if (options._includeSyntheticDatasets)
	{
		std::vector<SearchDataset> synthetic = makeSyntheticDatasets(options._syntheticScale);
		datasets.insert(datasets.end(), std::make_move_iterator(synthetic.begin()), std::make_move_iterator(synthetic.end()));
	}

	for (const std::string& inputPath : options._inputPaths)
	{
		SearchDataset dataset;
		dataset._name = datasetNameFromPath(inputPath);
		dataset._source = inputPath;
		dataset._cloud = PointCloud::load(inputPath, { options._useBinaryCache, options._rebuildBinaryCache });
		if (dataset._cloud.empty())
			throw std::runtime_error("Point cloud is empty: " + inputPath);
		datasets.push_back(std::move(dataset));
	}

	if (datasets.empty())
		throw std::runtime_error("Schema search requires at least one synthetic dataset or --input path");

	return datasets;
}

static std::vector<Experiments::SchemaCandidate> loadSchemas(const std::vector<std::string>& configuredPaths, bool includeConfiguredSchemas)
{
	if (!includeConfiguredSchemas)
		return {};

	const std::vector<std::string>& paths = configuredPaths.empty() ? defaultSchemaPaths() : configuredPaths;
	std::vector<Experiments::SchemaCandidate> schemas;
	schemas.reserve(paths.size());

	for (const std::string& schemaPath : paths)
	{
		Experiments::SchemaCandidate loaded;
		loaded._path = schemaPath;
		loaded._config = Config::loadSchemaConfig(schemaPath);
		loaded._name = loaded._config._name;
		schemas.push_back(std::move(loaded));
	}

	return schemas;
}

static void appendGeneratedSchemas(
	std::vector<Experiments::SchemaCandidate>& schemas,
	const Experiments::SchemaGenerationOptions& options,
	const Experiments::ConditionDomain* domain = nullptr)
{
	if (options._count == 0)
		return;

	// Pass the per-cloud condition domain so the generator calibrates conditional thresholds to actual cloud statistics instead of generic fallbacks that never fire.
	std::vector<Experiments::SchemaCandidate> generated = Experiments::generateSchemaCandidates(options, domain);
	schemas.insert(schemas.end(), std::make_move_iterator(generated.begin()), std::make_move_iterator(generated.end()));
}

static const std::vector<std::string>& baselineSchemaPaths(bool cudaEvaluator)
{
	static const std::vector<std::string> cpuPaths = {
		"configs/schemas/quadtree.json",
		"configs/schemas/octree.json",
		"configs/schemas/kdtree.json",
		"configs/schemas/bvh.json",
	};
	// CUDA keeps the GPU-native single-block controls (builders differ in query behavior); CPU discovery omits these aliases since they collapse to the base families.
	static const std::vector<std::string> cudaPaths = {
		"configs/schemas/quadtree.json",
		"configs/schemas/octree.json",
		"configs/schemas/kdtree.json",
		"configs/schemas/bvh.json",
		"configs/schemas/lbvh.json",
		"configs/schemas/karras_octree.json",
		"configs/schemas/regular_grid.json",
		"configs/schemas/hgrid.json",
		"configs/schemas/bih.json",
	};
	return cudaEvaluator ? cudaPaths : cpuPaths;
}

static void appendBaselineSchemas(std::vector<Experiments::SchemaCandidate>& schemas, bool cudaEvaluator)
{
	std::unordered_set<std::string> existing;
	for (const Experiments::SchemaCandidate& candidate : schemas)
		existing.insert(candidate._path);

	size_t added = 0;
	for (const std::string& path : baselineSchemaPaths(cudaEvaluator))
	{
		const std::filesystem::path resolved = resolveExistingPath(path);
		const std::string key = resolved.string();
		if (existing.find(key) != existing.end() || existing.find(path) != existing.end())
			continue;

		Experiments::SchemaCandidate baseline;
		baseline._path = key;
		try
		{
			baseline._config = Config::loadSchemaConfig(key);
		}
		catch (const std::exception& exception)
		{
			std::cerr << "Warning: skipped baseline schema " << path << " (" << exception.what() << ")\n";
			continue;
		}
		baseline._name = baseline._config._name;
		baseline._isBaseline = true;
		schemas.push_back(std::move(baseline));
		existing.insert(key);
		++added;
	}

	if (added > 0)
		std::cout << "  baselines: injected " << added << " single-block schema(s) as controls\n";
}

static std::string candidateKey(const std::string& name, const std::string& path)
{
	return name + "\n" + path;
}

static std::vector<Experiments::SchemaCandidate> selectBenchmarkSchemas(
	const Experiments::SchemaSearchOptions& options,
	const SearchDataset& dataset,
	const Experiments::WorkloadProfile& workload,
	const std::vector<Experiments::SchemaCandidate>& schemas,
	const std::optional<Experiments::SchemaSelectorModel>& rankModel)
{
	if (schemas.empty())
		return {};

	if (options._benchmarkTopK == 0 || options._benchmarkTopK >= schemas.size())
		return schemas;

	if (!rankModel.has_value())
	{
		if (!options._estimatePrefilter)
			return std::vector<Experiments::SchemaCandidate>(schemas.begin(), schemas.begin() + options._benchmarkTopK);

		// Cheap model-free pre-filter: rank by the zero-build query-cost estimate and keep the cheapest K rather than an arbitrary first-K prefix.
		const Experiments::ScoreWeights weights = effectiveScoreWeights(workload, options);
		const Experiments::PointCloudFeatures features = Experiments::extractPointCloudFeatures(dataset._cloud);
		const Experiments::WorkloadFeatures workloadFeatures = Experiments::extractWorkloadFeatures(workload, weights);

		std::vector<size_t> order(schemas.size());
		std::iota(order.begin(), order.end(), size_t(0));
		std::vector<double> cost(schemas.size());
		for (size_t i = 0; i < schemas.size(); ++i)
			cost[i] = Experiments::estimateSchemaQueryCost(schemas[i]._config, features, workloadFeatures, weights._visitProxyAlpha);
		std::stable_sort(order.begin(), order.end(), [&cost](size_t a, size_t b) { return cost[a] < cost[b]; });

		std::vector<Experiments::SchemaCandidate> selected;
		selected.reserve(options._benchmarkTopK);
		for (size_t k = 0; k < options._benchmarkTopK && k < order.size(); ++k)
			selected.push_back(schemas[order[k]]);
		std::cout << "  estimate pre-filter: kept " << selected.size() << " of " << schemas.size()
			<< " candidates by zero-build cost on '" << dataset._name << "'\n";
		return selected;
	}

	const std::vector<Experiments::CandidatePrediction> predictions = Experiments::scoreSchemaCandidates(
		rankModel.value(),
		workload,
		dataset._cloud,
		schemas);

	std::unordered_map<std::string, const Experiments::SchemaCandidate*> byKey;
	byKey.reserve(schemas.size());
	for (const Experiments::SchemaCandidate& schema : schemas)
		byKey[candidateKey(schema._config._name, schema._path)] = &schema;

	std::vector<Experiments::SchemaCandidate> selected;
	selected.reserve(std::min(options._benchmarkTopK, predictions.size()));
	for (const Experiments::CandidatePrediction& prediction : predictions)
	{
		const auto found = byKey.find(candidateKey(prediction._schemaName, prediction._schemaPath));
		if (found == byKey.end())
			continue;

		selected.push_back(*found->second);
		if (selected.size() >= options._benchmarkTopK)
			break;
	}

	return selected;
}

static std::vector<Experiments::WorkloadProfile> loadWorkloads(const Experiments::SchemaSearchOptions& options)
{
	const std::vector<std::string>& paths = options._workloadPaths.empty() ? defaultWorkloadPaths() : options._workloadPaths;
	std::vector<Experiments::WorkloadProfile> workloads;
	workloads.reserve(paths.size());

	for (const std::string& workloadPath : paths)
	{
		Experiments::WorkloadProfile profile = Experiments::loadWorkloadProfile(workloadPath);
		if (options._queryCountOverride > 0)
			profile._numQueries = options._queryCountOverride;
		if (options._knnKOverride > 0)
			profile._knnK = options._knnKOverride;
		if (options._querySeedOverride)
			profile._querySeed = options._querySeed;
		workloads.push_back(std::move(profile));
	}

	return workloads;
}

static std::vector<double> queryTypeWeights(const Experiments::WorkloadProfile& profile)
{
	std::vector<double> weights = {
		std::max(0.0, profile._rangeWeight),
		std::max(0.0, profile._radiusWeight),
		std::max(0.0, profile._knnWeight),
	};

	if (weights[0] == 0.0 && weights[1] == 0.0 && weights[2] == 0.0)
		weights = { 1.0, 1.0, 1.0 };

	return weights;
}

enum class QueryCenterMode
{
	Random,
	Dense,
	Boundary,
	Outside,
};

struct StratifiedQuerySpec
{
	PreparedQueryKind _kind = PreparedQueryKind::Range;
	std::string _name;
	double _scale = 0.01;
	QueryCenterMode _centerMode = QueryCenterMode::Random;
};

static std::vector<StratifiedQuerySpec> stratifiedQuerySpecs(const Experiments::WorkloadProfile& profile)
{
	std::vector<StratifiedQuerySpec> specs;
	const double rangeMin = std::min(profile._rangeScaleMin, profile._rangeScaleMax);
	const double rangeMax = std::max(profile._rangeScaleMin, profile._rangeScaleMax);
	const double rangeMid = 0.5 * (rangeMin + rangeMax);
	const double radiusMin = std::min(profile._radiusScaleMin, profile._radiusScaleMax);
	const double radiusMax = std::max(profile._radiusScaleMin, profile._radiusScaleMax);
	const double radiusMid = 0.5 * (radiusMin + radiusMax);

	if (profile._rangeWeight > 0.0)
	{
		specs.push_back({ PreparedQueryKind::Range, "range_small", rangeMin, QueryCenterMode::Random });
		specs.push_back({ PreparedQueryKind::Range, "range_medium", rangeMid, QueryCenterMode::Random });
		specs.push_back({ PreparedQueryKind::Range, "range_large", rangeMax, QueryCenterMode::Random });
		specs.push_back({ PreparedQueryKind::Range, "range_near_empty", rangeMin, QueryCenterMode::Outside });
		specs.push_back({ PreparedQueryKind::Range, "range_dense", rangeMin, QueryCenterMode::Dense });
	}
	if (profile._radiusWeight > 0.0)
	{
		specs.push_back({ PreparedQueryKind::Radius, "radius_small", radiusMin, QueryCenterMode::Random });
		specs.push_back({ PreparedQueryKind::Radius, "radius_medium", radiusMid, QueryCenterMode::Random });
		specs.push_back({ PreparedQueryKind::Radius, "radius_large", radiusMax, QueryCenterMode::Random });
		specs.push_back({ PreparedQueryKind::Radius, "radius_dense", radiusMin, QueryCenterMode::Dense });
		specs.push_back({ PreparedQueryKind::Radius, "radius_boundary", radiusMid, QueryCenterMode::Boundary });
	}
	if (profile._knnWeight > 0.0)
	{
		specs.push_back({ PreparedQueryKind::Knn, "knn_dense", 0.0, QueryCenterMode::Dense });
		specs.push_back({ PreparedQueryKind::Knn, "knn_outside", 0.0, QueryCenterMode::Outside });
		specs.push_back({ PreparedQueryKind::Knn, "knn_boundary", 0.0, QueryCenterMode::Boundary });
	}
	if (specs.empty())
	{
		specs.push_back({ PreparedQueryKind::Range, "range_medium", rangeMid, QueryCenterMode::Random });
		specs.push_back({ PreparedQueryKind::Radius, "radius_medium", radiusMid, QueryCenterMode::Random });
		specs.push_back({ PreparedQueryKind::Knn, "knn_dense", 0.0, QueryCenterMode::Dense });
	}
	return specs;
}

static glm::vec3 centerForStratum(std::mt19937& rng, const PointCloud& cloud, QueryCenterMode mode)
{
	switch (mode)
	{
	case QueryCenterMode::Dense:
		return sampledCloudPoint(rng, cloud);
	case QueryCenterMode::Boundary:
		return boundaryPoint(rng, cloud);
	case QueryCenterMode::Outside:
		return outsidePoint(rng, cloud);
	case QueryCenterMode::Random:
	default:
		return randomPointInBounds(rng, cloud.bounds());
	}
}

static PreparedWorkload prepareWorkloadProfile(
	const Experiments::WorkloadProfile& profile,
	const PointCloud& cloud,
	bool cudaEvaluator)
{
	PreparedWorkload prepared;
	if (profile._numQueries == 0)
		return prepared;

	std::mt19937 rng(profile._querySeed);
	if (profile._stratifyQueries)
	{
		const std::vector<StratifiedQuerySpec> specs = stratifiedQuerySpecs(profile);
		bool nextRangeIsCount = false;
		if (cudaEvaluator)
		{
			prepared._cudaQueries.reserve(profile._numQueries);
			prepared._cudaStrata.reserve(profile._numQueries);
			for (size_t i = 0; i < profile._numQueries; ++i)
			{
				const StratifiedQuerySpec& spec = specs[i % specs.size()];
				const glm::vec3 center = centerForStratum(rng, cloud, spec._centerMode);
				PointGpu::Query query;
				if (spec._kind == PreparedQueryKind::Range)
				{
					query._type = nextRangeIsCount ? PointGpu::QueryType::CountRange : PointGpu::QueryType::Range;
					query._bounds = queryBoxAtScale(center, cloud, spec._scale);
					if (nextRangeIsCount)
						++prepared._countRangeQueries;
					else
						++prepared._rangeQueries;
					nextRangeIsCount = !nextRangeIsCount;
				}
				else if (spec._kind == PreparedQueryKind::Radius)
				{
					query._type = PointGpu::QueryType::Radius;
					query.center = center;
					const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.001f));
					const float largestRange = std::max({ range.x, range.y, range.z, 1.0f });
					query._radius = largestRange * static_cast<float>(spec._scale);
					++prepared._radiusQueries;
				}
				else
				{
					query._type = PointGpu::QueryType::Knn;
					query.center = center;
					query._k = profile._knnK;
					++prepared._knnQueries;
				}
				prepared._cudaQueries.push_back(query);
				prepared._cudaStrata.push_back(spec._name);
			}
			return prepared;
		}

		prepared._cpuQueries.reserve(profile._numQueries);
		for (size_t i = 0; i < profile._numQueries; ++i)
		{
			const StratifiedQuerySpec& spec = specs[i % specs.size()];
			const glm::vec3 center = centerForStratum(rng, cloud, spec._centerMode);
			PreparedCpuQuery query;
			query._kind = spec._kind;
			query._stratum = spec._name;
			if (spec._kind == PreparedQueryKind::Range)
			{
				query._bounds = queryBoxAtScale(center, cloud, spec._scale);
				++prepared._rangeQueries;
			}
			else if (spec._kind == PreparedQueryKind::Radius)
			{
				query.center = center;
				const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.001f));
				const float largestRange = std::max({ range.x, range.y, range.z, 1.0f });
				query._radius = largestRange * static_cast<float>(spec._scale);
				++prepared._radiusQueries;
			}
			else
			{
				query.center = center;
				++prepared._knnQueries;
			}
			prepared._cpuQueries.push_back(query);
		}
		return prepared;
	}

	if (cudaEvaluator)
	{
		std::vector<double> weights = queryTypeWeights(profile);

		prepared._cudaQueries.reserve(profile._numQueries);
		std::discrete_distribution<size_t> queryType(weights.begin(), weights.end());
		bool nextRangeIsCount = false;
		for (size_t i = 0; i < profile._numQueries; ++i)
		{
			const size_t type = queryType(rng);
			PointGpu::Query query;
			if (type == 0)
			{
				query._type = nextRangeIsCount ? PointGpu::QueryType::CountRange : PointGpu::QueryType::Range;
				query._bounds = randomQueryBox(rng, cloud, profile);
				if (nextRangeIsCount)
					++prepared._countRangeQueries;
				else
					++prepared._rangeQueries;
				nextRangeIsCount = !nextRangeIsCount;
			}
			else if (type == 1)
			{
				query._type = PointGpu::QueryType::Radius;
				query.center = randomPointInBounds(rng, cloud.bounds());
				query._radius = randomQueryRadius(rng, cloud, profile);
				++prepared._radiusQueries;
			}
			else
			{
				query._type = PointGpu::QueryType::Knn;
				query.center = randomPointInBounds(rng, cloud.bounds());
				query._k = profile._knnK;
				++prepared._knnQueries;
			}
			prepared._cudaQueries.push_back(query);
			prepared._cudaStrata.push_back({});
		}

		return prepared;
	}

	prepared._cpuQueries.reserve(profile._numQueries);
	const std::vector<double> weights = queryTypeWeights(profile);
	std::discrete_distribution<size_t> queryType(weights.begin(), weights.end());
	for (size_t i = 0; i < profile._numQueries; ++i)
	{
		const size_t type = queryType(rng);
		PreparedCpuQuery query;
		if (type == 0)
		{
			query._kind = PreparedQueryKind::Range;
			query._bounds = randomQueryBox(rng, cloud, profile);
			query._stratum = "range";
			++prepared._rangeQueries;
		}
		else if (type == 1)
		{
			query._kind = PreparedQueryKind::Radius;
			query.center = randomPointInBounds(rng, cloud.bounds());
			query._radius = randomQueryRadius(rng, cloud, profile);
			query._stratum = "radius";
			++prepared._radiusQueries;
		}
		else
		{
			query._kind = PreparedQueryKind::Knn;
			query.center = randomPointInBounds(rng, cloud.bounds());
			query._stratum = "knn";
			++prepared._knnQueries;
		}
		prepared._cpuQueries.push_back(query);
	}

	return prepared;
}

static std::string queryKindName(PreparedQueryKind kind)
{
	switch (kind)
	{
	case PreparedQueryKind::Radius:
		return "radius";
	case PreparedQueryKind::Knn:
		return "knn";
	default:
		return "range";
	}
}

static std::string queryKindName(PointGpu::QueryType kind)
{
	switch (kind)
	{
	case PointGpu::QueryType::CountRange:
		return "count_range";
	case PointGpu::QueryType::Radius:
		return "radius";
	case PointGpu::QueryType::Knn:
		return "knn";
	default:
		return "range";
	}
}

static bool shouldWriteCsvHeader(const std::filesystem::path& path)
{
	std::error_code error;
	return !std::filesystem::exists(path, error) || std::filesystem::file_size(path, error) == 0;
}

using QueryBreakdown = PointSpatialIndex::QueryStats::QueryBreakdown;

static std::vector<std::pair<std::string, size_t>> sortedBreakdownMap(const std::unordered_map<std::string, size_t>& values)
{
	std::vector<std::pair<std::string, size_t>> sorted(values.begin(), values.end());
	std::sort(sorted.begin(), sorted.end(), [](const auto& left, const auto& right) {
		return left.first < right.first;
	});
	return sorted;
}

static std::string formatDepthBreakdown(const QueryBreakdown& breakdown)
{
	std::ostringstream output;
	bool first = true;
	for (size_t depth = 0; depth < breakdown.visitedByDepth.size(); ++depth)
	{
		const size_t count = breakdown.visitedByDepth[depth];
		if (count == 0)
			continue;
		if (!first)
			output << ';';
		first = false;
		output << depth << ':' << count;
	}
	return output.str();
}

static std::string formatMapBreakdown(const std::unordered_map<std::string, size_t>& values)
{
	std::ostringstream output;
	bool first = true;
	for (const auto& [name, count] : sortedBreakdownMap(values))
	{
		if (!first)
			output << ';';
		first = false;
		output << name << ':' << count;
	}
	return output.str();
}

static std::string formatStratumSummary(const std::map<std::string, Experiments::QueryMetrics>& metricsByStratum)
{
	std::ostringstream output;
	bool first = true;
	for (const auto& [name, metrics] : metricsByStratum)
	{
		if (metrics._totalQueries == 0)
			continue;
		if (!first)
			output << ';';
		first = false;
		output << name
			<< "|q=" << metrics._totalQueries
			<< "|avg_ms=" << metrics._averageLatencyMs
			<< "|p95_ms=" << metrics._p95LatencyMs
			<< "|visited=" << metrics._averageVisitedNodes
			<< "|tested=" << metrics._averageTestedPoints;
	}
	return output.str();
}

static void writeQueryTraceHeader(std::ostream& output)
{
	output
		<< "dataset_name,dataset_source,schema_name,schema_path,workload_name,score_stage,"
		<< "query_id,query_type,query_stratum,bounds_min_x,bounds_min_y,bounds_min_z,bounds_max_x,bounds_max_y,bounds_max_z,"
		<< "center_x,center_y,center_z,radius,k,latency_ms,visited_nodes,tested_points,returned_points,"
		<< "fully_contained_nodes,visited_by_depth,visited_by_structure,tested_points_by_structure,"
		<< "fully_contained_by_structure,backend,query_seed\n";
}

static void appendQueryTraceRow(
	std::ostream& output,
	const SearchDataset& dataset,
	const Experiments::SchemaCandidate& schema,
	const Experiments::WorkloadProfile& workload,
	const std::string& scoreStage,
	size_t queryId,
	const std::string& queryType,
	const std::string& queryStratum,
	const AABB* bounds,
	const glm::vec3* center,
	float radius,
	size_t k,
	const PointSpatialIndex::QueryStats& stats,
	const std::string& backend)
{
	output << std::fixed << std::setprecision(6)
		<< csvEscape(dataset._name) << ','
		<< csvEscape(dataset._source) << ','
		<< csvEscape(schema._config._name) << ','
		<< csvEscape(schema._path) << ','
		<< csvEscape(workload._name) << ','
		<< csvEscape(scoreStage) << ','
		<< queryId << ','
		<< csvEscape(queryType) << ',';
	output << csvEscape(queryStratum) << ',';

	if (bounds)
	{
		output
			<< bounds->min().x << ','
			<< bounds->min().y << ','
			<< bounds->min().z << ','
			<< bounds->max().x << ','
			<< bounds->max().y << ','
			<< bounds->max().z << ',';
	}
	else
	{
		output << ",,,,,,";
	}

	if (center)
	{
		output
			<< center->x << ','
			<< center->y << ','
			<< center->z << ',';
	}
	else
	{
		output << ",,,";
	}

	output
		<< radius << ','
		<< k << ','
		<< stats._elapsedMs << ','
		<< stats._visitedNodes << ','
		<< stats._testedPoints << ','
		<< stats._returnedPoints << ','
		<< stats._fullyContainedNodes << ','
		<< csvEscape(formatDepthBreakdown(stats._breakdown)) << ','
		<< csvEscape(formatMapBreakdown(stats._breakdown._visitedByStructure)) << ','
		<< csvEscape(formatMapBreakdown(stats._breakdown._testedPointsByStructure)) << ','
		<< csvEscape(formatMapBreakdown(stats._breakdown._fullyContainedByStructure)) << ','
		<< csvEscape(backend) << ','
		<< workload._querySeed << '\n';
}

static void appendSchemaQueryTrace(
	const std::string& tracePath,
	const SearchDataset& dataset,
	const Experiments::SchemaCandidate& schema,
	const Experiments::WorkloadProfile& workload,
	const PreparedWorkload& prepared,
	const WorkloadRun& run,
	const std::string& backend,
	const std::string& scoreStage)
{
	if (tracePath.empty() || run._samples.empty())
		return;

	createParentDirectory(tracePath);
	const std::filesystem::path path(tracePath);
	const bool writeHeader = shouldWriteCsvHeader(path);
	std::ofstream output(path, std::ios::app);
	if (!output.is_open())
		throw std::runtime_error("Unable to open query trace path: " + tracePath);
	if (writeHeader)
		writeQueryTraceHeader(output);

	if (!prepared._cpuQueries.empty())
	{
		const size_t count = std::min(prepared._cpuQueries.size(), run._samples.size());
		for (size_t i = 0; i < count; ++i)
		{
			const PreparedCpuQuery& query = prepared._cpuQueries[i];
			const bool hasBounds = query._kind == PreparedQueryKind::Range;
			const bool hasCenter = query._kind == PreparedQueryKind::Radius || query._kind == PreparedQueryKind::Knn;
			appendQueryTraceRow(
				output,
				dataset,
				schema,
				workload,
				scoreStage,
				i,
				queryKindName(query._kind),
				query._stratum,
				hasBounds ? &query._bounds : nullptr,
				hasCenter ? &query.center : nullptr,
				query._kind == PreparedQueryKind::Radius ? query._radius : 0.0f,
				query._kind == PreparedQueryKind::Knn ? workload._knnK : size_t(0),
				run._samples[i],
				backend);
		}
		return;
	}

	const size_t count = std::min(prepared._cudaQueries.size(), run._samples.size());
	for (size_t i = 0; i < count; ++i)
	{
		const PointGpu::Query& query = prepared._cudaQueries[i];
		const std::string queryStratum = i < prepared._cudaStrata.size() ? prepared._cudaStrata[i] : std::string();
		const bool hasBounds = query._type == PointGpu::QueryType::Range || query._type == PointGpu::QueryType::CountRange;
		const bool hasCenter = query._type == PointGpu::QueryType::Radius || query._type == PointGpu::QueryType::Knn;
		appendQueryTraceRow(
			output,
			dataset,
			schema,
			workload,
			scoreStage,
			i,
			queryKindName(query._type),
			queryStratum,
			hasBounds ? &query._bounds : nullptr,
			hasCenter ? &query.center : nullptr,
			query._type == PointGpu::QueryType::Radius ? query._radius : 0.0f,
			query._type == PointGpu::QueryType::Knn ? query._k : size_t(0),
			run._samples[i],
			backend);
	}
}

// Summarizes per-repeat latency noise into QueryMetrics reliability fields, deterministically (fixed bootstrap seed); a single repeat degenerates the CI to the point estimate.
static void fillLatencyReliability(Experiments::QueryMetrics& metrics, const std::vector<double>& repeatMeanLatencies)
{
	if (repeatMeanLatencies.empty())
		return;

	const double n = static_cast<double>(repeatMeanLatencies.size());
	const double mean = std::accumulate(repeatMeanLatencies.begin(), repeatMeanLatencies.end(), 0.0) / n;
	double variance = 0.0;
	for (double value : repeatMeanLatencies)
		variance += (value - mean) * (value - mean);
	variance /= n;

	metrics._measurementRepeats = repeatMeanLatencies.size();
	metrics._latencyMeanMs = mean;
	metrics._latencyStdDevMs = std::sqrt(variance);
	metrics._latencyCoeffVar = mean > 0.0 ? metrics._latencyStdDevMs / mean : 0.0;
	const auto [ciMean, ciLow, ciHigh] = Experiments::bootstrapMeanCI(repeatMeanLatencies, 1000, 0x5EED1234u);
	(void)ciMean;
	metrics._latencyCiLowMs = ciLow;
	metrics._latencyCiHighMs = ciHigh;
}

static WorkloadRun runWorkloadProfile(const PreparedWorkload& prepared, size_t knnK, const PointSpatialIndex& index, size_t repeats = 1)
{
	WorkloadRun result;
	if (prepared._cpuQueries.empty())
		return result;

	std::vector<PointSpatialIndex::QueryStats> samples;
	std::vector<PointSpatialIndex::QueryStats> rangeSamples;
	std::vector<PointSpatialIndex::QueryStats> radiusSamples;
	std::vector<PointSpatialIndex::QueryStats> knnSamples;
	std::map<std::string, std::vector<PointSpatialIndex::QueryStats>> stratumSamples;
	samples.reserve(prepared._cpuQueries.size());
	rangeSamples.reserve(prepared._rangeQueries);
	radiusSamples.reserve(prepared._radiusQueries);
	knnSamples.reserve(prepared._knnQueries);

	for (const PreparedCpuQuery& query : prepared._cpuQueries)
	{
		if (query._kind == PreparedQueryKind::Range)
		{
			PointSpatialIndex::QueryStats stats = index.rangeQuery(query._bounds)._stats;
			stratumSamples[query._stratum].push_back(stats);
			rangeSamples.push_back(stats);
			samples.push_back(std::move(stats));
			++result._rangeQueries;
			continue;
		}

		if (query._kind == PreparedQueryKind::Radius)
		{
			PointSpatialIndex::QueryStats stats = index.radiusQuery(query.center, query._radius)._stats;
			stratumSamples[query._stratum].push_back(stats);
			radiusSamples.push_back(stats);
			samples.push_back(std::move(stats));
			++result._radiusQueries;
			continue;
		}

		PointSpatialIndex::QueryStats stats = index.knnQuery(query.center, knnK)._stats;
		stratumSamples[query._stratum].push_back(stats);
		knnSamples.push_back(stats);
		samples.push_back(std::move(stats));
		++result._knnQueries;
	}

	result._samples = samples;
	result._metrics = Experiments::summarizeQueryStats(samples);
	result._rangeMetrics = Experiments::summarizeQueryStats(rangeSamples);
	result._radiusMetrics = Experiments::summarizeQueryStats(radiusSamples);
	result._knnMetrics = Experiments::summarizeQueryStats(knnSamples);
	for (const auto& [name, stratum] : stratumSamples)
	{
		if (!name.empty())
			result._stratumMetrics[name] = Experiments::summarizeQueryStats(stratum);
	}
	result._stratumSummary = formatStratumSummary(result._stratumMetrics);

	if (repeats > 1)
	{
		std::vector<double> repeatMeanLatencies;
		repeatMeanLatencies.reserve(repeats);
		repeatMeanLatencies.push_back(result._metrics._averageLatencyMs);
		for (size_t r = 1; r < repeats; ++r)
		{
			double totalMs = 0.0;
			for (const PreparedCpuQuery& query : prepared._cpuQueries)
			{
				if (query._kind == PreparedQueryKind::Range)
					totalMs += index.rangeQuery(query._bounds)._stats._elapsedMs;
				else if (query._kind == PreparedQueryKind::Radius)
					totalMs += index.radiusQuery(query.center, query._radius)._stats._elapsedMs;
				else
					totalMs += index.knnQuery(query.center, knnK)._stats._elapsedMs;
			}
			repeatMeanLatencies.push_back(totalMs / static_cast<double>(prepared._cpuQueries.size()));
		}
		fillLatencyReliability(result._metrics, repeatMeanLatencies);
	}

	return result;
}

template <typename IndexType>
static WorkloadRun runCudaWorkloadProfile(
	const PreparedWorkload& prepared,
	const IndexType& index,
	const PointGpu::Options& cudaOptions,
	size_t repeats = 1)
{
	WorkloadRun result;
	if (prepared._cudaQueries.empty())
		return result;

	const PointGpu::QueryResult queryResult = index.query(prepared._cudaQueries, cudaOptions);
	result._metrics = queryResult._metrics;
	result._gpuQueryMs = queryResult._gpuQueryTimeMs;
	result._samples.reserve(queryResult._samples.size());
	std::vector<PointSpatialIndex::QueryStats> rangeSamples;
	std::vector<PointSpatialIndex::QueryStats> countRangeSamples;
	std::vector<PointSpatialIndex::QueryStats> radiusSamples;
	std::vector<PointSpatialIndex::QueryStats> knnSamples;
	std::map<std::string, std::vector<PointSpatialIndex::QueryStats>> stratumSamples;
	rangeSamples.reserve(queryResult._rangeQueries);
	countRangeSamples.reserve(queryResult._countRangeQueries);
	radiusSamples.reserve(queryResult._radiusQueries);
	knnSamples.reserve(queryResult._knnQueries);
	const size_t sampleCount = std::min(queryResult._samples.size(), prepared._cudaQueries.size());
	for (size_t i = 0; i < sampleCount; ++i)
	{
		const PointGpu::QuerySample& sample = queryResult._samples[i];
		PointSpatialIndex::QueryStats stats;
		stats._visitedNodes = sample._visitedNodes;
		stats._testedPoints = sample._testedPoints;
		stats._returnedPoints = sample._returnedPoints;
		stats._elapsedMs = sample._elapsedMs;
		if (i < prepared._cudaStrata.size() && !prepared._cudaStrata[i].empty())
			stratumSamples[prepared._cudaStrata[i]].push_back(stats);
		switch (prepared._cudaQueries[i]._type)
		{
		case PointGpu::QueryType::Range:
			rangeSamples.push_back(stats);
			break;
		case PointGpu::QueryType::CountRange:
			countRangeSamples.push_back(stats);
			break;
		case PointGpu::QueryType::Radius:
			radiusSamples.push_back(stats);
			break;
		case PointGpu::QueryType::Knn:
			knnSamples.push_back(stats);
			break;
		}
		result._samples.push_back(stats);
	}
	result._rangeQueries = queryResult._rangeQueries;
	result._countRangeQueries = queryResult._countRangeQueries;
	result._radiusQueries = queryResult._radiusQueries;
	result._knnQueries = queryResult._knnQueries;
	result._rangeMetrics = Experiments::summarizeQueryStats(rangeSamples);
	result._countRangeMetrics = Experiments::summarizeQueryStats(countRangeSamples);
	result._radiusMetrics = Experiments::summarizeQueryStats(radiusSamples);
	result._knnMetrics = Experiments::summarizeQueryStats(knnSamples);
	for (const auto& [name, stratum] : stratumSamples)
		result._stratumMetrics[name] = Experiments::summarizeQueryStats(stratum);
	result._stratumSummary = formatStratumSummary(result._stratumMetrics);

	if (repeats > 1)
	{
		std::vector<double> repeatMeanLatencies;
		repeatMeanLatencies.reserve(repeats);
		repeatMeanLatencies.push_back(result._metrics._averageLatencyMs);
		for (size_t r = 1; r < repeats; ++r)
		{
			const PointGpu::QueryResult rerun = index.query(prepared._cudaQueries, cudaOptions);
			repeatMeanLatencies.push_back(rerun._metrics._averageLatencyMs);
		}
		fillLatencyReliability(result._metrics, repeatMeanLatencies);
	}

	return result;
}

static SchemaConfig effectiveSchemaForEvaluation(
	const Experiments::SchemaCandidate& schema,
	const Experiments::SchemaSearchOptions& options)
{
	SchemaConfig config = schema._config;
	if (!useCudaEvaluator(options) && options._enableLeafMicroIndexes)
	{
		config._buildPolicy._enableLeafMicroIndexes = true;
		config._buildPolicy._leafMicroIndexThreshold = options._leafMicroIndexThreshold;
	}
	return config;
}

static Experiments::SchemaSearchRecord benchmarkSchemaCandidate(
	const SearchDataset& dataset,
	const Experiments::PointCloudFeatures& pointFeatures,
	const Experiments::WorkloadProfile& workload,
	const Experiments::WorkloadFeatures& workloadFeatures,
	const PreparedWorkload& preparedWorkload,
	const Experiments::SchemaCandidate& schema,
	const Experiments::SchemaSearchOptions& options,
	CudaIndexCacheEntry* cudaCacheEntry = nullptr)
{
	Experiments::BuildMetrics buildMetrics;
	WorkloadRun workloadRun;
	std::string backend = "cpu";
	int cudaDevice = -1;
	std::string cudaBuilder;
	double gpuUploadMs = 0.0;
	double gpuBuildMs = 0.0;
	size_t gpuMemoryBytes = 0;
	ActiveStructureStats activeStats;
	const SchemaConfig effectiveConfig = effectiveSchemaForEvaluation(schema, options);

	// CUDA can't honor adaptive leaf capacity or occupancy-entropy conditions, so fall back to the CPU evaluator; the record keeps backend = "cpu" to make the fallback visible.
	const bool cpuFallbackForGpuFeature = useCudaEvaluator(options) && schemaUsesGpuUnsupportedFeature(effectiveConfig);

	if (useCudaEvaluator(options) && !cpuFallbackForGpuFeature)
	{
		activeStats = staticSchemaStructureStats(effectiveConfig);
		backend = "cuda";
		const PointGpu::Options cudaOptions = cudaOptionsFrom(options);
		const std::string currentSignature = schemaSignature(effectiveConfig);
		auto applyBuildResult = [&](const PointGpu::BuildResult& build) {
			buildMetrics = build._metrics;
			cudaDevice = build._device;
			cudaBuilder = build._builder;
			gpuUploadMs = build._uploadTimeMs;
			gpuBuildMs = build._gpuBuildTimeMs;
			gpuMemoryBytes = build._gpuMemoryBytes;
			if (build._activeStructureTypes > 0)
			{
				activeStats._activeStructureTypes = build._activeStructureTypes;
				activeStats._nestedActiveFraction = build._nestedActiveFraction;
				activeStats._summary = build._activeStructureSummary;
			}
		};

		if (isBIHBuilder(cudaOptions._builder))
		{
			PointGpu::BIH localIndex;
			PointGpu::BIH* indexPtr = &localIndex;
			PointGpu::BuildResult build;
			if (cudaCacheEntry)
				build = cudaBuildOrReuse(cudaCacheEntry->_bih, currentSignature, dataset._cloud, effectiveConfig, cudaOptions, *cudaCacheEntry, indexPtr);
			else
				build = localIndex.build(dataset._cloud, effectiveConfig, cudaOptions);
			applyBuildResult(build);
			workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions, options._measurementRepeats);
		}
		else if (isHGridBuilder(cudaOptions._builder))
		{
			PointGpu::HGrid localIndex;
			PointGpu::HGrid* indexPtr = &localIndex;
			PointGpu::BuildResult build;
			if (cudaCacheEntry)
				build = cudaBuildOrReuse(cudaCacheEntry->_hgrid, currentSignature, dataset._cloud, effectiveConfig, cudaOptions, *cudaCacheEntry, indexPtr);
			else
				build = localIndex.build(dataset._cloud, effectiveConfig, cudaOptions);
			applyBuildResult(build);
			workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions, options._measurementRepeats);
		}
		else if (isKDTreeBuilder(cudaOptions._builder))
		{
			PointGpu::KDTree localIndex;
			PointGpu::KDTree* indexPtr = &localIndex;
			PointGpu::BuildResult build;
			if (cudaCacheEntry)
				build = cudaBuildOrReuse(cudaCacheEntry->_kdTree, currentSignature, dataset._cloud, effectiveConfig, cudaOptions, *cudaCacheEntry, indexPtr);
			else
				build = localIndex.build(dataset._cloud, effectiveConfig, cudaOptions);
			applyBuildResult(build);
			workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions, options._measurementRepeats);
		}
		else if (isOctreeBuilder(cudaOptions._builder))
		{
			PointGpu::Octree localIndex;
			PointGpu::Octree* indexPtr = &localIndex;
			PointGpu::BuildResult build;
			if (cudaCacheEntry)
				build = cudaBuildOrReuse(cudaCacheEntry->_octree, currentSignature, dataset._cloud, effectiveConfig, cudaOptions, *cudaCacheEntry, indexPtr);
			else
				build = localIndex.build(dataset._cloud, effectiveConfig, cudaOptions);
			applyBuildResult(build);
			workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions, options._measurementRepeats);
		}
		else if (isQuadTreeBuilder(cudaOptions._builder))
		{
			PointGpu::QuadTree localIndex;
			PointGpu::QuadTree* indexPtr = &localIndex;
			PointGpu::BuildResult build;
			if (cudaCacheEntry)
				build = cudaBuildOrReuse(cudaCacheEntry->_quadTree, currentSignature, dataset._cloud, effectiveConfig, cudaOptions, *cudaCacheEntry, indexPtr);
			else
				build = localIndex.build(dataset._cloud, effectiveConfig, cudaOptions);
			applyBuildResult(build);
			workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions, options._measurementRepeats);
		}
		else if (isRegularGridBuilder(cudaOptions._builder))
		{
			PointGpu::RegularGrid localIndex;
			PointGpu::RegularGrid* indexPtr = &localIndex;
			PointGpu::BuildResult build;
			if (cudaCacheEntry)
				build = cudaBuildOrReuse(cudaCacheEntry->_regularGrid, currentSignature, dataset._cloud, effectiveConfig, cudaOptions, *cudaCacheEntry, indexPtr);
			else
				build = localIndex.build(dataset._cloud, effectiveConfig, cudaOptions);
			applyBuildResult(build);
			workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions, options._measurementRepeats);
		}
		else if (isMixedBuilder(cudaOptions._builder))
		{
			PointGpu::MixedTree localIndex;
			PointGpu::MixedTree* indexPtr = &localIndex;
			PointGpu::BuildResult build;
			if (cudaCacheEntry)
				build = cudaBuildOrReuse(cudaCacheEntry->_mixedTree, currentSignature, dataset._cloud, effectiveConfig, cudaOptions, *cudaCacheEntry, indexPtr);
			else
				build = localIndex.build(dataset._cloud, effectiveConfig, cudaOptions);
			applyBuildResult(build);
			workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions, options._measurementRepeats);
		}
		else
		{
			PointGpu::LBVH localIndex;
			PointGpu::LBVH* indexPtr = &localIndex;
			PointGpu::BuildResult build;
			if (cudaCacheEntry)
				build = cudaBuildOrReuse(cudaCacheEntry->_lbvh, currentSignature, dataset._cloud, effectiveConfig, cudaOptions, *cudaCacheEntry, indexPtr);
			else
				build = localIndex.build(dataset._cloud, effectiveConfig, cudaOptions);
			applyBuildResult(build);
			workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions, options._measurementRepeats);
		}
	}
	else
	{
		PointSpatialIndex index;
		const auto buildBegin = std::chrono::steady_clock::now();
		index.build(dataset._cloud, effectiveConfig);
		const auto buildEnd = std::chrono::steady_clock::now();

		buildMetrics = Experiments::collectBuildMetrics(index.stats(), index.root(), elapsedMilliseconds(buildBegin, buildEnd), effectiveConfig);
		activeStats = collectActiveStructureStats(index.root(), effectiveConfig);
		workloadRun = runWorkloadProfile(preparedWorkload, workload._knnK, index, options._measurementRepeats);
	}

	appendSchemaQueryTrace(
		options._queryTracePath,
		dataset,
		schema,
		workload,
		preparedWorkload,
		workloadRun,
		backend,
		options._scoreStage.empty() ? std::string("final") : options._scoreStage);

	Experiments::SchemaSearchRecord record;
	record._datasetName = dataset._name;
	record._datasetSource = dataset._source;
	record._numPoints = dataset._cloud.size();
	record._workloadName = workload._name;
	record._rangeWeight = workload._rangeWeight;
	record._radiusWeight = workload._radiusWeight;
	record._knnWeight = workload._knnWeight;
	record._numQueries = workload._numQueries;
	record._knnK = workload._knnK;
	record._querySeed = workload._querySeed;
	record._schemaName = effectiveConfig._name;
	record._schemaPath = schema._path;
	record._buildMetrics = buildMetrics;
	record._queryMetrics = workloadRun._metrics;
	record._rangeMetrics = workloadRun._rangeMetrics;
	record._countRangeMetrics = workloadRun._countRangeMetrics;
	record._radiusMetrics = workloadRun._radiusMetrics;
	record._knnMetrics = workloadRun._knnMetrics;
	record._rangeQueries = workloadRun._rangeQueries;
	record._countRangeQueries = workloadRun._countRangeQueries;
	record._radiusQueries = workloadRun._radiusQueries;
	record._knnQueries = workloadRun._knnQueries;
	record._queryStrataSummary = workloadRun._stratumSummary;
	record._weights = effectiveScoreWeights(workload, options);
	record._scoreMode = scoreModeForWeights(record._weights);
	record._scoreStage = options._scoreStage.empty() ? std::string("final") : options._scoreStage;
	record._scoreIsFinalLatency = options._scoreIsFinalLatency && scoreWeightsAreDefault(record._weights);
	record._score = Experiments::computeSchemaSearchScore(
		record._buildMetrics,
		record._queryMetrics,
		record._weights,
		record._scoreMemoryMb,
		record._scoreImbalancePenalty);
	record._pointFeatures = pointFeatures;
	record._workloadFeatures = workloadFeatures;
	record._estimatedQueryCost = Experiments::estimateSchemaQueryCost(
		effectiveConfig, pointFeatures, workloadFeatures, record._weights._visitProxyAlpha);
	record._backend = backend;
	record._knnBackend = record._knnQueries == 0
		? "none"
		: (backend == "cuda" ? "bruteforce_gpu_scan" : "cpu_tree_knn");
	record._cudaDevice = cudaDevice;
	record._cudaBuilder = cudaBuilder;
	record._gpuUploadMs = gpuUploadMs;
	record._gpuBuildMs = gpuBuildMs;
	record._gpuQueryMs = workloadRun._gpuQueryMs;
	record._gpuMemoryBytes = gpuMemoryBytes;
	record._conditionalLevels = schemaConditionalLevels(effectiveConfig);
	record._conditionFields = schemaConditionFields(effectiveConfig);
	record._conditionSummary = schemaConditionSummary(effectiveConfig);
	record._isBaseline = schema._isBaseline;
	record._activeStructureTypes = activeStats._activeStructureTypes;
	record._nestedActiveFraction = activeStats._nestedActiveFraction;
	record._activeStructureSummary = activeStats._summary;
	return record;
}

static Experiments::SchemaSearchRecord benchmarkSchemaCandidateCached(
	const SearchDataset& dataset,
	const Experiments::PointCloudFeatures& pointFeatures,
	const Experiments::WorkloadProfile& workload,
	const Experiments::WorkloadFeatures& workloadFeatures,
	const PreparedWorkload& preparedWorkload,
	const Experiments::SchemaCandidate& schema,
	const Experiments::SchemaSearchOptions& options,
	CudaIndexCacheEntry* cudaCacheEntry = nullptr)
{
	Experiments::EvaluationCache* cache = options._scoreCache;
	if (cache && cache->enabled() && !options._rebuildScoreCache && options._queryTracePath.empty())
	{
		const std::string backendName = useCudaEvaluator(options) ? "cuda" : "cpu";
		const SchemaConfig effectiveConfig = effectiveSchemaForEvaluation(schema, options);
		std::string cudaBuilder;
		if (useCudaEvaluator(options))
			cudaBuilder = cudaOptionsFrom(options)._builder;

		const Experiments::EvaluationCacheKey key = Experiments::makeEvaluationCacheKey(
			schemaSignature(effectiveConfig),
			dataset._name,
			dataset._cloud.size(),
			dataset._cloud.bounds().min(),
			dataset._cloud.bounds().max(),
			workload,
			backendName,
			cudaBuilder,
			effectiveScoreWeights(workload, options));

		Experiments::SchemaSearchRecord cached;
		cached._datasetName = dataset._name;
		cached._datasetSource = dataset._source;
		cached._numPoints = dataset._cloud.size();
		cached._workloadName = workload._name;
		cached._rangeWeight = workload._rangeWeight;
		cached._radiusWeight = workload._radiusWeight;
		cached._knnWeight = workload._knnWeight;
		cached._numQueries = workload._numQueries;
		cached._knnK = workload._knnK;
		cached._querySeed = workload._querySeed;
		cached._schemaName = schema._config._name;
		cached._schemaPath = schema._path;
		cached._weights = effectiveScoreWeights(workload, options);
		cached._scoreMode = scoreModeForWeights(cached._weights);
		cached._scoreStage = options._scoreStage.empty() ? std::string("final") : options._scoreStage;
		cached._scoreIsFinalLatency = options._scoreIsFinalLatency && scoreWeightsAreDefault(cached._weights);
		cached._pointFeatures = pointFeatures;
		cached._workloadFeatures = workloadFeatures;
		cached._isBaseline = schema._isBaseline;

		if (cache->tryGet(key, cached))
		{
			cached._isBaseline = schema._isBaseline;
			cached._scoreMode = scoreModeForWeights(effectiveScoreWeights(workload, options));
			cached._scoreStage = options._scoreStage.empty() ? std::string("final") : options._scoreStage;
			cached._scoreIsFinalLatency = options._scoreIsFinalLatency && scoreWeightsAreDefault(effectiveScoreWeights(workload, options));
			if (options._deepNestedSearch && !useCudaEvaluator(options) && cached._activeStructureTypes == 0)
			{
				Experiments::SchemaSearchRecord record = benchmarkSchemaCandidate(
					dataset, pointFeatures, workload, workloadFeatures, preparedWorkload, schema, options, cudaCacheEntry);
				cache->put(key, record);
				return record;
			}
			return cached;
		}

		Experiments::SchemaSearchRecord record = benchmarkSchemaCandidate(
			dataset, pointFeatures, workload, workloadFeatures, preparedWorkload, schema, options, cudaCacheEntry);
		cache->put(key, record);
		return record;
	}

	return benchmarkSchemaCandidate(
		dataset, pointFeatures, workload, workloadFeatures, preparedWorkload, schema, options, cudaCacheEntry);
}

static EvaluatedCandidate evaluateCandidate(
	const Experiments::SchemaCandidate& candidate,
	const std::vector<DatasetContext>& datasets,
	const std::vector<Experiments::WorkloadProfile>& workloads,
	const Experiments::SchemaSearchOptions& options,
	CudaIndexCache* cudaCache = nullptr)
{
	EvaluatedCandidate evaluation;
	evaluation._candidate = candidate;
	evaluation._records.reserve(datasets.size() * workloads.size());

	double scoreSum = 0.0;
	size_t scoreCount = 0;
	for (const DatasetContext& datasetContext : datasets)
	{
		for (size_t workloadIndex = 0; workloadIndex < workloads.size(); ++workloadIndex)
		{
			const Experiments::WorkloadProfile& workload = workloads[workloadIndex];
			const Experiments::ScoreWeights weights = effectiveScoreWeights(workload, options);
			const Experiments::WorkloadFeatures workloadFeatures = Experiments::extractWorkloadFeatures(workload, weights);
			CudaIndexCacheEntry* cudaEntry = cudaCache && useCudaEvaluator(options)
				? &(*cudaCache)[datasetContext._dataset]
				: nullptr;

			// Wrap each per-(dataset, workload) evaluation so a single failed candidate gets a sentinel infinite-score record instead of killing the whole optimizer run.
			try
			{
				Experiments::SchemaSearchRecord record = benchmarkSchemaCandidateCached(
					*datasetContext._dataset,
					datasetContext._features,
					workload,
					workloadFeatures,
					datasetContext._preparedWorkloads[workloadIndex],
					candidate,
					options,
					cudaEntry);
				scoreSum += record._score;
				++scoreCount;
				evaluation._records.push_back(std::move(record));
			}
			catch (const std::exception& exception)
			{
				std::cerr << "    candidate '" << candidate._config._name
					<< "' failed on dataset '" << datasetContext._dataset->_name
					<< "' / workload '" << workload._name << "': " << exception.what() << '\n';
				Experiments::SchemaSearchRecord failed;
				failed._datasetName = datasetContext._dataset->_name;
				failed._datasetSource = datasetContext._dataset->_source;
				failed._numPoints = datasetContext._dataset->_cloud.size();
				failed._workloadName = workload._name;
				failed._rangeWeight = workload._rangeWeight;
				failed._radiusWeight = workload._radiusWeight;
				failed._knnWeight = workload._knnWeight;
				failed._numQueries = workload._numQueries;
				failed._knnK = workload._knnK;
				failed._querySeed = workload._querySeed;
				failed._schemaName = candidate._config._name;
				failed._schemaPath = candidate._path;
				failed._weights = weights;
				failed._scoreMode = scoreModeForWeights(failed._weights);
				failed._scoreStage = options._scoreStage.empty() ? std::string("final") : options._scoreStage;
				failed._scoreIsFinalLatency = options._scoreIsFinalLatency && scoreWeightsAreDefault(failed._weights);
				failed._pointFeatures = datasetContext._features;
				failed._workloadFeatures = workloadFeatures;
				failed._backend = useCudaEvaluator(options) ? "cuda_failed" : "cpu_failed";
				failed._knnBackend = workload._knnWeight > 0.0
					? (useCudaEvaluator(options) ? "bruteforce_gpu_scan" : "cpu_tree_knn")
					: "none";
				failed._score = std::numeric_limits<double>::infinity();
				failed._isBaseline = candidate._isBaseline;
				evaluation._records.push_back(std::move(failed));
			}
		}
	}

	evaluation.aggregateScore = scoreCount > 0
		? scoreSum / static_cast<double>(scoreCount)
		: std::numeric_limits<double>::infinity();
	return evaluation;
}

static std::string csvEscape(const std::string& value)
{
	if (value.find_first_of(",\"\n\r") == std::string::npos)
		return value;

	std::string escaped = "\"";
	for (const char c : value)
	{
		if (c == '"')
			escaped += "\"\"";
		else
			escaped += c;
	}
	escaped += '"';
	return escaped;
}

static void createParentDirectory(const std::string& filename)
{
	const std::filesystem::path path(filename);
	if (path.has_parent_path())
		std::filesystem::create_directories(path.parent_path());
}

static void writeSearchHeader(std::ostream& output)
{
	output
		<< "dataset_name,dataset_source,num_points,workload_name,range_weight,radius_weight,knn_weight,num_queries,knn_k,query_seed,"
		<< "range_scale_min,range_scale_max,radius_scale_min,radius_scale_max,"
		<< "feature_sample_size,bbox_x,bbox_y,bbox_z,aspect_xy,aspect_xz,aspect_yz,density_bbox,height_mean,height_std,height_range,"
		<< "cov_eig_0,cov_eig_1,cov_eig_2,linearity,planarity,scattering,occupancy_ratio_8,occupancy_entropy_8,density_cv_8,verticality_score,flatness_score,"
		<< "w_range,w_radius,w_knn,query_scale_mean,query_scale_std,build_weight,memory_weight,"
		<< "schema_name,schema_path,build_time_ms,num_nodes,num_leaves,max_depth,avg_leaf_occupancy,max_leaf_occupancy,"
		<< "leaf_occupancy_p50,leaf_occupancy_p90,leaf_occupancy_p99,avg_depth,avg_fanout,max_fanout,empty_child_ratio,single_child_nodes,"
		<< "mean_tight_bounds_volume_ratio,micro_indexed_leaves,micro_indexed_points,node_fanout_summary,memory_estimate_bytes,"
		<< "total_queries,avg_latency_ms,median_latency_ms,p95_latency_ms,throughput_qps,avg_visited_nodes,avg_tested_points,avg_returned_points,"
		<< "range_queries,radius_queries,knn_queries,query_strata_summary,score,score_memory_mb,score_imbalance_penalty,lambda_latency,lambda_build,lambda_memory,lambda_imbalance,"
		<< "backend,knn_backend,cuda_device,cuda_builder,gpu_upload_ms,gpu_build_ms,gpu_query_ms,gpu_memory_bytes,count_range_queries,"
		<< "conditional_levels,condition_fields,condition_summary,is_baseline,active_structure_types,nested_active_fraction,active_structure_summary,"
		<< "best_baseline_schema,best_baseline_score,relative_speedup_vs_baseline,"
		<< "confirm_seeds_used,latency_mean_ms,latency_ci_low_ms,latency_ci_high_ms,"
		<< "p95_latency_mean_ms,p95_latency_ci_low_ms,p95_latency_ci_high_ms,"
		<< "gpu_build_mean_ms,gpu_build_ci_low_ms,gpu_build_ci_high_ms,"
		<< "score_mode,score_stage,score_is_final_latency,effective_queries,score_uses_visit_proxy,visit_proxy_alpha,"
		<< "measurement_repeats,latency_stddev_ms,latency_cv,latency_repeat_ci_low_ms,latency_repeat_ci_high_ms,ranking_confident,"
		<< "estimated_query_cost\n";
}

static void writeSearchRows(const std::string& csvPath, const std::vector<Experiments::SchemaSearchRecord>& records)
{
	if (csvPath.empty())
		return;

	createParentDirectory(csvPath);
	std::ofstream output(csvPath);
	if (!output.is_open())
		throw std::runtime_error("Unable to open schema-search CSV path: " + csvPath);

	writeSearchHeader(output);
	output << std::fixed << std::setprecision(6);
	for (const Experiments::SchemaSearchRecord& record : records)
	{
		output
			<< csvEscape(record._datasetName) << ','
			<< csvEscape(record._datasetSource) << ','
			<< record._numPoints << ','
			<< csvEscape(record._workloadName) << ','
			<< record._rangeWeight << ','
			<< record._radiusWeight << ','
			<< record._knnWeight << ','
			<< record._numQueries << ','
			<< record._knnK << ','
			<< record._querySeed << ','
			<< record._workloadFeatures._rangeScaleMin << ','
			<< record._workloadFeatures._rangeScaleMax << ','
			<< record._workloadFeatures._radiusScaleMin << ','
			<< record._workloadFeatures._radiusScaleMax << ','
			<< record._pointFeatures._sampleSize << ','
			<< record._pointFeatures._bboxX << ','
			<< record._pointFeatures._bboxY << ','
			<< record._pointFeatures._bboxZ << ','
			<< record._pointFeatures._aspectXY << ','
			<< record._pointFeatures._aspectXZ << ','
			<< record._pointFeatures._aspectYZ << ','
			<< record._pointFeatures._densityBbox << ','
			<< record._pointFeatures._heightMean << ','
			<< record._pointFeatures._heightStd << ','
			<< record._pointFeatures._heightRange << ','
			<< record._pointFeatures._covEig0 << ','
			<< record._pointFeatures._covEig1 << ','
			<< record._pointFeatures._covEig2 << ','
			<< record._pointFeatures._linearity << ','
			<< record._pointFeatures._planarity << ','
			<< record._pointFeatures._scattering << ','
			<< record._pointFeatures._occupancyRatio8 << ','
			<< record._pointFeatures._occupancyEntropy8 << ','
			<< record._pointFeatures._densityCv8 << ','
			<< record._pointFeatures._verticalityScore << ','
			<< record._pointFeatures._flatnessScore << ','
			<< record._workloadFeatures._wRange << ','
			<< record._workloadFeatures._wRadius << ','
			<< record._workloadFeatures._wKnn << ','
			<< record._workloadFeatures._queryScaleMean << ','
			<< record._workloadFeatures._queryScaleStd << ','
			<< record._workloadFeatures._buildWeight << ','
			<< record._workloadFeatures._memoryWeight << ','
			<< csvEscape(record._schemaName) << ','
			<< csvEscape(record._schemaPath) << ','
			<< record._buildMetrics._buildTimeMs << ','
			<< record._buildMetrics._numNodes << ','
			<< record._buildMetrics._numLeaves << ','
			<< record._buildMetrics._maxDepth << ','
			<< record._buildMetrics._averageLeafOccupancy << ','
			<< record._buildMetrics._maxLeafOccupancy << ','
			<< record._buildMetrics._leafOccupancyP50 << ','
			<< record._buildMetrics._leafOccupancyP90 << ','
			<< record._buildMetrics._leafOccupancyP99 << ','
			<< record._buildMetrics._averageDepth << ','
			<< record._buildMetrics._averageFanout << ','
			<< record._buildMetrics._maxFanout << ','
			<< record._buildMetrics._emptyChildRatio << ','
			<< record._buildMetrics._singleChildNodeCount << ','
			<< record._buildMetrics._meanTightBoundsVolumeRatio << ','
			<< record._buildMetrics._microIndexedLeaves << ','
			<< record._buildMetrics._microIndexedPoints << ','
			<< csvEscape(record._buildMetrics._nodeFanoutSummary) << ','
			<< record._buildMetrics._memoryEstimateBytes << ','
			<< record._queryMetrics._totalQueries << ','
			<< record._queryMetrics._averageLatencyMs << ','
			<< record._queryMetrics._medianLatencyMs << ','
			<< record._queryMetrics._p95LatencyMs << ','
			<< record._queryMetrics._throughputQueriesPerSecond << ','
			<< record._queryMetrics._averageVisitedNodes << ','
			<< record._queryMetrics._averageTestedPoints << ','
			<< record._queryMetrics._averageReturnedPoints << ','
			<< record._rangeQueries << ','
			<< record._radiusQueries << ','
			<< record._knnQueries << ','
			<< csvEscape(record._queryStrataSummary) << ','
			<< record._score << ','
			<< record._scoreMemoryMb << ','
			<< record._scoreImbalancePenalty << ','
			<< record._weights._lambdaLatency << ','
			<< record._weights._lambdaBuild << ','
			<< record._weights._lambdaMemory << ','
			<< record._weights._lambdaImbalance << ','
			<< csvEscape(record._backend) << ','
			<< csvEscape(record._knnBackend) << ','
			<< record._cudaDevice << ','
			<< csvEscape(record._cudaBuilder) << ','
			<< record._gpuUploadMs << ','
			<< record._gpuBuildMs << ','
			<< record._gpuQueryMs << ','
			<< record._gpuMemoryBytes << ','
			<< record._countRangeQueries << ','
			<< record._conditionalLevels << ','
			<< record._conditionFields << ','
			<< csvEscape(record._conditionSummary) << ','
			<< (record._isBaseline ? 1 : 0) << ','
			<< record._activeStructureTypes << ','
			<< record._nestedActiveFraction << ','
			<< csvEscape(record._activeStructureSummary) << ','
			<< csvEscape(record._bestBaselineSchema) << ','
			<< record._bestBaselineScore << ','
			<< record._relativeSpeedupVsBaseline << ','
			<< record._confirmSeedsUsed << ','
			<< record._latencyMean << ','
			<< record._latencyCiLow << ','
			<< record._latencyCiHigh << ','
			<< record._p95LatencyMean << ','
			<< record._p95LatencyCiLow << ','
			<< record._p95LatencyCiHigh << ','
			<< record._gpuBuildMean << ','
			<< record._gpuBuildCiLow << ','
			<< record._gpuBuildCiHigh << ','
			<< csvEscape(record._scoreMode) << ','
			<< csvEscape(record._scoreStage) << ','
			<< (record._scoreIsFinalLatency ? 1 : 0) << ','
			<< record._queryMetrics._totalQueries << ','
			<< (record._weights._useVisitProxy ? 1 : 0) << ','
			<< record._weights._visitProxyAlpha << ','
			<< record._queryMetrics._measurementRepeats << ','
			<< record._queryMetrics._latencyStdDevMs << ','
			<< record._queryMetrics._latencyCoeffVar << ','
			<< record._queryMetrics._latencyCiLowMs << ','
			<< record._queryMetrics._latencyCiHighMs << ','
			<< (record._rankingConfident ? 1 : 0) << ','
			<< record._estimatedQueryCost << '\n';
	}
}

static size_t countCandidatesForBest(const std::vector<Experiments::SchemaSearchRecord>& records, const Experiments::SchemaSearchRecord& best)
{
	size_t count = 0;
	for (const Experiments::SchemaSearchRecord& record : records)
	{
		if (record._datasetName == best._datasetName && record._workloadName == best._workloadName)
			++count;
	}
	return count;
}

static void writeBestRows(const std::string& csvPath, const std::vector<Experiments::SchemaSearchRecord>& records)
{
	if (csvPath.empty())
		return;

	const std::vector<Experiments::SchemaSearchRecord> bestRecords = Experiments::selectBestRecords(records);
	createParentDirectory(csvPath);
	std::ofstream output(csvPath);
	if (!output.is_open())
		throw std::runtime_error("Unable to open schema-search best CSV path: " + csvPath);

	output
		<< "dataset_name,workload_name,num_points,feature_sample_size,bbox_x,bbox_y,bbox_z,aspect_xy,aspect_xz,aspect_yz,density_bbox,"
		<< "height_mean,height_std,height_range,cov_eig_0,cov_eig_1,cov_eig_2,linearity,planarity,scattering,occupancy_ratio_8,"
		<< "occupancy_entropy_8,density_cv_8,verticality_score,flatness_score,w_range,w_radius,w_knn,knn_k,num_queries,query_strata_summary,"
		<< "range_scale_min,range_scale_max,radius_scale_min,radius_scale_max,query_scale_mean,query_scale_std,build_weight,memory_weight,best_schema_name,best_schema_path,best_score,"
		<< "best_avg_latency_ms,best_build_time_ms,best_memory_estimate_bytes,leaf_occupancy_p50,leaf_occupancy_p90,leaf_occupancy_p99,"
		<< "avg_depth,avg_fanout,max_fanout,empty_child_ratio,single_child_nodes,mean_tight_bounds_volume_ratio,"
		<< "micro_indexed_leaves,micro_indexed_points,node_fanout_summary,num_candidates,backend,knn_backend,cuda_device,cuda_builder,gpu_upload_ms,gpu_build_ms,gpu_query_ms,gpu_memory_bytes,"
		<< "conditional_levels,condition_fields,condition_summary,is_baseline,active_structure_types,nested_active_fraction,active_structure_summary,"
		<< "best_baseline_schema,best_baseline_score,relative_speedup_vs_baseline,"
		<< "confirm_seeds_used,latency_mean_ms,latency_ci_low_ms,latency_ci_high_ms,"
		<< "p95_latency_mean_ms,p95_latency_ci_low_ms,p95_latency_ci_high_ms,"
		<< "gpu_build_mean_ms,gpu_build_ci_low_ms,gpu_build_ci_high_ms,"
		<< "lambda_latency,lambda_build,lambda_memory,lambda_imbalance,"
		<< "score_mode,score_stage,score_is_final_latency,effective_queries,score_uses_visit_proxy,visit_proxy_alpha,"
		<< "measurement_repeats,latency_stddev_ms,latency_cv,latency_repeat_ci_low_ms,latency_repeat_ci_high_ms,ranking_confident,"
		<< "estimated_query_cost\n";
	output << std::fixed << std::setprecision(6);
	for (const Experiments::SchemaSearchRecord& record : bestRecords)
	{
		output
			<< csvEscape(record._datasetName) << ','
			<< csvEscape(record._workloadName) << ','
			<< record._numPoints << ','
			<< record._pointFeatures._sampleSize << ','
			<< record._pointFeatures._bboxX << ','
			<< record._pointFeatures._bboxY << ','
			<< record._pointFeatures._bboxZ << ','
			<< record._pointFeatures._aspectXY << ','
			<< record._pointFeatures._aspectXZ << ','
			<< record._pointFeatures._aspectYZ << ','
			<< record._pointFeatures._densityBbox << ','
			<< record._pointFeatures._heightMean << ','
			<< record._pointFeatures._heightStd << ','
			<< record._pointFeatures._heightRange << ','
			<< record._pointFeatures._covEig0 << ','
			<< record._pointFeatures._covEig1 << ','
			<< record._pointFeatures._covEig2 << ','
			<< record._pointFeatures._linearity << ','
			<< record._pointFeatures._planarity << ','
			<< record._pointFeatures._scattering << ','
			<< record._pointFeatures._occupancyRatio8 << ','
			<< record._pointFeatures._occupancyEntropy8 << ','
			<< record._pointFeatures._densityCv8 << ','
			<< record._pointFeatures._verticalityScore << ','
			<< record._pointFeatures._flatnessScore << ','
			<< record._workloadFeatures._wRange << ','
			<< record._workloadFeatures._wRadius << ','
			<< record._workloadFeatures._wKnn << ','
			<< record._workloadFeatures._knnK << ','
			<< record._workloadFeatures._numQueries << ','
			<< csvEscape(record._queryStrataSummary) << ','
			<< record._workloadFeatures._rangeScaleMin << ','
			<< record._workloadFeatures._rangeScaleMax << ','
			<< record._workloadFeatures._radiusScaleMin << ','
			<< record._workloadFeatures._radiusScaleMax << ','
			<< record._workloadFeatures._queryScaleMean << ','
			<< record._workloadFeatures._queryScaleStd << ','
			<< record._workloadFeatures._buildWeight << ','
			<< record._workloadFeatures._memoryWeight << ','
			<< csvEscape(record._schemaName) << ','
			<< csvEscape(record._schemaPath) << ','
			<< record._score << ','
			<< record._queryMetrics._averageLatencyMs << ','
			<< record._buildMetrics._buildTimeMs << ','
			<< record._buildMetrics._memoryEstimateBytes << ','
			<< record._buildMetrics._leafOccupancyP50 << ','
			<< record._buildMetrics._leafOccupancyP90 << ','
			<< record._buildMetrics._leafOccupancyP99 << ','
			<< record._buildMetrics._averageDepth << ','
			<< record._buildMetrics._averageFanout << ','
			<< record._buildMetrics._maxFanout << ','
			<< record._buildMetrics._emptyChildRatio << ','
			<< record._buildMetrics._singleChildNodeCount << ','
			<< record._buildMetrics._meanTightBoundsVolumeRatio << ','
			<< record._buildMetrics._microIndexedLeaves << ','
			<< record._buildMetrics._microIndexedPoints << ','
			<< csvEscape(record._buildMetrics._nodeFanoutSummary) << ','
			<< countCandidatesForBest(records, record) << ','
			<< csvEscape(record._backend) << ','
			<< csvEscape(record._knnBackend) << ','
			<< record._cudaDevice << ','
			<< csvEscape(record._cudaBuilder) << ','
			<< record._gpuUploadMs << ','
			<< record._gpuBuildMs << ','
			<< record._gpuQueryMs << ','
			<< record._gpuMemoryBytes << ','
			<< record._conditionalLevels << ','
			<< record._conditionFields << ','
			<< csvEscape(record._conditionSummary) << ','
			<< (record._isBaseline ? 1 : 0) << ','
			<< record._activeStructureTypes << ','
			<< record._nestedActiveFraction << ','
			<< csvEscape(record._activeStructureSummary) << ','
			<< csvEscape(record._bestBaselineSchema) << ','
			<< record._bestBaselineScore << ','
			<< record._relativeSpeedupVsBaseline << ','
			<< record._confirmSeedsUsed << ','
			<< record._latencyMean << ','
			<< record._latencyCiLow << ','
			<< record._latencyCiHigh << ','
			<< record._p95LatencyMean << ','
			<< record._p95LatencyCiLow << ','
			<< record._p95LatencyCiHigh << ','
			<< record._gpuBuildMean << ','
			<< record._gpuBuildCiLow << ','
			<< record._gpuBuildCiHigh << ','
			<< record._weights._lambdaLatency << ','
			<< record._weights._lambdaBuild << ','
			<< record._weights._lambdaMemory << ','
			<< record._weights._lambdaImbalance << ','
			<< csvEscape(record._scoreMode) << ','
			<< csvEscape(record._scoreStage) << ','
			<< (record._scoreIsFinalLatency ? 1 : 0) << ','
			<< record._queryMetrics._totalQueries << ','
			<< (record._weights._useVisitProxy ? 1 : 0) << ','
			<< record._weights._visitProxyAlpha << ','
			<< record._queryMetrics._measurementRepeats << ','
			<< record._queryMetrics._latencyStdDevMs << ','
			<< record._queryMetrics._latencyCoeffVar << ','
			<< record._queryMetrics._latencyCiLowMs << ','
			<< record._queryMetrics._latencyCiHighMs << ','
			<< (record._rankingConfident ? 1 : 0) << ','
			<< record._estimatedQueryCost << '\n';
	}
}

static void writeParetoRows(const std::string& csvPath, const std::vector<Experiments::SchemaSearchRecord>& records)
{
	if (csvPath.empty())
		return;

	const std::vector<Experiments::SchemaSearchRecord> front = Experiments::selectParetoRecords(records);
	createParentDirectory(csvPath);
	std::ofstream output(csvPath);
	if (!output.is_open())
		throw std::runtime_error("Unable to open schema-search Pareto CSV path: " + csvPath);

	// Compact column set: the four Pareto objectives plus enough identity to replay the candidate, meant for plotting the front (the full layout stays in the best CSV).
	output
		<< "dataset_name,workload_name,query_strata_summary,pareto_rank,is_knee,schema_name,schema_path,score,"
		<< "avg_latency_ms,build_time_ms,memory_mb,memory_estimate_bytes,imbalance_penalty,"
		<< "leaf_occupancy_p90,empty_child_ratio,mean_tight_bounds_volume_ratio,micro_indexed_leaves,micro_indexed_points,"
		<< "p95_latency_ms,throughput_qps,backend,knn_backend,cuda_builder,is_baseline,"
		<< "conditional_levels,condition_fields,active_structure_types,nested_active_fraction,"
		<< "confirm_seeds_used,latency_mean_ms,latency_ci_low_ms,latency_ci_high_ms,"
		<< "p95_latency_mean_ms,p95_latency_ci_low_ms,p95_latency_ci_high_ms,"
		<< "gpu_build_mean_ms,gpu_build_ci_low_ms,gpu_build_ci_high_ms,"
		<< "lambda_latency,lambda_build,lambda_memory,lambda_imbalance,"
		<< "score_mode,score_stage,score_is_final_latency,effective_queries,score_uses_visit_proxy,visit_proxy_alpha,"
		<< "measurement_repeats,latency_stddev_ms,latency_cv,latency_repeat_ci_low_ms,latency_repeat_ci_high_ms,ranking_confident,"
		<< "estimated_query_cost\n";
	output << std::fixed << std::setprecision(6);
	for (const Experiments::SchemaSearchRecord& record : front)
	{
		const double memoryMb = (record._backend == "cuda" && record._gpuMemoryBytes > 0)
			? static_cast<double>(record._gpuMemoryBytes) / (1024.0 * 1024.0)
			: static_cast<double>(record._buildMetrics._memoryEstimateBytes) / (1024.0 * 1024.0);
		const double imbalancePenalty = record._buildMetrics._averageLeafOccupancy > 0.0
			? static_cast<double>(record._buildMetrics._maxLeafOccupancy) / record._buildMetrics._averageLeafOccupancy
			: 0.0;
		output
			<< csvEscape(record._datasetName) << ','
			<< csvEscape(record._workloadName) << ','
			<< csvEscape(record._queryStrataSummary) << ','
			<< record._paretoRank << ','
			<< (record._paretoKnee ? 1 : 0) << ','
			<< csvEscape(record._schemaName) << ','
			<< csvEscape(record._schemaPath) << ','
			<< record._score << ','
			<< record._queryMetrics._averageLatencyMs << ','
			<< record._buildMetrics._buildTimeMs << ','
			<< memoryMb << ','
			<< record._buildMetrics._memoryEstimateBytes << ','
			<< imbalancePenalty << ','
			<< record._buildMetrics._leafOccupancyP90 << ','
			<< record._buildMetrics._emptyChildRatio << ','
			<< record._buildMetrics._meanTightBoundsVolumeRatio << ','
			<< record._buildMetrics._microIndexedLeaves << ','
			<< record._buildMetrics._microIndexedPoints << ','
			<< record._queryMetrics._p95LatencyMs << ','
			<< record._queryMetrics._throughputQueriesPerSecond << ','
			<< csvEscape(record._backend) << ','
			<< csvEscape(record._knnBackend) << ','
			<< csvEscape(record._cudaBuilder) << ','
			<< (record._isBaseline ? 1 : 0) << ','
			<< record._conditionalLevels << ','
			<< record._conditionFields << ','
			<< record._activeStructureTypes << ','
			<< record._nestedActiveFraction << ','
			<< record._confirmSeedsUsed << ','
			<< record._latencyMean << ','
			<< record._latencyCiLow << ','
			<< record._latencyCiHigh << ','
			<< record._p95LatencyMean << ','
			<< record._p95LatencyCiLow << ','
			<< record._p95LatencyCiHigh << ','
			<< record._gpuBuildMean << ','
			<< record._gpuBuildCiLow << ','
			<< record._gpuBuildCiHigh << ','
			<< record._weights._lambdaLatency << ','
			<< record._weights._lambdaBuild << ','
			<< record._weights._lambdaMemory << ','
			<< record._weights._lambdaImbalance << ','
			<< csvEscape(record._scoreMode) << ','
			<< csvEscape(record._scoreStage) << ','
			<< (record._scoreIsFinalLatency ? 1 : 0) << ','
			<< record._queryMetrics._totalQueries << ','
			<< (record._weights._useVisitProxy ? 1 : 0) << ','
			<< record._weights._visitProxyAlpha << ','
			<< record._queryMetrics._measurementRepeats << ','
			<< record._queryMetrics._latencyStdDevMs << ','
			<< record._queryMetrics._latencyCoeffVar << ','
			<< record._queryMetrics._latencyCiLowMs << ','
			<< record._queryMetrics._latencyCiHighMs << ','
			<< (record._rankingConfident ? 1 : 0) << ','
			<< record._estimatedQueryCost << '\n';
	}
}

static std::string recordGroupKey(const Experiments::SchemaSearchRecord& record)
{
	return record._datasetName + "\n" + record._workloadName;
}

static double reportLatencyMs(const Experiments::SchemaSearchRecord& record)
{
	return record._confirmSeedsUsed > 0 && record._latencyMean > 0.0
		? record._latencyMean
		: record._queryMetrics._averageLatencyMs;
}

static double reportBuildMs(const Experiments::SchemaSearchRecord& record)
{
	return record._confirmSeedsUsed > 0 && record._gpuBuildMean > 0.0
		? record._gpuBuildMean
		: record._buildMetrics._buildTimeMs;
}

static double reportMemoryMb(const Experiments::SchemaSearchRecord& record)
{
	return static_cast<double>(record._buildMetrics._memoryEstimateBytes) / (1024.0 * 1024.0);
}

static const Experiments::SchemaSearchRecord* bestRecordBy(
	const std::vector<const Experiments::SchemaSearchRecord*>& group,
	const std::function<double(const Experiments::SchemaSearchRecord&)>& metric)
{
	const Experiments::SchemaSearchRecord* best = nullptr;
	double bestValue = std::numeric_limits<double>::infinity();
	for (const Experiments::SchemaSearchRecord* record : group)
	{
		const double value = metric(*record);
		if (std::isfinite(value) && (!best || value < bestValue))
		{
			best = record;
			bestValue = value;
		}
	}
	return best;
}

static void printMetricWinner(
	const char* label,
	const Experiments::SchemaSearchRecord* record,
	const std::function<double(const Experiments::SchemaSearchRecord&)>& metric,
	const char* unit)
{
	if (!record)
	{
		std::cout << "    " << label << ": n/a\n";
		return;
	}

	std::cout << "    " << label << ": " << record->_schemaName
		<< " (" << metric(*record) << ' ' << unit
		<< ", score " << record->_score
		<< ", " << record->_scoreMode << ")\n";
}

static void reportMetricSummaries(
	const std::vector<Experiments::SchemaSearchRecord>& records,
	const Experiments::SchemaSearchOptions& options)
{
	if (records.empty())
		return;

	std::map<std::string, std::vector<const Experiments::SchemaSearchRecord*>> groups;
	for (const Experiments::SchemaSearchRecord& record : records)
		groups[recordGroupKey(record)].push_back(&record);

	std::cout << "  metric winners:\n";
	const double memoryBudgetMb = options._cuda._memoryBudgetMb > 0
		? static_cast<double>(options._cuda._memoryBudgetMb)
		: 0.0;
	for (const auto& [key, group] : groups)
	{
		const size_t separator = key.find('\n');
		const std::string dataset = separator == std::string::npos ? key : key.substr(0, separator);
		const std::string workload = separator == std::string::npos ? std::string() : key.substr(separator + 1);
		std::cout << "  [" << dataset << " / " << workload << "]\n";

		printMetricWinner("best latency", bestRecordBy(group, reportLatencyMs), reportLatencyMs, "ms");
		printMetricWinner("best build", bestRecordBy(group, reportBuildMs), reportBuildMs, "ms");
		printMetricWinner("best memory", bestRecordBy(group, reportMemoryMb), reportMemoryMb, "MB");
		if (memoryBudgetMb > 0.0)
		{
			std::vector<const Experiments::SchemaSearchRecord*> underBudget;
			for (const Experiments::SchemaSearchRecord* record : group)
			{
				if (reportMemoryMb(*record) <= memoryBudgetMb)
					underBudget.push_back(record);
			}
			printMetricWinner("best latency under memory budget", bestRecordBy(underBudget, reportLatencyMs), reportLatencyMs, "ms");
		}
	}
}

static bool recordCountsAsNested(const Experiments::SchemaSearchRecord& record);

static void annotateBaselineComparisons(std::vector<Experiments::SchemaSearchRecord>& records)
{
	std::unordered_map<std::string, const Experiments::SchemaSearchRecord*> bestBaselines;
	for (const Experiments::SchemaSearchRecord& record : records)
	{
		if (!record._isBaseline)
			continue;
		const std::string key = recordGroupKey(record);
		const auto existing = bestBaselines.find(key);
		if (existing == bestBaselines.end() || record._score < existing->second->_score)
			bestBaselines[key] = &record;
	}

	for (Experiments::SchemaSearchRecord& record : records)
	{
		const auto baseline = bestBaselines.find(recordGroupKey(record));
		if (baseline == bestBaselines.end())
			continue;

		const Experiments::SchemaSearchRecord& best = *baseline->second;
		record._bestBaselineSchema = best._schemaName;
		record._bestBaselineScore = best._score;
		record._relativeSpeedupVsBaseline = best._queryMetrics._averageLatencyMs > 0.0
			? (best._queryMetrics._averageLatencyMs - record._queryMetrics._averageLatencyMs) / best._queryMetrics._averageLatencyMs
			: 0.0;
	}
}

static void reportDeepNestedOutcome(const std::vector<Experiments::SchemaSearchRecord>& records)
{
	std::map<std::string, const Experiments::SchemaSearchRecord*> bestBaselines;
	std::map<std::string, const Experiments::SchemaSearchRecord*> bestNested;
	for (const Experiments::SchemaSearchRecord& record : records)
	{
		const std::string key = recordGroupKey(record);
		if (record._isBaseline)
		{
			const auto existing = bestBaselines.find(key);
			if (existing == bestBaselines.end() || record._queryMetrics._averageLatencyMs < existing->second->_queryMetrics._averageLatencyMs)
				bestBaselines[key] = &record;
		}
		else if (recordCountsAsNested(record))
		{
			const auto existing = bestNested.find(key);
			if (existing == bestNested.end() || record._queryMetrics._averageLatencyMs < existing->second->_queryMetrics._averageLatencyMs)
				bestNested[key] = &record;
		}
	}

	for (const auto& [key, baseline] : bestBaselines)
	{
		const auto nested = bestNested.find(key);
		if (nested == bestNested.end())
		{
			std::cout << "  deep nested outcome: " << baseline->_datasetName << " / " << baseline->_workloadName
				<< " has no runtime-nested finalist (>=2 active types and >=5% nested fraction)\n";
			continue;
		}

		const Experiments::SchemaSearchRecord& nestedRecord = *nested->second;
		const double meanSpeedup = baseline->_queryMetrics._averageLatencyMs > 0.0
			? (baseline->_queryMetrics._averageLatencyMs - nestedRecord._queryMetrics._averageLatencyMs) / baseline->_queryMetrics._averageLatencyMs
			: 0.0;
		const bool p95NoWorse = nestedRecord._queryMetrics._p95LatencyMs <= baseline->_queryMetrics._p95LatencyMs;
		std::cout << "  deep nested outcome: " << baseline->_datasetName << " / " << baseline->_workloadName
			<< " best nested " << nestedRecord._schemaName
			<< " vs baseline " << baseline->_schemaName
			<< ", mean speedup " << (meanSpeedup * 100.0) << "%"
			<< ", p95 " << (p95NoWorse ? "no worse" : "worse")
			<< ", nested active fraction " << nestedRecord._nestedActiveFraction;
		if (meanSpeedup >= 0.05 && p95NoWorse)
			std::cout << " (target met)";
		std::cout << '\n';
	}
}

struct ActiveExplainEntry
{
	std::string _typeName;
	size_t _nodes = 0;
	size_t _leafPoints = 0;
	bool _schemaOnly = false;
};

static size_t parseSizeField(const std::string& value, const std::string& key)
{
	const size_t begin = value.find(key);
	if (begin == std::string::npos)
		return 0;
	const size_t numberBegin = begin + key.size();
	size_t numberEnd = numberBegin;
	while (numberEnd < value.size() && std::isdigit(static_cast<unsigned char>(value[numberEnd])))
		++numberEnd;
	if (numberEnd == numberBegin)
		return 0;
	try
	{
		return static_cast<size_t>(std::stoull(value.substr(numberBegin, numberEnd - numberBegin)));
	}
	catch (...)
	{
		return 0;
	}
}

static std::string displayTypeName(const std::string& shortName)
{
	if (shortName == "qt") return "QuadTree";
	if (shortName == "ot") return "Octree";
	if (shortName == "kd") return "KDTree";
	if (shortName == "bvh") return "BVH";
	if (shortName == "lbvh") return "LBVH";
	if (shortName == "bih") return "BIH";
	if (shortName == "kot") return "KarrasOctree";
	if (shortName == "rg") return "RegularGrid";
	if (shortName == "hg") return "HGrid";
	return shortName.empty() ? "Unknown" : shortName;
}

static std::vector<ActiveExplainEntry> parseActiveStructureSummary(const std::string& summary)
{
	std::vector<ActiveExplainEntry> entries;
	size_t start = 0;
	while (start < summary.size())
	{
		const size_t end = summary.find(';', start);
		const std::string token = summary.substr(start, end == std::string::npos ? std::string::npos : end - start);
		const size_t separator = token.find(':');
		if (separator != std::string::npos)
		{
			ActiveExplainEntry entry;
			entry._typeName = token.substr(0, separator);
			const std::string fields = token.substr(separator + 1);
			entry._schemaOnly = fields.find("schema") != std::string::npos;
			entry._nodes = parseSizeField(fields, "nodes=");
			entry._leafPoints = parseSizeField(fields, "points=");
			entries.push_back(std::move(entry));
		}
		if (end == std::string::npos)
			break;
		start = end + 1;
	}
	return entries;
}

static std::string formatPercent(double fraction)
{
	std::ostringstream out;
	out << std::fixed << std::setprecision(1) << (fraction * 100.0) << "%";
	return out.str();
}

static std::string formatMs(double value)
{
	std::ostringstream out;
	out << std::fixed << std::setprecision(4) << value << " ms";
	return out.str();
}

static std::string schemaLevelDisplayName(const SchemaLevelConfig& level)
{
	std::string name = levelJsonTypeName(level);
	if (!level._axisPolicy.empty())
		name += "(" + level._axisPolicy + ")";
	return name;
}

static std::string schemaChainForReport(const Experiments::SchemaSearchRecord& record, SchemaConfig* loadedSchema)
{
	SchemaConfig schema;
	bool hasSchema = false;
	if (!record._schemaPath.empty() && record._schemaPath.rfind("generated:", 0) != 0)
	{
		std::error_code error;
		if (std::filesystem::exists(record._schemaPath, error))
		{
			try
			{
				schema = Config::loadSchemaConfig(record._schemaPath);
				hasSchema = !schema._levels.empty();
			}
			catch (...)
			{
				hasSchema = false;
			}
		}
	}

	if (hasSchema)
	{
		if (loadedSchema)
			*loadedSchema = schema;
		std::ostringstream out;
		for (size_t i = 0; i < schema._levels.size(); ++i)
		{
			if (i > 0)
				out << " -> ";
			out << schemaLevelDisplayName(schema._levels[i]);
		}
		return out.str();
	}

	if (loadedSchema)
		*loadedSchema = {};
	return record._schemaName;
}

static const ActiveExplainEntry* dominantLeafStructure(const std::vector<ActiveExplainEntry>& entries)
{
	const ActiveExplainEntry* best = nullptr;
	for (const ActiveExplainEntry& entry : entries)
	{
		if (!best || entry._leafPoints > best->_leafPoints)
			best = &entry;
	}
	return best;
}

static const SchemaLevelConfig* levelForDominantStructure(const SchemaConfig& schema, const std::string& shortName)
{
	for (auto it = schema._levels.rbegin(); it != schema._levels.rend(); ++it)
	{
		if (schemaTypeShortName(*it) == shortName)
			return &*it;
	}
	return schema._levels.empty() ? nullptr : &schema._levels.back();
}

static std::string metricComparisonPhrase(const char* verb, const char* noun, double baseline, double current)
{
	if (baseline <= 0.0 || !std::isfinite(baseline) || !std::isfinite(current))
		return {};
	const double reduction = (baseline - current) / baseline;
	std::ostringstream out;
	out << verb << ' ';
	if (std::abs(reduction) < 0.05)
		out << "similar " << noun << " to baseline";
	else if (reduction > 0.0)
		out << formatPercent(reduction) << " fewer " << noun << " than baseline";
	else
		out << formatPercent(-reduction) << " more " << noun << " than baseline";
	return out.str();
}

static void writeQueryBehaviorLine(
	std::ostream& output,
	const char* label,
	const Experiments::QueryMetrics& metrics,
	const Experiments::QueryMetrics* baselineMetrics)
{
	if (metrics._totalQueries == 0)
		return;

	output << "- " << label << ": ";
	if (baselineMetrics && baselineMetrics->_totalQueries > 0)
	{
		const std::string nodePhrase = metricComparisonPhrase(
			"visits", "nodes", baselineMetrics->_averageVisitedNodes, metrics._averageVisitedNodes);
		const std::string pointPhrase = metricComparisonPhrase(
			"tests", "points", baselineMetrics->_averageTestedPoints, metrics._averageTestedPoints);
		if (!nodePhrase.empty())
			output << nodePhrase;
		if (!nodePhrase.empty() && !pointPhrase.empty())
			output << "; ";
		if (!pointPhrase.empty())
			output << pointPhrase;
		if (nodePhrase.empty() && pointPhrase.empty())
			output << "baseline comparison unavailable";
	}
	else
	{
		output << "avg " << metrics._averageVisitedNodes << " visited nodes, "
			<< metrics._averageTestedPoints << " tested points";
	}
	if (metrics._averageFullyContainedNodes > 0.0)
		output << "; " << metrics._averageFullyContainedNodes << " fully-contained nodes/query";
	output << '\n';
}

static void writeDiagnosis(
	std::ostream& output,
	const Experiments::SchemaSearchRecord& record,
	const SchemaConfig& schema,
	const std::vector<ActiveExplainEntry>& activeEntries)
{
	Experiments::SchemaCandidate candidate;
	candidate._config = schema;
	candidate._name = record._schemaName;
	const Experiments::SchemaRepairDiagnostics diagnostics = diagnoseRepairInternal(candidate, { record });
	const ActiveExplainEntry* dominant = dominantLeafStructure(activeEntries);
	const SchemaLevelConfig* dominantLevel = dominant ? levelForDominantStructure(schema, dominant->_typeName) : nullptr;
	const std::string dominantName = dominant ? displayTypeName(dominant->_typeName) : "leaf";

	output << "Diagnosis:\n";
	if (record._isBaseline)
		output << "- Baseline control; use this row as the comparison reference for mixed schemas.\n";
	if (diagnostics._highLeafOccupancy && dominantLevel)
	{
		output << "- Reduce " << dominantName << " leafCapacity from " << dominantLevel->_leafCapacity
			<< " to " << std::max<size_t>(1, dominantLevel->_leafCapacity / 2)
			<< " or add one more level in that block.\n";
	}
	if (diagnostics._testedPointDominated)
	{
		output << "- Tested-points dominate traversal; add a gated leaf micro-index or tighten the deepest leaf capacity.\n";
		if (dominant && (dominant->_typeName == "kd" || dominant->_typeName == "bih") &&
			record._pointFeatures._verticalityScore >= record._pointFeatures._flatnessScore)
		{
			output << "- Try HGrid before " << dominantName << " for dense vertical regions.\n";
		}
	}
	if (diagnostics._visitedNodeDominated)
		output << "- Visited-node count is high while leaf scans are light; coarsen the root or switch the upper block.\n";
	if (diagnostics._fullContainmentDominated)
		output << "- Full-containment shortcuts are carrying the range workload; prefer coarser Grid/QuadTree blocks for this query scale.\n";
	if (diagnostics._likelySingleChildChains)
		output << "- Many nodes likely collapse into one-child chains; flip KDTree axisPolicy or use QuadTree xy/ignore_shortest for terrain-shaped regions.\n";
	if (!diagnostics._highLeafOccupancy &&
		!diagnostics._testedPointDominated &&
		!diagnostics._visitedNodeDominated &&
		!diagnostics._fullContainmentDominated &&
		!diagnostics._likelySingleChildChains &&
		!record._isBaseline)
	{
		output << "- No dominant counter bottleneck; confirm with more query seeds before structural changes.\n";
	}
}

static void writeSchemaExplainReport(
	const std::string& reportPath,
	const std::vector<Experiments::SchemaSearchRecord>& records)
{
	if (reportPath.empty() || records.empty())
		return;

	createParentDirectory(reportPath);
	std::ofstream output(reportPath);
	if (!output.is_open())
		throw std::runtime_error("Unable to open schema explain report path: " + reportPath);

	std::map<std::string, std::vector<const Experiments::SchemaSearchRecord*>> groups;
	for (const Experiments::SchemaSearchRecord& record : records)
		groups[recordGroupKey(record)].push_back(&record);

	std::map<std::string, const Experiments::SchemaSearchRecord*> bestBaselines;
	for (const Experiments::SchemaSearchRecord& record : records)
	{
		if (!record._isBaseline)
			continue;
		const std::string key = recordGroupKey(record);
		const auto existing = bestBaselines.find(key);
		if (existing == bestBaselines.end() || record._score < existing->second->_score)
			bestBaselines[key] = &record;
	}

	output << "# Schema Explain Report\n\n";
	output << "Rows: " << records.size() << "\n\n";

	for (auto& [key, group] : groups)
	{
		std::sort(group.begin(), group.end(), [](const auto* left, const auto* right) {
			if (left->_score == right->_score)
				return left->_schemaName < right->_schemaName;
			return left->_score < right->_score;
		});

		const size_t separator = key.find('\n');
		const std::string dataset = separator == std::string::npos ? key : key.substr(0, separator);
		const std::string workload = separator == std::string::npos ? std::string() : key.substr(separator + 1);
		const auto baselineIt = bestBaselines.find(key);
		const Experiments::SchemaSearchRecord* baseline = baselineIt == bestBaselines.end() ? nullptr : baselineIt->second;

		output << "## " << dataset << " / " << workload << "\n\n";
		if (baseline)
			output << "Best baseline: `" << baseline->_schemaName << "` (score " << baseline->_score << ")\n\n";

		for (const Experiments::SchemaSearchRecord* record : group)
		{
			SchemaConfig schema;
			const std::string chain = schemaChainForReport(*record, &schema);
			const std::vector<ActiveExplainEntry> activeEntries = parseActiveStructureSummary(record->_activeStructureSummary);
			output << "### " << record->_schemaName << "\n\n";
			output << "Schema: " << chain << "\n\n";
			output << "- Score: " << record->_score << " (" << record->_scoreMode << ", " << record->_scoreStage << ")\n";
			output << "- Avg latency: " << formatMs(record->_queryMetrics._averageLatencyMs)
				<< ", p95: " << formatMs(record->_queryMetrics._p95LatencyMs)
				<< ", build: " << formatMs(record->_buildMetrics._buildTimeMs) << "\n";
			if (!record->_bestBaselineSchema.empty())
				output << "- Speedup vs baseline: " << formatPercent(record->_relativeSpeedupVsBaseline) << "\n";
			output << '\n';

			output << "Active structures:\n";
			if (activeEntries.empty())
			{
				output << "- unavailable";
				if (!record->_activeStructureSummary.empty())
					output << " (`" << record->_activeStructureSummary << "`)";
				output << '\n';
			}
			else
			{
				const double totalNodes = static_cast<double>(std::max<size_t>(1, record->_buildMetrics._numNodes));
				const double totalPoints = static_cast<double>(std::max<size_t>(1, record->_buildMetrics._indexedPoints > 0
					? record->_buildMetrics._indexedPoints
					: record->_numPoints));
				for (const ActiveExplainEntry& entry : activeEntries)
				{
					output << "- " << displayTypeName(entry._typeName) << ": ";
					if (entry._schemaOnly)
						output << "schema-level present";
					else
						output << formatPercent(static_cast<double>(entry._nodes) / totalNodes) << " nodes, "
							<< formatPercent(static_cast<double>(entry._leafPoints) / totalPoints) << " leaf points";
					output << '\n';
				}
			}
			output << '\n';

			output << "Query behavior:\n";
			writeQueryBehaviorLine(output, "Range", record->_rangeMetrics, baseline ? &baseline->_rangeMetrics : nullptr);
			writeQueryBehaviorLine(output, "Count range", record->_countRangeMetrics, baseline ? &baseline->_countRangeMetrics : nullptr);
			writeQueryBehaviorLine(output, "Radius", record->_radiusMetrics, baseline ? &baseline->_radiusMetrics : nullptr);
			writeQueryBehaviorLine(output, "KNN", record->_knnMetrics, baseline ? &baseline->_knnMetrics : nullptr);
			if (record->_rangeMetrics._totalQueries == 0 &&
				record->_countRangeMetrics._totalQueries == 0 &&
				record->_radiusMetrics._totalQueries == 0 &&
				record->_knnMetrics._totalQueries == 0)
			{
				output << "- Per-query-family metrics unavailable; avg "
					<< record->_queryMetrics._averageVisitedNodes << " visited nodes and "
					<< record->_queryMetrics._averageTestedPoints << " tested points/query.\n";
			}
			output << '\n';

			writeDiagnosis(output, *record, schema, activeEntries);
			output << '\n';
		}
	}
}

static std::vector<Experiments::SchemaCandidate> uniqueCandidates(
	const std::vector<Experiments::SchemaCandidate>& candidates,
	std::unordered_set<std::string>& seenSignatures)
{
	std::vector<Experiments::SchemaCandidate> unique;
	unique.reserve(candidates.size());
	for (const Experiments::SchemaCandidate& candidate : candidates)
	{
		const std::string signature = schemaSignature(candidate._config);
		if (!seenSignatures.insert(signature).second)
			continue;

		unique.push_back(candidate);
	}
	return unique;
}

static void appendRecords(
	std::vector<Experiments::SchemaSearchRecord>& records,
	const EvaluatedCandidate& evaluation)
{
	records.insert(records.end(), evaluation._records.begin(), evaluation._records.end());
}

static void emitProgress(
	const Experiments::SchemaSearchOptions& options,
	const Experiments::SchemaSearchRecord& record)
{
	if (options.progressCallback)
		options.progressCallback(record);
}

static void sortEvaluations(std::vector<EvaluatedCandidate>& evaluations)
{
	std::sort(evaluations.begin(), evaluations.end(), [](const EvaluatedCandidate& left, const EvaluatedCandidate& right) {
		return left.aggregateScore < right.aggregateScore;
	});
}

struct CandidateObjectives
{
	double _avgLatencyMs = 0.0;
	double _buildTimeMs = 0.0;
	double _memoryMb = 0.0;
	double _imbalancePenalty = 0.0;
};

// Averages the four Pareto objectives across a candidate's per-(dataset, workload) records, weighting all datasets equally like aggregateScore does.
static CandidateObjectives objectivesOf(const EvaluatedCandidate& evaluation)
{
	CandidateObjectives obj;
	if (evaluation._records.empty())
		return obj;
	for (const Experiments::SchemaSearchRecord& record : evaluation._records)
	{
		// Seed-averaged mean when multi-seed confirmation ran, else the single-seed estimate; mirrors selectParetoRecords so optimizer and CSV agree on "latency".
		const double latency = record._confirmSeedsUsed > 0
			? record._latencyMean
			: record._queryMetrics._averageLatencyMs;
		const double memoryMb = static_cast<double>(record._buildMetrics._memoryEstimateBytes) / (1024.0 * 1024.0);
		const double imbalance = record._buildMetrics._averageLeafOccupancy > 0.0
			? static_cast<double>(record._buildMetrics._maxLeafOccupancy) / record._buildMetrics._averageLeafOccupancy
			: 0.0;
		obj._avgLatencyMs += latency;
		obj._buildTimeMs += record._buildMetrics._buildTimeMs;
		obj._memoryMb += memoryMb;
		obj._imbalancePenalty += imbalance;
	}
	const double n = static_cast<double>(evaluation._records.size());
	obj._avgLatencyMs /= n;
	obj._buildTimeMs /= n;
	obj._memoryMb /= n;
	obj._imbalancePenalty /= n;
	return obj;
}

static bool dominatesCandidate(const CandidateObjectives& a, const CandidateObjectives& b)
{
	const bool allLeq =
		a._avgLatencyMs <= b._avgLatencyMs &&
		a._buildTimeMs <= b._buildTimeMs &&
		a._memoryMb <= b._memoryMb &&
		a._imbalancePenalty <= b._imbalancePenalty;
	if (!allLeq)
		return false;
	return
		a._avgLatencyMs < b._avgLatencyMs ||
		a._buildTimeMs < b._buildTimeMs ||
		a._memoryMb < b._memoryMb ||
		a._imbalancePenalty < b._imbalancePenalty;
}

// NSGA-II fast non-dominated sort + crowding distance: populates paretoFront (0 = best) and crowdingDistance, sorting by front then larger crowding so elites stay spread across the front.
static void nsga2RankAndSort(std::vector<EvaluatedCandidate>& evaluations)
{
	const size_t n = evaluations.size();
	if (n == 0)
		return;

	std::vector<CandidateObjectives> objectives(n);
	for (size_t i = 0; i < n; ++i)
	{
		objectives[i] = objectivesOf(evaluations[i]);
		evaluations[i]._paretoFront = -1;
		evaluations[i]._crowdingDistance = 0.0;
	}

	std::vector<std::vector<size_t>> dominatedBy(n);
	std::vector<size_t> dominationCount(n, 0);
	std::vector<size_t> currentFront;
	currentFront.reserve(n);

	for (size_t p = 0; p < n; ++p)
	{
		for (size_t q = 0; q < n; ++q)
		{
			if (p == q)
				continue;
			if (dominatesCandidate(objectives[p], objectives[q]))
				dominatedBy[p].push_back(q);
			else if (dominatesCandidate(objectives[q], objectives[p]))
				++dominationCount[p];
		}
		if (dominationCount[p] == 0)
		{
			evaluations[p]._paretoFront = 0;
			currentFront.push_back(p);
		}
	}

	int rank = 0;
	while (!currentFront.empty())
	{
		std::vector<size_t> nextFront;
		for (const size_t p : currentFront)
		{
			for (const size_t q : dominatedBy[p])
			{
				if (dominationCount[q] == 0)
					continue;
				if (--dominationCount[q] == 0)
				{
					evaluations[q]._paretoFront = rank + 1;
					nextFront.push_back(q);
				}
			}
		}
		++rank;
		currentFront = std::move(nextFront);
	}

	// Crowding distance per front; each objective's extreme candidate gets +inf so front edges are always preserved.
	std::map<int, std::vector<size_t>> fronts;
	for (size_t i = 0; i < n; ++i)
		fronts[evaluations[i]._paretoFront].push_back(i);

	auto objectiveValue = [&](size_t idx, int axis) {
		switch (axis)
		{
		case 0: return objectives[idx]._avgLatencyMs;
		case 1: return objectives[idx]._buildTimeMs;
		case 2: return objectives[idx]._memoryMb;
		default: return objectives[idx]._imbalancePenalty;
		}
	};

	for (auto& [frontRank, indices] : fronts)
	{
		if (indices.size() <= 2)
		{
			for (const size_t idx : indices)
				evaluations[idx]._crowdingDistance = std::numeric_limits<double>::infinity();
			continue;
		}
		for (int axis = 0; axis < 4; ++axis)
		{
			std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
				return objectiveValue(a, axis) < objectiveValue(b, axis);
			});
			const double lo = objectiveValue(indices.front(), axis);
			const double hi = objectiveValue(indices.back(), axis);
			evaluations[indices.front()]._crowdingDistance = std::numeric_limits<double>::infinity();
			evaluations[indices.back()]._crowdingDistance = std::numeric_limits<double>::infinity();
			if (hi <= lo)
				continue;
			const double range = hi - lo;
			for (size_t i = 1; i + 1 < indices.size(); ++i)
			{
				if (std::isinf(evaluations[indices[i]]._crowdingDistance))
					continue;
				const double prev = objectiveValue(indices[i - 1], axis);
				const double next = objectiveValue(indices[i + 1], axis);
				evaluations[indices[i]]._crowdingDistance += (next - prev) / range;
			}
		}
	}

	std::sort(evaluations.begin(), evaluations.end(), [](const EvaluatedCandidate& a, const EvaluatedCandidate& b) {
		if (a._paretoFront != b._paretoFront)
			return a._paretoFront < b._paretoFront;
		if (a._crowdingDistance != b._crowdingDistance)
			return a._crowdingDistance > b._crowdingDistance;
		// Tie-break by scalar score: on a near-1D front every edge candidate gets +inf crowding, so without this the worst-but-edge one could land at archive[0] and poison the elite pool.
		return a.aggregateScore < b.aggregateScore;
	});

	// Hard-guarantee elitism: a final pass swaps the scalar champion into archive[0], since ranking or front geometry could otherwise leave a non-champion there.
	auto scalarBest = std::min_element(evaluations.begin(), evaluations.end(),
		[](const EvaluatedCandidate& a, const EvaluatedCandidate& b) {
			return a.aggregateScore < b.aggregateScore;
		});
	if (scalarBest != evaluations.end() && scalarBest != evaluations.begin())
		std::iter_swap(evaluations.begin(), scalarBest);
}

static std::vector<DatasetContext> makeDatasetContexts(
	const std::vector<SearchDataset>& datasets,
	const std::vector<Experiments::WorkloadProfile>& workloads,
	bool cudaEvaluator)
{
	std::vector<DatasetContext> contexts;
	contexts.reserve(datasets.size());
	for (const SearchDataset& dataset : datasets)
	{
		DatasetContext context;
		context._dataset = &dataset;
		context._features = Experiments::extractPointCloudFeatures(dataset._cloud);
		context._preparedWorkloads.reserve(workloads.size());
		for (const Experiments::WorkloadProfile& workload : workloads)
			context._preparedWorkloads.push_back(prepareWorkloadProfile(workload, dataset._cloud, cudaEvaluator));
		contexts.push_back(std::move(context));
	}
	return contexts;
}

static std::vector<DatasetContext> makeDatasetContextsFromPointers(
	const std::vector<const SearchDataset*>& datasets,
	const std::vector<Experiments::WorkloadProfile>& workloads,
	bool cudaEvaluator)
{
	std::vector<DatasetContext> contexts;
	contexts.reserve(datasets.size());
	for (const SearchDataset* dataset : datasets)
	{
		if (!dataset)
			continue;
		DatasetContext context;
		context._dataset = dataset;
		context._features = Experiments::extractPointCloudFeatures(dataset->_cloud);
		context._preparedWorkloads.reserve(workloads.size());
		for (const Experiments::WorkloadProfile& workload : workloads)
			context._preparedWorkloads.push_back(prepareWorkloadProfile(workload, dataset->_cloud, cudaEvaluator));
		contexts.push_back(std::move(context));
	}
	return contexts;
}

static PointCloud downsampleCloud(const PointCloud& cloud, size_t pointCap)
{
	if (pointCap == 0 || cloud.size() <= pointCap)
		return cloud;

	PointCloud downsampled;
	downsampled.reserve(pointCap);
	for (size_t i = 0; i < pointCap; ++i)
	{
		const size_t sourceIndex = pointCap <= 1
			? size_t(0)
			: std::min(cloud.size() - 1, (i * (cloud.size() - 1)) / (pointCap - 1));
		downsampled.addPoint(cloud.points()[sourceIndex]);
	}
	return downsampled;
}

static std::vector<SearchDataset> makeProxyDatasets(const std::vector<SearchDataset>& datasets, size_t pointCap)
{
	std::vector<SearchDataset> proxy;
	proxy.reserve(datasets.size());
	for (const SearchDataset& dataset : datasets)
	{
		SearchDataset item;
		item._name = dataset._name + "_proxy";
		item._source = dataset._source;
		item._cloud = downsampleCloud(dataset._cloud, pointCap);
		proxy.push_back(std::move(item));
	}
	return proxy;
}

static std::vector<Experiments::WorkloadProfile> withQueryCount(
	std::vector<Experiments::WorkloadProfile> workloads,
	size_t queryCount)
{
	if (queryCount == 0)
		return workloads;

	for (Experiments::WorkloadProfile& workload : workloads)
		workload._numQueries = queryCount;
	return workloads;
}

template <typename T>
static std::vector<T> thinSortedValues(std::vector<T> values, size_t maxValues)
{
	sortUniqueValues(values);
	if (values.size() <= maxValues || maxValues == 0)
		return values;

	std::vector<T> thinned;
	thinned.reserve(maxValues);
	for (size_t i = 0; i < maxValues; ++i)
	{
		const size_t sourceIndex = maxValues == 1
			? size_t(0)
			: (i * (values.size() - 1)) / (maxValues - 1);
		thinned.push_back(values[sourceIndex]);
	}
	sortUniqueValues(thinned);
	return thinned;
}

static std::vector<size_t> deepLeafCandidates(
	const Experiments::SchemaGenerationOptions& options,
	const Experiments::ConditionDomain& domain)
{
	const size_t minLeaf = std::max<size_t>(1, std::min(options._minLeafCapacity, options._maxLeafCapacity));
	const size_t maxLeaf = std::max(minLeaf, options._maxLeafCapacity);
	std::vector<size_t> values;
	for (size_t value = clampPowerOfTwo(minLeaf, minLeaf, maxLeaf); value <= maxLeaf; value *= 2)
	{
		values.push_back(value);
		if (value > maxLeaf / 2)
			break;
	}
	for (const size_t threshold : domain._pointThresholds)
		values.push_back(clampPowerOfTwo(threshold, minLeaf, maxLeaf));
	return thinSortedValues(values, 8);
}

static std::vector<size_t> deepPointThresholdCandidates(
	const Experiments::SchemaGenerationOptions& options,
	const Experiments::ConditionDomain& domain)
{
	const size_t minLeaf = std::max<size_t>(1, std::min(options._minLeafCapacity, options._maxLeafCapacity));
	const size_t maxThreshold = std::max<size_t>(minLeaf * 2, std::min<size_t>(options._maxLeafCapacity * 16, 1 << 20));
	std::vector<size_t> values = domain._pointThresholds.empty() ? fallbackPointThresholds() : domain._pointThresholds;
	for (const size_t fallback : fallbackPointThresholds())
		values.push_back(fallback);
	for (size_t& value : values)
		value = std::clamp(value, minLeaf, maxThreshold);
	return thinSortedValues(values, 8);
}

static std::vector<double> deepDoubleCandidates(std::vector<double> values, const std::vector<double>& fallback, size_t maxValues)
{
	if (values.empty())
		values = fallback;
	for (const double value : fallback)
		values.push_back(value);
	return thinSortedValues(values, maxValues);
}

static double representativeQueryScale(const std::vector<Experiments::WorkloadProfile>& workloads)
{
	double weighted = 0.0;
	double totalWeight = 0.0;
	for (const Experiments::WorkloadProfile& workload : workloads)
	{
		if (workload._rangeWeight > 0.0)
		{
			weighted += workload._rangeWeight * 0.5 * (workload._rangeScaleMin + workload._rangeScaleMax);
			totalWeight += workload._rangeWeight;
		}
		if (workload._radiusWeight > 0.0)
		{
			weighted += workload._radiusWeight * 0.5 * (workload._radiusScaleMin + workload._radiusScaleMax);
			totalWeight += workload._radiusWeight;
		}
	}
	return totalWeight > 0.0 ? weighted / totalWeight : 0.05;
}

static std::vector<size_t> deepHandoffDepthCandidates(
	const Experiments::SchemaGenerationOptions& options,
	const std::vector<Experiments::WorkloadProfile>& workloads)
{
	const size_t maxDepth = std::max<size_t>(2, options._maxDepth);
	const double scale = std::clamp(representativeQueryScale(workloads), 0.001, 0.5);
	const size_t suggested = std::clamp(
		static_cast<size_t>(std::ceil(std::log2(1.0 / scale))),
		size_t(1),
		maxDepth - 1);
	std::vector<size_t> values = { 1, 2, 3, 4, suggested };
	if (suggested > 1)
		values.push_back(suggested - 1);
	if (suggested + 1 < maxDepth)
		values.push_back(suggested + 1);
	for (size_t& value : values)
		value = std::clamp(value, size_t(1), maxDepth - 1);
	return thinSortedValues(values, 6);
}

static SchemaLevelConfig makeDeepLevel(const std::string& typeName, size_t numLevels, size_t leafCapacity)
{
	SchemaLevelConfig level;
	level._typeName = typeName;
	level._type = Config::parseDataStructureLevel(typeName);
	level._numLevels = std::max<size_t>(1, numLevels);
	level._leafCapacity = std::max<size_t>(2, leafCapacity);
	level._minPrimitivesToSplit = std::max<size_t>(2, level._leafCapacity / 4);
	if (level._type == MultiDataStructure::QuadTreeNode)
		level._axisPolicy = "xy";
	else if (level._type == MultiDataStructure::KDTreeNode)
		level._axisPolicy = "median_longest_axis";
	return level;
}

static bool deepTypeAllowedByProfile(const std::string& typeName, const Experiments::SchemaGenerationOptions& options)
{
	if (!queryMinimalPrimitiveProfile(options))
		return true;
	return !isBIHLevelName(typeName) &&
		!isKarrasOctreeLevelName(typeName) &&
		!isLBVHLevelName(typeName) &&
		!isRegularGridLevelName(typeName) &&
		!isHGridLevelName(typeName);
}

static bool isDeepNestedSchema(const SchemaConfig& schema)
{
	if (schema._levels.size() < 2)
		return false;
	std::set<std::string> types;
	for (const SchemaLevelConfig& level : schema._levels)
		types.insert(schemaTypeShortName(level));
	return types.size() >= 2;
}

static SchemaLevelCondition makeDeepSecondStageCondition(
	const std::string& firstType,
	const std::string& secondType,
	size_t minPoints,
	double heightRatio,
	double density,
	double extentX,
	double extentZ,
	size_t variant)
{
	SchemaLevelCondition condition;
	condition._minPoints = minPoints;

	if (isRegularGridLevelName(secondType) || isHGridLevelName(secondType) ||
		isRegularGridLevelName(firstType) || isHGridLevelName(firstType))
	{
		condition._minDensity = density;
	}
	else if (Config::parseDataStructureLevel(secondType) == MultiDataStructure::OctreeNode)
	{
		condition._minHeightRatio = heightRatio;
	}
	else if (Config::parseDataStructureLevel(firstType) == MultiDataStructure::QuadTreeNode)
	{
		condition._minHeightRatio = heightRatio;
	}

	if (variant % 4 == 1 && extentX > 0.0)
		condition._minExtentX = extentX;
	else if (variant % 4 == 2 && extentZ > 0.0)
		condition._minExtentZ = extentZ;

	return condition;
}

static std::vector<Experiments::SchemaCandidate> generateDeepNestedCandidates(
	const Experiments::SchemaGenerationOptions& baseOptions,
	const Experiments::ConditionDomain& domain,
	const std::vector<Experiments::WorkloadProfile>& workloads)
{
	Experiments::SchemaGenerationOptions options = baseOptions;
	options._minBlocks = std::max<size_t>(2, options._minBlocks);
	options._maxBlocks = std::max(options._minBlocks, options._maxBlocks);
	options._maxDepth = std::max<size_t>(2, options._maxDepth);
	options._conditionalLevels = true;
	options._conditionalProbability = std::max(0.75, options._conditionalProbability);

	struct DeepFamily
	{
		const char* first = "";
		std::vector<const char*> _seconds;
	};

	const std::array<DeepFamily, 5> families = {
		DeepFamily{ "QuadTree", { "Octree", "KarrasOctree", "RegularGrid" } },
		DeepFamily{ "RegularGrid", { "BIH", "KDTree", "LBVH" } },
		DeepFamily{ "HGrid", { "BIH", "KDTree", "LBVH" } },
		DeepFamily{ "Octree", { "KDTree", "BIH" } },
		DeepFamily{ "KarrasOctree", { "KDTree", "BIH" } },
	};

	const std::vector<size_t> firstDepths = deepHandoffDepthCandidates(options, workloads);
	const std::vector<size_t> leafValues = deepLeafCandidates(options, domain);
	const std::vector<size_t> pointThresholds = deepPointThresholdCandidates(options, domain);
	const std::vector<double> heightValues = deepDoubleCandidates(domain._heightRatioThresholds, fallbackHeightRatioThresholds(), 6);
	const std::vector<double> densityValues = deepDoubleCandidates(domain._densityThresholds, { 0.0001, 0.001, 0.01, 0.1 }, 6);
	const std::vector<double> extentXValues = deepDoubleCandidates(domain._extentXThresholds, {}, 4);
	const std::vector<double> extentZValues = deepDoubleCandidates(domain._extentZThresholds, {}, 4);
	const std::array<size_t, 5> secondDepthSeeds = { 1, 2, 3, 4, 6 };

	std::unordered_set<std::string> seen;
	std::vector<Experiments::SchemaCandidate> candidates;
	candidates.reserve(options._count);

	std::mt19937 rng(options._seed);
	size_t variant = 0;
	for (const DeepFamily& family : families)
	{
		if (!deepTypeAllowedByProfile(family.first, options))
			continue;
		for (const char* secondType : family._seconds)
		{
			if (!deepTypeAllowedByProfile(secondType, options))
				continue;
			for (const size_t firstDepth : firstDepths)
			{
				for (const size_t secondDepthSeed : secondDepthSeeds)
				{
					if (firstDepth + secondDepthSeed > options._maxDepth)
						continue;
					for (const size_t leaf : leafValues)
					{
						for (const size_t minPoints : pointThresholds)
						{
							const double height = heightValues.empty() ? 0.25 : heightValues[variant % heightValues.size()];
							const double density = densityValues.empty() ? 0.001 : densityValues[(variant / 2) % densityValues.size()];
							const double extentX = extentXValues.empty() ? 0.0 : extentXValues[(variant / 3) % extentXValues.size()];
							const double extentZ = extentZValues.empty() ? 0.0 : extentZValues[(variant / 5) % extentZValues.size()];

							SchemaConfig schema;
							schema._levels.push_back(makeDeepLevel(family.first, firstDepth, leaf));
							maybeAssignAdaptiveLeafCapacity(schema._levels.back(), rng, options);
							SchemaLevelConfig second = makeDeepLevel(
								secondType,
								secondDepthSeed,
								std::max<size_t>(options._minLeafCapacity, leaf / 2));
							maybeAssignAdaptiveLeafCapacity(second, rng, options);
							second._condition = makeDeepSecondStageCondition(
								family.first,
								secondType,
								minPoints,
								height,
								density,
								extentX,
								extentZ,
								variant);
							schema._levels.push_back(second);
							normalizeSchemaForGeneration(schema, options);
							if (!isDeepNestedSchema(schema))
							{
								++variant;
								continue;
							}

							const std::string signature = schemaSignature(schema);
							if (seen.insert(signature).second)
								candidates.push_back(materializeGeneratedSchema(schema, "deep", options._outputDirectory));
							++variant;

							if (candidates.size() >= options._count)
								return candidates;
						}
					}
				}
			}
		}
	}

	if (candidates.size() < options._count)
	{
		Experiments::SchemaGenerationOptions fallback = options;
		fallback._count = options._count - candidates.size();
		fallback._minBlocks = 2;
		std::vector<Experiments::SchemaCandidate> randomNested = Experiments::generateSchemaCandidates(fallback, &domain);
		for (Experiments::SchemaCandidate& candidate : randomNested)
		{
			if (!isDeepNestedSchema(candidate._config))
				continue;
			const std::string signature = schemaSignature(candidate._config);
			if (seen.insert(signature).second)
				candidates.push_back(std::move(candidate));
			if (candidates.size() >= options._count)
				break;
		}
	}

	if (candidates.size() < options._count)
		std::cerr << "Warning: generated " << candidates.size() << " deep nested schemas from requested " << options._count << '\n';
	return candidates;
}

// Key for the candidate's root (primary) block type, so the diversifier can keep at least one survivor per family when advancing top-K between rungs.
static std::string primaryBlockKey(const SchemaConfig& schema)
{
	if (schema._levels.empty())
		return "";
	return schemaTypeShortName(schema._levels.front());
}

static std::vector<Experiments::SchemaCandidate> topCandidates(
	const std::vector<EvaluatedCandidate>& evaluations,
	size_t count)
{
	// Score-sorted advancement with a per-primary-block diversity pass first, so visit-proxy bias toward HGrid can't make the shortlist mono-cultural.
	std::vector<Experiments::SchemaCandidate> result;
	const size_t limit = count == 0 ? evaluations.size() : std::min(count, evaluations.size());
	result.reserve(limit);
	std::unordered_set<std::string> picked;
	std::unordered_set<std::string> primarySeen;
	std::vector<size_t> deferred;
	deferred.reserve(evaluations.size());

	// Pass 1: take the best of each primary-block type, in score order.
	for (size_t i = 0; i < evaluations.size() && result.size() < limit; ++i)
	{
		const Experiments::SchemaCandidate& candidate = evaluations[i]._candidate;
		const std::string primary = primaryBlockKey(candidate._config);
		if (!primarySeen.insert(primary).second)
		{
			deferred.push_back(i);
			continue;
		}
		result.push_back(candidate);
		picked.insert(candidate._path.empty() ? candidate._name : candidate._path);
	}

	// Pass 2: fill remaining slots from the deferred pool, still in score order.
	for (const size_t i : deferred)
	{
		if (result.size() >= limit)
			break;
		const Experiments::SchemaCandidate& candidate = evaluations[i]._candidate;
		result.push_back(candidate);
		picked.insert(candidate._path.empty() ? candidate._name : candidate._path);
	}

	const size_t diverseCount = primarySeen.size();
	if (diverseCount > 1 && diverseCount < result.size())
		std::cout << "    + advanced " << diverseCount << " distinct primary-block type(s) into next rung\n";

	// Force-promote any baseline not already in the top-K so naive single-block structures are always compared at the next stage.
	size_t baselinesAdded = 0;
	for (const EvaluatedCandidate& evaluation : evaluations)
	{
		if (!evaluation._candidate._isBaseline)
			continue;
		const std::string key = evaluation._candidate._path.empty() ? evaluation._candidate._name : evaluation._candidate._path;
		if (!picked.insert(key).second)
			continue;
		result.push_back(evaluation._candidate);
		++baselinesAdded;
	}
	if (baselinesAdded > 0)
		std::cout << "    + force-promoted " << baselinesAdded << " baseline schema(s) past rank cut\n";
	return result;
}

static void appendBaselineControls(
	std::vector<Experiments::SchemaCandidate>& candidates,
	const std::vector<Experiments::SchemaCandidate>& baselines);

static std::vector<Experiments::SchemaCandidate> topSearchCandidatesWithBaselineControls(
	const std::vector<EvaluatedCandidate>& evaluations,
	size_t count,
	const std::vector<Experiments::SchemaCandidate>& baselines)
{
	std::vector<EvaluatedCandidate> searchEvaluations;
	searchEvaluations.reserve(evaluations.size());
	for (const EvaluatedCandidate& evaluation : evaluations)
	{
		if (!evaluation._candidate._isBaseline)
			searchEvaluations.push_back(evaluation);
	}

	std::vector<Experiments::SchemaCandidate> result = searchEvaluations.empty()
		? topCandidates(evaluations, count)
		: topCandidates(searchEvaluations, count);
	appendBaselineControls(result, baselines);
	return result;
}

static bool recordCountsAsNested(const Experiments::SchemaSearchRecord& record)
{
	return !record._isBaseline &&
		record._activeStructureTypes >= 2 &&
		record._nestedActiveFraction >= 0.05;
}

static bool evaluationCountsAsNested(const EvaluatedCandidate& evaluation)
{
	for (const Experiments::SchemaSearchRecord& record : evaluation._records)
	{
		if (recordCountsAsNested(record))
			return true;
	}
	return false;
}

static void appendUniqueCandidate(
	std::vector<Experiments::SchemaCandidate>& out,
	std::unordered_set<std::string>& picked,
	const Experiments::SchemaCandidate& candidate)
{
	const std::string key = candidate._path.empty() ? candidate._config._name : candidate._path;
	if (picked.insert(key).second)
		out.push_back(candidate);
}

static std::vector<Experiments::SchemaCandidate> topDeepNestedCandidates(
	const std::vector<EvaluatedCandidate>& evaluations,
	size_t count,
	bool includeBaselines)
{
	std::vector<Experiments::SchemaCandidate> result;
	std::unordered_set<std::string> picked;
	const size_t limit = count == 0 ? evaluations.size() : count;

	for (const EvaluatedCandidate& evaluation : evaluations)
	{
		if (result.size() >= limit)
			break;
		if (!evaluation._candidate._isBaseline && evaluationCountsAsNested(evaluation))
			appendUniqueCandidate(result, picked, evaluation._candidate);
	}

	if (result.size() < limit)
	{
		for (const EvaluatedCandidate& evaluation : evaluations)
		{
			if (result.size() >= limit)
				break;
			if (!evaluation._candidate._isBaseline)
				appendUniqueCandidate(result, picked, evaluation._candidate);
		}
	}

	if (includeBaselines)
	{
		size_t baselinesAdded = 0;
		for (const EvaluatedCandidate& evaluation : evaluations)
		{
			if (!evaluation._candidate._isBaseline)
				continue;
			const size_t previousSize = result.size();
			appendUniqueCandidate(result, picked, evaluation._candidate);
			if (result.size() != previousSize)
				++baselinesAdded;
		}
		if (baselinesAdded > 0)
			std::cout << "    + force-promoted " << baselinesAdded << " baseline schema(s) past deep-search rank cut\n";
	}

	return result;
}

static std::optional<Experiments::SchemaCandidate> bestBaselineCandidate(const std::vector<EvaluatedCandidate>& evaluations)
{
	for (const EvaluatedCandidate& evaluation : evaluations)
	{
		if (evaluation._candidate._isBaseline)
			return evaluation._candidate;
	}
	return std::nullopt;
}

struct LocalCellStats
{
	size_t _count = 0;
	glm::vec3 min = glm::vec3(std::numeric_limits<float>::max());
	glm::vec3 max = glm::vec3(std::numeric_limits<float>::lowest());
};

struct LocalOpportunityStats
{
	std::map<std::string, size_t> _suggestedTypes;
	double _dominantShare = 0.0;
	size_t _testedCells = 0;
};

static LocalOpportunityStats estimateLocalOpportunity(const PointCloud& cloud, size_t pointCap)
{
	LocalOpportunityStats result;
	if (cloud.empty())
		return result;

	const AABB bounds = cloud.bounds();
	const glm::vec3 extent = glm::max(bounds.size(), glm::vec3(static_cast<float>(EPSILON)));
	constexpr size_t gridResolution = 4;
	constexpr size_t cellCount = gridResolution * gridResolution * gridResolution;
	std::array<LocalCellStats, cellCount> cells;
	const size_t sampleCount = pointCap == 0 ? cloud.size() : std::min(pointCap, cloud.size());
	for (size_t i = 0; i < sampleCount; ++i)
	{
		const size_t sourceIndex = sampleCount <= 1
			? size_t(0)
			: std::min(cloud.size() - 1, (i * (cloud.size() - 1)) / (sampleCount - 1));
		const glm::vec3& position = cloud.points()[sourceIndex].position;
		glm::vec3 normalized = (position - bounds.min()) / extent;
		normalized = glm::clamp(normalized, glm::vec3(0.0f), glm::vec3(0.999999f));
		const size_t ix = static_cast<size_t>(normalized.x * gridResolution);
		const size_t iy = static_cast<size_t>(normalized.y * gridResolution);
		const size_t iz = static_cast<size_t>(normalized.z * gridResolution);
		LocalCellStats& cell = cells[ix + gridResolution * (iy + gridResolution * iz)];
		++cell._count;
		cell.min = glm::min(cell.min, position);
		cell.max = glm::max(cell.max, position);
	}

	std::vector<double> densities;
	std::vector<size_t> counts;
	for (const LocalCellStats& cell : cells)
	{
		if (cell._count == 0)
			continue;
		const glm::vec3 cellExtent = glm::max(cell.max - cell.min, glm::vec3(static_cast<float>(EPSILON)));
		const double volume = static_cast<double>(cellExtent.x) * cellExtent.y * cellExtent.z;
		densities.push_back(volume > EPSILON ? static_cast<double>(cell._count) / volume : 0.0);
		counts.push_back(cell._count);
	}
	if (counts.empty())
		return result;
	std::sort(counts.begin(), counts.end());
	std::sort(densities.begin(), densities.end());
	const size_t medianCount = counts[counts.size() / 2];
	const double medianDensity = densities[densities.size() / 2];

	for (const LocalCellStats& cell : cells)
	{
		if (cell._count < std::max<size_t>(4, medianCount / 2))
			continue;

		const glm::vec3 cellExtent = glm::max(cell.max - cell.min, glm::vec3(static_cast<float>(EPSILON)));
		const double horizontal = std::max({ static_cast<double>(cellExtent.x), static_cast<double>(cellExtent.y), EPSILON });
		const double heightRatio = static_cast<double>(cellExtent.z) / horizontal;
		const double minExtent = std::max(EPSILON, static_cast<double>(std::min({ cellExtent.x, cellExtent.y, cellExtent.z })));
		const double maxExtent = static_cast<double>(std::max({ cellExtent.x, cellExtent.y, cellExtent.z }));
		const double density = static_cast<double>(cell._count) /
			std::max(EPSILON, static_cast<double>(cellExtent.x) * cellExtent.y * cellExtent.z);

		std::string winner = "ot";
		if (heightRatio < 0.18)
			winner = "qt";
		else if (density > medianDensity * 1.75 && cell._count > medianCount)
			winner = "rg";
		else if (maxExtent / minExtent > 3.0)
			winner = "kd";

		++result._suggestedTypes[winner];
		++result._testedCells;
	}

	size_t dominant = 0;
	for (const auto& [typeName, count] : result._suggestedTypes)
		dominant = std::max(dominant, count);
	result._dominantShare = result._testedCells > 0
		? static_cast<double>(dominant) / static_cast<double>(result._testedCells)
		: 0.0;
	return result;
}

static void printLocalOpportunity(
	const SearchDataset& dataset,
	const Experiments::SchemaCandidate& bestBaseline,
	size_t pointCap)
{
	const LocalOpportunityStats local = estimateLocalOpportunity(dataset._cloud, pointCap);
	const std::string globalType = bestBaseline._config._levels.empty()
		? std::string()
		: schemaTypeShortName(bestBaseline._config._levels.front());

	std::cout << "    local opportunity: " << local._testedCells << " shallow occupied cells";
	if (local._testedCells == 0)
	{
		std::cout << " (insufficient local samples)\n";
		return;
	}

	std::cout << ", suggested second-stage types ";
	bool first = true;
	bool differsFromGlobal = false;
	for (const auto& [typeName, count] : local._suggestedTypes)
	{
		if (!first)
			std::cout << "; ";
		first = false;
		std::cout << typeName << "=" << count;
		if (typeName != globalType)
			differsFromGlobal = true;
	}
	std::cout << ", dominant share " << local._dominantShare;
	if (!differsFromGlobal || local._suggestedTypes.size() <= 1)
		std::cout << " (low nested-opportunity signal)";
	else
		std::cout << " (local winners differ from global " << globalType << ")";
	std::cout << '\n';
}

static std::vector<EvaluatedCandidate> evaluateAutoConditionStage(
	const std::string& label,
	const std::vector<Experiments::SchemaCandidate>& candidates,
	const std::vector<DatasetContext>& contexts,
	const std::vector<Experiments::WorkloadProfile>& workloads,
	const Experiments::SchemaSearchOptions& options);

static void runDeepNestedDiagnostics(
	const Experiments::SchemaSearchOptions& options,
	const std::vector<SearchDataset>& datasets,
	const std::vector<Experiments::WorkloadProfile>& workloads,
	const std::vector<Experiments::SchemaCandidate>& baselines)
{
	if (baselines.empty())
	{
		std::cout << "  deep nested diagnostics: skipped (no baseline schemas loaded)\n";
		return;
	}

	Experiments::SchemaSearchOptions diagnosticOptions = options;
	diagnosticOptions._evaluator = "cpu";
	diagnosticOptions._weights._useVisitProxy = false;
	const std::vector<Experiments::WorkloadProfile> diagnosticWorkloads = withQueryCount(workloads, 64);
	const std::vector<DatasetContext> contexts = makeDatasetContexts(datasets, diagnosticWorkloads, false);
	std::cout << "  deep nested diagnostics: pure baselines at 64 queries\n";
	std::vector<EvaluatedCandidate> baselineEvaluations = evaluateAutoConditionStage(
		"diagnostic baselines",
		baselines,
		contexts,
		diagnosticWorkloads,
		diagnosticOptions);
	if (baselineEvaluations.empty())
		return;

	const Experiments::SchemaCandidate& bestBaseline = baselineEvaluations.front()._candidate;
	std::cout << "    best pure baseline: " << bestBaseline._config._name
		<< " aggregate score " << baselineEvaluations.front().aggregateScore << '\n';
	for (const SearchDataset& dataset : datasets)
		printLocalOpportunity(dataset, bestBaseline, options._autoConditions._proxyPointCap);
}

static std::vector<Experiments::WorkloadProfile> makeRobustnessWorkloads(
	const std::vector<Experiments::WorkloadProfile>& workloads,
	size_t queryCount,
	size_t seedCount)
{
	std::vector<Experiments::WorkloadProfile> robust;
	robust.reserve(workloads.size() * seedCount);
	for (size_t seedIndex = 0; seedIndex < seedCount; ++seedIndex)
	{
		for (const Experiments::WorkloadProfile& workload : workloads)
		{
			Experiments::WorkloadProfile copy = workload;
			copy._name = workload._name + "_robust_seed" + std::to_string(seedIndex);
			copy._numQueries = queryCount;
			copy._querySeed = workload._querySeed + static_cast<uint32_t>(7919 * (seedIndex + 1));
			robust.push_back(copy);
		}
	}
	return robust;
}

static void appendBaselineControls(
	std::vector<Experiments::SchemaCandidate>& candidates,
	const std::vector<Experiments::SchemaCandidate>& baselines)
{
	std::unordered_set<std::string> picked;
	for (const Experiments::SchemaCandidate& candidate : candidates)
		picked.insert(candidate._path.empty() ? candidate._config._name : candidate._path);
	for (const Experiments::SchemaCandidate& baseline : baselines)
		appendUniqueCandidate(candidates, picked, baseline);
}

static void runDeepCudaConfirmation(
	const Experiments::SchemaSearchOptions& options,
	const std::vector<SearchDataset>& datasets,
	const std::vector<Experiments::WorkloadProfile>& workloads,
	const std::vector<EvaluatedCandidate>& sourceEvaluations)
{
	std::string cudaError;
	if (!PointGpu::MixedTree::isAvailable(&cudaError))
	{
		std::cout << "  deep nested CUDA confirmation: skipped (CUDA unavailable";
		if (!cudaError.empty())
			std::cout << ": " << cudaError;
		std::cout << ")\n";
		return;
	}

	std::vector<Experiments::SchemaCandidate> candidates = topDeepNestedCandidates(sourceEvaluations, 3, false);
	if (const std::optional<Experiments::SchemaCandidate> baseline = bestBaselineCandidate(sourceEvaluations))
	{
		std::unordered_set<std::string> picked;
		for (const Experiments::SchemaCandidate& candidate : candidates)
			picked.insert(candidate._path.empty() ? candidate._config._name : candidate._path);
		appendUniqueCandidate(candidates, picked, baseline.value());
	}
	if (candidates.empty())
		return;

	Experiments::SchemaSearchOptions cudaOptions = options;
	cudaOptions._evaluator = "cuda";
	cudaOptions._cuda._builder = "mixed";
	if (cudaOptions._cuda._device < 0)
		cudaOptions._cuda._device = 0;
	cudaOptions._weights._useVisitProxy = false;

	const std::vector<Experiments::WorkloadProfile> cudaWorkloads = withQueryCount(workloads, 64);
	const std::vector<DatasetContext> cudaContexts = makeDatasetContexts(datasets, cudaWorkloads, true);
	std::cout << "  deep nested CUDA confirmation: top nested candidates plus best baseline, mixed builder, 64 queries\n";
	const std::vector<EvaluatedCandidate> cudaEvaluations = evaluateAutoConditionStage(
		"cuda confirmation",
		candidates,
		cudaContexts,
		cudaWorkloads,
		cudaOptions);
	if (!cudaEvaluations.empty())
		std::cout << "    cuda confirmation best: " << cudaEvaluations.front()._candidate._config._name
			<< " aggregate score " << cudaEvaluations.front().aggregateScore << " (report only)\n";
}

static std::vector<EvaluatedCandidate> evaluateAutoConditionStage(
	const std::string& label,
	const std::vector<Experiments::SchemaCandidate>& candidates,
	const std::vector<DatasetContext>& contexts,
	const std::vector<Experiments::WorkloadProfile>& workloads,
	const Experiments::SchemaSearchOptions& options)
{
	const size_t parallelWorkers = (!useCudaEvaluator(options) && options._parallelDispatch > 1)
		? std::min(options._parallelDispatch, candidates.size())
		: 1;

	std::cout << "  auto-conditions " << label << ": " << candidates.size() << " candidates, "
		<< (workloads.empty() ? size_t(0) : workloads.front()._numQueries) << " queries";
	if (parallelWorkers > 1)
		std::cout << " (parallel CPU workers: " << parallelWorkers << ")";
	std::cout << '\n';

	std::vector<EvaluatedCandidate> evaluations(candidates.size());
	if (parallelWorkers > 1)
	{
		// CPU-only parallel dispatch; the CUDA per-builder cache isn't thread-safe so CUDA stays serial, and only the internally-locked score cache is shared here.
		std::atomic<size_t> nextIndex{0};
		std::vector<std::thread> workers;
		workers.reserve(parallelWorkers);
		for (size_t w = 0; w < parallelWorkers; ++w)
		{
			workers.emplace_back([&]() {
				for (;;)
				{
					const size_t i = nextIndex.fetch_add(1);
					if (i >= candidates.size())
						return;
					evaluations[i] = evaluateCandidate(candidates[i], contexts, workloads, options, nullptr);
				}
			});
		}
		for (std::thread& worker : workers)
			worker.join();

		for (size_t i = 0; i < candidates.size(); ++i)
		{
			std::cout << "    " << label << " [" << (i + 1) << "/" << candidates.size() << "] "
				<< candidates[i]._config._name << "  aggregate score " << evaluations[i].aggregateScore << '\n';
		}
	}
	else
	{
		CudaIndexCache cudaCache;
		for (size_t i = 0; i < candidates.size(); ++i)
		{
			const Experiments::SchemaCandidate& candidate = candidates[i];
			std::cout << "    " << label << " [" << (i + 1) << "/" << candidates.size() << "] " << candidate._config._name << '\n';
			evaluations[i] = evaluateCandidate(candidate, contexts, workloads, options, &cudaCache);
			std::cout << "      aggregate score " << evaluations[i].aggregateScore << '\n';
		}
	}

	sortEvaluations(evaluations);
	if (!evaluations.empty())
		std::cout << "    " << label << " best: " << evaluations.front()._candidate._config._name << " score " << evaluations.front().aggregateScore << '\n';
	return evaluations;
}

static std::string jsonEscape(const std::string& value)
{
	std::string escaped;
	escaped.reserve(value.size() + 8);
	for (const char c : value)
	{
		switch (c)
		{
		case '\\':
			escaped += "\\\\";
			break;
		case '"':
			escaped += "\\\"";
			break;
		case '\n':
			escaped += "\\n";
			break;
		case '\r':
			escaped += "\\r";
			break;
		case '\t':
			escaped += "\\t";
			break;
		default:
			escaped.push_back(c);
			break;
		}
	}
	return escaped;
}

static std::string safeFileStem(std::string value)
{
	for (char& c : value)
	{
		if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-')
			c = '_';
	}
	return value.empty() ? "auto_conditions" : value;
}

static void writeSchemaCopy(const Experiments::SchemaCandidate& candidate, const std::filesystem::path& path)
{
	if (path.has_parent_path())
		std::filesystem::create_directories(path.parent_path());
	std::ofstream output(path);
	if (!output.is_open())
		throw std::runtime_error("Unable to write auto-condition schema: " + path.string());
	output << schemaConfigToJson(candidate._config);
}

static void writeMeasuredSelectorArtifact(
	const std::string& path,
	const Experiments::SchemaSearchRecord& selected,
	const std::vector<Experiments::SchemaSearchRecord>& records,
	const std::string& sourceCsv)
{
	if (path.empty())
		return;

	createParentDirectory(path);
	std::ofstream output(path);
	if (!output.is_open())
		throw std::runtime_error("Unable to write auto-condition selector: " + path);

	output << std::fixed << std::setprecision(6);
	output << "{\n";
	output << "  \"model_type\": \"measured_best_schema\",\n";
	output << "  \"dataset_name\": \"" << jsonEscape(selected._datasetName) << "\",\n";
	output << "  \"dataset_path\": \"" << jsonEscape(selected._datasetSource) << "\",\n";
	output << "  \"workload_name\": \"" << jsonEscape(selected._workloadName) << "\",\n";
	output << "  \"workload_profile_path\": \"" << jsonEscape(selected._workloadName) << "\",\n";
	output << "  \"selected_schema\": {\n";
	output << "    \"name\": \"" << jsonEscape(selected._schemaName) << "\",\n";
	output << "    \"path\": \"" << jsonEscape(selected._schemaPath) << "\",\n";
	output << "    \"score\": " << selected._score << ",\n";
	output << "    \"score_mode\": \"" << jsonEscape(selected._scoreMode) << "\",\n";
	output << "    \"score_stage\": \"" << jsonEscape(selected._scoreStage) << "\",\n";
	output << "    \"score_is_final_latency\": " << (selected._scoreIsFinalLatency ? "true" : "false") << ",\n";
	output << "    \"avg_latency_ms\": " << selected._queryMetrics._averageLatencyMs << ",\n";
	output << "    \"build_time_ms\": " << selected._buildMetrics._buildTimeMs << ",\n";
	output << "    \"memory_estimate_bytes\": " << selected._buildMetrics._memoryEstimateBytes << "\n";
	output << "  },\n";
	output << "  \"candidate_scores\": [\n";
	bool first = true;
	for (const Experiments::SchemaSearchRecord& record : records)
	{
		if (record._datasetName != selected._datasetName || record._workloadName != selected._workloadName)
			continue;
		output << (first ? "" : ",\n");
		first = false;
		output << "    {\n";
		output << "      \"name\": \"" << jsonEscape(record._schemaName) << "\",\n";
		output << "      \"path\": \"" << jsonEscape(record._schemaPath) << "\",\n";
		output << "      \"score\": " << record._score << ",\n";
		output << "      \"score_mode\": \"" << jsonEscape(record._scoreMode) << "\",\n";
		output << "      \"score_stage\": \"" << jsonEscape(record._scoreStage) << "\",\n";
		output << "      \"score_is_final_latency\": " << (record._scoreIsFinalLatency ? "true" : "false") << ",\n";
		output << "      \"avg_latency_ms\": " << record._queryMetrics._averageLatencyMs << ",\n";
		output << "      \"build_time_ms\": " << record._buildMetrics._buildTimeMs << ",\n";
		output << "      \"memory_estimate_bytes\": " << record._buildMetrics._memoryEstimateBytes << "\n";
		output << "    }";
	}
	output << "\n  ],\n";
	output << "  \"source_csv\": \"" << jsonEscape(sourceCsv) << "\",\n";
	output << "  \"note\": \"Auto-condition measured selector: overfit to this point cloud and workload by staged conditional schema tuning.\"\n";
	output << "}\n";
}

static void writeAutoConditionArtifacts(
	std::vector<Experiments::SchemaSearchRecord>& records,
	const std::vector<EvaluatedCandidate>& finalEvaluations,
	const Experiments::SchemaSearchOptions& options)
{
	if (records.empty() || finalEvaluations.empty())
		return;

	std::unordered_map<std::string, const Experiments::SchemaCandidate*> candidatesByKey;
	for (const EvaluatedCandidate& evaluation : finalEvaluations)
		candidatesByKey[candidateKey(evaluation._candidate._config._name, evaluation._candidate._path)] = &evaluation._candidate;

	std::vector<Experiments::SchemaSearchRecord> bestRecords = Experiments::selectBestRecords(records);
	for (Experiments::SchemaSearchRecord& best : bestRecords)
	{
		const std::string originalPath = best._schemaPath;
		const auto found = candidatesByKey.find(candidateKey(best._schemaName, originalPath));
		if (found == candidatesByKey.end())
			continue;

		const std::filesystem::path outputPath =
			std::filesystem::path(options._autoConditions._outputDirectory) /
			(safeFileStem(best._datasetName + "_" + best._workloadName) + "_best_schema.json");
		writeSchemaCopy(*found->second, outputPath);

		for (Experiments::SchemaSearchRecord& record : records)
		{
			if (record._datasetName == best._datasetName &&
				record._workloadName == best._workloadName &&
				record._schemaName == best._schemaName &&
				record._schemaPath == originalPath)
			{
				record._schemaPath = outputPath.string();
			}
		}
		best._schemaPath = outputPath.string();
		std::cout << "  auto-condition schema: " << outputPath.string() << '\n';
	}

	bestRecords = Experiments::selectBestRecords(records);
	if (!bestRecords.empty())
	{
		writeMeasuredSelectorArtifact(options._autoConditions._selectorOutputPath, bestRecords.front(), records, options._csvPath);
		if (!options._autoConditions._selectorOutputPath.empty())
			std::cout << "  auto-condition selector: " << options._autoConditions._selectorOutputPath << '\n';
	}
}

static std::vector<Experiments::SchemaSearchRecord> runAutoConditionSearch(
	const Experiments::SchemaSearchOptions& options,
	const std::vector<SearchDataset>& datasets,
	const std::vector<Experiments::WorkloadProfile>& workloads,
	const std::vector<Experiments::SchemaCandidate>& configuredSchemas)
{
	const Experiments::AutoConditionOptions& autoOptions = options._autoConditions;
	if (datasets.empty())
		return {};

	for (const Experiments::WorkloadProfile& workload : workloads)
	{
		if (workload._knnWeight > 0.0)
			std::cout << "  warning: auto-condition tuning is optimized for range/radius workloads; KNN in workload '" << workload._name << "' may make search expensive\n";
	}

	Experiments::ConditionDomain domain = Experiments::estimateConditionDomain(datasets.front()._cloud, autoOptions._proxyPointCap);
	if (useCudaEvaluator(options))
		restrictConditionDomainToGpuSafe(domain);
	std::cout << "  auto-conditions: domain from " << domain._samplePoints << " sampled points, "
		<< domain._sketchNodes << " sketch nodes\n";
	std::cout << "    point thresholds: " << domain._pointThresholds.size()
		<< ", density thresholds: " << domain._densityThresholds.size()
		<< ", height thresholds: " << domain._heightRatioThresholds.size() << '\n';

	Experiments::SchemaGenerationOptions generation = options._generation;
	const size_t proxyBudget = std::max<size_t>(1, autoOptions._proxyCandidateCount);
	generation._conditionalLevels = true;
	generation._conditionalProbability = std::max(generation._conditionalProbability, 0.75);

	std::vector<Experiments::SchemaCandidate> baselines;
	for (const Experiments::SchemaCandidate& candidate : configuredSchemas)
	{
		if (candidate._isBaseline)
			baselines.push_back(candidate);
	}

	std::vector<Experiments::SchemaCandidate> candidates;
	if (options._deepNestedSearch)
	{
		runDeepNestedDiagnostics(options, datasets, workloads, baselines);
		generation._count = proxyBudget;
		generation._minBlocks = std::max<size_t>(2, generation._minBlocks);
		generation._maxBlocks = std::max(generation._maxBlocks, generation._minBlocks);
		std::vector<Experiments::SchemaCandidate> generated = generateDeepNestedCandidates(generation, domain, workloads);
		candidates.insert(candidates.end(), std::make_move_iterator(generated.begin()), std::make_move_iterator(generated.end()));
	}
	else
	{
		generation._count = configuredSchemas.size() >= proxyBudget
			? size_t(0)
			: proxyBudget - configuredSchemas.size();
		candidates = configuredSchemas;
		if (generation._count > 0)
		{
			std::vector<Experiments::SchemaCandidate> generated = Experiments::generateSchemaCandidates(generation, &domain);
			candidates.insert(candidates.end(), std::make_move_iterator(generated.begin()), std::make_move_iterator(generated.end()));
		}
	}

	std::unordered_set<std::string> seen;
	candidates = uniqueCandidates(candidates, seen);
	if (candidates.empty())
		throw std::runtime_error("Auto-condition search has no candidate schemas.");

	Experiments::SchemaSearchOptions discoveryOptions = options;
	if (options._deepNestedSearch)
		discoveryOptions._evaluator = "cpu";
	const bool cudaEvaluator = useCudaEvaluator(discoveryOptions);
	const std::vector<SearchDataset> proxyDatasets = makeProxyDatasets(datasets, autoOptions._proxyPointCap);
	const std::vector<Experiments::WorkloadProfile> proxyWorkloads = withQueryCount(workloads, autoOptions._proxyQueryCount);
	const std::vector<DatasetContext> proxyContexts = makeDatasetContexts(proxyDatasets, proxyWorkloads, cudaEvaluator);
	// Proxy stage ranks by a deterministic visit-count surrogate (cheap, noise-free); shortlist and confirmation stages fall back to latency.
	Experiments::SchemaSearchOptions proxyOptions = discoveryOptions;
	proxyOptions._weights._useVisitProxy = true;
	proxyOptions._scoreStage = "proxy";
	proxyOptions._scoreIsFinalLatency = false;
	if (proxyOptions._weights._visitProxyAlpha <= 0.0)
		proxyOptions._weights._visitProxyAlpha = 0.1;
	// A small lambdaBuild keeps build time honest in the proxy ranking, so schemas with low visit counts but catastrophic build cost can't sweep the proxy cloud and stall the shortlist.
	if (proxyOptions._weights._lambdaBuild <= 0.0)
		proxyOptions._weights._lambdaBuild = 1.0;
	std::cout << "  auto-conditions: proxy stage uses visit-count surrogate (alpha="
		<< proxyOptions._weights._visitProxyAlpha
		<< ", lambdaBuild=" << proxyOptions._weights._lambdaBuild << ")\n";
	std::vector<EvaluatedCandidate> proxyEvaluations = evaluateAutoConditionStage(
		"proxy",
		candidates,
		proxyContexts,
		proxyWorkloads,
		proxyOptions);

	std::vector<Experiments::SchemaCandidate> shortlist = options._deepNestedSearch
		? topDeepNestedCandidates(proxyEvaluations, autoOptions._finalTopK, false)
		: topSearchCandidatesWithBaselineControls(proxyEvaluations, autoOptions._finalTopK, baselines);
	if (options._deepNestedSearch)
		appendBaselineControls(shortlist, baselines);
	const std::vector<Experiments::WorkloadProfile> shortWorkloads = withQueryCount(workloads, options._deepNestedSearch ? 32 : 16);
	const std::vector<DatasetContext> shortContexts = makeDatasetContexts(datasets, shortWorkloads, cudaEvaluator);
	Experiments::SchemaSearchOptions shortOptions = discoveryOptions;
	shortOptions._scoreStage = "shortlist";
	shortOptions._scoreIsFinalLatency = false;
	std::vector<EvaluatedCandidate> shortEvaluations = evaluateAutoConditionStage(
		"shortlist",
		shortlist,
		shortContexts,
		shortWorkloads,
		shortOptions);

	std::vector<Experiments::SchemaCandidate> confirmation = options._deepNestedSearch
		? topDeepNestedCandidates(shortEvaluations, autoOptions._confirmationTopK, true)
		: topSearchCandidatesWithBaselineControls(shortEvaluations, autoOptions._confirmationTopK, baselines);
	const std::vector<DatasetContext> confirmationContexts = makeDatasetContexts(datasets, workloads, cudaEvaluator);
	Experiments::SchemaSearchOptions confirmationOptions = discoveryOptions;
	confirmationOptions._scoreStage = "confirmation";
	confirmationOptions._scoreIsFinalLatency = !confirmationOptions._weights._useVisitProxy;
	std::vector<EvaluatedCandidate> finalEvaluations = evaluateAutoConditionStage(
		"confirmation",
		confirmation,
		confirmationContexts,
		workloads,
		confirmationOptions);

	std::vector<Experiments::SchemaSearchRecord> records;
	for (const EvaluatedCandidate& evaluation : finalEvaluations)
	{
		appendRecords(records, evaluation);
		for (const Experiments::SchemaSearchRecord& record : evaluation._records)
			emitProgress(options, record);
	}

	std::vector<EvaluatedCandidate> artifactEvaluations = finalEvaluations;
	if (options._deepNestedSearch)
	{
		std::vector<Experiments::SchemaCandidate> robustCandidates = topDeepNestedCandidates(finalEvaluations, 6, false);
		if (const std::optional<Experiments::SchemaCandidate> baseline = bestBaselineCandidate(finalEvaluations))
		{
			std::unordered_set<std::string> picked;
			for (const Experiments::SchemaCandidate& candidate : robustCandidates)
				picked.insert(candidate._path.empty() ? candidate._config._name : candidate._path);
			appendUniqueCandidate(robustCandidates, picked, baseline.value());
		}

		const std::vector<Experiments::WorkloadProfile> robustWorkloads = makeRobustnessWorkloads(workloads, 256, 3);
		const std::vector<DatasetContext> robustContexts = makeDatasetContexts(datasets, robustWorkloads, false);
		std::vector<EvaluatedCandidate> robustEvaluations = evaluateAutoConditionStage(
			"robustness",
			robustCandidates,
			robustContexts,
			robustWorkloads,
			discoveryOptions);
		for (const EvaluatedCandidate& evaluation : robustEvaluations)
		{
			appendRecords(records, evaluation);
			for (const Experiments::SchemaSearchRecord& record : evaluation._records)
				emitProgress(options, record);
		}
		if (!robustEvaluations.empty())
			artifactEvaluations = robustEvaluations;

		runDeepCudaConfirmation(options, datasets, workloads, artifactEvaluations);
	}

	writeAutoConditionArtifacts(records, artifactEvaluations, options);
	return records;
}

// Runs a batch through the rung schedule (each rung has its own WorkloadProfile and ScoreWeights) and returns the final-rung evaluations best-first; finalRecordsOut gets only the highest-fidelity rung's records.
static std::vector<EvaluatedCandidate> runRungSchedule(
	const std::string& batchLabel,
	const std::vector<Experiments::SchemaCandidate>& inputBatch,
	const std::vector<const SearchDataset*>& datasetPtrs,
	const std::vector<Experiments::WorkloadProfile>& baseWorkloads,
	const Experiments::SchemaSearchOptions& options,
	std::vector<Experiments::SchemaSearchRecord>* finalRecordsOut)
{
	const Experiments::RungSchedule& schedule = options._evolution._rungSchedule;
	if (schedule._rungs.empty() || inputBatch.empty())
		return {};

	const bool cudaEvaluator = useCudaEvaluator(options);
	std::vector<Experiments::SchemaCandidate> current = inputBatch;
	std::vector<EvaluatedCandidate> evaluations;

	for (size_t r = 0; r < schedule._rungs.size(); ++r)
	{
		const Experiments::RungSpec& rung = schedule._rungs[r];

		const std::vector<Experiments::WorkloadProfile> rungWorkloads = withQueryCount(baseWorkloads, rung._queryCountOverride);
		const std::vector<DatasetContext> rungContexts = makeDatasetContextsFromPointers(datasetPtrs, rungWorkloads, cudaEvaluator);

		Experiments::SchemaSearchOptions rungOptions = options;
		rungOptions._weights._useVisitProxy = rung._useVisitProxy;
		if (rung._useVisitProxy && rung._visitProxyAlpha > 0.0)
			rungOptions._weights._visitProxyAlpha = rung._visitProxyAlpha;
		const bool isFinalRung = (r + 1 == schedule._rungs.size());
		rungOptions._scoreStage = rung._name.empty()
			? std::string("rung_") + std::to_string(r)
			: rung._name;
		rungOptions._scoreIsFinalLatency = isFinalRung && !rungOptions._weights._useVisitProxy;

		std::ostringstream label;
		label << batchLabel << " " << rung._name << " ("
			<< current.size() << " cand";
		if (rung._queryCountOverride > 0)
			label << ", " << rung._queryCountOverride << "q";
		label << (rung._useVisitProxy ? ", visit-proxy" : ", latency");
		label << ")";

		evaluations = evaluateAutoConditionStage(label.str(), current, rungContexts, rungWorkloads, rungOptions);

		if (isFinalRung && finalRecordsOut != nullptr)
		{
			for (const EvaluatedCandidate& evaluation : evaluations)
			{
				for (const Experiments::SchemaSearchRecord& record : evaluation._records)
				{
					finalRecordsOut->push_back(record);
					emitProgress(options, record);
				}
			}
		}

		if (!isFinalRung)
		{
			const size_t keep = (rung._advanceTopK == 0)
				? evaluations.size()
				: std::min(rung._advanceTopK, evaluations.size());

			// Diverse-by-primary advancement: each primary-block type keeps at least one survivor before slots fill by score, so proxy bias can't starve the latency rung of a whole family.
			std::vector<size_t> advanceOrder;
			advanceOrder.reserve(keep);
			std::unordered_set<std::string> primarySeen;
			std::vector<size_t> deferred;
			deferred.reserve(evaluations.size());
			for (size_t i = 0; i < evaluations.size() && advanceOrder.size() < keep; ++i)
			{
				const std::string primary = primaryBlockKey(evaluations[i]._candidate._config);
				if (primarySeen.insert(primary).second)
					advanceOrder.push_back(i);
				else
					deferred.push_back(i);
			}
			for (const size_t i : deferred)
			{
				if (advanceOrder.size() >= keep)
					break;
				advanceOrder.push_back(i);
			}

			current.clear();
			current.reserve(advanceOrder.size());
			std::vector<EvaluatedCandidate> trimmed;
			trimmed.reserve(advanceOrder.size());
			for (const size_t i : advanceOrder)
			{
				current.push_back(evaluations[i]._candidate);
				trimmed.push_back(std::move(evaluations[i]));
			}
			evaluations = std::move(trimmed);

			if (primarySeen.size() > 1)
				std::cout << "      rung '" << rung._name << "' advancing " << current.size()
					<< " candidate(s), " << primarySeen.size()
					<< " distinct primary-block type(s)\n";
		}
	}

	return evaluations;
}

static std::vector<Experiments::SchemaSearchRecord> runEvolutionarySchemaSearch(
	const Experiments::SchemaSearchOptions& options,
	const std::vector<DatasetContext>& datasets,
	const std::vector<Experiments::WorkloadProfile>& workloads,
	const std::vector<Experiments::SchemaCandidate>& initialCandidates,
	const Experiments::ConditionDomain* conditionDomain = nullptr)
{
	const Experiments::EvolutionOptions& evolution = options._evolution;
	const size_t populationSize = std::max<size_t>(1, evolution._populationSize);
	const size_t eliteCount = std::max<size_t>(1, evolution._eliteCount);
	const double randomFraction = std::clamp(evolution._randomImmigrationRate, 0.0, 1.0);

	std::mt19937 rng(evolution._seed);
	std::unordered_set<std::string> seenSignatures;
	std::vector<EvaluatedCandidate> archive;
	std::vector<Experiments::SchemaSearchRecord> records;
	CudaIndexCache cudaCache;

	// Diversity audit: count distinct topology hashes across evaluated candidates, reported at end of run to show whether the GA explored varied shapes or one corridor.
	std::map<std::string, size_t> topologyCounts;
	size_t totalEvaluatedCandidates = 0;

	// Picks the ordering helper per options._useNsga2Ranking: NSGA-II by front then crowding distance, else the scalar single-score path.
	auto rankArchive = [&]() {
		if (evolution._useNsga2Ranking)
			nsga2RankAndSort(archive);
		else
			sortEvaluations(archive);
	};

	std::vector<Experiments::SchemaCandidate> batch = uniqueCandidates(initialCandidates, seenSignatures);
	if (batch.empty())
		throw std::runtime_error("Evolutionary schema optimizer has no initial population.");

	std::cout << "  optimizer: evolutionary mutation search\n";
	std::cout << "    generations: " << evolution._generations << '\n';
	std::cout << "    population per generation: " << populationSize << '\n';
	std::cout << "    elites: " << eliteCount << '\n';
	std::cout << "    mutation rate: " << evolution._mutationRate << '\n';
	std::cout << "    random immigration: " << randomFraction << '\n';
	if (evolution._repairMutations)
	{
		std::cout << "    repair mutations: top-" << std::max<size_t>(1, evolution._repairTopK)
			<< ", " << std::max<size_t>(1, evolution._repairPerCandidate)
			<< " candidate(s)/parent\n";
	}

	const bool useRungSchedule = !evolution._rungSchedule._rungs.empty();
	std::vector<const SearchDataset*> datasetPtrs;
	Experiments::SurrogateAcquisition surrogate = Experiments::loadSurrogateAcquisition(evolution._rungSchedule._surrogateModelPath);
	if (surrogate._active)
	{
		std::cout << "    surrogate acquisition: " << surrogate._modelPath
			<< " (pool=" << evolution._rungSchedule._surrogateCandidatePool
			<< ", proposals/gen=" << evolution._rungSchedule._surrogateProposalsPerStep << ")\n";
	}
	if (useRungSchedule)
	{
		datasetPtrs.reserve(datasets.size());
		for (const DatasetContext& context : datasets)
			datasetPtrs.push_back(context._dataset);

		std::cout << "    rung schedule: " << evolution._rungSchedule._rungs.size() << " rungs\n";
		for (size_t r = 0; r < evolution._rungSchedule._rungs.size(); ++r)
		{
			const Experiments::RungSpec& rung = evolution._rungSchedule._rungs[r];
			std::cout << "      [" << r << "] " << rung._name
				<< (rung._useVisitProxy ? " (visit-proxy)" : " (latency)");
			if (rung._queryCountOverride > 0)
				std::cout << " queries=" << rung._queryCountOverride;
			if (rung._advanceTopK > 0 && r + 1 < evolution._rungSchedule._rungs.size())
				std::cout << " advance top-" << rung._advanceTopK;
			std::cout << '\n';
		}
	}

	const size_t parallelWorkers = (!useCudaEvaluator(options) && options._parallelDispatch > 1)
		? options._parallelDispatch
		: 1;
	if (parallelWorkers > 1)
		std::cout << "    parallel candidate dispatch: " << parallelWorkers << " CPU workers\n";

	auto evaluateBatch = [&](const std::vector<Experiments::SchemaCandidate>& candidates, const std::string& label) {
		if (candidates.empty())
			return;

		// Topology audit counts every candidate reaching here, so the diversity report captures attempted search breadth rather than only what the front rewarded.
		for (const Experiments::SchemaCandidate& candidate : candidates)
		{
			++topologyCounts[schemaTopologyKey(candidate._config)];
			++totalEvaluatedCandidates;
		}

		if (useRungSchedule)
		{
			std::vector<EvaluatedCandidate> finalEvaluations = runRungSchedule(
				label, candidates, datasetPtrs, workloads, options, &records);
			for (size_t i = 0; i < finalEvaluations.size(); ++i)
			{
				std::cout << "      " << label << " final [" << (i + 1) << "/" << finalEvaluations.size() << "] "
					<< finalEvaluations[i]._candidate._config._name
					<< "  aggregate score " << finalEvaluations[i].aggregateScore << '\n';
				archive.push_back(std::move(finalEvaluations[i]));
			}
			rankArchive();
			if (!archive.empty())
				std::cout << "      best so far: " << archive.front()._candidate._config._name
					<< " score " << archive.front().aggregateScore << '\n';
			return;
		}

		std::vector<EvaluatedCandidate> evaluations(candidates.size());
		if (parallelWorkers > 1)
		{
			// CPU-only parallel dispatch; the CUDA index cache is bypassed (per-builder build cache isn't thread-safe), while the internally-synchronised score cache is shared across workers.
			std::atomic<size_t> nextIndex{0};
			std::vector<std::thread> workers;
			workers.reserve(parallelWorkers);
			for (size_t w = 0; w < parallelWorkers; ++w)
			{
				workers.emplace_back([&]() {
					for (;;)
					{
						const size_t i = nextIndex.fetch_add(1);
						if (i >= candidates.size())
							return;
						evaluations[i] = evaluateCandidate(candidates[i], datasets, workloads, options, nullptr);
					}
				});
			}
			for (std::thread& worker : workers)
				worker.join();

			for (size_t i = 0; i < candidates.size(); ++i)
			{
				EvaluatedCandidate& evaluation = evaluations[i];
				std::cout << "      " << label << " [" << (i + 1) << "/" << candidates.size() << "] "
					<< candidates[i]._config._name << "  aggregate score " << evaluation.aggregateScore << '\n';
				appendRecords(records, evaluation);
				for (const Experiments::SchemaSearchRecord& record : evaluation._records)
					emitProgress(options, record);
				archive.push_back(std::move(evaluation));
			}
		}
		else
		{
			for (size_t i = 0; i < candidates.size(); ++i)
			{
				const Experiments::SchemaCandidate& candidate = candidates[i];
				std::cout << "      " << label << " [" << (i + 1) << "/" << candidates.size() << "] " << candidate._config._name << '\n';
				const auto candidateStart = std::chrono::steady_clock::now();
				EvaluatedCandidate evaluation = evaluateCandidate(candidate, datasets, workloads, options, &cudaCache);
				const auto candidateElapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - candidateStart).count();
				std::cout << "        aggregate score " << evaluation.aggregateScore;
				if (!evaluation._records.empty() && evaluation._records.front()._backend == "cuda")
					std::cout << " (" << cudaBuilderDisplayName(evaluation._records.front()._cudaBuilder) << ")";
				std::cout << " [" << candidateElapsed << " s]";
				std::cout << '\n';
				if (candidateElapsed > 30.0)
					std::cout << "        WARNING: candidate '" << candidate._config._name
						<< "' took " << candidateElapsed << " s. Raise --cuda-memory-budget-mb or restrict the schema generator if this repeats.\n";
				appendRecords(records, evaluation);
				for (const Experiments::SchemaSearchRecord& record : evaluation._records)
					emitProgress(options, record);
				archive.push_back(std::move(evaluation));
			}
		}
		rankArchive();
		if (!archive.empty())
			std::cout << "      best so far: " << archive.front()._candidate._config._name << " score " << archive.front().aggregateScore << '\n';
	};

	evaluateBatch(batch, "initial");

	for (size_t generation = 1; generation <= evolution._generations; ++generation)
	{
		rankArchive();
		const size_t currentEliteCount = std::min(eliteCount, archive.size());
		if (currentEliteCount == 0)
			break;

		std::vector<Experiments::SchemaCandidate> children;
		children.reserve(populationSize);

		size_t repairAccepted = 0;
		if (evolution._repairMutations && children.size() < populationSize)
		{
			const size_t repairParents = std::min({
				std::max<size_t>(1, evolution._repairTopK),
				currentEliteCount,
				archive.size()
			});
			const size_t repairsPerParent = std::max<size_t>(1, evolution._repairPerCandidate);
			for (size_t parentIndex = 0; parentIndex < repairParents && children.size() < populationSize; ++parentIndex)
			{
				const EvaluatedCandidate& parent = archive[parentIndex];
				std::vector<RepairSchemaMutation> repairs = generateRepairSchemaMutations(
					parent._candidate,
					parent._records,
					options._generation,
					conditionDomain,
					repairsPerParent,
					rng());
				for (RepairSchemaMutation& repair : repairs)
				{
					const std::string signature = schemaSignature(repair._schema);
					if (!seenSignatures.insert(signature).second)
						continue;

					const std::string prefix = "repair_g" + std::to_string(generation)
						+ "_i" + std::to_string(children.size())
						+ "_" + repair._reason;
					children.push_back(materializeGeneratedSchema(std::move(repair._schema), prefix, options._generation._outputDirectory));
					++repairAccepted;
					if (children.size() >= populationSize)
						break;
				}
			}
			if (repairAccepted > 0)
				std::cout << "      generation " << generation << " repairs accepted: " << repairAccepted << '\n';
		}

		const size_t remainingAfterRepairs = populationSize - children.size();
		const size_t randomCount = std::min(remainingAfterRepairs, static_cast<size_t>(std::round(static_cast<double>(populationSize) * randomFraction)));
		if (randomCount > 0)
		{
			Experiments::SchemaGenerationOptions randomOptions = options._generation;
			randomOptions._count = randomCount;
			randomOptions._seed = rng();
			std::vector<Experiments::SchemaCandidate> immigrants = Experiments::generateSchemaCandidates(randomOptions, conditionDomain);
			for (const Experiments::SchemaCandidate& immigrant : immigrants)
			{
				const std::string signature = schemaSignature(immigrant._config);
				if (!seenSignatures.insert(signature).second)
					continue;
				children.push_back(immigrant);
				if (children.size() >= populationSize)
					break;
			}
		}

		// Bayesian acquisition step: have the surrogate rank a fresh pool and inject its top-K predictions as extra children, cheaply decoupling progress from pure mutation noise.
		if (surrogate._active
			&& evolution._rungSchedule._surrogateCandidatePool > 0
			&& evolution._rungSchedule._surrogateProposalsPerStep > 0
			&& !datasets.empty()
			&& !workloads.empty()
			&& children.size() < populationSize)
		{
			const size_t budget = populationSize - children.size();
			const size_t proposalCount = std::min(evolution._rungSchedule._surrogateProposalsPerStep, budget);
			std::vector<Experiments::SchemaCandidate> proposals = Experiments::acquireSurrogateProposals(
				surrogate,
				options._generation,
				evolution._rungSchedule._surrogateCandidatePool,
				proposalCount,
				datasets.front()._dataset->_cloud,
				workloads.front(),
				rng());
			size_t accepted = 0;
			for (const Experiments::SchemaCandidate& proposal : proposals)
			{
				const std::string signature = schemaSignature(proposal._config);
				if (!seenSignatures.insert(signature).second)
					continue;
				children.push_back(proposal);
				++accepted;
				if (children.size() >= populationSize)
					break;
			}
			if (accepted > 0)
				std::cout << "      surrogate proposals accepted: " << accepted << '\n';
		}

		size_t attempts = 0;
		const size_t maxAttempts = std::max<size_t>(populationSize * 80, 512);
		const double crossoverProbability = std::clamp(evolution._crossoverRate, 0.0, 1.0);
		std::bernoulli_distribution useCrossover(crossoverProbability);
		size_t crossoverAccepted = 0;
		size_t mutationAccepted = 0;
		while (children.size() < populationSize && attempts++ < maxAttempts)
		{
			std::uniform_int_distribution<size_t> eliteDistribution(0, currentEliteCount - 1);
			SchemaConfig childSchema;
			bool wasCrossover = false;
			if (currentEliteCount >= 2 && useCrossover(rng))
			{
				// Champion-anchored crossover: archive[0] (the scalar champion) is always one parent, the other sampled from the elites, so every child inherits half the best-known schema.
				const size_t aIdx = 0;
				std::uniform_int_distribution<size_t> partnerDistribution(1, currentEliteCount - 1);
				const size_t bIdx = partnerDistribution(rng);
				childSchema = crossoverSchemaConfigs(
					archive[aIdx]._candidate._config,
					archive[bIdx]._candidate._config,
					rng,
					options._generation);
				wasCrossover = true;
			}
			else
			{
				const EvaluatedCandidate& parent = archive[eliteDistribution(rng)];
				childSchema = mutateSchemaConfig(parent._candidate._config, rng, options._generation, evolution, conditionDomain);
			}
			const std::string signature = schemaSignature(childSchema);
			if (!seenSignatures.insert(signature).second)
				continue;

			const std::string prefix = wasCrossover
				? std::string("xover_g") + std::to_string(generation) + "_i" + std::to_string(children.size())
				: std::string("evolved_g") + std::to_string(generation) + "_i" + std::to_string(children.size());
			children.push_back(materializeGeneratedSchema(childSchema, prefix, options._generation._outputDirectory));
			if (wasCrossover)
				++crossoverAccepted;
			else
				++mutationAccepted;
		}
		if (crossoverAccepted + mutationAccepted > 0)
			std::cout << "      generation " << generation
				<< " champion: " << archive[0]._candidate._config._name
				<< " (score " << archive[0].aggregateScore << ")\n"
				<< "      generation " << generation << " children: "
				<< crossoverAccepted << " crossover + "
				<< mutationAccepted << " mutation\n";

		if (children.empty())
		{
			std::cerr << "Warning: optimizer generation " << generation << " produced no unique children\n";
			break;
		}

		evaluateBatch(children, "generation " + std::to_string(generation));
	}

	rankArchive();
	if (!archive.empty())
		std::cout << "  optimizer best aggregate: " << archive.front()._candidate._config._name << " score " << archive.front().aggregateScore << '\n';

	// Diversity audit: distinct topology shapes vs. total candidates; a low ratio means the GA refined a few corridors instead of exploring the manifold.
	if (totalEvaluatedCandidates > 0)
	{
		std::cout << "  diversity audit: " << topologyCounts.size()
			<< " distinct topology shapes across " << totalEvaluatedCandidates
			<< " evaluated candidates"
			<< " (ratio " << std::fixed << std::setprecision(3)
			<< (static_cast<double>(topologyCounts.size()) / static_cast<double>(totalEvaluatedCandidates))
			<< ")\n";

		std::vector<std::pair<std::string, size_t>> sortedTopologies(topologyCounts.begin(), topologyCounts.end());
		std::sort(sortedTopologies.begin(), sortedTopologies.end(),
			[](const auto& a, const auto& b) { return a.second > b.second; });
		const size_t topReport = std::min<size_t>(8, sortedTopologies.size());
		std::cout << "    top topologies by evaluation count:\n";
		for (size_t i = 0; i < topReport; ++i)
		{
			std::cout << "      " << sortedTopologies[i].second << "x  "
				<< (sortedTopologies[i].first.empty() ? "<empty>" : sortedTopologies[i].first) << '\n';
		}

		// Front-0 diversity is the key signal: if every Pareto-front entry shares a topology, ranking/generation needs more pressure. Computed only when NSGA-II ran.
		if (evolution._useNsga2Ranking)
		{
			std::set<std::string> frontTopologies;
			for (const EvaluatedCandidate& evaluation : archive)
			{
				if (evaluation._paretoFront != 0)
					continue;
				frontTopologies.insert(schemaTopologyKey(evaluation._candidate._config));
			}
			if (!frontTopologies.empty())
				std::cout << "    Pareto-front-0 distinct topologies: " << frontTopologies.size() << '\n';
		}
	}

	// Threshold refinement: runs a (1+lambda)-ES on the top-K conditional candidates using the cheap visit-proxy, then re-measures each refined candidate at full fidelity.
	if (evolution._refineThresholds && !archive.empty() && !datasets.empty() && !workloads.empty())
	{
		const size_t topK = std::max<size_t>(1, evolution._refineThresholdsTopK);
		const size_t available = std::min(topK, archive.size());
		std::cout << "  threshold refinement: top-" << available
			<< ", budget " << evolution._refineThresholdsEvaluations << " evals/candidate, sigma0 "
			<< evolution._refineThresholdsSigma0 << '\n';

		Experiments::ConditionDomain domain = Experiments::estimateConditionDomain(datasets.front()._dataset->_cloud, options._autoConditions._proxyPointCap > 0 ? options._autoConditions._proxyPointCap : 262144);
		if (useCudaEvaluator(options))
			restrictConditionDomainToGpuSafe(domain);

		Experiments::ThresholdRefinementOptions refinerOptions;
		refinerOptions._enabled = true;
		refinerOptions._topK = available;
		refinerOptions._maxEvaluations = evolution._refineThresholdsEvaluations;
		refinerOptions._sigma0 = evolution._refineThresholdsSigma0;
		refinerOptions._seed = evolution._refineThresholdsSeed;
		refinerOptions._outputDirectory = options._generation._outputDirectory.empty()
			? std::string("results/refined_schemas")
			: options._generation._outputDirectory + "/refined";

		// Visit-proxy score function: scores against the first dataset with useVisitProxy forced on and a small query count, since refinement only needs to navigate the threshold landscape, not produce the final number.
		Experiments::SchemaSearchOptions proxyOptions = options;
		proxyOptions._weights._useVisitProxy = true;
		proxyOptions._scoreStage = "refine_proxy";
		proxyOptions._scoreIsFinalLatency = false;
		if (proxyOptions._weights._visitProxyAlpha <= 0.0)
			proxyOptions._weights._visitProxyAlpha = 0.1;

		const DatasetContext& primary = datasets.front();
		const Experiments::WorkloadProfile fullWorkload = workloads.front();
		Experiments::WorkloadProfile refinerWorkload = fullWorkload;
		// 8 queries matches the auto-conditions proxy stage default; the refiner just needs cheap deterministic ranking of neighboring threshold vectors.
		constexpr size_t kRefinerProxyQueryCount = 8;
		if (refinerWorkload._numQueries > kRefinerProxyQueryCount)
			refinerWorkload._numQueries = kRefinerProxyQueryCount;
		const Experiments::WorkloadFeatures primaryWorkloadFeatures = Experiments::extractWorkloadFeatures(refinerWorkload, proxyOptions._weights);

		// Separate PreparedWorkload for the trimmed query count so its cache fingerprint reflects the smaller workload and per-seed hits accumulate across iterations.
		const PreparedWorkload refinerPrepared = prepareWorkloadProfile(refinerWorkload, primary._dataset->_cloud, useCudaEvaluator(proxyOptions));

		std::cout << "  refinement workload: " << refinerWorkload._numQueries
			<< " queries (vs " << fullWorkload._numQueries << " full)\n";

		Experiments::ThresholdScoreFn scoreFn = [&](const Experiments::SchemaCandidate& trial) {
			Experiments::SchemaSearchRecord trialRecord = benchmarkSchemaCandidateCached(
				*primary._dataset,
				primary._features,
				refinerWorkload,
				primaryWorkloadFeatures,
				refinerPrepared,
				trial,
				proxyOptions,
				nullptr);
			return trialRecord._score;
		};

		std::vector<Experiments::SchemaCandidate> refinedSurvivors;
		refinedSurvivors.reserve(available);
		for (size_t i = 0; i < available; ++i)
		{
			const EvaluatedCandidate& source = archive[i];
			Experiments::ThresholdRefinementResult refinement = Experiments::refineSchemaThresholds(
				source._candidate, domain, refinerOptions, scoreFn);

			if (refinement._dimensions == 0)
			{
				std::cout << "    [" << (i + 1) << "/" << available << "] "
					<< source._candidate._config._name << ": no active threshold dimensions, skipping\n";
				continue;
			}

			std::cout << "    [" << (i + 1) << "/" << available << "] "
				<< source._candidate._config._name
				<< ": dims=" << refinement._dimensions
				<< " evals=" << refinement._evaluationsUsed
				<< " score " << refinement._initialScore << " -> " << refinement._refinedScore
				<< (refinement._refinedScore < refinement._initialScore ? " (improved)" : " (no improvement)")
				<< '\n';

			if (refinement._refinedScore < refinement._initialScore)
				refinedSurvivors.push_back(std::move(refinement._refinedCandidate));

			++refinerOptions._seed; // decorrelate the ES across survivors so they don't all walk in lockstep
		}

		if (!refinedSurvivors.empty())
		{
			std::cout << "  threshold refinement: measuring " << refinedSurvivors.size()
				<< " improved candidate(s) at full fidelity\n";
			evaluateBatch(refinedSurvivors, "refined");
		}
		else
		{
			std::cout << "  threshold refinement: no candidate improved on its parent\n";
		}
	}

	return records;
}

// Re-measures the top-K candidates per (dataset, workload) over multiple query seeds, storing seed-averaged mean + 95% bootstrap CI that the Pareto step prefers over the point estimate.
static const SearchDataset* findDatasetByName(const std::vector<SearchDataset>& datasets, const std::string& name)
{
	for (const SearchDataset& dataset : datasets)
	{
		if (dataset._name == name)
			return &dataset;
	}
	return nullptr;
}

static const Experiments::WorkloadProfile* findWorkloadByName(const std::vector<Experiments::WorkloadProfile>& workloads, const std::string& name)
{
	for (const Experiments::WorkloadProfile& workload : workloads)
	{
		if (workload._name == name)
			return &workload;
	}
	return nullptr;
}

static void runMultiSeedConfirmation(
	std::vector<Experiments::SchemaSearchRecord>& records,
	const std::vector<SearchDataset>& datasets,
	const std::vector<Experiments::WorkloadProfile>& workloads,
	const Experiments::SchemaSearchOptions& options)
{
	if (options._confirmSeeds < 2 || records.empty())
		return;

	const size_t topK = std::max<size_t>(1, options._confirmTopK);
	const bool cudaEvaluator = useCudaEvaluator(options);

	std::cout << "  multi-seed confirmation: top-" << topK
		<< " per (dataset, workload), " << options._confirmSeeds << " seeds each\n";

	std::map<std::pair<std::string, std::string>, std::vector<size_t>> groups;
	for (size_t i = 0; i < records.size(); ++i)
		groups[{ records[i]._datasetName, records[i]._workloadName }].push_back(i);

	for (auto& [key, indices] : groups)
	{
		const SearchDataset* dataset = findDatasetByName(datasets, key.first);
		const Experiments::WorkloadProfile* workload = findWorkloadByName(workloads, key.second);
		if (!dataset || !workload)
			continue;

		// Pick top-K by score within this group. Lower score wins.
		std::sort(indices.begin(), indices.end(), [&records](size_t a, size_t b) {
			return records[a]._score < records[b]._score;
		});
		if (indices.size() > topK)
			indices.resize(topK);

		const Experiments::PointCloudFeatures features = Experiments::extractPointCloudFeatures(dataset->_cloud);
		const Experiments::ScoreWeights weights = effectiveScoreWeights(*workload, options);
		const Experiments::WorkloadFeatures workloadFeatures = Experiments::extractWorkloadFeatures(*workload, weights);

		for (const size_t recordIndex : indices)
		{
			Experiments::SchemaSearchRecord& target = records[recordIndex];

			Experiments::SchemaCandidate candidate;
			candidate._name = target._schemaName;
			candidate._path = target._schemaPath;
			candidate._isBaseline = target._isBaseline;
			try
			{
				candidate._config = Config::loadSchemaConfig(target._schemaPath);
			}
			catch (const std::exception& exception)
			{
				std::cerr << "    multi-seed confirmation skipped '" << target._schemaName
					<< "' (" << exception.what() << ")\n";
				continue;
			}

			std::vector<double> latencies;
			std::vector<double> p95s;
			std::vector<double> gpuBuilds;
			latencies.reserve(options._confirmSeeds);
			p95s.reserve(options._confirmSeeds);
			gpuBuilds.reserve(options._confirmSeeds);

			for (size_t s = 0; s < options._confirmSeeds; ++s)
			{
				Experiments::WorkloadProfile seededWorkload = *workload;
				// 7919 is a prime offset that decorrelates seeded sub-runs even as the base querySeed is bumped between invocations.
				seededWorkload._querySeed = workload->_querySeed + static_cast<uint32_t>(7919u * (s + 1));
				const PreparedWorkload preparedWorkload = prepareWorkloadProfile(
					seededWorkload, dataset->_cloud, cudaEvaluator);

				Experiments::SchemaSearchRecord trialRecord = benchmarkSchemaCandidateCached(
					*dataset,
					features,
					seededWorkload,
					workloadFeatures,
					preparedWorkload,
					candidate,
					options,
					nullptr);

				latencies.push_back(trialRecord._queryMetrics._averageLatencyMs);
				p95s.push_back(trialRecord._queryMetrics._p95LatencyMs);
				gpuBuilds.push_back(trialRecord._gpuBuildMs);
			}

			const auto [latMean, latLo, latHi] = Experiments::bootstrapMeanCI(latencies);
			const auto [p95Mean, p95Lo, p95Hi] = Experiments::bootstrapMeanCI(p95s);
			const auto [bldMean, bldLo, bldHi] = Experiments::bootstrapMeanCI(gpuBuilds);

			target._confirmSeedsUsed = options._confirmSeeds;
			target._latencyMean = latMean;
			target._latencyCiLow = latLo;
			target._latencyCiHigh = latHi;
			target._p95LatencyMean = p95Mean;
			target._p95LatencyCiLow = p95Lo;
			target._p95LatencyCiHigh = p95Hi;
			target._gpuBuildMean = bldMean;
			target._gpuBuildCiLow = bldLo;
			target._gpuBuildCiHigh = bldHi;

			std::cout << "    confirm '" << target._schemaName << "' [" << key.first << "/" << key.second
				<< "]: latency " << latMean << " ms (95% CI " << latLo << "-" << latHi
				<< "), gpu build " << bldMean << " ms\n";
		}
	}
}

Experiments::EvaluatorResolution Experiments::resolveSchemaSearchEvaluator(
	const std::string& requestedEvaluator,
	bool cudaAvailable,
	const std::string& cudaError)
{
	const std::string evaluator = lowerCopy(requestedEvaluator);
	EvaluatorResolution resolution;
	resolution._requestedCuda = evaluator == "cuda" || evaluator == "gpu";
	if (!resolution._requestedCuda)
	{
		resolution._evaluator = "cpu";
		return resolution;
	}

	if (cudaAvailable)
	{
		resolution._evaluator = "cuda";
		resolution._usingCuda = true;
		return resolution;
	}

	resolution._evaluator = "cpu";
	resolution._fellBackToCpu = true;
	resolution._warning = "CUDA evaluator requested but unavailable";
	if (!cudaError.empty())
		resolution._warning += ": " + cudaError;
	resolution._warning += "; falling back to CPU.";
	return resolution;
}

std::string Experiments::resolvePrimitiveProfile(const std::string& requestedProfile, bool cudaEvaluator)
{
	return primitiveProfileName(parsePrimitiveProfileInternal(requestedProfile, cudaEvaluator));
}

static void pushAnisotropy(std::vector<double>& anisotropyValues, double x, double y, double z)
{
	const double minE = std::min({ x, y, z });
	const double maxE = std::max({ x, y, z });
	if (maxE > 1.0e-9)
		addUniqueDouble(anisotropyValues, 1.0 - (minE / maxE));
}

Experiments::ConditionDomain Experiments::estimateConditionDomain(const PointCloud& cloud, size_t maxSamplePoints)
{
	ConditionDomain domain;
	domain._samplePoints = std::min(cloud.size(), std::max<size_t>(1, maxSamplePoints));
	domain._estimatedFromCloud = !cloud.empty();
	if (cloud.empty())
	{
		domain._pointThresholds = fallbackPointThresholds();
		domain._heightRatioThresholds = fallbackHeightRatioThresholds();
		return domain;
	}

	struct SketchCell
	{
		size_t _count = 0;
	};

	std::vector<size_t> counts;
	std::vector<double> densities;
	std::vector<double> heightRatios;
	std::vector<double> extentXValues;
	std::vector<double> extentYValues;
	std::vector<double> extentZValues;
	// Anisotropy samples (1 - shortExtent / longExtent) per sketch cell; the root counts too so the quantile estimator has a sample even on uniform clouds.
	std::vector<double> anisotropyValues;

	const glm::vec3 rootExtent = glm::max(cloud.bounds().size(), glm::vec3(0.0f));
	const double horizontalExtent = std::max({ static_cast<double>(rootExtent.x), static_cast<double>(rootExtent.y), 1.0e-9 });
	const double rootHeightRatio = static_cast<double>(rootExtent.z) / horizontalExtent;
	const double rootVolume = static_cast<double>(rootExtent.x) * static_cast<double>(rootExtent.y) * static_cast<double>(rootExtent.z);
	const double rootDensity = rootVolume > 1.0e-9
		? static_cast<double>(cloud.size()) / rootVolume
		: 0.0;

	counts.push_back(cloud.size());
	addUniqueDouble(densities, rootDensity);
	addUniqueDouble(heightRatios, rootHeightRatio);
	addUniqueDouble(extentXValues, rootExtent.x);
	addUniqueDouble(extentYValues, rootExtent.y);
	addUniqueDouble(extentZValues, rootExtent.z);
	pushAnisotropy(anisotropyValues, rootExtent.x, rootExtent.y, rootExtent.z);

	const size_t sampleCount = domain._samplePoints;
	auto sampleIndexAt = [&](size_t sampleIndex) {
		if (sampleCount <= 1 || cloud.size() <= 1)
			return size_t(0);
		return std::min(cloud.size() - 1, (sampleIndex * (cloud.size() - 1)) / (sampleCount - 1));
	};

	auto addGridSketch = [&](size_t divisions) {
		if (divisions == 0)
			return;

		const size_t cellCount = divisions * divisions * divisions;
		std::vector<SketchCell> cells(cellCount);
		const glm::vec3 minPoint = cloud.bounds().min();
		const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(1.0e-6f));

		for (size_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
		{
			const PointPrimitive& point = cloud.points()[sampleIndexAt(sampleIndex)];
			glm::uvec3 cell(0);
			for (glm::uint axis = 0; axis < 3; ++axis)
			{
				const float normalized = (point.position[axis] - minPoint[axis]) / range[axis];
				const size_t coordinate = static_cast<size_t>(std::floor(std::clamp(normalized, 0.0f, 0.999999f) * static_cast<float>(divisions)));
				cell[axis] = static_cast<glm::uint>(std::min(coordinate, divisions - 1));
			}

			const size_t index = (static_cast<size_t>(cell.x) * divisions + static_cast<size_t>(cell.y)) * divisions + static_cast<size_t>(cell.z);
			++cells[index]._count;
		}

		const glm::dvec3 cellExtent(
			static_cast<double>(rootExtent.x) / static_cast<double>(divisions),
			static_cast<double>(rootExtent.y) / static_cast<double>(divisions),
			static_cast<double>(rootExtent.z) / static_cast<double>(divisions));
		const double cellHorizontalExtent = std::max({ cellExtent.x, cellExtent.y, 1.0e-9 });
		const double cellHeightRatio = cellExtent.z / cellHorizontalExtent;
		const double cellVolume = cellExtent.x * cellExtent.y * cellExtent.z;
		// All cells at this division share the same aspect ratio, so one sample is sufficient.
		pushAnisotropy(anisotropyValues, cellExtent.x, cellExtent.y, cellExtent.z);
		const double sampleScale = sampleCount > 0 ? static_cast<double>(cloud.size()) / static_cast<double>(sampleCount) : 1.0;

		for (const SketchCell& cell : cells)
		{
			if (cell._count == 0)
				continue;

			const size_t estimatedCount = std::max<size_t>(1, static_cast<size_t>(std::round(static_cast<double>(cell._count) * sampleScale)));
			counts.push_back(estimatedCount);
			addUniqueDouble(densities, cellVolume > 1.0e-9 ? static_cast<double>(estimatedCount) / cellVolume : 0.0);
			addUniqueDouble(heightRatios, cellHeightRatio);
			addUniqueDouble(extentXValues, cellExtent.x);
			addUniqueDouble(extentYValues, cellExtent.y);
			addUniqueDouble(extentZValues, cellExtent.z);
			++domain._sketchNodes;
		}
	};

	addGridSketch(2);
	addGridSketch(4);
	++domain._sketchNodes;

	static const std::array<double, 5> quantiles = { 0.10, 0.25, 0.50, 0.75, 0.90 };
	for (const double quantile : quantiles)
	{
		addPointThresholdFamily(domain._pointThresholds, quantileValue(counts, quantile));
		addUniqueDouble(domain._densityThresholds, quantileValue(densities, quantile));
		addUniqueDouble(domain._heightRatioThresholds, quantileValue(heightRatios, quantile));
		addUniqueDouble(domain._extentXThresholds, quantileValue(extentXValues, quantile));
		addUniqueDouble(domain._extentYThresholds, quantileValue(extentYValues, quantile));
		addUniqueDouble(domain._extentZThresholds, quantileValue(extentZValues, quantile));
		if (!anisotropyValues.empty())
			addUniqueDouble(domain._anisotropyThresholds, std::clamp(quantileValue(anisotropyValues, quantile), 0.0, 1.0));
	}
	// Fallback anisotropy gates spanning near-cubic to highly elongated, used when the sketch yielded too few samples.
	if (domain._anisotropyThresholds.size() < 3)
	{
		for (const double fallback : { 0.1, 0.25, 0.4, 0.55, 0.7, 0.85 })
			addUniqueDouble(domain._anisotropyThresholds, fallback);
	}
	// Occupancy-entropy thresholds: fixed spread over the normalized [0, 1] range (near 0 fires on clustered nodes, near 1 on uniform ones); no per-cloud anchoring since entropy is already normalized by ln(64).
	for (const double fallback : { 0.15, 0.30, 0.45, 0.60, 0.75, 0.90 })
		addUniqueDouble(domain._occupancyEntropyThresholds, fallback);

	for (const size_t fallback : fallbackPointThresholds())
	{
		if (fallback <= cloud.size() * 2)
			addUniqueSize(domain._pointThresholds, fallback);
	}
	for (const double fallback : fallbackHeightRatioThresholds())
		addUniqueDouble(domain._heightRatioThresholds, fallback);
	if (rootDensity > 0.0)
	{
		for (const double multiplier : { 0.25, 0.50, 1.0, 2.0, 4.0 })
			addUniqueDouble(domain._densityThresholds, rootDensity * multiplier);
	}
	for (const double divisor : { 1.0, 2.0, 4.0, 8.0 })
	{
		addUniqueDouble(domain._extentXThresholds, static_cast<double>(rootExtent.x) / divisor);
		addUniqueDouble(domain._extentYThresholds, static_cast<double>(rootExtent.y) / divisor);
		addUniqueDouble(domain._extentZThresholds, static_cast<double>(rootExtent.z) / divisor);
	}

	sortUniqueValues(domain._pointThresholds);
	sortUniqueValues(domain._densityThresholds);
	sortUniqueValues(domain._heightRatioThresholds);
	sortUniqueValues(domain._extentXThresholds);
	sortUniqueValues(domain._extentYThresholds);
	sortUniqueValues(domain._extentZThresholds);
	sortUniqueValues(domain._anisotropyThresholds);
	sortUniqueValues(domain._occupancyEntropyThresholds);
	return domain;
}

std::vector<Experiments::SchemaCandidate> Experiments::generateSchemaCandidates(const SchemaGenerationOptions& options)
{
	return generateSchemaCandidates(options, nullptr);
}

std::vector<Experiments::SchemaCandidate> Experiments::generateSchemaCandidates(
	const SchemaGenerationOptions& options,
	const ConditionDomain* conditionDomain)
{
	if (options._count == 0)
		return {};
	const size_t maxDepth = std::max<size_t>(1, options._maxDepth);
	const size_t maxBlocks = std::min(std::max<size_t>(1, options._maxBlocks), maxDepth);
	const size_t minBlocks = std::min(std::max<size_t>(1, options._minBlocks), maxBlocks);
	const size_t minLeaf = std::max<size_t>(1, std::min(options._minLeafCapacity, options._maxLeafCapacity));
	const size_t maxLeaf = std::max(minLeaf, options._maxLeafCapacity);
	const double conditionProbability = std::clamp(options._conditionalProbability, 0.0, 1.0);

	std::mt19937 rng(options._seed);
	std::bernoulli_distribution conditionDistribution(conditionProbability);
	std::unordered_set<std::string> seen;
	std::vector<SchemaCandidate> candidates;
	candidates.reserve(options._count);

	size_t attempts = 0;
	const size_t maxAttempts = std::max<size_t>(options._count * 50, 1024);
	while (candidates.size() < options._count && attempts++ < maxAttempts)
	{
		std::uniform_int_distribution<size_t> blockDistribution(minBlocks, maxBlocks);
		const size_t numBlocks = blockDistribution(rng);

		SchemaConfig schema;
		schema._buildPolicy._maxDepth = maxDepth;
		schema._buildPolicy._leafCapacity = randomPowerOfTwo(rng, minLeaf, maxLeaf);
		schema._buildPolicy._minPrimitivesToSplit = std::max<size_t>(2, schema._buildPolicy._leafCapacity / 4);
		schema._buildPolicy._collapseSingleChild = true;
		schema._buildPolicy._removeEmptyNodes = true;
		schema._buildPolicy._allowOverlapDuplication = false;

		size_t remainingDepth = maxDepth;
		std::optional<MultiDataStructure::DataStructureLevel> previousType;
		for (size_t block = 0; block < numBlocks; ++block)
		{
			const size_t remainingBlocks = numBlocks - block - 1;
			const size_t maxLevelForBlock = remainingDepth - remainingBlocks;
			std::uniform_int_distribution<size_t> levelDistribution(1, maxLevelForBlock);

			SchemaLevelConfig level;
			level._type = randomStructureType(rng, previousType);
			level._typeName = randomTypeNameForBase(level._type, rng, options);
			level._numLevels = levelDistribution(rng);
			level._leafCapacity = randomPowerOfTwo(rng, minLeaf, maxLeaf);
			level._minPrimitivesToSplit = std::max<size_t>(2, level._leafCapacity / 4);
			level._axisPolicy = sampleAxisPolicy(rng, level._type);
			maybeAssignAdaptiveLeafCapacity(level, rng, options);
			if (options._conditionalLevels && block > 0 && conditionDistribution(rng))
				level._condition = randomLevelCondition(rng, level, minLeaf, maxLeaf, conditionDomain);

			schema._levels.push_back(level);
			previousType = level._type;
			remainingDepth -= level._numLevels;
		}

		if (schema._levels.empty())
			continue;

		schema._buildPolicy._maxDepth = std::min(maxDepth, schema.totalLevels());
		const std::string signature = schemaSignature(schema);
		if (!seen.insert(signature).second)
			continue;

		SchemaCandidate candidate = materializeGeneratedSchema(schema, "generated", options._outputDirectory);
		candidates.push_back(std::move(candidate));
	}

	if (candidates.size() < options._count)
		std::cerr << "Warning: generated " << candidates.size() << " unique schemas from requested " << options._count << '\n';

	return candidates;
}

Experiments::SchemaRepairDiagnostics Experiments::diagnoseSchemaRepair(
	const SchemaCandidate& candidate,
	const std::vector<SchemaSearchRecord>& measuredRecords)
{
	return diagnoseRepairInternal(candidate, measuredRecords);
}

std::vector<Experiments::SchemaCandidate> Experiments::generateSchemaRepairCandidates(
	const SchemaCandidate& candidate,
	const std::vector<SchemaSearchRecord>& measuredRecords,
	const SchemaGenerationOptions& generationOptions,
	const ConditionDomain* conditionDomain,
	size_t maxCandidates,
	uint32_t seed,
	const std::string& outputDirectory)
{
	std::vector<RepairSchemaMutation> mutations = generateRepairSchemaMutations(
		candidate,
		measuredRecords,
		generationOptions,
		conditionDomain,
		maxCandidates,
		seed);

	std::vector<SchemaCandidate> candidates;
	candidates.reserve(mutations.size());
	for (size_t i = 0; i < mutations.size(); ++i)
	{
		const std::string prefix = "repair_" + mutations[i]._reason + "_" + std::to_string(i);
		candidates.push_back(materializeGeneratedSchema(std::move(mutations[i]._schema), prefix, outputDirectory));
	}
	return candidates;
}

Experiments::WorkloadProfile Experiments::parseWorkloadProfile(const std::string& jsonText, const std::string& sourceName)
{
	boost::system::error_code error;
	boost::json::value rootValue = boost::json::parse(jsonText, error);
	if (error)
		throw std::runtime_error("Invalid workload JSON" + (sourceName.empty() ? std::string() : " in " + sourceName) + ": " + error.message());

	if (!rootValue.is_object())
		throw std::runtime_error("Workload JSON root must be an object");

	const boost::json::object& root = rootValue.as_object();

	WorkloadProfile profile;
	profile._name = asString(root, "name", profile._name);
	profile._numQueries = asSize(root, "numQueries", profile._numQueries);
	profile._knnK = asSize(root, "knnK", profile._knnK);
	profile._querySeed = static_cast<uint32_t>(asSize(root, "querySeed", profile._querySeed));
	profile._stratifyQueries = asBool(root, "stratifyQueries", profile._stratifyQueries);
	profile._stratifyQueries = asBool(root, "stratifiedQueries", profile._stratifyQueries);
	profile._rangeScaleMin = asDouble(root, "rangeScaleMin", profile._rangeScaleMin);
	profile._rangeScaleMax = asDouble(root, "rangeScaleMax", profile._rangeScaleMax);
	profile._radiusScaleMin = asDouble(root, "radiusScaleMin", profile._radiusScaleMin);
	profile._radiusScaleMax = asDouble(root, "radiusScaleMax", profile._radiusScaleMax);

	if (const boost::json::value* queries = root.if_contains("queries"))
	{
		if (!queries->is_object())
			throw std::runtime_error("Workload queries must be an object");

		const boost::json::object& queryWeights = queries->as_object();
		profile._rangeWeight = asDouble(queryWeights, "aabb_range", profile._rangeWeight);
		profile._radiusWeight = asDouble(queryWeights, "radius", profile._radiusWeight);
		profile._knnWeight = asDouble(queryWeights, "knn", profile._knnWeight);
	}

	if (const boost::json::value* queryScales = root.if_contains("queryScales"))
	{
		if (!queryScales->is_object())
			throw std::runtime_error("Workload queryScales must be an object");

		const boost::json::object& scales = queryScales->as_object();
		parseScaleRange(scales, "aabb_range", profile._rangeScaleMin, profile._rangeScaleMax);
		parseScaleRange(scales, "range", profile._rangeScaleMin, profile._rangeScaleMax);
		parseScaleRange(scales, "volume", profile._rangeScaleMin, profile._rangeScaleMax);
		parseScaleRange(scales, "radius", profile._radiusScaleMin, profile._radiusScaleMax);
	}

	if (const boost::json::value* scoreWeights = root.if_contains("scoreWeights"))
	{
		if (!scoreWeights->is_object())
			throw std::runtime_error("Workload scoreWeights must be an object");

		profile._scoreWeights = parseScoreWeightsObject(scoreWeights->as_object());
		profile._hasScoreWeights = true;
	}

	normalizeScaleRange(profile._rangeScaleMin, profile._rangeScaleMax);
	normalizeScaleRange(profile._radiusScaleMin, profile._radiusScaleMax);

	if (profile._name.empty())
		throw std::runtime_error("Workload profile requires a non-empty name");

	return profile;
}

Experiments::WorkloadProfile Experiments::loadWorkloadProfile(const std::string& filename)
{
	const std::filesystem::path resolvedPath = resolveExistingPath(filename);
	std::ifstream file(resolvedPath);
	if (!file.is_open())
		throw std::runtime_error("Unable to open workload profile: " + filename);

	std::stringstream buffer;
	buffer << file.rdbuf();
	return parseWorkloadProfile(buffer.str(), resolvedPath.string());
}

static boost::json::object serializeBuildMetricsForOutput(const Experiments::BuildMetrics& metrics)
{
	boost::json::object out;
	out["buildTimeMs"] = metrics._buildTimeMs;
	out["numNodes"] = metrics._numNodes;
	out["numLeaves"] = metrics._numLeaves;
	out["indexedPoints"] = metrics._indexedPoints;
	out["maxDepth"] = metrics._maxDepth;
	out["averageLeafOccupancy"] = metrics._averageLeafOccupancy;
	out["maxLeafOccupancy"] = metrics._maxLeafOccupancy;
	out["leafOccupancyP50"] = metrics._leafOccupancyP50;
	out["leafOccupancyP90"] = metrics._leafOccupancyP90;
	out["leafOccupancyP99"] = metrics._leafOccupancyP99;
	out["averageDepth"] = metrics._averageDepth;
	out["averageFanout"] = metrics._averageFanout;
	out["maxFanout"] = metrics._maxFanout;
	out["emptyChildRatio"] = metrics._emptyChildRatio;
	out["singleChildNodeCount"] = metrics._singleChildNodeCount;
	out["meanTightBoundsVolumeRatio"] = metrics._meanTightBoundsVolumeRatio;
	out["microIndexedLeaves"] = metrics._microIndexedLeaves;
	out["microIndexedPoints"] = metrics._microIndexedPoints;
	out["nodeFanoutSummary"] = metrics._nodeFanoutSummary;
	out["memoryEstimateBytes"] = metrics._memoryEstimateBytes;
	return out;
}

static boost::json::object serializeQueryMetricsForOutput(const Experiments::QueryMetrics& metrics)
{
	boost::json::object out;
	out["totalQueries"] = metrics._totalQueries;
	out["totalLatencyMs"] = metrics._totalLatencyMs;
	out["averageLatencyMs"] = metrics._averageLatencyMs;
	out["medianLatencyMs"] = metrics._medianLatencyMs;
	out["p95LatencyMs"] = metrics._p95LatencyMs;
	out["throughputQueriesPerSecond"] = metrics._throughputQueriesPerSecond;
	out["averageVisitedNodes"] = metrics._averageVisitedNodes;
	out["averageTestedPoints"] = metrics._averageTestedPoints;
	out["averageReturnedPoints"] = metrics._averageReturnedPoints;
	out["averageFullyContainedNodes"] = metrics._averageFullyContainedNodes;
	out["totalVisitedNodes"] = metrics._totalVisitedNodes;
	out["totalTestedPoints"] = metrics._totalTestedPoints;
	out["totalReturnedPoints"] = metrics._totalReturnedPoints;
	out["totalFullyContainedNodes"] = metrics._totalFullyContainedNodes;
	return out;
}

static boost::json::object serializeRecordForStdout(const Experiments::SchemaSearchRecord& record)
{
	boost::json::object out;
	out["datasetName"] = record._datasetName;
	out["datasetSource"] = record._datasetSource;
	out["numPoints"] = record._numPoints;
	out["workloadName"] = record._workloadName;
	out["rangeWeight"] = record._rangeWeight;
	out["radiusWeight"] = record._radiusWeight;
	out["knnWeight"] = record._knnWeight;
	out["numQueries"] = record._numQueries;
	out["knnK"] = record._knnK;
	out["querySeed"] = record._querySeed;
	out["schemaName"] = record._schemaName;
	out["schemaPath"] = record._schemaPath;
	out["build"] = serializeBuildMetricsForOutput(record._buildMetrics);
	out["query"] = serializeQueryMetricsForOutput(record._queryMetrics);
	out["rangeQueries"] = record._rangeQueries;
	out["countRangeQueries"] = record._countRangeQueries;
	out["radiusQueries"] = record._radiusQueries;
	out["knnQueries"] = record._knnQueries;
	out["queryStrataSummary"] = record._queryStrataSummary;
	out["score"] = record._score;
	out["scoreMemoryMb"] = record._scoreMemoryMb;
	out["scoreImbalancePenalty"] = record._scoreImbalancePenalty;
	out["scoreMode"] = record._scoreMode;
	out["scoreStage"] = record._scoreStage;
	out["scoreIsFinalLatency"] = record._scoreIsFinalLatency;
	out["backend"] = record._backend;
	out["knnBackend"] = record._knnBackend;
	out["cudaDevice"] = record._cudaDevice;
	out["cudaBuilder"] = record._cudaBuilder;
	out["gpuUploadMs"] = record._gpuUploadMs;
	out["gpuBuildMs"] = record._gpuBuildMs;
	out["gpuQueryMs"] = record._gpuQueryMs;
	out["gpuMemoryBytes"] = record._gpuMemoryBytes;
	out["conditionalLevels"] = record._conditionalLevels;
	out["conditionFields"] = record._conditionFields;
	out["conditionSummary"] = record._conditionSummary;
	out["isBaseline"] = record._isBaseline;
	out["activeStructureTypes"] = record._activeStructureTypes;
	out["nestedActiveFraction"] = record._nestedActiveFraction;
	out["activeStructureSummary"] = record._activeStructureSummary;
	out["bestBaselineSchema"] = record._bestBaselineSchema;
	out["bestBaselineScore"] = record._bestBaselineScore;
	out["relativeSpeedupVsBaseline"] = record._relativeSpeedupVsBaseline;

	boost::json::object weights;
	weights["lambdaLatency"] = record._weights._lambdaLatency;
	weights["lambdaBuild"] = record._weights._lambdaBuild;
	weights["lambdaMemory"] = record._weights._lambdaMemory;
	weights["lambdaImbalance"] = record._weights._lambdaImbalance;
	weights["useVisitProxy"] = record._weights._useVisitProxy;
	weights["visitProxyAlpha"] = record._weights._visitProxyAlpha;
	out["weights"] = weights;
	return out;
}

int Experiments::runEvaluateOne(const SchemaSearchOptions& options)
{
	SchemaSearchOptions oneOptions = options;
	oneOptions._csvPath.clear();
	oneOptions._bestCsvPath.clear();
	oneOptions._includeSyntheticDatasets = false;
	oneOptions._includeConfiguredSchemas = true;
	oneOptions._generation._count = 0;
	oneOptions._autoConditions._enabled = false;
	oneOptions._evolution._enabled = false;
	oneOptions._benchmarkTopK = 0;
	oneOptions._pauseAtEnd = false;
	oneOptions._rankModelPath.clear();

	if (oneOptions._inputPaths.empty())
		throw std::runtime_error("evaluate-one requires --input <cloud>");
	if (oneOptions._schemaPaths.empty())
		throw std::runtime_error("evaluate-one requires --schema <schema.json>");
	if (oneOptions._workloadPaths.empty())
		throw std::runtime_error("evaluate-one requires --workloads <workload.json>");

	std::vector<boost::json::value> captured;
	oneOptions.progressCallback = [&captured](const SchemaSearchRecord& record) {
		captured.push_back(serializeRecordForStdout(record));
	};

	const int result = runSchemaSearch(oneOptions);

	boost::json::object payload;
	payload["recordCount"] = captured.size();
	boost::json::array records;
	records.reserve(captured.size());
	for (boost::json::value& record : captured)
		records.emplace_back(std::move(record));
	payload["records"] = std::move(records);
	std::cout << boost::json::serialize(payload) << '\n';

	return result;
}

Experiments::SchemaCandidate Experiments::materializeSchemaCandidate(
	SchemaConfig schema,
	const std::string& namePrefix,
	const std::string& outputDirectory)
{
	return ::materializeGeneratedSchema(std::move(schema), namePrefix, outputDirectory);
}

double Experiments::computeSchemaSearchScore(
	const BuildMetrics& buildMetrics,
	const QueryMetrics& queryMetrics,
	const ScoreWeights& weights,
	double& memoryMb,
	double& imbalancePenalty)
{
	memoryMb = static_cast<double>(buildMetrics._memoryEstimateBytes) / (1024.0 * 1024.0);
	imbalancePenalty = buildMetrics._averageLeafOccupancy > 0.0
		? static_cast<double>(buildMetrics._maxLeafOccupancy) / buildMetrics._averageLeafOccupancy
		: 0.0;

	const double primary = weights._useVisitProxy
		? queryMetrics._averageVisitedNodes + weights._visitProxyAlpha * queryMetrics._averageTestedPoints
		: weights._lambdaLatency * queryMetrics._averageLatencyMs;

	return primary +
		weights._lambdaBuild * buildMetrics._buildTimeMs +
		weights._lambdaMemory * memoryMb +
		weights._lambdaImbalance * imbalancePenalty;
}

// Memory footprint in MB, preferring the measured GPU allocation for CUDA records and falling back to the crude CPU-side estimate, so the memory objective stays honest for GPU runs.
static double recordMemoryMb(const Experiments::SchemaSearchRecord& record)
{
	const size_t bytes = (record._backend == "cuda" && record._gpuMemoryBytes > 0)
		? record._gpuMemoryBytes
		: record._buildMetrics._memoryEstimateBytes;
	return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static double bestSelectionScore(const Experiments::SchemaSearchRecord& record)
{
	if (record._weights._useVisitProxy)
		return record._score;

	if (record._confirmSeedsUsed == 0 && record._queryMetrics._averageLatencyMs <= 0.0)
		return record._score;

	const double primary = record._confirmSeedsUsed > 0 && record._latencyMean > 0.0
		? record._latencyMean
		: record._queryMetrics._averageLatencyMs;
	const double buildMs = record._confirmSeedsUsed > 0 && record._gpuBuildMean > 0.0
		? record._gpuBuildMean
		: record._buildMetrics._buildTimeMs;
	const double memoryMb = recordMemoryMb(record);
	const double imbalancePenalty = record._buildMetrics._averageLeafOccupancy > 0.0
		? static_cast<double>(record._buildMetrics._maxLeafOccupancy) / record._buildMetrics._averageLeafOccupancy
		: 0.0;

	return record._weights._lambdaLatency * primary +
		record._weights._lambdaBuild * buildMs +
		record._weights._lambdaMemory * memoryMb +
		record._weights._lambdaImbalance * imbalancePenalty;
}

std::vector<Experiments::SchemaSearchRecord> Experiments::selectBestRecords(const std::vector<SchemaSearchRecord>& records)
{
	std::vector<SchemaSearchRecord> best;
	for (const SchemaSearchRecord& record : records)
	{
		auto existing = std::find_if(best.begin(), best.end(), [&record](const SchemaSearchRecord& candidate) {
			return candidate._datasetName == record._datasetName && candidate._workloadName == record._workloadName;
		});

		if (existing == best.end())
		{
			best.push_back(record);
			continue;
		}

		if (bestSelectionScore(record) < bestSelectionScore(*existing))
			*existing = record;
	}

	return best;
}

bool Experiments::confidentlyBetter(const SchemaSearchRecord& a, const SchemaSearchRecord& b)
{
	// Prefer the multi-seed confirmation CI, falling back to the measurement-repeat CI; without either there is no interval to separate and no confidence to claim.
	double aHigh = 0.0;
	double bLow = 0.0;
	bool haveInterval = false;
	if (a._confirmSeedsUsed > 0 && b._confirmSeedsUsed > 0 && a._latencyCiHigh > 0.0 && b._latencyCiLow > 0.0)
	{
		aHigh = a._latencyCiHigh;
		bLow = b._latencyCiLow;
		haveInterval = true;
	}
	else if (a._queryMetrics._measurementRepeats > 1 && b._queryMetrics._measurementRepeats > 1)
	{
		aHigh = a._queryMetrics._latencyCiHighMs;
		bLow = b._queryMetrics._latencyCiLowMs;
		haveInterval = true;
	}

	if (!haveInterval)
		return false;
	return aHigh < bLow;
}

void Experiments::annotateRankingConfidence(std::vector<SchemaSearchRecord>& records)
{
	std::map<std::pair<std::string, std::string>, std::vector<size_t>> groups;
	for (size_t i = 0; i < records.size(); ++i)
		groups[{ records[i]._datasetName, records[i]._workloadName }].push_back(i);

	for (auto& [key, indices] : groups)
	{
		(void)key;
		size_t bestIdx = indices.front();
		size_t runnerIdx = bestIdx;
		bool haveRunner = false;
		for (size_t k = 1; k < indices.size(); ++k)
		{
			const size_t idx = indices[k];
			if (bestSelectionScore(records[idx]) < bestSelectionScore(records[bestIdx]))
			{
				runnerIdx = bestIdx;
				haveRunner = true;
				bestIdx = idx;
			}
			else if (!haveRunner || bestSelectionScore(records[idx]) < bestSelectionScore(records[runnerIdx]))
			{
				runnerIdx = idx;
				haveRunner = true;
			}
		}

		// A lone candidate wins trivially; otherwise require non-overlapping latency CIs between winner and runner-up.
		const bool confident = !haveRunner
			? true
			: Experiments::confidentlyBetter(records[bestIdx], records[runnerIdx]);
		for (size_t idx : indices)
			records[idx]._rankingConfident = confident;
	}
}

// Fractional (tie-averaged) 1-based ranks of the values in `v`.
static std::vector<double> averageRanks(const std::vector<double>& v)
{
	const size_t n = v.size();
	std::vector<size_t> order(n);
	std::iota(order.begin(), order.end(), size_t(0));
	std::sort(order.begin(), order.end(), [&v](size_t a, size_t b) { return v[a] < v[b]; });

	std::vector<double> ranks(n, 0.0);
	size_t i = 0;
	while (i < n)
	{
		size_t j = i;
		while (j + 1 < n && v[order[j + 1]] == v[order[i]])
			++j;
		const double averageRank = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
		for (size_t k = i; k <= j; ++k)
			ranks[order[k]] = averageRank;
		i = j + 1;
	}
	return ranks;
}

double Experiments::spearmanRankCorrelation(const std::vector<double>& a, const std::vector<double>& b)
{
	if (a.size() != b.size() || a.size() < 2)
		return 0.0;

	const std::vector<double> ra = averageRanks(a);
	const std::vector<double> rb = averageRanks(b);
	const double n = static_cast<double>(a.size());
	const double meanA = std::accumulate(ra.begin(), ra.end(), 0.0) / n;
	const double meanB = std::accumulate(rb.begin(), rb.end(), 0.0) / n;

	double cov = 0.0;
	double varA = 0.0;
	double varB = 0.0;
	for (size_t i = 0; i < ra.size(); ++i)
	{
		const double da = ra[i] - meanA;
		const double db = rb[i] - meanB;
		cov += da * db;
		varA += da * da;
		varB += db * db;
	}

	if (varA <= 0.0 || varB <= 0.0)
		return 0.0;
	return cov / std::sqrt(varA * varB);
}

void Experiments::reportProxyLatencyCorrelation(const std::vector<SchemaSearchRecord>& records, const std::string& csvPath)
{
	struct ProxyGroup
	{
		std::vector<double> _proxy;
		std::vector<double> _latency;
		double _alpha = 0.0;
	};

	std::map<std::pair<std::string, std::string>, ProxyGroup> groups;
	for (const SchemaSearchRecord& record : records)
	{
		// Compare on the real-latency records only; proxy-stage rows did not measure true latency.
		if (record._weights._useVisitProxy)
			continue;
		const double latency = record._confirmSeedsUsed > 0 && record._latencyMean > 0.0
			? record._latencyMean
			: record._queryMetrics._averageLatencyMs;
		if (latency <= 0.0)
			continue;

		const double proxy = record._queryMetrics._averageVisitedNodes
			+ record._weights._visitProxyAlpha * record._queryMetrics._averageTestedPoints;
		ProxyGroup& group = groups[{ record._datasetName, record._workloadName }];
		group._proxy.push_back(proxy);
		group._latency.push_back(latency);
		group._alpha = record._weights._visitProxyAlpha;
	}

	if (groups.empty())
		return;

	std::cout << "  proxy/latency rank correlation (Spearman; 1.0 = proxy perfectly predicts latency order):\n";

	std::ofstream csv;
	const bool writeCsv = !csvPath.empty();
	if (writeCsv)
	{
		createParentDirectory(csvPath);
		csv.open(csvPath);
		if (csv.is_open())
			csv << "dataset_name,workload_name,num_candidates,spearman_rho,visit_proxy_alpha\n";
	}

	for (const auto& [key, group] : groups)
	{
		if (group._proxy.size() < 3)
		{
			std::cout << "    [" << key.first << " / " << key.second << "] n/a ("
				<< group._proxy.size() << " candidate(s))\n";
			continue;
		}

		const double rho = Experiments::spearmanRankCorrelation(group._proxy, group._latency);
		std::cout << "    [" << key.first << " / " << key.second << "] rho="
			<< std::fixed << std::setprecision(3) << rho
			<< " over " << group._proxy.size() << " candidates\n";
		if (writeCsv && csv.is_open())
			csv << csvEscape(key.first) << ',' << csvEscape(key.second) << ','
				<< group._proxy.size() << ',' << rho << ',' << group._alpha << '\n';
	}
}

// Spatial branching factor and partitioned dimensionality for a primitive, used to turn a leaf count into a per-axis leaf count and an internal-node multiplier.
static std::pair<double, int> primitiveBranchingAndDims(SchemaPrimitiveKind kind)
{
	switch (kind)
	{
	case SchemaPrimitiveKind::QuadTree:     return { 4.0, 2 };
	case SchemaPrimitiveKind::Octree:       return { 8.0, 3 };
	case SchemaPrimitiveKind::KarrasOctree: return { 8.0, 3 };
	case SchemaPrimitiveKind::KDTree:       return { 2.0, 3 };
	case SchemaPrimitiveKind::BVH:
	case SchemaPrimitiveKind::LBVH:
	case SchemaPrimitiveKind::BIH:          return { 2.0, 3 };
	case SchemaPrimitiveKind::RegularGrid:  return { 27.0, 3 };
	case SchemaPrimitiveKind::HGrid:        return { 8.0, 3 };
	case SchemaPrimitiveKind::Mixed:        return { 8.0, 3 };
	default:                                return { 8.0, 3 };
	}
}

double Experiments::estimateSchemaQueryCost(
	const SchemaConfig& schema,
	const PointCloudFeatures& features,
	const WorkloadFeatures& workload,
	double visitProxyAlpha)
{
	const double n = std::max(1.0, static_cast<double>(features._numPoints));
	if (schema._levels.empty())
		return n; // no structure: a query scans the whole cloud

	const SchemaLevelConfig& leafBlock = schema._levels.back();
	const auto [branching, dims] = primitiveBranchingAndDims(leafBlock._primitiveKind);

	size_t totalDepth = 0;
	for (const SchemaLevelConfig& level : schema._levels)
		totalDepth += std::max<size_t>(1, level._numLevels);
	totalDepth = std::min<size_t>(totalDepth, 24); // cap to keep pow() finite

	size_t leafCapacity = leafBlock._leafCapacity > 0 ? leafBlock._leafCapacity : schema._buildPolicy._leafCapacity;
	leafCapacity = std::max<size_t>(1, leafCapacity);

	// Leaves: as many as needed to reach ~leafCapacity per leaf, but no more than the depth allows.
	const double maxLeaves = std::pow(branching, static_cast<double>(totalDepth));
	const double idealLeaves = n / static_cast<double>(leafCapacity);
	const double leaves = std::max(1.0, std::min(idealLeaves, maxLeaves));
	const double avgLeafPop = n / leaves;
	const double leavesPerSide = std::pow(leaves, 1.0 / static_cast<double>(dims));
	const double scale = std::clamp(workload._queryScaleMean, 0.0, 1.0);

	// A side-`scale` volume query touches ~(scale * leavesPerSide + 1) leaves per axis; internal ancestors add a geometric b/(b-1) factor.
	const double volumeVisitedLeaves = std::min(leaves, std::pow(scale * leavesPerSide + 1.0, static_cast<double>(dims)));
	const double volumeTested = volumeVisitedLeaves * avgLeafPop;
	const double internalFactor = branching > 1.0 ? branching / (branching - 1.0) : static_cast<double>(totalDepth);
	const double volumeVisitedNodes = volumeVisitedLeaves * internalFactor;
	const double volumeCost = volumeVisitedNodes + visitProxyAlpha * volumeTested;

	// KNN: descend to a leaf then expand to a couple of neighbouring leaves.
	const double knnCost = static_cast<double>(totalDepth)
		+ visitProxyAlpha * (2.0 * avgLeafPop + static_cast<double>(workload._knnK));

	const double wSum = workload._wRange + workload._wRadius + workload._wKnn;
	if (wSum <= 0.0)
		return volumeCost;
	return (workload._wRange * volumeCost + workload._wRadius * volumeCost + workload._wKnn * knnCost) / wSum;
}

void Experiments::reportEstimatedCostCorrelation(const std::vector<SchemaSearchRecord>& records, const std::string& csvPath)
{
	struct CostGroup
	{
		std::vector<double> _estimate;
		std::vector<double> _latency;
	};

	std::map<std::pair<std::string, std::string>, CostGroup> groups;
	for (const SchemaSearchRecord& record : records)
	{
		if (record._weights._useVisitProxy)
			continue;
		const double latency = record._confirmSeedsUsed > 0 && record._latencyMean > 0.0
			? record._latencyMean
			: record._queryMetrics._averageLatencyMs;
		if (latency <= 0.0 || record._estimatedQueryCost <= 0.0)
			continue;
		CostGroup& group = groups[{ record._datasetName, record._workloadName }];
		group._estimate.push_back(record._estimatedQueryCost);
		group._latency.push_back(latency);
	}

	if (groups.empty())
		return;

	std::cout << "  zero-build cost estimate vs latency rank correlation (Spearman):\n";
	std::ofstream csv;
	const bool writeCsv = !csvPath.empty();
	if (writeCsv)
	{
		createParentDirectory(csvPath);
		csv.open(csvPath);
		if (csv.is_open())
			csv << "dataset_name,workload_name,num_candidates,spearman_rho\n";
	}

	for (const auto& [key, group] : groups)
	{
		if (group._estimate.size() < 3)
		{
			std::cout << "    [" << key.first << " / " << key.second << "] n/a ("
				<< group._estimate.size() << " candidate(s))\n";
			continue;
		}
		const double rho = Experiments::spearmanRankCorrelation(group._estimate, group._latency);
		std::cout << "    [" << key.first << " / " << key.second << "] rho="
			<< std::fixed << std::setprecision(3) << rho
			<< " over " << group._estimate.size() << " candidates\n";
		if (writeCsv && csv.is_open())
			csv << csvEscape(key.first) << ',' << csvEscape(key.second) << ','
				<< group._estimate.size() << ',' << rho << '\n';
	}
}

struct ParetoMetrics
{
	double _avgLatencyMs = 0.0;
	double _buildTimeMs = 0.0;
	double _memoryMb = 0.0;
	double _imbalancePenalty = 0.0;
};

static ParetoMetrics extractParetoMetrics(const Experiments::SchemaSearchRecord& record)
{
	ParetoMetrics m;
	// Prefer the seed-averaged mean after multi-seed confirmation, so a single lucky seed can't fake a Pareto win.
	m._avgLatencyMs = record._confirmSeedsUsed > 0
		? record._latencyMean
		: record._queryMetrics._averageLatencyMs;
	m._buildTimeMs = record._buildMetrics._buildTimeMs;
	// Prefer the measured GPU footprint, falling back to the closed-form CPU estimate; derived here since scoreMemoryMb may be 0 for cached records.
	m._memoryMb = recordMemoryMb(record);
	m._imbalancePenalty = record._buildMetrics._averageLeafOccupancy > 0.0
		? static_cast<double>(record._buildMetrics._maxLeafOccupancy) / record._buildMetrics._averageLeafOccupancy
		: 0.0;
	return m;
}

static bool dominates(const ParetoMetrics& a, const ParetoMetrics& b)
{
	const bool allLEq =
		a._avgLatencyMs <= b._avgLatencyMs &&
		a._buildTimeMs <= b._buildTimeMs &&
		a._memoryMb <= b._memoryMb &&
		a._imbalancePenalty <= b._imbalancePenalty;
	if (!allLEq)
		return false;
	const bool anyStrict =
		a._avgLatencyMs < b._avgLatencyMs ||
		a._buildTimeMs < b._buildTimeMs ||
		a._memoryMb < b._memoryMb ||
		a._imbalancePenalty < b._imbalancePenalty;
	return anyStrict;
}

static double normalize01(double v, double lo, double hi)
{
	return hi > lo ? (v - lo) / (hi - lo) : 0.0;
}

static std::pair<double, double> paretoAxisRange(const std::vector<ParetoMetrics>& objectives, double ParetoMetrics::* axis)
{
	double lo = objectives.front().*axis;
	double hi = lo;
	for (const ParetoMetrics& o : objectives)
	{
		lo = std::min(lo, o.*axis);
		hi = std::max(hi, o.*axis);
	}
	return std::make_pair(lo, hi);
}

static std::string describeParetoEntry(const Experiments::SchemaSearchRecord& r)
{
	const double latency = r._confirmSeedsUsed > 0 ? r._latencyMean : r._queryMetrics._averageLatencyMs;
	const double memoryMb = (r._backend == "cuda" && r._gpuMemoryBytes > 0)
		? static_cast<double>(r._gpuMemoryBytes) / (1024.0 * 1024.0)
		: static_cast<double>(r._buildMetrics._memoryEstimateBytes) / (1024.0 * 1024.0);
	const double imbalance = r._buildMetrics._averageLeafOccupancy > 0.0
		? static_cast<double>(r._buildMetrics._maxLeafOccupancy) / r._buildMetrics._averageLeafOccupancy
		: 0.0;
	std::ostringstream out;
	out << std::fixed << std::setprecision(4)
		<< "latency=" << latency << "ms build=" << r._buildMetrics._buildTimeMs
		<< "ms mem=" << std::setprecision(2) << memoryMb << "MB imbalance="
		<< std::setprecision(2) << imbalance;
	return out.str();
}

std::vector<Experiments::SchemaSearchRecord> Experiments::selectParetoRecords(const std::vector<SchemaSearchRecord>& records)
{
	// Group by (dataset, workload) and run O(n^2) non-domination filtering per group; batch sizes stay small enough that the quadratic cost is far cheaper than any single measurement.
	std::map<std::pair<std::string, std::string>, std::vector<size_t>> groups;
	for (size_t i = 0; i < records.size(); ++i)
		groups[{ records[i]._datasetName, records[i]._workloadName }].push_back(i);

	std::vector<SchemaSearchRecord> front;
	for (const auto& [key, indices] : groups)
	{
		std::vector<size_t> nonDominated;
		nonDominated.reserve(indices.size());
		for (const size_t i : indices)
		{
			const ParetoMetrics mi = extractParetoMetrics(records[i]);
			bool dominated = false;
			for (const size_t j : indices)
			{
				if (i == j)
					continue;
				if (dominates(extractParetoMetrics(records[j]), mi))
				{
					dominated = true;
					break;
				}
			}
			if (!dominated)
				nonDominated.push_back(i);
		}

		// Rank non-dominated entries by avgLatencyMs (buildTimeMs as a stable tie-break) so the CSV's fastest front entry sits at rank 0 without re-sorting.
		std::sort(nonDominated.begin(), nonDominated.end(), [&records](size_t a, size_t b) {
			if (records[a]._queryMetrics._averageLatencyMs != records[b]._queryMetrics._averageLatencyMs)
				return records[a]._queryMetrics._averageLatencyMs < records[b]._queryMetrics._averageLatencyMs;
			return records[a]._buildMetrics._buildTimeMs < records[b]._buildMetrics._buildTimeMs;
		});

		// Knee = front entry closest to the normalized ideal across the four objectives (equal weighting); a single-entry front is its own knee.
		size_t kneeRank = 0;
		if (!nonDominated.empty())
		{
			std::vector<ParetoMetrics> objectives;
			objectives.reserve(nonDominated.size());
			for (const size_t idx : nonDominated)
				objectives.push_back(extractParetoMetrics(records[idx]));

			const auto [loL, hiL] = paretoAxisRange(objectives, &ParetoMetrics::_avgLatencyMs);
			const auto [loB, hiB] = paretoAxisRange(objectives, &ParetoMetrics::_buildTimeMs);
			const auto [loM, hiM] = paretoAxisRange(objectives, &ParetoMetrics::_memoryMb);
			const auto [loI, hiI] = paretoAxisRange(objectives, &ParetoMetrics::_imbalancePenalty);

			double bestDist = std::numeric_limits<double>::max();
			for (size_t r = 0; r < objectives.size(); ++r)
			{
				const ParetoMetrics& o = objectives[r];
				const double nl = normalize01(o._avgLatencyMs, loL, hiL);
				const double nb = normalize01(o._buildTimeMs, loB, hiB);
				const double nm = normalize01(o._memoryMb, loM, hiM);
				const double ni = normalize01(o._imbalancePenalty, loI, hiI);
				const double dist = nl * nl + nb * nb + nm * nm + ni * ni;
				if (dist < bestDist)
				{
					bestDist = dist;
					kneeRank = r;
				}
			}
		}

		for (size_t rank = 0; rank < nonDominated.size(); ++rank)
		{
			SchemaSearchRecord entry = records[nonDominated[rank]];
			entry._paretoRank = static_cast<int>(rank);
			entry._paretoKnee = (rank == kneeRank);
			front.push_back(std::move(entry));
		}
	}

	return front;
}

void Experiments::reportParetoKnee(const std::vector<SchemaSearchRecord>& records)
{
	const std::vector<SchemaSearchRecord> front = selectParetoRecords(records);
	if (front.empty())
		return;

	std::map<std::pair<std::string, std::string>, std::vector<size_t>> groups;
	for (size_t i = 0; i < front.size(); ++i)
		groups[{ front[i]._datasetName, front[i]._workloadName }].push_back(i);

	std::cout << "  Pareto recommendation (knee = balanced compromise across latency/build/memory/imbalance):\n";
	for (const auto& [key, indices] : groups)
	{
		size_t fastestIdx = indices.front();
		size_t kneeIdx = indices.front();
		for (const size_t i : indices)
		{
			if (front[i]._paretoRank == 0)
				fastestIdx = i;
			if (front[i]._paretoKnee)
				kneeIdx = i;
		}

		std::cout << "    [" << key.first << " / " << key.second << "] front size " << indices.size() << "\n";
		std::cout << "      fastest: " << front[fastestIdx]._schemaName << "  " << describeParetoEntry(front[fastestIdx]) << "\n";
		if (kneeIdx == fastestIdx)
			std::cout << "      knee:    (same as fastest)\n";
		else
			std::cout << "      knee:    " << front[kneeIdx]._schemaName << "  " << describeParetoEntry(front[kneeIdx]) << "\n";
	}
}

std::tuple<double, double, double> Experiments::bootstrapMeanCI(
	const std::vector<double>& samples,
	size_t resamples,
	uint32_t seed)
{
	if (samples.empty())
		return { 0.0, 0.0, 0.0 };

	const double pointMean = std::accumulate(samples.begin(), samples.end(), 0.0)
		/ static_cast<double>(samples.size());

	if (samples.size() == 1 || resamples == 0)
		return { pointMean, pointMean, pointMean };

	std::mt19937 rng(seed);
	std::uniform_int_distribution<size_t> pick(0, samples.size() - 1);
	std::vector<double> resampledMeans(resamples);
	for (size_t b = 0; b < resamples; ++b)
	{
		double sum = 0.0;
		for (size_t i = 0; i < samples.size(); ++i)
			sum += samples[pick(rng)];
		resampledMeans[b] = sum / static_cast<double>(samples.size());
	}
	std::sort(resampledMeans.begin(), resampledMeans.end());

	const size_t loIdx = static_cast<size_t>(static_cast<double>(resamples) * 0.025);
	const size_t hiIdx = std::min(resamples - 1, static_cast<size_t>(static_cast<double>(resamples) * 0.975));
	return { pointMean, resampledMeans[loIdx], resampledMeans[hiIdx] };
}

static SchemaConfig makeParitySchema(const std::string& typeName, MultiDataStructure::DataStructureLevel type)
{
	SchemaLevelConfig level;
	level._type = type;
	level._typeName = typeName;
	level._numLevels = 6;
	level._leafCapacity = 64;
	level._minPrimitivesToSplit = 8;

	SchemaConfig schema;
	schema._name = "parity_" + typeName;
	schema._levels.push_back(level);
	schema._buildPolicy._maxDepth = 6;
	schema._buildPolicy._leafCapacity = 64;
	schema._buildPolicy._minPrimitivesToSplit = 8;
	schema._buildPolicy._removeEmptyNodes = true;
	schema._buildPolicy._collapseSingleChild = false;
	return schema;
}

// Compares CPU-index vs. GPU-MixedTree returned-point counts for each range/radius query (KNN skipped since the GPU path brute-force scans); GPU queries derive from the CPU queries for identical inputs.
static void runCpuGpuParityCheck(
	const std::vector<SearchDataset>& datasets,
	const std::vector<Experiments::WorkloadProfile>& workloads,
	const Experiments::SchemaSearchOptions& options)
{
	std::string cudaError;
	if (!PointGpu::MixedTree::isAvailable(&cudaError))
	{
		std::cout << "  CPU/GPU parity check skipped: CUDA unavailable (" << cudaError << ")\n";
		return;
	}
	if (datasets.empty() || workloads.empty())
		return;

	const Experiments::WorkloadProfile& workload = workloads.front();
	PointGpu::Options cudaOptions = cudaOptionsFrom(options);
	cudaOptions._builder = "mixed";

	const std::vector<SchemaConfig> schemas = {
		makeParitySchema("Octree", MultiDataStructure::DataStructureLevel::OctreeNode),
		makeParitySchema("KDTree", MultiDataStructure::DataStructureLevel::KDTreeNode),
	};

	const std::string csvPath = "results/parity_report.csv";
	createParentDirectory(csvPath);
	std::ofstream csv(csvPath);
	if (csv.is_open())
		csv << "dataset_name,schema_name,workload_name,cpu_nodes,gpu_nodes,cpu_leaves,gpu_leaves,"
			   "compared_queries,count_matches,max_count_delta,parity_ok\n";

	std::cout << "  CPU/GPU parity check (range+radius returned-count agreement on '" << workload._name << "'):\n";

	for (const SearchDataset& dataset : datasets)
	{
		const PreparedWorkload prepared = prepareWorkloadProfile(workload, dataset._cloud, false);
		std::vector<PointGpu::Query> gpuQueries;
		gpuQueries.reserve(prepared._cpuQueries.size());
		for (const PreparedCpuQuery& q : prepared._cpuQueries)
		{
			PointGpu::Query gq;
			if (q._kind == PreparedQueryKind::Range)
			{
				gq._type = PointGpu::QueryType::Range;
				gq._bounds = q._bounds;
			}
			else if (q._kind == PreparedQueryKind::Radius)
			{
				gq._type = PointGpu::QueryType::Radius;
				gq.center = q.center;
				gq._radius = q._radius;
			}
			else
			{
				gq._type = PointGpu::QueryType::Knn;
				gq.center = q.center;
				gq._k = workload._knnK;
			}
			gpuQueries.push_back(gq);
		}

		for (const SchemaConfig& schema : schemas)
		{
			PointSpatialIndex cpu;
			cpu.build(dataset._cloud, schema);
			const PointSpatialIndex::Stats cpuStats = cpu.stats();

			PointGpu::MixedTree gpu;
			PointGpu::BuildResult gpuBuild;
			PointGpu::QueryResult gpuResult;
			try
			{
				gpuBuild = gpu.build(dataset._cloud, schema, cudaOptions);
				gpuResult = gpu.query(gpuQueries, cudaOptions);
			}
			catch (const std::exception& ex)
			{
				std::cout << "    [" << dataset._name << " / " << schema._name << "] GPU build/query failed: " << ex.what() << "\n";
				continue;
			}

			size_t compared = 0;
			size_t matches = 0;
			size_t maxDelta = 0;
			const size_t n = std::min(prepared._cpuQueries.size(), gpuResult._samples.size());
			for (size_t i = 0; i < n; ++i)
			{
				const PreparedCpuQuery& q = prepared._cpuQueries[i];
				size_t cpuCount = 0;
				if (q._kind == PreparedQueryKind::Range)
					cpuCount = cpu.rangeQuery(q._bounds)._pointIndices.size();
				else if (q._kind == PreparedQueryKind::Radius)
					cpuCount = cpu.radiusQuery(q.center, q._radius)._pointIndices.size();
				else
					continue;

				const size_t gpuCount = static_cast<size_t>(gpuResult._samples[i]._returnedPoints);
				const size_t delta = cpuCount > gpuCount ? cpuCount - gpuCount : gpuCount - cpuCount;
				++compared;
				if (delta == 0)
					++matches;
				maxDelta = std::max(maxDelta, delta);
			}

			const bool parityOk = compared > 0 && matches == compared;
			std::cout << "    [" << dataset._name << " / " << schema._name << "] "
				<< matches << "/" << compared << " range+radius counts match (max delta "
				<< maxDelta << ")" << (parityOk ? "  OK" : "  MISMATCH")
				<< "  [cpu nodes/leaves " << cpuStats._numNodes << "/" << cpuStats._numLeaves
				<< " vs gpu " << gpuBuild._metrics._numNodes << "/" << gpuBuild._metrics._numLeaves << "]\n";

			if (csv.is_open())
				csv << csvEscape(dataset._name) << ',' << csvEscape(schema._name) << ',' << csvEscape(workload._name) << ','
					<< cpuStats._numNodes << ',' << gpuBuild._metrics._numNodes << ','
					<< cpuStats._numLeaves << ',' << gpuBuild._metrics._numLeaves << ','
					<< compared << ',' << matches << ',' << maxDelta << ',' << (parityOk ? 1 : 0) << '\n';
		}
	}
}

int Experiments::runSchemaSearch(const SchemaSearchOptions& options)
{
	SchemaSearchOptions resolvedOptions = options;
	if (resolvedOptions._deepNestedSearch)
	{
		resolvedOptions._autoConditions._enabled = true;
		resolvedOptions._includeConfiguredSchemas = false;
		resolvedOptions._includeBaselineSchemas = true;
		resolvedOptions._generation._minBlocks = std::max<size_t>(2, resolvedOptions._generation._minBlocks);
		resolvedOptions._generation._maxBlocks = std::max(resolvedOptions._generation._maxBlocks, resolvedOptions._generation._minBlocks);
		resolvedOptions._generation._conditionalLevels = true;
		resolvedOptions._generation._conditionalProbability = std::max(0.75, resolvedOptions._generation._conditionalProbability);
	}
	std::string cudaError;
	bool cudaAvailable = false;
	if (useCudaEvaluator(options))
		cudaAvailable = PointGpu::MixedTree::isAvailable(&cudaError);
	const EvaluatorResolution evaluatorResolution = resolveSchemaSearchEvaluator(options._evaluator, cudaAvailable, cudaError);
	resolvedOptions._evaluator = evaluatorResolution._evaluator;
	resolvedOptions._generation._primitiveProfile = resolvePrimitiveProfile(
		resolvedOptions._generation._primitiveProfile,
		evaluatorResolution._usingCuda);
	if (resolvedOptions._deepNestedSearch)
		resolvedOptions._generation._primitiveProfile = resolvePrimitiveProfile("query_minimal_cpu", false);
	if (evaluatorResolution._usingCuda && resolvedOptions._generation._adaptiveLeafCapacity)
	{
		std::cout << "  warning: generated adaptive leaf capacity is CPU-only for now; disabling it for CUDA evaluation\n";
		resolvedOptions._generation._adaptiveLeafCapacity = false;
	}
	if (evaluatorResolution._usingCuda && (resolvedOptions._generation._conditionalLevels || resolvedOptions._autoConditions._enabled))
		std::cout << "  note: occupancy-entropy gates are CPU-only; excluded from CUDA schema generation\n";

	Experiments::EvaluationCache ownedScoreCache;
	if (!resolvedOptions._scoreCachePath.empty() && resolvedOptions._scoreCache == nullptr)
	{
		if (resolvedOptions._rebuildScoreCache)
		{
			std::error_code error;
			std::filesystem::remove(resolvedOptions._scoreCachePath, error);
		}
		const bool opened = ownedScoreCache.open(resolvedOptions._scoreCachePath, false);
		if (opened)
		{
			resolvedOptions._scoreCache = &ownedScoreCache;
			std::cout << "  score cache: " << resolvedOptions._scoreCachePath
				<< " (" << ownedScoreCache.entryCount() << " entries on load)\n";
		}
		else
		{
			std::cerr << "Warning: failed to open score cache at " << resolvedOptions._scoreCachePath << "; running uncached\n";
		}
	}

	const std::vector<SearchDataset> datasets = loadDatasets(resolvedOptions);

	// Per-cloud condition domain estimated once from the first dataset to calibrate conditional thresholds; only the GA / flat paths consult it (auto-conditions builds its own), and it's skipped on synthetic/empty runs.
	std::optional<ConditionDomain> sharedConditionDomain;
	if (!resolvedOptions._autoConditions._enabled && !datasets.empty() && !datasets.front()._cloud.empty())
	{
		sharedConditionDomain = estimateConditionDomain(datasets.front()._cloud,
			resolvedOptions._autoConditions._proxyPointCap > 0
				? resolvedOptions._autoConditions._proxyPointCap
				: size_t(262144));
		if (useCudaEvaluator(resolvedOptions))
			restrictConditionDomainToGpuSafe(sharedConditionDomain.value());
		std::cout << "  condition domain from '" << datasets.front()._name
			<< "': " << sharedConditionDomain->_pointThresholds.size() << " point, "
			<< sharedConditionDomain->_densityThresholds.size() << " density, "
			<< sharedConditionDomain->_heightRatioThresholds.size() << " height, "
			<< sharedConditionDomain->_anisotropyThresholds.size() << " anisotropy thresholds\n";
	}

	std::vector<SchemaCandidate> schemas = loadSchemas(resolvedOptions._schemaPaths, resolvedOptions._includeConfiguredSchemas);
	if (resolvedOptions._includeBaselineSchemas)
		appendBaselineSchemas(schemas, useCudaEvaluator(resolvedOptions));
	if (!resolvedOptions._autoConditions._enabled)
		appendGeneratedSchemas(schemas, resolvedOptions._generation,
			sharedConditionDomain.has_value() ? &sharedConditionDomain.value() : nullptr);
	if (schemas.empty())
	{
		if (!resolvedOptions._autoConditions._enabled || resolvedOptions._autoConditions._proxyCandidateCount == 0)
			throw std::runtime_error("Schema search has no schemas. Provide --schemas, omit --generated-only, or use --generate-schemas.");
	}
	const std::vector<WorkloadProfile> workloads = loadWorkloads(resolvedOptions);
	const std::optional<SchemaSelectorModel> rankModel = (resolvedOptions._rankModelPath.empty() || resolvedOptions._evolution._enabled || resolvedOptions._autoConditions._enabled)
		? std::optional<SchemaSelectorModel>()
		: std::optional<SchemaSelectorModel>(loadSchemaSelectorModel(resolvedOptions._rankModelPath));

	std::vector<SchemaSearchRecord> records;
	records.reserve(datasets.size() * workloads.size() * schemas.size());

	std::cout << std::fixed << std::setprecision(3);
	std::cout << "Schema search\n";
	std::cout << "  evaluator: " << (useCudaEvaluator(resolvedOptions) ? "cuda" : "cpu") << '\n';
	if (resolvedOptions._deepNestedSearch)
		std::cout << "  deep nested search: enabled (CPU discovery, nested candidates only, CUDA confirmation report when available)\n";
	if (!evaluatorResolution._warning.empty())
		std::cout << "  warning: " << evaluatorResolution._warning << '\n';
	if (useCudaEvaluator(resolvedOptions))
	{
		const PointGpu::Options cudaOptions = cudaOptionsFrom(resolvedOptions);
		std::cout << "  cuda builder: " << cudaBuilderDisplayName(cudaOptions._builder) << '\n';
		std::cout << "  cuda device: " << cudaDeviceDescription(cudaOptions) << '\n';
		std::cout << "  cuda warmup: " << warmUpCudaDevice(cudaOptions) << " ms\n";
		if (cudaOptions._queryBatchSize > 0)
			std::cout << "  cuda query batch: " << cudaOptions._queryBatchSize << '\n';
		if (cudaOptions._memoryBudgetMb > 0)
		{
			std::cout << "  cuda memory budget: " << cudaOptions._memoryBudgetMb << " MB";
			if (resolvedOptions._cuda._memoryBudgetMb == 0)
				std::cout << " (auto, 75% of total VRAM)";
			std::cout << '\n';
		}
	}
	std::cout << "  datasets: " << datasets.size() << '\n';
	std::cout << "  schemas: " << schemas.size() << '\n';
	if (resolvedOptions._deepNestedSearch)
	{
		std::cout << "  deep nested candidate budget: " << resolvedOptions._autoConditions._proxyCandidateCount << " requested\n";
		std::cout << "  generated min blocks: " << resolvedOptions._generation._minBlocks << '\n';
		std::cout << "  primitive profile: " << resolvedOptions._generation._primitiveProfile << '\n';
		if (resolvedOptions._generation._conditionalLevels)
			std::cout << "  generated conditions: probability " << resolvedOptions._generation._conditionalProbability << '\n';
		if (resolvedOptions._generation._adaptiveLeafCapacity)
			std::cout << "  adaptive leaf capacity: probability " << resolvedOptions._generation._adaptiveLeafProbability << '\n';
	}
	else if (resolvedOptions._generation._count > 0)
	{
		std::cout << "  generated schemas: " << resolvedOptions._generation._count << " requested\n";
		std::cout << "  primitive profile: " << resolvedOptions._generation._primitiveProfile << '\n';
		if (resolvedOptions._generation._conditionalLevels)
			std::cout << "  generated conditions: probability " << resolvedOptions._generation._conditionalProbability << '\n';
		if (resolvedOptions._generation._adaptiveLeafCapacity)
			std::cout << "  adaptive leaf capacity: probability " << resolvedOptions._generation._adaptiveLeafProbability << '\n';
	}
	if (!resolvedOptions._evolution._enabled && rankModel.has_value())
	{
		std::cout << "  surrogate rank model: " << resolvedOptions._rankModelPath << '\n';
		if (resolvedOptions._benchmarkTopK > 0)
			std::cout << "  benchmark top-k: " << resolvedOptions._benchmarkTopK << '\n';
	}
	else if (resolvedOptions._evolution._enabled && !resolvedOptions._rankModelPath.empty())
	{
		std::cout << "  surrogate rank model: not used inside evolutionary loop; measured scores drive selection\n";
	}
	std::cout << "  workloads: " << workloads.size() << '\n';
	if (resolvedOptions._autoConditions._enabled)
	{
		std::cout << "  auto-conditions: enabled\n";
		std::cout << "    proxy candidates: " << resolvedOptions._autoConditions._proxyCandidateCount << '\n';
		std::cout << "    proxy point cap: " << resolvedOptions._autoConditions._proxyPointCap << '\n';
		std::cout << "    proxy queries: " << resolvedOptions._autoConditions._proxyQueryCount << '\n';
		std::cout << "    final top-k: " << resolvedOptions._autoConditions._finalTopK << '\n';
		std::cout << "    confirmation top-k: " << resolvedOptions._autoConditions._confirmationTopK << '\n';
	}

	if (resolvedOptions._autoConditions._enabled)
	{
		records = runAutoConditionSearch(resolvedOptions, datasets, workloads, schemas);
	}
	else
	{
		std::vector<DatasetContext> datasetContexts = makeDatasetContexts(datasets, workloads, useCudaEvaluator(resolvedOptions));
		if (resolvedOptions._evolution._enabled)
		{
			records = runEvolutionarySchemaSearch(resolvedOptions, datasetContexts, workloads, schemas,
				sharedConditionDomain.has_value() ? &sharedConditionDomain.value() : nullptr);
		}
		else
		{
			CudaIndexCache cudaCache;
			for (const DatasetContext& datasetContext : datasetContexts)
			{
				const SearchDataset& dataset = *datasetContext._dataset;
				std::cout << "  dataset: " << dataset._name << " (" << dataset._cloud.size() << " points)\n";

				for (size_t workloadIndex = 0; workloadIndex < workloads.size(); ++workloadIndex)
				{
					const WorkloadProfile& workload = workloads[workloadIndex];
					const ScoreWeights weights = effectiveScoreWeights(workload, resolvedOptions);
					const WorkloadFeatures workloadFeatures = extractWorkloadFeatures(workload, weights);
					const std::vector<SchemaCandidate> benchmarkSchemas = selectBenchmarkSchemas(
						resolvedOptions,
						dataset,
						workload,
						schemas,
						rankModel);

					std::cout << "    workload: " << workload._name << " (" << workload._numQueries << " queries)\n";
					if (benchmarkSchemas.size() != schemas.size())
						std::cout << "      benchmarking " << benchmarkSchemas.size() << " / " << schemas.size() << " schemas after surrogate pruning\n";

					for (const SchemaCandidate& schema : benchmarkSchemas)
					{
						CudaIndexCacheEntry* cudaEntry = useCudaEvaluator(resolvedOptions)
							? &cudaCache[datasetContext._dataset]
							: nullptr;
						SchemaSearchRecord record = benchmarkSchemaCandidateCached(
							dataset,
							datasetContext._features,
							workload,
							workloadFeatures,
							datasetContext._preparedWorkloads[workloadIndex],
							schema,
							resolvedOptions,
							cudaEntry);
						records.push_back(record);
						emitProgress(resolvedOptions, record);

						std::cout << "      " << record._schemaName
							<< ": score " << record._score
							<< ", avg " << record._queryMetrics._averageLatencyMs
							<< " ms, build " << record._buildMetrics._buildTimeMs
							<< " ms";
						if (record._backend == "cuda")
							std::cout << ", gpu build " << record._gpuBuildMs
								<< " ms, upload " << record._gpuUploadMs
								<< " ms, gpu query " << record._gpuQueryMs << " ms";
						else if (useCudaEvaluator(resolvedOptions))
							std::cout << ", cpu fallback (GPU-unsupported feature)";
						std::cout << '\n';
					}
				}
			}
		}
	}

	runMultiSeedConfirmation(records, datasets, workloads, resolvedOptions);
	annotateBaselineComparisons(records);
	annotateRankingConfidence(records);
	reportProxyLatencyCorrelation(records, resolvedOptions._proxyCorrelationCsvPath);
	reportEstimatedCostCorrelation(records, std::string());
	reportParetoKnee(records);
	if (resolvedOptions._deepNestedSearch)
		reportDeepNestedOutcome(records);
	reportMetricSummaries(records, resolvedOptions);
	writeSchemaExplainReport(resolvedOptions._explainReportPath, records);
	writeSearchRows(resolvedOptions._csvPath, records);
	writeBestRows(resolvedOptions._bestCsvPath, records);
	writeParetoRows(resolvedOptions._paretoCsvPath, records);
	if (resolvedOptions._verifyParity)
		runCpuGpuParityCheck(datasets, workloads, resolvedOptions);

	std::cout << "  wrote rows: " << records.size() << '\n';
	if (!resolvedOptions._csvPath.empty())
		std::cout << "  csv: " << resolvedOptions._csvPath << '\n';
	if (!resolvedOptions._bestCsvPath.empty())
		std::cout << "  best csv: " << resolvedOptions._bestCsvPath << '\n';
	if (!resolvedOptions._paretoCsvPath.empty())
	{
		const std::vector<SchemaSearchRecord> front = selectParetoRecords(records);
		std::cout << "  pareto csv: " << resolvedOptions._paretoCsvPath
			<< " (" << front.size() << " non-dominated rows)\n";
	}
	if (!resolvedOptions._explainReportPath.empty())
		std::cout << "  explain report: " << resolvedOptions._explainReportPath << '\n';

	if (resolvedOptions._scoreCache != nullptr && resolvedOptions._scoreCache->enabled())
	{
		const size_t hits = resolvedOptions._scoreCache->hitCount();
		const size_t misses = resolvedOptions._scoreCache->missCount();
		std::cout << "  score cache: " << hits << " hits, " << misses << " misses (entries now " << resolvedOptions._scoreCache->entryCount() << ")\n";
	}

	if (resolvedOptions._pauseAtEnd)
		std::system("pause");

	return 0;
}

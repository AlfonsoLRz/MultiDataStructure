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

namespace
{
	constexpr double EPSILON = 1e-9;

	const std::vector<std::string>& defaultSchemaPaths()
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

	const std::vector<std::string>& defaultWorkloadPaths()
	{
		static const std::vector<std::string> paths = {
			"configs/workloads/volume_small_medium.json",
		};
		return paths;
	}

	struct SearchDataset
	{
		std::string name;
		std::string source;
		PointCloud cloud;
	};

	struct WorkloadRun
	{
		Experiments::QueryMetrics metrics;
		size_t rangeQueries = 0;
		size_t countRangeQueries = 0;
		size_t radiusQueries = 0;
		size_t knnQueries = 0;
		double gpuQueryMs = 0.0;
	};

	enum class PreparedQueryKind
	{
		Range,
		Radius,
		Knn,
	};

	struct PreparedCpuQuery
	{
		PreparedQueryKind kind = PreparedQueryKind::Range;
		AABB bounds;
		glm::vec3 center = glm::vec3(0.0f);
		float radius = 0.0f;
	};

	struct PreparedWorkload
	{
		std::vector<PreparedCpuQuery> cpuQueries;
		std::vector<PointGpu::Query> cudaQueries;
		size_t rangeQueries = 0;
		size_t countRangeQueries = 0;
		size_t radiusQueries = 0;
		size_t knnQueries = 0;
	};

	struct DatasetContext
	{
		const SearchDataset* dataset = nullptr;
		Experiments::PointCloudFeatures features;
		std::vector<PreparedWorkload> preparedWorkloads;
	};

	struct EvaluatedCandidate
	{
		Experiments::SchemaCandidate candidate;
		std::vector<Experiments::SchemaSearchRecord> records;
		double aggregateScore = std::numeric_limits<double>::infinity();
		// Phase B4 NSGA-II tagging. -1 = not ranked yet; otherwise 0 = current Pareto front,
		// 1 = next front after removing rank 0, etc. crowdingDistance is the sum of normalized
		// objective gaps to neighbors within the same front (used to break ties; larger = more
		// isolated, i.e. more diverse).
		int paretoFront = -1;
		double crowdingDistance = 0.0;
	};

	template <typename TIndex>
	struct CudaCachedBuilder
	{
		std::unique_ptr<TIndex> index;
		std::string lastSignature;
		PointGpu::BuildResult lastResult;
		bool hasResult = false;
	};

	struct CudaIndexCacheEntry
	{
		CudaCachedBuilder<PointGpu::BIH> bih;
		CudaCachedBuilder<PointGpu::HGrid> hgrid;
		CudaCachedBuilder<PointGpu::KDTree> kdTree;
		CudaCachedBuilder<PointGpu::LBVH> lbvh;
		CudaCachedBuilder<PointGpu::MixedTree> mixedTree;
		CudaCachedBuilder<PointGpu::Octree> octree;
		CudaCachedBuilder<PointGpu::QuadTree> quadTree;
		CudaCachedBuilder<PointGpu::RegularGrid> regularGrid;
		size_t buildHits = 0;
		size_t buildMisses = 0;
	};

	using CudaIndexCache = std::unordered_map<const SearchDataset*, CudaIndexCacheEntry>;

	template <typename TIndex>
	PointGpu::BuildResult cudaBuildOrReuse(
		CudaCachedBuilder<TIndex>& slot,
		const std::string& currentSignature,
		const PointCloud& cloud,
		const SchemaConfig& schemaConfig,
		const PointGpu::Options& cudaOptions,
		CudaIndexCacheEntry& parentEntry,
		TIndex*& outIndex)
	{
		if (!slot.index)
			slot.index = std::make_unique<TIndex>();
		outIndex = slot.index.get();

		if (slot.hasResult && slot.lastSignature == currentSignature)
		{
			++parentEntry.buildHits;
			return slot.lastResult;
		}

		slot.lastResult = slot.index->build(cloud, schemaConfig, cudaOptions);
		slot.lastSignature = currentSignature;
		slot.hasResult = true;
		++parentEntry.buildMisses;
		return slot.lastResult;
	}

	bool pathExists(const std::filesystem::path& path)
	{
		std::error_code error;
		return std::filesystem::exists(path, error);
	}

	std::filesystem::path resolveExistingPath(const std::string& filename)
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

	size_t asSize(const boost::json::object& object, const char* key, size_t fallback)
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

	double asDouble(const boost::json::object& object, const char* key, double fallback)
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

	std::string asString(const boost::json::object& object, const char* key, const std::string& fallback = {})
	{
		if (const boost::json::value* value = object.if_contains(key))
		{
			if (value->is_string())
				return std::string(value->as_string().c_str());
		}

		return fallback;
	}

	double elapsedMilliseconds(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
	{
		return std::chrono::duration<double, std::milli>(end - begin).count();
	}

	float randomFloat(std::mt19937& rng, float minValue, float maxValue)
	{
		if (minValue >= maxValue)
			return minValue;

		std::uniform_real_distribution<float> distribution(minValue, maxValue);
		return distribution(rng);
	}

	glm::vec3 randomPointInBounds(std::mt19937& rng, const AABB& bounds)
	{
		const glm::vec3 min = bounds.min();
		const glm::vec3 max = bounds.max();
		return glm::vec3(
			randomFloat(rng, min.x, max.x),
			randomFloat(rng, min.y, max.y),
			randomFloat(rng, min.z, max.z));
	}

	void normalizeScaleRange(double& minScale, double& maxScale)
	{
		minScale = std::max(0.0, minScale);
		maxScale = std::max(0.0, maxScale);
		if (maxScale < minScale)
			std::swap(minScale, maxScale);
	}

	void parseScaleRange(const boost::json::object& object, const char* key, double& minScale, double& maxScale)
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

	AABB randomQueryBox(std::mt19937& rng, const PointCloud& cloud, const Experiments::WorkloadProfile& profile)
	{
		const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.001f));
		const float scale = randomFloat(rng, static_cast<float>(profile.rangeScaleMin), static_cast<float>(profile.rangeScaleMax));
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

	float randomQueryRadius(std::mt19937& rng, const PointCloud& cloud, const Experiments::WorkloadProfile& profile)
	{
		const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.001f));
		const float largestRange = std::max({ range.x, range.y, range.z, 1.0f });
		return largestRange * randomFloat(rng, static_cast<float>(profile.radiusScaleMin), static_cast<float>(profile.radiusScaleMax));
	}

	std::string lowerCopy(std::string value)
	{
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return value;
	}

	bool useCudaEvaluator(const Experiments::SchemaSearchOptions& options)
	{
		const std::string evaluator = lowerCopy(options.evaluator);
		return evaluator == "cuda" || evaluator == "gpu";
	}

	bool isRegularGridBuilder(const std::string& builder)
	{
		return builder == "regular_grid" || builder == "regulargrid" || builder == "grid" || builder == "uniform_grid";
	}

	bool isHGridBuilder(const std::string& builder)
	{
		return builder == "hgrid" || builder == "hierarchical_grid" || builder == "hierarchicalgrid";
	}

	bool isBIHBuilder(const std::string& builder)
	{
		return builder == "bih" || builder == "interval_hierarchy" || builder == "binary_interval_hierarchy";
	}

	bool isKDTreeBuilder(const std::string& builder)
	{
		return builder == "kdtree" || builder == "kd_tree" || builder == "kd";
	}

	bool isOctreeBuilder(const std::string& builder)
	{
		return builder == "octree" || builder == "ot" ||
			builder == "karras_octree" || builder == "morton_octree" ||
			builder == "octree_karras" || builder == "octree_morton";
	}

	bool isKarrasOctreeBuilder(const std::string& builder)
	{
		return builder == "karras_octree" || builder == "morton_octree" ||
			builder == "octree_karras" || builder == "octree_morton";
	}

	bool isQuadTreeBuilder(const std::string& builder)
	{
		return builder == "quadtree" || builder == "quad_tree" || builder == "qt";
	}

	bool isMixedBuilder(const std::string& builder)
	{
		return builder == "mixed" || builder == "hybrid";
	}

	std::string canonicalCudaBuilder(std::string builder)
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

	std::string cudaBuilderDisplayName(const std::string& builder)
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

	std::string cudaDeviceDescription(const PointGpu::Options& cudaOptions)
	{
		int count = 0;
		const cudaError_t countResult = cudaGetDeviceCount(&count);
		if (countResult != cudaSuccess)
			return std::string("unavailable (") + cudaGetErrorString(countResult) + ")";
		if (count <= 0)
			return "unavailable (no CUDA devices)";

		int device = 0;
		if (cudaOptions.device >= 0)
		{
			device = std::min(cudaOptions.device, count - 1);
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
		if (cudaOptions.device >= 0)
		{
			if (cudaOptions.device != device)
				out << cudaOptions.device << " -> ";
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

	int resolvedCudaDevice(const PointGpu::Options& cudaOptions)
	{
		int count = 0;
		const cudaError_t countResult = cudaGetDeviceCount(&count);
		if (countResult != cudaSuccess)
			throw std::runtime_error(std::string("CUDA device query failed: ") + cudaGetErrorString(countResult));
		if (count <= 0)
			throw std::runtime_error("CUDA evaluator requested, but no CUDA devices are available.");

		if (cudaOptions.device >= 0)
			return std::min(cudaOptions.device, count - 1);

		int device = 0;
		const cudaError_t currentResult = cudaGetDevice(&device);
		if (currentResult != cudaSuccess || device < 0 || device >= count)
			return 0;
		return device;
	}

	double warmUpCudaDevice(const PointGpu::Options& cudaOptions)
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

	// Queries total VRAM on the selected (or default) CUDA device. Returns 0 if CUDA is
	// unavailable or the device cannot be queried. Used both for the schema-search header line
	// and to auto-default `memoryBudgetMb` when the caller leaves it at 0.
	size_t cudaTotalVramMb(int deviceHint)
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

	PointGpu::Options cudaOptionsFrom(const Experiments::SchemaSearchOptions& options)
	{
		PointGpu::Options cudaOptions;
		cudaOptions.device = options.cuda.device;
		cudaOptions.builder = canonicalCudaBuilder(options.cuda.builder);
		cudaOptions.queryBatchSize = options.cuda.queryBatchSize;
		cudaOptions.memoryBudgetMb = options.cuda.memoryBudgetMb;
		// Auto-default the budget to 75% of total VRAM when the caller leaves it at 0. The
		// HGrid aggregate pre-check and the RegularGrid per-level cap both consult this value,
		// so a sensible default protects users who haven't manually set it without preventing
		// the rest of the GPU stack from grabbing the remaining headroom.
		if (cudaOptions.memoryBudgetMb == 0)
		{
			const size_t totalMb = cudaTotalVramMb(cudaOptions.device);
			if (totalMb > 0)
				cudaOptions.memoryBudgetMb = (totalMb * 3) / 4;
		}
		return cudaOptions;
	}

	std::string datasetNameFromPath(const std::string& inputPath)
	{
		const std::filesystem::path path(inputPath);
		const std::string stem = path.stem().string();
		return stem.empty() ? "points" : stem;
	}

	std::string normalizedLevelTypeName(std::string value)
	{
		value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
			return std::isspace(c) || c == '_' || c == '-';
		}), value.end());
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return value;
	}

	bool isBIHLevelName(const std::string& typeName)
	{
		const std::string normalized = normalizedLevelTypeName(typeName);
		return normalized == "bih" || normalized == "binaryintervalhierarchy" || normalized == "intervalhierarchy";
	}

	bool isKarrasOctreeLevelName(const std::string& typeName)
	{
		const std::string normalized = normalizedLevelTypeName(typeName);
		return normalized == "karrasoctree" || normalized == "mortonoctree" || normalized == "octreekarras" || normalized == "octreemorton";
	}

	bool isLBVHLevelName(const std::string& typeName)
	{
		const std::string normalized = normalizedLevelTypeName(typeName);
		return normalized == "lbvh" || normalized == "linearbvh";
	}

	bool isRegularGridLevelName(const std::string& typeName)
	{
		const std::string normalized = normalizedLevelTypeName(typeName);
		return normalized == "regulargrid" || normalized == "uniformgrid" || normalized == "grid" || normalized == "grid3d";
	}

	bool isHGridLevelName(const std::string& typeName)
	{
		const std::string normalized = normalizedLevelTypeName(typeName);
		return normalized == "hgrid" || normalized == "hierarchicalgrid" || normalized == "hierarchicalgrid3d";
	}

	std::string schemaTypeShortName(const SchemaLevelConfig& level)
	{
		if (isBIHLevelName(level.typeName))
			return "bih";
		if (isKarrasOctreeLevelName(level.typeName))
			return "kot";
		if (isLBVHLevelName(level.typeName))
			return "lbvh";
		if (isRegularGridLevelName(level.typeName))
			return "rg";
		if (isHGridLevelName(level.typeName))
			return "hg";

		switch (level.type)
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

	std::string levelJsonTypeName(const SchemaLevelConfig& level)
	{
		return level.typeName.empty() ? Config::dataStructureLevelName(level.type) : level.typeName;
	}

	std::string randomTypeNameForBase(MultiDataStructure::DataStructureLevel type, std::mt19937& rng)
	{
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

	size_t clampPowerOfTwo(size_t value, size_t minValue, size_t maxValue)
	{
		value = std::max<size_t>(1, value);
		size_t power = 1;
		while (power < value && power < (std::numeric_limits<size_t>::max() / 2))
			power *= 2;
		return std::clamp(power, minValue, maxValue);
	}

	size_t randomPowerOfTwo(std::mt19937& rng, size_t minValue, size_t maxValue)
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

	MultiDataStructure::DataStructureLevel randomStructureType(
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
	void sortUniqueValues(std::vector<T>& values)
	{
		std::sort(values.begin(), values.end());
		values.erase(std::unique(values.begin(), values.end()), values.end());
	}

	void addUniqueSize(std::vector<size_t>& values, size_t value)
	{
		if (value > 0)
			values.push_back(value);
	}

	void addUniqueDouble(std::vector<double>& values, double value)
	{
		if (std::isfinite(value) && value > 0.0)
			values.push_back(value);
	}

	template <typename T>
	T quantileValue(std::vector<T> values, double quantile)
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

	size_t nearestPowerOfTwoAtLeastOne(size_t value)
	{
		value = std::max<size_t>(1, value);
		size_t power = 1;
		while (power < value && power < (std::numeric_limits<size_t>::max() / 2))
			power *= 2;
		const size_t lower = power > 1 ? power / 2 : power;
		return (value - lower) <= (power - value) ? lower : power;
	}

	void addPointThresholdFamily(std::vector<size_t>& values, size_t rawValue)
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

	const std::vector<size_t>& fallbackPointThresholds()
	{
		static const std::vector<size_t> values = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
		return values;
	}

	const std::vector<double>& fallbackHeightRatioThresholds()
	{
		static const std::vector<double> values = { 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00 };
		return values;
	}

	const std::vector<double>& vectorOrFallback(const std::vector<double>& values, const std::vector<double>& fallback)
	{
		return values.empty() ? fallback : values;
	}

	const std::vector<size_t>& vectorOrFallback(const std::vector<size_t>& values, const std::vector<size_t>& fallback)
	{
		return values.empty() ? fallback : values;
	}

	size_t chooseSizeValue(
		std::mt19937& rng,
		const std::vector<size_t>& values,
		const std::vector<size_t>& fallback)
	{
		const std::vector<size_t>& source = vectorOrFallback(values, fallback);
		std::uniform_int_distribution<size_t> distribution(0, source.size() - 1);
		return source[distribution(rng)];
	}

	double chooseDoubleValue(
		std::mt19937& rng,
		const std::vector<double>& values,
		const std::vector<double>& fallback)
	{
		const std::vector<double>& source = vectorOrFallback(values, fallback);
		std::uniform_int_distribution<size_t> distribution(0, source.size() - 1);
		return source[distribution(rng)];
	}

	template <typename T>
	T nearbyDomainValue(const std::vector<T>& values, T current, std::mt19937& rng)
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

	void appendOptionalSize(std::ostringstream& output, const char* jsonName, const std::optional<size_t>& value, bool& first)
	{
		if (!value.has_value())
			return;

		output << (first ? "" : ",\n") << "        \"" << jsonName << "\": " << value.value();
		first = false;
	}

	void appendOptionalDouble(std::ostringstream& output, const char* jsonName, const std::optional<double>& value, bool& first)
	{
		if (!value.has_value())
			return;

		output << (first ? "" : ",\n") << "        \"" << jsonName << "\": " << value.value();
		first = false;
	}

	void appendConditionSignature(std::ostringstream& output, const SchemaLevelCondition& condition)
	{
		if (condition.empty())
			return;

		output << "c";
		if (condition.minPoints) output << "p" << condition.minPoints.value();
		if (condition.maxPoints) output << "P" << condition.maxPoints.value();
		if (condition.minDensity) output << "d" << static_cast<size_t>(condition.minDensity.value() * 1000.0);
		if (condition.maxDensity) output << "D" << static_cast<size_t>(condition.maxDensity.value() * 1000.0);
		if (condition.minHeightRatio) output << "h" << static_cast<size_t>(condition.minHeightRatio.value() * 1000.0);
		if (condition.maxHeightRatio) output << "H" << static_cast<size_t>(condition.maxHeightRatio.value() * 1000.0);
		if (condition.minExtentX) output << "x" << static_cast<size_t>(condition.minExtentX.value() * 1000.0);
		if (condition.maxExtentX) output << "X" << static_cast<size_t>(condition.maxExtentX.value() * 1000.0);
		if (condition.minExtentY) output << "y" << static_cast<size_t>(condition.minExtentY.value() * 1000.0);
		if (condition.maxExtentY) output << "Y" << static_cast<size_t>(condition.maxExtentY.value() * 1000.0);
		if (condition.minExtentZ) output << "z" << static_cast<size_t>(condition.minExtentZ.value() * 1000.0);
		if (condition.maxExtentZ) output << "Z" << static_cast<size_t>(condition.maxExtentZ.value() * 1000.0);
		if (condition.minAnisotropy) output << "a" << static_cast<size_t>(condition.minAnisotropy.value() * 1000.0);
		if (condition.maxAnisotropy) output << "A" << static_cast<size_t>(condition.maxAnisotropy.value() * 1000.0);
		if (condition.minOccupancyEntropy) output << "e" << static_cast<size_t>(condition.minOccupancyEntropy.value() * 1000.0);
		if (condition.maxOccupancyEntropy) output << "E" << static_cast<size_t>(condition.maxOccupancyEntropy.value() * 1000.0);
	}

	size_t conditionFieldCount(const SchemaLevelCondition& condition)
	{
		size_t count = 0;
		count += condition.minPoints.has_value() ? 1 : 0;
		count += condition.maxPoints.has_value() ? 1 : 0;
		count += condition.minDensity.has_value() ? 1 : 0;
		count += condition.maxDensity.has_value() ? 1 : 0;
		count += condition.minHeightRatio.has_value() ? 1 : 0;
		count += condition.maxHeightRatio.has_value() ? 1 : 0;
		count += condition.minExtentX.has_value() ? 1 : 0;
		count += condition.maxExtentX.has_value() ? 1 : 0;
		count += condition.minExtentY.has_value() ? 1 : 0;
		count += condition.maxExtentY.has_value() ? 1 : 0;
		count += condition.minExtentZ.has_value() ? 1 : 0;
		count += condition.maxExtentZ.has_value() ? 1 : 0;
		count += condition.minAnisotropy.has_value() ? 1 : 0;
		count += condition.maxAnisotropy.has_value() ? 1 : 0;
		count += condition.minOccupancyEntropy.has_value() ? 1 : 0;
		count += condition.maxOccupancyEntropy.has_value() ? 1 : 0;
		return count;
	}

	size_t schemaConditionalLevels(const SchemaConfig& schema)
	{
		size_t count = 0;
		for (const SchemaLevelConfig& level : schema.levels)
		{
			if (!level.condition.empty())
				++count;
		}
		return count;
	}

	size_t schemaConditionFields(const SchemaConfig& schema)
	{
		size_t count = 0;
		for (const SchemaLevelConfig& level : schema.levels)
			count += conditionFieldCount(level.condition);
		return count;
	}

	std::string schemaConditionSummary(const SchemaConfig& schema)
	{
		std::ostringstream output;
		bool first = true;
		for (const SchemaLevelConfig& level : schema.levels)
		{
			if (level.condition.empty())
				continue;

			if (!first)
				output << ';';
			first = false;
			output << schemaTypeShortName(level) << ':';
			appendConditionSignature(output, level.condition);
		}
		return output.str();
	}

	struct ActiveTypeAccumulator
	{
		size_t nodes = 0;
		size_t leafPoints = 0;
	};

	struct ActiveStructureStats
	{
		size_t activeStructureTypes = 0;
		double nestedActiveFraction = 0.0;
		std::string summary;
	};

	const SchemaLevelConfig* schemaLevelForNode(const SchemaConfig& schema, const PointSpatialIndex::Node& node)
	{
		if (schema.levels.empty())
			return nullptr;
		const size_t totalLevels = std::max<size_t>(1, schema.totalLevels());
		const size_t schemaDepth = std::min(node.schemaDepth, totalLevels - 1);
		return &schema.levelForDepth(schemaDepth);
	}

	std::string activeTypeNameForNode(const SchemaConfig& schema, const PointSpatialIndex::Node& node)
	{
		const SchemaLevelConfig* level = schemaLevelForNode(schema, node);
		if (level)
			return schemaTypeShortName(*level);

		SchemaLevelConfig fallback;
		fallback.type = node.type;
		fallback.typeName = Config::dataStructureLevelName(node.type);
		return schemaTypeShortName(fallback);
	}

	void collectActiveStructureStatsRecursive(
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
		++accumulator.nodes;
		++totalNodes;
		if (node->isLeaf())
		{
			accumulator.leafPoints += node->pointIndices.size();
			totalLeafPoints += node->pointIndices.size();
		}

		for (const std::unique_ptr<PointSpatialIndex::Node>& child : node->children)
			collectActiveStructureStatsRecursive(child.get(), schema, byType, totalNodes, totalLeafPoints);
	}

	ActiveStructureStats collectActiveStructureStats(const PointSpatialIndex::Node* root, const SchemaConfig& schema)
	{
		ActiveStructureStats stats;
		if (!root)
			return stats;

		std::map<std::string, ActiveTypeAccumulator> byType;
		size_t totalNodes = 0;
		size_t totalLeafPoints = 0;
		collectActiveStructureStatsRecursive(root, schema, byType, totalNodes, totalLeafPoints);

		stats.activeStructureTypes = byType.size();
		const std::string primaryType = schema.levels.empty() ? activeTypeNameForNode(schema, *root) : schemaTypeShortName(schema.levels.front());
		size_t nonPrimaryNodes = 0;
		size_t nonPrimaryLeafPoints = 0;

		std::ostringstream summary;
		bool first = true;
		for (const auto& [typeName, accumulator] : byType)
		{
			if (!first)
				summary << ';';
			first = false;
			summary << typeName << ":nodes=" << accumulator.nodes << "|points=" << accumulator.leafPoints;

			if (typeName != primaryType)
			{
				nonPrimaryNodes += accumulator.nodes;
				nonPrimaryLeafPoints += accumulator.leafPoints;
			}
		}
		stats.summary = summary.str();

		const double pointFraction = totalLeafPoints > 0
			? static_cast<double>(nonPrimaryLeafPoints) / static_cast<double>(totalLeafPoints)
			: 0.0;
		const double nodeFraction = totalNodes > 0
			? static_cast<double>(nonPrimaryNodes) / static_cast<double>(totalNodes)
			: 0.0;
		stats.nestedActiveFraction = std::max(pointFraction, nodeFraction);
		return stats;
	}

	ActiveStructureStats staticSchemaStructureStats(const SchemaConfig& schema)
	{
		ActiveStructureStats stats;
		std::set<std::string> types;
		for (const SchemaLevelConfig& level : schema.levels)
			types.insert(schemaTypeShortName(level));
		stats.activeStructureTypes = types.size();

		std::ostringstream summary;
		bool first = true;
		for (const std::string& typeName : types)
		{
			if (!first)
				summary << ';';
			first = false;
			summary << typeName << ":schema";
		}
		stats.summary = summary.str();
		return stats;
	}

	SchemaLevelCondition randomLevelCondition(
		std::mt19937& rng,
		const SchemaLevelConfig& level,
		size_t minLeaf,
		size_t maxLeaf,
		const Experiments::ConditionDomain* domain)
	{
		SchemaLevelCondition condition;
		if (domain && !domain->pointThresholds.empty())
		{
			condition.minPoints = chooseSizeValue(rng, domain->pointThresholds, fallbackPointThresholds());
		}
		else
		{
			const size_t maxPointThreshold = std::max<size_t>(minLeaf * 2, std::min<size_t>(maxLeaf * 8, 1 << 20));
			condition.minPoints = randomPowerOfTwo(rng, std::max<size_t>(minLeaf, 32), maxPointThreshold);
		}

		if (level.type == MultiDataStructure::OctreeNode)
		{
			if (isRegularGridLevelName(level.typeName) || isHGridLevelName(level.typeName))
			{
				if (domain && !domain->densityThresholds.empty())
					condition.minDensity = chooseDoubleValue(rng, domain->densityThresholds, {});
			}
			else
			{
				condition.minHeightRatio = chooseDoubleValue(
					rng,
					domain ? domain->heightRatioThresholds : std::vector<double>{},
					fallbackHeightRatioThresholds());
			}
		}
		else if (level.type == MultiDataStructure::QuadTreeNode)
		{
			condition.maxHeightRatio = chooseDoubleValue(
				rng,
				domain ? domain->heightRatioThresholds : std::vector<double>{},
				fallbackHeightRatioThresholds());
		}

		// Phase B3: opportunistically add an anisotropy gate when the domain has one. Sampled
		// with probability 0.5 to keep half the generated candidates anisotropy-free, so the
		// optimizer can compare branches that gate on shape vs. branches that don't.
		if (domain && !domain->anisotropyThresholds.empty())
		{
			std::bernoulli_distribution coin(0.5);
			if (coin(rng))
			{
				std::bernoulli_distribution minOrMax(0.5);
				if (minOrMax(rng))
					condition.minAnisotropy = chooseDoubleValue(rng, domain->anisotropyThresholds, {});
				else
					condition.maxAnisotropy = chooseDoubleValue(rng, domain->anisotropyThresholds, {});
			}
		}

		// Same pattern for occupancy entropy: 0.5 chance to gate, 0.5 between min/max. Cheap to
		// add and gives the GA another shape-vs-uniform discriminator without ballooning the
		// search space (the matchesCondition fast-path skips entropy when no threshold is set).
		if (domain && !domain->occupancyEntropyThresholds.empty())
		{
			std::bernoulli_distribution coin(0.5);
			if (coin(rng))
			{
				std::bernoulli_distribution minOrMax(0.5);
				if (minOrMax(rng))
					condition.minOccupancyEntropy = chooseDoubleValue(rng, domain->occupancyEntropyThresholds, {});
				else
					condition.maxOccupancyEntropy = chooseDoubleValue(rng, domain->occupancyEntropyThresholds, {});
			}
		}

		return condition;
	}

	std::string schemaSignature(const SchemaConfig& schema)
	{
		std::ostringstream output;
		for (size_t i = 0; i < schema.levels.size(); ++i)
		{
			const SchemaLevelConfig& level = schema.levels[i];
			if (i > 0)
				output << '_';
			output
				<< schemaTypeShortName(level)
				<< level.numLevels
				<< "l"
				<< level.leafCapacity;
			// Phase B2: include the axis policy in the signature so two KDTree/BIH genomes that
			// differ only in policy land under distinct cache keys and seenSignatures slots. We
			// only emit the suffix when the policy meaningfully changes behavior — empty or the
			// long-standing default is treated as "no suffix" to keep legacy signatures stable.
			if (!level.axisPolicy.empty() && level.axisPolicy != "median_longest_axis")
			{
				if (level.axisPolicy == "round_robin")
					output << "ap=rr";
				else if (level.axisPolicy == "center_longest_axis")
					output << "ap=cl";
				else
					output << "ap=other";
			}
			appendConditionSignature(output, level.condition);
		}
		return output.str();
	}

	// Coarser key than schemaSignature: only the level-type sequence. Two schemas with the same
	// topology hash differ only in numeric details (leaf caps, depths, conditions). Used by the
	// diversity audit so we can ask "how many qualitatively distinct shapes did the optimizer
	// actually try, regardless of how many leaf-cap variants of each one it stamped out?".
	std::string schemaTopologyKey(const SchemaConfig& schema)
	{
		std::ostringstream output;
		for (size_t i = 0; i < schema.levels.size(); ++i)
		{
			if (i > 0)
				output << ':';
			output << schemaTypeShortName(schema.levels[i]);
		}
		return output.str();
	}

	std::string schemaConfigToJson(const SchemaConfig& schema)
	{
		std::ostringstream output;
		output << "{\n";
		output << "  \"name\": \"" << schema.name << "\",\n";
		output << "  \"levels\": [\n";
		for (size_t i = 0; i < schema.levels.size(); ++i)
		{
			const SchemaLevelConfig& level = schema.levels[i];
			output << "    {\n";
			output << "      \"type\": \"" << levelJsonTypeName(level) << "\",\n";
			output << "      \"numLevels\": " << level.numLevels << ",\n";
			output << "      \"leafCapacity\": " << level.leafCapacity << ",\n";
			output << "      \"minPointsToSplit\": " << level.minPrimitivesToSplit;
			if (!level.axisPolicy.empty())
				output << ",\n      \"axisPolicy\": \"" << level.axisPolicy << "\"";
			if (!level.condition.empty())
			{
				output << ",\n      \"condition\": {\n";
				bool first = true;
				appendOptionalSize(output, "minPoints", level.condition.minPoints, first);
				appendOptionalSize(output, "maxPoints", level.condition.maxPoints, first);
				appendOptionalDouble(output, "minDensity", level.condition.minDensity, first);
				appendOptionalDouble(output, "maxDensity", level.condition.maxDensity, first);
				appendOptionalDouble(output, "minHeightRatio", level.condition.minHeightRatio, first);
				appendOptionalDouble(output, "maxHeightRatio", level.condition.maxHeightRatio, first);
				appendOptionalDouble(output, "minExtentX", level.condition.minExtentX, first);
				appendOptionalDouble(output, "maxExtentX", level.condition.maxExtentX, first);
				appendOptionalDouble(output, "minExtentY", level.condition.minExtentY, first);
				appendOptionalDouble(output, "maxExtentY", level.condition.maxExtentY, first);
				appendOptionalDouble(output, "minExtentZ", level.condition.minExtentZ, first);
				appendOptionalDouble(output, "maxExtentZ", level.condition.maxExtentZ, first);
				appendOptionalDouble(output, "minAnisotropy", level.condition.minAnisotropy, first);
				appendOptionalDouble(output, "maxAnisotropy", level.condition.maxAnisotropy, first);
				appendOptionalDouble(output, "minOccupancyEntropy", level.condition.minOccupancyEntropy, first);
				appendOptionalDouble(output, "maxOccupancyEntropy", level.condition.maxOccupancyEntropy, first);
				output << "\n      }";
			}
			output << '\n';
			output << "    }" << (i + 1 < schema.levels.size() ? "," : "") << "\n";
		}
		output << "  ],\n";
		output << "  \"buildPolicy\": {\n";
		output << "    \"maxDepth\": " << schema.buildPolicy.maxDepth << ",\n";
		output << "    \"leafCapacity\": " << schema.buildPolicy.leafCapacity << ",\n";
		output << "    \"minPointsToSplit\": " << schema.buildPolicy.minPrimitivesToSplit << ",\n";
		output << "    \"collapseSingleChild\": " << (schema.buildPolicy.collapseSingleChild ? "true" : "false") << ",\n";
		output << "    \"removeEmptyNodes\": " << (schema.buildPolicy.removeEmptyNodes ? "true" : "false") << ",\n";
		output << "    \"allowOverlapDuplication\": " << (schema.buildPolicy.allowOverlapDuplication ? "true" : "false") << "\n";
		output << "  }\n";
		output << "}\n";
		return output.str();
	}

	Experiments::SchemaCandidate materializeGeneratedSchema(
		SchemaConfig schema,
		const std::string& namePrefix,
		const std::string& outputDirectory)
	{
		const std::string signature = schemaSignature(schema);
		schema.name = namePrefix + "_" + signature;

		Experiments::SchemaCandidate candidate;
		candidate.name = schema.name;
		candidate.config = schema;
		candidate.generated = true;

		if (!outputDirectory.empty())
		{
			const std::filesystem::path schemaPath = std::filesystem::path(outputDirectory) / (schema.name + ".json");
			if (schemaPath.has_parent_path())
				std::filesystem::create_directories(schemaPath.parent_path());

			std::ofstream output(schemaPath);
			if (!output.is_open())
				throw std::runtime_error("Unable to write generated schema: " + schemaPath.string());
			output << schemaConfigToJson(schema);
			candidate.path = schemaPath.string();
		}
		else
		{
			candidate.path = "generated:" + schema.name;
		}

		return candidate;
	}

	SchemaLevelConfig randomLevelConfig(
		std::mt19937& rng,
		const Experiments::SchemaGenerationOptions& options,
		std::optional<MultiDataStructure::DataStructureLevel> previousType = std::nullopt)
	{
		const size_t minLeaf = std::max<size_t>(1, std::min(options.minLeafCapacity, options.maxLeafCapacity));
		const size_t maxLeaf = std::max(minLeaf, options.maxLeafCapacity);

		SchemaLevelConfig level;
		level.type = randomStructureType(rng, previousType);
		level.typeName = randomTypeNameForBase(level.type, rng);
		level.numLevels = 1;
		level.leafCapacity = randomPowerOfTwo(rng, minLeaf, maxLeaf);
		level.minPrimitivesToSplit = std::max<size_t>(2, level.leafCapacity / 4);
		level.axisPolicy = level.type == MultiDataStructure::KDTreeNode
			? (std::bernoulli_distribution(0.5)(rng) ? "round_robin" : "median_longest_axis")
			: std::string();
		return level;
	}

	// Phase B2: KDTree/BIH levels (both share KDTreeNode under the hood) sample over the axis
	// policy. round_robin alternates X/Y/Z by depth; median_longest_axis is the legacy
	// extent-driven default. Coin-flip selection gives the optimizer both variants per topology
	// without exploding the search space.
	std::string sampleAxisPolicy(std::mt19937& rng, MultiDataStructure::DataStructureLevel type)
	{
		if (type != MultiDataStructure::KDTreeNode)
			return "";
		std::bernoulli_distribution coin(0.5);
		return coin(rng) ? "round_robin" : "median_longest_axis";
	}

	void refreshLevelTypeName(SchemaLevelConfig& level)
	{
		const bool compatibleBIH = level.type == MultiDataStructure::KDTreeNode && isBIHLevelName(level.typeName);
		const bool compatibleKarras = level.type == MultiDataStructure::OctreeNode && isKarrasOctreeLevelName(level.typeName);
		const bool compatibleLBVH = level.type == MultiDataStructure::BvhNode && isLBVHLevelName(level.typeName);
		const bool compatibleRegularGrid = level.type == MultiDataStructure::OctreeNode && isRegularGridLevelName(level.typeName);
		const bool compatibleHGrid = level.type == MultiDataStructure::OctreeNode && isHGridLevelName(level.typeName);
		if (!compatibleBIH && !compatibleKarras && !compatibleLBVH && !compatibleRegularGrid && !compatibleHGrid)
			level.typeName = Config::dataStructureLevelName(level.type);
		// Preserve axisPolicy if it's already set to a recognized value; only reset when the
		// type is not KDTree-family. Lets crossover/mutation children inherit their parent's
		// sampled policy instead of being clobbered back to median_longest_axis.
		if (level.type != MultiDataStructure::KDTreeNode)
			level.axisPolicy = "";
		else if (level.axisPolicy.empty())
			level.axisPolicy = "median_longest_axis";
	}

	void normalizeSchemaForGeneration(SchemaConfig& schema, const Experiments::SchemaGenerationOptions& options)
	{
		const size_t maxDepth = std::max<size_t>(1, options.maxDepth);
		const size_t maxBlocks = std::max<size_t>(1, std::min(options.maxBlocks, maxDepth));
		const size_t minLeaf = std::max<size_t>(1, std::min(options.minLeafCapacity, options.maxLeafCapacity));
		const size_t maxLeaf = std::max(minLeaf, options.maxLeafCapacity);

		if (schema.levels.empty())
		{
			std::mt19937 fallbackRng(options.seed);
			schema.levels.push_back(randomLevelConfig(fallbackRng, options));
		}

		while (schema.levels.size() > maxBlocks)
			schema.levels.pop_back();

		for (size_t i = 0; i < schema.levels.size(); ++i)
		{
			SchemaLevelConfig& level = schema.levels[i];
			level.numLevels = std::max<size_t>(1, level.numLevels);
			level.leafCapacity = clampPowerOfTwo(level.leafCapacity, minLeaf, maxLeaf);
			level.minPrimitivesToSplit = std::clamp(level.minPrimitivesToSplit, static_cast<size_t>(2), std::max<size_t>(2, level.leafCapacity));
			refreshLevelTypeName(level);
			if (i == 0)
				level.condition = {};
		}

		while (schema.totalLevels() > maxDepth && !schema.levels.empty())
		{
			auto reducible = std::find_if(schema.levels.rbegin(), schema.levels.rend(), [](const SchemaLevelConfig& level) {
				return level.numLevels > 1;
			});
			if (reducible != schema.levels.rend())
			{
				--reducible->numLevels;
				continue;
			}

			if (schema.levels.size() > 1)
				schema.levels.pop_back();
			else
				break;
		}

		if (schema.levels.empty())
		{
			std::mt19937 fallbackRng(options.seed);
			schema.levels.push_back(randomLevelConfig(fallbackRng, options));
		}

		schema.buildPolicy.maxDepth = std::min(maxDepth, schema.totalLevels());
		schema.buildPolicy.leafCapacity = schema.levels.front().leafCapacity;
		schema.buildPolicy.minPrimitivesToSplit = std::max<size_t>(2, schema.buildPolicy.leafCapacity / 4);
		schema.buildPolicy.collapseSingleChild = true;
		schema.buildPolicy.removeEmptyNodes = true;
		schema.buildPolicy.allowOverlapDuplication = false;
	}

	void mutateLevelCondition(
		SchemaLevelConfig& level,
		std::mt19937& rng,
		const Experiments::SchemaGenerationOptions& options,
		const Experiments::ConditionDomain* domain)
	{
		const size_t minLeaf = std::max<size_t>(1, std::min(options.minLeafCapacity, options.maxLeafCapacity));
		const size_t maxLeaf = std::max(minLeaf, options.maxLeafCapacity);
		if (level.condition.empty())
		{
			level.condition = randomLevelCondition(rng, level, minLeaf, maxLeaf, domain);
			return;
		}

		std::uniform_int_distribution<int> editDistribution(0, 10);
		switch (editDistribution(rng))
		{
		case 0:
			level.condition = randomLevelCondition(rng, level, minLeaf, maxLeaf, domain);
			break;
		case 1:
			if (domain && !domain->pointThresholds.empty())
				level.condition.minPoints = nearbyDomainValue(domain->pointThresholds, level.condition.minPoints.value_or(domain->pointThresholds.front()), rng);
			else
				level.condition.minPoints = randomPowerOfTwo(rng, minLeaf, std::max<size_t>(minLeaf * 2, std::min<size_t>(maxLeaf * 8, 1 << 20)));
			break;
		case 2:
		{
			const std::vector<double>& heightValues = domain && !domain->heightRatioThresholds.empty()
				? domain->heightRatioThresholds
				: fallbackHeightRatioThresholds();
			if (level.type == MultiDataStructure::QuadTreeNode)
				level.condition.maxHeightRatio = nearbyDomainValue(heightValues, level.condition.maxHeightRatio.value_or(heightValues.front()), rng);
			else
				level.condition.minHeightRatio = nearbyDomainValue(heightValues, level.condition.minHeightRatio.value_or(heightValues.front()), rng);
			break;
		}
		case 3:
			if (domain && !domain->pointThresholds.empty())
				level.condition.maxPoints = nearbyDomainValue(domain->pointThresholds, level.condition.maxPoints.value_or(domain->pointThresholds.back()), rng);
			break;
		case 4:
			if (domain && !domain->densityThresholds.empty())
			{
				if (isRegularGridLevelName(level.typeName) || isHGridLevelName(level.typeName))
					level.condition.minDensity = nearbyDomainValue(domain->densityThresholds, level.condition.minDensity.value_or(domain->densityThresholds.front()), rng);
				else
					level.condition.maxDensity = nearbyDomainValue(domain->densityThresholds, level.condition.maxDensity.value_or(domain->densityThresholds.back()), rng);
			}
			break;
		case 5:
			if (domain)
			{
				std::uniform_int_distribution<int> axisDistribution(0, 2);
				const int axis = axisDistribution(rng);
				if (axis == 0 && !domain->extentXThresholds.empty())
					level.condition.minExtentX = nearbyDomainValue(domain->extentXThresholds, level.condition.minExtentX.value_or(domain->extentXThresholds.front()), rng);
				else if (axis == 1 && !domain->extentYThresholds.empty())
					level.condition.minExtentY = nearbyDomainValue(domain->extentYThresholds, level.condition.minExtentY.value_or(domain->extentYThresholds.front()), rng);
				else if (axis == 2 && !domain->extentZThresholds.empty())
					level.condition.minExtentZ = nearbyDomainValue(domain->extentZThresholds, level.condition.minExtentZ.value_or(domain->extentZThresholds.front()), rng);
			}
			break;
		case 6:
			// Phase B3 anisotropy edit: toggle a min/max anisotropy threshold from the domain.
			if (domain && !domain->anisotropyThresholds.empty())
			{
				std::bernoulli_distribution minOrMax(0.5);
				if (minOrMax(rng))
					level.condition.minAnisotropy = nearbyDomainValue(domain->anisotropyThresholds,
						level.condition.minAnisotropy.value_or(domain->anisotropyThresholds.front()), rng);
				else
					level.condition.maxAnisotropy = nearbyDomainValue(domain->anisotropyThresholds,
						level.condition.maxAnisotropy.value_or(domain->anisotropyThresholds.back()), rng);
			}
			break;
		case 7:
			// Clear the anisotropy gate while keeping other threshold fields intact.
			level.condition.minAnisotropy.reset();
			level.condition.maxAnisotropy.reset();
			break;
		case 8:
			// Toggle a min/max occupancy entropy threshold from the domain.
			if (domain && !domain->occupancyEntropyThresholds.empty())
			{
				std::bernoulli_distribution minOrMax(0.5);
				if (minOrMax(rng))
					level.condition.minOccupancyEntropy = nearbyDomainValue(domain->occupancyEntropyThresholds,
						level.condition.minOccupancyEntropy.value_or(domain->occupancyEntropyThresholds.front()), rng);
				else
					level.condition.maxOccupancyEntropy = nearbyDomainValue(domain->occupancyEntropyThresholds,
						level.condition.maxOccupancyEntropy.value_or(domain->occupancyEntropyThresholds.back()), rng);
			}
			break;
		case 9:
			// Clear the entropy gate.
			level.condition.minOccupancyEntropy.reset();
			level.condition.maxOccupancyEntropy.reset();
			break;
		default:
			level.condition = {};
			break;
		}
	}

	// Single-point crossover on the level list. Splices the first `splitA` levels from parent A
	// with the tail of parent B starting at `splitB`. Lets the search reach topology combinations
	// neither parent had — e.g. parent A is "QuadTree -> Octree" and parent B is "BIH -> LBVH",
	// crossover can produce "QuadTree -> LBVH" which mutation alone would never reach because
	// it requires two simultaneous local edits in a narrow window.
	SchemaConfig crossoverSchemaConfigs(
		const SchemaConfig& parentA,
		const SchemaConfig& parentB,
		std::mt19937& rng,
		const Experiments::SchemaGenerationOptions& options)
	{
		if (parentA.levels.empty()) return parentB;
		if (parentB.levels.empty()) return parentA;

		std::uniform_int_distribution<size_t> cutADist(0, parentA.levels.size());
		std::uniform_int_distribution<size_t> cutBDist(0, parentB.levels.size());
		size_t splitA = cutADist(rng);
		size_t splitB = cutBDist(rng);

		// Reject the degenerate case "take all of A or all of B" — that just returns a parent
		// untouched and burns a child slot for nothing.
		if (splitA == 0 && splitB == 0)
			splitA = 1;
		else if (splitA == parentA.levels.size() && splitB == parentB.levels.size())
			splitB = std::max<size_t>(0, parentB.levels.size() - 1);

		SchemaConfig child;
		child.buildPolicy = parentA.buildPolicy;
		child.levels.reserve(splitA + (parentB.levels.size() - splitB));
		for (size_t i = 0; i < splitA; ++i)
			child.levels.push_back(parentA.levels[i]);
		for (size_t i = splitB; i < parentB.levels.size(); ++i)
			child.levels.push_back(parentB.levels[i]);

		// Hand off to the same normalization the mutation operator uses so depth caps, leaf-cap
		// monotonicity, and root-level condition stripping match the rest of the population.
		normalizeSchemaForGeneration(child, options);
		return child;
	}

	SchemaConfig mutateSchemaConfig(
		const SchemaConfig& parent,
		std::mt19937& rng,
		const Experiments::SchemaGenerationOptions& options,
		const Experiments::EvolutionOptions& evolution,
		const Experiments::ConditionDomain* domain)
	{
		SchemaConfig schema = parent;
		const size_t maxDepth = std::max<size_t>(1, options.maxDepth);
		const size_t maxBlocks = std::max<size_t>(1, std::min(options.maxBlocks, maxDepth));
		const size_t minLeaf = std::max<size_t>(1, std::min(options.minLeafCapacity, options.maxLeafCapacity));
		const size_t maxLeaf = std::max(minLeaf, options.maxLeafCapacity);

		std::bernoulli_distribution extraEdit(std::clamp(evolution.mutationRate, 0.0, 1.0));
		size_t edits = 1;
		while (edits < 5 && extraEdit(rng))
			++edits;

		for (size_t edit = 0; edit < edits; ++edit)
		{
			if (schema.levels.empty())
				schema.levels.push_back(randomLevelConfig(rng, options));

			if (domain && options.conditionalLevels && schema.levels.size() > 1)
			{
				std::bernoulli_distribution conditionEdit(0.5);
				if (conditionEdit(rng))
				{
					std::uniform_int_distribution<size_t> conditionalLevelDistribution(1, schema.levels.size() - 1);
					mutateLevelCondition(schema.levels[conditionalLevelDistribution(rng)], rng, options, domain);
					continue;
				}
			}

			std::uniform_int_distribution<size_t> levelDistribution(0, schema.levels.size() - 1);
			const size_t levelIndex = levelDistribution(rng);
			SchemaLevelConfig& level = schema.levels[levelIndex];

			std::uniform_int_distribution<int> mutationDistribution(0, 8);
			switch (mutationDistribution(rng))
			{
			case 0:
				level.type = randomStructureType(rng, std::nullopt);
				level.typeName = randomTypeNameForBase(level.type, rng);
				level.axisPolicy = sampleAxisPolicy(rng, level.type);
				if (levelIndex == 0)
					level.condition = {};
				break;
			case 1:
			{
				std::bernoulli_distribution grow(0.5);
				if (grow(rng) && schema.totalLevels() < maxDepth)
					++level.numLevels;
				else if (level.numLevels > 1)
					--level.numLevels;
				break;
			}
			case 2:
			{
				std::bernoulli_distribution grow(0.5);
				level.leafCapacity = grow(rng)
					? clampPowerOfTwo(level.leafCapacity * 2, minLeaf, maxLeaf)
					: clampPowerOfTwo(std::max<size_t>(1, level.leafCapacity / 2), minLeaf, maxLeaf);
				level.minPrimitivesToSplit = std::max<size_t>(2, level.leafCapacity / 4);
				break;
			}
			case 3:
			{
				static const std::array<size_t, 4> divisors = { 2, 4, 8, 16 };
				std::uniform_int_distribution<size_t> divisorDistribution(0, divisors.size() - 1);
				level.minPrimitivesToSplit = std::max<size_t>(2, level.leafCapacity / divisors[divisorDistribution(rng)]);
				break;
			}
			case 4:
				if (options.conditionalLevels && levelIndex > 0)
					mutateLevelCondition(level, rng, options, domain);
				break;
			case 5:
				if (schema.levels.size() < maxBlocks && schema.totalLevels() < maxDepth)
				{
					const auto insertAt = schema.levels.begin() + static_cast<std::ptrdiff_t>(levelIndex + 1);
					schema.levels.insert(insertAt, randomLevelConfig(rng, options, level.type));
				}
				break;
			case 6:
				if (schema.levels.size() > 1)
					schema.levels.erase(schema.levels.begin() + static_cast<std::ptrdiff_t>(levelIndex));
				break;
			case 7:
				if (schema.levels.size() > 1)
				{
					std::uniform_int_distribution<size_t> swapDistribution(0, schema.levels.size() - 1);
					const size_t other = swapDistribution(rng);
					if (other != levelIndex)
						std::swap(schema.levels[levelIndex], schema.levels[other]);
				}
				break;
			case 8:
				// Phase B2 axis-policy flip. Only meaningful for KDTree/BIH levels — for others
				// it's a no-op so the slot doesn't waste mutation budget.
				if (level.type == MultiDataStructure::KDTreeNode)
				{
					level.axisPolicy = (level.axisPolicy == "round_robin")
						? std::string("median_longest_axis")
						: std::string("round_robin");
				}
				break;
			}
		}

		normalizeSchemaForGeneration(schema, options);
		return schema;
	}

	std::vector<SearchDataset> makeSyntheticDatasets(size_t scale)
	{
		const size_t n = std::max<size_t>(64, scale);
		std::vector<SearchDataset> datasets;
		datasets.reserve(3);

		SearchDataset flat;
		flat.name = "synthetic_flat_terrain";
		flat.source = "synthetic";
		flat.cloud = SyntheticPointClouds::generateFlatTerrain(n, 100.0f, 100.0f, 0.05f, 101);
		datasets.push_back(std::move(flat));

		SearchDataset facade;
		facade.name = "synthetic_facade";
		facade.source = "synthetic";
		facade.cloud = SyntheticPointClouds::generateFacade(n, 60.0f, 40.0f, 0.2f, 202);
		datasets.push_back(std::move(facade));

		SearchDataset urban;
		urban.name = "synthetic_urban_mixed";
		urban.source = "synthetic";
		urban.cloud = SyntheticPointClouds::generateUrbanMixed(n, n, 5, 303);
		datasets.push_back(std::move(urban));

		return datasets;
	}

	std::vector<SearchDataset> loadDatasets(const Experiments::SchemaSearchOptions& options)
	{
		std::vector<SearchDataset> datasets;

		if (options.includeSyntheticDatasets)
		{
			std::vector<SearchDataset> synthetic = makeSyntheticDatasets(options.syntheticScale);
			datasets.insert(datasets.end(), std::make_move_iterator(synthetic.begin()), std::make_move_iterator(synthetic.end()));
		}

		for (const std::string& inputPath : options.inputPaths)
		{
			SearchDataset dataset;
			dataset.name = datasetNameFromPath(inputPath);
			dataset.source = inputPath;
			dataset.cloud = PointCloud::load(inputPath, { options.useBinaryCache, options.rebuildBinaryCache });
			if (dataset.cloud.empty())
				throw std::runtime_error("Point cloud is empty: " + inputPath);
			datasets.push_back(std::move(dataset));
		}

		if (datasets.empty())
			throw std::runtime_error("Schema search requires at least one synthetic dataset or --input path");

		return datasets;
	}

	std::vector<Experiments::SchemaCandidate> loadSchemas(const std::vector<std::string>& configuredPaths, bool includeConfiguredSchemas)
	{
		if (!includeConfiguredSchemas)
			return {};

		const std::vector<std::string>& paths = configuredPaths.empty() ? defaultSchemaPaths() : configuredPaths;
		std::vector<Experiments::SchemaCandidate> schemas;
		schemas.reserve(paths.size());

		for (const std::string& schemaPath : paths)
		{
			Experiments::SchemaCandidate loaded;
			loaded.path = schemaPath;
			loaded.config = Config::loadSchemaConfig(schemaPath);
			loaded.name = loaded.config.name;
			schemas.push_back(std::move(loaded));
		}

		return schemas;
	}

	void appendGeneratedSchemas(
		std::vector<Experiments::SchemaCandidate>& schemas,
		const Experiments::SchemaGenerationOptions& options,
		const Experiments::ConditionDomain* domain = nullptr)
	{
		if (options.count == 0)
			return;

		// Pass the (optional) per-cloud condition domain through so the generator's conditional
		// thresholds are calibrated to actual cloud statistics rather than the generic fallbacks.
		// Without this, on a real LiDAR cloud the GA's initial population samples conditions like
		// `minPoints >= 4096` that match basically every node, so conditional gates never fire and
		// the optimizer wastes generations polishing decoration. The auto-conditions path already
		// computes this domain at its top; runSchemaSearch now does the same and feeds it here.
		std::vector<Experiments::SchemaCandidate> generated = Experiments::generateSchemaCandidates(options, domain);
		schemas.insert(schemas.end(), std::make_move_iterator(generated.begin()), std::make_move_iterator(generated.end()));
	}

	const std::vector<std::string>& baselineSchemaPaths(bool cudaEvaluator)
	{
		(void)cudaEvaluator;
		// Canonical single-block controls. The CPU evaluator treats GPU-flavoured names through
		// their compatible CPU base families, while CUDA can dispatch them to native builders.
		static const std::vector<std::string> paths = {
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
		return paths;
	}

	void appendBaselineSchemas(std::vector<Experiments::SchemaCandidate>& schemas, bool cudaEvaluator)
	{
		std::unordered_set<std::string> existing;
		for (const Experiments::SchemaCandidate& candidate : schemas)
			existing.insert(candidate.path);

		size_t added = 0;
		for (const std::string& path : baselineSchemaPaths(cudaEvaluator))
		{
			const std::filesystem::path resolved = resolveExistingPath(path);
			const std::string key = resolved.string();
			if (existing.find(key) != existing.end() || existing.find(path) != existing.end())
				continue;

			Experiments::SchemaCandidate baseline;
			baseline.path = key;
			try
			{
				baseline.config = Config::loadSchemaConfig(key);
			}
			catch (const std::exception& exception)
			{
				std::cerr << "Warning: skipped baseline schema " << path << " (" << exception.what() << ")\n";
				continue;
			}
			baseline.name = baseline.config.name;
			baseline.isBaseline = true;
			schemas.push_back(std::move(baseline));
			existing.insert(key);
			++added;
		}

		if (added > 0)
			std::cout << "  baselines: injected " << added << " single-block schema(s) as controls\n";
	}

	std::string candidateKey(const std::string& name, const std::string& path)
	{
		return name + "\n" + path;
	}

	std::vector<Experiments::SchemaCandidate> selectBenchmarkSchemas(
		const Experiments::SchemaSearchOptions& options,
		const SearchDataset& dataset,
		const Experiments::WorkloadProfile& workload,
		const std::vector<Experiments::SchemaCandidate>& schemas,
		const std::optional<Experiments::SchemaSelectorModel>& rankModel)
	{
		if (schemas.empty())
			return {};

		if (options.benchmarkTopK == 0 || options.benchmarkTopK >= schemas.size())
			return schemas;

		if (!rankModel.has_value())
			return std::vector<Experiments::SchemaCandidate>(schemas.begin(), schemas.begin() + options.benchmarkTopK);

		const std::vector<Experiments::CandidatePrediction> predictions = Experiments::scoreSchemaCandidates(
			rankModel.value(),
			workload,
			dataset.cloud,
			schemas);

		std::unordered_map<std::string, const Experiments::SchemaCandidate*> byKey;
		byKey.reserve(schemas.size());
		for (const Experiments::SchemaCandidate& schema : schemas)
			byKey[candidateKey(schema.config.name, schema.path)] = &schema;

		std::vector<Experiments::SchemaCandidate> selected;
		selected.reserve(std::min(options.benchmarkTopK, predictions.size()));
		for (const Experiments::CandidatePrediction& prediction : predictions)
		{
			const auto found = byKey.find(candidateKey(prediction.schemaName, prediction.schemaPath));
			if (found == byKey.end())
				continue;

			selected.push_back(*found->second);
			if (selected.size() >= options.benchmarkTopK)
				break;
		}

		return selected;
	}

	std::vector<Experiments::WorkloadProfile> loadWorkloads(const Experiments::SchemaSearchOptions& options)
	{
		const std::vector<std::string>& paths = options.workloadPaths.empty() ? defaultWorkloadPaths() : options.workloadPaths;
		std::vector<Experiments::WorkloadProfile> workloads;
		workloads.reserve(paths.size());

		for (const std::string& workloadPath : paths)
		{
			Experiments::WorkloadProfile profile = Experiments::loadWorkloadProfile(workloadPath);
			if (options.queryCountOverride > 0)
				profile.numQueries = options.queryCountOverride;
			if (options.knnKOverride > 0)
				profile.knnK = options.knnKOverride;
			if (options.querySeedOverride)
				profile.querySeed = options.querySeed;
			workloads.push_back(std::move(profile));
		}

		return workloads;
	}

	std::vector<double> queryTypeWeights(const Experiments::WorkloadProfile& profile)
	{
		std::vector<double> weights = {
			std::max(0.0, profile.rangeWeight),
			std::max(0.0, profile.radiusWeight),
			std::max(0.0, profile.knnWeight),
		};

		if (weights[0] == 0.0 && weights[1] == 0.0 && weights[2] == 0.0)
			weights = { 1.0, 1.0, 1.0 };

		return weights;
	}

	PreparedWorkload prepareWorkloadProfile(
		const Experiments::WorkloadProfile& profile,
		const PointCloud& cloud,
		bool cudaEvaluator)
	{
		PreparedWorkload prepared;
		if (profile.numQueries == 0)
			return prepared;

		std::mt19937 rng(profile.querySeed);
		if (cudaEvaluator)
		{
			std::vector<double> weights = queryTypeWeights(profile);

			prepared.cudaQueries.reserve(profile.numQueries);
			std::discrete_distribution<size_t> queryType(weights.begin(), weights.end());
			bool nextRangeIsCount = false;
			for (size_t i = 0; i < profile.numQueries; ++i)
			{
				const size_t type = queryType(rng);
				PointGpu::Query query;
				if (type == 0)
				{
					query.type = nextRangeIsCount ? PointGpu::QueryType::CountRange : PointGpu::QueryType::Range;
					query.bounds = randomQueryBox(rng, cloud, profile);
					if (nextRangeIsCount)
						++prepared.countRangeQueries;
					else
						++prepared.rangeQueries;
					nextRangeIsCount = !nextRangeIsCount;
				}
				else if (type == 1)
				{
					query.type = PointGpu::QueryType::Radius;
					query.center = randomPointInBounds(rng, cloud.bounds());
					query.radius = randomQueryRadius(rng, cloud, profile);
					++prepared.radiusQueries;
				}
				else
				{
					query.type = PointGpu::QueryType::Knn;
					query.center = randomPointInBounds(rng, cloud.bounds());
					query.k = profile.knnK;
					++prepared.knnQueries;
				}
				prepared.cudaQueries.push_back(query);
			}

			return prepared;
		}

		prepared.cpuQueries.reserve(profile.numQueries);
		const std::vector<double> weights = queryTypeWeights(profile);
		std::discrete_distribution<size_t> queryType(weights.begin(), weights.end());
		for (size_t i = 0; i < profile.numQueries; ++i)
		{
			const size_t type = queryType(rng);
			PreparedCpuQuery query;
			if (type == 0)
			{
				query.kind = PreparedQueryKind::Range;
				query.bounds = randomQueryBox(rng, cloud, profile);
				++prepared.rangeQueries;
			}
			else if (type == 1)
			{
				query.kind = PreparedQueryKind::Radius;
				query.center = randomPointInBounds(rng, cloud.bounds());
				query.radius = randomQueryRadius(rng, cloud, profile);
				++prepared.radiusQueries;
			}
			else
			{
				query.kind = PreparedQueryKind::Knn;
				query.center = randomPointInBounds(rng, cloud.bounds());
				++prepared.knnQueries;
			}
			prepared.cpuQueries.push_back(query);
		}

		return prepared;
	}

	WorkloadRun runWorkloadProfile(const PreparedWorkload& prepared, size_t knnK, const PointSpatialIndex& index)
	{
		WorkloadRun result;
		if (prepared.cpuQueries.empty())
			return result;

		std::vector<PointSpatialIndex::QueryStats> samples;
		samples.reserve(prepared.cpuQueries.size());

		for (const PreparedCpuQuery& query : prepared.cpuQueries)
		{
			if (query.kind == PreparedQueryKind::Range)
			{
				samples.push_back(index.rangeQuery(query.bounds).stats);
				++result.rangeQueries;
				continue;
			}

			if (query.kind == PreparedQueryKind::Radius)
			{
				samples.push_back(index.radiusQuery(query.center, query.radius).stats);
				++result.radiusQueries;
				continue;
			}

			samples.push_back(index.knnQuery(query.center, knnK).stats);
			++result.knnQueries;
		}

		result.metrics = Experiments::summarizeQueryStats(samples);
		return result;
	}

	template <typename IndexType>
	WorkloadRun runCudaWorkloadProfile(
		const PreparedWorkload& prepared,
		const IndexType& index,
		const PointGpu::Options& cudaOptions)
	{
		WorkloadRun result;
		if (prepared.cudaQueries.empty())
			return result;

		const PointGpu::QueryResult queryResult = index.query(prepared.cudaQueries, cudaOptions);
		result.metrics = queryResult.metrics;
		result.gpuQueryMs = queryResult.gpuQueryTimeMs;
		result.rangeQueries = queryResult.rangeQueries;
		result.countRangeQueries = queryResult.countRangeQueries;
		result.radiusQueries = queryResult.radiusQueries;
		result.knnQueries = queryResult.knnQueries;
		return result;
	}

	Experiments::SchemaSearchRecord benchmarkSchemaCandidate(
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

		if (useCudaEvaluator(options))
		{
			activeStats = staticSchemaStructureStats(schema.config);
			backend = "cuda";
			const PointGpu::Options cudaOptions = cudaOptionsFrom(options);
			const std::string currentSignature = schemaSignature(schema.config);
			auto applyBuildResult = [&](const PointGpu::BuildResult& build) {
				buildMetrics = build.metrics;
				cudaDevice = build.device;
				cudaBuilder = build.builder;
				gpuUploadMs = build.uploadTimeMs;
				gpuBuildMs = build.gpuBuildTimeMs;
				gpuMemoryBytes = build.gpuMemoryBytes;
			};

			if (isBIHBuilder(cudaOptions.builder))
			{
				PointGpu::BIH localIndex;
				PointGpu::BIH* indexPtr = &localIndex;
				PointGpu::BuildResult build;
				if (cudaCacheEntry)
					build = cudaBuildOrReuse(cudaCacheEntry->bih, currentSignature, dataset.cloud, schema.config, cudaOptions, *cudaCacheEntry, indexPtr);
				else
					build = localIndex.build(dataset.cloud, schema.config, cudaOptions);
				applyBuildResult(build);
				workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions);
			}
			else if (isHGridBuilder(cudaOptions.builder))
			{
				PointGpu::HGrid localIndex;
				PointGpu::HGrid* indexPtr = &localIndex;
				PointGpu::BuildResult build;
				if (cudaCacheEntry)
					build = cudaBuildOrReuse(cudaCacheEntry->hgrid, currentSignature, dataset.cloud, schema.config, cudaOptions, *cudaCacheEntry, indexPtr);
				else
					build = localIndex.build(dataset.cloud, schema.config, cudaOptions);
				applyBuildResult(build);
				workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions);
			}
			else if (isKDTreeBuilder(cudaOptions.builder))
			{
				PointGpu::KDTree localIndex;
				PointGpu::KDTree* indexPtr = &localIndex;
				PointGpu::BuildResult build;
				if (cudaCacheEntry)
					build = cudaBuildOrReuse(cudaCacheEntry->kdTree, currentSignature, dataset.cloud, schema.config, cudaOptions, *cudaCacheEntry, indexPtr);
				else
					build = localIndex.build(dataset.cloud, schema.config, cudaOptions);
				applyBuildResult(build);
				workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions);
			}
			else if (isOctreeBuilder(cudaOptions.builder))
			{
				PointGpu::Octree localIndex;
				PointGpu::Octree* indexPtr = &localIndex;
				PointGpu::BuildResult build;
				if (cudaCacheEntry)
					build = cudaBuildOrReuse(cudaCacheEntry->octree, currentSignature, dataset.cloud, schema.config, cudaOptions, *cudaCacheEntry, indexPtr);
				else
					build = localIndex.build(dataset.cloud, schema.config, cudaOptions);
				applyBuildResult(build);
				workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions);
			}
			else if (isQuadTreeBuilder(cudaOptions.builder))
			{
				PointGpu::QuadTree localIndex;
				PointGpu::QuadTree* indexPtr = &localIndex;
				PointGpu::BuildResult build;
				if (cudaCacheEntry)
					build = cudaBuildOrReuse(cudaCacheEntry->quadTree, currentSignature, dataset.cloud, schema.config, cudaOptions, *cudaCacheEntry, indexPtr);
				else
					build = localIndex.build(dataset.cloud, schema.config, cudaOptions);
				applyBuildResult(build);
				workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions);
			}
			else if (isRegularGridBuilder(cudaOptions.builder))
			{
				PointGpu::RegularGrid localIndex;
				PointGpu::RegularGrid* indexPtr = &localIndex;
				PointGpu::BuildResult build;
				if (cudaCacheEntry)
					build = cudaBuildOrReuse(cudaCacheEntry->regularGrid, currentSignature, dataset.cloud, schema.config, cudaOptions, *cudaCacheEntry, indexPtr);
				else
					build = localIndex.build(dataset.cloud, schema.config, cudaOptions);
				applyBuildResult(build);
				workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions);
			}
			else if (isMixedBuilder(cudaOptions.builder))
			{
				PointGpu::MixedTree localIndex;
				PointGpu::MixedTree* indexPtr = &localIndex;
				PointGpu::BuildResult build;
				if (cudaCacheEntry)
					build = cudaBuildOrReuse(cudaCacheEntry->mixedTree, currentSignature, dataset.cloud, schema.config, cudaOptions, *cudaCacheEntry, indexPtr);
				else
					build = localIndex.build(dataset.cloud, schema.config, cudaOptions);
				applyBuildResult(build);
				workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions);
			}
			else
			{
				PointGpu::LBVH localIndex;
				PointGpu::LBVH* indexPtr = &localIndex;
				PointGpu::BuildResult build;
				if (cudaCacheEntry)
					build = cudaBuildOrReuse(cudaCacheEntry->lbvh, currentSignature, dataset.cloud, schema.config, cudaOptions, *cudaCacheEntry, indexPtr);
				else
					build = localIndex.build(dataset.cloud, schema.config, cudaOptions);
				applyBuildResult(build);
				workloadRun = runCudaWorkloadProfile(preparedWorkload, *indexPtr, cudaOptions);
			}
		}
		else
		{
			PointSpatialIndex index;
			const auto buildBegin = std::chrono::steady_clock::now();
			index.build(dataset.cloud, schema.config);
			const auto buildEnd = std::chrono::steady_clock::now();

			buildMetrics = Experiments::collectBuildMetrics(index.stats(), index.root(), elapsedMilliseconds(buildBegin, buildEnd));
			activeStats = collectActiveStructureStats(index.root(), schema.config);
			workloadRun = runWorkloadProfile(preparedWorkload, workload.knnK, index);
		}

		Experiments::SchemaSearchRecord record;
		record.datasetName = dataset.name;
		record.datasetSource = dataset.source;
		record.numPoints = dataset.cloud.size();
		record.workloadName = workload.name;
		record.rangeWeight = workload.rangeWeight;
		record.radiusWeight = workload.radiusWeight;
		record.knnWeight = workload.knnWeight;
		record.numQueries = workload.numQueries;
		record.knnK = workload.knnK;
		record.querySeed = workload.querySeed;
		record.schemaName = schema.config.name;
		record.schemaPath = schema.path;
		record.buildMetrics = buildMetrics;
		record.queryMetrics = workloadRun.metrics;
		record.rangeQueries = workloadRun.rangeQueries;
		record.countRangeQueries = workloadRun.countRangeQueries;
		record.radiusQueries = workloadRun.radiusQueries;
		record.knnQueries = workloadRun.knnQueries;
		record.weights = options.weights;
		record.score = Experiments::computeSchemaSearchScore(
			record.buildMetrics,
			record.queryMetrics,
			record.weights,
			record.scoreMemoryMb,
			record.scoreImbalancePenalty);
		record.pointFeatures = pointFeatures;
		record.workloadFeatures = workloadFeatures;
		record.backend = backend;
		record.cudaDevice = cudaDevice;
		record.cudaBuilder = cudaBuilder;
		record.gpuUploadMs = gpuUploadMs;
		record.gpuBuildMs = gpuBuildMs;
		record.gpuQueryMs = workloadRun.gpuQueryMs;
		record.gpuMemoryBytes = gpuMemoryBytes;
		record.conditionalLevels = schemaConditionalLevels(schema.config);
		record.conditionFields = schemaConditionFields(schema.config);
		record.conditionSummary = schemaConditionSummary(schema.config);
		record.isBaseline = schema.isBaseline;
		record.activeStructureTypes = activeStats.activeStructureTypes;
		record.nestedActiveFraction = activeStats.nestedActiveFraction;
		record.activeStructureSummary = activeStats.summary;
		return record;
	}

	Experiments::SchemaSearchRecord benchmarkSchemaCandidateCached(
		const SearchDataset& dataset,
		const Experiments::PointCloudFeatures& pointFeatures,
		const Experiments::WorkloadProfile& workload,
		const Experiments::WorkloadFeatures& workloadFeatures,
		const PreparedWorkload& preparedWorkload,
		const Experiments::SchemaCandidate& schema,
		const Experiments::SchemaSearchOptions& options,
		CudaIndexCacheEntry* cudaCacheEntry = nullptr)
	{
		Experiments::EvaluationCache* cache = options.scoreCache;
		if (cache && cache->enabled() && !options.rebuildScoreCache)
		{
			const std::string backendName = useCudaEvaluator(options) ? "cuda" : "cpu";
			std::string cudaBuilder;
			if (useCudaEvaluator(options))
				cudaBuilder = cudaOptionsFrom(options).builder;

			const Experiments::EvaluationCacheKey key = Experiments::makeEvaluationCacheKey(
				schemaSignature(schema.config),
				dataset.name,
				dataset.cloud.size(),
				dataset.cloud.bounds().min(),
				dataset.cloud.bounds().max(),
				workload,
				backendName,
				cudaBuilder,
				options.weights);

			Experiments::SchemaSearchRecord cached;
			cached.datasetName = dataset.name;
			cached.datasetSource = dataset.source;
			cached.numPoints = dataset.cloud.size();
			cached.workloadName = workload.name;
			cached.rangeWeight = workload.rangeWeight;
			cached.radiusWeight = workload.radiusWeight;
			cached.knnWeight = workload.knnWeight;
			cached.numQueries = workload.numQueries;
			cached.knnK = workload.knnK;
			cached.querySeed = workload.querySeed;
			cached.schemaName = schema.config.name;
			cached.schemaPath = schema.path;
			cached.weights = options.weights;
			cached.pointFeatures = pointFeatures;
			cached.workloadFeatures = workloadFeatures;
			cached.isBaseline = schema.isBaseline;

			if (cache->tryGet(key, cached))
			{
				cached.isBaseline = schema.isBaseline;
				if (options.deepNestedSearch && !useCudaEvaluator(options) && cached.activeStructureTypes == 0)
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

	EvaluatedCandidate evaluateCandidate(
		const Experiments::SchemaCandidate& candidate,
		const std::vector<DatasetContext>& datasets,
		const std::vector<Experiments::WorkloadProfile>& workloads,
		const Experiments::SchemaSearchOptions& options,
		CudaIndexCache* cudaCache = nullptr)
	{
		EvaluatedCandidate evaluation;
		evaluation.candidate = candidate;
		evaluation.records.reserve(datasets.size() * workloads.size());

		double scoreSum = 0.0;
		size_t scoreCount = 0;
		for (const DatasetContext& datasetContext : datasets)
		{
			for (size_t workloadIndex = 0; workloadIndex < workloads.size(); ++workloadIndex)
			{
				const Experiments::WorkloadProfile& workload = workloads[workloadIndex];
				const Experiments::WorkloadFeatures workloadFeatures = Experiments::extractWorkloadFeatures(workload, options.weights);
				CudaIndexCacheEntry* cudaEntry = cudaCache && useCudaEvaluator(options)
					? &(*cudaCache)[datasetContext.dataset]
					: nullptr;

				// Wrap the per-(dataset, workload) evaluation so a single failed candidate (e.g.
				// HGrid configurations that blow the memory budget, KDTree builds that hit a
				// CUDA OOM, malformed schema JSONs) cannot kill the whole optimizer run. The
				// failed candidate gets a sentinel record with infinite score and the GA moves
				// on to the next one.
				try
				{
					Experiments::SchemaSearchRecord record = benchmarkSchemaCandidateCached(
						*datasetContext.dataset,
						datasetContext.features,
						workload,
						workloadFeatures,
						datasetContext.preparedWorkloads[workloadIndex],
						candidate,
						options,
						cudaEntry);
					scoreSum += record.score;
					++scoreCount;
					evaluation.records.push_back(std::move(record));
				}
				catch (const std::exception& exception)
				{
					std::cerr << "    candidate '" << candidate.config.name
						<< "' failed on dataset '" << datasetContext.dataset->name
						<< "' / workload '" << workload.name << "': " << exception.what() << '\n';
					Experiments::SchemaSearchRecord failed;
					failed.datasetName = datasetContext.dataset->name;
					failed.datasetSource = datasetContext.dataset->source;
					failed.numPoints = datasetContext.dataset->cloud.size();
					failed.workloadName = workload.name;
					failed.rangeWeight = workload.rangeWeight;
					failed.radiusWeight = workload.radiusWeight;
					failed.knnWeight = workload.knnWeight;
					failed.numQueries = workload.numQueries;
					failed.knnK = workload.knnK;
					failed.querySeed = workload.querySeed;
					failed.schemaName = candidate.config.name;
					failed.schemaPath = candidate.path;
					failed.weights = options.weights;
					failed.pointFeatures = datasetContext.features;
					failed.workloadFeatures = workloadFeatures;
					failed.backend = useCudaEvaluator(options) ? "cuda_failed" : "cpu_failed";
					failed.score = std::numeric_limits<double>::infinity();
					failed.isBaseline = candidate.isBaseline;
					evaluation.records.push_back(std::move(failed));
				}
			}
		}

		evaluation.aggregateScore = scoreCount > 0
			? scoreSum / static_cast<double>(scoreCount)
			: std::numeric_limits<double>::infinity();
		return evaluation;
	}

	std::string csvEscape(const std::string& value)
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

	void createParentDirectory(const std::string& filename)
	{
		const std::filesystem::path path(filename);
		if (path.has_parent_path())
			std::filesystem::create_directories(path.parent_path());
	}

	void writeSearchHeader(std::ostream& output)
	{
		output
			<< "dataset_name,dataset_source,num_points,workload_name,range_weight,radius_weight,knn_weight,num_queries,knn_k,query_seed,"
			<< "range_scale_min,range_scale_max,radius_scale_min,radius_scale_max,"
			<< "feature_sample_size,bbox_x,bbox_y,bbox_z,aspect_xy,aspect_xz,aspect_yz,density_bbox,height_mean,height_std,height_range,"
			<< "cov_eig_0,cov_eig_1,cov_eig_2,linearity,planarity,scattering,occupancy_ratio_8,occupancy_entropy_8,density_cv_8,verticality_score,flatness_score,"
			<< "w_range,w_radius,w_knn,query_scale_mean,query_scale_std,build_weight,memory_weight,"
			<< "schema_name,schema_path,build_time_ms,num_nodes,num_leaves,max_depth,avg_leaf_occupancy,max_leaf_occupancy,memory_estimate_bytes,"
			<< "total_queries,avg_latency_ms,median_latency_ms,p95_latency_ms,throughput_qps,avg_visited_nodes,avg_tested_points,avg_returned_points,"
			<< "range_queries,radius_queries,knn_queries,score,score_memory_mb,score_imbalance_penalty,lambda_build,lambda_memory,lambda_imbalance,"
			<< "backend,cuda_device,cuda_builder,gpu_upload_ms,gpu_build_ms,gpu_query_ms,gpu_memory_bytes,count_range_queries,"
			<< "conditional_levels,condition_fields,condition_summary,is_baseline,active_structure_types,nested_active_fraction,active_structure_summary,"
			<< "best_baseline_schema,best_baseline_score,relative_speedup_vs_baseline,"
			<< "confirm_seeds_used,latency_mean_ms,latency_ci_low_ms,latency_ci_high_ms,"
			<< "p95_latency_mean_ms,p95_latency_ci_low_ms,p95_latency_ci_high_ms,"
			<< "gpu_build_mean_ms,gpu_build_ci_low_ms,gpu_build_ci_high_ms\n";
	}

	void writeSearchRows(const std::string& csvPath, const std::vector<Experiments::SchemaSearchRecord>& records)
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
				<< csvEscape(record.datasetName) << ','
				<< csvEscape(record.datasetSource) << ','
				<< record.numPoints << ','
				<< csvEscape(record.workloadName) << ','
				<< record.rangeWeight << ','
				<< record.radiusWeight << ','
				<< record.knnWeight << ','
				<< record.numQueries << ','
				<< record.knnK << ','
				<< record.querySeed << ','
				<< record.workloadFeatures.rangeScaleMin << ','
				<< record.workloadFeatures.rangeScaleMax << ','
				<< record.workloadFeatures.radiusScaleMin << ','
				<< record.workloadFeatures.radiusScaleMax << ','
				<< record.pointFeatures.sampleSize << ','
				<< record.pointFeatures.bboxX << ','
				<< record.pointFeatures.bboxY << ','
				<< record.pointFeatures.bboxZ << ','
				<< record.pointFeatures.aspectXY << ','
				<< record.pointFeatures.aspectXZ << ','
				<< record.pointFeatures.aspectYZ << ','
				<< record.pointFeatures.densityBbox << ','
				<< record.pointFeatures.heightMean << ','
				<< record.pointFeatures.heightStd << ','
				<< record.pointFeatures.heightRange << ','
				<< record.pointFeatures.covEig0 << ','
				<< record.pointFeatures.covEig1 << ','
				<< record.pointFeatures.covEig2 << ','
				<< record.pointFeatures.linearity << ','
				<< record.pointFeatures.planarity << ','
				<< record.pointFeatures.scattering << ','
				<< record.pointFeatures.occupancyRatio8 << ','
				<< record.pointFeatures.occupancyEntropy8 << ','
				<< record.pointFeatures.densityCv8 << ','
				<< record.pointFeatures.verticalityScore << ','
				<< record.pointFeatures.flatnessScore << ','
				<< record.workloadFeatures.wRange << ','
				<< record.workloadFeatures.wRadius << ','
				<< record.workloadFeatures.wKnn << ','
				<< record.workloadFeatures.queryScaleMean << ','
				<< record.workloadFeatures.queryScaleStd << ','
				<< record.workloadFeatures.buildWeight << ','
				<< record.workloadFeatures.memoryWeight << ','
				<< csvEscape(record.schemaName) << ','
				<< csvEscape(record.schemaPath) << ','
				<< record.buildMetrics.buildTimeMs << ','
				<< record.buildMetrics.numNodes << ','
				<< record.buildMetrics.numLeaves << ','
				<< record.buildMetrics.maxDepth << ','
				<< record.buildMetrics.averageLeafOccupancy << ','
				<< record.buildMetrics.maxLeafOccupancy << ','
				<< record.buildMetrics.memoryEstimateBytes << ','
				<< record.queryMetrics.totalQueries << ','
				<< record.queryMetrics.averageLatencyMs << ','
				<< record.queryMetrics.medianLatencyMs << ','
				<< record.queryMetrics.p95LatencyMs << ','
				<< record.queryMetrics.throughputQueriesPerSecond << ','
				<< record.queryMetrics.averageVisitedNodes << ','
				<< record.queryMetrics.averageTestedPoints << ','
				<< record.queryMetrics.averageReturnedPoints << ','
				<< record.rangeQueries << ','
				<< record.radiusQueries << ','
				<< record.knnQueries << ','
				<< record.score << ','
				<< record.scoreMemoryMb << ','
				<< record.scoreImbalancePenalty << ','
				<< record.weights.lambdaBuild << ','
				<< record.weights.lambdaMemory << ','
				<< record.weights.lambdaImbalance << ','
				<< csvEscape(record.backend) << ','
				<< record.cudaDevice << ','
				<< csvEscape(record.cudaBuilder) << ','
				<< record.gpuUploadMs << ','
				<< record.gpuBuildMs << ','
				<< record.gpuQueryMs << ','
				<< record.gpuMemoryBytes << ','
				<< record.countRangeQueries << ','
				<< record.conditionalLevels << ','
				<< record.conditionFields << ','
				<< csvEscape(record.conditionSummary) << ','
				<< (record.isBaseline ? 1 : 0) << ','
				<< record.activeStructureTypes << ','
				<< record.nestedActiveFraction << ','
				<< csvEscape(record.activeStructureSummary) << ','
				<< csvEscape(record.bestBaselineSchema) << ','
				<< record.bestBaselineScore << ','
				<< record.relativeSpeedupVsBaseline << ','
				<< record.confirmSeedsUsed << ','
				<< record.latencyMean << ','
				<< record.latencyCiLow << ','
				<< record.latencyCiHigh << ','
				<< record.p95LatencyMean << ','
				<< record.p95LatencyCiLow << ','
				<< record.p95LatencyCiHigh << ','
				<< record.gpuBuildMean << ','
				<< record.gpuBuildCiLow << ','
				<< record.gpuBuildCiHigh << '\n';
		}
	}

	size_t countCandidatesForBest(const std::vector<Experiments::SchemaSearchRecord>& records, const Experiments::SchemaSearchRecord& best)
	{
		size_t count = 0;
		for (const Experiments::SchemaSearchRecord& record : records)
		{
			if (record.datasetName == best.datasetName && record.workloadName == best.workloadName)
				++count;
		}
		return count;
	}

	void writeBestRows(const std::string& csvPath, const std::vector<Experiments::SchemaSearchRecord>& records)
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
			<< "occupancy_entropy_8,density_cv_8,verticality_score,flatness_score,w_range,w_radius,w_knn,knn_k,num_queries,"
			<< "range_scale_min,range_scale_max,radius_scale_min,radius_scale_max,query_scale_mean,query_scale_std,build_weight,memory_weight,best_schema_name,best_schema_path,best_score,"
			<< "best_avg_latency_ms,best_build_time_ms,best_memory_estimate_bytes,num_candidates,backend,cuda_device,cuda_builder,gpu_upload_ms,gpu_build_ms,gpu_query_ms,gpu_memory_bytes,"
			<< "conditional_levels,condition_fields,condition_summary,is_baseline,active_structure_types,nested_active_fraction,active_structure_summary,"
			<< "best_baseline_schema,best_baseline_score,relative_speedup_vs_baseline,"
			<< "confirm_seeds_used,latency_mean_ms,latency_ci_low_ms,latency_ci_high_ms,"
			<< "p95_latency_mean_ms,p95_latency_ci_low_ms,p95_latency_ci_high_ms,"
			<< "gpu_build_mean_ms,gpu_build_ci_low_ms,gpu_build_ci_high_ms\n";
		output << std::fixed << std::setprecision(6);
		for (const Experiments::SchemaSearchRecord& record : bestRecords)
		{
			output
				<< csvEscape(record.datasetName) << ','
				<< csvEscape(record.workloadName) << ','
				<< record.numPoints << ','
				<< record.pointFeatures.sampleSize << ','
				<< record.pointFeatures.bboxX << ','
				<< record.pointFeatures.bboxY << ','
				<< record.pointFeatures.bboxZ << ','
				<< record.pointFeatures.aspectXY << ','
				<< record.pointFeatures.aspectXZ << ','
				<< record.pointFeatures.aspectYZ << ','
				<< record.pointFeatures.densityBbox << ','
				<< record.pointFeatures.heightMean << ','
				<< record.pointFeatures.heightStd << ','
				<< record.pointFeatures.heightRange << ','
				<< record.pointFeatures.covEig0 << ','
				<< record.pointFeatures.covEig1 << ','
				<< record.pointFeatures.covEig2 << ','
				<< record.pointFeatures.linearity << ','
				<< record.pointFeatures.planarity << ','
				<< record.pointFeatures.scattering << ','
				<< record.pointFeatures.occupancyRatio8 << ','
				<< record.pointFeatures.occupancyEntropy8 << ','
				<< record.pointFeatures.densityCv8 << ','
				<< record.pointFeatures.verticalityScore << ','
				<< record.pointFeatures.flatnessScore << ','
				<< record.workloadFeatures.wRange << ','
				<< record.workloadFeatures.wRadius << ','
				<< record.workloadFeatures.wKnn << ','
				<< record.workloadFeatures.knnK << ','
				<< record.workloadFeatures.numQueries << ','
				<< record.workloadFeatures.rangeScaleMin << ','
				<< record.workloadFeatures.rangeScaleMax << ','
				<< record.workloadFeatures.radiusScaleMin << ','
				<< record.workloadFeatures.radiusScaleMax << ','
				<< record.workloadFeatures.queryScaleMean << ','
				<< record.workloadFeatures.queryScaleStd << ','
				<< record.workloadFeatures.buildWeight << ','
				<< record.workloadFeatures.memoryWeight << ','
				<< csvEscape(record.schemaName) << ','
				<< csvEscape(record.schemaPath) << ','
				<< record.score << ','
				<< record.queryMetrics.averageLatencyMs << ','
				<< record.buildMetrics.buildTimeMs << ','
				<< record.buildMetrics.memoryEstimateBytes << ','
				<< countCandidatesForBest(records, record) << ','
				<< csvEscape(record.backend) << ','
				<< record.cudaDevice << ','
				<< csvEscape(record.cudaBuilder) << ','
				<< record.gpuUploadMs << ','
				<< record.gpuBuildMs << ','
				<< record.gpuQueryMs << ','
				<< record.gpuMemoryBytes << ','
				<< record.conditionalLevels << ','
				<< record.conditionFields << ','
				<< csvEscape(record.conditionSummary) << ','
				<< (record.isBaseline ? 1 : 0) << ','
				<< record.activeStructureTypes << ','
				<< record.nestedActiveFraction << ','
				<< csvEscape(record.activeStructureSummary) << ','
				<< csvEscape(record.bestBaselineSchema) << ','
				<< record.bestBaselineScore << ','
				<< record.relativeSpeedupVsBaseline << ','
				<< record.confirmSeedsUsed << ','
				<< record.latencyMean << ','
				<< record.latencyCiLow << ','
				<< record.latencyCiHigh << ','
				<< record.p95LatencyMean << ','
				<< record.p95LatencyCiLow << ','
				<< record.p95LatencyCiHigh << ','
				<< record.gpuBuildMean << ','
				<< record.gpuBuildCiLow << ','
				<< record.gpuBuildCiHigh << '\n';
		}
	}

	void writeParetoRows(const std::string& csvPath, const std::vector<Experiments::SchemaSearchRecord>& records)
	{
		if (csvPath.empty())
			return;

		const std::vector<Experiments::SchemaSearchRecord> front = Experiments::selectParetoRecords(records);
		createParentDirectory(csvPath);
		std::ofstream output(csvPath);
		if (!output.is_open())
			throw std::runtime_error("Unable to open schema-search Pareto CSV path: " + csvPath);

		// Compact column set focused on the four Pareto objectives plus enough identity to replay
		// the candidate. The full feature-rich layout stays in the `best` CSV; this one is meant
		// for plotting the front, not training.
		output
			<< "dataset_name,workload_name,pareto_rank,schema_name,schema_path,score,"
			<< "avg_latency_ms,build_time_ms,memory_mb,memory_estimate_bytes,imbalance_penalty,"
			<< "p95_latency_ms,throughput_qps,backend,cuda_builder,is_baseline,"
			<< "conditional_levels,condition_fields,active_structure_types,nested_active_fraction,"
			<< "confirm_seeds_used,latency_mean_ms,latency_ci_low_ms,latency_ci_high_ms,"
			<< "p95_latency_mean_ms,p95_latency_ci_low_ms,p95_latency_ci_high_ms,"
			<< "gpu_build_mean_ms,gpu_build_ci_low_ms,gpu_build_ci_high_ms\n";
		output << std::fixed << std::setprecision(6);
		for (const Experiments::SchemaSearchRecord& record : front)
		{
			const double memoryMb = static_cast<double>(record.buildMetrics.memoryEstimateBytes) / (1024.0 * 1024.0);
			const double imbalancePenalty = record.buildMetrics.averageLeafOccupancy > 0.0
				? static_cast<double>(record.buildMetrics.maxLeafOccupancy) / record.buildMetrics.averageLeafOccupancy
				: 0.0;
			output
				<< csvEscape(record.datasetName) << ','
				<< csvEscape(record.workloadName) << ','
				<< record.paretoRank << ','
				<< csvEscape(record.schemaName) << ','
				<< csvEscape(record.schemaPath) << ','
				<< record.score << ','
				<< record.queryMetrics.averageLatencyMs << ','
				<< record.buildMetrics.buildTimeMs << ','
				<< memoryMb << ','
				<< record.buildMetrics.memoryEstimateBytes << ','
				<< imbalancePenalty << ','
				<< record.queryMetrics.p95LatencyMs << ','
				<< record.queryMetrics.throughputQueriesPerSecond << ','
				<< csvEscape(record.backend) << ','
				<< csvEscape(record.cudaBuilder) << ','
				<< (record.isBaseline ? 1 : 0) << ','
				<< record.conditionalLevels << ','
				<< record.conditionFields << ','
				<< record.activeStructureTypes << ','
				<< record.nestedActiveFraction << ','
				<< record.confirmSeedsUsed << ','
				<< record.latencyMean << ','
				<< record.latencyCiLow << ','
				<< record.latencyCiHigh << ','
				<< record.p95LatencyMean << ','
				<< record.p95LatencyCiLow << ','
				<< record.p95LatencyCiHigh << ','
				<< record.gpuBuildMean << ','
				<< record.gpuBuildCiLow << ','
				<< record.gpuBuildCiHigh << '\n';
		}
	}

	std::string recordGroupKey(const Experiments::SchemaSearchRecord& record)
	{
		return record.datasetName + "\n" + record.workloadName;
	}

	bool recordCountsAsNested(const Experiments::SchemaSearchRecord& record);

	void annotateBaselineComparisons(std::vector<Experiments::SchemaSearchRecord>& records)
	{
		std::unordered_map<std::string, const Experiments::SchemaSearchRecord*> bestBaselines;
		for (const Experiments::SchemaSearchRecord& record : records)
		{
			if (!record.isBaseline)
				continue;
			const std::string key = recordGroupKey(record);
			const auto existing = bestBaselines.find(key);
			if (existing == bestBaselines.end() || record.score < existing->second->score)
				bestBaselines[key] = &record;
		}

		for (Experiments::SchemaSearchRecord& record : records)
		{
			const auto baseline = bestBaselines.find(recordGroupKey(record));
			if (baseline == bestBaselines.end())
				continue;

			const Experiments::SchemaSearchRecord& best = *baseline->second;
			record.bestBaselineSchema = best.schemaName;
			record.bestBaselineScore = best.score;
			record.relativeSpeedupVsBaseline = best.queryMetrics.averageLatencyMs > 0.0
				? (best.queryMetrics.averageLatencyMs - record.queryMetrics.averageLatencyMs) / best.queryMetrics.averageLatencyMs
				: 0.0;
		}
	}

	void reportDeepNestedOutcome(const std::vector<Experiments::SchemaSearchRecord>& records)
	{
		std::map<std::string, const Experiments::SchemaSearchRecord*> bestBaselines;
		std::map<std::string, const Experiments::SchemaSearchRecord*> bestNested;
		for (const Experiments::SchemaSearchRecord& record : records)
		{
			const std::string key = recordGroupKey(record);
			if (record.isBaseline)
			{
				const auto existing = bestBaselines.find(key);
				if (existing == bestBaselines.end() || record.queryMetrics.averageLatencyMs < existing->second->queryMetrics.averageLatencyMs)
					bestBaselines[key] = &record;
			}
			else if (recordCountsAsNested(record))
			{
				const auto existing = bestNested.find(key);
				if (existing == bestNested.end() || record.queryMetrics.averageLatencyMs < existing->second->queryMetrics.averageLatencyMs)
					bestNested[key] = &record;
			}
		}

		for (const auto& [key, baseline] : bestBaselines)
		{
			const auto nested = bestNested.find(key);
			if (nested == bestNested.end())
			{
				std::cout << "  deep nested outcome: " << baseline->datasetName << " / " << baseline->workloadName
					<< " has no runtime-nested finalist (>=2 active types and >=5% nested fraction)\n";
				continue;
			}

			const Experiments::SchemaSearchRecord& nestedRecord = *nested->second;
			const double meanSpeedup = baseline->queryMetrics.averageLatencyMs > 0.0
				? (baseline->queryMetrics.averageLatencyMs - nestedRecord.queryMetrics.averageLatencyMs) / baseline->queryMetrics.averageLatencyMs
				: 0.0;
			const bool p95NoWorse = nestedRecord.queryMetrics.p95LatencyMs <= baseline->queryMetrics.p95LatencyMs;
			std::cout << "  deep nested outcome: " << baseline->datasetName << " / " << baseline->workloadName
				<< " best nested " << nestedRecord.schemaName
				<< " vs baseline " << baseline->schemaName
				<< ", mean speedup " << (meanSpeedup * 100.0) << "%"
				<< ", p95 " << (p95NoWorse ? "no worse" : "worse")
				<< ", nested active fraction " << nestedRecord.nestedActiveFraction;
			if (meanSpeedup >= 0.05 && p95NoWorse)
				std::cout << " (target met)";
			std::cout << '\n';
		}
	}

	std::vector<Experiments::SchemaCandidate> uniqueCandidates(
		const std::vector<Experiments::SchemaCandidate>& candidates,
		std::unordered_set<std::string>& seenSignatures)
	{
		std::vector<Experiments::SchemaCandidate> unique;
		unique.reserve(candidates.size());
		for (const Experiments::SchemaCandidate& candidate : candidates)
		{
			const std::string signature = schemaSignature(candidate.config);
			if (!seenSignatures.insert(signature).second)
				continue;

			unique.push_back(candidate);
		}
		return unique;
	}

	void appendRecords(
		std::vector<Experiments::SchemaSearchRecord>& records,
		const EvaluatedCandidate& evaluation)
	{
		records.insert(records.end(), evaluation.records.begin(), evaluation.records.end());
	}

	void emitProgress(
		const Experiments::SchemaSearchOptions& options,
		const Experiments::SchemaSearchRecord& record)
	{
		if (options.progressCallback)
			options.progressCallback(record);
	}

	void sortEvaluations(std::vector<EvaluatedCandidate>& evaluations)
	{
		std::sort(evaluations.begin(), evaluations.end(), [](const EvaluatedCandidate& left, const EvaluatedCandidate& right) {
			return left.aggregateScore < right.aggregateScore;
		});
	}

	struct CandidateObjectives
	{
		double avgLatencyMs = 0.0;
		double buildTimeMs = 0.0;
		double memoryMb = 0.0;
		double imbalancePenalty = 0.0;
	};

	// Averages the four Pareto objectives across an EvaluatedCandidate's records (one per
	// (dataset, workload)). For single-dataset / single-workload runs — the publication target —
	// this is the record's metrics verbatim. For multi-dataset sweeps, averaging treats all
	// datasets equally, which matches how `aggregateScore` already combines per-record scores.
	CandidateObjectives objectivesOf(const EvaluatedCandidate& evaluation)
	{
		CandidateObjectives obj;
		if (evaluation.records.empty())
			return obj;
		for (const Experiments::SchemaSearchRecord& record : evaluation.records)
		{
			// Use the seed-averaged mean when multi-seed confirmation was run, otherwise the
			// single-seed point estimate. Mirrors the Pareto step in selectParetoRecords so the
			// optimizer and the CSV agree on what "latency" means.
			const double latency = record.confirmSeedsUsed > 0
				? record.latencyMean
				: record.queryMetrics.averageLatencyMs;
			const double memoryMb = static_cast<double>(record.buildMetrics.memoryEstimateBytes) / (1024.0 * 1024.0);
			const double imbalance = record.buildMetrics.averageLeafOccupancy > 0.0
				? static_cast<double>(record.buildMetrics.maxLeafOccupancy) / record.buildMetrics.averageLeafOccupancy
				: 0.0;
			obj.avgLatencyMs += latency;
			obj.buildTimeMs += record.buildMetrics.buildTimeMs;
			obj.memoryMb += memoryMb;
			obj.imbalancePenalty += imbalance;
		}
		const double n = static_cast<double>(evaluation.records.size());
		obj.avgLatencyMs /= n;
		obj.buildTimeMs /= n;
		obj.memoryMb /= n;
		obj.imbalancePenalty /= n;
		return obj;
	}

	bool dominatesCandidate(const CandidateObjectives& a, const CandidateObjectives& b)
	{
		const bool allLeq =
			a.avgLatencyMs <= b.avgLatencyMs &&
			a.buildTimeMs <= b.buildTimeMs &&
			a.memoryMb <= b.memoryMb &&
			a.imbalancePenalty <= b.imbalancePenalty;
		if (!allLeq)
			return false;
		return
			a.avgLatencyMs < b.avgLatencyMs ||
			a.buildTimeMs < b.buildTimeMs ||
			a.memoryMb < b.memoryMb ||
			a.imbalancePenalty < b.imbalancePenalty;
	}

	// Standard NSGA-II fast non-dominated sort + crowding distance. After this call, each
	// EvaluatedCandidate has its `paretoFront` (0 = best, higher = worse) and `crowdingDistance`
	// fields populated, and the vector is sorted: smaller paretoFront first, ties broken by
	// larger crowdingDistance (more isolated = preferred). Lets the GA pick elites that are not
	// only good but spread across the front, which directly addresses the "one corridor" failure
	// mode of scalar-score elite selection.
	void nsga2RankAndSort(std::vector<EvaluatedCandidate>& evaluations)
	{
		const size_t n = evaluations.size();
		if (n == 0)
			return;

		std::vector<CandidateObjectives> objectives(n);
		for (size_t i = 0; i < n; ++i)
		{
			objectives[i] = objectivesOf(evaluations[i]);
			evaluations[i].paretoFront = -1;
			evaluations[i].crowdingDistance = 0.0;
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
				evaluations[p].paretoFront = 0;
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
						evaluations[q].paretoFront = rank + 1;
						nextFront.push_back(q);
					}
				}
			}
			++rank;
			currentFront = std::move(nextFront);
		}

		// Crowding distance per front. The candidate at each objective's extreme gets +inf so the
		// edges of the front are always preserved when elites are picked.
		std::map<int, std::vector<size_t>> fronts;
		for (size_t i = 0; i < n; ++i)
			fronts[evaluations[i].paretoFront].push_back(i);

		auto objectiveValue = [&](size_t idx, int axis) {
			switch (axis)
			{
			case 0: return objectives[idx].avgLatencyMs;
			case 1: return objectives[idx].buildTimeMs;
			case 2: return objectives[idx].memoryMb;
			default: return objectives[idx].imbalancePenalty;
			}
		};

		for (auto& [frontRank, indices] : fronts)
		{
			if (indices.size() <= 2)
			{
				for (const size_t idx : indices)
					evaluations[idx].crowdingDistance = std::numeric_limits<double>::infinity();
				continue;
			}
			for (int axis = 0; axis < 4; ++axis)
			{
				std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
					return objectiveValue(a, axis) < objectiveValue(b, axis);
				});
				const double lo = objectiveValue(indices.front(), axis);
				const double hi = objectiveValue(indices.back(), axis);
				evaluations[indices.front()].crowdingDistance = std::numeric_limits<double>::infinity();
				evaluations[indices.back()].crowdingDistance = std::numeric_limits<double>::infinity();
				if (hi <= lo)
					continue;
				const double range = hi - lo;
				for (size_t i = 1; i + 1 < indices.size(); ++i)
				{
					if (std::isinf(evaluations[indices[i]].crowdingDistance))
						continue;
					const double prev = objectiveValue(indices[i - 1], axis);
					const double next = objectiveValue(indices[i + 1], axis);
					evaluations[indices[i]].crowdingDistance += (next - prev) / range;
				}
			}
		}

		std::sort(evaluations.begin(), evaluations.end(), [](const EvaluatedCandidate& a, const EvaluatedCandidate& b) {
			if (a.paretoFront != b.paretoFront)
				return a.paretoFront < b.paretoFront;
			if (a.crowdingDistance != b.crowdingDistance)
				return a.crowdingDistance > b.crowdingDistance;
			// Tie-break by scalar score. Critical when the Pareto front is near-1D (e.g. when
			// lambdaBuild = lambdaMemory = lambdaImbalance = 0 so only avg_latency varies
			// meaningfully): every front-edge candidate gets crowdingDistance = +inf, including
			// the max-latency / max-build / max-memory candidates. Without this tie-break, the
			// std::sort comparator gives undefined ordering among +inf candidates and the
			// scalar-worst-but-edge candidate can land at archive[0], poisoning the elite pool.
			// Children of that elite drag subsequent generations worse — the regression the
			// user reported.
			return a.aggregateScore < b.aggregateScore;
		});

		// Hard-guarantee elitism: archive[0] is the actual scalar champion. Even with the
		// tie-break above, future ranking changes (or Pareto-front geometry that doesn't pin
		// the scalar best to a +inf crowding cell) could put a non-champion at index 0. Run a
		// final pass that swaps the scalar best into slot 0 — cost is one min_element, the
		// rest of the archive ordering is preserved.
		auto scalarBest = std::min_element(evaluations.begin(), evaluations.end(),
			[](const EvaluatedCandidate& a, const EvaluatedCandidate& b) {
				return a.aggregateScore < b.aggregateScore;
			});
		if (scalarBest != evaluations.end() && scalarBest != evaluations.begin())
			std::iter_swap(evaluations.begin(), scalarBest);
	}

	std::vector<DatasetContext> makeDatasetContexts(
		const std::vector<SearchDataset>& datasets,
		const std::vector<Experiments::WorkloadProfile>& workloads,
		bool cudaEvaluator)
	{
		std::vector<DatasetContext> contexts;
		contexts.reserve(datasets.size());
		for (const SearchDataset& dataset : datasets)
		{
			DatasetContext context;
			context.dataset = &dataset;
			context.features = Experiments::extractPointCloudFeatures(dataset.cloud);
			context.preparedWorkloads.reserve(workloads.size());
			for (const Experiments::WorkloadProfile& workload : workloads)
				context.preparedWorkloads.push_back(prepareWorkloadProfile(workload, dataset.cloud, cudaEvaluator));
			contexts.push_back(std::move(context));
		}
		return contexts;
	}

	std::vector<DatasetContext> makeDatasetContextsFromPointers(
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
			context.dataset = dataset;
			context.features = Experiments::extractPointCloudFeatures(dataset->cloud);
			context.preparedWorkloads.reserve(workloads.size());
			for (const Experiments::WorkloadProfile& workload : workloads)
				context.preparedWorkloads.push_back(prepareWorkloadProfile(workload, dataset->cloud, cudaEvaluator));
			contexts.push_back(std::move(context));
		}
		return contexts;
	}

	PointCloud downsampleCloud(const PointCloud& cloud, size_t pointCap)
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

	std::vector<SearchDataset> makeProxyDatasets(const std::vector<SearchDataset>& datasets, size_t pointCap)
	{
		std::vector<SearchDataset> proxy;
		proxy.reserve(datasets.size());
		for (const SearchDataset& dataset : datasets)
		{
			SearchDataset item;
			item.name = dataset.name + "_proxy";
			item.source = dataset.source;
			item.cloud = downsampleCloud(dataset.cloud, pointCap);
			proxy.push_back(std::move(item));
		}
		return proxy;
	}

	std::vector<Experiments::WorkloadProfile> withQueryCount(
		std::vector<Experiments::WorkloadProfile> workloads,
		size_t queryCount)
	{
		if (queryCount == 0)
			return workloads;

		for (Experiments::WorkloadProfile& workload : workloads)
			workload.numQueries = queryCount;
		return workloads;
	}

	template <typename T>
	std::vector<T> thinSortedValues(std::vector<T> values, size_t maxValues)
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

	std::vector<size_t> deepLeafCandidates(
		const Experiments::SchemaGenerationOptions& options,
		const Experiments::ConditionDomain& domain)
	{
		const size_t minLeaf = std::max<size_t>(1, std::min(options.minLeafCapacity, options.maxLeafCapacity));
		const size_t maxLeaf = std::max(minLeaf, options.maxLeafCapacity);
		std::vector<size_t> values;
		for (size_t value = clampPowerOfTwo(minLeaf, minLeaf, maxLeaf); value <= maxLeaf; value *= 2)
		{
			values.push_back(value);
			if (value > maxLeaf / 2)
				break;
		}
		for (const size_t threshold : domain.pointThresholds)
			values.push_back(clampPowerOfTwo(threshold, minLeaf, maxLeaf));
		return thinSortedValues(values, 8);
	}

	std::vector<size_t> deepPointThresholdCandidates(
		const Experiments::SchemaGenerationOptions& options,
		const Experiments::ConditionDomain& domain)
	{
		const size_t minLeaf = std::max<size_t>(1, std::min(options.minLeafCapacity, options.maxLeafCapacity));
		const size_t maxThreshold = std::max<size_t>(minLeaf * 2, std::min<size_t>(options.maxLeafCapacity * 16, 1 << 20));
		std::vector<size_t> values = domain.pointThresholds.empty() ? fallbackPointThresholds() : domain.pointThresholds;
		for (const size_t fallback : fallbackPointThresholds())
			values.push_back(fallback);
		for (size_t& value : values)
			value = std::clamp(value, minLeaf, maxThreshold);
		return thinSortedValues(values, 8);
	}

	std::vector<double> deepDoubleCandidates(std::vector<double> values, const std::vector<double>& fallback, size_t maxValues)
	{
		if (values.empty())
			values = fallback;
		for (const double value : fallback)
			values.push_back(value);
		return thinSortedValues(values, maxValues);
	}

	double representativeQueryScale(const std::vector<Experiments::WorkloadProfile>& workloads)
	{
		double weighted = 0.0;
		double totalWeight = 0.0;
		for (const Experiments::WorkloadProfile& workload : workloads)
		{
			if (workload.rangeWeight > 0.0)
			{
				weighted += workload.rangeWeight * 0.5 * (workload.rangeScaleMin + workload.rangeScaleMax);
				totalWeight += workload.rangeWeight;
			}
			if (workload.radiusWeight > 0.0)
			{
				weighted += workload.radiusWeight * 0.5 * (workload.radiusScaleMin + workload.radiusScaleMax);
				totalWeight += workload.radiusWeight;
			}
		}
		return totalWeight > 0.0 ? weighted / totalWeight : 0.05;
	}

	std::vector<size_t> deepHandoffDepthCandidates(
		const Experiments::SchemaGenerationOptions& options,
		const std::vector<Experiments::WorkloadProfile>& workloads)
	{
		const size_t maxDepth = std::max<size_t>(2, options.maxDepth);
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

	SchemaLevelConfig makeDeepLevel(const std::string& typeName, size_t numLevels, size_t leafCapacity)
	{
		SchemaLevelConfig level;
		level.typeName = typeName;
		level.type = Config::parseDataStructureLevel(typeName);
		level.numLevels = std::max<size_t>(1, numLevels);
		level.leafCapacity = std::max<size_t>(2, leafCapacity);
		level.minPrimitivesToSplit = std::max<size_t>(2, level.leafCapacity / 4);
		level.axisPolicy = level.type == MultiDataStructure::KDTreeNode ? "median_longest_axis" : "";
		return level;
	}

	bool isDeepNestedSchema(const SchemaConfig& schema)
	{
		if (schema.levels.size() < 2)
			return false;
		std::set<std::string> types;
		for (const SchemaLevelConfig& level : schema.levels)
			types.insert(schemaTypeShortName(level));
		return types.size() >= 2;
	}

	SchemaLevelCondition makeDeepSecondStageCondition(
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
		condition.minPoints = minPoints;

		if (isRegularGridLevelName(secondType) || isHGridLevelName(secondType) ||
			isRegularGridLevelName(firstType) || isHGridLevelName(firstType))
		{
			condition.minDensity = density;
		}
		else if (Config::parseDataStructureLevel(secondType) == MultiDataStructure::OctreeNode)
		{
			condition.minHeightRatio = heightRatio;
		}
		else if (Config::parseDataStructureLevel(firstType) == MultiDataStructure::QuadTreeNode)
		{
			condition.minHeightRatio = heightRatio;
		}

		if (variant % 4 == 1 && extentX > 0.0)
			condition.minExtentX = extentX;
		else if (variant % 4 == 2 && extentZ > 0.0)
			condition.minExtentZ = extentZ;

		return condition;
	}

	std::vector<Experiments::SchemaCandidate> generateDeepNestedCandidates(
		const Experiments::SchemaGenerationOptions& baseOptions,
		const Experiments::ConditionDomain& domain,
		const std::vector<Experiments::WorkloadProfile>& workloads)
	{
		Experiments::SchemaGenerationOptions options = baseOptions;
		options.minBlocks = std::max<size_t>(2, options.minBlocks);
		options.maxBlocks = std::max(options.minBlocks, options.maxBlocks);
		options.maxDepth = std::max<size_t>(2, options.maxDepth);
		options.conditionalLevels = true;
		options.conditionalProbability = std::max(0.75, options.conditionalProbability);

		struct DeepFamily
		{
			const char* first = "";
			std::vector<const char*> seconds;
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
		const std::vector<double> heightValues = deepDoubleCandidates(domain.heightRatioThresholds, fallbackHeightRatioThresholds(), 6);
		const std::vector<double> densityValues = deepDoubleCandidates(domain.densityThresholds, { 0.0001, 0.001, 0.01, 0.1 }, 6);
		const std::vector<double> extentXValues = deepDoubleCandidates(domain.extentXThresholds, {}, 4);
		const std::vector<double> extentZValues = deepDoubleCandidates(domain.extentZThresholds, {}, 4);
		const std::array<size_t, 5> secondDepthSeeds = { 1, 2, 3, 4, 6 };

		std::unordered_set<std::string> seen;
		std::vector<Experiments::SchemaCandidate> candidates;
		candidates.reserve(options.count);

		size_t variant = 0;
		for (const DeepFamily& family : families)
		{
			for (const char* secondType : family.seconds)
			{
				for (const size_t firstDepth : firstDepths)
				{
					for (const size_t secondDepthSeed : secondDepthSeeds)
					{
						if (firstDepth + secondDepthSeed > options.maxDepth)
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
								schema.levels.push_back(makeDeepLevel(family.first, firstDepth, leaf));
								SchemaLevelConfig second = makeDeepLevel(
									secondType,
									secondDepthSeed,
									std::max<size_t>(options.minLeafCapacity, leaf / 2));
								second.condition = makeDeepSecondStageCondition(
									family.first,
									secondType,
									minPoints,
									height,
									density,
									extentX,
									extentZ,
									variant);
								schema.levels.push_back(second);
								normalizeSchemaForGeneration(schema, options);
								if (!isDeepNestedSchema(schema))
								{
									++variant;
									continue;
								}

								const std::string signature = schemaSignature(schema);
								if (seen.insert(signature).second)
									candidates.push_back(materializeGeneratedSchema(schema, "deep", options.outputDirectory));
								++variant;

								if (candidates.size() >= options.count)
									return candidates;
							}
						}
					}
				}
			}
		}

		if (candidates.size() < options.count)
		{
			Experiments::SchemaGenerationOptions fallback = options;
			fallback.count = options.count - candidates.size();
			fallback.minBlocks = 2;
			std::vector<Experiments::SchemaCandidate> randomNested = Experiments::generateSchemaCandidates(fallback, &domain);
			for (Experiments::SchemaCandidate& candidate : randomNested)
			{
				if (!isDeepNestedSchema(candidate.config))
					continue;
				const std::string signature = schemaSignature(candidate.config);
				if (seen.insert(signature).second)
					candidates.push_back(std::move(candidate));
				if (candidates.size() >= options.count)
					break;
			}
		}

		if (candidates.size() < options.count)
			std::cerr << "Warning: generated " << candidates.size() << " deep nested schemas from requested " << options.count << '\n';
		return candidates;
	}

	// Returns a key identifying the candidate's primary block type — what sits at the root of
	// its tree. Used to guarantee primary-block diversity when advancing top-K between rungs.
	// Two candidates that both start with Octree (regardless of their nested blocks) share a
	// key; this lets the diversifier preserve "at least one Octree-rooted survivor" even when
	// the score-sort is dominated by a different family.
	std::string primaryBlockKey(const SchemaConfig& schema)
	{
		if (schema.levels.empty())
			return "";
		return schemaTypeShortName(schema.levels.front());
	}

	std::vector<Experiments::SchemaCandidate> topCandidates(
		const std::vector<EvaluatedCandidate>& evaluations,
		size_t count)
	{
		// Score-sorted advancement, but with a per-primary-block diversity pass first. The
		// motivation is the proxy-rung asymmetry: visit-proxy scoring underestimates HGrid's
		// real wall-clock cost, so HGrid candidates sweep the top of R0's score-sort and the
		// shortlist becomes mono-cultural. By guaranteeing one survivor per primary block type
		// before filling the rest by score, the latency rung gets to compare apples to apples
		// (an Octree-rooted candidate vs. a HGrid-rooted candidate) instead of comparing HGrid
		// variants to each other.
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
			const Experiments::SchemaCandidate& candidate = evaluations[i].candidate;
			const std::string primary = primaryBlockKey(candidate.config);
			if (!primarySeen.insert(primary).second)
			{
				deferred.push_back(i);
				continue;
			}
			result.push_back(candidate);
			picked.insert(candidate.path.empty() ? candidate.name : candidate.path);
		}

		// Pass 2: fill remaining slots from the deferred pool, still in score order.
		for (const size_t i : deferred)
		{
			if (result.size() >= limit)
				break;
			const Experiments::SchemaCandidate& candidate = evaluations[i].candidate;
			result.push_back(candidate);
			picked.insert(candidate.path.empty() ? candidate.name : candidate.path);
		}

		const size_t diverseCount = primarySeen.size();
		if (diverseCount > 1 && diverseCount < result.size())
			std::cout << "    + advanced " << diverseCount << " distinct primary-block type(s) into next rung\n";

		// Force-promote any baseline candidate that wasn't already in the top-K so the operator
		// always sees how naive single-block structures perform at the next stage.
		size_t baselinesAdded = 0;
		for (const EvaluatedCandidate& evaluation : evaluations)
		{
			if (!evaluation.candidate.isBaseline)
				continue;
			const std::string key = evaluation.candidate.path.empty() ? evaluation.candidate.name : evaluation.candidate.path;
			if (!picked.insert(key).second)
				continue;
			result.push_back(evaluation.candidate);
			++baselinesAdded;
		}
		if (baselinesAdded > 0)
			std::cout << "    + force-promoted " << baselinesAdded << " baseline schema(s) past rank cut\n";
		return result;
	}

	bool recordCountsAsNested(const Experiments::SchemaSearchRecord& record)
	{
		return !record.isBaseline &&
			record.activeStructureTypes >= 2 &&
			record.nestedActiveFraction >= 0.05;
	}

	bool evaluationCountsAsNested(const EvaluatedCandidate& evaluation)
	{
		for (const Experiments::SchemaSearchRecord& record : evaluation.records)
		{
			if (recordCountsAsNested(record))
				return true;
		}
		return false;
	}

	void appendUniqueCandidate(
		std::vector<Experiments::SchemaCandidate>& out,
		std::unordered_set<std::string>& picked,
		const Experiments::SchemaCandidate& candidate)
	{
		const std::string key = candidate.path.empty() ? candidate.config.name : candidate.path;
		if (picked.insert(key).second)
			out.push_back(candidate);
	}

	std::vector<Experiments::SchemaCandidate> topDeepNestedCandidates(
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
			if (!evaluation.candidate.isBaseline && evaluationCountsAsNested(evaluation))
				appendUniqueCandidate(result, picked, evaluation.candidate);
		}

		if (result.size() < limit)
		{
			for (const EvaluatedCandidate& evaluation : evaluations)
			{
				if (result.size() >= limit)
					break;
				if (!evaluation.candidate.isBaseline)
					appendUniqueCandidate(result, picked, evaluation.candidate);
			}
		}

		if (includeBaselines)
		{
			size_t baselinesAdded = 0;
			for (const EvaluatedCandidate& evaluation : evaluations)
			{
				if (!evaluation.candidate.isBaseline)
					continue;
				const size_t previousSize = result.size();
				appendUniqueCandidate(result, picked, evaluation.candidate);
				if (result.size() != previousSize)
					++baselinesAdded;
			}
			if (baselinesAdded > 0)
				std::cout << "    + force-promoted " << baselinesAdded << " baseline schema(s) past deep-search rank cut\n";
		}

		return result;
	}

	std::optional<Experiments::SchemaCandidate> bestBaselineCandidate(const std::vector<EvaluatedCandidate>& evaluations)
	{
		for (const EvaluatedCandidate& evaluation : evaluations)
		{
			if (evaluation.candidate.isBaseline)
				return evaluation.candidate;
		}
		return std::nullopt;
	}

	struct LocalCellStats
	{
		size_t count = 0;
		glm::vec3 min = glm::vec3(std::numeric_limits<float>::max());
		glm::vec3 max = glm::vec3(std::numeric_limits<float>::lowest());
	};

	struct LocalOpportunityStats
	{
		std::map<std::string, size_t> suggestedTypes;
		double dominantShare = 0.0;
		size_t testedCells = 0;
	};

	LocalOpportunityStats estimateLocalOpportunity(const PointCloud& cloud, size_t pointCap)
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
			++cell.count;
			cell.min = glm::min(cell.min, position);
			cell.max = glm::max(cell.max, position);
		}

		std::vector<double> densities;
		std::vector<size_t> counts;
		for (const LocalCellStats& cell : cells)
		{
			if (cell.count == 0)
				continue;
			const glm::vec3 cellExtent = glm::max(cell.max - cell.min, glm::vec3(static_cast<float>(EPSILON)));
			const double volume = static_cast<double>(cellExtent.x) * cellExtent.y * cellExtent.z;
			densities.push_back(volume > EPSILON ? static_cast<double>(cell.count) / volume : 0.0);
			counts.push_back(cell.count);
		}
		if (counts.empty())
			return result;
		std::sort(counts.begin(), counts.end());
		std::sort(densities.begin(), densities.end());
		const size_t medianCount = counts[counts.size() / 2];
		const double medianDensity = densities[densities.size() / 2];

		for (const LocalCellStats& cell : cells)
		{
			if (cell.count < std::max<size_t>(4, medianCount / 2))
				continue;

			const glm::vec3 cellExtent = glm::max(cell.max - cell.min, glm::vec3(static_cast<float>(EPSILON)));
			const double horizontal = std::max({ static_cast<double>(cellExtent.x), static_cast<double>(cellExtent.y), EPSILON });
			const double heightRatio = static_cast<double>(cellExtent.z) / horizontal;
			const double minExtent = std::max(EPSILON, static_cast<double>(std::min({ cellExtent.x, cellExtent.y, cellExtent.z })));
			const double maxExtent = static_cast<double>(std::max({ cellExtent.x, cellExtent.y, cellExtent.z }));
			const double density = static_cast<double>(cell.count) /
				std::max(EPSILON, static_cast<double>(cellExtent.x) * cellExtent.y * cellExtent.z);

			std::string winner = "ot";
			if (heightRatio < 0.18)
				winner = "qt";
			else if (density > medianDensity * 1.75 && cell.count > medianCount)
				winner = "rg";
			else if (maxExtent / minExtent > 3.0)
				winner = "kd";

			++result.suggestedTypes[winner];
			++result.testedCells;
		}

		size_t dominant = 0;
		for (const auto& [typeName, count] : result.suggestedTypes)
			dominant = std::max(dominant, count);
		result.dominantShare = result.testedCells > 0
			? static_cast<double>(dominant) / static_cast<double>(result.testedCells)
			: 0.0;
		return result;
	}

	void printLocalOpportunity(
		const SearchDataset& dataset,
		const Experiments::SchemaCandidate& bestBaseline,
		size_t pointCap)
	{
		const LocalOpportunityStats local = estimateLocalOpportunity(dataset.cloud, pointCap);
		const std::string globalType = bestBaseline.config.levels.empty()
			? std::string()
			: schemaTypeShortName(bestBaseline.config.levels.front());

		std::cout << "    local opportunity: " << local.testedCells << " shallow occupied cells";
		if (local.testedCells == 0)
		{
			std::cout << " (insufficient local samples)\n";
			return;
		}

		std::cout << ", suggested second-stage types ";
		bool first = true;
		bool differsFromGlobal = false;
		for (const auto& [typeName, count] : local.suggestedTypes)
		{
			if (!first)
				std::cout << "; ";
			first = false;
			std::cout << typeName << "=" << count;
			if (typeName != globalType)
				differsFromGlobal = true;
		}
		std::cout << ", dominant share " << local.dominantShare;
		if (!differsFromGlobal || local.suggestedTypes.size() <= 1)
			std::cout << " (low nested-opportunity signal)";
		else
			std::cout << " (local winners differ from global " << globalType << ")";
		std::cout << '\n';
	}

	std::vector<EvaluatedCandidate> evaluateAutoConditionStage(
		const std::string& label,
		const std::vector<Experiments::SchemaCandidate>& candidates,
		const std::vector<DatasetContext>& contexts,
		const std::vector<Experiments::WorkloadProfile>& workloads,
		const Experiments::SchemaSearchOptions& options);

	void runDeepNestedDiagnostics(
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
		diagnosticOptions.evaluator = "cpu";
		diagnosticOptions.weights.useVisitProxy = false;
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

		const Experiments::SchemaCandidate& bestBaseline = baselineEvaluations.front().candidate;
		std::cout << "    best pure baseline: " << bestBaseline.config.name
			<< " aggregate score " << baselineEvaluations.front().aggregateScore << '\n';
		for (const SearchDataset& dataset : datasets)
			printLocalOpportunity(dataset, bestBaseline, options.autoConditions.proxyPointCap);
	}

	std::vector<Experiments::WorkloadProfile> makeRobustnessWorkloads(
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
				copy.name = workload.name + "_robust_seed" + std::to_string(seedIndex);
				copy.numQueries = queryCount;
				copy.querySeed = workload.querySeed + static_cast<uint32_t>(7919 * (seedIndex + 1));
				robust.push_back(copy);
			}
		}
		return robust;
	}

	void appendBaselineControls(
		std::vector<Experiments::SchemaCandidate>& candidates,
		const std::vector<Experiments::SchemaCandidate>& baselines)
	{
		std::unordered_set<std::string> picked;
		for (const Experiments::SchemaCandidate& candidate : candidates)
			picked.insert(candidate.path.empty() ? candidate.config.name : candidate.path);
		for (const Experiments::SchemaCandidate& baseline : baselines)
			appendUniqueCandidate(candidates, picked, baseline);
	}

	void runDeepCudaConfirmation(
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
				picked.insert(candidate.path.empty() ? candidate.config.name : candidate.path);
			appendUniqueCandidate(candidates, picked, baseline.value());
		}
		if (candidates.empty())
			return;

		Experiments::SchemaSearchOptions cudaOptions = options;
		cudaOptions.evaluator = "cuda";
		cudaOptions.cuda.builder = "mixed";
		if (cudaOptions.cuda.device < 0)
			cudaOptions.cuda.device = 0;
		cudaOptions.weights.useVisitProxy = false;

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
			std::cout << "    cuda confirmation best: " << cudaEvaluations.front().candidate.config.name
				<< " aggregate score " << cudaEvaluations.front().aggregateScore << " (report only)\n";
	}

	std::vector<EvaluatedCandidate> evaluateAutoConditionStage(
		const std::string& label,
		const std::vector<Experiments::SchemaCandidate>& candidates,
		const std::vector<DatasetContext>& contexts,
		const std::vector<Experiments::WorkloadProfile>& workloads,
		const Experiments::SchemaSearchOptions& options)
	{
		const size_t parallelWorkers = (!useCudaEvaluator(options) && options.parallelDispatch > 1)
			? std::min(options.parallelDispatch, candidates.size())
			: 1;

		std::cout << "  auto-conditions " << label << ": " << candidates.size() << " candidates, "
			<< (workloads.empty() ? size_t(0) : workloads.front().numQueries) << " queries";
		if (parallelWorkers > 1)
			std::cout << " (parallel CPU workers: " << parallelWorkers << ")";
		std::cout << '\n';

		std::vector<EvaluatedCandidate> evaluations(candidates.size());
		if (parallelWorkers > 1)
		{
			// CPU-only parallel dispatch. The CUDA per-builder cache is not thread-safe so the
			// CUDA path stays serial; only the score cache (Phase 1) survives concurrency here,
			// and it is internally locked.
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
					<< candidates[i].config.name << "  aggregate score " << evaluations[i].aggregateScore << '\n';
			}
		}
		else
		{
			CudaIndexCache cudaCache;
			for (size_t i = 0; i < candidates.size(); ++i)
			{
				const Experiments::SchemaCandidate& candidate = candidates[i];
				std::cout << "    " << label << " [" << (i + 1) << "/" << candidates.size() << "] " << candidate.config.name << '\n';
				evaluations[i] = evaluateCandidate(candidate, contexts, workloads, options, &cudaCache);
				std::cout << "      aggregate score " << evaluations[i].aggregateScore << '\n';
			}
		}

		sortEvaluations(evaluations);
		if (!evaluations.empty())
			std::cout << "    " << label << " best: " << evaluations.front().candidate.config.name << " score " << evaluations.front().aggregateScore << '\n';
		return evaluations;
	}

	std::string jsonEscape(const std::string& value)
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

	std::string safeFileStem(std::string value)
	{
		for (char& c : value)
		{
			if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-')
				c = '_';
		}
		return value.empty() ? "auto_conditions" : value;
	}

	void writeSchemaCopy(const Experiments::SchemaCandidate& candidate, const std::filesystem::path& path)
	{
		if (path.has_parent_path())
			std::filesystem::create_directories(path.parent_path());
		std::ofstream output(path);
		if (!output.is_open())
			throw std::runtime_error("Unable to write auto-condition schema: " + path.string());
		output << schemaConfigToJson(candidate.config);
	}

	void writeMeasuredSelectorArtifact(
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
		output << "  \"dataset_name\": \"" << jsonEscape(selected.datasetName) << "\",\n";
		output << "  \"dataset_path\": \"" << jsonEscape(selected.datasetSource) << "\",\n";
		output << "  \"workload_name\": \"" << jsonEscape(selected.workloadName) << "\",\n";
		output << "  \"workload_profile_path\": \"" << jsonEscape(selected.workloadName) << "\",\n";
		output << "  \"selected_schema\": {\n";
		output << "    \"name\": \"" << jsonEscape(selected.schemaName) << "\",\n";
		output << "    \"path\": \"" << jsonEscape(selected.schemaPath) << "\",\n";
		output << "    \"score\": " << selected.score << ",\n";
		output << "    \"avg_latency_ms\": " << selected.queryMetrics.averageLatencyMs << ",\n";
		output << "    \"build_time_ms\": " << selected.buildMetrics.buildTimeMs << ",\n";
		output << "    \"memory_estimate_bytes\": " << selected.buildMetrics.memoryEstimateBytes << "\n";
		output << "  },\n";
		output << "  \"candidate_scores\": [\n";
		bool first = true;
		for (const Experiments::SchemaSearchRecord& record : records)
		{
			if (record.datasetName != selected.datasetName || record.workloadName != selected.workloadName)
				continue;
			output << (first ? "" : ",\n");
			first = false;
			output << "    {\n";
			output << "      \"name\": \"" << jsonEscape(record.schemaName) << "\",\n";
			output << "      \"path\": \"" << jsonEscape(record.schemaPath) << "\",\n";
			output << "      \"score\": " << record.score << ",\n";
			output << "      \"avg_latency_ms\": " << record.queryMetrics.averageLatencyMs << ",\n";
			output << "      \"build_time_ms\": " << record.buildMetrics.buildTimeMs << ",\n";
			output << "      \"memory_estimate_bytes\": " << record.buildMetrics.memoryEstimateBytes << "\n";
			output << "    }";
		}
		output << "\n  ],\n";
		output << "  \"source_csv\": \"" << jsonEscape(sourceCsv) << "\",\n";
		output << "  \"note\": \"Auto-condition measured selector: overfit to this point cloud and workload by staged conditional schema tuning.\"\n";
		output << "}\n";
	}

	void writeAutoConditionArtifacts(
		std::vector<Experiments::SchemaSearchRecord>& records,
		const std::vector<EvaluatedCandidate>& finalEvaluations,
		const Experiments::SchemaSearchOptions& options)
	{
		if (records.empty() || finalEvaluations.empty())
			return;

		std::unordered_map<std::string, const Experiments::SchemaCandidate*> candidatesByKey;
		for (const EvaluatedCandidate& evaluation : finalEvaluations)
			candidatesByKey[candidateKey(evaluation.candidate.config.name, evaluation.candidate.path)] = &evaluation.candidate;

		std::vector<Experiments::SchemaSearchRecord> bestRecords = Experiments::selectBestRecords(records);
		for (Experiments::SchemaSearchRecord& best : bestRecords)
		{
			const std::string originalPath = best.schemaPath;
			const auto found = candidatesByKey.find(candidateKey(best.schemaName, originalPath));
			if (found == candidatesByKey.end())
				continue;

			const std::filesystem::path outputPath =
				std::filesystem::path(options.autoConditions.outputDirectory) /
				(safeFileStem(best.datasetName + "_" + best.workloadName) + "_best_schema.json");
			writeSchemaCopy(*found->second, outputPath);

			for (Experiments::SchemaSearchRecord& record : records)
			{
				if (record.datasetName == best.datasetName &&
					record.workloadName == best.workloadName &&
					record.schemaName == best.schemaName &&
					record.schemaPath == originalPath)
				{
					record.schemaPath = outputPath.string();
				}
			}
			best.schemaPath = outputPath.string();
			std::cout << "  auto-condition schema: " << outputPath.string() << '\n';
		}

		bestRecords = Experiments::selectBestRecords(records);
		if (!bestRecords.empty())
		{
			writeMeasuredSelectorArtifact(options.autoConditions.selectorOutputPath, bestRecords.front(), records, options.csvPath);
			if (!options.autoConditions.selectorOutputPath.empty())
				std::cout << "  auto-condition selector: " << options.autoConditions.selectorOutputPath << '\n';
		}
	}

	std::vector<Experiments::SchemaSearchRecord> runAutoConditionSearch(
		const Experiments::SchemaSearchOptions& options,
		const std::vector<SearchDataset>& datasets,
		const std::vector<Experiments::WorkloadProfile>& workloads,
		const std::vector<Experiments::SchemaCandidate>& configuredSchemas)
	{
		const Experiments::AutoConditionOptions& autoOptions = options.autoConditions;
		if (datasets.empty())
			return {};

		for (const Experiments::WorkloadProfile& workload : workloads)
		{
			if (workload.knnWeight > 0.0)
				std::cout << "  warning: auto-condition tuning is optimized for range/radius workloads; KNN in workload '" << workload.name << "' may make search expensive\n";
		}

		Experiments::ConditionDomain domain = Experiments::estimateConditionDomain(datasets.front().cloud, autoOptions.proxyPointCap);
		std::cout << "  auto-conditions: domain from " << domain.samplePoints << " sampled points, "
			<< domain.sketchNodes << " sketch nodes\n";
		std::cout << "    point thresholds: " << domain.pointThresholds.size()
			<< ", density thresholds: " << domain.densityThresholds.size()
			<< ", height thresholds: " << domain.heightRatioThresholds.size() << '\n';

		Experiments::SchemaGenerationOptions generation = options.generation;
		const size_t proxyBudget = std::max<size_t>(1, autoOptions.proxyCandidateCount);
		generation.conditionalLevels = true;
		generation.conditionalProbability = std::max(generation.conditionalProbability, 0.75);

		std::vector<Experiments::SchemaCandidate> baselines;
		for (const Experiments::SchemaCandidate& candidate : configuredSchemas)
		{
			if (candidate.isBaseline)
				baselines.push_back(candidate);
		}

		std::vector<Experiments::SchemaCandidate> candidates;
		if (options.deepNestedSearch)
		{
			runDeepNestedDiagnostics(options, datasets, workloads, baselines);
			generation.count = proxyBudget;
			generation.minBlocks = std::max<size_t>(2, generation.minBlocks);
			generation.maxBlocks = std::max(generation.maxBlocks, generation.minBlocks);
			std::vector<Experiments::SchemaCandidate> generated = generateDeepNestedCandidates(generation, domain, workloads);
			candidates.insert(candidates.end(), std::make_move_iterator(generated.begin()), std::make_move_iterator(generated.end()));
		}
		else
		{
			generation.count = configuredSchemas.size() >= proxyBudget
				? size_t(0)
				: proxyBudget - configuredSchemas.size();
			candidates = configuredSchemas;
			if (generation.count > 0)
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
		if (options.deepNestedSearch)
			discoveryOptions.evaluator = "cpu";
		const bool cudaEvaluator = useCudaEvaluator(discoveryOptions);
		const std::vector<SearchDataset> proxyDatasets = makeProxyDatasets(datasets, autoOptions.proxyPointCap);
		const std::vector<Experiments::WorkloadProfile> proxyWorkloads = withQueryCount(workloads, autoOptions.proxyQueryCount);
		const std::vector<DatasetContext> proxyContexts = makeDatasetContexts(proxyDatasets, proxyWorkloads, cudaEvaluator);
		// Proxy stage ranks candidates by deterministic visit-count surrogate (cheap, noise-free)
		// rather than wall-clock latency. The shortlist and confirmation stages fall back to latency.
		Experiments::SchemaSearchOptions proxyOptions = discoveryOptions;
		proxyOptions.weights.useVisitProxy = true;
		if (proxyOptions.weights.visitProxyAlpha <= 0.0)
			proxyOptions.weights.visitProxyAlpha = 0.1;
		// Without a build-time penalty, the visit-proxy can promote pathological schemas with
		// low per-query visit counts but catastrophic build cost (deep hierarchical grids, etc.).
		// They look great on the downsampled proxy cloud and then stall the shortlist when rebuilt
		// on the full cloud. A small lambdaBuild keeps build time honest in the proxy ranking.
		if (proxyOptions.weights.lambdaBuild <= 0.0)
			proxyOptions.weights.lambdaBuild = 1.0;
		std::cout << "  auto-conditions: proxy stage uses visit-count surrogate (alpha="
			<< proxyOptions.weights.visitProxyAlpha
			<< ", lambdaBuild=" << proxyOptions.weights.lambdaBuild << ")\n";
		std::vector<EvaluatedCandidate> proxyEvaluations = evaluateAutoConditionStage(
			"proxy",
			candidates,
			proxyContexts,
			proxyWorkloads,
			proxyOptions);

		std::vector<Experiments::SchemaCandidate> shortlist = options.deepNestedSearch
			? topDeepNestedCandidates(proxyEvaluations, autoOptions.finalTopK, false)
			: topCandidates(proxyEvaluations, autoOptions.finalTopK);
		if (options.deepNestedSearch)
			appendBaselineControls(shortlist, baselines);
		const std::vector<Experiments::WorkloadProfile> shortWorkloads = withQueryCount(workloads, options.deepNestedSearch ? 32 : 16);
		const std::vector<DatasetContext> shortContexts = makeDatasetContexts(datasets, shortWorkloads, cudaEvaluator);
		std::vector<EvaluatedCandidate> shortEvaluations = evaluateAutoConditionStage(
			"shortlist",
			shortlist,
			shortContexts,
			shortWorkloads,
			discoveryOptions);

		std::vector<Experiments::SchemaCandidate> confirmation = options.deepNestedSearch
			? topDeepNestedCandidates(shortEvaluations, autoOptions.confirmationTopK, true)
			: topCandidates(shortEvaluations, autoOptions.confirmationTopK);
		const std::vector<DatasetContext> confirmationContexts = makeDatasetContexts(datasets, workloads, cudaEvaluator);
		std::vector<EvaluatedCandidate> finalEvaluations = evaluateAutoConditionStage(
			"confirmation",
			confirmation,
			confirmationContexts,
			workloads,
			discoveryOptions);

		std::vector<Experiments::SchemaSearchRecord> records;
		for (const EvaluatedCandidate& evaluation : finalEvaluations)
		{
			appendRecords(records, evaluation);
			for (const Experiments::SchemaSearchRecord& record : evaluation.records)
				emitProgress(options, record);
		}

		std::vector<EvaluatedCandidate> artifactEvaluations = finalEvaluations;
		if (options.deepNestedSearch)
		{
			std::vector<Experiments::SchemaCandidate> robustCandidates = topDeepNestedCandidates(finalEvaluations, 6, false);
			if (const std::optional<Experiments::SchemaCandidate> baseline = bestBaselineCandidate(finalEvaluations))
			{
				std::unordered_set<std::string> picked;
				for (const Experiments::SchemaCandidate& candidate : robustCandidates)
					picked.insert(candidate.path.empty() ? candidate.config.name : candidate.path);
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
				for (const Experiments::SchemaSearchRecord& record : evaluation.records)
					emitProgress(options, record);
			}
			if (!robustEvaluations.empty())
				artifactEvaluations = robustEvaluations;

			runDeepCudaConfirmation(options, datasets, workloads, artifactEvaluations);
		}

		writeAutoConditionArtifacts(records, artifactEvaluations, options);
		return records;
	}

	// Runs one batch of candidates through the configured rung schedule. Each rung gets its own
	// `WorkloadProfile` (via withQueryCount) and its own `ScoreWeights` override (visit-proxy on/off,
	// alpha). The cache fingerprint already mixes the visit-proxy flag and the effective query count
	// (workload.numQueries) so cached rows do not bleed between rungs even though the underlying
	// schema/dataset are the same.
	//
	// Returns the final-rung evaluations (sorted best-first). `finalRecordsOut` receives the
	// SchemaSearchRecords from the highest-fidelity rung only — intermediate rungs still populate
	// the score cache but their rows are not duplicated into the CSV.
	std::vector<EvaluatedCandidate> runRungSchedule(
		const std::string& batchLabel,
		const std::vector<Experiments::SchemaCandidate>& inputBatch,
		const std::vector<const SearchDataset*>& datasetPtrs,
		const std::vector<Experiments::WorkloadProfile>& baseWorkloads,
		const Experiments::SchemaSearchOptions& options,
		std::vector<Experiments::SchemaSearchRecord>* finalRecordsOut)
	{
		const Experiments::RungSchedule& schedule = options.evolution.rungSchedule;
		if (schedule.rungs.empty() || inputBatch.empty())
			return {};

		const bool cudaEvaluator = useCudaEvaluator(options);
		std::vector<Experiments::SchemaCandidate> current = inputBatch;
		std::vector<EvaluatedCandidate> evaluations;

		for (size_t r = 0; r < schedule.rungs.size(); ++r)
		{
			const Experiments::RungSpec& rung = schedule.rungs[r];

			const std::vector<Experiments::WorkloadProfile> rungWorkloads = withQueryCount(baseWorkloads, rung.queryCountOverride);
			const std::vector<DatasetContext> rungContexts = makeDatasetContextsFromPointers(datasetPtrs, rungWorkloads, cudaEvaluator);

			Experiments::SchemaSearchOptions rungOptions = options;
			rungOptions.weights.useVisitProxy = rung.useVisitProxy;
			if (rung.useVisitProxy && rung.visitProxyAlpha > 0.0)
				rungOptions.weights.visitProxyAlpha = rung.visitProxyAlpha;

			std::ostringstream label;
			label << batchLabel << " " << rung.name << " ("
				<< current.size() << " cand";
			if (rung.queryCountOverride > 0)
				label << ", " << rung.queryCountOverride << "q";
			label << (rung.useVisitProxy ? ", visit-proxy" : ", latency");
			label << ")";

			evaluations = evaluateAutoConditionStage(label.str(), current, rungContexts, rungWorkloads, rungOptions);

			const bool isFinalRung = (r + 1 == schedule.rungs.size());
			if (isFinalRung && finalRecordsOut != nullptr)
			{
				for (const EvaluatedCandidate& evaluation : evaluations)
				{
					for (const Experiments::SchemaSearchRecord& record : evaluation.records)
					{
						finalRecordsOut->push_back(record);
						emitProgress(options, record);
					}
				}
			}

			if (!isFinalRung)
			{
				const size_t keep = (rung.advanceTopK == 0)
					? evaluations.size()
					: std::min(rung.advanceTopK, evaluations.size());

				// Diverse-by-primary advancement: guarantee each distinct primary-block type
				// has at least one survivor passing to the next rung before filling the rest by
				// score. Without this, a proxy-rung asymmetry (visit-proxy under-counts HGrid's
				// real GPU cost, for example) can sweep HGrid descendants into every promotion
				// slot, starving the latency rung of any Octree/KDTree-rooted candidates to
				// compare against. Reuses the same `primaryBlockKey` helper as the auto-condition
				// `topCandidates` filter for consistency.
				std::vector<size_t> advanceOrder;
				advanceOrder.reserve(keep);
				std::unordered_set<std::string> primarySeen;
				std::vector<size_t> deferred;
				deferred.reserve(evaluations.size());
				for (size_t i = 0; i < evaluations.size() && advanceOrder.size() < keep; ++i)
				{
					const std::string primary = primaryBlockKey(evaluations[i].candidate.config);
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
					current.push_back(evaluations[i].candidate);
					trimmed.push_back(std::move(evaluations[i]));
				}
				evaluations = std::move(trimmed);

				if (primarySeen.size() > 1)
					std::cout << "      rung '" << rung.name << "' advancing " << current.size()
						<< " candidate(s), " << primarySeen.size()
						<< " distinct primary-block type(s)\n";
			}
		}

		return evaluations;
	}

	std::vector<Experiments::SchemaSearchRecord> runEvolutionarySchemaSearch(
		const Experiments::SchemaSearchOptions& options,
		const std::vector<DatasetContext>& datasets,
		const std::vector<Experiments::WorkloadProfile>& workloads,
		const std::vector<Experiments::SchemaCandidate>& initialCandidates,
		const Experiments::ConditionDomain* conditionDomain = nullptr)
	{
		const Experiments::EvolutionOptions& evolution = options.evolution;
		const size_t populationSize = std::max<size_t>(1, evolution.populationSize);
		const size_t eliteCount = std::max<size_t>(1, evolution.eliteCount);
		const double randomFraction = std::clamp(evolution.randomImmigrationRate, 0.0, 1.0);

		std::mt19937 rng(evolution.seed);
		std::unordered_set<std::string> seenSignatures;
		std::vector<EvaluatedCandidate> archive;
		std::vector<Experiments::SchemaSearchRecord> records;
		CudaIndexCache cudaCache;

		// Phase B4 diversity audit: count distinct topology hashes (level-type sequence only,
		// ignoring leaf caps / depth / conditions) across every evaluated candidate. Reported at
		// end of run so we can tell whether the GA explored qualitatively different shapes or
		// just polished a single corridor of the search space.
		std::map<std::string, size_t> topologyCounts;
		size_t totalEvaluatedCandidates = 0;

		// Picks the right ordering helper based on options.useNsga2Ranking. NSGA-II ranks by
		// non-dominated front first (smaller = better), ties broken by larger crowding distance
		// (more isolated = preferred). Scalar fallback keeps the single-score path bit-for-bit
		// identical to pre-B4 behavior.
		auto rankArchive = [&]() {
			if (evolution.useNsga2Ranking)
				nsga2RankAndSort(archive);
			else
				sortEvaluations(archive);
		};

		std::vector<Experiments::SchemaCandidate> batch = uniqueCandidates(initialCandidates, seenSignatures);
		if (batch.empty())
			throw std::runtime_error("Evolutionary schema optimizer has no initial population.");

		std::cout << "  optimizer: evolutionary mutation search\n";
		std::cout << "    generations: " << evolution.generations << '\n';
		std::cout << "    population per generation: " << populationSize << '\n';
		std::cout << "    elites: " << eliteCount << '\n';
		std::cout << "    mutation rate: " << evolution.mutationRate << '\n';
		std::cout << "    random immigration: " << randomFraction << '\n';

		const bool useRungSchedule = !evolution.rungSchedule.rungs.empty();
		std::vector<const SearchDataset*> datasetPtrs;
		Experiments::SurrogateAcquisition surrogate = Experiments::loadSurrogateAcquisition(evolution.rungSchedule.surrogateModelPath);
		if (surrogate.active)
		{
			std::cout << "    surrogate acquisition: " << surrogate.modelPath
				<< " (pool=" << evolution.rungSchedule.surrogateCandidatePool
				<< ", proposals/gen=" << evolution.rungSchedule.surrogateProposalsPerStep << ")\n";
		}
		if (useRungSchedule)
		{
			datasetPtrs.reserve(datasets.size());
			for (const DatasetContext& context : datasets)
				datasetPtrs.push_back(context.dataset);

			std::cout << "    rung schedule: " << evolution.rungSchedule.rungs.size() << " rungs\n";
			for (size_t r = 0; r < evolution.rungSchedule.rungs.size(); ++r)
			{
				const Experiments::RungSpec& rung = evolution.rungSchedule.rungs[r];
				std::cout << "      [" << r << "] " << rung.name
					<< (rung.useVisitProxy ? " (visit-proxy)" : " (latency)");
				if (rung.queryCountOverride > 0)
					std::cout << " queries=" << rung.queryCountOverride;
				if (rung.advanceTopK > 0 && r + 1 < evolution.rungSchedule.rungs.size())
					std::cout << " advance top-" << rung.advanceTopK;
				std::cout << '\n';
			}
		}

		const size_t parallelWorkers = (!useCudaEvaluator(options) && options.parallelDispatch > 1)
			? options.parallelDispatch
			: 1;
		if (parallelWorkers > 1)
			std::cout << "    parallel candidate dispatch: " << parallelWorkers << " CPU workers\n";

		auto evaluateBatch = [&](const std::vector<Experiments::SchemaCandidate>& candidates, const std::string& label) {
			if (candidates.empty())
				return;

			// Topology audit: every candidate that makes it this far counts, whether or not it
			// survives the rung schedule's downstream filters. Lets the diversity report capture
			// the *attempted* search breadth, not just what the front rewarded.
			for (const Experiments::SchemaCandidate& candidate : candidates)
			{
				++topologyCounts[schemaTopologyKey(candidate.config)];
				++totalEvaluatedCandidates;
			}

			if (useRungSchedule)
			{
				std::vector<EvaluatedCandidate> finalEvaluations = runRungSchedule(
					label, candidates, datasetPtrs, workloads, options, &records);
				for (size_t i = 0; i < finalEvaluations.size(); ++i)
				{
					std::cout << "      " << label << " final [" << (i + 1) << "/" << finalEvaluations.size() << "] "
						<< finalEvaluations[i].candidate.config.name
						<< "  aggregate score " << finalEvaluations[i].aggregateScore << '\n';
					archive.push_back(std::move(finalEvaluations[i]));
				}
				rankArchive();
				if (!archive.empty())
					std::cout << "      best so far: " << archive.front().candidate.config.name
						<< " score " << archive.front().aggregateScore << '\n';
				return;
			}

			std::vector<EvaluatedCandidate> evaluations(candidates.size());
			if (parallelWorkers > 1)
			{
				// CPU-only parallel dispatch. The CUDA index cache is bypassed because its
				// per-builder build cache is not thread-safe. The score cache is internally
				// synchronised so it is safe to share across workers.
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
						<< candidates[i].config.name << "  aggregate score " << evaluation.aggregateScore << '\n';
					appendRecords(records, evaluation);
					for (const Experiments::SchemaSearchRecord& record : evaluation.records)
						emitProgress(options, record);
					archive.push_back(std::move(evaluation));
				}
			}
			else
			{
				for (size_t i = 0; i < candidates.size(); ++i)
				{
					const Experiments::SchemaCandidate& candidate = candidates[i];
					std::cout << "      " << label << " [" << (i + 1) << "/" << candidates.size() << "] " << candidate.config.name << '\n';
					const auto candidateStart = std::chrono::steady_clock::now();
					EvaluatedCandidate evaluation = evaluateCandidate(candidate, datasets, workloads, options, &cudaCache);
					const auto candidateElapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - candidateStart).count();
					std::cout << "        aggregate score " << evaluation.aggregateScore;
					if (!evaluation.records.empty() && evaluation.records.front().backend == "cuda")
						std::cout << " (" << cudaBuilderDisplayName(evaluation.records.front().cudaBuilder) << ")";
					std::cout << " [" << candidateElapsed << " s]";
					std::cout << '\n';
					if (candidateElapsed > 30.0)
						std::cout << "        WARNING: candidate '" << candidate.config.name
							<< "' took " << candidateElapsed << " s. Raise --cuda-memory-budget-mb or restrict the schema generator if this repeats.\n";
					appendRecords(records, evaluation);
					for (const Experiments::SchemaSearchRecord& record : evaluation.records)
						emitProgress(options, record);
					archive.push_back(std::move(evaluation));
				}
			}
			rankArchive();
			if (!archive.empty())
				std::cout << "      best so far: " << archive.front().candidate.config.name << " score " << archive.front().aggregateScore << '\n';
		};

		evaluateBatch(batch, "initial");

		for (size_t generation = 1; generation <= evolution.generations; ++generation)
		{
			rankArchive();
			const size_t currentEliteCount = std::min(eliteCount, archive.size());
			if (currentEliteCount == 0)
				break;

			std::vector<Experiments::SchemaCandidate> children;
			children.reserve(populationSize);

			const size_t randomCount = std::min(populationSize, static_cast<size_t>(std::round(static_cast<double>(populationSize) * randomFraction)));
			if (randomCount > 0)
			{
				Experiments::SchemaGenerationOptions randomOptions = options.generation;
				randomOptions.count = randomCount;
				randomOptions.seed = rng();
				std::vector<Experiments::SchemaCandidate> immigrants = Experiments::generateSchemaCandidates(randomOptions, conditionDomain);
				for (const Experiments::SchemaCandidate& immigrant : immigrants)
				{
					const std::string signature = schemaSignature(immigrant.config);
					if (!seenSignatures.insert(signature).second)
						continue;
					children.push_back(immigrant);
					if (children.size() >= populationSize)
						break;
				}
			}

			// Bayesian acquisition step: ask the surrogate to rank a fresh pool and inject its
			// top-K predictions as additional children alongside random immigrants. Cheap (no
			// measurement) and decouples optimizer progress from the GA's pure mutation noise.
			if (surrogate.active
				&& evolution.rungSchedule.surrogateCandidatePool > 0
				&& evolution.rungSchedule.surrogateProposalsPerStep > 0
				&& !datasets.empty()
				&& !workloads.empty()
				&& children.size() < populationSize)
			{
				const size_t budget = populationSize - children.size();
				const size_t proposalCount = std::min(evolution.rungSchedule.surrogateProposalsPerStep, budget);
				std::vector<Experiments::SchemaCandidate> proposals = Experiments::acquireSurrogateProposals(
					surrogate,
					options.generation,
					evolution.rungSchedule.surrogateCandidatePool,
					proposalCount,
					datasets.front().dataset->cloud,
					workloads.front(),
					rng());
				size_t accepted = 0;
				for (const Experiments::SchemaCandidate& proposal : proposals)
				{
					const std::string signature = schemaSignature(proposal.config);
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
			const double crossoverProbability = std::clamp(evolution.crossoverRate, 0.0, 1.0);
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
					// Champion-anchored crossover: archive[0] is always one parent (it's the
					// scalar champion after NSGA-II's tie-break + the post-sort iter_swap), the
					// other is sampled from the remaining elites. This guarantees every
					// crossover child inherits half its blocks from the current best-known
					// schema and explores the other half. Without anchoring, random elite-elite
					// splices often produce children worse than either parent because crossover
					// has no signal about which blocks are good at which depths — and the user
					// observed the baselines beating evolved candidates exactly because of
					// this. Anchoring keeps the champion's DNA in every crossover child while
					// the second parent provides the variation.
					const size_t aIdx = 0;
					std::uniform_int_distribution<size_t> partnerDistribution(1, currentEliteCount - 1);
					const size_t bIdx = partnerDistribution(rng);
					childSchema = crossoverSchemaConfigs(
						archive[aIdx].candidate.config,
						archive[bIdx].candidate.config,
						rng,
						options.generation);
					wasCrossover = true;
				}
				else
				{
					const EvaluatedCandidate& parent = archive[eliteDistribution(rng)];
					childSchema = mutateSchemaConfig(parent.candidate.config, rng, options.generation, evolution, conditionDomain);
				}
				const std::string signature = schemaSignature(childSchema);
				if (!seenSignatures.insert(signature).second)
					continue;

				const std::string prefix = wasCrossover
					? std::string("xover_g") + std::to_string(generation) + "_i" + std::to_string(children.size())
					: std::string("evolved_g") + std::to_string(generation) + "_i" + std::to_string(children.size());
				children.push_back(materializeGeneratedSchema(childSchema, prefix, options.generation.outputDirectory));
				if (wasCrossover)
					++crossoverAccepted;
				else
					++mutationAccepted;
			}
			if (crossoverAccepted + mutationAccepted > 0)
				std::cout << "      generation " << generation
					<< " champion: " << archive[0].candidate.config.name
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
			std::cout << "  optimizer best aggregate: " << archive.front().candidate.config.name << " score " << archive.front().aggregateScore << '\n';

		// Phase B4 diversity audit. Reports distinct topology shapes attempted across the run
		// vs. the total candidate count. A low distinct-to-total ratio means the GA spent its
		// budget refining a few corridors instead of exploring the manifold — fix is then to
		// raise crossoverRate, enable NSGA-II ranking, or broaden the generator's bounds.
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

			// Front-0 diversity is the load-bearing number: if every Pareto-front entry has the
			// same topology, B4 (or richer generation in B2/B3) needs more pressure. Computed
			// only when NSGA-II ran, so the front ranks are populated.
			if (evolution.useNsga2Ranking)
			{
				std::set<std::string> frontTopologies;
				for (const EvaluatedCandidate& evaluation : archive)
				{
					if (evaluation.paretoFront != 0)
						continue;
					frontTopologies.insert(schemaTopologyKey(evaluation.candidate.config));
				}
				if (!frontTopologies.empty())
					std::cout << "    Pareto-front-0 distinct topologies: " << frontTopologies.size() << '\n';
			}
		}

		// Threshold refinement (Phase B1). Picks the top-K archive entries with conditional
		// levels and runs a continuous-parameter (1+lambda)-ES on the active threshold vector.
		// Each refiner evaluation uses the visit-proxy on the first dataset+workload (cheap and
		// deterministic), and then the resulting refined candidate is measured at full fidelity
		// so it lands in the records/CSV like any other candidate.
		if (evolution.refineThresholds && !archive.empty() && !datasets.empty() && !workloads.empty())
		{
			const size_t topK = std::max<size_t>(1, evolution.refineThresholdsTopK);
			const size_t available = std::min(topK, archive.size());
			std::cout << "  threshold refinement: top-" << available
				<< ", budget " << evolution.refineThresholdsEvaluations << " evals/candidate, sigma0 "
				<< evolution.refineThresholdsSigma0 << '\n';

			const Experiments::ConditionDomain domain = Experiments::estimateConditionDomain(datasets.front().dataset->cloud, options.autoConditions.proxyPointCap > 0 ? options.autoConditions.proxyPointCap : 262144);

			Experiments::ThresholdRefinementOptions refinerOptions;
			refinerOptions.enabled = true;
			refinerOptions.topK = available;
			refinerOptions.maxEvaluations = evolution.refineThresholdsEvaluations;
			refinerOptions.sigma0 = evolution.refineThresholdsSigma0;
			refinerOptions.seed = evolution.refineThresholdsSeed;
			refinerOptions.outputDirectory = options.generation.outputDirectory.empty()
				? std::string("results/refined_schemas")
				: options.generation.outputDirectory + "/refined";

			// Visit-proxy score function: builds the candidate against the first dataset's
			// PreparedWorkload and returns the resulting `record.score`. Reuses
			// `benchmarkSchemaCandidateCached` so cache hits between refinement iterations come
			// for free. We force `weights.useVisitProxy = true` (cheap, deterministic) regardless
			// of the user's measurement-level score weights, AND clamp the workload to a small
			// query count for refinement evaluations. The refiner does up to maxEvaluations
			// (default 60) full-fidelity builds per candidate; with the user's full workload
			// (often 64+ queries on a 100M-point cloud) each evaluation runs the full sweep and
			// total refinement time balloons. Refinement only needs to navigate the threshold
			// landscape, not produce a final number — 8 queries are enough to discriminate
			// neighboring threshold vectors, and the post-refinement re-measurement at full
			// fidelity gives the final reported score.
			Experiments::SchemaSearchOptions proxyOptions = options;
			proxyOptions.weights.useVisitProxy = true;
			if (proxyOptions.weights.visitProxyAlpha <= 0.0)
				proxyOptions.weights.visitProxyAlpha = 0.1;

			const DatasetContext& primary = datasets.front();
			const Experiments::WorkloadProfile fullWorkload = workloads.front();
			Experiments::WorkloadProfile refinerWorkload = fullWorkload;
			// 8 queries is the same query budget the auto-conditions proxy stage uses by default;
			// the threshold refiner's job is qualitatively identical (cheap deterministic ranking
			// of neighboring threshold vectors).
			constexpr size_t kRefinerProxyQueryCount = 8;
			if (refinerWorkload.numQueries > kRefinerProxyQueryCount)
				refinerWorkload.numQueries = kRefinerProxyQueryCount;
			const Experiments::WorkloadFeatures primaryWorkloadFeatures = Experiments::extractWorkloadFeatures(refinerWorkload, proxyOptions.weights);

			// Prepare a separate PreparedWorkload for the trimmed query count so the cache
			// fingerprint reflects the smaller workload and per-seed cache hits accumulate
			// across refinement iterations.
			const PreparedWorkload refinerPrepared = prepareWorkloadProfile(refinerWorkload, primary.dataset->cloud, useCudaEvaluator(proxyOptions));

			std::cout << "  refinement workload: " << refinerWorkload.numQueries
				<< " queries (vs " << fullWorkload.numQueries << " full)\n";

			Experiments::ThresholdScoreFn scoreFn = [&](const Experiments::SchemaCandidate& trial) {
				Experiments::SchemaSearchRecord trialRecord = benchmarkSchemaCandidateCached(
					*primary.dataset,
					primary.features,
					refinerWorkload,
					primaryWorkloadFeatures,
					refinerPrepared,
					trial,
					proxyOptions,
					nullptr);
				return trialRecord.score;
			};

			std::vector<Experiments::SchemaCandidate> refinedSurvivors;
			refinedSurvivors.reserve(available);
			for (size_t i = 0; i < available; ++i)
			{
				const EvaluatedCandidate& source = archive[i];
				Experiments::ThresholdRefinementResult refinement = Experiments::refineSchemaThresholds(
					source.candidate, domain, refinerOptions, scoreFn);

				if (refinement.dimensions == 0)
				{
					std::cout << "    [" << (i + 1) << "/" << available << "] "
						<< source.candidate.config.name << ": no active threshold dimensions, skipping\n";
					continue;
				}

				std::cout << "    [" << (i + 1) << "/" << available << "] "
					<< source.candidate.config.name
					<< ": dims=" << refinement.dimensions
					<< " evals=" << refinement.evaluationsUsed
					<< " score " << refinement.initialScore << " -> " << refinement.refinedScore
					<< (refinement.refinedScore < refinement.initialScore ? " (improved)" : " (no improvement)")
					<< '\n';

				if (refinement.refinedScore < refinement.initialScore)
					refinedSurvivors.push_back(std::move(refinement.refinedCandidate));

				++refinerOptions.seed; // decorrelate the ES across survivors so they don't all walk in lockstep
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

	// Phase C2: re-measure the top-K candidates per (dataset, workload) with multiple query
	// seeds, then store seed-averaged mean + 95% bootstrap CI on (avgLatency, p95Latency,
	// gpuBuild). The Pareto step (and the best-CSV view of latency) will prefer these means
	// over the single-seed point estimate.
	void runMultiSeedConfirmation(
		std::vector<Experiments::SchemaSearchRecord>& records,
		const std::vector<SearchDataset>& datasets,
		const std::vector<Experiments::WorkloadProfile>& workloads,
		const Experiments::SchemaSearchOptions& options)
	{
		if (options.confirmSeeds < 2 || records.empty())
			return;

		const size_t topK = std::max<size_t>(1, options.confirmTopK);
		const bool cudaEvaluator = useCudaEvaluator(options);

		std::cout << "  multi-seed confirmation: top-" << topK
			<< " per (dataset, workload), " << options.confirmSeeds << " seeds each\n";

		std::map<std::pair<std::string, std::string>, std::vector<size_t>> groups;
		for (size_t i = 0; i < records.size(); ++i)
			groups[{ records[i].datasetName, records[i].workloadName }].push_back(i);

		auto findDataset = [&](const std::string& name) -> const SearchDataset* {
			for (const SearchDataset& dataset : datasets)
			{
				if (dataset.name == name)
					return &dataset;
			}
			return nullptr;
		};
		auto findWorkload = [&](const std::string& name) -> const Experiments::WorkloadProfile* {
			for (const Experiments::WorkloadProfile& workload : workloads)
			{
				if (workload.name == name)
					return &workload;
			}
			return nullptr;
		};

		for (auto& [key, indices] : groups)
		{
			const SearchDataset* dataset = findDataset(key.first);
			const Experiments::WorkloadProfile* workload = findWorkload(key.second);
			if (!dataset || !workload)
				continue;

			// Pick top-K by score within this group. Lower score wins.
			std::sort(indices.begin(), indices.end(), [&records](size_t a, size_t b) {
				return records[a].score < records[b].score;
			});
			if (indices.size() > topK)
				indices.resize(topK);

			const Experiments::PointCloudFeatures features = Experiments::extractPointCloudFeatures(dataset->cloud);
			const Experiments::WorkloadFeatures workloadFeatures = Experiments::extractWorkloadFeatures(*workload, options.weights);

			for (const size_t recordIndex : indices)
			{
				Experiments::SchemaSearchRecord& target = records[recordIndex];

				Experiments::SchemaCandidate candidate;
				candidate.name = target.schemaName;
				candidate.path = target.schemaPath;
				candidate.isBaseline = target.isBaseline;
				try
				{
					candidate.config = Config::loadSchemaConfig(target.schemaPath);
				}
				catch (const std::exception& exception)
				{
					std::cerr << "    multi-seed confirmation skipped '" << target.schemaName
						<< "' (" << exception.what() << ")\n";
					continue;
				}

				std::vector<double> latencies;
				std::vector<double> p95s;
				std::vector<double> gpuBuilds;
				latencies.reserve(options.confirmSeeds);
				p95s.reserve(options.confirmSeeds);
				gpuBuilds.reserve(options.confirmSeeds);

				for (size_t s = 0; s < options.confirmSeeds; ++s)
				{
					Experiments::WorkloadProfile seededWorkload = *workload;
					// 7919 is a prime offset that decorrelates seeded sub-runs even when the
					// caller's base querySeed is also bumped between invocations.
					seededWorkload.querySeed = workload->querySeed + static_cast<uint32_t>(7919u * (s + 1));
					const PreparedWorkload preparedWorkload = prepareWorkloadProfile(
						seededWorkload, dataset->cloud, cudaEvaluator);

					Experiments::SchemaSearchRecord trialRecord = benchmarkSchemaCandidateCached(
						*dataset,
						features,
						seededWorkload,
						workloadFeatures,
						preparedWorkload,
						candidate,
						options,
						nullptr);

					latencies.push_back(trialRecord.queryMetrics.averageLatencyMs);
					p95s.push_back(trialRecord.queryMetrics.p95LatencyMs);
					gpuBuilds.push_back(trialRecord.gpuBuildMs);
				}

				const auto [latMean, latLo, latHi] = Experiments::bootstrapMeanCI(latencies);
				const auto [p95Mean, p95Lo, p95Hi] = Experiments::bootstrapMeanCI(p95s);
				const auto [bldMean, bldLo, bldHi] = Experiments::bootstrapMeanCI(gpuBuilds);

				target.confirmSeedsUsed = options.confirmSeeds;
				target.latencyMean = latMean;
				target.latencyCiLow = latLo;
				target.latencyCiHigh = latHi;
				target.p95LatencyMean = p95Mean;
				target.p95LatencyCiLow = p95Lo;
				target.p95LatencyCiHigh = p95Hi;
				target.gpuBuildMean = bldMean;
				target.gpuBuildCiLow = bldLo;
				target.gpuBuildCiHigh = bldHi;

				std::cout << "    confirm '" << target.schemaName << "' [" << key.first << "/" << key.second
					<< "]: latency " << latMean << " ms (95% CI " << latLo << "-" << latHi
					<< "), gpu build " << bldMean << " ms\n";
			}
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
	resolution.requestedCuda = evaluator == "cuda" || evaluator == "gpu";
	if (!resolution.requestedCuda)
	{
		resolution.evaluator = "cpu";
		return resolution;
	}

	if (cudaAvailable)
	{
		resolution.evaluator = "cuda";
		resolution.usingCuda = true;
		return resolution;
	}

	resolution.evaluator = "cpu";
	resolution.fellBackToCpu = true;
	resolution.warning = "CUDA evaluator requested but unavailable";
	if (!cudaError.empty())
		resolution.warning += ": " + cudaError;
	resolution.warning += "; falling back to CPU.";
	return resolution;
}

Experiments::ConditionDomain Experiments::estimateConditionDomain(const PointCloud& cloud, size_t maxSamplePoints)
{
	ConditionDomain domain;
	domain.samplePoints = std::min(cloud.size(), std::max<size_t>(1, maxSamplePoints));
	domain.estimatedFromCloud = !cloud.empty();
	if (cloud.empty())
	{
		domain.pointThresholds = fallbackPointThresholds();
		domain.heightRatioThresholds = fallbackHeightRatioThresholds();
		return domain;
	}

	struct SketchCell
	{
		size_t count = 0;
	};

	std::vector<size_t> counts;
	std::vector<double> densities;
	std::vector<double> heightRatios;
	std::vector<double> extentXValues;
	std::vector<double> extentYValues;
	std::vector<double> extentZValues;
	// Phase B3 anisotropy samples: `1 - shortExtent / longExtent` per sketch cell. Root counts too
	// so the quantile estimator has at least one sample on extremely uniform clouds.
	std::vector<double> anisotropyValues;
	auto pushAnisotropy = [&anisotropyValues](double x, double y, double z) {
		const double minE = std::min({ x, y, z });
		const double maxE = std::max({ x, y, z });
		if (maxE > 1.0e-9)
			addUniqueDouble(anisotropyValues, 1.0 - (minE / maxE));
	};

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
	pushAnisotropy(rootExtent.x, rootExtent.y, rootExtent.z);

	const size_t sampleCount = domain.samplePoints;
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
			++cells[index].count;
		}

		const glm::dvec3 cellExtent(
			static_cast<double>(rootExtent.x) / static_cast<double>(divisions),
			static_cast<double>(rootExtent.y) / static_cast<double>(divisions),
			static_cast<double>(rootExtent.z) / static_cast<double>(divisions));
		const double cellHorizontalExtent = std::max({ cellExtent.x, cellExtent.y, 1.0e-9 });
		const double cellHeightRatio = cellExtent.z / cellHorizontalExtent;
		const double cellVolume = cellExtent.x * cellExtent.y * cellExtent.z;
		// All cells at this division share the same aspect ratio, so one sample is sufficient.
		pushAnisotropy(cellExtent.x, cellExtent.y, cellExtent.z);
		const double sampleScale = sampleCount > 0 ? static_cast<double>(cloud.size()) / static_cast<double>(sampleCount) : 1.0;

		for (const SketchCell& cell : cells)
		{
			if (cell.count == 0)
				continue;

			const size_t estimatedCount = std::max<size_t>(1, static_cast<size_t>(std::round(static_cast<double>(cell.count) * sampleScale)));
			counts.push_back(estimatedCount);
			addUniqueDouble(densities, cellVolume > 1.0e-9 ? static_cast<double>(estimatedCount) / cellVolume : 0.0);
			addUniqueDouble(heightRatios, cellHeightRatio);
			addUniqueDouble(extentXValues, cellExtent.x);
			addUniqueDouble(extentYValues, cellExtent.y);
			addUniqueDouble(extentZValues, cellExtent.z);
			++domain.sketchNodes;
		}
	};

	addGridSketch(2);
	addGridSketch(4);
	++domain.sketchNodes;

	static const std::array<double, 5> quantiles = { 0.10, 0.25, 0.50, 0.75, 0.90 };
	for (const double quantile : quantiles)
	{
		addPointThresholdFamily(domain.pointThresholds, quantileValue(counts, quantile));
		addUniqueDouble(domain.densityThresholds, quantileValue(densities, quantile));
		addUniqueDouble(domain.heightRatioThresholds, quantileValue(heightRatios, quantile));
		addUniqueDouble(domain.extentXThresholds, quantileValue(extentXValues, quantile));
		addUniqueDouble(domain.extentYThresholds, quantileValue(extentYValues, quantile));
		addUniqueDouble(domain.extentZThresholds, quantileValue(extentZValues, quantile));
		if (!anisotropyValues.empty())
			addUniqueDouble(domain.anisotropyThresholds, std::clamp(quantileValue(anisotropyValues, quantile), 0.0, 1.0));
	}
	// Default fallback anisotropy gates when the sketch yielded too few samples — covers a
	// reasonable spread from "near-cubic" to "highly elongated" so the generator always has
	// something to sample from.
	if (domain.anisotropyThresholds.size() < 3)
	{
		for (const double fallback : { 0.1, 0.25, 0.4, 0.55, 0.7, 0.85 })
			addUniqueDouble(domain.anisotropyThresholds, fallback);
	}
	// Occupancy-entropy thresholds. Fixed spread over [0.1, 0.9] in the [0, 1] normalized space
	// covers the useful gating range — values near 0 fire on extremely clustered nodes, values
	// near 1 fire on near-uniform ones. No per-cloud anchoring needed since the per-node entropy
	// is itself normalized by ln(64) at evaluation time.
	for (const double fallback : { 0.15, 0.30, 0.45, 0.60, 0.75, 0.90 })
		addUniqueDouble(domain.occupancyEntropyThresholds, fallback);

	for (const size_t fallback : fallbackPointThresholds())
	{
		if (fallback <= cloud.size() * 2)
			addUniqueSize(domain.pointThresholds, fallback);
	}
	for (const double fallback : fallbackHeightRatioThresholds())
		addUniqueDouble(domain.heightRatioThresholds, fallback);
	if (rootDensity > 0.0)
	{
		for (const double multiplier : { 0.25, 0.50, 1.0, 2.0, 4.0 })
			addUniqueDouble(domain.densityThresholds, rootDensity * multiplier);
	}
	for (const double divisor : { 1.0, 2.0, 4.0, 8.0 })
	{
		addUniqueDouble(domain.extentXThresholds, static_cast<double>(rootExtent.x) / divisor);
		addUniqueDouble(domain.extentYThresholds, static_cast<double>(rootExtent.y) / divisor);
		addUniqueDouble(domain.extentZThresholds, static_cast<double>(rootExtent.z) / divisor);
	}

	sortUniqueValues(domain.pointThresholds);
	sortUniqueValues(domain.densityThresholds);
	sortUniqueValues(domain.heightRatioThresholds);
	sortUniqueValues(domain.extentXThresholds);
	sortUniqueValues(domain.extentYThresholds);
	sortUniqueValues(domain.extentZThresholds);
	sortUniqueValues(domain.anisotropyThresholds);
	sortUniqueValues(domain.occupancyEntropyThresholds);
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
	if (options.count == 0)
		return {};
	const size_t maxDepth = std::max<size_t>(1, options.maxDepth);
	const size_t maxBlocks = std::min(std::max<size_t>(1, options.maxBlocks), maxDepth);
	const size_t minBlocks = std::min(std::max<size_t>(1, options.minBlocks), maxBlocks);
	const size_t minLeaf = std::max<size_t>(1, std::min(options.minLeafCapacity, options.maxLeafCapacity));
	const size_t maxLeaf = std::max(minLeaf, options.maxLeafCapacity);
	const double conditionProbability = std::clamp(options.conditionalProbability, 0.0, 1.0);

	std::mt19937 rng(options.seed);
	std::bernoulli_distribution conditionDistribution(conditionProbability);
	std::unordered_set<std::string> seen;
	std::vector<SchemaCandidate> candidates;
	candidates.reserve(options.count);

	size_t attempts = 0;
	const size_t maxAttempts = std::max<size_t>(options.count * 50, 1024);
	while (candidates.size() < options.count && attempts++ < maxAttempts)
	{
		std::uniform_int_distribution<size_t> blockDistribution(minBlocks, maxBlocks);
		const size_t numBlocks = blockDistribution(rng);

		SchemaConfig schema;
		schema.buildPolicy.maxDepth = maxDepth;
		schema.buildPolicy.leafCapacity = randomPowerOfTwo(rng, minLeaf, maxLeaf);
		schema.buildPolicy.minPrimitivesToSplit = std::max<size_t>(2, schema.buildPolicy.leafCapacity / 4);
		schema.buildPolicy.collapseSingleChild = true;
		schema.buildPolicy.removeEmptyNodes = true;
		schema.buildPolicy.allowOverlapDuplication = false;

		size_t remainingDepth = maxDepth;
		std::optional<MultiDataStructure::DataStructureLevel> previousType;
		for (size_t block = 0; block < numBlocks; ++block)
		{
			const size_t remainingBlocks = numBlocks - block - 1;
			const size_t maxLevelForBlock = remainingDepth - remainingBlocks;
			std::uniform_int_distribution<size_t> levelDistribution(1, maxLevelForBlock);

			SchemaLevelConfig level;
			level.type = randomStructureType(rng, previousType);
			level.typeName = randomTypeNameForBase(level.type, rng);
			level.numLevels = levelDistribution(rng);
			level.leafCapacity = randomPowerOfTwo(rng, minLeaf, maxLeaf);
			level.minPrimitivesToSplit = std::max<size_t>(2, level.leafCapacity / 4);
			level.axisPolicy = sampleAxisPolicy(rng, level.type);
			if (options.conditionalLevels && block > 0 && conditionDistribution(rng))
				level.condition = randomLevelCondition(rng, level, minLeaf, maxLeaf, conditionDomain);

			schema.levels.push_back(level);
			previousType = level.type;
			remainingDepth -= level.numLevels;
		}

		if (schema.levels.empty())
			continue;

		schema.buildPolicy.maxDepth = std::min(maxDepth, schema.totalLevels());
		const std::string signature = schemaSignature(schema);
		if (!seen.insert(signature).second)
			continue;

		SchemaCandidate candidate = materializeGeneratedSchema(schema, "generated", options.outputDirectory);
		candidates.push_back(std::move(candidate));
	}

	if (candidates.size() < options.count)
		std::cerr << "Warning: generated " << candidates.size() << " unique schemas from requested " << options.count << '\n';

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
	profile.name = asString(root, "name", profile.name);
	profile.numQueries = asSize(root, "numQueries", profile.numQueries);
	profile.knnK = asSize(root, "knnK", profile.knnK);
	profile.querySeed = static_cast<uint32_t>(asSize(root, "querySeed", profile.querySeed));
	profile.rangeScaleMin = asDouble(root, "rangeScaleMin", profile.rangeScaleMin);
	profile.rangeScaleMax = asDouble(root, "rangeScaleMax", profile.rangeScaleMax);
	profile.radiusScaleMin = asDouble(root, "radiusScaleMin", profile.radiusScaleMin);
	profile.radiusScaleMax = asDouble(root, "radiusScaleMax", profile.radiusScaleMax);

	if (const boost::json::value* queries = root.if_contains("queries"))
	{
		if (!queries->is_object())
			throw std::runtime_error("Workload queries must be an object");

		const boost::json::object& queryWeights = queries->as_object();
		profile.rangeWeight = asDouble(queryWeights, "aabb_range", profile.rangeWeight);
		profile.radiusWeight = asDouble(queryWeights, "radius", profile.radiusWeight);
		profile.knnWeight = asDouble(queryWeights, "knn", profile.knnWeight);
	}

	if (const boost::json::value* queryScales = root.if_contains("queryScales"))
	{
		if (!queryScales->is_object())
			throw std::runtime_error("Workload queryScales must be an object");

		const boost::json::object& scales = queryScales->as_object();
		parseScaleRange(scales, "aabb_range", profile.rangeScaleMin, profile.rangeScaleMax);
		parseScaleRange(scales, "range", profile.rangeScaleMin, profile.rangeScaleMax);
		parseScaleRange(scales, "volume", profile.rangeScaleMin, profile.rangeScaleMax);
		parseScaleRange(scales, "radius", profile.radiusScaleMin, profile.radiusScaleMax);
	}

	normalizeScaleRange(profile.rangeScaleMin, profile.rangeScaleMax);
	normalizeScaleRange(profile.radiusScaleMin, profile.radiusScaleMax);

	if (profile.name.empty())
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

namespace
{
	boost::json::object serializeBuildMetricsForOutput(const Experiments::BuildMetrics& metrics)
	{
		boost::json::object out;
		out["buildTimeMs"] = metrics.buildTimeMs;
		out["numNodes"] = metrics.numNodes;
		out["numLeaves"] = metrics.numLeaves;
		out["indexedPoints"] = metrics.indexedPoints;
		out["maxDepth"] = metrics.maxDepth;
		out["averageLeafOccupancy"] = metrics.averageLeafOccupancy;
		out["maxLeafOccupancy"] = metrics.maxLeafOccupancy;
		out["memoryEstimateBytes"] = metrics.memoryEstimateBytes;
		return out;
	}

	boost::json::object serializeQueryMetricsForOutput(const Experiments::QueryMetrics& metrics)
	{
		boost::json::object out;
		out["totalQueries"] = metrics.totalQueries;
		out["totalLatencyMs"] = metrics.totalLatencyMs;
		out["averageLatencyMs"] = metrics.averageLatencyMs;
		out["medianLatencyMs"] = metrics.medianLatencyMs;
		out["p95LatencyMs"] = metrics.p95LatencyMs;
		out["throughputQueriesPerSecond"] = metrics.throughputQueriesPerSecond;
		out["averageVisitedNodes"] = metrics.averageVisitedNodes;
		out["averageTestedPoints"] = metrics.averageTestedPoints;
		out["averageReturnedPoints"] = metrics.averageReturnedPoints;
		out["totalVisitedNodes"] = metrics.totalVisitedNodes;
		out["totalTestedPoints"] = metrics.totalTestedPoints;
		out["totalReturnedPoints"] = metrics.totalReturnedPoints;
		return out;
	}

	boost::json::object serializeRecordForStdout(const Experiments::SchemaSearchRecord& record)
	{
		boost::json::object out;
		out["datasetName"] = record.datasetName;
		out["datasetSource"] = record.datasetSource;
		out["numPoints"] = record.numPoints;
		out["workloadName"] = record.workloadName;
		out["rangeWeight"] = record.rangeWeight;
		out["radiusWeight"] = record.radiusWeight;
		out["knnWeight"] = record.knnWeight;
		out["numQueries"] = record.numQueries;
		out["knnK"] = record.knnK;
		out["querySeed"] = record.querySeed;
		out["schemaName"] = record.schemaName;
		out["schemaPath"] = record.schemaPath;
		out["build"] = serializeBuildMetricsForOutput(record.buildMetrics);
		out["query"] = serializeQueryMetricsForOutput(record.queryMetrics);
		out["rangeQueries"] = record.rangeQueries;
		out["countRangeQueries"] = record.countRangeQueries;
		out["radiusQueries"] = record.radiusQueries;
		out["knnQueries"] = record.knnQueries;
		out["score"] = record.score;
		out["scoreMemoryMb"] = record.scoreMemoryMb;
		out["scoreImbalancePenalty"] = record.scoreImbalancePenalty;
		out["backend"] = record.backend;
		out["cudaDevice"] = record.cudaDevice;
		out["cudaBuilder"] = record.cudaBuilder;
		out["gpuUploadMs"] = record.gpuUploadMs;
		out["gpuBuildMs"] = record.gpuBuildMs;
		out["gpuQueryMs"] = record.gpuQueryMs;
		out["gpuMemoryBytes"] = record.gpuMemoryBytes;
		out["conditionalLevels"] = record.conditionalLevels;
		out["conditionFields"] = record.conditionFields;
		out["conditionSummary"] = record.conditionSummary;
		out["isBaseline"] = record.isBaseline;
		out["activeStructureTypes"] = record.activeStructureTypes;
		out["nestedActiveFraction"] = record.nestedActiveFraction;
		out["activeStructureSummary"] = record.activeStructureSummary;
		out["bestBaselineSchema"] = record.bestBaselineSchema;
		out["bestBaselineScore"] = record.bestBaselineScore;
		out["relativeSpeedupVsBaseline"] = record.relativeSpeedupVsBaseline;

		boost::json::object weights;
		weights["lambdaBuild"] = record.weights.lambdaBuild;
		weights["lambdaMemory"] = record.weights.lambdaMemory;
		weights["lambdaImbalance"] = record.weights.lambdaImbalance;
		weights["useVisitProxy"] = record.weights.useVisitProxy;
		weights["visitProxyAlpha"] = record.weights.visitProxyAlpha;
		out["weights"] = weights;
		return out;
	}
}

int Experiments::runEvaluateOne(const SchemaSearchOptions& options)
{
	SchemaSearchOptions oneOptions = options;
	oneOptions.csvPath.clear();
	oneOptions.bestCsvPath.clear();
	oneOptions.includeSyntheticDatasets = false;
	oneOptions.includeConfiguredSchemas = true;
	oneOptions.generation.count = 0;
	oneOptions.autoConditions.enabled = false;
	oneOptions.evolution.enabled = false;
	oneOptions.benchmarkTopK = 0;
	oneOptions.pauseAtEnd = false;
	oneOptions.rankModelPath.clear();

	if (oneOptions.inputPaths.empty())
		throw std::runtime_error("evaluate-one requires --input <cloud>");
	if (oneOptions.schemaPaths.empty())
		throw std::runtime_error("evaluate-one requires --schema <schema.json>");
	if (oneOptions.workloadPaths.empty())
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
	memoryMb = static_cast<double>(buildMetrics.memoryEstimateBytes) / (1024.0 * 1024.0);
	imbalancePenalty = buildMetrics.averageLeafOccupancy > 0.0
		? static_cast<double>(buildMetrics.maxLeafOccupancy) / buildMetrics.averageLeafOccupancy
		: 0.0;

	const double primary = weights.useVisitProxy
		? queryMetrics.averageVisitedNodes + weights.visitProxyAlpha * queryMetrics.averageTestedPoints
		: queryMetrics.averageLatencyMs;

	return primary +
		weights.lambdaBuild * buildMetrics.buildTimeMs +
		weights.lambdaMemory * memoryMb +
		weights.lambdaImbalance * imbalancePenalty;
}

std::vector<Experiments::SchemaSearchRecord> Experiments::selectBestRecords(const std::vector<SchemaSearchRecord>& records)
{
	std::vector<SchemaSearchRecord> best;
	for (const SchemaSearchRecord& record : records)
	{
		auto existing = std::find_if(best.begin(), best.end(), [&record](const SchemaSearchRecord& candidate) {
			return candidate.datasetName == record.datasetName && candidate.workloadName == record.workloadName;
		});

		if (existing == best.end())
		{
			best.push_back(record);
			continue;
		}

		if (record.score < existing->score)
			*existing = record;
	}

	return best;
}

namespace
{
	struct ParetoMetrics
	{
		double avgLatencyMs = 0.0;
		double buildTimeMs = 0.0;
		double memoryMb = 0.0;
		double imbalancePenalty = 0.0;
	};

	ParetoMetrics extractParetoMetrics(const Experiments::SchemaSearchRecord& record)
	{
		ParetoMetrics m;
		// When the record went through multi-seed confirmation, prefer the seed-averaged mean.
		// A single noisy seed that got lucky can't fake a Pareto win because its win has to hold
		// up against the average over confirmSeedsUsed independent draws.
		m.avgLatencyMs = record.confirmSeedsUsed > 0
			? record.latencyMean
			: record.queryMetrics.averageLatencyMs;
		m.buildTimeMs = record.buildMetrics.buildTimeMs;
		// memoryEstimateBytes is always populated; scoreMemoryMb may be 0 when the record is
		// loaded from the cache without recomputing computeSchemaSearchScore. Always derive
		// directly from buildMetrics for consistency.
		m.memoryMb = static_cast<double>(record.buildMetrics.memoryEstimateBytes) / (1024.0 * 1024.0);
		m.imbalancePenalty = record.buildMetrics.averageLeafOccupancy > 0.0
			? static_cast<double>(record.buildMetrics.maxLeafOccupancy) / record.buildMetrics.averageLeafOccupancy
			: 0.0;
		return m;
	}

	bool dominates(const ParetoMetrics& a, const ParetoMetrics& b)
	{
		const bool allLEq =
			a.avgLatencyMs <= b.avgLatencyMs &&
			a.buildTimeMs <= b.buildTimeMs &&
			a.memoryMb <= b.memoryMb &&
			a.imbalancePenalty <= b.imbalancePenalty;
		if (!allLEq)
			return false;
		const bool anyStrict =
			a.avgLatencyMs < b.avgLatencyMs ||
			a.buildTimeMs < b.buildTimeMs ||
			a.memoryMb < b.memoryMb ||
			a.imbalancePenalty < b.imbalancePenalty;
		return anyStrict;
	}
}

std::vector<Experiments::SchemaSearchRecord> Experiments::selectParetoRecords(const std::vector<SchemaSearchRecord>& records)
{
	// Group by (dataset, workload). For each group we run O(n^2) non-domination filtering;
	// schema-search batches stay well under a few thousand candidates so the quadratic cost is
	// far cheaper than any single measurement. If that ever stops being true, switch to a
	// Kung-style sweep — the API contract here doesn't change.
	std::map<std::pair<std::string, std::string>, std::vector<size_t>> groups;
	for (size_t i = 0; i < records.size(); ++i)
		groups[{ records[i].datasetName, records[i].workloadName }].push_back(i);

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

		// Rank non-dominated entries by avgLatencyMs (then buildTimeMs as a stable tie-break) so
		// the CSV reader can find the "fastest" front entry at rank 0 without re-sorting.
		std::sort(nonDominated.begin(), nonDominated.end(), [&records](size_t a, size_t b) {
			if (records[a].queryMetrics.averageLatencyMs != records[b].queryMetrics.averageLatencyMs)
				return records[a].queryMetrics.averageLatencyMs < records[b].queryMetrics.averageLatencyMs;
			return records[a].buildMetrics.buildTimeMs < records[b].buildMetrics.buildTimeMs;
		});

		for (size_t rank = 0; rank < nonDominated.size(); ++rank)
		{
			SchemaSearchRecord entry = records[nonDominated[rank]];
			entry.paretoRank = static_cast<int>(rank);
			front.push_back(std::move(entry));
		}
	}

	return front;
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

int Experiments::runSchemaSearch(const SchemaSearchOptions& options)
{
	SchemaSearchOptions resolvedOptions = options;
	if (resolvedOptions.deepNestedSearch)
	{
		resolvedOptions.autoConditions.enabled = true;
		resolvedOptions.includeConfiguredSchemas = false;
		resolvedOptions.includeBaselineSchemas = true;
		resolvedOptions.generation.minBlocks = std::max<size_t>(2, resolvedOptions.generation.minBlocks);
		resolvedOptions.generation.maxBlocks = std::max(resolvedOptions.generation.maxBlocks, resolvedOptions.generation.minBlocks);
		resolvedOptions.generation.conditionalLevels = true;
		resolvedOptions.generation.conditionalProbability = std::max(0.75, resolvedOptions.generation.conditionalProbability);
	}
	std::string cudaError;
	bool cudaAvailable = false;
	if (useCudaEvaluator(options))
		cudaAvailable = PointGpu::MixedTree::isAvailable(&cudaError);
	const EvaluatorResolution evaluatorResolution = resolveSchemaSearchEvaluator(options.evaluator, cudaAvailable, cudaError);
	resolvedOptions.evaluator = evaluatorResolution.evaluator;

	Experiments::EvaluationCache ownedScoreCache;
	if (!resolvedOptions.scoreCachePath.empty() && resolvedOptions.scoreCache == nullptr)
	{
		if (resolvedOptions.rebuildScoreCache)
		{
			std::error_code error;
			std::filesystem::remove(resolvedOptions.scoreCachePath, error);
		}
		const bool opened = ownedScoreCache.open(resolvedOptions.scoreCachePath, false);
		if (opened)
		{
			resolvedOptions.scoreCache = &ownedScoreCache;
			std::cout << "  score cache: " << resolvedOptions.scoreCachePath
				<< " (" << ownedScoreCache.entryCount() << " entries on load)\n";
		}
		else
		{
			std::cerr << "Warning: failed to open score cache at " << resolvedOptions.scoreCachePath << "; running uncached\n";
		}
	}

	const std::vector<SearchDataset> datasets = loadDatasets(resolvedOptions);

	// Per-cloud condition domain — estimated once from the first dataset so the initial
	// generator + the GA's mutation operator can sample conditional thresholds calibrated to
	// actual cloud statistics. The auto-conditions path computes its own domain internally
	// (and uses tighter sketch parameters), so this estimate is only consulted in the GA /
	// flat measurement paths. Skipped on synthetic-only / empty-dataset runs.
	std::optional<ConditionDomain> sharedConditionDomain;
	if (!resolvedOptions.autoConditions.enabled && !datasets.empty() && !datasets.front().cloud.empty())
	{
		sharedConditionDomain = estimateConditionDomain(datasets.front().cloud,
			resolvedOptions.autoConditions.proxyPointCap > 0
				? resolvedOptions.autoConditions.proxyPointCap
				: size_t(262144));
		std::cout << "  condition domain from '" << datasets.front().name
			<< "': " << sharedConditionDomain->pointThresholds.size() << " point, "
			<< sharedConditionDomain->densityThresholds.size() << " density, "
			<< sharedConditionDomain->heightRatioThresholds.size() << " height, "
			<< sharedConditionDomain->anisotropyThresholds.size() << " anisotropy thresholds\n";
	}

	std::vector<SchemaCandidate> schemas = loadSchemas(resolvedOptions.schemaPaths, resolvedOptions.includeConfiguredSchemas);
	if (resolvedOptions.includeBaselineSchemas)
		appendBaselineSchemas(schemas, useCudaEvaluator(resolvedOptions));
	if (!resolvedOptions.autoConditions.enabled)
		appendGeneratedSchemas(schemas, resolvedOptions.generation,
			sharedConditionDomain.has_value() ? &sharedConditionDomain.value() : nullptr);
	if (schemas.empty())
	{
		if (!resolvedOptions.autoConditions.enabled || resolvedOptions.autoConditions.proxyCandidateCount == 0)
			throw std::runtime_error("Schema search has no schemas. Provide --schemas, omit --generated-only, or use --generate-schemas.");
	}
	const std::vector<WorkloadProfile> workloads = loadWorkloads(resolvedOptions);
	const std::optional<SchemaSelectorModel> rankModel = (resolvedOptions.rankModelPath.empty() || resolvedOptions.evolution.enabled || resolvedOptions.autoConditions.enabled)
		? std::optional<SchemaSelectorModel>()
		: std::optional<SchemaSelectorModel>(loadSchemaSelectorModel(resolvedOptions.rankModelPath));

	std::vector<SchemaSearchRecord> records;
	records.reserve(datasets.size() * workloads.size() * schemas.size());

	std::cout << std::fixed << std::setprecision(3);
	std::cout << "Schema search\n";
	std::cout << "  evaluator: " << (useCudaEvaluator(resolvedOptions) ? "cuda" : "cpu") << '\n';
	if (resolvedOptions.deepNestedSearch)
		std::cout << "  deep nested search: enabled (CPU discovery, nested candidates only, CUDA confirmation report when available)\n";
	if (!evaluatorResolution.warning.empty())
		std::cout << "  warning: " << evaluatorResolution.warning << '\n';
	if (useCudaEvaluator(resolvedOptions))
	{
		const PointGpu::Options cudaOptions = cudaOptionsFrom(resolvedOptions);
		std::cout << "  cuda builder: " << cudaBuilderDisplayName(cudaOptions.builder) << '\n';
		std::cout << "  cuda device: " << cudaDeviceDescription(cudaOptions) << '\n';
		std::cout << "  cuda warmup: " << warmUpCudaDevice(cudaOptions) << " ms\n";
		if (cudaOptions.queryBatchSize > 0)
			std::cout << "  cuda query batch: " << cudaOptions.queryBatchSize << '\n';
		if (cudaOptions.memoryBudgetMb > 0)
		{
			std::cout << "  cuda memory budget: " << cudaOptions.memoryBudgetMb << " MB";
			if (resolvedOptions.cuda.memoryBudgetMb == 0)
				std::cout << " (auto, 75% of total VRAM)";
			std::cout << '\n';
		}
	}
	std::cout << "  datasets: " << datasets.size() << '\n';
	std::cout << "  schemas: " << schemas.size() << '\n';
	if (resolvedOptions.deepNestedSearch)
	{
		std::cout << "  deep nested candidate budget: " << resolvedOptions.autoConditions.proxyCandidateCount << " requested\n";
		std::cout << "  generated min blocks: " << resolvedOptions.generation.minBlocks << '\n';
		if (resolvedOptions.generation.conditionalLevels)
			std::cout << "  generated conditions: probability " << resolvedOptions.generation.conditionalProbability << '\n';
	}
	else if (resolvedOptions.generation.count > 0)
	{
		std::cout << "  generated schemas: " << resolvedOptions.generation.count << " requested\n";
		if (resolvedOptions.generation.conditionalLevels)
			std::cout << "  generated conditions: probability " << resolvedOptions.generation.conditionalProbability << '\n';
	}
	if (!resolvedOptions.evolution.enabled && rankModel.has_value())
	{
		std::cout << "  surrogate rank model: " << resolvedOptions.rankModelPath << '\n';
		if (resolvedOptions.benchmarkTopK > 0)
			std::cout << "  benchmark top-k: " << resolvedOptions.benchmarkTopK << '\n';
	}
	else if (resolvedOptions.evolution.enabled && !resolvedOptions.rankModelPath.empty())
	{
		std::cout << "  surrogate rank model: not used inside evolutionary loop; measured scores drive selection\n";
	}
	std::cout << "  workloads: " << workloads.size() << '\n';
	if (resolvedOptions.autoConditions.enabled)
	{
		std::cout << "  auto-conditions: enabled\n";
		std::cout << "    proxy candidates: " << resolvedOptions.autoConditions.proxyCandidateCount << '\n';
		std::cout << "    proxy point cap: " << resolvedOptions.autoConditions.proxyPointCap << '\n';
		std::cout << "    proxy queries: " << resolvedOptions.autoConditions.proxyQueryCount << '\n';
		std::cout << "    final top-k: " << resolvedOptions.autoConditions.finalTopK << '\n';
		std::cout << "    confirmation top-k: " << resolvedOptions.autoConditions.confirmationTopK << '\n';
	}

	if (resolvedOptions.autoConditions.enabled)
	{
		records = runAutoConditionSearch(resolvedOptions, datasets, workloads, schemas);
	}
	else
	{
		std::vector<DatasetContext> datasetContexts = makeDatasetContexts(datasets, workloads, useCudaEvaluator(resolvedOptions));
		if (resolvedOptions.evolution.enabled)
		{
			records = runEvolutionarySchemaSearch(resolvedOptions, datasetContexts, workloads, schemas,
				sharedConditionDomain.has_value() ? &sharedConditionDomain.value() : nullptr);
		}
		else
		{
			CudaIndexCache cudaCache;
			for (const DatasetContext& datasetContext : datasetContexts)
			{
				const SearchDataset& dataset = *datasetContext.dataset;
				std::cout << "  dataset: " << dataset.name << " (" << dataset.cloud.size() << " points)\n";

				for (size_t workloadIndex = 0; workloadIndex < workloads.size(); ++workloadIndex)
				{
					const WorkloadProfile& workload = workloads[workloadIndex];
					const WorkloadFeatures workloadFeatures = extractWorkloadFeatures(workload, resolvedOptions.weights);
					const std::vector<SchemaCandidate> benchmarkSchemas = selectBenchmarkSchemas(
						resolvedOptions,
						dataset,
						workload,
						schemas,
						rankModel);

					std::cout << "    workload: " << workload.name << " (" << workload.numQueries << " queries)\n";
					if (benchmarkSchemas.size() != schemas.size())
						std::cout << "      benchmarking " << benchmarkSchemas.size() << " / " << schemas.size() << " schemas after surrogate pruning\n";

					for (const SchemaCandidate& schema : benchmarkSchemas)
					{
						CudaIndexCacheEntry* cudaEntry = useCudaEvaluator(resolvedOptions)
							? &cudaCache[datasetContext.dataset]
							: nullptr;
						SchemaSearchRecord record = benchmarkSchemaCandidateCached(
							dataset,
							datasetContext.features,
							workload,
							workloadFeatures,
							datasetContext.preparedWorkloads[workloadIndex],
							schema,
							resolvedOptions,
							cudaEntry);
						records.push_back(record);
						emitProgress(resolvedOptions, record);

						std::cout << "      " << record.schemaName
							<< ": score " << record.score
							<< ", avg " << record.queryMetrics.averageLatencyMs
							<< " ms, build " << record.buildMetrics.buildTimeMs
							<< " ms";
						if (record.backend == "cuda")
							std::cout << ", gpu build " << record.gpuBuildMs
								<< " ms, upload " << record.gpuUploadMs
								<< " ms, gpu query " << record.gpuQueryMs << " ms";
						std::cout << '\n';
					}
				}
			}
		}
	}

	runMultiSeedConfirmation(records, datasets, workloads, resolvedOptions);
	annotateBaselineComparisons(records);
	if (resolvedOptions.deepNestedSearch)
		reportDeepNestedOutcome(records);
	writeSearchRows(resolvedOptions.csvPath, records);
	writeBestRows(resolvedOptions.bestCsvPath, records);
	writeParetoRows(resolvedOptions.paretoCsvPath, records);

	std::cout << "  wrote rows: " << records.size() << '\n';
	if (!resolvedOptions.csvPath.empty())
		std::cout << "  csv: " << resolvedOptions.csvPath << '\n';
	if (!resolvedOptions.bestCsvPath.empty())
		std::cout << "  best csv: " << resolvedOptions.bestCsvPath << '\n';
	if (!resolvedOptions.paretoCsvPath.empty())
	{
		const std::vector<SchemaSearchRecord> front = selectParetoRecords(records);
		std::cout << "  pareto csv: " << resolvedOptions.paretoCsvPath
			<< " (" << front.size() << " non-dominated rows)\n";
	}

	if (resolvedOptions.scoreCache != nullptr && resolvedOptions.scoreCache->enabled())
	{
		const size_t hits = resolvedOptions.scoreCache->hitCount();
		const size_t misses = resolvedOptions.scoreCache->missCount();
		std::cout << "  score cache: " << hits << " hits, " << misses << " misses (entries now " << resolvedOptions.scoreCache->entryCount() << ")\n";
	}

	if (resolvedOptions.pauseAtEnd)
		std::system("pause");

	return 0;
}

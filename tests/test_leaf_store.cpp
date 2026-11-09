#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/workloads/points/LeafStore.h"
#include "../MultiDataStructure/workloads/points/PointSpatialIndex.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		SchemaLevelConfig storeTestLevel(const std::string& typeName, size_t numLevels, size_t leafCapacity, const std::string& axisPolicy = {})
		{
			SchemaLevelConfig level;
			level._typeName = typeName;
			level._primitiveKind = Config::parseSchemaPrimitiveKind(typeName);
			level._cpuFallbackType = Config::cpuFallbackForPrimitiveKind(level._primitiveKind);
			level._type = level._cpuFallbackType;
			level._numLevels = numLevels;
			level._leafCapacity = leafCapacity;
			level._minPrimitivesToSplit = 2;
			level._axisPolicy = axisPolicy;
			return level;
		}

		SchemaConfig storeTestSchema(const std::string& name, std::vector<SchemaLevelConfig> levels, size_t leafCapacity)
		{
			SchemaConfig schema;
			schema._name = name;
			schema._levels = std::move(levels);
			schema._buildPolicy._maxDepth = 0;
			schema._buildPolicy._leafCapacity = leafCapacity;
			schema._buildPolicy._minPrimitivesToSplit = 2;
			schema._buildPolicy._collapseSingleChild = true;
			schema._buildPolicy._removeEmptyNodes = true;
			schema._buildPolicy._allowOverlapDuplication = false;
			return schema;
		}

		std::vector<size_t> sortedIndices(std::vector<size_t> values)
		{
			std::sort(values.begin(), values.end());
			return values;
		}

		struct StoreAnswers
		{
			std::vector<std::vector<size_t>>	_ranges;
			std::vector<size_t>					_counts;
			std::vector<std::vector<size_t>>	_radii;
			std::vector<std::vector<size_t>>	_knn;
		};

		StoreAnswers answer(const PointSpatialIndex& index, const std::vector<AABB>& boxes, const std::vector<glm::vec3>& centers, float radius)
		{
			StoreAnswers answers;
			for (const AABB& box : boxes)
			{
				answers._ranges.push_back(sortedIndices(index.rangeQuery(box)._pointIndices));
				answers._counts.push_back(index.countRange(box, PointSpatialIndex::QueryTelemetry::Off)._count);
			}
			for (const glm::vec3& center : centers)
			{
				answers._radii.push_back(sortedIndices(index.radiusQuery(center, radius, PointSpatialIndex::QueryTelemetry::Off)._pointIndices));
				// k = 9 takes the depth-first kNN, k = 100 the best-first one.
				answers._knn.push_back(index.knnQuery(center, 9)._pointIndices);
				answers._knn.push_back(index.knnQuery(center, 100, PointSpatialIndex::QueryTelemetry::Off)._pointIndices);
			}
			return answers;
		}
	}

	// Out of core: a stored index must answer exactly as the same index in memory, with or without
	// a cache and with the hierarchy in memory or paged, and must actually read from its file.
	void runLeafStoreTests()
	{
		const PointCloud cloud = SyntheticPointClouds::generateUrbanMixed(3000, 3000, 6, 7);
		const AABB bounds = cloud.bounds();
		const glm::vec3 extent = bounds.size();

		std::vector<AABB> boxes;
		std::vector<glm::vec3> centers;
		for (size_t i = 0; i < 8; ++i)
		{
			const glm::vec3 center = cloud.points()[(i * cloud.size()) / 8].position;
			centers.push_back(center);
			boxes.emplace_back(center - extent * 0.05f, center + extent * 0.05f);
		}
		// Large boxes take the fully contained branch, whole subtrees read leaf by leaf.
		boxes.emplace_back(bounds.min() - glm::vec3(0.01f), bounds.max() + glm::vec3(0.01f));
		boxes.emplace_back(bounds.min() + extent * 0.2f, bounds.max() - extent * 0.2f);
		const float radius = 0.05f * std::max({ extent.x, extent.y, extent.z });

		const std::vector<SchemaConfig> schemas = {
			storeTestSchema("store_kd", { storeTestLevel("KDTree", 24, 16, "sliding_midpoint") }, 16),
			storeTestSchema("store_qt_ot", { storeTestLevel("QuadTree", 3, 16), storeTestLevel("Octree", 8, 16) }, 16),
			storeTestSchema("store_grid_ot", { storeTestLevel("RegularGrid", 1, 16), storeTestLevel("Octree", 8, 16) }, 16),
		};

		const std::filesystem::path directory = std::filesystem::temp_directory_path() /
			("mds_leaf_store_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
		for (const SchemaConfig& schema : schemas)
		{
			PointSpatialIndex index;
			index.build(cloud, schema);
			const uint64_t digest = index.structureDigest();
			const PointSpatialIndex::PoolView pool = index.poolView();
			const std::string path = (directory / (schema._name + ".store")).string();
			LeafStore::write(path, pool, digest);
			expect(LeafStore::matches(path, pool, digest), schema._name + ": a written store matches its tree");
			expect(!LeafStore::matches(path, pool, digest ^ 1u), schema._name + ": a store does not match another digest");

			const StoreAnswers inMemory = answer(index, boxes, centers, radius);
			uint64_t memoryHierarchyReads = 0;
			for (const bool paged : { false, true })
			{
				for (const double cacheFraction : { 0.0, 0.5 })
				{
					LeafStore::Options options;
					options._pagedHierarchy = paged;
					options._cacheBytes = static_cast<size_t>(cacheFraction * static_cast<double>(std::filesystem::file_size(path)));
					options._requireDirect = false;
					LeafStore store(path, pool, digest, options);
					index.attachStore(&store);
					const std::string label = schema._name + (paged ? ", paged" : ", in memory") + (cacheFraction > 0.0 ? ", cached" : ", no cache");

					const StoreAnswers stored = answer(index, boxes, centers, radius);
					expect(stored._ranges == inMemory._ranges, label + ": stored range queries match in memory");
					expect(stored._counts == inMemory._counts, label + ": stored count queries match in memory");
					expect(stored._radii == inMemory._radii, label + ": stored radius queries match in memory");
					expect(stored._knn == inMemory._knn, label + ": stored kNN queries match in memory, in order");
					expect(store.counters()._reads > 0 && store.counters()._bytes >= store.counters()._reads * LeafStore::PageBytes,
						label + ": stored queries read whole pages from the file");
					if (cacheFraction > 0.0)
						expect(store.counters()._hits > 0, label + ": repeated queries hit the cache");
					else
						expect(store.counters()._hits == 0, label + ": no cache, no hits");

					if (!paged && cacheFraction == 0.0)
						memoryHierarchyReads = store.counters()._reads;
					if (paged && cacheFraction == 0.0)
						expect(store.counters()._reads > memoryHierarchyReads, label + ": a paged hierarchy reads its child groups too");

					store.clearCache();
					store.resetCounters();
					index.attachStore(nullptr);
				}
			}

			// Direct (unbuffered) reads answer like buffered ones. Required on Windows; on Linux a
			// temporary directory on tmpfs refuses O_DIRECT, so there it is tried, not required.
			{
				LeafStore::Options options;
				options._pagedHierarchy = true;
				options._requireDirect = true;
				std::unique_ptr<LeafStore> direct;
				try
				{
					direct = std::make_unique<LeafStore>(path, pool, digest, options);
				}
				catch (const std::exception&)
				{
#if defined(_WIN32)
					expect(false, schema._name + ": the store opens unbuffered on Windows");
#endif
				}
				if (direct)
				{
					expect(direct->direct(), schema._name + ": a store opened with direct reads required reads directly");
					index.attachStore(direct.get());
					const StoreAnswers stored = answer(index, boxes, centers, radius);
					expect(stored._ranges == inMemory._ranges && stored._counts == inMemory._counts && stored._radii == inMemory._radii &&
						   stored._knn == inMemory._knn, schema._name + ": direct reads answer as in memory");
					expect(direct->counters()._reads > 0, schema._name + ": direct reads reach the file");
					index.attachStore(nullptr);
				}
			}

			bool refused = false;
			try
			{
				LeafStore wrong(path, pool, digest ^ 1u, LeafStore::Options{ 0, false, false });
			}
			catch (const std::exception&)
			{
				refused = true;
			}
			expect(refused, schema._name + ": opening a store for another tree is refused");
		}
		std::error_code ignored;
		std::filesystem::remove_all(directory, ignored);
	}
}

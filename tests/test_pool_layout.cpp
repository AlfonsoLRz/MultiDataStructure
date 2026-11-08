#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/workloads/points/PointSpatialIndex.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		SchemaLevelConfig layoutTestLevel(const std::string& typeName, size_t numLevels, size_t leafCapacity, const std::string& axisPolicy = {})
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

		SchemaConfig layoutTestSchema(const std::string& name, std::vector<SchemaLevelConfig> levels, size_t leafCapacity)
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

		std::vector<size_t> sortedIds(std::vector<size_t> values)
		{
			std::sort(values.begin(), values.end());
			return values;
		}

		// Every query kind, instrumented, so the entered nodes are counted alongside the answers.
		struct LayoutAnswers
		{
			std::vector<std::vector<size_t>>	_ranges;
			std::vector<size_t>					_counts;
			std::vector<std::vector<size_t>>	_radii;
			std::vector<std::vector<size_t>>	_knn;
			size_t								_visited = 0;
		};

		LayoutAnswers layoutAnswer(const PointSpatialIndex& index, const std::vector<AABB>& boxes, const std::vector<glm::vec3>& centers, float radius)
		{
			LayoutAnswers answers;
			for (const AABB& box : boxes)
			{
				const PointSpatialIndex::QueryResult range = index.rangeQuery(box);
				answers._ranges.push_back(sortedIds(range._pointIndices));
				answers._visited += range._stats._visitedNodes;
				const PointSpatialIndex::CountResult count = index.countRange(box);
				answers._counts.push_back(count._count);
				answers._visited += count._stats._visitedNodes;
			}
			for (const glm::vec3& center : centers)
			{
				const PointSpatialIndex::QueryResult around = index.radiusQuery(center, radius);
				answers._radii.push_back(sortedIds(around._pointIndices));
				answers._visited += around._stats._visitedNodes;
				// k = 9 takes the depth-first kNN, k = 100 the best-first one.
				for (const size_t k : { size_t(9), size_t(100) })
				{
					const PointSpatialIndex::QueryResult nearest = index.knnQuery(center, k);
					answers._knn.push_back(nearest._pointIndices);
					answers._visited += nearest._stats._visitedNodes;
				}
			}
			return answers;
		}
	}

	// Memory layout dials (testing.md 2s): huge pages and child prefetch change where the bytes live
	// and when they are fetched, never the answer; depth timing prices every entered node.
	void runPoolLayoutTests()
	{
		using Depths = std::array<double, PointSpatialIndex::QueryStats::MaxBreakdownDepth>;
		using Visits = std::array<uint64_t, PointSpatialIndex::QueryStats::MaxBreakdownDepth>;
		const bool hugeBefore = PointSpatialIndex::hugePages();
		const bool prefetchBefore = PointSpatialIndex::prefetchChildren();
		const bool timingBefore = PointSpatialIndex::depthTiming();

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
		boxes.emplace_back(bounds.min() + extent * 0.2f, bounds.max() - extent * 0.2f);
		const float radius = 0.05f * std::max({ extent.x, extent.y, extent.z });

		// Binary, 8-way and 27-way child groups, so every prefetch width is exercised.
		const std::vector<SchemaConfig> schemas = {
			layoutTestSchema("layout_kd", { layoutTestLevel("KDTree", 24, 16, "sliding_midpoint") }, 16),
			layoutTestSchema("layout_qt_ot", { layoutTestLevel("QuadTree", 3, 16), layoutTestLevel("Octree", 8, 16) }, 16),
			layoutTestSchema("layout_grid_ot", { layoutTestLevel("RegularGrid", 1, 16), layoutTestLevel("Octree", 8, 16) }, 16),
		};
		for (const SchemaConfig& schema : schemas)
		{
			PointSpatialIndex::setHugePages(false);
			PointSpatialIndex::setPrefetchChildren(false);
			PointSpatialIndex::setDepthTiming(false);
			PointSpatialIndex plain;
			plain.build(cloud, schema);
			expect(reinterpret_cast<uintptr_t>(plain.poolView()._nodes) % 64 == 0, schema._name + ": the node pool is 64-byte aligned");
			const LayoutAnswers before = layoutAnswer(plain, boxes, centers, radius);

			PointSpatialIndex::setHugePages(true);
			PointSpatialIndex::setPrefetchChildren(true);
			PointSpatialIndex::setDepthTiming(true);
			Depths ns{};
			Visits visits{};
			PointSpatialIndex::takeDepthTiming(ns, visits);
			PointSpatialIndex tuned;
			tuned.build(cloud, schema);
			const LayoutAnswers after = layoutAnswer(tuned, boxes, centers, radius);
			expect(after._ranges == before._ranges, schema._name + ": range answers are unchanged by the layout dials");
			expect(after._counts == before._counts, schema._name + ": count answers are unchanged by the layout dials");
			expect(after._radii == before._radii, schema._name + ": radius answers are unchanged by the layout dials");
			expect(after._knn == before._knn, schema._name + ": kNN answers are unchanged by the layout dials, in order");
			expect(after._visited == before._visited, schema._name + ": prefetching enters the same nodes");

			PointSpatialIndex::takeDepthTiming(ns, visits);
			uint64_t timedVisits = 0;
			double timedNs = 0.0;
			size_t deepest = 0;
			for (size_t depth = 0; depth < visits.size(); ++depth)
			{
				timedVisits += visits[depth];
				timedNs += ns[depth];
				if (visits[depth] > 0)
					deepest = depth;
			}
			expect(timedVisits == after._visited, schema._name + ": depth timing saw every entered node");
			expect(timedNs > 0.0, schema._name + ": depth timing charged time to the entered nodes");
			expect(deepest <= tuned.stats()._maxDepth, schema._name + ": visits are charged to depths the tree has");
			expect(visits[0] > 0, schema._name + ": the root is entered");
		}

		// A pool above 4 MB asks for huge pages and comes back 2 MB aligned; small ones keep 64 bytes.
		PointSpatialIndex::setHugePages(true);
		void* big = PointSpatialIndex::allocatePool(size_t(8) << 20, 64);
		expect(reinterpret_cast<uintptr_t>(big) % (size_t(2) << 20) == 0, "a large pool is 2 MB aligned with huge pages on");
		PointSpatialIndex::freePool(big);
		PointSpatialIndex::setHugePages(false);
		void* small = PointSpatialIndex::allocatePool(1024, 64);
		expect(reinterpret_cast<uintptr_t>(small) % 64 == 0, "a small pool is 64-byte aligned");
		PointSpatialIndex::freePool(small);

		PointSpatialIndex::setHugePages(hugeBefore);
		PointSpatialIndex::setPrefetchChildren(prefetchBefore);
		PointSpatialIndex::setDepthTiming(timingBefore);
	}
}

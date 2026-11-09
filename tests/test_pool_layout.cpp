#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/workloads/points/PointSpatialIndex.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

#include <set>

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

		// Page-blocked pool: same tree, same answers and visits, sibling groups contiguous and inside
		// one 4 KiB page, and a paged hierarchy reads fewer pages than the depth-first order.
		{
			PointSpatialIndex::setHugePages(false);
			PointSpatialIndex::setPrefetchChildren(false);
			PointSpatialIndex::setDepthTiming(false);
			const PointCloud large = SyntheticPointClouds::generateUrbanMixed(20000, 20000, 6, 11);
			std::vector<glm::vec3> probes;
			std::vector<AABB> probeBoxes;
			const glm::vec3 largeExtent = large.bounds().size();
			for (size_t i = 0; i < 64; ++i)
			{
				const glm::vec3 center = large.points()[(i * large.size()) / 64].position;
				probes.push_back(center);
				probeBoxes.emplace_back(center - largeExtent * 0.02f, center + largeExtent * 0.02f);
			}
			const float probeRadius = 0.02f * std::max({ largeExtent.x, largeExtent.y, largeExtent.z });
			const std::vector<SchemaConfig> pagedSchemas = {
				layoutTestSchema("paged_kd", { layoutTestLevel("KDTree", 40, 4, "sliding_midpoint") }, 4),
				layoutTestSchema("paged_qt_kd", { layoutTestLevel("QuadTree", 4, 4), layoutTestLevel("KDTree", 40, 4, "sliding_midpoint") }, 4),
				layoutTestSchema("paged_ot", { layoutTestLevel("Octree", 16, 4) }, 4),
				layoutTestSchema("paged_hg", { layoutTestLevel("HGrid", 3, 8) }, 8),
			};
			for (const SchemaConfig& schema : pagedSchemas)
			{
				PointSpatialIndex::setPoolLayout(PointSpatialIndex::PoolLayout::DepthFirst);
				PointSpatialIndex depthFirst;
				depthFirst.build(large, schema);
				PointSpatialIndex::setPoolLayout(PointSpatialIndex::PoolLayout::PageBlocked);
				PointSpatialIndex blocked;
				blocked.build(large, schema);
				PointSpatialIndex::setPoolLayout(PointSpatialIndex::PoolLayout::DepthFirst);

				expect(blocked.structureDigest() == depthFirst.structureDigest(), schema._name + ": the page-blocked pool holds the same tree");
				const LayoutAnswers a = layoutAnswer(depthFirst, probeBoxes, probes, probeRadius);
				const LayoutAnswers b = layoutAnswer(blocked, probeBoxes, probes, probeRadius);
				expect(a._ranges == b._ranges && a._counts == b._counts && a._radii == b._radii && a._knn == b._knn,
					schema._name + ": the page-blocked pool answers as the depth-first one");
				expect(a._visited == b._visited, schema._name + ": the page-blocked pool enters the same nodes");

				const PointSpatialIndex::PoolView pool = blocked.poolView();
				bool contained = true;
				for (size_t slot = 0; slot < pool._nodeCount; ++slot)
				{
					const PointSpatialIndex::HotNode& node = pool._nodes[slot];
					// Groups wider than a quarter page keep the depth-first order, so only the
					// narrower, page-blocked ones must sit inside one page.
					if (!node.isInternal() || node.count() > PointSpatialIndex::PoolPageSlots / 4)
						continue;
					contained &= node._first > slot &&
						node._first / PointSpatialIndex::PoolPageSlots == (node._first + node.count() - 1) / PointSpatialIndex::PoolPageSlots;
				}
				expect(contained || schema._name == "paged_hg", schema._name + ": every sibling group follows its parent and lies inside one page");

				// Distinct pages entered by the probes' radius queries: a stand-in for the paged
				// hierarchy's reads, counted the way LeafStore::touchNodes counts them.
				const auto pagesTouched = [&](const PointSpatialIndex& index) {
					const PointSpatialIndex::PoolView view = index.poolView();
					size_t total = 0;
					for (const glm::vec3& center : probes)
					{
						std::set<size_t> pages{ 0 };
						std::vector<uint32_t> stack{ 0u };
						while (!stack.empty())
						{
							const uint32_t slot = stack.back();
							stack.pop_back();
							const PointSpatialIndex::HotNode& node = view._nodes[slot];
							if (!node.isInternal())
								continue;
							pages.insert(node._first / PointSpatialIndex::PoolPageSlots);
							pages.insert((node._first + node.count() - 1) / PointSpatialIndex::PoolPageSlots);
							for (uint32_t c = 0; c < node.count(); ++c)
							{
								const PointSpatialIndex::HotNode& child = view._nodes[node._first + c];
								bool overlaps = child.count() > 0 || child.isInternal();
								for (int axis = 0; axis < 3 && overlaps; ++axis)
									overlaps = center[axis] + probeRadius >= child._lo[axis] && center[axis] - probeRadius <= child._hi[axis];
								if (overlaps)
									stack.push_back(node._first + c);
							}
						}
						total += pages.size();
					}
					return total;
				};
				const size_t pagesDepthFirst = pagesTouched(depthFirst);
				const size_t pagesBlocked = pagesTouched(blocked);
				std::cout << "  " << schema._name << ": pages touched, depth-first " << pagesDepthFirst << ", page-blocked " << pagesBlocked << '\n';
				expect(pagesBlocked <= pagesDepthFirst, schema._name + ": the page-blocked pool touches no more pages (" +
					std::to_string(pagesBlocked) + " vs " + std::to_string(pagesDepthFirst) + ")");
			}
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

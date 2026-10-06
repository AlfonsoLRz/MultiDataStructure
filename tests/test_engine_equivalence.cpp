#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/workloads/points/PointGpuIndex.h"
#include "../MultiDataStructure/workloads/points/PointSpatialIndex.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

// Brute-force equivalence for the CPU query engine.
//
// Every query kind, on randomly generated nested schemas, over clouds chosen to break things
// (planar, collinear, duplicate-heavy, tiny), is checked against a linear scan. Range and radius
// must return exactly the brute-force set, kNN exactly the brute-force order, and each result's
// checksum must equal the checksum of the brute-force set. This is the gate the engine rewrite
// has to pass at every step: layout, traversal order and counters may change, results may not.
namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		struct Rng
		{
			uint64_t _state;

			uint64_t next()
			{
				uint64_t z = (_state += 0x9E3779B97F4A7C15ull);
				z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
				z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
				return z ^ (z >> 31);
			}

			size_t below(size_t bound) { return bound ? static_cast<size_t>(next() % bound) : 0; }
			double unit() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
			bool chance(double p) { return unit() < p; }

			template <typename T, size_t N>
			const T& pick(const T (&values)[N]) { return values[below(N)]; }
		};

		SchemaLevelConfig randomLevel(Rng& rng, bool allowCondition)
		{
			static const char* const Types[] = {
				"Octree", "QuadTree", "KDTree", "BVH", "BIH", "RegularGrid", "HGrid", "LBVH", "KarrasOctree", "STR" };
			static const char* const KdPolicies[] = { "", "median_longest_axis", "round_robin", "center_longest_axis", "sliding_midpoint" };
			static const char* const QuadPolicies[] = { "xy", "xz", "yz", "ignore_shortest" };
			static const size_t Capacities[] = { 1, 2, 3, 4, 8, 16, 64 };

			SchemaLevelConfig level;
			level._typeName = rng.pick(Types);
			level._primitiveKind = Config::parseSchemaPrimitiveKind(level._typeName);
			level._cpuFallbackType = Config::cpuFallbackForPrimitiveKind(level._primitiveKind);
			level._type = level._cpuFallbackType;
			level._numLevels = 1 + rng.below(5);
			level._leafCapacity = rng.pick(Capacities);
			level._minPrimitivesToSplit = 1 + rng.below(4);
			if (level._type == MultiDataStructure::DataStructureLevel::KDTreeNode)
				level._axisPolicy = rng.pick(KdPolicies);
			else if (level._type == MultiDataStructure::DataStructureLevel::QuadTreeNode)
				level._axisPolicy = rng.pick(QuadPolicies);
			static const size_t Fanouts[] = { 8, 27, 64 };
			if (level._primitiveKind == SchemaPrimitiveKind::STR)
				level._fanout = rng.pick(Fanouts);
			else if ((level._type == MultiDataStructure::DataStructureLevel::OctreeNode ||
				 level._type == MultiDataStructure::DataStructureLevel::QuadTreeNode) && rng.chance(0.4))
				level._tightCenter = true;

			if (allowCondition && rng.chance(0.35))
			{
				SchemaLevelCondition& condition = level._condition;
				switch (rng.below(6))
				{
				case 0: condition._minPoints = 4 + rng.below(64); break;
				case 1: condition._maxPoints = 16 + rng.below(256); break;
				case 2: condition._maxHeightRatio = 0.05 + rng.unit(); break;
				case 3: condition._minAnisotropy = rng.unit() * 0.8; break;
				case 4: condition._maxOccupancyEntropy = 0.3 + 0.6 * rng.unit(); break;
				default: condition._minDensity = rng.unit() * 10.0; break;
				}
			}

			if (rng.chance(0.15))
			{
				level._adaptiveLeafCapacity._enabled = true;
				level._adaptiveLeafCapacity._densityWeight = rng.unit();
				level._adaptiveLeafCapacity._anisotropyWeight = rng.unit();
				level._adaptiveLeafCapacity._heightRatioWeight = rng.unit() * 0.5;
			}
			return level;
		}

		SchemaConfig randomSchema(Rng& rng, size_t index)
		{
			SchemaConfig schema;
			schema._name = "fuzz_" + std::to_string(index);
			const size_t blocks = 1 + rng.below(3);
			for (size_t b = 0; b < blocks; ++b)
				schema._levels.push_back(randomLevel(rng, b > 0));

			schema._buildPolicy._maxDepth = rng.chance(0.3) ? 1 + rng.below(schema.totalLevels() + 2) : 0;
			schema._buildPolicy._leafCapacity = 1 + rng.below(16);
			schema._buildPolicy._minPrimitivesToSplit = 1 + rng.below(4);
			schema._buildPolicy._collapseSingleChild = rng.chance(0.6);
			schema._buildPolicy._removeEmptyNodes = rng.chance(0.75);
			schema._buildPolicy._allowOverlapDuplication = false;
			if (rng.chance(0.2))
			{
				schema._buildPolicy._enableLeafMicroIndexes = true;
				schema._buildPolicy._leafMicroIndexThreshold = 4 + rng.below(32);
			}
			return schema;
		}

		PointCloud cloudFrom(const std::vector<glm::vec3>& positions)
		{
			PointCloud cloud;
			for (const glm::vec3& position : positions)
			{
				PointPrimitive point;
				point.position = position;
				cloud.addPoint(point);
			}
			return cloud;
		}

		std::vector<std::pair<std::string, PointCloud>> edgeClouds()
		{
			std::vector<std::pair<std::string, PointCloud>> clouds;
			clouds.emplace_back("sparse_dense", SyntheticPointClouds::generateSparseDenseMixture(150, 450, 7));
			clouds.emplace_back("urban", SyntheticPointClouds::generateUrbanMixed(300, 300, 3, 11));

			Rng rng{ 99 };
			std::vector<glm::vec3> planar, line, duplicates;
			for (int i = 0; i < 400; ++i)
			{
				planar.emplace_back(static_cast<float>(rng.unit() * 50.0), static_cast<float>(rng.unit() * 20.0), 3.0f);
				line.emplace_back(static_cast<float>(i) * 0.25f, 1.0f, 2.0f);
			}
			for (int i = 0; i < 300; ++i)
				duplicates.emplace_back(1.0f, 2.0f, 3.0f);
			for (int i = 0; i < 40; ++i)
				duplicates.emplace_back(static_cast<float>(rng.unit() * 4.0), static_cast<float>(rng.unit()), 3.0f);
			clouds.emplace_back("planar", cloudFrom(planar));
			clouds.emplace_back("collinear", cloudFrom(line));
			clouds.emplace_back("duplicates", cloudFrom(duplicates));
			clouds.emplace_back("ties", cloudFrom({ { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }));
			clouds.emplace_back("single", cloudFrom({ { 5, 5, 5 } }));
			clouds.emplace_back("pair", cloudFrom({ { 0, 0, 0 }, { 1, 1, 1 } }));
			clouds.emplace_back("empty", PointCloud());
			return clouds;
		}

		uint64_t checksumOf(const std::vector<size_t>& indices)
		{
			uint64_t checksum = 0;
			for (const size_t index : indices)
				PointSpatialIndex::accumulateResultChecksum(checksum, index);
			return checksum;
		}

		bool inBox(const AABB& box, const glm::vec3& p)
		{
			const glm::vec3 min = box.min();
			const glm::vec3 max = box.max();
			return p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y && p.z >= min.z && p.z <= max.z;
		}

		// The engine's exact float expression, so ties and boundary points resolve identically.
		float distanceSquared(const glm::vec3& p, const glm::vec3& center)
		{
			const float dx = p.x - center.x;
			const float dy = p.y - center.y;
			const float dz = p.z - center.z;
			return dx * dx + dy * dy + dz * dz;
		}

		std::vector<size_t> sortedCopy(std::vector<size_t> values)
		{
			std::sort(values.begin(), values.end());
			return values;
		}

		void checkCloudSchema(const std::string& label, const PointCloud& cloud, const SchemaConfig& schema, Rng& rng)
		{
			PointSpatialIndex index;
			index.build(cloud, schema);
			const std::vector<PointPrimitive>& points = cloud.points();
			const AABB bounds = cloud.empty() ? AABB(glm::vec3(0.0f), glm::vec3(1.0f)) : cloud.bounds();
			const glm::vec3 size = glm::max(bounds.size(), glm::vec3(1.0f));

			for (int q = 0; q < 24; ++q)
			{
				glm::vec3 center;
				if (!points.empty() && rng.chance(0.7))
					center = points[rng.below(points.size())].position;
				else
					for (int axis = 0; axis < 3; ++axis)
						center[axis] = bounds.min()[axis] - 0.2f * size[axis] + static_cast<float>(rng.unit()) * 1.4f * size[axis];

				const float scale = static_cast<float>(glm::length(size) * std::pow(10.0, -2.5 + 2.5 * rng.unit()));
				const std::string where = label + " / " + schema._name + " / q" + std::to_string(q);

				// Range and count over an anisotropic box.
				glm::vec3 half;
				for (int axis = 0; axis < 3; ++axis)
					half[axis] = scale * static_cast<float>(0.1 + rng.unit());
				const AABB box(center - half, center + half);
				std::vector<size_t> expectedRange;
				for (size_t i = 0; i < points.size(); ++i)
					if (inBox(box, points[i].position))
						expectedRange.push_back(i);
				const PointSpatialIndex::QueryResult range = index.rangeQuery(box);
				expect(sortedCopy(range._pointIndices) == expectedRange, "engine range == brute force: " + where);
				expect(range._stats._resultChecksum == checksumOf(expectedRange), "engine range checksum: " + where);
				expect(index.countRange(box)._count == expectedRange.size(), "engine count == brute force: " + where);

				// Radius, including radius 0.
				const float radius = rng.chance(0.1) ? 0.0f : scale;
				const float radiusSquared = radius * radius;
				std::vector<size_t> expectedRadius;
				for (size_t i = 0; i < points.size(); ++i)
					if (distanceSquared(points[i].position, center) <= radiusSquared)
						expectedRadius.push_back(i);
				const PointSpatialIndex::QueryResult sphere = index.radiusQuery(center, radius);
				expect(sortedCopy(sphere._pointIndices) == expectedRadius, "engine radius == brute force: " + where);
				expect(sphere._stats._resultChecksum == checksumOf(expectedRadius), "engine radius checksum: " + where);

				// kNN in exact (distance, index) order, including k larger than the cloud.
				static const size_t Ks[] = { 1, 3, 16 };
				const size_t k = rng.chance(0.1) ? points.size() + 3 : rng.pick(Ks);
				std::vector<std::pair<float, size_t>> ranked;
				for (size_t i = 0; i < points.size(); ++i)
					ranked.emplace_back(distanceSquared(points[i].position, center), i);
				std::sort(ranked.begin(), ranked.end());
				std::vector<size_t> expectedKnn;
				for (size_t i = 0; i < std::min(k, ranked.size()); ++i)
					expectedKnn.push_back(ranked[i].second);
				const PointSpatialIndex::QueryResult knn = index.knnQuery(center, k);
				expect(knn._pointIndices == expectedKnn, "engine knn == brute force order: " + where);
				expect(knn._stats._resultChecksum == checksumOf(expectedKnn), "engine knn checksum: " + where);

				// The uninstrumented timed path must answer identically and record no counters.
				constexpr PointSpatialIndex::QueryTelemetry off = PointSpatialIndex::QueryTelemetry::Off;
				const PointSpatialIndex::QueryResult rangeOff = index.rangeQuery(box, off);
				expect(rangeOff._stats._resultChecksum == range._stats._resultChecksum &&
					   rangeOff._stats._visitedNodes == 0 && rangeOff._stats._testedPoints == 0,
					   "telemetry-off range matches and records nothing: " + where);
				expect(index.countRange(box, off)._count == expectedRange.size(), "telemetry-off count matches: " + where);
				expect(index.radiusQuery(center, radius, off)._stats._resultChecksum == sphere._stats._resultChecksum,
					   "telemetry-off radius matches: " + where);
				expect(index.knnQuery(center, k, off)._pointIndices == expectedKnn, "telemetry-off knn matches: " + where);

				// Entered nodes are a subset of bounds-tested ones.
				expect(range._stats._visitedNodes <= range._stats._testedBoxes &&
					   sphere._stats._visitedNodes <= sphere._stats._testedBoxes &&
					   knn._stats._visitedNodes <= knn._stats._testedBoxes,
					   "visited nodes never exceed tested boxes: " + where);
			}
		}
	}

	void runEngineEquivalenceTests()
	{
		// The checksum discriminates swaps that preserve the index sum, and ignores order.
		expect(checksumOf({ 3, 8 }) != checksumOf({ 4, 7 }), "result checksum separates {3,8} from {4,7}");
		expect(checksumOf({ 1, 2, 3 }) == checksumOf({ 3, 1, 2 }), "result checksum is order-independent");
		expect(checksumOf({}) == 0, "empty result checksum is 0");

		const std::vector<std::pair<std::string, PointCloud>> clouds = edgeClouds();
		Rng rng{ 20261003 };
		for (size_t s = 0; s < 48; ++s)
		{
			const SchemaConfig schema = randomSchema(rng, s);
			for (const auto& [label, cloud] : clouds)
				checkCloudSchema(label, cloud, schema, rng);
		}

		// The structure digest is a pure function of (cloud, schema): rebuilding reproduces it,
		// and a different schema on the same cloud does not.
		const PointCloud& cloud = clouds.front().second;
		Rng schemaRng{ 5 };
		SchemaConfig first = randomSchema(schemaRng, 0);
		first._levels.resize(1);
		first._levels.front()._condition = SchemaLevelCondition();
		first._levels.front()._adaptiveLeafCapacity = AdaptiveLeafCapacityConfig();
		// An explicit type: the fuzz type list grows, and a drawn type need not split this cloud
		// differently at these two capacities.
		first._levels.front()._typeName = "Octree";
		first._levels.front()._primitiveKind = SchemaPrimitiveKind::Octree;
		first._levels.front()._type = MultiDataStructure::DataStructureLevel::OctreeNode;
		first._levels.front()._tightCenter = false;
		first._levels.front()._numLevels = 4;
		first._levels.front()._leafCapacity = 64;
		first._levels.front()._minPrimitivesToSplit = 1;
		first._buildPolicy._maxDepth = 0;
		SchemaConfig second = first;
		second._levels.front()._leafCapacity = 4;
		PointSpatialIndex a, b, c;
		a.build(cloud, first);
		b.build(cloud, first);
		c.build(cloud, second);
		expect(a.structureDigest() == b.structureDigest(), "structure digest is reproducible");
		expect(a.structureDigest() != c.structureDigest(), "structure digest separates different trees");

		// Geometry-only digest: the same tree under different labels matches. An Octree level
		// and a KarrasOctree level split identically on the CPU.
		const auto level = [](const char* type, size_t depth, bool tight = false, const char* policy = "") {
			SchemaLevelConfig config;
			config._typeName = type;
			config._primitiveKind = Config::parseSchemaPrimitiveKind(config._typeName);
			config._cpuFallbackType = Config::cpuFallbackForPrimitiveKind(config._primitiveKind);
			config._type = config._cpuFallbackType;
			config._numLevels = depth;
			config._leafCapacity = 4;
			config._minPrimitivesToSplit = 1;
			config._axisPolicy = policy;
			config._tightCenter = tight;
			return config;
		};
		SchemaConfig plain, relabelled;
		plain._levels = { level("Octree", 6) };
		relabelled._levels = { level("Octree", 2), level("KarrasOctree", 4) };
		PointSpatialIndex plainIndex, relabelledIndex;
		plainIndex.build(cloud, plain);
		relabelledIndex.build(cloud, relabelled);
		expect(plainIndex.structureDigest(true) == relabelledIndex.structureDigest(true), "geometry digest ignores structure labels");
		expect(plainIndex.structureDigest() != relabelledIndex.structureDigest(), "full digest keeps structure labels");

		// The data-adaptive splits: every point lies in its leaf's cell (the tight-centre child
		// cells match the routing), a sliding-midpoint node never leaves a side empty, and a node
		// whose points all coincide is a leaf (with depth caps of 48 and 24, splitting it would
		// chain empty nodes down to the cap).
		for (const auto& [label, edgeCloud] : clouds)
		{
			for (int variant = 0; variant < 4; ++variant)
			{
				SchemaConfig schema;
				schema._levels = { variant == 0 ? level("Octree", 24, true)
					: variant == 1 ? level("QuadTree", 24, true, "xy")
					: variant == 2 ? level("KDTree", 40, false, "sliding_midpoint")
					: level("STR", 12) };
				schema._levels.front()._fanout = 64;
				schema._buildPolicy._removeEmptyNodes = false;
				schema._buildPolicy._collapseSingleChild = false;
				PointSpatialIndex index;
				index.build(edgeCloud, schema);
				bool contained = true, bothSides = true, coincidentLeaf = true;
				std::vector<PointSpatialIndex::NodeView> stack;
				if (const PointSpatialIndex::NodeView root = index.root())
					stack.push_back(root);
				while (!stack.empty())
				{
					const PointSpatialIndex::NodeView node = stack.back();
					stack.pop_back();
					if (node.isLeaf())
					{
						const AABB cell = node.bounds();
						for (size_t i = 0; i < node.pointCount(); ++i)
						{
							const glm::vec3 p = edgeCloud.points()[index.pointOrder()[node.pointOffset() + i]].position;
							contained = contained && p.x >= cell.min().x && p.y >= cell.min().y && p.z >= cell.min().z &&
								p.x <= cell.max().x && p.y <= cell.max().y && p.z <= cell.max().z;
						}
						continue;
					}
					expect(node.childCount() <= 64, "no node has more than 64 children: " + label);
					const glm::vec3 spread = node.tightBounds().size();
					coincidentLeaf = coincidentLeaf && (spread.x > 0.0f || spread.y > 0.0f || spread.z > 0.0f);
					if (variant == 2 && node.childCount() == 2)
						bothSides = bothSides && node.child(0).subtreePointCount() > 0 && node.child(1).subtreePointCount() > 0;
					for (size_t c = 0; c < node.childCount(); ++c)
						stack.push_back(node.child(c));
				}
				const std::string where = label + (variant == 0 ? " tight octree" : variant == 1 ? " tight quadtree"
					: variant == 2 ? " sliding midpoint" : " STR 64");
				expect(contained, "every point lies in its leaf's cell: " + where);
				expect(bothSides, "sliding midpoint never leaves a side empty: " + where);
				expect(coincidentLeaf, "a node whose points all coincide is a leaf: " + where);
			}
		}

		// The GPU batch engine queries the CPU's own tree, so every answer of a batch must be the
		// CPU's: counts and checksums, range and radius as sets, kNN in order, Morton-sorted or not,
		// with and without indices. Skipped without a CUDA device.
		if (!PointGpuIndex::deviceAvailable())
			return;
		Rng gpuRng{ 20261006 };
		for (size_t s = 0; s < 24; ++s)
		{
			const SchemaConfig schema = randomSchema(gpuRng, s);
			for (const auto& [label, gpuCloud] : clouds)
			{
				PointSpatialIndex index;
				index.build(gpuCloud, schema);
				PointGpuIndex device;
				device.upload(index);

				const std::vector<PointPrimitive>& points = gpuCloud.points();
				const AABB bounds = gpuCloud.empty() ? AABB(glm::vec3(0.0f), glm::vec3(1.0f)) : gpuCloud.bounds();
				const glm::vec3 size = glm::max(bounds.size(), glm::vec3(1.0f));
				std::vector<PointGpuIndex::Query> queries;
				for (int q = 0; q < 64; ++q)
				{
					glm::vec3 center;
					if (!points.empty() && gpuRng.chance(0.7))
						center = points[gpuRng.below(points.size())].position;
					else
						for (int axis = 0; axis < 3; ++axis)
							center[axis] = bounds.min()[axis] - 0.2f * size[axis] + static_cast<float>(gpuRng.unit()) * 1.4f * size[axis];
					const float scale = static_cast<float>(glm::length(size) * std::pow(10.0, -2.5 + 2.5 * gpuRng.unit()));
					PointGpuIndex::Query query;
					query._kind = static_cast<PointGpuIndex::Kind>(q % 4);
					switch (query._kind)
					{
					case PointGpuIndex::Kind::Range:
					case PointGpuIndex::Kind::CountRange:
					{
						glm::vec3 half;
						for (int axis = 0; axis < 3; ++axis)
							half[axis] = scale * static_cast<float>(0.1 + gpuRng.unit());
						query._min = center - half;
						query._max = center + half;
						break;
					}
					case PointGpuIndex::Kind::Radius:
						query._min = center;
						query._radius = gpuRng.chance(0.1) ? 0.0f : scale;
						break;
					case PointGpuIndex::Kind::Knn:
					{
						static const uint32_t Ks[] = { 1, 3, 16, 64 };
						query._min = center;
						query._k = points.size() + 3 <= PointGpuIndex::MaxK && gpuRng.chance(0.2)
							? static_cast<uint32_t>(points.size() + 3) : gpuRng.pick(Ks);
						break;
					}
					}
					queries.push_back(query);
				}

				const bool sorted = s % 2 == 0;
				const PointGpuIndex::BatchResult full = device.run(queries, true, sorted);
				const PointGpuIndex::BatchResult counted = device.run(queries, false, !sorted);
				for (size_t q = 0; q < queries.size(); ++q)
				{
					const PointGpuIndex::Query& query = queries[q];
					const std::string where = label + " / " + schema._name + " / gpu q" + std::to_string(q);
					expect(full._counts[q] == counted._counts[q] && full._checksums[q] == counted._checksums[q],
						   "gpu answers agree with and without indices and sorting: " + where);
					const auto slice = [&](size_t count) {
						const auto begin = full._indices.begin() + static_cast<std::ptrdiff_t>(full._offsets[q]);
						return std::vector<size_t>(begin, begin + static_cast<std::ptrdiff_t>(count));
					};
					switch (query._kind)
					{
					case PointGpuIndex::Kind::Range:
					{
						const PointSpatialIndex::QueryResult cpu = index.rangeQuery(AABB(query._min, query._max));
						expect(full._counts[q] == cpu._pointIndices.size() && full._checksums[q] == cpu._stats._resultChecksum,
							   "gpu range count and checksum == cpu: " + where);
						expect(sortedCopy(slice(full._counts[q])) == sortedCopy(cpu._pointIndices), "gpu range set == cpu: " + where);
						break;
					}
					case PointGpuIndex::Kind::CountRange:
						expect(full._counts[q] == index.countRange(AABB(query._min, query._max))._count && full._checksums[q] == 0,
							   "gpu count == cpu: " + where);
						break;
					case PointGpuIndex::Kind::Radius:
					{
						const PointSpatialIndex::QueryResult cpu = index.radiusQuery(query._min, query._radius);
						expect(full._counts[q] == cpu._pointIndices.size() && full._checksums[q] == cpu._stats._resultChecksum,
							   "gpu radius count and checksum == cpu: " + where);
						expect(sortedCopy(slice(full._counts[q])) == sortedCopy(cpu._pointIndices), "gpu radius set == cpu: " + where);
						break;
					}
					case PointGpuIndex::Kind::Knn:
					{
						const PointSpatialIndex::QueryResult cpu = index.knnQuery(query._min, query._k);
						expect(full._counts[q] == cpu._pointIndices.size() && slice(full._counts[q]) == cpu._pointIndices,
							   "gpu knn == cpu, in order: " + where);
						break;
					}
					}
				}
			}
		}
	}
}

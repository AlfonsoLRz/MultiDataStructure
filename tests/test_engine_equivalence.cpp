#include "../MultiDataStructure/stdafx.h"
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
				"Octree", "QuadTree", "KDTree", "BVH", "BIH", "RegularGrid", "HGrid", "LBVH", "KarrasOctree" };
			static const char* const KdPolicies[] = { "", "median_longest_axis", "round_robin", "center_longest_axis" };
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
	}
}

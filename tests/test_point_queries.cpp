#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/workloads/points/PointSpatialIndex.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		SchemaConfig makeQuerySchema()
		{
			SchemaLevelConfig level;
			level.type = MultiDataStructure::DataStructureLevel::OctreeNode;
			level.typeName = "Octree";
			level.numLevels = 6;
			level.leafCapacity = 6;
			level.minPrimitivesToSplit = 4;

			SchemaConfig schema;
			schema.name = "point_query_test";
			schema.levels.push_back(level);
			schema.buildPolicy.maxDepth = 6;
			schema.buildPolicy.leafCapacity = 6;
			schema.buildPolicy.minPrimitivesToSplit = 4;
			schema.buildPolicy.collapseSingleChild = false;
			schema.buildPolicy.removeEmptyNodes = true;
			schema.buildPolicy.allowOverlapDuplication = false;
			return schema;
		}

		bool containsPoint(const AABB& bounds, const glm::vec3& point)
		{
			const glm::vec3 min = bounds.min();
			const glm::vec3 max = bounds.max();
			return point.x >= min.x && point.x <= max.x &&
				   point.y >= min.y && point.y <= max.y &&
				   point.z >= min.z && point.z <= max.z;
		}

		std::vector<size_t> sorted(std::vector<size_t> values)
		{
			std::sort(values.begin(), values.end());
			return values;
		}

		std::vector<size_t> bruteForceRange(const PointCloud& cloud, const AABB& bounds)
		{
			std::vector<size_t> result;
			for (size_t pointIndex = 0; pointIndex < cloud.size(); ++pointIndex)
			{
				if (containsPoint(bounds, cloud.points()[pointIndex].position))
					result.push_back(pointIndex);
			}
			return result;
		}

		std::vector<size_t> bruteForceRadius(const PointCloud& cloud, const glm::vec3& center, float radius)
		{
			const float radiusSquared = radius * radius;
			std::vector<size_t> result;
			for (size_t pointIndex = 0; pointIndex < cloud.size(); ++pointIndex)
			{
				if (glm::length2(cloud.points()[pointIndex].position - center) <= radiusSquared)
					result.push_back(pointIndex);
			}
			return result;
		}

		std::vector<size_t> bruteForceKnn(const PointCloud& cloud, const glm::vec3& center, size_t k)
		{
			std::vector<std::pair<float, size_t>> distances;
			distances.reserve(cloud.size());
			for (size_t pointIndex = 0; pointIndex < cloud.size(); ++pointIndex)
				distances.push_back({ glm::length2(cloud.points()[pointIndex].position - center), pointIndex });

			std::sort(distances.begin(), distances.end(), [](const auto& left, const auto& right) {
				if (left.first == right.first)
					return left.second < right.second;
				return left.first < right.first;
			});

			std::vector<size_t> result;
			const size_t count = std::min(k, distances.size());
			result.reserve(count);
			for (size_t i = 0; i < count; ++i)
				result.push_back(distances[i].second);

			return result;
		}

		void expectSameSet(const std::vector<size_t>& actual, const std::vector<size_t>& expected, const std::string& message)
		{
			expect(sorted(actual) == sorted(expected), message);
		}
	}

	void runPointQueryTests()
	{
		const PointCloud cloud = SyntheticPointClouds::generateSparseDenseMixture(48, 96, 13);
		const SchemaConfig schema = makeQuerySchema();

		PointSpatialIndex index;
		index.build(cloud, schema);

		const PointSpatialIndex::Stats buildStats = index.stats();
		expect(buildStats.numPoints == cloud.size(), "point query index keeps every point in leaves");
		expect(buildStats.numNodes > 1, "point query index creates a searchable hierarchy");

		const AABB rangeBounds(glm::vec3(7.0f, -7.0f, 2.0f), glm::vec3(9.0f, -5.0f, 4.0f));
		const std::vector<size_t> bruteRange = bruteForceRange(cloud, rangeBounds);
		const PointSpatialIndex::QueryResult range = index.rangeQuery(rangeBounds);
		expectSameSet(range.pointIndices, bruteRange, "range query matches brute force");
		expect(range.stats.visitedNodes > 0, "range query records visited nodes");
		expect(range.stats.testedPoints > 0, "range query records tested points");
		expect(range.stats.returnedPoints == range.pointIndices.size(), "range query records returned points");

		const PointSpatialIndex::CountResult count = index.countRange(rangeBounds);
		expect(count.count == bruteRange.size(), "count range query matches brute force");
		expect(count.stats.returnedPoints == count.count, "count query records returned count");

		const glm::vec3 denseCenter(8.0f, -6.0f, 3.0f);
		const float radius = 2.0f;
		const std::vector<size_t> bruteRadius = bruteForceRadius(cloud, denseCenter, radius);
		const PointSpatialIndex::QueryResult radiusResult = index.radiusQuery(denseCenter, radius);
		expectSameSet(radiusResult.pointIndices, bruteRadius, "radius query matches brute force");
		expect(radiusResult.stats.visitedNodes > 0, "radius query records visited nodes");
		expect(radiusResult.stats.testedPoints > 0, "radius query records tested points");

		const size_t k = 7;
		const std::vector<size_t> bruteKnn = bruteForceKnn(cloud, denseCenter, k);
		const PointSpatialIndex::QueryResult knn = index.knnQuery(denseCenter, k);
		expect(knn.pointIndices == bruteKnn, "KNN query matches brute force ordering");
		expect(knn.stats.returnedPoints == k, "KNN query records returned neighbors");
		expect(knn.stats.visitedNodes > 0, "KNN query records visited nodes");
		expect(knn.stats.testedPoints >= k, "KNN query tests enough points to fill neighbors");

		const PointSpatialIndex::QueryResult emptyKnn = index.knnQuery(denseCenter, 0);
		expect(emptyKnn.pointIndices.empty(), "KNN query with k=0 returns no points");
		expect(emptyKnn.stats.returnedPoints == 0, "KNN query with k=0 records no returned points");
	}
}

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
			level._type = MultiDataStructure::DataStructureLevel::OctreeNode;
			level._typeName = "Octree";
			level._numLevels = 6;
			level._leafCapacity = 6;
			level._minPrimitivesToSplit = 4;

			SchemaConfig schema;
			schema._name = "point_query_test";
			schema._levels.push_back(level);
			schema._buildPolicy._maxDepth = 6;
			schema._buildPolicy._leafCapacity = 6;
			schema._buildPolicy._minPrimitivesToSplit = 4;
			schema._buildPolicy._collapseSingleChild = false;
			schema._buildPolicy._removeEmptyNodes = true;
			schema._buildPolicy._allowOverlapDuplication = false;
			return schema;
		}

		SchemaConfig makePrimitiveSchema(const std::string& typeName, size_t numLevels = 2)
		{
			SchemaLevelConfig level;
			level._typeName = typeName;
			level._primitiveKind = Config::parseSchemaPrimitiveKind(typeName);
			level._cpuFallbackType = Config::cpuFallbackForPrimitiveKind(level._primitiveKind);
			level._type = level._cpuFallbackType;
			level._numLevels = numLevels;
			level._leafCapacity = 6;
			level._minPrimitivesToSplit = 4;

			SchemaConfig schema;
			schema._name = typeName + "_point_query_test";
			schema._levels.push_back(level);
			schema._buildPolicy._maxDepth = numLevels;
			schema._buildPolicy._leafCapacity = 6;
			schema._buildPolicy._minPrimitivesToSplit = 4;
			schema._buildPolicy._collapseSingleChild = false;
			schema._buildPolicy._removeEmptyNodes = false;
			schema._buildPolicy._allowOverlapDuplication = false;
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

		void expectQueryCorrectness(const PointCloud& cloud, const PointSpatialIndex& index, const AABB& rangeBounds, const glm::vec3& center, float radius, const std::string& label)
		{
			expectSameSet(index.rangeQuery(rangeBounds)._pointIndices, bruteForceRange(cloud, rangeBounds), label + " range query matches brute force");
			expect(index.countRange(rangeBounds)._count == bruteForceRange(cloud, rangeBounds).size(), label + " count range query matches brute force");
			expectSameSet(index.radiusQuery(center, radius)._pointIndices, bruteForceRadius(cloud, center, radius), label + " radius query matches brute force");
			expect(index.knnQuery(center, 9)._pointIndices == bruteForceKnn(cloud, center, 9), label + " KNN query matches brute force");
		}

		PointCloud makeTieCloud()
		{
			PointCloud cloud;
			for (const glm::vec3& position : {
				glm::vec3(1.0f, 0.0f, 0.0f),
				glm::vec3(-1.0f, 0.0f, 0.0f),
				glm::vec3(0.0f, 1.0f, 0.0f),
				glm::vec3(0.0f, 0.0f, 0.0f),
				glm::vec3(0.0f, 0.0f, 0.0f) })
			{
				PointPrimitive point;
				point.position = position;
				cloud.addPoint(point);
			}
			return cloud;
		}

		PointCloud makeTightBoundsRadiusCloud()
		{
			PointCloud cloud;
			for (int x = 0; x < 4; ++x)
			{
				for (int y = 0; y < 4; ++y)
				{
					for (int z = 0; z < 2; ++z)
					{
						PointPrimitive point;
						point.position = glm::vec3(
							0.1f + static_cast<float>(x) * 0.05f,
							0.1f + static_cast<float>(y) * 0.05f,
							0.1f + static_cast<float>(z) * 0.05f);
						cloud.addPoint(point);
					}
				}
			}

			PointPrimitive outlier;
			outlier.position = glm::vec3(100.0f, 100.0f, 100.0f);
			cloud.addPoint(outlier);
			return cloud;
		}
	}

	void runPointQueryTests()
	{
		const PointCloud cloud = SyntheticPointClouds::generateSparseDenseMixture(48, 96, 13);
		const SchemaConfig schema = makeQuerySchema();

		PointSpatialIndex index;
		index.build(cloud, schema);

		const PointSpatialIndex::Stats buildStats = index.stats();
		expect(index.root() && index.root()->_pointCount == cloud.size(), "point query index stores root as a contiguous point range");
		expect(buildStats._numPoints == cloud.size(), "point query index keeps every point in leaves");
		expect(buildStats._numNodes > 1, "point query index creates a searchable hierarchy");

		const AABB rangeBounds(glm::vec3(7.0f, -7.0f, 2.0f), glm::vec3(9.0f, -5.0f, 4.0f));
		const std::vector<size_t> bruteRange = bruteForceRange(cloud, rangeBounds);
		const PointSpatialIndex::QueryResult range = index.rangeQuery(rangeBounds);
		expectSameSet(range._pointIndices, bruteRange, "range query matches brute force");
		expect(range._stats._visitedNodes > 0, "range query records visited nodes");
		expect(range._stats._testedPoints > 0 || range._stats._fullyContainedNodes > 0, "range query records tested points or containment shortcuts");
		expect(range._stats._returnedPoints == range._pointIndices.size(), "range query records returned points");
		const size_t rangeVisitedByDepth = std::accumulate(
			range._stats._breakdown.visitedByDepth.begin(),
			range._stats._breakdown.visitedByDepth.end(),
			size_t(0));
		expect(rangeVisitedByDepth == range._stats._visitedNodes, "range query breakdown accounts for visited nodes by depth");
		expect(range._stats._breakdown._visitedByStructure.at("Octree") == range._stats._visitedNodes,
			"range query breakdown accounts for visited nodes by structure");

		const PointSpatialIndex::CountResult count = index.countRange(rangeBounds);
		expect(count._count == bruteRange.size(), "count range query matches brute force");
		expect(count._stats._returnedPoints == count._count, "count query records returned count");

		const AABB largeBounds(cloud.bounds().min() - glm::vec3(0.01f), cloud.bounds().max() + glm::vec3(0.01f));
		const PointSpatialIndex::QueryResult largeRange = index.rangeQuery(largeBounds);
		expectSameSet(largeRange._pointIndices, bruteForceRange(cloud, largeBounds), "large range query with containment fast path matches brute force");
		expect(largeRange._stats._fullyContainedNodes > 0, "large range query records fully contained nodes");
		expect(largeRange._stats._testedPoints < cloud.size(), "large range query tests fewer points through containment fast path");
		const PointSpatialIndex::CountResult largeCount = index.countRange(largeBounds);
		expect(largeCount._count == cloud.size(), "large count query with containment fast path matches brute force");
		expect(largeCount._stats._fullyContainedNodes > 0, "large count query records fully contained nodes");
		expect(largeCount._stats._testedPoints < cloud.size(), "large count query tests fewer points through containment fast path");

		const glm::vec3 denseCenter(8.0f, -6.0f, 3.0f);
		const float radius = 2.0f;
		const std::vector<size_t> bruteRadius = bruteForceRadius(cloud, denseCenter, radius);
		const PointSpatialIndex::QueryResult radiusResult = index.radiusQuery(denseCenter, radius);
		expectSameSet(radiusResult._pointIndices, bruteRadius, "radius query matches brute force");
		expect(radiusResult._stats._visitedNodes > 0, "radius query records visited nodes");
		expect(radiusResult._stats._testedPoints > 0, "radius query records tested points");
		expect(radiusResult._stats._breakdown._testedPointsByStructure.at("Octree") == radiusResult._stats._testedPoints,
			"radius query breakdown accounts for tested points by structure");

		const size_t k = 7;
		const std::vector<size_t> bruteKnn = bruteForceKnn(cloud, denseCenter, k);
		const PointSpatialIndex::QueryResult knn = index.knnQuery(denseCenter, k);
		expect(knn._pointIndices == bruteKnn, "KNN query matches brute force ordering");
		expect(knn._stats._returnedPoints == k, "KNN query records returned neighbors");
		expect(knn._stats._visitedNodes > 0, "KNN query records visited nodes");
		expect(knn._stats._testedPoints >= k, "KNN query tests enough points to fill neighbors");

		const PointSpatialIndex::QueryResult emptyKnn = index.knnQuery(denseCenter, 0);
		expect(emptyKnn._pointIndices.empty(), "KNN query with k=0 returns no points");
		expect(emptyKnn._stats._returnedPoints == 0, "KNN query with k=0 records no returned points");

		const PointCloud tieCloud = makeTieCloud();
		PointSpatialIndex tieIndex;
		tieIndex.build(tieCloud, makeQuerySchema());
		const glm::vec3 origin(0.0f);
		const PointSpatialIndex::QueryResult duplicateKnn = tieIndex.knnQuery(origin, 2);
		expect((duplicateKnn._pointIndices == std::vector<size_t>{ 3, 4 }), "KNN returns duplicate zero-distance points in index order");
		const PointSpatialIndex::QueryResult tieKnn = tieIndex.knnQuery(origin, 5);
		expect(tieKnn._pointIndices == bruteForceKnn(tieCloud, origin, 5), "KNN preserves equal-distance tie ordering");
		const PointSpatialIndex::QueryResult oversizedKnn = tieIndex.knnQuery(origin, 10);
		expect(oversizedKnn._pointIndices == bruteForceKnn(tieCloud, origin, 10), "KNN with k > num points returns all points in brute-force order");
		const glm::vec3 outsideCenter(100.0f, 100.0f, 100.0f);
		const PointSpatialIndex::QueryResult outsideKnn = tieIndex.knnQuery(outsideCenter, 3);
		expect(outsideKnn._pointIndices == bruteForceKnn(tieCloud, outsideCenter, 3), "KNN outside cloud bounds matches brute force");

		PointSpatialIndex rebuiltIndex;
		rebuiltIndex.build(cloud, schema);
		rebuiltIndex.build(tieCloud, makeQuerySchema());
		expectSameSet(rebuiltIndex.radiusQuery(origin, 1.01f)._pointIndices, bruteForceRadius(tieCloud, origin, 1.01f),
			"point query index rebuild refreshes ordered SoA point storage");

		const PointCloud tightBoundsCloud = makeTightBoundsRadiusCloud();
		PointSpatialIndex tightBoundsIndex;
		tightBoundsIndex.build(tightBoundsCloud, makePrimitiveSchema("Octree", 4));
		const PointSpatialIndex::QueryResult tightBoundsRadius = tightBoundsIndex.radiusQuery(glm::vec3(49.0f, 49.0f, 49.0f), 0.25f);
		expect(tightBoundsRadius._pointIndices.empty(), "radius query outside tight node bounds returns no points");
		expect(tightBoundsIndex.root() && tightBoundsRadius._stats._visitedNodes <= tightBoundsIndex.root()->_children.size() + 1,
			"radius query prunes sparse nodes with tight bounds before visiting grandchildren");

		SchemaConfig microSchema = makePrimitiveSchema("Octree", 1);
		microSchema._levels[0]._leafCapacity = 4096;
		microSchema._levels[0]._minPrimitivesToSplit = 4096;
		microSchema._buildPolicy._leafCapacity = 4096;
		microSchema._buildPolicy._minPrimitivesToSplit = 4096;
		microSchema._buildPolicy._enableLeafMicroIndexes = true;
		microSchema._buildPolicy._leafMicroIndexThreshold = 16;
		PointSpatialIndex microIndex;
		microIndex.build(cloud, microSchema);
		expect(microIndex.root() && microIndex.root()->isLeaf(), "micro-index test keeps one heavy leaf");
		expect(microIndex.root() && microIndex.root()->_microIndex, "heavy leaf builds optional micro-index");
		expectQueryCorrectness(cloud, microIndex, rangeBounds, denseCenter, radius, "heavy-leaf micro-index");

		const AABB gridRange(glm::vec3(-15.0f, -15.0f, -4.0f), glm::vec3(15.0f, 15.0f, 8.0f));
		for (const std::string& typeName : { std::string("RegularGrid"), std::string("HGrid") })
		{
			PointSpatialIndex gridIndex;
			gridIndex.build(cloud, makePrimitiveSchema(typeName, 2));
			expect(gridIndex.root() && gridIndex.root()->_children.size() > 8, typeName + " CPU split creates grid cells instead of collapsing to Octree");
			expect(gridIndex.stats()._numPoints == cloud.size(), typeName + " CPU split preserves point count");
			expect(gridIndex.root()->_subtreePointCount == cloud.size(), typeName + " CPU split preserves subtree point count");
			expectQueryCorrectness(cloud, gridIndex, gridRange, denseCenter, radius, typeName + " CPU grid");
		}
	}
}

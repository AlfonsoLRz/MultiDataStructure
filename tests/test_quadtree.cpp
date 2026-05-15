#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/workloads/points/QuadTree.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		SchemaConfig makeQuadTreeSchema()
		{
			SchemaLevelConfig level;
			level.type = MultiDataStructure::DataStructureLevel::QuadTreeNode;
			level.typeName = "QuadTree";
			level.numLevels = 5;
			level.leafCapacity = 16;
			level.minPrimitivesToSplit = 4;

			SchemaConfig schema;
			schema.name = "quadtree_test";
			schema.levels.push_back(level);
			schema.buildPolicy.maxDepth = 5;
			schema.buildPolicy.leafCapacity = 16;
			schema.buildPolicy.minPrimitivesToSplit = 4;
			schema.buildPolicy.collapseSingleChild = true;
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

		size_t bruteForceRangeCount(const PointCloud& cloud, const AABB& bounds)
		{
			size_t count = 0;
			for (const PointPrimitive& point : cloud.points())
			{
				if (containsPoint(bounds, point.position))
					++count;
			}
			return count;
		}

		size_t bruteForceRadiusCount(const PointCloud& cloud, const glm::vec3& center, float radius)
		{
			const float radiusSquared = radius * radius;
			size_t count = 0;
			for (const PointPrimitive& point : cloud.points())
			{
				if (glm::length2(point.position - center) <= radiusSquared)
					++count;
			}
			return count;
		}
	}

	void runQuadTreeTests()
	{
		std::string cudaError;
		if (!PointGpu::QuadTree::isAvailable(&cudaError))
		{
			std::cout << "QuadTree tests skipped: " << cudaError << '\n';
			return;
		}

		const PointCloud cloud = SyntheticPointClouds::generateSparseDenseMixture(64, 128, 41);
		PointGpu::QuadTree index;
		PointGpu::Options options;
		options.builder = "quadtree";
		const PointGpu::BuildResult build = index.build(cloud, makeQuadTreeSchema(), options);
		expect(build.metrics.indexedPoints == cloud.size(), "QuadTree indexes every point");
		expect(build.metrics.numLeaves > 0, "QuadTree creates leaves");
		expect(build.metrics.numNodes >= build.metrics.numLeaves, "QuadTree creates a valid node array");

		std::vector<PointGpu::Query> queries;

		PointGpu::Query range;
		range.type = PointGpu::QueryType::Range;
		range.bounds = AABB(glm::vec3(7.0f, -7.0f, 2.0f), glm::vec3(9.0f, -5.0f, 4.0f));
		queries.push_back(range);

		PointGpu::Query countRange = range;
		countRange.type = PointGpu::QueryType::CountRange;
		queries.push_back(countRange);

		PointGpu::Query radius;
		radius.type = PointGpu::QueryType::Radius;
		radius.center = glm::vec3(8.0f, -6.0f, 3.0f);
		radius.radius = 2.0f;
		queries.push_back(radius);

		PointGpu::Query knn;
		knn.type = PointGpu::QueryType::Knn;
		knn.center = glm::vec3(8.0f, -6.0f, 3.0f);
		knn.k = 7;
		queries.push_back(knn);

		const PointGpu::QueryResult result = index.query(queries, options);
		expect(result.samples.size() == queries.size(), "QuadTree returns one sample per query");
		expect(result.samples[0].returnedPoints == bruteForceRangeCount(cloud, range.bounds), "QuadTree range count matches brute force");
		expect(result.samples[1].returnedPoints == bruteForceRangeCount(cloud, countRange.bounds), "QuadTree count-range matches brute force");
		expect(result.samples[2].returnedPoints == bruteForceRadiusCount(cloud, radius.center, radius.radius), "QuadTree radius count matches brute force");
		expect(result.samples[3].returnedPoints == std::min(knn.k, cloud.size()), "QuadTree KNN returns requested neighbor count");
		expect(result.samples[3].testedPoints == cloud.size(), "QuadTree KNN scans the GPU point buffer");
		expect(result.knnQueries == 1, "QuadTree counts KNN queries");
		expect(result.metrics.totalQueries == queries.size(), "QuadTree summarizes query samples");
	}
}

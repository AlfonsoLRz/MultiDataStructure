#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/workloads/points/BIH.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		SchemaConfig makeBIHSchema()
		{
			SchemaLevelConfig level;
			level.type = MultiDataStructure::DataStructureLevel::KDTreeNode;
			level.typeName = "KDTree";
			level.numLevels = 8;
			level.leafCapacity = 16;
			level.minPrimitivesToSplit = 4;

			SchemaConfig schema;
			schema.name = "bih_test";
			schema.levels.push_back(level);
			schema.buildPolicy.maxDepth = 8;
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

	void runBIHTests()
	{
		std::string cudaError;
		if (!PointGpu::BIH::isAvailable(&cudaError))
		{
			std::cout << "BIH tests skipped: " << cudaError << '\n';
			return;
		}

		const PointCloud cloud = SyntheticPointClouds::generateSparseDenseMixture(64, 128, 37);
		PointGpu::BIH index;
		PointGpu::Options options;
		options.builder = "bih";
		const PointGpu::BuildResult build = index.build(cloud, makeBIHSchema(), options);
		expect(build.builder == "bih", "BIH reports its builder name");
		expect(build.metrics.indexedPoints == cloud.size(), "BIH indexes every point");
		expect(build.metrics.numLeaves > 0, "BIH creates leaves");
		expect(build.metrics.numNodes >= build.metrics.numLeaves, "BIH creates a valid node array");

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
		expect(result.samples.size() == queries.size(), "BIH returns one sample per query");
		expect(result.samples[0].returnedPoints == bruteForceRangeCount(cloud, range.bounds), "BIH range count matches brute force");
		expect(result.samples[1].returnedPoints == bruteForceRangeCount(cloud, countRange.bounds), "BIH count-range matches brute force");
		expect(result.samples[2].returnedPoints == bruteForceRadiusCount(cloud, radius.center, radius.radius), "BIH radius count matches brute force");
		expect(result.samples[3].returnedPoints == std::min(knn.k, cloud.size()), "BIH KNN returns requested neighbor count");
		expect(result.samples[3].testedPoints == cloud.size(), "BIH KNN scans the GPU point buffer");
		expect(result.knnQueries == 1, "BIH counts KNN queries");
		expect(result.metrics.totalQueries == queries.size(), "BIH summarizes query samples");
	}
}

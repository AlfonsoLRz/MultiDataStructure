#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/workloads/points/RegularGrid.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		SchemaConfig makeRegularGridSchema()
		{
			SchemaLevelConfig level;
			level._type = MultiDataStructure::DataStructureLevel::BvhNode;
			level._typeName = "BVH";
			level._numLevels = 1;
			level._leafCapacity = 16;
			level._minPrimitivesToSplit = 4;

			SchemaConfig schema;
			schema._name = "regular_grid_test";
			schema._levels.push_back(level);
			schema._buildPolicy._maxDepth = 1;
			schema._buildPolicy._leafCapacity = 16;
			schema._buildPolicy._minPrimitivesToSplit = 4;
			schema._buildPolicy._collapseSingleChild = true;
			schema._buildPolicy._removeEmptyNodes = true;
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

	void runRegularGridTests()
	{
		std::string cudaError;
		if (!PointGpu::RegularGrid::isAvailable(&cudaError))
		{
			std::cout << "RegularGrid tests skipped: " << cudaError << '\n';
			return;
		}

		const PointCloud cloud = SyntheticPointClouds::generateSparseDenseMixture(64, 128, 31);
		PointGpu::RegularGrid index;
		PointGpu::Options options;
		options._builder = "regular_grid";
		const PointGpu::BuildResult build = index.build(cloud, makeRegularGridSchema(), options);
		expect(build._metrics._indexedPoints == cloud.size(), "RegularGrid indexes every point");
		expect(build._metrics._numNodes >= build._metrics._numLeaves, "RegularGrid reports cells and occupied cells");
		expect(index.cellCount() == build._metrics._numNodes, "RegularGrid exposes its cell count");

		std::vector<PointGpu::Query> queries;

		PointGpu::Query range;
		range._type = PointGpu::QueryType::Range;
		range._bounds = AABB(glm::vec3(7.0f, -7.0f, 2.0f), glm::vec3(9.0f, -5.0f, 4.0f));
		queries.push_back(range);

		PointGpu::Query countRange = range;
		countRange._type = PointGpu::QueryType::CountRange;
		queries.push_back(countRange);

		PointGpu::Query radius;
		radius._type = PointGpu::QueryType::Radius;
		radius.center = glm::vec3(8.0f, -6.0f, 3.0f);
		radius._radius = 2.0f;
		queries.push_back(radius);

		PointGpu::Query knn;
		knn._type = PointGpu::QueryType::Knn;
		knn.center = glm::vec3(8.0f, -6.0f, 3.0f);
		knn._k = 7;
		queries.push_back(knn);

		const PointGpu::QueryResult result = index.query(queries, options);
		expect(result._samples.size() == queries.size(), "RegularGrid returns one sample per query");
		expect(result._samples[0]._returnedPoints == bruteForceRangeCount(cloud, range._bounds), "RegularGrid range count matches brute force");
		expect(result._samples[1]._returnedPoints == bruteForceRangeCount(cloud, countRange._bounds), "RegularGrid count-range matches brute force");
		expect(result._samples[2]._returnedPoints == bruteForceRadiusCount(cloud, radius.center, radius._radius), "RegularGrid radius count matches brute force");
		expect(result._samples[3]._returnedPoints == std::min(knn._k, cloud.size()), "RegularGrid KNN returns requested neighbor count");
		expect(result._samples[3]._testedPoints == cloud.size(), "RegularGrid KNN scans the GPU point buffer");
		expect(result._knnQueries == 1, "RegularGrid counts KNN queries");
		expect(result._metrics._totalQueries == queries.size(), "RegularGrid summarizes query samples");
	}
}

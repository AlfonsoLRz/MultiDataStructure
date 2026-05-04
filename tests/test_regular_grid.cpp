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
			level.type = MultiDataStructure::DataStructureLevel::BvhNode;
			level.typeName = "BVH";
			level.numLevels = 1;
			level.leafCapacity = 16;
			level.minPrimitivesToSplit = 4;

			SchemaConfig schema;
			schema.name = "regular_grid_test";
			schema.levels.push_back(level);
			schema.buildPolicy.maxDepth = 1;
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
		options.builder = "regular_grid";
		const PointGpu::BuildResult build = index.build(cloud, makeRegularGridSchema(), options);
		expect(build.metrics.indexedPoints == cloud.size(), "RegularGrid indexes every point");
		expect(build.metrics.numNodes >= build.metrics.numLeaves, "RegularGrid reports cells and occupied cells");
		expect(index.cellCount() == build.metrics.numNodes, "RegularGrid exposes its cell count");

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

		const PointGpu::QueryResult result = index.query(queries, options);
		expect(result.samples.size() == queries.size(), "RegularGrid returns one sample per query");
		expect(result.samples[0].returnedPoints == bruteForceRangeCount(cloud, range.bounds), "RegularGrid range count matches brute force");
		expect(result.samples[1].returnedPoints == bruteForceRangeCount(cloud, countRange.bounds), "RegularGrid count-range matches brute force");
		expect(result.samples[2].returnedPoints == bruteForceRadiusCount(cloud, radius.center, radius.radius), "RegularGrid radius count matches brute force");
		expect(result.metrics.totalQueries == queries.size(), "RegularGrid summarizes query samples");
	}
}

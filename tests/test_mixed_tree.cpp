#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/workloads/points/MixedTree.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		SchemaConfig makeMixedSchema()
		{
			SchemaConfig schema;
			schema.name = "mixed_gpu_test";

			SchemaLevelConfig quad;
			quad.type = MultiDataStructure::DataStructureLevel::QuadTreeNode;
			quad.typeName = "QuadTree";
			quad.numLevels = 1;
			quad.leafCapacity = 32;
			quad.minPrimitivesToSplit = 8;
			schema.levels.push_back(quad);

			SchemaLevelConfig regularGrid;
			regularGrid.type = MultiDataStructure::DataStructureLevel::OctreeNode;
			regularGrid.typeName = "RegularGrid";
			regularGrid.numLevels = 1;
			regularGrid.leafCapacity = 24;
			regularGrid.minPrimitivesToSplit = 8;
			schema.levels.push_back(regularGrid);

			SchemaLevelConfig hgrid;
			hgrid.type = MultiDataStructure::DataStructureLevel::OctreeNode;
			hgrid.typeName = "HGrid";
			hgrid.numLevels = 1;
			hgrid.leafCapacity = 20;
			hgrid.minPrimitivesToSplit = 8;
			schema.levels.push_back(hgrid);

			SchemaLevelConfig karras;
			karras.type = MultiDataStructure::DataStructureLevel::OctreeNode;
			karras.typeName = "KarrasOctree";
			karras.numLevels = 1;
			karras.leafCapacity = 24;
			karras.minPrimitivesToSplit = 8;
			schema.levels.push_back(karras);

			SchemaLevelConfig octree;
			octree.type = MultiDataStructure::DataStructureLevel::OctreeNode;
			octree.typeName = "Octree";
			octree.numLevels = 1;
			octree.leafCapacity = 16;
			octree.minPrimitivesToSplit = 4;
			octree.condition.minPoints = 8;
			schema.levels.push_back(octree);

			SchemaLevelConfig bih;
			bih.type = MultiDataStructure::DataStructureLevel::KDTreeNode;
			bih.typeName = "BIH";
			bih.numLevels = 1;
			bih.leafCapacity = 12;
			bih.minPrimitivesToSplit = 4;
			schema.levels.push_back(bih);

			SchemaLevelConfig kd;
			kd.type = MultiDataStructure::DataStructureLevel::KDTreeNode;
			kd.typeName = "KDTree";
			kd.numLevels = 1;
			kd.leafCapacity = 12;
			kd.minPrimitivesToSplit = 4;
			kd.axisPolicy = "center_longest_axis";
			schema.levels.push_back(kd);

			SchemaLevelConfig lbvh;
			lbvh.type = MultiDataStructure::DataStructureLevel::BvhNode;
			lbvh.typeName = "LBVH";
			lbvh.numLevels = 1;
			lbvh.leafCapacity = 12;
			lbvh.minPrimitivesToSplit = 4;
			schema.levels.push_back(lbvh);

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

	void runMixedTreeTests()
	{
		std::string cudaError;
		if (!PointGpu::MixedTree::isAvailable(&cudaError))
		{
			std::cout << "MixedTree tests skipped: " << cudaError << '\n';
			return;
		}

		const PointCloud cloud = SyntheticPointClouds::generateUrbanMixed(96, 96, 4, 77);
		PointGpu::MixedTree index;
		PointGpu::Options options;
		options.builder = "mixed";
		const PointGpu::BuildResult build = index.build(cloud, makeMixedSchema(), options);
		expect(build.metrics.indexedPoints == cloud.size(), "MixedTree indexes every point");
		expect(build.metrics.numLeaves > 0, "MixedTree creates leaves");
		expect(build.metrics.numNodes >= build.metrics.numLeaves, "MixedTree creates a valid node array");

		std::vector<PointGpu::Query> queries;

		PointGpu::Query range;
		range.type = PointGpu::QueryType::Range;
		range.bounds = AABB(glm::vec3(-15.0f, -15.0f, -2.0f), glm::vec3(15.0f, 15.0f, 8.0f));
		queries.push_back(range);

		PointGpu::Query countRange = range;
		countRange.type = PointGpu::QueryType::CountRange;
		queries.push_back(countRange);

		PointGpu::Query radius;
		radius.type = PointGpu::QueryType::Radius;
		radius.center = glm::vec3(0.0f, 0.0f, 2.0f);
		radius.radius = 18.0f;
		queries.push_back(radius);

		const PointGpu::QueryResult result = index.query(queries, options);
		expect(result.samples.size() == queries.size(), "MixedTree returns one sample per query");
		expect(result.samples[0].returnedPoints == bruteForceRangeCount(cloud, range.bounds), "MixedTree range count matches brute force");
		expect(result.samples[1].returnedPoints == bruteForceRangeCount(cloud, countRange.bounds), "MixedTree count-range matches brute force");
		expect(result.samples[2].returnedPoints == bruteForceRadiusCount(cloud, radius.center, radius.radius), "MixedTree radius count matches brute force");
		expect(result.metrics.totalQueries == queries.size(), "MixedTree summarizes query samples");
	}
}

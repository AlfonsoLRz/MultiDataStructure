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
			schema._name = "mixed_gpu_test";

			SchemaLevelConfig quad;
			quad._type = MultiDataStructure::DataStructureLevel::QuadTreeNode;
			quad._typeName = "QuadTree";
			quad._numLevels = 1;
			quad._leafCapacity = 32;
			quad._minPrimitivesToSplit = 8;
			schema._levels.push_back(quad);

			SchemaLevelConfig regularGrid;
			regularGrid._type = MultiDataStructure::DataStructureLevel::OctreeNode;
			regularGrid._typeName = "RegularGrid";
			regularGrid._numLevels = 1;
			regularGrid._leafCapacity = 24;
			regularGrid._minPrimitivesToSplit = 8;
			schema._levels.push_back(regularGrid);

			SchemaLevelConfig hgrid;
			hgrid._type = MultiDataStructure::DataStructureLevel::OctreeNode;
			hgrid._typeName = "HGrid";
			hgrid._numLevels = 1;
			hgrid._leafCapacity = 20;
			hgrid._minPrimitivesToSplit = 8;
			schema._levels.push_back(hgrid);

			SchemaLevelConfig karras;
			karras._type = MultiDataStructure::DataStructureLevel::OctreeNode;
			karras._typeName = "KarrasOctree";
			karras._numLevels = 1;
			karras._leafCapacity = 24;
			karras._minPrimitivesToSplit = 8;
			schema._levels.push_back(karras);

			SchemaLevelConfig octree;
			octree._type = MultiDataStructure::DataStructureLevel::OctreeNode;
			octree._typeName = "Octree";
			octree._numLevels = 1;
			octree._leafCapacity = 16;
			octree._minPrimitivesToSplit = 4;
			octree._condition._minPoints = 8;
			schema._levels.push_back(octree);

			SchemaLevelConfig bih;
			bih._type = MultiDataStructure::DataStructureLevel::KDTreeNode;
			bih._typeName = "BIH";
			bih._numLevels = 1;
			bih._leafCapacity = 12;
			bih._minPrimitivesToSplit = 4;
			schema._levels.push_back(bih);

			SchemaLevelConfig kd;
			kd._type = MultiDataStructure::DataStructureLevel::KDTreeNode;
			kd._typeName = "KDTree";
			kd._numLevels = 1;
			kd._leafCapacity = 12;
			kd._minPrimitivesToSplit = 4;
			kd._axisPolicy = "center_longest_axis";
			schema._levels.push_back(kd);

			SchemaLevelConfig lbvh;
			lbvh._type = MultiDataStructure::DataStructureLevel::BvhNode;
			lbvh._typeName = "LBVH";
			lbvh._numLevels = 1;
			lbvh._leafCapacity = 12;
			lbvh._minPrimitivesToSplit = 4;
			schema._levels.push_back(lbvh);

			schema._buildPolicy._maxDepth = 8;
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
		options._builder = "mixed";
		const PointGpu::BuildResult build = index.build(cloud, makeMixedSchema(), options);
		expect(build._metrics._indexedPoints == cloud.size(), "MixedTree indexes every point");
		expect(build._metrics._numLeaves > 0, "MixedTree creates leaves");
		expect(build._metrics._numNodes >= build._metrics._numLeaves, "MixedTree creates a valid node array");

		std::vector<PointGpu::Query> queries;

		PointGpu::Query range;
		range._type = PointGpu::QueryType::Range;
		range._bounds = AABB(glm::vec3(-15.0f, -15.0f, -2.0f), glm::vec3(15.0f, 15.0f, 8.0f));
		queries.push_back(range);

		PointGpu::Query countRange = range;
		countRange._type = PointGpu::QueryType::CountRange;
		queries.push_back(countRange);

		PointGpu::Query radius;
		radius._type = PointGpu::QueryType::Radius;
		radius.center = glm::vec3(0.0f, 0.0f, 2.0f);
		radius._radius = 18.0f;
		queries.push_back(radius);

		PointGpu::Query knn;
		knn._type = PointGpu::QueryType::Knn;
		knn.center = glm::vec3(0.0f, 0.0f, 2.0f);
		knn._k = 7;
		queries.push_back(knn);

		const PointGpu::QueryResult result = index.query(queries, options);
		expect(result._samples.size() == queries.size(), "MixedTree returns one sample per query");
		expect(result._samples[0]._returnedPoints == bruteForceRangeCount(cloud, range._bounds), "MixedTree range count matches brute force");
		expect(result._samples[1]._returnedPoints == bruteForceRangeCount(cloud, countRange._bounds), "MixedTree count-range matches brute force");
		expect(result._samples[2]._returnedPoints == bruteForceRadiusCount(cloud, radius.center, radius._radius), "MixedTree radius count matches brute force");
		expect(result._samples[3]._returnedPoints == std::min(knn._k, cloud.size()), "MixedTree KNN returns requested neighbor count");
		expect(result._samples[3]._testedPoints == cloud.size(), "MixedTree KNN scans the GPU point buffer");
		expect(result._knnQueries == 1, "MixedTree counts KNN queries");
		expect(result._metrics._totalQueries == queries.size(), "MixedTree summarizes query samples");
	}
}

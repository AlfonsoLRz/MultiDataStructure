#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/workloads/points/KDTree.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		SchemaConfig makeKDTreeSchema()
		{
			SchemaLevelConfig level;
			level.type = MultiDataStructure::DataStructureLevel::KDTreeNode;
			level.typeName = "KDTree";
			level.numLevels = 8;
			level.leafCapacity = 16;
			level.minPrimitivesToSplit = 4;

			SchemaConfig schema;
			schema.name = "kdtree_test";
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

		std::vector<uint32_t> bruteForceKnn(const PointCloud& cloud, const glm::vec3& center, size_t k)
		{
			std::vector<std::pair<float, uint32_t>> distances;
			distances.reserve(cloud.size());
			for (size_t pointIndex = 0; pointIndex < cloud.size(); ++pointIndex)
			{
				distances.push_back({
					glm::length2(cloud.points()[pointIndex].position - center),
					static_cast<uint32_t>(pointIndex)
				});
			}
			std::sort(distances.begin(), distances.end(), [](const auto& left, const auto& right) {
				if (left.first == right.first)
					return left.second < right.second;
				return left.first < right.first;
			});

			std::vector<uint32_t> result;
			const size_t count = std::min(k, distances.size());
			result.reserve(count);
			for (size_t i = 0; i < count; ++i)
				result.push_back(distances[i].second);
			return result;
		}
	}

	void runKDTreeTests()
	{
		std::string cudaError;
		if (!PointGpu::KDTree::isAvailable(&cudaError))
		{
			std::cout << "KDTree tests skipped: " << cudaError << '\n';
			return;
		}

		const PointCloud cloud = SyntheticPointClouds::generateSparseDenseMixture(64, 128, 37);
		PointGpu::KDTree index;
		PointGpu::Options options;
		options.builder = "kdtree";
		const PointGpu::BuildResult build = index.build(cloud, makeKDTreeSchema(), options);
		expect(build.metrics.indexedPoints == cloud.size(), "KDTree indexes every point");
		expect(build.metrics.numLeaves > 0, "KDTree creates leaves");
		expect(build.metrics.numNodes >= build.metrics.numLeaves, "KDTree creates a valid node array");

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
		expect(result.samples.size() == queries.size(), "KDTree returns one sample per query");
		expect(result.samples[0].returnedPoints == bruteForceRangeCount(cloud, range.bounds), "KDTree range count matches brute force");
		expect(result.samples[1].returnedPoints == bruteForceRangeCount(cloud, countRange.bounds), "KDTree count-range matches brute force");
		expect(result.samples[2].returnedPoints == bruteForceRadiusCount(cloud, radius.center, radius.radius), "KDTree radius count matches brute force");
		expect(result.samples[3].returnedPoints == std::min(knn.k, cloud.size()), "KDTree KNN returns requested neighbor count");
		expect(result.knnBackend == "gpu_tree_knn", "KDTree auto KNN uses tree backend for small k");
		expect(result.knnPointIndices.size() == queries.size(), "KDTree tree KNN returns per-query hit buffers");
		expect(result.knnPointIndices[3] == bruteForceKnn(cloud, knn.center, knn.k), "KDTree tree KNN returns exact nearest-neighbor ordering");
		expect(result.knnQueries == 1, "KDTree counts KNN queries");
		expect(result.metrics.totalQueries == queries.size(), "KDTree summarizes query samples");

		PointGpu::Options bruteOptions = options;
		bruteOptions.knnBackend = "gpu_bruteforce_knn";
		const PointGpu::QueryResult bruteResult = index.query(std::vector<PointGpu::Query>{ knn }, bruteOptions);
		expect(bruteResult.knnBackend == "gpu_bruteforce_knn", "KDTree can still use the explicit brute-force GPU KNN backend");
		expect(bruteResult.samples.size() == 1 && bruteResult.samples[0].testedPoints == cloud.size(),
			"KDTree brute-force KNN backend scans the GPU point buffer");
	}
}

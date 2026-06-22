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
			level._type = MultiDataStructure::DataStructureLevel::KDTreeNode;
			level._typeName = "KDTree";
			level._numLevels = 8;
			level._leafCapacity = 16;
			level._minPrimitivesToSplit = 4;

			SchemaConfig schema;
			schema._name = "bih_test";
			schema._levels.push_back(level);
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
		options._builder = "bih";
		const PointGpu::BuildResult build = index.build(cloud, makeBIHSchema(), options);
		expect(build._builder == "bih", "BIH reports its builder name");
		expect(build._metrics._indexedPoints == cloud.size(), "BIH indexes every point");
		expect(build._metrics._numLeaves > 0, "BIH creates leaves");
		expect(build._metrics._numNodes >= build._metrics._numLeaves, "BIH creates a valid node array");

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
		expect(result._samples.size() == queries.size(), "BIH returns one sample per query");
		expect(result._samples[0]._returnedPoints == bruteForceRangeCount(cloud, range._bounds), "BIH range count matches brute force");
		expect(result._samples[1]._returnedPoints == bruteForceRangeCount(cloud, countRange._bounds), "BIH count-range matches brute force");
		expect(result._samples[2]._returnedPoints == bruteForceRadiusCount(cloud, radius.center, radius._radius), "BIH radius count matches brute force");
		expect(result._samples[3]._returnedPoints == std::min(knn._k, cloud.size()), "BIH KNN returns requested neighbor count");
		expect(result._knnBackend == "gpu_tree_knn", "BIH auto KNN uses tree backend for small k");
		expect(result._knnPointIndices.size() == queries.size(), "BIH tree KNN returns per-query hit buffers");
		expect(result._knnPointIndices[3] == bruteForceKnn(cloud, knn.center, knn._k), "BIH tree KNN returns exact nearest-neighbor ordering");
		expect(result._knnQueries == 1, "BIH counts KNN queries");
		expect(result._metrics._totalQueries == queries.size(), "BIH summarizes query samples");

		PointGpu::Options bruteOptions = options;
		bruteOptions._knnBackend = "gpu_bruteforce_knn";
		const PointGpu::QueryResult bruteResult = index.query(std::vector<PointGpu::Query>{ knn }, bruteOptions);
		expect(bruteResult._knnBackend == "gpu_bruteforce_knn", "BIH can still use the explicit brute-force GPU KNN backend");
		expect(bruteResult._samples.size() == 1 && bruteResult._samples[0]._testedPoints == cloud.size(),
			"BIH brute-force KNN backend scans the GPU point buffer");
	}
}

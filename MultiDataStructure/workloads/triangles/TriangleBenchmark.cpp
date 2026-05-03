#include "../../stdafx.h"

#include "../../Camera.h"
#include "../../CudaHelper.h"
#include "../../ExternalBvh.h"
#include "../../Image.h"
#include "../../KdTree.h"
#include "../../MultiDataStructure.h"
#include "../../Octree.h"
#include "../../QuadTree.h"
#include "../../SceneContent.h"
#include "../../timeit.hpp"
#include "../../TriangleMesh.h"
#include "TriangleBenchmark.h"

namespace
{
	struct BenchmarkResult
	{
		std::string mode = "triangles";
		std::string scene;
		std::string schemaName;
		std::string backend;
		std::string imagePath;
		double buildTimeMs = 0.0;
		double queryTimeMs = 0.0;
		glm::uint numLevels = 0;
		glm::uint numNodes = 0;
		glm::uint numLeaves = 0;
		glm::uint numPrimitives = 0;
		float averageLeafPrimitives = 0.0f;
		float sahCost = 0.0f;
		size_t numRays = 0;
	};

	std::string levelToString(MultiDataStructure::DataStructureLevel level)
	{
		switch (level)
		{
		case MultiDataStructure::DataStructureLevel::QuadTreeNode:
			return "quadtree";
		case MultiDataStructure::DataStructureLevel::KDTreeNode:
			return "kdtree";
		case MultiDataStructure::DataStructureLevel::OctreeNode:
			return "octree";
		case MultiDataStructure::DataStructureLevel::BvhNode:
			return "bvh";
		default:
			return "unknown";
		}
	}

	std::string schemaNameFromConfig(const std::vector<MultiDataStructure::LevelConfig>& config)
	{
		std::ostringstream stream;
		for (size_t i = 0; i < config.size(); ++i)
		{
			if (i > 0)
				stream << "_";

			stream << levelToString(config[i]._levelType) << "x" << config[i]._numLevels;
		}

		return stream.str();
	}

	std::string jsonEscape(const std::string& value)
	{
		std::ostringstream stream;
		for (const char c : value)
		{
			switch (c)
			{
			case '\\':
				stream << "\\\\";
				break;
			case '"':
				stream << "\\\"";
				break;
			case '\n':
				stream << "\\n";
				break;
			case '\r':
				stream << "\\r";
				break;
			case '\t':
				stream << "\\t";
				break;
			default:
				stream << c;
				break;
			}
		}

		return stream.str();
	}

	std::string timestampUtc()
	{
		const std::time_t now = std::time(nullptr);
		std::tm utcTime{};

	#if defined(_WIN32)
		gmtime_s(&utcTime, &now);
	#else
		gmtime_r(&now, &utcTime);
	#endif

		std::ostringstream stream;
		stream << std::put_time(&utcTime, "%Y-%m-%dT%H:%M:%SZ");
		return stream.str();
	}

	double elapsedMilliseconds(std::chrono::high_resolution_clock::time_point start)
	{
		const auto end = std::chrono::high_resolution_clock::now();
		return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
	}

	template <typename TimeitStats>
	double meanTimeitMilliseconds(const TimeitStats& stats)
	{
		const long double meanNs = stats.mean / static_cast<long double>(stats._n_repetitions);
		return static_cast<double>(meanNs / 1000000.0L);
	}

	void writeBenchmarkJson(
		const std::string& filename,
		const TriangleBenchmark::Options& options,
		const glm::uvec2& windowSize,
		const std::vector<BenchmarkResult>& results)
	{
		const std::filesystem::path outputPath(filename);
		if (outputPath.has_parent_path())
			std::filesystem::create_directories(outputPath.parent_path());

		std::ofstream file(filename);
		if (!file.is_open())
			throw std::runtime_error("Unable to open JSON output path: " + filename);

		file << std::fixed << std::setprecision(4);
		file << "{\n";
		file << "  \"mode\": \"triangles\",\n";
		file << "  \"benchmark\": \"" << jsonEscape(options.benchmark) << "\",\n";
		file << "  \"scene\": \"" << jsonEscape(options.scenePath) << "\",\n";
		file << "  \"timestamp\": \"" << timestampUtc() << "\",\n";
		file << "  \"deterministic_settings\": {\n";
		file << "    \"random_seed\": " << TriangleBenchmark::DEFAULT_RANDOM_SEED << ",\n";
		file << "    \"window_width\": " << windowSize.x << ",\n";
		file << "    \"window_height\": " << windowSize.y << ",\n";
		file << "    \"samples_per_pixel\": 1,\n";
		file << "    \"num_rays\": " << static_cast<size_t>(windowSize.x) * static_cast<size_t>(windowSize.y) << "\n";
		file << "  },\n";
		file << "  \"results\": [\n";

		for (size_t i = 0; i < results.size(); ++i)
		{
			const BenchmarkResult& result = results[i];
			file << "    {\n";
			file << "      \"mode\": \"" << jsonEscape(result.mode) << "\",\n";
			file << "      \"scene\": \"" << jsonEscape(result.scene) << "\",\n";
			file << "      \"schema_name\": \"" << jsonEscape(result.schemaName) << "\",\n";
			file << "      \"backend\": \"" << jsonEscape(result.backend) << "\",\n";
			file << "      \"image_path\": \"" << jsonEscape(result.imagePath) << "\",\n";
			file << "      \"metrics\": {\n";
			file << "        \"build_time_ms\": " << result.buildTimeMs << ",\n";
			file << "        \"query_time_ms\": " << result.queryTimeMs << ",\n";
			file << "        \"num_levels\": " << result.numLevels << ",\n";
			file << "        \"num_nodes\": " << result.numNodes << ",\n";
			file << "        \"num_leaves\": " << result.numLeaves << ",\n";
			file << "        \"num_primitives\": " << result.numPrimitives << ",\n";
			file << "        \"average_leaf_primitives\": " << result.averageLeafPrimitives << ",\n";
			file << "        \"sah_cost\": " << result.sahCost << ",\n";
			file << "        \"num_rays\": " << result.numRays << "\n";
			file << "      }\n";
			file << "    }";
			file << (i + 1 == results.size() ? "\n" : ",\n");
		}

		file << "  ]\n";
		file << "}\n";
	}

	BenchmarkResult testMultiDS(
		const std::vector<MultiDataStructure::LevelConfig>& config,
		const std::string& schemaName,
		SceneContent* scene,
		const Node* bvhNodes,
		const std::vector<Ray>& rays,
		const glm::uvec2& windowSize)
	{
		std::vector<float> depth(rays.size());
		BenchmarkResult result;
		result.schemaName = schemaName;
		result.backend = "MultiDataStructure";
		result.imagePath = "output/depth_mds.png";
		result.numRays = rays.size();

		const auto buildStart = std::chrono::high_resolution_clock::now();
		MultiDataStructure multiDS(config);
		multiDS.build(bvhNodes, scene->getNumTriangles(), scene->_sceneAABB);

		multiDS.applyConfiguredCleanup();
		multiDS.checkSanity();

		result.buildTimeMs = elapsedMilliseconds(buildStart);
		const MultiDataStructure::Stats stats = multiDS.getStats();
		result.numLevels = stats.numLevels;
		result.numNodes = stats.numNodes;
		result.numLeaves = stats.numLeaves;
		result.numPrimitives = stats.numPrimitives;
		result.averageLeafPrimitives = stats.averageLeafPrimitives;

		std::cout << "MultiDS build time: " << result.buildTimeMs / 1000.0 << " seconds" << '\n';
		//multiDS.exportNodes("output/nodes.csv");

		std::cout << "\n-----------------------------------";
		{
			std::cout << "MultiDS tests..." << '\n';
			multiDS.printStats();
			const auto queryStats = timeit([&] {
				multiDS.resolveRayQueries(rays, depth, scene->getVertices(), scene->getIndices());
				});
			result.queryTimeMs = meanTimeitMilliseconds(queryStats);

			{
				Image image;
				image.fill(depth.data(), windowSize.x, windowSize.y, 1);
				image.normalize();
				image.save(result.imagePath);
			}
		}
		std::cout << "-----------------------------------\n";

		return result;
	}
}

int TriangleBenchmark::run(const Options& options)
{
	if (options.benchmark != "default")
		throw std::invalid_argument("Only --benchmark default is supported in the current triangle benchmark");

	CudaHelper::setDevice();

	NodeFactory::registerType<QuadTreeNode>(MultiDataStructure::DataStructureLevel::QuadTreeNode);
	NodeFactory::registerType<OctreeNode>(MultiDataStructure::DataStructureLevel::OctreeNode);
	NodeFactory::registerType<BvhNode>(MultiDataStructure::DataStructureLevel::BvhNode);
	NodeFactory::registerType<KdTreeNode>(MultiDataStructure::DataStructureLevel::KDTreeNode);

	std::filesystem::create_directories("output");

	glm::uvec2 windowSize(200, 200);
	Camera camera(windowSize.x, windowSize.y);
	camera.setFovX(glm::radians(60.0f));

	std::vector<Ray> rays;
	camera.setPosition(glm::vec3(0.1f, 0.0f, -15.0f));
	camera.setRaspect(windowSize.x, windowSize.y);
	camera.buildRays(rays, windowSize, 1);

	std::vector<float> depth(rays.size());

	std::unique_ptr<TriangleMesh> mesh = std::make_unique<TriangleMesh>();
	if (!mesh->load(options.scenePath))
		throw std::runtime_error("Unable to load scene: " + options.scenePath);

	mesh->moveGeometryToOrigin(glm::mat4(1.0f), 5.0f);

	std::unique_ptr<SceneContent> scene = std::make_unique<SceneContent>();
	scene->addNewModel(mesh.release());
	scene->buildScenario();
	std::unique_ptr<Node[]> bvhNodes(scene->getBvhNodesExplicitly());
	if (!bvhNodes)
		throw std::runtime_error("Unable to download BVH nodes from the scene");

	std::cout << scene->getNumTriangles() << '\n';

	std::vector<BenchmarkResult> results;

	const std::vector<MultiDataStructure::LevelConfig> hybridConfig = {
		{ ._levelType = MultiDataStructure::DataStructureLevel::OctreeNode, ._numLevels = 1 },
		{ ._levelType = MultiDataStructure::DataStructureLevel::BvhNode, ._numLevels = 3 },
		{ ._levelType = MultiDataStructure::DataStructureLevel::OctreeNode, ._numLevels = 2 },
		{ ._levelType = MultiDataStructure::DataStructureLevel::BvhNode, ._numLevels = 10 },
	};
	results.push_back(testMultiDS(hybridConfig, schemaNameFromConfig(hybridConfig), scene.get(), bvhNodes.get(), rays, windowSize));

	const std::vector<MultiDataStructure::LevelConfig> bvhConfig = {
		{ ._levelType = MultiDataStructure::DataStructureLevel::BvhNode, ._numLevels = 20 },
	};
	results.push_back(testMultiDS(bvhConfig, schemaNameFromConfig(bvhConfig), scene.get(), bvhNodes.get(), rays, windowSize));

	const auto tinyBvhBuildStart = std::chrono::high_resolution_clock::now();
	ExternalBvh tinyBvh(
		scene->getVertices(),
		scene->getIndices(),
		static_cast<glm::uint>(scene->getNumVertices()),
		static_cast<glm::uint>(scene->getNumTriangles()));
	BenchmarkResult tinyBvhResult;
	tinyBvhResult.scene = options.scenePath;
	tinyBvhResult.schemaName = "tinybvh_hq";
	tinyBvhResult.backend = "TinyBVH";
	tinyBvhResult.imagePath = "output/depth_ebvh.png";
	tinyBvhResult.buildTimeMs = elapsedMilliseconds(tinyBvhBuildStart);
	tinyBvhResult.numLeaves = tinyBvh.getNumLeaves();
	tinyBvhResult.numNodes = tinyBvh.getNumNodes();
	tinyBvhResult.numPrimitives = tinyBvh.getNumPrimitives();
	tinyBvhResult.sahCost = tinyBvh.getSAHCost();
	tinyBvhResult.numRays = rays.size();
	std::cout << "BVH build time: " << tinyBvhResult.buildTimeMs / 1000.0 << " seconds" << '\n';

	std::cout << "\n-----------------------------------";
	{
		std::cout << "TinyBVH tests..." << '\n';
		tinyBvh.printStats();
		const auto queryStats = timeit([&] {
			tinyBvh.resolveRayQueries(rays, depth);
			});
		tinyBvhResult.queryTimeMs = meanTimeitMilliseconds(queryStats);

		{
			Image image;
			image.fill(depth.data(), windowSize.x, windowSize.y, 1);
			image.normalize();
			image.save(tinyBvhResult.imagePath);
		}
	}
	std::cout << "-----------------------------------\n";
	results.push_back(tinyBvhResult);

	for (auto& result : results)
		result.scene = options.scenePath;

	if (!options.outputPath.empty())
	{
		writeBenchmarkJson(options.outputPath, options, windowSize, results);
		std::cout << "Benchmark JSON written to " << options.outputPath << '\n';
	}

	if (options.pauseAtEnd)
		system("pause");

	return 0;
}

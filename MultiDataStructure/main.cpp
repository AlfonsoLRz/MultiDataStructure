#define TINYBVH_IMPLEMENTATION

#include "stdafx.h"

#include "Camera.h"
#include "ChronoUtilities.h"
#include "ExternalBvh.h"
#include "Image.h"
#include "KdTree.h"
#include "MultiDataStructure.h"
#include "Octree.h"
#include "QuadTree.h"
#include "SceneContent.h"
#include "timeit.hpp"
#include "TriangleMesh.h"

int main(int argc, char* argv)
{
    CudaHelper::setDevice();

	NodeFactory::registerType<QuadTreeNode>(MultiDataStructure::DataStructureLevel::QuadTreeNode);
	NodeFactory::registerType<OctreeNode>(MultiDataStructure::DataStructureLevel::OctreeNode);
	NodeFactory::registerType<BvhNode>(MultiDataStructure::DataStructureLevel::BvhNode);
	NodeFactory::registerType<KdTreeNode>(MultiDataStructure::DataStructureLevel::KDTreeNode);

	glm::uvec2 windowSize(200, 200);
	Camera camera (windowSize.x, windowSize.y);
	camera.setFovX(glm::radians(60.0f));

	std::vector<Ray> rays;
	camera.setPosition(glm::vec3(0.1f, 0.0f, -15.0f));
	camera.setRaspect(windowSize.x, windowSize.y);
	camera.buildRays(rays, windowSize, 1);

	std::vector<float> depth(rays.size());

    TriangleMesh* mesh = new TriangleMesh();
    mesh->load("C:/Datasets/models/CornellBox/CornellKnightDragon.obj");
    mesh->moveGeometryToOrigin(glm::mat4(1.0f), 5.0f);

	SceneContent* scene = new SceneContent();
	scene->addNewModel(mesh);
	scene->buildScenario();
	Node* bvhNodes = scene->getBvhNodesExplicitly();

	std::cout << scene->getNumTriangles() << '\n';

	ChronoUtilities::initChrono();
	MultiDataStructure multiDS({
		{ ._levelType = MultiDataStructure::DataStructureLevel::QuadTreeNode, ._numLevels = 1 },
		{ ._levelType = MultiDataStructure::DataStructureLevel::OctreeNode, ._numLevels = 4 },
		{ ._levelType = MultiDataStructure::DataStructureLevel::BvhNode, ._numLevels = 8 },
	});
	multiDS.build(bvhNodes, scene->getNumTriangles(), mesh->getAABB());

	glm::uint deletedNodes;
	do
	{
		multiDS.removeEmptyNodes(deletedNodes);
	}
	while (deletedNodes > 0);
	multiDS.collapseNodes();
	multiDS.checkSanity();

	std::cout << "MultiDS build time: " << ChronoUtilities::getDuration(ChronoUtilities::SECONDS) << " seconds" << '\n';
    //multiDS.exportNodes("output/nodes.csv");

	ChronoUtilities::initChrono();
	ExternalBvh tinyBvh(scene->getVertices(), scene->getIndices(), scene->getNumVertices(), scene->getNumTriangles());
	std::cout << "BVH build time: " << ChronoUtilities::getDuration(ChronoUtilities::SECONDS) << " seconds" << '\n';

	// Test 1
	std::cout << "\n-----------------------------------";
	{
		std::cout << "MultiDS tests..." << '\n';
		multiDS.printStats();
		timeit([&] {
			multiDS.resolveRayQueries(rays, depth, scene->getVertices(), scene->getIndices());
		});

		{
			Image image;
			image.fill(depth.data(), windowSize.x, windowSize.y, 1);
			image.normalize();
			image.save("output/depth_mds.png");
		}
	}
	std::cout << "-----------------------------------\n";

	// Test 2
	std::cout << "\n-----------------------------------";
	{
		std::cout << "TinyBVH tests..." << '\n';
		tinyBvh.printStats();
		timeit([&] {
			tinyBvh.resolveRayQueries(rays, depth);
		});

		{
			Image image;
			image.fill(depth.data(), windowSize.x, windowSize.y, 1);
			image.normalize();
			image.save("output/depth_ebvh.png");
		}
	}
	std::cout << "-----------------------------------\n";

	delete[] bvhNodes;

    // - Esta llamada es para impedir que la consola se cierre inmediatamente tras la
    // ejecución y poder leer los mensajes. Se puede usar también getChar();
    system("pause");
}

#define TINYBVH_IMPLEMENTATION

#include "stdafx.h"

#include "Camera.h"
#include "ChronoUtilities.h"
#include "ExternalBvh.h"
#include "GeometryUtils.h"
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
	camera.setFovX(glm::radians(40.0f));

	std::vector<RayGPU> rays;
	camera.setPosition(glm::vec3(0.1f, 0.0f, -15.0f));
	camera.setRaspect(windowSize.x, windowSize.y);
	camera.buildRays(rays, windowSize, 1);

	std::vector<float> depth(rays.size());
    TriangleMesh* mesh = new TriangleMesh();
    mesh->load("C:/Datasets/models/CornellBox/CornellKnight.obj");
    mesh->moveGeometryToOrigin(glm::mat4(1.0f), 5.0f);

	SceneContent* scene = new SceneContent();
	scene->addNewModel(mesh);
	scene->buildScenario();

	VertexGPU* vertices = scene->getVertices();
	glm::u32* indices = scene->getIndices();
	size_t numVertices = scene->getNumVertices();
	size_t numTriangles = scene->getNumTriangles();

	Node* nodes = GeometryUtils::fillNodeBuffer(vertices, indices, numTriangles * 3);
	GeometryUtils::reorderNodes(nodes, static_cast<glm::uint>(numTriangles), GeometryUtils::SortMethod::SortMethodMorton);

	glm::uint numNodes = static_cast<glm::uint>(numTriangles);
	Bvh bvh(nodes, numNodes);
	bvh.build(numNodes);
	//bvh.exportNodes("output/nodes.csv");

	// Test 1
	std::cout << "\n-----------------------------------";
	{
		std::cout << "MultiDS tests..." << '\n';
		timeit([&] {
			bvh.resolveRayQueries(rays, depth, scene->getVertices(), scene->getIndices());
		});

		{
			Image image;
			image.fill(depth.data(), windowSize.x, windowSize.y, 1);
			image.normalize();
			image.save("output/depth_mds.png");
		}
	}
	std::cout << "-----------------------------------\n";


	//MultiDataStructure multiDS({
	//	{ ._levelType = MultiDataStructure::DataStructureLevel::QuadTreeNode, ._numLevels = 1 },
	//	{ ._levelType = MultiDataStructure::DataStructureLevel::OctreeNode, ._numLevels = 4 },
	//	{ ._levelType = MultiDataStructure::DataStructureLevel::BvhNode, ._numLevels = 8 },
	//});
	//multiDS.build(bvhNodes, scene->getNumTriangles(), mesh->getAABB());

	//glm::uint deletedNodes;
	//do
	//{
	//	multiDS.removeEmptyNodes(deletedNodes);
	//}
	//while (deletedNodes > 0);
	//multiDS.collapseNodes();
	//multiDS.checkSanity();

 //   //multiDS.exportNodes("output/nodes.csv");

	//ChronoUtilities::initChrono();


	//// Test 1
	//std::cout << "\n-----------------------------------";
	//{
	//	std::cout << "MultiDS tests..." << '\n';
	//	multiDS.printStats();
	//	timeit([&] {
	//		multiDS.resolveRayQueries(rays, depth, scene->getVertices(), scene->getIndices());
	//	});

	//	{
	//		Image image;
	//		image.fill(depth.data(), windowSize.x, windowSize.y, 1);
	//		image.normalize();
	//		image.save("output/depth_mds.png");
	//	}
	//}
	//std::cout << "-----------------------------------\n";

	ChronoUtilities::initChrono();
	ExternalBvh tinyBvh(scene->getVertices(), scene->getIndices(), scene->getNumVertices(), scene->getNumTriangles());
	std::cout << "BVH build time: " << ChronoUtilities::getDuration(ChronoUtilities::SECONDS) << " seconds" << '\n';

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

	//delete[] bvhNodes;

    // - Esta llamada es para impedir que la consola se cierre inmediatamente tras la
    // ejecución y poder leer los mensajes. Se puede usar también getChar();
    system("pause");
}

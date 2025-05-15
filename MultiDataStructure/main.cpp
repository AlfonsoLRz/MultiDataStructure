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
#include "TriangleMesh.h"

int main(int argc, char* argv)
{
    CudaHelper::setDevice();

	NodeFactory::registerType<QuadTreeNode>(MultiDataStructure::DataStructureLevel::QuadTreeNode);
	NodeFactory::registerType<OctreeNode>(MultiDataStructure::DataStructureLevel::OctreeNode);
	NodeFactory::registerType<BvhNode>(MultiDataStructure::DataStructureLevel::BvhNode);
	NodeFactory::registerType<KdTreeNode>(MultiDataStructure::DataStructureLevel::KDTreeNode);

	glm::uvec2 windowSize(800, 600);
	Camera camera (windowSize.x, windowSize.y);
	camera.setFovX(60.0f);

    TriangleMesh* mesh = new TriangleMesh();
    mesh->load("C:/Datasets/models/CornellBox/CornellKnightDragon.obj");
    mesh->moveGeometryToOrigin(glm::mat4(1.0f), 5.0f);

	SceneContent* scene = new SceneContent();
	scene->addNewModel(mesh);
	scene->buildScenario();

	Node* bvhNodes = scene->getBvhNodesExplicitly();

	ChronoUtilities::initChrono();
	MultiDataStructure multiDS({
		{._levelType = MultiDataStructure::DataStructureLevel::QuadTreeNode, ._numLevels = 4 },
		{._levelType = MultiDataStructure::DataStructureLevel::OctreeNode, ._numLevels = 4 },
		{._levelType = MultiDataStructure::DataStructureLevel::BvhNode, ._numLevels = 8 } }
		);
	multiDS.build({
		, 
		bvhNodes, scene->getNumTriangles(), mesh->getAABB(), true);


	multiDS.build(MultiDataStructure::QuadTreeNode, 4, bvhNodes, scene->getNumTriangles(), mesh->getAABB());
	multiDS.check(MultiDataStructure::OctreeNode, 4);
	multiDS.check(MultiDataStructure::BvhNode, 8);
	std::cout << "MultiDS build time: " << ChronoUtilities::getDuration(ChronoUtilities::SECONDS) << " seconds" << std::endl;
    multiDS.exportNodes("output/nodes.csv");

	ChronoUtilities::initChrono();
	ExternalBvh tinyBvh(scene->getVertices(), scene->getIndices(), scene->getNumVertices(), scene->getNumTriangles());
	std::cout << "BVH build time: " << ChronoUtilities::getDuration(ChronoUtilities::SECONDS) << " seconds" << std::endl;

	std::vector<Ray> rays;
	std::vector<float> depth;
	camera.setRaspect(windowSize.x, windowSize.y);
	camera.buildRays(rays, windowSize, 1);

	// Test 1
	ChronoUtilities::initChrono();
	multiDS.resolveRayQueries(rays, depth, scene->getVertices(), scene->getIndices());
	std::cout << "MultiDS resolve time: " << ChronoUtilities::getDuration(ChronoUtilities::NANOSECONDS) << " nanoseconds" << std::endl;

	//multiDS.resolveRayQueriesBruteForce(rays, depth, scene->getVertices(), scene->getIndices(), bvhNodes, scene->getNumTriangles() * 2 - 1);

	// Test 2
	ChronoUtilities::initChrono();
	tinyBvh.resolveRayQueries(rays, depth);
	std::cout << "TinyBVH resolve time: " << ChronoUtilities::getDuration(ChronoUtilities::NANOSECONDS) << " nanoseconds" << std::endl;

	delete[] bvhNodes;

	std::vector<glm::vec3> directions;
	for (const auto& ray : rays)
	{
		glm::vec3 direction = ray._direction;
		directions.push_back(glm::normalize(direction));
	}

	// Save depth map
	Image image;
	image.fill(depth.data(), windowSize.x, windowSize.y, 1);
	image.normalize();
	image.save("output/depth.png");

    // - Esta llamada es para impedir que la consola se cierre inmediatamente tras la
    // ejecución y poder leer los mensajes. Se puede usar también getChar();
    system("pause");
}

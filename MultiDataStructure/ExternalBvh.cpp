#include "stdafx.h"
#include "ExternalBvh.h"

//

ExternalBvh::ExternalBvh(const VertexGPU* vertices, const glm::u32* indices, glm::uint numVertices, glm::uint numTriangles)
{
	tinybvh::bvhvec4* bvhVertices = new tinybvh::bvhvec4[numVertices];
	for (glm::uint i = 0; i < numVertices; i++)
	{
		bvhVertices[i].x = vertices[i]._position.x;
		bvhVertices[i].y = vertices[i]._position.y;
		bvhVertices[i].z = vertices[i]._position.z;
		bvhVertices[i].w = 1.0f;
	}

	_bvh.BuildHQ(bvhVertices, indices, numTriangles);

	//delete[] bvhVertices;
}

ExternalBvh::~ExternalBvh() = default;

void ExternalBvh::printStats()
{
	std::cout << "BVH stats:" << std::endl;
	std::cout << "  - Number of leaves: " << _bvh.LeafCount() << std::endl;
	std::cout << "  - Number of triangles: " << _bvh.PrimCount() << std::endl;
	std::cout << "  - Number of nodes: " << _bvh.NodeCount() << std::endl;
	std::cout << "  - SAH: " << _bvh.SAHCost(0) << std::endl;
}

void ExternalBvh::resolveRayQueries(const std::vector<Ray>& rays, std::vector<float>& depth) const
{
	depth.resize(rays.size());

	#pragma omp parallel for
	for (int rayIdx = 0; rayIdx < static_cast<int>(rays.size()); ++rayIdx)
	{
		tinybvh::bvhvec3 O(rays[rayIdx]._origin.x, rays[rayIdx]._origin.y, rays[rayIdx]._origin.z);
		tinybvh::bvhvec3 D(rays[rayIdx]._direction.x, rays[rayIdx]._direction.y, rays[rayIdx]._direction.z);
		tinybvh::Ray bvhRay(O, D);
		int steps = _bvh.Intersect(bvhRay);

		if (bvhRay.hit.t < BVH_FAR)
			depth[rayIdx] = bvhRay.hit.t;
		else
			depth[rayIdx] = .0f;
	}
}

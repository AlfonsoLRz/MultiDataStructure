#pragma once

#include "AABB.h"
#include "GPUStructs.h"

class Bvh
{
protected:
	Node*		_triangles;
	glm::uint	_numTriangles;
	Node*		_nodes;

protected:
	void resolveRayQueries(
		const Node& node, const RayGPU& ray,
		const VertexGPU* vertices, const glm::u32* indices,
		HitInfo& hitInfo
	) const;

public:
	Bvh(Node* triangles, glm::uint numTriangles);
	~Bvh();

	void build(glm::uint& numNodes);
	void resolveRayQueries(
		const std::vector<RayGPU>& rays, std::vector<float>& depth,
		const VertexGPU* vertices, const glm::u32* indices
	) const;
	void resolveRayQueriesBruteForce(
		const std::vector<RayGPU>& rays, std::vector<float>& depth,
		const VertexGPU* vertices, const glm::u32* indices
	) const;

	bool exportNodes(const std::string& filename) const;
};
#pragma once

#include "GPUStructs.h"
#include "Ray.h"
#include "tiny_bvh.h"

class ExternalBvh
{
private:
	tinybvh::BVH _bvh;
	std::vector<tinybvh::bvhvec4> _bvhVertices;

public:
	ExternalBvh(const VertexGPU* vertices, const glm::u32* indices, glm::uint numVertices, glm::uint numTriangle);
	virtual ~ExternalBvh();

	glm::uint getNumLeaves() const;
	glm::uint getNumNodes() const;
	glm::uint getNumPrimitives() const;
	float getSAHCost() const;
	void printStats();
	void resolveRayQueries(const std::vector<Ray>& rays, std::vector<float>& depth) const;
};


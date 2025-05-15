#pragma once

#include "GPUStructs.h"
#include "Ray.h"
#include "tiny_bvh.h"

class ExternalBvh
{
private:
	tinybvh::BVH _bvh;

public:
	ExternalBvh(const VertexGPU* vertices, const glm::u32* indices, glm::uint numVertices, glm::uint numTriangle);
	virtual ~ExternalBvh();

	void resolveRayQueries(const std::vector<Ray>& rays, std::vector<float>& depth) const;
};


#include "stdafx.h"
#include "cuda_globals.cuh"

__device__ void calculateBounds(const VertexGPU* vertices, const glm::u32* indices, const glm::u32 index, glm::vec3* maxPoint, glm::vec3* minPoint)
{
	*maxPoint = glm::max(vertices[indices[index + 0]]._position, glm::max(vertices[indices[index + 1]]._position, vertices[indices[index + 2]]._position));
	*minPoint = glm::min(vertices[indices[index + 0]]._position, glm::min(vertices[indices[index + 1]]._position, vertices[indices[index + 2]]._position));
}


#pragma once

#include "GPUStructs.h"

__device__ void calculateBounds(const VertexGPU* vertices, const glm::u32* indices, const glm::u32 index, glm::vec3* maxPoint, glm::vec3* minPoint);
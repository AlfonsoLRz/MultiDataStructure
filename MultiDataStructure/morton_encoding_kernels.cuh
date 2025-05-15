#pragma once

#include "GPUStructs.h"

__device__ glm::u32 expandBits(glm::u32 v);

// We computes Morton codes for points located within the unit cube [0, 1].
__device__ glm::vec3 normalize(glm::vec3 position, const glm::vec3 aabbMax, const glm::vec3 aabbMin);

// Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1]. 2^10 = 1024, 10 bits for each coordinate.
__device__ glm::u32 morton3D(glm::vec3 position, const glm::vec3 aabbMax, const glm::vec3 aabbMin);

__global__ void computeMortonCodesCuda(const VertexGPU* vertices, const glm::u32* indices, glm::u32 numIndices, glm::u32* mortonCodes, const glm::vec3 aabbMax, const glm::vec3 aabbMin);


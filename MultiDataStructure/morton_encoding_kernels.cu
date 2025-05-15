#include "stdafx.h"
#include "morton_encoding_kernels.cuh"
#include "cuda_globals.cuh"

__device__ glm::u32 expandBits(glm::u32 v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;

	return v;
}

__device__ glm::vec3 normalize(const glm::vec3 position, const glm::vec3 aabbMax, const glm::vec3 aabbMin)
{
	return (position - aabbMin) / (aabbMax - aabbMin);
}

__device__ glm::u32 morton3D(const glm::vec3 position, const glm::vec3 aabbMax, const glm::vec3 aabbMin)
{
	const glm::vec3 normVertex = normalize(position, aabbMax, aabbMin);
	const float x = normVertex.x * 1024.0f;
	const float y = normVertex.y * 1024.0f;
	const float z = normVertex.z * 1024.0f;

	glm::u32 xx = expandBits(glm::u32(x));
	glm::u32 yy = expandBits(glm::u32(y));
	glm::u32 zz = expandBits(glm::u32(z));

	return xx * 4 + yy * 2 + zz;
}

__global__ void computeMortonCodesCuda(const VertexGPU* vertices, const glm::u32* indices, glm::u32 numIndices, glm::u32* mortonCodes, const glm::vec3 aabbMax, const glm::vec3 aabbMin)
{
	const glm::u32 index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= numIndices / 3) 
		return;

	glm::vec3 minPoint, maxPoint;
	calculateBounds(vertices, indices, index * 3, &maxPoint, &minPoint);
	mortonCodes[index] = morton3D((minPoint + maxPoint) / 2.0f, aabbMax, aabbMin);
}
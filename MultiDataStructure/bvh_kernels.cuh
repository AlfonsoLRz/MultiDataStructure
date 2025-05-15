#pragma once

#include "GPUStructs.h"

#define SEARCH_RADIUS 128

__device__ glm::u32 getMesh(const glm::u32 triangleIndex, const MeshGPU* meshes, const glm::u32 numMeshes);

__global__ void buildClusterBufferCuda(
	const VertexGPU* vertices, const glm::u32* indices, const MeshGPU* meshes, 
	const glm::uint32* sortedIndices, 
	Node* cluster, Node* tempCluster, 
	glm::uint32 numTriangles, glm::uint32 numMeshes
);

//

__device__ float heuristic(const Node* cluster, const glm::u32 index1, const glm::u32 index2);

__global__ void findBestNeighbors(const Node* cluster, glm::u32* neighbor, const glm::u32 numTriangles);

//

__device__ void buildCluster(Node* tempCluster, Node* cluster, glm::u32* currentPosition, glm::u32* numNodes, const glm::u32 index1, const glm::u32 index2);

__global__ void mergeClusters(
	Node* tempCluster, Node* cluster,
	const glm::u32* neighbor,
	glm::u32* validCluster, glm::u32* mergedCluster, glm::u32* prefixScan, glm::u32* currentPosition,
	glm::u32* numNodes, const glm::u32 numTriangles
);

//

__global__ void reallocateClusters(
	const Node* inCluster, Node* outCluster, 
	const glm::u32* validCluster, 
	const glm::u32* prefixScan, 
	const glm::u32* inPosition, glm::u32* outPosition,
	const glm::u32 numTriangles);

//

__global__ void endLoop(glm::u32* arraySizeCount, const glm::u32* prefixScan, const glm::u32* validCluster, const glm::u32 numTriangles);
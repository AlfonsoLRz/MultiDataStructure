#include "stdafx.h"
#include "bvh_kernels.cuh"

#include "cuda_globals.cuh"

//

__device__ glm::u32 getMesh(glm::u32 triangleIndex, const MeshGPU* meshes, glm::u32 numMeshes)
{
	for (glm::u32 i = 0; i < numMeshes; ++i)
	{
		if (triangleIndex >= meshes[i]._startIndex && triangleIndex < meshes[i]._startIndex + meshes[i]._length)
			return i;
	}

	return 0;
}

__global__ void buildClusterBufferCuda(const VertexGPU* vertices, const glm::u32* indices, const MeshGPU* meshes, const glm::uint32* sortedIndices, Node* cluster, Node* tempCluster, glm::uint32 numTriangles, glm::uint32 numMeshes)
{
	const glm::u32 index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= numTriangles)
		return;

	// Indices buffer (sorted morton codes) links to a face different than index
	const glm::u32 triangleIndex = sortedIndices[index];

	glm::vec3 minPoint, maxPoint;
	calculateBounds(vertices, indices, triangleIndex * 3, &maxPoint, &minPoint);

	cluster[index]._minPoint = glm::vec4(minPoint, .0f);
	cluster[index]._maxPoint = glm::vec4(maxPoint, .0f);
	cluster[index]._prevIndex1 = cluster[index]._prevIndex2 = INT_MAX;
	cluster[index]._triangleIndex = triangleIndex;
	cluster[index]._meshIndex = getMesh(triangleIndex * 3, meshes, numMeshes);
	cluster[index]._numTriangles = 1;
	tempCluster[index] = cluster[index];
}

//

__device__ float heuristic(const Node* cluster, const glm::u32 index1, const glm::u32 index2)
{
	const glm::vec3 minPoint = glm::min(cluster[index1]._minPoint, cluster[index2]._minPoint);
	const glm::vec3 maxPoint = glm::max(cluster[index1]._maxPoint, cluster[index2]._maxPoint);
	const glm::vec3 length = maxPoint - minPoint;

	return 2.0f * length.x * length.y + 2.0f * length.z * length.y + 2.0f * length.x * length.z;
}

__global__ void findBestNeighbors(const Node* cluster, glm::u32* neighbor, const glm::u32 numTriangles)
{
	const glm::u32 index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= numTriangles)
		return;

	const glm::u32 lowerIndex = glm::max(int(index - SEARCH_RADIUS), 0), upperIndex = glm::min(index + SEARCH_RADIUS, numTriangles - 1);
	glm::u32 currentIndex = lowerIndex;
	float distance, minDistance = INT_MAX;

	while (currentIndex < index)
	{
		distance = heuristic(cluster, index, currentIndex);
		if (distance < minDistance)
		{
			neighbor[index] = currentIndex;
			minDistance = distance;
		}

		++currentIndex;
	}

	++currentIndex;

	while (currentIndex <= upperIndex)
	{
		distance = heuristic(cluster, index, currentIndex);
		if (distance < minDistance)
		{
			neighbor[index] = currentIndex;
			minDistance = distance;
		}

		++currentIndex;
	}
}

//

__device__ void buildCluster(Node* tempCluster, Node* cluster, glm::u32* currentPosition, glm::u32* numNodes, const glm::u32 index1, const glm::u32 index2)
{
	// Thread which manages index1 must call this method, then index1 is treated as the min index
	tempCluster[index1]._minPoint = min(tempCluster[index1]._minPoint, tempCluster[index2]._minPoint);
	tempCluster[index1]._maxPoint = max(tempCluster[index1]._maxPoint, tempCluster[index2]._maxPoint);
	tempCluster[index1]._prevIndex1 = currentPosition[index1];
	tempCluster[index1]._prevIndex2 = currentPosition[index2];
	tempCluster[index1]._triangleIndex = INT_MAX;
	tempCluster[index1]._numTriangles = tempCluster[index1]._numTriangles + tempCluster[index2]._numTriangles;

	// Compact buffer operation
	glm::u32 newPosition = atomicAdd(&(numNodes[0]), 1);
	cluster[newPosition] = tempCluster[index1];
	currentPosition[index1] = newPosition;
}

__global__ void mergeClusters(
	Node* tempCluster, Node* cluster, 
	const glm::u32* neighbor, 
	glm::u32* validCluster, glm::u32* mergedCluster, glm::u32* prefixScan, glm::u32* currentPosition, 
	glm::u32* numNodes, const glm::u32 numTriangles)
{
	const glm::u32 index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= numTriangles)
		return;

	if (neighbor[neighbor[index]] == index)
	{
		if (index < neighbor[index])
		{
			buildCluster(tempCluster, cluster, currentPosition, numNodes, index, neighbor[index]);
			validCluster[index] = prefixScan[index] = 1;
			validCluster[neighbor[index]] = prefixScan[neighbor[index]] = 0;
			mergedCluster[index] = 1;
		}
		else
		{
			mergedCluster[index] = 0;
			return;
		}
	}
	else
	{
		validCluster[index] = prefixScan[index] = 1;				// By default it's valid
		mergedCluster[index] = 0;
	}
}

__global__ void reallocateClusters(const Node* inCluster, Node* outCluster, const glm::u32* validCluster, const glm::u32* prefixScan, const glm::u32* inPosition, glm::u32* outPosition, const glm::u32 numTriangles)
{
	const glm::u32 index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= numTriangles)
		return;

	if (validCluster[index] == 1)
	{
		outCluster[prefixScan[index]] = inCluster[index];
		outPosition[prefixScan[index]] = inPosition[index];
	}
}

__global__ void endLoop(glm::u32* arraySizeCount, const glm::u32* prefixScan, const glm::u32* validCluster, const glm::u32 numTriangles)
{
	arraySizeCount[0] = prefixScan[numTriangles - 1] + validCluster[numTriangles - 1];
}




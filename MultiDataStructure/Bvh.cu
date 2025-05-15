// ReSharper disable CppExpressionWithoutSideEffects
#include "stdafx.h"
#include "Bvh.h"

#include "CudaHelper.h"

#include <cub/cub.cuh>
#include "bvh_kernels.cuh"
#include "morton_encoding_kernels.cuh"

//

Bvh::Bvh() :
	_vertexBufferGPU(nullptr), _indexBufferGPU(nullptr), _meshBufferGPU(nullptr), _clusterBufferGPU(nullptr),
	_numIndices(0), _numMeshes(0)
{
}

Bvh::~Bvh()
{
}

void Bvh::build()
{
	glm::u32* mortonCodesBufferGPU = this->computeMortonCodes();
	glm::u32* sortedIndicesBufferGPU = this->sortMortonCodes(mortonCodesBufferGPU);
	Node* tempClusterBufferGPU = this->buildClusterBuffer(sortedIndicesBufferGPU);
	this->buildTree(tempClusterBufferGPU);
}

void Bvh::initialize(VertexGPU* vertexBuffer, glm::u32* indexBuffer, MeshGPU* meshBuffer, const AABB& aabb, glm::uint numIndices, glm::uint numMeshes)
{
	_vertexBufferGPU = vertexBuffer;
	_indexBufferGPU = indexBuffer;
	_meshBufferGPU = meshBuffer;
	_aabb = aabb;
	_numIndices = numIndices;
	_numMeshes = numMeshes;
}

//

Node* Bvh::buildClusterBuffer(glm::u32* sortedFacesBufferGPU)
{
	const size_t numTriangles = _numIndices / 3;
	const size_t clusterSize = numTriangles * 2 - 1;
	const size_t blockSize = CudaHelper::getMaxThreadsBlock(), numBlocks = CudaHelper::getNumBlocks(numTriangles, blockSize);

	_clusterBufferGPU = nullptr;
	CudaHelper::initializeBufferGPU(_clusterBufferGPU, clusterSize);

	Node* tempClusterBufferGPU = nullptr;
	CudaHelper::initializeBufferGPU(tempClusterBufferGPU, numTriangles);

	buildClusterBufferCuda<<<numBlocks, blockSize>>>(
		_vertexBufferGPU, _indexBufferGPU, _meshBufferGPU, sortedFacesBufferGPU, _clusterBufferGPU, tempClusterBufferGPU, numTriangles, _numMeshes);

	CudaHelper::free(sortedFacesBufferGPU);

	//std::vector<Node> clusterBuffer;
	//clusterBuffer.resize(clusterSize);
	//CudaHelper::downloadBufferGPU(_clusterBufferGPU, clusterBuffer.data(), clusterSize);

	return tempClusterBufferGPU;
}

inline void Bvh::buildTree(Node* tempClusterBufferGPU)
{
	// Compute shader execution data: groups and iteration control
	glm::u32 arraySize = _numIndices / 3;
	const size_t maxBlockSize = CudaHelper::getMaxThreadsBlock();

	// Compact cluster buffer support
	glm::u32* currentPosBufferOut = new glm::u32[arraySize];
	std::iota(currentPosBufferOut, currentPosBufferOut + arraySize, 0);

	Node* coutBuffer = tempClusterBufferGPU;		// Swapped during loop => not const
	Node* cinBuffer = nullptr;
	CudaHelper::initializeBufferGPU(cinBuffer, arraySize);

	glm::u32* inCurrentPosition = nullptr;
	CudaHelper::initializeBufferGPU(inCurrentPosition, arraySize);		// Position of compact buffer where a cluster is saved
	glm::u32* outCurrentPosition = nullptr;
	CudaHelper::initializeBufferGPU(outCurrentPosition, arraySize, currentPosBufferOut);

	glm::u32* neighborIndex = nullptr, *prefixScan = nullptr, *validCluster = nullptr, *mergedCluster = nullptr, *numNodesCount = nullptr, *arraySizeCount = nullptr;
	CudaHelper::initializeBufferGPU(neighborIndex, arraySize);					// Nearest neighbor search
	CudaHelper::initializeBufferGPU(prefixScan, arraySize);						// Final position of each valid cluster for the next loop iteration
	CudaHelper::initializeBufferGPU(validCluster, arraySize);						// Clusters which takes part of next loop iteration
	CudaHelper::initializeBufferGPU(mergedCluster, arraySize);					// A merged cluster is always valid, but the opposite situation is not fitting
	CudaHelper::initializeBufferGPU(numNodesCount, 1, &arraySize);		// Number of currently added nodes, which increases as the clusters are merged
	CudaHelper::initializeBufferGPU(arraySizeCount, 1);

	while (arraySize > 1)
	{
		size_t numBlocks	= CudaHelper::getNumBlocks(arraySize, maxBlockSize);
		size_t startThreads	= static_cast<size_t>(std::ceil(static_cast<float>(arraySize) / 2.0f));
		size_t numExec		= static_cast<size_t>(std::ceil(std::log2(arraySize)));

		// Thread sizes are repeated on reduce and sweep down phases
		std::vector<size_t> threadCount;
		threadCount.reserve(numExec);
		threadCount.push_back(startThreads);

		std::swap(coutBuffer, cinBuffer);
		std::swap(inCurrentPosition, outCurrentPosition);

		findBestNeighbors<<<numBlocks, maxBlockSize>>>(cinBuffer, neighborIndex, arraySize);
		mergeClusters<<<numBlocks, maxBlockSize>>>(cinBuffer, _clusterBufferGPU, neighborIndex, validCluster, mergedCluster, prefixScan, inCurrentPosition, numNodesCount, arraySize);

		// Prefix scan
		glm::u32* tempStorage = nullptr;								
		size_t tempStorageBytes = 0;

		cub::DeviceScan::ExclusiveSum(tempStorage, tempStorageBytes, validCluster, prefixScan, arraySize);
		CudaHelper::initializeBufferGPU(tempStorage, tempStorageBytes);
		cub::DeviceScan::ExclusiveSum(tempStorage, tempStorageBytes, validCluster, prefixScan, arraySize);

		reallocateClusters<<<numBlocks, maxBlockSize>>>(cinBuffer, coutBuffer, validCluster, prefixScan, inCurrentPosition, outCurrentPosition, arraySize);
		endLoop<<<1, 1>>> (arraySizeCount, prefixScan, validCluster, arraySize);

		CudaHelper::downloadBufferGPU(arraySizeCount, &arraySize, 1);
		CudaHelper::free(tempStorage);
	}

	CudaHelper::free(cinBuffer);
	CudaHelper::free(coutBuffer);
	CudaHelper::free(inCurrentPosition);
	CudaHelper::free(outCurrentPosition);
	CudaHelper::free(neighborIndex);
	CudaHelper::free(prefixScan);
	CudaHelper::free(validCluster);
	CudaHelper::free(mergedCluster);
	CudaHelper::free(numNodesCount);
	CudaHelper::free(arraySizeCount);
}

glm::u32* Bvh::computeMortonCodes() const
{
	glm::u32* mortonCodesBufferGPU = nullptr;
	CudaHelper::initializeBufferGPU(mortonCodesBufferGPU, _numIndices / 3);

	size_t maxNumThreadsBlock = CudaHelper::getMaxThreadsBlock(), numBlocks = CudaHelper::getNumBlocks(_numIndices / 3, maxNumThreadsBlock);
	glm::vec3 aabbMax = _aabb.max(), aabbMin = _aabb.min();

	computeMortonCodesCuda<<<numBlocks, maxNumThreadsBlock>>>(_vertexBufferGPU, _indexBufferGPU, _numIndices, mortonCodesBufferGPU, aabbMax, aabbMin);

	//std::vector<glm::u32> codes;
	//codes.resize(_numIndices / 3);
	//CudaHelper::downloadBufferGPU(mortonCodesBufferGPU, codes.data(), _numIndices / 3);

	return mortonCodesBufferGPU;
}

glm::u32* Bvh::sortMortonCodes(glm::u32* mortonCodesBufferGPU) const
{
	GLuint numTriangles = _numIndices / 3;

	std::vector<glm::u32> indices(numTriangles);
	std::iota(indices.begin(), indices.end(), 0);
	glm::u32* indicesBufferGPU = nullptr;
	CudaHelper::initializeBufferGPU(indicesBufferGPU, numTriangles, indices.data());

	size_t maxNumThreadsBlock = CudaHelper::getMaxThreadsBlock(), numBlocks = CudaHelper::getNumBlocks(numTriangles, maxNumThreadsBlock);

	glm::u32* dTemp_storage = nullptr;
	size_t tempStorageBytes = 0;

	glm::u32* d_keys_in = mortonCodesBufferGPU;
	glm::u32* d_keys_out = nullptr;
	glm::u32* d_values_in = indicesBufferGPU;
	glm::u32* d_values_out = nullptr;
	CudaHelper::initializeBufferGPU(d_keys_out, numTriangles);
	CudaHelper::initializeBufferGPU(d_values_out, numTriangles);

	cub::DeviceRadixSort::SortPairs(dTemp_storage, tempStorageBytes, d_keys_in, d_keys_out, d_values_in, d_values_out, numTriangles);
	CudaHelper::initializeBufferGPU(dTemp_storage, tempStorageBytes);
	cub::DeviceRadixSort::SortPairs(dTemp_storage, tempStorageBytes, d_keys_in, d_keys_out, d_values_in, d_values_out, numTriangles);

	CudaHelper::downloadBufferGPU(d_values_out, indices.data(), numTriangles);

	CudaHelper::free(d_keys_in);
	CudaHelper::free(d_values_in);
	CudaHelper::free(d_keys_out);
	CudaHelper::free(dTemp_storage);

	return d_values_out;
}

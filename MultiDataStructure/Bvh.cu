// ReSharper disable CppExpressionWithoutSideEffects
#include "stdafx.h"
#include "Bvh.h"

#include "CudaHelper.h"

#include <cub/cub.cuh>
#include <numeric>
#include "bvh_kernels.cuh"
#include "morton_encoding_kernels.cuh"

//

void Bvh::resolveRayQueries(const Node& node, const RayGPU& ray, const VertexGPU* vertices, const glm::u32* indices, HitInfo& hitInfo) const
{
	if (node.intersectsRay(ray) < hitInfo._t)
	{
		if (node._prevIndex1 == INT_MAX && node._prevIndex2 == INT_MAX)
		{
			HitInfo localHitInfo;

			ray.intersectsTriangle(
				vertices[indices[node._triangleIndex + 0]]._position,
				vertices[indices[node._triangleIndex + 1]]._position,
				vertices[indices[node._triangleIndex + 2]]._position,
				localHitInfo, hitInfo._t
			);

			if (localHitInfo._hit == 1 && localHitInfo._t < hitInfo._t)
			{
				hitInfo._hit = 1;
				hitInfo._position = localHitInfo._position;
				hitInfo._normal = localHitInfo._normal;
				hitInfo._t = localHitInfo._t;
			}
		}
		else
		{
			const float distanceA = _nodes[node._prevIndex1].intersectsRay(ray);
			const float distanceB = _nodes[node._prevIndex2].intersectsRay(ray);

			const bool isNearestA = distanceA < distanceB;
			const float distanceNear = isNearestA ? distanceA : distanceB;
			const float distanceFar = isNearestA ? distanceB : distanceA;
			const glm::uint childNear = isNearestA ? node._prevIndex1 : node._prevIndex2;
			const glm::uint childFar = isNearestA ? node._prevIndex2 : node._prevIndex1;

			if (distanceFar < hitInfo._t)
				this->resolveRayQueries(_nodes[childFar], ray, vertices, indices, hitInfo);

			if (distanceNear < hitInfo._t)
				this->resolveRayQueries(_nodes[childNear], ray, vertices, indices, hitInfo);
		}
	}
}

Bvh::Bvh(Node* triangles, glm::uint numTriangles) : _triangles(triangles), _numTriangles(numTriangles), _nodes(nullptr)
{
}

Bvh::~Bvh()
{
	delete[] _nodes;
}

void Bvh::build(glm::uint& numNodes)
{
	constexpr float C_t = 1.0f; // Cost of traversing a node
	constexpr float C_i = 2.0f; // Cost of intersecting a primitive

	auto mergeHeuristic = [=](const Node* nodes, const glm::uint index1, const glm::uint index2)
	{
		const glm::vec3 minPoint = glm::min(nodes[index1]._minPoint, nodes[index2]._minPoint);
		const glm::vec3 maxPoint = glm::max(nodes[index1]._maxPoint, nodes[index2]._maxPoint);
		const glm::vec3 length = maxPoint - minPoint;
		const float SA = 2.0f * (length.x * length.y + length.z * length.y + length.x * length.z);

		return C_t + (static_cast<float>(nodes[index1]._numTriangles) + static_cast<float>(nodes[index2]._numTriangles)) * C_i * SA;
	};

	auto mergeCluster = [](Node* tempCluster, Node* cluster, glm::uint* currentPosition, glm::uint& numNodes, const glm::uint index1, const glm::uint index2)
	{
		// Thread which manages index1 is the one intersectsBoundingBox charge of doing all this
		tempCluster[index1]._minPoint = min(tempCluster[index1]._minPoint, tempCluster[index2]._minPoint);
		tempCluster[index1]._maxPoint = max(tempCluster[index1]._maxPoint, tempCluster[index2]._maxPoint);
		tempCluster[index1]._prevIndex1 = currentPosition[index1];
		tempCluster[index1]._prevIndex2 = currentPosition[index2];
		tempCluster[index1]._triangleIndex = INT_MAX;
		tempCluster[index1]._numTriangles = tempCluster[index1]._numTriangles + tempCluster[index2]._numTriangles;

		// Compact buffer operation
		#pragma omp atomic
		glm::uint newPosition = numNodes++;

		cluster[newPosition] = tempCluster[index1];
		currentPosition[index1] = newPosition;
	};

	//
	std::vector<glm::uint> neighborIndex(_numTriangles), prefixScan(_numTriangles), validCluster(_numTriangles), mergedCluster(_numTriangles);
	numNodes = _numTriangles;
	glm::uint arraySize = static_cast<glm::uint>(_numTriangles);

	//
	Node* inTempNodes = new Node[_numTriangles];
	Node* outTempNodes = _triangles;
	Node* bvhNodes = new Node[_numTriangles * 2 - 1];
	for (int i = 0; i < _numTriangles; ++i)
		bvhNodes[i] = _triangles[i];

	//
	glm::uint* inPosition = new glm::uint[_numTriangles];
	glm::uint* outPosition = new glm::uint[_numTriangles];
	std::iota(outPosition, outPosition + _numTriangles, 0);

	while (arraySize > 1)
	{
		std::swap(outTempNodes, inTempNodes);
		std::swap(inPosition, outPosition);

		// Find best neighbor
		#pragma omp paralell for
		for (int index = 0; index < arraySize; ++index)
		{
			const glm::u32 lowerIndex = glm::max(index - SEARCH_RADIUS, 0), upperIndex = glm::min(index + SEARCH_RADIUS, int(arraySize) - 1);
			glm::u32 currentIndex = lowerIndex;
			float distance, minDistance = INT_MAX;

			while (currentIndex < index)
			{
				distance = mergeHeuristic(inTempNodes, index, currentIndex);
				if (distance < minDistance)
				{
					neighborIndex[index] = currentIndex;
					minDistance = distance;
				}

				++currentIndex;
			}

			++currentIndex;

			while (currentIndex <= upperIndex)
			{
				distance = mergeHeuristic(inTempNodes, index, currentIndex);
				if (distance < minDistance)
				{
					neighborIndex[index] = currentIndex;
					minDistance = distance;
				}

				++currentIndex;
			}
		}

		// Merge clusters if possible
		#pragma omp paralell for
		for (int index = 0; index < arraySize; ++index)
		{
			if (neighborIndex[neighborIndex[index]] == index)
			{
				if (index < neighborIndex[index])
				{
					mergeCluster(inTempNodes, bvhNodes, inPosition, numNodes, index, neighborIndex[index]);
					validCluster[index] = prefixScan[index] = 1;
					validCluster[neighborIndex[index]] = prefixScan[neighborIndex[index]] = 0;
					mergedCluster[index] = 1;
				}
				else
				{
					mergedCluster[index] = 0;
				}
			}
			else
			{
				validCluster[index] = prefixScan[index] = 1;				
				mergedCluster[index] = 0;
			}
		}

		// Exclusive sum
		glm::uint sum = 0u;
		for (int index = 0; index < arraySize; ++index)
		{
			prefixScan[index] = sum;
			sum += validCluster[index];
		}

		// Reallocate clusters
		#pragma omp paralell for
		for (int index = 0; index < arraySize; ++index)
		{
			if (validCluster[index] == 1)
			{
				outTempNodes[prefixScan[index]] = inTempNodes[index];
				outPosition[prefixScan[index]] = inPosition[index];
			}
		}

		// Update size
		arraySize = prefixScan[arraySize - 1] + validCluster[arraySize - 1];
	}

	delete[] inTempNodes;
	delete[] outTempNodes;
	delete[] inPosition;
	delete[] outPosition;

	this->_nodes = bvhNodes;
}

void Bvh::resolveRayQueries(const std::vector<RayGPU>& rays, std::vector<float>& depth, const VertexGPU* vertices, const glm::u32* indices) const
{
	const glm::uint numNodes = _numTriangles * 2 - 1;

	#pragma omp parallel for
	for (int rayIdx = 0; rayIdx < static_cast<int>(rays.size()); ++rayIdx)
	{
		HitInfo hitInfo, localHitInfo;
		hitInfo._t = FLT_MAX;
		hitInfo._hit = 0;

		const RayGPU& ray = rays[rayIdx];
		this->resolveRayQueries(_nodes[numNodes - 1], ray, vertices, indices, hitInfo);

		/*while (stackIndex > 0)
		{
			Node& node = _nodes[stack[--stackIndex]];

			if (node.intersectsRay(ray) < hitInfo._t)
			{
				if (node._prevIndex1 == INT_MAX && node._prevIndex2 == INT_MAX)
				{
					const glm::uint triangleIndex = node._triangleIndex;
					ray.intersectsTriangle(
						vertices[indices[triangleIndex + 0]]._position,
						vertices[indices[triangleIndex + 1]]._position,
						vertices[indices[triangleIndex + 2]]._position,
						localHitInfo
					);

					if (localHitInfo._hit == 1 && localHitInfo._t < hitInfo._t)
					{
						hitInfo._hit = 1;
						hitInfo._position = localHitInfo._position;
						hitInfo._normal = localHitInfo._normal;
						hitInfo._t = localHitInfo._t;
					}
				}
				else
				{
					const float distanceA = _nodes[node._prevIndex1].intersectsRay(ray);
					const float distanceB = _nodes[node._prevIndex2].intersectsRay(ray);

					const bool isNearestA = distanceA < distanceB;
					const float distanceNear = isNearestA ? distanceA : distanceB;
					const float distanceFar = isNearestA ? distanceB : distanceA;
					const glm::uint childNear = isNearestA ? node._prevIndex1 : node._prevIndex2;
					const glm::uint childFar = isNearestA ? node._prevIndex2 : node._prevIndex1;

					if (distanceFar < hitInfo._t)
						stack[stackIndex++] = childFar;

					if (distanceNear < hitInfo._t)
						stack[stackIndex++] = childNear;
				}
			}
		}*/

		depth[rayIdx] = hitInfo._hit == 1 ? hitInfo._t : FLT_MAX;
	}
}

void Bvh::resolveRayQueriesBruteForce(const std::vector<RayGPU>& rays, std::vector<float>& depth,
	const VertexGPU* vertices, const glm::u32* indices) const
{
	const glm::uint numNodes = _numTriangles * 2 - 1;

	#pragma omp parallel for
	for (int rayIdx = 0; rayIdx < static_cast<int>(rays.size()); ++rayIdx)
	{
		const RayGPU& ray = rays[rayIdx];
		HitInfo hitInfo, localHitInfo;
		hitInfo._t = FLT_MAX;
		hitInfo._hit = 0;

		for (int idx = 0; idx < numNodes; ++idx)
		{
			const Node& node = _nodes[idx];

			if (node._prevIndex1 == INT_MAX && node._prevIndex2 == INT_MAX)
			{
				const glm::uint triangleIndex = node._triangleIndex;
				ray.intersectsTriangle(
					vertices[indices[triangleIndex + 0]]._position,
					vertices[indices[triangleIndex + 1]]._position,
					vertices[indices[triangleIndex + 2]]._position,
					localHitInfo, hitInfo._t
				);

				if (localHitInfo._hit == 1 && localHitInfo._t < hitInfo._t)
				{
					hitInfo._hit = 1;
					hitInfo._position = localHitInfo._position;
					hitInfo._normal = localHitInfo._normal;
					hitInfo._t = localHitInfo._t;
				}
			}
		}

		depth[rayIdx] = hitInfo._hit == 1 ? hitInfo._t : FLT_MAX;
	}
}

bool Bvh::exportNodes(const std::string& filename) const
{
	std::vector<AABB> nodes;
	for (int idx = 0; idx < _numTriangles * 2 - 2; ++idx)
		nodes.emplace_back(_nodes[idx]._minPoint, _nodes[idx]._maxPoint);

	// Export as obj
	std::ofstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Error opening file: " << filename << '\n';
		return false;
	}

	file << "x_min,y_min,z_min,x_max,y_max,z_max" << '\n';
	for (const auto& node : nodes)
	{
		file << node.min().x << "," << node.min().y << "," << node.min().z << ",";
		file << node.max().x << "," << node.max().y << "," << node.max().z << '\n';
	}

	file.close();
	return true;
}

//

//inline void Bvh::buildTree(Node* tempClusterBufferGPU) const
//{
//	// Compute shader execution data: groups and iteration control
//	glm::u32 arraySize = _numIndices / 3;
//	const size_t maxBlockSize = CudaHelper::getMaxThreadsBlock();
//
//	// Compact cluster buffer support
//	glm::u32* currentPosBufferOut = new glm::u32[arraySize];
//	std::iota(currentPosBufferOut, currentPosBufferOut + arraySize, 0);
//
//	Node* coutBuffer = tempClusterBufferGPU;		// Swapped during loop => not const
//	Node* cinBuffer = nullptr;
//	CudaHelper::initializeBufferGPU(cinBuffer, arraySize);
//
//	glm::u32* inCurrentPosition = nullptr;
//	CudaHelper::initializeBufferGPU(inCurrentPosition, arraySize);		// Position of compact buffer where a cluster is saved
//	glm::u32* outCurrentPosition = nullptr;
//	CudaHelper::initializeBufferGPU(outCurrentPosition, arraySize, currentPosBufferOut);
//
//	glm::u32* neighborIndex = nullptr, *prefixScan = nullptr, *validCluster = nullptr, *mergedCluster = nullptr, *numNodesCount = nullptr, *arraySizeCount = nullptr;
//	CudaHelper::initializeBufferGPU(neighborIndex, arraySize);					// Nearest neighbor search
//	CudaHelper::initializeBufferGPU(prefixScan, arraySize);						// Final position of each valid cluster for the next loop iteration
//	CudaHelper::initializeBufferGPU(validCluster, arraySize);						// Clusters which takes part of next loop iteration
//	CudaHelper::initializeBufferGPU(mergedCluster, arraySize);					// A merged cluster is always valid, but the opposite situation is not fitting
//	CudaHelper::initializeBufferGPU(numNodesCount, 1, &arraySize);		// Number of currently added nodes, which increases as the clusters are merged
//	CudaHelper::initializeBufferGPU(arraySizeCount, 1);
//
//	while (arraySize > 1)
//	{
//		size_t numBlocks	= CudaHelper::getNumBlocks(arraySize, maxBlockSize);
//		size_t startThreads	= static_cast<size_t>(std::ceil(static_cast<float>(arraySize) / 2.0f));
//		size_t numExec		= static_cast<size_t>(std::ceil(std::log2(arraySize)));
//
//		// Thread sizes are repeated on reduce and sweep down phases
//		std::vector<size_t> threadCount;
//		threadCount.reserve(numExec);
//		threadCount.push_back(startThreads);
//
//		std::swap(coutBuffer, cinBuffer);
//		std::swap(inCurrentPosition, outCurrentPosition);
//
//		findBestNeighbors<<<numBlocks, maxBlockSize>>>(cinBuffer, neighborIndex, arraySize);
//		mergeClusters<<<numBlocks, maxBlockSize>>>(cinBuffer, _clusterBufferGPU, neighborIndex, validCluster, mergedCluster, prefixScan, inCurrentPosition, numNodesCount, arraySize);
//
//		// Prefix scan
//		glm::u32* tempStorage = nullptr;								
//		size_t tempStorageBytes = 0;
//
//		cub::DeviceScan::ExclusiveSum(tempStorage, tempStorageBytes, validCluster, prefixScan, arraySize);
//		CudaHelper::initializeBufferGPU(tempStorage, tempStorageBytes);
//		cub::DeviceScan::ExclusiveSum(tempStorage, tempStorageBytes, validCluster, prefixScan, arraySize);
//
//		reallocateClusters<<<numBlocks, maxBlockSize>>>(cinBuffer, coutBuffer, validCluster, prefixScan, inCurrentPosition, outCurrentPosition, arraySize);
//		endLoop<<<1, 1>>> (arraySizeCount, prefixScan, validCluster, arraySize);
//
//		CudaHelper::downloadBufferGPU(arraySizeCount, &arraySize, 1);
//		CudaHelper::free(tempStorage);
//	}
//
//	CudaHelper::free(cinBuffer);
//	CudaHelper::free(coutBuffer);
//	CudaHelper::free(inCurrentPosition);
//	CudaHelper::free(outCurrentPosition);
//	CudaHelper::free(neighborIndex);
//	CudaHelper::free(prefixScan);
//	CudaHelper::free(validCluster);
//	CudaHelper::free(mergedCluster);
//	CudaHelper::free(numNodesCount);
//	CudaHelper::free(arraySizeCount);
//}
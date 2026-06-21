#pragma once

#include "../../stdafx.h"
#include "../../AABB.h"
#include "../../experiments/Metrics.h"
#include "PointPrimitive.h"

namespace PointGpu
{
	// Shared host/device payloads for point-cloud GPU structures such as LBVH, grids, and future hybrids.
	enum class QueryType
	{
		Range = 0,
		CountRange = 1,
		Radius = 2,
		Knn = 3,
	};

	// Phase B2 split-axis policy for KDTree/BIH. Generators sample over this enum so the same
	// topology can express both extent-driven and depth-driven partitioning, which produces
	// genuinely different trees on anisotropic clouds. The CPU build path falls back to
	// longest-extent regardless; CUDA KDTree/BIH honor the policy.
	enum class KdAxisPolicy
	{
		LongestExtent = 0,
		RoundRobin = 1,
	};

	struct Options
	{
		int device = -1;
		std::string builder = "lbvh";
		size_t queryBatchSize = 0;
		size_t memoryBudgetMb = 0;
		KdAxisPolicy kdAxisPolicy = KdAxisPolicy::LongestExtent;
		// "auto" picks the structure's native choice. KDTree/BIH use gpu_tree_knn for
		// K <= MaxTrackedKnnK; other builders currently use gpu_bruteforce_knn.
		std::string knnBackend = "auto";
	};

	struct Query
	{
		QueryType type = QueryType::Range;
		AABB bounds;
		glm::vec3 center = glm::vec3(0.0f);
		float radius = 0.0f;
		size_t k = 0;
	};

	struct QuerySample
	{
		size_t visitedNodes = 0;
		size_t testedPoints = 0;
		size_t returnedPoints = 0;
		double elapsedMs = 0.0;
	};

	struct BuildResult
	{
		Experiments::BuildMetrics metrics;
		double uploadTimeMs = 0.0;
		double gpuBuildTimeMs = 0.0;
		size_t gpuMemoryBytes = 0;
		int device = 0;
		std::string builder = "lbvh";
		size_t activeStructureTypes = 0;
		double nestedActiveFraction = 0.0;
		std::string activeStructureSummary;
	};

	struct QueryResult
	{
		Experiments::QueryMetrics metrics;
		double gpuQueryTimeMs = 0.0;
		size_t rangeQueries = 0;
		size_t countRangeQueries = 0;
		size_t radiusQueries = 0;
		size_t knnQueries = 0;
		std::string knnBackend = "none";
		std::vector<std::vector<uint32_t>> knnPointIndices;
		std::vector<std::vector<float>> knnDistancesSquared;
		std::vector<QuerySample> samples;
	};

	struct DevicePoint
	{
		float x;
		float y;
		float z;
	};

	static_assert(sizeof(DevicePoint) == sizeof(PointPrimitive), "GPU points must match the position-only CPU point payload for direct upload.");

	struct LinearNode
	{
		float minX;
		float minY;
		float minZ;
		float maxX;
		float maxY;
		float maxZ;
		int left;
		int right;
		int parent;
		uint32_t pointOffset;
		uint32_t pointCount;
		// flags bit layout (KDTree/BIH builders):
		//   bit  0       : 1 = leaf, 0 = internal (existing semantics)
		//   bits 1..2    : stored split axis (0 = X, 1 = Y, 2 = Z) — only valid when bit 3 is set
		//   bit  3       : 1 = the split axis is explicitly stored in bits 1..2;
		//                  0 = derive axis from node extents at query time (longest-extent policy)
		//   bits 4..11   : node depth from root (0..255). Lets round-robin queries compute
		//                  axis = depth % 3 without recursing parents; also useful for diagnostics.
		//   bits 12..31  : reserved
		uint32_t flags;
	};

	struct LinearOctreeNode
	{
		float minX;
		float minY;
		float minZ;
		float maxX;
		float maxY;
		float maxZ;
		int parent;
		int childBase;
		uint32_t childMask;
		uint32_t pointOffset;
		uint32_t pointCount;
		uint32_t flags;
		uint32_t depth;
		uint32_t schemaDepth;
	};

	using LinearQuadTreeNode = LinearOctreeNode;
	using LinearMixedTreeNode = LinearOctreeNode;

	struct DeviceQuery
	{
		int type;
		float minX;
		float minY;
		float minZ;
		float maxX;
		float maxY;
		float maxZ;
		float centerX;
		float centerY;
		float centerZ;
		float radius;
		uint32_t knnK;
	};

	struct DeviceQuerySample
	{
		unsigned long long visitedNodes;
		unsigned long long testedPoints;
		unsigned long long returnedPoints;
		float elapsedMs;
	};

#ifdef __CUDACC__
	inline constexpr uint32_t MaxTrackedKnnK = 16;

	__device__ inline bool knnHitLess(float distanceSquared, uint32_t pointIndex, float otherDistanceSquared, uint32_t otherPointIndex)
	{
		return distanceSquared < otherDistanceSquared ||
			(distanceSquared == otherDistanceSquared && pointIndex < otherPointIndex);
	}

	__device__ inline void insertKnnHit(float* bestDistances, uint32_t* bestIndices, uint32_t& found, uint32_t trackedK, float distanceSquared, uint32_t pointIndex)
	{
		if (trackedK == 0)
			return;

		if (found < trackedK)
		{
			bestDistances[found] = distanceSquared;
			bestIndices[found] = pointIndex;
			++found;
			return;
		}

		uint32_t worstIndex = 0;
		for (uint32_t i = 1; i < trackedK; ++i)
		{
			if (knnHitLess(bestDistances[worstIndex], bestIndices[worstIndex], bestDistances[i], bestIndices[i]))
				worstIndex = i;
		}

		if (knnHitLess(distanceSquared, pointIndex, bestDistances[worstIndex], bestIndices[worstIndex]))
		{
			bestDistances[worstIndex] = distanceSquared;
			bestIndices[worstIndex] = pointIndex;
		}
	}

	__device__ inline float worstKnnDistance(const float* bestDistances, uint32_t found, uint32_t trackedK)
	{
		if (found < trackedK)
			return 3.402823466e+38F;

		float worstDistance = bestDistances[0];
		for (uint32_t i = 1; i < trackedK; ++i)
			worstDistance = fmaxf(worstDistance, bestDistances[i]);
		return worstDistance;
	}

	__device__ inline void sortKnnHits(float* distances, uint32_t* indices, uint32_t count)
	{
		for (uint32_t i = 1; i < count; ++i)
		{
			const float distance = distances[i];
			const uint32_t index = indices[i];
			uint32_t j = i;
			while (j > 0 && knnHitLess(distance, index, distances[j - 1], indices[j - 1]))
			{
				distances[j] = distances[j - 1];
				indices[j] = indices[j - 1];
				--j;
			}
			distances[j] = distance;
			indices[j] = index;
		}
	}

	__device__ inline float pointDistanceSquared(const DevicePoint& point, const DeviceQuery& query)
	{
		const float dx = point.x - query.centerX;
		const float dy = point.y - query.centerY;
		const float dz = point.z - query.centerZ;
		return dx * dx + dy * dy + dz * dz;
	}

	__device__ inline void insertKnnDistance(volatile float* bestDistances, uint32_t& found, uint32_t trackedK, float distanceSquared)
	{
		if (trackedK == 0)
			return;

		if (found < trackedK)
		{
			bestDistances[found++] = distanceSquared;
			return;
		}

		uint32_t worstIndex = 0;
		float worstDistance = bestDistances[0];
		for (uint32_t i = 1; i < trackedK; ++i)
		{
			const float candidate = bestDistances[i];
			if (candidate > worstDistance)
			{
				worstDistance = candidate;
				worstIndex = i;
			}
		}

		if (distanceSquared < worstDistance)
			bestDistances[worstIndex] = distanceSquared;
	}

	static __global__ void bruteForceKnnKernel(
		const DevicePoint* points,
		size_t pointCount,
		const DeviceQuery* queries,
		size_t queryCount,
		float clockRateKHz,
		DeviceQuerySample* samples)
	{
		const size_t queryIndex = static_cast<size_t>(blockIdx.x);
		if (queryIndex >= queryCount)
			return;

		const DeviceQuery query = queries[queryIndex];
		if (query.type != static_cast<int>(QueryType::Knn))
			return;

		const unsigned long long begin = clock64();
		DeviceQuerySample sample{};
		sample.visitedNodes = pointCount > 0 ? 1 : 0;

		const uint32_t requestedK = query.knnK;
		if (requestedK == 0 || pointCount == 0)
		{
			const unsigned long long end = clock64();
			sample.elapsedMs = clockRateKHz > 0.0f ? static_cast<float>(end - begin) / clockRateKHz : 0.0f;
			if (threadIdx.x == 0)
				samples[queryIndex] = sample;
			return;
		}

		const uint32_t trackedK = requestedK < MaxTrackedKnnK ? requestedK : MaxTrackedKnnK;
		volatile float bestDistances[MaxTrackedKnnK];
		uint32_t found = 0;
		for (size_t pointIndex = static_cast<size_t>(threadIdx.x); pointIndex < pointCount; pointIndex += blockDim.x)
		{
			const float distanceSquared = pointDistanceSquared(points[pointIndex], query);
			insertKnnDistance(bestDistances, found, trackedK, distanceSquared);
		}

		float checksum = 0.0f;
		for (uint32_t i = 0; i < found; ++i)
			checksum += bestDistances[i];

		extern __shared__ float sharedChecksums[];
		sharedChecksums[threadIdx.x] = checksum;
		__syncthreads();

		for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
		{
			if (threadIdx.x < stride)
				sharedChecksums[threadIdx.x] += sharedChecksums[threadIdx.x + stride];
			__syncthreads();
		}

		const unsigned long long end = clock64();
		if (threadIdx.x == 0)
		{
			const size_t requested = static_cast<size_t>(requestedK);
			sample.testedPoints = static_cast<unsigned long long>(pointCount);
			sample.returnedPoints = static_cast<unsigned long long>(pointCount < requested ? pointCount : requested);
			if (!(sharedChecksums[0] >= 0.0f))
				sample.returnedPoints = 0;
			sample.elapsedMs = clockRateKHz > 0.0f ? static_cast<float>(end - begin) / clockRateKHz : 0.0f;
			samples[queryIndex] = sample;
		}
	}
#endif
}

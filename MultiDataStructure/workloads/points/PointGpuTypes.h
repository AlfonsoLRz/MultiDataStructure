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

	// Split-axis policy for KDTree/BIH; CPU build falls back to longest-extent, CUDA honors it.
	enum class KdAxisPolicy
	{
		LongestExtent = 0,
		RoundRobin = 1,
	};

	struct Options
	{
		int				_device = -1;
		std::string		_builder = "lbvh";
		size_t			_queryBatchSize = 0;
		size_t			_memoryBudgetMb = 0;
		KdAxisPolicy	_kdAxisPolicy = KdAxisPolicy::LongestExtent;
		// "auto" picks the native backend: KDTree/BIH use gpu_tree_knn for K <= MaxTrackedKnnK, else bruteforce.
		std::string	_knnBackend = "auto";
	};

	struct Query
	{
		QueryType	_type = QueryType::Range;
		AABB		_bounds;
		glm::vec3 center = glm::vec3(0.0f);
		float	_radius = 0.0f;
		size_t	_k = 0;
	};

	struct QuerySample
	{
		size_t	_visitedNodes = 0;
		size_t	_testedPoints = 0;
		size_t	_returnedPoints = 0;
		double	_elapsedMs = 0.0;
	};

	struct BuildResult
	{
		Experiments::BuildMetrics	_metrics;
		double	_uploadTimeMs = 0.0;
		double	_gpuBuildTimeMs = 0.0;
		size_t	_gpuMemoryBytes = 0;
		int	_device = 0;
		std::string	_builder = "lbvh";
		size_t	_activeStructureTypes = 0;
		double	_nestedActiveFraction = 0.0;
		std::string	_activeStructureSummary;
	};

	struct QueryResult
	{
		Experiments::QueryMetrics	_metrics;
		double	_gpuQueryTimeMs = 0.0;
		size_t	_rangeQueries = 0;
		size_t	_countRangeQueries = 0;
		size_t	_radiusQueries = 0;
		size_t	_knnQueries = 0;
		std::string	_knnBackend = "none";
		std::vector<std::vector<uint32_t>>	_knnPointIndices;
		std::vector<std::vector<float>>	_knnDistancesSquared;
		std::vector<QuerySample>	_samples;
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
		float		_minX;
		float		_minY;
		float		_minZ;
		float		_maxX;
		float		_maxY;
		float		_maxZ;
		int			_left;
		int			_right;
		int			_parent;
		uint32_t	_pointOffset;
		uint32_t	_pointCount;
		// flags (KDTree/BIH): bit0 leaf; bits1..2 split axis (valid if bit3); bit3 axis stored; bits4..11 depth; bits12..31 reserved.
		uint32_t	_flags;
	};

	struct LinearOctreeNode
	{
		float		_minX;
		float		_minY;
		float		_minZ;
		float		_maxX;
		float		_maxY;
		float		_maxZ;
		int			_parent;
		int			_childBase;
		uint32_t	_childMask;
		uint32_t	_pointOffset;
		uint32_t	_pointCount;
		uint32_t	_flags;
		uint32_t	_depth;
		uint32_t	_schemaDepth;
	};

	using LinearQuadTreeNode = LinearOctreeNode;
	using LinearMixedTreeNode = LinearOctreeNode;

	struct DeviceQuery
	{
		int			_type;
		float		_minX;
		float		_minY;
		float		_minZ;
		float		_maxX;
		float		_maxY;
		float		_maxZ;
		float		_centerX;
		float		_centerY;
		float		_centerZ;
		float		_radius;
		uint32_t	_knnK;
	};

	struct DeviceQuerySample
	{
		unsigned long long	_visitedNodes;
		unsigned long long	_testedPoints;
		unsigned long long	_returnedPoints;
		float				_elapsedMs;
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
		const float dx = point.x - query._centerX;
		const float dy = point.y - query._centerY;
		const float dz = point.z - query._centerZ;
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
		if (query._type != static_cast<int>(QueryType::Knn))
			return;

		const unsigned long long begin = clock64();
		DeviceQuerySample sample{};
		sample._visitedNodes = pointCount > 0 ? 1 : 0;

		const uint32_t requestedK = query._knnK;
		if (requestedK == 0 || pointCount == 0)
		{
			const unsigned long long end = clock64();
			sample._elapsedMs = clockRateKHz > 0.0f ? static_cast<float>(end - begin) / clockRateKHz : 0.0f;
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
			sample._testedPoints = static_cast<unsigned long long>(pointCount);
			sample._returnedPoints = static_cast<unsigned long long>(pointCount < requested ? pointCount : requested);
			if (!(sharedChecksums[0] >= 0.0f))
				sample._returnedPoints = 0;
			sample._elapsedMs = clockRateKHz > 0.0f ? static_cast<float>(end - begin) / clockRateKHz : 0.0f;
			samples[queryIndex] = sample;
		}
	}
#endif
}

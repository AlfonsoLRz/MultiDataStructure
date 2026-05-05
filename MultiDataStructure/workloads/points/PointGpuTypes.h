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
	};

	struct Options
	{
		int device = -1;
		std::string builder = "lbvh";
		size_t queryBatchSize = 0;
		size_t memoryBudgetMb = 0;
	};

	struct Query
	{
		QueryType type = QueryType::Range;
		AABB bounds;
		glm::vec3 center = glm::vec3(0.0f);
		float radius = 0.0f;
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
	};

	struct QueryResult
	{
		Experiments::QueryMetrics metrics;
		double gpuQueryTimeMs = 0.0;
		size_t rangeQueries = 0;
		size_t countRangeQueries = 0;
		size_t radiusQueries = 0;
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
	};

	struct DeviceQuerySample
	{
		unsigned long long visitedNodes;
		unsigned long long testedPoints;
		unsigned long long returnedPoints;
		float elapsedMs;
	};
}

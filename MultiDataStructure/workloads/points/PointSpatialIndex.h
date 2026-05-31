#pragma once

#include "../../stdafx.h"
#include "../../core/Config.h"
#include "PointCloud.h"

class PointSpatialIndex
{
public:
	struct Node
	{
		AABB bounds;
		MultiDataStructure::DataStructureLevel type = MultiDataStructure::DataStructureLevel::OctreeNode;
		size_t depth = 0;
		size_t schemaDepth = 0;
		AABB tightBounds;
		size_t subtreePointCount = 0;
		size_t pointOffset = 0;
		size_t pointCount = 0;
		std::vector<std::unique_ptr<Node>> children;

		bool isLeaf() const { return children.empty(); }
	};

	struct Stats
	{
		size_t numNodes = 0;
		size_t numLeaves = 0;
		size_t numPoints = 0;
		size_t maxDepth = 0;
	};

	struct QueryStats
	{
		size_t visitedNodes = 0;
		size_t testedPoints = 0;
		size_t returnedPoints = 0;
		size_t fullyContainedNodes = 0;
		double elapsedMs = 0.0;
	};

	struct QueryResult
	{
		std::vector<size_t> pointIndices;
		QueryStats stats;
	};

	struct CountResult
	{
		size_t count = 0;
		QueryStats stats;
	};

	void build(const PointCloud& cloud, const SchemaConfig& schema);
	const Node* root() const { return _root.get(); }
	Stats stats() const;
	QueryResult rangeQuery(const AABB& bounds) const;
	CountResult countRange(const AABB& bounds) const;
	QueryResult radiusQuery(const glm::vec3& center, float radius) const;
	QueryResult knnQuery(const glm::vec3& center, size_t k) const;

private:
	struct ActiveLevel
	{
		const SchemaLevelConfig* config = nullptr;
		size_t schemaDepth = 0;
	};

	const PointCloud* _cloud = nullptr;
	SchemaConfig _schema;
	std::unique_ptr<Node> _root;
	std::vector<uint32_t> _pointOrder;

	void buildNode(std::unique_ptr<Node>& node);
	void computeNodeAggregates(Node* node);
	std::optional<ActiveLevel> activeLevelForNode(const Node& node) const;
	bool matchesCondition(const Node& node, const SchemaLevelCondition& condition) const;
	bool shouldSplit(const Node& node, const SchemaLevelConfig& levelConfig) const;
	std::vector<AABB> childBounds(const Node& node, const SchemaLevelConfig& levelConfig, float& splitValue, glm::uint& splitAxis) const;
	size_t locateChild(const glm::vec3& point, const SchemaLevelConfig& levelConfig, const Node& node, float splitValue, glm::uint splitAxis, size_t numChildren) const;
	void collectStats(const Node* node, Stats& stats) const;
	void appendSubtreePoints(const Node* node, std::vector<size_t>& pointIndices) const;
	void rangeQueryNode(const Node* node, const AABB& bounds, QueryResult& result) const;
	void countRangeNode(const Node* node, const AABB& bounds, CountResult& result) const;
	void radiusQueryNode(const Node* node, const glm::vec3& center, float radiusSquared, QueryResult& result) const;
};

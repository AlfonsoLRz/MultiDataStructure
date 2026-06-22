#pragma once

#include "../../stdafx.h"
#include "../../core/Config.h"
#include "PointCloud.h"

class PointSpatialIndex
{
public:
	struct LeafMicroIndex
	{
		struct KdNode
		{
			AABB _bounds;
			uint32_t _begin = 0;
			uint32_t _count = 0;
			uint32_t _left = std::numeric_limits<uint32_t>::max();
			uint32_t _right = std::numeric_limits<uint32_t>::max();

			bool isLeaf() const
			{
				return _left == std::numeric_limits<uint32_t>::max() &&
					_right == std::numeric_limits<uint32_t>::max();
			}
		};

		AABB _bounds;
		glm::uvec3 gridResolution = glm::uvec3(1, 1, 1);
		std::vector<uint32_t> _gridOffsets;
		std::vector<uint32_t> _gridOrder;
		std::vector<uint32_t> _kdOrder;
		std::vector<KdNode> _kdNodes;

		bool hasGrid() const { return _gridOffsets.size() > 1 && !_gridOrder.empty(); }
		bool hasKdTree() const { return !_kdNodes.empty() && !_kdOrder.empty(); }
	};

	struct Node
	{
		AABB _bounds;
		MultiDataStructure::DataStructureLevel _type = MultiDataStructure::DataStructureLevel::OctreeNode;
		size_t _depth = 0;
		size_t _schemaDepth = 0;
		AABB _tightBounds;
		size_t _subtreePointCount = 0;
		size_t _pointOffset = 0;
		size_t _pointCount = 0;
		std::unique_ptr<LeafMicroIndex> _microIndex;
		std::vector<std::unique_ptr<Node>> _children;

		bool isLeaf() const { return _children.empty(); }
	};

	struct Stats
	{
		size_t _numNodes = 0;
		size_t _numLeaves = 0;
		size_t _numPoints = 0;
		size_t _maxDepth = 0;
	};

	struct QueryStats
	{
		static constexpr size_t MaxBreakdownDepth = 64;

		struct QueryBreakdown
		{
			std::array<size_t, MaxBreakdownDepth> visitedByDepth{};
			std::unordered_map<std::string, size_t> _visitedByStructure;
			std::unordered_map<std::string, size_t> _testedPointsByStructure;
			std::unordered_map<std::string, size_t> _fullyContainedByStructure;
		};

		size_t _visitedNodes = 0;
		size_t _testedPoints = 0;
		size_t _returnedPoints = 0;
		size_t _fullyContainedNodes = 0;
		double _elapsedMs = 0.0;
		QueryBreakdown _breakdown;
	};

	struct QueryResult
	{
		std::vector<size_t> _pointIndices;
		QueryStats _stats;
	};

	struct CountResult
	{
		size_t _count = 0;
		QueryStats _stats;
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
		const SchemaLevelConfig* _config = nullptr;
		size_t _schemaDepth = 0;
	};

	struct BuildScratch
	{
		std::vector<uint32_t> _tempOrder;
		std::vector<uint32_t> _childCounts;
		std::vector<size_t> _childOffsets;
		std::vector<AABB> _childBounds;
	};

	const PointCloud* _cloud = nullptr;
	SchemaConfig _schema;
	std::unique_ptr<Node> _root;
	std::vector<uint32_t> _pointOrder;
	std::vector<float> _orderedX;
	std::vector<float> _orderedY;
	std::vector<float> _orderedZ;

	void buildNode(std::unique_ptr<Node>& node, BuildScratch& scratch);
	void rebuildOrderedPointSoA();
	void computeNodeAggregates(Node* node);
	void buildLeafMicroIndexes(Node* node);
	std::unique_ptr<LeafMicroIndex> buildLeafMicroIndex(const Node& node) const;
	uint32_t buildLeafMicroKdNode(LeafMicroIndex& index, uint32_t begin, uint32_t end, uint32_t depth) const;
	std::optional<ActiveLevel> activeLevelForNode(const Node& node) const;
	bool matchesCondition(const Node& node, const SchemaLevelCondition& condition) const;
	bool shouldSplit(const Node& node, const SchemaLevelConfig& levelConfig) const;
	size_t effectiveLeafCapacity(const Node& node, const SchemaLevelConfig& levelConfig) const;
	double nodeDensity(const Node& node) const;
	double nodeHeightRatio(const Node& node) const;
	double nodeAnisotropy(const Node& node) const;
	void childBounds(const Node& node, const SchemaLevelConfig& levelConfig, float& splitValue, glm::uint& splitAxis, std::vector<AABB>& bounds) const;
	size_t locateChild(const glm::vec3& point, const SchemaLevelConfig& levelConfig, const Node& node, float splitValue, glm::uint splitAxis, size_t numChildren) const;
	void collectStats(const Node* node, Stats& stats) const;
	void appendSubtreePoints(const Node* node, std::vector<size_t>& pointIndices) const;
	std::string structureNameForNode(const Node& node) const;
	void recordNodeVisit(const Node& node, QueryStats& stats) const;
	void recordTestedPoints(const Node& node, size_t count, QueryStats& stats) const;
	void recordFullyContainedNode(const Node& node, QueryStats& stats) const;
	void rangeQueryMicroGrid(const Node& node, const AABB& bounds, QueryResult& result) const;
	void countRangeMicroGrid(const Node& node, const AABB& bounds, CountResult& result) const;
	void radiusQueryMicroGrid(const Node& node, const glm::vec3& center, float radiusSquared, QueryResult& result) const;
	void rangeQueryNode(const Node* node, const AABB& bounds, QueryResult& result) const;
	void countRangeNode(const Node* node, const AABB& bounds, CountResult& result) const;
	void radiusQueryNode(const Node* node, const glm::vec3& center, float radiusSquared, QueryResult& result) const;
};

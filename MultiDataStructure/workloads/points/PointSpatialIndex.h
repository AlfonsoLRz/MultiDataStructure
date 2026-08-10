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
			AABB		_bounds;
			uint32_t	_begin = 0;
			uint32_t	_count = 0;
			uint32_t	_left = std::numeric_limits<uint32_t>::max();
			uint32_t	_right = std::numeric_limits<uint32_t>::max();

			bool isLeaf() const
			{
				return _left == std::numeric_limits<uint32_t>::max() &&
					_right == std::numeric_limits<uint32_t>::max();
			}
		};

		AABB	_bounds;
		glm::uvec3 gridResolution = glm::uvec3(1, 1, 1);
		std::vector<uint32_t>	_gridOffsets;
		std::vector<uint32_t>	_gridOrder;
		std::vector<uint32_t>	_kdOrder;
		std::vector<KdNode>		_kdNodes;

		bool hasGrid() const { return _gridOffsets.size() > 1 && !_gridOrder.empty(); }
		bool hasKdTree() const { return !_kdNodes.empty() && !_kdOrder.empty(); }
	};

	struct Node
	{
		AABB	_bounds;
		MultiDataStructure::DataStructureLevel	_type = MultiDataStructure::DataStructureLevel::OctreeNode;
		size_t	_depth = 0;
		size_t	_schemaDepth = 0;
		// Index into the breakdown counter arrays, resolved once during build so that query
		// traversal never has to name a structure. See QueryStats::QueryBreakdown.
		uint32_t _structureIndex = 0;
		AABB	_tightBounds;
		size_t	_subtreePointCount = 0;
		size_t	_pointOffset = 0;
		size_t	_pointCount = 0;
		std::unique_ptr<LeafMicroIndex>	_microIndex;
		std::vector<std::unique_ptr<Node>>	_children;

		bool isLeaf() const { return _children.empty(); }
	};

	struct Stats
	{
		size_t	_numNodes = 0;
		size_t	_numLeaves = 0;
		size_t	_numPoints = 0;
		size_t	_maxDepth = 0;
	};

	struct QueryStats
	{
		static constexpr size_t MaxBreakdownDepth = 64;
		// Distinct structure names a single schema can contribute. One slot per schema level
		// plus one fallback for nodes whose level cannot be resolved; schemas far smaller than
		// this in practice, and overflow degrades to the fallback slot rather than misreporting.
		static constexpr size_t MaxBreakdownStructures = 32;

		// Per-structure counters are plain arrays indexed by Node::_structureIndex, which is
		// resolved once at build time. They used to be unordered_map<std::string, size_t>
		// keyed by a by-value std::string built per node visit, inside the timed query region:
		// that cost ~106 ns per visited node against ~1.2 ns per point tested, i.e. it
		// dominated the measurement it was supposed to explain. The emitted CSV/JSON keeps the
		// same name:count form - names are resolved on read via structureBreakdownNames().
		struct QueryBreakdown
		{
			std::array<size_t, MaxBreakdownDepth> visitedByDepth{};
			std::array<size_t, MaxBreakdownStructures>	_visitedByStructure{};
			std::array<size_t, MaxBreakdownStructures>	_testedPointsByStructure{};
			std::array<size_t, MaxBreakdownStructures>	_fullyContainedByStructure{};
			// Slot names, shared with the owning index (copying a shared_ptr, not the strings).
			// Null on a default-constructed QueryStats, which formats as an empty breakdown.
			std::shared_ptr<const std::vector<std::string>>	_structureNames;

			// Count for one structure by name; 0 when the schema never produced such a node.
			size_t countFor(const std::array<size_t, MaxBreakdownStructures>& counters,
							const std::string& name) const
			{
				if (!_structureNames)
					return 0;
				for (size_t i = 0; i < _structureNames->size() && i < counters.size(); ++i)
				{
					if ((*_structureNames)[i] == name)
						return counters[i];
				}
				return 0;
			}

			// Recovers the old name:count pairs from a counter array, dropping zero slots and
			// sorting by name so emitted CSV/JSON keeps the ordering readers already expect.
			std::vector<std::pair<std::string, size_t>> namedCounts(
				const std::array<size_t, MaxBreakdownStructures>& counters) const
			{
				std::vector<std::pair<std::string, size_t>> named;
				if (!_structureNames)
					return named;
				const size_t count = std::min(_structureNames->size(), counters.size());
				for (size_t i = 0; i < count; ++i)
				{
					if (counters[i] > 0)
						named.emplace_back((*_structureNames)[i], counters[i]);
				}
				std::sort(named.begin(), named.end(), [](const auto& left, const auto& right) {
					return left.first < right.first;
				});
				return named;
			}
		};

		size_t			_visitedNodes = 0;
		size_t			_testedPoints = 0;
		size_t			_returnedPoints = 0;
		size_t			_fullyContainedNodes = 0;
		double			_elapsedMs = 0.0;
		QueryBreakdown	_breakdown;
	};

	struct QueryResult
	{
		std::vector<size_t>	_pointIndices;
		QueryStats			_stats;
	};

	struct CountResult
	{
		size_t		_count = 0;
		QueryStats	_stats;
	};

	void build(const PointCloud& cloud, const SchemaConfig& schema);
	const Node* root() const { return _root.get(); }
	Stats stats() const;
	// Names for the per-structure breakdown slots, in slot order. Readers zip this against
	// QueryBreakdown's arrays to recover the old name:count pairs.
	const std::vector<std::string>& structureBreakdownNames() const
	{
		static const std::vector<std::string> empty;
		return _structureNames ? *_structureNames : empty;
	}
	QueryResult rangeQuery(const AABB& bounds) const;
	CountResult countRange(const AABB& bounds) const;
	QueryResult radiusQuery(const glm::vec3& center, float radius) const;
	QueryResult knnQuery(const glm::vec3& center, size_t k) const;

private:
	struct ActiveLevel
	{
		const SchemaLevelConfig*	_config = nullptr;
		size_t						_schemaDepth = 0;
	};

	struct BuildScratch
	{
		std::vector<uint32_t>	_tempOrder;
		std::vector<uint32_t>	_childCounts;
		std::vector<size_t>		_childOffsets;
		std::vector<AABB>		_childBounds;
	};

	const PointCloud*		_cloud = nullptr;
	SchemaConfig			_schema;
	std::unique_ptr<Node>	_root;
	std::vector<uint32_t>	_pointOrder;
	std::vector<float>		_orderedX;
	std::vector<float>		_orderedY;
	std::vector<float>		_orderedZ;
	std::shared_ptr<std::vector<std::string>>	_structureNames;

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
	uint32_t structureIndexForNode(const Node& node);
	void assignStructureIndexes(Node* node);
	// Hot path: called once per visited node inside the timed region. Must stay allocation-
	// free and branch-light - see QueryStats::QueryBreakdown for what this used to cost.
	void recordNodeVisit(const Node& node, QueryStats& stats) const
	{
		++stats._visitedNodes;
		++stats._breakdown.visitedByDepth[std::min(node._depth, QueryStats::MaxBreakdownDepth - 1)];
		++stats._breakdown._visitedByStructure[node._structureIndex];
	}
	void recordTestedPoints(const Node& node, size_t count, QueryStats& stats) const
	{
		stats._testedPoints += count;
		stats._breakdown._testedPointsByStructure[node._structureIndex] += count;
	}
	void recordFullyContainedNode(const Node& node, QueryStats& stats) const
	{
		++stats._fullyContainedNodes;
		++stats._breakdown._fullyContainedByStructure[node._structureIndex];
	}
	void rangeQueryMicroGrid(const Node& node, const AABB& bounds, QueryResult& result) const;
	void countRangeMicroGrid(const Node& node, const AABB& bounds, CountResult& result) const;
	void radiusQueryMicroGrid(const Node& node, const glm::vec3& center, float radiusSquared, QueryResult& result) const;
	void rangeQueryNode(const Node* node, const AABB& bounds, QueryResult& result) const;
	void countRangeNode(const Node* node, const AABB& bounds, CountResult& result) const;
	void radiusQueryNode(const Node* node, const glm::vec3& center, float radiusSquared, QueryResult& result) const;
};

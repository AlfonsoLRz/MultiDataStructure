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

		// Nodes ENTERED: their bounds passed the query's prune test and the traversal went on to
		// read their children or points. Same definition for range, count, radius and kNN since
		// engine 2; before that, range/count/radius also counted nodes that were tested and
		// pruned, which is what _testedBoxes now holds.
		size_t			_visitedNodes = 0;
		// Bounds tests performed on non-empty nodes, entered or pruned. Equal to the pre-engine-2
		// "visited" count for range/count/radius on any schema that drops empty nodes.
		size_t			_testedBoxes = 0;
		size_t			_testedPoints = 0;
		size_t			_returnedPoints = 0;
		size_t			_fullyContainedNodes = 0;
		double			_elapsedMs = 0.0;
		// Order-independent digest of the returned point identities, computed OUTSIDE the timed
		// region. This is the equivalence gate for traversal work: latency, visited-node and
		// tested-point counts are all expected to move when the traversal is optimized, so none
		// of them can tell an optimization from a regression. The result set must not move, and
		// this is the column that proves it. Order-independent on purpose - a legitimate change
		// to descent order must not read as a behaviour change, while any gained or lost point
		// does. kNN output is additionally sorted, so for kNN this is exact set equality.
		uint64_t		_resultChecksum = 0;
		QueryBreakdown	_breakdown;
	};

	// Mixes one returned point identity into an order-independent digest: each identity is passed
	// through a bit-avalanche (splitmix64's finalizer) and the results are SUMMED. Addition is
	// commutative, so the digest depends on the multiset of returned points and not on the order
	// the traversal happened to emit them. Avalanching first is what makes it discriminating -
	// summing raw indices would collide on any swap preserving the total, e.g. {3, 8} vs {4, 7}.
	static void accumulateResultChecksum(uint64_t& checksum, size_t pointIndex)
	{
		uint64_t mixed = static_cast<uint64_t>(pointIndex) + 0x9E3779B97F4A7C15ull;
		mixed ^= mixed >> 30;
		mixed *= 0xBF58476D1CE4E5B9ull;
		mixed ^= mixed >> 27;
		mixed *= 0x94D049BB133111EBull;
		mixed ^= mixed >> 31;
		checksum += mixed;
	}

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

	// Names the query/build engine revision in every measurement that depends on it (score-cache
	// keys, engine-check output), so numbers from different engines are never silently mixed.
	static constexpr const char* EngineVersion = "3-build-plan";

	// Full fills every QueryStats counter and the per-depth/per-structure breakdown; it is what
	// scoring, the visit proxy and diagnostics read. Off compiles all of that out of the
	// traversal and leaves only the result, its checksum and the elapsed time, so a timed pass
	// measures the same work an uninstrumented library does. Results are identical in both.
	enum class QueryTelemetry
	{
		Full,
		Off,
	};

	// A schema level's split rule, resolved from its strings once per build. The build used to
	// re-derive this per POINT per level - Config::parseSchemaPrimitiveKind and the axis-policy
	// normalizer allocate and lowercase strings - twice per level (count pass and scatter pass).
	struct ResolvedLevel
	{
		enum class Split : uint8_t { Grid, Octree, Quad, BinaryKd, Binary };
		enum class KdPolicy : uint8_t { Median, RoundRobin, Center };

		Split		_split = Split::Octree;
		glm::uvec3	_gridSubdivisions = glm::uvec3(1);
		// Axis a quadtree level does not split; -1 = the node's shortest axis, decided per node.
		int			_quadIgnoredAxis = 2;
		KdPolicy	_kdPolicy = KdPolicy::Median;
	};

	// One node's split, computed once before its points are routed to children.
	struct SplitPlan
	{
		ResolvedLevel::Split	_split = ResolvedLevel::Split::Octree;
		glm::vec3	_min = glm::vec3(0.0f);
		glm::vec3	_extent = glm::vec3(0.0f);
		glm::vec3	_center = glm::vec3(0.0f);
		glm::uvec3	_subdivisions = glm::uvec3(1);
		glm::uint	_axis = 0;
		float		_value = 0.0f;
		size_t		_childCount = 0;
	};

	void build(const PointCloud& cloud, const SchemaConfig& schema);
	const Node* root() const { return _root.get(); }
	Stats stats() const;
	// Order-independent digest of the built tree: every node's shape, bounds bits, type and
	// structure name, plus each leaf's point-identity set. Two builds with equal digests are the
	// same tree, which is what lets a storage rewrite prove it changed layout and nothing else.
	// Walks the whole index; not for timed paths.
	uint64_t structureDigest() const;
	// Names for the per-structure breakdown slots, in slot order. Readers zip this against
	// QueryBreakdown's arrays to recover the old name:count pairs.
	const std::vector<std::string>& structureBreakdownNames() const
	{
		static const std::vector<std::string> empty;
		return _structureNames ? *_structureNames : empty;
	}
	QueryResult rangeQuery(const AABB& bounds, QueryTelemetry telemetry = QueryTelemetry::Full) const;
	CountResult countRange(const AABB& bounds, QueryTelemetry telemetry = QueryTelemetry::Full) const;
	QueryResult radiusQuery(const glm::vec3& center, float radius, QueryTelemetry telemetry = QueryTelemetry::Full) const;
	QueryResult knnQuery(const glm::vec3& center, size_t k, QueryTelemetry telemetry = QueryTelemetry::Full) const;

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
		// Child slot of each point of the node being split (at most 64 children: HGrid 4^3),
		// written by the counting pass so the scatter pass does not locate every point again.
		std::vector<uint8_t>	_childOf;
		std::vector<float>		_medianScratch;
	};

	const PointCloud*		_cloud = nullptr;
	SchemaConfig			_schema;
	std::unique_ptr<Node>	_root;
	// Output-buffer sizing hints, per index and per query kind, so a query can reserve its result
	// vector BEFORE the clock starts and then allocate nothing while being timed. See the query
	// entry points for why that matters. mutable + atomic because the query methods are const and
	// callers may query one index from several threads; the value is only a hint, so relaxed
	// ordering is sufficient and a lost update just costs one reallocation.
	mutable std::atomic<size_t>	_rangeReserveHint{ 0 };
	mutable std::atomic<size_t>	_radiusReserveHint{ 0 };
	std::vector<uint32_t>	_pointOrder;
	std::vector<float>		_orderedX;
	std::vector<float>		_orderedY;
	std::vector<float>		_orderedZ;
	std::shared_ptr<std::vector<std::string>>	_structureNames;
	std::vector<ResolvedLevel>	_resolvedLevels;	// parallel to _schema._levels
	std::vector<int32_t>		_structureSlotBySchemaDepth;	// build-time cache, -1 = unresolved

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
	const ResolvedLevel& resolvedLevelFor(const SchemaLevelConfig& levelConfig) const;
	void makeSplitPlan(const Node& node, const ResolvedLevel& level, SplitPlan& plan, BuildScratch& scratch) const;
	void routePointsToChildren(const Node& node, const SplitPlan& plan, BuildScratch& scratch) const;
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
	// Telemetry=false is the timed path: every record*() call below is compiled out of it.
	template <bool Telemetry> void rangeQueryMicroGrid(const Node& node, const AABB& bounds, QueryResult& result) const;
	template <bool Telemetry> void countRangeMicroGrid(const Node& node, const AABB& bounds, CountResult& result) const;
	template <bool Telemetry> void radiusQueryMicroGrid(const Node& node, const glm::vec3& center, float radiusSquared, QueryResult& result) const;
	template <bool Telemetry> void rangeQueryNode(const Node* node, const AABB& bounds, QueryResult& result) const;
	template <bool Telemetry> void countRangeNode(const Node* node, const AABB& bounds, CountResult& result) const;
	template <bool Telemetry> void radiusQueryNode(const Node* node, const glm::vec3& center, float radiusSquared, QueryResult& result) const;
	template <bool Telemetry> void knnSearch(const glm::vec3& center, size_t k, QueryResult& result) const;
};

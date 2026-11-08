#include "../../stdafx.h"
#include "PointSpatialIndex.h"
#include "LeafStore.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#if defined(_WIN32)
#include <malloc.h>
#else
#include <cstdlib>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif

static constexpr double EPSILON = 1e-9;

static glm::uint longestAxis(const AABB& bounds)
{
	const glm::vec3 size = bounds.size();
	if (size.x >= size.y && size.x >= size.z)
		return 0;
	if (size.y >= size.z)
		return 1;
	return 2;
}

static glm::uint shortestAxis(const AABB& bounds)
{
	const glm::vec3 size = bounds.size();
	if (size.x <= size.y && size.x <= size.z)
		return 0;
	if (size.y <= size.z)
		return 1;
	return 2;
}

static std::string normalizedAxisPolicy(std::string value)
{
	value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
		return std::isspace(c) || c == '_' || c == '-';
	}), value.end());
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	});
	return value;
}

static glm::uint ignoredQuadTreeAxis(const SchemaLevelConfig& levelConfig, const AABB& bounds)
{
	const std::string policy = normalizedAxisPolicy(levelConfig._axisPolicy.empty() ? std::string("xy") : levelConfig._axisPolicy);
	if (policy == "yz" || policy == "ignorex" || policy == "x")
		return 0;
	if (policy == "xz" || policy == "ignorey" || policy == "y")
		return 1;
	if (policy == "ignoreshortest" || policy == "shortest")
		return shortestAxis(bounds);
	return 2;
}

static bool containsAABB(const AABB& outer, const AABB& inner)
{
	const glm::vec3 outerMin = outer.min();
	const glm::vec3 outerMax = outer.max();
	const glm::vec3 innerMin = inner.min();
	const glm::vec3 innerMax = inner.max();
	return innerMin.x >= outerMin.x && innerMax.x <= outerMax.x &&
		   innerMin.y >= outerMin.y && innerMax.y <= outerMax.y &&
		   innerMin.z >= outerMin.z && innerMax.z <= outerMax.z;
}

static float distanceSquaredToAABB(const AABB& bounds, const glm::vec3& point)
{
	// Flat per-axis scalar accumulation, no vector temporaries. This runs once per
	// visited node in every kNN/radius traversal; the previous glm::clamp/length2
	// form built three temporaries per call. The pattern follows the min-dist
	// helpers in lean reference implementations (e.g. Indexicon's
	// kdMindistToRegion), which the external baseline showed to matter.
	const glm::vec3 min = bounds.min();
	const glm::vec3 max = bounds.max();
	float distanceSquared = 0.0f;
	if (point.x < min.x) { const float d = min.x - point.x; distanceSquared += d * d; }
	else if (point.x > max.x) { const float d = point.x - max.x; distanceSquared += d * d; }
	if (point.y < min.y) { const float d = min.y - point.y; distanceSquared += d * d; }
	else if (point.y > max.y) { const float d = point.y - max.y; distanceSquared += d * d; }
	if (point.z < min.z) { const float d = min.z - point.z; distanceSquared += d * d; }
	else if (point.z > max.z) { const float d = point.z - max.z; distanceSquared += d * d; }
	return distanceSquared;
}

static double clampedAdaptiveFactor(double value)
{
	if (!std::isfinite(value))
		return 1.0;
	return std::clamp(value, 0.25, 4.0);
}

static double elapsedMilliseconds(const std::chrono::steady_clock::time_point& start)
{
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

static size_t schemaBlockEndDepth(const SchemaConfig& schema, size_t schemaDepth)
{
	size_t cumulative = 0;
	for (const SchemaLevelConfig& level : schema._levels)
	{
		cumulative += level._numLevels;
		if (schemaDepth < cumulative)
			return cumulative;
	}

	return schema.totalLevels();
}

static bool belowMin(size_t value, const std::optional<size_t>& minValue)
{
	return minValue.has_value() && value < minValue.value();
}

static bool aboveMax(size_t value, const std::optional<size_t>& maxValue)
{
	return maxValue.has_value() && value > maxValue.value();
}

static bool belowMin(double value, const std::optional<double>& minValue)
{
	return minValue.has_value() && value < minValue.value();
}

static bool aboveMax(double value, const std::optional<double>& maxValue)
{
	return maxValue.has_value() && value > maxValue.value();
}

static SchemaPrimitiveKind primitiveKindForLevel(const SchemaLevelConfig& levelConfig)
{
	try
	{
		if (!levelConfig._typeName.empty())
			return Config::parseSchemaPrimitiveKind(levelConfig._typeName);
	}
	catch (const std::exception&)
	{
	}

	return levelConfig._primitiveKind;
}

static glm::uvec3 gridSubdivisionsForLevel(const SchemaLevelConfig& levelConfig)
{
	const SchemaPrimitiveKind kind = primitiveKindForLevel(levelConfig);
	if (kind == SchemaPrimitiveKind::HGrid)
		return glm::uvec3(4, 4, 4);
	return glm::uvec3(3, 3, 3);
}

static glm::uint strSlabsForLevel(const SchemaLevelConfig& levelConfig)
{
	return levelConfig._fanout == 8 ? 2u : levelConfig._fanout == 64 ? 4u : 3u;
}

static bool isGridPrimitive(const SchemaLevelConfig& levelConfig)
{
	const SchemaPrimitiveKind kind = primitiveKindForLevel(levelConfig);
	return kind == SchemaPrimitiveKind::RegularGrid || kind == SchemaPrimitiveKind::HGrid;
}

// Mirrors, decision for decision, what childBounds/locateChild used to work out from the level's
// strings on every call: grid kinds first (by parsed type name, then primitive kind), then the
// legacy level type. Any divergence here changes trees, which the structure digest would catch.
static PointSpatialIndex::ResolvedLevel resolveLevel(const SchemaLevelConfig& levelConfig)
{
	using ResolvedLevel = PointSpatialIndex::ResolvedLevel;
	ResolvedLevel resolved;
	if (primitiveKindForLevel(levelConfig) == SchemaPrimitiveKind::STR)
	{
		resolved._split = ResolvedLevel::Split::Str;
		resolved._strSlabs = strSlabsForLevel(levelConfig);
	}
	else if (isGridPrimitive(levelConfig))
	{
		resolved._split = ResolvedLevel::Split::Grid;
		resolved._gridSubdivisions = gridSubdivisionsForLevel(levelConfig);
	}
	else if (levelConfig._type == MultiDataStructure::DataStructureLevel::OctreeNode)
	{
		resolved._split = ResolvedLevel::Split::Octree;
		resolved._tightCenter = levelConfig._tightCenter;
	}
	else if (levelConfig._type == MultiDataStructure::DataStructureLevel::QuadTreeNode)
	{
		resolved._split = ResolvedLevel::Split::Quad;
		const std::string policy = normalizedAxisPolicy(levelConfig._axisPolicy.empty() ? std::string("xy") : levelConfig._axisPolicy);
		if (policy == "yz" || policy == "ignorex" || policy == "x")
			resolved._quadIgnoredAxis = 0;
		else if (policy == "xz" || policy == "ignorey" || policy == "y")
			resolved._quadIgnoredAxis = 1;
		else if (policy == "ignoreshortest" || policy == "shortest")
			resolved._quadIgnoredAxis = -1;
		else
			resolved._quadIgnoredAxis = 2;
		resolved._tightCenter = levelConfig._tightCenter;
	}
	else if (levelConfig._type == MultiDataStructure::DataStructureLevel::KDTreeNode)
	{
		resolved._split = ResolvedLevel::Split::BinaryKd;
		const std::string policy = normalizedAxisPolicy(levelConfig._axisPolicy.empty() ? std::string("median_longest_axis") : levelConfig._axisPolicy);
		if (policy == "roundrobin")
			resolved._kdPolicy = ResolvedLevel::KdPolicy::RoundRobin;
		else if (policy == "medianlongestaxis")
			resolved._kdPolicy = ResolvedLevel::KdPolicy::Median;
		else if (policy == "slidingmidpoint")
			resolved._kdPolicy = ResolvedLevel::KdPolicy::SlidingMidpoint;
		else
			resolved._kdPolicy = ResolvedLevel::KdPolicy::Center;
	}
	else
	{
		resolved._split = ResolvedLevel::Split::Binary;
	}
	return resolved;
}

static size_t gridChildIndex(const glm::vec3& point, const AABB& bounds, const glm::uvec3& subdivisions)
{
	const glm::vec3 minBound = bounds.min();
	const glm::vec3 extent = bounds.size();
	glm::uvec3 coordinates(0);
	for (glm::uint axis = 0; axis < 3; ++axis)
	{
		const uint32_t cells = subdivisions[axis];
		if (cells <= 1 || extent[axis] <= 0.0f)
			continue;

		const float normalized = (point[axis] - minBound[axis]) / extent[axis];
		const int raw = static_cast<int>(std::floor(normalized * static_cast<float>(cells)));
		coordinates[axis] = static_cast<glm::uint>(std::clamp(raw, 0, static_cast<int>(cells) - 1));
	}

	return coordinates.x * subdivisions.y * subdivisions.z +
		coordinates.y * subdivisions.z +
		coordinates.z;
}

static glm::uvec3 gridCoordinates(const glm::vec3& point, const AABB& bounds, const glm::uvec3& subdivisions)
{
	const glm::vec3 minBound = bounds.min();
	const glm::vec3 extent = bounds.size();
	glm::uvec3 coordinates(0);
	for (glm::uint axis = 0; axis < 3; ++axis)
	{
		const uint32_t cells = subdivisions[axis];
		if (cells <= 1 || extent[axis] <= 0.0f)
			continue;

		const float normalized = (point[axis] - minBound[axis]) / extent[axis];
		const int raw = static_cast<int>(std::floor(normalized * static_cast<float>(cells)));
		coordinates[axis] = static_cast<glm::uint>(std::clamp(raw, 0, static_cast<int>(cells) - 1));
	}
	return coordinates;
}

static AABB gridCellBounds(const AABB& bounds, const glm::uvec3& subdivisions, const glm::uvec3& coordinates)
{
	const glm::vec3 minBound = bounds.min();
	const glm::vec3 extent = bounds.size();
	glm::vec3 cellMin = minBound;
	glm::vec3 cellMax = bounds.max();
	for (glm::uint axis = 0; axis < 3; ++axis)
	{
		const uint32_t cells = subdivisions[axis];
		if (cells <= 1 || extent[axis] <= 0.0f)
			continue;

		const float cellSize = extent[axis] / static_cast<float>(cells);
		cellMin[axis] = minBound[axis] + static_cast<float>(coordinates[axis]) * cellSize;
		cellMax[axis] = coordinates[axis] + 1u >= cells
			? bounds.max()[axis]
			: cellMin[axis] + cellSize;
	}
	return AABB(cellMin, cellMax);
}

void PointSpatialIndex::build(const PointCloud& cloud, const SchemaConfig& schema)
{
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("CPU point index currently supports up to 2^32 - 1 points.");

	_cloud = &cloud;
	_schema = schema;
	_store = nullptr;
	_structureNames = std::make_shared<std::vector<std::string>>();
	_resolvedLevels.clear();
	for (const SchemaLevelConfig& level : _schema._levels)
		_resolvedLevels.push_back(resolveLevel(level));
	_structureSlotBySchemaDepth.assign(_schema.totalLevels(), -1);
	_pointOrder.resize(cloud.size());
	std::iota(_pointOrder.begin(), _pointOrder.end(), 0u);

	_root = std::make_unique<Node>();
	_root->_bounds = cloud.bounds();
	_root->_depth = 0;
	_root->_schemaDepth = 0;
	_root->_pointOffset = 0;
	_root->_pointCount = cloud.size();

	BuildScratch scratch;
	scratch._tempOrder.resize(cloud.size());
	scratch._childOf.resize(cloud.size());
	scratch._childCounts.reserve(64);
	scratch._childOffsets.reserve(64);
	scratch._childBounds.reserve(64);

	if (!cloud.empty())
		buildNode(_root, scratch);

	rebuildOrderedPointSoA();
	computeNodeAggregates(_root.get());
	buildLeafMicroIndexes(_root.get());
	assignStructureIndexes(_root.get());
	flattenTree();
	_root.reset();
}

PointSpatialIndex::Stats PointSpatialIndex::stats() const
{
	Stats result;
	if (_hot.empty())
		return result;

	std::vector<uint32_t> stack{ 0u };
	while (!stack.empty())
	{
		const uint32_t slot = stack.back();
		stack.pop_back();
		++result._numNodes;
		result._maxDepth = std::max<size_t>(result._maxDepth, _cold[slot]._depth);
		const HotNode& node = _hot[slot];
		if (!node.isInternal())
		{
			++result._numLeaves;
			result._numPoints += node.count();
			continue;
		}
		for (uint32_t c = 0; c < node.count(); ++c)
			stack.push_back(node._first + c);
	}
	return result;
}

size_t PointSpatialIndex::bytesPerNode()
{
	return sizeof(HotNode) + sizeof(ColdNode);
}

size_t PointSpatialIndex::memoryFootprintBytes() const
{
	size_t bytes = _hot.size() * sizeof(HotNode) + _cold.size() * sizeof(ColdNode);
	bytes += _pointOrder.size() * sizeof(uint32_t);
	bytes += (_orderedX.size() + _orderedY.size() + _orderedZ.size()) * sizeof(float);
	for (const std::unique_ptr<LeafMicroIndex>& micro : _microIndexes)
	{
		bytes += sizeof(LeafMicroIndex);
		bytes += (micro->_gridOffsets.size() + micro->_gridOrder.size() + micro->_kdOrder.size()) * sizeof(uint32_t);
		bytes += micro->_kdNodes.size() * sizeof(LeafMicroIndex::KdNode);
	}
	return bytes;
}

// ---------------------------------------------------------------- node views

bool PointSpatialIndex::NodeView::isLeaf() const { return !_index->_hot[_slot].isInternal(); }

size_t PointSpatialIndex::NodeView::childCount() const
{
	const HotNode& node = _index->_hot[_slot];
	return node.isInternal() ? node.count() : 0;
}

PointSpatialIndex::NodeView PointSpatialIndex::NodeView::child(size_t i) const
{
	return NodeView(_index, _index->_hot[_slot]._first + static_cast<uint32_t>(i));
}

AABB PointSpatialIndex::NodeView::bounds() const
{
	const ColdNode& cold = _index->_cold[_slot];
	return AABB(glm::vec3(cold._looseLo[0], cold._looseLo[1], cold._looseLo[2]),
				glm::vec3(cold._looseHi[0], cold._looseHi[1], cold._looseHi[2]));
}

AABB PointSpatialIndex::NodeView::tightBounds() const
{
	if (_index->_cold[_slot]._subtreeCount == 0)
		return bounds();
	const HotNode& node = _index->_hot[_slot];
	return AABB(glm::vec3(node._lo[0], node._lo[1], node._lo[2]), glm::vec3(node._hi[0], node._hi[1], node._hi[2]));
}

size_t PointSpatialIndex::NodeView::depth() const { return _index->_cold[_slot]._depth; }
size_t PointSpatialIndex::NodeView::schemaDepth() const { return _index->_cold[_slot]._schemaDepth; }

MultiDataStructure::DataStructureLevel PointSpatialIndex::NodeView::type() const
{
	return static_cast<MultiDataStructure::DataStructureLevel>(_index->_cold[_slot]._type);
}

uint32_t PointSpatialIndex::NodeView::structureIndex() const { return _index->_cold[_slot]._structureIndex; }
size_t PointSpatialIndex::NodeView::pointOffset() const { return _index->_cold[_slot]._subtreeOffset; }
size_t PointSpatialIndex::NodeView::pointCount() const { return _index->_cold[_slot]._subtreeCount; }

const PointSpatialIndex::LeafMicroIndex* PointSpatialIndex::NodeView::microIndex() const
{
	const uint32_t microSlot = _index->_cold[_slot]._microSlot;
	return microSlot == NoSlot ? nullptr : _index->_microIndexes[microSlot].get();
}

// ---------------------------------------------------------------- pool layout

// Lays the build tree out depth-first by sibling GROUP: a node's children are written together,
// then each child's subtree follows in child order. A subtree therefore occupies one contiguous
// slot range, so a spatially local query walks nearby memory all the way down, and each group
// starts on an even slot so that, with the pool 64-byte aligned, it begins on a cache line.
void PointSpatialIndex::flattenTree()
{
	_hot.clear();
	_cold.clear();
	_microIndexes.clear();
	if (!_root)
		return;

	_hot.resize(1);
	_cold.resize(1);
	writePoolNode(*_root, 0);
	layoutChildren(*_root, 0);
	if (_hot.size() >= static_cast<size_t>(NoSlot))
		throw std::runtime_error("CPU point index node pool exceeds the 32-bit slot space; use a larger leaf capacity.");
}

PointSpatialIndex::PoolView PointSpatialIndex::poolView() const
{
	PoolView view;
	view._nodes = _hot.data();
	view._nodeCount = _hot.size();
	view._x = _orderedX.data();
	view._y = _orderedY.data();
	view._z = _orderedZ.data();
	view._order = _pointOrder.data();
	view._pointCount = _pointOrder.size();
	view._subtree.resize(2 * _cold.size());
	for (size_t slot = 0; slot < _cold.size(); ++slot)
	{
		view._subtree[2 * slot] = _cold[slot]._subtreeOffset;
		view._subtree[2 * slot + 1] = _cold[slot]._subtreeCount;
	}

	// Children follow their parent in the pool, so one pass from the last slot up sees every
	// child's need before its parent's.
	std::vector<uint32_t> need(_hot.size(), 1);
	for (size_t slot = _hot.size(); slot-- > 0;)
	{
		const HotNode& node = _hot[slot];
		if (!node.isInternal() || node.count() == 0)
			continue;
		uint32_t deepest = 1;
		for (uint32_t c = 0; c < node.count(); ++c)
			deepest = std::max(deepest, need[node._first + c]);
		need[slot] = node.count() - 1 + deepest;
	}
	view._stackBound = need.empty() ? 0 : need.front();
	return view;
}

void PointSpatialIndex::writePoolNode(Node& node, uint32_t slot)
{
	HotNode& hot = _hot[slot];
	const glm::vec3 tightMin = node._tightBounds.min();
	const glm::vec3 tightMax = node._tightBounds.max();
	for (int axis = 0; axis < 3; ++axis)
	{
		hot._lo[axis] = tightMin[axis];
		hot._hi[axis] = tightMax[axis];
	}
	if (node.isLeaf())
	{
		if (node._pointCount >= InternalBit)
			throw std::runtime_error("CPU point index leaf holds 2^31 or more points; use a schema that splits further.");
		hot._first = static_cast<uint32_t>(node._pointOffset);
		hot._word = static_cast<uint32_t>(node._pointCount);
	}

	ColdNode& cold = _cold[slot];
	const glm::vec3 looseMin = node._bounds.min();
	const glm::vec3 looseMax = node._bounds.max();
	for (int axis = 0; axis < 3; ++axis)
	{
		cold._looseLo[axis] = looseMin[axis];
		cold._looseHi[axis] = looseMax[axis];
	}
	cold._subtreeOffset = static_cast<uint32_t>(node._pointOffset);
	cold._subtreeCount = static_cast<uint32_t>(node._subtreePointCount);
	if (node._depth > std::numeric_limits<uint16_t>::max() || node._schemaDepth > std::numeric_limits<uint16_t>::max())
		throw std::runtime_error("CPU point index depth exceeds 65535.");
	cold._depth = static_cast<uint16_t>(node._depth);
	cold._schemaDepth = static_cast<uint16_t>(node._schemaDepth);
	cold._type = static_cast<uint8_t>(node._type);
	cold._structureIndex = static_cast<uint8_t>(node._structureIndex);
	if (node._microIndex)
	{
		cold._microSlot = static_cast<uint32_t>(_microIndexes.size());
		_microIndexes.push_back(std::move(node._microIndex));
	}
}

void PointSpatialIndex::layoutChildren(Node& node, uint32_t slot)
{
	if (node.isLeaf())
		return;

	if (_hot.size() % 2 != 0)
	{
		// Padding slot: a zero-count leaf nothing points to, so the group below starts aligned.
		_hot.emplace_back();
		_cold.emplace_back();
	}

	const uint32_t first = static_cast<uint32_t>(_hot.size());
	const uint32_t count = static_cast<uint32_t>(node._children.size());
	_hot.resize(_hot.size() + count);
	_cold.resize(_cold.size() + count);
	_hot[slot]._first = first;
	_hot[slot]._word = InternalBit | count;

	for (uint32_t c = 0; c < count; ++c)
		writePoolNode(*node._children[c], first + c);
	for (uint32_t c = 0; c < count; ++c)
		layoutChildren(*node._children[c], first + c);
}

// Output-buffer sizing hints, one per query kind, carried across calls on the calling thread.
//
// The timed region has to contain the traversal and nothing else. It used to contain the
// QueryResult construction - a 1336-byte QueryStats zero-init - plus every reallocation of an
// output vector that started at capacity 0, so a radius query returning N points paid ~log2(N)
// mallocs and copy chains against the clock. The external baseline we are measured against
// (tools/indexicon_point_baseline.cpp) declares its hit buffer once outside both loops and only
// clear()s it, so after a handful of queries its timed region allocates nothing at all. Reserving
// from a high-water mark before the clock starts reproduces that: once the hint settles, the
// traversal does zero allocations, and the two sides are timing the same work.
//
// The hints live on the index (see the mutable atomics in the header), not in a thread_local: a
// thread_local hint is shared across every index that thread touches, so querying a cloud that
// returns 100k points would make every later query on a small cloud reserve 100k slots.
namespace
{
	// Raises a hint to at least `observed`, without ever lowering it. Relaxed throughout: this is
	// a sizing heuristic, and the worst outcome of a lost update is one extra reallocation.
	void raiseHint(std::atomic<size_t>& hint, size_t observed)
	{
		size_t current = hint.load(std::memory_order_relaxed);
		while (current < observed &&
			   !hint.compare_exchange_weak(current, observed, std::memory_order_relaxed))
		{
		}
	}

	// Calls call(full, micro, charged, stored, prefetch) with each flag as a compile-time constant,
	// so every combination is its own traversal and the plain timed path tests no flag per node. A
	// stored index has no micro-indexes (attachStore refuses them) and reads from a file, so Stored
	// never meets Micro or Prefetch.
	template <typename Call>
	void withTraversalFlags(bool full, bool micro, bool charged, bool stored, bool prefetch, Call&& call)
	{
		using T = std::true_type;
		using F = std::false_type;
		const auto withCharge = [&](auto f, auto m, auto s, auto p) {
			if (charged)
				call(f, m, T{}, s, p);
			else
				call(f, m, F{}, s, p);
		};
		const auto withPrefetch = [&](auto f, auto m) {
			if (prefetch)
				withCharge(f, m, F{}, T{});
			else
				withCharge(f, m, F{}, F{});
		};
		const auto withStore = [&](auto f) {
			if (stored)
				withCharge(f, F{}, T{}, F{});
			else if (micro)
				withPrefetch(f, T{});
			else
				withPrefetch(f, F{});
		};
		if (full)
			withStore(T{});
		else
			withStore(F{});
	}

	// Both helpers must be inlined into the traversal: a prefetch reads and writes no memory the
	// compiler can see, so a function made only of prefetches is classed pure, and GCC's early DCE
	// then deletes every call to it (the whole --prefetch-children arm compiled to nothing once).
	MDS_ALWAYS_INLINE void prefetchLine(const void* address)
	{
#if defined(_MSC_VER) || defined(__SSE__)
		_mm_prefetch(reinterpret_cast<const char*>(address), _MM_HINT_T0);
#else
		__builtin_prefetch(address);
#endif
	}

	// What a pushed child will need next: an internal child its own child group (its first cache
	// lines, at most four), a leaf the start of its point arrays. The child itself is already in
	// cache, since its box was just tested in the parent's group.
	MDS_ALWAYS_INLINE void prefetchChild(const PointSpatialIndex::HotNode& child, const PointSpatialIndex::HotNode* hot,
										 const float* xs, const float* ys, const float* zs, const uint32_t* order)
	{
		if (child.isInternal())
		{
			const char* group = reinterpret_cast<const char*>(hot + child._first);
			const size_t lines = std::min<size_t>(4, (child.count() * sizeof(PointSpatialIndex::HotNode) + 63) / 64);
			for (size_t line = 0; line < lines; ++line)
				prefetchLine(group + 64 * line);
			return;
		}
		prefetchLine(xs + child._first);
		prefetchLine(ys + child._first);
		prefetchLine(zs + child._first);
		if (order)
			prefetchLine(order + child._first);
	}

	// Per-thread depth timing (PointSpatialIndex::setDepthTiming): _last is the tick of the
	// previous entered node and _depth its depth; the gap to the next entered node, or to the end
	// of the query, is that node's cost.
	struct DepthTiming
	{
		std::array<uint64_t, PointSpatialIndex::QueryStats::MaxBreakdownDepth>	_ticks{};
		std::array<uint64_t, PointSpatialIndex::QueryStats::MaxBreakdownDepth>	_visits{};
		uint64_t	_last = 0;
		size_t		_depth = 0;
	};
	thread_local DepthTiming t_depthTiming;
}

void* PointSpatialIndex::allocatePool(size_t bytes, size_t alignment)
{
	constexpr size_t HugePage = size_t(2) << 20;
	constexpr size_t HugeThreshold = size_t(4) << 20;
	const bool huge = _hugePages && bytes >= HugeThreshold;
	if (huge)
	{
		alignment = HugePage;
		bytes = (bytes + HugePage - 1) / HugePage * HugePage;
	}
	bytes = std::max<size_t>(bytes, 1);
#if defined(_WIN32)
	void* pointer = _aligned_malloc(bytes, alignment);
	if (!pointer)
		throw std::bad_alloc();
#else
	void* pointer = nullptr;
	if (posix_memalign(&pointer, std::max(alignment, sizeof(void*)), bytes) != 0)
		throw std::bad_alloc();
#if defined(__linux__)
	// Advisory: with transparent huge pages in madvise mode the first touch of each 2 MB block
	// faults in one huge page; in never mode the call changes nothing.
	if (huge)
		madvise(pointer, bytes, MADV_HUGEPAGE);
#endif
#endif
	return pointer;
}

void PointSpatialIndex::freePool(void* pointer)
{
#if defined(_WIN32)
	_aligned_free(pointer);
#else
	free(pointer);
#endif
}

double PointSpatialIndex::ticksPerNs()
{
	if (_ticksPerNs > 0.0)
		return _ticksPerNs;
	// Over 50 ms of the steady clock. The counter runs at a fixed rate on x86-64 CPUs with an
	// invariant TSC, whatever the core clock does.
	const auto start = std::chrono::steady_clock::now();
	const uint64_t startTicks = __rdtsc();
	auto end = start;
	uint64_t endTicks = startTicks;
	do
	{
		end = std::chrono::steady_clock::now();
		endTicks = __rdtsc();
	} while (end - start < std::chrono::milliseconds(50));
	_ticksPerNs = static_cast<double>(endTicks - startTicks) / std::chrono::duration<double, std::nano>(end - start).count();
	return _ticksPerNs;
}

void PointSpatialIndex::noteVisitTiming(size_t depth)
{
	DepthTiming& timing = t_depthTiming;
	const uint64_t now = __rdtsc();
	depth = std::min<size_t>(depth, QueryStats::MaxBreakdownDepth - 1);
	if (timing._last)
		timing._ticks[timing._depth] += now - timing._last;
	timing._last = now;
	timing._depth = depth;
	++timing._visits[depth];
}

void PointSpatialIndex::finishDepthTiming()
{
	DepthTiming& timing = t_depthTiming;
	if (timing._last)
		timing._ticks[timing._depth] += __rdtsc() - timing._last;
	timing._last = 0;
}

void PointSpatialIndex::takeDepthTiming(std::array<double, QueryStats::MaxBreakdownDepth>& nsByDepth,
										std::array<uint64_t, QueryStats::MaxBreakdownDepth>& visitsByDepth)
{
	DepthTiming& timing = t_depthTiming;
	const double perNs = ticksPerNs();
	for (size_t depth = 0; depth < QueryStats::MaxBreakdownDepth; ++depth)
		nsByDepth[depth] = static_cast<double>(timing._ticks[depth]) / perNs;
	visitsByDepth = timing._visits;
	timing = DepthTiming{};
}

void PointSpatialIndex::attachStore(LeafStore* store)
{
	if (store && !_microIndexes.empty())
		throw std::invalid_argument("a leaf store cannot serve an index with leaf micro-indexes");
	_store = store;
}

void PointSpatialIndex::setNodeVisitCost(double nanoseconds)
{
	_nodeVisitNs = std::max(0.0, nanoseconds);
	_nodeVisitTicks = 0;
	if (_nodeVisitNs <= 0.0)
		return;
	_nodeVisitTicks = std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(_nodeVisitNs * ticksPerNs())));
}

inline void PointSpatialIndex::chargeNodeVisit()
{
	const uint64_t until = __rdtsc() + _nodeVisitTicks;
	while (__rdtsc() < until)
	{
	}
}

PointSpatialIndex::QueryResult PointSpatialIndex::rangeQuery(const AABB& bounds, QueryTelemetry telemetry) const
{
	QueryResult result;
	result._pointIndices.reserve(_rangeReserveHint.load(std::memory_order_relaxed));

	const bool full = telemetry == QueryTelemetry::Full;
	const bool micro = !_microIndexes.empty();
	const auto start = std::chrono::steady_clock::now();
	if (!_hot.empty())
	{
		withTraversalFlags(full, micro, _nodeVisitTicks != 0, _store != nullptr, _prefetchChildren, [&](auto f, auto m, auto c, auto s, auto p) {
			this->template rangeSearch<decltype(f)::value, decltype(m)::value, decltype(c)::value, decltype(s)::value, decltype(p)::value>(bounds, result);
		});
		if (full && _depthTiming)
			finishDepthTiming();
	}
	const double elapsedMs = elapsedMilliseconds(start);

	raiseHint(_rangeReserveHint, result._pointIndices.size());
	result._stats._returnedPoints = result._pointIndices.size();
	result._stats._elapsedMs = elapsedMs;
	for (size_t pointIndex : result._pointIndices)
		accumulateResultChecksum(result._stats._resultChecksum, pointIndex);
	result._stats._breakdown._structureNames = _structureNames;
	return result;
}

PointSpatialIndex::CountResult PointSpatialIndex::countRange(const AABB& bounds, QueryTelemetry telemetry) const
{
	CountResult result;

	const bool full = telemetry == QueryTelemetry::Full;
	const bool micro = !_microIndexes.empty();
	const auto start = std::chrono::steady_clock::now();
	if (!_hot.empty())
	{
		withTraversalFlags(full, micro, _nodeVisitTicks != 0, _store != nullptr, _prefetchChildren, [&](auto f, auto m, auto c, auto s, auto p) {
			this->template countSearch<decltype(f)::value, decltype(m)::value, decltype(c)::value, decltype(s)::value, decltype(p)::value>(bounds, result);
		});
		if (full && _depthTiming)
			finishDepthTiming();
	}
	const double elapsedMs = elapsedMilliseconds(start);

	result._stats._returnedPoints = result._count;
	result._stats._elapsedMs = elapsedMs;
	result._stats._breakdown._structureNames = _structureNames;
	return result;
}

PointSpatialIndex::QueryResult PointSpatialIndex::radiusQuery(const glm::vec3& center, float radius, QueryTelemetry telemetry) const
{
	QueryResult result;
	result._pointIndices.reserve(_radiusReserveHint.load(std::memory_order_relaxed));

	const bool full = telemetry == QueryTelemetry::Full;
	const bool micro = !_microIndexes.empty();
	const float radiusSquared = radius * radius;
	const auto start = std::chrono::steady_clock::now();
	if (radius >= 0.0f && !_hot.empty())
	{
		withTraversalFlags(full, micro, _nodeVisitTicks != 0, _store != nullptr, _prefetchChildren, [&](auto f, auto m, auto c, auto s, auto p) {
			this->template radiusSearch<decltype(f)::value, decltype(m)::value, decltype(c)::value, decltype(s)::value, decltype(p)::value>(center, radiusSquared, result);
		});
		if (full && _depthTiming)
			finishDepthTiming();
	}
	const double elapsedMs = elapsedMilliseconds(start);

	raiseHint(_radiusReserveHint, result._pointIndices.size());
	result._stats._returnedPoints = result._pointIndices.size();
	result._stats._elapsedMs = elapsedMs;
	for (size_t pointIndex : result._pointIndices)
		accumulateResultChecksum(result._stats._resultChecksum, pointIndex);
	result._stats._breakdown._structureNames = _structureNames;
	return result;
}

namespace
{
	// Pending-node entry of the best-first kNN: (lower-bound distance, push sequence) is a total
	// order, the sequence breaking distance ties in push order exactly as before the pool.
	struct KnnNodeCandidate
	{
		float		_distance = 0.0f;
		uint32_t	_sequence = 0;
		uint32_t	_slot = 0;
	};

	struct KnnNodeCandidateGreater
	{
		bool operator()(const KnnNodeCandidate& left, const KnnNodeCandidate& right) const
		{
			if (left._distance == right._distance)
				return left._sequence > right._sequence;
			return left._distance > right._distance;
		}
	};

	struct KnnMicroCandidate
	{
		float		_distance = 0.0f;
		uint32_t	_nodeIndex = 0;
	};

	struct KnnMicroCandidateGreater
	{
		bool operator()(const KnnMicroCandidate& left, const KnnMicroCandidate& right) const
		{
			return left._distance > right._distance;
		}
	};

	// Working sets reused across queries on the calling thread, so a warm thread allocates
	// nothing inside the timed region.
	std::vector<std::pair<float, size_t>>& knnBestScratch()
	{
		static thread_local std::vector<std::pair<float, size_t>> best;
		return best;
	}

	std::vector<KnnNodeCandidate>& knnPendingScratch()
	{
		static thread_local std::vector<KnnNodeCandidate> pending;
		return pending;
	}

	std::vector<KnnMicroCandidate>& knnMicroPendingScratch()
	{
		static thread_local std::vector<KnnMicroCandidate> microPending;
		return microPending;
	}

	std::vector<std::pair<float, uint32_t>>& knnStackScratch()
	{
		static thread_local std::vector<std::pair<float, uint32_t>> stack;
		return stack;
	}

	std::vector<uint32_t>& traversalStackScratch()
	{
		static thread_local std::vector<uint32_t> stack;
		return stack;
	}

	// The exact arithmetic of distanceSquaredToAABB / AABB::collides / containsAABB, on raw
	// floats. Pruning decisions must not move, or counters would.
	inline float boxDistanceSquared(const float* lo, const float* hi, const glm::vec3& point)
	{
		float distanceSquared = 0.0f;
		if (point.x < lo[0]) { const float d = lo[0] - point.x; distanceSquared += d * d; }
		else if (point.x > hi[0]) { const float d = point.x - hi[0]; distanceSquared += d * d; }
		if (point.y < lo[1]) { const float d = lo[1] - point.y; distanceSquared += d * d; }
		else if (point.y > hi[1]) { const float d = point.y - hi[1]; distanceSquared += d * d; }
		if (point.z < lo[2]) { const float d = lo[2] - point.z; distanceSquared += d * d; }
		else if (point.z > hi[2]) { const float d = point.z - hi[2]; distanceSquared += d * d; }
		return distanceSquared;
	}

	inline bool boxOverlaps(const float* lo, const float* hi, const glm::vec3& queryMin, const glm::vec3& queryMax)
	{
		return !(hi[0] < queryMin.x || lo[0] > queryMax.x ||
				 hi[1] < queryMin.y || lo[1] > queryMax.y ||
				 hi[2] < queryMin.z || lo[2] > queryMax.z);
	}

	inline bool boxInside(const float* lo, const float* hi, const glm::vec3& queryMin, const glm::vec3& queryMax)
	{
		return lo[0] >= queryMin.x && hi[0] <= queryMax.x &&
			   lo[1] >= queryMin.y && hi[1] <= queryMax.y &&
			   lo[2] >= queryMin.z && hi[2] <= queryMax.z;
	}

	// Leaf scans, eight points per step under AVX2. Results must be bit-identical to the scalar
	// loops they replace, so the arithmetic is spelled out rather than left to the compiler:
	// (dx*dx + dy*dy) + dz*dz with separate multiplies and adds (an FMA rounds once, not twice,
	// and would move points on the radius boundary), ordered compares (false on NaN, like the
	// scalar operators), and hits emitted in ascending index order. The remainder runs scalar.
	inline void scanLeafRadius(const float* xs, const float* ys, const float* zs, const uint32_t* order,
							   uint32_t begin, uint32_t end, const glm::vec3& center, float radiusSquared,
							   std::vector<size_t>& out)
	{
		uint32_t i = begin;
#if defined(__AVX2__)
		const __m256 cx = _mm256_set1_ps(center.x);
		const __m256 cy = _mm256_set1_ps(center.y);
		const __m256 cz = _mm256_set1_ps(center.z);
		const __m256 r2 = _mm256_set1_ps(radiusSquared);
		for (; i + 8 <= end; i += 8)
		{
			const __m256 dx = _mm256_sub_ps(_mm256_loadu_ps(xs + i), cx);
			const __m256 dy = _mm256_sub_ps(_mm256_loadu_ps(ys + i), cy);
			const __m256 dz = _mm256_sub_ps(_mm256_loadu_ps(zs + i), cz);
			const __m256 d2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)), _mm256_mul_ps(dz, dz));
			unsigned mask = static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(d2, r2, _CMP_LE_OQ)));
			while (mask)
			{
				const unsigned lane = static_cast<unsigned>(_tzcnt_u32(mask));
				out.push_back(order[i + lane]);
				mask &= mask - 1;
			}
		}
#endif
		for (; i < end; ++i)
		{
			const float dx = xs[i] - center.x;
			const float dy = ys[i] - center.y;
			const float dz = zs[i] - center.z;
			if (dx * dx + dy * dy + dz * dz <= radiusSquared)
				out.push_back(order[i]);
		}
	}

#if defined(__AVX2__)
	inline unsigned boxMask8(const float* xs, const float* ys, const float* zs, uint32_t i,
							 const __m256& minX, const __m256& minY, const __m256& minZ,
							 const __m256& maxX, const __m256& maxY, const __m256& maxZ)
	{
		const __m256 x = _mm256_loadu_ps(xs + i);
		const __m256 y = _mm256_loadu_ps(ys + i);
		const __m256 z = _mm256_loadu_ps(zs + i);
		__m256 inside = _mm256_and_ps(_mm256_cmp_ps(x, minX, _CMP_GE_OQ), _mm256_cmp_ps(x, maxX, _CMP_LE_OQ));
		inside = _mm256_and_ps(inside, _mm256_and_ps(_mm256_cmp_ps(y, minY, _CMP_GE_OQ), _mm256_cmp_ps(y, maxY, _CMP_LE_OQ)));
		inside = _mm256_and_ps(inside, _mm256_and_ps(_mm256_cmp_ps(z, minZ, _CMP_GE_OQ), _mm256_cmp_ps(z, maxZ, _CMP_LE_OQ)));
		return static_cast<unsigned>(_mm256_movemask_ps(inside));
	}
#endif

	// Offers one candidate to the k-best set. `best` is a max-heap on (distance, pointIndex) once
	// it holds k entries; below that it is an unordered bucket that make_heap orders as it fills.
	inline void offerKnnPoint(std::vector<std::pair<float, size_t>>& best, size_t k, const uint32_t* order,
							  float distance, uint32_t orderedIndex)
	{
		if (best.size() < k)
		{
			best.emplace_back(distance, static_cast<size_t>(order[orderedIndex]));
			if (best.size() == k)
				std::make_heap(best.begin(), best.end());
			return;
		}

		const std::pair<float, size_t>& worst = best.front();
		// Fast reject before touching the identity array: only an exact distance tie or an
		// accepted point needs it.
		if (distance > worst.first)
			return;

		const size_t pointIndex = static_cast<size_t>(order[orderedIndex]);
		if (distance < worst.first || pointIndex < worst.second)
		{
			std::pop_heap(best.begin(), best.end());
			best.back() = { distance, pointIndex };
			std::push_heap(best.begin(), best.end());
		}
	}

	inline bool pointInBox(float x, float y, float z, const glm::vec3& queryMin, const glm::vec3& queryMax)
	{
		return x >= queryMin.x && x <= queryMax.x &&
			   y >= queryMin.y && y <= queryMax.y &&
			   z >= queryMin.z && z <= queryMax.z;
	}

	inline void scanLeafRange(const float* xs, const float* ys, const float* zs, const uint32_t* order,
							  uint32_t begin, uint32_t end, const glm::vec3& queryMin, const glm::vec3& queryMax,
							  std::vector<size_t>& out)
	{
		uint32_t i = begin;
#if defined(__AVX2__)
		const __m256 minX = _mm256_set1_ps(queryMin.x), maxX = _mm256_set1_ps(queryMax.x);
		const __m256 minY = _mm256_set1_ps(queryMin.y), maxY = _mm256_set1_ps(queryMax.y);
		const __m256 minZ = _mm256_set1_ps(queryMin.z), maxZ = _mm256_set1_ps(queryMax.z);
		for (; i + 8 <= end; i += 8)
		{
			unsigned mask = boxMask8(xs, ys, zs, i, minX, minY, minZ, maxX, maxY, maxZ);
			while (mask)
			{
				const unsigned lane = static_cast<unsigned>(_tzcnt_u32(mask));
				out.push_back(order[i + lane]);
				mask &= mask - 1;
			}
		}
#endif
		for (; i < end; ++i)
		{
			if (pointInBox(xs[i], ys[i], zs[i], queryMin, queryMax))
				out.push_back(order[i]);
		}
	}

	inline size_t countLeafRange(const float* xs, const float* ys, const float* zs,
								 uint32_t begin, uint32_t end, const glm::vec3& queryMin, const glm::vec3& queryMax)
	{
		size_t contained = 0;
		uint32_t i = begin;
#if defined(__AVX2__)
		const __m256 minX = _mm256_set1_ps(queryMin.x), maxX = _mm256_set1_ps(queryMax.x);
		const __m256 minY = _mm256_set1_ps(queryMin.y), maxY = _mm256_set1_ps(queryMax.y);
		const __m256 minZ = _mm256_set1_ps(queryMin.z), maxZ = _mm256_set1_ps(queryMax.z);
		for (; i + 8 <= end; i += 8)
			contained += static_cast<size_t>(_mm_popcnt_u32(boxMask8(xs, ys, zs, i, minX, minY, minZ, maxX, maxY, maxZ)));
#endif
		for (; i < end; ++i)
		{
			if (pointInBox(xs[i], ys[i], zs[i], queryMin, queryMax))
				++contained;
		}
		return contained;
	}
}

PointSpatialIndex::QueryResult PointSpatialIndex::knnQuery(const glm::vec3& center, size_t k, QueryTelemetry telemetry) const
{
	QueryResult result;

	std::vector<std::pair<float, size_t>>& best = knnBestScratch();
	best.clear();
	knnPendingScratch().clear();
	const size_t indexedPoints = _cold.empty() ? 0 : _cold.front()._subtreeCount;
	// Clamped: k comes from a replayed trace and nothing bounds it against the cloud, so a
	// pathological k must not turn a reserve into a multi-gigabyte allocation.
	if (k > 0)
	{
		best.reserve(std::min<size_t>(k, indexedPoints));
		result._pointIndices.reserve(std::min<size_t>(k, indexedPoints));
	}

	const bool full = telemetry == QueryTelemetry::Full;
	const bool micro = !_microIndexes.empty();
	const auto start = std::chrono::steady_clock::now();
	if (_cloud && k > 0 && indexedPoints > 0)
	{
		// Depth-first for the small k of every captured workload; best-first above, where its
		// node-optimality pays for the heap.
		constexpr size_t DepthFirstMaxK = 64;
		withTraversalFlags(full, micro, _nodeVisitTicks != 0, _store != nullptr, _prefetchChildren, [&](auto f, auto m, auto c, auto s, auto p) {
			if (k <= DepthFirstMaxK)
				this->template knnSearchDepthFirst<decltype(f)::value, decltype(m)::value, decltype(c)::value, decltype(s)::value, decltype(p)::value>(center, k, result);
			else
				this->template knnSearch<decltype(f)::value, decltype(m)::value, decltype(c)::value, decltype(s)::value, decltype(p)::value>(center, k, result);
		});
		if (full && _depthTiming)
			finishDepthTiming();

		// The answer leaves the traversal in heap order, into storage reserved before the clock.
		// That is where the timed region ends, as it does for the external baselines: Indexicon's
		// kNN also returns its neighbours unsorted.
		for (const auto& entry : best)
			result._pointIndices.push_back(entry.second);
	}
	const double elapsedMs = elapsedMilliseconds(start);

	// Deterministic (distance, pointIndex) order, which callers and tests rely on, established
	// after the clock. The comparator is a total order, so the output is identical to sorting
	// inside the timed region; only what the clock covers changed.
	if (!best.empty())
	{
		std::sort(best.begin(), best.end(), [](const auto& left, const auto& right) {
			if (left.first == right.first)
				return left.second < right.second;
			return left.first < right.first;
		});
		for (size_t i = 0; i < best.size(); ++i)
			result._pointIndices[i] = best[i].second;
	}
	result._stats._returnedPoints = result._pointIndices.size();
	result._stats._elapsedMs = elapsedMs;
	for (size_t pointIndex : result._pointIndices)
		accumulateResultChecksum(result._stats._resultChecksum, pointIndex);
	result._stats._breakdown._structureNames = _structureNames;
	return result;
}

// ---------------------------------------------------------------- traversals

// Depth-first with an explicit stack. Each entered node tests all its children's boxes while
// their group is in cache and pushes only the survivors, in reverse so they pop in child order
// (the old recursion's order). Counter semantics are those of engine 2: a bounds test on a
// non-empty node is a tested box, a node whose test passed is visited.
template <bool Telemetry, bool Micro, bool Charged, bool Stored, bool Prefetch>
void PointSpatialIndex::rangeSearch(const AABB& bounds, QueryResult& result) const
{
	const glm::vec3 queryMin = bounds.min();
	const glm::vec3 queryMax = bounds.max();
	const HotNode* hot = _hot.data();
	const uint32_t* order = _pointOrder.data();
	const float* xs = _orderedX.data();
	const float* ys = _orderedY.data();
	const float* zs = _orderedZ.data();
	std::vector<size_t>& out = result._pointIndices;

	if (!hot[0].isInternal() && hot[0].count() == 0)
		return;
	if constexpr (Stored)
		_store->touchNodes(0, 1);
	if constexpr (Telemetry)
		++result._stats._testedBoxes;
	if (!boxOverlaps(hot[0]._lo, hot[0]._hi, queryMin, queryMax))
		return;

	std::vector<uint32_t>& stack = traversalStackScratch();
	stack.clear();
	stack.push_back(0);
	while (!stack.empty())
	{
		const uint32_t slot = stack.back();
		stack.pop_back();
		const HotNode& node = hot[slot];
		if constexpr (Telemetry)
			recordNodeVisit(slot, result._stats);
		if constexpr (Charged)
			chargeNodeVisit();

		if (boxInside(node._lo, node._hi, queryMin, queryMax))
		{
			if constexpr (Telemetry)
				recordFullyContainedNode(slot, result._stats);
			// Stored: one read per leaf of the subtree, the per-node load of a file-per-node store.
			if constexpr (Stored)
				_store->appendOrder(_cold[slot]._subtreeOffset, _cold[slot]._subtreeCount, out);
			else
				appendSubtreePoints(slot, out);
			continue;
		}

		if (!node.isInternal())
		{
			if constexpr (Micro && !Stored)
			{
				const uint32_t microSlot = _cold[slot]._microSlot;
				if (microSlot != NoSlot && _microIndexes[microSlot]->hasGrid())
				{
					rangeQueryMicroGrid<Telemetry>(slot, *_microIndexes[microSlot], bounds, result);
					continue;
				}
			}
			if constexpr (Telemetry)
				recordTestedPoints(slot, node.count(), result._stats);
			if constexpr (Stored)
			{
				const LeafStore::LeafView leaf = _store->leaf(node._first, node.count());
				scanLeafRange(leaf._x, leaf._y, leaf._z, leaf._order, 0, node.count(), queryMin, queryMax, out);
			}
			else
				scanLeafRange(xs, ys, zs, order, node._first, node._first + node.count(), queryMin, queryMax, out);
			continue;
		}

		if constexpr (Stored)
			_store->touchNodes(node._first, node.count());

		for (uint32_t c = node.count(); c-- > 0;)
		{
			const uint32_t childSlot = node._first + c;
			const HotNode& child = hot[childSlot];
			if (!child.isInternal() && child.count() == 0)
				continue;
			if constexpr (Telemetry)
				++result._stats._testedBoxes;
			if (boxOverlaps(child._lo, child._hi, queryMin, queryMax))
			{
				if constexpr (Prefetch)
					prefetchChild(child, hot, xs, ys, zs, order);
				stack.push_back(childSlot);
			}
		}
	}
}

template <bool Telemetry, bool Micro, bool Charged, bool Stored, bool Prefetch>
void PointSpatialIndex::countSearch(const AABB& bounds, CountResult& result) const
{
	const glm::vec3 queryMin = bounds.min();
	const glm::vec3 queryMax = bounds.max();
	const HotNode* hot = _hot.data();
	const float* xs = _orderedX.data();
	const float* ys = _orderedY.data();
	const float* zs = _orderedZ.data();

	if (!hot[0].isInternal() && hot[0].count() == 0)
		return;
	if constexpr (Stored)
		_store->touchNodes(0, 1);
	if constexpr (Telemetry)
		++result._stats._testedBoxes;
	if (!boxOverlaps(hot[0]._lo, hot[0]._hi, queryMin, queryMax))
		return;

	std::vector<uint32_t>& stack = traversalStackScratch();
	stack.clear();
	stack.push_back(0);
	while (!stack.empty())
	{
		const uint32_t slot = stack.back();
		stack.pop_back();
		const HotNode& node = hot[slot];
		if constexpr (Telemetry)
			recordNodeVisit(slot, result._stats);
		if constexpr (Charged)
			chargeNodeVisit();

		if (boxInside(node._lo, node._hi, queryMin, queryMax))
		{
			if constexpr (Telemetry)
				recordFullyContainedNode(slot, result._stats);
			result._count += _cold[slot]._subtreeCount;
			continue;
		}

		if (!node.isInternal())
		{
			if constexpr (Micro && !Stored)
			{
				const uint32_t microSlot = _cold[slot]._microSlot;
				if (microSlot != NoSlot && _microIndexes[microSlot]->hasGrid())
				{
					countRangeMicroGrid<Telemetry>(slot, *_microIndexes[microSlot], bounds, result);
					continue;
				}
			}
			if constexpr (Telemetry)
				recordTestedPoints(slot, node.count(), result._stats);
			if constexpr (Stored)
			{
				const LeafStore::LeafView leaf = _store->leaf(node._first, node.count());
				result._count += countLeafRange(leaf._x, leaf._y, leaf._z, 0, node.count(), queryMin, queryMax);
			}
			else
				result._count += countLeafRange(xs, ys, zs, node._first, node._first + node.count(), queryMin, queryMax);
			continue;
		}

		if constexpr (Stored)
			_store->touchNodes(node._first, node.count());

		for (uint32_t c = node.count(); c-- > 0;)
		{
			const uint32_t childSlot = node._first + c;
			const HotNode& child = hot[childSlot];
			if (!child.isInternal() && child.count() == 0)
				continue;
			if constexpr (Telemetry)
				++result._stats._testedBoxes;
			if (boxOverlaps(child._lo, child._hi, queryMin, queryMax))
			{
				if constexpr (Prefetch)
					prefetchChild(child, hot, xs, ys, zs, nullptr);
				stack.push_back(childSlot);
			}
		}
	}
}

template <bool Telemetry, bool Micro, bool Charged, bool Stored, bool Prefetch>
void PointSpatialIndex::radiusSearch(const glm::vec3& center, float radiusSquared, QueryResult& result) const
{
	const HotNode* hot = _hot.data();
	const uint32_t* order = _pointOrder.data();
	const float* xs = _orderedX.data();
	const float* ys = _orderedY.data();
	const float* zs = _orderedZ.data();
	std::vector<size_t>& out = result._pointIndices;

	if (!hot[0].isInternal() && hot[0].count() == 0)
		return;
	if constexpr (Stored)
		_store->touchNodes(0, 1);
	if constexpr (Telemetry)
		++result._stats._testedBoxes;
	if (boxDistanceSquared(hot[0]._lo, hot[0]._hi, center) > radiusSquared)
		return;

	std::vector<uint32_t>& stack = traversalStackScratch();
	stack.clear();
	stack.push_back(0);
	while (!stack.empty())
	{
		const uint32_t slot = stack.back();
		stack.pop_back();
		const HotNode& node = hot[slot];
		if constexpr (Telemetry)
			recordNodeVisit(slot, result._stats);
		if constexpr (Charged)
			chargeNodeVisit();

		if (!node.isInternal())
		{
			if constexpr (Micro && !Stored)
			{
				const uint32_t microSlot = _cold[slot]._microSlot;
				if (microSlot != NoSlot && _microIndexes[microSlot]->hasGrid())
				{
					radiusQueryMicroGrid<Telemetry>(slot, *_microIndexes[microSlot], center, radiusSquared, result);
					continue;
				}
			}
			if constexpr (Telemetry)
				recordTestedPoints(slot, node.count(), result._stats);
			if constexpr (Stored)
			{
				const LeafStore::LeafView leaf = _store->leaf(node._first, node.count());
				scanLeafRadius(leaf._x, leaf._y, leaf._z, leaf._order, 0, node.count(), center, radiusSquared, out);
			}
			else
				scanLeafRadius(xs, ys, zs, order, node._first, node._first + node.count(), center, radiusSquared, out);
			continue;
		}

		if constexpr (Stored)
			_store->touchNodes(node._first, node.count());

		for (uint32_t c = node.count(); c-- > 0;)
		{
			const uint32_t childSlot = node._first + c;
			const HotNode& child = hot[childSlot];
			if (!child.isInternal() && child.count() == 0)
				continue;
			if constexpr (Telemetry)
				++result._stats._testedBoxes;
			if (boxDistanceSquared(child._lo, child._hi, center) <= radiusSquared)
			{
				if constexpr (Prefetch)
					prefetchChild(child, hot, xs, ys, zs, order);
				stack.push_back(childSlot);
			}
		}
	}
}

// One leaf of a kNN traversal: its micro-index if it has one, else every point, eight at a time.
// Shared by both traversals, so they offer identical candidates in identical order per leaf.
template <bool Telemetry, bool Micro, bool Stored>
void PointSpatialIndex::knnLeaf(uint32_t slot, const glm::vec3& center, size_t k, QueryResult& result) const
{
	std::vector<std::pair<float, size_t>>& best = knnBestScratch();
	const HotNode& node = _hot[slot];
	const uint32_t* order = _pointOrder.data();
	const float* xs = _orderedX.data();
	const float* ys = _orderedY.data();
	const float* zs = _orderedZ.data();
	uint32_t begin = node._first;
	uint32_t end = node._first + node.count();
	if constexpr (Stored)
	{
		const LeafStore::LeafView leaf = _store->leaf(node._first, node.count());
		order = leaf._order;
		xs = leaf._x;
		ys = leaf._y;
		zs = leaf._z;
		begin = 0;
		end = node.count();
	}

	if constexpr (Micro && !Stored)
	{
		const uint32_t microSlot = _cold[slot]._microSlot;
		if (microSlot != NoSlot && _microIndexes[microSlot]->hasKdTree())
		{
			const LeafMicroIndex& micro = *_microIndexes[microSlot];
			std::vector<KnnMicroCandidate>& microPending = knnMicroPendingScratch();
			microPending.clear();
			microPending.push_back({ distanceSquaredToAABB(micro._kdNodes.front()._bounds, center), 0u });

			while (!microPending.empty())
			{
				std::pop_heap(microPending.begin(), microPending.end(), KnnMicroCandidateGreater{});
				const KnnMicroCandidate microCandidate = microPending.back();
				microPending.pop_back();
				if (best.size() == k && microCandidate._distance > best.front().first)
					break;

				const LeafMicroIndex::KdNode& microNode = micro._kdNodes[microCandidate._nodeIndex];
				if (microNode.isLeaf())
				{
					if constexpr (Telemetry)
						recordTestedPoints(slot, microNode._count, result._stats);
					const uint32_t microEnd = microNode._begin + microNode._count;
					for (uint32_t i = microNode._begin; i < microEnd; ++i)
					{
						const uint32_t orderedIndex = micro._kdOrder[i];
						const float dx = xs[orderedIndex] - center.x;
						const float dy = ys[orderedIndex] - center.y;
						const float dz = zs[orderedIndex] - center.z;
						offerKnnPoint(best, k, order, dx * dx + dy * dy + dz * dz, orderedIndex);
					}
					continue;
				}

				for (const uint32_t childIndex : { microNode._left, microNode._right })
				{
					if (childIndex == std::numeric_limits<uint32_t>::max())
						continue;

					const float childDistance = distanceSquaredToAABB(micro._kdNodes[childIndex]._bounds, center);
					if (best.size() == k && childDistance > best.front().first)
						continue;
					microPending.push_back({ childDistance, childIndex });
					std::push_heap(microPending.begin(), microPending.end(), KnnMicroCandidateGreater{});
				}
			}
			return;
		}
	}

	if constexpr (Telemetry)
		recordTestedPoints(slot, node.count(), result._stats);
	uint32_t i = begin;
#if defined(__AVX2__)
	const __m256 cx = _mm256_set1_ps(center.x);
	const __m256 cy = _mm256_set1_ps(center.y);
	const __m256 cz = _mm256_set1_ps(center.z);
	alignas(32) float distances[8];
	for (; i + 8 <= end; i += 8)
	{
		const __m256 dx = _mm256_sub_ps(_mm256_loadu_ps(xs + i), cx);
		const __m256 dy = _mm256_sub_ps(_mm256_loadu_ps(ys + i), cy);
		const __m256 dz = _mm256_sub_ps(_mm256_loadu_ps(zs + i), cz);
		const __m256 d2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)), _mm256_mul_ps(dz, dz));
		unsigned mask = 0xFFu;
		if (best.size() == k)
			mask = static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(d2, _mm256_set1_ps(best.front().first), _CMP_NGT_UQ)));
		if (!mask)
			continue;
		_mm256_store_ps(distances, d2);
		while (mask)
		{
			const unsigned lane = static_cast<unsigned>(_tzcnt_u32(mask));
			offerKnnPoint(best, k, order, distances[lane], i + lane);
			mask &= mask - 1;
		}
	}
#endif
	for (; i < end; ++i)
	{
		const float dx = xs[i] - center.x;
		const float dy = ys[i] - center.y;
		const float dz = zs[i] - center.z;
		offerKnnPoint(best, k, order, dx * dx + dy * dy + dz * dz, i);
	}
}

// Best-first kNN: a global min-heap of pending nodes by lower-bound distance, so nodes are
// entered in distance order and the search stops at the first that cannot improve the answer.
// Node-optimal, but every push and pop is an O(log n) sift. Used for large k.
template <bool Telemetry, bool Micro, bool Charged, bool Stored, bool Prefetch>
void PointSpatialIndex::knnSearch(const glm::vec3& center, size_t k, QueryResult& result) const
{
	std::vector<std::pair<float, size_t>>& best = knnBestScratch();
	std::vector<KnnNodeCandidate>& pending = knnPendingScratch();
	const HotNode* hot = _hot.data();

	uint32_t sequence = 0;
	if constexpr (Stored)
		_store->touchNodes(0, 1);
	if constexpr (Telemetry)
		++result._stats._testedBoxes;
	pending.push_back({ boxDistanceSquared(hot[0]._lo, hot[0]._hi, center), sequence++, 0u });

	while (!pending.empty())
	{
		std::pop_heap(pending.begin(), pending.end(), KnnNodeCandidateGreater{});
		const KnnNodeCandidate candidate = pending.back();
		pending.pop_back();

		// `pending` is a min-heap on lower-bound distance and the kth-best only shrinks, so the
		// first candidate that cannot improve the answer proves no remaining one can.
		if (best.size() == k && candidate._distance > best.front().first)
			break;

		const uint32_t slot = candidate._slot;
		const HotNode& node = hot[slot];
		if constexpr (Telemetry)
			recordNodeVisit(slot, result._stats);
		if constexpr (Charged)
			chargeNodeVisit();

		if (!node.isInternal())
		{
			knnLeaf<Telemetry, Micro, Stored>(slot, center, k, result);
			continue;
		}

		const uint32_t first = node._first;
		const uint32_t count = node.count();
		if constexpr (Stored)
			_store->touchNodes(first, count);
		for (uint32_t c = 0; c < count; ++c)
		{
			const HotNode& child = hot[first + c];
			if (!child.isInternal() && child.count() == 0)
				continue;

			if constexpr (Telemetry)
				++result._stats._testedBoxes;
			const float childDistance = boxDistanceSquared(child._lo, child._hi, center);
			if (best.size() == k && childDistance > best.front().first)
				continue;

			if constexpr (Prefetch)
				prefetchChild(child, hot, _orderedX.data(), _orderedY.data(), _orderedZ.data(), _pointOrder.data());
			pending.push_back({ childDistance, sequence++, first + c });
			std::push_heap(pending.begin(), pending.end(), KnnNodeCandidateGreater{});
		}
	}
}

// Depth-first kNN, nearest child first, for small k. At each internal node the surviving
// children are ordered by lower-bound distance and pushed farthest-first, so the nearest subtree
// is searched first and tightens the k-th distance before its siblings are reconsidered; a node
// is skipped when popped if its bound no longer beats that distance. This may enter a few nodes
// best-first would not, but trades the global heap for a stack and a sort of at most 64 entries.
// The answer is identical: the prune is strict and (distance, index) is a total order, so the k
// best are the same whatever order the leaves are offered in.
template <bool Telemetry, bool Micro, bool Charged, bool Stored, bool Prefetch>
void PointSpatialIndex::knnSearchDepthFirst(const glm::vec3& center, size_t k, QueryResult& result) const
{
	std::vector<std::pair<float, size_t>>& best = knnBestScratch();
	std::vector<std::pair<float, uint32_t>>& stack = knnStackScratch();
	stack.clear();
	const HotNode* hot = _hot.data();

	if constexpr (Stored)
		_store->touchNodes(0, 1);
	if constexpr (Telemetry)
		++result._stats._testedBoxes;
	stack.emplace_back(boxDistanceSquared(hot[0]._lo, hot[0]._hi, center), 0u);

	std::pair<float, uint32_t> children[64];
	while (!stack.empty())
	{
		const auto [distance, slot] = stack.back();
		stack.pop_back();
		if (best.size() == k && distance > best.front().first)
			continue;

		const HotNode& node = hot[slot];
		if constexpr (Telemetry)
			recordNodeVisit(slot, result._stats);
		if constexpr (Charged)
			chargeNodeVisit();

		if (!node.isInternal())
		{
			knnLeaf<Telemetry, Micro, Stored>(slot, center, k, result);
			continue;
		}

		size_t survivors = 0;
		const uint32_t first = node._first;
		const uint32_t count = node.count();
		if constexpr (Stored)
			_store->touchNodes(first, count);
		for (uint32_t c = 0; c < count; ++c)
		{
			const HotNode& child = hot[first + c];
			if (!child.isInternal() && child.count() == 0)
				continue;
			if constexpr (Telemetry)
				++result._stats._testedBoxes;
			const float childDistance = boxDistanceSquared(child._lo, child._hi, center);
			if (best.size() == k && childDistance > best.front().first)
				continue;
			// Insertion into descending order (at most 64 children): the nearest ends up last, on
			// top of the stack. Ties keep child order, so the traversal is deterministic.
			size_t at = survivors++;
			while (at > 0 && children[at - 1].first < childDistance)
			{
				children[at] = children[at - 1];
				--at;
			}
			children[at] = { childDistance, first + c };
		}
		for (size_t i = 0; i < survivors; ++i)
		{
			if constexpr (Prefetch)
				prefetchChild(hot[children[i].second], hot, _orderedX.data(), _orderedY.data(), _orderedZ.data(), _pointOrder.data());
			stack.push_back(children[i]);
		}
	}
}

// A node whose points all coincide stays a leaf above its capacity, the only leaf such data
// allows: no split separates them. The tight-bounds splits (sliding-midpoint kd, tight-centre
// octree/quadtree) see it in their plan; every other split once it routes all points to one child.
static bool coincident(const AABB& points)
{
	const glm::vec3 size = points.size();
	return !(size.x > 0.0f) && !(size.y > 0.0f) && !(size.z > 0.0f);
}

void PointSpatialIndex::buildNode(std::unique_ptr<Node>& node, BuildScratch& scratch)
{
	const std::optional<ActiveLevel> activeLevel = activeLevelForNode(*node);
	if (!activeLevel.has_value())
		return;

	const SchemaLevelConfig& levelConfig = *activeLevel->_config;
	node->_schemaDepth = activeLevel->_schemaDepth;
	node->_type = levelConfig._type;

	if (!shouldSplit(*node, levelConfig))
		return;

	SplitPlan plan;
	makeSplitPlan(*node, resolvedLevelFor(levelConfig), plan, scratch);
	const size_t childCount = plan._childCount;
	if (childCount == 0)
		return;

	routePointsToChildren(*node, plan, scratch);

	size_t nonEmptyChildren = 0;
	for (size_t childIndex = 0; childIndex < childCount; ++childIndex)
	{
		if (scratch._childCounts[childIndex] > 0)
			++nonEmptyChildren;
	}

	if (nonEmptyChildren == 0)
		return;
	// Every point went to one child. If they all coincide no split will ever separate them, and with
	// open-ended depths of 48/24 the node would chain down to the cap; it stays a leaf instead.
	if (nonEmptyChildren == 1 && coincident(pointBounds(*node)))
		return;

	scratch._childOffsets.resize(childCount);
	size_t writeOffset = node->_pointOffset;
	for (size_t childIndex = 0; childIndex < childCount; ++childIndex)
	{
		scratch._childOffsets[childIndex] = writeOffset;
		writeOffset += static_cast<size_t>(scratch._childCounts[childIndex]);
	}

	for (size_t i = 0; i < node->_pointCount; ++i)
		scratch._tempOrder[scratch._childOffsets[scratch._childOf[i]]++] = _pointOrder[node->_pointOffset + i];

	writeOffset = node->_pointOffset;
	for (size_t childIndex = 0; childIndex < childCount; ++childIndex)
	{
		scratch._childOffsets[childIndex] = writeOffset;
		writeOffset += static_cast<size_t>(scratch._childCounts[childIndex]);
	}

	std::copy(
		scratch._tempOrder.begin() + static_cast<std::ptrdiff_t>(node->_pointOffset),
		scratch._tempOrder.begin() + static_cast<std::ptrdiff_t>(node->_pointOffset + node->_pointCount),
		_pointOrder.begin() + static_cast<std::ptrdiff_t>(node->_pointOffset));

	std::vector<std::unique_ptr<Node>> children;
	children.reserve(childCount);
	for (size_t childIndex = 0; childIndex < childCount; ++childIndex)
	{
		if (scratch._childCounts[childIndex] == 0 && _schema._buildPolicy._removeEmptyNodes)
			continue;

		std::unique_ptr<Node> child = std::make_unique<Node>();
		child->_bounds = scratch._childBounds[childIndex];
		child->_type = levelConfig._type;
		child->_depth = node->_depth + 1;
		child->_schemaDepth = activeLevel->_schemaDepth + 1;
		child->_pointOffset = scratch._childOffsets[childIndex];
		child->_pointCount = static_cast<size_t>(scratch._childCounts[childIndex]);
		children.push_back(std::move(child));
	}

	for (std::unique_ptr<Node>& child : children)
		buildNode(child, scratch);

	if (_schema._buildPolicy._collapseSingleChild && nonEmptyChildren == 1 && _schema._buildPolicy._removeEmptyNodes)
	{
		for (std::unique_ptr<Node>& child : children)
		{
			if (child->_pointCount > 0 || !child->_children.empty())
			{
				node = std::move(child);
				return;
			}
		}
	}

	node->_children = std::move(children);
}

void PointSpatialIndex::rebuildOrderedPointSoA()
{
	const size_t pointCount = _pointOrder.size();
	_orderedX.resize(pointCount);
	_orderedY.resize(pointCount);
	_orderedZ.resize(pointCount);

	if (!_cloud)
		return;

	const std::vector<PointPrimitive>& points = _cloud->points();
	for (size_t orderedIndex = 0; orderedIndex < pointCount; ++orderedIndex)
	{
		const glm::vec3& position = points[_pointOrder[orderedIndex]].position;
		_orderedX[orderedIndex] = position.x;
		_orderedY[orderedIndex] = position.y;
		_orderedZ[orderedIndex] = position.z;
	}
}

void PointSpatialIndex::computeNodeAggregates(Node* node)
{
	if (!node || !_cloud)
		return;

	node->_tightBounds = AABB();
	node->_subtreePointCount = 0;

	if (node->isLeaf())
	{
		node->_subtreePointCount = node->_pointCount;
		for (size_t i = 0; i < node->_pointCount; ++i)
		{
			const size_t orderedIndex = node->_pointOffset + i;
			node->_tightBounds.update(glm::vec3(_orderedX[orderedIndex], _orderedY[orderedIndex], _orderedZ[orderedIndex]));
		}

		if (node->_subtreePointCount == 0)
			node->_tightBounds = node->_bounds;
		return;
	}

	for (const std::unique_ptr<Node>& child : node->_children)
	{
		computeNodeAggregates(child.get());
		if (!child || child->_subtreePointCount == 0)
			continue;

		node->_subtreePointCount += child->_subtreePointCount;
		node->_tightBounds.update(child->_tightBounds);
	}

	if (node->_subtreePointCount == 0)
		node->_tightBounds = node->_bounds;
}

void PointSpatialIndex::buildLeafMicroIndexes(Node* node)
{
	if (!node)
		return;

	node->_microIndex.reset();
	if (node->isLeaf())
	{
		if (_schema._buildPolicy._enableLeafMicroIndexes &&
			node->_pointCount > _schema._buildPolicy._leafMicroIndexThreshold)
		{
			node->_microIndex = buildLeafMicroIndex(*node);
		}
		return;
	}

	for (const std::unique_ptr<Node>& child : node->_children)
		buildLeafMicroIndexes(child.get());
}

std::unique_ptr<PointSpatialIndex::LeafMicroIndex> PointSpatialIndex::buildLeafMicroIndex(const Node& node) const
{
	if (!_cloud || node._pointCount == 0)
		return {};

	std::unique_ptr<LeafMicroIndex> index = std::make_unique<LeafMicroIndex>();
	index->_bounds = node._tightBounds;

	const double targetCells = std::max(1.0, static_cast<double>(node._pointCount) / 32.0);
	const uint32_t side = static_cast<uint32_t>(std::clamp(
		static_cast<int>(std::ceil(std::cbrt(targetCells))),
		2,
		16));
	const glm::vec3 extent = index->_bounds.size();
	index->gridResolution = glm::uvec3(
		extent.x > static_cast<float>(EPSILON) ? side : 1u,
		extent.y > static_cast<float>(EPSILON) ? side : 1u,
		extent.z > static_cast<float>(EPSILON) ? side : 1u);
	const size_t cellCount =
		static_cast<size_t>(index->gridResolution.x) *
		static_cast<size_t>(index->gridResolution.y) *
		static_cast<size_t>(index->gridResolution.z);

	index->_gridOffsets.assign(cellCount + 1, 0u);
	index->_gridOrder.resize(node._pointCount);
	for (size_t i = 0; i < node._pointCount; ++i)
	{
		const uint32_t orderedIndex = static_cast<uint32_t>(node._pointOffset + i);
		const glm::vec3 position(_orderedX[orderedIndex], _orderedY[orderedIndex], _orderedZ[orderedIndex]);
		const size_t cell = gridChildIndex(position, index->_bounds, index->gridResolution);
		++index->_gridOffsets[cell + 1];
	}

	for (size_t cell = 0; cell < cellCount; ++cell)
		index->_gridOffsets[cell + 1] += index->_gridOffsets[cell];

	std::vector<uint32_t> writeOffsets = index->_gridOffsets;
	for (size_t i = 0; i < node._pointCount; ++i)
	{
		const uint32_t orderedIndex = static_cast<uint32_t>(node._pointOffset + i);
		const glm::vec3 position(_orderedX[orderedIndex], _orderedY[orderedIndex], _orderedZ[orderedIndex]);
		const size_t cell = gridChildIndex(position, index->_bounds, index->gridResolution);
		index->_gridOrder[writeOffsets[cell]++] = orderedIndex;
	}

	index->_kdOrder.resize(node._pointCount);
	for (size_t i = 0; i < node._pointCount; ++i)
		index->_kdOrder[i] = static_cast<uint32_t>(node._pointOffset + i);
	buildLeafMicroKdNode(*index, 0u, static_cast<uint32_t>(index->_kdOrder.size()), 0u);
	return index;
}

uint32_t PointSpatialIndex::buildLeafMicroKdNode(LeafMicroIndex& index, uint32_t begin, uint32_t end, uint32_t depth) const
{
	const uint32_t nodeIndex = static_cast<uint32_t>(index._kdNodes.size());
	index._kdNodes.push_back({});
	LeafMicroIndex::KdNode& kdNode = index._kdNodes.back();
	kdNode._begin = begin;
	kdNode._count = end - begin;

	for (uint32_t i = begin; i < end; ++i)
	{
		const uint32_t orderedIndex = index._kdOrder[i];
		kdNode._bounds.update(glm::vec3(_orderedX[orderedIndex], _orderedY[orderedIndex], _orderedZ[orderedIndex]));
	}

	constexpr uint32_t MicroKdLeafCapacity = 32;
	if (kdNode._count <= MicroKdLeafCapacity)
		return nodeIndex;

	const glm::uint axis = longestAxis(kdNode._bounds);
	const uint32_t mid = begin + kdNode._count / 2u;
	const auto& orderedCoord = (axis == 0) ? _orderedX : (axis == 1) ? _orderedY : _orderedZ;
	std::nth_element(
		index._kdOrder.begin() + begin,
		index._kdOrder.begin() + mid,
		index._kdOrder.begin() + end,
		[&](uint32_t left, uint32_t right) {
			const float leftValue = orderedCoord[left];
			const float rightValue = orderedCoord[right];
			if (leftValue == rightValue)
				return _pointOrder[left] < _pointOrder[right];
			return leftValue < rightValue;
		});

	index._kdNodes[nodeIndex]._left = buildLeafMicroKdNode(index, begin, mid, depth + 1u);
	index._kdNodes[nodeIndex]._right = buildLeafMicroKdNode(index, mid, end, depth + 1u);
	return nodeIndex;
}

std::optional<PointSpatialIndex::ActiveLevel> PointSpatialIndex::activeLevelForNode(const Node& node) const
{
	const size_t totalLevels = _schema.totalLevels();
	size_t schemaDepth = node._schemaDepth;

	while (schemaDepth < totalLevels)
	{
		const SchemaLevelConfig& levelConfig = _schema.levelForDepth(schemaDepth);
		if (matchesCondition(node, levelConfig._condition))
			return ActiveLevel{ &levelConfig, schemaDepth };

		schemaDepth = schemaBlockEndDepth(_schema, schemaDepth);
	}

	return std::nullopt;
}

bool PointSpatialIndex::matchesCondition(const Node& node, const SchemaLevelCondition& condition) const
{
	if (condition.empty())
		return true;

	const size_t pointCount = node._pointCount;
	if (belowMin(pointCount, condition._minPoints) || aboveMax(pointCount, condition._maxPoints))
		return false;

	const glm::vec3 extent = glm::max(node._bounds.size(), glm::vec3(0.0f));
	const double horizontalExtent = std::max({ static_cast<double>(extent.x), static_cast<double>(extent.y), EPSILON });
	const double heightRatio = static_cast<double>(extent.z) / horizontalExtent;
	if (belowMin(heightRatio, condition._minHeightRatio) || aboveMax(heightRatio, condition._maxHeightRatio))
		return false;

	const double volume = static_cast<double>(extent.x) * static_cast<double>(extent.y) * static_cast<double>(extent.z);
	const double density = volume > EPSILON ? static_cast<double>(pointCount) / volume : 0.0;
	if (belowMin(density, condition._minDensity) || aboveMax(density, condition._maxDensity))
		return false;

	const double extentX = static_cast<double>(extent.x);
	const double extentY = static_cast<double>(extent.y);
	const double extentZ = static_cast<double>(extent.z);
	if (belowMin(extentX, condition._minExtentX) || aboveMax(extentX, condition._maxExtentX))
		return false;
	if (belowMin(extentY, condition._minExtentY) || aboveMax(extentY, condition._maxExtentY))
		return false;
	if (belowMin(extentZ, condition._minExtentZ) || aboveMax(extentZ, condition._maxExtentZ))
		return false;

	if (condition._minAnisotropy.has_value() || condition._maxAnisotropy.has_value())
	{
		// Anisotropy 1 - shortExtent / longExtent in [0, 1]; skip when the bbox is degenerate.
		const double minE = std::min({ extentX, extentY, extentZ });
		const double maxE = std::max({ extentX, extentY, extentZ });
		if (maxE > EPSILON)
		{
			const double anisotropy = 1.0 - (minE / maxE);
			if (belowMin(anisotropy, condition._minAnisotropy) || aboveMax(anisotropy, condition._maxAnisotropy))
				return false;
		}
	}

	if (condition._minOccupancyEntropy.has_value() || condition._maxOccupancyEntropy.has_value())
	{
		// Shannon entropy over a 4x4x4 sub-grid, normalized to [0, 1]; skip when too few points.
		const size_t pointThreshold = 16;
		if (pointCount >= pointThreshold && extentX > EPSILON && extentY > EPSILON && extentZ > EPSILON)
		{
			constexpr int kDivisions = 4;
			constexpr int kCellCount = kDivisions * kDivisions * kDivisions;
			std::array<size_t, kCellCount> cells{};
			const glm::vec3 minBound = node._bounds.min();
			for (size_t i = 0; i < node._pointCount; ++i)
			{
				const uint32_t pointIndex = _pointOrder[node._pointOffset + i];
				const glm::vec3& position = _cloud->points()[pointIndex].position;
				int cx = static_cast<int>((static_cast<double>(position.x - minBound.x) / extentX) * kDivisions);
				int cy = static_cast<int>((static_cast<double>(position.y - minBound.y) / extentY) * kDivisions);
				int cz = static_cast<int>((static_cast<double>(position.z - minBound.z) / extentZ) * kDivisions);
				cx = std::clamp(cx, 0, kDivisions - 1);
				cy = std::clamp(cy, 0, kDivisions - 1);
				cz = std::clamp(cz, 0, kDivisions - 1);
				++cells[(cx * kDivisions + cy) * kDivisions + cz];
			}
			double entropy = 0.0;
			const double total = static_cast<double>(pointCount);
			for (const size_t cellCount : cells)
			{
				if (cellCount == 0)
					continue;
				const double probability = static_cast<double>(cellCount) / total;
				entropy -= probability * std::log(probability);
			}
			const double normalized = entropy / std::log(static_cast<double>(kCellCount));
			if (belowMin(normalized, condition._minOccupancyEntropy) || aboveMax(normalized, condition._maxOccupancyEntropy))
				return false;
		}
	}

	return true;
}

double PointSpatialIndex::nodeDensity(const Node& node) const
{
	const glm::vec3 extent = glm::max(node._bounds.size(), glm::vec3(0.0f));
	const double volume = static_cast<double>(extent.x) * static_cast<double>(extent.y) * static_cast<double>(extent.z);
	return volume > EPSILON ? static_cast<double>(node._pointCount) / volume : 0.0;
}

double PointSpatialIndex::nodeHeightRatio(const Node& node) const
{
	const glm::vec3 extent = glm::max(node._bounds.size(), glm::vec3(0.0f));
	const double horizontalExtent = std::max({ static_cast<double>(extent.x), static_cast<double>(extent.y), EPSILON });
	return static_cast<double>(extent.z) / horizontalExtent;
}

double PointSpatialIndex::nodeAnisotropy(const Node& node) const
{
	const glm::vec3 extent = glm::max(node._bounds.size(), glm::vec3(0.0f));
	const double minExtent = std::min({ static_cast<double>(extent.x), static_cast<double>(extent.y), static_cast<double>(extent.z) });
	const double maxExtent = std::max({ static_cast<double>(extent.x), static_cast<double>(extent.y), static_cast<double>(extent.z) });
	if (maxExtent <= EPSILON)
		return 0.0;
	return 1.0 - (minExtent / maxExtent);
}

size_t PointSpatialIndex::effectiveLeafCapacity(const Node& node, const SchemaLevelConfig& levelConfig) const
{
	const size_t baseCapacity = std::max<size_t>(1, levelConfig._leafCapacity > 0 ? levelConfig._leafCapacity : _schema._buildPolicy._leafCapacity);
	const AdaptiveLeafCapacityConfig& adaptive = levelConfig._adaptiveLeafCapacity;
	if (!adaptive._enabled)
		return baseCapacity;

	double capacity = static_cast<double>(baseCapacity) * adaptive._queryMixFactor;
	const Node* root = _root.get();
	if (root && adaptive._densityWeight != 0.0)
	{
		const double referenceDensity = nodeDensity(*root);
		const double currentDensity = nodeDensity(node);
		if (referenceDensity > EPSILON && currentDensity > EPSILON)
			capacity *= std::pow(clampedAdaptiveFactor(referenceDensity / currentDensity), adaptive._densityWeight);
	}

	if (root && adaptive._heightRatioWeight != 0.0)
	{
		const double referenceHeightRatio = std::max(nodeHeightRatio(*root), EPSILON);
		const double currentHeightRatio = std::max(nodeHeightRatio(node), EPSILON);
		capacity *= std::pow(clampedAdaptiveFactor(referenceHeightRatio / currentHeightRatio), adaptive._heightRatioWeight);
	}

	if (adaptive._anisotropyWeight != 0.0)
	{
		const double regularity = std::clamp(1.0 - nodeAnisotropy(node), 0.25, 1.0);
		capacity *= std::pow(regularity, adaptive._anisotropyWeight);
	}

	const size_t fallbackMin = std::max<size_t>(1, baseCapacity / 4);
	const size_t fallbackMax = baseCapacity > std::numeric_limits<size_t>::max() / 4
		? std::numeric_limits<size_t>::max()
		: std::max<size_t>(baseCapacity, baseCapacity * 4);
	const size_t minCapacity = adaptive._minCapacity > 0 ? adaptive._minCapacity : fallbackMin;
	const size_t maxCapacity = std::max(minCapacity, adaptive._maxCapacity > 0 ? adaptive._maxCapacity : fallbackMax);
	const double clamped = std::clamp(capacity, static_cast<double>(minCapacity), static_cast<double>(maxCapacity));
	return static_cast<size_t>(std::max<long long>(1, std::llround(clamped)));
}

bool PointSpatialIndex::shouldSplit(const Node& node, const SchemaLevelConfig& levelConfig) const
{
	const size_t maxDepth = _schema._buildPolicy._maxDepth > 0 ? _schema._buildPolicy._maxDepth : _schema.totalLevels();
	if (node._depth >= maxDepth || node._schemaDepth >= _schema.totalLevels())
		return false;

	const size_t leafCapacity = effectiveLeafCapacity(node, levelConfig);
	const size_t minPointsToSplit = levelConfig._minPrimitivesToSplit > 0 ? levelConfig._minPrimitivesToSplit : _schema._buildPolicy._minPrimitivesToSplit;

	return node._pointCount > leafCapacity && node._pointCount >= minPointsToSplit;
}

const PointSpatialIndex::ResolvedLevel& PointSpatialIndex::resolvedLevelFor(const SchemaLevelConfig& levelConfig) const
{
	// Active levels are always elements of _schema._levels (SchemaConfig::levelForDepth returns
	// references into it), so the offset is the level's slot.
	return _resolvedLevels[static_cast<size_t>(&levelConfig - _schema._levels.data())];
}

// Child bounds and the routing geometry for one node. Arithmetic is exactly the old childBounds /
// locateChild: real division in the grid cell index, (max + min) / 2 centres via AABB::center(),
// the kd median as element n/2. Trees must come out bit-identical, not merely equivalent.
// Children of `cell` cut at `center` on every axis with two subdivisions, in split3D's order
// (x outermost, z innermost), so locateInPlan's index formula addresses them unchanged.
static void splitCellAt(const AABB& cell, const glm::vec3& center, const glm::uvec3& subdivisions, AABB* out)
{
	size_t index = 0;
	for (glm::uint x = 0; x < subdivisions.x; ++x)
		for (glm::uint y = 0; y < subdivisions.y; ++y)
			for (glm::uint z = 0; z < subdivisions.z; ++z)
			{
				const glm::uvec3 slot(x, y, z);
				glm::vec3 low = cell.min(), high = cell.max();
				for (glm::uint axis = 0; axis < 3; ++axis)
				{
					if (subdivisions[axis] != 2)
						continue;
					if (slot[axis] == 0)
						high[axis] = center[axis];
					else
						low[axis] = center[axis];
				}
				out[index++] = AABB(low, high);
			}
}

AABB PointSpatialIndex::pointBounds(const Node& node) const
{
	const std::vector<PointPrimitive>& points = _cloud->points();
	glm::vec3 low(std::numeric_limits<float>::max()), high(-std::numeric_limits<float>::max());
	for (size_t i = 0; i < node._pointCount; ++i)
	{
		const glm::vec3& position = points[_pointOrder[node._pointOffset + i]].position;
		low = glm::min(low, position);
		high = glm::max(high, position);
	}
	return AABB(low, high);
}

// Sort-Tile-Recursive. Every part is defined by the same rule routing uses (a point belongs to
// part c = how many cuts are <= its coordinate), so the quantile cuts only steer the counts: a run
// of equal coordinates stays together and can leave a part empty (removed like any empty child).
void PointSpatialIndex::makeStrPlan(const Node& node, glm::uint slabs, SplitPlan& plan, BuildScratch& scratch) const
{
	std::vector<glm::vec3>& positions = scratch._strPositions;
	const std::vector<PointPrimitive>& points = _cloud->points();
	positions.resize(node._pointCount);
	for (size_t i = 0; i < node._pointCount; ++i)
		positions[i] = points[_pointOrder[node._pointOffset + i]].position;

	plan._slabs = slabs;
	plan._childCount = static_cast<size_t>(slabs) * slabs * slabs;
	float* const xCuts = plan._cuts.data();
	float* const yCuts = xCuts + (slabs - 1);
	float* const zCuts = yCuts + slabs * (slabs - 1);

	// Cuts [begin, end) along `axis` at its quantiles and groups it part by part; parts[c] is
	// where part c starts. An empty range gets cuts at the cell's low side (no point uses them).
	const auto tile = [&](size_t begin, size_t end, int axis, float* cuts, size_t* parts) {
		const auto less = [axis](const glm::vec3& a, const glm::vec3& b) { return a[axis] < b[axis]; };
		for (glm::uint c = 1; c < slabs; ++c)
		{
			if (end == begin)
			{
				cuts[c - 1] = node._bounds.min()[axis];
				continue;
			}
			const size_t k = begin + (end - begin) * c / slabs;
			std::nth_element(positions.begin() + begin, positions.begin() + k, positions.begin() + end, less);
			cuts[c - 1] = positions[k][axis];
		}
		parts[0] = begin;
		for (glm::uint c = 1; c < slabs; ++c)
		{
			const float cut = cuts[c - 1];
			const auto middle = std::partition(positions.begin() + parts[c - 1], positions.begin() + end,
				[axis, cut](const glm::vec3& p) { return p[axis] < cut; });
			parts[c] = static_cast<size_t>(middle - positions.begin());
		}
		parts[slabs] = end;
	};

	size_t xParts[5], yParts[5], zParts[5];
	tile(0, positions.size(), 0, xCuts, xParts);
	for (glm::uint i = 0; i < slabs; ++i)
	{
		tile(xParts[i], xParts[i + 1], 1, yCuts + i * (slabs - 1), yParts);
		for (glm::uint j = 0; j < slabs; ++j)
			tile(yParts[j], yParts[j + 1], 2, zCuts + (i * slabs + j) * (slabs - 1), zParts);
	}

	// Child cells: the parent's cell cut at the same values, x outermost as in split3D.
	std::vector<AABB>& bounds = scratch._childBounds;
	bounds.resize(plan._childCount);
	const glm::vec3 low = node._bounds.min(), high = node._bounds.max();
	const auto range = [](const float* cuts, glm::uint part, glm::uint parts, float lo, float hi) {
		const float a = part == 0 ? lo : std::clamp(cuts[part - 1], lo, hi);
		const float b = part + 1 == parts ? hi : std::clamp(cuts[part], lo, hi);
		return std::pair<float, float>(a, std::max(a, b));
	};
	size_t index = 0;
	for (glm::uint i = 0; i < slabs; ++i)
	{
		const auto [x0, x1] = range(xCuts, i, slabs, low.x, high.x);
		for (glm::uint j = 0; j < slabs; ++j)
		{
			const auto [y0, y1] = range(yCuts + i * (slabs - 1), j, slabs, low.y, high.y);
			for (glm::uint k = 0; k < slabs; ++k)
			{
				const auto [z0, z1] = range(zCuts + (i * slabs + j) * (slabs - 1), k, slabs, low.z, high.z);
				bounds[index++] = AABB(glm::vec3(x0, y0, z0), glm::vec3(x1, y1, z1));
			}
		}
	}
}

void PointSpatialIndex::makeSplitPlan(const Node& node, const ResolvedLevel& level, SplitPlan& plan, BuildScratch& scratch) const
{
	std::vector<AABB>& bounds = scratch._childBounds;
	bounds.clear();
	plan._split = level._split;

	switch (level._split)
	{
	case ResolvedLevel::Split::Grid:
	{
		plan._subdivisions = level._gridSubdivisions;
		plan._min = node._bounds.min();
		plan._extent = node._bounds.size();
		plan._childCount = static_cast<size_t>(plan._subdivisions.x) * plan._subdivisions.y * plan._subdivisions.z;
		bounds.resize(plan._childCount);
		node._bounds.split3D(plan._subdivisions, bounds.data());
		return;
	}
	case ResolvedLevel::Split::Octree:
		plan._childCount = 8;
		bounds.resize(8);
		if (level._tightCenter && node._pointCount > 0)
		{
			const AABB points = pointBounds(node);
			if (coincident(points))
			{
				plan._childCount = 0;
				return;
			}
			plan._center = points.center();
			splitCellAt(node._bounds, plan._center, glm::uvec3(2, 2, 2), bounds.data());
			return;
		}
		plan._center = node._bounds.center();
		node._bounds.split3D(glm::uvec3(2, 2, 2), bounds.data());
		return;
	case ResolvedLevel::Split::Quad:
	{
		plan._axis = level._quadIgnoredAxis < 0 ? shortestAxis(node._bounds) : static_cast<glm::uint>(level._quadIgnoredAxis);
		plan._subdivisions = glm::uvec3(2, 2, 2);
		plan._subdivisions[plan._axis] = 1;
		plan._childCount = 4;
		bounds.resize(4);
		if (level._tightCenter && node._pointCount > 0)
		{
			const AABB points = pointBounds(node);
			if (coincident(points))
			{
				plan._childCount = 0;
				return;
			}
			plan._center = points.center();
			splitCellAt(node._bounds, plan._center, plan._subdivisions, bounds.data());
			return;
		}
		plan._center = node._bounds.center();
		node._bounds.split3D(plan._subdivisions, bounds.data());
		return;
	}
	case ResolvedLevel::Split::Str:
		makeStrPlan(node, level._strSlabs, plan, scratch);
		return;
	case ResolvedLevel::Split::BinaryKd:
	case ResolvedLevel::Split::Binary:
		break;
	}

	plan._axis = longestAxis(node._bounds);
	plan._value = node._bounds.center()[plan._axis];
	if (level._split == ResolvedLevel::Split::BinaryKd)
	{
		if (level._kdPolicy == ResolvedLevel::KdPolicy::RoundRobin)
		{
			plan._axis = static_cast<glm::uint>(node._depth % 3);
			plan._value = node._bounds.center()[plan._axis];
		}
		else if (level._kdPolicy == ResolvedLevel::KdPolicy::Median && node._pointCount > 0)
		{
			std::vector<float>& coordinates = scratch._medianScratch;
			coordinates.resize(node._pointCount);
			const std::vector<PointPrimitive>& points = _cloud->points();
			for (size_t i = 0; i < node._pointCount; ++i)
				coordinates[i] = points[_pointOrder[node._pointOffset + i]].position[plan._axis];

			const size_t median = coordinates.size() / 2;
			std::nth_element(coordinates.begin(), coordinates.begin() + median, coordinates.end());
			plan._value = coordinates[median];
		}
		else if (level._kdPolicy == ResolvedLevel::KdPolicy::SlidingMidpoint && node._pointCount > 0)
		{
			// nanoflann's middle split. Its node box is the bounding box of the node's points, so
			// the cut is the midpoint of the points' extent on their longest side: strictly inside
			// the points whenever they spread, so both sides keep points and no level is spent
			// peeling a few off an edge (cutting the CELL's midpoint did that, and at depth 22
			// left Toronto 5M with 10x the points tested per query).
			const AABB points = pointBounds(node);
			const glm::vec3 spread = points.size();
			plan._axis = spread.x >= spread.y && spread.x >= spread.z ? 0u : spread.y >= spread.z ? 1u : 2u;
			const float low = points.min()[plan._axis];
			const float high = points.max()[plan._axis];
			// No spread on the longest side: every point coincides, no cut separates them, and
			// splitting on would only chain empty nodes down to the depth cap.
			if (!(high > low))
			{
				plan._childCount = 0;
				return;
			}
			// Points on the cut go left (<=); the midpoint stays below `high`, so the right side
			// keeps the points at the far edge.
			plan._value = std::min(low + (high - low) * 0.5f, std::nextafter(high, low));
		}
	}

	plan._childCount = 2;
	bounds.resize(2);
	node._bounds.split2D(plan._axis, plan._value, bounds.data());
}

namespace
{
	using Split = PointSpatialIndex::ResolvedLevel::Split;

	template <Split S>
	size_t locateInPlan(const glm::vec3& point, const PointSpatialIndex::SplitPlan& plan)
	{
		if constexpr (S == Split::Grid)
		{
			glm::uvec3 coordinates(0);
			for (glm::uint axis = 0; axis < 3; ++axis)
			{
				const uint32_t cells = plan._subdivisions[axis];
				if (cells <= 1 || plan._extent[axis] <= 0.0f)
					continue;

				const float normalized = (point[axis] - plan._min[axis]) / plan._extent[axis];
				const int raw = static_cast<int>(std::floor(normalized * static_cast<float>(cells)));
				coordinates[axis] = static_cast<glm::uint>(std::clamp(raw, 0, static_cast<int>(cells) - 1));
			}
			const size_t index = coordinates.x * plan._subdivisions.y * plan._subdivisions.z +
				coordinates.y * plan._subdivisions.z +
				coordinates.z;
			return std::min(index, plan._childCount - 1);
		}
		else if constexpr (S == Split::Octree)
		{
			const size_t x = point.x >= plan._center.x ? 1 : 0;
			const size_t y = point.y >= plan._center.y ? 1 : 0;
			const size_t z = point.z >= plan._center.z ? 1 : 0;
			return x * 4 + y * 2 + z;
		}
		else if constexpr (S == Split::Quad)
		{
			glm::uvec3 coordinates(0);
			for (glm::uint axis = 0; axis < 3; ++axis)
			{
				if (axis != plan._axis && point[axis] >= plan._center[axis])
					coordinates[axis] = 1;
			}
			const size_t index =
				coordinates.x * plan._subdivisions.y * plan._subdivisions.z +
				coordinates.y * plan._subdivisions.z +
				coordinates.z;
			return std::min(index, plan._childCount - 1);
		}
		else if constexpr (S == Split::BinaryKd)
		{
			return point[plan._axis] <= plan._value ? 0 : 1;
		}
		else if constexpr (S == Split::Str)
		{
			const glm::uint slabs = plan._slabs;
			const float* xCuts = plan._cuts.data();
			glm::uint i = 0;
			while (i + 1 < slabs && point.x >= xCuts[i])
				++i;
			const float* yCuts = xCuts + (slabs - 1) + i * (slabs - 1);
			glm::uint j = 0;
			while (j + 1 < slabs && point.y >= yCuts[j])
				++j;
			const float* zCuts = xCuts + (slabs - 1) + slabs * (slabs - 1) + (i * slabs + j) * (slabs - 1);
			glm::uint k = 0;
			while (k + 1 < slabs && point.z >= zCuts[k])
				++k;
			return (static_cast<size_t>(i) * slabs + j) * slabs + k;
		}
		else
		{
			return point[plan._axis] < plan._value ? 0 : 1;
		}
	}

	template <Split S>
	void routeWith(const PointSpatialIndex::SplitPlan& plan, const std::vector<PointPrimitive>& points,
				   const uint32_t* order, size_t count, uint8_t* childOf, uint32_t* counts)
	{
		for (size_t i = 0; i < count; ++i)
		{
			const size_t child = locateInPlan<S>(points[order[i]].position, plan);
			childOf[i] = static_cast<uint8_t>(child);
			++counts[child];
		}
	}
}

// Counting pass: each point is located exactly once, its child slot kept for the scatter pass.
void PointSpatialIndex::routePointsToChildren(const Node& node, const SplitPlan& plan, BuildScratch& scratch) const
{
	scratch._childCounts.assign(plan._childCount, 0u);
	const uint32_t* order = _pointOrder.data() + node._pointOffset;
	uint8_t* childOf = scratch._childOf.data();
	uint32_t* counts = scratch._childCounts.data();
	const std::vector<PointPrimitive>& points = _cloud->points();
	switch (plan._split)
	{
	case Split::Grid: routeWith<Split::Grid>(plan, points, order, node._pointCount, childOf, counts); break;
	case Split::Octree: routeWith<Split::Octree>(plan, points, order, node._pointCount, childOf, counts); break;
	case Split::Quad: routeWith<Split::Quad>(plan, points, order, node._pointCount, childOf, counts); break;
	case Split::BinaryKd: routeWith<Split::BinaryKd>(plan, points, order, node._pointCount, childOf, counts); break;
	case Split::Binary: routeWith<Split::Binary>(plan, points, order, node._pointCount, childOf, counts); break;
	case Split::Str: routeWith<Split::Str>(plan, points, order, node._pointCount, childOf, counts); break;
	}
}

namespace
{
	uint64_t avalanche64(uint64_t value)
	{
		value += 0x9E3779B97F4A7C15ull;
		value ^= value >> 30;
		value *= 0xBF58476D1CE4E5B9ull;
		value ^= value >> 27;
		value *= 0x94D049BB133111EBull;
		value ^= value >> 31;
		return value;
	}

	// Order-dependent fold of one field into a node's hash. Within a node the field order is
	// fixed, so position-sensitivity is wanted; ACROSS nodes the digests are summed instead.
	void foldDigest(uint64_t& hash, uint64_t value)
	{
		hash = avalanche64(hash ^ avalanche64(value));
	}

	// FNV-1a, not std::hash: the digest is compared across machines (Windows goldens replayed on
	// the Linux runner), and std::hash<std::string> differs between standard libraries.
	uint64_t stableStringHash(const std::string& value)
	{
		uint64_t hash = 0xCBF29CE484222325ull;
		for (const char c : value)
		{
			hash ^= static_cast<unsigned char>(c);
			hash *= 0x100000001B3ull;
		}
		return hash;
	}

	uint64_t floatBits(float value)
	{
		uint32_t bits = 0;
		std::memcpy(&bits, &value, sizeof(bits));
		return bits;
	}

	void foldBounds(uint64_t& hash, const AABB& bounds)
	{
		const glm::vec3 min = bounds.min();
		const glm::vec3 max = bounds.max();
		for (int axis = 0; axis < 3; ++axis)
		{
			foldDigest(hash, floatBits(min[axis]));
			foldDigest(hash, floatBits(max[axis]));
		}
	}
}

uint64_t PointSpatialIndex::structureDigest(bool geometryOnly) const
{
	uint64_t digest = 0;
	std::vector<NodeView> stack;
	if (const NodeView rootNode = root())
		stack.push_back(rootNode);

	while (!stack.empty())
	{
		const NodeView node = stack.back();
		stack.pop_back();

		uint64_t hash = 0;
		foldDigest(hash, node.isLeaf() ? 1u : 0u);
		foldDigest(hash, node.depth());
		if (!geometryOnly)
		{
			foldDigest(hash, node.schemaDepth());
			foldDigest(hash, static_cast<uint64_t>(node.type()));
			foldDigest(hash, stableStringHash(structureNameFor(node.schemaDepth(), node.type())));
		}
		foldDigest(hash, node.childCount());
		foldDigest(hash, node.subtreePointCount());
		foldDigest(hash, node.microIndex() ? 1u : 0u);
		foldBounds(hash, node.bounds());
		foldBounds(hash, node.tightBounds());

		if (node.isLeaf())
		{
			uint64_t leafIds = 0;
			for (size_t i = 0; i < node.pointCount(); ++i)
				accumulateResultChecksum(leafIds, _pointOrder[node.pointOffset() + i]);
			foldDigest(hash, leafIds);
		}

		digest += hash;
		for (size_t c = 0; c < node.childCount(); ++c)
			stack.push_back(node.child(c));
	}

	return digest;
}

// A subtree's points are one contiguous slice of _pointOrder: the build partitions each node's
// range among its children in child order, so a depth-first leaf walk visits exactly
// [_pointOffset, _pointOffset + _pointCount) front to back. One widening copy replaces a
// recursive descent that touched every node of the contained subtree.
void PointSpatialIndex::appendSubtreePoints(uint32_t slot, std::vector<size_t>& pointIndices) const
{
	const ColdNode& cold = _cold[slot];
	if (cold._subtreeCount == 0)
		return;

	const auto first = _pointOrder.begin() + static_cast<std::ptrdiff_t>(cold._subtreeOffset);
	pointIndices.insert(pointIndices.end(), first, first + static_cast<std::ptrdiff_t>(cold._subtreeCount));
}

std::string PointSpatialIndex::structureNameFor(size_t schemaDepth, MultiDataStructure::DataStructureLevel type) const
{
	if (!_schema._levels.empty() && _schema.totalLevels() > 0)
	{
		const SchemaLevelConfig& level = _schema.levelForDepth(std::min(schemaDepth, _schema.totalLevels() - 1));
		if (!level._typeName.empty())
			return level._typeName;
	}

	return Config::dataStructureLevelName(type);
}

std::string PointSpatialIndex::structureNameForNode(const Node& node) const
{
	return structureNameFor(node._schemaDepth, node._type);
}

// Resolves the node's structure name to a slot in the breakdown arrays, registering it on
// first sight. Build-time only: the query path just reads Node::_structureIndex.
uint32_t PointSpatialIndex::structureIndexForNode(const Node& node)
{
	const std::string name = structureNameForNode(node);
	std::vector<std::string>& names = *_structureNames;
	for (size_t i = 0; i < names.size(); ++i)
	{
		if (names[i] == name)
			return static_cast<uint32_t>(i);
	}

	if (names.size() >= QueryStats::MaxBreakdownStructures)
		return static_cast<uint32_t>(QueryStats::MaxBreakdownStructures - 1);

	names.push_back(name);
	return static_cast<uint32_t>(names.size() - 1);
}

void PointSpatialIndex::assignStructureIndexes(Node* node)
{
	if (!node)
		return;

	// A node's structure name depends only on its (clamped) schema depth whenever that level
	// names its type, which every loaded schema does; the cache keeps first-sight registration
	// order, so slot numbering is unchanged. Unnamed levels fall back to the per-node path.
	const size_t totalLevels = _schema.totalLevels();
	const size_t depthSlot = totalLevels > 0 ? std::min(node->_schemaDepth, totalLevels - 1) : 0;
	const bool cacheable = totalLevels > 0 && !_schema.levelForDepth(depthSlot)._typeName.empty();
	if (cacheable && _structureSlotBySchemaDepth[depthSlot] >= 0)
	{
		node->_structureIndex = static_cast<uint32_t>(_structureSlotBySchemaDepth[depthSlot]);
	}
	else
	{
		node->_structureIndex = structureIndexForNode(*node);
		if (cacheable)
			_structureSlotBySchemaDepth[depthSlot] = static_cast<int32_t>(node->_structureIndex);
	}
	for (const std::unique_ptr<Node>& child : node->_children)
		assignStructureIndexes(child.get());
}

template <bool Telemetry>
void PointSpatialIndex::rangeQueryMicroGrid(uint32_t slot, const LeafMicroIndex& micro, const AABB& bounds, QueryResult& result) const
{
	const glm::uvec3 minCell = gridCoordinates(bounds.min(), micro._bounds, micro.gridResolution);
	const glm::uvec3 maxCell = gridCoordinates(bounds.max(), micro._bounds, micro.gridResolution);
	const glm::vec3 minBound = bounds.min();
	const glm::vec3 maxBound = bounds.max();
	size_t tested = 0;

	for (uint32_t x = minCell.x; x <= maxCell.x; ++x)
	{
		for (uint32_t y = minCell.y; y <= maxCell.y; ++y)
		{
			for (uint32_t z = minCell.z; z <= maxCell.z; ++z)
			{
				const size_t cell = x * micro.gridResolution.y * micro.gridResolution.z + y * micro.gridResolution.z + z;
				const uint32_t begin = micro._gridOffsets[cell];
				const uint32_t end = micro._gridOffsets[cell + 1];
				tested += static_cast<size_t>(end - begin);
				for (uint32_t offset = begin; offset < end; ++offset)
				{
					const uint32_t orderedIndex = micro._gridOrder[offset];
					const float px = _orderedX[orderedIndex];
					const float py = _orderedY[orderedIndex];
					const float pz = _orderedZ[orderedIndex];
					if (px >= minBound.x && px <= maxBound.x &&
						py >= minBound.y && py <= maxBound.y &&
						pz >= minBound.z && pz <= maxBound.z)
					{
						result._pointIndices.push_back(_pointOrder[orderedIndex]);
					}
				}
			}
		}
	}

	if constexpr (Telemetry)
		recordTestedPoints(slot, tested, result._stats);
}

template <bool Telemetry>
void PointSpatialIndex::countRangeMicroGrid(uint32_t slot, const LeafMicroIndex& micro, const AABB& bounds, CountResult& result) const
{
	const glm::uvec3 minCell = gridCoordinates(bounds.min(), micro._bounds, micro.gridResolution);
	const glm::uvec3 maxCell = gridCoordinates(bounds.max(), micro._bounds, micro.gridResolution);
	const glm::vec3 minBound = bounds.min();
	const glm::vec3 maxBound = bounds.max();
	size_t tested = 0;
	size_t contained = 0;

	for (uint32_t x = minCell.x; x <= maxCell.x; ++x)
	{
		for (uint32_t y = minCell.y; y <= maxCell.y; ++y)
		{
			for (uint32_t z = minCell.z; z <= maxCell.z; ++z)
			{
				const size_t cell = x * micro.gridResolution.y * micro.gridResolution.z + y * micro.gridResolution.z + z;
				const uint32_t begin = micro._gridOffsets[cell];
				const uint32_t end = micro._gridOffsets[cell + 1];
				tested += static_cast<size_t>(end - begin);
				for (uint32_t offset = begin; offset < end; ++offset)
				{
					const uint32_t orderedIndex = micro._gridOrder[offset];
					const float px = _orderedX[orderedIndex];
					const float py = _orderedY[orderedIndex];
					const float pz = _orderedZ[orderedIndex];
					if (px >= minBound.x && px <= maxBound.x &&
						py >= minBound.y && py <= maxBound.y &&
						pz >= minBound.z && pz <= maxBound.z)
					{
						++contained;
					}
				}
			}
		}
	}

	if constexpr (Telemetry)
		recordTestedPoints(slot, tested, result._stats);
	result._count += contained;
}

template <bool Telemetry>
void PointSpatialIndex::radiusQueryMicroGrid(uint32_t slot, const LeafMicroIndex& micro, const glm::vec3& center, float radiusSquared, QueryResult& result) const
{
	const float radius = std::sqrt(radiusSquared);
	const AABB queryBounds(center - glm::vec3(radius), center + glm::vec3(radius));
	const glm::uvec3 minCell = gridCoordinates(queryBounds.min(), micro._bounds, micro.gridResolution);
	const glm::uvec3 maxCell = gridCoordinates(queryBounds.max(), micro._bounds, micro.gridResolution);
	size_t tested = 0;

	for (uint32_t x = minCell.x; x <= maxCell.x; ++x)
	{
		for (uint32_t y = minCell.y; y <= maxCell.y; ++y)
		{
			for (uint32_t z = minCell.z; z <= maxCell.z; ++z)
			{
				const glm::uvec3 coordinates(x, y, z);
				if (distanceSquaredToAABB(gridCellBounds(micro._bounds, micro.gridResolution, coordinates), center) > radiusSquared)
					continue;

				const size_t cell = x * micro.gridResolution.y * micro.gridResolution.z + y * micro.gridResolution.z + z;
				const uint32_t begin = micro._gridOffsets[cell];
				const uint32_t end = micro._gridOffsets[cell + 1];
				tested += static_cast<size_t>(end - begin);
				for (uint32_t offset = begin; offset < end; ++offset)
				{
					const uint32_t orderedIndex = micro._gridOrder[offset];
					const float dx = _orderedX[orderedIndex] - center.x;
					const float dy = _orderedY[orderedIndex] - center.y;
					const float dz = _orderedZ[orderedIndex] - center.z;
					if (dx * dx + dy * dy + dz * dz <= radiusSquared)
						result._pointIndices.push_back(_pointOrder[orderedIndex]);
				}
			}
		}
	}

	if constexpr (Telemetry)
		recordTestedPoints(slot, tested, result._stats);
}

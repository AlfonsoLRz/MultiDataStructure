#include "../../stdafx.h"
#include "PointSpatialIndex.h"

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

static bool isGridPrimitive(const SchemaLevelConfig& levelConfig)
{
	const SchemaPrimitiveKind kind = primitiveKindForLevel(levelConfig);
	return kind == SchemaPrimitiveKind::RegularGrid || kind == SchemaPrimitiveKind::HGrid;
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
	_structureNames = std::make_shared<std::vector<std::string>>();
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
	scratch._childCounts.reserve(64);
	scratch._childOffsets.reserve(64);
	scratch._childBounds.reserve(64);

	if (!cloud.empty())
		buildNode(_root, scratch);

	rebuildOrderedPointSoA();
	computeNodeAggregates(_root.get());
	buildLeafMicroIndexes(_root.get());
	assignStructureIndexes(_root.get());
}

PointSpatialIndex::Stats PointSpatialIndex::stats() const
{
	Stats result;
	collectStats(_root.get(), result);
	return result;
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
}

PointSpatialIndex::QueryResult PointSpatialIndex::rangeQuery(const AABB& bounds) const
{
	QueryResult result;
	result._pointIndices.reserve(_rangeReserveHint.load(std::memory_order_relaxed));

	const auto start = std::chrono::steady_clock::now();
	rangeQueryNode(_root.get(), bounds, result);
	const double elapsedMs = elapsedMilliseconds(start);

	raiseHint(_rangeReserveHint, result._pointIndices.size());
	result._stats._returnedPoints = result._pointIndices.size();
	result._stats._elapsedMs = elapsedMs;
	for (size_t pointIndex : result._pointIndices)
		accumulateResultChecksum(result._stats._resultChecksum, pointIndex);
	result._stats._breakdown._structureNames = _structureNames;
	return result;
}

PointSpatialIndex::CountResult PointSpatialIndex::countRange(const AABB& bounds) const
{
	CountResult result;

	const auto start = std::chrono::steady_clock::now();
	countRangeNode(_root.get(), bounds, result);
	const double elapsedMs = elapsedMilliseconds(start);

	result._stats._returnedPoints = result._count;
	result._stats._elapsedMs = elapsedMs;
	result._stats._breakdown._structureNames = _structureNames;
	return result;
}

PointSpatialIndex::QueryResult PointSpatialIndex::radiusQuery(const glm::vec3& center, float radius) const
{
	QueryResult result;
	result._pointIndices.reserve(_radiusReserveHint.load(std::memory_order_relaxed));

	const auto start = std::chrono::steady_clock::now();
	if (radius >= 0.0f)
		radiusQueryNode(_root.get(), center, radius * radius, result);
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
	struct KnnNodeCandidate
	{
		float								_distance = 0.0f;
		size_t								_sequence = 0;
		const PointSpatialIndex::Node*		_node = nullptr;
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

	// The three kNN working sets, reused across queries on the calling thread.
	//
	// These were std::priority_queue locals: `best` and `pending` allocated once per query with no
	// reserve (so they reallocated during fill), and `microPending` was constructed fresh for
	// EVERY micro-indexed leaf the traversal reached - an operator new/delete apiece to search a
	// bucket of at most MicroKdLeafCapacity points. Vector + heap algorithms give identical
	// ordering with clear() preserving capacity, so a warm thread does no allocation at all.
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
}

PointSpatialIndex::QueryResult PointSpatialIndex::knnQuery(const glm::vec3& center, size_t k) const
{
	QueryResult result;

	std::vector<std::pair<float, size_t>>& best = knnBestScratch();
	std::vector<KnnNodeCandidate>& pending = knnPendingScratch();
	std::vector<KnnMicroCandidate>& microPending = knnMicroPendingScratch();
	best.clear();
	pending.clear();
	// Clamped: k comes from a replayed trace and nothing bounds it against the cloud, so a
	// pathological k must not turn a reserve into a multi-gigabyte allocation.
	if (k > 0 && _root)
		best.reserve(std::min<size_t>(k, _root->_subtreePointCount));

	const auto start = std::chrono::steady_clock::now();

	if (_root && _cloud && k > 0 && _root->_subtreePointCount > 0)
	{
		// `best` is a max-heap on (distance, pointIndex) once it is full, exactly as the
		// std::priority_queue it replaces. Below k elements it is an unordered bucket: nothing
		// reads front() until size() == k, and make_heap runs at the moment it fills.
		const auto bestWorst = [&best]() -> const std::pair<float, size_t>& { return best.front(); };
		const auto offerPoint = [&best, k](float distance, const uint32_t* order, size_t orderedIndex) {
			if (best.size() < k)
			{
				best.emplace_back(distance, static_cast<size_t>(order[orderedIndex]));
				if (best.size() == k)
					std::make_heap(best.begin(), best.end());
				return;
			}

			const std::pair<float, size_t>& worst = best.front();
			// Fast reject before touching _pointOrder. The identity is only needed to break an
			// exact distance tie or to store an accepted point, so the common case - a candidate
			// strictly worse than the current kth - never pays the extra memory stream.
			if (distance > worst.first)
				return;

			const size_t pointIndex = static_cast<size_t>(order[orderedIndex]);
			if (distance < worst.first || pointIndex < worst.second)
			{
				std::pop_heap(best.begin(), best.end());
				best.back() = { distance, pointIndex };
				std::push_heap(best.begin(), best.end());
			}
		};

		size_t sequence = 0;
		pending.push_back({ distanceSquaredToAABB(_root->_tightBounds, center), sequence++, _root.get() });

		while (!pending.empty())
		{
			std::pop_heap(pending.begin(), pending.end(), KnnNodeCandidateGreater{});
			const KnnNodeCandidate candidate = pending.back();
			pending.pop_back();
			const Node* node = candidate._node;
			if (!node || node->_subtreePointCount == 0)
				continue;

			// `pending` is a min-heap on node lower-bound distance, so a candidate that fails this
			// test proves every remaining entry fails it too: their distances are all >= this one
			// and the kth-best distance only shrinks. Draining the rest with `continue` cost an
			// O(log n) sift-down per leftover node and, because recordNodeVisit used to run before
			// the test, inflated the visited-node telemetry with nodes that were never examined.
			if (best.size() == k && candidate._distance > bestWorst().first)
				break;

			recordNodeVisit(*node, result._stats);

			if (node->isLeaf())
			{
				if (node->_microIndex && node->_microIndex->hasKdTree())
				{
					const LeafMicroIndex& micro = *node->_microIndex;
					microPending.clear();
					microPending.push_back({ distanceSquaredToAABB(micro._kdNodes.front()._bounds, center), 0u });

					while (!microPending.empty())
					{
						std::pop_heap(microPending.begin(), microPending.end(), KnnMicroCandidateGreater{});
						const KnnMicroCandidate microCandidate = microPending.back();
						microPending.pop_back();
						if (best.size() == k && microCandidate._distance > bestWorst().first)
							break;

						const LeafMicroIndex::KdNode& microNode = micro._kdNodes[microCandidate._nodeIndex];
						if (microNode.isLeaf())
						{
							recordTestedPoints(*node, microNode._count, result._stats);
							const uint32_t microEnd = microNode._begin + microNode._count;
							for (uint32_t i = microNode._begin; i < microEnd; ++i)
							{
								const uint32_t orderedIndex = micro._kdOrder[i];
								const float dx = _orderedX[orderedIndex] - center.x;
								const float dy = _orderedY[orderedIndex] - center.y;
								const float dz = _orderedZ[orderedIndex] - center.z;
								offerPoint(dx * dx + dy * dy + dz * dz, _pointOrder.data(), orderedIndex);
							}
							continue;
						}

						for (const uint32_t childIndex : { microNode._left, microNode._right })
						{
							if (childIndex == std::numeric_limits<uint32_t>::max())
								continue;

							const float childDistance = distanceSquaredToAABB(micro._kdNodes[childIndex]._bounds, center);
							if (best.size() == k && childDistance > bestWorst().first)
								continue;
							microPending.push_back({ childDistance, childIndex });
							std::push_heap(microPending.begin(), microPending.end(), KnnMicroCandidateGreater{});
						}
					}
					continue;
				}

				recordTestedPoints(*node, node->_pointCount, result._stats);
				const size_t leafEnd = node->_pointOffset + node->_pointCount;
				for (size_t orderedIndex = node->_pointOffset; orderedIndex < leafEnd; ++orderedIndex)
				{
					const float dx = _orderedX[orderedIndex] - center.x;
					const float dy = _orderedY[orderedIndex] - center.y;
					const float dz = _orderedZ[orderedIndex] - center.z;
					offerPoint(dx * dx + dy * dy + dz * dz, _pointOrder.data(), orderedIndex);
				}
				continue;
			}

			for (const std::unique_ptr<Node>& child : node->_children)
			{
				if (!child || child->_subtreePointCount == 0)
					continue;

				const float childDistance = distanceSquaredToAABB(child->_tightBounds, center);
				if (best.size() == k && childDistance > bestWorst().first)
					continue;

				pending.push_back({ childDistance, sequence++, child.get() });
				std::push_heap(pending.begin(), pending.end(), KnnNodeCandidateGreater{});
			}
		}

		// Sorted, deterministic output. The comparator is a total order on (distance, pointIndex),
		// so this reproduces the previous drain-then-sort exactly; the intermediate `ordered`
		// vector it used to build is gone.
		std::sort(best.begin(), best.end(), [](const auto& left, const auto& right) {
			if (left.first == right.first)
				return left.second < right.second;
			return left.first < right.first;
		});

		result._pointIndices.reserve(best.size());
		for (const auto& entry : best)
			result._pointIndices.push_back(entry.second);
	}

	const double elapsedMs = elapsedMilliseconds(start);
	result._stats._returnedPoints = result._pointIndices.size();
	result._stats._elapsedMs = elapsedMs;
	for (size_t pointIndex : result._pointIndices)
		accumulateResultChecksum(result._stats._resultChecksum, pointIndex);
	result._stats._breakdown._structureNames = _structureNames;
	return result;
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

	float splitValue = 0.0f;
	glm::uint splitAxis = 0;
	childBounds(*node, levelConfig, splitValue, splitAxis, scratch._childBounds);
	const size_t childCount = scratch._childBounds.size();
	if (childCount == 0)
		return;

	scratch._childCounts.assign(childCount, 0u);
	for (size_t i = 0; i < node->_pointCount; ++i)
	{
		const uint32_t pointIndex = _pointOrder[node->_pointOffset + i];
		const glm::vec3& position = _cloud->points()[pointIndex].position;
		const size_t childIndex = locateChild(position, levelConfig, *node, splitValue, splitAxis, childCount);
		++scratch._childCounts[childIndex];
	}

	size_t nonEmptyChildren = 0;
	for (size_t childIndex = 0; childIndex < childCount; ++childIndex)
	{
		if (scratch._childCounts[childIndex] > 0)
			++nonEmptyChildren;
	}

	if (nonEmptyChildren == 0)
		return;

	scratch._childOffsets.resize(childCount);
	size_t writeOffset = node->_pointOffset;
	for (size_t childIndex = 0; childIndex < childCount; ++childIndex)
	{
		scratch._childOffsets[childIndex] = writeOffset;
		writeOffset += static_cast<size_t>(scratch._childCounts[childIndex]);
	}

	for (size_t i = 0; i < node->_pointCount; ++i)
	{
		const uint32_t pointIndex = _pointOrder[node->_pointOffset + i];
		const glm::vec3& position = _cloud->points()[pointIndex].position;
		const size_t childIndex = locateChild(position, levelConfig, *node, splitValue, splitAxis, childCount);
		scratch._tempOrder[scratch._childOffsets[childIndex]++] = pointIndex;
	}

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

void PointSpatialIndex::childBounds(const Node& node, const SchemaLevelConfig& levelConfig, float& splitValue, glm::uint& splitAxis, std::vector<AABB>& bounds) const
{
	bounds.clear();
	if (isGridPrimitive(levelConfig))
	{
		const glm::uvec3 subdivisions = gridSubdivisionsForLevel(levelConfig);
		const size_t childCount = static_cast<size_t>(subdivisions.x) * subdivisions.y * subdivisions.z;
		bounds.resize(childCount);
		node._bounds.split3D(subdivisions, bounds.data());
		return;
	}

	if (levelConfig._type == MultiDataStructure::DataStructureLevel::OctreeNode)
	{
		bounds.resize(8);
		node._bounds.split3D(glm::uvec3(2, 2, 2), bounds.data());
		return;
	}

	if (levelConfig._type == MultiDataStructure::DataStructureLevel::QuadTreeNode)
	{
		glm::uvec3 subdivisions(2, 2, 2);
		subdivisions[ignoredQuadTreeAxis(levelConfig, node._bounds)] = 1;
		bounds.resize(4);
		node._bounds.split3D(subdivisions, bounds.data());
		return;
	}

	splitAxis = longestAxis(node._bounds);
	splitValue = node._bounds.center()[splitAxis];
	if (levelConfig._type == MultiDataStructure::DataStructureLevel::KDTreeNode)
	{
		const std::string policy = normalizedAxisPolicy(levelConfig._axisPolicy.empty() ? std::string("median_longest_axis") : levelConfig._axisPolicy);
		if (policy == "roundrobin")
		{
			splitAxis = static_cast<glm::uint>(node._depth % 3);
			splitValue = node._bounds.center()[splitAxis];
		}
		else if (policy == "medianlongestaxis" && node._pointCount > 0)
		{
			std::vector<float> coordinates;
			coordinates.reserve(node._pointCount);
			for (size_t i = 0; i < node._pointCount; ++i)
			{
				const uint32_t pointIndex = _pointOrder[node._pointOffset + i];
				coordinates.push_back(_cloud->points()[pointIndex].position[splitAxis]);
			}

			const size_t median = coordinates.size() / 2;
			std::nth_element(coordinates.begin(), coordinates.begin() + median, coordinates.end());
			splitValue = coordinates[median];
		}
	}

	bounds.resize(2);
	node._bounds.split2D(splitAxis, splitValue, bounds.data());
}

size_t PointSpatialIndex::locateChild(const glm::vec3& point, const SchemaLevelConfig& levelConfig, const Node& node, float splitValue, glm::uint splitAxis, size_t numChildren) const
{
	if (isGridPrimitive(levelConfig))
		return std::min(gridChildIndex(point, node._bounds, gridSubdivisionsForLevel(levelConfig)), numChildren - 1);

	if (levelConfig._type == MultiDataStructure::DataStructureLevel::OctreeNode)
	{
		const glm::vec3 center = node._bounds.center();
		const size_t x = point.x >= center.x ? 1 : 0;
		const size_t y = point.y >= center.y ? 1 : 0;
		const size_t z = point.z >= center.z ? 1 : 0;
		return x * 4 + y * 2 + z;
	}

	if (levelConfig._type == MultiDataStructure::DataStructureLevel::QuadTreeNode)
	{
		const glm::uint planarAxis = ignoredQuadTreeAxis(levelConfig, node._bounds);
		const glm::vec3 center = node._bounds.center();
		glm::uvec3 subdivisions(2, 2, 2);
		subdivisions[planarAxis] = 1;
		glm::uvec3 coordinates(0);
		for (glm::uint axis = 0; axis < 3; ++axis)
		{
			if (axis == planarAxis)
				continue;

			if (point[axis] >= center[axis])
				coordinates[axis] = 1;
		}
		const size_t index =
			coordinates.x * subdivisions.y * subdivisions.z +
			coordinates.y * subdivisions.z +
			coordinates.z;
		return std::min(index, numChildren - 1);
	}

	if (levelConfig._type == MultiDataStructure::DataStructureLevel::KDTreeNode)
		return point[splitAxis] <= splitValue ? 0 : 1;
	return point[splitAxis] < splitValue ? 0 : 1;
}

void PointSpatialIndex::collectStats(const Node* node, Stats& stats) const
{
	if (!node)
		return;

	++stats._numNodes;
	stats._maxDepth = std::max(stats._maxDepth, node->_depth);

	if (node->_children.empty())
	{
		++stats._numLeaves;
		stats._numPoints += node->_pointCount;
		return;
	}

	for (const std::unique_ptr<Node>& child : node->_children)
		collectStats(child.get(), stats);
}

void PointSpatialIndex::appendSubtreePoints(const Node* node, std::vector<size_t>& pointIndices) const
{
	if (!node)
		return;

	if (node->isLeaf())
	{
		pointIndices.reserve(pointIndices.size() + node->_pointCount);
		for (size_t i = 0; i < node->_pointCount; ++i)
			pointIndices.push_back(_pointOrder[node->_pointOffset + i]);
		return;
	}

	for (const std::unique_ptr<Node>& child : node->_children)
		appendSubtreePoints(child.get(), pointIndices);
}

std::string PointSpatialIndex::structureNameForNode(const Node& node) const
{
	if (!_schema._levels.empty() && _schema.totalLevels() > 0)
	{
		const size_t schemaDepth = std::min(node._schemaDepth, _schema.totalLevels() - 1);
		const SchemaLevelConfig& level = _schema.levelForDepth(schemaDepth);
		if (!level._typeName.empty())
			return level._typeName;
	}

	return Config::dataStructureLevelName(node._type);
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

	node->_structureIndex = structureIndexForNode(*node);
	for (const std::unique_ptr<Node>& child : node->_children)
		assignStructureIndexes(child.get());
}

void PointSpatialIndex::rangeQueryMicroGrid(const Node& node, const AABB& bounds, QueryResult& result) const
{
	const LeafMicroIndex& micro = *node._microIndex;
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

	recordTestedPoints(node, tested, result._stats);
}

void PointSpatialIndex::countRangeMicroGrid(const Node& node, const AABB& bounds, CountResult& result) const
{
	const LeafMicroIndex& micro = *node._microIndex;
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

	recordTestedPoints(node, tested, result._stats);
	result._count += contained;
}

void PointSpatialIndex::radiusQueryMicroGrid(const Node& node, const glm::vec3& center, float radiusSquared, QueryResult& result) const
{
	const LeafMicroIndex& micro = *node._microIndex;
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

	recordTestedPoints(node, tested, result._stats);
}

void PointSpatialIndex::rangeQueryNode(const Node* node, const AABB& bounds, QueryResult& result) const
{
	if (!node || !_cloud)
		return;

	recordNodeVisit(*node, result._stats);
	if (node->_subtreePointCount == 0 || !node->_tightBounds.collides(bounds))
		return;

	if (containsAABB(bounds, node->_tightBounds))
	{
		recordFullyContainedNode(*node, result._stats);
		appendSubtreePoints(node, result._pointIndices);
		return;
	}

	if (node->isLeaf())
	{
		if (node->_microIndex && node->_microIndex->hasGrid())
		{
			rangeQueryMicroGrid(*node, bounds, result);
			return;
		}

		recordTestedPoints(*node, node->_pointCount, result._stats);
		const glm::vec3 minBound = bounds.min();
		const glm::vec3 maxBound = bounds.max();
		for (size_t i = 0; i < node->_pointCount; ++i)
		{
			const size_t orderedIndex = node->_pointOffset + i;
			const float x = _orderedX[orderedIndex];
			const float y = _orderedY[orderedIndex];
			const float z = _orderedZ[orderedIndex];
			if (x >= minBound.x && x <= maxBound.x &&
				y >= minBound.y && y <= maxBound.y &&
				z >= minBound.z && z <= maxBound.z)
			{
				result._pointIndices.push_back(_pointOrder[orderedIndex]);
			}
		}
		return;
	}

	for (const std::unique_ptr<Node>& child : node->_children)
		rangeQueryNode(child.get(), bounds, result);
}

void PointSpatialIndex::countRangeNode(const Node* node, const AABB& bounds, CountResult& result) const
{
	if (!node || !_cloud)
		return;

	recordNodeVisit(*node, result._stats);
	if (node->_subtreePointCount == 0 || !node->_tightBounds.collides(bounds))
		return;

	if (containsAABB(bounds, node->_tightBounds))
	{
		recordFullyContainedNode(*node, result._stats);
		result._count += node->_subtreePointCount;
		return;
	}

	if (node->isLeaf())
	{
		if (node->_microIndex && node->_microIndex->hasGrid())
		{
			countRangeMicroGrid(*node, bounds, result);
			return;
		}

		recordTestedPoints(*node, node->_pointCount, result._stats);
		const glm::vec3 minBound = bounds.min();
		const glm::vec3 maxBound = bounds.max();
		size_t contained = 0;
		for (size_t i = 0; i < node->_pointCount; ++i)
		{
			const size_t orderedIndex = node->_pointOffset + i;
			const float x = _orderedX[orderedIndex];
			const float y = _orderedY[orderedIndex];
			const float z = _orderedZ[orderedIndex];
			if (x >= minBound.x && x <= maxBound.x &&
				y >= minBound.y && y <= maxBound.y &&
				z >= minBound.z && z <= maxBound.z)
			{
				++contained;
			}
		}
		result._count += contained;
		return;
	}

	for (const std::unique_ptr<Node>& child : node->_children)
		countRangeNode(child.get(), bounds, result);
}

void PointSpatialIndex::radiusQueryNode(const Node* node, const glm::vec3& center, float radiusSquared, QueryResult& result) const
{
	if (!node || !_cloud)
		return;

	recordNodeVisit(*node, result._stats);
	if (distanceSquaredToAABB(node->_tightBounds, center) > radiusSquared)
		return;

	if (node->isLeaf())
	{
		if (node->_microIndex && node->_microIndex->hasGrid())
		{
			radiusQueryMicroGrid(*node, center, radiusSquared, result);
			return;
		}

		recordTestedPoints(*node, node->_pointCount, result._stats);
		for (size_t i = 0; i < node->_pointCount; ++i)
		{
			const size_t orderedIndex = node->_pointOffset + i;
			const float dx = _orderedX[orderedIndex] - center.x;
			const float dy = _orderedY[orderedIndex] - center.y;
			const float dz = _orderedZ[orderedIndex] - center.z;
			if (dx * dx + dy * dy + dz * dz <= radiusSquared)
				result._pointIndices.push_back(_pointOrder[orderedIndex]);
		}
		return;
	}

	for (const std::unique_ptr<Node>& child : node->_children)
		radiusQueryNode(child.get(), center, radiusSquared, result);
}

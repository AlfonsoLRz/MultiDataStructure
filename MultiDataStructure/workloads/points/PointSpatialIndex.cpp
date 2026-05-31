#include "../../stdafx.h"
#include "PointSpatialIndex.h"

namespace
{
	constexpr double EPSILON = 1e-9;

	glm::uint longestAxis(const AABB& bounds)
	{
		const glm::vec3 size = bounds.size();
		if (size.x >= size.y && size.x >= size.z)
			return 0;
		if (size.y >= size.z)
			return 1;
		return 2;
	}

	glm::uint shortestAxis(const AABB& bounds)
	{
		const glm::vec3 size = bounds.size();
		if (size.x <= size.y && size.x <= size.z)
			return 0;
		if (size.y <= size.z)
			return 1;
		return 2;
	}

	std::string normalizedAxisPolicy(std::string value)
	{
		value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
			return std::isspace(c) || c == '_' || c == '-';
		}), value.end());
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return value;
	}

	glm::uint ignoredQuadTreeAxis(const SchemaLevelConfig& levelConfig, const AABB& bounds)
	{
		const std::string policy = normalizedAxisPolicy(levelConfig.axisPolicy.empty() ? std::string("xy") : levelConfig.axisPolicy);
		if (policy == "yz" || policy == "ignorex" || policy == "x")
			return 0;
		if (policy == "xz" || policy == "ignorey" || policy == "y")
			return 1;
		if (policy == "ignoreshortest" || policy == "shortest")
			return shortestAxis(bounds);
		return 2;
	}

	bool containsPoint(const AABB& bounds, const glm::vec3& point)
	{
		const glm::vec3 min = bounds.min();
		const glm::vec3 max = bounds.max();
		return point.x >= min.x && point.x <= max.x &&
			   point.y >= min.y && point.y <= max.y &&
			   point.z >= min.z && point.z <= max.z;
	}

	bool containsAABB(const AABB& outer, const AABB& inner)
	{
		const glm::vec3 outerMin = outer.min();
		const glm::vec3 outerMax = outer.max();
		const glm::vec3 innerMin = inner.min();
		const glm::vec3 innerMax = inner.max();
		return innerMin.x >= outerMin.x && innerMax.x <= outerMax.x &&
			   innerMin.y >= outerMin.y && innerMax.y <= outerMax.y &&
			   innerMin.z >= outerMin.z && innerMax.z <= outerMax.z;
	}

	float distanceSquaredToAABB(const AABB& bounds, const glm::vec3& point)
	{
		const glm::vec3 min = bounds.min();
		const glm::vec3 max = bounds.max();
		const glm::vec3 clamped(
			std::clamp(point.x, min.x, max.x),
			std::clamp(point.y, min.y, max.y),
			std::clamp(point.z, min.z, max.z));
		return glm::length2(point - clamped);
	}

	double elapsedMilliseconds(const std::chrono::steady_clock::time_point& start)
	{
		return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	}

	size_t schemaBlockEndDepth(const SchemaConfig& schema, size_t schemaDepth)
	{
		size_t cumulative = 0;
		for (const SchemaLevelConfig& level : schema.levels)
		{
			cumulative += level.numLevels;
			if (schemaDepth < cumulative)
				return cumulative;
		}

		return schema.totalLevels();
	}

	bool belowMin(size_t value, const std::optional<size_t>& minValue)
	{
		return minValue.has_value() && value < minValue.value();
	}

	bool aboveMax(size_t value, const std::optional<size_t>& maxValue)
	{
		return maxValue.has_value() && value > maxValue.value();
	}

	bool belowMin(double value, const std::optional<double>& minValue)
	{
		return minValue.has_value() && value < minValue.value();
	}

	bool aboveMax(double value, const std::optional<double>& maxValue)
	{
		return maxValue.has_value() && value > maxValue.value();
	}

	SchemaPrimitiveKind primitiveKindForLevel(const SchemaLevelConfig& levelConfig)
	{
		try
		{
			if (!levelConfig.typeName.empty())
				return Config::parseSchemaPrimitiveKind(levelConfig.typeName);
		}
		catch (const std::exception&)
		{
		}

		return levelConfig.primitiveKind;
	}

	glm::uvec3 gridSubdivisionsForLevel(const SchemaLevelConfig& levelConfig)
	{
		const SchemaPrimitiveKind kind = primitiveKindForLevel(levelConfig);
		if (kind == SchemaPrimitiveKind::HGrid)
			return glm::uvec3(4, 4, 4);
		return glm::uvec3(3, 3, 3);
	}

	bool isGridPrimitive(const SchemaLevelConfig& levelConfig)
	{
		const SchemaPrimitiveKind kind = primitiveKindForLevel(levelConfig);
		return kind == SchemaPrimitiveKind::RegularGrid || kind == SchemaPrimitiveKind::HGrid;
	}

	size_t gridChildIndex(const glm::vec3& point, const AABB& bounds, const glm::uvec3& subdivisions)
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
}

void PointSpatialIndex::build(const PointCloud& cloud, const SchemaConfig& schema)
{
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("CPU point index currently supports up to 2^32 - 1 points.");

	_cloud = &cloud;
	_schema = schema;
	_pointOrder.resize(cloud.size());
	std::iota(_pointOrder.begin(), _pointOrder.end(), 0u);

	_root = std::make_unique<Node>();
	_root->bounds = cloud.bounds();
	_root->depth = 0;
	_root->schemaDepth = 0;
	_root->pointOffset = 0;
	_root->pointCount = cloud.size();

	if (!cloud.empty())
		buildNode(_root);

	computeNodeAggregates(_root.get());
}

PointSpatialIndex::Stats PointSpatialIndex::stats() const
{
	Stats result;
	collectStats(_root.get(), result);
	return result;
}

PointSpatialIndex::QueryResult PointSpatialIndex::rangeQuery(const AABB& bounds) const
{
	const auto start = std::chrono::steady_clock::now();
	QueryResult result;

	rangeQueryNode(_root.get(), bounds, result);

	result.stats.returnedPoints = result.pointIndices.size();
	result.stats.elapsedMs = elapsedMilliseconds(start);
	return result;
}

PointSpatialIndex::CountResult PointSpatialIndex::countRange(const AABB& bounds) const
{
	const auto start = std::chrono::steady_clock::now();
	CountResult result;

	countRangeNode(_root.get(), bounds, result);

	result.stats.returnedPoints = result.count;
	result.stats.elapsedMs = elapsedMilliseconds(start);
	return result;
}

PointSpatialIndex::QueryResult PointSpatialIndex::radiusQuery(const glm::vec3& center, float radius) const
{
	const auto start = std::chrono::steady_clock::now();
	QueryResult result;

	if (radius >= 0.0f)
		radiusQueryNode(_root.get(), center, radius * radius, result);

	result.stats.returnedPoints = result.pointIndices.size();
	result.stats.elapsedMs = elapsedMilliseconds(start);
	return result;
}

PointSpatialIndex::QueryResult PointSpatialIndex::knnQuery(const glm::vec3& center, size_t k) const
{
	const auto start = std::chrono::steady_clock::now();
	QueryResult result;

	if (_root && _cloud && k > 0 && _root->subtreePointCount > 0)
	{
		struct NodeCandidate
		{
			float distance = 0.0f;
			size_t sequence = 0;
			const Node* node = nullptr;
		};

		struct NodeCandidateGreater
		{
			bool operator()(const NodeCandidate& left, const NodeCandidate& right) const
			{
				if (left.distance == right.distance)
					return left.sequence > right.sequence;
				return left.distance > right.distance;
			}
		};

		std::priority_queue<std::pair<float, size_t>> best;
		std::priority_queue<NodeCandidate, std::vector<NodeCandidate>, NodeCandidateGreater> pending;
		size_t sequence = 0;
		pending.push({ distanceSquaredToAABB(_root->tightBounds, center), sequence++, _root.get() });

		while (!pending.empty())
		{
			const NodeCandidate candidate = pending.top();
			pending.pop();
			const Node* node = candidate.node;
			if (!node || node->subtreePointCount == 0)
				continue;

			++result.stats.visitedNodes;
			if (best.size() == k && candidate.distance > best.top().first)
				continue;

			if (node->isLeaf())
			{
				for (size_t i = 0; i < node->pointCount; ++i)
				{
					const uint32_t pointIndex = _pointOrder[node->pointOffset + i];
					++result.stats.testedPoints;
					const glm::vec3& position = _cloud->points()[pointIndex].position;
					const float distance = glm::length2(position - center);
					if (best.size() < k)
					{
						best.push({ distance, pointIndex });
						continue;
					}

					const std::pair<float, size_t>& worst = best.top();
					if (distance < worst.first || (distance == worst.first && pointIndex < worst.second))
					{
						best.pop();
						best.push({ distance, pointIndex });
					}
				}
				continue;
			}

			for (const std::unique_ptr<Node>& child : node->children)
			{
				if (!child || child->subtreePointCount == 0)
					continue;

				const float childDistance = distanceSquaredToAABB(child->tightBounds, center);
				if (best.size() == k && childDistance > best.top().first)
					continue;

				pending.push({ childDistance, sequence++, child.get() });
			}
		}

		std::vector<std::pair<float, size_t>> ordered;
		ordered.reserve(best.size());
		while (!best.empty())
		{
			ordered.push_back(best.top());
			best.pop();
		}

		std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
			if (left.first == right.first)
				return left.second < right.second;
			return left.first < right.first;
		});

		result.pointIndices.reserve(ordered.size());
		for (const auto& entry : ordered)
			result.pointIndices.push_back(entry.second);
	}

	result.stats.returnedPoints = result.pointIndices.size();
	result.stats.elapsedMs = elapsedMilliseconds(start);
	return result;
}

void PointSpatialIndex::buildNode(std::unique_ptr<Node>& node)
{
	const std::optional<ActiveLevel> activeLevel = activeLevelForNode(*node);
	if (!activeLevel.has_value())
		return;

	const SchemaLevelConfig& levelConfig = *activeLevel->config;
	node->schemaDepth = activeLevel->schemaDepth;
	node->type = levelConfig.type;

	if (!shouldSplit(*node, levelConfig))
		return;

	float splitValue = 0.0f;
	glm::uint splitAxis = 0;
	const std::vector<AABB> bounds = childBounds(*node, levelConfig, splitValue, splitAxis);
	std::vector<std::vector<uint32_t>> childPoints(bounds.size());

	for (size_t i = 0; i < node->pointCount; ++i)
	{
		const uint32_t pointIndex = _pointOrder[node->pointOffset + i];
		const glm::vec3& position = _cloud->points()[pointIndex].position;
		const size_t childIndex = locateChild(position, levelConfig, *node, splitValue, splitAxis, bounds.size());
		childPoints[childIndex].push_back(pointIndex);
	}

	size_t nonEmptyChildren = 0;
	for (size_t childIndex = 0; childIndex < childPoints.size(); ++childIndex)
	{
		if (!childPoints[childIndex].empty())
			++nonEmptyChildren;
	}

	if (nonEmptyChildren == 0)
		return;

	std::vector<size_t> childOffsets(childPoints.size(), node->pointOffset);
	size_t writeOffset = node->pointOffset;
	for (size_t childIndex = 0; childIndex < childPoints.size(); ++childIndex)
	{
		childOffsets[childIndex] = writeOffset;
		for (const uint32_t pointIndex : childPoints[childIndex])
			_pointOrder[writeOffset++] = pointIndex;
	}

	std::vector<std::unique_ptr<Node>> children;
	children.reserve(bounds.size());
	for (size_t childIndex = 0; childIndex < bounds.size(); ++childIndex)
	{
		if (childPoints[childIndex].empty() && _schema.buildPolicy.removeEmptyNodes)
			continue;

		std::unique_ptr<Node> child = std::make_unique<Node>();
		child->bounds = bounds[childIndex];
		child->type = levelConfig.type;
		child->depth = node->depth + 1;
		child->schemaDepth = activeLevel->schemaDepth + 1;
		child->pointOffset = childOffsets[childIndex];
		child->pointCount = childPoints[childIndex].size();
		buildNode(child);
		children.push_back(std::move(child));
	}

	if (_schema.buildPolicy.collapseSingleChild && nonEmptyChildren == 1 && _schema.buildPolicy.removeEmptyNodes)
	{
		for (std::unique_ptr<Node>& child : children)
		{
			if (child->pointCount > 0 || !child->children.empty())
			{
				node = std::move(child);
				return;
			}
		}
	}

	node->children = std::move(children);
}

void PointSpatialIndex::computeNodeAggregates(Node* node)
{
	if (!node || !_cloud)
		return;

	node->tightBounds = AABB();
	node->subtreePointCount = 0;

	if (node->isLeaf())
	{
		node->subtreePointCount = node->pointCount;
		for (size_t i = 0; i < node->pointCount; ++i)
		{
			const uint32_t pointIndex = _pointOrder[node->pointOffset + i];
			node->tightBounds.update(_cloud->points()[pointIndex].position);
		}

		if (node->subtreePointCount == 0)
			node->tightBounds = node->bounds;
		return;
	}

	for (const std::unique_ptr<Node>& child : node->children)
	{
		computeNodeAggregates(child.get());
		if (!child || child->subtreePointCount == 0)
			continue;

		node->subtreePointCount += child->subtreePointCount;
		node->tightBounds.update(child->tightBounds);
	}

	if (node->subtreePointCount == 0)
		node->tightBounds = node->bounds;
}

std::optional<PointSpatialIndex::ActiveLevel> PointSpatialIndex::activeLevelForNode(const Node& node) const
{
	const size_t totalLevels = _schema.totalLevels();
	size_t schemaDepth = node.schemaDepth;

	while (schemaDepth < totalLevels)
	{
		const SchemaLevelConfig& levelConfig = _schema.levelForDepth(schemaDepth);
		if (matchesCondition(node, levelConfig.condition))
			return ActiveLevel{ &levelConfig, schemaDepth };

		schemaDepth = schemaBlockEndDepth(_schema, schemaDepth);
	}

	return std::nullopt;
}

bool PointSpatialIndex::matchesCondition(const Node& node, const SchemaLevelCondition& condition) const
{
	if (condition.empty())
		return true;

	const size_t pointCount = node.pointCount;
	if (belowMin(pointCount, condition.minPoints) || aboveMax(pointCount, condition.maxPoints))
		return false;

	const glm::vec3 extent = glm::max(node.bounds.size(), glm::vec3(0.0f));
	const double horizontalExtent = std::max({ static_cast<double>(extent.x), static_cast<double>(extent.y), EPSILON });
	const double heightRatio = static_cast<double>(extent.z) / horizontalExtent;
	if (belowMin(heightRatio, condition.minHeightRatio) || aboveMax(heightRatio, condition.maxHeightRatio))
		return false;

	const double volume = static_cast<double>(extent.x) * static_cast<double>(extent.y) * static_cast<double>(extent.z);
	const double density = volume > EPSILON ? static_cast<double>(pointCount) / volume : 0.0;
	if (belowMin(density, condition.minDensity) || aboveMax(density, condition.maxDensity))
		return false;

	const double extentX = static_cast<double>(extent.x);
	const double extentY = static_cast<double>(extent.y);
	const double extentZ = static_cast<double>(extent.z);
	if (belowMin(extentX, condition.minExtentX) || aboveMax(extentX, condition.maxExtentX))
		return false;
	if (belowMin(extentY, condition.minExtentY) || aboveMax(extentY, condition.maxExtentY))
		return false;
	if (belowMin(extentZ, condition.minExtentZ) || aboveMax(extentZ, condition.maxExtentZ))
		return false;

	if (condition.minAnisotropy.has_value() || condition.maxAnisotropy.has_value())
	{
		// Per-node anisotropy proxy: 1 - shortExtent / longExtent over the bbox. Range [0, 1].
		// 0 = perfectly cubic, 1 = degenerate line. Skip the test when the bbox is degenerate
		// (e.g. zero-volume slice) — there's no meaningful aspect ratio to compare against the
		// threshold there, and gating on a singular ratio would cause condition cliffs.
		const double minE = std::min({ extentX, extentY, extentZ });
		const double maxE = std::max({ extentX, extentY, extentZ });
		if (maxE > EPSILON)
		{
			const double anisotropy = 1.0 - (minE / maxE);
			if (belowMin(anisotropy, condition.minAnisotropy) || aboveMax(anisotropy, condition.maxAnisotropy))
				return false;
		}
	}

	if (condition.minOccupancyEntropy.has_value() || condition.maxOccupancyEntropy.has_value())
	{
		// Per-node Shannon entropy over a 4x4x4 = 64-cell sub-grid of this node's bbox.
		// Normalized to [0, 1] by dividing by ln(64). 0 = all points in one sub-cell (perfectly
		// clustered), 1 = uniformly spread. Skip the test when the node has too few points to
		// estimate entropy meaningfully (a couple of points always gives near-zero entropy and
		// would trigger maxOccupancyEntropy gates spuriously).
		const size_t pointThreshold = 16;
		if (pointCount >= pointThreshold && extentX > EPSILON && extentY > EPSILON && extentZ > EPSILON)
		{
			constexpr int kDivisions = 4;
			constexpr int kCellCount = kDivisions * kDivisions * kDivisions;
			std::array<size_t, kCellCount> cells{};
			const glm::vec3 minBound = node.bounds.min();
			for (size_t i = 0; i < node.pointCount; ++i)
			{
				const uint32_t pointIndex = _pointOrder[node.pointOffset + i];
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
			if (belowMin(normalized, condition.minOccupancyEntropy) || aboveMax(normalized, condition.maxOccupancyEntropy))
				return false;
		}
	}

	return true;
}

bool PointSpatialIndex::shouldSplit(const Node& node, const SchemaLevelConfig& levelConfig) const
{
	const size_t maxDepth = _schema.buildPolicy.maxDepth > 0 ? _schema.buildPolicy.maxDepth : _schema.totalLevels();
	if (node.depth >= maxDepth || node.schemaDepth >= _schema.totalLevels())
		return false;

	const size_t leafCapacity = levelConfig.leafCapacity > 0 ? levelConfig.leafCapacity : _schema.buildPolicy.leafCapacity;
	const size_t minPointsToSplit = levelConfig.minPrimitivesToSplit > 0 ? levelConfig.minPrimitivesToSplit : _schema.buildPolicy.minPrimitivesToSplit;

	return node.pointCount > leafCapacity && node.pointCount >= minPointsToSplit;
}

std::vector<AABB> PointSpatialIndex::childBounds(const Node& node, const SchemaLevelConfig& levelConfig, float& splitValue, glm::uint& splitAxis) const
{
	if (isGridPrimitive(levelConfig))
	{
		const glm::uvec3 subdivisions = gridSubdivisionsForLevel(levelConfig);
		const size_t childCount = static_cast<size_t>(subdivisions.x) * subdivisions.y * subdivisions.z;
		std::vector<AABB> children(childCount);
		node.bounds.split3D(subdivisions, children.data());
		return children;
	}

	if (levelConfig.type == MultiDataStructure::DataStructureLevel::OctreeNode)
	{
		AABB children[8];
		node.bounds.split3D(glm::uvec3(2, 2, 2), children);
		return std::vector<AABB>(std::begin(children), std::end(children));
	}

	if (levelConfig.type == MultiDataStructure::DataStructureLevel::QuadTreeNode)
	{
		AABB children[4];
		glm::uvec3 subdivisions(2, 2, 2);
		subdivisions[ignoredQuadTreeAxis(levelConfig, node.bounds)] = 1;
		node.bounds.split3D(subdivisions, children);
		return std::vector<AABB>(std::begin(children), std::end(children));
	}

	splitAxis = longestAxis(node.bounds);
	splitValue = node.bounds.center()[splitAxis];
	if (levelConfig.type == MultiDataStructure::DataStructureLevel::KDTreeNode && node.pointCount > 0)
	{
		std::vector<float> coordinates;
		coordinates.reserve(node.pointCount);
		for (size_t i = 0; i < node.pointCount; ++i)
		{
			const uint32_t pointIndex = _pointOrder[node.pointOffset + i];
			coordinates.push_back(_cloud->points()[pointIndex].position[splitAxis]);
		}

		const size_t median = coordinates.size() / 2;
		std::nth_element(coordinates.begin(), coordinates.begin() + median, coordinates.end());
		splitValue = coordinates[median];
	}

	AABB children[2];
	node.bounds.split2D(splitAxis, splitValue, children);
	return std::vector<AABB>(std::begin(children), std::end(children));
}

size_t PointSpatialIndex::locateChild(const glm::vec3& point, const SchemaLevelConfig& levelConfig, const Node& node, float splitValue, glm::uint splitAxis, size_t numChildren) const
{
	if (isGridPrimitive(levelConfig))
		return std::min(gridChildIndex(point, node.bounds, gridSubdivisionsForLevel(levelConfig)), numChildren - 1);

	if (levelConfig.type == MultiDataStructure::DataStructureLevel::OctreeNode)
	{
		const glm::vec3 center = node.bounds.center();
		const size_t x = point.x >= center.x ? 1 : 0;
		const size_t y = point.y >= center.y ? 1 : 0;
		const size_t z = point.z >= center.z ? 1 : 0;
		return x * 4 + y * 2 + z;
	}

	if (levelConfig.type == MultiDataStructure::DataStructureLevel::QuadTreeNode)
	{
		const glm::uint planarAxis = ignoredQuadTreeAxis(levelConfig, node.bounds);
		const glm::vec3 center = node.bounds.center();
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

	return point[splitAxis] < splitValue ? 0 : 1;
}

void PointSpatialIndex::collectStats(const Node* node, Stats& stats) const
{
	if (!node)
		return;

	++stats.numNodes;
	stats.maxDepth = std::max(stats.maxDepth, node->depth);

	if (node->children.empty())
	{
		++stats.numLeaves;
		stats.numPoints += node->pointCount;
		return;
	}

	for (const std::unique_ptr<Node>& child : node->children)
		collectStats(child.get(), stats);
}

void PointSpatialIndex::appendSubtreePoints(const Node* node, std::vector<size_t>& pointIndices) const
{
	if (!node)
		return;

	if (node->isLeaf())
	{
		pointIndices.reserve(pointIndices.size() + node->pointCount);
		for (size_t i = 0; i < node->pointCount; ++i)
			pointIndices.push_back(_pointOrder[node->pointOffset + i]);
		return;
	}

	for (const std::unique_ptr<Node>& child : node->children)
		appendSubtreePoints(child.get(), pointIndices);
}

void PointSpatialIndex::rangeQueryNode(const Node* node, const AABB& bounds, QueryResult& result) const
{
	if (!node || !_cloud)
		return;

	++result.stats.visitedNodes;
	if (node->subtreePointCount == 0 || !node->tightBounds.collides(bounds))
		return;

	if (containsAABB(bounds, node->tightBounds))
	{
		++result.stats.fullyContainedNodes;
		appendSubtreePoints(node, result.pointIndices);
		return;
	}

	if (node->isLeaf())
	{
		for (size_t i = 0; i < node->pointCount; ++i)
		{
			const uint32_t pointIndex = _pointOrder[node->pointOffset + i];
			++result.stats.testedPoints;
			if (containsPoint(bounds, _cloud->points()[pointIndex].position))
				result.pointIndices.push_back(pointIndex);
		}
		return;
	}

	for (const std::unique_ptr<Node>& child : node->children)
		rangeQueryNode(child.get(), bounds, result);
}

void PointSpatialIndex::countRangeNode(const Node* node, const AABB& bounds, CountResult& result) const
{
	if (!node || !_cloud)
		return;

	++result.stats.visitedNodes;
	if (node->subtreePointCount == 0 || !node->tightBounds.collides(bounds))
		return;

	if (containsAABB(bounds, node->tightBounds))
	{
		++result.stats.fullyContainedNodes;
		result.count += node->subtreePointCount;
		return;
	}

	if (node->isLeaf())
	{
		for (size_t i = 0; i < node->pointCount; ++i)
		{
			const uint32_t pointIndex = _pointOrder[node->pointOffset + i];
			++result.stats.testedPoints;
			if (containsPoint(bounds, _cloud->points()[pointIndex].position))
				++result.count;
		}
		return;
	}

	for (const std::unique_ptr<Node>& child : node->children)
		countRangeNode(child.get(), bounds, result);
}

void PointSpatialIndex::radiusQueryNode(const Node* node, const glm::vec3& center, float radiusSquared, QueryResult& result) const
{
	if (!node || !_cloud)
		return;

	++result.stats.visitedNodes;
	if (distanceSquaredToAABB(node->tightBounds, center) > radiusSquared)
		return;

	if (node->isLeaf())
	{
		for (size_t i = 0; i < node->pointCount; ++i)
		{
			const uint32_t pointIndex = _pointOrder[node->pointOffset + i];
			++result.stats.testedPoints;
			const glm::vec3& position = _cloud->points()[pointIndex].position;
			if (glm::length2(position - center) <= radiusSquared)
				result.pointIndices.push_back(pointIndex);
		}
		return;
	}

	for (const std::unique_ptr<Node>& child : node->children)
		radiusQueryNode(child.get(), center, radiusSquared, result);
}

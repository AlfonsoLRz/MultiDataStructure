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

	bool containsPoint(const AABB& bounds, const glm::vec3& point)
	{
		const glm::vec3 min = bounds.min();
		const glm::vec3 max = bounds.max();
		return point.x >= min.x && point.x <= max.x &&
			   point.y >= min.y && point.y <= max.y &&
			   point.z >= min.z && point.z <= max.z;
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
}

void PointSpatialIndex::build(const PointCloud& cloud, const SchemaConfig& schema)
{
	if (cloud.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		throw std::runtime_error("CPU point index currently supports up to 2^32 - 1 points.");

	_cloud = &cloud;
	_schema = schema;

	_root = std::make_unique<Node>();
	_root->bounds = cloud.bounds();
	_root->depth = 0;
	_root->schemaDepth = 0;
	_root->pointIndices.resize(cloud.size());
	std::iota(_root->pointIndices.begin(), _root->pointIndices.end(), 0u);

	if (!cloud.empty())
		buildNode(_root);
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

	if (_root && k > 0)
	{
		std::priority_queue<std::pair<float, size_t>> best;
		knnQueryNode(_root.get(), center, k, best, result.stats);

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

	for (const uint32_t pointIndex : node->pointIndices)
	{
		const glm::vec3& position = _cloud->points()[pointIndex].position;
		const size_t childIndex = locateChild(position, levelConfig, *node, splitValue, splitAxis, bounds.size());
		childPoints[childIndex].push_back(pointIndex);
	}

	size_t nonEmptyChildren = 0;
	size_t onlyNonEmptyChild = 0;
	for (size_t childIndex = 0; childIndex < childPoints.size(); ++childIndex)
	{
		if (!childPoints[childIndex].empty())
		{
			++nonEmptyChildren;
			onlyNonEmptyChild = childIndex;
		}
	}

	if (nonEmptyChildren == 0)
		return;

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
		child->pointIndices = std::move(childPoints[childIndex]);
		buildNode(child);
		children.push_back(std::move(child));
	}

	if (_schema.buildPolicy.collapseSingleChild && nonEmptyChildren == 1 && _schema.buildPolicy.removeEmptyNodes)
	{
		for (std::unique_ptr<Node>& child : children)
		{
			if (!child->pointIndices.empty() || !child->children.empty())
			{
				node = std::move(child);
				return;
			}
		}
	}

	node->pointIndices.clear();
	node->children = std::move(children);
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

	const size_t pointCount = node.pointIndices.size();
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

	return true;
}

bool PointSpatialIndex::shouldSplit(const Node& node, const SchemaLevelConfig& levelConfig) const
{
	const size_t maxDepth = _schema.buildPolicy.maxDepth > 0 ? _schema.buildPolicy.maxDepth : _schema.totalLevels();
	if (node.depth >= maxDepth || node.schemaDepth >= _schema.totalLevels())
		return false;

	const size_t leafCapacity = levelConfig.leafCapacity > 0 ? levelConfig.leafCapacity : _schema.buildPolicy.leafCapacity;
	const size_t minPointsToSplit = levelConfig.minPrimitivesToSplit > 0 ? levelConfig.minPrimitivesToSplit : _schema.buildPolicy.minPrimitivesToSplit;

	return node.pointIndices.size() > leafCapacity && node.pointIndices.size() >= minPointsToSplit;
}

std::vector<AABB> PointSpatialIndex::childBounds(const Node& node, const SchemaLevelConfig& levelConfig, float& splitValue, glm::uint& splitAxis) const
{
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
		subdivisions[longestAxis(node.bounds)] = 1;
		node.bounds.split3D(subdivisions, children);
		return std::vector<AABB>(std::begin(children), std::end(children));
	}

	splitAxis = longestAxis(node.bounds);
	splitValue = node.bounds.center()[splitAxis];
	if (levelConfig.type == MultiDataStructure::DataStructureLevel::KDTreeNode && !node.pointIndices.empty())
	{
		std::vector<float> coordinates;
		coordinates.reserve(node.pointIndices.size());
		for (const uint32_t pointIndex : node.pointIndices)
			coordinates.push_back(_cloud->points()[pointIndex].position[splitAxis]);

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
		const glm::uint planarAxis = longestAxis(node.bounds);
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
		stats.numPoints += node->pointIndices.size();
		return;
	}

	for (const std::unique_ptr<Node>& child : node->children)
		collectStats(child.get(), stats);
}

void PointSpatialIndex::rangeQueryNode(const Node* node, const AABB& bounds, QueryResult& result) const
{
	if (!node || !_cloud)
		return;

	++result.stats.visitedNodes;
	if (!node->bounds.collides(bounds))
		return;

	if (node->isLeaf())
	{
		for (const uint32_t pointIndex : node->pointIndices)
		{
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
	if (!node->bounds.collides(bounds))
		return;

	if (node->isLeaf())
	{
		for (const uint32_t pointIndex : node->pointIndices)
		{
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
	if (distanceSquaredToAABB(node->bounds, center) > radiusSquared)
		return;

	if (node->isLeaf())
	{
		for (const uint32_t pointIndex : node->pointIndices)
		{
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

void PointSpatialIndex::knnQueryNode(const Node* node, const glm::vec3& center, size_t k, std::priority_queue<std::pair<float, size_t>>& best, QueryStats& stats) const
{
	if (!node || !_cloud || k == 0)
		return;

	++stats.visitedNodes;
	const float nodeDistance = distanceSquaredToAABB(node->bounds, center);
	if (best.size() == k && nodeDistance > best.top().first)
		return;

	if (node->isLeaf())
	{
		for (const uint32_t pointIndex : node->pointIndices)
		{
			++stats.testedPoints;
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
		return;
	}

	std::vector<std::pair<float, const Node*>> children;
	children.reserve(node->children.size());
	for (const std::unique_ptr<Node>& child : node->children)
	{
		if (child)
			children.push_back({ distanceSquaredToAABB(child->bounds, center), child.get() });
	}

	std::sort(children.begin(), children.end(), [](const auto& left, const auto& right) {
		return left.first < right.first;
	});

	for (const auto& child : children)
	{
		if (best.size() == k && child.first > best.top().first)
			break;

		knnQueryNode(child.second, center, k, best, stats);
	}
}

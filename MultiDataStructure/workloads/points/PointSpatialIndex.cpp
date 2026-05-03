#include "../../stdafx.h"
#include "PointSpatialIndex.h"

namespace
{
	glm::uint longestAxis(const AABB& bounds)
	{
		const glm::vec3 size = bounds.size();
		if (size.x >= size.y && size.x >= size.z)
			return 0;
		if (size.y >= size.z)
			return 1;
		return 2;
	}
}

void PointSpatialIndex::build(const PointCloud& cloud, const SchemaConfig& schema)
{
	_cloud = &cloud;
	_schema = schema;

	_root = std::make_unique<Node>();
	_root->bounds = cloud.bounds();
	_root->depth = 0;
	_root->pointIndices.resize(cloud.size());
	std::iota(_root->pointIndices.begin(), _root->pointIndices.end(), 0);

	if (!cloud.empty())
		buildNode(_root);
}

PointSpatialIndex::Stats PointSpatialIndex::stats() const
{
	Stats result;
	collectStats(_root.get(), result);
	return result;
}

void PointSpatialIndex::buildNode(std::unique_ptr<Node>& node)
{
	const SchemaLevelConfig& levelConfig = _schema.levelForDepth(node->depth);
	node->type = levelConfig.type;

	if (!shouldSplit(*node, levelConfig))
		return;

	float splitValue = 0.0f;
	glm::uint splitAxis = 0;
	const std::vector<AABB> bounds = childBounds(*node, levelConfig, splitValue, splitAxis);
	std::vector<std::vector<size_t>> childPoints(bounds.size());

	for (const size_t pointIndex : node->pointIndices)
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

bool PointSpatialIndex::shouldSplit(const Node& node, const SchemaLevelConfig& levelConfig) const
{
	const size_t maxDepth = _schema.buildPolicy.maxDepth > 0 ? _schema.buildPolicy.maxDepth : _schema.totalLevels();
	if (node.depth >= maxDepth || node.depth >= _schema.totalLevels())
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
		for (const size_t pointIndex : node.pointIndices)
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

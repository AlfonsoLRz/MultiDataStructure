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
		std::vector<size_t> pointIndices;
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

	void build(const PointCloud& cloud, const SchemaConfig& schema);
	const Node* root() const { return _root.get(); }
	Stats stats() const;

private:
	const PointCloud* _cloud = nullptr;
	SchemaConfig _schema;
	std::unique_ptr<Node> _root;

	void buildNode(std::unique_ptr<Node>& node);
	bool shouldSplit(const Node& node, const SchemaLevelConfig& levelConfig) const;
	std::vector<AABB> childBounds(const Node& node, const SchemaLevelConfig& levelConfig, float& splitValue, glm::uint& splitAxis) const;
	size_t locateChild(const glm::vec3& point, const SchemaLevelConfig& levelConfig, const Node& node, float splitValue, glm::uint splitAxis, size_t numChildren) const;
	void collectStats(const Node* node, Stats& stats) const;
};


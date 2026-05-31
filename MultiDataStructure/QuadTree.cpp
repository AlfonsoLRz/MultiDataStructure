#include "stdafx.h"
#include "QuadTree.h"

//

QuadTreeNode::QuadTreeNode(const AABB& aabb) : SpatialDSNode(aabb), _planarAxis(2)
{
}

std::unique_ptr<MultiDataStructure::SpatialDSNode> QuadTreeNode::copy(const AABB& aabb) const
{
	return std::make_unique<QuadTreeNode>(aabb);
}

//bool QuadTreeNode::intersects(const Ray& ray)
//{
//	return SpatialDSNode::intersects(ray);
//}

void QuadTreeNode::split(MultiDataStructure::DataStructureLevel nodeType)
{
	std::array<AABB, 4> aabbs;

	glm::uvec3 subdivisions{ 2, 2, 2 };
	subdivisions[_planarAxis] = 1;
	_aabb.split3D(subdivisions, aabbs.data());

	for (const auto& aabb : aabbs)
	{
		_children.push_back(NodeFactory::create(nodeType, aabb));
	}
}

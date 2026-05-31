#include "stdafx.h"
#include "QuadTree.h"

//

QuadTreeNode::QuadTreeNode(const AABB& aabb) : SpatialDSNode(aabb), _planarAxis(-1)
{
	_planarAxis = _aabb.size().x > _aabb.size().y and _aabb.size().x > _aabb.size().z ? 0 :
				  _aabb.size().y > _aabb.size().z ? 1 : 2;
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
	static AABB aabbs[4];

	glm::uvec3 subdivisions{ 2, 2, 2 };
	subdivisions[_planarAxis] = 1;
	_aabb.split3D(subdivisions, aabbs);

	for (const auto& aabb : aabbs)
	{
		_children.push_back(NodeFactory::create(nodeType, aabb));
	}
}

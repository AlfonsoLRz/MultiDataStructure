#include "stdafx.h"
#include "Octree.h"

OctreeNode::OctreeNode(const AABB& aabb) : MultiDataStructure::SpatialDSNode(aabb)
{
}

std::unique_ptr<MultiDataStructure::SpatialDSNode> OctreeNode::copy(const AABB& aabb) const
{
	return std::make_unique<OctreeNode>(aabb);
}

void OctreeNode::split(MultiDataStructure::DataStructureLevel nodeType)
{
	static AABB aabbs[8];
	_aabb.split3D({2, 2, 2}, aabbs);

	for (const auto& aabb : aabbs)
	{
		_children.push_back(NodeFactory::create(nodeType, aabb));
	}
}

BvhNode::BvhNode(const AABB& aabb) : MultiDataStructure::SpatialDSNode(aabb)
{
	_splitAxis = _aabb.size().x > _aabb.size().y and _aabb.size().x > _aabb.size().z ? 0 :
				 _aabb.size().y > _aabb.size().z ? 1 : 2;
}

std::unique_ptr<MultiDataStructure::SpatialDSNode> BvhNode::copy(const AABB& aabb) const
{
	return std::make_unique<BvhNode>(aabb);
}

void BvhNode::split(MultiDataStructure::DataStructureLevel nodeType)
{
	static AABB aabbs[2];
	_aabb.split2D(_splitAxis, aabbs);

	_children.resize(2);
	_children[0] = NodeFactory::create(nodeType, aabbs[0]);
	_children[1] = NodeFactory::create(nodeType, aabbs[1]);
}

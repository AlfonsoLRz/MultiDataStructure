#include "stdafx.h"
#include "Octree.h"

OctreeNode::OctreeNode(const AABB& aabb) : MultiDataStructure::SpatialDSNode(aabb)
{
}

OctreeNode* OctreeNode::copy(const AABB& aabb) const
{
	return new OctreeNode(aabb);
}

void OctreeNode::split(MultiDataStructure::DataStructureLevel nodeType)
{
	static AABB aabbs[8];
	_aabb.split3D({2, 2, 2}, aabbs);

	for (const auto& aabb : aabbs)
	{
		MultiDataStructure::SpatialDSNode* newNode = NodeFactory::create(nodeType, aabb);
		_children.push_back(newNode);
	}
}

BvhNode::BvhNode(const AABB& aabb) : MultiDataStructure::SpatialDSNode(aabb)
{
	_splitAxis = _aabb.size().x > _aabb.size().y and _aabb.size().x > _aabb.size().z ? 0 :
				 _aabb.size().y > _aabb.size().z ? 1 : 2;
}

BvhNode* BvhNode::copy(const AABB& aabb) const
{
	return new BvhNode(aabb);
}

void BvhNode::split(MultiDataStructure::DataStructureLevel nodeType)
{
	static AABB aabbs[2];
	_aabb.split2D(_splitAxis, aabbs);

	_children.resize(2);
	_children[0] = NodeFactory::create(nodeType, aabbs[0]);
	_children[1] = NodeFactory::create(nodeType, aabbs[1]);
}
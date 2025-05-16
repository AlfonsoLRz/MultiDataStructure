#include "stdafx.h"
#include "KdTree.h"

//

KdTreeNode::KdTreeNode(const AABB& aabb) : SpatialDSNode(aabb), _splitAxis(-1)
{
	_splitAxis = _aabb.size().x > _aabb.size().y and _aabb.size().x > _aabb.size().z ? 0 :
				 _aabb.size().y > _aabb.size().z ? 1 : 2;
}

KdTreeNode* KdTreeNode::copy(const AABB& aabb) const
{
	return new KdTreeNode(aabb);
}

void KdTreeNode::split(MultiDataStructure::DataStructureLevel nodeType)
{
	static AABB aabbs[2];

	// Reorder primitives according to the split axis
	std::ranges::sort(_primitives,
	                  [this](const Node* a, const Node* b)
	                  {
		                  return (a->_minPoint[_splitAxis] + a->_maxPoint[_splitAxis]) / 2.0f < (b->_minPoint[_splitAxis] + b->_maxPoint[_splitAxis]) / 2.0f;
	                  });

	float splitValue = (_primitives[_primitives.size() / 2]->_minPoint[_splitAxis] + _primitives[_primitives.size() / 2]->_maxPoint[_splitAxis]) / 2.0f;
	_aabb.split2D(_splitAxis, splitValue, aabbs);

	for (const auto& aabb: aabbs)
	{
		MultiDataStructure::SpatialDSNode* newNode = NodeFactory::create(nodeType, aabb);
		_children.push_back(newNode);
	}
}

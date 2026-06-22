#pragma once

#include "MultiDataStructure.h"

class KdTreeNode : public MultiDataStructure::SpatialDSNode
{
private:
	int	_splitAxis;

public:
	KdTreeNode(const AABB& aabb = AABB());

	virtual std::unique_ptr<MultiDataStructure::SpatialDSNode> copy(const AABB& aabb) const override;
	virtual void split(MultiDataStructure::DataStructureLevel nodeType) override;
};


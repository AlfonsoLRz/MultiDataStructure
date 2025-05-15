#pragma once

#include "MultiDataStructure.h"

class QuadTreeNode : public MultiDataStructure::SpatialDSNode
{
private:
	int _planarAxis;

public:
	QuadTreeNode(const AABB& aabb = AABB());

	virtual QuadTreeNode* copy(const AABB& aabb) const override;
	virtual void split(MultiDataStructure::DataStructureLevel nodeType) override;
};



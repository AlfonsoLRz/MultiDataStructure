#pragma once

#include "MultiDataStructure.h"

class OctreeNode : public MultiDataStructure::SpatialDSNode
{
public:
	OctreeNode(const AABB& aabb = AABB());

	virtual std::unique_ptr<MultiDataStructure::SpatialDSNode> copy(const AABB& aabb) const override;
	virtual void split(MultiDataStructure::DataStructureLevel nodeType) override;
};

class BvhNode : public MultiDataStructure::SpatialDSNode
{
private:
	int	_splitAxis;

public:
	BvhNode(const AABB& aabb = AABB());

	virtual std::unique_ptr<MultiDataStructure::SpatialDSNode> copy(const AABB& aabb) const override;
	virtual void split(MultiDataStructure::DataStructureLevel nodeType) override;
};

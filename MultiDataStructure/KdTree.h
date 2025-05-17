#pragma once

#include "MultiDataStructure.h"

class KdTreeNode : public MultiDataStructure::SpatialDSNode
{
private:
	int _splitAxis;

public:
	KdTreeNode(const AABB& aabb = AABB());

	virtual KdTreeNode* copy(const AABB& aabb) const override;
	virtual void split(MultiDataStructure::DataStructureLevel nodeType) override;
};

class KdTreePrimitive
{
public:
	virtual float centroid(glm::uint axis) const = 0;
	virtual float distanceToPoint(const glm::vec3& point) const = 0;
	virtual bool inBoundingBox(const AABB& aabb) const = 0;
};

//class KdTree
//{
//protected:
//	struct Node
//	{
//		AABB								_aabb;
//		Node*								_left;
//		Node*								_right;
//		std::vector<KdTreePrimitive*>		_primitives;
//
//		Node() : _left(nullptr), _right(nullptr) {}
//
//		void findClosest(const glm::vec3& point, KdTreePrimitive*& closest, float& minDistance) const
//		{
//			for (auto primitive : _primitives)
//			{
//				float distance = primitive->distanceToPoint(point);
//				if (distance < minDistance)
//				{
//					minDistance = distance;
//					closest = primitive;
//				}
//			}
//		}
//
//		bool hasChildren() const { return _left != nullptr; }
//
//		void split(glm::uint axis, float median)
//		{
//			_left = new Node;
//			_right = new Node;
//			_aabb.split(axis, _left->_aabb, _right->_aabb, median);
//		}
//	};
//
//protected:
//	glm::uint	_maxLevel;
//	Node* _root;
//
//protected:
//	void getClosestDistance(Node* node, const glm::vec2& point, KdTreePrimitive*& closest, float& minDistance) const;
//	void insert(Node* node, const std::vector<KdTreePrimitive*>& primitives, glm::uint level);
//	void sort(std::vector<KdTreePrimitive*>& primitives, glm::uint axis);
//
//public:
//	KdTree(const AABB& aabb, const glm::uint maxLevel);
//	~KdTree();
//
//	void empty();
//
//	float getClosestDistance(const glm::vec2& point, KdTreePrimitive*& closest) const;
//	void insert(const std::vector<KdTreePrimitive*>& primitives);
//};


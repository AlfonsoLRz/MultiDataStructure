#pragma once

#include "GPUStructs.h"

class GeometryUtils
{
public:
	enum class SortMethod : int
	{
		SortMethodMorton = 0,
		SortMethodKdTree = 1
	};

protected:
	static glm::uint mortonCode(const glm::vec3& point, const glm::vec3& minPoint, const glm::vec3& maxPoint);

public:
	static float triangleArea(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3);

	static Node* fillNodeBuffer(const VertexGPU* vertices, const glm::u32* indices, glm::uint numIndices);
	static void reorderNodes(Node*& nodes, glm::uint numNodes, SortMethod sortMethod);
};


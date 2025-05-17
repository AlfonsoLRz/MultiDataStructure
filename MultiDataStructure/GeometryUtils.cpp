#include "stdafx.h"
#include "GeometryUtils.h"

#include <libmorton/morton3D.h>

//

glm::uint GeometryUtils::mortonCode(const glm::vec3& point, const glm::vec3& minPoint, const glm::vec3& maxPoint)
{
	glm::vec3 normalizedPoint = (point - minPoint) / (maxPoint - minPoint);
	glm::uint x = static_cast<glm::uint>(normalizedPoint.x * 1023.0f);
	glm::uint y = static_cast<glm::uint>(normalizedPoint.y * 1023.0f);
	glm::uint z = static_cast<glm::uint>(normalizedPoint.z * 1023.0f);

	return libmorton::m3D_e_LUT<glm::uint, glm::uint>(x, y, z);
}

float GeometryUtils::triangleArea(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3)
{
	return glm::length(glm::cross(v2 - v1, v3 - v1)) / 2.0f;
}

Node* GeometryUtils::fillNodeBuffer(const VertexGPU* vertices, const glm::u32* indices, glm::uint numIndices)
{
	auto calculateBounds = [=](const glm::u32 index, glm::vec3* maxPoint, glm::vec3* minPoint)
	{
		*maxPoint = glm::max(vertices[indices[index + 0]]._position, glm::max(vertices[indices[index + 1]]._position, vertices[indices[index + 2]]._position));
		*minPoint = glm::min(vertices[indices[index + 0]]._position, glm::min(vertices[indices[index + 1]]._position, vertices[indices[index + 2]]._position));
	};

	Node* nodes = new Node[numIndices / 3];
	for (glm::uint i = 0; i < numIndices / 3; ++i)
	{
		glm::vec3 maxPoint, minPoint;
		calculateBounds(i * 3, &maxPoint, &minPoint);
		nodes[i]._maxPoint = maxPoint;
		nodes[i]._minPoint = minPoint;
		nodes[i]._triangleIndex = i * 3;
		nodes[i]._prevIndex1 = nodes[i]._prevIndex2 = INT_MAX;
		nodes[i]._meshIndex = 0;		// TODO
		nodes[i]._numTriangles = 1;
	}

	return nodes;
}

void GeometryUtils::reorderNodes(Node*& nodes, glm::uint numNodes, SortMethod sortMethod)
{
	if (sortMethod == SortMethod::SortMethodMorton)
	{
		std::vector<glm::uint> mortonCodes(numNodes);
		std::vector<glm::uint> indices(numNodes);

		for (glm::uint i = 0; i < numNodes; ++i)
		{
			mortonCodes[i] = mortonCode(nodes[i].centroid(), nodes[i]._minPoint, nodes[i]._maxPoint);
			indices[i] = i;
		}

		// Reorder indices based on morton codes
		std::ranges::sort(indices,
		                  [&mortonCodes](const glm::uint a, const glm::uint b)
		                  {
			                  return mortonCodes[a] < mortonCodes[b];
		                  });

		Node* reorderedNodes = new Node[numNodes];
		for (glm::uint i = 0; i < numNodes; ++i)
			reorderedNodes[i] = nodes[indices[i]];

		std::swap(nodes, reorderedNodes);
		delete[] reorderedNodes;
	}
}

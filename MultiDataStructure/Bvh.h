#pragma once

#include "AABB.h"
#include "GPUStructs.h"

class Bvh
{
protected:
	// Required but not ours (do not free by ourselves)
	VertexGPU*	_vertexBufferGPU;
	glm::u32*	_indexBufferGPU;
	MeshGPU*	_meshBufferGPU;
	Node*		_clusterBufferGPU;

	//
	AABB	_aabb;
	GLuint	_numIndices;
	GLuint	_numMeshes;

protected:
	Node* buildClusterBuffer(glm::u32* sortedFacesBufferGPU);
	void buildTree(Node* tempClusterBufferGP) const;
	glm::u32* computeMortonCodes() const;
	glm::u32* sortMortonCodes(glm::u32* mortonCodesBufferGPU) const;

public:
	Bvh();
	~Bvh();

	void build();
	void initialize(VertexGPU* vertexBuffer, glm::u32* indexBuffer, MeshGPU* meshBuffer, const AABB& aabb, glm::uint numIndices, glm::uint numMeshes);

	Node* getClusterBuffer() const { return _clusterBufferGPU; }
	glm::uint getNumNodes() const { return _numIndices / 3 * 2 - 1; }
};
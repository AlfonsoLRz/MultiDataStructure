#pragma once

#include "stdafx.h"
#include "AABB.h"
#include "ApplicationState.h"
#include "Bvh.h"
#include "Camera.h"
#include "Model3D.h"

class CudaTexture;
class FloatImage;
class Texture;

class SceneContent
{
public:
	std::vector<std::unique_ptr<Camera>>	_camera;
	std::vector<std::unique_ptr<Model3D>>	_model;
	AABB									_sceneAABB;

	//
	std::vector<VertexGPU>					_vertices;
	std::vector<uint32_t>					_indices;

	// GPU buffers
	VertexGPU*								_vertexBufferGPU;
	glm::u32*								_indexBufferGPU;
	glm::u32*								_emissiveIndexBufferGPU;
	MeshGPU*								_meshBufferGPU;

	float*									_cdfIndicesBuffer;

	std::vector<CudaTexture*>				_diffuseTextures;

	glm::uint								_numVertices;
	glm::uint								_numMeshes;
	glm::uint								_numTextures;
	glm::uint								_numTriangles;

private:
	static void buildCDFIndices(
		const std::vector<VertexGPU>& vertices, const std::vector<uint32_t>& indices, const std::vector<MeshGPU>& meshes, 
		std::vector<float>& triangleArea);

public:
	SceneContent();
	virtual ~SceneContent();

	void addNewCamera(ApplicationState* appState);
	void addNewModel(Model3D* model);
	void buildScenario();

	void gatherModelGPUData(bool clearData);

	VertexGPU* getVertices() { return _vertices.data(); }
	glm::u32* getIndices() { return _indices.data(); }

	size_t getNumVertices() const { return _numVertices; }
	size_t getNumTriangles() const { return _numTriangles; }
};

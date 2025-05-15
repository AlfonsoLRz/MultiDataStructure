#include "stdafx.h"
#include "SceneContent.h"

#include "Bvh.h"
#include "CudaHelper.h"
#include "GeometricUtilities.h"
#include "GPUStructs.h"
#include "TriangleMesh.h"


// ----------------------------- BUILD YOUR SCENARIO HERE -----------------------------------

void SceneContent::buildScenario()
{
	this->gatherModelGPUData(true);
}

void SceneContent::gatherModelGPUData(bool moveData)
{
	std::vector<MeshGPU> materials;
	std::vector<std::string> diffuseTexturePaths;

	for (auto& model : _model)
		model->gatherGPUData(_vertices, _indices, materials, diffuseTexturePaths);

	_numVertices = static_cast<GLuint>(_vertices.size());
	_numMeshes = static_cast<GLuint>(materials.size());
	_numTextures = static_cast<GLuint>(diffuseTexturePaths.size());
	_numTriangles = static_cast<GLuint>(_indices.size()) / 3;

	if (moveData)
		for (auto& model : _model)
			model->clearData();

	// Build CDFs
	std::vector<float> triangleArea;

	//buildCDFIndices(vertices, indices, materials, triangleArea);
	//CudaHelper::initializeBufferGPU(_cdfIndicesBuffer, triangleArea.size(), triangleArea.data());

	// To GPU
	CudaHelper::initializeBufferGPU(_vertexBufferGPU, _vertices.size(), _vertices.data());
	CudaHelper::initializeBufferGPU(_indexBufferGPU, _indices.size(), _indices.data());
	CudaHelper::initializeBufferGPU(_meshBufferGPU, materials.size(), materials.data());

	// Build BVH
	_bvh.initialize(
		_vertexBufferGPU, _indexBufferGPU, _meshBufferGPU,
		_sceneAABB, _numTriangles * 3, _numMeshes);
	_bvh.build();
}

Node* SceneContent::getBvhNodesExplicitly() const
{
	if (_bvh.getClusterBuffer() == nullptr)
	{
		std::cout << "BVH not built yet!" << std::endl;
		return nullptr;
	}

	size_t size = _bvh.getNumNodes();
	Node* bvhNodes = new Node[size];
	Node* bvhNodesPointer = _bvh.getClusterBuffer();
	CudaHelper::downloadBufferGPU(bvhNodesPointer, bvhNodes, size);

	return bvhNodes;
}


// ------------------------------------------------------------------------------------------


void SceneContent::buildCDFIndices(
	const std::vector<VertexGPU>& vertices, const std::vector<uint32_t>& indices, const std::vector<MeshGPU>& meshes, std::vector<float>& triangleArea
)
{
	float sumArea = .0f;
	triangleArea.resize(indices.size() / 3);

	#pragma omp parallel for reduction(+:sumArea)
	for (int idx = 0; idx < static_cast<int>(indices.size()); idx += 3)
	{
		float area = GeometricUtilities::triangleArea(
			vertices[indices[idx + 0]]._position,
			vertices[indices[idx + 1]]._position,
			vertices[indices[idx + 2]]._position);

		triangleArea[idx / 3] = area;
		sumArea += area;
	}

	for (auto& area : triangleArea)
		area /= sumArea;

	for (int i = 1; i < static_cast<int>(triangleArea.size()); ++i)
		triangleArea[i] += triangleArea[i - 1];
}

SceneContent::SceneContent() :
	_vertexBufferGPU(nullptr), _indexBufferGPU(nullptr), _emissiveIndexBufferGPU(nullptr), _meshBufferGPU(nullptr),
	_cdfIndicesBuffer(nullptr), 
	_numVertices(0), _numMeshes(0), _numTextures(0),
	_numTriangles(0)
{
}

SceneContent::~SceneContent()
{
    _camera.clear();
    _model.clear();

	CudaHelper::free(_vertexBufferGPU);
	CudaHelper::free(_indexBufferGPU);
	CudaHelper::free(_emissiveIndexBufferGPU);
	CudaHelper::free(_meshBufferGPU);
	CudaHelper::free(_cdfIndicesBuffer);
}

void SceneContent::addNewCamera(ApplicationState* appState)
{
    _camera.push_back(std::make_unique<Camera>(appState->_viewportSize.x, appState->_viewportSize.y, false));
}

void SceneContent::addNewModel(Model3D* model)
{
    _sceneAABB.update(model->getAABB());
    _model.push_back(std::unique_ptr<Model3D>(model));
}

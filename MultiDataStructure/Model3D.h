#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "AABB.h"
#include "GPUStructs.h"
#include "Material.h"

class Model3D
{
	friend class SceneContent;

public:
	class Component
	{
	public:
		bool					_enabled;
		std::string				_name;

		std::vector<VertexGPU>	_vertices;
		std::vector<uint32_t>	_indices;
		Material				_material;
		AABB					_aabb;

		float area() const;
	};

	struct HitInformation
	{
		bool		_hit;
		glm::vec3	_point, _normal;
		float		_distance;
		Material*	_material;
	};

protected:
	static const std::string				BINARY_EXTENSION;
	static std::unordered_set<std::string>	USED_NAMES;

protected:
	AABB						_aabb;
	std::vector<Component>		_components;
	glm::mat4					_modelMatrix;

protected:
	void calculateAABB();
	void clearData();
	void gatherGPUData(
		std::vector<VertexGPU>& vertices, 
		std::vector<uint32_t>& indices,
		std::vector<MeshGPU>& meshes,
		std::vector<std::string>& diffuseTextures);
	void loadModelBinaryFile(const std::string& path);
	void writeBinaryFile(const std::string& path);

public:
	Model3D();
	virtual ~Model3D();

	virtual AABB getAABB() const { return _aabb.dot(_modelMatrix); }
	std::vector<Component>& getComponents() { return _components; }
	std::vector<Material*> getMaterials();
	glm::mat4 getModelMatrix() const { return _modelMatrix; }
	Model3D* moveGeometryToOrigin(const glm::mat4& origMatrix = glm::mat4(1.0f), float maxScale = FLT_MAX);
	Model3D* overrideModelName();
	Model3D* setModelMatrix(const glm::mat4& modelMatrix) { _modelMatrix = modelMatrix; return this; }
	Model3D* setTriangleColor(const glm::vec3& color);
};


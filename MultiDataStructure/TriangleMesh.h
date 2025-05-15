#pragma once

#include <assimp/Importer.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Model3D.h"


class TriangleMesh : public Model3D
{
protected:
	Assimp::Importer		_assimpImporter;

protected:
	void printStats() const;
	static Component processMesh(aiMesh* mesh, const aiScene* scene, const std::string& folder);
	void processNode(const aiNode* node, const aiScene* scene, const std::string& folder);

public:
	TriangleMesh();
	~TriangleMesh() override;

	bool load(const std::string& filename);
};


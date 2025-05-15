#include "stdafx.h"
#include "TriangleMesh.h"

// Public methods

TriangleMesh::TriangleMesh()
= default;

TriangleMesh::~TriangleMesh()
{
}

bool TriangleMesh::load(const std::string& filename)
{
    std::string binaryFile = filename.substr(0, filename.find_last_of('.')) + BINARY_EXTENSION;

    if (std::filesystem::exists(binaryFile))
    {
        this->loadModelBinaryFile(binaryFile);
    }
    else
    {
        const aiScene* scene = _assimpImporter.ReadFile(filename, aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            std::cout << "ERROR::ASSIMP::" << _assimpImporter.GetErrorString() << std::endl;
            return false;
        }

        std::string shortName = aiScene::GetShortFilename(filename.c_str());
        std::string folder = filename.substr(0, filename.length() - shortName.length());

        this->processNode(scene->mRootNode, scene, folder);
        this->writeBinaryFile(binaryFile);
    }

    this->calculateAABB();
    this->printStats();

    return true;
}

// Protected methods

void TriangleMesh::printStats() const
{
	for (auto& component : _components)
	{
		std::cout << "Component: " << component._name << std::endl;
		std::cout << "Vertices: " << component._vertices.size() << std::endl;
		std::cout << "Indices: " << component._indices.size() / 3 << std::endl;
		std::cout << "AABB: " << component._aabb << std::endl;
        std::cout << "\n";
	}
}

Model3D::Component TriangleMesh::processMesh(aiMesh* mesh, const aiScene* scene, const std::string& folder)
{
    AABB aabb;
    std::vector<VertexGPU> vertices(mesh->mNumVertices);
    std::vector<GLuint> indices(mesh->mNumFaces * 3);
    int numVertices = static_cast<int>(mesh->mNumVertices);

    for (int i = 0; i < numVertices; i++)
    {
        VertexGPU vertex;
        vertex._position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
        vertex._normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        if (mesh->mTextureCoords[0]) vertex._textCoord = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);

        vertices[i] = vertex;
        aabb.update(vertex._position);
    }

    // Indices
    for (unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++)
            indices[i * 3 + j] = face.mIndices[j];
    }

    Component component;
	component._name = mesh->mName.C_Str();
    component._vertices = std::move(vertices);
    component._indices = std::move(indices);
    component._aabb = aabb;
	component._material = Material::createMaterial(scene->mMaterials[mesh->mMaterialIndex]);

    return component;
}

void TriangleMesh::processNode(const aiNode* node, const aiScene* scene, const std::string& folder)
{
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        _components.push_back(processMesh(mesh, scene, folder));
        _aabb.update(_components[_components.size() - 1]._aabb);
    }

    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        this->processNode(node->mChildren[i], scene, folder);
    }
}
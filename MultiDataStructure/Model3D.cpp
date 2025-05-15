#include "stdafx.h"
#include "Model3D.h"
#include "GeometricUtilities.h"

// Static properties

const std::string Model3D::BINARY_EXTENSION = ".bin";
std::unordered_set<std::string> Model3D::USED_NAMES;

// Public methods

Model3D::Model3D() : _modelMatrix(1.0f)
{
    this->overrideModelName();
}

Model3D::~Model3D()
= default;

Model3D::HitInformation Model3D::hit(const Ray& ray)
{
    return {};
}

std::vector<Material*> Model3D::getMaterials()
{
	std::vector<Material*> materials;
	for (auto& component : _components)
		materials.push_back(&component._material);
	return materials;
}

Model3D* Model3D::moveGeometryToOrigin(const glm::mat4& origMatrix, float maxScale)
{
    AABB aabb = this->getAABB();

    glm::vec3 translate = -aabb.center();
    glm::vec3 extent = aabb.extent();
    float maxScaleAABB = std::max(extent.x, std::max(extent.y, extent.z));
    glm::vec3 scale = (maxScale < FLT_MAX) ? ((maxScale > maxScaleAABB) ? glm::vec3(1.0f) : glm::vec3(maxScale / maxScaleAABB)) : glm::vec3(1.0f);

    _modelMatrix = glm::scale(glm::mat4(1.0f), scale) * glm::translate(glm::mat4(1.0f), translate) * origMatrix;

    return this;
}

Model3D* Model3D::overrideModelName()
{
    std::string className = typeid(*this).name();
    std::string classTarget = "class ";
    size_t classIndex = className.find(classTarget);
    if (classIndex != std::string::npos)
    {
        className = className.substr(classIndex + classTarget.size(), className.size() - classIndex - classTarget.size());
    }

    unsigned modelIdx = 0;
	for (Component& component : _components)
	{
        bool nameValid = false;
        while (!nameValid)
        {
            component._name = className + " " + std::to_string(modelIdx);
            nameValid = !USED_NAMES.contains(component._name);
            ++modelIdx;
        }

        USED_NAMES.insert(component._name);
	}

    return this;
}

Model3D* Model3D::setTriangleColor(const glm::vec3& color)
{
    for (auto& component : _components)
    {
        component._material.setDiffuseColor(color);
    }

    return this;
}

// Private methods

float Model3D::Component::area() const
{
	float area = 0.0f;
    for (int idx = 0; idx < static_cast<int>(_indices.size()); idx += 3)
    {
        area += GeometricUtilities::triangleArea(
            _vertices[_indices[idx + 0]]._position,
            _vertices[_indices[idx + 1]]._position,
            _vertices[_indices[idx + 2]]._position);
    }

	return area;
}

void Model3D::calculateAABB()
{
    _aabb = AABB();

    for (const Component& component : _components)
        for (const VertexGPU& vertex : component._vertices)
            _aabb.update(vertex._position);
}

void Model3D::clearData()
{
	for (auto& component : _components)
	{
		component._vertices.clear();
		component._indices.clear();
	}
}

void Model3D::gatherGPUData(std::vector<VertexGPU>& vertices, std::vector<uint32_t>& indices,
                            std::vector<MeshGPU>& meshes, std::vector<std::string>& diffuseTextures)
{
	for (auto& component : _components)
	{
		size_t numVertices = vertices.size(), startIndex = indices.size();

        // Vertices
        for (auto& vertex : component._vertices)
        {
            vertices.emplace_back(vertex);
			vertices.back()._position = glm::vec3(_modelMatrix * glm::vec4(vertex._position, 1.0f));
        }

        // Indices
        for (auto index : component._indices)
            indices.push_back(index + numVertices);

        // Textures
		if (!component._material.getDiffuseTexturePath().empty())
			diffuseTextures.push_back(component._material.getDiffuseTexturePath());

		// Mesh info
		MeshGPU mesh;
		mesh._startIndex = static_cast<glm::uint>(startIndex);
		mesh._length = static_cast<glm::uint>(indices.size() - startIndex);
		mesh._diffuseColour = component._material.getDiffuseColor();
		mesh._specularColor = component._material.getSpecularColor();
		mesh._emissionColor = component._material.getEmissionColor();
		mesh._emissionStrength = component._material.getEmissionStrength();
		mesh._metallic = component._material.getMetallic();
		mesh._smoothness = component._material.getSmoothness();
        mesh._textureIndex = component._material.getDiffuseTexturePath().empty() ? -1 : static_cast<int>(diffuseTextures.size() - 1);
		mesh._max = glm::vec3(_modelMatrix * glm::vec4(component._aabb.max(), 1.0f));
		mesh._min = glm::vec3(_modelMatrix * glm::vec4(component._aabb.min(), 1.0f));

		meshes.push_back(mesh);
	}
}

// Protected methods

void Model3D::loadModelBinaryFile(const std::string& path)
{
    std::ifstream fin(path, std::ios::in | std::ios::binary);
    if (!fin.is_open())
    {
        std::cout << "Failed to open the binary file " << path << "!" << std::endl;
        return;
    }

    size_t numComponents = _components.size();
    fin.read(reinterpret_cast<char*>(&numComponents), sizeof(size_t));

    for (size_t compIdx = 0; compIdx < numComponents; ++compIdx)
    {
        Component component;
        size_t numVertices, numIndices;

        fin.read(reinterpret_cast<char*>(&numVertices), sizeof(size_t));
        component._vertices.resize(numVertices);
        fin.read(reinterpret_cast<char*>(component._vertices.data()), sizeof(VertexGPU) * numVertices);

        fin.read(reinterpret_cast<char*>(&numIndices), sizeof(size_t));
        if (numIndices)
        {
            component._indices.resize(numIndices);
            fin.read(reinterpret_cast<char*>(component._indices.data()), sizeof(GLuint) * numIndices);
        }

        fin.read(reinterpret_cast<char*>(&component._aabb), sizeof(AABB));
		fin.read(reinterpret_cast<char*>(&component._material), sizeof(Material));

		size_t nameLength;
		fin.read(reinterpret_cast<char*>(&nameLength), sizeof(size_t));
		component._name.resize(nameLength);
		fin.read(component._name.data(), nameLength);

        _components.emplace_back(std::move(component));
        _aabb.update(_components[compIdx]._aabb);
    }
}

void Model3D::writeBinaryFile(const std::string& path)
{
    std::ofstream fout(path, std::ios::out | std::ios::binary);
    if (!fout.is_open())
    {
        std::cout << "Failed to write the binary file!" << std::endl;
    }

    size_t numComponents = _components.size();
    fout.write(reinterpret_cast<char*>(&numComponents), sizeof(size_t));

    for (auto& component : _components)
    {
        size_t numVertices = component._vertices.size();

        fout.write(reinterpret_cast<char*>(&numVertices), sizeof(size_t));
        fout.write(reinterpret_cast<char*>(component._vertices.data()), numVertices * sizeof(VertexGPU));

        size_t numIndices = component._indices.size();
        fout.write(reinterpret_cast<char*>(&numIndices), sizeof(size_t));
        if (numIndices)
            fout.write(reinterpret_cast<char*>(component._indices.data()), numIndices * sizeof(GLuint));

        fout.write(reinterpret_cast<char*>(&component._aabb), sizeof(AABB));
		fout.write(reinterpret_cast<char*>(&component._material), sizeof(Material));

		size_t nameLength = component._name.size();
		fout.write(reinterpret_cast<char*>(&nameLength), sizeof(size_t));
		fout.write(component._name.c_str(), nameLength);
    }

    fout.close();
}

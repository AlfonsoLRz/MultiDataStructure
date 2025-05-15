#pragma once

#include <assimp/scene.h>

class Material
{
private:
	glm::vec3		_diffuseColor;
	glm::vec3		_specularColour;
	glm::vec3		_emissionColor;

	char			_diffuseTexturePath[500];

	float			_emissionStrength;
	float 			_metallic;
	float			_smoothness;

public:
	Material();
	static Material createMaterial(const aiMaterial* material);
	virtual ~Material() = default;

	// Setters
	void setDiffuseColor(const glm::vec3& color) { _diffuseColor = color; }
	void setSpecularColour(const glm::vec3& color) { _specularColour = color; }
	void setEmissionColor(const glm::vec3& color) { _emissionColor = color; }
	void setEmissionStrength(const float strength) { _emissionStrength = strength; }
	void setSmoothness(const float smoothness) { _smoothness = smoothness; }
	void setMetallic(const float metallic) { _metallic = metallic; }

	// Getters
	glm::vec3 getDiffuseColor() const { return _diffuseColor; }
	std::string getDiffuseTexturePath() const { return _diffuseTexturePath; }
	glm::vec3 getSpecularColor() const { return _specularColour; }
	glm::vec3 getEmissionColor() const { return _emissionColor; }
	float getEmissionStrength() const { return _emissionStrength; }
	float getSmoothness() const { return _smoothness; }
	float getMetallic() const { return _metallic; }
};


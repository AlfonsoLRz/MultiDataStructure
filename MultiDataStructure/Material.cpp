#include "stdafx.h"
#include "Material.h"

Material::Material() : _diffuseColor(glm::vec4(.0f)), _specularColour(1.0f), _emissionColor(.0f),
                       _emissionStrength(.0f),
                       _metallic(1.0f),
                       _smoothness(0.5f)
{
	for (char& i : _diffuseTexturePath)
		i = '\0';
}

Material Material::createMaterial(const aiMaterial* material)
{
	Material mat;
	aiColor4D colorAssimp;

	if (material->Get(AI_MATKEY_COLOR_DIFFUSE, colorAssimp) == AI_SUCCESS)
	{
		mat._diffuseColor = glm::vec3(colorAssimp.r, colorAssimp.g, colorAssimp.b);
	}

	if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
	{
		aiString path;
		material->GetTexture(aiTextureType_DIFFUSE, 0, &path);

		for (unsigned int i = 0; i < path.length; ++i)
			mat._diffuseTexturePath[i] = path.data[i];
	}

	if (material->Get(AI_MATKEY_COLOR_EMISSIVE, colorAssimp) == AI_SUCCESS)
	{
		glm::vec3 colorGLM (colorAssimp.r, colorAssimp.g, colorAssimp.b);

		mat._emissionStrength = glm::max(colorGLM.r, glm::max(colorGLM.g, colorGLM.b));
		if (mat._emissionStrength > 1.0f)
			mat._emissionColor = colorGLM * mat._emissionStrength;
		else
			mat._emissionColor = colorGLM;
	}

	if (material->Get(AI_MATKEY_SHININESS, colorAssimp) == AI_SUCCESS)
		mat._smoothness = colorAssimp.r / 1000.0f;

	if (material->Get(AI_MATKEY_COLOR_AMBIENT, colorAssimp) == AI_SUCCESS)
	mat._metallic = colorAssimp.r;

	if (material->Get(AI_MATKEY_COLOR_SPECULAR, colorAssimp) == AI_SUCCESS)
		mat._specularColour = glm::vec3(colorAssimp.r, colorAssimp.g, colorAssimp.b);

	return mat;
}

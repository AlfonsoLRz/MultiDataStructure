#include "stdafx.h"
#include "GeometricUtilities.h"

//

float GeometricUtilities::triangleArea(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3)
{
	return glm::length(glm::cross(v2 - v1, v3 - v1)) / 2.0f;
}

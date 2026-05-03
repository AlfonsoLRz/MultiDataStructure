#pragma once

#include "../../stdafx.h"
#include "../../AABB.h"

struct PointPrimitive
{
	glm::vec3 position = glm::vec3(0.0f);
	float intensity = 0.0f;
	uint32_t classification = 0;
	uint64_t id = 0;

	AABB bounds(float epsilon = 0.0f) const;
};


#pragma once

#include "../../stdafx.h"
#include "../../AABB.h"

struct PointPrimitive
{
	glm::vec3 position = glm::vec3(0.0f);

	AABB bounds(float epsilon = 0.0f) const;
};

static_assert(sizeof(PointPrimitive) == sizeof(float) * 3, "PointPrimitive must stay position-only for point-cloud memory use.");

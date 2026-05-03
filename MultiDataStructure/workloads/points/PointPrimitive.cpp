#include "../../stdafx.h"
#include "PointPrimitive.h"

AABB PointPrimitive::bounds(float epsilon) const
{
	const glm::vec3 delta(epsilon);
	return AABB(position - delta, position + delta);
}


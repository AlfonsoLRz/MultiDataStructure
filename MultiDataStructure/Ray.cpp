#include "stdafx.h"
#include "Ray.h"

Ray::Ray() : _origin(0.0f), _direction(0.0f)
{
}

Ray::Ray(const glm::vec3& origin, const glm::vec3& direction) : _origin(origin), _direction(direction)
{
}

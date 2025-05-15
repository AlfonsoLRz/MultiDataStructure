#include "stdafx.h"
#include "AABB.h"

// Public methods

AABB::AABB(const glm::vec3& min, const glm::vec3& max) : _max(max), _min(min)
{
}

AABB::AABB(const AABB& aabb) 
= default;

AABB::~AABB()
= default;

AABB& AABB::operator=(const AABB& aabb)
= default;

AABB AABB::dot(const glm::mat4& matrix) const
{
	return { matrix * glm::vec4(_min, 1.0f), matrix * glm::vec4(_max, 1.0f) };
}

void AABB::update(const AABB& aabb)
{
	this->update(aabb.max());
	this->update(aabb.min());
}

void AABB::update(const glm::vec3& point)
{
	_min.x = std::min(point.x, _min.x);
	_min.y = std::min(point.y, _min.y);
	_min.z = std::min(point.z, _min.z);

	_max.x = std::max(point.x, _max.x);
	_max.y = std::max(point.y, _max.y);
	_max.z = std::max(point.z, _max.z);
}

void AABB::split2D(glm::uint axis, AABB* aabb) const
{
	glm::vec3 center = this->center();

	if (axis == 0)
	{
		aabb[0] = AABB(_min, glm::vec3(center.x, _max.y, _max.z));
		aabb[1] = AABB(glm::vec3(center.x, _min.y, _min.z), _max);
	}
	else if (axis == 1)
	{
		aabb[0] = AABB(_min, glm::vec3(_max.x, center.y, _max.z));
		aabb[1] = AABB(glm::vec3(_min.x, center.y, _min.z), _max);
	}
	else
	{
		aabb[0] = AABB(_min, glm::vec3(_max.x, _max.y, center.z));
		aabb[1] = AABB(glm::vec3(_min.x, _min.y, center.z), _max);
	}
}

void AABB::split2D(glm::uint axis, float value, AABB* aabb) const
{
	if (axis == 0)
	{
		aabb[0] = AABB(_min, glm::vec3(value, _max.y, _max.z));
		aabb[1] = AABB(glm::vec3(value, _min.y, _min.z), _max);
	}
	else if (axis == 1)
	{
		aabb[0] = AABB(_min, glm::vec3(_max.x, value, _max.z));
		aabb[1] = AABB(glm::vec3(_min.x, value, _min.z), _max);
	}
	else
	{
		aabb[0] = AABB(_min, glm::vec3(_max.x, _max.y, value));
		aabb[1] = AABB(glm::vec3(_min.x, _min.y, value), _max);
	}
}

void AABB::split3D(glm::uvec3 numSubdivisions, AABB* aabb) const
{
	glm::uint idx = 0;
	glm::vec3 size = this->size();
	glm::vec3 step = size / glm::vec3(numSubdivisions);

	for (glm::uint i = 0; i < numSubdivisions.x; ++i)
	{
		for (glm::uint j = 0; j < numSubdivisions.y; ++j)
		{
			for (glm::uint k = 0; k < numSubdivisions.z; ++k)
			{
				glm::vec3 min = _min + glm::vec3(i * step.x, j * step.y, k * step.z);
				glm::vec3 max = min + step;
				aabb[idx++] = AABB(min, max);
			}
		}
	}
}

bool AABB::collides(const AABB& aabb) const
{
	return !(_max.x < aabb.min().x || _min.x > aabb.max().x ||
		_max.y < aabb.min().y || _min.y > aabb.max().y ||
		_max.z < aabb.min().z || _min.z > aabb.max().z);
}

bool AABB::collides(const glm::vec3& minPoint, const glm::vec3& maxPoint) const
{
	return !(_max.x < minPoint.x || _min.x > maxPoint.x ||
		_max.y < minPoint.y || _min.y > maxPoint.y ||
		_max.z < minPoint.z || _min.z > maxPoint.z);
}

bool AABB::intersects(const Ray& ray, float& tFar) const
{
	glm::vec3 invDir = 1.0f / ray._direction;
	glm::vec3 t0 = (_min - ray._origin) * invDir;
	glm::vec3 t1 = (_max - ray._origin) * invDir;
	glm::vec3 tMin = glm::min(t0, t1);
	glm::vec3 tMax = glm::max(t0, t1);
	float tNear = std::max({tMin.x, tMin.y, tMin.z});
	tFar = std::min({tMax.x, tMax.y, tMax.z});
	return (tNear < tFar && tFar > 0.0f);
}

bool AABB::intersects(const Ray& ray) const
{
	float tFar;
	return this->intersects(ray, tFar);
}

std::ostream& operator<<(std::ostream& os, const AABB& aabb)
{
	os << "max. " << aabb.max().x << ", " << aabb.max().y << ", " << aabb.max().z << "; ";
	os << "min. " << aabb.min().x << ", " << aabb.min().y << ", " << aabb.min().z;

	return os;
}

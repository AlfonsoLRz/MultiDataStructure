#pragma once

class Ray
{
public:
	glm::vec3	_origin, _direction;

public:
	Ray();
	Ray(const glm::vec3& origin, const glm::vec3& direction);
	virtual ~Ray() = default;

	// Getters
	glm::vec3 getDirection() const { return _direction; }
	glm::vec3 getOrigin() const { return _origin; }
};


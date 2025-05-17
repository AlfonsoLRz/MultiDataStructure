#pragma once

#include "GPUStructs.h"

class AABB
{
protected:
	glm::vec3	_max, _min;

public:
	AABB(const glm::vec3& min = glm::vec3(INFINITY), const glm::vec3& max = glm::vec3(-INFINITY));
	AABB(const AABB& aabb);
	virtual ~AABB();
	AABB& operator=(const AABB& aabb);

	glm::vec3 center() const { return (_max + _min) / 2.0f; }
	AABB dot(const glm::mat4& matrix) const;
	glm::vec3 extent() const { return _max - center(); }
	glm::vec3 max() const { return _max; }
	glm::vec3 min() const { return _min; }
	glm::vec3 size() const { return _max - _min; }

	void update(const AABB& aabb);
	void update(const glm::vec3& point);

	void split2D(glm::uint axis, AABB* aabb) const;
	std::vector<AABB> split2D(glm::uint axis) const;
	void split2D(glm::uint axis, float value, AABB* aabb) const;
	void split3D(glm::uvec3 numSubdivisions, AABB* aabb) const;

	bool collides(const AABB& aabb) const;
	bool collides(const glm::vec3& minPoint, const glm::vec3& maxPoint) const;
	bool collides(const Node* node) const;
	bool intersects(const RayGPU& ray, float& tFar) const;
	float intersects(const RayGPU& ray) const;

	friend std::ostream& operator<<(std::ostream& os, const AABB& aabb);
};


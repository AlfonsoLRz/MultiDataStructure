#pragma once

#include "stdafx.h"

typedef std::mt19937							RandomNumberGenerator;
typedef std::uniform_real_distribution<float>	DoubleUniformDistribution;

namespace RandomUtilities
{
	glm::vec3 getRandomToSphere(float radius, float distanceSquared);
	float getUniformRandom();
	float getUniformRandom(float min, float max);
	glm::vec3 getUniformRandomColor();
	glm::vec3 getUniformRandomColor(float min, float max);
	glm::vec3 getUniformRandomCosineDirection();
	glm::vec3 getUniformRandomInHemisphere(const glm::vec3& normal);
	int getUniformRandomInt(int min, int max);
	glm::vec3 getUniformRandomInUnitDisk();
	glm::vec3 getUniformRandomInUnitSphere();

	static float* getNoiseBuffer(size_t numPoints);
}

inline glm::vec3 RandomUtilities::getRandomToSphere(float radius, float distanceSquared)
{
	const float r1 = getUniformRandom();
	const float r2 = getUniformRandom();
	const float z = 1 + r2 * (sqrt(1.0f - radius * radius / distanceSquared) - 1.0f);
	const float phi = 2.0f * glm::pi<float>() * r1;
	const float x = std::cos(phi) * sqrt(1 - z * z);
	const float y = std::sin(phi) * sqrt(1 - z * z);

	return {x, y, z};
}

inline float RandomUtilities::getUniformRandom()
{
	static RandomNumberGenerator generator;
	static DoubleUniformDistribution distribution(.0f, 1.0f);

	return distribution(generator);
}

inline float RandomUtilities::getUniformRandom(float min, float max)
{
	return min + (max - min) * getUniformRandom();
}

inline glm::vec3 RandomUtilities::getUniformRandomColor()
{
	return {RandomUtilities::getUniformRandom(), RandomUtilities::getUniformRandom(), RandomUtilities::getUniformRandom()
	};
}
inline glm::vec3 RandomUtilities::getUniformRandomColor(float min, float max)
{
	return {RandomUtilities::getUniformRandom(min, max), RandomUtilities::getUniformRandom(min, max), RandomUtilities::getUniformRandom(min, max)
	};
}

inline glm::vec3 RandomUtilities::getUniformRandomCosineDirection()
{
	const float r1 = RandomUtilities::getUniformRandom(), r2 = RandomUtilities::getUniformRandom();
	const float z = sqrt(1 - r2);
	const float phi = 2.0f * glm::pi<float>() * r1;
	const float x = std::cos(phi) * sqrt(r2);
	const float y = std::sin(phi) * sqrt(r2);

	return {x, y, z};
}

inline glm::vec3 RandomUtilities::getUniformRandomInHemisphere(const glm::vec3& normal)
{
	glm::vec3 unitSphere = getUniformRandomInUnitSphere();
	return unitSphere * glm::sign(glm::dot(unitSphere, normal));
}

inline int RandomUtilities::getUniformRandomInt(int min, int max)
{
	return static_cast<int>(getUniformRandom(static_cast<float>(min), static_cast<float>(max)));
}

inline glm::vec3 RandomUtilities::getUniformRandomInUnitDisk()
{
	while (true)
	{
		auto point = glm::vec3(getUniformRandom(-1.0f, 1.0f), getUniformRandom(-1.0f, 1.0f), .0f);
		if (glm::length2(point) >= 1) continue;

		return point;
	}
}

inline glm::vec3 RandomUtilities::getUniformRandomInUnitSphere()
{
	while (true)
	{
		glm::vec3 point = glm::vec3(getUniformRandom(-1.0f, 1.0f), getUniformRandom(-1.0f, 1.0f),
		                            getUniformRandom(-1.0f, 1.0f));
		if (glm::length2(point) >= 1) continue;

		return point;
	}
}

inline float* RandomUtilities::getNoiseBuffer(size_t numPoints)
{
	std::vector<float> noise(numPoints);
	for (float& n : noise)
		n = RandomUtilities::getUniformRandom();

	float* noiseBuffer = nullptr;
	CudaHelper::initializeBufferGPU(noiseBuffer, numPoints, noise.data());

	return noiseBuffer;
}

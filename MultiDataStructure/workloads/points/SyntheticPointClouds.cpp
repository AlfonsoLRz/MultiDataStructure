#include "../../stdafx.h"
#include "SyntheticPointClouds.h"

namespace
{
	float uniform(std::mt19937& rng, float minValue, float maxValue)
	{
		std::uniform_real_distribution<float> distribution(minValue, maxValue);
		return distribution(rng);
	}

	PointPrimitive makePoint(const glm::vec3& position, uint64_t, uint32_t = 0, float = 0.0f)
	{
		PointPrimitive point;
		point.position = position;
		return point;
	}

	void appendTranslated(PointCloud& target, const PointCloud& source, const glm::vec3& offset)
	{
		for (const PointPrimitive& sourcePoint : source.points())
		{
			PointPrimitive point = sourcePoint;
			point.position += offset;
			target.addPoint(point);
		}
	}
}

PointCloud SyntheticPointClouds::generateFlatTerrain(size_t n, float width, float depth, float noiseZ, uint32_t seed)
{
	std::mt19937 rng(seed);
	PointCloud cloud;
	cloud.reserve(n);

	for (size_t i = 0; i < n; ++i)
	{
		const float x = uniform(rng, -width * 0.5f, width * 0.5f);
		const float y = uniform(rng, -depth * 0.5f, depth * 0.5f);
		const float z = uniform(rng, -noiseZ, noiseZ);
		cloud.addPoint(makePoint(glm::vec3(x, y, z), static_cast<uint64_t>(i), 2));
	}

	return cloud;
}

PointCloud SyntheticPointClouds::generateSlopedTerrain(size_t n, float width, float depth, float slopeX, float slopeY, float noiseZ, uint32_t seed)
{
	std::mt19937 rng(seed);
	PointCloud cloud;
	cloud.reserve(n);

	for (size_t i = 0; i < n; ++i)
	{
		const float x = uniform(rng, -width * 0.5f, width * 0.5f);
		const float y = uniform(rng, -depth * 0.5f, depth * 0.5f);
		const float z = slopeX * x + slopeY * y + uniform(rng, -noiseZ, noiseZ);
		cloud.addPoint(makePoint(glm::vec3(x, y, z), static_cast<uint64_t>(i), 2));
	}

	return cloud;
}

PointCloud SyntheticPointClouds::generateFacade(size_t n, float width, float height, float thickness, uint32_t seed)
{
	std::mt19937 rng(seed);
	PointCloud cloud;
	cloud.reserve(n);

	for (size_t i = 0; i < n; ++i)
	{
		const float x = uniform(rng, -width * 0.5f, width * 0.5f);
		const float y = uniform(rng, -thickness * 0.5f, thickness * 0.5f);
		const float z = uniform(rng, 0.0f, height);
		cloud.addPoint(makePoint(glm::vec3(x, y, z), static_cast<uint64_t>(i), 6));
	}

	return cloud;
}

PointCloud SyntheticPointClouds::generateBuildingShell(size_t n, float width, float depth, float height, uint32_t seed)
{
	std::mt19937 rng(seed);
	PointCloud cloud;
	cloud.reserve(n);

	for (size_t i = 0; i < n; ++i)
	{
		const int face = static_cast<int>(uniform(rng, 0.0f, 5.999f));
		float x = uniform(rng, -width * 0.5f, width * 0.5f);
		float y = uniform(rng, -depth * 0.5f, depth * 0.5f);
		float z = uniform(rng, 0.0f, height);

		if (face == 0)
			x = -width * 0.5f;
		else if (face == 1)
			x = width * 0.5f;
		else if (face == 2)
			y = -depth * 0.5f;
		else if (face == 3)
			y = depth * 0.5f;
		else if (face == 4)
			z = height;
		else
			z = 0.0f;

		cloud.addPoint(makePoint(glm::vec3(x, y, z), static_cast<uint64_t>(i), 6));
	}

	return cloud;
}

PointCloud SyntheticPointClouds::generateUrbanMixed(size_t terrainPoints, size_t buildingPoints, int numBuildings, uint32_t seed)
{
	PointCloud cloud = generateFlatTerrain(terrainPoints, 80.0f, 80.0f, 0.05f, seed);
	if (numBuildings <= 0 || buildingPoints == 0)
		return cloud;

	std::mt19937 rng(seed + 1);
	const size_t pointsPerBuilding = std::max<size_t>(1, buildingPoints / static_cast<size_t>(numBuildings));

	for (int building = 0; building < numBuildings; ++building)
	{
		PointCloud shell = generateBuildingShell(
			pointsPerBuilding,
			uniform(rng, 4.0f, 12.0f),
			uniform(rng, 4.0f, 12.0f),
			uniform(rng, 6.0f, 30.0f),
			seed + 10 + static_cast<uint32_t>(building));

		const glm::vec3 offset(
			uniform(rng, -30.0f, 30.0f),
			uniform(rng, -30.0f, 30.0f),
			0.0f);
		appendTranslated(cloud, shell, offset);
	}

	while (cloud.size() < terrainPoints + buildingPoints)
	{
		const glm::vec3 position(uniform(rng, -30.0f, 30.0f), uniform(rng, -30.0f, 30.0f), uniform(rng, 0.0f, 20.0f));
		cloud.addPoint(makePoint(position, static_cast<uint64_t>(cloud.size()), 6));
	}

	return cloud;
}

PointCloud SyntheticPointClouds::generateSparseDenseMixture(size_t sparsePoints, size_t densePoints, uint32_t seed)
{
	std::mt19937 rng(seed);
	PointCloud cloud;
	cloud.reserve(sparsePoints + densePoints);

	for (size_t i = 0; i < sparsePoints; ++i)
	{
		const glm::vec3 position(
			uniform(rng, -50.0f, 50.0f),
			uniform(rng, -50.0f, 50.0f),
			uniform(rng, -5.0f, 20.0f));
		cloud.addPoint(makePoint(position, static_cast<uint64_t>(cloud.size()), 1));
	}

	const glm::vec3 denseCenter(8.0f, -6.0f, 3.0f);
	for (size_t i = 0; i < densePoints; ++i)
	{
		const glm::vec3 position = denseCenter + glm::vec3(
			uniform(rng, -1.0f, 1.0f),
			uniform(rng, -1.0f, 1.0f),
			uniform(rng, -1.0f, 1.0f));
		cloud.addPoint(makePoint(position, static_cast<uint64_t>(cloud.size()), 1));
	}

	return cloud;
}

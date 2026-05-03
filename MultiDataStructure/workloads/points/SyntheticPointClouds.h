#pragma once

#include "../../stdafx.h"
#include "PointCloud.h"

namespace SyntheticPointClouds
{
	PointCloud generateFlatTerrain(size_t n, float width, float depth, float noiseZ, uint32_t seed = 1337);
	PointCloud generateSlopedTerrain(size_t n, float width, float depth, float slopeX, float slopeY, float noiseZ, uint32_t seed = 1337);
	PointCloud generateFacade(size_t n, float width, float height, float thickness, uint32_t seed = 1337);
	PointCloud generateBuildingShell(size_t n, float width, float depth, float height, uint32_t seed = 1337);
	PointCloud generateUrbanMixed(size_t terrainPoints, size_t buildingPoints, int numBuildings, uint32_t seed = 1337);
	PointCloud generateSparseDenseMixture(size_t sparsePoints, size_t densePoints, uint32_t seed = 1337);
}


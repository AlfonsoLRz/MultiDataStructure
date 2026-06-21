#pragma once

#include "../stdafx.h"
#include "../MultiDataStructure.h"
#include "BuildPolicy.h"

enum class SchemaPrimitiveKind
{
	QuadTree,
	Octree,
	KDTree,
	BVH,
	LBVH,
	BIH,
	KarrasOctree,
	RegularGrid,
	HGrid,
	Mixed
};

struct SchemaLevelCondition
{
	std::optional<size_t> minPoints;
	std::optional<size_t> maxPoints;
	std::optional<double> minDensity;
	std::optional<double> maxDensity;
	std::optional<double> minHeightRatio;
	std::optional<double> maxHeightRatio;
	std::optional<double> minExtentX;
	std::optional<double> maxExtentX;
	std::optional<double> minExtentY;
	std::optional<double> maxExtentY;
	std::optional<double> minExtentZ;
	std::optional<double> maxExtentZ;
	// Per-node anisotropy `1 - shortExtent / longExtent`: 0 = cubic, 1 = elongated.
	std::optional<double> minAnisotropy;
	std::optional<double> maxAnisotropy;
	// Per-node Shannon entropy over a 4x4x4 sub-grid, normalized to [0, 1]: 0 = clustered, 1 = even.
	std::optional<double> minOccupancyEntropy;
	std::optional<double> maxOccupancyEntropy;

	bool empty() const;
};

struct AdaptiveLeafCapacityConfig
{
	bool enabled = false;
	size_t minCapacity = 0;
	size_t maxCapacity = 0;
	double densityWeight = 0.0;
	double anisotropyWeight = 0.0;
	double heightRatioWeight = 0.0;
	double queryMixFactor = 1.0;
};

struct SchemaLevelConfig
{
	SchemaPrimitiveKind primitiveKind = SchemaPrimitiveKind::Octree;
	MultiDataStructure::DataStructureLevel cpuFallbackType = MultiDataStructure::DataStructureLevel::OctreeNode;
	// Compatibility alias for older CPU/build code; prefer primitiveKind or cpuFallbackType.
	MultiDataStructure::DataStructureLevel type = MultiDataStructure::DataStructureLevel::OctreeNode;
	std::string typeName = "Octree";
	size_t numLevels = 1;
	size_t leafCapacity = 1;
	size_t minPrimitivesToSplit = 2;
	std::string axisPolicy;
	SchemaLevelCondition condition;
	AdaptiveLeafCapacityConfig adaptiveLeafCapacity;
};

struct SchemaConfig
{
	std::string name;
	std::vector<SchemaLevelConfig> levels;
	BuildPolicy buildPolicy;

	size_t totalLevels() const;
	std::vector<MultiDataStructure::LevelConfig> toLevelConfigs() const;
	const SchemaLevelConfig& levelForDepth(size_t depth) const;
};

namespace Config
{
	SchemaConfig loadSchemaConfig(const std::string& filename);
	SchemaConfig parseSchemaConfig(const std::string& jsonText, const std::string& sourceName = {});
	SchemaPrimitiveKind parseSchemaPrimitiveKind(const std::string& value);
	std::string schemaPrimitiveKindName(SchemaPrimitiveKind kind);
	MultiDataStructure::DataStructureLevel cpuFallbackForPrimitiveKind(SchemaPrimitiveKind kind);
	MultiDataStructure::DataStructureLevel parseDataStructureLevel(const std::string& value);
	std::string dataStructureLevelName(MultiDataStructure::DataStructureLevel level);
}

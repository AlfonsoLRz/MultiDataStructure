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
	std::optional<size_t>	_minPoints;
	std::optional<size_t>	_maxPoints;
	std::optional<double>	_minDensity;
	std::optional<double>	_maxDensity;
	std::optional<double>	_minHeightRatio;
	std::optional<double>	_maxHeightRatio;
	std::optional<double>	_minExtentX;
	std::optional<double>	_maxExtentX;
	std::optional<double>	_minExtentY;
	std::optional<double>	_maxExtentY;
	std::optional<double>	_minExtentZ;
	std::optional<double>	_maxExtentZ;
	// Per-node anisotropy `1 - shortExtent / longExtent`: 0 = cubic, 1 = elongated.
	std::optional<double>	_minAnisotropy;
	std::optional<double>	_maxAnisotropy;
	// Per-node Shannon entropy over a 4x4x4 sub-grid, normalized to [0, 1]: 0 = clustered, 1 = even.
	std::optional<double>	_minOccupancyEntropy;
	std::optional<double>	_maxOccupancyEntropy;

	bool empty() const;
};

struct AdaptiveLeafCapacityConfig
{
	bool	_enabled = false;
	size_t	_minCapacity = 0;
	size_t	_maxCapacity = 0;
	double	_densityWeight = 0.0;
	double	_anisotropyWeight = 0.0;
	double	_heightRatioWeight = 0.0;
	double	_queryMixFactor = 1.0;
};

struct SchemaLevelConfig
{
	SchemaPrimitiveKind	_primitiveKind = SchemaPrimitiveKind::Octree;
	MultiDataStructure::DataStructureLevel	_cpuFallbackType = MultiDataStructure::DataStructureLevel::OctreeNode;
	// Compatibility alias for older CPU/build code; prefer primitiveKind or cpuFallbackType.
	MultiDataStructure::DataStructureLevel	_type = MultiDataStructure::DataStructureLevel::OctreeNode;
	std::string	_typeName = "Octree";
	size_t	_numLevels = 1;
	size_t	_leafCapacity = 1;
	size_t	_minPrimitivesToSplit = 2;
	std::string	_axisPolicy;
	SchemaLevelCondition	_condition;
	AdaptiveLeafCapacityConfig	_adaptiveLeafCapacity;
};

struct SchemaConfig
{
	std::string	_name;
	std::vector<SchemaLevelConfig>	_levels;
	BuildPolicy	_buildPolicy;

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

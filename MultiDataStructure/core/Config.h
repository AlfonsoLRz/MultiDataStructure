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
	// Phase B3 anisotropy predicate. Defined per-node as `1 - shortExtent / longExtent` over
	// the node's bbox extents: 0.0 = perfectly cubic / regular, 1.0 = infinitely elongated.
	// Lets branches specialize on shape rather than just on size or density (e.g. a tall thin
	// facade region triggers a nested LBVH while a roughly-cubic region falls through).
	std::optional<double> minAnisotropy;
	std::optional<double> maxAnisotropy;
	// Phase B3 occupancy-entropy predicate. Per-node Shannon entropy over a 4x4x4 sub-grid of
	// the node's points, normalized to [0, 1] by dividing by ln(64). 0 = all points in one
	// sub-cell (perfectly clustered), 1 = uniformly spread across all sub-cells. Lets branches
	// distinguish "clumpy" regions (low entropy — favor coarse leaves) from "even" regions
	// (high entropy — favor fine recursion).
	std::optional<double> minOccupancyEntropy;
	std::optional<double> maxOccupancyEntropy;

	bool empty() const;
};

struct SchemaLevelConfig
{
	SchemaPrimitiveKind primitiveKind = SchemaPrimitiveKind::Octree;
	MultiDataStructure::DataStructureLevel cpuFallbackType = MultiDataStructure::DataStructureLevel::OctreeNode;
	// Compatibility alias for older CPU/build code. New code should prefer primitiveKind for
	// schema semantics and cpuFallbackType when it intentionally needs the legacy CPU enum.
	MultiDataStructure::DataStructureLevel type = MultiDataStructure::DataStructureLevel::OctreeNode;
	std::string typeName = "Octree";
	size_t numLevels = 1;
	size_t leafCapacity = 1;
	size_t minPrimitivesToSplit = 2;
	std::string axisPolicy;
	SchemaLevelCondition condition;
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

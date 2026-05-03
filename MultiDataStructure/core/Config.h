#pragma once

#include "../stdafx.h"
#include "../MultiDataStructure.h"
#include "BuildPolicy.h"

struct SchemaLevelConfig
{
	MultiDataStructure::DataStructureLevel type = MultiDataStructure::DataStructureLevel::OctreeNode;
	std::string typeName = "Octree";
	size_t numLevels = 1;
	size_t leafCapacity = 1;
	size_t minPrimitivesToSplit = 2;
	std::string axisPolicy;
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
	MultiDataStructure::DataStructureLevel parseDataStructureLevel(const std::string& value);
	std::string dataStructureLevelName(MultiDataStructure::DataStructureLevel level);
}


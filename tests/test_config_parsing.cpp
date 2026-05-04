#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/core/Config.h"
#include "../MultiDataStructure/workloads/points/PointSpatialIndex.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		SchemaConfig loadSchema(const std::string& name)
		{
			return Config::loadSchemaConfig("configs/schemas/" + name + ".json");
		}

		void expectPointBuildPreservesCounts(const SchemaConfig& schema, const PointCloud& cloud, const std::string& label)
		{
			PointSpatialIndex index;
			index.build(cloud, schema);
			const PointSpatialIndex::Stats stats = index.stats();

			expect(stats.numNodes >= 1, label + " creates at least one node");
			expect(stats.numLeaves >= 1, label + " creates at least one leaf");
			expect(stats.numPoints == cloud.size(), label + " preserves point count without duplication");
		}
	}

	void runConfigParsingTests()
	{
		const SchemaConfig octree = loadSchema("octree");
		expect(octree.name == "octree_default", "octree schema name parsed");
		expect(octree.levels.size() == 1, "octree schema has one level block");
		expect(octree.totalLevels() == 8, "octree schema total levels parsed");
		expect(octree.levels[0].type == MultiDataStructure::DataStructureLevel::OctreeNode, "octree type parsed");
		expect(octree.buildPolicy.allowOverlapDuplication == false, "point schema disables overlap duplication");

		const SchemaConfig quadtree = loadSchema("quadtree");
		expect(quadtree.levels[0].type == MultiDataStructure::DataStructureLevel::QuadTreeNode, "quadtree type parsed");

		const SchemaConfig kdtree = loadSchema("kdtree");
		expect(kdtree.levels[0].type == MultiDataStructure::DataStructureLevel::KDTreeNode, "kd-tree type parsed");
		expect(kdtree.levels[0].axisPolicy == "median_longest_axis", "kd-tree axis policy parsed");

		const SchemaConfig hybrid = loadSchema("quadtree_octree");
		expect(hybrid.levels.size() == 2, "hybrid schema has two level blocks");
		expect(hybrid.totalLevels() == 8, "hybrid total levels parsed");

		const std::vector<MultiDataStructure::LevelConfig> legacyLevels = hybrid.toLevelConfigs();
		expect(legacyLevels.size() == 2, "hybrid converts to legacy level configs");
		expect(legacyLevels[0]._leafCapacity == 64, "legacy level config carries leaf capacity");
		expect(legacyLevels[1]._minPrimitivesToSplit == 8, "legacy level config carries min split threshold");

		const PointCloud cloud = SyntheticPointClouds::generateUrbanMixed(128, 128, 4);
		expectPointBuildPreservesCounts(octree, cloud, "octree point build");
		expectPointBuildPreservesCounts(quadtree, cloud, "quadtree point build");
		expectPointBuildPreservesCounts(hybrid, cloud, "hybrid point build");

		const char* noSplitJson = R"json(
		{
		  "name": "no_split",
		  "levels": [
		    { "type": "Octree", "numLevels": 4, "leafCapacity": 10000, "minPointsToSplit": 10000 }
		  ],
		  "buildPolicy": {
		    "maxDepth": 4,
		    "leafCapacity": 10000,
		    "minPointsToSplit": 10000,
		    "collapseSingleChild": true,
		    "removeEmptyNodes": true,
		    "allowOverlapDuplication": false
		  }
		}
		)json";

		const char* splitJson = R"json(
		{
		  "name": "split",
		  "levels": [
		    { "type": "Octree", "numLevels": 4, "leafCapacity": 4, "minPointsToSplit": 2 }
		  ],
		  "buildPolicy": {
		    "maxDepth": 4,
		    "leafCapacity": 4,
		    "minPointsToSplit": 2,
		    "collapseSingleChild": false,
		    "removeEmptyNodes": true,
		    "allowOverlapDuplication": false
		  }
		}
		)json";

		PointSpatialIndex noSplitIndex;
		noSplitIndex.build(cloud, Config::parseSchemaConfig(noSplitJson, "no_split"));
		expect(noSplitIndex.stats().numNodes == 1, "large leaf capacity prevents point subdivision");

		PointSpatialIndex splitIndex;
		splitIndex.build(cloud, Config::parseSchemaConfig(splitJson, "split"));
		expect(splitIndex.stats().numNodes > 1, "small leaf capacity allows point subdivision");
		expect(splitIndex.stats().numPoints == cloud.size(), "policy-controlled split preserves point count");

		const char* conditionalFalseJson = R"json(
		{
		  "name": "conditional_false",
		  "levels": [
		    { "type": "QuadTree", "numLevels": 1, "leafCapacity": 4, "minPointsToSplit": 2 },
		    {
		      "type": "Octree",
		      "numLevels": 3,
		      "leafCapacity": 4,
		      "minPointsToSplit": 2,
		      "condition": {
		        "minHeightRatio": 999.0,
		        "minPoints": 2
		      }
		    }
		  ],
		  "buildPolicy": {
		    "maxDepth": 4,
		    "leafCapacity": 4,
		    "minPointsToSplit": 2,
		    "collapseSingleChild": false,
		    "removeEmptyNodes": true,
		    "allowOverlapDuplication": false
		  }
		}
		)json";

		const char* conditionalTrueJson = R"json(
		{
		  "name": "conditional_true",
		  "levels": [
		    { "type": "QuadTree", "numLevels": 1, "leafCapacity": 4, "minPointsToSplit": 2 },
		    {
		      "type": "Octree",
		      "numLevels": 3,
		      "leafCapacity": 4,
		      "minPointsToSplit": 2,
		      "condition": {
		        "minHeightRatio": 0.0,
		        "minPoints": 2
		      }
		    }
		  ],
		  "buildPolicy": {
		    "maxDepth": 4,
		    "leafCapacity": 4,
		    "minPointsToSplit": 2,
		    "collapseSingleChild": false,
		    "removeEmptyNodes": true,
		    "allowOverlapDuplication": false
		  }
		}
		)json";

		const SchemaConfig conditionalFalse = Config::parseSchemaConfig(conditionalFalseJson, "conditional_false");
		expect(conditionalFalse.levels[1].condition.minHeightRatio.has_value(), "schema condition parses height ratio");
		expect(conditionalFalse.levels[1].condition.minPoints.has_value(), "schema condition parses min points");

		PointSpatialIndex conditionalFalseIndex;
		conditionalFalseIndex.build(cloud, conditionalFalse);

		PointSpatialIndex conditionalTrueIndex;
		conditionalTrueIndex.build(cloud, Config::parseSchemaConfig(conditionalTrueJson, "conditional_true"));

		expect(conditionalFalseIndex.stats().numPoints == cloud.size(), "conditional false build preserves point count");
		expect(conditionalTrueIndex.stats().numPoints == cloud.size(), "conditional true build preserves point count");
		expect(conditionalTrueIndex.stats().maxDepth > conditionalFalseIndex.stats().maxDepth, "matching schema condition unfolds deeper levels");
	}
}

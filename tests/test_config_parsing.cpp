#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/core/Config.h"
#include "../MultiDataStructure/workloads/points/PointSpatialIndex.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		bool nearlyEqual(float left, float right, float epsilon = 0.0001f)
		{
			return std::abs(left - right) <= epsilon;
		}

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
		expect(octree.levels[0].primitiveKind == SchemaPrimitiveKind::Octree, "octree primitive kind parsed");
		expect(octree.levels[0].cpuFallbackType == MultiDataStructure::DataStructureLevel::OctreeNode, "octree CPU fallback parsed");
		expect(octree.buildPolicy.allowOverlapDuplication == false, "point schema disables overlap duplication");

		const SchemaConfig quadtree = loadSchema("quadtree");
		expect(quadtree.levels[0].type == MultiDataStructure::DataStructureLevel::QuadTreeNode, "quadtree type parsed");
		expect(quadtree.levels[0].axisPolicy == "xy", "quadtree defaults to XY axis policy");

		const char* quadtreeXzJson = R"json(
		{
		  "name": "quadtree_xz",
		  "levels": [
		    { "type": "QuadTree", "axisPolicy": "xz", "numLevels": 1, "leafCapacity": 1, "minPointsToSplit": 2 }
		  ],
		  "buildPolicy": {
		    "maxDepth": 1,
		    "leafCapacity": 1,
		    "minPointsToSplit": 2,
		    "collapseSingleChild": false,
		    "removeEmptyNodes": false,
		    "allowOverlapDuplication": false
		  }
		}
		)json";
		const SchemaConfig quadtreeXz = Config::parseSchemaConfig(quadtreeXzJson, "quadtree_xz");
		expect(quadtreeXz.levels[0].axisPolicy == "xz", "quadtree parses explicit XZ axis policy");

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

		PointCloud wideTerrain;
		wideTerrain.addPoint({ glm::vec3(0.0f, 0.0f, 0.0f) });
		wideTerrain.addPoint({ glm::vec3(100.0f, 0.0f, 0.1f) });
		wideTerrain.addPoint({ glm::vec3(0.0f, 10.0f, 0.2f) });
		wideTerrain.addPoint({ glm::vec3(100.0f, 10.0f, 0.3f) });
		wideTerrain.addPoint({ glm::vec3(50.0f, 5.0f, 0.4f) });
		const char* quadtreeSplitJson = R"json(
		{
		  "name": "quadtree_xy_split",
		  "levels": [
		    { "type": "QuadTree", "numLevels": 1, "leafCapacity": 1, "minPointsToSplit": 2 }
		  ],
		  "buildPolicy": {
		    "maxDepth": 1,
		    "leafCapacity": 1,
		    "minPointsToSplit": 2,
		    "collapseSingleChild": false,
		    "removeEmptyNodes": false,
		    "allowOverlapDuplication": false
		  }
		}
		)json";
		PointSpatialIndex wideIndex;
		wideIndex.build(wideTerrain, Config::parseSchemaConfig(quadtreeSplitJson, "quadtree_xy_split"));
		const PointSpatialIndex::Node* wideRoot = wideIndex.root();
		expect(wideRoot != nullptr && wideRoot->children.size() == 4, "quadtree XY split creates four root children");
		if (wideRoot && !wideRoot->children.empty())
		{
			const glm::vec3 rootSize = wideRoot->bounds.size();
			const glm::vec3 childSize = wideRoot->children.front()->bounds.size();
			expect(childSize.x < rootSize.x && childSize.y < rootSize.y,
				"quadtree default splits X and Y for terrain-shaped clouds");
			expect(nearlyEqual(childSize.z, rootSize.z),
				"quadtree default leaves Z unsplit");
		}

		const char* gpuFlavorJson = R"json(
		{
		  "name": "gpu_flavors",
		  "levels": [
		    { "type": "BIH", "numLevels": 1, "leafCapacity": 16, "minPointsToSplit": 4 },
		    { "type": "KarrasOctree", "numLevels": 1, "leafCapacity": 16, "minPointsToSplit": 4 },
		    { "type": "LBVH", "numLevels": 1, "leafCapacity": 16, "minPointsToSplit": 4 },
		    { "type": "RegularGrid", "numLevels": 1, "leafCapacity": 16, "minPointsToSplit": 4 },
		    { "type": "HGrid", "numLevels": 1, "leafCapacity": 16, "minPointsToSplit": 4 }
		  ],
		  "buildPolicy": {
		    "maxDepth": 3,
		    "leafCapacity": 16,
		    "minPointsToSplit": 4,
		    "collapseSingleChild": true,
		    "removeEmptyNodes": true,
		    "allowOverlapDuplication": false
		  }
		}
		)json";
		const SchemaConfig gpuFlavors = Config::parseSchemaConfig(gpuFlavorJson, "gpu_flavors");
		expect(gpuFlavors.levels[0].type == MultiDataStructure::DataStructureLevel::KDTreeNode, "BIH parses as KD-compatible family");
		expect(gpuFlavors.levels[0].primitiveKind == SchemaPrimitiveKind::BIH, "BIH primitive kind is preserved");
		expect(gpuFlavors.levels[0].cpuFallbackType == MultiDataStructure::DataStructureLevel::KDTreeNode, "BIH CPU fallback is explicit");
		expect(gpuFlavors.levels[0].typeName == "BIH", "BIH schema name is preserved");
		expect(gpuFlavors.levels[1].type == MultiDataStructure::DataStructureLevel::OctreeNode, "KarrasOctree parses as Octree-compatible family");
		expect(gpuFlavors.levels[1].primitiveKind == SchemaPrimitiveKind::KarrasOctree, "KarrasOctree primitive kind is preserved");
		expect(gpuFlavors.levels[1].typeName == "KarrasOctree", "KarrasOctree schema name is preserved");
		expect(gpuFlavors.levels[2].type == MultiDataStructure::DataStructureLevel::BvhNode, "LBVH parses as BVH-compatible family");
		expect(gpuFlavors.levels[2].primitiveKind == SchemaPrimitiveKind::LBVH, "LBVH primitive kind is preserved");
		expect(gpuFlavors.levels[2].cpuFallbackType == MultiDataStructure::DataStructureLevel::BvhNode, "LBVH CPU fallback is explicit");
		expect(gpuFlavors.levels[3].type == MultiDataStructure::DataStructureLevel::OctreeNode, "RegularGrid parses as Octree-compatible family");
		expect(gpuFlavors.levels[3].primitiveKind == SchemaPrimitiveKind::RegularGrid, "RegularGrid primitive kind is preserved");
		expect(gpuFlavors.levels[3].cpuFallbackType == MultiDataStructure::DataStructureLevel::OctreeNode, "RegularGrid CPU fallback is explicit");
		expect(gpuFlavors.levels[3].typeName == "RegularGrid", "RegularGrid schema name is preserved");
		expect(gpuFlavors.levels[4].type == MultiDataStructure::DataStructureLevel::OctreeNode, "HGrid parses as Octree-compatible family");
		expect(gpuFlavors.levels[4].primitiveKind == SchemaPrimitiveKind::HGrid, "HGrid primitive kind is preserved");
		expect(gpuFlavors.levels[4].typeName == "HGrid", "HGrid schema name is preserved");

		const char* mixedJson = R"json(
		{
		  "name": "mixed_primitive",
		  "levels": [
		    { "type": "Mixed", "numLevels": 1, "leafCapacity": 16, "minPointsToSplit": 4 }
		  ]
		}
		)json";
		const SchemaConfig mixedPrimitive = Config::parseSchemaConfig(mixedJson, "mixed_primitive");
		expect(mixedPrimitive.levels[0].primitiveKind == SchemaPrimitiveKind::Mixed, "Mixed primitive kind is parsed");
		expect(mixedPrimitive.levels[0].cpuFallbackType == MultiDataStructure::DataStructureLevel::OctreeNode, "Mixed CPU fallback is explicit");

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

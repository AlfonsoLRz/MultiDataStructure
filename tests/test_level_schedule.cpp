#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/MultiDataStructure.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	void runLevelScheduleTests()
	{
		using Level = MultiDataStructure::DataStructureLevel;

		const MultiDataStructure singleBlock({
			{ ._levelType = Level::KDTreeNode, ._numLevels = 3 },
		});
		expect(singleBlock.getConfiguredNodeType(0) == Level::KDTreeNode, "single-block schedule maps level 0 to its type");
		expect(singleBlock.getConfiguredNodeType(1) == Level::KDTreeNode, "single-block schedule maps level 1 to its type");
		expect(singleBlock.getConfiguredNodeType(2) == Level::KDTreeNode, "single-block schedule maps its last configured level");
		expect(singleBlock.getConfiguredNodeType(3) == Level::KDTreeNode, "single-block schedule clamps out-of-range depths to its last type");

		const MultiDataStructure twoBlock({
			{ ._levelType = Level::OctreeNode, ._numLevels = 1 },
			{ ._levelType = Level::BvhNode, ._numLevels = 3 },
		});
		expect(twoBlock.getConfiguredNodeType(0) == Level::OctreeNode, "two-block schedule maps level 0 to the first type");
		expect(twoBlock.getConfiguredNodeType(1) == Level::BvhNode, "two-block schedule advances at the first boundary");
		expect(twoBlock.getConfiguredNodeType(2) == Level::BvhNode, "two-block schedule keeps the second type within its range");
		expect(twoBlock.getConfiguredNodeType(3) == Level::BvhNode, "two-block schedule maps the second block's last level");

		const MultiDataStructure threeBlock({
			{ ._levelType = Level::OctreeNode, ._numLevels = 1 },
			{ ._levelType = Level::BvhNode, ._numLevels = 3 },
			{ ._levelType = Level::KDTreeNode, ._numLevels = 2 },
		});

		expect(threeBlock.getConfiguredNodeType(0) == Level::OctreeNode, "level 0 uses the first configured type");
		expect(threeBlock.getConfiguredNodeType(1) == Level::BvhNode, "level 1 advances to the second configured type");
		expect(threeBlock.getConfiguredNodeType(2) == Level::BvhNode, "level 2 remains in the second configured range");
		expect(threeBlock.getConfiguredNodeType(3) == Level::BvhNode, "level 3 maps to the second block's last configured level");
		expect(threeBlock.getConfiguredNodeType(4) == Level::KDTreeNode, "level 4 advances to the third configured type");
		expect(threeBlock.getConfiguredNodeType(5) == Level::KDTreeNode, "level 5 remains in the third configured range");

		BuildPolicy cappedPolicy;
		cappedPolicy._maxDepth = 3;
		const MultiDataStructure capped({
			{ ._levelType = Level::QuadTreeNode, ._numLevels = 2 },
			{ ._levelType = Level::OctreeNode, ._numLevels = 4 },
		}, cappedPolicy);
		expect(capped.getStats()._numLevels == 3, "maxDepth caps total build levels without changing schedule boundaries");
	}
}

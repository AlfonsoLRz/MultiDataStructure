#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/MultiDataStructure.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	void runLevelScheduleTests()
	{
		using Level = MultiDataStructure::DataStructureLevel;

		const MultiDataStructure index({
			{ ._levelType = Level::OctreeNode, ._numLevels = 1 },
			{ ._levelType = Level::BvhNode, ._numLevels = 3 },
			{ ._levelType = Level::KDTreeNode, ._numLevels = 2 },
		});

		expect(index.getConfiguredNodeType(0) == Level::OctreeNode, "level 0 uses the first configured type");
		expect(index.getConfiguredNodeType(1) == Level::OctreeNode, "current boundary behavior keeps level 1 on the first type");
		expect(index.getConfiguredNodeType(2) == Level::BvhNode, "level 2 advances to the second configured type");
		expect(index.getConfiguredNodeType(4) == Level::BvhNode, "level 4 remains in the second configured range");
		expect(index.getConfiguredNodeType(5) == Level::KDTreeNode, "level 5 advances to the third configured type");
	}
}


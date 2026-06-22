#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/MultiDataStructure.h"
#include "BaselineTests.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message)
	{
		if (!condition)
			throw std::runtime_error(message);
	}

	void runLevelScheduleTests();
	void runPointCloudTests();
	void runPointQueryTests();
	void runMetricsTests();
	void runConfigParsingTests();
	void runSchemaSearchTests();
	void runFeatureExtractionTests();
	void runSchemaSelectorTests();
	void runBIHTests();
	void runHGridTests();
	void runKDTreeTests();
	void runLBVHTests();
	void runMixedTreeTests();
	void runOctreeTests();
	void runQuadTreeTests();
	void runRegularGridTests();

	bool nearlyEqual(float left, float right, float epsilon = 1e-5f)
	{
		return std::abs(left - right) <= epsilon;
	}

	void runAABBTests()
	{
		const AABB box(glm::vec3(-1.0f, -2.0f, -3.0f), glm::vec3(2.0f, 1.0f, 4.0f));
		const glm::mat4 transform =
			glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, -3.0f, 2.0f)) *
			glm::rotate(glm::mat4(1.0f), glm::half_pi<float>(), glm::vec3(0.0f, 0.0f, 1.0f));

		const AABB transformed = box.dot(transform);
		expect(nearlyEqual(transformed.min().x, 4.0f), "transformed AABB min x includes all rotated corners");
		expect(nearlyEqual(transformed.max().x, 7.0f), "transformed AABB max x includes all rotated corners");
		expect(nearlyEqual(transformed.min().y, -4.0f), "transformed AABB min y includes all rotated corners");
		expect(nearlyEqual(transformed.max().y, -1.0f), "transformed AABB max y includes all rotated corners");
		expect(nearlyEqual(transformed.min().z, -1.0f), "transformed AABB min z includes translation");
		expect(nearlyEqual(transformed.max().z, 6.0f), "transformed AABB max z includes translation");
	}

	class TestBinaryNode final : public MultiDataStructure::SpatialDSNode
	{
	public:
		TestBinaryNode(const AABB& aabb = AABB()) : SpatialDSNode(aabb) {}

		std::unique_ptr<MultiDataStructure::SpatialDSNode> copy(const AABB& aabb) const override
		{
			return std::make_unique<TestBinaryNode>(aabb);
		}

		void split(MultiDataStructure::DataStructureLevel nodeType) override
		{
			AABB halves[2];
			_aabb.split2D(0, halves);
			_children.push_back(NodeFactory::create(nodeType, halves[0]));
			_children.push_back(NodeFactory::create(nodeType, halves[1]));
		}
	};

	Node makeTestPrimitive(float minX, float maxX)
	{
		Node node{};
		node._minPoint = glm::vec4(minX, -0.1f, -0.1f, 0.0f);
		node._maxPoint = glm::vec4(maxX, 0.1f, 0.1f, 0.0f);
		node._triangleIndex = 0;
		return node;
	}

	void runTreeCleanupTests()
	{
		using Level = MultiDataStructure::DataStructureLevel;

		NodeFactory::registerType<TestBinaryNode>(Level::BvhNode);

		Node primitives[] = {
			makeTestPrimitive(-0.9f, -0.8f),
			makeTestPrimitive(-0.7f, -0.6f),
		};

		MultiDataStructure index({
			{ ._levelType = Level::BvhNode, ._numLevels = 1 },
		});

		index.build(primitives, 2, AABB(glm::vec3(-1.0f), glm::vec3(1.0f)));

		MultiDataStructure::Stats stats = index.getStats();
		expect(stats._numNodes == 3, "build creates a root and two children before cleanup");
		expect(stats._numLeaves == 2, "build leaves one populated and one empty child before cleanup");
		expect(stats._numPrimitives == 2, "build keeps both primitives in the populated child");

		glm::uint deletedNodes = 0;
		index.removeEmptyNodes(deletedNodes);
		expect(deletedNodes == 1, "removeEmptyNodes deletes the empty child");

		stats = index.getStats();
		expect(stats._numNodes == 2, "removeEmptyNodes keeps the root and populated child");
		expect(stats._numLeaves == 1, "removeEmptyNodes leaves a single populated leaf");
		expect(stats._numPrimitives == 2, "removeEmptyNodes preserves primitive membership");

		index.collapseNodes();
		stats = index.getStats();
		expect(stats._numNodes == 1, "collapseNodes replaces the root with its only child");
		expect(stats._numLeaves == 1, "collapseNodes leaves one leaf");
		expect(stats._numPrimitives == 2, "collapseNodes preserves primitive membership");
	}
}

int runBaselineTests()
{
	try
	{
		BaselineTests::runLevelScheduleTests();
		BaselineTests::runTreeCleanupTests();
		BaselineTests::runAABBTests();
		BaselineTests::runPointCloudTests();
		BaselineTests::runPointQueryTests();
		BaselineTests::runMetricsTests();
		BaselineTests::runConfigParsingTests();
		BaselineTests::runSchemaSearchTests();
		BaselineTests::runFeatureExtractionTests();
		BaselineTests::runSchemaSelectorTests();
		BaselineTests::runBIHTests();
		BaselineTests::runHGridTests();
		BaselineTests::runKDTreeTests();
		BaselineTests::runLBVHTests();
		BaselineTests::runMixedTreeTests();
		BaselineTests::runOctreeTests();
		BaselineTests::runQuadTreeTests();
		BaselineTests::runRegularGridTests();
		std::cout << "Baseline tests passed" << '\n';
		return 0;
	}
	catch (const std::exception& exception)
	{
		std::cerr << "Baseline tests failed: " << exception.what() << '\n';
		return 1;
	}
}

#pragma once

#include "AABB.h"
#include "core/BuildPolicy.h"
#include "GPUStructs.h"

class MultiDataStructure
{
public:
	enum DataStructureLevel
	{
		QuadTreeNode,
		KDTreeNode,
		OctreeNode,
		BvhNode
	};

	struct LevelConfig
	{
		DataStructureLevel	_levelType;
		glm::uint			_numLevels;
		size_t				_leafCapacity = 1;
		size_t				_minPrimitivesToSplit = 2;
	};

	struct Stats
	{
		glm::uint numLevels = 0;
		glm::uint numNodes = 0;
		glm::uint numLeaves = 0;
		glm::uint numPrimitives = 0;
		float averageLeafPrimitives = 0.0f;
	};

public:
	class SpatialDSNode
	{
	public:
		AABB _aabb;
		std::vector<const Node*> _primitives;
		std::vector<SpatialDSNode*> _children;

		SpatialDSNode(const AABB& aabb = AABB()) : _aabb(aabb) {}
		virtual ~SpatialDSNode() = default;

		virtual SpatialDSNode* copy(const AABB& aabb) const = 0;
		virtual bool in(const Node* node) const;
		virtual bool intersects(const Ray& ray);
		virtual void split(DataStructureLevel nodeType) = 0;

		virtual bool isLeaf() const { return _children.empty(); }
	};

private:
	glm::uint					_maxLevels;
	SpatialDSNode*				_rootNode;
	glm::uint					_numPrimitives;
	BuildPolicy					_buildPolicy;

	//
	std::vector<LevelConfig>	_levels;
	std::vector<glm::uint> 	    _levelCDF;

private:
	void check(SpatialDSNode* dsNode, glm::uint level);
	void insert(SpatialDSNode* dsNode, const Node* node, glm::uint level);

	void collectNodes(const SpatialDSNode* node, std::vector<AABB>& nodes);

	void checkSanity(const SpatialDSNode* dsNode, glm::uint level);

	void collapseNodes(SpatialDSNode*& dsNode, glm::uint level);
	void removeEmptyNodes(SpatialDSNode* dsNode, glm::uint level, glm::uint& deletedNodes);

	DataStructureLevel getNodeType(const glm::uint level) const;
	const LevelConfig& getLevelConfig(const glm::uint level) const;
	bool shouldSplit(const LevelConfig& levelConfig, size_t primitiveCount, glm::uint level) const;
	void getAverageLeafPrimitives(const SpatialDSNode* dsNode, float& sum, glm::uint& count) const;
	void getNumLeaves(const SpatialDSNode* dsNode, glm::uint& numLeaves) const;
	void getNumNodes(const SpatialDSNode* dsNode, glm::uint& numNodes) const;
	void getNumPrimitives(const SpatialDSNode* dsNode, glm::uint& numPrimitives) const;

	static void resolveNodeCollisions(
		const Ray& ray, HitInfo& hitInfo, const Node* node, 
		const VertexGPU* vertices, const glm::u32* indices
	);
	void resolveRayQuery(
		const SpatialDSNode* node, const Ray& ray, HitInfo& hitInfo, 
		const VertexGPU* vertices, const glm::u32* indices
	);

public:
	MultiDataStructure(const std::vector<LevelConfig>& levels);
	MultiDataStructure(const std::vector<LevelConfig>& levels, const BuildPolicy& buildPolicy);
	virtual ~MultiDataStructure();

	void build(const Node* nodes, size_t numNodes, const AABB& aabb);

	void checkSanity() { this->checkSanity(_rootNode, 0); }
	void collapseNodes() { this->collapseNodes(_rootNode, 0); }
	void removeEmptyNodes(glm::uint& deletedNodes);
	void applyConfiguredCleanup();

	void resolveRayQueries(
		const std::vector<Ray>& rays, std::vector<float>& depth, 
		const VertexGPU* vertices, const glm::u32* indices
	);

	static void resolveRayQueriesBruteForce(
		const std::vector<Ray>& rays, std::vector<float>& depth,
		const VertexGPU* vertices, const glm::u32* indices, const Node* nodes, size_t numNodes
	);

	bool exportNodes(const std::string &filename);
	DataStructureLevel getConfiguredNodeType(glm::uint level) const { return getNodeType(level); }
	const BuildPolicy& getBuildPolicy() const { return _buildPolicy; }
	Stats getStats() const;
	void printStats() const;
};

//

class NodeFactory
{
public:
	using Creator = std::function<std::unique_ptr<MultiDataStructure::SpatialDSNode>()>;

	// Register a new type
	template <typename T>
	static void registerType(MultiDataStructure::DataStructureLevel type)
	{
		_creators[type] = []() { return std::make_unique<T>(); };
	}

	// Create an instance from enum
	static MultiDataStructure::SpatialDSNode* create(MultiDataStructure::DataStructureLevel type, const AABB& aabb)
	{
		auto it = _creators.find(type);
		if (it == _creators.end())
			throw std::runtime_error("Unknown type");

		return it->second()->copy(aabb);
	}

private:
	static inline std::unordered_map<MultiDataStructure::DataStructureLevel, Creator> _creators;
};

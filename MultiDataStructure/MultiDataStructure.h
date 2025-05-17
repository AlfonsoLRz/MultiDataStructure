#pragma once

#include "AABB.h"
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
	};

public:
	class SpatialDSNode
	{
	public:
		AABB						_aabb;
		std::vector<const Node*>	_primitives;
		std::vector<SpatialDSNode*> _children;

		SpatialDSNode(const AABB& aabb = AABB()) : _aabb(aabb) {}
		virtual ~SpatialDSNode() = default;

		virtual SpatialDSNode* copy(const AABB& aabb) const = 0;
		virtual bool in(const Node* node) const;
		virtual bool isLeaf() const { return _children.empty(); }
		virtual bool intersects(const RayGPU& ray);
		virtual void split(DataStructureLevel nodeType) = 0;
	};

private:
	glm::uint					_maxLevels;
	SpatialDSNode*				_rootNode;
	glm::uint					_numPrimitives;

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
	void getAverageLeafPrimitives(const SpatialDSNode* dsNode, float& sum, glm::uint& count) const;
	void getNumLeaves(const SpatialDSNode* dsNode, glm::uint& numLeaves) const;
	void getNumNodes(const SpatialDSNode* dsNode, glm::uint& numNodes) const;
	void getNumPrimitives(const SpatialDSNode* dsNode, glm::uint& numPrimitives) const;

	static void resolveNodeCollisions(
		const RayGPU& ray, HitInfo& hitInfo, const Node* node,
		const VertexGPU* vertices, const glm::u32* indices
	);
	void resolveRayQuery(
		const SpatialDSNode* node, const RayGPU& ray, HitInfo& hitInfo,
		const VertexGPU* vertices, const glm::u32* indices
	);

public:
	MultiDataStructure(const std::vector<LevelConfig>& levels);
	virtual ~MultiDataStructure();

	void build(const Node* nodes, size_t numNodes, const AABB& aabb);

	void checkSanity() { this->checkSanity(_rootNode, 0); }
	void collapseNodes() { this->collapseNodes(_rootNode, 0); }
	void removeEmptyNodes(glm::uint& deletedNodes);

	void resolveRayQueries(
		const std::vector<RayGPU>& rays, std::vector<float>& depth,
		const VertexGPU* vertices, const glm::u32* indices
	);

	static void resolveRayQueriesBruteForce(
		const std::vector<RayGPU>& rays, std::vector<float>& depth,
		const VertexGPU* vertices, const glm::u32* indices, const Node* nodes, size_t numNodes
	);

	bool exportNodes(const std::string &filename);
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

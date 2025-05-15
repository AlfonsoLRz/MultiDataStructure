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
		AABB _aabb;
		std::vector<const Node*> _primitives;
		std::vector<SpatialDSNode*> _children;

		SpatialDSNode(const AABB& aabb = AABB()) : _aabb(aabb) {}
		virtual ~SpatialDSNode() = default;

		virtual SpatialDSNode* copy(const AABB& aabb) const = 0;
		virtual bool in(const Node* node) const;
		virtual bool intersects(const Ray& ray);
		virtual void split(DataStructureLevel nodeType) = 0;
	};

private:
	glm::uint					_maxLevels;
	SpatialDSNode*				_rootNode;

	//
	std::vector<LevelConfig>	_levels;
	std::vector<glm::uint> 	    _levelCDF;

private:
	void check(SpatialDSNode* dsNode, glm::uint level);
	void insert(SpatialDSNode* dsNode, const Node* node, glm::uint level);

	void collectNodes(SpatialDSNode* node, std::vector<AABB>& nodes);

	DataStructureLevel getNodeType(const glm::uint level) const;

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
	virtual ~MultiDataStructure();

	void build(DataStructureLevel nodeType, glm::uint maxLevels, const Node* nodes, size_t numNodes, const AABB& aabb);
	void check(DataStructureLevel nodeType, glm::uint maxLevels);

	void build(const std::vector<LevelConfig>& levels, const Node* nodes, size_t numNodes, const AABB& aabb, bool reverse=true);

	void resolveRayQueries(
		const std::vector<Ray>& rays, std::vector<float>& depth, 
		const VertexGPU* vertices, const glm::u32* indices
	);

	static void resolveRayQueriesBruteForce(
		const std::vector<Ray>& rays, std::vector<float>& depth,
		const VertexGPU* vertices, const glm::u32* indices, const Node* nodes, size_t numNodes
	);
	bool exportNodes(const std::string &filename);
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

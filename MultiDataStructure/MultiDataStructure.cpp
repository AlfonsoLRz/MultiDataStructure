#include "stdafx.h"
#include "MultiDataStructure.h"

#include "progressbar.hpp"

//

MultiDataStructure::MultiDataStructure(const std::vector<LevelConfig>& levels) : _maxLevels(0), _rootNode(nullptr), _levels(levels)
{
	_levelCDF.resize(levels.size());
	for (size_t i = 0; i < levels.size(); ++i)
	{
		if (i == 0)
			_levelCDF[i] = levels[i]._numLevels;
		else
			_levelCDF[i] = _levelCDF[i - 1] + levels[i]._numLevels;
	}
}

MultiDataStructure::~MultiDataStructure()
{
	delete _rootNode;
}

void MultiDataStructure::build(DataStructureLevel nodeType, glm::uint maxLevels, const Node* nodes, size_t numNodes, const AABB& aabb)
{
	_maxLevels += maxLevels;

	if (!_rootNode)
		_rootNode = NodeFactory::create(nodeType, aabb);

	for (size_t i = 0; i < numNodes; ++i)
	{
		if (_rootNode->in(&nodes[i]))
			this->insert(nodeType, _rootNode, &nodes[i], 0);
	}
}

void MultiDataStructure::check(DataStructureLevel nodeType, glm::uint maxLevels)
{
	_maxLevels += maxLevels;

	if (!_rootNode)
		throw std::runtime_error("MultiDataStructure not built yet!");

	this->check(nodeType, _rootNode, 0);
}

void MultiDataStructure::build(const std::vector<LevelConfig>& levels, const Node* nodes, size_t numNodes, const AABB& aabb, bool reverse)
{
	for (const auto& level : levels)
	{
		_maxLevels += level._numLevels;

		if (!_rootNode)
			_rootNode = NodeFactory::create(level._levelType, aabb);

		for (size_t i = 0; i < numNodes; ++i)
		{
			if (_rootNode->in(&nodes[i]))
				this->insert(level._levelType, _rootNode, &nodes[i], 0);
		}
	}
}

void MultiDataStructure::resolveRayQueries(
	const std::vector<Ray>& rays, std::vector<float>& depth,
	const VertexGPU* vertices, const glm::u32* indices
)
{
	depth.resize(rays.size());

	#pragma omp parallel for
	for (int rayIdx = 0; rayIdx < static_cast<int>(rays.size()); ++rayIdx)
	{
		HitInfo hitInfo;
		hitInfo._t = FLT_MAX;
		hitInfo._hit = 0;

		resolveRayQuery(_rootNode, rays[rayIdx], hitInfo, vertices, indices);
		depth[rayIdx] = hitInfo._hit == 1 ? hitInfo._t : FLT_MAX;
	}
}

void MultiDataStructure::resolveRayQueriesBruteForce(
	const std::vector<Ray>& rays, std::vector<float>& depth,
	const VertexGPU* vertices, const glm::u32* indices, const Node* nodes, size_t numNodes)
{
	depth.resize(rays.size());

	#pragma omp parallel for
	for (int rayIdx = 0; rayIdx < static_cast<int>(rays.size()); ++rayIdx)
	{
		HitInfo hitInfo;
		hitInfo._t = FLT_MAX;
		hitInfo._hit = 0;

		for (int nodeIdx = 0; nodeIdx < static_cast<int>(numNodes); ++nodeIdx)
		{
			resolveNodeCollisions(rays[rayIdx], hitInfo, &nodes[nodeIdx], vertices, indices);
		}

		depth[rayIdx] = hitInfo._hit == 1 ? hitInfo._t : FLT_MAX;
	}
}

bool MultiDataStructure::exportNodes(const std::string& filename)
{
	std::vector<AABB> nodes;
	this->collectNodes(_rootNode, nodes);

	// Export as obj
	std::ofstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Error opening file: " << filename << std::endl;
		return false;
	}

	file << "x_min,y_min,z_min,x_max,y_max,z_max" << std::endl;
	for (const auto& node : nodes)
	{
		file << node.min().x << "," << node.min().y << "," << node.min().z << ",";
		file << node.max().x << "," << node.max().y << "," << node.max().z << std::endl;
	}

	file.close();
	return true;
}

bool MultiDataStructure::SpatialDSNode::in(const Node* node) const
{
	return _aabb.collides(node->_minPoint, node->_maxPoint);
}

bool MultiDataStructure::SpatialDSNode::intersects(const Ray& ray)
{
	return _aabb.intersects(ray);
}

void MultiDataStructure::check(SpatialDSNode* dsNode, glm::uint level)
{
	DataStructureLevel nodeType = this->getNodeType(level);

	if (dsNode->_children.empty())
	{
		if (level < _maxLevels)
		{
			dsNode->split(nodeType);

			for (auto& child : dsNode->_children)
			{
				for (auto& primitive : dsNode->_primitives)
					if (child->in(primitive))
						this->insert(child, primitive, level + 1);
			}

			dsNode->_primitives.clear();
		}
		else
		{
			for (auto& primitive : dsNode->_primitives)
			{
				if (dsNode->in(primitive))
					dsNode->_primitives.push_back(primitive);
			}
		}
	}
	else
	{
		for (auto& child : dsNode->_children)
			this->check(child, level + 1);
	}
}

void MultiDataStructure::insert(SpatialDSNode* dsNode, const Node* node, glm::uint level)
{
	DataStructureLevel nodeType = this->getNodeType(level);

	if (dsNode->_children.empty()) 
	{
		dsNode->_primitives.push_back(node);
		if (level < _maxLevels)
		{
			dsNode->split(nodeType);

			#pragma omp parallel for
			for (int childrenIdx = 0; childrenIdx < dsNode->_children.size(); ++childrenIdx)
				for (auto& primitive : dsNode->_primitives)
					if (dsNode->_children[childrenIdx]->in(primitive))
						this->insert(dsNode->_children[childrenIdx], primitive, level + 1);

			dsNode->_primitives.clear();
		}
	}
	else 
		for (auto& child : dsNode->_children)
			this->insert(child, node, level + 1);
}

void MultiDataStructure::collectNodes(SpatialDSNode* node, std::vector<AABB>& nodes)
{
	nodes.push_back(node->_aabb);

	for (auto& child : node->_children)
		this->collectNodes(child, nodes);
}

MultiDataStructure::DataStructureLevel MultiDataStructure::getNodeType(const glm::uint level) const
{
	glm::uint idx = 0;
	const DataStructureLevel nodeType = _levels.front()._levelType;

	while (_levelCDF[idx] < level)
		++idx;

	return _levels[idx]._levelType;
}

void MultiDataStructure::resolveNodeCollisions(
	const Ray& ray, HitInfo& hitInfo, const Node* node,
	const VertexGPU* vertices, const glm::u32* indices
)
{
	if (node->_triangleIndex != INT_MAX)
	{
		const glm::vec3 a = vertices[indices[node->_triangleIndex * 3 + 0]]._position;
		const glm::vec3 b = vertices[indices[node->_triangleIndex * 3 + 1]]._position;
		const glm::vec3 c = vertices[indices[node->_triangleIndex * 3 + 2]]._position;

		const glm::vec3 normal = glm::normalize(cross(b - a, c - a));
		const float t = -(dot(normal, ray._origin) + -glm::dot(normal, a)) / dot(normal, ray._direction);
		if (t > 0.0f)
		{
			const glm::vec3 p = ray._origin + t * ray._direction;
			const glm::vec3 v0 = c - a;
			const glm::vec3 v1 = b - a;
			const glm::vec3 v2 = p - a;
			const float dot00 = glm::dot(v0, v0);
			const float dot01 = glm::dot(v0, v1);
			const float dot02 = glm::dot(v0, v2);
			const float dot11 = glm::dot(v1, v1);
			const float dot12 = glm::dot(v1, v2);
			const float inverseDenominator = 1.0f / (dot00 * dot11 - dot01 * dot01);
			const float u = (dot11 * dot02 - dot01 * dot12) * inverseDenominator;
			const float v = (dot00 * dot12 - dot01 * dot02) * inverseDenominator;

			if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f && t < hitInfo._t)
			{
				hitInfo._hit = 1;
				hitInfo._t = t;
				hitInfo._position = p;
				hitInfo._normal = normal;
			}
		}
	}
}

void MultiDataStructure::resolveRayQuery(
	const SpatialDSNode* node, const Ray& ray, HitInfo& hitInfo, 
	const VertexGPU* vertices, const glm::u32* indices
)
{
	float tFar;
	if (node->_aabb.intersects(ray, tFar))
	{
		if (tFar > hitInfo._t)
			return;

		if (node->_children.empty())
		{
			for (auto& primitive : node->_primitives)
			{
				resolveNodeCollisions(ray, hitInfo, primitive, vertices, indices);
			}
		}
		else
		{
			for (auto& child : node->_children)
				this->resolveRayQuery(child, ray, hitInfo, vertices, indices);
		}
	}
}



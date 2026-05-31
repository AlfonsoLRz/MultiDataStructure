#include "stdafx.h"
#include "MultiDataStructure.h"

#include "progressbar.hpp"

//

MultiDataStructure::MultiDataStructure(const std::vector<LevelConfig>& levels) :
	MultiDataStructure(levels, BuildPolicy())
{
}

MultiDataStructure::MultiDataStructure(const std::vector<LevelConfig>& levels, const BuildPolicy& buildPolicy) :
	_maxLevels(0), _rootNode(nullptr), _numPrimitives(0), _buildPolicy(buildPolicy), _levels(levels)
{
	if (levels.empty())
		throw std::invalid_argument("MultiDataStructure requires at least one level configuration");

	_levelCDF.resize(levels.size());
	for (size_t i = 0; i < levels.size(); ++i)
	{
		if (i == 0)
			_levelCDF[i] = levels[i]._numLevels;
		else
			_levelCDF[i] = _levelCDF[i - 1] + levels[i]._numLevels;

		_maxLevels += levels[i]._numLevels;
	}

	if (_buildPolicy.maxDepth > 0)
		_maxLevels = static_cast<glm::uint>(std::min<size_t>(_maxLevels, _buildPolicy.maxDepth));
}

MultiDataStructure::~MultiDataStructure()
{
}

//void MultiDataStructure::build(DataStructureLevel nodeType, glm::uint maxLevels, const Node* nodes, size_t numNodes, const AABB& aabb)
//{
//	_maxLevels += maxLevels;
//
//	if (!_rootNode)
//		_rootNode = NodeFactory::create(nodeType, aabb);
//
//	for (size_t i = 0; i < numNodes; ++i)
//	{
//		if (_rootNode->in(&nodes[i]))
//			this->insert(nodeType, _rootNode, &nodes[i], 0);
//	}
//}
//
void MultiDataStructure::build(const Node* nodes, size_t numNodes, const AABB& aabb)
{
	_numPrimitives = numNodes;

	if (!_rootNode)
		_rootNode = NodeFactory::create(getNodeType(0), aabb);

	for (size_t i = 0; i < numNodes; ++i)
	{
		if (_rootNode->in(&nodes[i]))
			this->insert(_rootNode.get(), &nodes[i], 0);
	}
}

void MultiDataStructure::removeEmptyNodes(glm::uint& deletedNodes)
{
	deletedNodes = 0;
	if (!_rootNode)
		return;

	this->removeEmptyNodes(_rootNode.get(), 0, deletedNodes);
}

void MultiDataStructure::applyConfiguredCleanup()
{
	if (_buildPolicy.removeEmptyNodes)
	{
		glm::uint deletedNodes = 0;
		do
		{
			removeEmptyNodes(deletedNodes);
		} while (deletedNodes > 0);
	}

	if (_buildPolicy.collapseSingleChild)
		collapseNodes();
}

void MultiDataStructure::resolveRayQueries(
	const std::vector<Ray>& rays, std::vector<float>& depth,
	const VertexGPU* vertices, const glm::u32* indices
)
{
	#pragma omp parallel for
	for (int rayIdx = 0; rayIdx < static_cast<int>(rays.size()); ++rayIdx)
	{
		float tFar;
		HitInfo hitInfo;
		hitInfo._hit = 0;
		hitInfo._t = FLT_MAX;

		if (_rootNode->_aabb.intersects(rays[rayIdx], tFar))
			resolveRayQuery(_rootNode.get(), rays[rayIdx], hitInfo, vertices, indices);

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
	this->collectNodes(_rootNode.get(), nodes);

	// Export as obj
	std::ofstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Error opening file: " << filename << '\n';
		return false;
	}

	file << "x_min,y_min,z_min,x_max,y_max,z_max" << '\n';
	for (const auto& node : nodes)
	{
		file << node.min().x << "," << node.min().y << "," << node.min().z << ",";
		file << node.max().x << "," << node.max().y << "," << node.max().z << '\n';
	}

	file.close();
	return true;
}

MultiDataStructure::Stats MultiDataStructure::getStats() const
{
	Stats stats;
	stats.numLevels = _maxLevels;

	if (!_rootNode)
		return stats;

	float sum = 0.0f;
	glm::uint leafPrimitiveSamples = 0;
	this->getNumLeaves(_rootNode.get(), stats.numLeaves);
	this->getNumNodes(_rootNode.get(), stats.numNodes);
	this->getNumPrimitives(_rootNode.get(), stats.numPrimitives);
	this->getAverageLeafPrimitives(_rootNode.get(), sum, leafPrimitiveSamples);
	stats.averageLeafPrimitives = leafPrimitiveSamples > 0 ? sum / static_cast<float>(leafPrimitiveSamples) : 0.0f;

	return stats;
}

void MultiDataStructure::printStats() const
{
	const Stats stats = getStats();

	std::cout << "MultiDataStructure stats:" << '\n';
	std::cout << "  - Number of levels: " << stats.numLevels << '\n';
	std::cout << "  - Number of primitives: " << stats.numPrimitives << '\n';
	std::cout << "  - Number of nodes: " << stats.numNodes << '\n';
	std::cout << "  - Number of leaves: " << stats.numLeaves << '\n';
	std::cout << "  - Average number of primitives per leaf: " << stats.averageLeafPrimitives << '\n';
}

bool MultiDataStructure::SpatialDSNode::in(const Node* node) const
{
	return _aabb.collides(node->_minPoint, node->_maxPoint);
}

bool MultiDataStructure::SpatialDSNode::intersects(const Ray& ray)
{
	return _aabb.intersects(ray);
}

void MultiDataStructure::insert(SpatialDSNode* dsNode, const Node* node, glm::uint level)
{
	if (dsNode->in(node))
	{
		if (dsNode->_children.empty())
		{
			dsNode->_primitives.push_back(node);

			const LevelConfig& levelConfig = this->getLevelConfig(level);
			if (this->shouldSplit(levelConfig, dsNode->_primitives.size(), level))
			{
				DataStructureLevel nodeType = levelConfig._levelType;
				dsNode->split(nodeType);

				for (auto& child : dsNode->_children)
				{
					for (auto& primitive : dsNode->_primitives)
						if (child->in(primitive))
							this->insert(child.get(), primitive, level + 1);
				}

				dsNode->_primitives.clear();
			}
		}
		else
		{
			for (auto& child : dsNode->_children)
				this->insert(child.get(), node, level + 1);
		}
	}
	//if (level < _maxLevels)
	//{
	//	DataStructureLevel nodeType = this->getNodeType(level);

	//	if (dsNode->_children.empty())
	//		dsNode->split(nodeType);

	//	#pragma omp parallel for
	//	for (int childrenIdx = 0; childrenIdx < dsNode->_children.size(); ++childrenIdx)
	//		if (dsNode->_children[childrenIdx]->in(node))
	//			this->insert(dsNode->_children[childrenIdx], node, level + 1);
	//}
	//else
	//{
	//	dsNode->_primitives.push_back(node);
	//}
}

void MultiDataStructure::collectNodes(const SpatialDSNode* node, std::vector<AABB>& nodes)
{
	nodes.push_back(node->_aabb);

	for (auto& child : node->_children)
		this->collectNodes(child.get(), nodes);
}

void MultiDataStructure::checkSanity(const SpatialDSNode* dsNode, glm::uint level)
{
	assert(dsNode->_children.empty() != dsNode->_primitives.empty());
	//assert(dsNode->_primitives.size() <= 1 || level >= _maxLevels);
	assert(level <= _maxLevels);

	for (auto& child : dsNode->_children)
		this->checkSanity(child.get(), level + 1);
}

void MultiDataStructure::collapseNodes(std::unique_ptr<SpatialDSNode>& dsNode, glm::uint level)
{
	if (dsNode->_children.size() == 1)
	{
		std::unique_ptr<SpatialDSNode> child = std::move(dsNode->_children.front());
		dsNode = std::move(child);
	}
	else
	{
		for (auto& child : dsNode->_children)
			this->collapseNodes(child, level + 1);
	}
}

void MultiDataStructure::removeEmptyNodes(SpatialDSNode* dsNode, glm::uint level, glm::uint& deletedNodes)
{
	for (auto it = dsNode->_children.begin(); it != dsNode->_children.end();)
	{
		if ((*it)->_primitives.empty() == (*it)->_children.empty())
		{
			it = dsNode->_children.erase(it);
			++deletedNodes;
		}
		else
			++it;
	}

	for (auto& child : dsNode->_children)
		this->removeEmptyNodes(child.get(), level + 1, deletedNodes);

	//if (dsNode->_children.size() == 1)
	//{
	//	dsNode->_primitives = dsNode->_children.front()->_primitives;
	//	//dsNode->_aabb = dsNode->_children.front()->_aabb;

	//	delete dsNode->_children.front();
	//	++deletedNodes;

	//	dsNode->_children.clear();

	//	if (level < _maxLevels)
	//	{
	//		DataStructureLevel nodeType = this->getNodeType(level);
	//		dsNode->split(nodeType);

	//		for (auto& child : dsNode->_children)
	//		{
	//			for (auto& primitive : dsNode->_primitives)
	//				if (child->in(primitive))
	//					this->insert(child, primitive, level + 1);
	//		}

	//		dsNode->_primitives.clear();
	//	}
	//}
	//else
	//{
	//	for (auto& child : dsNode->_children)
	//		this->removeEmptyNodes(child, level + 1, deletedNodes);
	//}
}

MultiDataStructure::DataStructureLevel MultiDataStructure::getNodeType(const glm::uint level) const
{
	return getLevelConfig(level)._levelType;
}

const MultiDataStructure::LevelConfig& MultiDataStructure::getLevelConfig(const glm::uint level) const
{
	glm::uint idx = 0;

	while (idx + 1 < _levelCDF.size() && level >= _levelCDF[idx])
		++idx;

	return _levels[idx];
}

bool MultiDataStructure::shouldSplit(const LevelConfig& levelConfig, size_t primitiveCount, glm::uint level) const
{
	if (level >= _maxLevels)
		return false;

	const size_t leafCapacity = levelConfig._leafCapacity > 0 ? levelConfig._leafCapacity : _buildPolicy.leafCapacity;
	const size_t minPrimitivesToSplit = levelConfig._minPrimitivesToSplit > 0 ? levelConfig._minPrimitivesToSplit : _buildPolicy.minPrimitivesToSplit;

	return primitiveCount > leafCapacity && primitiveCount >= minPrimitivesToSplit;
}

void MultiDataStructure::getAverageLeafPrimitives(
	const SpatialDSNode* dsNode, float& sum, glm::uint& count) const
{
	if (dsNode->_children.empty())
	{
		sum += static_cast<float>(dsNode->_primitives.size());
		++count;
	}
	else
	{
		for (const auto& child : dsNode->_children)
			this->getAverageLeafPrimitives(child.get(), sum, count);
	}
}

void MultiDataStructure::getNumLeaves(const SpatialDSNode* dsNode, glm::uint& numLeaves) const
{
	if (dsNode->isLeaf())
	{
		numLeaves += 1;
	}
	else
	{
		for (const auto& child : dsNode->_children)
			this->getNumLeaves(child.get(), numLeaves);
	}
}

void MultiDataStructure::getNumNodes(const SpatialDSNode* dsNode, glm::uint& numNodes) const
{
	++numNodes;
	for (const auto& child : dsNode->_children)
		this->getNumNodes(child.get(), numNodes);
}

void MultiDataStructure::getNumPrimitives(const SpatialDSNode* dsNode, glm::uint& numPrimitives) const
{
	if (dsNode->isLeaf())
	{
		numPrimitives += static_cast<glm::uint>(dsNode->_primitives.size());
	}
	else
	{
		for (const auto& child : dsNode->_children)
			this->getNumPrimitives(child.get(), numPrimitives);
	}
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
		if (t >= 0.0f)
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
	//if (node->_aabb.intersects(ray, tFar) && tFar < hitInfo._t)
	//{
		if (node->_children.empty())
		{
			for (auto& primitive : node->_primitives)
			{
				resolveNodeCollisions(ray, hitInfo, primitive, vertices, indices);
			}
		}
		else
		{
			glm::uint minDistanceIdx = 0;
			float minDistance = FLT_MAX, tFar;
			float tReservoir[8];

			for (size_t i = 0; i < node->_children.size(); ++i)
			{
				tReservoir[i] = node->_children[i]->_aabb.intersects(ray, tFar) ? tFar : FLT_MAX;

				if (tReservoir[i] < minDistance)
				{
					minDistance = tReservoir[i];
					minDistanceIdx = i;
				}
			}

			for (glm::uint i = 0; i < node->_children.size(); ++i)
			{
				glm::uint idx = (minDistanceIdx + i) % node->_children.size();
				if (tReservoir[idx] < hitInfo._t)
					this->resolveRayQuery(node->_children[idx].get(), ray, hitInfo, vertices, indices);
			}

			//for (auto& child : node->_children)
			//{
			//	this->resolveRayQuery(child, ray, hitInfo, vertices, indices);
			//}
		}
	//}
}



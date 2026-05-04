#include "../../stdafx.h"
#include "BIH.h"

#include "KDTree.h"

PointGpu::BIH::BIH()
	: _index(std::make_unique<KDTree>())
{
}

PointGpu::BIH::~BIH() = default;

bool PointGpu::BIH::isAvailable(std::string* error)
{
	return KDTree::isAvailable(error);
}

int PointGpu::BIH::deviceCount()
{
	return KDTree::deviceCount();
}

PointGpu::BuildResult PointGpu::BIH::build(const PointCloud& cloud, const SchemaConfig& schema, const Options& options)
{
	Options bihOptions = options;
	bihOptions.builder = "bih";
	return _index->build(cloud, schema, bihOptions);
}

PointGpu::QueryResult PointGpu::BIH::query(const std::vector<Query>& queries, const Options& options) const
{
	return _index ? _index->query(queries, options) : QueryResult();
}

bool PointGpu::BIH::built() const
{
	return _index && _index->built();
}

size_t PointGpu::BIH::pointCount() const
{
	return _index ? _index->pointCount() : 0;
}

size_t PointGpu::BIH::nodeCount() const
{
	return _index ? _index->nodeCount() : 0;
}

size_t PointGpu::BIH::leafCount() const
{
	return _index ? _index->leafCount() : 0;
}

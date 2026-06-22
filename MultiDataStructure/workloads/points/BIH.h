#pragma once

#include "../../stdafx.h"
#include "../../core/Config.h"
#include "PointCloud.h"
#include "PointGpuTypes.h"

namespace PointGpu
{
	class KDTree;

	class BIH
	{
	public:
		BIH();
		~BIH();

		BIH(const BIH&) = delete;
		BIH& operator=(const BIH&) = delete;

		static bool isAvailable(std::string* error = nullptr);
		static int deviceCount();

		BuildResult build(const PointCloud& cloud, const SchemaConfig& schema, const Options& options = {});
		QueryResult query(const std::vector<Query>& queries, const Options& options = {}) const;

		bool built() const;
		size_t pointCount() const;
		size_t nodeCount() const;
		size_t leafCount() const;

	private:
		std::unique_ptr<KDTree>	_index;
	};
}

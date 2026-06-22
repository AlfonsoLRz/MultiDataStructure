#pragma once

#include "../../stdafx.h"
#include "../../core/Config.h"
#include "PointCloud.h"
#include "PointGpuTypes.h"

namespace PointGpu
{
	class MixedTree
	{
	public:
		MixedTree();
		~MixedTree();

		MixedTree(const MixedTree&) = delete;
		MixedTree& operator=(const MixedTree&) = delete;

		static bool isAvailable(std::string* error = nullptr);
		static int deviceCount();

		BuildResult build(const PointCloud& cloud, const SchemaConfig& schema, const Options& options = {});
		QueryResult query(const std::vector<Query>& queries, const Options& options = {}) const;

		bool built() const;
		size_t pointCount() const;
		size_t nodeCount() const;
		size_t leafCount() const;

	private:
		struct DeviceState;
		void release();
		void releaseTree();

		std::unique_ptr<DeviceState>	_state;
	};
}

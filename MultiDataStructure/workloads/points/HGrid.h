#pragma once

#include "../../stdafx.h"
#include "../../core/Config.h"
#include "PointCloud.h"
#include "PointGpuTypes.h"

namespace PointGpu
{
	class HGrid
	{
	public:
		HGrid();
		~HGrid();

		HGrid(const HGrid&) = delete;
		HGrid& operator=(const HGrid&) = delete;

		static bool isAvailable(std::string* error = nullptr);
		static int deviceCount();

		BuildResult build(const PointCloud& cloud, const SchemaConfig& schema, const Options& options = {});
		QueryResult query(const std::vector<Query>& queries, const Options& options = {}) const;

		bool built() const;
		size_t pointCount() const;
		size_t levelCount() const;
		size_t cellCount() const;
		glm::uvec3 dimensions(size_t level) const;

	private:
		struct DeviceState;
		void release();

		std::unique_ptr<DeviceState>	_state;
	};
}

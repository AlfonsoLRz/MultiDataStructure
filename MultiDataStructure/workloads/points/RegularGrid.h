#pragma once

#include "../../stdafx.h"
#include "../../core/Config.h"
#include "PointCloud.h"
#include "PointGpuTypes.h"

namespace PointGpu
{
	class RegularGrid
	{
	public:
		RegularGrid();
		~RegularGrid();

		RegularGrid(const RegularGrid&) = delete;
		RegularGrid& operator=(const RegularGrid&) = delete;

		static bool isAvailable(std::string* error = nullptr);
		static int deviceCount();

		BuildResult build(const PointCloud& cloud, const SchemaConfig& schema, const Options& options = {});
		QueryResult query(const std::vector<Query>& queries, const Options& options = {}) const;

		bool built() const;
		size_t pointCount() const;
		size_t cellCount() const;
		glm::uvec3 dimensions() const;

	private:
		struct DeviceState;
		void release();
		void releaseGrid();

		std::unique_ptr<DeviceState> _state;
	};
}

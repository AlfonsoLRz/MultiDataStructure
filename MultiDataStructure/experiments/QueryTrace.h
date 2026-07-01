#pragma once

#include "../stdafx.h"

namespace Experiments
{
	// One recorded query from an application trace CSV — the same header-keyed format
	// written by --query-trace and replayed by tools/pcl_point_baseline.cpp and
	// scripts/compare_frameworks.py, so traces round-trip across all tools.
	struct TraceQuery
	{
		enum class Kind
		{
			Range,
			CountRange,
			Radius,
			Knn,
		};

		Kind		_kind = Kind::Range;
		glm::vec3	_minBound = glm::vec3(0.0f);
		glm::vec3	_maxBound = glm::vec3(0.0f);
		glm::vec3	_center = glm::vec3(0.0f);
		float		_radius = 0.0f;
		size_t		_k = 0;
	};

	// Loads a query trace CSV. Columns are keyed by header name: query_type
	// (range|count_range|radius|knn), bounds_min_x/y/z, bounds_max_x/y/z,
	// center_x/y/z, radius, k. Unknown query types and extra columns are ignored.
	// Throws on missing file or empty/parseless content.
	std::vector<TraceQuery> loadQueryTrace(const std::string& path);

	// Most frequent kNN k in the trace (0 when the trace has no kNN queries). Used as
	// the workload-level default so CPU evaluation matches the recorded application.
	size_t dominantKnnK(const std::vector<TraceQuery>& trace);
}

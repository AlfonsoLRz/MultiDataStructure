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

		// World coordinates, kept in double until the cloud's local transform. They used to be
		// glm::vec3, and float has a 0.25 m step at UTM northings (~4e6): every replayed center
		// moved by up to 12 cm, more than the median query radius on some curated scenes, while
		// the external baselines read the same CSV as doubles and answered the recorded queries.
		Kind		_kind = Kind::Range;
		glm::dvec3	_minBound = glm::dvec3(0.0);
		glm::dvec3	_maxBound = glm::dvec3(0.0);
		glm::dvec3	_center = glm::dvec3(0.0);
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

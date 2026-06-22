#pragma once

#include "../stdafx.h"
#include "SchemaSearch.h"
#include "SchemaSelector.h"

namespace Experiments
{
	// A loaded surrogate model built once per run and cheap to query thereafter; active is false when no path was configured or the model failed to load.
	struct SurrogateAcquisition
	{
		SchemaSelectorModel	_model;
		std::string			_modelPath;
		bool				_active = false;
	};

	// Loads the selector JSON at modelPath; on any failure returns an inactive acquisition rather than throwing, so the GA falls back to its mutation/immigration path.
	SurrogateAcquisition loadSurrogateAcquisition(const std::string& modelPath);

	// Generates a fresh genome pool, scores it with the surrogate against (cloud, workload), and returns the lowest-score topK (already materialised to disk); the caller handles dedup and seed decorrelates the pool.
	std::vector<SchemaCandidate> acquireSurrogateProposals(
		const SurrogateAcquisition& acquisition,
		const SchemaGenerationOptions& generationOptions,
		size_t poolSize,
		size_t topK,
		const PointCloud& cloud,
		const WorkloadProfile& workload,
		uint32_t seed);
}

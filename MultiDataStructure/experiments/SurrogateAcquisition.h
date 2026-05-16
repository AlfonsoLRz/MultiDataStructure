#pragma once

#include "../stdafx.h"
#include "SchemaSearch.h"
#include "SchemaSelector.h"

namespace Experiments
{
	// Holds a loaded surrogate model + the cloud/workload it should be evaluated against. Built
	// once per evolutionary run; cheap to query thereafter (no I/O, no feature pipeline rebuild).
	// `active` is false when no surrogate path was configured or when the model failed to load.
	struct SurrogateAcquisition
	{
		SchemaSelectorModel model;
		std::string modelPath;
		bool active = false;
	};

	// Attempts to load the exported selector JSON at `modelPath`. On any failure (empty path,
	// missing file, parse error, measured-best artifact) returns an inactive acquisition state
	// rather than throwing — the GA falls back to its mutation/immigration path automatically.
	SurrogateAcquisition loadSurrogateAcquisition(const std::string& modelPath);

	// Generates a pool of fresh genomes through the existing `generateSchemaCandidates` path,
	// scores each one with the surrogate model against (cloud, workload), and returns the lowest
	// predicted-score `topK` candidates. Returned candidates have already been materialised to
	// disk under generationOptions.outputDirectory by the generator, so they can be replayed.
	//
	// Dedup against the GA's running `seenSignatures` set is the caller's responsibility — this
	// helper does not know about the GA's state. `seed` lets the GA decorrelate the surrogate's
	// pool from its random-immigration RNG.
	std::vector<SchemaCandidate> acquireSurrogateProposals(
		const SurrogateAcquisition& acquisition,
		const SchemaGenerationOptions& generationOptions,
		size_t poolSize,
		size_t topK,
		const PointCloud& cloud,
		const WorkloadProfile& workload,
		uint32_t seed);
}

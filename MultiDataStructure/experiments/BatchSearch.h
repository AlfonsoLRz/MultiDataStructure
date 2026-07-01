#pragma once

#include "../stdafx.h"
#include "SchemaSearch.h"

// Batch-distribution schema search (mode "batch-search").
//
// The deep-learning dataloader regime: a workload is a MANIFEST of many small
// (cloud, trace) pairs — e.g. PointNet++ blocks with their per-layer neighborhood
// queries (scripts/capture_dl_workload.py). One schema is tuned for the whole
// distribution; every candidate is scored by the amortized wall cost of a full pass
// (index build PLUS query replay for every pair), because real dataloaders rebuild
// the structure per block. This is the regime where build time is first-class,
// unlike the one-big-cloud benchmark.
namespace Experiments
{
	int runBatchSearch(const SchemaSearchOptions& options);
}

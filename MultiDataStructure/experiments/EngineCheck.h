#pragma once

#include "../stdafx.h"

namespace Experiments
{
	// `--mode engine-check`: build each schema on one cloud, replay one fixed query set, and write
	// what the engine returned and how long it took. It exists for two jobs the schema search is
	// the wrong tool for:
	//
	//   1. Equivalence. Per-query result checksums and counters plus a per-schema structure digest,
	//      so a binary that changes the engine's layout can be diffed query-for-query against one
	//      that did not (scripts/diff_engine_equivalence.py).
	//   2. A/B timing. Per-type latency over warm-up + timed passes of the same queries, the same
	//      protocol as tools/indexicon_point_baseline.cpp, without any search machinery between the
	//      clock and the traversal.
	//
	// Queries come from --input-trace (world frame, strided to --queries exactly like the search's
	// replay) or, without a trace, from a seeded generator covering every query kind including
	// range and count, which application traces do not exercise.
	struct EngineCheckOptions
	{
		std::string					_inputPath;
		std::vector<std::string>	_schemaPaths;
		std::string					_inputTracePath;
		std::string					_queryCsvPath;
		std::string					_summaryCsvPath;
		size_t						_queryCount = 0;
		uint32_t					_querySeed = 1337;
		size_t						_warmupPasses = 1;
		size_t						_timedPasses = 3;
		size_t						_buildRepeats = 1;
		// Timed passes run with QueryTelemetry::Off unless this is set; pass 0 is always Full.
		bool						_timedTelemetryFull = false;
		bool						_useBinaryCache = true;
		bool						_enableLeafMicroIndexes = false;
		size_t						_leafMicroIndexThreshold = 512;
		// Build every schema first and rotate one pass per schema (--interleave), so a slow stretch
		// of a shared machine hits all schemas instead of whichever one it coincided with.
		bool						_interleave = false;
		// Out of core (testing.md 2q): each schema's tree goes to a LeafStore in this directory, named
		// by its structure digest and reused when one already matches, and every query reads its
		// leaves from there. Empty = in memory.
		std::string					_leafStoreDir;
		// LRU cache as a fraction of the store's size; 0 = every leaf is a device read.
		double						_storeCacheFraction = 0.0;
		// Also read the page of every child group a query tests (else the pool stays in memory).
		bool						_storePagedHierarchy = false;
		// Block device of the store (e.g. sdg): its reads per timed pass, beyond the store's own,
		// are reported, so passes shared with other I/O on the device can be flagged.
		std::string					_storeDevice;
		bool						_storeRequireDirect = true;
		// --store-regimes: several storage regimes timed in turn on the same built trees, so a 100M
		// cell builds each schema once instead of once per regime. Empty = the single regime the
		// flags above describe.
		struct StoreRegime
		{
			bool	_inMemory = true;
			double	_cacheFraction = 0.0;
			bool	_pagedHierarchy = false;

			// "memory", or "<cache fraction>:<memory|paged>" (the CLI form).
			std::string name() const;
		};
		std::vector<StoreRegime>	_storeRegimes;
	};

	int runEngineCheck(const EngineCheckOptions& options);
}

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
		bool						_useBinaryCache = true;
		bool						_enableLeafMicroIndexes = false;
		size_t						_leafMicroIndexThreshold = 512;
	};

	int runEngineCheck(const EngineCheckOptions& options);
}

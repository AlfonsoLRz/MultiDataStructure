#include "../stdafx.h"
#include "EngineCheck.h"

#include "QueryTrace.h"
#include "../core/Config.h"
#include "../workloads/points/LeafStore.h"
#include "../workloads/points/PointCloud.h"
#include "../workloads/points/PointSpatialIndex.h"

namespace
{
	enum class CheckKind
	{
		Range,
		CountRange,
		Radius,
		Knn,
	};

	const char* kindName(CheckKind kind)
	{
		switch (kind)
		{
		case CheckKind::Range: return "range";
		case CheckKind::CountRange: return "count_range";
		case CheckKind::Radius: return "radius";
		case CheckKind::Knn: return "knn";
		}
		return "unknown";
	}

	// Queries are held in the cloud's LOCAL frame, the frame the index answers in.
	struct CheckQuery
	{
		CheckKind	_kind = CheckKind::Range;
		AABB		_bounds;
		glm::vec3	_center = glm::vec3(0.0f);
		float		_radius = 0.0f;
		size_t		_k = 0;
	};

	// splitmix64. Used instead of <random> distributions because those are not specified bit-for-bit
	// across standard libraries, and a golden file captured on Windows must replay on Linux.
	struct SplitMix
	{
		uint64_t _state;

		uint64_t next()
		{
			uint64_t z = (_state += 0x9E3779B97F4A7C15ull);
			z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
			z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
			return z ^ (z >> 31);
		}

		double unit() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
	};

	std::vector<CheckQuery> queriesFromTrace(const std::string& path, size_t target, const PointCloud& cloud)
	{
		const std::vector<Experiments::TraceQuery> trace = Experiments::loadQueryTrace(path);
		const size_t total = trace.size();
		const size_t count = (target > 0 && target < total) ? target : total;

		std::vector<CheckQuery> queries;
		queries.reserve(count);
		for (size_t i = 0; i < count; ++i)
		{
			// Same deterministic stride as the schema search's trace replay.
			const Experiments::TraceQuery& source = trace[(count == total) ? i : (i * total) / count];
			CheckQuery query;
			switch (source._kind)
			{
			case Experiments::TraceQuery::Kind::Radius: query._kind = CheckKind::Radius; break;
			case Experiments::TraceQuery::Kind::Knn: query._kind = CheckKind::Knn; break;
			case Experiments::TraceQuery::Kind::CountRange: query._kind = CheckKind::CountRange; break;
			default: query._kind = CheckKind::Range; break;
			}
			query._center = cloud.toLocalPosition(source._center);
			query._bounds = AABB(cloud.toLocalPosition(source._minBound), cloud.toLocalPosition(source._maxBound));
			query._radius = source._radius;
			query._k = source._k;
			queries.push_back(query);
		}
		return queries;
	}

	// Every query kind, at scales from a few points to a sizeable fraction of the cloud, centred
	// mostly on real points with a share placed anywhere in (and slightly beyond) the bounds.
	std::vector<CheckQuery> generatedQueries(size_t count, uint32_t seed, const PointCloud& cloud)
	{
		std::vector<CheckQuery> queries;
		if (cloud.empty())
			return queries;

		SplitMix rng{ 0xD1B54A32D192ED03ull ^ seed };
		const AABB bounds = cloud.bounds();
		const glm::vec3 size = bounds.size();
		const glm::vec3 margin = size * 0.1f;
		const double diagonal = std::max(static_cast<double>(glm::length(size)), 1e-6);
		static constexpr size_t KnnKs[] = { 1, 8, 16, 64 };

		queries.reserve(count);
		for (size_t i = 0; i < count; ++i)
		{
			glm::vec3 center;
			if (rng.next() % 4 != 0)
			{
				center = cloud.points()[rng.next() % cloud.size()].position;
			}
			else
			{
				for (int axis = 0; axis < 3; ++axis)
					center[axis] = bounds.min()[axis] - margin[axis] +
						static_cast<float>(rng.unit()) * (size[axis] + 2.0f * margin[axis]);
			}

			// Log-uniform in [1e-4, 3e-2] of the diagonal, with one query in 16 much larger.
			double scale = diagonal * std::pow(10.0, -4.0 + rng.unit() * (std::log10(3e-2) + 4.0));
			if (rng.next() % 16 == 0)
				scale = diagonal * 0.1;

			CheckQuery query;
			query._kind = static_cast<CheckKind>(i % 4);
			query._center = center;
			switch (query._kind)
			{
			case CheckKind::Range:
			case CheckKind::CountRange:
			{
				glm::vec3 half;
				for (int axis = 0; axis < 3; ++axis)
					half[axis] = static_cast<float>(scale * (0.25 + rng.unit()));
				query._bounds = AABB(center - half, center + half);
				break;
			}
			case CheckKind::Radius:
				query._radius = (rng.next() % 32 == 0) ? 0.0f : static_cast<float>(scale);
				break;
			case CheckKind::Knn:
				query._k = KnnKs[(i / 4) % std::size(KnnKs)];
				break;
			}
			queries.push_back(query);
		}
		return queries;
	}

	struct QueryRecord
	{
		size_t		_returned = 0;
		uint64_t	_checksum = 0;
		size_t		_visited = 0;
		size_t		_testedBoxes = 0;
		size_t		_tested = 0;
		size_t		_fullyContained = 0;
	};

	PointSpatialIndex::QueryStats runQuery(const PointSpatialIndex& index, const CheckQuery& query,
										   PointSpatialIndex::QueryTelemetry telemetry)
	{
		switch (query._kind)
		{
		case CheckKind::Range: return index.rangeQuery(query._bounds, telemetry)._stats;
		case CheckKind::CountRange: return index.countRange(query._bounds, telemetry)._stats;
		case CheckKind::Radius: return index.radiusQuery(query._center, query._radius, telemetry)._stats;
		case CheckKind::Knn: return index.knnQuery(query._center, query._k, telemetry)._stats;
		}
		return {};
	}

	// Huge pages backing this process right now, in MB (Linux; -1 elsewhere), so a --pool-huge-pages
	// run can show its pools really got them.
	double anonHugePagesMb()
	{
#if defined(__linux__)
		std::ifstream rollup("/proc/self/smaps_rollup");
		std::string line;
		while (std::getline(rollup, line))
		{
			if (line.rfind("AnonHugePages:", 0) == 0)
			{
				std::istringstream fields(line.substr(14));
				double kb = 0.0;
				fields >> kb;
				return kb / 1024.0;
			}
		}
		return 0.0;
#else
		return -1.0;
#endif
	}

	std::string csvField(const std::string& value)
	{
		if (value.find_first_of(",\"\n") == std::string::npos)
			return value;
		std::string escaped = "\"";
		for (const char c : value)
		{
			if (c == '"')
				escaped += '"';
			escaped += c;
		}
		return escaped + '"';
	}
}

namespace
{
	// One schema under measurement: its index, the per-query record of the instrumented pass and
	// the timed latency gathered so far.
	struct Measured
	{
		std::string							_schemaPath;
		std::string							_schemaName;
		std::unique_ptr<PointSpatialIndex>	_index;
		std::vector<double>					_buildMs;
		PointSpatialIndex::Stats			_stats;
		uint64_t							_digest = 0;
		std::vector<QueryRecord>			_records;
		std::array<double, 4>				_latencySum{};
		std::array<size_t, 4>				_latencyCount{};
		// From the instrumented pass: entered nodes per depth summed over the query set, and the
		// nanoseconds charged to each depth (PointSpatialIndex::takeDepthTiming), so each level's
		// visits can be priced (testing.md 2s).
		std::array<double, PointSpatialIndex::QueryStats::MaxBreakdownDepth>	_visitsByDepth{};
		std::array<double, PointSpatialIndex::QueryStats::MaxBreakdownDepth>	_visitNsByDepth{};
		// Mean latency over all queries of each timed pass, in pass order.
		std::vector<double>					_passMeansMs;
		size_t								_passMismatches = 0;
		// Out of core: the store file, the store the index reads from in the current regime, and its
		// counters summed over that regime's timed passes.
		std::string							_storePath;
		Experiments::EngineCheckOptions::StoreRegime	_regime;
		std::unique_ptr<LeafStore>			_store;
		LeafStore::Counters					_storeTotals;
		double								_deviceExtraBytes = 0.0;
		size_t								_storePasses = 0;
	};

	// Writes the index's store, or finds it already written.
	void writeLeafStore(Measured& measured, const Experiments::EngineCheckOptions& options)
	{
		const PointSpatialIndex::PoolView pool = measured._index->poolView();
		std::ostringstream name;
		// The digest describes the tree, not the order of its pool, so the layout names the file too.
		name << std::hex << std::setw(16) << std::setfill('0') << measured._digest
			 << (PointSpatialIndex::poolLayout() == PointSpatialIndex::PoolLayout::PageBlocked ? "_paged" : "") << ".store";
		measured._storePath = (std::filesystem::path(options._leafStoreDir) / name.str()).string();
		if (!LeafStore::matches(measured._storePath, pool, measured._digest))
		{
			const auto start = std::chrono::steady_clock::now();
			LeafStore::write(measured._storePath, pool, measured._digest);
			std::cout << "  store " << measured._storePath << " written in " << std::fixed << std::setprecision(1)
					  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
			std::cout.unsetf(std::ios::floatfield);
		}
	}

	// Puts a built schema in one storage regime with fresh measurement state: detached and in
	// memory, or reading from its store with the regime's cache and hierarchy.
	void applyRegime(Measured& measured, const Experiments::EngineCheckOptions::StoreRegime& regime,
					 const Experiments::EngineCheckOptions& options)
	{
		measured._index->attachStore(nullptr);
		measured._store.reset();
		measured._regime = regime;
		std::fill(measured._records.begin(), measured._records.end(), QueryRecord{});
		measured._latencySum = {};
		measured._latencyCount = {};
		measured._visitsByDepth = {};
		measured._visitNsByDepth = {};
		measured._passMeansMs.clear();
		measured._passMismatches = 0;
		measured._storeTotals = {};
		measured._deviceExtraBytes = 0.0;
		measured._storePasses = 0;
		if (regime._inMemory)
			return;

		LeafStore::Options storeOptions;
		storeOptions._cacheBytes = static_cast<size_t>(regime._cacheFraction * static_cast<double>(std::filesystem::file_size(measured._storePath)));
		storeOptions._pagedHierarchy = regime._pagedHierarchy;
		storeOptions._requireDirect = options._storeRequireDirect;
		measured._store = std::make_unique<LeafStore>(measured._storePath, measured._index->poolView(), measured._digest, storeOptions);
		measured._index->attachStore(measured._store.get());
	}

	Measured buildMeasured(const std::string& schemaPath, const PointCloud& cloud, const Experiments::EngineCheckOptions& options, size_t queryCount)
	{
		SchemaConfig schema = Config::loadSchemaConfig(schemaPath);
		if (options._enableLeafMicroIndexes)
		{
			schema._buildPolicy._enableLeafMicroIndexes = true;
			schema._buildPolicy._leafMicroIndexThreshold = options._leafMicroIndexThreshold;
		}

		Measured measured;
		measured._schemaPath = schemaPath;
		measured._schemaName = schema._name.empty() ? std::filesystem::path(schemaPath).stem().string() : schema._name;
		// Rebuilt from scratch per repeat (the index holds atomics, so it is not reassignable).
		for (size_t b = 0; b < std::max<size_t>(1, options._buildRepeats); ++b)
		{
			measured._index.reset();
			measured._index = std::make_unique<PointSpatialIndex>();
			const auto start = std::chrono::steady_clock::now();
			measured._index->build(cloud, schema);
			measured._buildMs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
		}
		measured._stats = measured._index->stats();
		measured._digest = measured._index->structureDigest();
		measured._records.resize(queryCount);
		if (!options._leafStoreDir.empty())
			writeLeafStore(measured, options);
		return measured;
	}

	void runPass(Measured& measured, const std::vector<CheckQuery>& queries, size_t pass, const Experiments::EngineCheckOptions& options,
				 bool warmSweep = false)
	{
		const PointSpatialIndex& index = *measured._index;
		const bool timed = pass >= options._warmupPasses;
		// Interleaved, the previous turn belonged to another schema, so its data filled the caches.
		// An untimed sweep first puts this schema back in the warm state a sequential measurement
		// times; without it interleaving measured a colder regime (kdtree/qt16 2.7 vs 3.1).
		// Out of core there is no warm state to restore: every pass starts with an empty store
		// cache, and fills it only from its own queries.
		LeafStore* store = measured._store.get();
		if (timed && warmSweep && !store)
			for (const CheckQuery& query : queries)
				runQuery(index, query, PointSpatialIndex::QueryTelemetry::Off);
		if (store)
		{
			store->clearCache();
			store->resetCounters();
		}
		const uint64_t deviceSectorsBefore = store ? LeafStore::deviceSectorsRead(options._storeDevice) : 0;
		// Pass 0 is always instrumented, since the per-query counters come from it; timed passes
		// run uninstrumented unless --timed-telemetry full asks otherwise.
		const PointSpatialIndex::QueryTelemetry telemetry = (pass == 0 || !timed || options._timedTelemetryFull)
			? PointSpatialIndex::QueryTelemetry::Full
			: PointSpatialIndex::QueryTelemetry::Off;
		std::array<double, PointSpatialIndex::QueryStats::MaxBreakdownDepth> depthNs{};
		std::array<uint64_t, PointSpatialIndex::QueryStats::MaxBreakdownDepth> depthVisits{};
		if (pass == 0)
			PointSpatialIndex::takeDepthTiming(depthNs, depthVisits);	// discard whatever came before this pass
		double passMs = 0.0;
		for (size_t q = 0; q < queries.size(); ++q)
		{
			const PointSpatialIndex::QueryStats queryStats = runQuery(index, queries[q], telemetry);
			QueryRecord& record = measured._records[q];
			if (pass == 0)
			{
				record._returned = queryStats._returnedPoints;
				record._checksum = queryStats._resultChecksum;
				record._visited = queryStats._visitedNodes;
				record._testedBoxes = queryStats._testedBoxes;
				record._tested = queryStats._testedPoints;
				record._fullyContained = queryStats._fullyContainedNodes;
				for (size_t depth = 0; depth < measured._visitsByDepth.size(); ++depth)
					measured._visitsByDepth[depth] += static_cast<double>(queryStats._breakdown.visitedByDepth[depth]);
			}
			else if (record._checksum != queryStats._resultChecksum || record._returned != queryStats._returnedPoints)
			{
				++measured._passMismatches;
			}

			if (timed)
			{
				const size_t slot = static_cast<size_t>(queries[q]._kind);
				measured._latencySum[slot] += queryStats._elapsedMs;
				++measured._latencyCount[slot];
				passMs += queryStats._elapsedMs;
			}
		}
		if (pass == 0)
		{
			PointSpatialIndex::takeDepthTiming(depthNs, depthVisits);
			measured._visitNsByDepth = depthNs;
		}
		if (timed && !queries.empty())
			measured._passMeansMs.push_back(passMs / static_cast<double>(queries.size()));
		if (timed && store)
		{
			const LeafStore::Counters& counters = store->counters();
			measured._storeTotals._reads += counters._reads;
			measured._storeTotals._bytes += counters._bytes;
			measured._storeTotals._hits += counters._hits;
			measured._storeTotals._ioNs += counters._ioNs;
			if (!options._storeDevice.empty())
			{
				const uint64_t deviceSectors = LeafStore::deviceSectorsRead(options._storeDevice) - deviceSectorsBefore;
				measured._deviceExtraBytes += static_cast<double>(deviceSectors) * 512.0 - static_cast<double>(counters._bytes);
			}
			++measured._storePasses;
		}
	}

	struct Outputs
	{
		std::ofstream	_queryCsv;
		std::ofstream	_summaryCsv;
	};

	// Writes one schema's rows and console line; false if its passes disagreed.
	bool report(const Measured& measured, const std::vector<CheckQuery>& queries, const PointCloud& cloud,
				const Experiments::EngineCheckOptions& options, Outputs& outputs)
	{
		const PointSpatialIndex::Stats& stats = measured._stats;
		double visited = 0.0, testedBoxes = 0.0, tested = 0.0, returned = 0.0;
		uint64_t querySetChecksum = 0;
		for (size_t q = 0; q < queries.size(); ++q)
		{
			const QueryRecord& record = measured._records[q];
			visited += static_cast<double>(record._visited);
			testedBoxes += static_cast<double>(record._testedBoxes);
			tested += static_cast<double>(record._tested);
			returned += static_cast<double>(record._returned);
			// Folded with the query position so a result moving between queries still registers.
			querySetChecksum += (record._checksum ^ (static_cast<uint64_t>(q) * 0x9E3779B97F4A7C15ull)) + record._returned;
			if (outputs._queryCsv)
			{
				outputs._queryCsv << csvField(measured._schemaName) << ',' << measured._digest << ',' << stats._numNodes << ','
								  << stats._numLeaves << ',' << stats._maxDepth << ',' << q << ',' << kindName(queries[q]._kind) << ','
								  << queries[q]._k << ',' << record._returned << ',' << record._checksum << ',' << record._visited << ','
								  << record._testedBoxes << ',' << record._tested << ',' << record._fullyContained << '\n';
			}
		}

		const auto& sum = measured._latencySum;
		const auto& count = measured._latencyCount;
		const auto mean = [](double total, size_t n) { return n ? total / static_cast<double>(n) : 0.0; };
		const double allSum = sum[0] + sum[1] + sum[2] + sum[3];
		const size_t allCount = count[0] + count[1] + count[2] + count[3];
		const double queryCount = std::max<double>(1.0, static_cast<double>(queries.size()));
		const double buildMean = std::accumulate(measured._buildMs.begin(), measured._buildMs.end(), 0.0) / static_cast<double>(measured._buildMs.size());
		const double buildMin = *std::min_element(measured._buildMs.begin(), measured._buildMs.end());
		std::vector<double> passMeans = measured._passMeansMs;
		std::sort(passMeans.begin(), passMeans.end());
		const double passMedian = passMeans.empty() ? 0.0 : passMeans[passMeans.size() / 2];
		const double passMin = passMeans.empty() ? 0.0 : passMeans.front();

		std::cout << "  " << measured._schemaName << ": build " << std::fixed << std::setprecision(1) << buildMean << " ms, "
				  << stats._numNodes << " nodes, digest " << measured._digest << std::setprecision(6)
				  << " | radius " << mean(sum[2], count[2]) * 1000.0 << " us"
				  << " | knn " << mean(sum[3], count[3]) * 1000.0 << " us"
				  << " | range " << mean(sum[0], count[0]) * 1000.0 << " us"
				  << " | all " << mean(allSum, allCount) * 1000.0 << " us (pass median " << passMedian * 1000.0 << ")"
				  << (measured._passMismatches ? " | PASS MISMATCHES: " + std::to_string(measured._passMismatches) : std::string()) << '\n';
		const LeafStore* store = measured._store.get();
		const double storeQueries = std::max<double>(1.0, static_cast<double>(measured._storePasses) * static_cast<double>(queries.size()));
		const double storePasses = std::max<double>(1.0, static_cast<double>(measured._storePasses));
		if (store)
		{
			std::cout << "    store " << measured._regime.name() << ": " << static_cast<double>(measured._storeTotals._reads) / storeQueries << " reads, "
					  << static_cast<double>(measured._storeTotals._bytes) / storeQueries / 1024.0 << " KiB, "
					  << static_cast<double>(measured._storeTotals._hits) / storeQueries << " hits, "
					  << static_cast<double>(measured._storeTotals._ioNs) / storeQueries / 1000.0 << " us of I/O per query"
					  << (store->direct() ? "" : " (BUFFERED, not direct)");
			if (!options._storeDevice.empty())
				std::cout << " | other reads on " << options._storeDevice << ": " << measured._deviceExtraBytes / storePasses / 1e6 << " MB per pass";
			std::cout << '\n';
		}
		// Per depth, from the instrumented pass: mean entered nodes per query and nanoseconds per
		// entered node (the gap to the next entered node, so a leaf's figure includes its scan).
		// Depths beyond the deepest visited are left out.
		size_t deepest = 0;
		for (size_t depth = 0; depth < measured._visitsByDepth.size(); ++depth)
			if (measured._visitsByDepth[depth] > 0.0)
				deepest = depth;
		std::ostringstream visitsByDepth, nsByDepth;
		visitsByDepth << std::setprecision(6);
		nsByDepth << std::setprecision(6);
		std::cout << "    visits/query and ns/visit by depth:";
		for (size_t depth = 0; depth <= deepest; ++depth)
		{
			const double visits = measured._visitsByDepth[depth];
			const double ns = visits > 0.0 ? measured._visitNsByDepth[depth] / visits : 0.0;
			visitsByDepth << (depth ? ";" : "") << visits / queryCount;
			nsByDepth << (depth ? ";" : "") << ns;
			if (visits > 0.0)
				std::cout << ' ' << depth << ':' << std::fixed << std::setprecision(2) << visits / queryCount << '/' << std::setprecision(0) << ns;
		}
		std::cout << '\n';
		std::cout.unsetf(std::ios::floatfield);

		if (outputs._summaryCsv)
		{
			std::ostringstream passList;
			passList << std::setprecision(10);
			for (size_t i = 0; i < measured._passMeansMs.size(); ++i)
				passList << (i ? ";" : "") << measured._passMeansMs[i];
			outputs._summaryCsv << std::setprecision(10)
								<< PointSpatialIndex::EngineVersion << ',' << csvField(options._inputPath) << ',' << cloud.size() << ','
								<< csvField(measured._schemaName) << ',' << csvField(measured._schemaPath) << ','
								<< (options._inputTracePath.empty() ? "generated" : csvField(options._inputTracePath)) << ','
								<< queries.size() << ',' << options._warmupPasses << ',' << options._timedPasses << ','
								<< buildMean << ',' << buildMin << ',' << stats._numNodes << ',' << stats._numLeaves << ','
								<< stats._maxDepth << ',' << measured._digest << ','
								<< count[0] / options._timedPasses << ',' << mean(sum[0], count[0]) << ','
								<< count[1] / options._timedPasses << ',' << mean(sum[1], count[1]) << ','
								<< count[2] / options._timedPasses << ',' << mean(sum[2], count[2]) << ','
								<< count[3] / options._timedPasses << ',' << mean(sum[3], count[3]) << ','
								<< mean(allSum, allCount) << ',' << visited / queryCount << ',' << tested / queryCount << ','
								<< returned / queryCount << ',' << querySetChecksum << ',' << measured._passMismatches << ','
								<< (options._timedTelemetryFull ? "full" : "off") << ',' << testedBoxes / queryCount << ','
								<< (options._interleave ? 1 : 0) << ',' << passMedian << ',' << passMin << ',' << passList.str() << ','
								<< (store ? store->fileBytes() : 0) << ','
								<< (store ? static_cast<uint64_t>(measured._regime._cacheFraction * static_cast<double>(store->fileBytes())) : 0) << ','
								<< (store && measured._regime._pagedHierarchy ? 1 : 0) << ',' << (store && store->direct() ? 1 : 0) << ','
								<< static_cast<double>(measured._storeTotals._reads) / storeQueries << ','
								<< static_cast<double>(measured._storeTotals._bytes) / storeQueries << ','
								<< static_cast<double>(measured._storeTotals._hits) / storeQueries << ','
								<< static_cast<double>(measured._storeTotals._ioNs) / storeQueries / 1e6 << ','
								<< measured._deviceExtraBytes / storePasses / 1e6 << ',' << measured._regime.name() << ','
								<< visitsByDepth.str() << ',' << nsByDepth.str() << ','
								<< (PointSpatialIndex::hugePages() ? 1 : 0) << ',' << (PointSpatialIndex::prefetchChildren() ? 1 : 0) << ','
								<< anonHugePagesMb() << ','
								<< (PointSpatialIndex::poolLayout() == PointSpatialIndex::PoolLayout::PageBlocked ? "paged" : "dfs") << '\n';
			outputs._summaryCsv.flush();
		}

		if (measured._passMismatches)
		{
			std::cerr << "engine-check: " << measured._schemaName << " returned different results across passes\n";
			return false;
		}
		return true;
	}
}

int Experiments::runEngineCheck(const EngineCheckOptions& options)
{
	if (options._inputPath.empty())
		throw std::invalid_argument("engine-check needs --input <cloud>");
	if (options._schemaPaths.empty())
		throw std::invalid_argument("engine-check needs --schema or --schemas");
	if (options._timedPasses == 0)
		throw std::invalid_argument("engine-check needs at least one timed pass (--measure-repeats >= 1)");

	PointCloud::LoadOptions loadOptions;
	loadOptions._useBinaryCache = options._useBinaryCache;
	const PointCloud cloud = PointCloud::load(options._inputPath, loadOptions);

	const std::vector<CheckQuery> queries = options._inputTracePath.empty()
		? generatedQueries(options._queryCount > 0 ? options._queryCount : 4096, options._querySeed, cloud)
		: queriesFromTrace(options._inputTracePath, options._queryCount, cloud);

	// Price each level's visits on the instrumented pass (the timed passes never see it).
	PointSpatialIndex::setDepthTiming(true);
	std::cout << "engine-check: engine " << PointSpatialIndex::EngineVersion
			  << ", " << cloud.size() << " points, " << queries.size() << " queries ("
			  << (options._inputTracePath.empty() ? "generated" : "trace") << "), "
			  << options._warmupPasses << " warm-up + " << options._timedPasses << " timed passes"
			  << (options._interleave ? ", interleaved across schemas" : "") << '\n';
	// The storage regimes, each timed on the same built trees.
	std::vector<EngineCheckOptions::StoreRegime> regimes = options._storeRegimes;
	if (regimes.empty())
	{
		EngineCheckOptions::StoreRegime regime;
		regime._inMemory = options._leafStoreDir.empty();
		regime._cacheFraction = options._storeCacheFraction;
		regime._pagedHierarchy = options._storePagedHierarchy;
		regimes.push_back(regime);
	}
	if (options._leafStoreDir.empty() && std::any_of(regimes.begin(), regimes.end(), [](const auto& regime) { return !regime._inMemory; }))
		throw std::invalid_argument("engine-check: out-of-core regimes need --leaf-store <dir>");
	if (!options._leafStoreDir.empty())
	{
		std::cout << "engine-check: stores in " << options._leafStoreDir << ", regimes";
		for (const EngineCheckOptions::StoreRegime& regime : regimes)
			std::cout << ' ' << regime.name();
		std::cout << '\n';
	}

	Outputs outputs;
	if (!options._queryCsvPath.empty())
	{
		std::filesystem::create_directories(std::filesystem::absolute(options._queryCsvPath).parent_path());
		outputs._queryCsv.open(options._queryCsvPath);
		if (!outputs._queryCsv)
			throw std::runtime_error("cannot write " + options._queryCsvPath);
		outputs._queryCsv << "schema_name,structure_digest,num_nodes,num_leaves,max_depth,query_id,query_type,k,"
						  << "returned_points,result_checksum,visited_nodes,tested_boxes,tested_points,fully_contained_nodes\n";
	}

	if (!options._summaryCsvPath.empty())
	{
		std::filesystem::create_directories(std::filesystem::absolute(options._summaryCsvPath).parent_path());
		const bool exists = std::filesystem::exists(options._summaryCsvPath);
		outputs._summaryCsv.open(options._summaryCsvPath, std::ios::app);
		if (!outputs._summaryCsv)
			throw std::runtime_error("cannot write " + options._summaryCsvPath);
		if (!exists)
		{
			outputs._summaryCsv << "engine_version,cloud,points,schema_name,schema_path,query_source,queries,warmup_passes,timed_passes,"
								<< "build_ms_mean,build_ms_min,num_nodes,num_leaves,max_depth,structure_digest,"
								<< "range_queries,range_mean_ms,count_queries,count_mean_ms,radius_queries,radius_mean_ms,"
								<< "knn_queries,knn_mean_ms,all_mean_ms,mean_visited_nodes,mean_tested_points,"
								<< "mean_returned_points,query_set_checksum,pass_checksum_mismatches,timed_telemetry,mean_tested_boxes,"
								<< "interleaved,pass_median_ms,pass_min_ms,pass_means_ms,"
								<< "store_file_bytes,store_cache_bytes,store_paged,store_direct,store_reads,store_bytes,store_hits,store_io_ms,"
								<< "store_device_extra_mb,store_regime,visits_by_depth,visit_ns_by_depth,huge_pages,prefetch_children,anon_huge_mb,pool_layout\n";
		}
	}

	const size_t totalPasses = options._warmupPasses + options._timedPasses;
	if (!options._interleave)
	{
		for (const std::string& schemaPath : options._schemaPaths)
		{
			Measured measured = buildMeasured(schemaPath, cloud, options, queries.size());
			for (const EngineCheckOptions::StoreRegime& regime : regimes)
			{
				applyRegime(measured, regime, options);
				for (size_t pass = 0; pass < totalPasses; ++pass)
					runPass(measured, queries, pass, options);
				if (!report(measured, queries, cloud, options, outputs))
					return 2;
			}
		}
		return 0;
	}

	// Interleaved: every index is built first, then each pass runs one sweep of the query set per
	// schema, starting one schema later each pass. On a shared machine a slow stretch of the core
	// (another job on the SMT sibling, a frequency dip) lasts longer than one schema's whole
	// sequential measurement and made identical trees time up to 1.5x apart (testing.md 2h);
	// interleaved, the stretch is spread over every schema and their ratios survive it.
	std::vector<Measured> all;
	all.reserve(options._schemaPaths.size());
	for (const std::string& schemaPath : options._schemaPaths)
		all.push_back(buildMeasured(schemaPath, cloud, options, queries.size()));
	int status = 0;
	for (const EngineCheckOptions::StoreRegime& regime : regimes)
	{
		for (Measured& measured : all)
			applyRegime(measured, regime, options);
		for (size_t pass = 0; pass < totalPasses; ++pass)
			for (size_t i = 0; i < all.size(); ++i)
				runPass(all[(i + pass) % all.size()], queries, pass, options, true);
		for (const Measured& measured : all)
			if (!report(measured, queries, cloud, options, outputs))
				status = 2;
	}
	return status;
}

std::string Experiments::EngineCheckOptions::StoreRegime::name() const
{
	if (_inMemory)
		return "memory";
	std::ostringstream text;
	text << _cacheFraction << ':' << (_pagedHierarchy ? "paged" : "memory");
	return text.str();
}

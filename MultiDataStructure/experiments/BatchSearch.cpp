#include "../stdafx.h"
#include "BatchSearch.h"

#include "Metrics.h"
#include "QueryTrace.h"
#include "../core/Config.h"
#include "../workloads/points/PointCloud.h"
#include "../workloads/points/PointSpatialIndex.h"

#include <boost/json.hpp>

namespace
{
	struct BatchPair
	{
		std::string	_cloudName;
		PointCloud	_cloud;
		std::vector<Experiments::TraceQuery>	_trace;
	};

	struct BatchEval
	{
		Experiments::SchemaCandidate	_candidate;
		double		_totalBuildMs = 0.0;
		double		_totalQueryMs = 0.0;
		double		_score = std::numeric_limits<double>::infinity();	// build + query, one full pass
		Experiments::BuildMetrics	_buildMetrics;						// from the largest pair (repair diagnostics)
		Experiments::QueryMetrics	_queryMetrics;						// across all pairs
		bool		_baseline = false;
	};

	std::vector<BatchPair> loadManifest(const std::string& manifestPath)
	{
		std::ifstream file(manifestPath);
		if (!file.is_open())
			throw std::runtime_error("Unable to open batch manifest: " + manifestPath);
		std::stringstream buffer;
		buffer << file.rdbuf();

		boost::system::error_code error;
		const boost::json::value root = boost::json::parse(buffer.str(), error);
		if (error || !root.is_object())
			throw std::runtime_error("Invalid batch manifest JSON: " + manifestPath);

		const std::filesystem::path baseDir = std::filesystem::path(manifestPath).parent_path();
		std::vector<BatchPair> pairs;

		const boost::json::value* blocks = root.as_object().if_contains("blocks");
		if (!blocks || !blocks->is_array())
			throw std::runtime_error("Batch manifest has no blocks[]: " + manifestPath);

		PointCloud::LoadOptions loadOptions;
		loadOptions._useBinaryCache = false;

		for (const boost::json::value& blockValue : blocks->as_array())
		{
			const boost::json::value* pairList = blockValue.as_object().if_contains("pairs");
			if (!pairList || !pairList->is_array())
				continue;
			for (const boost::json::value& pairValue : pairList->as_array())
			{
				const boost::json::object& pairObject = pairValue.as_object();
				BatchPair pair;
				pair._cloudName = boost::json::value_to<std::string>(pairObject.at("cloud"));
				pair._cloud = PointCloud::load((baseDir / pair._cloudName).string(), loadOptions);
				pair._trace = Experiments::loadQueryTrace((baseDir / boost::json::value_to<std::string>(pairObject.at("trace"))).string());
				pairs.push_back(std::move(pair));
			}
		}

		if (pairs.empty())
			throw std::runtime_error("Batch manifest yielded no (cloud, trace) pairs: " + manifestPath);
		return pairs;
	}

	// One full dataloader pass with this schema: rebuild the index per pair and replay
	// the pair's recorded queries. Returns wall cost decomposed into build and query.
	BatchEval evaluateCandidate(const Experiments::SchemaCandidate& candidate, const std::vector<BatchPair>& pairs, size_t fallbackKnnK)
	{
		BatchEval eval;
		eval._candidate = candidate;
		eval._baseline = candidate._isBaseline;

		std::vector<PointSpatialIndex::QueryStats> samples;
		size_t largestPairPoints = 0;

		for (const BatchPair& pair : pairs)
		{
			PointSpatialIndex index;
			const auto buildStart = std::chrono::high_resolution_clock::now();
			index.build(pair._cloud, candidate._config);
			const auto buildEnd = std::chrono::high_resolution_clock::now();
			const double buildMs = std::chrono::duration<double, std::milli>(buildEnd - buildStart).count();
			eval._totalBuildMs += buildMs;

			for (const Experiments::TraceQuery& query : pair._trace)
			{
				// Traces are in world coordinates; transform into the pair cloud's local
				// frame (identity for the XYZ block files the DL capture emits).
				const glm::vec3 center = pair._cloud.toLocalPosition(glm::dvec3(query._center));
				PointSpatialIndex::QueryStats stats;
				if (query._kind == Experiments::TraceQuery::Kind::Radius)
					stats = index.radiusQuery(center, query._radius)._stats;
				else if (query._kind == Experiments::TraceQuery::Kind::Knn)
					stats = index.knnQuery(center, query._k > 0 ? query._k : fallbackKnnK)._stats;
				else if (query._kind == Experiments::TraceQuery::Kind::CountRange)
					stats = index.countRange(AABB(pair._cloud.toLocalPosition(glm::dvec3(query._minBound)), pair._cloud.toLocalPosition(glm::dvec3(query._maxBound))))._stats;
				else
					stats = index.rangeQuery(AABB(pair._cloud.toLocalPosition(glm::dvec3(query._minBound)), pair._cloud.toLocalPosition(glm::dvec3(query._maxBound))))._stats;
				eval._totalQueryMs += stats._elapsedMs;
				samples.push_back(std::move(stats));
			}

			if (pair._cloud.size() > largestPairPoints)
			{
				largestPairPoints = pair._cloud.size();
				eval._buildMetrics = Experiments::collectBuildMetrics(index.stats(), index.root(), buildMs, candidate._config);
			}
		}

		eval._queryMetrics = Experiments::summarizeQueryStats(samples);
		eval._score = eval._totalBuildMs + eval._totalQueryMs;
		return eval;
	}

	std::vector<Experiments::SchemaCandidate> loadBaselineCandidates(const std::vector<std::string>& schemaPaths)
	{
		// Canonical single-primitive controls; missing files are skipped quietly so the
		// mode works from any working directory.
		static const char* kDefaults[] = {
			"configs/schemas/quadtree.json", "configs/schemas/octree.json",
			"configs/schemas/kdtree.json", "configs/schemas/bvh.json",
		};
		std::vector<std::string> paths = schemaPaths;
		if (paths.empty())
			paths.assign(std::begin(kDefaults), std::end(kDefaults));

		std::vector<Experiments::SchemaCandidate> baselines;
		for (const std::string& path : paths)
		{
			std::error_code error;
			if (!std::filesystem::exists(path, error))
				continue;
			Experiments::SchemaCandidate candidate;
			candidate._config = Config::loadSchemaConfig(path);
			candidate._name = candidate._config._name.empty() ? std::filesystem::path(path).stem().string() : candidate._config._name;
			candidate._path = path;
			candidate._isBaseline = true;
			baselines.push_back(std::move(candidate));
		}
		return baselines;
	}
}

int Experiments::runBatchSearch(const SchemaSearchOptions& options)
{
	if (options._batchManifestPath.empty())
		throw std::invalid_argument("batch-search requires --batch-manifest <json>");

	std::cout << "Loading batch manifest: " << options._batchManifestPath << "\n";
	const std::vector<BatchPair> pairs = loadManifest(options._batchManifestPath);
	size_t totalQueries = 0, totalPoints = 0, dominantK = 0;
	{
		std::vector<TraceQuery> merged;
		for (const BatchPair& pair : pairs)
		{
			totalQueries += pair._trace.size();
			totalPoints += pair._cloud.size();
			merged.insert(merged.end(), pair._trace.begin(), pair._trace.end());
		}
		dominantK = dominantKnnK(merged);
	}
	std::cout << "  " << pairs.size() << " (cloud, trace) pairs, " << totalPoints << " points, "
			  << totalQueries << " queries, dominant kNN k = " << dominantK << "\n";

	// Candidate pool: canonical baselines + generated schemas from the CPU profile.
	std::vector<SchemaCandidate> pool = loadBaselineCandidates(options._schemaPaths);
	const size_t baselineCount = pool.size();

	SchemaGenerationOptions generation = options._generation;
	if (generation._count == 0)
		generation._count = 64;
	generation._primitiveProfile = resolvePrimitiveProfile(generation._primitiveProfile, false);
	generation._outputDirectory.clear();				// no JSON spam for the search pool
	{
		std::vector<SchemaCandidate> generated = generateSchemaCandidates(generation);
		pool.insert(pool.end(), std::make_move_iterator(generated.begin()), std::make_move_iterator(generated.end()));
	}
	std::cout << "  pool: " << baselineCount << " baselines + " << (pool.size() - baselineCount) << " generated\n";

	std::unordered_map<std::string, BatchEval> archive;
	const auto evaluateCached = [&](const SchemaCandidate& candidate) -> const BatchEval&
	{
		auto found = archive.find(candidate._name);
		if (found == archive.end())
			found = archive.emplace(candidate._name, evaluateCandidate(candidate, pairs, dominantK)).first;
		return found->second;
	};

	const size_t rounds = std::max<size_t>(1, options._evolution._generations);
	const size_t elites = std::max<size_t>(1, options._evolution._eliteCount);
	const size_t repairPerCandidate = std::max<size_t>(1, options._evolution._repairPerCandidate);

	for (size_t round = 0; round < rounds; ++round)
	{
		for (const SchemaCandidate& candidate : pool)
			evaluateCached(candidate);

		std::vector<const BatchEval*> ranked;
		for (const auto& [name, eval] : archive)
			ranked.push_back(&eval);
		std::sort(ranked.begin(), ranked.end(), [](const BatchEval* a, const BatchEval* b) { return a->_score < b->_score; });

		std::cout << "  round " << round << ": best " << ranked.front()->_candidate._name
				  << " pass=" << std::fixed << std::setprecision(1) << ranked.front()->_score << " ms"
				  << " (build " << ranked.front()->_totalBuildMs << " + query " << ranked.front()->_totalQueryMs << ")\n";

		if (round + 1 == rounds)
			break;

		// Measured-feedback mutation: repair children of the current elites.
		for (size_t e = 0; e < elites && e < ranked.size(); ++e)
		{
			const BatchEval* parent = ranked[e];
			SchemaSearchRecord record;
			record._datasetName = "batch";
			record._workloadName = "manifest";
			record._buildMetrics = parent->_buildMetrics;
			record._queryMetrics = parent->_queryMetrics;
			record._score = parent->_score;
			std::vector<SchemaCandidate> repairs = generateSchemaRepairCandidates(
				parent->_candidate, { record }, generation, nullptr, repairPerCandidate,
				static_cast<uint32_t>(generation._seed + 101 * round + e), std::string());
			for (SchemaCandidate& repair : repairs)
			{
				if (archive.find(repair._name) == archive.end())
					pool.push_back(std::move(repair));
			}
		}
	}

	// Final ranking + report.
	std::vector<const BatchEval*> ranked;
	for (const auto& [name, eval] : archive)
		ranked.push_back(&eval);
	std::sort(ranked.begin(), ranked.end(), [](const BatchEval* a, const BatchEval* b) { return a->_score < b->_score; });

	double bestBaseline = std::numeric_limits<double>::infinity();
	std::string bestBaselineName;
	for (const BatchEval* eval : ranked)
	{
		if (eval->_baseline && eval->_score < bestBaseline)
		{
			bestBaseline = eval->_score;
			bestBaselineName = eval->_candidate._name;
		}
	}

	std::cout << "\n=== Top schemas by full-pass wall cost (build + query over all pairs; lower is better) ===\n";
	std::cout << std::left << std::setw(5) << "rank" << std::setw(42) << "schema" << std::right
			  << std::setw(12) << "pass ms" << std::setw(12) << "build ms" << std::setw(12) << "query ms" << "  tag\n";
	const size_t topK = 15;
	for (size_t i = 0; i < ranked.size() && i < topK; ++i)
	{
		const BatchEval* r = ranked[i];
		std::cout << std::left << std::setw(5) << (i + 1) << std::setw(42) << r->_candidate._name << std::right
				  << std::setw(12) << std::fixed << std::setprecision(1) << r->_score
				  << std::setw(12) << r->_totalBuildMs << std::setw(12) << r->_totalQueryMs
				  << "  " << (r->_baseline ? "baseline" : "searched") << "\n";
	}

	const BatchEval* best = ranked.empty() ? nullptr : ranked.front();
	std::cout << "\n=== Verdict ===\n";
	if (!bestBaselineName.empty())
		std::cout << "best baseline           : " << bestBaselineName << " @ " << std::setprecision(1) << bestBaseline << " ms/pass\n";
	if (best)
	{
		std::cout << "best schema overall     : " << best->_candidate._name << " @ " << best->_score << " ms/pass"
				  << (best->_baseline ? "  [baseline]" : "  [searched]") << "\n";
		if (!best->_baseline && bestBaseline > 0.0 && std::isfinite(bestBaseline))
			std::cout << ">> The searched schema cuts a full dataloader pass by "
					  << std::setprecision(1) << (100.0 * (bestBaseline - best->_score) / bestBaseline)
					  << "% vs the best single-primitive baseline.\n";
	}

	if (!options._csvPath.empty())
	{
		const bool exists = std::filesystem::exists(options._csvPath) && std::filesystem::file_size(options._csvPath) > 0;
		std::filesystem::create_directories(std::filesystem::path(options._csvPath).parent_path());
		std::ofstream stream(options._csvPath, std::ios::app);
		if (stream.is_open())
		{
			if (!exists)
				stream << "schema,tag,pass_ms,build_ms,query_ms,pairs,total_queries,manifest\n";
			for (const BatchEval* r : ranked)
				stream << '"' << r->_candidate._name << "\"," << (r->_baseline ? "baseline" : "searched") << ','
					   << std::fixed << std::setprecision(3) << r->_score << ',' << r->_totalBuildMs << ',' << r->_totalQueryMs << ','
					   << pairs.size() << ',' << totalQueries << ",\"" << options._batchManifestPath << "\"\n";
		}
	}

	if (options._pauseAtEnd)
	{
		std::cout << "\nPress Enter to exit...";
		std::cin.get();
	}
	return 0;
}

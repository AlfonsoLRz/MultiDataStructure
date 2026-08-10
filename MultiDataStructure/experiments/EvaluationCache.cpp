#include "../stdafx.h"
#include "EvaluationCache.h"

#include <boost/json.hpp>

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>

static constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
static constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

static uint64_t fnv1aBytes(const void* data, size_t length, uint64_t seed = kFnvOffset)
{
	const uint8_t* bytes = static_cast<const uint8_t*>(data);
	uint64_t hash = seed;
	for (size_t i = 0; i < length; ++i)
	{
		hash ^= static_cast<uint64_t>(bytes[i]);
		hash *= kFnvPrime;
	}
	return hash;
}

static uint64_t fnv1aString(const std::string& value, uint64_t seed = kFnvOffset)
{
	return fnv1aBytes(value.data(), value.size(), seed);
}

static std::string hex64(uint64_t value)
{
	std::ostringstream output;
	output << std::hex << std::setw(16) << std::setfill('0') << value;
	return output.str();
}

static std::string formatDouble(double value, int precision = 6)
{
	std::ostringstream output;
	output << std::setprecision(precision) << std::fixed << value;
	return output.str();
}

static const boost::json::value* find(const boost::json::object& object, const char* key)
{
	return object.if_contains(key);
}

static double asDouble(const boost::json::object& object, const char* key, double fallback = 0.0)
{
	const boost::json::value* value = find(object, key);
	if (!value)
		return fallback;
	if (value->is_double())
		return value->as_double();
	if (value->is_int64())
		return static_cast<double>(value->as_int64());
	if (value->is_uint64())
		return static_cast<double>(value->as_uint64());
	return fallback;
}

static size_t asSize(const boost::json::object& object, const char* key, size_t fallback = 0)
{
	const boost::json::value* value = find(object, key);
	if (!value)
		return fallback;
	if (value->is_int64())
		return static_cast<size_t>(value->as_int64());
	if (value->is_uint64())
		return static_cast<size_t>(value->as_uint64());
	if (value->is_double())
		return static_cast<size_t>(value->as_double());
	return fallback;
}

static std::string asString(const boost::json::object& object, const char* key, const std::string& fallback = {})
{
	const boost::json::value* value = find(object, key);
	if (!value || !value->is_string())
		return fallback;
	return std::string(value->as_string().c_str());
}

static boost::json::object encodeBuildMetrics(const Experiments::BuildMetrics& metrics)
{
	boost::json::object out;
	out["buildTimeMs"] = metrics._buildTimeMs;
	out["numNodes"] = metrics._numNodes;
	out["numLeaves"] = metrics._numLeaves;
	out["indexedPoints"] = metrics._indexedPoints;
	out["maxDepth"] = metrics._maxDepth;
	out["averageLeafOccupancy"] = metrics._averageLeafOccupancy;
	out["maxLeafOccupancy"] = metrics._maxLeafOccupancy;
	out["leafOccupancyP50"] = metrics._leafOccupancyP50;
	out["leafOccupancyP90"] = metrics._leafOccupancyP90;
	out["leafOccupancyP99"] = metrics._leafOccupancyP99;
	out["averageDepth"] = metrics._averageDepth;
	out["averageFanout"] = metrics._averageFanout;
	out["maxFanout"] = metrics._maxFanout;
	out["emptyChildRatio"] = metrics._emptyChildRatio;
	out["singleChildNodeCount"] = metrics._singleChildNodeCount;
	out["meanTightBoundsVolumeRatio"] = metrics._meanTightBoundsVolumeRatio;
	out["microIndexedLeaves"] = metrics._microIndexedLeaves;
	out["microIndexedPoints"] = metrics._microIndexedPoints;
	out["nodeFanoutSummary"] = metrics._nodeFanoutSummary;
	out["memoryEstimateBytes"] = metrics._memoryEstimateBytes;
	return out;
}

static void decodeBuildMetrics(const boost::json::object& source, Experiments::BuildMetrics& out)
{
	out._buildTimeMs = asDouble(source, "buildTimeMs");
	out._numNodes = asSize(source, "numNodes");
	out._numLeaves = asSize(source, "numLeaves");
	out._indexedPoints = asSize(source, "indexedPoints");
	out._maxDepth = asSize(source, "maxDepth");
	out._averageLeafOccupancy = asDouble(source, "averageLeafOccupancy");
	out._maxLeafOccupancy = asSize(source, "maxLeafOccupancy");
	out._leafOccupancyP50 = asDouble(source, "leafOccupancyP50");
	out._leafOccupancyP90 = asDouble(source, "leafOccupancyP90");
	out._leafOccupancyP99 = asDouble(source, "leafOccupancyP99");
	out._averageDepth = asDouble(source, "averageDepth");
	out._averageFanout = asDouble(source, "averageFanout");
	out._maxFanout = asSize(source, "maxFanout");
	out._emptyChildRatio = asDouble(source, "emptyChildRatio");
	out._singleChildNodeCount = asSize(source, "singleChildNodeCount");
	out._meanTightBoundsVolumeRatio = asDouble(source, "meanTightBoundsVolumeRatio");
	out._microIndexedLeaves = asSize(source, "microIndexedLeaves");
	out._microIndexedPoints = asSize(source, "microIndexedPoints");
	out._nodeFanoutSummary = asString(source, "nodeFanoutSummary");
	out._memoryEstimateBytes = asSize(source, "memoryEstimateBytes");
}

static boost::json::object encodeQueryMetrics(const Experiments::QueryMetrics& metrics)
{
	boost::json::object out;
	out["totalQueries"] = metrics._totalQueries;
	out["totalLatencyMs"] = metrics._totalLatencyMs;
	out["averageLatencyMs"] = metrics._averageLatencyMs;
	out["medianLatencyMs"] = metrics._medianLatencyMs;
	out["p95LatencyMs"] = metrics._p95LatencyMs;
	out["throughputQueriesPerSecond"] = metrics._throughputQueriesPerSecond;
	out["averageVisitedNodes"] = metrics._averageVisitedNodes;
	out["averageTestedPoints"] = metrics._averageTestedPoints;
	out["averageReturnedPoints"] = metrics._averageReturnedPoints;
	out["averageFullyContainedNodes"] = metrics._averageFullyContainedNodes;
	out["totalVisitedNodes"] = metrics._totalVisitedNodes;
	out["totalTestedPoints"] = metrics._totalTestedPoints;
	out["totalReturnedPoints"] = metrics._totalReturnedPoints;
	out["totalFullyContainedNodes"] = metrics._totalFullyContainedNodes;
	return out;
}

static void decodeQueryMetrics(const boost::json::object& source, Experiments::QueryMetrics& out)
{
	out._totalQueries = asSize(source, "totalQueries");
	out._totalLatencyMs = asDouble(source, "totalLatencyMs");
	out._averageLatencyMs = asDouble(source, "averageLatencyMs");
	out._medianLatencyMs = asDouble(source, "medianLatencyMs");
	out._p95LatencyMs = asDouble(source, "p95LatencyMs");
	out._throughputQueriesPerSecond = asDouble(source, "throughputQueriesPerSecond");
	out._averageVisitedNodes = asDouble(source, "averageVisitedNodes");
	out._averageTestedPoints = asDouble(source, "averageTestedPoints");
	out._averageReturnedPoints = asDouble(source, "averageReturnedPoints");
	out._averageFullyContainedNodes = asDouble(source, "averageFullyContainedNodes");
	out._totalVisitedNodes = asSize(source, "totalVisitedNodes");
	out._totalTestedPoints = asSize(source, "totalTestedPoints");
	out._totalReturnedPoints = asSize(source, "totalReturnedPoints");
	out._totalFullyContainedNodes = asSize(source, "totalFullyContainedNodes");
}

static boost::json::object encodeRecord(const std::string& canonicalKey, const Experiments::SchemaSearchRecord& record)
{
	boost::json::object out;
	out["key"] = canonicalKey;
	out["build"] = encodeBuildMetrics(record._buildMetrics);
	out["query"] = encodeQueryMetrics(record._queryMetrics);
	out["rangeQuery"] = encodeQueryMetrics(record._rangeMetrics);
	out["countRangeQuery"] = encodeQueryMetrics(record._countRangeMetrics);
	out["radiusQuery"] = encodeQueryMetrics(record._radiusMetrics);
	out["knnQuery"] = encodeQueryMetrics(record._knnMetrics);
	out["rangeQueries"] = record._rangeQueries;
	out["countRangeQueries"] = record._countRangeQueries;
	out["radiusQueries"] = record._radiusQueries;
	out["knnQueries"] = record._knnQueries;
	out["queryStrataSummary"] = record._queryStrataSummary;
	out["score"] = record._score;
	out["scoreMemoryMb"] = record._scoreMemoryMb;
	out["scoreImbalancePenalty"] = record._scoreImbalancePenalty;
	out["scoreObjective"] = record._scoreObjective;
	out["scoreMode"] = record._scoreMode;
	out["scoreStage"] = record._scoreStage;
	out["scoreIsFinalLatency"] = record._scoreIsFinalLatency;
	out["backend"] = record._backend;
	out["gpuSupportStatus"] = record._gpuSupportStatus;
	out["knnBackend"] = record._knnBackend;
	out["cudaDevice"] = record._cudaDevice;
	out["cudaBuilder"] = record._cudaBuilder;
	out["gpuUploadMs"] = record._gpuUploadMs;
	out["gpuBuildMs"] = record._gpuBuildMs;
	out["gpuQueryMs"] = record._gpuQueryMs;
	out["gpuMemoryBytes"] = record._gpuMemoryBytes;
	out["conditionalLevels"] = record._conditionalLevels;
	out["conditionFields"] = record._conditionFields;
	out["conditionSummary"] = record._conditionSummary;
	out["isBaseline"] = record._isBaseline;
	out["activeStructureTypes"] = record._activeStructureTypes;
	out["nestedActiveFraction"] = record._nestedActiveFraction;
	out["activeStructureSummary"] = record._activeStructureSummary;
	return out;
}

static void decodeRecord(const boost::json::object& source, Experiments::SchemaSearchRecord& out)
{
	const boost::json::value* build = find(source, "build");
	if (build && build->is_object())
		decodeBuildMetrics(build->as_object(), out._buildMetrics);

	const boost::json::value* query = find(source, "query");
	if (query && query->is_object())
		decodeQueryMetrics(query->as_object(), out._queryMetrics);

	const boost::json::value* rangeQuery = find(source, "rangeQuery");
	if (rangeQuery && rangeQuery->is_object())
		decodeQueryMetrics(rangeQuery->as_object(), out._rangeMetrics);
	const boost::json::value* countRangeQuery = find(source, "countRangeQuery");
	if (countRangeQuery && countRangeQuery->is_object())
		decodeQueryMetrics(countRangeQuery->as_object(), out._countRangeMetrics);
	const boost::json::value* radiusQuery = find(source, "radiusQuery");
	if (radiusQuery && radiusQuery->is_object())
		decodeQueryMetrics(radiusQuery->as_object(), out._radiusMetrics);
	const boost::json::value* knnQuery = find(source, "knnQuery");
	if (knnQuery && knnQuery->is_object())
		decodeQueryMetrics(knnQuery->as_object(), out._knnMetrics);

	out._rangeQueries = asSize(source, "rangeQueries");
	out._countRangeQueries = asSize(source, "countRangeQueries");
	out._radiusQueries = asSize(source, "radiusQueries");
	out._knnQueries = asSize(source, "knnQueries");
	out._queryStrataSummary = asString(source, "queryStrataSummary");
	out._score = asDouble(source, "score");
	out._scoreMemoryMb = asDouble(source, "scoreMemoryMb");
	out._scoreImbalancePenalty = asDouble(source, "scoreImbalancePenalty");
	out._scoreObjective = asString(source, "scoreObjective", out._scoreObjective);
	out._scoreMode = asString(source, "scoreMode", out._scoreMode);
	out._scoreStage = asString(source, "scoreStage", out._scoreStage);
	const boost::json::value* finalLatency = find(source, "scoreIsFinalLatency");
	if (finalLatency && finalLatency->is_bool())
		out._scoreIsFinalLatency = finalLatency->as_bool();
	out._backend = asString(source, "backend", "cpu");
	out._gpuSupportStatus = asString(source, "gpuSupportStatus", out._gpuSupportStatus);
	out._knnBackend = asString(source, "knnBackend", out._knnBackend);
	const boost::json::value* device = find(source, "cudaDevice");
	if (device && device->is_int64())
		out._cudaDevice = static_cast<int>(device->as_int64());
	out._cudaBuilder = asString(source, "cudaBuilder");
	out._gpuUploadMs = asDouble(source, "gpuUploadMs");
	out._gpuBuildMs = asDouble(source, "gpuBuildMs");
	out._gpuQueryMs = asDouble(source, "gpuQueryMs");
	out._gpuMemoryBytes = asSize(source, "gpuMemoryBytes");
	out._conditionalLevels = asSize(source, "conditionalLevels");
	out._conditionFields = asSize(source, "conditionFields");
	out._conditionSummary = asString(source, "conditionSummary");
	const boost::json::value* baseline = find(source, "isBaseline");
	if (baseline && baseline->is_bool())
		out._isBaseline = baseline->as_bool();
	out._activeStructureTypes = asSize(source, "activeStructureTypes");
	out._nestedActiveFraction = asDouble(source, "nestedActiveFraction");
	out._activeStructureSummary = asString(source, "activeStructureSummary");
}

namespace Experiments
{
	std::string EvaluationCacheKey::canonical() const
	{
		return _schemaSignature + "|" + _datasetFingerprint + "|" + _workloadFingerprint + "|" + _evaluatorFingerprint;
	}

	EvaluationCache::EvaluationCache() = default;

	EvaluationCache::~EvaluationCache()
	{
		close();
	}

	bool EvaluationCache::open(const std::string& filePath, bool readOnly)
	{
		std::lock_guard<std::mutex> lock(_mutex);
		close();
		_filePath = filePath;
		_readOnly = readOnly;
		if (_filePath.empty())
			return false;

		std::error_code error;
		const std::filesystem::path path(_filePath);
		if (path.has_parent_path())
			std::filesystem::create_directories(path.parent_path(), error);

		loadFromDisk();

		if (!_readOnly)
		{
			_appendStream.open(_filePath, std::ios::out | std::ios::app);
			if (!_appendStream.is_open())
			{
				_readOnly = true;
				return false;
			}
		}
		return true;
	}

	void EvaluationCache::close()
	{
		if (_appendStream.is_open())
			_appendStream.close();
		_filePath.clear();
		_entries.clear();
		_hits.store(0);
		_misses.store(0);
		_readOnly = false;
	}

	bool EvaluationCache::loadFromDisk()
	{
		_entries.clear();
		std::ifstream input(_filePath);
		if (!input.is_open())
			return false;

		std::string line;
		while (std::getline(input, line))
		{
			if (line.empty())
				continue;
			boost::system::error_code parseError;
			boost::json::value parsed = boost::json::parse(line, parseError);
			if (parseError || !parsed.is_object())
				continue;

			const boost::json::object& object = parsed.as_object();
			const std::string keyText = asString(object, "key");
			if (keyText.empty())
				continue;

			Entry entry;
			decodeRecord(object, entry._record);
			_entries[keyText] = std::move(entry);
		}
		return true;
	}

	bool EvaluationCache::tryGet(const EvaluationCacheKey& key, SchemaSearchRecord& outRecord) const
	{
		if (_filePath.empty())
			return false;

		std::lock_guard<std::mutex> lock(_mutex);
		const std::string canonicalKey = key.canonical();
		const auto it = _entries.find(canonicalKey);
		if (it == _entries.end())
		{
			_misses.fetch_add(1);
			return false;
		}

		const SchemaSearchRecord& cached = it->second._record;
		// Preserve caller-provided context (names, weights, counts) filled at the call site after lookup.
		const std::string datasetName = std::move(outRecord._datasetName);
		const std::string datasetSource = std::move(outRecord._datasetSource);
		const size_t numPoints = outRecord._numPoints;
		const std::string workloadName = std::move(outRecord._workloadName);
		const double rangeWeight = outRecord._rangeWeight;
		const double radiusWeight = outRecord._radiusWeight;
		const double knnWeight = outRecord._knnWeight;
		const size_t numQueries = outRecord._numQueries;
		const size_t knnK = outRecord._knnK;
		const uint32_t querySeed = outRecord._querySeed;
		const std::string schemaName = std::move(outRecord._schemaName);
		const std::string schemaPath = std::move(outRecord._schemaPath);
		const bool isBaseline = outRecord._isBaseline;
		const ScoreWeights weights = outRecord._weights;
		const std::string scoreObjective = outRecord._scoreObjective;
		const std::string scoreMode = outRecord._scoreMode;
		const std::string scoreStage = outRecord._scoreStage;
		const bool scoreIsFinalLatency = outRecord._scoreIsFinalLatency;
		const PointCloudFeatures pointFeatures = outRecord._pointFeatures;
		const WorkloadFeatures workloadFeatures = outRecord._workloadFeatures;
		const std::string gpuSupportStatus = outRecord._gpuSupportStatus;

		outRecord = cached;
		outRecord._datasetName = datasetName;
		outRecord._datasetSource = datasetSource;
		outRecord._numPoints = numPoints;
		outRecord._workloadName = workloadName;
		outRecord._rangeWeight = rangeWeight;
		outRecord._radiusWeight = radiusWeight;
		outRecord._knnWeight = knnWeight;
		outRecord._numQueries = numQueries;
		outRecord._knnK = knnK;
		outRecord._querySeed = querySeed;
		outRecord._schemaName = schemaName;
		outRecord._schemaPath = schemaPath;
		outRecord._isBaseline = isBaseline;
		outRecord._weights = weights;
		outRecord._scoreObjective = scoreObjective;
		outRecord._scoreMode = scoreMode;
		outRecord._scoreStage = scoreStage;
		outRecord._scoreIsFinalLatency = scoreIsFinalLatency;
		outRecord._pointFeatures = pointFeatures;
		outRecord._workloadFeatures = workloadFeatures;
		outRecord._gpuSupportStatus = gpuSupportStatus;

		_hits.fetch_add(1);
		return true;
	}

	void EvaluationCache::put(const EvaluationCacheKey& key, const SchemaSearchRecord& record)
	{
		if (_filePath.empty() || _readOnly)
			return;

		std::lock_guard<std::mutex> lock(_mutex);
		const std::string canonicalKey = key.canonical();
		Entry entry;
		entry._record = record;
		_entries[canonicalKey] = std::move(entry);
		appendLine(canonicalKey, record);
	}

	void EvaluationCache::appendLine(const std::string& canonicalKey, const SchemaSearchRecord& record)
	{
		if (!_appendStream.is_open())
			return;
		_appendStream << boost::json::serialize(encodeRecord(canonicalKey, record)) << '\n';
		_appendStream.flush();
	}

	size_t EvaluationCache::entryCount() const
	{
		std::lock_guard<std::mutex> lock(_mutex);
		return _entries.size();
	}

	EvaluationCacheKey makeEvaluationCacheKey(
		const std::string& schemaSignature,
		const std::string& datasetName,
		size_t numPoints,
		const glm::vec3& bboxMin,
		const glm::vec3& bboxMax,
		const WorkloadProfile& workload,
		const std::string& evaluatorBackend,
		const std::string& cudaBuilder,
		const ScoreWeights& weights)
	{
		EvaluationCacheKey key;
		key._schemaSignature = schemaSignature;

		std::ostringstream datasetText;
		datasetText << datasetName << "|" << numPoints
			<< "|" << formatDouble(bboxMin.x) << "," << formatDouble(bboxMin.y) << "," << formatDouble(bboxMin.z)
			<< "|" << formatDouble(bboxMax.x) << "," << formatDouble(bboxMax.y) << "," << formatDouble(bboxMax.z);
		key._datasetFingerprint = hex64(fnv1aString(datasetText.str()));

		std::ostringstream workloadText;
		workloadText << workload._name
			<< "|r=" << formatDouble(workload._rangeWeight, 4)
			<< ",d=" << formatDouble(workload._radiusWeight, 4)
			<< ",k=" << formatDouble(workload._knnWeight, 4)
			<< "|rs=" << formatDouble(workload._rangeScaleMin, 4) << "-" << formatDouble(workload._rangeScaleMax, 4)
			<< "|ds=" << formatDouble(workload._radiusScaleMin, 4) << "-" << formatDouble(workload._radiusScaleMax, 4)
			<< "|n=" << workload._numQueries
			<< "|kk=" << workload._knnK
			<< "|seed=" << workload._querySeed
			<< "|strata=" << (workload._stratifyQueries ? "1" : "0");
		// A replayed trace IS the workload, so two runs over different traces must not share a
		// key. Without this the fingerprint collapsed to the workload JSON's name and weights,
		// and replaying trace B after trace A silently returned trace A's cached scores for
		// every candidate. Size and write time are included so editing a trace in place (e.g.
		// re-running split_trace.py) also invalidates.
		if (!workload._tracePath.empty())
		{
			workloadText << "|trace=" << workload._tracePath;
			std::error_code ec;
			const std::filesystem::path tracePath(workload._tracePath);
			const auto size = std::filesystem::file_size(tracePath, ec);
			if (!ec)
				workloadText << "@" << size;
			const auto written = std::filesystem::last_write_time(tracePath, ec);
			if (!ec)
				workloadText << "#" << written.time_since_epoch().count();
		}
		key._workloadFingerprint = hex64(fnv1aString(workloadText.str()));

		std::ostringstream evaluatorText;
		evaluatorText << evaluatorBackend;
		if (evaluatorBackend == "cuda" && !cudaBuilder.empty())
			evaluatorText << ":" << cudaBuilder;
		evaluatorText << "|ll=" << formatDouble(weights._lambdaLatency, 6)
			<< "|lb=" << formatDouble(weights._lambdaBuild, 6)
			<< "|lm=" << formatDouble(weights._lambdaMemory, 6)
			<< "|li=" << formatDouble(weights._lambdaImbalance, 6);
		// Visit-proxy and latency score the same counters differently; this bit stops the proxy rung clobbering the latency rung's cached score.
		evaluatorText << "|vp=" << (weights._useVisitProxy ? "1" : "0");
		if (weights._useVisitProxy)
			evaluatorText << "@" << formatDouble(weights._visitProxyAlpha, 6);
		key._evaluatorFingerprint = hex64(fnv1aString(evaluatorText.str()));

		return key;
	}
}

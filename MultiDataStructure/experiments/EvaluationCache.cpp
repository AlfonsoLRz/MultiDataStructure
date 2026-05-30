#include "../stdafx.h"
#include "EvaluationCache.h"

#include <boost/json.hpp>

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace
{
	constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
	constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

	uint64_t fnv1aBytes(const void* data, size_t length, uint64_t seed = kFnvOffset)
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

	uint64_t fnv1aString(const std::string& value, uint64_t seed = kFnvOffset)
	{
		return fnv1aBytes(value.data(), value.size(), seed);
	}

	std::string hex64(uint64_t value)
	{
		std::ostringstream output;
		output << std::hex << std::setw(16) << std::setfill('0') << value;
		return output.str();
	}

	std::string formatDouble(double value, int precision = 6)
	{
		std::ostringstream output;
		output << std::setprecision(precision) << std::fixed << value;
		return output.str();
	}

	const boost::json::value* find(const boost::json::object& object, const char* key)
	{
		return object.if_contains(key);
	}

	double asDouble(const boost::json::object& object, const char* key, double fallback = 0.0)
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

	size_t asSize(const boost::json::object& object, const char* key, size_t fallback = 0)
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

	std::string asString(const boost::json::object& object, const char* key, const std::string& fallback = {})
	{
		const boost::json::value* value = find(object, key);
		if (!value || !value->is_string())
			return fallback;
		return std::string(value->as_string().c_str());
	}

	boost::json::object encodeBuildMetrics(const Experiments::BuildMetrics& metrics)
	{
		boost::json::object out;
		out["buildTimeMs"] = metrics.buildTimeMs;
		out["numNodes"] = metrics.numNodes;
		out["numLeaves"] = metrics.numLeaves;
		out["indexedPoints"] = metrics.indexedPoints;
		out["maxDepth"] = metrics.maxDepth;
		out["averageLeafOccupancy"] = metrics.averageLeafOccupancy;
		out["maxLeafOccupancy"] = metrics.maxLeafOccupancy;
		out["memoryEstimateBytes"] = metrics.memoryEstimateBytes;
		return out;
	}

	void decodeBuildMetrics(const boost::json::object& source, Experiments::BuildMetrics& out)
	{
		out.buildTimeMs = asDouble(source, "buildTimeMs");
		out.numNodes = asSize(source, "numNodes");
		out.numLeaves = asSize(source, "numLeaves");
		out.indexedPoints = asSize(source, "indexedPoints");
		out.maxDepth = asSize(source, "maxDepth");
		out.averageLeafOccupancy = asDouble(source, "averageLeafOccupancy");
		out.maxLeafOccupancy = asSize(source, "maxLeafOccupancy");
		out.memoryEstimateBytes = asSize(source, "memoryEstimateBytes");
	}

	boost::json::object encodeQueryMetrics(const Experiments::QueryMetrics& metrics)
	{
		boost::json::object out;
		out["totalQueries"] = metrics.totalQueries;
		out["totalLatencyMs"] = metrics.totalLatencyMs;
		out["averageLatencyMs"] = metrics.averageLatencyMs;
		out["medianLatencyMs"] = metrics.medianLatencyMs;
		out["p95LatencyMs"] = metrics.p95LatencyMs;
		out["throughputQueriesPerSecond"] = metrics.throughputQueriesPerSecond;
		out["averageVisitedNodes"] = metrics.averageVisitedNodes;
		out["averageTestedPoints"] = metrics.averageTestedPoints;
		out["averageReturnedPoints"] = metrics.averageReturnedPoints;
		out["totalVisitedNodes"] = metrics.totalVisitedNodes;
		out["totalTestedPoints"] = metrics.totalTestedPoints;
		out["totalReturnedPoints"] = metrics.totalReturnedPoints;
		return out;
	}

	void decodeQueryMetrics(const boost::json::object& source, Experiments::QueryMetrics& out)
	{
		out.totalQueries = asSize(source, "totalQueries");
		out.totalLatencyMs = asDouble(source, "totalLatencyMs");
		out.averageLatencyMs = asDouble(source, "averageLatencyMs");
		out.medianLatencyMs = asDouble(source, "medianLatencyMs");
		out.p95LatencyMs = asDouble(source, "p95LatencyMs");
		out.throughputQueriesPerSecond = asDouble(source, "throughputQueriesPerSecond");
		out.averageVisitedNodes = asDouble(source, "averageVisitedNodes");
		out.averageTestedPoints = asDouble(source, "averageTestedPoints");
		out.averageReturnedPoints = asDouble(source, "averageReturnedPoints");
		out.totalVisitedNodes = asSize(source, "totalVisitedNodes");
		out.totalTestedPoints = asSize(source, "totalTestedPoints");
		out.totalReturnedPoints = asSize(source, "totalReturnedPoints");
	}

	boost::json::object encodeRecord(const std::string& canonicalKey, const Experiments::SchemaSearchRecord& record)
	{
		boost::json::object out;
		out["key"] = canonicalKey;
		out["build"] = encodeBuildMetrics(record.buildMetrics);
		out["query"] = encodeQueryMetrics(record.queryMetrics);
		out["rangeQueries"] = record.rangeQueries;
		out["countRangeQueries"] = record.countRangeQueries;
		out["radiusQueries"] = record.radiusQueries;
		out["knnQueries"] = record.knnQueries;
		out["score"] = record.score;
		out["scoreMemoryMb"] = record.scoreMemoryMb;
		out["scoreImbalancePenalty"] = record.scoreImbalancePenalty;
		out["scoreMode"] = record.scoreMode;
		out["scoreStage"] = record.scoreStage;
		out["scoreIsFinalLatency"] = record.scoreIsFinalLatency;
		out["backend"] = record.backend;
		out["cudaDevice"] = record.cudaDevice;
		out["cudaBuilder"] = record.cudaBuilder;
		out["gpuUploadMs"] = record.gpuUploadMs;
		out["gpuBuildMs"] = record.gpuBuildMs;
		out["gpuQueryMs"] = record.gpuQueryMs;
		out["gpuMemoryBytes"] = record.gpuMemoryBytes;
		out["conditionalLevels"] = record.conditionalLevels;
		out["conditionFields"] = record.conditionFields;
		out["conditionSummary"] = record.conditionSummary;
		out["isBaseline"] = record.isBaseline;
		out["activeStructureTypes"] = record.activeStructureTypes;
		out["nestedActiveFraction"] = record.nestedActiveFraction;
		out["activeStructureSummary"] = record.activeStructureSummary;
		return out;
	}

	void decodeRecord(const boost::json::object& source, Experiments::SchemaSearchRecord& out)
	{
		const boost::json::value* build = find(source, "build");
		if (build && build->is_object())
			decodeBuildMetrics(build->as_object(), out.buildMetrics);

		const boost::json::value* query = find(source, "query");
		if (query && query->is_object())
			decodeQueryMetrics(query->as_object(), out.queryMetrics);

		out.rangeQueries = asSize(source, "rangeQueries");
		out.countRangeQueries = asSize(source, "countRangeQueries");
		out.radiusQueries = asSize(source, "radiusQueries");
		out.knnQueries = asSize(source, "knnQueries");
		out.score = asDouble(source, "score");
		out.scoreMemoryMb = asDouble(source, "scoreMemoryMb");
		out.scoreImbalancePenalty = asDouble(source, "scoreImbalancePenalty");
		out.scoreMode = asString(source, "scoreMode", out.scoreMode);
		out.scoreStage = asString(source, "scoreStage", out.scoreStage);
		const boost::json::value* finalLatency = find(source, "scoreIsFinalLatency");
		if (finalLatency && finalLatency->is_bool())
			out.scoreIsFinalLatency = finalLatency->as_bool();
		out.backend = asString(source, "backend", "cpu");
		const boost::json::value* device = find(source, "cudaDevice");
		if (device && device->is_int64())
			out.cudaDevice = static_cast<int>(device->as_int64());
		out.cudaBuilder = asString(source, "cudaBuilder");
		out.gpuUploadMs = asDouble(source, "gpuUploadMs");
		out.gpuBuildMs = asDouble(source, "gpuBuildMs");
		out.gpuQueryMs = asDouble(source, "gpuQueryMs");
		out.gpuMemoryBytes = asSize(source, "gpuMemoryBytes");
		out.conditionalLevels = asSize(source, "conditionalLevels");
		out.conditionFields = asSize(source, "conditionFields");
		out.conditionSummary = asString(source, "conditionSummary");
		const boost::json::value* baseline = find(source, "isBaseline");
		if (baseline && baseline->is_bool())
			out.isBaseline = baseline->as_bool();
		out.activeStructureTypes = asSize(source, "activeStructureTypes");
		out.nestedActiveFraction = asDouble(source, "nestedActiveFraction");
		out.activeStructureSummary = asString(source, "activeStructureSummary");
	}
}

namespace Experiments
{
	std::string EvaluationCacheKey::canonical() const
	{
		return schemaSignature + "|" + datasetFingerprint + "|" + workloadFingerprint + "|" + evaluatorFingerprint;
	}

	EvaluationCache::EvaluationCache() = default;

	EvaluationCache::~EvaluationCache()
	{
		close();
	}

	bool EvaluationCache::open(const std::string& filePath, bool readOnly)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		close();
		filePath_ = filePath;
		readOnly_ = readOnly;
		if (filePath_.empty())
			return false;

		std::error_code error;
		const std::filesystem::path path(filePath_);
		if (path.has_parent_path())
			std::filesystem::create_directories(path.parent_path(), error);

		loadFromDisk();

		if (!readOnly_)
		{
			appendStream_.open(filePath_, std::ios::out | std::ios::app);
			if (!appendStream_.is_open())
			{
				readOnly_ = true;
				return false;
			}
		}
		return true;
	}

	void EvaluationCache::close()
	{
		if (appendStream_.is_open())
			appendStream_.close();
		filePath_.clear();
		entries_.clear();
		hits_.store(0);
		misses_.store(0);
		readOnly_ = false;
	}

	bool EvaluationCache::loadFromDisk()
	{
		entries_.clear();
		std::ifstream input(filePath_);
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
			decodeRecord(object, entry.record);
			entries_[keyText] = std::move(entry);
		}
		return true;
	}

	bool EvaluationCache::tryGet(const EvaluationCacheKey& key, SchemaSearchRecord& outRecord) const
	{
		if (filePath_.empty())
			return false;

		std::lock_guard<std::mutex> lock(mutex_);
		const std::string canonicalKey = key.canonical();
		const auto it = entries_.find(canonicalKey);
		if (it == entries_.end())
		{
			misses_.fetch_add(1);
			return false;
		}

		const SchemaSearchRecord& cached = it->second.record;
		// Preserve caller-provided context fields (dataset / schema names, weights, point count,
		// workload weights) which are filled at the call site after lookup.
		const std::string datasetName = std::move(outRecord.datasetName);
		const std::string datasetSource = std::move(outRecord.datasetSource);
		const size_t numPoints = outRecord.numPoints;
		const std::string workloadName = std::move(outRecord.workloadName);
		const double rangeWeight = outRecord.rangeWeight;
		const double radiusWeight = outRecord.radiusWeight;
		const double knnWeight = outRecord.knnWeight;
		const size_t numQueries = outRecord.numQueries;
		const size_t knnK = outRecord.knnK;
		const uint32_t querySeed = outRecord.querySeed;
		const std::string schemaName = std::move(outRecord.schemaName);
		const std::string schemaPath = std::move(outRecord.schemaPath);
		const bool isBaseline = outRecord.isBaseline;
		const ScoreWeights weights = outRecord.weights;
		const std::string scoreMode = outRecord.scoreMode;
		const std::string scoreStage = outRecord.scoreStage;
		const bool scoreIsFinalLatency = outRecord.scoreIsFinalLatency;
		const PointCloudFeatures pointFeatures = outRecord.pointFeatures;
		const WorkloadFeatures workloadFeatures = outRecord.workloadFeatures;

		outRecord = cached;
		outRecord.datasetName = datasetName;
		outRecord.datasetSource = datasetSource;
		outRecord.numPoints = numPoints;
		outRecord.workloadName = workloadName;
		outRecord.rangeWeight = rangeWeight;
		outRecord.radiusWeight = radiusWeight;
		outRecord.knnWeight = knnWeight;
		outRecord.numQueries = numQueries;
		outRecord.knnK = knnK;
		outRecord.querySeed = querySeed;
		outRecord.schemaName = schemaName;
		outRecord.schemaPath = schemaPath;
		outRecord.isBaseline = isBaseline;
		outRecord.weights = weights;
		outRecord.scoreMode = scoreMode;
		outRecord.scoreStage = scoreStage;
		outRecord.scoreIsFinalLatency = scoreIsFinalLatency;
		outRecord.pointFeatures = pointFeatures;
		outRecord.workloadFeatures = workloadFeatures;

		hits_.fetch_add(1);
		return true;
	}

	void EvaluationCache::put(const EvaluationCacheKey& key, const SchemaSearchRecord& record)
	{
		if (filePath_.empty() || readOnly_)
			return;

		std::lock_guard<std::mutex> lock(mutex_);
		const std::string canonicalKey = key.canonical();
		Entry entry;
		entry.record = record;
		entries_[canonicalKey] = std::move(entry);
		appendLine(canonicalKey, record);
	}

	void EvaluationCache::appendLine(const std::string& canonicalKey, const SchemaSearchRecord& record)
	{
		if (!appendStream_.is_open())
			return;
		appendStream_ << boost::json::serialize(encodeRecord(canonicalKey, record)) << '\n';
		appendStream_.flush();
	}

	size_t EvaluationCache::entryCount() const
	{
		std::lock_guard<std::mutex> lock(mutex_);
		return entries_.size();
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
		key.schemaSignature = schemaSignature;

		std::ostringstream datasetText;
		datasetText << datasetName << "|" << numPoints
			<< "|" << formatDouble(bboxMin.x) << "," << formatDouble(bboxMin.y) << "," << formatDouble(bboxMin.z)
			<< "|" << formatDouble(bboxMax.x) << "," << formatDouble(bboxMax.y) << "," << formatDouble(bboxMax.z);
		key.datasetFingerprint = hex64(fnv1aString(datasetText.str()));

		std::ostringstream workloadText;
		workloadText << workload.name
			<< "|r=" << formatDouble(workload.rangeWeight, 4)
			<< ",d=" << formatDouble(workload.radiusWeight, 4)
			<< ",k=" << formatDouble(workload.knnWeight, 4)
			<< "|rs=" << formatDouble(workload.rangeScaleMin, 4) << "-" << formatDouble(workload.rangeScaleMax, 4)
			<< "|ds=" << formatDouble(workload.radiusScaleMin, 4) << "-" << formatDouble(workload.radiusScaleMax, 4)
			<< "|n=" << workload.numQueries
			<< "|kk=" << workload.knnK
			<< "|seed=" << workload.querySeed;
		key.workloadFingerprint = hex64(fnv1aString(workloadText.str()));

		std::ostringstream evaluatorText;
		evaluatorText << evaluatorBackend;
		if (evaluatorBackend == "cuda" && !cudaBuilder.empty())
			evaluatorText << ":" << cudaBuilder;
		evaluatorText << "|lb=" << formatDouble(weights.lambdaBuild, 6)
			<< "|lm=" << formatDouble(weights.lambdaMemory, 6)
			<< "|li=" << formatDouble(weights.lambdaImbalance, 6);
		// Visit-proxy and latency are different scoring functions over the same kernel counters,
		// but the cached record's `score` field reflects the choice; without this bit the cheap
		// proxy rung would clobber the latency rung's cached score for the same (schema, dataset,
		// workload) tuple.
		evaluatorText << "|vp=" << (weights.useVisitProxy ? "1" : "0");
		if (weights.useVisitProxy)
			evaluatorText << "@" << formatDouble(weights.visitProxyAlpha, 6);
		key.evaluatorFingerprint = hex64(fnv1aString(evaluatorText.str()));

		return key;
	}
}

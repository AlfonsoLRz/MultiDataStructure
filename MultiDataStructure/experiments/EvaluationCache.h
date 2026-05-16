#pragma once

#include "../stdafx.h"
#include "SchemaSearch.h"

namespace Experiments
{
	struct EvaluationCacheKey
	{
		std::string schemaSignature;
		std::string datasetFingerprint;
		std::string workloadFingerprint;
		std::string evaluatorFingerprint;

		std::string canonical() const;
	};

	class EvaluationCache
	{
	public:
		EvaluationCache();
		EvaluationCache(const EvaluationCache&) = delete;
		EvaluationCache& operator=(const EvaluationCache&) = delete;
		~EvaluationCache();

		// Opens (creating if missing) a JSONL-backed cache at the given path. When readOnly is
		// true, lookups still work but writes are silently dropped. Returns false if the file
		// could not be opened for writing (read-only stays usable).
		bool open(const std::string& filePath, bool readOnly = false);
		void close();

		bool enabled() const { return !filePath_.empty(); }
		bool readOnly() const { return readOnly_; }
		const std::string& filePath() const { return filePath_; }

		// On hit, copies the cached record into outRecord and returns true. The caller is
		// responsible for filling fields not persisted by the cache (datasetName / datasetSource /
		// schemaName / schemaPath / weights / numPoints / workload weights), which depend on the
		// surrounding run context.
		bool tryGet(const EvaluationCacheKey& key, SchemaSearchRecord& outRecord) const;

		// Appends a record under the given key. No-op when the cache is not open or is read-only.
		void put(const EvaluationCacheKey& key, const SchemaSearchRecord& record);

		size_t hitCount() const { return hits_; }
		size_t missCount() const { return misses_; }
		size_t entryCount() const;

	private:
		struct Entry
		{
			SchemaSearchRecord record;
		};

		bool loadFromDisk();
		void appendLine(const std::string& canonicalKey, const SchemaSearchRecord& record);

		std::string filePath_;
		bool readOnly_ = false;
		mutable std::mutex mutex_;
		std::unordered_map<std::string, Entry> entries_;
		std::ofstream appendStream_;
		mutable std::atomic<size_t> hits_{0};
		mutable std::atomic<size_t> misses_{0};
	};

	// Builds a cache key from the live evaluation context. The dataset fingerprint summarises the
	// cloud identity (name + point count + bounds) so that the same cloud opened from a different
	// path or rebuilt from the same synthetic seed produces the same key. The schema signature is
	// supplied by the caller (computed via the existing schemaSignature() helper).
	EvaluationCacheKey makeEvaluationCacheKey(
		const std::string& schemaSignature,
		const std::string& datasetName,
		size_t numPoints,
		const glm::vec3& bboxMin,
		const glm::vec3& bboxMax,
		const WorkloadProfile& workload,
		const std::string& evaluatorBackend,
		const std::string& cudaBuilder,
		const ScoreWeights& weights);
}

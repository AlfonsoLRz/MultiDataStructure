#pragma once

#include "../stdafx.h"
#include "SchemaSearch.h"

namespace Experiments
{
	struct EvaluationCacheKey
	{
		std::string _schemaSignature;
		std::string _datasetFingerprint;
		std::string _workloadFingerprint;
		std::string _evaluatorFingerprint;

		std::string canonical() const;
	};

	class EvaluationCache
	{
	public:
		EvaluationCache();
		EvaluationCache(const EvaluationCache&) = delete;
		EvaluationCache& operator=(const EvaluationCache&) = delete;
		~EvaluationCache();

		// Opens (creating if missing) a JSONL-backed cache; readOnly keeps lookups working but drops writes. Returns false if the file could not be opened for writing.
		bool open(const std::string& filePath, bool readOnly = false);
		void close();

		bool enabled() const { return !_filePath.empty(); }
		bool readOnly() const { return _readOnly; }
		const std::string& filePath() const { return _filePath; }

		// On hit, copies the cached record into outRecord and returns true; the caller fills run-context fields the cache doesn't persist (dataset/schema names, paths, weights, point counts).
		bool tryGet(const EvaluationCacheKey& key, SchemaSearchRecord& outRecord) const;

		// Appends a record under the given key. No-op when the cache is not open or is read-only.
		void put(const EvaluationCacheKey& key, const SchemaSearchRecord& record);

		size_t hitCount() const { return _hits; }
		size_t missCount() const { return _misses; }
		size_t entryCount() const;

	private:
		struct Entry
		{
			SchemaSearchRecord _record;
		};

		bool loadFromDisk();
		void appendLine(const std::string& canonicalKey, const SchemaSearchRecord& record);

		std::string								_filePath;
		bool									_readOnly = false;
		mutable std::mutex						_mutex;
		std::unordered_map<std::string, Entry>	_entries;
		std::ofstream							_appendStream;
		mutable std::atomic<size_t>				_hits{0};
		mutable std::atomic<size_t>				_misses{0};
	};

	// Builds a cache key from the live evaluation context; the dataset fingerprint (name + point count + bounds) keeps the key stable across paths/seeds, and the caller supplies the schema signature.
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

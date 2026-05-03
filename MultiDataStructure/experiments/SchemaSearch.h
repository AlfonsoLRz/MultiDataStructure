#pragma once

#include "../stdafx.h"
#include "FeatureExtraction.h"
#include "Metrics.h"

namespace Experiments
{
	struct WorkloadProfile
	{
		std::string name = "mixed";
		double rangeWeight = 0.4;
		double radiusWeight = 0.3;
		double knnWeight = 0.3;
		size_t numQueries = 1000;
		size_t knnK = 16;
		uint32_t querySeed = 1337;
	};

	struct ScoreWeights
	{
		double lambdaBuild = 0.001;
		double lambdaMemory = 0.01;
		double lambdaImbalance = 0.01;
	};

	struct SchemaSearchOptions
	{
		std::vector<std::string> inputPaths;
		std::vector<std::string> schemaPaths;
		std::vector<std::string> workloadPaths;
		std::string csvPath = "results/schema_search.csv";
		std::string bestCsvPath = "results/schema_search_best.csv";
		bool includeSyntheticDatasets = true;
		bool useBinaryCache = true;
		bool rebuildBinaryCache = false;
		bool pauseAtEnd = false;
		size_t syntheticScale = 512;
		size_t queryCountOverride = 0;
		size_t knnKOverride = 0;
		uint32_t querySeed = 1337;
		bool querySeedOverride = false;
		ScoreWeights weights;
	};

	struct SchemaSearchRecord
	{
		std::string datasetName;
		std::string datasetSource;
		size_t numPoints = 0;
		std::string workloadName;
		double rangeWeight = 0.0;
		double radiusWeight = 0.0;
		double knnWeight = 0.0;
		size_t numQueries = 0;
		size_t knnK = 0;
		uint32_t querySeed = 0;
		std::string schemaName;
		std::string schemaPath;
		BuildMetrics buildMetrics;
		QueryMetrics queryMetrics;
		size_t rangeQueries = 0;
		size_t radiusQueries = 0;
		size_t knnQueries = 0;
		double score = 0.0;
		double scoreMemoryMb = 0.0;
		double scoreImbalancePenalty = 0.0;
		ScoreWeights weights;
		PointCloudFeatures pointFeatures;
		WorkloadFeatures workloadFeatures;
	};

	WorkloadProfile parseWorkloadProfile(const std::string& jsonText, const std::string& sourceName = {});
	WorkloadProfile loadWorkloadProfile(const std::string& filename);
	double computeSchemaSearchScore(
		const BuildMetrics& buildMetrics,
		const QueryMetrics& queryMetrics,
		const ScoreWeights& weights,
		double& memoryMb,
		double& imbalancePenalty);
	std::vector<SchemaSearchRecord> selectBestRecords(const std::vector<SchemaSearchRecord>& records);
	int runSchemaSearch(const SchemaSearchOptions& options);
}

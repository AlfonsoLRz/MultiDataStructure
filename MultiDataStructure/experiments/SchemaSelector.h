#pragma once

#include "../stdafx.h"
#include "../core/Config.h"
#include "../workloads/points/PointCloud.h"
#include "FeatureExtraction.h"
#include "SchemaSearch.h"

namespace Experiments
{
	struct CandidatePrediction
	{
		std::string schemaName;
		std::string schemaPath;
		double predictedScore = 0.0;
	};

	struct SchemaSelection
	{
		std::string modelPath;
		std::string workloadProfilePath;
		std::string schemaName;
		std::string schemaPath;
		double predictedScore = 0.0;
		bool measuredScore = false;
		std::vector<CandidatePrediction> candidates;
	};

	struct SchemaSelectorModel
	{
		std::string modelType;
		std::string sourceModel;
		bool measuredBestSelector = false;
		bool onnxScoreRanker = false;
		std::string measuredDatasetName;
		std::string measuredDatasetPath;
		std::string measuredWorkloadName;
		std::string fixedSchemaName;
		std::string fixedSchemaPath;
		double fixedMeasuredScore = 0.0;
		std::vector<CandidatePrediction> measuredCandidates;
		std::string onnxModelPath;
		std::string onnxInputName = "features";
		std::string onnxOutputName = "score";
		std::string onnxExecutionProvider = "cpu";
		int onnxDeviceId = 0;
		std::vector<std::string> featureNames;
		std::vector<double> coefficients;
		double intercept = 0.0;
		std::vector<SchemaCandidate> candidates;
	};

	SchemaSelectorModel loadSchemaSelectorModel(const std::string& filename);
	std::vector<double> schemaFeatureVector(const SchemaConfig& schema);
	std::vector<double> selectorFeatureVector(
		const std::vector<std::string>& featureNames,
		const PointCloudFeatures& pointFeatures,
		const WorkloadFeatures& workloadFeatures,
		const SchemaConfig& schema);
	double predictScore(
		const SchemaSelectorModel& model,
		const PointCloudFeatures& pointFeatures,
		const WorkloadFeatures& workloadFeatures,
		const SchemaConfig& schema);
	std::vector<CandidatePrediction> scoreSchemaCandidates(
		const SchemaSelectorModel& model,
		const WorkloadProfile& workload,
		const PointCloud& cloud,
		const std::vector<SchemaCandidate>& candidates);
	SchemaSelection selectSchemaForCloud(
		const std::string& modelPath,
		const std::string& workloadProfilePath,
		const PointCloud& cloud);
}

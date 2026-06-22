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
		std::string	_schemaName;
		std::string	_schemaPath;
		double		_predictedScore = 0.0;
	};

	struct SchemaSelection
	{
		std::string	_modelPath;
		std::string	_workloadProfilePath;
		std::string	_schemaName;
		std::string	_schemaPath;
		double	_predictedScore = 0.0;
		bool	_measuredScore = false;
		std::vector<CandidatePrediction>	_candidates;
	};

	struct SchemaSelectorModel
	{
		std::string	_modelType;
		std::string	_sourceModel;
		bool	_measuredBestSelector = false;
		bool	_onnxScoreRanker = false;
		std::string	_measuredDatasetName;
		std::string	_measuredDatasetPath;
		std::string	_measuredWorkloadName;
		std::string	_fixedSchemaName;
		std::string	_fixedSchemaPath;
		double	_fixedMeasuredScore = 0.0;
		std::vector<CandidatePrediction>	_measuredCandidates;
		std::string	_onnxModelPath;
		std::string	_onnxInputName = "features";
		std::string	_onnxOutputName = "score";
		std::string	_onnxExecutionProvider = "cpu";
		int	_onnxDeviceId = 0;
		std::vector<std::string>	_featureNames;
		std::vector<double>	_coefficients;
		double	_intercept = 0.0;
		std::vector<SchemaCandidate>	_candidates;
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

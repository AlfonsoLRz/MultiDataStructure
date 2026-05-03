#include "../stdafx.h"
#include "SchemaSelector.h"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

namespace
{
	bool pathExists(const std::filesystem::path& path)
	{
		std::error_code error;
		return std::filesystem::exists(path, error);
	}

	std::filesystem::path resolveExistingPath(const std::string& filename)
	{
		const std::filesystem::path configuredPath(filename);
		if (pathExists(configuredPath) || configuredPath.is_absolute())
			return configuredPath;

		std::error_code error;
		std::filesystem::path current = std::filesystem::absolute(std::filesystem::current_path(), error);
		if (error)
			return configuredPath;

		for (;;)
		{
			const std::filesystem::path candidate = (current / configuredPath).lexically_normal();
			if (pathExists(candidate))
				return candidate;

			if (!current.has_parent_path() || current == current.parent_path())
				break;

			current = current.parent_path();
		}

		return configuredPath;
	}

	std::string asString(const boost::json::object& object, const char* key, const std::string& fallback = {})
	{
		if (const boost::json::value* value = object.if_contains(key))
		{
			if (value->is_string())
				return std::string(value->as_string().c_str());
		}

		return fallback;
	}

	double asDouble(const boost::json::value& value)
	{
		if (value.is_double())
			return value.as_double();
		if (value.is_int64())
			return static_cast<double>(value.as_int64());
		if (value.is_uint64())
			return static_cast<double>(value.as_uint64());
		throw std::runtime_error("Expected numeric model coefficient");
	}

	std::vector<std::string> parseStringArray(const boost::json::value& value, const char* fieldName)
	{
		if (!value.is_array())
			throw std::runtime_error(std::string(fieldName) + " must be an array");

		std::vector<std::string> result;
		for (const boost::json::value& entry : value.as_array())
		{
			if (!entry.is_string())
				throw std::runtime_error(std::string(fieldName) + " entries must be strings");
			result.push_back(std::string(entry.as_string().c_str()));
		}
		return result;
	}

	std::vector<double> parseDoubleArray(const boost::json::value& value, const char* fieldName)
	{
		if (!value.is_array())
			throw std::runtime_error(std::string(fieldName) + " must be an array");

		std::vector<double> result;
		for (const boost::json::value& entry : value.as_array())
			result.push_back(asDouble(entry));
		return result;
	}

	double featureValue(
		const std::string& name,
		const Experiments::PointCloudFeatures& point,
		const Experiments::WorkloadFeatures& workload,
		const std::vector<double>& schema)
	{
		if (name == "num_points") return static_cast<double>(point.numPoints);
		if (name == "feature_sample_size") return static_cast<double>(point.sampleSize);
		if (name == "bbox_x") return point.bboxX;
		if (name == "bbox_y") return point.bboxY;
		if (name == "bbox_z") return point.bboxZ;
		if (name == "aspect_xy") return point.aspectXY;
		if (name == "aspect_xz") return point.aspectXZ;
		if (name == "aspect_yz") return point.aspectYZ;
		if (name == "density_bbox") return point.densityBbox;
		if (name == "height_mean") return point.heightMean;
		if (name == "height_std") return point.heightStd;
		if (name == "height_range") return point.heightRange;
		if (name == "cov_eig_0") return point.covEig0;
		if (name == "cov_eig_1") return point.covEig1;
		if (name == "cov_eig_2") return point.covEig2;
		if (name == "linearity") return point.linearity;
		if (name == "planarity") return point.planarity;
		if (name == "scattering") return point.scattering;
		if (name == "occupancy_ratio_8") return point.occupancyRatio8;
		if (name == "occupancy_entropy_8") return point.occupancyEntropy8;
		if (name == "density_cv_8") return point.densityCv8;
		if (name == "verticality_score") return point.verticalityScore;
		if (name == "flatness_score") return point.flatnessScore;
		if (name == "w_range") return workload.wRange;
		if (name == "w_radius") return workload.wRadius;
		if (name == "w_knn") return workload.wKnn;
		if (name == "knn_k") return static_cast<double>(workload.knnK);
		if (name == "num_queries") return static_cast<double>(workload.numQueries);
		if (name == "query_scale_mean") return workload.queryScaleMean;
		if (name == "query_scale_std") return workload.queryScaleStd;
		if (name == "build_weight") return workload.buildWeight;
		if (name == "memory_weight") return workload.memoryWeight;
		if (name == "schema_has_quadtree") return schema[0];
		if (name == "schema_has_octree") return schema[1];
		if (name == "schema_has_kdtree") return schema[2];
		if (name == "schema_has_grid2d") return schema[3];
		if (name == "schema_has_grid3d") return schema[4];
		if (name == "schema_num_blocks") return schema[5];
		if (name == "schema_total_levels") return schema[6];
		if (name == "schema_max_leaf_capacity") return schema[7];
		if (name == "schema_min_leaf_capacity") return schema[8];
		throw std::runtime_error("Unsupported schema-selector feature: " + name);
	}

	std::vector<Experiments::SchemaCandidate> parseCandidates(const boost::json::value& value)
	{
		if (!value.is_array())
			throw std::runtime_error("candidate_schemas must be an array");

		std::vector<Experiments::SchemaCandidate> candidates;
		for (const boost::json::value& entry : value.as_array())
		{
			if (!entry.is_object())
				throw std::runtime_error("candidate_schemas entries must be objects");

			const boost::json::object& object = entry.as_object();
			Experiments::SchemaCandidate candidate;
			candidate.name = asString(object, "name");
			candidate.path = asString(object, "path");
			if (candidate.path.empty())
				throw std::runtime_error("candidate schema path is empty for " + candidate.name);
			candidate.config = Config::loadSchemaConfig(candidate.path);
			if (candidate.name.empty())
				candidate.name = candidate.config.name;
			candidates.push_back(std::move(candidate));
		}
		return candidates;
	}

	Experiments::CandidatePrediction parseCandidatePrediction(const boost::json::object& object)
	{
		Experiments::CandidatePrediction prediction;
		prediction.schemaName = asString(object, "name", asString(object, "schema_name"));
		prediction.schemaPath = asString(object, "path", asString(object, "schema_path"));
		if (const boost::json::value* score = object.if_contains("score"))
			prediction.predictedScore = asDouble(*score);
		else if (const boost::json::value* predicted = object.if_contains("predicted_score"))
			prediction.predictedScore = asDouble(*predicted);
		return prediction;
	}

	std::vector<Experiments::CandidatePrediction> parseCandidateScores(const boost::json::value& value)
	{
		if (!value.is_array())
			throw std::runtime_error("candidate_scores must be an array");

		std::vector<Experiments::CandidatePrediction> candidates;
		for (const boost::json::value& entry : value.as_array())
		{
			if (!entry.is_object())
				throw std::runtime_error("candidate_scores entries must be objects");
			candidates.push_back(parseCandidatePrediction(entry.as_object()));
		}
		return candidates;
	}
}

Experiments::SchemaSelectorModel Experiments::loadSchemaSelectorModel(const std::string& filename)
{
	const std::filesystem::path resolvedPath = resolveExistingPath(filename);
	std::ifstream file(resolvedPath);
	if (!file.is_open())
		throw std::runtime_error("Unable to open schema selector model: " + filename);

	std::stringstream buffer;
	buffer << file.rdbuf();

	boost::system::error_code error;
	boost::json::value rootValue = boost::json::parse(buffer.str(), error);
	if (error)
		throw std::runtime_error("Invalid schema selector JSON in " + resolvedPath.string() + ": " + error.message());
	if (!rootValue.is_object())
		throw std::runtime_error("Schema selector JSON root must be an object");

	const boost::json::object& root = rootValue.as_object();
	SchemaSelectorModel model;
	model.modelType = asString(root, "model_type");
	model.sourceModel = asString(root, "source_model");
	if (model.modelType == "measured_best_schema")
	{
		const boost::json::value* selectedSchema = root.if_contains("selected_schema");
		if (!selectedSchema || !selectedSchema->is_object())
			throw std::runtime_error("measured_best_schema model requires selected_schema");

		const Experiments::CandidatePrediction selected = parseCandidatePrediction(selectedSchema->as_object());
		if (selected.schemaPath.empty())
			throw std::runtime_error("measured_best_schema selected_schema path is empty");

		model.measuredBestSelector = true;
		model.measuredDatasetName = asString(root, "dataset_name");
		model.measuredDatasetPath = asString(root, "dataset_path");
		model.measuredWorkloadName = asString(root, "workload_name");
		model.fixedSchemaName = selected.schemaName;
		model.fixedSchemaPath = selected.schemaPath;
		model.fixedMeasuredScore = selected.predictedScore;

		if (const boost::json::value* candidateScores = root.if_contains("candidate_scores"))
			model.measuredCandidates = parseCandidateScores(*candidateScores);

		if (model.measuredCandidates.empty())
			model.measuredCandidates.push_back(selected);

		std::sort(model.measuredCandidates.begin(), model.measuredCandidates.end(), [](const Experiments::CandidatePrediction& left, const Experiments::CandidatePrediction& right) {
			if (left.predictedScore == right.predictedScore)
				return left.schemaName < right.schemaName;
			return left.predictedScore < right.predictedScore;
		});

		return model;
	}

	if (model.modelType != "linear_score_ranker")
		throw std::runtime_error("Unsupported schema selector model_type: " + model.modelType);

	const boost::json::value* featureNames = root.if_contains("feature_names");
	const boost::json::value* coefficients = root.if_contains("coefficients");
	const boost::json::value* candidates = root.if_contains("candidate_schemas");
	if (!featureNames || !coefficients || !candidates)
		throw std::runtime_error("Schema selector JSON requires feature_names, coefficients, and candidate_schemas");

	model.featureNames = parseStringArray(*featureNames, "feature_names");
	model.coefficients = parseDoubleArray(*coefficients, "coefficients");
	if (model.featureNames.size() != model.coefficients.size())
		throw std::runtime_error("Schema selector feature_names and coefficients size mismatch");

	if (const boost::json::value* intercept = root.if_contains("intercept"))
		model.intercept = asDouble(*intercept);

	model.candidates = parseCandidates(*candidates);
	if (model.candidates.empty())
		throw std::runtime_error("Schema selector model has no candidate schemas");

	return model;
}

std::vector<double> Experiments::schemaFeatureVector(const SchemaConfig& schema)
{
	std::vector<double> features(9, 0.0);
	features[5] = static_cast<double>(schema.levels.size());
	features[6] = static_cast<double>(schema.totalLevels());

	double maxLeafCapacity = 0.0;
	double minLeafCapacity = std::numeric_limits<double>::max();
	for (const SchemaLevelConfig& level : schema.levels)
	{
		if (level.type == MultiDataStructure::QuadTreeNode)
			features[0] = 1.0;
		else if (level.type == MultiDataStructure::OctreeNode)
			features[1] = 1.0;
		else if (level.type == MultiDataStructure::KDTreeNode)
			features[2] = 1.0;

		if (level.leafCapacity > 0)
		{
			const double capacity = static_cast<double>(level.leafCapacity);
			maxLeafCapacity = std::max(maxLeafCapacity, capacity);
			minLeafCapacity = std::min(minLeafCapacity, capacity);
		}
	}

	features[7] = maxLeafCapacity;
	features[8] = minLeafCapacity == std::numeric_limits<double>::max() ? 0.0 : minLeafCapacity;
	return features;
}

double Experiments::predictScore(
	const SchemaSelectorModel& model,
	const PointCloudFeatures& pointFeatures,
	const WorkloadFeatures& workloadFeatures,
	const SchemaConfig& schema)
{
	const std::vector<double> schemaFeatures = schemaFeatureVector(schema);
	double score = model.intercept;
	for (size_t i = 0; i < model.featureNames.size(); ++i)
	{
		score += model.coefficients[i] * featureValue(
			model.featureNames[i],
			pointFeatures,
			workloadFeatures,
			schemaFeatures);
	}
	return score;
}

Experiments::SchemaSelection Experiments::selectSchemaForCloud(
	const std::string& modelPath,
	const std::string& workloadProfilePath,
	const PointCloud& cloud)
{
	const SchemaSelectorModel model = loadSchemaSelectorModel(modelPath);
	SchemaSelection selection;
	selection.modelPath = modelPath;
	selection.workloadProfilePath = workloadProfilePath;

	if (model.measuredBestSelector)
	{
		selection.schemaName = model.fixedSchemaName;
		selection.schemaPath = model.fixedSchemaPath;
		selection.predictedScore = model.fixedMeasuredScore;
		selection.measuredScore = true;
		selection.candidates = model.measuredCandidates;
		return selection;
	}

	const WorkloadProfile workload = loadWorkloadProfile(workloadProfilePath);
	const PointCloudFeatures pointFeatures = extractPointCloudFeatures(cloud);
	const WorkloadFeatures workloadFeatures = extractWorkloadFeatures(workload, ScoreWeights{});

	for (const SchemaCandidate& candidate : model.candidates)
	{
		CandidatePrediction prediction;
		prediction.schemaName = candidate.config.name;
		prediction.schemaPath = candidate.path;
		prediction.predictedScore = predictScore(model, pointFeatures, workloadFeatures, candidate.config);
		selection.candidates.push_back(prediction);
	}

	std::sort(selection.candidates.begin(), selection.candidates.end(), [](const CandidatePrediction& left, const CandidatePrediction& right) {
		if (left.predictedScore == right.predictedScore)
			return left.schemaName < right.schemaName;
		return left.predictedScore < right.predictedScore;
	});

	const CandidatePrediction& best = selection.candidates.front();
	selection.schemaName = best.schemaName;
	selection.schemaPath = best.schemaPath;
	selection.predictedScore = best.predictedScore;
	return selection;
}

#include "../stdafx.h"
#include "SchemaSelector.h"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

#if defined(MDSPC_ENABLE_ONNX)
#include <onnxruntime_cxx_api.h>
#define MDSPC_ONNX_AVAILABLE 1
#else
#define MDSPC_ONNX_AVAILABLE 0
#endif

static bool pathExists(const std::filesystem::path& path)
{
	std::error_code error;
	return std::filesystem::exists(path, error);
}

static std::filesystem::path resolveExistingPath(const std::string& filename)
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

static std::string asString(const boost::json::object& object, const char* key, const std::string& fallback = {})
{
	if (const boost::json::value* value = object.if_contains(key))
	{
		if (value->is_string())
			return std::string(value->as_string().c_str());
	}

	return fallback;
}

static double asDouble(const boost::json::value& value)
{
	if (value.is_double())
		return value.as_double();
	if (value.is_int64())
		return static_cast<double>(value.as_int64());
	if (value.is_uint64())
		return static_cast<double>(value.as_uint64());
	throw std::runtime_error("Expected numeric model coefficient");
}

static int asInt(const boost::json::object& object, const char* key, int fallback = 0)
{
	if (const boost::json::value* value = object.if_contains(key))
	{
		if (value->is_int64())
			return static_cast<int>(value->as_int64());
		if (value->is_uint64())
			return static_cast<int>(value->as_uint64());
	}
	return fallback;
}

static std::vector<std::string> parseStringArray(const boost::json::value& value, const char* fieldName)
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

static std::vector<double> parseDoubleArray(const boost::json::value& value, const char* fieldName)
{
	if (!value.is_array())
		throw std::runtime_error(std::string(fieldName) + " must be an array");

	std::vector<double> result;
	for (const boost::json::value& entry : value.as_array())
		result.push_back(asDouble(entry));
	return result;
}

static double featureValue(
	const std::string& name,
	const Experiments::PointCloudFeatures& point,
	const Experiments::WorkloadFeatures& workload,
	const std::vector<double>& schema)
{
	if (name == "num_points") return static_cast<double>(point._numPoints);
	if (name == "feature_sample_size") return static_cast<double>(point._sampleSize);
	if (name == "bbox_x") return point._bboxX;
	if (name == "bbox_y") return point._bboxY;
	if (name == "bbox_z") return point._bboxZ;
	if (name == "aspect_xy") return point._aspectXY;
	if (name == "aspect_xz") return point._aspectXZ;
	if (name == "aspect_yz") return point._aspectYZ;
	if (name == "density_bbox") return point._densityBbox;
	if (name == "height_mean") return point._heightMean;
	if (name == "height_std") return point._heightStd;
	if (name == "height_range") return point._heightRange;
	if (name == "cov_eig_0") return point._covEig0;
	if (name == "cov_eig_1") return point._covEig1;
	if (name == "cov_eig_2") return point._covEig2;
	if (name == "linearity") return point._linearity;
	if (name == "planarity") return point._planarity;
	if (name == "scattering") return point._scattering;
	if (name == "occupancy_ratio_8") return point._occupancyRatio8;
	if (name == "occupancy_entropy_8") return point._occupancyEntropy8;
	if (name == "density_cv_8") return point._densityCv8;
	if (name == "verticality_score") return point._verticalityScore;
	if (name == "flatness_score") return point._flatnessScore;
	if (name == "w_range") return workload._wRange;
	if (name == "w_radius") return workload._wRadius;
	if (name == "w_knn") return workload._wKnn;
	if (name == "knn_k") return static_cast<double>(workload._knnK);
	if (name == "num_queries") return static_cast<double>(workload._numQueries);
	if (name == "range_scale_min") return workload._rangeScaleMin;
	if (name == "range_scale_max") return workload._rangeScaleMax;
	if (name == "radius_scale_min") return workload._radiusScaleMin;
	if (name == "radius_scale_max") return workload._radiusScaleMax;
	if (name == "query_scale_mean") return workload._queryScaleMean;
	if (name == "query_scale_std") return workload._queryScaleStd;
	if (name == "build_weight") return workload._buildWeight;
	if (name == "memory_weight") return workload._memoryWeight;
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

static std::vector<Experiments::SchemaCandidate> parseCandidates(const boost::json::value& value)
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
		candidate._name = asString(object, "name");
		candidate._path = asString(object, "path");
		if (candidate._path.empty())
			throw std::runtime_error("candidate schema path is empty for " + candidate._name);
		candidate._config = Config::loadSchemaConfig(candidate._path);
		if (candidate._name.empty())
			candidate._name = candidate._config._name;
		candidates.push_back(std::move(candidate));
	}
	return candidates;
}

static Experiments::CandidatePrediction parseCandidatePrediction(const boost::json::object& object)
{
	Experiments::CandidatePrediction prediction;
	prediction._schemaName = asString(object, "name", asString(object, "schema_name"));
	prediction._schemaPath = asString(object, "path", asString(object, "schema_path"));
	if (const boost::json::value* score = object.if_contains("score"))
		prediction._predictedScore = asDouble(*score);
	else if (const boost::json::value* predicted = object.if_contains("predicted_score"))
		prediction._predictedScore = asDouble(*predicted);
	return prediction;
}

static std::vector<Experiments::CandidatePrediction> parseCandidateScores(const boost::json::value& value)
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

#if MDSPC_ONNX_AVAILABLE
static std::string onnxStatusMessage(const OrtApi& api, OrtStatus* status)
{
	if (status == nullptr)
		return {};
	const std::string message = api.GetErrorMessage(status);
	api.ReleaseStatus(status);
	return message;
}

static void throwOnOnnxStatus(const OrtApi& api, OrtStatus* status, const char* operation)
{
	if (status == nullptr)
		return;
	throw std::runtime_error(std::string("ONNX Runtime ") + operation + " failed: " + onnxStatusMessage(api, status));
}

static std::filesystem::path resolveOnnxModelPath(const Experiments::SchemaSelectorModel& model)
{
	const std::string path = model._onnxModelPath.empty() ? model._sourceModel : model._onnxModelPath;
	if (path.empty())
		throw std::runtime_error("onnx_score_ranker requires source_model or onnx_model");
	return resolveExistingPath(path);
}

static const ORTCHAR_T* onnxPathChars(const std::filesystem::path& path, std::wstring& widePath, std::string& narrowPath)
{
#if defined(_WIN32)
	(void)narrowPath;
	widePath = path.wstring();
	return widePath.c_str();
#else
	(void)widePath;
	narrowPath = path.string();
	return narrowPath.c_str();
#endif
}

class OnnxScoreRanker
{
public:
	explicit OnnxScoreRanker(const Experiments::SchemaSelectorModel& model)
		: model_(model),
		  env_(ORT_LOGGING_LEVEL_WARNING, "mdspc_schema_selector"),
		  sessionOptions_(),
		  session_(nullptr)
	{
		sessionOptions_.SetIntraOpNumThreads(1);
		sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

		if (model_._onnxExecutionProvider == "cuda")
			appendCudaProvider();
		else if (model_._onnxExecutionProvider != "cpu")
			throw std::runtime_error("Unsupported ONNX execution_provider: " + model_._onnxExecutionProvider);

		const std::filesystem::path modelPath = resolveOnnxModelPath(model_);
		std::wstring _widePath;
		std::string _narrowPath;
		session_ = Ort::Session(env_, onnxPathChars(modelPath, widePath, narrowPath), sessionOptions_);
	}

	double predict(const std::vector<double>& features)
	{
		if (model_._onnxInputName.empty() || model_._onnxOutputName.empty())
			throw std::runtime_error("onnx_score_ranker requires input_name and output_name");

		std::vector<float> _input;
		input.reserve(features.size());
		for (const double value : features)
			input.push_back(static_cast<float>(value));

		std::array<int64_t, 2> shape = { 1, static_cast<int64_t>(input.size()) };
		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
			memoryInfo,
			input.data(),
			input.size(),
			shape.data(),
			shape.size());

		const char* inputNames[] = { model_._onnxInputName.c_str() };
		const char* outputNames[] = { model_._onnxOutputName.c_str() };
		std::vector<Ort::Value> outputs = session_.Run(
			Ort::RunOptions{ nullptr },
			inputNames,
			&inputTensor,
			1,
			outputNames,
			1);

		if (outputs.empty() || !outputs[0].IsTensor())
			throw std::runtime_error("ONNX score ranker did not return a tensor output");

		const Ort::TensorTypeAndShapeInfo shapeInfo = outputs[0].GetTensorTypeAndShapeInfo();
		if (shapeInfo.GetElementCount() < 1)
			throw std::runtime_error("ONNX score ranker returned an empty tensor");

		const ONNXTensorElementDataType elementType = shapeInfo.GetElementType();
		if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
			return static_cast<double>(outputs[0].GetTensorData<float>()[0]);
		if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
			return outputs[0].GetTensorData<double>()[0];

		throw std::runtime_error("ONNX score ranker output must be float or double");
	}

private:
	void appendCudaProvider()
	{
		const OrtApi& api = Ort::GetApi();
		OrtCUDAProviderOptionsV2* _cudaOptions = nullptr;
		throwOnOnnxStatus(api, api.CreateCUDAProviderOptions(&cudaOptions), "CreateCUDAProviderOptions");

		std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> optionsGuard(
			cudaOptions,
			api.ReleaseCUDAProviderOptions);

		const std::string deviceId = std::to_string(model_._onnxDeviceId);
		const char* keys[] = { "device_id" };
		const char* values[] = { deviceId.c_str() };
		throwOnOnnxStatus(api, api.UpdateCUDAProviderOptions(cudaOptions, keys, values, 1), "UpdateCUDAProviderOptions");
		throwOnOnnxStatus(
			api,
			api.SessionOptionsAppendExecutionProvider_CUDA_V2(static_cast<OrtSessionOptions*>(sessionOptions_), cudaOptions),
			"SessionOptionsAppendExecutionProvider_CUDA_V2");
	}

	const Experiments::SchemaSelectorModel& model_;
	Ort::Env env_;
	Ort::SessionOptions sessionOptions_;
	Ort::Session session_;
};
#endif

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
	model._modelType = asString(root, "model_type");
	model._sourceModel = asString(root, "source_model");
	if (model._modelType == "measured_best_schema")
	{
		const boost::json::value* selectedSchema = root.if_contains("selected_schema");
		if (!selectedSchema || !selectedSchema->is_object())
			throw std::runtime_error("measured_best_schema model requires selected_schema");

		const Experiments::CandidatePrediction selected = parseCandidatePrediction(selectedSchema->as_object());
		if (selected._schemaPath.empty())
			throw std::runtime_error("measured_best_schema selected_schema path is empty");

		model._measuredBestSelector = true;
		model._measuredDatasetName = asString(root, "dataset_name");
		model._measuredDatasetPath = asString(root, "dataset_path");
		model._measuredWorkloadName = asString(root, "workload_name");
		model._fixedSchemaName = selected._schemaName;
		model._fixedSchemaPath = selected._schemaPath;
		model._fixedMeasuredScore = selected._predictedScore;

		if (const boost::json::value* candidateScores = root.if_contains("candidate_scores"))
			model._measuredCandidates = parseCandidateScores(*candidateScores);

		if (model._measuredCandidates.empty())
			model._measuredCandidates.push_back(selected);

		std::sort(model._measuredCandidates.begin(), model._measuredCandidates.end(), [](const Experiments::CandidatePrediction& left, const Experiments::CandidatePrediction& right) {
			if (left._predictedScore == right._predictedScore)
				return left._schemaName < right._schemaName;
			return left._predictedScore < right._predictedScore;
		});

		return model;
	}

	if (model._modelType == "onnx_score_ranker")
	{
		model._onnxScoreRanker = true;
		model._onnxModelPath = asString(root, "onnx_model", model._sourceModel);
		model._onnxInputName = asString(root, "input_name", "features");
		model._onnxOutputName = asString(root, "output_name", "score");
		model._onnxExecutionProvider = asString(root, "execution_provider", "cpu");
		model._onnxDeviceId = asInt(root, "device_id", 0);

		const boost::json::value* featureNames = root.if_contains("feature_names");
		const boost::json::value* candidates = root.if_contains("candidate_schemas");
		if (!featureNames || !candidates)
			throw std::runtime_error("ONNX schema selector JSON requires feature_names and candidate_schemas");
		if (model._onnxModelPath.empty())
			throw std::runtime_error("ONNX schema selector JSON requires source_model or onnx_model");

		model._featureNames = parseStringArray(*featureNames, "feature_names");
		model._candidates = parseCandidates(*candidates);
		if (model._candidates.empty())
			throw std::runtime_error("ONNX schema selector model has no candidate schemas");
		return model;
	}

	if (model._modelType != "linear_score_ranker")
		throw std::runtime_error("Unsupported schema selector model_type: " + model._modelType);

	const boost::json::value* featureNames = root.if_contains("feature_names");
	const boost::json::value* coefficients = root.if_contains("coefficients");
	const boost::json::value* candidates = root.if_contains("candidate_schemas");
	if (!featureNames || !coefficients || !candidates)
		throw std::runtime_error("Schema selector JSON requires feature_names, coefficients, and candidate_schemas");

	model._featureNames = parseStringArray(*featureNames, "feature_names");
	model._coefficients = parseDoubleArray(*coefficients, "coefficients");
	if (model._featureNames.size() != model._coefficients.size())
		throw std::runtime_error("Schema selector feature_names and coefficients size mismatch");

	if (const boost::json::value* intercept = root.if_contains("intercept"))
		model._intercept = asDouble(*intercept);

	model._candidates = parseCandidates(*candidates);
	if (model._candidates.empty())
		throw std::runtime_error("Schema selector model has no candidate schemas");

	return model;
}

std::vector<double> Experiments::schemaFeatureVector(const SchemaConfig& schema)
{
	std::vector<double> features(9, 0.0);
	features[5] = static_cast<double>(schema._levels.size());
	features[6] = static_cast<double>(schema.totalLevels());

	double maxLeafCapacity = 0.0;
	double minLeafCapacity = std::numeric_limits<double>::max();
	for (const SchemaLevelConfig& level : schema._levels)
	{
		SchemaPrimitiveKind kind = level._primitiveKind;
		try
		{
			if (!level._typeName.empty())
				kind = Config::parseSchemaPrimitiveKind(level._typeName);
		}
		catch (const std::exception&)
		{
			kind = level._primitiveKind;
		}
		if (kind == SchemaPrimitiveKind::QuadTree)
			features[0] = 1.0;
		else if (kind == SchemaPrimitiveKind::Octree ||
				 kind == SchemaPrimitiveKind::KarrasOctree ||
				 kind == SchemaPrimitiveKind::RegularGrid ||
				 kind == SchemaPrimitiveKind::HGrid ||
				 kind == SchemaPrimitiveKind::Mixed)
			features[1] = 1.0;
		else if (kind == SchemaPrimitiveKind::KDTree || kind == SchemaPrimitiveKind::BIH)
			features[2] = 1.0;

		if (level._leafCapacity > 0)
		{
			const double capacity = static_cast<double>(level._leafCapacity);
			maxLeafCapacity = std::max(maxLeafCapacity, capacity);
			minLeafCapacity = std::min(minLeafCapacity, capacity);
		}
	}

	features[7] = maxLeafCapacity;
	features[8] = minLeafCapacity == std::numeric_limits<double>::max() ? 0.0 : minLeafCapacity;
	return features;
}

std::vector<double> Experiments::selectorFeatureVector(
	const std::vector<std::string>& featureNames,
	const PointCloudFeatures& pointFeatures,
	const WorkloadFeatures& workloadFeatures,
	const SchemaConfig& schema)
{
	const std::vector<double> schemaFeatures = schemaFeatureVector(schema);
	std::vector<double> features;
	features.reserve(featureNames.size());
	for (const std::string& name : featureNames)
	{
		features.push_back(featureValue(
			name,
			pointFeatures,
			workloadFeatures,
			schemaFeatures));
	}
	return features;
}

double Experiments::predictScore(
	const SchemaSelectorModel& model,
	const PointCloudFeatures& pointFeatures,
	const WorkloadFeatures& workloadFeatures,
	const SchemaConfig& schema)
{
	if (model._onnxScoreRanker)
		throw std::runtime_error("ONNX score rankers are evaluated through selectSchemaForCloud");

	double score = model._intercept;
	const std::vector<double> features = selectorFeatureVector(model._featureNames, pointFeatures, workloadFeatures, schema);
	for (size_t i = 0; i < features.size(); ++i)
	{
		score += model._coefficients[i] * features[i];
	}
	return score;
}

std::vector<Experiments::CandidatePrediction> Experiments::scoreSchemaCandidates(
	const SchemaSelectorModel& model,
	const WorkloadProfile& workload,
	const PointCloud& cloud,
	const std::vector<SchemaCandidate>& candidates)
{
	if (model._measuredBestSelector)
		throw std::runtime_error("measured_best_schema cannot score arbitrary generated schema candidates");

	const PointCloudFeatures pointFeatures = extractPointCloudFeatures(cloud);
	const WorkloadFeatures workloadFeatures = extractWorkloadFeatures(
		workload,
		workload._hasScoreWeights ? workload._scoreWeights : ScoreWeights{});

#if MDSPC_ONNX_AVAILABLE
	std::unique_ptr<OnnxScoreRanker> onnxRanker;
	if (model._onnxScoreRanker)
		onnxRanker = std::make_unique<OnnxScoreRanker>(model);
#else
	if (model._onnxScoreRanker)
		throw std::runtime_error("ONNX schema selector requested, but this build was not compiled with MDSPC_ENABLE_ONNX. Set OnnxRuntimeDir in the Visual Studio project or use the dependency-free linear/measured selector.");
#endif

	std::vector<CandidatePrediction> predictions;
	predictions.reserve(candidates.size());
	for (const SchemaCandidate& candidate : candidates)
	{
		CandidatePrediction prediction;
		prediction._schemaName = candidate._config._name.empty() ? candidate._name : candidate._config._name;
		prediction._schemaPath = candidate._path;
#if MDSPC_ONNX_AVAILABLE
		if (model._onnxScoreRanker)
		{
			const std::vector<double> features = selectorFeatureVector(
				model._featureNames,
				pointFeatures,
				workloadFeatures,
				candidate._config);
			prediction._predictedScore = onnxRanker->predict(features);
		}
		else
#endif
		prediction._predictedScore = predictScore(model, pointFeatures, workloadFeatures, candidate._config);
		predictions.push_back(prediction);
	}

	std::sort(predictions.begin(), predictions.end(), [](const CandidatePrediction& left, const CandidatePrediction& right) {
		if (left._predictedScore == right._predictedScore)
			return left._schemaName < right._schemaName;
		return left._predictedScore < right._predictedScore;
	});

	return predictions;
}

Experiments::SchemaSelection Experiments::selectSchemaForCloud(
	const std::string& modelPath,
	const std::string& workloadProfilePath,
	const PointCloud& cloud)
{
	const SchemaSelectorModel model = loadSchemaSelectorModel(modelPath);
	SchemaSelection selection;
	selection._modelPath = modelPath;
	selection._workloadProfilePath = workloadProfilePath;

	if (model._measuredBestSelector)
	{
		selection._schemaName = model._fixedSchemaName;
		selection._schemaPath = model._fixedSchemaPath;
		selection._predictedScore = model._fixedMeasuredScore;
		selection._measuredScore = true;
		selection._candidates = model._measuredCandidates;
		return selection;
	}

	const WorkloadProfile workload = loadWorkloadProfile(workloadProfilePath);
	selection._candidates = scoreSchemaCandidates(model, workload, cloud, model._candidates);

	const CandidatePrediction& best = selection._candidates.front();
	selection._schemaName = best._schemaName;
	selection._schemaPath = best._schemaPath;
	selection._predictedScore = best._predictedScore;
	return selection;
}

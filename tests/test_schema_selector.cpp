#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/experiments/SchemaSelector.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		std::filesystem::path tempFile(const std::string& filename)
		{
			return std::filesystem::temp_directory_path() / filename;
		}
	}

	void runSchemaSelectorTests()
	{
		SchemaConfig quadtree = Config::loadSchemaConfig("configs/schemas/quadtree.json");
		SchemaConfig kdtree = Config::loadSchemaConfig("configs/schemas/kdtree.json");

		const std::vector<double> quadtreeFeatures = Experiments::schemaFeatureVector(quadtree);
		const std::vector<double> kdtreeFeatures = Experiments::schemaFeatureVector(kdtree);
		expect(quadtreeFeatures[0] == 1.0, "schema selector marks quadtree block");
		expect(quadtreeFeatures[2] == 0.0, "schema selector leaves kdtree absent for quadtree");
		expect(kdtreeFeatures[2] == 1.0, "schema selector marks kdtree block");
		expect(kdtreeFeatures[6] == static_cast<double>(kdtree.totalLevels()), "schema selector records total levels");

		const std::filesystem::path modelPath = tempFile("multids_schema_selector_test.json");
		{
			std::ofstream file(modelPath);
			file << R"json(
			{
			  "model_type": "linear_score_ranker",
			  "feature_names": ["schema_has_kdtree"],
			  "coefficients": [-1.0],
			  "intercept": 0.0,
			  "candidate_schemas": [
			    { "name": "quadtree_default", "path": "configs/schemas/quadtree.json" },
			    { "name": "kdtree_default", "path": "configs/schemas/kdtree.json" }
			  ]
			}
			)json";
		}

		const PointCloud cloud = SyntheticPointClouds::generateSparseDenseMixture(16, 16, 77);
		const Experiments::SchemaSelection selection = Experiments::selectSchemaForCloud(
			modelPath.string(),
			"configs/workloads/mixed.json",
			cloud);

		expect(selection._schemaName == "kdtree_default", "schema selector chooses lowest predicted score");
		expect(selection._candidates.size() == 2, "schema selector reports candidate predictions");
		expect(selection._candidates[0]._schemaName == "kdtree_default", "schema selector sorts candidates by score");
		expect(selection._candidates[0]._predictedScore < selection._candidates[1]._predictedScore, "schema selector score ordering is deterministic");

		std::filesystem::remove(modelPath);

		const std::filesystem::path measuredModelPath = tempFile("multids_measured_schema_selector_test.json");
		{
			std::ofstream file(measuredModelPath);
			file << R"json(
			{
			  "model_type": "measured_best_schema",
			  "dataset_name": "local_cloud",
			  "dataset_path": "local.xyz",
			  "workload_name": "mixed",
			  "selected_schema": {
			    "name": "quadtree_default",
			    "path": "configs/schemas/quadtree.json",
			    "score": 0.25
			  },
			  "candidate_scores": [
			    { "name": "kdtree_default", "path": "configs/schemas/kdtree.json", "score": 0.50 },
			    { "name": "quadtree_default", "path": "configs/schemas/quadtree.json", "score": 0.25 }
			  ]
			}
			)json";
		}

		const Experiments::SchemaSelection measuredSelection = Experiments::selectSchemaForCloud(
			measuredModelPath.string(),
			"configs/workloads/mixed.json",
			cloud);
		expect(measuredSelection._schemaName == "quadtree_default", "measured local selector uses benchmark winner");
		expect(measuredSelection._candidates[0]._schemaName == "quadtree_default", "measured local selector preserves ranked candidates");
		expect(measuredSelection._predictedScore == 0.25, "measured local selector exposes measured score as selection score");

		std::filesystem::remove(measuredModelPath);

		const std::filesystem::path onnxModelPath = tempFile("multids_onnx_schema_selector_test.json");
		{
			std::ofstream file(onnxModelPath);
			file << R"json(
			{
			  "model_type": "onnx_score_ranker",
			  "source_model": "models/schema_selector.onnx",
			  "input_name": "features",
			  "output_name": "score",
			  "execution_provider": "cpu",
			  "feature_names": ["num_points", "w_range", "schema_has_quadtree"],
			  "candidate_schemas": [
			    { "name": "quadtree_default", "path": "configs/schemas/quadtree.json" },
			    { "name": "kdtree_default", "path": "configs/schemas/kdtree.json" }
			  ]
			}
			)json";
		}

		const Experiments::SchemaSelectorModel onnxModel = Experiments::loadSchemaSelectorModel(onnxModelPath.string());
		expect(onnxModel._onnxScoreRanker, "schema selector parses ONNX ranker model type");
		expect(onnxModel._onnxInputName == "features", "schema selector parses ONNX input name");
		expect(onnxModel._onnxOutputName == "score", "schema selector parses ONNX output name");
		expect(onnxModel._candidates.size() == 2, "schema selector parses ONNX candidate schemas");

		std::filesystem::remove(onnxModelPath);
	}
}

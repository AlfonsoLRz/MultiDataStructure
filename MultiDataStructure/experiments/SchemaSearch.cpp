#include "../stdafx.h"
#include "SchemaSearch.h"

#include "SchemaSelector.h"
#include "../workloads/points/PointCloud.h"
#include "../workloads/points/PointSpatialIndex.h"
#include "../workloads/points/SyntheticPointClouds.h"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

namespace
{
	const std::vector<std::string>& defaultSchemaPaths()
	{
		static const std::vector<std::string> paths = {
			"configs/schemas/quadtree.json",
			"configs/schemas/octree.json",
			"configs/schemas/kdtree.json",
			"configs/schemas/quadtree_octree.json",
			"configs/schemas/octree_kdtree.json",
			"configs/schemas/urban_hybrid.json",
		};
		return paths;
	}

	const std::vector<std::string>& defaultWorkloadPaths()
	{
		static const std::vector<std::string> paths = {
			"configs/workloads/range_heavy.json",
			"configs/workloads/knn_heavy.json",
			"configs/workloads/mixed.json",
		};
		return paths;
	}

	struct SearchDataset
	{
		std::string name;
		std::string source;
		PointCloud cloud;
	};

	struct WorkloadRun
	{
		Experiments::QueryMetrics metrics;
		size_t rangeQueries = 0;
		size_t radiusQueries = 0;
		size_t knnQueries = 0;
	};

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

	size_t asSize(const boost::json::object& object, const char* key, size_t fallback)
	{
		if (const boost::json::value* value = object.if_contains(key))
		{
			if (value->is_int64())
				return static_cast<size_t>(value->as_int64());
			if (value->is_uint64())
				return static_cast<size_t>(value->as_uint64());
			if (value->is_double())
				return static_cast<size_t>(value->as_double());
		}

		return fallback;
	}

	double asDouble(const boost::json::object& object, const char* key, double fallback)
	{
		if (const boost::json::value* value = object.if_contains(key))
		{
			if (value->is_double())
				return value->as_double();
			if (value->is_int64())
				return static_cast<double>(value->as_int64());
			if (value->is_uint64())
				return static_cast<double>(value->as_uint64());
		}

		return fallback;
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

	double elapsedMilliseconds(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
	{
		return std::chrono::duration<double, std::milli>(end - begin).count();
	}

	float randomFloat(std::mt19937& rng, float minValue, float maxValue)
	{
		if (minValue >= maxValue)
			return minValue;

		std::uniform_real_distribution<float> distribution(minValue, maxValue);
		return distribution(rng);
	}

	glm::vec3 randomPointInBounds(std::mt19937& rng, const AABB& bounds)
	{
		const glm::vec3 min = bounds.min();
		const glm::vec3 max = bounds.max();
		return glm::vec3(
			randomFloat(rng, min.x, max.x),
			randomFloat(rng, min.y, max.y),
			randomFloat(rng, min.z, max.z));
	}

	AABB randomQueryBox(std::mt19937& rng, const PointCloud& cloud)
	{
		const glm::vec3 center = randomPointInBounds(rng, cloud.bounds());
		const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.001f));
		const float scale = randomFloat(rng, 0.01f, 0.05f);
		const glm::vec3 halfExtent = glm::max(range * scale * 0.5f, glm::vec3(0.0005f));
		return AABB(center - halfExtent, center + halfExtent);
	}

	float randomQueryRadius(std::mt19937& rng, const PointCloud& cloud)
	{
		const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.001f));
		const float largestRange = std::max({ range.x, range.y, range.z, 1.0f });
		return largestRange * randomFloat(rng, 0.01f, 0.04f);
	}

	std::string datasetNameFromPath(const std::string& inputPath)
	{
		const std::filesystem::path path(inputPath);
		const std::string stem = path.stem().string();
		return stem.empty() ? "points" : stem;
	}

	std::string schemaTypeShortName(MultiDataStructure::DataStructureLevel type)
	{
		switch (type)
		{
		case MultiDataStructure::QuadTreeNode:
			return "qt";
		case MultiDataStructure::OctreeNode:
			return "ot";
		case MultiDataStructure::KDTreeNode:
			return "kd";
		case MultiDataStructure::BvhNode:
			return "bvh";
		default:
			return "x";
		}
	}

	size_t clampPowerOfTwo(size_t value, size_t minValue, size_t maxValue)
	{
		value = std::max<size_t>(1, value);
		size_t power = 1;
		while (power < value && power < (std::numeric_limits<size_t>::max() / 2))
			power *= 2;
		return std::clamp(power, minValue, maxValue);
	}

	size_t randomPowerOfTwo(std::mt19937& rng, size_t minValue, size_t maxValue)
	{
		minValue = std::max<size_t>(1, minValue);
		maxValue = std::max(minValue, maxValue);

		std::vector<size_t> values;
		for (size_t value = clampPowerOfTwo(minValue, minValue, maxValue); value <= maxValue; value *= 2)
		{
			values.push_back(value);
			if (value > maxValue / 2)
				break;
		}

		std::uniform_int_distribution<size_t> distribution(0, values.size() - 1);
		return values[distribution(rng)];
	}

	MultiDataStructure::DataStructureLevel randomStructureType(
		std::mt19937& rng,
		std::optional<MultiDataStructure::DataStructureLevel> previous)
	{
		static const std::array<MultiDataStructure::DataStructureLevel, 3> types = {
			MultiDataStructure::QuadTreeNode,
			MultiDataStructure::OctreeNode,
			MultiDataStructure::KDTreeNode,
		};

		for (;;)
		{
			std::uniform_int_distribution<size_t> distribution(0, types.size() - 1);
			const MultiDataStructure::DataStructureLevel type = types[distribution(rng)];
			if (!previous.has_value() || type != previous.value())
				return type;
		}
	}

	std::string schemaSignature(const SchemaConfig& schema)
	{
		std::ostringstream output;
		for (size_t i = 0; i < schema.levels.size(); ++i)
		{
			const SchemaLevelConfig& level = schema.levels[i];
			if (i > 0)
				output << '_';
			output
				<< schemaTypeShortName(level.type)
				<< level.numLevels
				<< "l"
				<< level.leafCapacity;
		}
		return output.str();
	}

	std::string schemaConfigToJson(const SchemaConfig& schema)
	{
		std::ostringstream output;
		output << "{\n";
		output << "  \"name\": \"" << schema.name << "\",\n";
		output << "  \"levels\": [\n";
		for (size_t i = 0; i < schema.levels.size(); ++i)
		{
			const SchemaLevelConfig& level = schema.levels[i];
			output << "    {\n";
			output << "      \"type\": \"" << Config::dataStructureLevelName(level.type) << "\",\n";
			output << "      \"numLevels\": " << level.numLevels << ",\n";
			output << "      \"leafCapacity\": " << level.leafCapacity << ",\n";
			output << "      \"minPointsToSplit\": " << level.minPrimitivesToSplit;
			if (!level.axisPolicy.empty())
				output << ",\n      \"axisPolicy\": \"" << level.axisPolicy << "\"\n";
			else
				output << '\n';
			output << "    }" << (i + 1 < schema.levels.size() ? "," : "") << "\n";
		}
		output << "  ],\n";
		output << "  \"buildPolicy\": {\n";
		output << "    \"maxDepth\": " << schema.buildPolicy.maxDepth << ",\n";
		output << "    \"leafCapacity\": " << schema.buildPolicy.leafCapacity << ",\n";
		output << "    \"minPointsToSplit\": " << schema.buildPolicy.minPrimitivesToSplit << ",\n";
		output << "    \"collapseSingleChild\": " << (schema.buildPolicy.collapseSingleChild ? "true" : "false") << ",\n";
		output << "    \"removeEmptyNodes\": " << (schema.buildPolicy.removeEmptyNodes ? "true" : "false") << ",\n";
		output << "    \"allowOverlapDuplication\": " << (schema.buildPolicy.allowOverlapDuplication ? "true" : "false") << "\n";
		output << "  }\n";
		output << "}\n";
		return output.str();
	}

	std::vector<SearchDataset> makeSyntheticDatasets(size_t scale)
	{
		const size_t n = std::max<size_t>(64, scale);
		std::vector<SearchDataset> datasets;
		datasets.reserve(3);

		SearchDataset flat;
		flat.name = "synthetic_flat_terrain";
		flat.source = "synthetic";
		flat.cloud = SyntheticPointClouds::generateFlatTerrain(n, 100.0f, 100.0f, 0.05f, 101);
		datasets.push_back(std::move(flat));

		SearchDataset facade;
		facade.name = "synthetic_facade";
		facade.source = "synthetic";
		facade.cloud = SyntheticPointClouds::generateFacade(n, 60.0f, 40.0f, 0.2f, 202);
		datasets.push_back(std::move(facade));

		SearchDataset urban;
		urban.name = "synthetic_urban_mixed";
		urban.source = "synthetic";
		urban.cloud = SyntheticPointClouds::generateUrbanMixed(n, n, 5, 303);
		datasets.push_back(std::move(urban));

		return datasets;
	}

	std::vector<SearchDataset> loadDatasets(const Experiments::SchemaSearchOptions& options)
	{
		std::vector<SearchDataset> datasets;

		if (options.includeSyntheticDatasets)
		{
			std::vector<SearchDataset> synthetic = makeSyntheticDatasets(options.syntheticScale);
			datasets.insert(datasets.end(), std::make_move_iterator(synthetic.begin()), std::make_move_iterator(synthetic.end()));
		}

		for (const std::string& inputPath : options.inputPaths)
		{
			SearchDataset dataset;
			dataset.name = datasetNameFromPath(inputPath);
			dataset.source = inputPath;
			dataset.cloud = PointCloud::load(inputPath, { options.useBinaryCache, options.rebuildBinaryCache });
			if (dataset.cloud.empty())
				throw std::runtime_error("Point cloud is empty: " + inputPath);
			datasets.push_back(std::move(dataset));
		}

		if (datasets.empty())
			throw std::runtime_error("Schema search requires at least one synthetic dataset or --input path");

		return datasets;
	}

	std::vector<Experiments::SchemaCandidate> loadSchemas(const std::vector<std::string>& configuredPaths, bool includeConfiguredSchemas)
	{
		if (!includeConfiguredSchemas)
			return {};

		const std::vector<std::string>& paths = configuredPaths.empty() ? defaultSchemaPaths() : configuredPaths;
		std::vector<Experiments::SchemaCandidate> schemas;
		schemas.reserve(paths.size());

		for (const std::string& schemaPath : paths)
		{
			Experiments::SchemaCandidate loaded;
			loaded.path = schemaPath;
			loaded.config = Config::loadSchemaConfig(schemaPath);
			loaded.name = loaded.config.name;
			schemas.push_back(std::move(loaded));
		}

		return schemas;
	}

	void appendGeneratedSchemas(std::vector<Experiments::SchemaCandidate>& schemas, const Experiments::SchemaGenerationOptions& options)
	{
		if (options.count == 0)
			return;

		std::vector<Experiments::SchemaCandidate> generated = Experiments::generateSchemaCandidates(options);
		schemas.insert(schemas.end(), std::make_move_iterator(generated.begin()), std::make_move_iterator(generated.end()));
	}

	std::string candidateKey(const std::string& name, const std::string& path)
	{
		return name + "\n" + path;
	}

	std::vector<Experiments::SchemaCandidate> selectBenchmarkSchemas(
		const Experiments::SchemaSearchOptions& options,
		const SearchDataset& dataset,
		const Experiments::WorkloadProfile& workload,
		const std::vector<Experiments::SchemaCandidate>& schemas,
		const std::optional<Experiments::SchemaSelectorModel>& rankModel)
	{
		if (schemas.empty())
			return {};

		if (options.benchmarkTopK == 0 || options.benchmarkTopK >= schemas.size())
			return schemas;

		if (!rankModel.has_value())
			return std::vector<Experiments::SchemaCandidate>(schemas.begin(), schemas.begin() + options.benchmarkTopK);

		const std::vector<Experiments::CandidatePrediction> predictions = Experiments::scoreSchemaCandidates(
			rankModel.value(),
			workload,
			dataset.cloud,
			schemas);

		std::unordered_map<std::string, const Experiments::SchemaCandidate*> byKey;
		byKey.reserve(schemas.size());
		for (const Experiments::SchemaCandidate& schema : schemas)
			byKey[candidateKey(schema.config.name, schema.path)] = &schema;

		std::vector<Experiments::SchemaCandidate> selected;
		selected.reserve(std::min(options.benchmarkTopK, predictions.size()));
		for (const Experiments::CandidatePrediction& prediction : predictions)
		{
			const auto found = byKey.find(candidateKey(prediction.schemaName, prediction.schemaPath));
			if (found == byKey.end())
				continue;

			selected.push_back(*found->second);
			if (selected.size() >= options.benchmarkTopK)
				break;
		}

		return selected;
	}

	std::vector<Experiments::WorkloadProfile> loadWorkloads(const Experiments::SchemaSearchOptions& options)
	{
		const std::vector<std::string>& paths = options.workloadPaths.empty() ? defaultWorkloadPaths() : options.workloadPaths;
		std::vector<Experiments::WorkloadProfile> workloads;
		workloads.reserve(paths.size());

		for (const std::string& workloadPath : paths)
		{
			Experiments::WorkloadProfile profile = Experiments::loadWorkloadProfile(workloadPath);
			if (options.queryCountOverride > 0)
				profile.numQueries = options.queryCountOverride;
			if (options.knnKOverride > 0)
				profile.knnK = options.knnKOverride;
			if (options.querySeedOverride)
				profile.querySeed = options.querySeed;
			workloads.push_back(std::move(profile));
		}

		return workloads;
	}

	std::vector<double> queryTypeWeights(const Experiments::WorkloadProfile& profile)
	{
		std::vector<double> weights = {
			std::max(0.0, profile.rangeWeight),
			std::max(0.0, profile.radiusWeight),
			std::max(0.0, profile.knnWeight),
		};

		if (weights[0] == 0.0 && weights[1] == 0.0 && weights[2] == 0.0)
			weights = { 1.0, 1.0, 1.0 };

		return weights;
	}

	WorkloadRun runWorkloadProfile(const Experiments::WorkloadProfile& profile, const PointCloud& cloud, const PointSpatialIndex& index)
	{
		WorkloadRun result;
		if (profile.numQueries == 0)
			return result;

		std::vector<PointSpatialIndex::QueryStats> samples;
		samples.reserve(profile.numQueries);

		std::mt19937 rng(profile.querySeed);
		const std::vector<double> weights = queryTypeWeights(profile);
		std::discrete_distribution<size_t> queryType(weights.begin(), weights.end());

		for (size_t i = 0; i < profile.numQueries; ++i)
		{
			const size_t type = queryType(rng);
			if (type == 0)
			{
				samples.push_back(index.rangeQuery(randomQueryBox(rng, cloud)).stats);
				++result.rangeQueries;
				continue;
			}

			if (type == 1)
			{
				const glm::vec3 center = randomPointInBounds(rng, cloud.bounds());
				samples.push_back(index.radiusQuery(center, randomQueryRadius(rng, cloud)).stats);
				++result.radiusQueries;
				continue;
			}

			const glm::vec3 center = randomPointInBounds(rng, cloud.bounds());
			samples.push_back(index.knnQuery(center, profile.knnK).stats);
			++result.knnQueries;
		}

		result.metrics = Experiments::summarizeQueryStats(samples);
		return result;
	}

	std::string csvEscape(const std::string& value)
	{
		if (value.find_first_of(",\"\n\r") == std::string::npos)
			return value;

		std::string escaped = "\"";
		for (const char c : value)
		{
			if (c == '"')
				escaped += "\"\"";
			else
				escaped += c;
		}
		escaped += '"';
		return escaped;
	}

	void createParentDirectory(const std::string& filename)
	{
		const std::filesystem::path path(filename);
		if (path.has_parent_path())
			std::filesystem::create_directories(path.parent_path());
	}

	void writeSearchHeader(std::ostream& output)
	{
		output
			<< "dataset_name,dataset_source,num_points,workload_name,range_weight,radius_weight,knn_weight,num_queries,knn_k,query_seed,"
			<< "feature_sample_size,bbox_x,bbox_y,bbox_z,aspect_xy,aspect_xz,aspect_yz,density_bbox,height_mean,height_std,height_range,"
			<< "cov_eig_0,cov_eig_1,cov_eig_2,linearity,planarity,scattering,occupancy_ratio_8,occupancy_entropy_8,density_cv_8,verticality_score,flatness_score,"
			<< "w_range,w_radius,w_knn,query_scale_mean,query_scale_std,build_weight,memory_weight,"
			<< "schema_name,schema_path,build_time_ms,num_nodes,num_leaves,max_depth,avg_leaf_occupancy,max_leaf_occupancy,memory_estimate_bytes,"
			<< "total_queries,avg_latency_ms,median_latency_ms,p95_latency_ms,throughput_qps,avg_visited_nodes,avg_tested_points,avg_returned_points,"
			<< "range_queries,radius_queries,knn_queries,score,score_memory_mb,score_imbalance_penalty,lambda_build,lambda_memory,lambda_imbalance\n";
	}

	void writeSearchRows(const std::string& csvPath, const std::vector<Experiments::SchemaSearchRecord>& records)
	{
		if (csvPath.empty())
			return;

		createParentDirectory(csvPath);
		std::ofstream output(csvPath);
		if (!output.is_open())
			throw std::runtime_error("Unable to open schema-search CSV path: " + csvPath);

		writeSearchHeader(output);
		output << std::fixed << std::setprecision(6);
		for (const Experiments::SchemaSearchRecord& record : records)
		{
			output
				<< csvEscape(record.datasetName) << ','
				<< csvEscape(record.datasetSource) << ','
				<< record.numPoints << ','
				<< csvEscape(record.workloadName) << ','
				<< record.rangeWeight << ','
				<< record.radiusWeight << ','
				<< record.knnWeight << ','
				<< record.numQueries << ','
				<< record.knnK << ','
				<< record.querySeed << ','
				<< record.pointFeatures.sampleSize << ','
				<< record.pointFeatures.bboxX << ','
				<< record.pointFeatures.bboxY << ','
				<< record.pointFeatures.bboxZ << ','
				<< record.pointFeatures.aspectXY << ','
				<< record.pointFeatures.aspectXZ << ','
				<< record.pointFeatures.aspectYZ << ','
				<< record.pointFeatures.densityBbox << ','
				<< record.pointFeatures.heightMean << ','
				<< record.pointFeatures.heightStd << ','
				<< record.pointFeatures.heightRange << ','
				<< record.pointFeatures.covEig0 << ','
				<< record.pointFeatures.covEig1 << ','
				<< record.pointFeatures.covEig2 << ','
				<< record.pointFeatures.linearity << ','
				<< record.pointFeatures.planarity << ','
				<< record.pointFeatures.scattering << ','
				<< record.pointFeatures.occupancyRatio8 << ','
				<< record.pointFeatures.occupancyEntropy8 << ','
				<< record.pointFeatures.densityCv8 << ','
				<< record.pointFeatures.verticalityScore << ','
				<< record.pointFeatures.flatnessScore << ','
				<< record.workloadFeatures.wRange << ','
				<< record.workloadFeatures.wRadius << ','
				<< record.workloadFeatures.wKnn << ','
				<< record.workloadFeatures.queryScaleMean << ','
				<< record.workloadFeatures.queryScaleStd << ','
				<< record.workloadFeatures.buildWeight << ','
				<< record.workloadFeatures.memoryWeight << ','
				<< csvEscape(record.schemaName) << ','
				<< csvEscape(record.schemaPath) << ','
				<< record.buildMetrics.buildTimeMs << ','
				<< record.buildMetrics.numNodes << ','
				<< record.buildMetrics.numLeaves << ','
				<< record.buildMetrics.maxDepth << ','
				<< record.buildMetrics.averageLeafOccupancy << ','
				<< record.buildMetrics.maxLeafOccupancy << ','
				<< record.buildMetrics.memoryEstimateBytes << ','
				<< record.queryMetrics.totalQueries << ','
				<< record.queryMetrics.averageLatencyMs << ','
				<< record.queryMetrics.medianLatencyMs << ','
				<< record.queryMetrics.p95LatencyMs << ','
				<< record.queryMetrics.throughputQueriesPerSecond << ','
				<< record.queryMetrics.averageVisitedNodes << ','
				<< record.queryMetrics.averageTestedPoints << ','
				<< record.queryMetrics.averageReturnedPoints << ','
				<< record.rangeQueries << ','
				<< record.radiusQueries << ','
				<< record.knnQueries << ','
				<< record.score << ','
				<< record.scoreMemoryMb << ','
				<< record.scoreImbalancePenalty << ','
				<< record.weights.lambdaBuild << ','
				<< record.weights.lambdaMemory << ','
				<< record.weights.lambdaImbalance << '\n';
		}
	}

	size_t countCandidatesForBest(const std::vector<Experiments::SchemaSearchRecord>& records, const Experiments::SchemaSearchRecord& best)
	{
		size_t count = 0;
		for (const Experiments::SchemaSearchRecord& record : records)
		{
			if (record.datasetName == best.datasetName && record.workloadName == best.workloadName)
				++count;
		}
		return count;
	}

	void writeBestRows(const std::string& csvPath, const std::vector<Experiments::SchemaSearchRecord>& records)
	{
		if (csvPath.empty())
			return;

		const std::vector<Experiments::SchemaSearchRecord> bestRecords = Experiments::selectBestRecords(records);
		createParentDirectory(csvPath);
		std::ofstream output(csvPath);
		if (!output.is_open())
			throw std::runtime_error("Unable to open schema-search best CSV path: " + csvPath);

		output
			<< "dataset_name,workload_name,num_points,feature_sample_size,bbox_x,bbox_y,bbox_z,aspect_xy,aspect_xz,aspect_yz,density_bbox,"
			<< "height_mean,height_std,height_range,cov_eig_0,cov_eig_1,cov_eig_2,linearity,planarity,scattering,occupancy_ratio_8,"
			<< "occupancy_entropy_8,density_cv_8,verticality_score,flatness_score,w_range,w_radius,w_knn,knn_k,num_queries,"
			<< "query_scale_mean,query_scale_std,build_weight,memory_weight,best_schema_name,best_schema_path,best_score,"
			<< "best_avg_latency_ms,best_build_time_ms,best_memory_estimate_bytes,num_candidates\n";
		output << std::fixed << std::setprecision(6);
		for (const Experiments::SchemaSearchRecord& record : bestRecords)
		{
			output
				<< csvEscape(record.datasetName) << ','
				<< csvEscape(record.workloadName) << ','
				<< record.numPoints << ','
				<< record.pointFeatures.sampleSize << ','
				<< record.pointFeatures.bboxX << ','
				<< record.pointFeatures.bboxY << ','
				<< record.pointFeatures.bboxZ << ','
				<< record.pointFeatures.aspectXY << ','
				<< record.pointFeatures.aspectXZ << ','
				<< record.pointFeatures.aspectYZ << ','
				<< record.pointFeatures.densityBbox << ','
				<< record.pointFeatures.heightMean << ','
				<< record.pointFeatures.heightStd << ','
				<< record.pointFeatures.heightRange << ','
				<< record.pointFeatures.covEig0 << ','
				<< record.pointFeatures.covEig1 << ','
				<< record.pointFeatures.covEig2 << ','
				<< record.pointFeatures.linearity << ','
				<< record.pointFeatures.planarity << ','
				<< record.pointFeatures.scattering << ','
				<< record.pointFeatures.occupancyRatio8 << ','
				<< record.pointFeatures.occupancyEntropy8 << ','
				<< record.pointFeatures.densityCv8 << ','
				<< record.pointFeatures.verticalityScore << ','
				<< record.pointFeatures.flatnessScore << ','
				<< record.workloadFeatures.wRange << ','
				<< record.workloadFeatures.wRadius << ','
				<< record.workloadFeatures.wKnn << ','
				<< record.workloadFeatures.knnK << ','
				<< record.workloadFeatures.numQueries << ','
				<< record.workloadFeatures.queryScaleMean << ','
				<< record.workloadFeatures.queryScaleStd << ','
				<< record.workloadFeatures.buildWeight << ','
				<< record.workloadFeatures.memoryWeight << ','
				<< csvEscape(record.schemaName) << ','
				<< csvEscape(record.schemaPath) << ','
				<< record.score << ','
				<< record.queryMetrics.averageLatencyMs << ','
				<< record.buildMetrics.buildTimeMs << ','
				<< record.buildMetrics.memoryEstimateBytes << ','
				<< countCandidatesForBest(records, record) << '\n';
		}
	}
}

std::vector<Experiments::SchemaCandidate> Experiments::generateSchemaCandidates(const SchemaGenerationOptions& options)
{
	if (options.count == 0)
		return {};

	const size_t maxDepth = std::max<size_t>(1, options.maxDepth);
	const size_t maxBlocks = std::min(std::max<size_t>(1, options.maxBlocks), maxDepth);
	const size_t minLeaf = std::max<size_t>(1, std::min(options.minLeafCapacity, options.maxLeafCapacity));
	const size_t maxLeaf = std::max(minLeaf, options.maxLeafCapacity);

	std::mt19937 rng(options.seed);
	std::unordered_set<std::string> seen;
	std::vector<SchemaCandidate> candidates;
	candidates.reserve(options.count);

	size_t attempts = 0;
	const size_t maxAttempts = std::max<size_t>(options.count * 50, 1024);
	while (candidates.size() < options.count && attempts++ < maxAttempts)
	{
		std::uniform_int_distribution<size_t> blockDistribution(1, maxBlocks);
		const size_t numBlocks = blockDistribution(rng);

		SchemaConfig schema;
		schema.buildPolicy.maxDepth = maxDepth;
		schema.buildPolicy.leafCapacity = randomPowerOfTwo(rng, minLeaf, maxLeaf);
		schema.buildPolicy.minPrimitivesToSplit = std::max<size_t>(2, schema.buildPolicy.leafCapacity / 4);
		schema.buildPolicy.collapseSingleChild = true;
		schema.buildPolicy.removeEmptyNodes = true;
		schema.buildPolicy.allowOverlapDuplication = false;

		size_t remainingDepth = maxDepth;
		std::optional<MultiDataStructure::DataStructureLevel> previousType;
		for (size_t block = 0; block < numBlocks; ++block)
		{
			const size_t remainingBlocks = numBlocks - block - 1;
			const size_t maxLevelForBlock = remainingDepth - remainingBlocks;
			std::uniform_int_distribution<size_t> levelDistribution(1, maxLevelForBlock);

			SchemaLevelConfig level;
			level.type = randomStructureType(rng, previousType);
			level.typeName = Config::dataStructureLevelName(level.type);
			level.numLevels = levelDistribution(rng);
			level.leafCapacity = randomPowerOfTwo(rng, minLeaf, maxLeaf);
			level.minPrimitivesToSplit = std::max<size_t>(2, level.leafCapacity / 4);
			if (level.type == MultiDataStructure::KDTreeNode)
				level.axisPolicy = "median_longest_axis";

			schema.levels.push_back(level);
			previousType = level.type;
			remainingDepth -= level.numLevels;
		}

		if (schema.levels.empty())
			continue;

		schema.buildPolicy.maxDepth = std::min(maxDepth, schema.totalLevels());
		const std::string signature = schemaSignature(schema);
		if (!seen.insert(signature).second)
			continue;

		schema.name = "generated_" + signature;

		SchemaCandidate candidate;
		candidate.name = schema.name;
		if (!options.outputDirectory.empty())
		{
			const std::filesystem::path schemaPath = std::filesystem::path(options.outputDirectory) / (schema.name + ".json");
			if (schemaPath.has_parent_path())
				std::filesystem::create_directories(schemaPath.parent_path());

			std::ofstream output(schemaPath);
			if (!output.is_open())
				throw std::runtime_error("Unable to write generated schema: " + schemaPath.string());
			output << schemaConfigToJson(schema);
			candidate.path = schemaPath.string();
		}
		else
		{
			candidate.path = "generated:" + signature;
		}
		candidate.config = schema;
		candidate.generated = true;
		candidates.push_back(std::move(candidate));
	}

	if (candidates.size() < options.count)
		std::cerr << "Warning: generated " << candidates.size() << " unique schemas from requested " << options.count << '\n';

	return candidates;
}

Experiments::WorkloadProfile Experiments::parseWorkloadProfile(const std::string& jsonText, const std::string& sourceName)
{
	boost::system::error_code error;
	boost::json::value rootValue = boost::json::parse(jsonText, error);
	if (error)
		throw std::runtime_error("Invalid workload JSON" + (sourceName.empty() ? std::string() : " in " + sourceName) + ": " + error.message());

	if (!rootValue.is_object())
		throw std::runtime_error("Workload JSON root must be an object");

	const boost::json::object& root = rootValue.as_object();

	WorkloadProfile profile;
	profile.name = asString(root, "name", profile.name);
	profile.numQueries = asSize(root, "numQueries", profile.numQueries);
	profile.knnK = asSize(root, "knnK", profile.knnK);
	profile.querySeed = static_cast<uint32_t>(asSize(root, "querySeed", profile.querySeed));

	if (const boost::json::value* queries = root.if_contains("queries"))
	{
		if (!queries->is_object())
			throw std::runtime_error("Workload queries must be an object");

		const boost::json::object& queryWeights = queries->as_object();
		profile.rangeWeight = asDouble(queryWeights, "aabb_range", profile.rangeWeight);
		profile.radiusWeight = asDouble(queryWeights, "radius", profile.radiusWeight);
		profile.knnWeight = asDouble(queryWeights, "knn", profile.knnWeight);
	}

	if (profile.name.empty())
		throw std::runtime_error("Workload profile requires a non-empty name");

	return profile;
}

Experiments::WorkloadProfile Experiments::loadWorkloadProfile(const std::string& filename)
{
	const std::filesystem::path resolvedPath = resolveExistingPath(filename);
	std::ifstream file(resolvedPath);
	if (!file.is_open())
		throw std::runtime_error("Unable to open workload profile: " + filename);

	std::stringstream buffer;
	buffer << file.rdbuf();
	return parseWorkloadProfile(buffer.str(), resolvedPath.string());
}

double Experiments::computeSchemaSearchScore(
	const BuildMetrics& buildMetrics,
	const QueryMetrics& queryMetrics,
	const ScoreWeights& weights,
	double& memoryMb,
	double& imbalancePenalty)
{
	memoryMb = static_cast<double>(buildMetrics.memoryEstimateBytes) / (1024.0 * 1024.0);
	imbalancePenalty = buildMetrics.averageLeafOccupancy > 0.0
		? static_cast<double>(buildMetrics.maxLeafOccupancy) / buildMetrics.averageLeafOccupancy
		: 0.0;

	return queryMetrics.averageLatencyMs +
		weights.lambdaBuild * buildMetrics.buildTimeMs +
		weights.lambdaMemory * memoryMb +
		weights.lambdaImbalance * imbalancePenalty;
}

std::vector<Experiments::SchemaSearchRecord> Experiments::selectBestRecords(const std::vector<SchemaSearchRecord>& records)
{
	std::vector<SchemaSearchRecord> best;
	for (const SchemaSearchRecord& record : records)
	{
		auto existing = std::find_if(best.begin(), best.end(), [&record](const SchemaSearchRecord& candidate) {
			return candidate.datasetName == record.datasetName && candidate.workloadName == record.workloadName;
		});

		if (existing == best.end())
		{
			best.push_back(record);
			continue;
		}

		if (record.score < existing->score)
			*existing = record;
	}

	return best;
}

int Experiments::runSchemaSearch(const SchemaSearchOptions& options)
{
	const std::vector<SearchDataset> datasets = loadDatasets(options);
	std::vector<SchemaCandidate> schemas = loadSchemas(options.schemaPaths, options.includeConfiguredSchemas);
	appendGeneratedSchemas(schemas, options.generation);
	if (schemas.empty())
		throw std::runtime_error("Schema search has no schemas. Provide --schemas, omit --generated-only, or use --generate-schemas.");
	const std::vector<WorkloadProfile> workloads = loadWorkloads(options);
	const std::optional<SchemaSelectorModel> rankModel = options.rankModelPath.empty()
		? std::optional<SchemaSelectorModel>()
		: std::optional<SchemaSelectorModel>(loadSchemaSelectorModel(options.rankModelPath));

	std::vector<SchemaSearchRecord> records;
	records.reserve(datasets.size() * workloads.size() * schemas.size());

	std::cout << std::fixed << std::setprecision(3);
	std::cout << "Schema search\n";
	std::cout << "  datasets: " << datasets.size() << '\n';
	std::cout << "  schemas: " << schemas.size() << '\n';
	if (options.generation.count > 0)
		std::cout << "  generated schemas: " << options.generation.count << " requested\n";
	if (rankModel.has_value())
	{
		std::cout << "  surrogate rank model: " << options.rankModelPath << '\n';
		if (options.benchmarkTopK > 0)
			std::cout << "  benchmark top-k: " << options.benchmarkTopK << '\n';
	}
	std::cout << "  workloads: " << workloads.size() << '\n';

	for (const SearchDataset& dataset : datasets)
	{
		const PointCloudFeatures pointFeatures = extractPointCloudFeatures(dataset.cloud);
		std::cout << "  dataset: " << dataset.name << " (" << dataset.cloud.size() << " points)\n";

		for (const WorkloadProfile& workload : workloads)
		{
			const WorkloadFeatures workloadFeatures = extractWorkloadFeatures(workload, options.weights);
			const std::vector<SchemaCandidate> benchmarkSchemas = selectBenchmarkSchemas(
				options,
				dataset,
				workload,
				schemas,
				rankModel);

			std::cout << "    workload: " << workload.name << " (" << workload.numQueries << " queries)\n";
			if (benchmarkSchemas.size() != schemas.size())
				std::cout << "      benchmarking " << benchmarkSchemas.size() << " / " << schemas.size() << " schemas after surrogate pruning\n";

			for (const SchemaCandidate& schema : benchmarkSchemas)
			{
				PointSpatialIndex index;
				const auto buildBegin = std::chrono::steady_clock::now();
				index.build(dataset.cloud, schema.config);
				const auto buildEnd = std::chrono::steady_clock::now();

				const BuildMetrics buildMetrics = collectBuildMetrics(index.stats(), index.root(), elapsedMilliseconds(buildBegin, buildEnd));
				const WorkloadRun workloadRun = runWorkloadProfile(workload, dataset.cloud, index);

				SchemaSearchRecord record;
				record.datasetName = dataset.name;
				record.datasetSource = dataset.source;
				record.numPoints = dataset.cloud.size();
				record.workloadName = workload.name;
				record.rangeWeight = workload.rangeWeight;
				record.radiusWeight = workload.radiusWeight;
				record.knnWeight = workload.knnWeight;
				record.numQueries = workload.numQueries;
				record.knnK = workload.knnK;
				record.querySeed = workload.querySeed;
				record.schemaName = schema.config.name;
				record.schemaPath = schema.path;
				record.buildMetrics = buildMetrics;
				record.queryMetrics = workloadRun.metrics;
				record.rangeQueries = workloadRun.rangeQueries;
				record.radiusQueries = workloadRun.radiusQueries;
				record.knnQueries = workloadRun.knnQueries;
				record.weights = options.weights;
				record.score = computeSchemaSearchScore(
					record.buildMetrics,
					record.queryMetrics,
					record.weights,
					record.scoreMemoryMb,
					record.scoreImbalancePenalty);
				record.pointFeatures = pointFeatures;
				record.workloadFeatures = workloadFeatures;
				records.push_back(record);

				std::cout << "      " << record.schemaName
					<< ": score " << record.score
					<< ", avg " << record.queryMetrics.averageLatencyMs
					<< " ms, build " << record.buildMetrics.buildTimeMs
					<< " ms\n";
			}
		}
	}

	writeSearchRows(options.csvPath, records);
	writeBestRows(options.bestCsvPath, records);

	std::cout << "  wrote rows: " << records.size() << '\n';
	if (!options.csvPath.empty())
		std::cout << "  csv: " << options.csvPath << '\n';
	if (!options.bestCsvPath.empty())
		std::cout << "  best csv: " << options.bestCsvPath << '\n';

	if (options.pauseAtEnd)
		std::system("pause");

	return 0;
}

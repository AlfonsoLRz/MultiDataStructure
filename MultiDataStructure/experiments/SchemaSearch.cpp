#include "../stdafx.h"
#include "SchemaSearch.h"

#include "../core/Config.h"
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

	struct LoadedSchema
	{
		std::string path;
		SchemaConfig config;
	};

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

	std::vector<LoadedSchema> loadSchemas(const std::vector<std::string>& configuredPaths)
	{
		const std::vector<std::string>& paths = configuredPaths.empty() ? defaultSchemaPaths() : configuredPaths;
		std::vector<LoadedSchema> schemas;
		schemas.reserve(paths.size());

		for (const std::string& schemaPath : paths)
		{
			LoadedSchema loaded;
			loaded.path = schemaPath;
			loaded.config = Config::loadSchemaConfig(schemaPath);
			schemas.push_back(std::move(loaded));
		}

		return schemas;
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
	const std::vector<LoadedSchema> schemas = loadSchemas(options.schemaPaths);
	const std::vector<WorkloadProfile> workloads = loadWorkloads(options);

	std::vector<SchemaSearchRecord> records;
	records.reserve(datasets.size() * workloads.size() * schemas.size());

	std::cout << std::fixed << std::setprecision(3);
	std::cout << "Schema search\n";
	std::cout << "  datasets: " << datasets.size() << '\n';
	std::cout << "  schemas: " << schemas.size() << '\n';
	std::cout << "  workloads: " << workloads.size() << '\n';

	for (const SearchDataset& dataset : datasets)
	{
		const PointCloudFeatures pointFeatures = extractPointCloudFeatures(dataset.cloud);
		std::cout << "  dataset: " << dataset.name << " (" << dataset.cloud.size() << " points)\n";

		for (const WorkloadProfile& workload : workloads)
		{
			const WorkloadFeatures workloadFeatures = extractWorkloadFeatures(workload, options.weights);
			std::cout << "    workload: " << workload.name << " (" << workload.numQueries << " queries)\n";

			for (const LoadedSchema& schema : schemas)
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

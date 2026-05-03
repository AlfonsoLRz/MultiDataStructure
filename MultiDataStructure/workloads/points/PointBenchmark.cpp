#include "../../stdafx.h"
#include "PointBenchmark.h"

#include "../../core/Config.h"
#include "../../experiments/Metrics.h"
#include "../../experiments/SchemaSelector.h"
#include "PointCloud.h"
#include "PointSpatialIndex.h"

namespace
{
	double elapsedMilliseconds(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
	{
		return std::chrono::duration<double, std::milli>(end - begin).count();
	}

	std::string jsonEscape(const std::string& value)
	{
		std::ostringstream escaped;
		for (const char c : value)
		{
			switch (c)
			{
			case '\\':
				escaped << "\\\\";
				break;
			case '"':
				escaped << "\\\"";
				break;
			case '\n':
				escaped << "\\n";
				break;
			case '\r':
				escaped << "\\r";
				break;
			case '\t':
				escaped << "\\t";
				break;
			default:
				escaped << c;
				break;
			}
		}

		return escaped.str();
	}

	void writeVec3Json(std::ostream& stream, const glm::vec3& value)
	{
		stream << '[' << value.x << ", " << value.y << ", " << value.z << ']';
	}

	struct QueryProfileSection
	{
		std::vector<PointSpatialIndex::QueryStats> samples;
		Experiments::QueryMetrics metrics;

		void add(const PointSpatialIndex::QueryStats& stats)
		{
			samples.push_back(stats);
		}

		void finalize()
		{
			metrics = Experiments::summarizeQueryStats(samples);
		}
	};

	struct QueryProfileSummary
	{
		size_t queryCount = 0;
		size_t queryK = 0;
		uint32_t seed = 0;
		QueryProfileSection range;
		QueryProfileSection countRange;
		QueryProfileSection radius;
		QueryProfileSection knn;
		Experiments::QueryMetrics mixed;

		size_t totalQueries() const
		{
			return mixed.totalQueries;
		}

		void finalize()
		{
			range.finalize();
			countRange.finalize();
			radius.finalize();
			knn.finalize();

			std::vector<PointSpatialIndex::QueryStats> allSamples;
			allSamples.reserve(range.samples.size() + countRange.samples.size() + radius.samples.size() + knn.samples.size());
			allSamples.insert(allSamples.end(), range.samples.begin(), range.samples.end());
			allSamples.insert(allSamples.end(), countRange.samples.begin(), countRange.samples.end());
			allSamples.insert(allSamples.end(), radius.samples.begin(), radius.samples.end());
			allSamples.insert(allSamples.end(), knn.samples.begin(), knn.samples.end());
			mixed = Experiments::summarizeQueryStats(allSamples);
		}
	};

	struct LoadedSchema
	{
		std::string path;
		SchemaConfig config;
		double schemaLoadMs = 0.0;
	};

	struct AutoSelectionLog
	{
		bool enabled = false;
		Experiments::SchemaSelection selection;
	};

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

	QueryProfileSummary runQueryProfile(const PointBenchmark::Options& options, const PointCloud& cloud, const PointSpatialIndex& index)
	{
		QueryProfileSummary summary;
		summary.queryCount = options.queryCount;
		summary.queryK = options.queryK;
		summary.seed = options.querySeed;

		if (options.queryCount == 0)
			return summary;

		std::mt19937 rng(options.querySeed);
		for (size_t i = 0; i < options.queryCount; ++i)
		{
			const AABB rangeBounds = randomQueryBox(rng, cloud);
			summary.range.add(index.rangeQuery(rangeBounds).stats);
			summary.countRange.add(index.countRange(rangeBounds).stats);

			const glm::vec3 radiusCenter = randomPointInBounds(rng, cloud.bounds());
			const float radius = randomQueryRadius(rng, cloud);
			summary.radius.add(index.radiusQuery(radiusCenter, radius).stats);

			const glm::vec3 knnCenter = randomPointInBounds(rng, cloud.bounds());
			summary.knn.add(index.knnQuery(knnCenter, options.queryK).stats);
		}

		summary.finalize();
		return summary;
	}

	std::string sanitizeName(std::string value)
	{
		for (char& c : value)
		{
			const unsigned char byte = static_cast<unsigned char>(c);
			if (!std::isalnum(byte) && c != '-' && c != '_')
				c = '_';
		}

		if (value.empty())
			return "run";
		return value;
	}

	std::string datasetName(const std::string& inputPath)
	{
		const std::filesystem::path path(inputPath);
		const std::string stem = path.stem().string();
		return stem.empty() ? "points" : stem;
	}

	std::string makeRunId(const std::string& inputPath, const std::string& schemaName, size_t sequence)
	{
		const auto now = std::chrono::system_clock::now();
		const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
		return "points_" + sanitizeName(datasetName(inputPath)) + "_" + sanitizeName(schemaName) + "_" + std::to_string(milliseconds) + "_" + std::to_string(sequence);
	}

	std::filesystem::path outputPathForRun(const std::string& configuredOutputPath, const std::string& schemaName, bool multipleSchemas)
	{
		if (configuredOutputPath.empty())
			return {};

		std::filesystem::path outputPath(configuredOutputPath);
		if (!multipleSchemas)
			return outputPath;

		const std::filesystem::path parent = outputPath.parent_path();
		const std::string stem = outputPath.stem().empty() ? "points" : outputPath.stem().string();
		const std::string extension = outputPath.extension().empty() ? ".json" : outputPath.extension().string();
		return parent / (stem + "_" + sanitizeName(schemaName) + extension);
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

	bool isEmptyFile(const std::filesystem::path& path)
	{
		std::error_code error;
		return !std::filesystem::exists(path, error) || std::filesystem::file_size(path, error) == 0;
	}

	void writeQueryMetricsJson(std::ostream& stream, const char* name, const Experiments::QueryMetrics& metrics, bool trailingComma)
	{
		stream << "    \"" << name << "\": {\n";
		stream << "      \"total_queries\": " << metrics.totalQueries << ",\n";
		stream << "      \"total_latency_ms\": " << metrics.totalLatencyMs << ",\n";
		stream << "      \"avg_latency_ms\": " << metrics.averageLatencyMs << ",\n";
		stream << "      \"median_latency_ms\": " << metrics.medianLatencyMs << ",\n";
		stream << "      \"p95_latency_ms\": " << metrics.p95LatencyMs << ",\n";
		stream << "      \"throughput_queries_per_sec\": " << metrics.throughputQueriesPerSecond << ",\n";
		stream << "      \"avg_visited_nodes\": " << metrics.averageVisitedNodes << ",\n";
		stream << "      \"avg_tested_points\": " << metrics.averageTestedPoints << ",\n";
		stream << "      \"avg_returned_points\": " << metrics.averageReturnedPoints << ",\n";
		stream << "      \"total_visited_nodes\": " << metrics.totalVisitedNodes << ",\n";
		stream << "      \"total_tested_points\": " << metrics.totalTestedPoints << ",\n";
		stream << "      \"total_returned_points\": " << metrics.totalReturnedPoints << "\n";
		stream << "    }" << (trailingComma ? "," : "") << "\n";
	}

	void printQueryMetrics(const char* label, const Experiments::QueryMetrics& metrics)
	{
		std::cout << "    " << label << ": "
			<< "avg " << metrics.averageLatencyMs << " ms, "
			<< "p95 " << metrics.p95LatencyMs << " ms, "
			<< "visited " << metrics.averageVisitedNodes << ", "
			<< "tested " << metrics.averageTestedPoints << ", "
			<< "returned " << metrics.averageReturnedPoints << '\n';
	}

	void printBounds(const PointCloud& cloud)
	{
		std::cout << "  bounds min: ["
			<< cloud.bounds().min().x << ", "
			<< cloud.bounds().min().y << ", "
			<< cloud.bounds().min().z << "]\n";
		std::cout << "  bounds max: ["
			<< cloud.bounds().max().x << ", "
			<< cloud.bounds().max().y << ", "
			<< cloud.bounds().max().z << "]\n";
	}

	void writeResults(
		const std::filesystem::path& outputPath,
		const std::string& runId,
		const PointBenchmark::Options& options,
		const std::string& schemaPath,
		const SchemaConfig& schema,
		const PointCloud& cloud,
		const Experiments::BuildMetrics& buildMetrics,
		const QueryProfileSummary& queryProfile,
		double schemaMs,
		double loadMs,
		bool multipleSchemas,
		const AutoSelectionLog& autoSelection)
	{
		if (outputPath.empty())
			return;

		if (outputPath.has_parent_path())
			std::filesystem::create_directories(outputPath.parent_path());

		std::ofstream output(outputPath);
		if (!output.is_open())
			throw std::runtime_error("Unable to open point benchmark output path: " + options.outputPath);

		output << std::fixed << std::setprecision(4);
		output << "{\n";
		output << "  \"run_id\": \"" << jsonEscape(runId) << "\",\n";
		output << "  \"mode\": \"points\",\n";
		output << "  \"dataset\": {\n";
		output << "    \"name\": \"" << jsonEscape(datasetName(options.inputPath)) << "\",\n";
		output << "    \"source\": \"file\",\n";
		output << "    \"path\": \"" << jsonEscape(options.inputPath) << "\",\n";
		output << "    \"num_points\": " << cloud.size() << ",\n";
		output << "    \"approximate_density\": " << cloud.approximateDensity() << ",\n";
		output << "    \"bounds_min\": ";
		writeVec3Json(output, cloud.bounds().min());
		output << ",\n";
		output << "    \"bounds_max\": ";
		writeVec3Json(output, cloud.bounds().max());
		output << ",\n";
		output << "    \"coordinate_range\": ";
		writeVec3Json(output, cloud.coordinateRange());
		output << "\n";
		output << "  },\n";
		output << "  \"cache\": {\n";
		output << "    \"enabled\": " << (options.useBinaryCache ? "true" : "false") << ",\n";
		output << "    \"path\": \"" << jsonEscape(cloud.cachePath()) << "\",\n";
		output << "    \"loaded_from_cache\": " << (cloud.loadedFromCache() ? "true" : "false") << "\n";
		output << "  },\n";
		output << "  \"schema\": {\n";
		output << "    \"path\": \"" << jsonEscape(schemaPath) << "\",\n";
		output << "    \"name\": \"" << jsonEscape(schema.name) << "\",\n";
		output << "    \"total_levels\": " << schema.totalLevels() << "\n";
		output << "  },\n";
		if (autoSelection.enabled)
		{
			output << "  \"schema_selection\": {\n";
			output << "    \"mode\": \"auto\",\n";
			output << "    \"model_path\": \"" << jsonEscape(autoSelection.selection.modelPath) << "\",\n";
			output << "    \"workload_profile_path\": \"" << jsonEscape(autoSelection.selection.workloadProfilePath) << "\",\n";
			output << "    \"selected_schema_name\": \"" << jsonEscape(autoSelection.selection.schemaName) << "\",\n";
			output << "    \"selected_schema_path\": \"" << jsonEscape(autoSelection.selection.schemaPath) << "\",\n";
			output << "    \"score_source\": \"" << (autoSelection.selection.measuredScore ? "measured" : "predicted") << "\",\n";
			output << "    \"predicted_score\": " << autoSelection.selection.predictedScore << ",\n";
			output << "    \"candidates\": [\n";
			for (size_t i = 0; i < autoSelection.selection.candidates.size(); ++i)
			{
				const Experiments::CandidatePrediction& candidate = autoSelection.selection.candidates[i];
				output << "      {\"schema_name\": \"" << jsonEscape(candidate.schemaName)
					<< "\", \"schema_path\": \"" << jsonEscape(candidate.schemaPath)
					<< "\", \"predicted_score\": " << candidate.predictedScore << "}";
				output << (i + 1 < autoSelection.selection.candidates.size() ? "," : "") << "\n";
			}
			output << "    ]\n";
			output << "  },\n";
		}
		output << "  \"workload\": {\n";
		output << "    \"name\": \"generated_mixed\",\n";
		output << "    \"queries_per_type\": " << queryProfile.queryCount << ",\n";
		output << "    \"total_queries\": " << queryProfile.totalQueries() << ",\n";
		output << "    \"knn_k\": " << queryProfile.queryK << ",\n";
		output << "    \"seed\": " << queryProfile.seed << "\n";
		output << "  },\n";
		output << "  \"timings\": {\n";
		output << "    \"schema_load_time_ms\": " << schemaMs << ",\n";
		output << "    \"point_load_time_ms\": " << loadMs << "\n";
		output << "  },\n";
		output << "  \"build_metrics\": {\n";
		output << "    \"build_time_ms\": " << buildMetrics.buildTimeMs << ",\n";
		output << "    \"num_nodes\": " << buildMetrics.numNodes << ",\n";
		output << "    \"num_leaves\": " << buildMetrics.numLeaves << ",\n";
		output << "    \"indexed_points\": " << buildMetrics.indexedPoints << ",\n";
		output << "    \"max_depth\": " << buildMetrics.maxDepth << ",\n";
		output << "    \"avg_leaf_occupancy\": " << buildMetrics.averageLeafOccupancy << ",\n";
		output << "    \"max_leaf_occupancy\": " << buildMetrics.maxLeafOccupancy << ",\n";
		output << "    \"memory_estimate_bytes\": " << buildMetrics.memoryEstimateBytes << "\n";
		output << "  },\n";
		output << "  \"query_metrics\": {\n";
		writeQueryMetricsJson(output, "mixed", queryProfile.mixed, true);
		writeQueryMetricsJson(output, "range", queryProfile.range.metrics, true);
		writeQueryMetricsJson(output, "count_range", queryProfile.countRange.metrics, true);
		writeQueryMetricsJson(output, "radius", queryProfile.radius.metrics, true);
		writeQueryMetricsJson(output, "knn", queryProfile.knn.metrics, false);
		output << "\n";
		output << "  },\n";
		output << "  \"outputs\": {\n";
		output << "    \"json_path\": \"" << jsonEscape(outputPath.string()) << "\",\n";
		output << "    \"csv_path\": \"" << jsonEscape(options.csvPath) << "\",\n";
		output << "    \"multiple_schemas\": " << (multipleSchemas ? "true" : "false") << "\n";
		output << "  }\n";
		output << "}\n";
	}

	void writeCsvHeader(std::ostream& output)
	{
		output
			<< "run_id,dataset_name,dataset_path,num_points,schema_name,schema_path,workload_name,queries_per_type,total_queries,knn_k,query_seed,"
			<< "schema_load_time_ms,point_load_time_ms,build_time_ms,num_nodes,num_leaves,max_depth,avg_leaf_occupancy,max_leaf_occupancy,memory_estimate_bytes,"
			<< "mixed_avg_latency_ms,mixed_median_latency_ms,mixed_p95_latency_ms,mixed_throughput_qps,mixed_avg_visited_nodes,mixed_avg_tested_points,mixed_avg_returned_points,"
			<< "range_avg_latency_ms,range_p95_latency_ms,count_range_avg_latency_ms,count_range_p95_latency_ms,radius_avg_latency_ms,radius_p95_latency_ms,knn_avg_latency_ms,knn_p95_latency_ms\n";
	}

	void appendCsvSummary(
		const std::string& csvPath,
		const std::string& runId,
		const PointBenchmark::Options& options,
		const std::string& schemaPath,
		const SchemaConfig& schema,
		const PointCloud& cloud,
		const Experiments::BuildMetrics& buildMetrics,
		const QueryProfileSummary& queryProfile,
		double schemaMs,
		double loadMs)
	{
		if (csvPath.empty())
			return;

		const std::filesystem::path path(csvPath);
		if (path.has_parent_path())
			std::filesystem::create_directories(path.parent_path());

		const bool writeHeader = isEmptyFile(path);
		std::ofstream output(path, std::ios::app);
		if (!output.is_open())
			throw std::runtime_error("Unable to open point benchmark CSV path: " + csvPath);

		if (writeHeader)
			writeCsvHeader(output);

		output << std::fixed << std::setprecision(6)
			<< csvEscape(runId) << ','
			<< csvEscape(datasetName(options.inputPath)) << ','
			<< csvEscape(options.inputPath) << ','
			<< cloud.size() << ','
			<< csvEscape(schema.name) << ','
			<< csvEscape(schemaPath) << ','
			<< "generated_mixed" << ','
			<< queryProfile.queryCount << ','
			<< queryProfile.totalQueries() << ','
			<< queryProfile.queryK << ','
			<< queryProfile.seed << ','
			<< schemaMs << ','
			<< loadMs << ','
			<< buildMetrics.buildTimeMs << ','
			<< buildMetrics.numNodes << ','
			<< buildMetrics.numLeaves << ','
			<< buildMetrics.maxDepth << ','
			<< buildMetrics.averageLeafOccupancy << ','
			<< buildMetrics.maxLeafOccupancy << ','
			<< buildMetrics.memoryEstimateBytes << ','
			<< queryProfile.mixed.averageLatencyMs << ','
			<< queryProfile.mixed.medianLatencyMs << ','
			<< queryProfile.mixed.p95LatencyMs << ','
			<< queryProfile.mixed.throughputQueriesPerSecond << ','
			<< queryProfile.mixed.averageVisitedNodes << ','
			<< queryProfile.mixed.averageTestedPoints << ','
			<< queryProfile.mixed.averageReturnedPoints << ','
			<< queryProfile.range.metrics.averageLatencyMs << ','
			<< queryProfile.range.metrics.p95LatencyMs << ','
			<< queryProfile.countRange.metrics.averageLatencyMs << ','
			<< queryProfile.countRange.metrics.p95LatencyMs << ','
			<< queryProfile.radius.metrics.averageLatencyMs << ','
			<< queryProfile.radius.metrics.p95LatencyMs << ','
			<< queryProfile.knn.metrics.averageLatencyMs << ','
			<< queryProfile.knn.metrics.p95LatencyMs << '\n';
	}
}

int PointBenchmark::run(const Options& options)
{
	if (options.inputPath.empty())
		throw std::invalid_argument("Point mode requires --input <file.las|file.ply>");

	const auto loadBegin = std::chrono::steady_clock::now();
	const PointCloud cloud = PointCloud::load(options.inputPath, { options.useBinaryCache, options.rebuildBinaryCache });
	const auto loadEnd = std::chrono::steady_clock::now();
	if (cloud.empty())
		throw std::runtime_error("Point cloud is empty: " + options.inputPath);

	const double loadMs = elapsedMilliseconds(loadBegin, loadEnd);
	AutoSelectionLog autoSelection;
	std::vector<std::string> schemaPaths = options.schemaPaths;
	if (schemaPaths.empty())
	{
		if (options.schemaPath == "auto")
		{
			if (options.modelPath.empty())
				throw std::invalid_argument("--schema auto requires --model <schema_selector.json>");
			if (options.workloadProfilePath.empty())
				throw std::invalid_argument("--schema auto requires --workload-profile <profile.json>");

			autoSelection.enabled = true;
			autoSelection.selection = Experiments::selectSchemaForCloud(options.modelPath, options.workloadProfilePath, cloud);
			schemaPaths.push_back(autoSelection.selection.schemaPath);
		}
		else
		{
			schemaPaths.push_back(options.schemaPath);
		}
	}

	std::vector<LoadedSchema> schemas;
	schemas.reserve(schemaPaths.size());
	for (const std::string& schemaPath : schemaPaths)
	{
		const auto schemaBegin = std::chrono::steady_clock::now();
		LoadedSchema loaded;
		loaded.path = schemaPath;
		loaded.config = Config::loadSchemaConfig(schemaPath);
		const auto schemaEnd = std::chrono::steady_clock::now();
		loaded.schemaLoadMs = elapsedMilliseconds(schemaBegin, schemaEnd);
		schemas.push_back(std::move(loaded));
	}

	const bool multipleSchemas = schemas.size() > 1;

	std::cout << std::fixed << std::setprecision(3);
	std::cout << "Point cloud benchmark\n";
	std::cout << "  input: " << options.inputPath << '\n';
	if (options.useBinaryCache)
		std::cout << "  cache: " << (cloud.loadedFromCache() ? "loaded " : "written ") << cloud.cachePath() << '\n';
	else
		std::cout << "  cache: disabled\n";
	std::cout << "  points: " << cloud.size() << '\n';
	printBounds(cloud);
	std::cout << "  point load: " << loadMs << " ms\n";
	if (autoSelection.enabled)
	{
		std::cout << "  schema selection: auto\n";
		std::cout << "    model: " << autoSelection.selection.modelPath << '\n';
		std::cout << "    workload profile: " << autoSelection.selection.workloadProfilePath << '\n';
		std::cout << "    selected: " << autoSelection.selection.schemaName
			<< " (" << autoSelection.selection.schemaPath << ")"
			<< (autoSelection.selection.measuredScore ? " measured score " : " predicted score ")
			<< autoSelection.selection.predictedScore << '\n';
		for (const Experiments::CandidatePrediction& candidate : autoSelection.selection.candidates)
		{
			std::cout << "      candidate: " << candidate.schemaName
				<< " score " << candidate.predictedScore << '\n';
		}
	}

	for (size_t schemaIndex = 0; schemaIndex < schemas.size(); ++schemaIndex)
	{
		const LoadedSchema& loadedSchema = schemas[schemaIndex];
		const SchemaConfig& schema = loadedSchema.config;

		PointSpatialIndex index;
		const auto buildBegin = std::chrono::steady_clock::now();
		index.build(cloud, schema);
		const auto buildEnd = std::chrono::steady_clock::now();
		const double buildMs = elapsedMilliseconds(buildBegin, buildEnd);
		const PointSpatialIndex::Stats indexStats = index.stats();
		const Experiments::BuildMetrics buildMetrics = Experiments::collectBuildMetrics(indexStats, index.root(), buildMs);
		const QueryProfileSummary queryProfile = runQueryProfile(options, cloud, index);
		const std::string runId = makeRunId(options.inputPath, schema.name, schemaIndex);
		const std::filesystem::path jsonPath = outputPathForRun(options.outputPath, schema.name, multipleSchemas);

		std::cout << "  schema[" << (schemaIndex + 1) << "/" << schemas.size() << "]: "
			<< schema.name << " (" << loadedSchema.path << ")\n";
		std::cout << "    run id: " << runId << '\n';
		std::cout << "    schema load: " << loadedSchema.schemaLoadMs << " ms\n";
		std::cout << "    index build: " << buildMetrics.buildTimeMs << " ms\n";
		std::cout << "    nodes/leaves/maxDepth: "
			<< buildMetrics.numNodes << " / "
			<< buildMetrics.numLeaves << " / "
			<< buildMetrics.maxDepth << '\n';
		std::cout << "    leaf occupancy avg/max: "
			<< buildMetrics.averageLeafOccupancy << " / "
			<< buildMetrics.maxLeafOccupancy << '\n';
		std::cout << "    memory estimate: " << buildMetrics.memoryEstimateBytes << " bytes\n";
		std::cout << "    indexed points: " << buildMetrics.indexedPoints << '\n';
		if (options.queryCount > 0)
		{
			std::cout << "    query profile: " << options.queryCount << " generated queries per type"
				<< " (knn k=" << options.queryK << ", seed=" << options.querySeed << ")\n";
			printQueryMetrics("mixed", queryProfile.mixed);
			printQueryMetrics("range", queryProfile.range.metrics);
			printQueryMetrics("count range", queryProfile.countRange.metrics);
			printQueryMetrics("radius", queryProfile.radius.metrics);
			printQueryMetrics("knn", queryProfile.knn.metrics);
		}
		else
		{
			std::cout << "    query profile: disabled\n";
		}

		writeResults(
			jsonPath,
			runId,
			options,
			loadedSchema.path,
			schema,
			cloud,
			buildMetrics,
			queryProfile,
			loadedSchema.schemaLoadMs,
			loadMs,
			multipleSchemas,
			autoSelection);
		appendCsvSummary(
			options.csvPath,
			runId,
			options,
			loadedSchema.path,
			schema,
			cloud,
			buildMetrics,
			queryProfile,
			loadedSchema.schemaLoadMs,
			loadMs);
	}

	if (options.pauseAtEnd)
		std::system("pause");

	return 0;
}

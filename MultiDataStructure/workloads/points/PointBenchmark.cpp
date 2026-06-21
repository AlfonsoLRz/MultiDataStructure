#include "../../stdafx.h"
#include "PointBenchmark.h"

#include "../../core/Config.h"
#include "../../experiments/Metrics.h"
#include "../../experiments/SchemaSelector.h"
#include "PointCloud.h"
#include "PointSpatialIndex.h"

static double elapsedMilliseconds(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
{
	return std::chrono::duration<double, std::milli>(end - begin).count();
}

static std::string jsonEscape(const std::string& value)
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

static void writeVec3Json(std::ostream& stream, const glm::vec3& value)
{
	stream << '[' << value.x << ", " << value.y << ", " << value.z << ']';
}

static void writeDVec3Json(std::ostream& stream, const glm::dvec3& value)
{
	stream << '[' << value.x << ", " << value.y << ", " << value.z << ']';
}

using QueryBreakdown = PointSpatialIndex::QueryStats::QueryBreakdown;

static void mergeBreakdown(QueryBreakdown& target, const QueryBreakdown& source)
{
	for (size_t i = 0; i < target.visitedByDepth.size(); ++i)
		target.visitedByDepth[i] += source.visitedByDepth[i];

	for (const auto& [name, count] : source.visitedByStructure)
		target.visitedByStructure[name] += count;
	for (const auto& [name, count] : source.testedPointsByStructure)
		target.testedPointsByStructure[name] += count;
	for (const auto& [name, count] : source.fullyContainedByStructure)
		target.fullyContainedByStructure[name] += count;
}

static QueryBreakdown aggregateBreakdowns(const std::vector<PointSpatialIndex::QueryStats>& samples)
{
	QueryBreakdown result;
	for (const PointSpatialIndex::QueryStats& sample : samples)
		mergeBreakdown(result, sample.breakdown);
	return result;
}

static std::vector<std::pair<std::string, size_t>> sortedBreakdownMap(const std::unordered_map<std::string, size_t>& values)
{
	std::vector<std::pair<std::string, size_t>> sorted(values.begin(), values.end());
	std::sort(sorted.begin(), sorted.end(), [](const auto& left, const auto& right) {
		return left.first < right.first;
	});
	return sorted;
}

static std::string formatDepthBreakdown(const QueryBreakdown& breakdown)
{
	std::ostringstream output;
	bool first = true;
	for (size_t depth = 0; depth < breakdown.visitedByDepth.size(); ++depth)
	{
		const size_t count = breakdown.visitedByDepth[depth];
		if (count == 0)
			continue;
		if (!first)
			output << ';';
		first = false;
		output << depth << ':' << count;
	}
	return output.str();
}

static std::string formatMapBreakdown(const std::unordered_map<std::string, size_t>& values)
{
	std::ostringstream output;
	bool first = true;
	for (const auto& [name, count] : sortedBreakdownMap(values))
	{
		if (!first)
			output << ';';
		first = false;
		output << name << ':' << count;
	}
	return output.str();
}

static void writeDepthBreakdownJson(std::ostream& stream, const QueryBreakdown& breakdown)
{
	stream << '{';
	bool first = true;
	for (size_t depth = 0; depth < breakdown.visitedByDepth.size(); ++depth)
	{
		const size_t count = breakdown.visitedByDepth[depth];
		if (count == 0)
			continue;
		if (!first)
			stream << ", ";
		first = false;
		stream << '"' << depth << "\": " << count;
	}
	stream << '}';
}

static void writeMapBreakdownJson(std::ostream& stream, const std::unordered_map<std::string, size_t>& values)
{
	stream << '{';
	bool first = true;
	for (const auto& [name, count] : sortedBreakdownMap(values))
	{
		if (!first)
			stream << ", ";
		first = false;
		stream << '"' << jsonEscape(name) << "\": " << count;
	}
	stream << '}';
}

static void writeQueryBreakdownJson(std::ostream& stream, const char* name, const QueryBreakdown& breakdown, bool trailingComma)
{
	stream << "    \"" << name << "\": {\n";
	stream << "      \"visited_by_depth\": ";
	writeDepthBreakdownJson(stream, breakdown);
	stream << ",\n";
	stream << "      \"visited_by_structure\": ";
	writeMapBreakdownJson(stream, breakdown.visitedByStructure);
	stream << ",\n";
	stream << "      \"tested_points_by_structure\": ";
	writeMapBreakdownJson(stream, breakdown.testedPointsByStructure);
	stream << ",\n";
	stream << "      \"fully_contained_by_structure\": ";
	writeMapBreakdownJson(stream, breakdown.fullyContainedByStructure);
	stream << "\n";
	stream << "    }" << (trailingComma ? "," : "") << "\n";
}

struct QueryProfileSection
{
	std::vector<PointSpatialIndex::QueryStats> samples;
	Experiments::QueryMetrics metrics;
	QueryBreakdown breakdown;

	void add(const PointSpatialIndex::QueryStats& stats)
	{
		samples.push_back(stats);
	}

	void finalize()
	{
		metrics = Experiments::summarizeQueryStats(samples);
		breakdown = aggregateBreakdowns(samples);
	}
};

struct QueryTraceSample
{
	size_t queryId = 0;
	std::string queryType;
	bool hasBounds = false;
	AABB bounds;
	bool hasCenter = false;
	glm::vec3 center = glm::vec3(0.0f);
	float radius = 0.0f;
	size_t k = 0;
	PointSpatialIndex::QueryStats stats;
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
	QueryBreakdown mixedBreakdown;
	std::vector<QueryTraceSample> traces;

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
		mixedBreakdown = aggregateBreakdowns(allSamples);
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

static float randomFloat(std::mt19937& rng, float minValue, float maxValue)
{
	if (minValue >= maxValue)
		return minValue;

	std::uniform_real_distribution<float> distribution(minValue, maxValue);
	return distribution(rng);
}

static glm::vec3 randomPointInBounds(std::mt19937& rng, const AABB& bounds)
{
	const glm::vec3 min = bounds.min();
	const glm::vec3 max = bounds.max();
	return glm::vec3(
		randomFloat(rng, min.x, max.x),
		randomFloat(rng, min.y, max.y),
		randomFloat(rng, min.z, max.z));
}

static AABB randomQueryBox(std::mt19937& rng, const PointCloud& cloud)
{
	const glm::vec3 center = randomPointInBounds(rng, cloud.bounds());
	const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.001f));
	const float scale = randomFloat(rng, 0.01f, 0.05f);
	const glm::vec3 halfExtent = glm::max(range * scale * 0.5f, glm::vec3(0.0005f));
	return AABB(center - halfExtent, center + halfExtent);
}

static float randomQueryRadius(std::mt19937& rng, const PointCloud& cloud)
{
	const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.001f));
	const float largestRange = std::max({ range.x, range.y, range.z, 1.0f });
	return largestRange * randomFloat(rng, 0.01f, 0.04f);
}

static QueryProfileSummary runQueryProfile(const PointBenchmark::Options& options, const PointCloud& cloud, const PointSpatialIndex& index)
{
	QueryProfileSummary summary;
	summary.queryCount = options.queryCount;
	summary.queryK = options.queryK;
	summary.seed = options.querySeed;

	if (options.queryCount == 0)
		return summary;

	std::mt19937 rng(options.querySeed);
	size_t traceId = 0;
	for (size_t i = 0; i < options.queryCount; ++i)
	{
		const AABB rangeBounds = randomQueryBox(rng, cloud);
		const PointSpatialIndex::QueryStats rangeStats = index.rangeQuery(rangeBounds).stats;
		summary.range.add(rangeStats);
		summary.traces.push_back({ traceId++, "range", true, rangeBounds, false, {}, 0.0f, 0, rangeStats });

		const PointSpatialIndex::QueryStats countStats = index.countRange(rangeBounds).stats;
		summary.countRange.add(countStats);
		summary.traces.push_back({ traceId++, "count_range", true, rangeBounds, false, {}, 0.0f, 0, countStats });

		const glm::vec3 radiusCenter = randomPointInBounds(rng, cloud.bounds());
		const float radius = randomQueryRadius(rng, cloud);
		const PointSpatialIndex::QueryStats radiusStats = index.radiusQuery(radiusCenter, radius).stats;
		summary.radius.add(radiusStats);
		summary.traces.push_back({ traceId++, "radius", false, {}, true, radiusCenter, radius, 0, radiusStats });

		const glm::vec3 knnCenter = randomPointInBounds(rng, cloud.bounds());
		const PointSpatialIndex::QueryStats knnStats = index.knnQuery(knnCenter, options.queryK).stats;
		summary.knn.add(knnStats);
		summary.traces.push_back({ traceId++, "knn", false, {}, true, knnCenter, 0.0f, options.queryK, knnStats });
	}

	summary.finalize();
	return summary;
}

static std::string sanitizeName(std::string value)
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

static std::string datasetName(const std::string& inputPath)
{
	const std::filesystem::path path(inputPath);
	const std::string stem = path.stem().string();
	return stem.empty() ? "points" : stem;
}

static std::string makeRunId(const std::string& inputPath, const std::string& schemaName, size_t sequence)
{
	const auto now = std::chrono::system_clock::now();
	const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
	return "points_" + sanitizeName(datasetName(inputPath)) + "_" + sanitizeName(schemaName) + "_" + std::to_string(milliseconds) + "_" + std::to_string(sequence);
}

static std::filesystem::path outputPathForRun(const std::string& configuredOutputPath, const std::string& schemaName, bool multipleSchemas)
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

static std::string csvEscape(const std::string& value)
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

static bool isEmptyFile(const std::filesystem::path& path)
{
	std::error_code error;
	return !std::filesystem::exists(path, error) || std::filesystem::file_size(path, error) == 0;
}

static void writeQueryMetricsJson(std::ostream& stream, const char* name, const Experiments::QueryMetrics& metrics, bool trailingComma)
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
	stream << "      \"avg_fully_contained_nodes\": " << metrics.averageFullyContainedNodes << ",\n";
	stream << "      \"total_visited_nodes\": " << metrics.totalVisitedNodes << ",\n";
	stream << "      \"total_tested_points\": " << metrics.totalTestedPoints << ",\n";
	stream << "      \"total_returned_points\": " << metrics.totalReturnedPoints << ",\n";
	stream << "      \"total_fully_contained_nodes\": " << metrics.totalFullyContainedNodes << "\n";
	stream << "    }" << (trailingComma ? "," : "") << "\n";
}

static void printQueryMetrics(const char* label, const Experiments::QueryMetrics& metrics)
{
	std::cout << "    " << label << ": "
		<< "avg " << metrics.averageLatencyMs << " ms, "
		<< "p95 " << metrics.p95LatencyMs << " ms, "
		<< "visited " << metrics.averageVisitedNodes << ", "
		<< "tested " << metrics.averageTestedPoints << ", "
		<< "returned " << metrics.averageReturnedPoints << '\n';
}

static void printBounds(const PointCloud& cloud)
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

static void writeResults(
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
	output << ",\n";
	output << "    \"coordinate_space\": \"local\",\n";
	output << "    \"coordinate_frame\": {\n";
	output << "      \"origin\": ";
	writeDVec3Json(output, cloud.coordinateFrame().origin);
	output << ",\n";
	output << "      \"scale\": ";
	writeDVec3Json(output, cloud.coordinateFrame().scale);
	output << "\n";
	output << "    }\n";
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
	output << "    \"knn_backend\": \"" << (queryProfile.knn.metrics.totalQueries > 0 ? "cpu_tree_knn" : "none") << "\",\n";
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
	output << "    \"leaf_occupancy_p50\": " << buildMetrics.leafOccupancyP50 << ",\n";
	output << "    \"leaf_occupancy_p90\": " << buildMetrics.leafOccupancyP90 << ",\n";
	output << "    \"leaf_occupancy_p99\": " << buildMetrics.leafOccupancyP99 << ",\n";
	output << "    \"avg_depth\": " << buildMetrics.averageDepth << ",\n";
	output << "    \"avg_fanout\": " << buildMetrics.averageFanout << ",\n";
	output << "    \"max_fanout\": " << buildMetrics.maxFanout << ",\n";
	output << "    \"empty_child_ratio\": " << buildMetrics.emptyChildRatio << ",\n";
	output << "    \"single_child_nodes\": " << buildMetrics.singleChildNodeCount << ",\n";
	output << "    \"mean_tight_bounds_volume_ratio\": " << buildMetrics.meanTightBoundsVolumeRatio << ",\n";
	output << "    \"micro_indexed_leaves\": " << buildMetrics.microIndexedLeaves << ",\n";
	output << "    \"micro_indexed_points\": " << buildMetrics.microIndexedPoints << ",\n";
	output << "    \"node_fanout_summary\": \"" << jsonEscape(buildMetrics.nodeFanoutSummary) << "\",\n";
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
	output << "  \"query_breakdowns\": {\n";
	writeQueryBreakdownJson(output, "mixed", queryProfile.mixedBreakdown, true);
	writeQueryBreakdownJson(output, "range", queryProfile.range.breakdown, true);
	writeQueryBreakdownJson(output, "count_range", queryProfile.countRange.breakdown, true);
	writeQueryBreakdownJson(output, "radius", queryProfile.radius.breakdown, true);
	writeQueryBreakdownJson(output, "knn", queryProfile.knn.breakdown, false);
	output << "  },\n";
	output << "  \"outputs\": {\n";
	output << "    \"json_path\": \"" << jsonEscape(outputPath.string()) << "\",\n";
	output << "    \"csv_path\": \"" << jsonEscape(options.csvPath) << "\",\n";
	output << "    \"query_trace_path\": \"" << jsonEscape(options.queryTracePath) << "\",\n";
	output << "    \"multiple_schemas\": " << (multipleSchemas ? "true" : "false") << "\n";
	output << "  }\n";
	output << "}\n";
}

static void writeCsvHeader(std::ostream& output)
{
	output
		<< "run_id,dataset_name,dataset_path,num_points,schema_name,schema_path,workload_name,queries_per_type,total_queries,knn_k,query_seed,"
		<< "schema_load_time_ms,point_load_time_ms,build_time_ms,num_nodes,num_leaves,max_depth,avg_leaf_occupancy,max_leaf_occupancy,memory_estimate_bytes,"
		<< "leaf_occupancy_p50,leaf_occupancy_p90,leaf_occupancy_p99,avg_depth,avg_fanout,max_fanout,empty_child_ratio,single_child_nodes,"
		<< "mean_tight_bounds_volume_ratio,micro_indexed_leaves,micro_indexed_points,node_fanout_summary,"
		<< "mixed_avg_latency_ms,mixed_median_latency_ms,mixed_p95_latency_ms,mixed_throughput_qps,mixed_avg_visited_nodes,mixed_avg_tested_points,mixed_avg_returned_points,"
		<< "range_avg_latency_ms,range_p95_latency_ms,count_range_avg_latency_ms,count_range_p95_latency_ms,radius_avg_latency_ms,radius_p95_latency_ms,knn_avg_latency_ms,knn_p95_latency_ms,knn_backend\n";
}

static void appendCsvSummary(
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
		<< buildMetrics.leafOccupancyP50 << ','
		<< buildMetrics.leafOccupancyP90 << ','
		<< buildMetrics.leafOccupancyP99 << ','
		<< buildMetrics.averageDepth << ','
		<< buildMetrics.averageFanout << ','
		<< buildMetrics.maxFanout << ','
		<< buildMetrics.emptyChildRatio << ','
		<< buildMetrics.singleChildNodeCount << ','
		<< buildMetrics.meanTightBoundsVolumeRatio << ','
		<< buildMetrics.microIndexedLeaves << ','
		<< buildMetrics.microIndexedPoints << ','
		<< csvEscape(buildMetrics.nodeFanoutSummary) << ','
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
		<< queryProfile.knn.metrics.p95LatencyMs << ','
		<< (queryProfile.knn.metrics.totalQueries > 0 ? "cpu_tree_knn" : "none") << '\n';
}

static void writeQueryTraceHeader(std::ostream& output)
{
	output
		<< "run_id,dataset_name,dataset_path,schema_name,schema_path,workload_name,"
		<< "query_id,query_type,bounds_min_x,bounds_min_y,bounds_min_z,bounds_max_x,bounds_max_y,bounds_max_z,"
		<< "center_x,center_y,center_z,radius,k,latency_ms,visited_nodes,tested_points,returned_points,"
		<< "fully_contained_nodes,visited_by_depth,visited_by_structure,tested_points_by_structure,"
		<< "fully_contained_by_structure,backend,query_seed\n";
}

static void appendPointQueryTrace(
	const std::string& tracePath,
	const std::string& runId,
	const PointBenchmark::Options& options,
	const std::string& schemaPath,
	const SchemaConfig& schema,
	const QueryProfileSummary& queryProfile)
{
	if (tracePath.empty() || queryProfile.traces.empty())
		return;

	const std::filesystem::path path(tracePath);
	if (path.has_parent_path())
		std::filesystem::create_directories(path.parent_path());

	const bool writeHeader = isEmptyFile(path);
	std::ofstream output(path, std::ios::app);
	if (!output.is_open())
		throw std::runtime_error("Unable to open point query trace path: " + tracePath);
	if (writeHeader)
		writeQueryTraceHeader(output);

	output << std::fixed << std::setprecision(6);
	for (const QueryTraceSample& trace : queryProfile.traces)
	{
		output
			<< csvEscape(runId) << ','
			<< csvEscape(datasetName(options.inputPath)) << ','
			<< csvEscape(options.inputPath) << ','
			<< csvEscape(schema.name) << ','
			<< csvEscape(schemaPath) << ','
			<< "generated_mixed" << ','
			<< trace.queryId << ','
			<< csvEscape(trace.queryType) << ',';

		if (trace.hasBounds)
		{
			output
				<< trace.bounds.min().x << ','
				<< trace.bounds.min().y << ','
				<< trace.bounds.min().z << ','
				<< trace.bounds.max().x << ','
				<< trace.bounds.max().y << ','
				<< trace.bounds.max().z << ',';
		}
		else
		{
			output << ",,,,,,";
		}

		if (trace.hasCenter)
		{
			output
				<< trace.center.x << ','
				<< trace.center.y << ','
				<< trace.center.z << ',';
		}
		else
		{
			output << ",,,";
		}

		output
			<< trace.radius << ','
			<< trace.k << ','
			<< trace.stats.elapsedMs << ','
			<< trace.stats.visitedNodes << ','
			<< trace.stats.testedPoints << ','
			<< trace.stats.returnedPoints << ','
			<< trace.stats.fullyContainedNodes << ','
			<< csvEscape(formatDepthBreakdown(trace.stats.breakdown)) << ','
			<< csvEscape(formatMapBreakdown(trace.stats.breakdown.visitedByStructure)) << ','
			<< csvEscape(formatMapBreakdown(trace.stats.breakdown.testedPointsByStructure)) << ','
			<< csvEscape(formatMapBreakdown(trace.stats.breakdown.fullyContainedByStructure)) << ','
			<< "cpu" << ','
			<< queryProfile.seed << '\n';
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
		SchemaConfig schema = loadedSchema.config;
		if (options.enableLeafMicroIndexes)
		{
			schema.buildPolicy.enableLeafMicroIndexes = true;
			schema.buildPolicy.leafMicroIndexThreshold = options.leafMicroIndexThreshold;
		}

		PointSpatialIndex index;
		const auto buildBegin = std::chrono::steady_clock::now();
		index.build(cloud, schema);
		const auto buildEnd = std::chrono::steady_clock::now();
		const double buildMs = elapsedMilliseconds(buildBegin, buildEnd);
		const PointSpatialIndex::Stats indexStats = index.stats();
		const Experiments::BuildMetrics buildMetrics = Experiments::collectBuildMetrics(indexStats, index.root(), buildMs, schema);
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
		appendPointQueryTrace(
			options.queryTracePath,
			runId,
			options,
			loadedSchema.path,
			schema,
			queryProfile);
	}

	if (options.pauseAtEnd)
		std::system("pause");

	return 0;
}

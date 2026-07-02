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

	for (const auto& [name, count] : source._visitedByStructure)
		target._visitedByStructure[name] += count;
	for (const auto& [name, count] : source._testedPointsByStructure)
		target._testedPointsByStructure[name] += count;
	for (const auto& [name, count] : source._fullyContainedByStructure)
		target._fullyContainedByStructure[name] += count;
}

static QueryBreakdown aggregateBreakdowns(const std::vector<PointSpatialIndex::QueryStats>& samples)
{
	QueryBreakdown result;
	for (const PointSpatialIndex::QueryStats& sample : samples)
		mergeBreakdown(result, sample._breakdown);
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
	writeMapBreakdownJson(stream, breakdown._visitedByStructure);
	stream << ",\n";
	stream << "      \"tested_points_by_structure\": ";
	writeMapBreakdownJson(stream, breakdown._testedPointsByStructure);
	stream << ",\n";
	stream << "      \"fully_contained_by_structure\": ";
	writeMapBreakdownJson(stream, breakdown._fullyContainedByStructure);
	stream << "\n";
	stream << "    }" << (trailingComma ? "," : "") << "\n";
}

struct QueryProfileSection
{
	std::vector<PointSpatialIndex::QueryStats>	_samples;
	Experiments::QueryMetrics	_metrics;
	QueryBreakdown	_breakdown;

	void add(const PointSpatialIndex::QueryStats& stats)
	{
		_samples.push_back(stats);
	}

	void finalize()
	{
		_metrics = Experiments::summarizeQueryStats(_samples);
		_breakdown = aggregateBreakdowns(_samples);
	}
};

struct QueryTraceSample
{
	size_t		_queryId = 0;
	std::string	_queryType;
	bool		_hasBounds = false;
	AABB		_bounds;
	bool		_hasCenter = false;
	glm::vec3 center = glm::vec3(0.0f);
	float	_radius = 0.0f;
	size_t	_k = 0;
	PointSpatialIndex::QueryStats	_stats;
};

struct QueryProfileSummary
{
	size_t	_queryCount = 0;
	size_t	_queryK = 0;
	uint32_t	_seed = 0;
	QueryProfileSection	_range;
	QueryProfileSection	_countRange;
	QueryProfileSection	_radius;
	QueryProfileSection	_knn;
	Experiments::QueryMetrics	_mixed;
	QueryBreakdown	_mixedBreakdown;
	std::vector<QueryTraceSample>	_traces;

	size_t totalQueries() const
	{
		return _mixed._totalQueries;
	}

	void finalize()
	{
		_range.finalize();
		_countRange.finalize();
		_radius.finalize();
		_knn.finalize();

		std::vector<PointSpatialIndex::QueryStats> allSamples;
		allSamples.reserve(_range._samples.size() + _countRange._samples.size() + _radius._samples.size() + _knn._samples.size());
		allSamples.insert(allSamples.end(), _range._samples.begin(), _range._samples.end());
		allSamples.insert(allSamples.end(), _countRange._samples.begin(), _countRange._samples.end());
		allSamples.insert(allSamples.end(), _radius._samples.begin(), _radius._samples.end());
		allSamples.insert(allSamples.end(), _knn._samples.begin(), _knn._samples.end());
		_mixed = Experiments::summarizeQueryStats(allSamples);
		_mixedBreakdown = aggregateBreakdowns(allSamples);
	}
};

struct LoadedSchema
{
	std::string		_path;
	SchemaConfig	_config;
	double			_schemaLoadMs = 0.0;
};

struct AutoSelectionLog
{
	bool	_enabled = false;
	Experiments::SchemaSelection	_selection;
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
	summary._queryCount = options._queryCount;
	summary._queryK = options._queryK;
	summary._seed = options._querySeed;

	if (options._queryCount == 0)
		return summary;

	std::mt19937 rng(options._querySeed);
	size_t traceId = 0;
	for (size_t i = 0; i < options._queryCount; ++i)
	{
		const AABB rangeBounds = randomQueryBox(rng, cloud);
		const PointSpatialIndex::QueryStats rangeStats = index.rangeQuery(rangeBounds)._stats;
		summary._range.add(rangeStats);
		summary._traces.push_back({ traceId++, "range", true, rangeBounds, false, {}, 0.0f, 0, rangeStats });

		const PointSpatialIndex::QueryStats countStats = index.countRange(rangeBounds)._stats;
		summary._countRange.add(countStats);
		summary._traces.push_back({ traceId++, "count_range", true, rangeBounds, false, {}, 0.0f, 0, countStats });

		const glm::vec3 radiusCenter = randomPointInBounds(rng, cloud.bounds());
		const float radius = randomQueryRadius(rng, cloud);
		const PointSpatialIndex::QueryStats radiusStats = index.radiusQuery(radiusCenter, radius)._stats;
		summary._radius.add(radiusStats);
		summary._traces.push_back({ traceId++, "radius", false, {}, true, radiusCenter, radius, 0, radiusStats });

		const glm::vec3 knnCenter = randomPointInBounds(rng, cloud.bounds());
		const PointSpatialIndex::QueryStats knnStats = index.knnQuery(knnCenter, options._queryK)._stats;
		summary._knn.add(knnStats);
		summary._traces.push_back({ traceId++, "knn", false, {}, true, knnCenter, 0.0f, options._queryK, knnStats });
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
	stream << "      \"total_queries\": " << metrics._totalQueries << ",\n";
	stream << "      \"total_latency_ms\": " << metrics._totalLatencyMs << ",\n";
	stream << "      \"avg_latency_ms\": " << metrics._averageLatencyMs << ",\n";
	stream << "      \"median_latency_ms\": " << metrics._medianLatencyMs << ",\n";
	stream << "      \"p95_latency_ms\": " << metrics._p95LatencyMs << ",\n";
	stream << "      \"throughput_queries_per_sec\": " << metrics._throughputQueriesPerSecond << ",\n";
	stream << "      \"avg_visited_nodes\": " << metrics._averageVisitedNodes << ",\n";
	stream << "      \"avg_tested_points\": " << metrics._averageTestedPoints << ",\n";
	stream << "      \"avg_returned_points\": " << metrics._averageReturnedPoints << ",\n";
	stream << "      \"avg_fully_contained_nodes\": " << metrics._averageFullyContainedNodes << ",\n";
	stream << "      \"total_visited_nodes\": " << metrics._totalVisitedNodes << ",\n";
	stream << "      \"total_tested_points\": " << metrics._totalTestedPoints << ",\n";
	stream << "      \"total_returned_points\": " << metrics._totalReturnedPoints << ",\n";
	stream << "      \"total_fully_contained_nodes\": " << metrics._totalFullyContainedNodes << "\n";
	stream << "    }" << (trailingComma ? "," : "") << "\n";
}

static void printQueryMetrics(const char* label, const Experiments::QueryMetrics& metrics)
{
	std::cout << "    " << label << ": "
		<< "avg " << metrics._averageLatencyMs << " ms, "
		<< "p95 " << metrics._p95LatencyMs << " ms, "
		<< "visited " << metrics._averageVisitedNodes << ", "
		<< "tested " << metrics._averageTestedPoints << ", "
		<< "returned " << metrics._averageReturnedPoints << '\n';
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
		throw std::runtime_error("Unable to open point benchmark output path: " + options._outputPath);

	output << std::fixed << std::setprecision(4);
	output << "{\n";
	output << "  \"run_id\": \"" << jsonEscape(runId) << "\",\n";
	output << "  \"mode\": \"points\",\n";
	output << "  \"dataset\": {\n";
	output << "    \"name\": \"" << jsonEscape(datasetName(options._inputPath)) << "\",\n";
	output << "    \"source\": \"file\",\n";
	output << "    \"path\": \"" << jsonEscape(options._inputPath) << "\",\n";
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
	writeDVec3Json(output, cloud.coordinateFrame()._scale);
	output << "\n";
	output << "    }\n";
	output << "  },\n";
	output << "  \"cache\": {\n";
	output << "    \"enabled\": " << (options._useBinaryCache ? "true" : "false") << ",\n";
	output << "    \"path\": \"" << jsonEscape(cloud.cachePath()) << "\",\n";
	output << "    \"loaded_from_cache\": " << (cloud.loadedFromCache() ? "true" : "false") << "\n";
	output << "  },\n";
	output << "  \"schema\": {\n";
	output << "    \"path\": \"" << jsonEscape(schemaPath) << "\",\n";
	output << "    \"name\": \"" << jsonEscape(schema._name) << "\",\n";
	output << "    \"total_levels\": " << schema.totalLevels() << "\n";
	output << "  },\n";
	if (autoSelection._enabled)
	{
		output << "  \"schema_selection\": {\n";
		output << "    \"mode\": \"auto\",\n";
		output << "    \"model_path\": \"" << jsonEscape(autoSelection._selection._modelPath) << "\",\n";
		output << "    \"workload_profile_path\": \"" << jsonEscape(autoSelection._selection._workloadProfilePath) << "\",\n";
		output << "    \"selected_schema_name\": \"" << jsonEscape(autoSelection._selection._schemaName) << "\",\n";
		output << "    \"selected_schema_path\": \"" << jsonEscape(autoSelection._selection._schemaPath) << "\",\n";
		output << "    \"score_source\": \"" << (autoSelection._selection._measuredScore ? "measured" : "predicted") << "\",\n";
		output << "    \"predicted_score\": " << autoSelection._selection._predictedScore << ",\n";
		output << "    \"candidates\": [\n";
		for (size_t i = 0; i < autoSelection._selection._candidates.size(); ++i)
		{
			const Experiments::CandidatePrediction& candidate = autoSelection._selection._candidates[i];
			output << "      {\"schema_name\": \"" << jsonEscape(candidate._schemaName)
				<< "\", \"schema_path\": \"" << jsonEscape(candidate._schemaPath)
				<< "\", \"predicted_score\": " << candidate._predictedScore << "}";
			output << (i + 1 < autoSelection._selection._candidates.size() ? "," : "") << "\n";
		}
		output << "    ]\n";
		output << "  },\n";
	}
	output << "  \"workload\": {\n";
	output << "    \"name\": \"generated_mixed\",\n";
	output << "    \"queries_per_type\": " << queryProfile._queryCount << ",\n";
	output << "    \"total_queries\": " << queryProfile.totalQueries() << ",\n";
	output << "    \"knn_k\": " << queryProfile._queryK << ",\n";
	output << "    \"knn_backend\": \"" << (queryProfile._knn._metrics._totalQueries > 0 ? "cpu_tree_knn" : "none") << "\",\n";
	output << "    \"seed\": " << queryProfile._seed << "\n";
	output << "  },\n";
	output << "  \"timings\": {\n";
	output << "    \"schema_load_time_ms\": " << schemaMs << ",\n";
	output << "    \"point_load_time_ms\": " << loadMs << "\n";
	output << "  },\n";
	output << "  \"build_metrics\": {\n";
	output << "    \"build_time_ms\": " << buildMetrics._buildTimeMs << ",\n";
	output << "    \"num_nodes\": " << buildMetrics._numNodes << ",\n";
	output << "    \"num_leaves\": " << buildMetrics._numLeaves << ",\n";
	output << "    \"indexed_points\": " << buildMetrics._indexedPoints << ",\n";
	output << "    \"max_depth\": " << buildMetrics._maxDepth << ",\n";
	output << "    \"avg_leaf_occupancy\": " << buildMetrics._averageLeafOccupancy << ",\n";
	output << "    \"max_leaf_occupancy\": " << buildMetrics._maxLeafOccupancy << ",\n";
	output << "    \"leaf_occupancy_p50\": " << buildMetrics._leafOccupancyP50 << ",\n";
	output << "    \"leaf_occupancy_p90\": " << buildMetrics._leafOccupancyP90 << ",\n";
	output << "    \"leaf_occupancy_p99\": " << buildMetrics._leafOccupancyP99 << ",\n";
	output << "    \"avg_depth\": " << buildMetrics._averageDepth << ",\n";
	output << "    \"avg_fanout\": " << buildMetrics._averageFanout << ",\n";
	output << "    \"max_fanout\": " << buildMetrics._maxFanout << ",\n";
	output << "    \"empty_child_ratio\": " << buildMetrics._emptyChildRatio << ",\n";
	output << "    \"single_child_nodes\": " << buildMetrics._singleChildNodeCount << ",\n";
	output << "    \"mean_tight_bounds_volume_ratio\": " << buildMetrics._meanTightBoundsVolumeRatio << ",\n";
	output << "    \"micro_indexed_leaves\": " << buildMetrics._microIndexedLeaves << ",\n";
	output << "    \"micro_indexed_points\": " << buildMetrics._microIndexedPoints << ",\n";
	output << "    \"node_fanout_summary\": \"" << jsonEscape(buildMetrics._nodeFanoutSummary) << "\",\n";
	output << "    \"memory_estimate_bytes\": " << buildMetrics._memoryEstimateBytes << "\n";
	output << "  },\n";
	output << "  \"query_metrics\": {\n";
	writeQueryMetricsJson(output, "mixed", queryProfile._mixed, true);
	writeQueryMetricsJson(output, "range", queryProfile._range._metrics, true);
	writeQueryMetricsJson(output, "count_range", queryProfile._countRange._metrics, true);
	writeQueryMetricsJson(output, "radius", queryProfile._radius._metrics, true);
	writeQueryMetricsJson(output, "knn", queryProfile._knn._metrics, false);
	output << "\n";
	output << "  },\n";
	output << "  \"query_breakdowns\": {\n";
	writeQueryBreakdownJson(output, "mixed", queryProfile._mixedBreakdown, true);
	writeQueryBreakdownJson(output, "range", queryProfile._range._breakdown, true);
	writeQueryBreakdownJson(output, "count_range", queryProfile._countRange._breakdown, true);
	writeQueryBreakdownJson(output, "radius", queryProfile._radius._breakdown, true);
	writeQueryBreakdownJson(output, "knn", queryProfile._knn._breakdown, false);
	output << "  },\n";
	output << "  \"outputs\": {\n";
	output << "    \"json_path\": \"" << jsonEscape(outputPath.string()) << "\",\n";
	output << "    \"csv_path\": \"" << jsonEscape(options._csvPath) << "\",\n";
	output << "    \"query_trace_path\": \"" << jsonEscape(options._queryTracePath) << "\",\n";
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
		<< csvEscape(datasetName(options._inputPath)) << ','
		<< csvEscape(options._inputPath) << ','
		<< cloud.size() << ','
		<< csvEscape(schema._name) << ','
		<< csvEscape(schemaPath) << ','
		<< "generated_mixed" << ','
		<< queryProfile._queryCount << ','
		<< queryProfile.totalQueries() << ','
		<< queryProfile._queryK << ','
		<< queryProfile._seed << ','
		<< schemaMs << ','
		<< loadMs << ','
		<< buildMetrics._buildTimeMs << ','
		<< buildMetrics._numNodes << ','
		<< buildMetrics._numLeaves << ','
		<< buildMetrics._maxDepth << ','
		<< buildMetrics._averageLeafOccupancy << ','
		<< buildMetrics._maxLeafOccupancy << ','
		<< buildMetrics._memoryEstimateBytes << ','
		<< buildMetrics._leafOccupancyP50 << ','
		<< buildMetrics._leafOccupancyP90 << ','
		<< buildMetrics._leafOccupancyP99 << ','
		<< buildMetrics._averageDepth << ','
		<< buildMetrics._averageFanout << ','
		<< buildMetrics._maxFanout << ','
		<< buildMetrics._emptyChildRatio << ','
		<< buildMetrics._singleChildNodeCount << ','
		<< buildMetrics._meanTightBoundsVolumeRatio << ','
		<< buildMetrics._microIndexedLeaves << ','
		<< buildMetrics._microIndexedPoints << ','
		<< csvEscape(buildMetrics._nodeFanoutSummary) << ','
		<< queryProfile._mixed._averageLatencyMs << ','
		<< queryProfile._mixed._medianLatencyMs << ','
		<< queryProfile._mixed._p95LatencyMs << ','
		<< queryProfile._mixed._throughputQueriesPerSecond << ','
		<< queryProfile._mixed._averageVisitedNodes << ','
		<< queryProfile._mixed._averageTestedPoints << ','
		<< queryProfile._mixed._averageReturnedPoints << ','
		<< queryProfile._range._metrics._averageLatencyMs << ','
		<< queryProfile._range._metrics._p95LatencyMs << ','
		<< queryProfile._countRange._metrics._averageLatencyMs << ','
		<< queryProfile._countRange._metrics._p95LatencyMs << ','
		<< queryProfile._radius._metrics._averageLatencyMs << ','
		<< queryProfile._radius._metrics._p95LatencyMs << ','
		<< queryProfile._knn._metrics._averageLatencyMs << ','
		<< queryProfile._knn._metrics._p95LatencyMs << ','
		<< (queryProfile._knn._metrics._totalQueries > 0 ? "cpu_tree_knn" : "none") << '\n';
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
	const QueryProfileSummary& queryProfile,
	const PointCloud& cloud)
{
	if (tracePath.empty() || queryProfile._traces.empty())
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
	for (const QueryTraceSample& trace : queryProfile._traces)
	{
		output
			<< csvEscape(runId) << ','
			<< csvEscape(datasetName(options._inputPath)) << ','
			<< csvEscape(options._inputPath) << ','
			<< csvEscape(schema._name) << ','
			<< csvEscape(schemaPath) << ','
			<< "generated_mixed" << ','
			<< trace._queryId << ','
			<< csvEscape(trace._queryType) << ',';

		// World coordinates, so external tools can replay against the raw cloud file
		// (identity transform for XYZ/PLY/CSV clouds; origin translation for LAS).
		if (trace._hasBounds)
		{
			const glm::dvec3 worldMin = cloud.toWorldPosition(trace._bounds.min());
			const glm::dvec3 worldMax = cloud.toWorldPosition(trace._bounds.max());
			output
				<< worldMin.x << ','
				<< worldMin.y << ','
				<< worldMin.z << ','
				<< worldMax.x << ','
				<< worldMax.y << ','
				<< worldMax.z << ',';
		}
		else
		{
			output << ",,,,,,";
		}

		if (trace._hasCenter)
		{
			const glm::dvec3 worldCenter = cloud.toWorldPosition(trace.center);
			output
				<< worldCenter.x << ','
				<< worldCenter.y << ','
				<< worldCenter.z << ',';
		}
		else
		{
			output << ",,,";
		}

		output
			<< trace._radius << ','
			<< trace._k << ','
			<< trace._stats._elapsedMs << ','
			<< trace._stats._visitedNodes << ','
			<< trace._stats._testedPoints << ','
			<< trace._stats._returnedPoints << ','
			<< trace._stats._fullyContainedNodes << ','
			<< csvEscape(formatDepthBreakdown(trace._stats._breakdown)) << ','
			<< csvEscape(formatMapBreakdown(trace._stats._breakdown._visitedByStructure)) << ','
			<< csvEscape(formatMapBreakdown(trace._stats._breakdown._testedPointsByStructure)) << ','
			<< csvEscape(formatMapBreakdown(trace._stats._breakdown._fullyContainedByStructure)) << ','
			<< "cpu" << ','
			<< queryProfile._seed << '\n';
	}
}

int PointBenchmark::run(const Options& options)
{
	if (options._inputPath.empty())
		throw std::invalid_argument("Point mode requires --input <file.las|file.ply>");

	const auto loadBegin = std::chrono::steady_clock::now();
	const PointCloud cloud = PointCloud::load(options._inputPath, { options._useBinaryCache, options._rebuildBinaryCache });
	const auto loadEnd = std::chrono::steady_clock::now();
	if (cloud.empty())
		throw std::runtime_error("Point cloud is empty: " + options._inputPath);

	const double loadMs = elapsedMilliseconds(loadBegin, loadEnd);
	AutoSelectionLog autoSelection;
	std::vector<std::string> schemaPaths = options._schemaPaths;
	if (schemaPaths.empty())
	{
		if (options._schemaPath == "auto")
		{
			if (options._modelPath.empty())
				throw std::invalid_argument("--schema auto requires --model <schema_selector.json>");
			if (options._workloadProfilePath.empty())
				throw std::invalid_argument("--schema auto requires --workload-profile <profile.json>");

			autoSelection._enabled = true;
			autoSelection._selection = Experiments::selectSchemaForCloud(options._modelPath, options._workloadProfilePath, cloud);
			schemaPaths.push_back(autoSelection._selection._schemaPath);
		}
		else
		{
			schemaPaths.push_back(options._schemaPath);
		}
	}

	std::vector<LoadedSchema> schemas;
	schemas.reserve(schemaPaths.size());
	for (const std::string& schemaPath : schemaPaths)
	{
		const auto schemaBegin = std::chrono::steady_clock::now();
		LoadedSchema loaded;
		loaded._path = schemaPath;
		loaded._config = Config::loadSchemaConfig(schemaPath);
		const auto schemaEnd = std::chrono::steady_clock::now();
		loaded._schemaLoadMs = elapsedMilliseconds(schemaBegin, schemaEnd);
		schemas.push_back(std::move(loaded));
	}

	const bool multipleSchemas = schemas.size() > 1;

	std::cout << std::fixed << std::setprecision(3);
	std::cout << "Point cloud benchmark\n";
	std::cout << "  input: " << options._inputPath << '\n';
	if (options._useBinaryCache)
		std::cout << "  cache: " << (cloud.loadedFromCache() ? "loaded " : "written ") << cloud.cachePath() << '\n';
	else
		std::cout << "  cache: disabled\n";
	std::cout << "  points: " << cloud.size() << '\n';
	printBounds(cloud);
	std::cout << "  point load: " << loadMs << " ms\n";
	if (autoSelection._enabled)
	{
		std::cout << "  schema selection: auto\n";
		std::cout << "    model: " << autoSelection._selection._modelPath << '\n';
		std::cout << "    workload profile: " << autoSelection._selection._workloadProfilePath << '\n';
		std::cout << "    selected: " << autoSelection._selection._schemaName
			<< " (" << autoSelection._selection._schemaPath << ")"
			<< (autoSelection._selection._measuredScore ? " measured score " : " predicted score ")
			<< autoSelection._selection._predictedScore << '\n';
		for (const Experiments::CandidatePrediction& candidate : autoSelection._selection._candidates)
		{
			std::cout << "      candidate: " << candidate._schemaName
				<< " score " << candidate._predictedScore << '\n';
		}
	}

	for (size_t schemaIndex = 0; schemaIndex < schemas.size(); ++schemaIndex)
	{
		const LoadedSchema& loadedSchema = schemas[schemaIndex];
		SchemaConfig schema = loadedSchema._config;
		if (options._enableLeafMicroIndexes)
		{
			schema._buildPolicy._enableLeafMicroIndexes = true;
			schema._buildPolicy._leafMicroIndexThreshold = options._leafMicroIndexThreshold;
		}

		PointSpatialIndex index;
		const auto buildBegin = std::chrono::steady_clock::now();
		index.build(cloud, schema);
		const auto buildEnd = std::chrono::steady_clock::now();
		const double buildMs = elapsedMilliseconds(buildBegin, buildEnd);
		const PointSpatialIndex::Stats indexStats = index.stats();
		const Experiments::BuildMetrics buildMetrics = Experiments::collectBuildMetrics(indexStats, index.root(), buildMs, schema);
		const QueryProfileSummary queryProfile = runQueryProfile(options, cloud, index);
		const std::string runId = makeRunId(options._inputPath, schema._name, schemaIndex);
		const std::filesystem::path jsonPath = outputPathForRun(options._outputPath, schema._name, multipleSchemas);

		std::cout << "  schema[" << (schemaIndex + 1) << "/" << schemas.size() << "]: "
			<< schema._name << " (" << loadedSchema._path << ")\n";
		std::cout << "    run id: " << runId << '\n';
		std::cout << "    schema load: " << loadedSchema._schemaLoadMs << " ms\n";
		std::cout << "    index build: " << buildMetrics._buildTimeMs << " ms\n";
		std::cout << "    nodes/leaves/maxDepth: "
			<< buildMetrics._numNodes << " / "
			<< buildMetrics._numLeaves << " / "
			<< buildMetrics._maxDepth << '\n';
		std::cout << "    leaf occupancy avg/max: "
			<< buildMetrics._averageLeafOccupancy << " / "
			<< buildMetrics._maxLeafOccupancy << '\n';
		std::cout << "    memory estimate: " << buildMetrics._memoryEstimateBytes << " bytes\n";
		std::cout << "    indexed points: " << buildMetrics._indexedPoints << '\n';
		if (options._queryCount > 0)
		{
			std::cout << "    query profile: " << options._queryCount << " generated queries per type"
				<< " (knn k=" << options._queryK << ", seed=" << options._querySeed << ")\n";
			printQueryMetrics("mixed", queryProfile._mixed);
			printQueryMetrics("range", queryProfile._range._metrics);
			printQueryMetrics("count range", queryProfile._countRange._metrics);
			printQueryMetrics("radius", queryProfile._radius._metrics);
			printQueryMetrics("knn", queryProfile._knn._metrics);
		}
		else
		{
			std::cout << "    query profile: disabled\n";
		}

		writeResults(
			jsonPath,
			runId,
			options,
			loadedSchema._path,
			schema,
			cloud,
			buildMetrics,
			queryProfile,
			loadedSchema._schemaLoadMs,
			loadMs,
			multipleSchemas,
			autoSelection);
		appendCsvSummary(
			options._csvPath,
			runId,
			options,
			loadedSchema._path,
			schema,
			cloud,
			buildMetrics,
			queryProfile,
			loadedSchema._schemaLoadMs,
			loadMs);
		appendPointQueryTrace(
			options._queryTracePath,
			runId,
			options,
			loadedSchema._path,
			schema,
			queryProfile,
			cloud);
	}

	if (options._pauseAtEnd)
		std::system("pause");

	return 0;
}

#include "../../stdafx.h"
#include "PointBenchmark.h"

#include "../../core/Config.h"
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
		const PointBenchmark::Options& options,
		const SchemaConfig& schema,
		const PointCloud& cloud,
		const PointSpatialIndex::Stats& indexStats,
		double loadMs,
		double buildMs)
	{
		if (options.outputPath.empty())
			return;

		const std::filesystem::path outputPath(options.outputPath);
		if (outputPath.has_parent_path())
			std::filesystem::create_directories(outputPath.parent_path());

		std::ofstream output(outputPath);
		if (!output.is_open())
			throw std::runtime_error("Unable to open point benchmark output path: " + options.outputPath);

		output << std::fixed << std::setprecision(4);
		output << "{\n";
		output << "  \"mode\": \"points\",\n";
		output << "  \"input\": \"" << jsonEscape(options.inputPath) << "\",\n";
		output << "  \"cache_enabled\": " << (options.useBinaryCache ? "true" : "false") << ",\n";
		output << "  \"cache_path\": \"" << jsonEscape(cloud.cachePath()) << "\",\n";
		output << "  \"loaded_from_cache\": " << (cloud.loadedFromCache() ? "true" : "false") << ",\n";
		output << "  \"schema\": {\n";
		output << "    \"path\": \"" << jsonEscape(options.schemaPath) << "\",\n";
		output << "    \"name\": \"" << jsonEscape(schema.name) << "\",\n";
		output << "    \"total_levels\": " << schema.totalLevels() << "\n";
		output << "  },\n";
		output << "  \"metrics\": {\n";
		output << "    \"load_time_ms\": " << loadMs << ",\n";
		output << "    \"build_time_ms\": " << buildMs << ",\n";
		output << "    \"num_points\": " << cloud.size() << ",\n";
		output << "    \"num_nodes\": " << indexStats.numNodes << ",\n";
		output << "    \"num_leaves\": " << indexStats.numLeaves << ",\n";
		output << "    \"indexed_points\": " << indexStats.numPoints << ",\n";
		output << "    \"max_depth\": " << indexStats.maxDepth << ",\n";
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
		output << "  }\n";
		output << "}\n";
	}
}

int PointBenchmark::run(const Options& options)
{
	if (options.inputPath.empty())
		throw std::invalid_argument("Point mode requires --input <file.las|file.ply>");

	const auto schemaBegin = std::chrono::steady_clock::now();
	const SchemaConfig schema = Config::loadSchemaConfig(options.schemaPath);
	const auto schemaEnd = std::chrono::steady_clock::now();

	const auto loadBegin = std::chrono::steady_clock::now();
	const PointCloud cloud = PointCloud::load(options.inputPath, { options.useBinaryCache, options.rebuildBinaryCache });
	const auto loadEnd = std::chrono::steady_clock::now();
	if (cloud.empty())
		throw std::runtime_error("Point cloud is empty: " + options.inputPath);

	PointSpatialIndex index;
	const auto buildBegin = std::chrono::steady_clock::now();
	index.build(cloud, schema);
	const auto buildEnd = std::chrono::steady_clock::now();
	const PointSpatialIndex::Stats indexStats = index.stats();

	const double schemaMs = elapsedMilliseconds(schemaBegin, schemaEnd);
	const double loadMs = elapsedMilliseconds(loadBegin, loadEnd);
	const double buildMs = elapsedMilliseconds(buildBegin, buildEnd);

	std::cout << std::fixed << std::setprecision(3);
	std::cout << "Point cloud benchmark\n";
	std::cout << "  input: " << options.inputPath << '\n';
	if (options.useBinaryCache)
		std::cout << "  cache: " << (cloud.loadedFromCache() ? "loaded " : "written ") << cloud.cachePath() << '\n';
	else
		std::cout << "  cache: disabled\n";
	std::cout << "  schema: " << schema.name << " (" << options.schemaPath << ")\n";
	std::cout << "  points: " << cloud.size() << '\n';
	printBounds(cloud);
	std::cout << "  schema load: " << schemaMs << " ms\n";
	std::cout << "  point load: " << loadMs << " ms\n";
	std::cout << "  index build: " << buildMs << " ms\n";
	std::cout << "  nodes/leaves/maxDepth: "
		<< indexStats.numNodes << " / "
		<< indexStats.numLeaves << " / "
		<< indexStats.maxDepth << '\n';
	std::cout << "  indexed points: " << indexStats.numPoints << '\n';

	writeResults(options, schema, cloud, indexStats, loadMs, buildMs);

	if (options.pauseAtEnd)
		std::system("pause");

	return 0;
}

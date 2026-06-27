// Optional standalone PCL baseline helper for scripts/compare_frameworks.py.
//
// Build outside the main project when PCL is installed, for example with your
// preferred CMake/vcpkg/PCL setup, then pass the executable as --pcl-exe.

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
	using Clock = std::chrono::high_resolution_clock;
	using Point = pcl::PointXYZ;
	using Cloud = pcl::PointCloud<Point>;

	struct Args
	{
		std::string input;
		std::string queryTrace;
		std::string output;
		double octreeResolution = 0.0;
	};

	struct Vec3
	{
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;
	};

	struct Query
	{
		size_t id = 0;
		std::string kind;
		Vec3 minBound;
		Vec3 maxBound;
		Vec3 center;
		float radius = 0.0f;
		int k = 0;
		int expectedCount = -1;
	};

	struct Sample
	{
		std::string kind;
		double latencyMs = 0.0;
		int returnedCount = 0;
		int expectedCount = -1;
	};

	double elapsedMs(const Clock::time_point& start, const Clock::time_point& end)
	{
		return std::chrono::duration<double, std::milli>(end - start).count();
	}

	std::string lowerCopy(std::string value)
	{
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return value;
	}

	std::vector<std::string> splitCsvLine(const std::string& line)
	{
		std::vector<std::string> cells;
		std::string current;
		bool quoted = false;
		for (size_t i = 0; i < line.size(); ++i)
		{
			const char c = line[i];
			if (c == '"')
			{
				if (quoted && i + 1 < line.size() && line[i + 1] == '"')
				{
					current.push_back('"');
					++i;
				}
				else
				{
					quoted = !quoted;
				}
				continue;
			}
			if (c == ',' && !quoted)
			{
				cells.push_back(current);
				current.clear();
				continue;
			}
			current.push_back(c);
		}
		cells.push_back(current);
		return cells;
	}

	float parseFloat(const std::string& value, float fallback = 0.0f)
	{
		if (value.empty())
			return fallback;
		return std::strtof(value.c_str(), nullptr);
	}

	int parseInt(const std::string& value, int fallback = 0)
	{
		if (value.empty())
			return fallback;
		return static_cast<int>(std::strtol(value.c_str(), nullptr, 10));
	}

	std::string jsonEscape(const std::string& value)
	{
		std::ostringstream out;
		for (const char c : value)
		{
			switch (c)
			{
			case '\\': out << "\\\\"; break;
			case '"': out << "\\\""; break;
			case '\n': out << "\\n"; break;
			case '\r': out << "\\r"; break;
			case '\t': out << "\\t"; break;
			default: out << c; break;
			}
		}
		return out.str();
	}

	Args parseArgs(int argc, char** argv)
	{
		Args args;
		for (int i = 1; i < argc; ++i)
		{
			const std::string arg = argv[i];
			if (arg == "--input" && i + 1 < argc)
				args.input = argv[++i];
			else if (arg == "--query-trace" && i + 1 < argc)
				args.queryTrace = argv[++i];
			else if (arg == "--output" && i + 1 < argc)
				args.output = argv[++i];
			else if (arg == "--octree-resolution" && i + 1 < argc)
				args.octreeResolution = std::stod(argv[++i]);
			else if (arg == "--help" || arg == "-h")
				throw std::runtime_error("usage: pcl_point_baseline --input cloud.ply|cloud.xyz|cloud.csv|cloud.pcd --query-trace query_trace.csv --output result.json [--octree-resolution r]");
			else
				throw std::runtime_error("unknown argument: " + arg);
		}
		if (args.input.empty() || args.queryTrace.empty() || args.output.empty())
			throw std::runtime_error("--input, --query-trace, and --output are required");
		return args;
	}

	bool parsePointLine(const std::string& line, Point& point)
	{
		std::string cleaned = line;
		std::replace(cleaned.begin(), cleaned.end(), ',', ' ');
		std::replace(cleaned.begin(), cleaned.end(), ';', ' ');
		std::istringstream in(cleaned);
		return static_cast<bool>(in >> point.x >> point.y >> point.z);
	}

	Cloud::Ptr loadTextCloud(const std::string& filename)
	{
		std::ifstream input(filename);
		if (!input.is_open())
			throw std::runtime_error("unable to open point file: " + filename);
		Cloud::Ptr cloud(new Cloud());
		std::string line;
		while (std::getline(input, line))
		{
			Point point;
			if (parsePointLine(line, point))
				cloud->push_back(point);
		}
		return cloud;
	}

	Cloud::Ptr loadCloud(const std::string& filename)
	{
		const std::string ext = lowerCopy(std::filesystem::path(filename).extension().string());
		Cloud::Ptr cloud(new Cloud());
		if (ext == ".pcd")
		{
			if (pcl::io::loadPCDFile<Point>(filename, *cloud) != 0)
				throw std::runtime_error("PCL failed to load PCD: " + filename);
			return cloud;
		}
		if (ext == ".ply")
		{
			if (pcl::io::loadPLYFile<Point>(filename, *cloud) != 0)
				throw std::runtime_error("PCL failed to load PLY: " + filename);
			return cloud;
		}
		if (ext == ".xyz" || ext == ".txt" || ext == ".csv")
			return loadTextCloud(filename);
		throw std::runtime_error("PCL helper supports PCD, PLY, XYZ, TXT, and CSV only");
	}

	std::vector<Query> loadQueries(const std::string& filename)
	{
		std::ifstream input(filename);
		if (!input.is_open())
			throw std::runtime_error("unable to open query trace: " + filename);
		std::string line;
		if (!std::getline(input, line))
			throw std::runtime_error("empty query trace: " + filename);
		const std::vector<std::string> header = splitCsvLine(line);
		std::map<std::string, size_t> index;
		for (size_t i = 0; i < header.size(); ++i)
			index[header[i]] = i;

		auto cell = [&](const std::vector<std::string>& row, const std::string& name) -> std::string {
			const auto found = index.find(name);
			if (found == index.end() || found->second >= row.size())
				return {};
			return row[found->second];
		};

		std::vector<Query> queries;
		while (std::getline(input, line))
		{
			const std::vector<std::string> row = splitCsvLine(line);
			Query query;
			query.id = static_cast<size_t>(parseInt(cell(row, "query_id"), static_cast<int>(queries.size())));
			query.kind = lowerCopy(cell(row, "query_type"));
			if (query.kind == "count_range")
				query.kind = "range";
			if (query.kind != "range" && query.kind != "radius" && query.kind != "knn")
				continue;
			query.minBound = {
				parseFloat(cell(row, "bounds_min_x")),
				parseFloat(cell(row, "bounds_min_y")),
				parseFloat(cell(row, "bounds_min_z")),
			};
			query.maxBound = {
				parseFloat(cell(row, "bounds_max_x")),
				parseFloat(cell(row, "bounds_max_y")),
				parseFloat(cell(row, "bounds_max_z")),
			};
			query.center = {
				parseFloat(cell(row, "center_x")),
				parseFloat(cell(row, "center_y")),
				parseFloat(cell(row, "center_z")),
			};
			query.radius = parseFloat(cell(row, "radius"));
			query.k = parseInt(cell(row, "k"));
			query.expectedCount = parseInt(cell(row, "returned_points"), -1);
			queries.push_back(query);
		}
		return queries;
	}

	double defaultOctreeResolution(const Cloud& cloud)
	{
		if (cloud.empty())
			return 1.0;
		float minX = cloud.front().x, maxX = cloud.front().x;
		float minY = cloud.front().y, maxY = cloud.front().y;
		float minZ = cloud.front().z, maxZ = cloud.front().z;
		for (const Point& point : cloud)
		{
			minX = std::min(minX, point.x);
			maxX = std::max(maxX, point.x);
			minY = std::min(minY, point.y);
			maxY = std::max(maxY, point.y);
			minZ = std::min(minZ, point.z);
			maxZ = std::max(maxZ, point.z);
		}
		const double dx = maxX - minX;
		const double dy = maxY - minY;
		const double dz = maxZ - minZ;
		const double longest = std::max({ dx, dy, dz, 1.0 });
		return std::max(longest / 128.0, 1.0e-4);
	}

	std::vector<Sample> runKdTree(const Cloud::Ptr& cloud, const std::vector<Query>& queries, double& buildMs)
	{
		pcl::KdTreeFLANN<Point> tree;
		const auto buildStart = Clock::now();
		tree.setInputCloud(cloud);
		buildMs = elapsedMs(buildStart, Clock::now());

		std::vector<Sample> samples;
		std::vector<int> indices;
		std::vector<float> distances;
		for (const Query& query : queries)
		{
			if (query.kind == "range")
				continue;
			const Point center(query.center.x, query.center.y, query.center.z);
			const auto start = Clock::now();
			int returned = 0;
			if (query.kind == "radius")
				returned = tree.radiusSearch(center, query.radius, indices, distances);
			else if (query.kind == "knn")
				returned = tree.nearestKSearch(center, query.k, indices, distances);
			const double latency = elapsedMs(start, Clock::now());
			samples.push_back({ query.kind, latency, returned, query.expectedCount });
		}
		return samples;
	}

	std::vector<Sample> runOctree(const Cloud::Ptr& cloud, const std::vector<Query>& queries, double resolution, double& buildMs)
	{
		pcl::octree::OctreePointCloudSearch<Point> octree(static_cast<float>(resolution));
		const auto buildStart = Clock::now();
		octree.setInputCloud(cloud);
		octree.addPointsFromInputCloud();
		buildMs = elapsedMs(buildStart, Clock::now());

		std::vector<Sample> samples;
		std::vector<int> indices;
		std::vector<float> distances;
		for (const Query& query : queries)
		{
			const auto start = Clock::now();
			int returned = 0;
			if (query.kind == "range")
			{
				const Point minPoint(query.minBound.x, query.minBound.y, query.minBound.z);
				const Point maxPoint(query.maxBound.x, query.maxBound.y, query.maxBound.z);
				returned = static_cast<int>(octree.boxSearch(minPoint, maxPoint, indices));
			}
			else if (query.kind == "radius")
			{
				const Point center(query.center.x, query.center.y, query.center.z);
				returned = static_cast<int>(octree.radiusSearch(center, query.radius, indices, distances));
			}
			else if (query.kind == "knn")
			{
				const Point center(query.center.x, query.center.y, query.center.z);
				returned = static_cast<int>(octree.nearestKSearch(center, query.k, indices, distances));
			}
			const double latency = elapsedMs(start, Clock::now());
			samples.push_back({ query.kind, latency, returned, query.expectedCount });
		}
		return samples;
	}

	double percentile(std::vector<double> values, double pct)
	{
		if (values.empty())
			return std::numeric_limits<double>::quiet_NaN();
		std::sort(values.begin(), values.end());
		const double pos = (values.size() - 1) * pct;
		const size_t lo = static_cast<size_t>(std::floor(pos));
		const size_t hi = static_cast<size_t>(std::ceil(pos));
		if (lo == hi)
			return values[lo];
		return values[lo] * (hi - pos) + values[hi] * (pos - lo);
	}

	void writeResult(std::ostream& out, const std::string& method, const std::vector<Sample>& samples, double buildMs, bool trailingComma)
	{
		if (samples.empty())
		{
			out << "    {\n";
			out << "      \"method\": \"" << jsonEscape(method) << "\",\n";
			out << "      \"status\": \"skipped\",\n";
			out << "      \"reason\": \"no supported queries for this PCL baseline\"\n";
			out << "    }" << (trailingComma ? "," : "") << "\n";
			return;
		}

		std::vector<double> latencies;
		int mismatches = 0;
		int maxDelta = 0;
		std::map<std::string, std::vector<double>> byType;
		for (const Sample& sample : samples)
		{
			latencies.push_back(sample.latencyMs);
			byType[sample.kind].push_back(sample.latencyMs);
			if (sample.expectedCount >= 0 && sample.expectedCount != sample.returnedCount)
			{
				++mismatches;
				maxDelta = std::max(maxDelta, std::abs(sample.expectedCount - sample.returnedCount));
			}
		}
		const double avg = latencies.empty()
			? std::numeric_limits<double>::quiet_NaN()
			: std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

		out << "    {\n";
		out << "      \"method\": \"" << jsonEscape(method) << "\",\n";
		out << "      \"status\": \"ok\",\n";
		out << "      \"build_ms\": " << buildMs << ",\n";
		out << "      \"queries\": " << samples.size() << ",\n";
		out << "      \"avg_latency_ms\": " << avg << ",\n";
		out << "      \"p95_latency_ms\": " << percentile(latencies, 0.95) << ",\n";
		out << "      \"count_mismatches\": " << mismatches << ",\n";
		out << "      \"max_count_delta\": " << maxDelta << ",\n";
		out << "      \"by_type\": {\n";
		bool first = true;
		for (const auto& item : byType)
		{
			if (!first)
				out << ",\n";
			first = false;
			const double typeAvg = std::accumulate(item.second.begin(), item.second.end(), 0.0) / item.second.size();
			out << "        \"" << jsonEscape(item.first) << "\": {\"queries\": " << item.second.size()
				<< ", \"avg_latency_ms\": " << typeAvg
				<< ", \"p95_latency_ms\": " << percentile(item.second, 0.95) << "}";
		}
		out << "\n      },\n";
		out << "      \"note\": \"" << (method == "pcl_kdtree" ? "KdTreeFLANN radius/KNN only." : "OctreePointCloudSearch range/radius/KNN.") << "\"\n";
		out << "    }" << (trailingComma ? "," : "") << "\n";
	}
}

int main(int argc, char** argv)
{
	try
	{
		const Args args = parseArgs(argc, argv);
		Cloud::Ptr cloud = loadCloud(args.input);
		if (cloud->empty())
			throw std::runtime_error("point cloud is empty");
		const std::vector<Query> queries = loadQueries(args.queryTrace);
		const double resolution = args.octreeResolution > 0.0 ? args.octreeResolution : defaultOctreeResolution(*cloud);

		double kdBuildMs = 0.0;
		double octreeBuildMs = 0.0;
		const std::vector<Sample> kdSamples = runKdTree(cloud, queries, kdBuildMs);
		const std::vector<Sample> octreeSamples = runOctree(cloud, queries, resolution, octreeBuildMs);

		std::ofstream output(args.output);
		if (!output.is_open())
			throw std::runtime_error("unable to open output: " + args.output);
		output << "{\n";
		output << "  \"input\": \"" << jsonEscape(args.input) << "\",\n";
		output << "  \"query_trace\": \"" << jsonEscape(args.queryTrace) << "\",\n";
		output << "  \"octree_resolution\": " << resolution << ",\n";
		output << "  \"results\": [\n";
		writeResult(output, "pcl_kdtree", kdSamples, kdBuildMs, true);
		writeResult(output, "pcl_octree", octreeSamples, octreeBuildMs, false);
		output << "  ]\n";
		output << "}\n";
		return 0;
	}
	catch (const std::exception& exception)
	{
		std::cerr << "Error: " << exception.what() << '\n';
		return 1;
	}
}

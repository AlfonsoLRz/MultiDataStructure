// External replay baseline against nanoflann (github.com/jlblancoc/nanoflann, BSD, the header from
// vcpkg, 1.10.x): the widely used header-only C++ kd-tree, single-threaded build and queries.
//
// Same protocol as tools/indexicon_point_baseline.cpp, so scripts/head_to_head.ps1 can put the
// three libraries in one table: the cloud from its .mdspc cache, the recorded trace subset with
// the same integer stride, untimed warm-up passes, then timed passes pooled per query kind, each
// query timed individually around the search call with result buffers reused across queries.
//
// Points are kept as float in the cloud's local frame - the representation MultiDataStructure
// queries - and trace centres (world, double) are moved into that frame in double first. Radius
// queries ask for unsorted results (nanoflann sorts by default, which is extra work the other
// libraries are not asked to do); kNN returns the k nearest as nanoflann always does.
//
// Build:
//   powershell -File tools\build_nanoflann_baseline.ps1
// Usage:
//   nanoflann_point_baseline --input <cloud.mdspc|.las> --query-trace <trace.csv> --output <result.json>
//                            [--max-queries N] [--warmup W] [--repeats R] [--leaf-sizes 10,128]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanoflann.hpp>

namespace
{
	using Clock = std::chrono::steady_clock;

	struct Cloud
	{
		std::vector<float> xyz;	// interleaved, local frame
		double origin[3] = { 0, 0, 0 };
		double scale[3] = { 1, 1, 1 };

		size_t kdtree_get_point_count() const { return xyz.size() / 3; }
		float kdtree_get_pt(size_t i, size_t dim) const { return xyz[i * 3 + dim]; }
		template <class BBox> bool kdtree_get_bbox(BBox&) const { return false; }
	};

	Cloud loadMdspc(const std::string& path)
	{
		std::ifstream in(path, std::ios::binary);
		if (!in) throw std::runtime_error("cannot open " + path);
		char header[88] = {};
		in.read(header, 40);
		if (!in || std::memcmp(header, "MDSPC01", 7) != 0) throw std::runtime_error("not an .mdspc cache: " + path);
		uint32_t version = 0;
		uint64_t count = 0;
		std::memcpy(&version, header + 8, sizeof(version));
		std::memcpy(&count, header + 32, sizeof(count));
		Cloud cloud;
		if (version == 3)
		{
			in.read(header + 40, 48);
			std::memcpy(cloud.origin, header + 40, sizeof(cloud.origin));
			std::memcpy(cloud.scale, header + 64, sizeof(cloud.scale));
		}
		else if (version != 2)
			throw std::runtime_error("unsupported .mdspc version " + std::to_string(version));
		cloud.xyz.resize(static_cast<size_t>(count) * 3);
		in.read(reinterpret_cast<char*>(cloud.xyz.data()), static_cast<std::streamsize>(cloud.xyz.size() * sizeof(float)));
		if (!in) throw std::runtime_error("truncated .mdspc payload: " + path);
		return cloud;
	}

	struct Query
	{
		std::string kind;
		float center[3] = { 0, 0, 0 };
		float radius = 0.0f;
		size_t k = 0;
	};

	std::vector<std::string> split(const std::string& line)
	{
		std::vector<std::string> out;
		std::string cell;
		std::istringstream ss(line);
		while (std::getline(ss, cell, ','))
			out.push_back(cell);
		return out;
	}

	std::vector<Query> loadTrace(const std::string& path, const Cloud& cloud, size_t maxQueries)
	{
		std::ifstream in(path);
		if (!in) throw std::runtime_error("cannot open " + path);
		std::string line;
		std::getline(in, line);
		const std::vector<std::string> header = split(line);
		const auto col = [&](const char* name) {
			const auto it = std::find(header.begin(), header.end(), name);
			if (it == header.end()) throw std::runtime_error(std::string("trace missing column ") + name);
			return static_cast<size_t>(it - header.begin());
		};
		const size_t cType = col("query_type"), cx = col("center_x"), cy = col("center_y"), cz = col("center_z");
		const size_t cRadius = col("radius"), cK = col("k");
		std::vector<Query> all;
		while (std::getline(in, line))
		{
			const std::vector<std::string> f = split(line);
			if (f.size() <= std::max({ cType, cx, cy, cz, cRadius, cK }))
				continue;
			Query q;
			q.kind = f[cType];
			const double world[3] = { std::stod(f[cx]), std::stod(f[cy]), std::stod(f[cz]) };
			for (int a = 0; a < 3; ++a)
				q.center[a] = static_cast<float>((world[a] - cloud.origin[a]) / cloud.scale[a]);
			q.radius = static_cast<float>(std::stod(f[cRadius]));
			q.k = static_cast<size_t>(std::stod(f[cK]));
			all.push_back(q);
		}
		if (maxQueries == 0 || all.size() <= maxQueries)
			return all;
		std::vector<Query> subset;
		for (size_t i = 0; i < maxQueries; ++i)
			subset.push_back(all[(i * all.size()) / maxQueries]);
		return subset;
	}

	using Tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, Cloud>, Cloud, 3, uint32_t>;

	struct Measured
	{
		std::string method;
		double buildMs = 0.0;
		size_t queries = 0, repeats = 0;
		std::map<std::string, std::vector<double>> latencies;
	};

	Measured run(const Cloud& cloud, const std::vector<Query>& queries, size_t leafSize, size_t warmup, size_t repeats)
	{
		Measured m;
		m.method = "nanoflann_kdtree_leaf" + std::to_string(leafSize);
		const auto t0 = Clock::now();
		Tree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(leafSize));
		m.buildMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

		std::vector<nanoflann::ResultItem<uint32_t, float>> hits;
		std::vector<uint32_t> knnIndices;
		std::vector<float> knnDistances;
		nanoflann::SearchParameters unsorted;
		unsorted.sorted = false;
		for (size_t pass = 0; pass < warmup + repeats; ++pass)
		{
			for (const Query& q : queries)
			{
				double ms = 0.0;
				if (q.kind == "radius")
				{
					const auto start = Clock::now();
					// nanoflann's L2 metric works in squared distances.
					(void)tree.radiusSearch(q.center, q.radius * q.radius, hits, unsorted);
					ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
				}
				else if (q.kind == "knn")
				{
					knnIndices.resize(q.k);
					knnDistances.resize(q.k);
					const auto start = Clock::now();
					(void)tree.knnSearch(q.center, q.k, knnIndices.data(), knnDistances.data());
					ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
				}
				else
					continue;
				if (pass >= warmup)
					m.latencies[q.kind].push_back(ms);
			}
		}
		m.queries = queries.size();
		m.repeats = repeats;
		return m;
	}
}

int main(int argc, char** argv)
{
	try
	{
		std::string input, trace, output;
		size_t maxQueries = 0, warmup = 1, repeats = 3;
		std::vector<size_t> leafSizes = { 10, 128 };
		for (int i = 1; i < argc; ++i)
		{
			const std::string arg = argv[i];
			const auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg); return argv[++i]; };
			if (arg == "--input") input = next();
			else if (arg == "--query-trace") trace = next();
			else if (arg == "--output") output = next();
			else if (arg == "--max-queries") maxQueries = std::stoull(next());
			else if (arg == "--warmup") warmup = std::stoull(next());
			else if (arg == "--repeats") repeats = std::stoull(next());
			else if (arg == "--leaf-sizes")
			{
				leafSizes.clear();
				std::istringstream ss(next());
				std::string item;
				while (std::getline(ss, item, ','))
					leafSizes.push_back(std::stoull(item));
			}
			else throw std::runtime_error("unknown argument " + arg);
		}
		if (input.size() > 4 && input.substr(input.size() - 4) == ".las")
			input += ".mdspc";
		const Cloud cloud = loadMdspc(input);
		const std::vector<Query> queries = loadTrace(trace, cloud, maxQueries);
		std::cout << "nanoflann " << NANOFLANN_VERSION_STRING << ": " << cloud.kdtree_get_point_count() << " points, "
				  << queries.size() << " queries, " << warmup << " warm-up + " << repeats << " timed passes\n";

		std::vector<Measured> results;
		for (const size_t leaf : leafSizes)
		{
			results.push_back(run(cloud, queries, leaf, warmup, repeats));
			const Measured& m = results.back();
			std::cout << m.method << ": build " << m.buildMs << " ms";
			for (const auto& [kind, values] : m.latencies)
				std::cout << " | " << kind << " " << 1000.0 * std::accumulate(values.begin(), values.end(), 0.0) / values.size() << " us";
			std::cout << '\n';
		}

		if (!output.empty())
		{
			std::ofstream out(output);
			out.precision(10);
			out << "{\n  \"library\": \"nanoflann\",\n  \"library_version\": \"" << NANOFLANN_VERSION_STRING << "\",\n";
			out << "  \"points\": " << cloud.kdtree_get_point_count() << ",\n  \"results\": [\n";
			for (size_t r = 0; r < results.size(); ++r)
			{
				const Measured& m = results[r];
				out << "    {\n      \"method\": \"" << m.method << "\",\n      \"status\": \"ok\",\n";
				out << "      \"build_ms\": " << m.buildMs << ",\n      \"queries\": " << m.queries << ",\n      \"repeats\": " << m.repeats << ",\n";
				out << "      \"by_type\": {";
				bool first = true;
				for (const auto& [kind, values] : m.latencies)
				{
					out << (first ? "\n" : ",\n") << "        \"" << kind << "\": {\"queries\": " << values.size()
						<< ", \"avg_latency_ms\": " << std::accumulate(values.begin(), values.end(), 0.0) / values.size() << "}";
					first = false;
				}
				out << "\n      }\n    }" << (r + 1 < results.size() ? "," : "") << "\n";
			}
			out << "  ]\n}\n";
		}
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << '\n';
		return 1;
	}
}

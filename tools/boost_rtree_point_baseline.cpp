// External replay baseline against Boost.Geometry's R-tree (boost::geometry::index::rtree, Boost
// Software License, header-only, from vcpkg): the R-tree reviewers of a spatial-index paper expect,
// bulk-loaded with the STR packing algorithm its range constructor uses, single-threaded.
//
// Same protocol as tools/nanoflann_point_baseline.cpp, so scripts/head_to_head.ps1 can put it in
// the same table: the cloud from its .mdspc cache, the recorded trace subset with the same integer
// stride, untimed warm-up passes, then timed passes pooled per query kind, each query timed
// individually around the search call with the result buffer reused across queries.
//
// Points are float in the cloud's local frame, as MultiDataStructure queries them. Boost has no
// sphere predicate, so a radius query is the usual box intersection plus a distance filter
// (bgi::intersects + bgi::satisfies); kNN uses bgi::nearest, which returns the k nearest unsorted.
//
// Build:
//   powershell -File tools\build_boost_rtree_baseline.ps1
// Usage:
//   boost_rtree_point_baseline --input <cloud.mdspc|.las> --query-trace <trace.csv> --output <result.json>
//                              [--max-queries N] [--warmup W] [--repeats R] [--max-elements 16,32]

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
#include <utility>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/version.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace
{
	using Clock = std::chrono::steady_clock;
	using Point = bg::model::point<float, 3, bg::cs::cartesian>;
	using Box = bg::model::box<Point>;
	using Value = std::pair<Point, uint32_t>;

	struct Cloud
	{
		std::vector<float> xyz;	// interleaved, local frame
		double origin[3] = { 0, 0, 0 };
		double scale[3] = { 1, 1, 1 };
		size_t size() const { return xyz.size() / 3; }
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

	struct Measured
	{
		std::string method;
		double buildMs = 0.0;
		size_t queries = 0, repeats = 0;
		std::map<std::string, std::vector<double>> latencies;
	};

	template <typename Parameters>
	Measured run(const std::string& method, const Cloud& cloud, const std::vector<Query>& queries, size_t warmup, size_t repeats)
	{
		Measured m;
		m.method = method;
		std::vector<Value> values;
		values.reserve(cloud.size());
		for (size_t i = 0; i < cloud.size(); ++i)
			values.emplace_back(Point(cloud.xyz[i * 3], cloud.xyz[i * 3 + 1], cloud.xyz[i * 3 + 2]), static_cast<uint32_t>(i));

		// The range constructor bulk-loads with STR packing; the value copy above is not timed.
		const auto t0 = Clock::now();
		bgi::rtree<Value, Parameters> tree(values.begin(), values.end());
		m.buildMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
		values.clear();
		values.shrink_to_fit();

		std::vector<Value> hits;
		for (size_t pass = 0; pass < warmup + repeats; ++pass)
		{
			for (const Query& q : queries)
			{
				hits.clear();
				const Point center(q.center[0], q.center[1], q.center[2]);
				double ms = 0.0;
				if (q.kind == "radius")
				{
					const float r = q.radius, r2 = q.radius * q.radius;
					const Box box(Point(q.center[0] - r, q.center[1] - r, q.center[2] - r), Point(q.center[0] + r, q.center[1] + r, q.center[2] + r));
					const auto start = Clock::now();
					tree.query(bgi::intersects(box) && bgi::satisfies([&](const Value& v) {
						const float dx = bg::get<0>(v.first) - q.center[0];
						const float dy = bg::get<1>(v.first) - q.center[1];
						const float dz = bg::get<2>(v.first) - q.center[2];
						return dx * dx + dy * dy + dz * dz <= r2;
					}), std::back_inserter(hits));
					ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
				}
				else if (q.kind == "knn")
				{
					const auto start = Clock::now();
					tree.query(bgi::nearest(center, static_cast<unsigned>(q.k)), std::back_inserter(hits));
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

	Measured runVariant(const std::string& kind, size_t maxElements, const Cloud& cloud, const std::vector<Query>& queries, size_t warmup, size_t repeats)
	{
		const std::string name = "boost_rtree_" + kind + std::to_string(maxElements) + "_str";
		// Node capacity is a compile-time parameter in Boost; instantiate the supported ones.
		if (kind == "rstar")
		{
			if (maxElements == 8) return run<bgi::rstar<8>>(name, cloud, queries, warmup, repeats);
			if (maxElements == 16) return run<bgi::rstar<16>>(name, cloud, queries, warmup, repeats);
			if (maxElements == 32) return run<bgi::rstar<32>>(name, cloud, queries, warmup, repeats);
			if (maxElements == 64) return run<bgi::rstar<64>>(name, cloud, queries, warmup, repeats);
		}
		else if (kind == "quadratic")
		{
			if (maxElements == 8) return run<bgi::quadratic<8>>(name, cloud, queries, warmup, repeats);
			if (maxElements == 16) return run<bgi::quadratic<16>>(name, cloud, queries, warmup, repeats);
			if (maxElements == 32) return run<bgi::quadratic<32>>(name, cloud, queries, warmup, repeats);
			if (maxElements == 64) return run<bgi::quadratic<64>>(name, cloud, queries, warmup, repeats);
		}
		throw std::runtime_error("unsupported R-tree variant " + kind + " " + std::to_string(maxElements) + " (8, 16, 32 or 64)");
	}
}

int main(int argc, char** argv)
{
	try
	{
		std::string input, trace, output;
		size_t maxQueries = 0, warmup = 1, repeats = 3;
		std::vector<size_t> maxElements = { 16, 32 };
		std::vector<std::string> kinds = { "rstar", "quadratic" };
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
			else if (arg == "--max-elements")
			{
				maxElements.clear();
				std::istringstream ss(next());
				std::string item;
				while (std::getline(ss, item, ','))
					maxElements.push_back(std::stoull(item));
			}
			else throw std::runtime_error("unknown argument " + arg);
		}
		if (input.size() > 4 && input.substr(input.size() - 4) == ".las")
			input += ".mdspc";
		const Cloud cloud = loadMdspc(input);
		const std::vector<Query> queries = loadTrace(trace, cloud, maxQueries);
		const std::string version = std::to_string(BOOST_VERSION / 100000) + "." + std::to_string(BOOST_VERSION / 100 % 1000);
		std::cout << "boost rtree " << version << ": " << cloud.size() << " points, " << queries.size() << " queries, "
				  << warmup << " warm-up + " << repeats << " timed passes\n";

		std::vector<Measured> results;
		for (const std::string& kind : kinds)
			for (const size_t elements : maxElements)
			{
				results.push_back(runVariant(kind, elements, cloud, queries, warmup, repeats));
				const Measured& m = results.back();
				std::cout << m.method << ": build " << m.buildMs << " ms";
				for (const auto& [k, values] : m.latencies)
					std::cout << " | " << k << " " << 1000.0 * std::accumulate(values.begin(), values.end(), 0.0) / values.size() << " us";
				std::cout << '\n';
			}

		if (!output.empty())
		{
			std::ofstream out(output);
			out.precision(10);
			out << "{\n  \"library\": \"boost_rtree\",\n  \"library_version\": \"" << version << "\",\n";
			out << "  \"points\": " << cloud.size() << ",\n  \"results\": [\n";
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

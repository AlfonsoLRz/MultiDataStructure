// External replay baseline against Indexicon (github.com/psimatis/Indexicon-Spatial-Library,
// MIT). Replays a recorded MDS query trace (--query-trace format) against Indexicon's
// bulk-loaded octree and kd-tree and reports per-type latency plus returned-count parity
// versus the trace's recorded counts. Indexicon has no radius queries, so radius rows are
// skipped and counted; the comparable subset is range + kNN.
//
// Build (needs the clone at external/Indexicon):
//   cl /nologo /std:c++17 /EHsc /O2 /MD /DNOMINMAX /I external\Indexicon
//     tools\indexicon_point_baseline.cpp /Fe:tools\bin\indexicon_point_baseline.exe
// Usage:
//   indexicon_point_baseline.exe <cloud.ply|.xyz> <trace.csv> [max_queries]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "indexes/octtree.hpp"
#include "indexes/kdtree.hpp"

namespace
{
	using Clock = std::chrono::steady_clock;

	double elapsedMs(Clock::time_point start, Clock::time_point end)
	{
		return std::chrono::duration<double, std::milli>(end - start).count();
	}

	struct XYZ { double x, y, z; };

	// Minimal PLY reader: ascii or binary_little_endian, float/double properties,
	// unknown vertex properties skipped by stride.
	std::vector<XYZ> loadPly(const std::string& path)
	{
		std::ifstream in(path, std::ios::binary);
		if (!in) throw std::runtime_error("cannot open " + path);
		std::string line, format;
		size_t vertexCount = 0;
		struct Prop { std::string type; std::string name; };
		std::vector<Prop> props;
		bool inVertex = false;
		while (std::getline(in, line))
		{
			if (!line.empty() && line.back() == '\r') line.pop_back();
			std::istringstream ss(line);
			std::string tok; ss >> tok;
			if (tok == "format") { ss >> format; }
			else if (tok == "element") { std::string n; ss >> n >> vertexCount; inVertex = (n == "vertex"); }
			else if (tok == "property" && inVertex) { Prop p; ss >> p.type >> p.name; if (p.type != "list") props.push_back(p); }
			else if (tok == "end_header") break;
		}
		auto typeSize = [](const std::string& t) -> size_t {
			if (t == "double" || t == "float64") return 8;
			if (t == "float" || t == "float32" || t == "int" || t == "uint" || t == "int32" || t == "uint32") return 4;
			if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
			return 1;
		};
		int xi = -1, yi = -1, zi = -1;
		size_t stride = 0;
		std::vector<size_t> offsets;
		for (size_t i = 0; i < props.size(); ++i)
		{
			offsets.push_back(stride);
			if (props[i].name == "x") xi = static_cast<int>(i);
			if (props[i].name == "y") yi = static_cast<int>(i);
			if (props[i].name == "z") zi = static_cast<int>(i);
			stride += typeSize(props[i].type);
		}
		if (xi < 0 || yi < 0 || zi < 0) throw std::runtime_error("PLY missing x/y/z");

		std::vector<XYZ> pts;
		pts.reserve(vertexCount);
		auto readAt = [&](const char* row, size_t propIndex) -> double {
			const char* p = row + offsets[propIndex];
			const std::string& t = props[propIndex].type;
			if (t == "double" || t == "float64") { double v; std::memcpy(&v, p, 8); return v; }
			float v; std::memcpy(&v, p, 4); return static_cast<double>(v);
		};
		if (format == "binary_little_endian")
		{
			std::vector<char> row(stride);
			for (size_t i = 0; i < vertexCount; ++i)
			{
				in.read(row.data(), static_cast<std::streamsize>(stride));
				if (!in) throw std::runtime_error("truncated PLY");
				pts.push_back({ readAt(row.data(), xi), readAt(row.data(), yi), readAt(row.data(), zi) });
			}
		}
		else
		{
			for (size_t i = 0; i < vertexCount && std::getline(in, line); ++i)
			{
				std::istringstream ss(line);
				std::vector<double> vals(props.size());
				for (double& v : vals) ss >> v;
				pts.push_back({ vals[xi], vals[yi], vals[zi] });
			}
		}
		return pts;
	}

	std::vector<XYZ> loadXyz(const std::string& path)
	{
		std::ifstream in(path);
		if (!in) throw std::runtime_error("cannot open " + path);
		std::vector<XYZ> pts;
		double x, y, z;
		std::string rest;
		while (in >> x >> y >> z) { pts.push_back({ x, y, z }); std::getline(in, rest); }
		return pts;
	}

	struct TraceQuery
	{
		std::string kind;
		double minB[3] = {}, maxB[3] = {}, center[3] = {};
		double radius = 0.0;
		size_t k = 0;
		long long expected = -1;
	};

	std::vector<TraceQuery> loadTrace(const std::string& path)
	{
		std::ifstream in(path);
		if (!in) throw std::runtime_error("cannot open " + path);
		std::string header;
		std::getline(in, header);
		std::vector<std::string> cols;
		{
			std::istringstream ss(header);
			std::string c;
			while (std::getline(ss, c, ',')) cols.push_back(c);
		}
		std::map<std::string, int> idx;
		for (size_t i = 0; i < cols.size(); ++i) idx[cols[i]] = static_cast<int>(i);
		auto need = [&](const char* n) {
			auto it = idx.find(n);
			if (it == idx.end()) throw std::runtime_error(std::string("trace missing column ") + n);
			return it->second;
		};
		const int cType = need("query_type");
		const int cMin[3] = { need("bounds_min_x"), need("bounds_min_y"), need("bounds_min_z") };
		const int cMax[3] = { need("bounds_max_x"), need("bounds_max_y"), need("bounds_max_z") };
		const int cCen[3] = { need("center_x"), need("center_y"), need("center_z") };
		const int cRad = need("radius"), cK = need("k");
		const int cRet = idx.count("returned_points") ? idx["returned_points"] : -1;

		std::vector<TraceQuery> out;
		std::string line;
		while (std::getline(in, line))
		{
			if (line.empty()) continue;
			std::vector<std::string> f;
			{
				std::istringstream ss(line);
				std::string c;
				while (std::getline(ss, c, ',')) f.push_back(c);
			}
			if (f.size() < cols.size() - 1) continue;
			TraceQuery q;
			q.kind = f[cType];
			auto num = [&](int c) { return f[c].empty() ? 0.0 : std::stod(f[c]); };
			for (int a = 0; a < 3; ++a) { q.minB[a] = num(cMin[a]); q.maxB[a] = num(cMax[a]); q.center[a] = num(cCen[a]); }
			q.radius = num(cRad);
			q.k = static_cast<size_t>(num(cK));
			if (cRet >= 0 && !f[cRet].empty()) q.expected = static_cast<long long>(std::stoll(f[cRet]));
			out.push_back(std::move(q));
		}
		return out;
	}

	struct TypeAgg { double totalMs = 0; size_t n = 0, mismatches = 0; long long maxDelta = 0; };

	void report(const char* method, double buildMs, const std::map<std::string, TypeAgg>& agg, size_t skippedRadius)
	{
		std::cout << method << ": build " << buildMs << " ms";
		if (skippedRadius) std::cout << " | radius queries skipped (unsupported): " << skippedRadius;
		std::cout << '\n';
		for (const auto& [kind, a] : agg)
		{
			std::cout << "  " << kind << ": n=" << a.n
					  << " avg_ms=" << (a.n ? a.totalMs / a.n : 0.0)
					  << " count_mismatches=" << a.mismatches
					  << " max_delta=" << a.maxDelta << '\n';
		}
	}
}

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		std::cerr << "usage: indexicon_point_baseline <cloud.ply|.xyz> <trace.csv> [max_queries]\n";
		return 1;
	}
	try
	{
		const std::string cloudPath = argv[1], tracePath = argv[2];
		const size_t maxQueries = argc > 3 ? std::stoull(argv[3]) : 0;

		const std::vector<XYZ> pts = cloudPath.size() > 4 && cloudPath.substr(cloudPath.size() - 4) == ".ply"
			? loadPly(cloudPath) : loadXyz(cloudPath);
		std::cout << "loaded " << pts.size() << " points\n";
		std::vector<TraceQuery> queries = loadTrace(tracePath);
		if (maxQueries && queries.size() > maxQueries)
		{
			// deterministic even stride, matching the MDS replayer's subsetting
			std::vector<TraceQuery> subset;
			const double step = static_cast<double>(queries.size()) / static_cast<double>(maxQueries);
			for (size_t i = 0; i < maxQueries; ++i) subset.push_back(queries[static_cast<size_t>(i * step)]);
			queries.swap(subset);
		}
		std::cout << "replaying " << queries.size() << " queries from " << tracePath << '\n';

		constexpr size_t Capacity = 128; // Indexicon's default leaf capacity

		// ---------- Indexicon octree ----------
		{
			using Tree = OctreeNode<double, size_t, Capacity>;
			using Pt = OPoint<double, size_t>;
			std::vector<Pt> bulk;
			bulk.reserve(pts.size());
			for (size_t i = 0; i < pts.size(); ++i) bulk.emplace_back(i, pts[i].x, pts[i].y, pts[i].z);
			const auto b0 = Clock::now();
			Tree tree(bulk.begin(), bulk.end());
			const double buildMs = elapsedMs(b0, Clock::now());

			std::map<std::string, TypeAgg> agg;
			size_t skipped = 0;
			std::vector<size_t> results;
			for (const TraceQuery& q : queries)
			{
				if (q.kind == "range" || q.kind == "count_range")
				{
					results.clear();
					const OBox<double, size_t> box(q.minB[0], q.minB[1], q.minB[2], q.maxB[0], q.maxB[1], q.maxB[2]);
					const auto s = Clock::now();
					tree.query(box, std::back_inserter(results));
					TypeAgg& a = agg["range"];
					a.totalMs += elapsedMs(s, Clock::now());
					++a.n;
					if (q.expected >= 0 && static_cast<long long>(results.size()) != q.expected)
					{ ++a.mismatches; a.maxDelta = std::max(a.maxDelta, std::llabs(static_cast<long long>(results.size()) - q.expected)); }
				}
				else if (q.kind == "knn")
				{
					results.clear();
					const std::array<double, 3> c{ q.center[0], q.center[1], q.center[2] };
					const auto s = Clock::now();
					tree.knnQuery(c, q.k, std::back_inserter(results));
					TypeAgg& a = agg["knn"];
					a.totalMs += elapsedMs(s, Clock::now());
					++a.n;
					if (static_cast<long long>(results.size()) != static_cast<long long>(q.k))
					{ ++a.mismatches; a.maxDelta = std::max(a.maxDelta, std::llabs(static_cast<long long>(results.size()) - static_cast<long long>(q.k))); }
				}
				else ++skipped;
			}
			report("indexicon_octree", buildMs, agg, skipped);
		}

		// ---------- Indexicon kd-tree ----------
		{
			using Tree = KDTree<double, size_t, 3, Capacity, KDSplit::ADAPTIVE>;
			using Pt = Tree::Point;
			std::vector<std::pair<Pt, size_t>> bulk;
			bulk.reserve(pts.size());
			for (size_t i = 0; i < pts.size(); ++i) bulk.emplace_back(Pt(pts[i].x, pts[i].y, pts[i].z), i);
			const auto b0 = Clock::now();
			Tree tree(bulk.begin(), bulk.end());
			const double buildMs = elapsedMs(b0, Clock::now());

			std::map<std::string, TypeAgg> agg;
			size_t skipped = 0;
			std::vector<size_t> results;
			for (const TraceQuery& q : queries)
			{
				if (q.kind == "range" || q.kind == "count_range")
				{
					results.clear();
					typename Tree::Region box;
					box.low = Pt(q.minB[0], q.minB[1], q.minB[2]);
					box.high = Pt(q.maxB[0], q.maxB[1], q.maxB[2]);
					const auto s = Clock::now();
					tree.query(box, std::back_inserter(results));
					TypeAgg& a = agg["range"];
					a.totalMs += elapsedMs(s, Clock::now());
					++a.n;
					if (q.expected >= 0 && static_cast<long long>(results.size()) != q.expected)
					{ ++a.mismatches; a.maxDelta = std::max(a.maxDelta, std::llabs(static_cast<long long>(results.size()) - q.expected)); }
				}
				else if (q.kind == "knn")
				{
					results.clear();
					const auto s = Clock::now();
					tree.knnQuery(Pt(q.center[0], q.center[1], q.center[2]), q.k, std::back_inserter(results));
					TypeAgg& a = agg["knn"];
					a.totalMs += elapsedMs(s, Clock::now());
					++a.n;
					if (static_cast<long long>(results.size()) != static_cast<long long>(q.k))
					{ ++a.mismatches; a.maxDelta = std::max(a.maxDelta, std::llabs(static_cast<long long>(results.size()) - static_cast<long long>(q.k))); }
				}
				else ++skipped;
			}
			report("indexicon_kdtree", buildMs, agg, skipped);
		}
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << '\n';
		return 1;
	}
}

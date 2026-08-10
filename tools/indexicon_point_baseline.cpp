// External replay baseline against Indexicon (github.com/psimatis/Indexicon-Spatial-Library,
// MIT, pinned at c9f9b1d). Replays a recorded MDS query trace against Indexicon's bulk-loaded
// octree, kd-tree and R-tree, reporting per-type latency plus returned-count parity versus the
// trace's recorded counts.
//
// Radius queries. Indexicon exposes no radius primitive, so earlier revisions of this driver
// skipped them - which dropped the majority query type of our pipeline traces. Two emulations
// are implemented here and selected with --radius-mode:
//   native_mbr_prune (default) - a recursive traversal written against each structure's own
//       public node interface, pruning a subtree when its minimum squared distance to the query
//       centre exceeds r^2. Each structure is driven through the very same min-distance helper
//       its own kNN search uses (OctreeNode::minSqrDist, kdMindistToRegion, mindistPointToBox),
//       so this is the sphere query their own code would write, not a handicapped stand-in.
//   aabb_filter - native box query over the sphere's bounding box, then an exact distance
//       filter. Simpler, and what a user of the library would reach for, but a sphere occupies
//       only pi/6 (~52%) of its bounding box, so it tests roughly twice the points it needs.
// The mode is recorded per result; numbers from different modes are not comparable.
//
// Build (fetches/pins the clone if missing):
//   powershell -File tools\build_indexicon_baseline.ps1
// Manual equivalent:
//   cl /nologo /std:c++17 /EHsc /O2 /MD /DNOMINMAX /I external\Indexicon
//     tools\indexicon_point_baseline.cpp /Fe:tools\bin\indexicon_point_baseline.exe
//
// Usage:
//   indexicon_point_baseline --input <cloud> --query-trace <trace.csv> --output <result.json>
//                            [--max-queries N] [--repeats R] [--structures octree,kdtree,rtree]
//                            [--radius-mode native_mbr_prune|aabb_filter] [--verify-bruteforce N]
//   indexicon_point_baseline <cloud> <trace.csv> [max_queries]      (legacy positional form)
//
// <cloud> may be .ply, .xyz or .mdspc. A .las/.laz path resolves to its sibling <path>.mdspc
// binary cache, which MultiDataStructure writes beside every source it loads; this is the only
// way to feed the driver the LAS cells the evaluation battery actually uses.
//
// The JSON payload matches tools/pcl_point_baseline.cpp so scripts/compare_frameworks.py can
// splice Indexicon rows into the same table as Open3D and PCL.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "indexes/octtree.hpp"
#include "indexes/kdtree.hpp"
#include "indexes/rtree_point.hpp"

namespace
{
	using Clock = std::chrono::steady_clock;

	constexpr size_t Capacity = 128; // Indexicon's default leaf capacity

	double elapsedMs(Clock::time_point start, Clock::time_point end)
	{
		return std::chrono::duration<double, std::milli>(end - start).count();
	}

	struct XYZ { double x, y, z; };

	// ---------------------------------------------------------------- cloud loading

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

	// MultiDataStructure's .mdspc position cache (PointCloud.cpp). Layout, read at explicit
	// offsets rather than trusting struct packing:
	//   [0]  char   magic[8] = "MDSPC01\0"
	//   [8]  uint32 version                (3 = position-only + coordinate frame)
	//   [16] uint64 sourceSize
	//   [24] int64  sourceWriteTime
	//   [32] uint64 numPoints
	//   [40] double origin[3], double scale[3]        (version 3 only)
	//   [88] float  x, y, z  per point
	// Stored positions are in the cloud's local frame; world = origin + local * scale, matching
	// PointCloud::toWorldPosition. Traces are recorded in world coordinates, so this conversion
	// is mandatory - skipping it is the frame bug that invalidated an entire round of results.
	std::vector<XYZ> loadMdspc(const std::string& path)
	{
		std::ifstream in(path, std::ios::binary);
		if (!in) throw std::runtime_error("cannot open " + path);

		char header[88] = {};
		in.read(header, 40);
		if (!in) throw std::runtime_error("truncated .mdspc header: " + path);
		if (std::memcmp(header, "MDSPC01", 7) != 0) throw std::runtime_error("not an .mdspc cache: " + path);

		uint32_t version = 0;
		uint64_t numPoints = 0;
		std::memcpy(&version, header + 8, sizeof(version));
		std::memcpy(&numPoints, header + 32, sizeof(numPoints));
		if (version != 3 && version != 2)
			throw std::runtime_error("unsupported .mdspc version " + std::to_string(version) +
				" (expected 2 or 3); rebuild the cache with the current MultiDataStructure build");

		double origin[3] = { 0.0, 0.0, 0.0 };
		double scale[3] = { 1.0, 1.0, 1.0 };
		if (version == 3)
		{
			in.read(header + 40, 48);
			if (!in) throw std::runtime_error("truncated .mdspc metadata: " + path);
			std::memcpy(origin, header + 40, sizeof(origin));
			std::memcpy(scale, header + 64, sizeof(scale));
		}

		std::vector<XYZ> pts;
		pts.reserve(static_cast<size_t>(numPoints));
		constexpr size_t ChunkPoints = 1 << 20;
		std::vector<float> chunk(ChunkPoints * 3);
		uint64_t remaining = numPoints;
		while (remaining > 0)
		{
			const size_t current = static_cast<size_t>(std::min<uint64_t>(remaining, ChunkPoints));
			in.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(current * 3 * sizeof(float)));
			if (!in) throw std::runtime_error("truncated .mdspc payload: " + path);
			for (size_t i = 0; i < current; ++i)
			{
				pts.push_back({ origin[0] + static_cast<double>(chunk[i * 3 + 0]) * scale[0],
								origin[1] + static_cast<double>(chunk[i * 3 + 1]) * scale[1],
								origin[2] + static_cast<double>(chunk[i * 3 + 2]) * scale[2] });
			}
			remaining -= current;
		}
		return pts;
	}

	std::string lowerExtension(const std::string& path)
	{
		const size_t dot = path.find_last_of('.');
		if (dot == std::string::npos) return {};
		std::string ext = path.substr(dot);
		std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return ext;
	}

	bool fileExists(const std::string& path)
	{
		std::ifstream probe(path, std::ios::binary);
		return probe.good();
	}

	std::vector<XYZ> loadCloud(const std::string& path, std::string& resolvedPath)
	{
		const std::string ext = lowerExtension(path);
		if (ext == ".las" || ext == ".laz")
		{
			const std::string cache = path + ".mdspc";
			if (!fileExists(cache))
				throw std::runtime_error("no .mdspc cache beside " + path +
					"; run MultiDataStructure once on this cloud to generate it");
			resolvedPath = cache;
			return loadMdspc(cache);
		}
		resolvedPath = path;
		if (ext == ".mdspc") return loadMdspc(path);
		if (ext == ".ply") return loadPly(path);
		return loadXyz(path);
	}

	// ---------------------------------------------------------------- trace loading

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

	// ---------------------------------------------------------------- radius traversals
	//
	// One per structure, each pruning with the structure's own kNN min-distance helper. Results
	// are collected (not merely counted) so the cost model matches the native range path.

	using OTree = OctreeNode<double, size_t, Capacity>;
	using KTree = KDTree<double, size_t, 3, Capacity, KDSplit::ADAPTIVE>;
	using RTreeT = RTree<size_t, double, 3, Capacity>;

	void octreeRadius(const OTree* node, const std::array<double, 6>& probe, double r2, std::vector<size_t>& out)
	{
		if (node->minSqrDist(probe) > r2) return;
		if (node->isLeaf())
		{
			for (const auto& p : node->data)
			{
				const double dx = p.x - probe[0], dy = p.y - probe[1], dz = p.z - probe[2];
				if (dx * dx + dy * dy + dz * dz <= r2) out.push_back(p.id);
			}
			return;
		}
		for (const OTree* child : node->children)
			if (child) octreeRadius(child, probe, r2, out);
	}

	void kdRadius(const KTree::Node* node, const KTree::Point& centre, double r2, std::vector<size_t>& out)
	{
		if (!node) return;
		if (kdMindistToRegion(centre, node->region) > r2) return;
		if (node->isLeaf)
		{
			for (const auto& item : KTree::asLeaf(node)->data)
				if (kdPointDistSq(centre, item.first) <= r2) out.push_back(item.second);
			return;
		}
		const auto* internal = KTree::asInternal(node);
		kdRadius(internal->left.get(), centre, r2, out);
		kdRadius(internal->right.get(), centre, r2, out);
	}

	void rtreeRadius(const RTreeT::NodeType* node, const RTreeT::PointType& centre, double r2, std::vector<size_t>& out)
	{
		if (!node) return;
		if (node->isLeaf)
		{
			for (const auto& element : static_cast<const RTreeT::LeafType*>(node)->elements)
				if (squaredDistance(centre, element.first) <= r2) out.push_back(element.second);
			return;
		}
		for (const auto& child : static_cast<const RTreeT::InternalType*>(node)->children)
			if (mindistPointToBox(centre, child.first) <= r2) rtreeRadius(child.second, centre, r2, out);
	}

	void bruteForceRadius(const std::vector<XYZ>& pts, const TraceQuery& q, std::vector<size_t>& out)
	{
		const double r2 = q.radius * q.radius;
		for (size_t i = 0; i < pts.size(); ++i)
		{
			const double dx = pts[i].x - q.center[0], dy = pts[i].y - q.center[1], dz = pts[i].z - q.center[2];
			if (dx * dx + dy * dy + dz * dz <= r2) out.push_back(i);
		}
	}

	// ---------------------------------------------------------------- measurement

	struct Sample
	{
		std::string kind;
		double latencyMs = 0.0;
		long long count = 0;
		long long expected = -1;
	};

	struct Result
	{
		std::string method;
		std::string status = "ok";
		std::string reason;
		std::string note;
		std::string radiusMode;
		double buildMs = 0.0;
		size_t queries = 0;
		size_t repeats = 1;
		size_t skipped = 0;
		size_t height = 0, leaves = 0, internalNodes = 0, sizeBytes = 0;
		std::vector<Sample> samples;
	};

	enum class RadiusMode { NativeMbrPrune, AabbFilter };

	const char* radiusModeName(RadiusMode mode)
	{
		return mode == RadiusMode::NativeMbrPrune ? "native_mbr_prune" : "aabb_filter";
	}

	double percentile(std::vector<double> values, double fraction)
	{
		if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
		std::sort(values.begin(), values.end());
		const double pos = fraction * static_cast<double>(values.size() - 1);
		const size_t lo = static_cast<size_t>(pos), hi = std::min(lo + 1, values.size() - 1);
		if (lo == hi) return values[lo];
		return values[lo] * (static_cast<double>(hi) - pos) + values[hi] * (pos - static_cast<double>(lo));
	}

	std::string jsonEscape(const std::string& value)
	{
		std::string out;
		for (const char c : value)
		{
			if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
			else if (c == '\n') out += "\\n";
			else out.push_back(c);
		}
		return out;
	}

	void writeResult(std::ostream& out, const Result& result, bool trailingComma)
	{
		out << "    {\n";
		out << "      \"method\": \"" << jsonEscape(result.method) << "\",\n";
		if (result.status != "ok")
		{
			out << "      \"status\": \"" << jsonEscape(result.status) << "\",\n";
			out << "      \"reason\": \"" << jsonEscape(result.reason) << "\"\n";
			out << "    }" << (trailingComma ? "," : "") << "\n";
			return;
		}

		std::vector<double> latencies;
		std::map<std::string, std::vector<double>> byType;
		size_t mismatches = 0;
		long long maxDelta = 0;
		for (const Sample& sample : result.samples)
		{
			latencies.push_back(sample.latencyMs);
			byType[sample.kind].push_back(sample.latencyMs);
			if (sample.expected >= 0 && sample.count != sample.expected)
			{
				++mismatches;
				maxDelta = std::max(maxDelta, std::llabs(sample.count - sample.expected));
			}
		}
		const double avg = latencies.empty()
			? std::numeric_limits<double>::quiet_NaN()
			: std::accumulate(latencies.begin(), latencies.end(), 0.0) / static_cast<double>(latencies.size());

		out << "      \"status\": \"ok\",\n";
		out << "      \"build_ms\": " << result.buildMs << ",\n";
		out << "      \"queries\": " << result.queries << ",\n";
		out << "      \"repeats\": " << result.repeats << ",\n";
		out << "      \"avg_latency_ms\": " << avg << ",\n";
		out << "      \"p95_latency_ms\": " << percentile(latencies, 0.95) << ",\n";
		out << "      \"count_mismatches\": " << mismatches << ",\n";
		out << "      \"max_count_delta\": " << maxDelta << ",\n";
		out << "      \"unsupported_queries\": " << result.skipped << ",\n";
		out << "      \"radius_mode\": \"" << jsonEscape(result.radiusMode) << "\",\n";
		out << "      \"height\": " << result.height << ",\n";
		out << "      \"leaves\": " << result.leaves << ",\n";
		out << "      \"internal_nodes\": " << result.internalNodes << ",\n";
		out << "      \"size_bytes\": " << result.sizeBytes << ",\n";
		out << "      \"by_type\": {\n";
		bool first = true;
		for (const auto& item : byType)
		{
			if (!first) out << ",\n";
			first = false;
			const double typeAvg = std::accumulate(item.second.begin(), item.second.end(), 0.0) / static_cast<double>(item.second.size());
			out << "        \"" << jsonEscape(item.first) << "\": {\"queries\": " << item.second.size()
				<< ", \"avg_latency_ms\": " << typeAvg
				<< ", \"p95_latency_ms\": " << percentile(item.second, 0.95) << "}";
		}
		out << "\n      },\n";
		out << "      \"note\": \"" << jsonEscape(result.note) << "\"\n";
		out << "    }" << (trailingComma ? "," : "") << "\n";
	}

	void printResult(const Result& result)
	{
		if (result.status != "ok")
		{
			std::cout << result.method << ": " << result.status << " (" << result.reason << ")\n";
			return;
		}
		std::map<std::string, std::pair<double, size_t>> byType;
		std::map<std::string, std::pair<size_t, long long>> parity;
		for (const Sample& sample : result.samples)
		{
			auto& agg = byType[sample.kind];
			agg.first += sample.latencyMs;
			++agg.second;
			if (sample.expected >= 0 && sample.count != sample.expected)
			{
				auto& p = parity[sample.kind];
				++p.first;
				p.second = std::max(p.second, std::llabs(sample.count - sample.expected));
			}
		}
		std::cout << result.method << ": build " << result.buildMs << " ms, "
				  << result.queries << " queries x " << result.repeats << " repeats"
				  << " | radius=" << result.radiusMode;
		if (result.skipped) std::cout << " | unsupported queries skipped: " << result.skipped;
		std::cout << '\n';
		for (const auto& item : byType)
		{
			const auto& p = parity[item.first];
			std::cout << "  " << item.first << ": n=" << item.second.second
					  << " avg_ms=" << (item.second.first / static_cast<double>(item.second.second))
					  << " count_mismatches=" << p.first
					  << " max_delta=" << p.second << '\n';
		}
	}

	// Replay driver shared by all three structures. `range`, `radius` and `knn` are supplied as
	// callables so the timing loop, parity accounting and repeat handling stay identical across
	// structures - the only thing that varies is the index being measured.
	template <typename RangeFn, typename RadiusFn, typename KnnFn>
	void replay(Result& result, const std::vector<TraceQuery>& queries, size_t repeats,
				RangeFn range, RadiusFn radius, KnnFn knn)
	{
		std::vector<size_t> hits;
		result.samples.reserve(queries.size() * repeats);
		for (size_t repeat = 0; repeat < repeats; ++repeat)
		{
			for (const TraceQuery& q : queries)
			{
				hits.clear();
				Sample sample;
				sample.expected = q.expected;
				if (q.kind == "range" || q.kind == "count_range")
				{
					sample.kind = "range";
					const auto start = Clock::now();
					range(q, hits);
					sample.latencyMs = elapsedMs(start, Clock::now());
				}
				else if (q.kind == "radius")
				{
					sample.kind = "radius";
					const auto start = Clock::now();
					radius(q, hits);
					sample.latencyMs = elapsedMs(start, Clock::now());
				}
				else if (q.kind == "knn")
				{
					sample.kind = "knn";
					const auto start = Clock::now();
					knn(q, hits);
					sample.latencyMs = elapsedMs(start, Clock::now());
					// kNN returns exactly k whenever the cloud holds at least k points.
					sample.expected = static_cast<long long>(q.k);
				}
				else
				{
					if (repeat == 0) ++result.skipped;
					continue;
				}
				sample.count = static_cast<long long>(hits.size());
				result.samples.push_back(std::move(sample));
			}
		}
		result.repeats = repeats;
		result.queries = repeats ? result.samples.size() / repeats : 0;
	}

	// ---------------------------------------------------------------- per-structure runs

	Result runOctree(const std::vector<XYZ>& pts, const std::vector<TraceQuery>& queries, size_t repeats, RadiusMode mode)
	{
		Result result;
		result.method = "indexicon_octree";
		result.radiusMode = radiusModeName(mode);
		result.note = "Indexicon bulk-loaded octree (leaf capacity 128); radius via " + result.radiusMode + ".";

		using Pt = OPoint<double, size_t>;
		std::vector<Pt> bulk;
		bulk.reserve(pts.size());
		for (size_t i = 0; i < pts.size(); ++i) bulk.emplace_back(i, pts[i].x, pts[i].y, pts[i].z);
		const auto buildStart = Clock::now();
		OTree tree(bulk.begin(), bulk.end());
		result.buildMs = elapsedMs(buildStart, Clock::now());

		replay(result, queries, repeats,
			[&](const TraceQuery& q, std::vector<size_t>& out) {
				tree.query(OBox<double, size_t>(q.minB[0], q.minB[1], q.minB[2], q.maxB[0], q.maxB[1], q.maxB[2]),
						   std::back_inserter(out));
			},
			[&](const TraceQuery& q, std::vector<size_t>& out) {
				const double r2 = q.radius * q.radius;
				if (mode == RadiusMode::NativeMbrPrune)
				{
					const std::array<double, 6> probe{ q.center[0], q.center[1], q.center[2], q.center[0], q.center[1], q.center[2] };
					octreeRadius(&tree, probe, r2, out);
					return;
				}
				tree.query(OBox<double, size_t>(q.center[0] - q.radius, q.center[1] - q.radius, q.center[2] - q.radius,
												q.center[0] + q.radius, q.center[1] + q.radius, q.center[2] + q.radius),
						   std::back_inserter(out));
				out.erase(std::remove_if(out.begin(), out.end(), [&](size_t id) {
					const double dx = pts[id].x - q.center[0], dy = pts[id].y - q.center[1], dz = pts[id].z - q.center[2];
					return dx * dx + dy * dy + dz * dz > r2;
				}), out.end());
			},
			[&](const TraceQuery& q, std::vector<size_t>& out) {
				tree.knnQuery(std::array<double, 3>{ q.center[0], q.center[1], q.center[2] }, q.k, std::back_inserter(out));
			});

		const OctreeStats stats = tree.getStatistics();
		result.height = stats.height;
		result.leaves = stats.numLeaves;
		result.internalNodes = stats.numInternalNodes;
		result.sizeBytes = stats.sizeBytes;
		return result;
	}

	Result runKdTree(const std::vector<XYZ>& pts, const std::vector<TraceQuery>& queries, size_t repeats, RadiusMode mode)
	{
		Result result;
		result.method = "indexicon_kdtree";
		result.radiusMode = radiusModeName(mode);
		result.note = "Indexicon bulk-loaded kd-tree (ADAPTIVE split, bucket 128); radius via " + result.radiusMode + ".";

		using Pt = KTree::Point;
		std::vector<std::pair<Pt, size_t>> bulk;
		bulk.reserve(pts.size());
		for (size_t i = 0; i < pts.size(); ++i) bulk.emplace_back(Pt(pts[i].x, pts[i].y, pts[i].z), i);
		const auto buildStart = Clock::now();
		KTree tree(bulk.begin(), bulk.end());
		result.buildMs = elapsedMs(buildStart, Clock::now());

		replay(result, queries, repeats,
			[&](const TraceQuery& q, std::vector<size_t>& out) {
				KTree::Region box;
				box.low = Pt(q.minB[0], q.minB[1], q.minB[2]);
				box.high = Pt(q.maxB[0], q.maxB[1], q.maxB[2]);
				tree.query(box, std::back_inserter(out));
			},
			[&](const TraceQuery& q, std::vector<size_t>& out) {
				const double r2 = q.radius * q.radius;
				const Pt centre(q.center[0], q.center[1], q.center[2]);
				if (mode == RadiusMode::NativeMbrPrune)
				{
					kdRadius(tree.root.get(), centre, r2, out);
					return;
				}
				KTree::Region box;
				box.low = Pt(q.center[0] - q.radius, q.center[1] - q.radius, q.center[2] - q.radius);
				box.high = Pt(q.center[0] + q.radius, q.center[1] + q.radius, q.center[2] + q.radius);
				tree.query(box, std::back_inserter(out));
				out.erase(std::remove_if(out.begin(), out.end(), [&](size_t id) {
					const double dx = pts[id].x - q.center[0], dy = pts[id].y - q.center[1], dz = pts[id].z - q.center[2];
					return dx * dx + dy * dy + dz * dz > r2;
				}), out.end());
			},
			[&](const TraceQuery& q, std::vector<size_t>& out) {
				tree.knnQuery(Pt(q.center[0], q.center[1], q.center[2]), q.k, std::back_inserter(out));
			});

		const KDTreeStats stats = tree.getStatistics();
		result.height = stats.height;
		result.leaves = stats.numLeaves;
		result.internalNodes = stats.numInternalNodes;
		result.sizeBytes = stats.sizeBytes;
		return result;
	}

	Result runRTree(const std::vector<XYZ>& pts, const std::vector<TraceQuery>& queries, size_t repeats, RadiusMode mode)
	{
		Result result;
		result.method = "indexicon_rtree";
		result.radiusMode = radiusModeName(mode);
		result.note = "Indexicon packed R-tree (max 128 entries/node); radius via " + result.radiusMode + ".";

		using Pt = RTreeT::PointType;
		std::vector<std::pair<Pt, size_t>> bulk;
		bulk.reserve(pts.size());
		for (size_t i = 0; i < pts.size(); ++i) bulk.emplace_back(Pt(pts[i].x, pts[i].y, pts[i].z), i);
		const auto buildStart = Clock::now();
		RTreeT tree(bulk.begin(), bulk.end());
		result.buildMs = elapsedMs(buildStart, Clock::now());

		replay(result, queries, repeats,
			[&](const TraceQuery& q, std::vector<size_t>& out) {
				RTreeT::BoxType box;
				box.minCorner = Pt(q.minB[0], q.minB[1], q.minB[2]);
				box.maxCorner = Pt(q.maxB[0], q.maxB[1], q.maxB[2]);
				tree.query(box, std::back_inserter(out));
			},
			[&](const TraceQuery& q, std::vector<size_t>& out) {
				const double r2 = q.radius * q.radius;
				const Pt centre(q.center[0], q.center[1], q.center[2]);
				if (mode == RadiusMode::NativeMbrPrune)
				{
					rtreeRadius(tree.root, centre, r2, out);
					return;
				}
				RTreeT::BoxType box;
				box.minCorner = Pt(q.center[0] - q.radius, q.center[1] - q.radius, q.center[2] - q.radius);
				box.maxCorner = Pt(q.center[0] + q.radius, q.center[1] + q.radius, q.center[2] + q.radius);
				tree.query(box, std::back_inserter(out));
				out.erase(std::remove_if(out.begin(), out.end(), [&](size_t id) {
					const double dx = pts[id].x - q.center[0], dy = pts[id].y - q.center[1], dz = pts[id].z - q.center[2];
					return dx * dx + dy * dy + dz * dz > r2;
				}), out.end());
			},
			[&](const TraceQuery& q, std::vector<size_t>& out) {
				tree.knnQuery(Pt(q.center[0], q.center[1], q.center[2]), q.k, std::back_inserter(out));
			});

		const RTreeT::RTreeStats stats = tree.getStatistics();
		result.height = stats.height;
		result.leaves = stats.numLeaves;
		result.internalNodes = stats.numInternalNodes;
		result.sizeBytes = stats.sizeBytes;
		return result;
	}

	// Independent check that the three radius traversals are exact: brute-force the first
	// `limit` radius queries and require identical hit counts. This isolates a traversal bug
	// from a convention mismatch with the trace's recorded counts.
	size_t verifyRadius(const std::vector<XYZ>& pts, const std::vector<TraceQuery>& queries, size_t limit)
	{
		using Pt = OPoint<double, size_t>;
		std::vector<Pt> obulk;
		std::vector<std::pair<KTree::Point, size_t>> kbulk;
		std::vector<std::pair<RTreeT::PointType, size_t>> rbulk;
		obulk.reserve(pts.size()); kbulk.reserve(pts.size()); rbulk.reserve(pts.size());
		for (size_t i = 0; i < pts.size(); ++i)
		{
			obulk.emplace_back(i, pts[i].x, pts[i].y, pts[i].z);
			kbulk.emplace_back(KTree::Point(pts[i].x, pts[i].y, pts[i].z), i);
			rbulk.emplace_back(RTreeT::PointType(pts[i].x, pts[i].y, pts[i].z), i);
		}
		OTree octree(obulk.begin(), obulk.end());
		KTree kdtree(kbulk.begin(), kbulk.end());
		RTreeT rtree(rbulk.begin(), rbulk.end());

		size_t checked = 0, failures = 0;
		std::vector<size_t> truth, got;
		for (const TraceQuery& q : queries)
		{
			if (q.kind != "radius" || checked >= limit) continue;
			++checked;
			const double r2 = q.radius * q.radius;
			const std::array<double, 6> probe{ q.center[0], q.center[1], q.center[2], q.center[0], q.center[1], q.center[2] };
			const KTree::Point kcentre(q.center[0], q.center[1], q.center[2]);
			const RTreeT::PointType rcentre(q.center[0], q.center[1], q.center[2]);

			truth.clear();
			bruteForceRadius(pts, q, truth);

			struct Check { const char* name; size_t count; };
			got.clear(); octreeRadius(&octree, probe, r2, got);
			const size_t octreeCount = got.size();
			got.clear(); kdRadius(kdtree.root.get(), kcentre, r2, got);
			const size_t kdCount = got.size();
			got.clear(); rtreeRadius(rtree.root, rcentre, r2, got);
			const size_t rtreeCount = got.size();

			const Check checks[3] = { {"octree", octreeCount}, {"kdtree", kdCount}, {"rtree", rtreeCount} };
			for (const Check& check : checks)
			{
				if (check.count != truth.size())
				{
					++failures;
					std::cout << "  MISMATCH " << check.name << " radius query " << checked
							  << ": got " << check.count << ", brute force " << truth.size() << '\n';
				}
			}
		}
		std::cout << "radius brute-force verification: " << checked << " queries, " << failures << " mismatches\n";
		return failures;
	}

	struct Args
	{
		std::string input, queryTrace, output;
		std::string structures = "octree,kdtree,rtree";
		size_t maxQueries = 0;
		size_t repeats = 1;
		size_t verifyBruteforce = 0;
		RadiusMode radiusMode = RadiusMode::NativeMbrPrune;
	};

	constexpr const char* Usage =
		"usage: indexicon_point_baseline --input <cloud.ply|.xyz|.mdspc|.las> --query-trace <trace.csv>\n"
		"       [--output result.json] [--structures octree,kdtree,rtree] [--max-queries N]\n"
		"       [--repeats R] [--radius-mode native_mbr_prune|aabb_filter] [--verify-bruteforce N]\n"
		"       indexicon_point_baseline <cloud> <trace.csv> [max_queries]   (legacy positional form)\n";

	// Thrown for --help so main() can print usage on stdout and exit 0 rather than treating a
	// help request as a failure.
	struct HelpRequested {};

	Args parseArgs(int argc, char** argv)
	{
		Args args;
		// Legacy positional form: <cloud> <trace> [max_queries]
		if (argc >= 3 && argv[1][0] != '-')
		{
			args.input = argv[1];
			args.queryTrace = argv[2];
			if (argc > 3) args.maxQueries = std::stoull(argv[3]);
			return args;
		}
		for (int i = 1; i < argc; ++i)
		{
			const std::string arg = argv[i];
			if (arg == "--input" && i + 1 < argc) args.input = argv[++i];
			else if (arg == "--query-trace" && i + 1 < argc) args.queryTrace = argv[++i];
			else if (arg == "--output" && i + 1 < argc) args.output = argv[++i];
			else if (arg == "--structures" && i + 1 < argc) args.structures = argv[++i];
			else if (arg == "--max-queries" && i + 1 < argc) args.maxQueries = std::stoull(argv[++i]);
			else if (arg == "--repeats" && i + 1 < argc) args.repeats = std::stoull(argv[++i]);
			else if (arg == "--verify-bruteforce" && i + 1 < argc) args.verifyBruteforce = std::stoull(argv[++i]);
			else if (arg == "--radius-mode" && i + 1 < argc)
			{
				const std::string mode = argv[++i];
				if (mode == "native_mbr_prune") args.radiusMode = RadiusMode::NativeMbrPrune;
				else if (mode == "aabb_filter") args.radiusMode = RadiusMode::AabbFilter;
				else throw std::runtime_error("unknown --radius-mode: " + mode + " (native_mbr_prune|aabb_filter)");
			}
			else if (arg == "--help" || arg == "-h") throw HelpRequested{};
			else throw std::runtime_error("unknown argument: " + arg + "\n" + Usage);
		}
		if (args.input.empty() || args.queryTrace.empty())
			throw std::runtime_error(std::string("--input and --query-trace are required\n") + Usage);
		if (args.repeats == 0) throw std::runtime_error("--repeats must be at least 1");
		return args;
	}

	bool wants(const std::string& list, const std::string& name)
	{
		return list.find(name) != std::string::npos;
	}
}

int main(int argc, char** argv)
{
	try
	{
		const Args args = parseArgs(argc, argv);

		std::string resolvedCloud;
		const std::vector<XYZ> pts = loadCloud(args.input, resolvedCloud);
		std::cout << "loaded " << pts.size() << " points from " << resolvedCloud << '\n';
		if (pts.empty()) throw std::runtime_error("point cloud is empty");

		std::vector<TraceQuery> queries = loadTrace(args.queryTrace);
		if (args.maxQueries && queries.size() > args.maxQueries)
		{
			// deterministic even stride, matching the MDS replayer's subsetting
			std::vector<TraceQuery> subset;
			const double step = static_cast<double>(queries.size()) / static_cast<double>(args.maxQueries);
			for (size_t i = 0; i < args.maxQueries; ++i) subset.push_back(queries[static_cast<size_t>(i * step)]);
			queries.swap(subset);
		}
		std::cout << "replaying " << queries.size() << " queries from " << args.queryTrace
				  << " (radius mode: " << radiusModeName(args.radiusMode) << ", repeats: " << args.repeats << ")\n";

		if (args.verifyBruteforce)
		{
			if (verifyRadius(pts, queries, args.verifyBruteforce) != 0)
			{
				std::cerr << "error: radius traversal disagrees with brute force; refusing to report numbers\n";
				return 2;
			}
		}

		std::vector<Result> results;
		if (wants(args.structures, "octree")) results.push_back(runOctree(pts, queries, args.repeats, args.radiusMode));
		if (wants(args.structures, "kdtree")) results.push_back(runKdTree(pts, queries, args.repeats, args.radiusMode));
		if (wants(args.structures, "rtree")) results.push_back(runRTree(pts, queries, args.repeats, args.radiusMode));
		if (results.empty()) throw std::runtime_error("--structures selected nothing (octree,kdtree,rtree)");

		for (const Result& result : results) printResult(result);

		if (!args.output.empty())
		{
			std::ofstream out(args.output);
			if (!out) throw std::runtime_error("cannot write " + args.output);
			out.setf(std::ios::fmtflags(0), std::ios::floatfield);
			out.precision(10);
			out << "{\n";
			out << "  \"library\": \"Indexicon\",\n";
			out << "  \"library_commit\": \"c9f9b1d\",\n";
			out << "  \"input\": \"" << jsonEscape(resolvedCloud) << "\",\n";
			out << "  \"points\": " << pts.size() << ",\n";
			out << "  \"query_trace\": \"" << jsonEscape(args.queryTrace) << "\",\n";
			out << "  \"results\": [\n";
			for (size_t i = 0; i < results.size(); ++i) writeResult(out, results[i], i + 1 < results.size());
			out << "  ]\n";
			out << "}\n";
			std::cout << "wrote " << args.output << '\n';
		}
		return 0;
	}
	catch (const HelpRequested&)
	{
		std::cout << Usage;
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << '\n';
		return 1;
	}
}

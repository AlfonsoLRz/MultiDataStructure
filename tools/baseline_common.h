// One measurement protocol for every external replay baseline (testing.md 2r), so their rows sit
// next to engine-check's without a conversion step:
//
//   - the cloud from its .mdspc cache, as float in the cloud's local frame (the representation
//     MultiDataStructure queries); trace centres (world, double) are moved into that frame in double;
//   - the trace subset with engine-check's integer stride (i * total) / count;
//   - one index build, timed alone, then --warmup untimed passes and --repeats timed passes, each
//     query timed individually around the library's search call only;
//   - after the clock, each query's returned count and the same order-independent checksum of point
//     ids engine-check computes, so every library's answers can be checked against the engine's
//     (the held-out traces carry no ground truth);
//   - one row per library configuration with engine-check's --summary-csv columns, and per-query rows
//     with its --query-csv columns.
//
// Header-only; each driver includes it, defines its Methods and calls Baseline::runAll.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Baseline
{
	using Clock = std::chrono::steady_clock;

	inline double msSince(Clock::time_point start)
	{
		return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
	}

	struct Cloud
	{
		std::vector<float>	xyz;	// interleaved, local frame
		double				origin[3] = { 0, 0, 0 };
		double				scale[3] = { 1, 1, 1 };

		size_t size() const { return xyz.size() / 3; }
		const float* point(size_t i) const { return xyz.data() + 3 * i; }
	};

	inline Cloud loadMdspc(const std::string& path)
	{
		std::ifstream in(path, std::ios::binary);
		if (!in)
			throw std::runtime_error("cannot open " + path);
		char header[88] = {};
		in.read(header, 40);
		if (!in || std::memcmp(header, "MDSPC01", 7) != 0)
			throw std::runtime_error("not an .mdspc cache: " + path);
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
		if (!in)
			throw std::runtime_error("truncated .mdspc payload: " + path);
		return cloud;
	}

	enum class Kind { Range = 0, Count = 1, Radius = 2, Knn = 3, Other = 4 };

	inline const char* kindName(Kind kind)
	{
		switch (kind)
		{
		case Kind::Range: return "range";
		case Kind::Count: return "count_range";
		case Kind::Radius: return "radius";
		case Kind::Knn: return "knn";
		default: return "other";
		}
	}

	struct Query
	{
		Kind	kind = Kind::Other;
		float	center[3] = { 0, 0, 0 };
		float	lo[3] = { 0, 0, 0 };
		float	hi[3] = { 0, 0, 0 };
		float	radius = 0.0f;
		size_t	k = 0;
	};

	inline std::vector<std::string> splitCsv(const std::string& line)
	{
		std::vector<std::string> out;
		std::string cell;
		std::istringstream ss(line);
		while (std::getline(ss, cell, ','))
			out.push_back(cell);
		return out;
	}

	inline std::vector<Query> loadTrace(const std::string& path, const Cloud& cloud, size_t maxQueries)
	{
		std::ifstream in(path);
		if (!in)
			throw std::runtime_error("cannot open " + path);
		std::string line;
		std::getline(in, line);
		if (!line.empty() && line.back() == '\r')
			line.pop_back();
		const std::vector<std::string> header = splitCsv(line);
		const auto col = [&](const char* name) {
			const auto it = std::find(header.begin(), header.end(), name);
			if (it == header.end())
				throw std::runtime_error(std::string("trace missing column ") + name);
			return static_cast<size_t>(it - header.begin());
		};
		const size_t cType = col("query_type"), cRadius = col("radius"), cK = col("k");
		const size_t cCenter[3] = { col("center_x"), col("center_y"), col("center_z") };
		const size_t cLo[3] = { col("bounds_min_x"), col("bounds_min_y"), col("bounds_min_z") };
		const size_t cHi[3] = { col("bounds_max_x"), col("bounds_max_y"), col("bounds_max_z") };
		const auto local = [&](double world, int axis) { return static_cast<float>((world - cloud.origin[axis]) / cloud.scale[axis]); };

		std::vector<Query> all;
		while (std::getline(in, line))
		{
			if (!line.empty() && line.back() == '\r')
				line.pop_back();
			const std::vector<std::string> f = splitCsv(line);
			if (f.size() < header.size())
				continue;
			Query q;
			const std::string& type = f[cType];
			q.kind = type == "radius" ? Kind::Radius : type == "knn" ? Kind::Knn : type == "count_range" ? Kind::Count
				   : (type == "range" || type == "aabb_range") ? Kind::Range : Kind::Other;
			for (int a = 0; a < 3; ++a)
			{
				q.center[a] = local(std::stod(f[cCenter[a]]), a);
				q.lo[a] = local(std::stod(f[cLo[a]]), a);
				q.hi[a] = local(std::stod(f[cHi[a]]), a);
			}
			q.radius = static_cast<float>(std::stod(f[cRadius]));
			q.k = static_cast<size_t>(std::stod(f[cK]));
			all.push_back(q);
		}
		const size_t total = all.size();
		const size_t count = (maxQueries > 0 && maxQueries < total) ? maxQueries : total;
		if (count == total)
			return all;
		std::vector<Query> subset;
		subset.reserve(count);
		for (size_t i = 0; i < count; ++i)
			subset.push_back(all[(i * total) / count]);
		return subset;
	}

	// PointSpatialIndex::accumulateResultChecksum: splitmix64's finalizer per id, summed, so the
	// digest is order-independent.
	inline void mixChecksum(uint64_t& checksum, size_t pointIndex)
	{
		uint64_t mixed = static_cast<uint64_t>(pointIndex) + 0x9E3779B97F4A7C15ull;
		mixed ^= mixed >> 30;
		mixed *= 0xBF58476D1CE4E5B9ull;
		mixed ^= mixed >> 27;
		mixed *= 0x94D049BB133111EBull;
		mixed ^= mixed >> 31;
		checksum += mixed;
	}

	struct Options
	{
		std::string	input;
		std::string	trace;
		std::string	summaryCsv;
		std::string	queryCsv;
		size_t		maxQueries = 0;
		size_t		warmup = 1;
		size_t		repeats = 10;
		std::map<std::string, std::string>	extra;	// driver flags, e.g. --leaf-sizes

		std::vector<double> list(const std::string& flag, const std::string& fallback) const
		{
			const auto it = extra.find(flag);
			std::vector<double> values;
			std::istringstream ss(it == extra.end() ? fallback : it->second);
			std::string item;
			while (std::getline(ss, item, ','))
				if (!item.empty())
					values.push_back(std::stod(item));
			return values;
		}
	};

	// --input --query-trace --summary-csv [--query-csv] [--max-queries] [--warmup] [--repeats], plus the
	// driver's own flags (each takes one value).
	inline Options parseOptions(int argc, char** argv, const std::vector<std::string>& driverFlags)
	{
		Options options;
		for (int i = 1; i < argc; ++i)
		{
			const std::string arg = argv[i];
			const auto next = [&]() -> std::string {
				if (i + 1 >= argc)
					throw std::runtime_error("missing value for " + arg);
				return argv[++i];
			};
			if (arg == "--input") options.input = next();
			else if (arg == "--query-trace") options.trace = next();
			else if (arg == "--summary-csv") options.summaryCsv = next();
			else if (arg == "--query-csv") options.queryCsv = next();
			else if (arg == "--max-queries") options.maxQueries = std::stoull(next());
			else if (arg == "--warmup") options.warmup = std::stoull(next());
			else if (arg == "--repeats") options.repeats = std::max<size_t>(1, std::stoull(next()));
			else if (std::find(driverFlags.begin(), driverFlags.end(), arg) != driverFlags.end()) options.extra[arg] = next();
			else throw std::runtime_error("unknown argument " + arg);
		}
		if (options.input.empty() || options.trace.empty())
			throw std::runtime_error("needs --input <cloud.mdspc> and --query-trace <trace.csv>");
		if (options.input.size() > 4 && options.input.substr(options.input.size() - 4) == ".las")
			options.input += ".mdspc";
		return options;
	}

	// One configuration of one library. build() constructs the index and returns its build time in
	// ms (the driver times only the library call, not its own data copies). query() runs one search,
	// timing only the library's call, and returns the ms, or a negative value for a kind it does not
	// support. answer() reports, after the clock, how many points the last query returned and their
	// checksum. release() frees the index before the next configuration is built.
	struct Method
	{
		std::string	name;
		std::function<double()>					build;
		std::function<double(const Query&)>		query;
		std::function<void(size_t&, uint64_t&)>	answer;
		std::function<void()>					release;
	};

	inline std::string csvField(const std::string& value)
	{
		return value.find_first_of(",\"\n") == std::string::npos ? value : "\"" + value + "\"";
	}

	inline void runAll(const std::string& library, const Options& options, const Cloud& cloud, const std::vector<Query>& queries,
					   std::vector<Method>& methods)
	{
		std::cout << library << ": " << cloud.size() << " points, " << queries.size() << " queries, " << options.warmup
				  << " warm-up + " << options.repeats << " timed passes\n";

		std::ofstream summary, perQuery;
		if (!options.summaryCsv.empty())
		{
			const bool exists = std::filesystem::exists(options.summaryCsv);
			summary.open(options.summaryCsv, std::ios::app);
			if (!summary)
				throw std::runtime_error("cannot write " + options.summaryCsv);
			if (!exists)
				summary << "engine_version,cloud,points,schema_name,schema_path,query_source,queries,warmup_passes,timed_passes,"
						<< "build_ms_mean,build_ms_min,num_nodes,num_leaves,max_depth,structure_digest,"
						<< "range_queries,range_mean_ms,count_queries,count_mean_ms,radius_queries,radius_mean_ms,"
						<< "knn_queries,knn_mean_ms,all_mean_ms,mean_visited_nodes,mean_tested_points,"
						<< "mean_returned_points,query_set_checksum,pass_checksum_mismatches,timed_telemetry,mean_tested_boxes,"
						<< "interleaved,pass_median_ms,pass_min_ms,pass_means_ms,unsupported_queries\n";
		}
		if (!options.queryCsv.empty())
		{
			const bool exists = std::filesystem::exists(options.queryCsv);
			perQuery.open(options.queryCsv, std::ios::app);
			if (!exists)
				perQuery << "schema_name,structure_digest,num_nodes,num_leaves,max_depth,query_id,query_type,k,"
						 << "returned_points,result_checksum,visited_nodes,tested_boxes,tested_points,fully_contained_nodes\n";
		}

		for (Method& method : methods)
		{
			const double buildMs = method.build();
			std::vector<size_t> returned(queries.size(), 0);
			std::vector<uint64_t> checksums(queries.size(), 0);
			std::vector<char> supported(queries.size(), 1);
			std::array<double, 4> sum{};
			std::array<size_t, 4> count{};
			std::vector<double> passMeans;
			size_t mismatches = 0;
			for (size_t pass = 0; pass < options.warmup + options.repeats; ++pass)
			{
				const bool timed = pass >= options.warmup;
				double passMs = 0.0;
				size_t passQueries = 0;
				for (size_t q = 0; q < queries.size(); ++q)
				{
					if (!supported[q])
						continue;
					const double ms = method.query(queries[q]);
					if (ms < 0.0)
					{
						supported[q] = 0;
						continue;
					}
					size_t n = 0;
					uint64_t checksum = 0;
					method.answer(n, checksum);
					if (pass == 0)
					{
						returned[q] = n;
						checksums[q] = checksum;
					}
					else if (returned[q] != n || checksums[q] != checksum)
						++mismatches;
					if (timed)
					{
						const size_t slot = static_cast<size_t>(queries[q].kind);
						sum[slot] += ms;
						++count[slot];
						passMs += ms;
						++passQueries;
					}
				}
				if (timed && passQueries)
					passMeans.push_back(passMs / static_cast<double>(passQueries));
			}
			method.release();

			uint64_t querySetChecksum = 0;
			double returnedSum = 0.0;
			size_t unsupported = 0;
			for (size_t q = 0; q < queries.size(); ++q)
			{
				if (!supported[q])
				{
					++unsupported;
					continue;
				}
				returnedSum += static_cast<double>(returned[q]);
				querySetChecksum += (checksums[q] ^ (static_cast<uint64_t>(q) * 0x9E3779B97F4A7C15ull)) + returned[q];
				if (perQuery)
					perQuery << csvField(method.name) << ",0,0,0,0," << q << ',' << kindName(queries[q].kind) << ',' << queries[q].k << ','
							 << returned[q] << ',' << checksums[q] << ",0,0,0,0\n";
			}
			const auto mean = [](double total, size_t n) { return n ? total / static_cast<double>(n) : 0.0; };
			const double allSum = sum[0] + sum[1] + sum[2] + sum[3];
			const size_t allCount = count[0] + count[1] + count[2] + count[3];
			std::vector<double> sorted = passMeans;
			std::sort(sorted.begin(), sorted.end());
			const double passMedian = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];

			std::cout << "  " << method.name << ": build " << std::fixed << std::setprecision(1) << buildMs << " ms" << std::setprecision(3)
					  << " | radius " << mean(sum[2], count[2]) * 1000.0 << " us | knn " << mean(sum[3], count[3]) * 1000.0
					  << " us | range " << mean(sum[0], count[0]) * 1000.0 << " us | all " << mean(allSum, allCount) * 1000.0 << " us"
					  << (unsupported ? " | " + std::to_string(unsupported) + " unsupported" : std::string())
					  << (mismatches ? " | PASS MISMATCHES " + std::to_string(mismatches) : std::string()) << '\n';
			std::cout.unsetf(std::ios::floatfield);

			if (summary)
			{
				std::ostringstream passList;
				passList << std::setprecision(10);
				for (size_t i = 0; i < passMeans.size(); ++i)
					passList << (i ? ";" : "") << passMeans[i];
				const size_t timedPasses = options.repeats;
				summary << std::setprecision(10) << csvField(library) << ',' << csvField(options.input) << ',' << cloud.size() << ','
						<< csvField(method.name) << ',' << csvField(method.name) << ',' << csvField(options.trace) << ',' << queries.size() << ','
						<< options.warmup << ',' << timedPasses << ',' << buildMs << ',' << buildMs << ",0,0,0,0,"
						<< count[0] / timedPasses << ',' << mean(sum[0], count[0]) << ',' << count[1] / timedPasses << ',' << mean(sum[1], count[1]) << ','
						<< count[2] / timedPasses << ',' << mean(sum[2], count[2]) << ',' << count[3] / timedPasses << ',' << mean(sum[3], count[3]) << ','
						<< mean(allSum, allCount) << ",0,0," << returnedSum / std::max<double>(1.0, static_cast<double>(queries.size() - unsupported)) << ','
						<< querySetChecksum << ',' << mismatches << ",off,0,0," << passMedian << ',' << (sorted.empty() ? 0.0 : sorted.front()) << ','
						<< passList.str() << ',' << unsupported << '\n';
				summary.flush();
			}
		}
	}
}

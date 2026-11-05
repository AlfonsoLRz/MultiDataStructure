// External replay baseline against the unibn octree (Behley, Steinhage and Cremers, ICRA 2015;
// github.com/jbehley/octree, MIT, header-only): the octree built for radius search in LiDAR scans,
// single-threaded. It answers radius queries (and 1-NN, not used), so kNN queries count as
// unsupported. Protocol and output: tools/baseline_common.h.
//
// One configuration per bucket size (the leaf capacity; the library's default is 32). Points are
// float in the cloud's local frame, not copied (copyPoints = false), as the paper's benchmark ran.
//
// Build: tools/build_baselines_linux.sh
// Usage:
//   unibn_point_baseline --input <cloud.mdspc> --query-trace <trace.csv> --summary-csv <out.csv>
//                        [--query-csv <q.csv>] [--max-queries N] [--warmup W] [--repeats R]
//                        [--bucket-sizes 32,8,16,64,128,256,512]

#include "baseline_common.h"

#include <memory>

#include <Octree.hpp>

namespace
{
	struct Point
	{
		float x, y, z;
	};
}

int main(int argc, char** argv)
{
	try
	{
		const Baseline::Options options = Baseline::parseOptions(argc, argv, { "--bucket-sizes" });
		const Baseline::Cloud cloud = Baseline::loadMdspc(options.input);
		const std::vector<Baseline::Query> queries = Baseline::loadTrace(options.trace, cloud, options.maxQueries);

		std::vector<Point> points(cloud.size());
		for (size_t i = 0; i < cloud.size(); ++i)
			points[i] = { cloud.point(i)[0], cloud.point(i)[1], cloud.point(i)[2] };

		std::unique_ptr<unibn::Octree<Point>> octree;
		std::vector<uint32_t> indices;
		std::vector<Baseline::Method> methods;
		for (const double bucket : options.list("--bucket-sizes", "32,8,16,64,128,256,512"))
		{
			Baseline::Method method;
			method.name = "unibn_octree_bucket" + std::to_string(static_cast<size_t>(bucket));
			method.build = [&, bucket]() {
				const auto start = Baseline::Clock::now();
				octree = std::make_unique<unibn::Octree<Point>>();
				octree->initialize(points, unibn::OctreeParams(static_cast<uint32_t>(bucket), false, 0.0f));
				return Baseline::msSince(start);
			};
			method.query = [&](const Baseline::Query& q) {
				if (q.kind != Baseline::Kind::Radius)
					return -1.0;
				const Point center{ q.center[0], q.center[1], q.center[2] };
				const auto start = Baseline::Clock::now();
				octree->radiusNeighbors<unibn::L2Distance<Point>>(center, q.radius, indices);
				return Baseline::msSince(start);
			};
			method.answer = [&](size_t& count, uint64_t& checksum) {
				count = indices.size();
				for (const uint32_t index : indices)
					Baseline::mixChecksum(checksum, index);
			};
			method.release = [&]() { octree.reset(); };
			methods.push_back(std::move(method));
		}
		Baseline::runAll("unibn octree 8e3927d", options, cloud, queries, methods);
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << '\n';
		return 1;
	}
}

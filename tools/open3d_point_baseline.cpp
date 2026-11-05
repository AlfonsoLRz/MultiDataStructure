// External replay baseline against Open3D's geometry::KDTreeFlann (open3d.org, MIT), the kd-tree
// behind Open3D's normal estimation, outlier removal and registration; single-threaded queries.
// Protocol and output: tools/baseline_common.h.
//
// KDTreeFlann holds the points as an Eigen 3 x N matrix of doubles and has no parameters (a
// nanoflann tree inside, with Open3D's own leaf size). Coordinates are the cloud's local frame,
// widened from the same floats the other libraries index. Radius results are requested unsorted.
//
// Build: tools/build_baselines_linux.sh
// Usage:
//   open3d_point_baseline --input <cloud.mdspc> --query-trace <trace.csv> --summary-csv <out.csv>
//                         [--query-csv <q.csv>] [--max-queries N] [--warmup W] [--repeats R]

#include "baseline_common.h"

#include <memory>

#include <Eigen/Core>
#include <open3d/Open3DConfig.h>
#include <open3d/geometry/KDTreeFlann.h>

int main(int argc, char** argv)
{
	try
	{
		const Baseline::Options options = Baseline::parseOptions(argc, argv, {});
		const Baseline::Cloud cloud = Baseline::loadMdspc(options.input);
		const std::vector<Baseline::Query> queries = Baseline::loadTrace(options.trace, cloud, options.maxQueries);

		Eigen::MatrixXd data(3, static_cast<Eigen::Index>(cloud.size()));
		for (size_t i = 0; i < cloud.size(); ++i)
			for (int a = 0; a < 3; ++a)
				data(a, static_cast<Eigen::Index>(i)) = cloud.point(i)[a];

		std::unique_ptr<open3d::geometry::KDTreeFlann> tree;
		std::vector<int> indices;
		std::vector<double> distances;
		std::vector<Baseline::Method> methods(1);
		Baseline::Method& method = methods.front();
		method.name = "open3d_kdtreeflann";
		method.build = [&]() {
			const auto start = Baseline::Clock::now();
			tree = std::make_unique<open3d::geometry::KDTreeFlann>(data);
			return Baseline::msSince(start);
		};
		method.query = [&](const Baseline::Query& q) {
			const Eigen::Vector3d center(q.center[0], q.center[1], q.center[2]);
			if (q.kind == Baseline::Kind::Radius)
			{
				const auto start = Baseline::Clock::now();
				tree->SearchRadius(center, static_cast<double>(q.radius), indices, distances, false);
				return Baseline::msSince(start);
			}
			if (q.kind == Baseline::Kind::Knn)
			{
				const auto start = Baseline::Clock::now();
				tree->SearchKNN(center, static_cast<int>(q.k), indices, distances);
				return Baseline::msSince(start);
			}
			return -1.0;
		};
		method.answer = [&](size_t& count, uint64_t& checksum) {
			count = indices.size();
			for (const int index : indices)
				Baseline::mixChecksum(checksum, static_cast<size_t>(index));
		};
		method.release = [&]() { tree.reset(); };
		Baseline::runAll(std::string("Open3D ") + OPEN3D_VERSION, options, cloud, queries, methods);
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << '\n';
		return 1;
	}
}

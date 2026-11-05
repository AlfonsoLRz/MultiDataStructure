// External replay baseline against Boost.Geometry's R-tree (boost::geometry::index::rtree, Boost
// Software License, header-only): the R-tree reviewers of a spatial-index paper expect, bulk-loaded
// with the STR packing its range constructor uses, single-threaded. Protocol and output:
// tools/baseline_common.h.
//
// One configuration per node capacity (a compile-time parameter; 16 is the documented example).
// Packing ignores the insertion algorithm, so only bgi::rstar is built. Points are float in the
// cloud's local frame. Boost has no sphere predicate, so a radius query is the usual box intersection
// plus a distance filter (bgi::intersects + bgi::satisfies); kNN uses bgi::nearest, unsorted.
//
// Build: tools/build_baselines_linux.sh (Linux), tools/build_boost_rtree_baseline.ps1 (Windows)
// Usage:
//   boost_rtree_point_baseline --input <cloud.mdspc> --query-trace <trace.csv> --summary-csv <out.csv>
//                              [--query-csv <q.csv>] [--max-queries N] [--warmup W] [--repeats R]
//                              [--max-elements 16,8,32,64]

#include "baseline_common.h"

#include <memory>
#include <utility>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/version.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace
{
	using Point = bg::model::point<float, 3, bg::cs::cartesian>;
	using Box = bg::model::box<Point>;
	using Value = std::pair<Point, uint32_t>;

	template <typename Parameters>
	Baseline::Method makeMethod(size_t maxElements, const Baseline::Cloud& cloud)
	{
		struct State
		{
			std::unique_ptr<bgi::rtree<Value, Parameters>> tree;
			std::vector<Value> hits;
		};
		auto state = std::make_shared<State>();
		Baseline::Method method;
		method.name = "boost_rtree_rstar" + std::to_string(maxElements) + "_str";
		method.build = [state, &cloud]() {
			std::vector<Value> values;
			values.reserve(cloud.size());
			for (size_t i = 0; i < cloud.size(); ++i)
				values.emplace_back(Point(cloud.point(i)[0], cloud.point(i)[1], cloud.point(i)[2]), static_cast<uint32_t>(i));
			// The range constructor bulk-loads with STR packing; the value copy above is not timed.
			const auto start = Baseline::Clock::now();
			state->tree = std::make_unique<bgi::rtree<Value, Parameters>>(values.begin(), values.end());
			return Baseline::msSince(start);
		};
		method.query = [state](const Baseline::Query& q) {
			state->hits.clear();
			if (q.kind == Baseline::Kind::Radius)
			{
				const float r = q.radius, r2 = q.radius * q.radius;
				const float cx = q.center[0], cy = q.center[1], cz = q.center[2];
				const Box box(Point(cx - r, cy - r, cz - r), Point(cx + r, cy + r, cz + r));
				const auto start = Baseline::Clock::now();
				state->tree->query(bgi::intersects(box) && bgi::satisfies([&](const Value& v) {
					const float dx = bg::get<0>(v.first) - cx;
					const float dy = bg::get<1>(v.first) - cy;
					const float dz = bg::get<2>(v.first) - cz;
					return dx * dx + dy * dy + dz * dz <= r2;
				}), std::back_inserter(state->hits));
				return Baseline::msSince(start);
			}
			if (q.kind == Baseline::Kind::Knn)
			{
				const Point center(q.center[0], q.center[1], q.center[2]);
				const auto start = Baseline::Clock::now();
				state->tree->query(bgi::nearest(center, static_cast<unsigned>(q.k)), std::back_inserter(state->hits));
				return Baseline::msSince(start);
			}
			return -1.0;
		};
		method.answer = [state](size_t& count, uint64_t& checksum) {
			count = state->hits.size();
			for (const Value& v : state->hits)
				Baseline::mixChecksum(checksum, v.second);
		};
		method.release = [state]() { state->tree.reset(); };
		return method;
	}
}

int main(int argc, char** argv)
{
	try
	{
		const Baseline::Options options = Baseline::parseOptions(argc, argv, { "--max-elements" });
		const Baseline::Cloud cloud = Baseline::loadMdspc(options.input);
		const std::vector<Baseline::Query> queries = Baseline::loadTrace(options.trace, cloud, options.maxQueries);

		std::vector<Baseline::Method> methods;
		for (const double value : options.list("--max-elements", "16,8,32,64"))
		{
			const size_t maxElements = static_cast<size_t>(value);
			if (maxElements == 8) methods.push_back(makeMethod<bgi::rstar<8>>(8, cloud));
			else if (maxElements == 16) methods.push_back(makeMethod<bgi::rstar<16>>(16, cloud));
			else if (maxElements == 32) methods.push_back(makeMethod<bgi::rstar<32>>(32, cloud));
			else if (maxElements == 64) methods.push_back(makeMethod<bgi::rstar<64>>(64, cloud));
			else throw std::runtime_error("unsupported R-tree node capacity " + std::to_string(maxElements) + " (8, 16, 32 or 64)");
		}
		const std::string version = std::to_string(BOOST_VERSION / 100000) + "." + std::to_string(BOOST_VERSION / 100 % 1000);
		Baseline::runAll("Boost.Geometry rtree " + version, options, cloud, queries, methods);
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << '\n';
		return 1;
	}
}

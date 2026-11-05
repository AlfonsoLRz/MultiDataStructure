// External replay baseline against nanoflann (github.com/jlblancoc/nanoflann, BSD, header-only):
// the widely used C++ kd-tree, single-threaded build and queries, one configuration per leaf size
// (nanoflann's default leaf_max_size is 10). Protocol and output: tools/baseline_common.h.
//
// Points are float in the cloud's local frame, as MultiDataStructure queries them. Radius queries ask
// for unsorted results (nanoflann sorts by default, which is extra work the other libraries are not
// asked to do); kNN returns the k nearest as nanoflann always does. nanoflann's L2 metric works in
// squared distances.
//
// Build: tools/build_baselines_linux.sh (Linux), tools/build_nanoflann_baseline.ps1 (Windows)
// Usage:
//   nanoflann_point_baseline --input <cloud.mdspc> --query-trace <trace.csv> --summary-csv <out.csv>
//                            [--query-csv <q.csv>] [--max-queries N] [--warmup W] [--repeats R]
//                            [--leaf-sizes 10,16,32,64,128,256]

#include "baseline_common.h"

#include <memory>

#include <nanoflann.hpp>

namespace
{
	struct Adaptor
	{
		const Baseline::Cloud* cloud = nullptr;

		size_t kdtree_get_point_count() const { return cloud->size(); }
		float kdtree_get_pt(size_t i, size_t dim) const { return cloud->xyz[i * 3 + dim]; }
		template <class BBox> bool kdtree_get_bbox(BBox&) const { return false; }
	};

	using Tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, Adaptor>, Adaptor, 3, uint32_t>;
}

int main(int argc, char** argv)
{
	try
	{
		const Baseline::Options options = Baseline::parseOptions(argc, argv, { "--leaf-sizes" });
		const Baseline::Cloud cloud = Baseline::loadMdspc(options.input);
		const std::vector<Baseline::Query> queries = Baseline::loadTrace(options.trace, cloud, options.maxQueries);
		const Adaptor adaptor{ &cloud };

		std::unique_ptr<Tree> tree;
		std::vector<nanoflann::ResultItem<uint32_t, float>> hits;
		std::vector<uint32_t> knnIndices;
		std::vector<float> knnDistances;
		size_t lastCount = 0;
		bool lastWasKnn = false;
		nanoflann::SearchParameters unsorted;
		unsorted.sorted = false;

		std::vector<Baseline::Method> methods;
		for (const double leaf : options.list("--leaf-sizes", "10,16,32,64,128,256"))
		{
			Baseline::Method method;
			method.name = "nanoflann_kd_leaf" + std::to_string(static_cast<size_t>(leaf));
			method.build = [&, leaf]() {
				const auto start = Baseline::Clock::now();
				tree = std::make_unique<Tree>(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(static_cast<size_t>(leaf)));
				return Baseline::msSince(start);
			};
			method.query = [&](const Baseline::Query& q) {
				if (q.kind == Baseline::Kind::Radius)
				{
					const auto start = Baseline::Clock::now();
					lastCount = tree->radiusSearch(q.center, q.radius * q.radius, hits, unsorted);
					const double ms = Baseline::msSince(start);
					lastWasKnn = false;
					return ms;
				}
				if (q.kind == Baseline::Kind::Knn)
				{
					knnIndices.resize(q.k);
					knnDistances.resize(q.k);
					const auto start = Baseline::Clock::now();
					lastCount = tree->knnSearch(q.center, q.k, knnIndices.data(), knnDistances.data());
					const double ms = Baseline::msSince(start);
					lastWasKnn = true;
					return ms;
				}
				return -1.0;
			};
			method.answer = [&](size_t& count, uint64_t& checksum) {
				count = lastCount;
				for (size_t i = 0; i < lastCount; ++i)
					Baseline::mixChecksum(checksum, lastWasKnn ? knnIndices[i] : hits[i].first);
			};
			method.release = [&]() { tree.reset(); };
			methods.push_back(std::move(method));
		}
		Baseline::runAll(std::string("nanoflann ") + NANOFLANN_VERSION_STRING, options, cloud, queries, methods);
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << '\n';
		return 1;
	}
}

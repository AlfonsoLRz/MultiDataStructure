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
#if defined(_OPENMP)
#include <omp.h>
#endif

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

	// Throughput mode (--threads N): the trace's queries, cycled to --batch, answered by N OpenMP
	// threads over one tree (schedule dynamic, 64), as engine batch-check's cpuN backend does. One
	// untimed run, then --repeats timed runs; the CSV row matches batch-check's columns that apply.
	int runThroughput(const Baseline::Options& options, const Baseline::Cloud& cloud, const std::vector<Baseline::Query>& trace,
					  const Adaptor& adaptor)
	{
#if !defined(_OPENMP)
		throw std::runtime_error("--threads needs a build with OpenMP (-fopenmp)");
#else
		const int threads = static_cast<int>(options.list("--threads", "1").front());
		const size_t batch = static_cast<size_t>(options.list("--batch", std::to_string(trace.size())).front());
		std::vector<const Baseline::Query*> queries;
		for (size_t i = 0; i < batch && !trace.empty(); ++i)
		{
			const Baseline::Query& q = trace[batch <= trace.size() ? (i * trace.size()) / batch : i % trace.size()];
			if (q.kind == Baseline::Kind::Radius || q.kind == Baseline::Kind::Knn)
				queries.push_back(&q);
		}
		std::ofstream summary;
		if (!options.summaryCsv.empty())
		{
			const bool exists = std::filesystem::exists(options.summaryCsv);
			summary.open(options.summaryCsv, std::ios::app);
			if (!exists)
				summary << "library,cloud,points,schema_name,query_source,backend,threads,batch,median_ms,min_ms,throughput_qps,"
						<< "checksum,mean_returned,build_ms\n";
		}
		nanoflann::SearchParameters unsorted;
		unsorted.sorted = false;
		for (const double leaf : options.list("--leaf-sizes", "10,16,32,64,128,256"))
		{
			const auto buildStart = Baseline::Clock::now();
			Tree tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(static_cast<size_t>(leaf)));
			const double buildMs = Baseline::msSince(buildStart);
			std::vector<double> runs;
			uint64_t checksum = 0;
			double returned = 0.0;
			for (size_t run = 0; run < 1 + options.repeats; ++run)
			{
				uint64_t runChecksum = 0;
				double runReturned = 0.0;
				const auto start = Baseline::Clock::now();
#pragma omp parallel num_threads(threads) reduction(+ : runChecksum, runReturned)
				{
					std::vector<nanoflann::ResultItem<uint32_t, float>> hits;
					std::vector<uint32_t> indices;
					std::vector<float> distances;
#pragma omp for schedule(dynamic, 64)
					for (long long i = 0; i < static_cast<long long>(queries.size()); ++i)
					{
						const Baseline::Query& q = *queries[static_cast<size_t>(i)];
						uint64_t one = 0;
						size_t n = 0;
						if (q.kind == Baseline::Kind::Radius)
						{
							n = tree.radiusSearch(q.center, q.radius * q.radius, hits, unsorted);
							for (size_t h = 0; h < n; ++h)
								Baseline::mixChecksum(one, hits[h].first);
						}
						else
						{
							indices.resize(q.k);
							distances.resize(q.k);
							n = tree.knnSearch(q.center, q.k, indices.data(), distances.data());
							for (size_t h = 0; h < n; ++h)
								Baseline::mixChecksum(one, indices[h]);
						}
						runChecksum += one;
						runReturned += static_cast<double>(n);
					}
				}
				const double ms = Baseline::msSince(start);
				if (run == 0)
				{
					checksum = runChecksum;
					returned = runReturned;
					continue;
				}
				if (runChecksum != checksum)
					throw std::runtime_error("nanoflann throughput: answers changed between runs");
				runs.push_back(ms);
			}
			std::sort(runs.begin(), runs.end());
			const double median = runs[runs.size() / 2];
			const std::string name = "nanoflann_kd_leaf" + std::to_string(static_cast<size_t>(leaf));
			std::cout << "  " << name << " x" << threads << ": " << queries.size() << " queries in " << median << " ms ("
					  << static_cast<double>(queries.size()) / (median * 1e-3) << " q/s)\n";
			if (summary)
				summary << "nanoflann " << NANOFLANN_VERSION_STRING << ',' << Baseline::csvField(options.input) << ',' << cloud.size() << ','
						<< name << ',' << Baseline::csvField(options.trace) << ",nanoflannN," << threads << ',' << queries.size() << ','
						<< median << ',' << runs.front() << ',' << static_cast<double>(queries.size()) / (median * 1e-3) << ','
						<< checksum << ',' << returned / static_cast<double>(std::max<size_t>(1, queries.size())) << ',' << buildMs << '\n';
		}
		return 0;
#endif
	}
}

int main(int argc, char** argv)
{
	try
	{
		const Baseline::Options options = Baseline::parseOptions(argc, argv, { "--leaf-sizes", "--threads", "--batch" });
		const Baseline::Cloud cloud = Baseline::loadMdspc(options.input);
		const std::vector<Baseline::Query> queries = Baseline::loadTrace(options.trace, cloud, options.maxQueries);
		const Adaptor adaptor{ &cloud };
		if (options.extra.count("--threads"))
			return runThroughput(options, cloud, queries, adaptor);

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

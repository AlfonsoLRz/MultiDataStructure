#include "../stdafx.h"
#include "BatchCheck.h"

#include "QueryTrace.h"
#include "../core/Config.h"
#include "../workloads/points/PointCloud.h"
#include "../workloads/points/CudaKdTreeBaseline.h"
#include "../workloads/points/GpuUniformGrid.h"
#include "../workloads/points/PointGpuIndex.h"
#include "../workloads/points/PointNormals.h"
#include "../workloads/points/PointSpatialIndex.h"

#include <omp.h>

namespace
{
	using Kind = PointGpuIndex::Kind;
	using Query = PointGpuIndex::Query;

	// splitmix64, as engine-check: bit-identical batches on every platform.
	struct SplitMix
	{
		uint64_t _state;

		uint64_t next()
		{
			uint64_t z = (_state += 0x9E3779B97F4A7C15ull);
			z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
			z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
			return z ^ (z >> 31);
		}
	};

	// A query's shape without its position: kind, radius, k and the box's half extent.
	struct Template
	{
		Kind		_kind = Kind::Radius;
		glm::vec3	_half = glm::vec3(0.0f);
		float		_radius = 0.0f;
		uint32_t	_k = 0;
	};

	std::vector<Template> templatesFrom(const std::string& tracePath, const PointCloud& cloud, size_t& dropped)
	{
		std::vector<Template> templates;
		if (!tracePath.empty())
		{
			for (const Experiments::TraceQuery& source : Experiments::loadQueryTrace(tracePath))
			{
				Template t;
				switch (source._kind)
				{
				case Experiments::TraceQuery::Kind::Radius: t._kind = Kind::Radius; break;
				case Experiments::TraceQuery::Kind::Knn: t._kind = Kind::Knn; break;
				case Experiments::TraceQuery::Kind::CountRange: t._kind = Kind::CountRange; break;
				default: t._kind = Kind::Range; break;
				}
				if (t._kind == Kind::Knn && source._k > PointGpuIndex::MaxK)
				{
					++dropped;
					continue;
				}
				t._half = glm::abs(glm::vec3(source._maxBound - source._minBound)) * 0.5f;
				t._radius = source._radius;
				t._k = static_cast<uint32_t>(source._k);
				templates.push_back(t);
			}
			return templates;
		}

		// No trace: the common pipeline shapes, radius and kNN, at scales set by the extent.
		const float diagonal = glm::length(cloud.bounds().size());
		for (const float fraction : { 1e-3f, 3e-3f })
		{
			Template t;
			t._kind = Kind::Radius;
			t._radius = diagonal * fraction;
			templates.push_back(t);
		}
		for (const uint32_t k : { 8u, 16u })
		{
			Template t;
			t._kind = Kind::Knn;
			t._k = k;
			templates.push_back(t);
		}
		return templates;
	}

	// Centred on points of the cloud, in its local frame, as pipeline queries are.
	std::vector<Query> batchQueries(const std::vector<Template>& templates, const PointCloud& cloud, size_t count, uint32_t seed)
	{
		std::vector<Query> queries;
		if (templates.empty() || cloud.empty())
			return queries;
		SplitMix rng{ 0xA0761D6478BD642Full ^ seed };
		queries.reserve(count);
		for (size_t i = 0; i < count; ++i)
		{
			const Template& t = templates[rng.next() % templates.size()];
			const glm::vec3 center = cloud.points()[rng.next() % cloud.size()].position;
			Query query;
			query._kind = t._kind;
			switch (t._kind)
			{
			case Kind::Range:
			case Kind::CountRange:
				query._min = center - t._half;
				query._max = center + t._half;
				break;
			case Kind::Radius:
				query._min = center;
				query._radius = t._radius;
				break;
			case Kind::Knn:
				query._min = center;
				query._k = t._k;
				break;
			}
			queries.push_back(query);
		}
		return queries;
	}

	// The k-th neighbourhood's normal from the CPU's own answer (cloud indices in (distance, index) order).
	PointNormals::Estimate cpuNormal(const PointCloud& cloud, const std::vector<size_t>& indices)
	{
		const std::vector<PointPrimitive>& points = cloud.points();
		return PointNormals::fromPoints(indices.size(), [&](size_t i, float* p) {
			const glm::vec3& v = points[indices[i]].position;
			p[0] = v.x;
			p[1] = v.y;
			p[2] = v.z;
		});
	}

	void cpuQuery(const PointSpatialIndex& index, const PointCloud& cloud, const Query& query, uint32_t& count, uint64_t& checksum,
				  PointNormals::Estimate* normal, float& kth)
	{
		constexpr PointSpatialIndex::QueryTelemetry off = PointSpatialIndex::QueryTelemetry::Off;
		switch (query._kind)
		{
		case Kind::Range:
		{
			const PointSpatialIndex::QueryResult result = index.rangeQuery(AABB(query._min, query._max), off);
			count = static_cast<uint32_t>(result._stats._returnedPoints);
			checksum = result._stats._resultChecksum;
			return;
		}
		case Kind::CountRange:
			count = static_cast<uint32_t>(index.countRange(AABB(query._min, query._max), off)._count);
			checksum = 0;
			return;
		case Kind::Radius:
		{
			const PointSpatialIndex::QueryResult result = index.radiusQuery(query._min, query._radius, off);
			count = static_cast<uint32_t>(result._stats._returnedPoints);
			checksum = result._stats._resultChecksum;
			return;
		}
		case Kind::Knn:
		{
			const PointSpatialIndex::QueryResult result = index.knnQuery(query._min, query._k, off);
			count = static_cast<uint32_t>(result._stats._returnedPoints);
			checksum = result._stats._resultChecksum;
			if (normal)
				*normal = cpuNormal(cloud, result._pointIndices);
			if (!result._pointIndices.empty())
			{
				// (dx*dx + dy*dy) + dz*dz, the engine's arithmetic.
				const glm::vec3 d = cloud.points()[result._pointIndices.back()].position - query._min;
				kth = (d.x * d.x + d.y * d.y) + d.z * d.z;
			}
			return;
		}
		}
	}

	struct Answers
	{
		std::vector<uint32_t>				_counts;
		std::vector<uint64_t>				_checksums;
		std::vector<PointNormals::Estimate>	_normals;	// with normals: per query (kNN only)
		std::vector<float>					_kth;		// kNN: squared distance to the farthest neighbour
	};

	double runCpu(const PointSpatialIndex& index, const PointCloud& cloud, const std::vector<Query>& queries, int threads, bool normals,
				  Answers& answers)
	{
		answers._counts.assign(queries.size(), 0);
		answers._checksums.assign(queries.size(), 0);
		answers._normals.assign(normals ? queries.size() : 0, PointNormals::Estimate{});
		answers._kth.assign(queries.size(), 0.0f);
		const int n = static_cast<int>(queries.size());
		const auto start = std::chrono::steady_clock::now();
		const auto one = [&](int q) {
			cpuQuery(index, cloud, queries[q], answers._counts[q], answers._checksums[q], normals ? &answers._normals[q] : nullptr,
					 answers._kth[q]);
		};
		if (threads <= 1)
		{
			for (int q = 0; q < n; ++q)
				one(q);
		}
		else
		{
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
			for (int q = 0; q < n; ++q)
				one(q);
		}
		return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	}

	// Tie-tolerant kNN comparison, for a library that breaks distance ties its own way: the count and
	// the squared distance to the farthest neighbour must match (to rounding).
	size_t kthMismatches(const Answers& reference, const std::vector<uint32_t>& counts, const std::vector<float>& kth)
	{
		size_t different = 0;
		for (size_t q = 0; q < reference._counts.size(); ++q)
		{
			const double expected = reference._kth[q];
			if (reference._counts[q] != counts[q] || std::abs(kth[q] - expected) > 1e-5 * expected + 1e-12)
				++different;
		}
		return different;
	}

	// kNN queries whose normal differs from the reference's (PointNormals::agrees).
	size_t normalMismatches(const Answers& reference, const std::vector<Query>& queries, const std::vector<float>& normals)
	{
		if (reference._normals.empty() || normals.size() < 3 * queries.size())
			return 0;
		size_t different = 0;
		for (size_t q = 0; q < queries.size(); ++q)
			if (queries[q]._kind == Kind::Knn && !PointNormals::agrees(reference._normals[q], reference._counts[q], &normals[3 * q]))
				++different;
		return different;
	}

	size_t normalMismatches(const Answers& reference, const std::vector<Query>& queries, const std::vector<PointNormals::Estimate>& normals)
	{
		std::vector<float> flat(3 * normals.size());
		for (size_t q = 0; q < normals.size(); ++q)
			for (int axis = 0; axis < 3; ++axis)
				flat[3 * q + axis] = normals[q]._normal[axis];
		return normalMismatches(reference, queries, flat);
	}

	// A kNN query at `count` evenly spaced points of the cloud, in cloud order: a normal-estimation
	// pass over the cloud (all of it when count is its size).
	std::vector<Query> selfQueries(const PointCloud& cloud, uint32_t k, size_t count)
	{
		std::vector<Query> queries;
		const size_t n = cloud.size();
		if (n == 0)
			return queries;
		queries.reserve(count);
		for (size_t j = 0; j < count; ++j)
		{
			Query query;
			query._kind = Kind::Knn;
			query._k = k;
			query._min = cloud.points()[static_cast<size_t>((static_cast<uint64_t>(j) * n) / count) % n].position;
			queries.push_back(query);
		}
		return queries;
	}

	// The median distance to the k-th neighbour over up to 1,024 of the batch's kNN queries: the
	// scale of a uniform grid's cells.
	double medianKnnRadius(const PointSpatialIndex& index, const PointCloud& cloud, const std::vector<Query>& batch)
	{
		std::vector<const Query*> knn;
		for (const Query& query : batch)
			if (query._kind == Kind::Knn && query._k > 0)
				knn.push_back(&query);
		if (knn.empty())
			return 0.0;
		std::vector<double> radii;
		const size_t samples = std::min<size_t>(1024, knn.size());
		for (size_t s = 0; s < samples; ++s)
		{
			const Query& query = *knn[(s * knn.size()) / samples];
			const PointSpatialIndex::QueryResult result = index.knnQuery(query._min, query._k, PointSpatialIndex::QueryTelemetry::Off);
			if (!result._pointIndices.empty())
				radii.push_back(glm::length(cloud.points()[result._pointIndices.back()].position - query._min));
		}
		if (radii.empty())
			return 0.0;
		std::nth_element(radii.begin(), radii.begin() + radii.size() / 2, radii.end());
		return radii[radii.size() / 2];
	}

	size_t mismatches(const Answers& reference, const std::vector<uint32_t>& counts, const std::vector<uint64_t>& checksums)
	{
		size_t different = 0;
		for (size_t q = 0; q < reference._counts.size(); ++q)
			if (reference._counts[q] != counts[q] || reference._checksums[q] != checksums[q])
				++different;
		return different;
	}

	double median(std::vector<double> values)
	{
		if (values.empty())
			return 0.0;
		std::sort(values.begin(), values.end());
		return values[values.size() / 2];
	}

	std::string csvField(const std::string& value)
	{
		if (value.find_first_of(",\"\n") == std::string::npos)
			return value;
		std::string escaped = "\"";
		for (const char c : value)
		{
			if (c == '"')
				escaped += '"';
			escaped += c;
		}
		return escaped + '"';
	}

	struct Row
	{
		std::string	_backend;
		int			_threads = 1;
		size_t		_batch = 0;
		double		_medianMs = 0.0;
		double		_minMs = 0.0;
		double		_kernelMs = 0.0;	// GPU: counting + writing traversal (median)
		double		_sortMs = 0.0;
		size_t		_mismatches = 0;
		size_t		_normalMismatches = 0;
		double		_meanReturned = 0.0;
		// The structure: the schema's tree, or a baseline's own (gpugrid).
		std::string	_name;
		std::string	_path;
		uint64_t	_digest = 0;
		size_t		_nodes = 0;
		size_t		_depth = 0;
		size_t		_stackBound = 0;
		double		_buildMs = 0.0;		// tree: CPU build; baseline: device build
		double		_uploadMs = 0.0;
		size_t		_deviceBytes = 0;
		double		_cellSize = 0.0;
	};
}

int Experiments::runBatchCheck(const BatchCheckOptions& options)
{
	if (options._inputPath.empty())
		throw std::invalid_argument("batch-check needs --input <cloud>");
	if (options._schemaPaths.empty())
		throw std::invalid_argument("batch-check needs --schema or --schemas");
	if (options._batchSizes.empty())
		throw std::invalid_argument("batch-check needs --batch-sizes");

	PointCloud::LoadOptions loadOptions;
	loadOptions._useBinaryCache = options._useBinaryCache;
	const PointCloud cloud = PointCloud::load(options._inputPath, loadOptions);

	size_t dropped = 0;
	std::vector<Template> templates;
	if (options._selfKnn == 0)
		templates = templatesFrom(options._inputTracePath, cloud, dropped);
	else if (options._selfKnn > PointGpuIndex::MaxK)
		throw std::invalid_argument("batch-check: --self-knn at most " + std::to_string(PointGpuIndex::MaxK));
	const size_t largest = *std::max_element(options._batchSizes.begin(), options._batchSizes.end());
	const std::vector<Query> queries = options._selfKnn ? std::vector<Query>{} : batchQueries(templates, cloud, largest, options._querySeed);
	// Every schema answers the same batches. A trace or generated batch of size n is the first n
	// queries; a self-query batch spreads its n queries over the whole cloud.
	std::vector<std::vector<Query>> batches;
	for (const size_t size : options._batchSizes)
		batches.push_back(options._selfKnn ? selfQueries(cloud, options._selfKnn, size)
										   : std::vector<Query>(queries.begin(), queries.begin() + std::min(size, queries.size())));

	const auto wants = [&options](const char* backend) {
		return std::find(options._backends.begin(), options._backends.end(), backend) != options._backends.end();
	};
	const bool wantGpu = wants("gpu") || wants("gpuwarp") || wants("gpugrid") || wants("gpucudakd");
	const bool gpu = wantGpu && PointGpuIndex::deviceAvailable();
	if (gpu)
		PointGpuIndex::selectDevice(options._device);
	const int allThreads = options._threads > 0 ? static_cast<int>(options._threads) : omp_get_max_threads();

	std::cout << "batch-check: engine " << PointSpatialIndex::EngineVersion << ", " << cloud.size() << " points, "
			  << (options._selfKnn ? "self-query kNN, k = " + std::to_string(options._selfKnn)
								   : std::to_string(queries.size()) + " queries (" + (options._inputTracePath.empty() ? "generated mix" : "trace mix") +
									 ", centred on cloud points" + (dropped ? ", " + std::to_string(dropped) + " trace kNN with k > 64 left out" : "") + ")")
			  << (options._normals ? ", with normals" : "") << ", cpuN " << allThreads << " threads, GPU "
			  << (gpu ? PointGpuIndex::deviceName() : (wantGpu ? "unavailable" : "off")) << ", " << options._repeats << " timed runs\n";

	std::ofstream summary;
	if (!options._summaryCsvPath.empty())
	{
		std::filesystem::create_directories(std::filesystem::absolute(options._summaryCsvPath).parent_path());
		const bool exists = std::filesystem::exists(options._summaryCsvPath);
		summary.open(options._summaryCsvPath, std::ios::app);
		if (!summary)
			throw std::runtime_error("cannot write " + options._summaryCsvPath);
		if (!exists)
			summary << "engine_version,cloud,points,schema_name,schema_path,structure_digest,num_nodes,max_depth,stack_bound,"
					<< "query_source,backend,threads,batch,gpu_indices,sorted,median_ms,min_ms,throughput_qps,kernel_ms,sort_ms,"
					<< "upload_ms,device_bytes,device,mismatches,mean_returned,normals,normal_mismatches,build_ms,cell_size\n";
	}

	int status = 0;
	const std::string querySource = options._selfKnn ? "self-knn-" + std::to_string(options._selfKnn)
													 : (options._inputTracePath.empty() ? "generated" : options._inputTracePath);
	const auto record = [&](const Row& row) {
		const double qps = row._medianMs > 0.0 ? static_cast<double>(row._batch) / (row._medianMs * 1e-3) : 0.0;
		std::cout << "    " << std::setw(7) << row._backend << " x" << std::setw(2) << (row._threads ? row._threads : 1)
				  << "  batch " << std::setw(8) << row._batch << ": " << std::fixed << std::setprecision(3)
				  << std::setw(10) << row._medianMs << " ms  " << std::setprecision(0) << std::setw(12) << qps << " q/s"
				  << (row._threads == 0 ? "  (kernels " + std::to_string(row._kernelMs) + " ms, sort " + std::to_string(row._sortMs) + " ms)" : std::string())
				  << (row._mismatches ? "  MISMATCHES " + std::to_string(row._mismatches) : std::string())
				  << (row._normalMismatches ? "  NORMAL MISMATCHES " + std::to_string(row._normalMismatches) : std::string()) << '\n';
		std::cout.unsetf(std::ios::floatfield);
		if (row._mismatches || row._normalMismatches)
			status = 2;
		if (summary)
		{
			summary << std::setprecision(10) << PointSpatialIndex::EngineVersion << ',' << csvField(options._inputPath) << ','
					<< cloud.size() << ',' << csvField(row._name) << ',' << csvField(row._path) << ',' << row._digest << ','
					<< row._nodes << ',' << row._depth << ',' << row._stackBound << ',' << csvField(querySource) << ','
					<< row._backend << ',' << row._threads << ',' << row._batch << ',' << (options._gpuIndices ? 1 : 0) << ','
					<< (options._sortQueries ? 1 : 0) << ',' << row._medianMs << ',' << row._minMs << ',' << qps << ','
					<< row._kernelMs << ',' << row._sortMs << ',' << row._uploadMs << ',' << row._deviceBytes << ','
					<< csvField(row._threads == 0 ? PointGpuIndex::deviceName() : "cpu") << ',' << row._mismatches << ','
					<< row._meanReturned << ',' << (options._normals ? 1 : 0) << ',' << row._normalMismatches << ','
					<< row._buildMs << ',' << row._cellSize << '\n';
			summary.flush();
		}
	};

	// The reference answers, from the first schema on every core: each query is answered on its own,
	// and every schema gives the same answers (tests/test_engine_equivalence.cpp), so they are
	// computed once per batch and every backend of every schema is checked against them.
	std::vector<Answers> references(batches.size());
	std::vector<double> meanReturned(batches.size(), 0.0);

	for (size_t schemaNumber = 0; schemaNumber < options._schemaPaths.size(); ++schemaNumber)
	{
		const std::string& schemaPath = options._schemaPaths[schemaNumber];
		const SchemaConfig schema = Config::loadSchemaConfig(schemaPath);
		const std::string name = schema._name.empty() ? std::filesystem::path(schemaPath).stem().string() : schema._name;
		PointSpatialIndex index;
		const auto buildStart = std::chrono::steady_clock::now();
		index.build(cloud, schema);
		const double buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count();
		const PointSpatialIndex::Stats stats = index.stats();
		const uint64_t digest = index.structureDigest();

		PointGpuIndex device;
		bool gpuReady = false;
		if (gpu && (wants("gpu") || wants("gpuwarp")))
		{
			try
			{
				device.upload(index);
				gpuReady = true;
			}
			catch (const std::exception& error)
			{
				std::cout << "  " << name << ": GPU skipped (" << error.what() << ")\n";
			}
		}
		std::cout << "  " << name << ": " << stats._numNodes << " nodes, depth " << stats._maxDepth << ", build " << buildMs << " ms"
				  << (gpuReady ? ", stack bound " + std::to_string(device.stackBound()) + ", upload " +
						  std::to_string(device.uploadMs()) + " ms" : std::string()) << '\n';

		if (schemaNumber == 0)
			for (size_t b = 0; b < batches.size(); ++b)
			{
				runCpu(index, cloud, batches[b], allThreads, options._normals, references[b]);
				for (const uint32_t count : references[b]._counts)
					meanReturned[b] += count;
				meanReturned[b] /= std::max<size_t>(1, batches[b].size());
			}

		// The schema-free baselines run once, alongside the first schema; their cell size comes
		// from that schema's kNN answers.
		std::vector<std::pair<double, std::unique_ptr<GpuUniformGrid>>> grids;
		if (schemaNumber == 0 && gpu && wants("gpugrid"))
		{
			const double radius = medianKnnRadius(index, cloud, batches.back());
			if (radius <= 0.0)
				std::cout << "  gpugrid skipped: no kNN queries with an answer\n";
			for (const double factor : radius > 0.0 ? options._gridCellFactors : std::vector<double>{})
			{
				auto grid = std::make_unique<GpuUniformGrid>();
				try
				{
					grid->build(cloud, static_cast<float>(radius * factor));
					std::cout << "  grid x" << factor << ": cell " << grid->cellSize() << ", " << grid->cells() << " cells, upload "
							  << grid->uploadMs() << " ms, build " << grid->buildMs() << " ms\n";
					grids.emplace_back(factor, std::move(grid));
				}
				catch (const std::exception& error)
				{
					std::cout << "  grid x" << factor << " skipped (" << error.what() << ")\n";
				}
			}
		}

		std::unique_ptr<CudaKdTreeBaseline> cudaKd;
		if (schemaNumber == 0 && gpu && wants("gpucudakd"))
		{
			if (!CudaKdTreeBaseline::available())
				std::cout << "  gpucudakd skipped: built without MDS_CUDAKDTREE_DIR\n";
			else
			{
				try
				{
					cudaKd = std::make_unique<CudaKdTreeBaseline>();
					cudaKd->build(cloud);
					std::cout << "  cudaKDTree: upload " << cudaKd->uploadMs() << " ms, build " << cudaKd->buildMs() << " ms\n";
				}
				catch (const std::exception& error)
				{
					std::cout << "  gpucudakd skipped (" << error.what() << ")\n";
					cudaKd.reset();
				}
			}
		}

		for (size_t b = 0; b < batches.size(); ++b)
		{
			const std::vector<Query>& batch = batches[b];
			const Answers& reference = references[b];

			for (const std::string& backend : options._backends)
			{
				Row row;
				row._backend = backend;
				row._batch = batch.size();
				row._meanReturned = meanReturned[b];
				row._name = name;
				row._path = schemaPath;
				row._digest = digest;
				row._nodes = stats._numNodes;
				row._depth = stats._maxDepth;
				row._stackBound = gpuReady ? device.stackBound() : 0;
				row._buildMs = buildMs;
				row._uploadMs = gpuReady ? device.uploadMs() : 0.0;
				row._deviceBytes = gpuReady ? device.deviceBytes() : 0;
				std::vector<double> times, kernels, sorts;
				if (backend == "cpu1" || backend == "cpuN")
				{
					// One core's throughput does not depend on the batch size; timing it on a million
					// queries only costs minutes.
					if (backend == "cpu1" && options._cpu1MaxBatch > 0 && batch.size() > options._cpu1MaxBatch)
						continue;
					row._threads = backend == "cpu1" ? 1 : allThreads;
					Answers answers;
					if (backend == "cpuN")
						runCpu(index, cloud, batch, row._threads, options._normals, answers);
					for (size_t r = 0; r < std::max<size_t>(1, options._repeats); ++r)
						times.push_back(runCpu(index, cloud, batch, row._threads, options._normals, answers));
					row._mismatches = mismatches(reference, answers._counts, answers._checksums);
					row._normalMismatches = normalMismatches(reference, batch, answers._normals);
				}
				else if (backend == "gpu" || backend == "gpuwarp")
				{
					if (!gpuReady)
						continue;
					row._threads = 0;
					const PointGpuIndex::Traversal traversal = backend == "gpuwarp" ? PointGpuIndex::Traversal::Warp
																					: PointGpuIndex::Traversal::Thread;
					// A batch whose results do not fit on the device (a million large queries can return
					// tens of GB of indices) is skipped; the other batches and schemas still run.
					try
					{
						PointGpuIndex::BatchResult result = device.run(batch, options._gpuIndices, options._sortQueries, traversal, options._normals);
						for (size_t r = 0; r < std::max<size_t>(1, options._repeats); ++r)
						{
							result = device.run(batch, options._gpuIndices, options._sortQueries, traversal, options._normals);
							times.push_back(result._endToEndMs);
							kernels.push_back(result._countMs + result._writeMs);
							sorts.push_back(result._sortMs);
						}
						row._mismatches = mismatches(reference, result._counts, result._checksums);
						row._normalMismatches = normalMismatches(reference, batch, result._normals);
					}
					catch (const std::exception& error)
					{
						std::cout << "    " << std::setw(7) << backend << "  batch " << std::setw(8) << batch.size()
								  << ": skipped (" << error.what() << ")\n";
						continue;
					}
				}
				else if (backend == "gpugrid")
				{
					for (const auto& [factor, grid] : grids)
					{
						Row gridRow = row;
						gridRow._threads = 0;
						std::ostringstream label;
						label << "grid_x" << factor;
						gridRow._name = label.str();
						gridRow._path.clear();
						gridRow._digest = 0;
						gridRow._nodes = grid->cells();
						gridRow._depth = 0;
						gridRow._stackBound = 0;
						gridRow._buildMs = grid->buildMs();
						gridRow._uploadMs = grid->uploadMs();
						gridRow._deviceBytes = grid->deviceBytes();
						gridRow._cellSize = grid->cellSize();
						std::vector<double> gridTimes, gridKernels, gridSorts;
						try
						{
							PointGpuIndex::BatchResult result = grid->run(batch, options._sortQueries, options._normals);
							for (size_t r = 0; r < std::max<size_t>(1, options._repeats); ++r)
							{
								result = grid->run(batch, options._sortQueries, options._normals);
								gridTimes.push_back(result._endToEndMs);
								gridKernels.push_back(result._countMs);
								gridSorts.push_back(result._sortMs);
							}
							gridRow._mismatches = mismatches(reference, result._counts, result._checksums);
							gridRow._normalMismatches = normalMismatches(reference, batch, result._normals);
						}
						catch (const std::exception& error)
						{
							std::cout << "    " << std::setw(7) << backend << "  batch " << std::setw(8) << batch.size()
									  << ": skipped (" << error.what() << ")\n";
							continue;
						}
						gridRow._medianMs = median(gridTimes);
						gridRow._minMs = *std::min_element(gridTimes.begin(), gridTimes.end());
						gridRow._kernelMs = median(gridKernels);
						gridRow._sortMs = median(gridSorts);
						record(gridRow);
					}
					continue;
				}
				else if (backend == "gpucudakd")
				{
					if (!cudaKd)
						continue;
					row._threads = 0;
					row._name = "cudakdtree";
					row._path.clear();
					row._digest = 0;
					row._nodes = cloud.size();
					row._depth = 0;
					row._stackBound = 0;
					row._buildMs = cudaKd->buildMs();
					row._uploadMs = cudaKd->uploadMs();
					row._deviceBytes = cudaKd->deviceBytes();
					try
					{
						std::vector<float> kth;
						PointGpuIndex::BatchResult result = cudaKd->run(batch, options._sortQueries, options._normals, kth);
						for (size_t r = 0; r < std::max<size_t>(1, options._repeats); ++r)
						{
							result = cudaKd->run(batch, options._sortQueries, options._normals, kth);
							times.push_back(result._endToEndMs);
							kernels.push_back(result._countMs);
							sorts.push_back(result._sortMs);
						}
						// Exact sets differ only where cudaKDTree broke a distance tie differently.
						const size_t exact = mismatches(reference, result._counts, result._checksums);
						row._mismatches = kthMismatches(reference, result._counts, kth);
						row._normalMismatches = normalMismatches(reference, batch, result._normals);
						if (exact)
							std::cout << "    gpucudakd  batch " << std::setw(8) << batch.size() << ": " << exact
									  << " neighbour sets differ from the CPU's (distance ties broken by cudaKDTree's point order)\n";
					}
					catch (const std::exception& error)
					{
						std::cout << "    " << std::setw(7) << backend << "  batch " << std::setw(8) << batch.size()
								  << ": skipped (" << error.what() << ")\n";
						continue;
					}
				}
				else
				{
					throw std::invalid_argument("batch-check: unknown backend " + backend + " (cpu1, cpuN, gpu, gpuwarp, gpugrid, gpucudakd)");
				}
				row._medianMs = median(times);
				row._minMs = *std::min_element(times.begin(), times.end());
				row._kernelMs = median(kernels);
				row._sortMs = median(sorts);
				record(row);
			}
		}
	}
	return status;
}

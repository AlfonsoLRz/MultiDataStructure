#include "../stdafx.h"
#include "BatchCheck.h"

#include "QueryTrace.h"
#include "../core/Config.h"
#include "../workloads/points/PointCloud.h"
#include "../workloads/points/PointGpuIndex.h"
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

	void cpuQuery(const PointSpatialIndex& index, const Query& query, uint32_t& count, uint64_t& checksum)
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
			return;
		}
		}
	}

	struct Answers
	{
		std::vector<uint32_t>	_counts;
		std::vector<uint64_t>	_checksums;
	};

	double runCpu(const PointSpatialIndex& index, const std::vector<Query>& queries, int threads, Answers& answers)
	{
		answers._counts.assign(queries.size(), 0);
		answers._checksums.assign(queries.size(), 0);
		const int n = static_cast<int>(queries.size());
		const auto start = std::chrono::steady_clock::now();
		if (threads <= 1)
		{
			for (int q = 0; q < n; ++q)
				cpuQuery(index, queries[q], answers._counts[q], answers._checksums[q]);
		}
		else
		{
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
			for (int q = 0; q < n; ++q)
				cpuQuery(index, queries[q], answers._counts[q], answers._checksums[q]);
		}
		return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
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
		double		_meanReturned = 0.0;
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
	const std::vector<Template> templates = templatesFrom(options._inputTracePath, cloud, dropped);
	const size_t largest = *std::max_element(options._batchSizes.begin(), options._batchSizes.end());
	const std::vector<Query> queries = batchQueries(templates, cloud, largest, options._querySeed);

	const auto wants = [&options](const char* backend) {
		return std::find(options._backends.begin(), options._backends.end(), backend) != options._backends.end();
	};
	const bool wantGpu = wants("gpu") || wants("gpuwarp");
	const bool gpu = wantGpu && PointGpuIndex::deviceAvailable();
	if (gpu)
		PointGpuIndex::selectDevice(options._device);
	const int allThreads = options._threads > 0 ? static_cast<int>(options._threads) : omp_get_max_threads();

	std::cout << "batch-check: engine " << PointSpatialIndex::EngineVersion << ", " << cloud.size() << " points, "
			  << queries.size() << " queries (" << (options._inputTracePath.empty() ? "generated mix" : "trace mix")
			  << ", centred on cloud points" << (dropped ? ", " + std::to_string(dropped) + " trace kNN with k > 64 left out" : "")
			  << "), cpuN " << allThreads << " threads, GPU " << (gpu ? PointGpuIndex::deviceName() : (wantGpu ? "unavailable" : "off"))
			  << ", " << options._repeats << " timed runs\n";

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
					<< "upload_ms,device_bytes,device,mismatches,mean_returned\n";
	}

	int status = 0;
	for (const std::string& schemaPath : options._schemaPaths)
	{
		const SchemaConfig schema = Config::loadSchemaConfig(schemaPath);
		const std::string name = schema._name.empty() ? std::filesystem::path(schemaPath).stem().string() : schema._name;
		PointSpatialIndex index;
		index.build(cloud, schema);
		const PointSpatialIndex::Stats stats = index.stats();
		const uint64_t digest = index.structureDigest();

		PointGpuIndex device;
		bool gpuReady = false;
		if (gpu)
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
		std::cout << "  " << name << ": " << stats._numNodes << " nodes, depth " << stats._maxDepth
				  << (gpuReady ? ", stack bound " + std::to_string(device.stackBound()) + ", upload " +
						  std::to_string(device.uploadMs()) + " ms" : std::string()) << '\n';

		std::vector<Row> rows;
		for (const size_t size : options._batchSizes)
		{
			const std::vector<Query> batch(queries.begin(), queries.begin() + std::min(size, queries.size()));
			// The reference answer, on every core: each query is answered on its own, so the counts
			// and checksums are the single-core ones (cpu1 is checked against them like any backend).
			// Also the cpuN warm-up.
			Answers reference;
			runCpu(index, batch, allThreads, reference);
			double returned = 0.0;
			for (const uint32_t count : reference._counts)
				returned += count;
			returned /= std::max<size_t>(1, batch.size());

			for (const std::string& backend : options._backends)
			{
				Row row;
				row._backend = backend;
				row._batch = batch.size();
				row._meanReturned = returned;
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
						runCpu(index, batch, row._threads, answers);
					for (size_t r = 0; r < std::max<size_t>(1, options._repeats); ++r)
						times.push_back(runCpu(index, batch, row._threads, answers));
					row._mismatches = mismatches(reference, answers._counts, answers._checksums);
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
						PointGpuIndex::BatchResult result = device.run(batch, options._gpuIndices, options._sortQueries, traversal);
						for (size_t r = 0; r < std::max<size_t>(1, options._repeats); ++r)
						{
							result = device.run(batch, options._gpuIndices, options._sortQueries, traversal);
							times.push_back(result._endToEndMs);
							kernels.push_back(result._countMs + result._writeMs);
							sorts.push_back(result._sortMs);
						}
						row._mismatches = mismatches(reference, result._counts, result._checksums);
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
					throw std::invalid_argument("batch-check: unknown backend " + backend + " (cpu1, cpuN, gpu, gpuwarp)");
				}
				row._medianMs = median(times);
				row._minMs = *std::min_element(times.begin(), times.end());
				row._kernelMs = median(kernels);
				row._sortMs = median(sorts);
				rows.push_back(row);

				const double qps = row._medianMs > 0.0 ? static_cast<double>(row._batch) / (row._medianMs * 1e-3) : 0.0;
				std::cout << "    " << std::setw(7) << backend << " x" << std::setw(2) << (row._threads ? row._threads : 1)
						  << "  batch " << std::setw(8) << row._batch << ": " << std::fixed << std::setprecision(3)
						  << std::setw(10) << row._medianMs << " ms  " << std::setprecision(0) << std::setw(12) << qps << " q/s"
						  << (row._threads == 0 ? "  (kernels " + std::to_string(row._kernelMs) + " ms, sort " + std::to_string(row._sortMs) + " ms)" : std::string())
						  << (row._mismatches ? "  MISMATCHES " + std::to_string(row._mismatches) : std::string()) << '\n';
				std::cout.unsetf(std::ios::floatfield);
				if (row._mismatches)
					status = 2;

				if (summary)
				{
					summary << std::setprecision(10) << PointSpatialIndex::EngineVersion << ',' << csvField(options._inputPath) << ','
							<< cloud.size() << ',' << csvField(name) << ',' << csvField(schemaPath) << ',' << digest << ','
							<< stats._numNodes << ',' << stats._maxDepth << ',' << (gpuReady ? device.stackBound() : 0) << ','
							<< (options._inputTracePath.empty() ? "generated" : csvField(options._inputTracePath)) << ','
							<< backend << ',' << row._threads << ',' << row._batch << ',' << (options._gpuIndices ? 1 : 0) << ','
							<< (options._sortQueries ? 1 : 0) << ',' << row._medianMs << ',' << row._minMs << ',' << qps << ','
							<< row._kernelMs << ',' << row._sortMs << ',' << (gpuReady ? device.uploadMs() : 0.0) << ','
							<< (gpuReady ? device.deviceBytes() : 0) << ',' << csvField(row._threads == 0 ? PointGpuIndex::deviceName() : "cpu") << ','
							<< row._mismatches << ',' << row._meanReturned << '\n';
					summary.flush();
				}
			}
		}
	}
	return status;
}

#pragma once

#include "../stdafx.h"

namespace Experiments
{
	// `--mode batch-check`: throughput of whole query batches on one core, on all cores and on the
	// GPU, over the same trees and the same queries. engine-check times queries one at a time, the
	// way an interactive caller issues them; a DL or LiDAR pipeline issues thousands at once (a kNN
	// graph over every point, a ball query at every sampled centre), which is where a GPU should
	// pay. Each schema is built once on the CPU; the GPU gets that exact tree (PointGpuIndex), and
	// every backend's per-query count and checksum must equal the single-core answer.
	//
	// Batch queries keep the trace's mix (kind, radius, k, box size) but are centred on points drawn
	// from the cloud, as pipeline queries are; without a trace, radius and kNN queries at scales set
	// by the cloud's extent. With selfKnn the batch is instead a kNN query at evenly spaced points of
	// the cloud itself, the queries of a normal-estimation pass, and with normals every backend also
	// computes each neighbourhood's PCA normal (checked against the single-core one).
	struct BatchCheckOptions
	{
		std::string					_inputPath;
		std::vector<std::string>	_schemaPaths;
		std::string					_inputTracePath;
		std::string					_summaryCsvPath;
		std::vector<size_t>			_batchSizes{ 1000, 10000, 100000, 1000000 };
		// cpu1, cpuN, gpu (one thread per query), gpuwarp (one warp per query); kNN-only baselines
		// that ignore the schema and run once, with the first schema: gpugrid (GpuUniformGrid).
		std::vector<std::string>	_backends{ "cpu1", "cpuN", "gpu", "gpuwarp" };
		// cpu1 is timed only on batches up to this size (0 = all): its throughput is flat in the
		// batch size, and a million queries on one core take minutes per schema.
		size_t						_cpu1MaxBatch = 100000;
		// Threads of the cpuN backend; 0 = every hardware thread.
		size_t						_threads = 0;
		// Timed runs per (schema, backend, batch size), after one untimed warm-up run.
		size_t						_repeats = 3;
		uint32_t					_querySeed = 1337;
		// CUDA device for the gpu backend (--cuda-device); -1 = the default one.
		int							_device = -1;
		// The GPU also writes every result index (two passes), as the CPU backends do.
		bool						_gpuIndices = true;
		bool						_sortQueries = true;
		bool						_useBinaryCache = true;
		// k of the self-query kNN batch (0 = trace or generated mix).
		uint32_t					_selfKnn = 0;
		bool						_normals = false;
		// gpugrid cell sides, as multiples of the median k-th neighbour distance of the batch.
		std::vector<double>			_gridCellFactors{ 1.0 };
	};

	int runBatchCheck(const BatchCheckOptions& options);
}

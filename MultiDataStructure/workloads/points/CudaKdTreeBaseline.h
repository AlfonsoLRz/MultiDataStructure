#pragma once

#include "PointGpuIndex.h"

// External GPU kNN baseline: Ingo Wald's cudaKDTree (github.com/ingowald/cudaKDTree, Apache-2.0), a
// left-balanced point k-d tree built on the device, here with the widest-dimension split
// (has_explicit_dim), the thrust builder and the closest-corner-tracking kNN traversal, one thread
// per query with a FixedCandidateList. Compiled only when CMake is given MDS_CUDAKDTREE_DIR
// (MDS_WITH_CUDAKDTREE); available() says whether it was.
//
// cudaKDTree breaks distance ties by its own point order, not by cloud index, so its neighbour set
// can differ from the CPU's when several points are exactly as far as the k-th; batch-check
// therefore compares its counts and k-th neighbour distances (kthDistances).
class CudaKdTreeBaseline
{
public:
	CudaKdTreeBaseline();
	~CudaKdTreeBaseline();
	CudaKdTreeBaseline(const CudaKdTreeBaseline&) = delete;
	CudaKdTreeBaseline& operator=(const CudaKdTreeBaseline&) = delete;

	static bool available();

	void build(const PointCloud& cloud);
	// kNN queries with k in {8, 16, 32}. Fills counts, checksums (of cloud indices), timings and,
	// with normals, the neighbourhood normals; kthDistances gets each query's squared distance to
	// its farthest neighbour (0 when it has none).
	PointGpuIndex::BatchResult run(const std::vector<PointGpuIndex::Query>& queries, bool sortQueries, bool normals,
								   std::vector<float>& kthDistances);

	double uploadMs() const;
	double buildMs() const;
	size_t deviceBytes() const;

private:
	struct State;
	std::unique_ptr<State>	_state;
};

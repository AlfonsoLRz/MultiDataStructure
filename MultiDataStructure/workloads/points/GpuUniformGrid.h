#pragma once

#include "PointGpuIndex.h"

// The tree-free baseline for GPU kNN: a uniform grid of cubic cells, built on the device the usual
// way (a cell key per point, a radix sort, one run per occupied cell) with the occupied cells found
// through an open-addressing hash table, so empty space costs nothing. A query is one warp: it
// visits the rings of cells around its own cell, nearest ring first, skipping cells farther than its
// current k-th neighbour, and stops once the next ring cannot hold anything nearer. It shares the
// tree engine's candidate list and normal step (GpuKnnCommon.cuh), so its answers are the CPU's.
// kNN only, k <= 32.
class GpuUniformGrid
{
public:
	GpuUniformGrid();
	~GpuUniformGrid();
	GpuUniformGrid(const GpuUniformGrid&) = delete;
	GpuUniformGrid& operator=(const GpuUniformGrid&) = delete;

	// Uploads the cloud's positions and builds the grid with cells of side cellSize (> 0). Throws
	// when an axis would need more than 2^21 cells.
	void build(const PointCloud& cloud, float cellSize);
	// One batch of kNN queries (any other kind, or k > 32, throws). Fills counts, checksums, the
	// timings and, with normals, the neighbourhood normals, as PointGpuIndex::run does; no indices.
	PointGpuIndex::BatchResult run(const std::vector<PointGpuIndex::Query>& queries, bool sortQueries, bool normals);

	double uploadMs() const;	// host to device copy of the positions
	double buildMs() const;		// device build: keys, sort, cells, hash table (allocations included)
	size_t cells() const;		// occupied cells
	float cellSize() const;
	size_t deviceBytes() const;

private:
	struct State;
	std::unique_ptr<State>	_state;
};

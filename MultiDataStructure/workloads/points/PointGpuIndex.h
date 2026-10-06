#pragma once

#include "PointSpatialIndex.h"

// A built PointSpatialIndex copied to the GPU as it is - the same node pool, the same tree-ordered
// points - and queried there in batches, one thread per query. Any schema the CPU builds (nested,
// sliding-midpoint, STR, tight-centre, any depth) runs unchanged, and every answer equals the CPU's
// bit for bit: the traversals are the CPU's, and distances are computed with separately rounded
// multiplies and adds (no FMA), as the CPU build does.
//
// This replaces nothing in the legacy GPU path (KDTree.cu, MixedTree.cu, ...), which builds its own
// trees on the device; that path cannot measure batch throughput (testing.md 2j).
class PointGpuIndex
{
public:
	enum class Kind : uint32_t
	{
		Range = 0,
		CountRange = 1,
		Radius = 2,
		Knn = 3,
	};

	struct Query
	{
		Kind		_kind = Kind::Range;
		glm::vec3	_min = glm::vec3(0.0f);	// range/count: box minimum; radius/kNN: centre
		glm::vec3	_max = glm::vec3(0.0f);	// range/count: box maximum
		float		_radius = 0.0f;
		uint32_t	_k = 0;
	};

	struct BatchResult
	{
		// Per query, in input order: points returned and the order-independent checksum of their
		// cloud indices (PointSpatialIndex::accumulateResultChecksum), as the CPU reports them; a
		// count query returns no points, so its checksum is 0 there too.
		std::vector<uint32_t>	_counts;
		std::vector<uint64_t>	_checksums;
		// With indices: query q's cloud indices are _indices[_offsets[q], _offsets[q] + _counts[q]),
		// kNN in (distance, index) order like the CPU, range and radius in traversal order. A count
		// query writes none (its range is empty).
		std::vector<uint64_t>	_offsets;
		std::vector<uint32_t>	_indices;
		// Device time from CUDA events: sorting the queries along a Morton curve, the counting
		// traversal (counts, checksums, kNN answers), and with indices the writing traversal.
		double	_sortMs = 0.0;
		double	_countMs = 0.0;
		double	_writeMs = 0.0;
		// Queries uploaded to results downloaded, everything included.
		double	_endToEndMs = 0.0;
	};

	// kNN keeps its k best in per-thread storage; the CPU's depth-first kNN has the same bound.
	static constexpr uint32_t MaxK = 64;
	// Deepest traversal stack the kernels hold per thread (PointSpatialIndex::PoolView::_stackBound).
	static constexpr size_t MaxStack = 1024;

	PointGpuIndex();
	~PointGpuIndex();
	PointGpuIndex(const PointGpuIndex&) = delete;
	PointGpuIndex& operator=(const PointGpuIndex&) = delete;

	static bool deviceAvailable();
	static std::string deviceName();
	// Makes `device` current for this thread's later uploads and runs; -1 keeps the current one.
	static void selectDevice(int device);

	// Copies the index to the device, replacing any previous upload. Throws when the tree needs a
	// deeper stack than MaxStack.
	void upload(const PointSpatialIndex& index);
	// Runs one batch. Throws on a kNN query with k > MaxK. sortQueries orders the batch along a
	// Morton curve of the query centres first, so neighbouring threads walk neighbouring paths.
	BatchResult run(const std::vector<Query>& queries, bool withIndices, bool sortQueries = true);

	double uploadMs() const;
	size_t deviceBytes() const;
	size_t stackBound() const;

private:
	struct State;
	std::unique_ptr<State>	_state;
};

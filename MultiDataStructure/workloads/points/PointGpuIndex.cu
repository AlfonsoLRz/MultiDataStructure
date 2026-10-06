#include "../../stdafx.h"
#include "PointGpuIndex.h"

#include "../../CudaHelper.h"

#include <cub/cub.cuh>

// Every kernel is one thread per query, and is the CPU traversal of PointSpatialIndex.cpp line for
// line: the same root test, the same prune expressions, children pushed in reverse so they pop in
// child order, kNN depth-first with the nearest child first and the same (distance, index) heap.
// Float arithmetic goes through __fmul_rn / __fadd_rn / __fsub_rn, which nvcc never contracts into
// an FMA: the CPU build has no FMA either, and a fused multiply-add rounds once instead of twice,
// which would move points that sit exactly on a radius boundary.
namespace
{
	constexpr uint32_t InternalBit = 0x80000000u;
	constexpr int ThreadsPerBlock = 128;
	constexpr uint32_t MaxK = PointGpuIndex::MaxK;

	struct GpuNode
	{
		float		_lo[3];
		float		_hi[3];
		uint32_t	_first;
		uint32_t	_word;
	};
	static_assert(sizeof(GpuNode) == sizeof(PointSpatialIndex::HotNode), "the device copy is the CPU pool byte for byte");
	static_assert(offsetof(PointSpatialIndex::HotNode, _first) == offsetof(GpuNode, _first), "HotNode layout");
	static_assert(offsetof(PointSpatialIndex::HotNode, _word) == offsetof(GpuNode, _word), "HotNode layout");

	struct GpuQuery
	{
		uint32_t	_kind;
		float		_a[3];	// range/count: minimum; radius/kNN: centre
		float		_b[3];	// range/count: maximum
		float		_radius;
		uint32_t	_k;
	};

	// Kernel groups, which are also the high bits of the sort key: range and count share a kernel.
	enum Group : uint32_t { RangeGroup = 0, RadiusGroup = 1, KnnGroup = 2, GroupCount = 3 };

	uint32_t groupOf(PointGpuIndex::Kind kind)
	{
		switch (kind)
		{
		case PointGpuIndex::Kind::Radius: return RadiusGroup;
		case PointGpuIndex::Kind::Knn: return KnnGroup;
		default: return RangeGroup;
		}
	}

	struct Tree
	{
		const GpuNode* __restrict__		_nodes;
		const float* __restrict__		_x;
		const float* __restrict__		_y;
		const float* __restrict__		_z;
		const uint32_t* __restrict__	_order;
		const uint32_t* __restrict__	_subtree;
		uint32_t						_indexed;
	};

	struct Outputs
	{
		uint64_t*		_counts;
		// Indices each query writes: its count, except a count query, which writes none.
		uint64_t*		_emitted;
		uint64_t*		_checksums;
		const uint64_t*	_offsets;
		uint32_t*		_indices;
		uint32_t*		_knnScratch;	// MaxK answers per kNN thread, (distance, index) order
		int*			_overflow;
	};

	__device__ __forceinline__ GpuNode loadNode(const GpuNode* __restrict__ nodes, uint32_t slot)
	{
		const float4* p = reinterpret_cast<const float4*>(nodes + slot);
		const float4 a = __ldg(p);
		const float4 b = __ldg(p + 1);
		GpuNode node;
		node._lo[0] = a.x; node._lo[1] = a.y; node._lo[2] = a.z;
		node._hi[0] = a.w; node._hi[1] = b.x; node._hi[2] = b.y;
		node._first = __float_as_uint(b.z);
		node._word = __float_as_uint(b.w);
		return node;
	}

	__device__ __forceinline__ bool isInternal(const GpuNode& node) { return (node._word & InternalBit) != 0; }
	__device__ __forceinline__ uint32_t countOf(const GpuNode& node) { return node._word & ~InternalBit; }
	__device__ __forceinline__ bool emptyLeaf(const GpuNode& node) { return node._word == 0; }

	// PointSpatialIndex::accumulateResultChecksum's mixer.
	__device__ __forceinline__ uint64_t mixIndex(uint32_t index)
	{
		uint64_t mixed = static_cast<uint64_t>(index) + 0x9E3779B97F4A7C15ull;
		mixed ^= mixed >> 30;
		mixed *= 0xBF58476D1CE4E5B9ull;
		mixed ^= mixed >> 27;
		mixed *= 0x94D049BB133111EBull;
		mixed ^= mixed >> 31;
		return mixed;
	}

	__device__ __forceinline__ float squared(float d) { return __fmul_rn(d, d); }

	__device__ __forceinline__ float boxDistanceSquared(const GpuNode& node, const float* p)
	{
		float distanceSquared = 0.0f;
		for (int axis = 0; axis < 3; ++axis)
		{
			if (p[axis] < node._lo[axis])
				distanceSquared = __fadd_rn(distanceSquared, squared(__fsub_rn(node._lo[axis], p[axis])));
			else if (p[axis] > node._hi[axis])
				distanceSquared = __fadd_rn(distanceSquared, squared(__fsub_rn(p[axis], node._hi[axis])));
		}
		return distanceSquared;
	}

	__device__ __forceinline__ bool boxOverlaps(const GpuNode& node, const float* mn, const float* mx)
	{
		return !(node._hi[0] < mn[0] || node._lo[0] > mx[0] ||
				 node._hi[1] < mn[1] || node._lo[1] > mx[1] ||
				 node._hi[2] < mn[2] || node._lo[2] > mx[2]);
	}

	__device__ __forceinline__ bool boxInside(const GpuNode& node, const float* mn, const float* mx)
	{
		return node._lo[0] >= mn[0] && node._hi[0] <= mx[0] &&
			   node._lo[1] >= mn[1] && node._hi[1] <= mx[1] &&
			   node._lo[2] >= mn[2] && node._hi[2] <= mx[2];
	}

	// (dx*dx + dy*dy) + dz*dz, the CPU's order.
	__device__ __forceinline__ float pointDistanceSquared(const Tree& tree, uint32_t i, const float* c)
	{
		const float dx = __fsub_rn(__ldg(tree._x + i), c[0]);
		const float dy = __fsub_rn(__ldg(tree._y + i), c[1]);
		const float dz = __fsub_rn(__ldg(tree._z + i), c[2]);
		return __fadd_rn(__fadd_rn(squared(dx), squared(dy)), squared(dz));
	}

	__device__ __forceinline__ bool pointInBox(const Tree& tree, uint32_t i, const float* mn, const float* mx)
	{
		const float x = __ldg(tree._x + i), y = __ldg(tree._y + i), z = __ldg(tree._z + i);
		return x >= mn[0] && x <= mx[0] && y >= mn[1] && y <= mx[1] && z >= mn[2] && z <= mx[2];
	}

	// Range and count. Write = false: counts and checksums; Write = true: the indices, at offsets
	// the counting pass produced.
	template <int StackSize, bool Write>
	__global__ void rangeKernel(Tree tree, const GpuQuery* __restrict__ queries, const uint32_t* __restrict__ permutation,
								uint32_t begin, uint32_t count, Outputs out)
	{
		const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
		if (t >= count)
			return;
		const uint32_t q = permutation[begin + t];
		const GpuQuery query = queries[q];
		const bool countOnly = query._kind == static_cast<uint32_t>(PointGpuIndex::Kind::CountRange);
		if (Write && countOnly)
			return;

		uint64_t found = 0;
		uint64_t checksum = 0;
		uint32_t* destination = Write ? out._indices + out._offsets[q] : nullptr;
		uint32_t stack[StackSize];
		int top = 0;

		const GpuNode root = loadNode(tree._nodes, 0);
		if (!emptyLeaf(root) && boxOverlaps(root, query._a, query._b))
			stack[top++] = 0;
		while (top > 0)
		{
			const uint32_t slot = stack[--top];
			const GpuNode node = loadNode(tree._nodes, slot);

			if (boxInside(node, query._a, query._b))
			{
				const uint32_t first = tree._subtree[2 * slot];
				const uint32_t points = tree._subtree[2 * slot + 1];
				if (countOnly)
				{
					found += points;
					continue;
				}
				for (uint32_t i = first; i < first + points; ++i)
				{
					const uint32_t index = __ldg(tree._order + i);
					if (Write)
						destination[found] = index;
					else
						checksum += mixIndex(index);
					++found;
				}
				continue;
			}

			if (!isInternal(node))
			{
				const uint32_t end = node._first + countOf(node);
				for (uint32_t i = node._first; i < end; ++i)
				{
					if (!pointInBox(tree, i, query._a, query._b))
						continue;
					if (!countOnly)
					{
						const uint32_t index = __ldg(tree._order + i);
						if (Write)
							destination[found] = index;
						else
							checksum += mixIndex(index);
					}
					++found;
				}
				continue;
			}

			for (uint32_t c = countOf(node); c-- > 0;)
			{
				const GpuNode child = loadNode(tree._nodes, node._first + c);
				if (emptyLeaf(child) || !boxOverlaps(child, query._a, query._b))
					continue;
				if (top == StackSize)
				{
					atomicExch(out._overflow, 1);
					return;
				}
				stack[top++] = node._first + c;
			}
		}

		if (!Write)
		{
			out._counts[q] = found;
			out._emitted[q] = countOnly ? 0 : found;
			out._checksums[q] = countOnly ? 0 : checksum;
		}
	}

	template <int StackSize, bool Write>
	__global__ void radiusKernel(Tree tree, const GpuQuery* __restrict__ queries, const uint32_t* __restrict__ permutation,
								 uint32_t begin, uint32_t count, Outputs out)
	{
		const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
		if (t >= count)
			return;
		const uint32_t q = permutation[begin + t];
		const GpuQuery query = queries[q];

		uint64_t found = 0;
		uint64_t checksum = 0;
		uint32_t* destination = Write ? out._indices + out._offsets[q] : nullptr;
		uint32_t stack[StackSize];
		int top = 0;

		// radius >= 0 is false for NaN as well, as on the CPU.
		if (query._radius >= 0.0f)
		{
			const float radiusSquared = __fmul_rn(query._radius, query._radius);
			const GpuNode root = loadNode(tree._nodes, 0);
			if (!emptyLeaf(root) && boxDistanceSquared(root, query._a) <= radiusSquared)
				stack[top++] = 0;
			while (top > 0)
			{
				const uint32_t slot = stack[--top];
				const GpuNode node = loadNode(tree._nodes, slot);
				if (!isInternal(node))
				{
					const uint32_t end = node._first + countOf(node);
					for (uint32_t i = node._first; i < end; ++i)
					{
						if (pointDistanceSquared(tree, i, query._a) > radiusSquared)
							continue;
						const uint32_t index = __ldg(tree._order + i);
						if (Write)
							destination[found] = index;
						else
							checksum += mixIndex(index);
						++found;
					}
					continue;
				}
				for (uint32_t c = countOf(node); c-- > 0;)
				{
					const GpuNode child = loadNode(tree._nodes, node._first + c);
					if (emptyLeaf(child) || boxDistanceSquared(child, query._a) > radiusSquared)
						continue;
					if (top == StackSize)
					{
						atomicExch(out._overflow, 1);
						return;
					}
					stack[top++] = node._first + c;
				}
			}
		}

		if (!Write)
		{
			out._counts[q] = found;
			out._emitted[q] = found;
			out._checksums[q] = checksum;
		}
	}

	struct Candidate
	{
		float		_distance;
		uint32_t	_slot;
	};

	// (distance, index) order, std::pair's, which the CPU's heap uses.
	__device__ __forceinline__ bool less(float da, uint32_t ia, float db, uint32_t ib)
	{
		return da < db || (da == db && ia < ib);
	}

	__device__ void siftDown(float* distances, uint32_t* indices, uint32_t size, uint32_t at)
	{
		for (;;)
		{
			uint32_t largest = at;
			const uint32_t left = 2 * at + 1, right = left + 1;
			if (left < size && less(distances[largest], indices[largest], distances[left], indices[left]))
				largest = left;
			if (right < size && less(distances[largest], indices[largest], distances[right], indices[right]))
				largest = right;
			if (largest == at)
				return;
			const float d = distances[at]; distances[at] = distances[largest]; distances[largest] = d;
			const uint32_t i = indices[at]; indices[at] = indices[largest]; indices[largest] = i;
			at = largest;
		}
	}

	// PointSpatialIndex's offerKnnPoint: an unordered bucket until k entries, then a max-heap on
	// (distance, index). Which heap shape holds the k best does not matter; the set does.
	__device__ __forceinline__ void offer(float* distances, uint32_t* indices, uint32_t& size, uint32_t k, float distance, uint32_t index)
	{
		if (size < k)
		{
			distances[size] = distance;
			indices[size] = index;
			if (++size == k)
				for (uint32_t at = k / 2; at-- > 0;)
					siftDown(distances, indices, k, at);
			return;
		}
		if (distance > distances[0])
			return;
		if (distance < distances[0] || index < indices[0])
		{
			distances[0] = distance;
			indices[0] = index;
			siftDown(distances, indices, k, 0);
		}
	}

	// Depth-first, nearest child first: knnSearchDepthFirst. Write = true also stores the answer,
	// in (distance, index) order, in the thread's MaxK slots of the scratch buffer.
	template <int StackSize, bool Write>
	__global__ void knnKernel(Tree tree, const GpuQuery* __restrict__ queries, const uint32_t* __restrict__ permutation,
							  uint32_t begin, uint32_t count, Outputs out)
	{
		const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
		if (t >= count)
			return;
		const uint32_t q = permutation[begin + t];
		const GpuQuery query = queries[q];
		const uint32_t k = query._k;

		float distances[MaxK];
		uint32_t indices[MaxK];
		uint32_t size = 0;
		if (k > 0 && tree._indexed > 0)
		{
			Candidate stack[StackSize];
			Candidate children[64];
			int top = 0;
			stack[top++] = { boxDistanceSquared(loadNode(tree._nodes, 0), query._a), 0u };
			while (top > 0)
			{
				const Candidate entry = stack[--top];
				if (size == k && entry._distance > distances[0])
					continue;
				const GpuNode node = loadNode(tree._nodes, entry._slot);
				if (!isInternal(node))
				{
					const uint32_t end = node._first + countOf(node);
					for (uint32_t i = node._first; i < end; ++i)
					{
						const float distance = pointDistanceSquared(tree, i, query._a);
						if (size == k && distance > distances[0])
							continue;
						offer(distances, indices, size, k, distance, __ldg(tree._order + i));
					}
					continue;
				}

				// Survivors in descending distance, ties in child order, so the nearest pops first.
				uint32_t survivors = 0;
				const uint32_t childCount = countOf(node);
				for (uint32_t c = 0; c < childCount; ++c)
				{
					const GpuNode child = loadNode(tree._nodes, node._first + c);
					if (emptyLeaf(child))
						continue;
					const float childDistance = boxDistanceSquared(child, query._a);
					if (size == k && childDistance > distances[0])
						continue;
					uint32_t at = survivors++;
					while (at > 0 && children[at - 1]._distance < childDistance)
					{
						children[at] = children[at - 1];
						--at;
					}
					children[at] = { childDistance, node._first + c };
				}
				if (top + static_cast<int>(survivors) > StackSize)
				{
					atomicExch(out._overflow, 1);
					return;
				}
				for (uint32_t i = 0; i < survivors; ++i)
					stack[top++] = children[i];
			}
		}

		// Ascending (distance, index): the CPU's output order.
		for (uint32_t i = 1; i < size; ++i)
		{
			const float d = distances[i];
			const uint32_t index = indices[i];
			uint32_t at = i;
			while (at > 0 && less(d, index, distances[at - 1], indices[at - 1]))
			{
				distances[at] = distances[at - 1];
				indices[at] = indices[at - 1];
				--at;
			}
			distances[at] = d;
			indices[at] = index;
		}

		uint64_t checksum = 0;
		for (uint32_t i = 0; i < size; ++i)
			checksum += mixIndex(indices[i]);
		out._counts[q] = size;
		out._emitted[q] = size;
		out._checksums[q] = checksum;
		if (Write)
		{
			uint32_t* scratch = out._knnScratch + static_cast<size_t>(t) * MaxK;
			for (uint32_t i = 0; i < size; ++i)
				scratch[i] = indices[i];
		}
	}

	__global__ void copyKnnKernel(const uint32_t* __restrict__ permutation, uint32_t begin, uint32_t count, Outputs out)
	{
		const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
		if (t >= count)
			return;
		const uint32_t q = permutation[begin + t];
		const uint32_t* scratch = out._knnScratch + static_cast<size_t>(t) * MaxK;
		uint32_t* destination = out._indices + out._offsets[q];
		for (uint64_t i = 0; i < out._emitted[q]; ++i)
			destination[i] = scratch[i];
	}

	__device__ __forceinline__ uint32_t spreadBits(uint32_t v)
	{
		v = (v | (v << 16)) & 0x030000FFu;
		v = (v | (v << 8)) & 0x0300F00Fu;
		v = (v | (v << 4)) & 0x030C30C3u;
		v = (v | (v << 2)) & 0x09249249u;
		return v;
	}

	// Kernel group in bits 30-31, then 10 bits per axis of the query centre's Morton code (zero
	// when unsorted, so the stable sort only groups).
	__global__ void sortKeyKernel(const GpuQuery* __restrict__ queries, uint32_t count, float3 lo, float3 scale, bool morton,
								  uint32_t* keys, uint32_t* values)
	{
		const uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
		if (q >= count)
			return;
		const GpuQuery query = queries[q];
		const uint32_t kind = query._kind;
		const uint32_t group = kind == static_cast<uint32_t>(PointGpuIndex::Kind::Radius) ? RadiusGroup
			: kind == static_cast<uint32_t>(PointGpuIndex::Kind::Knn) ? KnnGroup : RangeGroup;
		uint32_t code = 0;
		if (morton)
		{
			const bool box = group == RangeGroup;
			const float c[3] = { box ? 0.5f * (query._a[0] + query._b[0]) : query._a[0],
								 box ? 0.5f * (query._a[1] + query._b[1]) : query._a[1],
								 box ? 0.5f * (query._a[2] + query._b[2]) : query._a[2] };
			const float l[3] = { lo.x, lo.y, lo.z };
			const float s[3] = { scale.x, scale.y, scale.z };
			uint32_t cell[3];
			for (int axis = 0; axis < 3; ++axis)
				cell[axis] = static_cast<uint32_t>(fminf(fmaxf((c[axis] - l[axis]) * s[axis], 0.0f), 1023.0f));
			code = (spreadBits(cell[0]) << 2) | (spreadBits(cell[1]) << 1) | spreadBits(cell[2]);
		}
		keys[q] = (group << 30) | code;
		values[q] = q;
	}

	template <typename T>
	void ensure(T*& pointer, size_t& capacity, size_t needed, size_t& bytes)
	{
		if (needed <= capacity)
			return;
		if (pointer)
		{
			CudaHelper::checkError(cudaFree(pointer));
			bytes -= capacity * sizeof(T);
		}
		pointer = nullptr;
		capacity = std::max<size_t>(needed, capacity + capacity / 2);
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&pointer), capacity * sizeof(T)));
		bytes += capacity * sizeof(T);
	}

	template <typename T>
	void ensureHost(T*& pointer, size_t& capacity, size_t needed)
	{
		if (needed <= capacity)
			return;
		if (pointer)
			CudaHelper::checkError(cudaFreeHost(pointer));
		pointer = nullptr;
		capacity = std::max<size_t>(needed, capacity + capacity / 2);
		CudaHelper::checkError(cudaMallocHost(reinterpret_cast<void**>(&pointer), capacity * sizeof(T)));
	}

	template <typename T>
	void release(T*& pointer)
	{
		if (pointer)
			cudaFree(pointer);
		pointer = nullptr;
	}

	unsigned blocksFor(uint32_t count)
	{
		return static_cast<unsigned>((count + ThreadsPerBlock - 1) / ThreadsPerBlock);
	}

	template <int StackSize>
	void launchGroup(uint32_t group, bool write, const Tree& tree, const GpuQuery* queries, const uint32_t* permutation,
					 uint32_t begin, uint32_t count, const Outputs& out)
	{
		const unsigned blocks = blocksFor(count);
		switch (group)
		{
		case RangeGroup:
			write ? rangeKernel<StackSize, true><<<blocks, ThreadsPerBlock>>>(tree, queries, permutation, begin, count, out)
				  : rangeKernel<StackSize, false><<<blocks, ThreadsPerBlock>>>(tree, queries, permutation, begin, count, out);
			break;
		case RadiusGroup:
			write ? radiusKernel<StackSize, true><<<blocks, ThreadsPerBlock>>>(tree, queries, permutation, begin, count, out)
				  : radiusKernel<StackSize, false><<<blocks, ThreadsPerBlock>>>(tree, queries, permutation, begin, count, out);
			break;
		case KnnGroup:
			// The kNN answer is final after the counting pass; with indices it also lands in scratch.
			write ? knnKernel<StackSize, true><<<blocks, ThreadsPerBlock>>>(tree, queries, permutation, begin, count, out)
				  : knnKernel<StackSize, false><<<blocks, ThreadsPerBlock>>>(tree, queries, permutation, begin, count, out);
			break;
		}
		CudaHelper::checkError(cudaGetLastError());
	}

	void launch(size_t stackBound, uint32_t group, bool write, const Tree& tree, const GpuQuery* queries, const uint32_t* permutation,
				uint32_t begin, uint32_t count, const Outputs& out)
	{
		if (stackBound <= 64)
			launchGroup<64>(group, write, tree, queries, permutation, begin, count, out);
		else if (stackBound <= 256)
			launchGroup<256>(group, write, tree, queries, permutation, begin, count, out);
		else
			launchGroup<1024>(group, write, tree, queries, permutation, begin, count, out);
	}

	float elapsed(cudaEvent_t from, cudaEvent_t to)
	{
		float ms = 0.0f;
		CudaHelper::checkError(cudaEventElapsedTime(&ms, from, to));
		return ms;
	}
}

struct PointGpuIndex::State
{
	// The tree.
	GpuNode*	_nodes = nullptr;
	float*		_x = nullptr;
	float*		_y = nullptr;
	float*		_z = nullptr;
	uint32_t*	_order = nullptr;
	uint32_t*	_subtree = nullptr;
	size_t		_nodeCount = 0;
	size_t		_pointCount = 0;
	uint32_t	_indexed = 0;
	size_t		_stackBound = 0;
	float3		_lo = { 0.0f, 0.0f, 0.0f };
	float3		_scale = { 0.0f, 0.0f, 0.0f };
	double		_uploadMs = 0.0;
	size_t		_treeBytes = 0;

	// Batch buffers, grown on demand and kept.
	size_t		_batchBytes = 0;
	GpuQuery*	_queries = nullptr;		size_t _queryCapacity = 0;
	uint32_t*	_keys = nullptr;		size_t _keyCapacity = 0;
	uint32_t*	_sortedKeys = nullptr;	size_t _sortedKeyCapacity = 0;
	uint32_t*	_values = nullptr;		size_t _valueCapacity = 0;
	uint32_t*	_permutation = nullptr;	size_t _permutationCapacity = 0;
	uint64_t*	_counts = nullptr;		size_t _countCapacity = 0;
	uint64_t*	_emitted = nullptr;		size_t _emittedCapacity = 0;
	uint64_t*	_checksums = nullptr;	size_t _checksumCapacity = 0;
	uint64_t*	_offsets = nullptr;		size_t _offsetCapacity = 0;
	uint32_t*	_indices = nullptr;		size_t _indexCapacity = 0;
	uint32_t*	_knnScratch = nullptr;	size_t _knnScratchCapacity = 0;
	uint8_t*	_cubTemp = nullptr;		size_t _cubTempCapacity = 0;
	int*		_overflow = nullptr;

	// Pinned staging, so transfers run at full bus speed.
	GpuQuery*	_hostQueries = nullptr;		size_t _hostQueryCapacity = 0;
	uint64_t*	_hostCounts = nullptr;		size_t _hostCountCapacity = 0;
	uint64_t*	_hostChecksums = nullptr;	size_t _hostChecksumCapacity = 0;
	uint64_t*	_hostOffsets = nullptr;		size_t _hostOffsetCapacity = 0;
	uint32_t*	_hostIndices = nullptr;		size_t _hostIndexCapacity = 0;

	cudaEvent_t	_events[6] = {};

	State()
	{
		for (cudaEvent_t& event : _events)
			CudaHelper::checkError(cudaEventCreate(&event));
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&_overflow), sizeof(int)));
	}

	~State()
	{
		releaseTree();
		release(_queries); release(_keys); release(_sortedKeys); release(_values); release(_permutation);
		release(_counts); release(_emitted); release(_checksums); release(_offsets); release(_indices); release(_knnScratch);
		release(_cubTemp); release(_overflow);
		for (void* host : { static_cast<void*>(_hostQueries), static_cast<void*>(_hostCounts), static_cast<void*>(_hostChecksums),
							static_cast<void*>(_hostOffsets), static_cast<void*>(_hostIndices) })
			if (host)
				cudaFreeHost(host);
		for (cudaEvent_t event : _events)
			cudaEventDestroy(event);
	}

	void releaseTree()
	{
		release(_nodes); release(_x); release(_y); release(_z); release(_order); release(_subtree);
		_nodeCount = _pointCount = 0;
		_treeBytes = 0;
	}

	Tree tree() const
	{
		return Tree{ _nodes, _x, _y, _z, _order, _subtree, _indexed };
	}
};

PointGpuIndex::PointGpuIndex() = default;
PointGpuIndex::~PointGpuIndex() = default;

bool PointGpuIndex::deviceAvailable()
{
	int count = 0;
	return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string PointGpuIndex::deviceName()
{
	int device = 0;
	cudaDeviceProp properties{};
	if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess)
		return "none";
	return properties.name;
}

void PointGpuIndex::upload(const PointSpatialIndex& index)
{
	const PointSpatialIndex::PoolView view = index.poolView();
	if (view._stackBound > MaxStack)
		throw std::runtime_error("PointGpuIndex: this tree needs a traversal stack of " + std::to_string(view._stackBound) +
								 " entries; the GPU kernels hold at most " + std::to_string(MaxStack));
	if (!_state)
		_state = std::make_unique<State>();
	State& s = *_state;
	s.releaseTree();

	const auto start = std::chrono::steady_clock::now();
	const auto copy = [&s](auto*& device, const auto* host, size_t count) {
		using T = std::remove_pointer_t<std::remove_reference_t<decltype(device)>>;
		const size_t bytes = std::max<size_t>(1, count) * sizeof(T);
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&device), bytes));
		if (count)
			CudaHelper::checkError(cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice));
		s._treeBytes += bytes;
	};
	copy(s._nodes, reinterpret_cast<const GpuNode*>(view._nodes), view._nodeCount);
	copy(s._x, view._x, view._pointCount);
	copy(s._y, view._y, view._pointCount);
	copy(s._z, view._z, view._pointCount);
	copy(s._order, view._order, view._pointCount);
	copy(s._subtree, view._subtree.data(), view._subtree.size());
	CudaHelper::checkError(cudaDeviceSynchronize());
	s._uploadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

	s._nodeCount = view._nodeCount;
	s._pointCount = view._pointCount;
	s._indexed = view._subtree.size() >= 2 ? view._subtree[1] : 0;
	s._stackBound = std::max<size_t>(1, view._stackBound);
	if (view._nodeCount > 0)
	{
		const PointSpatialIndex::HotNode& root = view._nodes[0];
		s._lo = make_float3(root._lo[0], root._lo[1], root._lo[2]);
		const auto inverse = [](float lo, float hi) { return hi > lo ? 1024.0f / (hi - lo) : 0.0f; };
		s._scale = make_float3(inverse(root._lo[0], root._hi[0]), inverse(root._lo[1], root._hi[1]), inverse(root._lo[2], root._hi[2]));
	}
}

PointGpuIndex::BatchResult PointGpuIndex::run(const std::vector<Query>& queries, bool withIndices, bool sortQueries)
{
	if (!_state)
		throw std::runtime_error("PointGpuIndex::run before upload");
	State& s = *_state;
	const size_t n = queries.size();
	BatchResult result;
	result._counts.resize(n);
	result._checksums.resize(n);
	if (withIndices)
		result._offsets.resize(n);
	// An empty index answers every query with nothing, as the CPU does.
	if (n == 0 || s._nodeCount == 0)
		return result;
	if (n >= (size_t(1) << 31))
		throw std::runtime_error("PointGpuIndex: batches are limited to 2^31 queries");

	// Host side, untimed: the queries in device form and the size of each kernel group.
	std::array<uint32_t, GroupCount> groupSize{};
	ensureHost(s._hostQueries, s._hostQueryCapacity, n);
	uint32_t knnQueries = 0;
	for (size_t q = 0; q < n; ++q)
	{
		const Query& query = queries[q];
		if (query._kind == Kind::Knn && query._k > MaxK)
			throw std::runtime_error("PointGpuIndex: kNN with k = " + std::to_string(query._k) + " > " + std::to_string(MaxK));
		GpuQuery& g = s._hostQueries[q];
		g._kind = static_cast<uint32_t>(query._kind);
		for (int axis = 0; axis < 3; ++axis)
		{
			g._a[axis] = query._min[axis];
			g._b[axis] = query._max[axis];
		}
		g._radius = query._radius;
		g._k = query._k;
		++groupSize[groupOf(query._kind)];
		knnQueries += query._kind == Kind::Knn ? 1u : 0u;
	}

	// Device buffers, untimed after the first batch of a given size.
	ensure(s._queries, s._queryCapacity, n, s._batchBytes);
	ensure(s._keys, s._keyCapacity, n, s._batchBytes);
	ensure(s._sortedKeys, s._sortedKeyCapacity, n, s._batchBytes);
	ensure(s._values, s._valueCapacity, n, s._batchBytes);
	ensure(s._permutation, s._permutationCapacity, n, s._batchBytes);
	ensure(s._counts, s._countCapacity, n, s._batchBytes);
	ensure(s._emitted, s._emittedCapacity, n, s._batchBytes);
	ensure(s._checksums, s._checksumCapacity, n, s._batchBytes);
	if (withIndices)
	{
		ensure(s._offsets, s._offsetCapacity, n, s._batchBytes);
		ensure(s._knnScratch, s._knnScratchCapacity, std::max<size_t>(1, static_cast<size_t>(knnQueries) * MaxK), s._batchBytes);
	}
	size_t sortBytes = 0, scanBytes = 0;
	CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, s._keys, s._sortedKeys, s._values, s._permutation, static_cast<int>(n)));
	CudaHelper::checkError(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, s._emitted, s._offsets, static_cast<int>(n)));
	ensure(s._cubTemp, s._cubTempCapacity, std::max(sortBytes, scanBytes), s._batchBytes);
	ensureHost(s._hostCounts, s._hostCountCapacity, n);
	ensureHost(s._hostChecksums, s._hostChecksumCapacity, n);
	if (withIndices)
		ensureHost(s._hostOffsets, s._hostOffsetCapacity, n);
	CudaHelper::checkError(cudaMemset(s._overflow, 0, sizeof(int)));
	CudaHelper::checkError(cudaDeviceSynchronize());

	cudaEvent_t* e = s._events;
	const Tree tree = s.tree();
	Outputs out{ s._counts, s._emitted, s._checksums, s._offsets, s._indices, s._knnScratch, s._overflow };

	CudaHelper::checkError(cudaEventRecord(e[0]));
	CudaHelper::checkError(cudaMemcpy(s._queries, s._hostQueries, n * sizeof(GpuQuery), cudaMemcpyHostToDevice));

	// Group, and with sortQueries order each group along a Morton curve.
	CudaHelper::checkError(cudaEventRecord(e[1]));
	sortKeyKernel<<<blocksFor(static_cast<uint32_t>(n)), ThreadsPerBlock>>>(s._queries, static_cast<uint32_t>(n), s._lo, s._scale, sortQueries,
																			 s._keys, s._values);
	CudaHelper::checkError(cudaGetLastError());
	size_t tempBytes = s._cubTempCapacity;
	CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(s._cubTemp, tempBytes, s._keys, s._sortedKeys, s._values, s._permutation,
														   static_cast<int>(n), sortQueries ? 0 : 30, 32));
	CudaHelper::checkError(cudaEventRecord(e[2]));

	// Counting pass: counts and checksums of every query, and the final kNN answers.
	uint32_t begin = 0;
	for (uint32_t group = 0; group < GroupCount; ++group)
	{
		if (groupSize[group])
			launch(s._stackBound, group, withIndices && group == KnnGroup, tree, s._queries, s._permutation, begin, groupSize[group], out);
		begin += groupSize[group];
	}
	CudaHelper::checkError(cudaEventRecord(e[3]));

	uint64_t total = 0;
	if (withIndices)
	{
		tempBytes = s._cubTempCapacity;
		CudaHelper::checkError(cub::DeviceScan::ExclusiveSum(s._cubTemp, tempBytes, s._emitted, s._offsets, static_cast<int>(n)));
		uint64_t lastOffset = 0, lastCount = 0;
		CudaHelper::checkError(cudaMemcpy(&lastOffset, s._offsets + n - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost));
		CudaHelper::checkError(cudaMemcpy(&lastCount, s._emitted + n - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost));
		total = lastOffset + lastCount;
		// Grows only when a batch returns more than any before it: a warm-up batch keeps this out
		// of the measured ones.
		ensure(s._indices, s._indexCapacity, std::max<uint64_t>(1, total), s._batchBytes);
		ensureHost(s._hostIndices, s._hostIndexCapacity, std::max<uint64_t>(1, total));
		out._indices = s._indices;
		CudaHelper::checkError(cudaEventRecord(e[4]));
		begin = 0;
		for (uint32_t group = 0; group < GroupCount; ++group)
		{
			if (groupSize[group] && group != KnnGroup)
				launch(s._stackBound, group, true, tree, s._queries, s._permutation, begin, groupSize[group], out);
			else if (groupSize[group])
			{
				copyKnnKernel<<<blocksFor(groupSize[group]), ThreadsPerBlock>>>(s._permutation, begin, groupSize[group], out);
				CudaHelper::checkError(cudaGetLastError());
			}
			begin += groupSize[group];
		}
	}
	else
	{
		CudaHelper::checkError(cudaEventRecord(e[4]));
	}
	CudaHelper::checkError(cudaEventRecord(e[5]));

	CudaHelper::checkError(cudaMemcpy(s._hostCounts, s._counts, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	CudaHelper::checkError(cudaMemcpy(s._hostChecksums, s._checksums, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	if (withIndices)
	{
		CudaHelper::checkError(cudaMemcpy(s._hostOffsets, s._offsets, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
		if (total)
			CudaHelper::checkError(cudaMemcpy(s._hostIndices, s._indices, total * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	}
	cudaEvent_t done = nullptr;
	CudaHelper::checkError(cudaEventCreate(&done));
	CudaHelper::checkError(cudaEventRecord(done));
	CudaHelper::checkError(cudaEventSynchronize(done));

	int overflow = 0;
	CudaHelper::checkError(cudaMemcpy(&overflow, s._overflow, sizeof(int), cudaMemcpyDeviceToHost));
	if (overflow)
	{
		cudaEventDestroy(done);
		throw std::runtime_error("PointGpuIndex: a traversal outgrew its stack (bound " + std::to_string(s._stackBound) + ")");
	}

	result._sortMs = elapsed(e[1], e[2]);
	result._countMs = elapsed(e[2], e[3]);
	result._writeMs = withIndices ? elapsed(e[4], e[5]) : 0.0;
	result._endToEndMs = elapsed(e[0], done);
	cudaEventDestroy(done);

	for (size_t q = 0; q < n; ++q)
	{
		result._counts[q] = static_cast<uint32_t>(s._hostCounts[q]);
		result._checksums[q] = s._hostChecksums[q];
	}
	if (withIndices)
	{
		result._offsets.assign(s._hostOffsets, s._hostOffsets + n);
		result._indices.assign(s._hostIndices, s._hostIndices + total);
	}
	return result;
}

double PointGpuIndex::uploadMs() const
{
	return _state ? _state->_uploadMs : 0.0;
}

size_t PointGpuIndex::deviceBytes() const
{
	return _state ? _state->_treeBytes + _state->_batchBytes : 0;
}

size_t PointGpuIndex::stackBound() const
{
	return _state ? _state->_stackBound : 0;
}

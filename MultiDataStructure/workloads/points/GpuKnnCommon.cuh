#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "PointNormals.h"

// Device pieces shared by the GPU batch engines (PointGpuIndex on the CPU's own trees, GpuUniformGrid):
// the result checksum, the (distance, index) order and a warp-wide k-nearest candidate list.
namespace GpuKnn
{
	constexpr uint32_t FullMask = 0xFFFFFFFFu;
	constexpr uint32_t NoIndex = 0xFFFFFFFFu;
	// The warp kNN list holds one candidate per lane.
	constexpr uint32_t WarpMaxK = 32;

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

	// (distance, index) order, std::pair's, which the CPU's kNN heap uses.
	__device__ __forceinline__ bool less(float da, uint32_t ia, float db, uint32_t ib)
	{
		return da < db || (da == db && ia < ib);
	}

	__device__ __forceinline__ uint32_t lanesBelow(uint32_t lane) { return (1u << lane) - 1u; }

	// Every lane gets the warp's total.
	__device__ __forceinline__ uint64_t warpSum(uint64_t value)
	{
		for (int offset = 16; offset > 0; offset >>= 1)
			value += __shfl_xor_sync(FullMask, value, offset);
		return value;
	}

	__device__ __forceinline__ float warpSum(float value)
	{
		for (int offset = 16; offset > 0; offset >>= 1)
			value += __shfl_xor_sync(FullMask, value, offset);
		return value;
	}

	// The k best (distance, index) pairs of one query, k <= 32, one per lane and ascending across the
	// warp: lane j holds the j-th best, empty slots are (+inf, NoIndex). It keeps exactly the k
	// smallest pairs offered, so any traversal that offers every point not provably farther than the
	// current k-th gives the CPU's answer. Every member is called by all 32 lanes.
	struct WarpKnn
	{
		float		_distance;
		uint32_t	_index;		// cloud index: the answer and the tie-break
		uint32_t	_slot;		// position in the engine's point arrays, for the coordinates
		uint32_t	_k;
		uint32_t	_lane;

		__device__ __forceinline__ void init(uint32_t k, uint32_t lane)
		{
			_distance = __int_as_float(0x7F800000);	// +inf
			_index = NoIndex;
			_slot = 0;
			_k = k;
			_lane = lane;
		}

		__device__ __forceinline__ float worstDistance() const { return __shfl_sync(FullMask, _distance, _k - 1); }
		__device__ __forceinline__ uint32_t worstIndex() const { return __shfl_sync(FullMask, _index, _k - 1); }

		__device__ __forceinline__ void insert(float distance, uint32_t index, uint32_t slot)
		{
			// The list is sorted, so the lanes whose entry the candidate precedes are a suffix.
			const uint32_t after = __ballot_sync(FullMask, less(distance, index, _distance, _index));
			const uint32_t at = after ? static_cast<uint32_t>(__ffs(after) - 1) : 32u;
			const float upDistance = __shfl_up_sync(FullMask, _distance, 1);
			const uint32_t upIndex = __shfl_up_sync(FullMask, _index, 1);
			const uint32_t upSlot = __shfl_up_sync(FullMask, _slot, 1);
			if (at < _k)
			{
				if (_lane == at)
				{
					_distance = distance;
					_index = index;
					_slot = slot;
				}
				else if (_lane > at)
				{
					_distance = upDistance;
					_index = upIndex;
					_slot = upSlot;
				}
			}
		}

		// Inserts the candidate of every lane where `candidate` is true, lowest lane first.
		__device__ __forceinline__ void offer(bool candidate, float distance, uint32_t index, uint32_t slot)
		{
			for (uint32_t pending = __ballot_sync(FullMask, candidate); pending; pending &= pending - 1)
			{
				const int source = __ffs(pending) - 1;
				insert(__shfl_sync(FullMask, distance, source), __shfl_sync(FullMask, index, source),
					   __shfl_sync(FullMask, slot, source));
			}
		}

		// Offers one point at engine position `slot` (lanes with inRange false offer nothing). The
		// cloud index is loaded only for points at or inside the current k-th distance.
		__device__ __forceinline__ void offerPoint(bool inRange, float distance, uint32_t slot, const uint32_t* __restrict__ order)
		{
			const float worst = worstDistance();
			const uint32_t worstAt = worstIndex();
			bool candidate = false;
			uint32_t index = NoIndex;
			if (inRange && distance <= worst)
			{
				index = __ldg(order + slot);
				candidate = distance < worst || index < worstAt;
			}
			offer(candidate, distance, index, slot);
		}

		__device__ __forceinline__ bool holds() const { return _lane < _k && _index != NoIndex; }
		__device__ __forceinline__ uint32_t count() const { return __popc(__ballot_sync(FullMask, holds())); }
		__device__ __forceinline__ uint64_t checksum() const { return warpSum(holds() ? mixIndex(_index) : uint64_t{ 0 }); }

		// The PCA normal of the neighbourhood, in every lane. Mean, then centred covariance, by warp sums.
		__device__ __forceinline__ PointNormals::Estimate normal(const float* __restrict__ x, const float* __restrict__ y,
																 const float* __restrict__ z) const
		{
			const bool mine = holds();
			const uint32_t n = count();
			if (n == 0)
				return PointNormals::Estimate{};
			const float px = mine ? __ldg(x + _slot) : 0.0f;
			const float py = mine ? __ldg(y + _slot) : 0.0f;
			const float pz = mine ? __ldg(z + _slot) : 0.0f;
			const float inverse = 1.0f / static_cast<float>(n);
			const float mx = warpSum(px) * inverse, my = warpSum(py) * inverse, mz = warpSum(pz) * inverse;
			const float dx = mine ? px - mx : 0.0f, dy = mine ? py - my : 0.0f, dz = mine ? pz - mz : 0.0f;
			const float c[6] = { warpSum(dx * dx), warpSum(dx * dy), warpSum(dx * dz),
								 warpSum(dy * dy), warpSum(dy * dz), warpSum(dz * dz) };
			return PointNormals::fromCovariance(c);
		}
	};
}

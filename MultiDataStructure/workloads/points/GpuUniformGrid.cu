#include "../../stdafx.h"
#include "GpuUniformGrid.h"

#include "../../CudaHelper.h"
#include "GpuKnnCommon.cuh"
#include "PointCloud.h"

#include <cub/cub.cuh>

namespace
{
	using GpuKnn::FullMask;

	constexpr int WarpsPerBlock = 4;
	constexpr int ThreadsPerBlock = 128;
	constexpr uint64_t EmptyKey = ~0ull;
	constexpr int AxisLimit = 1 << 21;

	struct GridQuery
	{
		float		_c[3];
		uint32_t	_k;
	};

	struct Grid
	{
		const float* __restrict__		_x;			// points sorted by cell
		const float* __restrict__		_y;
		const float* __restrict__		_z;
		const uint32_t* __restrict__	_order;		// cloud index of each sorted point
		const uint64_t* __restrict__	_tableKeys;	// cell key, or EmptyKey
		const uint32_t* __restrict__	_tableCells;
		const uint32_t* __restrict__	_cellStart;
		const uint32_t* __restrict__	_cellCount;
		uint64_t						_tableMask;
		float							_lo[3];
		float							_cell;
		float							_inverse;
		int								_dims[3];
	};

	__host__ __device__ __forceinline__ int cellOf(float v, float lo, float inverse, int dim)
	{
		const int c = static_cast<int>(floorf((v - lo) * inverse));
		return c < 0 ? 0 : (c >= dim ? dim - 1 : c);
	}

	__device__ __forceinline__ uint64_t cellKey(int x, int y, int z)
	{
		return (static_cast<uint64_t>(x) << 42) | (static_cast<uint64_t>(y) << 21) | static_cast<uint64_t>(z);
	}

	__device__ __forceinline__ uint64_t hashKey(uint64_t key)
	{
		key ^= key >> 33;
		key *= 0xFF51AFD7ED558CCDull;
		key ^= key >> 33;
		key *= 0xC4CEB9FE1A85EC53ull;
		key ^= key >> 33;
		return key;
	}

	// (dx*dx + dy*dy) + dz*dz with separately rounded operations: the tree engine's and the CPU's.
	__device__ __forceinline__ float distanceSquared(const Grid& g, uint32_t i, const float* c)
	{
		const float dx = __fsub_rn(__ldg(g._x + i), c[0]);
		const float dy = __fsub_rn(__ldg(g._y + i), c[1]);
		const float dz = __fsub_rn(__ldg(g._z + i), c[2]);
		return __fadd_rn(__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)), __fmul_rn(dz, dz));
	}

	__global__ void keyKernel(const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z, uint32_t n,
							  float3 lo, float inverse, int3 dims, uint64_t* keys, uint32_t* values)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n)
			return;
		keys[i] = cellKey(cellOf(x[i], lo.x, inverse, dims.x), cellOf(y[i], lo.y, inverse, dims.y), cellOf(z[i], lo.z, inverse, dims.z));
		values[i] = i;
	}

	__global__ void gatherKernel(const uint32_t* __restrict__ sorted, const float* __restrict__ x, const float* __restrict__ y,
								 const float* __restrict__ z, uint32_t n, float* sx, float* sy, float* sz, uint32_t* order)
	{
		const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
		if (j >= n)
			return;
		const uint32_t i = sorted[j];
		sx[j] = x[i];
		sy[j] = y[i];
		sz[j] = z[i];
		order[j] = i;
	}

	__global__ void insertKernel(const uint64_t* __restrict__ keys, uint32_t cells, uint64_t* tableKeys, uint32_t* tableCells, uint64_t mask)
	{
		const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
		if (c >= cells)
			return;
		const uint64_t key = keys[c];
		for (uint64_t slot = hashKey(key) & mask;; slot = (slot + 1) & mask)
		{
			const unsigned long long previous = atomicCAS(reinterpret_cast<unsigned long long*>(tableKeys + slot),
														  static_cast<unsigned long long>(EmptyKey), static_cast<unsigned long long>(key));
			if (previous == EmptyKey)
			{
				tableCells[slot] = c;
				return;
			}
		}
	}

	__device__ __forceinline__ bool findCell(const Grid& g, uint64_t key, uint32_t& cell)
	{
		for (uint64_t slot = hashKey(key) & g._tableMask;; slot = (slot + 1) & g._tableMask)
		{
			const uint64_t stored = __ldg(reinterpret_cast<const unsigned long long*>(g._tableKeys + slot));
			if (stored == key)
			{
				cell = __ldg(g._tableCells + slot);
				return true;
			}
			if (stored == EmptyKey)
				return false;
		}
	}

	// The id-th cell on the shell at Chebyshev distance r of the 3D block around the query's cell:
	// the two z faces, then the two y faces without their z rows, then the two x faces inside.
	// A shell has 24 r^2 + 2 cells (1 for r = 0).
	__device__ __forceinline__ void shellCell(int r, uint32_t id, int& dx, int& dy, int& dz)
	{
		if (r == 0)
		{
			dx = dy = dz = 0;
			return;
		}
		const uint32_t s = 2 * r + 1, inner = 2 * r - 1;
		if (id < 2 * s * s)
		{
			const uint32_t rest = id % (s * s);
			dz = id < s * s ? -r : r;
			dx = static_cast<int>(rest % s) - r;
			dy = static_cast<int>(rest / s) - r;
			return;
		}
		id -= 2 * s * s;
		if (id < 2 * s * inner)
		{
			const uint32_t rest = id % (s * inner);
			dy = id < s * inner ? -r : r;
			dx = static_cast<int>(rest % s) - r;
			dz = static_cast<int>(rest / s) - (r - 1);
			return;
		}
		id -= 2 * s * inner;
		const uint32_t rest = id % (inner * inner);
		dx = id < inner * inner ? -r : r;
		dy = static_cast<int>(rest % inner) - (r - 1);
		dz = static_cast<int>(rest / inner) - (r - 1);
	}

	template <bool Normals>
	__global__ void __launch_bounds__(WarpsPerBlock * 32)
	gridKnnKernel(Grid g, const GridQuery* __restrict__ queries, const uint32_t* __restrict__ permutation, uint32_t count,
				  uint32_t* counts, uint64_t* checksums, float* normals)
	{
		const uint32_t lane = threadIdx.x & 31u;
		const uint32_t w = blockIdx.x * WarpsPerBlock + (threadIdx.x >> 5);
		if (w >= count)
			return;
		const uint32_t q = permutation[w];
		const GridQuery query = queries[q];
		const uint32_t k = min(query._k, GpuKnn::WarpMaxK);
		GpuKnn::WarpKnn knn;
		knn.init(max(k, 1u), lane);

		if (k > 0)
		{
			int home[3];
			float margin = __int_as_float(0x7F800000);
			for (int axis = 0; axis < 3; ++axis)
			{
				home[axis] = cellOf(query._c[axis], g._lo[axis], g._inverse, g._dims[axis]);
				const float low = g._lo[axis] + home[axis] * g._cell;
				margin = fminf(margin, fminf(query._c[axis] - low, low + g._cell - query._c[axis]));
			}
			margin = fmaxf(margin, 0.0f);
			// Rounding can place a point a hair outside its cell: cell boxes and ring bounds get slack.
			const float slack = 1e-5f * g._cell;
			const int lastRing = max(g._dims[0], max(g._dims[1], g._dims[2]));
			for (int r = 0; r <= lastRing; ++r)
			{
				const uint32_t shell = r == 0 ? 1u : 24u * r * r + 2u;
				for (uint32_t base = 0; base < shell; base += 32)
				{
					const float worst = knn.worstDistance();
					bool found = false;
					uint32_t cell = 0;
					if (base + lane < shell)
					{
						int d[3];
						shellCell(r, base + lane, d[0], d[1], d[2]);
						int c[3];
						bool inside = true;
						float boxDistance = 0.0f;
						for (int axis = 0; axis < 3; ++axis)
						{
							c[axis] = home[axis] + d[axis];
							inside = inside && c[axis] >= 0 && c[axis] < g._dims[axis];
							const float low = g._lo[axis] + c[axis] * g._cell - slack, high = low + g._cell + 2.0f * slack;
							const float gap = query._c[axis] < low ? low - query._c[axis] : (query._c[axis] > high ? query._c[axis] - high : 0.0f);
							boxDistance += gap * gap;
						}
						if (inside && boxDistance <= worst)
							found = findCell(g, cellKey(c[0], c[1], c[2]), cell);
					}
					for (uint32_t pending = __ballot_sync(FullMask, found); pending; pending &= pending - 1)
					{
						const uint32_t visit = __shfl_sync(FullMask, cell, __ffs(pending) - 1);
						const uint32_t start = __ldg(g._cellStart + visit), points = __ldg(g._cellCount + visit);
						for (uint32_t offset = 0; offset < points; offset += 32)
						{
							const bool inRange = offset + lane < points;
							const uint32_t j = start + offset + lane;
							knn.offerPoint(inRange, inRange ? distanceSquared(g, j, query._c) : 0.0f, j, g._order);
						}
					}
				}
				// Every point outside rings 0..r is at least r cells plus the margin away.
				const float bound = fmaxf(0.0f, r * g._cell + margin - slack);
				if (knn.worstDistance() < bound * bound)
					break;
			}
		}

		const uint32_t found = k > 0 ? knn.count() : 0u;
		const uint64_t checksum = k > 0 ? knn.checksum() : 0ull;
		if (lane == 0)
		{
			counts[q] = found;
			checksums[q] = checksum;
		}
		if (Normals)
		{
			const PointNormals::Estimate estimate = k > 0 ? knn.normal(g._x, g._y, g._z) : PointNormals::Estimate{};
			if (lane == 0)
				for (int axis = 0; axis < 3; ++axis)
					normals[3 * static_cast<size_t>(q) + axis] = estimate._normal[axis];
		}
	}

	__device__ __forceinline__ uint32_t spreadBits(uint32_t v)
	{
		v = (v | (v << 16)) & 0x030000FFu;
		v = (v | (v << 8)) & 0x0300F00Fu;
		v = (v | (v << 4)) & 0x030C30C3u;
		v = (v | (v << 2)) & 0x09249249u;
		return v;
	}

	__global__ void mortonKernel(const GridQuery* __restrict__ queries, uint32_t count, float3 lo, float3 scale, uint32_t* keys, uint32_t* values)
	{
		const uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
		if (q >= count)
			return;
		const float c[3] = { queries[q]._c[0], queries[q]._c[1], queries[q]._c[2] };
		const float l[3] = { lo.x, lo.y, lo.z }, s[3] = { scale.x, scale.y, scale.z };
		uint32_t cell[3];
		for (int axis = 0; axis < 3; ++axis)
			cell[axis] = static_cast<uint32_t>(fminf(fmaxf((c[axis] - l[axis]) * s[axis], 0.0f), 1023.0f));
		keys[q] = (spreadBits(cell[0]) << 2) | (spreadBits(cell[1]) << 1) | spreadBits(cell[2]);
		values[q] = q;
	}

	unsigned blocksFor(size_t count, int threads)
	{
		return static_cast<unsigned>((count + threads - 1) / threads);
	}

	template <typename T>
	void allocate(T*& pointer, size_t count, size_t& bytes)
	{
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&pointer), std::max<size_t>(1, count) * sizeof(T)));
		bytes += std::max<size_t>(1, count) * sizeof(T);
	}

	template <typename T>
	void release(T*& pointer)
	{
		if (pointer)
			cudaFree(pointer);
		pointer = nullptr;
	}

	// Grow-only device buffer for batch data (contents are not kept).
	template <typename T>
	void ensure(T*& pointer, size_t& capacity, size_t needed)
	{
		if (needed <= capacity)
			return;
		release(pointer);
		capacity = 0;
		T* allocated = nullptr;
		CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&allocated), needed * sizeof(T)));
		pointer = allocated;
		capacity = needed;
	}

	template <typename T>
	void ensureHost(T*& pointer, size_t& capacity, size_t needed)
	{
		if (needed <= capacity)
			return;
		if (pointer)
			cudaFreeHost(pointer);
		pointer = nullptr;
		capacity = 0;
		T* allocated = nullptr;
		CudaHelper::checkError(cudaMallocHost(reinterpret_cast<void**>(&allocated), needed * sizeof(T)));
		pointer = allocated;
		capacity = needed;
	}

	float elapsed(cudaEvent_t from, cudaEvent_t to)
	{
		float ms = 0.0f;
		CudaHelper::checkError(cudaEventElapsedTime(&ms, from, to));
		return ms;
	}
}

struct GpuUniformGrid::State
{
	float*		_x = nullptr;
	float*		_y = nullptr;
	float*		_z = nullptr;
	uint32_t*	_order = nullptr;
	uint64_t*	_tableKeys = nullptr;
	uint32_t*	_tableCells = nullptr;
	uint32_t*	_cellStart = nullptr;
	uint32_t*	_cellCount = nullptr;
	uint64_t	_tableMask = 0;
	size_t		_points = 0;
	size_t		_cells = 0;
	size_t		_gridBytes = 0;
	float		_lo[3] = { 0.0f, 0.0f, 0.0f };
	float		_extent[3] = { 0.0f, 0.0f, 0.0f };
	float		_cell = 0.0f;
	int			_dims[3] = { 1, 1, 1 };
	double		_uploadMs = 0.0;
	double		_buildMs = 0.0;

	GridQuery*	_queries = nullptr;		size_t _queryCapacity = 0;
	uint32_t*	_keys = nullptr;		size_t _keyCapacity = 0;
	uint32_t*	_sortedKeys = nullptr;	size_t _sortedKeyCapacity = 0;
	uint32_t*	_values = nullptr;		size_t _valueCapacity = 0;
	uint32_t*	_permutation = nullptr;	size_t _permutationCapacity = 0;
	uint32_t*	_counts = nullptr;		size_t _countCapacity = 0;
	uint64_t*	_checksums = nullptr;	size_t _checksumCapacity = 0;
	float*		_normals = nullptr;		size_t _normalCapacity = 0;
	uint8_t*	_temp = nullptr;		size_t _tempCapacity = 0;
	GridQuery*	_hostQueries = nullptr;	size_t _hostQueryCapacity = 0;
	uint32_t*	_hostCounts = nullptr;	size_t _hostCountCapacity = 0;
	uint64_t*	_hostChecksums = nullptr;	size_t _hostChecksumCapacity = 0;
	float*		_hostNormals = nullptr;	size_t _hostNormalCapacity = 0;
	cudaEvent_t	_events[4] = {};

	State()
	{
		for (cudaEvent_t& event : _events)
			CudaHelper::checkError(cudaEventCreate(&event));
	}

	~State()
	{
		releaseGrid();
		release(_queries); release(_keys); release(_sortedKeys); release(_values); release(_permutation);
		release(_counts); release(_checksums); release(_normals); release(_temp);
		for (void* host : { static_cast<void*>(_hostQueries), static_cast<void*>(_hostCounts), static_cast<void*>(_hostChecksums),
							static_cast<void*>(_hostNormals) })
			if (host)
				cudaFreeHost(host);
		for (cudaEvent_t event : _events)
			cudaEventDestroy(event);
	}

	void releaseGrid()
	{
		release(_x); release(_y); release(_z); release(_order);
		release(_tableKeys); release(_tableCells); release(_cellStart); release(_cellCount);
		_points = _cells = _gridBytes = 0;
	}

	Grid grid() const
	{
		Grid g{ _x, _y, _z, _order, _tableKeys, _tableCells, _cellStart, _cellCount, _tableMask,
				{ _lo[0], _lo[1], _lo[2] }, _cell, 1.0f / _cell, { _dims[0], _dims[1], _dims[2] } };
		return g;
	}
};

GpuUniformGrid::GpuUniformGrid() = default;
GpuUniformGrid::~GpuUniformGrid() = default;

void GpuUniformGrid::build(const PointCloud& cloud, float cellSize)
{
	if (!(cellSize > 0.0f))
		throw std::invalid_argument("GpuUniformGrid: the cell size must be positive");
	if (!_state)
		_state = std::make_unique<State>();
	State& s = *_state;
	s.releaseGrid();

	const std::vector<PointPrimitive>& points = cloud.points();
	const size_t n = points.size();
	if (n >= (size_t(1) << 32) - 1)
		throw std::runtime_error("GpuUniformGrid: at most 2^32 - 2 points");
	std::vector<float> x(n), y(n), z(n);
	float lo[3] = { 0.0f, 0.0f, 0.0f }, hi[3] = { 0.0f, 0.0f, 0.0f };
	for (size_t i = 0; i < n; ++i)
	{
		const glm::vec3& p = points[i].position;
		x[i] = p.x; y[i] = p.y; z[i] = p.z;
		for (int axis = 0; axis < 3; ++axis)
		{
			lo[axis] = i == 0 ? p[axis] : std::min(lo[axis], p[axis]);
			hi[axis] = i == 0 ? p[axis] : std::max(hi[axis], p[axis]);
		}
	}
	s._cell = cellSize;
	for (int axis = 0; axis < 3; ++axis)
	{
		s._lo[axis] = lo[axis];
		s._extent[axis] = hi[axis] - lo[axis];
		const double cells = std::floor(static_cast<double>(hi[axis] - lo[axis]) / cellSize) + 1.0;
		if (cells >= AxisLimit)
			throw std::runtime_error("GpuUniformGrid: cell size " + std::to_string(cellSize) + " needs " + std::to_string(cells) +
									 " cells on an axis (limit 2^21)");
		s._dims[axis] = static_cast<int>(cells);
	}
	s._points = n;
	if (n == 0)
		return;

	auto start = std::chrono::steady_clock::now();
	float *px = nullptr, *py = nullptr, *pz = nullptr;
	size_t scratchBytes = 0;
	allocate(px, n, scratchBytes); allocate(py, n, scratchBytes); allocate(pz, n, scratchBytes);
	CudaHelper::checkError(cudaMemcpy(px, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaMemcpy(py, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaMemcpy(pz, z.data(), n * sizeof(float), cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaDeviceSynchronize());
	s._uploadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

	start = std::chrono::steady_clock::now();
	uint64_t *keys = nullptr, *sortedKeys = nullptr, *uniqueKeys = nullptr;
	uint32_t *values = nullptr, *sortedValues = nullptr, *cellCount = nullptr, *cellTotal = nullptr;
	allocate(keys, n, scratchBytes); allocate(sortedKeys, n, scratchBytes); allocate(uniqueKeys, n, scratchBytes);
	allocate(values, n, scratchBytes); allocate(sortedValues, n, scratchBytes); allocate(cellCount, n, scratchBytes);
	allocate(cellTotal, 1, scratchBytes);
	keyKernel<<<blocksFor(n, ThreadsPerBlock), ThreadsPerBlock>>>(px, py, pz, static_cast<uint32_t>(n), make_float3(lo[0], lo[1], lo[2]),
																  1.0f / cellSize, make_int3(s._dims[0], s._dims[1], s._dims[2]), keys, values);
	CudaHelper::checkError(cudaGetLastError());
	size_t sortBytes = 0, encodeBytes = 0, scanBytes = 0;
	CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, keys, sortedKeys, values, sortedValues, static_cast<int>(n), 0, 63));
	CudaHelper::checkError(cub::DeviceRunLengthEncode::Encode(nullptr, encodeBytes, sortedKeys, uniqueKeys, cellCount, cellTotal, static_cast<int>(n)));
	CudaHelper::checkError(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, cellCount, cellCount, static_cast<int>(n)));
	uint8_t* temp = nullptr;
	const size_t tempBytes = std::max(sortBytes, std::max(encodeBytes, scanBytes));
	allocate(temp, tempBytes, scratchBytes);
	size_t bytes = tempBytes;
	CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(temp, bytes, keys, sortedKeys, values, sortedValues, static_cast<int>(n), 0, 63));
	bytes = tempBytes;
	CudaHelper::checkError(cub::DeviceRunLengthEncode::Encode(temp, bytes, sortedKeys, uniqueKeys, cellCount, cellTotal, static_cast<int>(n)));
	uint32_t cells = 0;
	CudaHelper::checkError(cudaMemcpy(&cells, cellTotal, sizeof(uint32_t), cudaMemcpyDeviceToHost));
	s._cells = cells;

	allocate(s._x, n, s._gridBytes); allocate(s._y, n, s._gridBytes); allocate(s._z, n, s._gridBytes); allocate(s._order, n, s._gridBytes);
	allocate(s._cellStart, cells, s._gridBytes); allocate(s._cellCount, cells, s._gridBytes);
	CudaHelper::checkError(cudaMemcpy(s._cellCount, cellCount, cells * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
	bytes = tempBytes;
	CudaHelper::checkError(cub::DeviceScan::ExclusiveSum(temp, bytes, s._cellCount, s._cellStart, static_cast<int>(cells)));
	gatherKernel<<<blocksFor(n, ThreadsPerBlock), ThreadsPerBlock>>>(sortedValues, px, py, pz, static_cast<uint32_t>(n), s._x, s._y, s._z, s._order);
	CudaHelper::checkError(cudaGetLastError());

	// Load factor at most one half.
	uint64_t tableSize = 2;
	while (tableSize < 2ull * cells)
		tableSize <<= 1;
	s._tableMask = tableSize - 1;
	allocate(s._tableKeys, tableSize, s._gridBytes);
	allocate(s._tableCells, tableSize, s._gridBytes);
	CudaHelper::checkError(cudaMemset(s._tableKeys, 0xFF, tableSize * sizeof(uint64_t)));
	insertKernel<<<blocksFor(cells, ThreadsPerBlock), ThreadsPerBlock>>>(uniqueKeys, cells, s._tableKeys, s._tableCells, s._tableMask);
	CudaHelper::checkError(cudaGetLastError());
	CudaHelper::checkError(cudaDeviceSynchronize());
	s._buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

	for (void* scratch : { static_cast<void*>(px), static_cast<void*>(py), static_cast<void*>(pz), static_cast<void*>(keys),
						   static_cast<void*>(sortedKeys), static_cast<void*>(uniqueKeys), static_cast<void*>(values),
						   static_cast<void*>(sortedValues), static_cast<void*>(cellCount), static_cast<void*>(cellTotal),
						   static_cast<void*>(temp) })
		cudaFree(scratch);
}

PointGpuIndex::BatchResult GpuUniformGrid::run(const std::vector<PointGpuIndex::Query>& queries, bool sortQueries, bool normals)
{
	if (!_state)
		throw std::runtime_error("GpuUniformGrid::run before build");
	State& s = *_state;
	const size_t n = queries.size();
	PointGpuIndex::BatchResult result;
	result._counts.assign(n, 0);
	result._checksums.assign(n, 0);
	if (normals)
		result._normals.assign(3 * n, 0.0f);
	if (n == 0 || s._points == 0)
		return result;
	if (n >= (size_t(1) << 31))
		throw std::runtime_error("GpuUniformGrid: batches are limited to 2^31 queries");

	ensureHost(s._hostQueries, s._hostQueryCapacity, n);
	for (size_t q = 0; q < n; ++q)
	{
		const PointGpuIndex::Query& query = queries[q];
		if (query._kind != PointGpuIndex::Kind::Knn || query._k > GpuKnn::WarpMaxK)
			throw std::runtime_error("GpuUniformGrid: kNN queries with k <= 32 only");
		GridQuery& g = s._hostQueries[q];
		g._c[0] = query._min.x; g._c[1] = query._min.y; g._c[2] = query._min.z;
		g._k = query._k;
	}
	ensure(s._queries, s._queryCapacity, n);
	ensure(s._keys, s._keyCapacity, n);
	ensure(s._sortedKeys, s._sortedKeyCapacity, n);
	ensure(s._values, s._valueCapacity, n);
	ensure(s._permutation, s._permutationCapacity, n);
	ensure(s._counts, s._countCapacity, n);
	ensure(s._checksums, s._checksumCapacity, n);
	ensureHost(s._hostCounts, s._hostCountCapacity, n);
	ensureHost(s._hostChecksums, s._hostChecksumCapacity, n);
	if (normals)
	{
		ensure(s._normals, s._normalCapacity, 3 * n);
		ensureHost(s._hostNormals, s._hostNormalCapacity, 3 * n);
	}
	size_t sortBytes = 0;
	CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, s._keys, s._sortedKeys, s._values, s._permutation, static_cast<int>(n)));
	ensure(s._temp, s._tempCapacity, std::max<size_t>(1, sortBytes));
	CudaHelper::checkError(cudaDeviceSynchronize());

	cudaEvent_t* e = s._events;
	CudaHelper::checkError(cudaEventRecord(e[0]));
	CudaHelper::checkError(cudaMemcpy(s._queries, s._hostQueries, n * sizeof(GridQuery), cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaEventRecord(e[1]));
	const auto inverse = [](float extent) { return extent > 0.0f ? 1024.0f / extent : 0.0f; };
	const float3 scale = sortQueries ? make_float3(inverse(s._extent[0]), inverse(s._extent[1]), inverse(s._extent[2])) : make_float3(0.0f, 0.0f, 0.0f);
	mortonKernel<<<blocksFor(n, ThreadsPerBlock), ThreadsPerBlock>>>(s._queries, static_cast<uint32_t>(n), make_float3(s._lo[0], s._lo[1], s._lo[2]),
																	 scale, s._keys, s._values);
	CudaHelper::checkError(cudaGetLastError());
	size_t tempBytes = s._tempCapacity;
	CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(s._temp, tempBytes, s._keys, s._sortedKeys, s._values, s._permutation, static_cast<int>(n),
														   0, sortQueries ? 30 : 1));
	CudaHelper::checkError(cudaEventRecord(e[2]));
	const unsigned blocks = blocksFor(n, WarpsPerBlock);
	if (normals)
		gridKnnKernel<true><<<blocks, WarpsPerBlock * 32>>>(s.grid(), s._queries, s._permutation, static_cast<uint32_t>(n), s._counts, s._checksums, s._normals);
	else
		gridKnnKernel<false><<<blocks, WarpsPerBlock * 32>>>(s.grid(), s._queries, s._permutation, static_cast<uint32_t>(n), s._counts, s._checksums, nullptr);
	CudaHelper::checkError(cudaGetLastError());
	CudaHelper::checkError(cudaEventRecord(e[3]));
	CudaHelper::checkError(cudaMemcpy(s._hostCounts, s._counts, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	CudaHelper::checkError(cudaMemcpy(s._hostChecksums, s._checksums, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	if (normals)
		CudaHelper::checkError(cudaMemcpy(s._hostNormals, s._normals, 3 * n * sizeof(float), cudaMemcpyDeviceToHost));
	cudaEvent_t done = nullptr;
	CudaHelper::checkError(cudaEventCreate(&done));
	CudaHelper::checkError(cudaEventRecord(done));
	CudaHelper::checkError(cudaEventSynchronize(done));
	result._sortMs = elapsed(e[1], e[2]);
	result._countMs = elapsed(e[2], e[3]);
	result._endToEndMs = elapsed(e[0], done);
	cudaEventDestroy(done);

	result._counts.assign(s._hostCounts, s._hostCounts + n);
	result._checksums.assign(s._hostChecksums, s._hostChecksums + n);
	if (normals)
		result._normals.assign(s._hostNormals, s._hostNormals + 3 * n);
	return result;
}

double GpuUniformGrid::uploadMs() const { return _state ? _state->_uploadMs : 0.0; }
double GpuUniformGrid::buildMs() const { return _state ? _state->_buildMs : 0.0; }
size_t GpuUniformGrid::cells() const { return _state ? _state->_cells : 0; }
float GpuUniformGrid::cellSize() const { return _state ? _state->_cell : 0.0f; }
size_t GpuUniformGrid::deviceBytes() const { return _state ? _state->_gridBytes : 0; }

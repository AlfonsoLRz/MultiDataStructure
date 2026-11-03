#include "../../stdafx.h"
#include "CudaKdTreeBaseline.h"

#include "../../CudaHelper.h"
#include "GpuKnnCommon.cuh"
#include "PointCloud.h"

#include "cukd/builder.h"
#include "cukd/knn.h"

#include <cub/cub.cuh>

namespace
{
	constexpr int ThreadsPerBlock = 128;

	struct KdPoint
	{
		float3		_position;
		uint32_t	_id;	// cloud index
		int			_dim;	// split dimension, set by the builder
	};

	struct KdPointTraits : public cukd::default_data_traits<float3>
	{
		using data_t = KdPoint;
		using point_t = float3;
		static inline __host__ __device__ const float3& get_point(const KdPoint& n) { return n._position; }
		static inline __host__ __device__ float get_coord(const KdPoint& n, int d) { return cukd::get_coord(n._position, d); }
		enum { has_explicit_dim = true };
		static inline __host__ __device__ int get_dim(const KdPoint& n) { return n._dim; }
		static inline __host__ __device__ void set_dim(KdPoint& n, int dim) { n._dim = dim; }
	};

	template <int K, bool Normals>
	__global__ void knnKernel(const KdPoint* __restrict__ nodes, int n, const cukd::box_t<float3>* bounds, const float3* __restrict__ queries,
							  const uint32_t* __restrict__ permutation, uint32_t count, uint32_t* counts, uint64_t* checksums, float* kth,
							  float* normals)
	{
		const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
		if (t >= count)
			return;
		const uint32_t q = permutation[t];
		cukd::FixedCandidateList<K> result(__int_as_float(0x7F800000));
		cukd::cct::knn<cukd::FixedCandidateList<K>, KdPoint, KdPointTraits>(result, queries[q], *bounds, nodes, n);

		uint32_t found = 0;
		uint64_t checksum = 0;
		float farthest = 0.0f;
		for (int i = 0; i < K; ++i)
		{
			const int id = result.get_pointID(i);
			if (id < 0)
				continue;
			++found;
			checksum += GpuKnn::mixIndex(nodes[id]._id);
			farthest = fmaxf(farthest, result.get_dist2(i));
		}
		counts[q] = found;
		checksums[q] = checksum;
		kth[q] = farthest;
		if (Normals)
		{
			int ids[K];
			uint32_t m = 0;
			for (int i = 0; i < K; ++i)
				if (result.get_pointID(i) >= 0)
					ids[m++] = result.get_pointID(i);
			const PointNormals::Estimate estimate = PointNormals::fromPoints(m, [&](size_t i, float* p) {
				const float3 v = nodes[ids[i]]._position;
				p[0] = v.x;
				p[1] = v.y;
				p[2] = v.z;
			});
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

	__global__ void mortonKernel(const float3* __restrict__ queries, uint32_t count, float3 lo, float3 scale, uint32_t* keys, uint32_t* values)
	{
		const uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
		if (q >= count)
			return;
		const float c[3] = { queries[q].x, queries[q].y, queries[q].z };
		const float l[3] = { lo.x, lo.y, lo.z }, s[3] = { scale.x, scale.y, scale.z };
		uint32_t cell[3];
		for (int axis = 0; axis < 3; ++axis)
			cell[axis] = static_cast<uint32_t>(fminf(fmaxf((c[axis] - l[axis]) * s[axis], 0.0f), 1023.0f));
		keys[q] = (spreadBits(cell[0]) << 2) | (spreadBits(cell[1]) << 1) | spreadBits(cell[2]);
		values[q] = q;
	}

	template <typename T>
	void release(T*& pointer)
	{
		if (pointer)
			cudaFree(pointer);
		pointer = nullptr;
	}

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

struct CudaKdTreeBaseline::State
{
	KdPoint*				_nodes = nullptr;
	cukd::box_t<float3>*	_bounds = nullptr;
	int						_points = 0;
	float					_lo[3] = { 0.0f, 0.0f, 0.0f };
	float					_extent[3] = { 0.0f, 0.0f, 0.0f };
	double					_uploadMs = 0.0;
	double					_buildMs = 0.0;

	float3*		_queries = nullptr;		size_t _queryCapacity = 0;
	uint32_t*	_keys = nullptr;		size_t _keyCapacity = 0;
	uint32_t*	_sortedKeys = nullptr;	size_t _sortedKeyCapacity = 0;
	uint32_t*	_values = nullptr;		size_t _valueCapacity = 0;
	uint32_t*	_permutation = nullptr;	size_t _permutationCapacity = 0;
	uint32_t*	_counts = nullptr;		size_t _countCapacity = 0;
	uint64_t*	_checksums = nullptr;	size_t _checksumCapacity = 0;
	float*		_kth = nullptr;			size_t _kthCapacity = 0;
	float*		_normals = nullptr;		size_t _normalCapacity = 0;
	uint8_t*	_temp = nullptr;		size_t _tempCapacity = 0;
	float3*		_hostQueries = nullptr;	size_t _hostQueryCapacity = 0;
	uint32_t*	_hostCounts = nullptr;	size_t _hostCountCapacity = 0;
	uint64_t*	_hostChecksums = nullptr;	size_t _hostChecksumCapacity = 0;
	float*		_hostKth = nullptr;		size_t _hostKthCapacity = 0;
	float*		_hostNormals = nullptr;	size_t _hostNormalCapacity = 0;
	cudaEvent_t	_events[4] = {};

	State()
	{
		for (cudaEvent_t& event : _events)
			CudaHelper::checkError(cudaEventCreate(&event));
	}

	~State()
	{
		release(_nodes); release(_bounds);
		release(_queries); release(_keys); release(_sortedKeys); release(_values); release(_permutation);
		release(_counts); release(_checksums); release(_kth); release(_normals); release(_temp);
		for (void* host : { static_cast<void*>(_hostQueries), static_cast<void*>(_hostCounts), static_cast<void*>(_hostChecksums),
							static_cast<void*>(_hostKth), static_cast<void*>(_hostNormals) })
			if (host)
				cudaFreeHost(host);
		for (cudaEvent_t event : _events)
			cudaEventDestroy(event);
	}
};

CudaKdTreeBaseline::CudaKdTreeBaseline() = default;
CudaKdTreeBaseline::~CudaKdTreeBaseline() = default;

bool CudaKdTreeBaseline::available()
{
	return true;
}

void CudaKdTreeBaseline::build(const PointCloud& cloud)
{
	if (!_state)
		_state = std::make_unique<State>();
	State& s = *_state;
	release(s._nodes);
	release(s._bounds);
	const std::vector<PointPrimitive>& points = cloud.points();
	if (points.size() >= (size_t(1) << 31))
		throw std::runtime_error("CudaKdTreeBaseline: at most 2^31 - 1 points");
	s._points = static_cast<int>(points.size());
	std::vector<KdPoint> host(points.size());
	float hi[3] = { 0.0f, 0.0f, 0.0f };
	for (size_t i = 0; i < points.size(); ++i)
	{
		const glm::vec3& p = points[i].position;
		host[i] = KdPoint{ make_float3(p.x, p.y, p.z), static_cast<uint32_t>(i), 0 };
		for (int axis = 0; axis < 3; ++axis)
		{
			s._lo[axis] = i == 0 ? p[axis] : std::min(s._lo[axis], p[axis]);
			hi[axis] = i == 0 ? p[axis] : std::max(hi[axis], p[axis]);
		}
	}
	for (int axis = 0; axis < 3; ++axis)
		s._extent[axis] = hi[axis] - s._lo[axis];

	auto start = std::chrono::steady_clock::now();
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&s._nodes), std::max<size_t>(1, host.size()) * sizeof(KdPoint)));
	CudaHelper::checkError(cudaMalloc(reinterpret_cast<void**>(&s._bounds), sizeof(cukd::box_t<float3>)));
	if (!host.empty())
		CudaHelper::checkError(cudaMemcpy(s._nodes, host.data(), host.size() * sizeof(KdPoint), cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaDeviceSynchronize());
	s._uploadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

	start = std::chrono::steady_clock::now();
	if (s._points > 0)
		cukd::buildTree<KdPoint, KdPointTraits>(s._nodes, s._points, s._bounds);
	CudaHelper::checkError(cudaGetLastError());
	CudaHelper::checkError(cudaDeviceSynchronize());
	s._buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

PointGpuIndex::BatchResult CudaKdTreeBaseline::run(const std::vector<PointGpuIndex::Query>& queries, bool sortQueries, bool normals,
												   std::vector<float>& kthDistances)
{
	if (!_state)
		throw std::runtime_error("CudaKdTreeBaseline::run before build");
	State& s = *_state;
	const size_t n = queries.size();
	PointGpuIndex::BatchResult result;
	result._counts.assign(n, 0);
	result._checksums.assign(n, 0);
	kthDistances.assign(n, 0.0f);
	if (normals)
		result._normals.assign(3 * n, 0.0f);
	if (n == 0 || s._points == 0)
		return result;
	const uint32_t k = queries.front()._k;
	if (k != 8 && k != 16 && k != 32)
		throw std::runtime_error("CudaKdTreeBaseline: kNN with k = 8, 16 or 32");
	ensureHost(s._hostQueries, s._hostQueryCapacity, n);
	for (size_t q = 0; q < n; ++q)
	{
		if (queries[q]._kind != PointGpuIndex::Kind::Knn || queries[q]._k != k)
			throw std::runtime_error("CudaKdTreeBaseline: one kNN k per batch");
		s._hostQueries[q] = make_float3(queries[q]._min.x, queries[q]._min.y, queries[q]._min.z);
	}
	ensure(s._queries, s._queryCapacity, n);
	ensure(s._keys, s._keyCapacity, n);
	ensure(s._sortedKeys, s._sortedKeyCapacity, n);
	ensure(s._values, s._valueCapacity, n);
	ensure(s._permutation, s._permutationCapacity, n);
	ensure(s._counts, s._countCapacity, n);
	ensure(s._checksums, s._checksumCapacity, n);
	ensure(s._kth, s._kthCapacity, n);
	ensureHost(s._hostCounts, s._hostCountCapacity, n);
	ensureHost(s._hostChecksums, s._hostChecksumCapacity, n);
	ensureHost(s._hostKth, s._hostKthCapacity, n);
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
	const unsigned blocks = static_cast<unsigned>((n + ThreadsPerBlock - 1) / ThreadsPerBlock);
	CudaHelper::checkError(cudaEventRecord(e[0]));
	CudaHelper::checkError(cudaMemcpy(s._queries, s._hostQueries, n * sizeof(float3), cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaEventRecord(e[1]));
	const auto inverse = [](float extent) { return extent > 0.0f ? 1024.0f / extent : 0.0f; };
	const float3 scale = sortQueries ? make_float3(inverse(s._extent[0]), inverse(s._extent[1]), inverse(s._extent[2])) : make_float3(0.0f, 0.0f, 0.0f);
	mortonKernel<<<blocks, ThreadsPerBlock>>>(s._queries, static_cast<uint32_t>(n), make_float3(s._lo[0], s._lo[1], s._lo[2]), scale, s._keys, s._values);
	CudaHelper::checkError(cudaGetLastError());
	size_t tempBytes = s._tempCapacity;
	CudaHelper::checkError(cub::DeviceRadixSort::SortPairs(s._temp, tempBytes, s._keys, s._sortedKeys, s._values, s._permutation, static_cast<int>(n),
														   0, sortQueries ? 30 : 1));
	CudaHelper::checkError(cudaEventRecord(e[2]));
	const auto launch = [&](auto kTag, auto normalsTag) {
		constexpr int K = decltype(kTag)::value;
		constexpr bool N = decltype(normalsTag)::value;
		knnKernel<K, N><<<blocks, ThreadsPerBlock>>>(s._nodes, s._points, s._bounds, s._queries, s._permutation, static_cast<uint32_t>(n),
													 s._counts, s._checksums, s._kth, s._normals);
	};
	using True = std::integral_constant<bool, true>;
	using False = std::integral_constant<bool, false>;
	if (k == 8)
		normals ? launch(std::integral_constant<int, 8>{}, True{}) : launch(std::integral_constant<int, 8>{}, False{});
	else if (k == 16)
		normals ? launch(std::integral_constant<int, 16>{}, True{}) : launch(std::integral_constant<int, 16>{}, False{});
	else
		normals ? launch(std::integral_constant<int, 32>{}, True{}) : launch(std::integral_constant<int, 32>{}, False{});
	CudaHelper::checkError(cudaGetLastError());
	CudaHelper::checkError(cudaEventRecord(e[3]));
	CudaHelper::checkError(cudaMemcpy(s._hostCounts, s._counts, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	CudaHelper::checkError(cudaMemcpy(s._hostChecksums, s._checksums, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	CudaHelper::checkError(cudaMemcpy(s._hostKth, s._kth, n * sizeof(float), cudaMemcpyDeviceToHost));
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
	kthDistances.assign(s._hostKth, s._hostKth + n);
	if (normals)
		result._normals.assign(s._hostNormals, s._hostNormals + 3 * n);
	return result;
}

double CudaKdTreeBaseline::uploadMs() const { return _state ? _state->_uploadMs : 0.0; }
double CudaKdTreeBaseline::buildMs() const { return _state ? _state->_buildMs : 0.0; }
size_t CudaKdTreeBaseline::deviceBytes() const
{
	return _state ? static_cast<size_t>(_state->_points) * sizeof(KdPoint) : 0;
}

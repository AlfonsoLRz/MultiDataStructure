#include "../../stdafx.h"
#include "CudaKdTreeBaseline.h"

// Without MDS_CUDAKDTREE_DIR the cudaKDTree baseline is not compiled; this stub keeps batch-check
// linking and reports the backend as unavailable.
#if !defined(MDS_WITH_CUDAKDTREE)
struct CudaKdTreeBaseline::State
{
};

CudaKdTreeBaseline::CudaKdTreeBaseline() = default;
CudaKdTreeBaseline::~CudaKdTreeBaseline() = default;

bool CudaKdTreeBaseline::available()
{
	return false;
}

void CudaKdTreeBaseline::build(const PointCloud&)
{
	throw std::runtime_error("cudaKDTree baseline not built (configure with -DMDS_CUDAKDTREE_DIR=<cudaKDTree checkout>)");
}

PointGpuIndex::BatchResult CudaKdTreeBaseline::run(const std::vector<PointGpuIndex::Query>&, bool, bool, std::vector<float>&)
{
	throw std::runtime_error("cudaKDTree baseline not built");
}

double CudaKdTreeBaseline::uploadMs() const { return 0.0; }
double CudaKdTreeBaseline::buildMs() const { return 0.0; }
size_t CudaKdTreeBaseline::deviceBytes() const { return 0; }
#endif

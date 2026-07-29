# Framework Baseline Comparison

- Input: `D:\Datasets\Point Clouds\Alhambra\Alhambra_100M_binary.ply`
- Workload: `configs/workloads/pipeline_replay.json`
- Schema: `C:\Github\MultiDataStructure\results\eval_traces\v2\best_schemas\alhambra_100m_ga\Alhambra_100M_pipeline_replay_best_schema.json`
- Query trace: `C:\Github\MultiDataStructure\results\framework_compare\v2_alhambra_100m_pcl\query_trace.csv`

| Method | Status | Queries | Build ms | Avg query ms | P95 query ms | Mismatches | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| multidatastructure_cpu | ok | 3000 | 67453.683400 | 0.012930 | 0.023700 | 0 | Project CPU evaluator using the resolved schema. |
| pcl_kdtree | ok | 3000 | 12385.200000 | 0.035645 | 0.102600 | 0 | KdTreeFLANN radius/KNN only. |
| pcl_octree | ok | 3000 | 2109.890000 | 0.058959 | 0.121505 | 0 | OctreePointCloudSearch range/radius/KNN. |

## Per-Type Detail

### multidatastructure_cpu

| Query type | Queries | Avg ms | P95 ms |
|---|---:|---:|---:|
| knn | 1000 | 0.008414 | 0.011400 |
| radius | 2000 | 0.015188 | 0.025705 |

### pcl_kdtree

| Query type | Queries | Avg ms | P95 ms |
|---|---:|---:|---:|
| knn | 1000 | 0.002976 | 0.004305 |
| radius | 2000 | 0.051980 | 0.118800 |

### pcl_octree

| Query type | Queries | Avg ms | P95 ms |
|---|---:|---:|---:|
| knn | 1000 | 0.070205 | 0.127605 |
| radius | 2000 | 0.053336 | 0.115210 |

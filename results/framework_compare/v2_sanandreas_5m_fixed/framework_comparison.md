# Framework Baseline Comparison

- Input: `D:\Datasets\Point Clouds\SanAndreas\5M.las`
- Workload: `configs/workloads/pipeline_replay.json`
- Schema: `C:\Github\MultiDataStructure\results\eval_traces\v2\best_schemas\sanandreas_5m_ga\5M_pipeline_replay_best_schema.json`
- Query trace: `C:\Github\MultiDataStructure\results\framework_compare\v2_sanandreas_5m_fixed\query_trace.csv`

| Method | Status | Queries | Build ms | Avg query ms | P95 query ms | Mismatches | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| multidatastructure_cpu | ok | 3000 | 6538.105900 | 0.003632 | 0.005800 | 0 | Project CPU evaluator using the resolved schema. |
| open3d | ok | 3000 | 3055.589200 | 0.013298 | 0.010800 | 1 | KDTreeFlann for radius/KNN; AxisAlignedBoundingBox crop for range. |

## Per-Type Detail

### multidatastructure_cpu

| Query type | Queries | Avg ms | P95 ms |
|---|---:|---:|---:|
| knn | 1000 | 0.005046 | 0.006300 |
| radius | 2000 | 0.002925 | 0.004000 |

### open3d

| Query type | Queries | Avg ms | P95 ms |
|---|---:|---:|---:|
| knn | 1000 | 0.006218 | 0.007800 |
| radius | 2000 | 0.016839 | 0.011205 |

# Framework Baseline Comparison

- Input: `D:\Datasets\Point Clouds\Alhambra\Alhambra_100M.las`
- Workload: `configs/workloads/pipeline_replay.json`
- Schema: `C:\Github\MultiDataStructure\results\eval_traces\v2\best_schemas\alhambra_100m_ga\Alhambra_100M_pipeline_replay_best_schema.json`
- Query trace: `C:\Github\MultiDataStructure\results\framework_compare\v2_alhambra_100m_fixed\query_trace.csv`

| Method | Status | Queries | Build ms | Avg query ms | P95 query ms | Mismatches | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| multidatastructure_cpu | ok | 3000 | 68132.599800 | 0.012922 | 0.023900 | 0 | Project CPU evaluator using the resolved schema. |
| open3d | ok | 3000 | 32527.089500 | 0.047927 | 0.111660 | 10 | KDTreeFlann for radius/KNN; AxisAlignedBoundingBox crop for range. |

## Per-Type Detail

### multidatastructure_cpu

| Query type | Queries | Avg ms | P95 ms |
|---|---:|---:|---:|
| knn | 1000 | 0.008477 | 0.011900 |
| radius | 2000 | 0.015144 | 0.025805 |

### open3d

| Query type | Queries | Avg ms | P95 ms |
|---|---:|---:|---:|
| knn | 1000 | 0.006799 | 0.009200 |
| radius | 2000 | 0.068492 | 0.131700 |

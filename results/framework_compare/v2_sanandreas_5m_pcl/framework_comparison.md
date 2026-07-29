# Framework Baseline Comparison

- Input: `C:\Github\MultiDataStructure\results\traces\sanandreas_5m\5M_binary.ply`
- Workload: `configs/workloads/pipeline_replay.json`
- Schema: `C:\Github\MultiDataStructure\configs\schemas\quadtree.json`
- Query trace: `C:\Github\MultiDataStructure\results\framework_compare\v2_sanandreas_5m_pcl\query_trace.csv`

| Method | Status | Queries | Build ms | Avg query ms | P95 query ms | Mismatches | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| multidatastructure_cpu | ok | 6000 | 6444.932700 | 0.003369 | 0.005400 | 0 | Project CPU evaluator using the resolved schema. |
| pcl_kdtree | ok | 6000 | 1818.580000 | 0.002848 | 0.004900 | 0 | KdTreeFLANN radius/KNN only. |
| pcl_octree | ok | 6000 | 154.473000 | 0.014698 | 0.029400 | 0 | OctreePointCloudSearch range/radius/KNN. |

## Per-Type Detail

### multidatastructure_cpu

| Query type | Queries | Avg ms | P95 ms |
|---|---:|---:|---:|
| knn | 2000 | 0.005133 | 0.006200 |
| radius | 4000 | 0.002577 | 0.003800 |

### pcl_kdtree

| Query type | Queries | Avg ms | P95 ms |
|---|---:|---:|---:|
| knn | 2000 | 0.002305 | 0.003300 |
| radius | 4000 | 0.003119 | 0.005300 |

### pcl_octree

| Query type | Queries | Avg ms | P95 ms |
|---|---:|---:|---:|
| knn | 2000 | 0.022415 | 0.034100 |
| radius | 4000 | 0.010840 | 0.022605 |

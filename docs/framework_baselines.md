# Framework Baselines

`scripts/compare_frameworks.py` compares one project schema against optional open
point-cloud frameworks without making them mandatory dependencies.

The script accepts either:

- a normal schema JSON, such as `configs/schemas/octree_kdtree.json`;
- a measured selector JSON written by `scripts/tune_schema_for_cloud.py`.

For selector JSON, the harness reads `selected_schema.path` and benchmarks that
concrete schema. This is the current "optimal nested data structure" artifact in
the project: the optimizer exports an ordinary replayable schema JSON.

Example:

```powershell
python scripts\compare_frameworks.py `
  --input C:\data\cloud.ply `
  --schema models\local_schema_selector.json `
  --workload-profile configs\workloads\mixed.json `
  --queries 64 `
  --frameworks open3d pdal pcl `
  --pcl-exe C:\tools\pcl_point_baseline.exe
```

Outputs are written under `results/framework_compare/<cloud>_<timestamp>/`:

- `mds_raw.csv`: the project CPU schema-search row.
- `query_trace.csv`: exact generated queries and returned counts.
- `framework_comparison.json`: machine-readable comparison.
- `framework_comparison.md`: readable summary table.

## Baseline Coverage

Open3D:

- radius and KNN use `KDTreeFlann`;
- AABB range uses `AxisAlignedBoundingBox` crop, which is useful but not the
  same indexed primitive as PCL octree box search.

PDAL:

- AABB range uses `filters.crop`;
- arbitrary radius-center and KNN query replay are not matched by PDAL stages
  in this harness;
- each range query runs a PDAL pipeline, so the default run limits PDAL to the
  first 32 range queries (`--pdal-query-limit 0` disables that limit).

PCL:

- build the optional helper from `tools/pcl_point_baseline.cpp` in an
  environment with PCL installed;
- the helper reports `pcl_kdtree` for radius/KNN and `pcl_octree` for
  range/radius/KNN;
- the helper reads PCD, PLY, XYZ, TXT, and CSV. It does not read LAS/LAZ.

## Interpretation

Do not mix backend selection and backend comparison. For a CPU point-cloud
analytics claim, run the optimizer/final schema on the CPU and compare against
CPU framework baselines. GPU schema search should be reported as a separate
backend target or appendix.

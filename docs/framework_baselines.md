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

Indexicon (`tools/indexicon_point_baseline.cpp`, driven by
`scripts/run_external_baselines.ps1`):

- an independent MIT, header-only C++17 spatial library
  (github.com/psimatis/Indexicon-Spatial-Library, arXiv:2606.04676), used as a
  primitive-quality control: if our octree/kd-tree were weak, a composition
  built on them would win for the wrong reason;
- it is **not vendored**. `external/` is gitignored, so
  `tools/build_indexicon_baseline.ps1` clones it and pins commit
  `c9f9b1da7a55c44fcb30435bcfe101f1f699784f`, then compiles the driver against
  it. Run that script rather than the raw `cl` line, or the pin is lost;
- structures reported: `indexicon_octree`, `indexicon_kdtree`,
  `indexicon_rtree` (a packed R-tree — the R-tree family our own schema grammar
  deliberately does not contain, see `plans/research_direction_v2.md` §F.3);
- the driver reads PLY, XYZ, and `.mdspc`. A `.las`/`.laz` path resolves to the
  sibling `<path>.mdspc` cache that `PointCloud::load` writes on first load, so
  a cloud must be opened once by `MultiDataStructure.exe` before it can be
  replayed here. Positions are converted back to world coordinates with the
  cache's own origin/scale, matching `PointCloud::toWorldPosition`;
- it emits the same JSON payload shape as the PCL helper, so
  `compare_frameworks.py` *could* splice its rows into the frameworks table.
  It is deliberately not wired in: Indexicon has no radius primitive, so its
  radius numbers depend on which emulation was selected, and mixing them into a
  table alongside natively-supported libraries invites a misreading.
  `run_external_baselines.ps1` keeps it in its own output instead, with the
  visited-node and tested-point counts needed to decompose the cost (see
  Interpretation).

### Radius queries on Indexicon

Indexicon exposes no radius primitive. `--radius-mode` selects the emulation and
the choice is recorded in every payload; the two modes are **not** comparable:

- `native_mbr_prune` (default): a traversal written against each structure's own
  public node interface, pruning a subtree when its minimum squared distance to
  the query centre exceeds r². Each structure is driven through the very same
  min-distance helper its own kNN search uses (`OctreeNode::minSqrDist`,
  `kdMindistToRegion`, `mindistPointToBox`), so this is the sphere query their
  code would write, not a handicapped stand-in;
- `aabb_filter`: native box query over the sphere's bounding box, then an exact
  distance filter. A sphere is only π/6 (~52%) of its bounding box, so this
  tests roughly twice the points it needs.

Measured on `indoor_1M` (3000 queries, 3 repeats), `native_mbr_prune` is
1.39–1.59× faster than `aabb_filter` across the three structures. The default is
therefore the charitable one: reporting Indexicon on the naive emulation would
have inflated our own position by roughly 1.5× on every radius query, which is
the majority of a pipeline trace.

`--verify-bruteforce N` brute-forces the first N radius queries and refuses to
report numbers if any structure disagrees. Keep it on: this is query code
written against a third-party tree, and a silent traversal bug would look like a
performance result. `run_external_baselines.ps1` skips it above 30M points,
where building each structure a second time will not fit in memory; correctness
is a property of the traversal rather than of the cloud, so verifying on the
small cells covers the large ones.

`count_mismatches` is counted over every measured sample, so with `--repeats R` a
persistently disagreeing query contributes R. The denominator is `queries` ×
`repeats`, both of which are in the payload.

## Interpretation

Do not mix backend selection and backend comparison. For a CPU point-cloud
analytics claim, run the optimizer/final schema on the CPU and compare against
CPU framework baselines. GPU schema search should be reported as a separate
backend target or appendix.

### A visited node costs ~90× a tested point in our CPU traversal

Regressing MDS latency on `avg_visited_nodes` and `avg_tested_points` across the
schemas measured in one cell separates the two cost terms. Across all 13 matrix
cells the fit is stable at **51–131 ns per visited node** against **1.1–2.8 ns
per point tested**; an independent six-schema depth sweep on `terrain_5M` gives
106 ns/node vs 1.17 ns/point. The per-point figure is what a float distance test
should cost. The per-node figure is far above a bounds check.

**About a tenth of that was instrumentation, and it has been removed.**
`recordNodeVisit` used to build a `std::string` by value and probe an
`unordered_map` on every visited node, inside the timed region. On 2026-08-06
those counters were rewritten to plain arrays indexed by a build-time-resolved
`Node::_structureIndex`. Measured A/B on `terrain_5M` with **identical node and
point counters**, proving the traversal itself is unchanged: **octree 1.21×**
(66.7 visited nodes/query) and **kd-tree 1.03×** (28.7), i.e. roughly
**6–14 ns/node** of the ~106 ns/node total. The emitted `visited_by_structure` /
`tested_points_by_structure` breakdowns are unchanged.

The remaining ~90 ns/node is genuine traversal work: recursion, bounds tests,
and above all cache misses over node structs — a 524k-node kd-tree spans roughly
50 MB, and visiting 66 scattered nodes in it is dominated by memory stalls.

An intermediate version of this document claimed the rewrite changed nothing.
That measurement ran a **stale binary** (MSBuild links to `<repo>\x64\Release`;
the scripts pointed at a legacy `MultiDataStructure\x64\Release` copy frozen at
2026-07-28) which did not contain the change. `scripts/resolve_exe.ps1` now
resolves the correct path and refuses a binary older than the sources.

What this means for baseline comparisons:

1. **Our pre-fix numbers were pessimistic by ~10% per node.** Indexicon was
   never charged this, so the octree ratios in
   `results/framework_compare/indexicon/` understate us by roughly 1.2×; our
   octree likely beats theirs at 25M once re-measured.
2. **Our results favour shallow, wide structures.** With nodes still ~90× more
   expensive than points, the search correctly prefers designs that touch fewer
   nodes. That is implementation-specific: an index with better node locality
   would shift the trade-off.
3. `run_external_baselines.ps1` records `mds_*_visited` / `mds_*_tested`
   alongside the latencies so this decomposition stays reproducible.

Node locality is the remaining optimization opportunity.

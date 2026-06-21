# Current Behavior

This document tracks the point-cloud smoke path from `CODEX_PLAN_MultiDataStructure.md`.

## Executable

The repository builds a Visual Studio C++/CUDA console application from:

```text
MultiDataStructure.sln
MultiDataStructure/MultiDataStructure.vcxproj
```

The executable entry point is `MultiDataStructure/main.cpp`. It is now point-cloud focused and intentionally small. Editable defaults live in `MultiDataStructure/AppConfig.h` under `AppDefaults`, so local runs can be changed by editing/rebuilding instead of changing Visual Studio debug arguments.

Default knobs:

| Constant | Meaning |
|---|---|
| `DEFAULT_MODE` | `"points"`, `"schema-search"`, or `"tests"`. |
| `POINT_INPUT_PATH` | Default `.las`, `.ply`, `.xyz`, or `.csv` input path. |
| `POINT_SCHEMA_PATH` | Default schema JSON path. |
| `POINT_SCHEMA_PATHS` | Optional `;` or `,` separated schema list for multi-schema runs. |
| `POINT_OUTPUT_PATH` | Default JSON output path. |
| `POINT_CSV_OUTPUT_PATH` | Default appendable CSV summary path. |
| `POINT_MODEL_PATH` | Default exported selector JSON for `--schema auto`. |
| `POINT_WORKLOAD_PROFILE_PATH` | Default workload profile for `--schema auto`. |
| `POINT_USE_BINARY_CACHE` | Reads/writes the sibling `.mdspc` point cache when `true`. |
| `POINT_REBUILD_BINARY_CACHE` | Forces source parsing and cache replacement when `true`. |
| `POINT_QUERY_COUNT` | Number of generated queries per query type; `0` disables query profiling. |
| `POINT_QUERY_K` | Neighbor count for generated KNN queries. |
| `POINT_QUERY_SEED` | Seed for generated range/radius/KNN query profiles. |
| `ENABLE_LEAF_MICRO_INDEXES` | Opts CPU point indexes into heavy-leaf micro-indexes. Disabled by default. |
| `LEAF_MICRO_INDEX_THRESHOLD` | Minimum leaf point count before building a leaf-local micro-index. |
| `SCHEMA_SEARCH_SCHEMA_PATHS` | Default finite candidate schema list. |
| `SCHEMA_SEARCH_WORKLOAD_PATHS` | Default workload profile JSON list. |
| `SCHEMA_SEARCH_CSV_PATH` | Default raw schema-search CSV output. |
| `SCHEMA_SEARCH_BEST_CSV_PATH` | Default best-schema CSV output. |
| `SCHEMA_SEARCH_SYNTHETIC_SCALE` | Point-count scale for built-in synthetic schema-search datasets. |
| `RUN_TESTS` | Runs smoke tests by default when `true`. |
| `PAUSE_AT_END` | Keeps the console pause when `true`. |

If `POINT_INPUT_PATH` is empty, pass an input path on the command line:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:\data\sample.las --schema configs\schemas\octree.json --output results\points.json --no-pause
```

Supported command-line overrides:

| Argument | Meaning |
|---|---|
| `--input <path>` | Loads `.las`, ASCII `.ply`, `.xyz`, or `.csv` point data. |
| `--schema <path>` | Selects a schema JSON for point mode. |
| `--schema auto` | Selects a schema with the exported score-ranker model. |
| `--schemas <a;b;c>` | Runs one loaded point cloud across multiple schema JSON files. |
| `--model <path>` | Selector JSON for `--schema auto`. |
| `--workload-profile <path>` | Workload profile JSON for `--schema auto`. |
| `--output <path>` | Writes point benchmark metrics as JSON; multi-schema runs suffix the schema name. |
| `--csv <path>` | Appends one machine-readable summary row per schema run. |
| `--best-csv <path>` | Writes best-schema rows for schema-search mode. |
| `--no-csv` | Disables CSV summary output. |
| `--mode points` | Runs the point load/build path. |
| `--mode schema-search` | Runs candidate schemas over datasets and workload profiles. |
| `--mode tests` | Runs the built-in smoke tests. |
| `--run-tests` | Runs the built-in smoke tests. |
| `--queries <count>` | Runs generated range/count/radius/KNN query profiles; `0` disables them. |
| `--knn-k <count>` | Sets the generated KNN query neighbor count. |
| `--query-seed <seed>` | Sets the generated query profile seed. |
| `--leaf-micro-indexes` | Enables optional CPU leaf-local micro-indexes for leaves above the configured threshold. |
| `--leaf-micro-threshold <count>` | Sets the heavy-leaf threshold for micro-index construction; default is `512`. |
| `--workloads <a;b;c>` | Selects workload profile JSON files for schema-search mode. |
| `--synthetic-scale <count>` | Sets the synthetic dataset size scale for schema-search mode. |
| `--no-synthetic` | Uses only `--input` datasets in schema-search mode. |
| `--evaluator cpu\|cuda` | Selects CPU index benchmarking or a GPU evaluator for schema-search mode; default is CUDA with CPU fallback. |
| `--cuda-device <id>` | Selects the CUDA device for `--evaluator cuda`; default is `0`. |
| `--cuda-builder lbvh\|kdtree\|bih\|octree\|karras_octree\|quadtree\|regular_grid\|hgrid\|mixed` | Selects the CUDA structure; `lbvh`, `kdtree`, `bih`, `octree`, `karras_octree`, `quadtree`, `regular_grid`, `hgrid`, and static `mixed` schemas are implemented. |
| `--cuda-query-batch <count>` | Sets CUDA query batch size; `0` runs each generated workload as one batch. |
| `--cuda-memory-budget-mb <mb>` | Optional CUDA memory budget guard. |
| `--no-cache` | Reads source point data without reading or writing `.mdspc`. |
| `--rebuild-cache` | Re-reads source point data and replaces the `.mdspc` cache. |
| `--no-pause` | Skips the final console pause. |

## Point Workload

Point mode:

- loads the point cloud,
- reads or writes a sibling `.mdspc` binary cache next to the source point cloud,
- stores LAS positions as local `float3` coordinates relative to a double-precision coordinate-frame origin,
- loads a schema JSON,
- builds the CPU point index,
- optionally builds heavy-leaf micro-indexes when `--leaf-micro-indexes` or schema `buildPolicy.enableLeafMicroIndexes` is enabled,
- optionally runs generated AABB range, count-range, radius, and KNN query profiles,
- prints load/build stats,
- writes JSON metrics when `POINT_OUTPUT_PATH` or `--output` is set,
- appends CSV summary rows when `POINT_CSV_OUTPUT_PATH` or `--csv` is set.

The JSON document includes:

- `run_id`, dataset, local bounds, coordinate-frame, cache, schema, and workload metadata,
- schema/point load timings,
- build metrics: build time, node/leaf/depth counts, leaf occupancy, indexed points, memory estimate, and tree-health diagnostics,
- query metrics for mixed/range/count/radius/KNN workloads: total queries, average/median/p95 latency, throughput, visited nodes, tested points, returned points.

The CSV summary uses one row per `(dataset, schema, workload)` run and is meant for later schema comparison or learner training.

When enabled, heavy-leaf micro-indexes are built only for leaves above the threshold. Range/count/radius use a leaf-local uniform grid, while KNN uses a leaf-local KD tree. This is CPU-only and disabled by default so existing point and schema-search behavior stays comparable.

Multi-schema experiment helper:

```powershell
python scripts\run_experiments.py --input C:\data\sample.las --queries 64
```

## Schema Search

Schema-search mode creates the first selector-training table. It defaults to CUDA evaluation on device `0` with the `mixed` CUDA builder, `configs/workloads/volume_small_medium.json`, 64 prepared workload queries, 256 generated candidates, `models/schema_selector.json` as the lightweight ranker, and measured benchmarking of the top 32 ranked candidates. If CUDA is unavailable, schema-search logs a warning and falls back to the CPU evaluator.

The GUI is oriented around the per-cloud publication workflow by default: real-cloud-only tuning, CUDA/Mixed on device `0`, the volume workload, generated-only conditional candidates, and `--auto-conditions` with the staged proxy/shortlist/confirmation budget. Broader schema lists, surrogate ranking, score weights, and CUDA device/builder knobs remain available under secondary sections.

It loads candidate schemas, workload profiles from `configs/workloads/`, and at least three deterministic synthetic point datasets by default:

- `synthetic_flat_terrain`
- `synthetic_facade`
- `synthetic_urban_mixed`

Each run logs one raw CSV row per `(dataset, workload, schema)` and writes a second CSV containing the lowest-score schema for each `(dataset, workload)`.
Rows include deterministic point-cloud features, workload features, raw metrics, score components, and final score. The feature protocol is documented in `docs/experiment_protocol.md`.

Workload JSON files configure the query mix, query count, seed, KNN `k`, and optional generated query scale intervals:

```json
{
  "queryScales": {
    "aabb_range": { "min": 0.01, "max": 0.25 },
    "radius": { "min": 0.01, "max": 0.08 }
  }
}
```

For AABB range queries the scale is a linear fraction of each dataset bounding-box extent. `configs/workloads/volume_small_medium.json` uses only 3D volume queries and samples continuously from 1% to 25% of each dimension.

Workload JSON files can also enable query-set stratification with `"stratifyQueries": true`. Stratified runs sample deterministic difficulty buckets such as small/medium/large ranges, near-empty ranges, dense-region ranges, small/medium/large radii, boundary radii, dense KNN, outside-cloud KNN, and boundary KNN. Raw schema-search rows include `query_strata_summary`, and query traces include `query_stratum`.

Example:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --csv results\schema_search.csv --best-csv results\schema_search_best.csv --no-pause
```

Generated hyperspace search is available in the same mode. The generator samples valid nested schemas from bounded intervals over the currently implemented node families. CPU discovery now defaults to the query-minimal primitive profile (`QuadTree`, `Octree`, `KDTree`, `BVH`) so CPU-equivalent aliases do not waste search budget. CUDA/full sweeps can opt into GPU-specific variants (`KarrasOctree`, `BIH`, `LBVH`, `RegularGrid`, `HGrid`) with `--primitive-profile cuda_query_full` or `--all-primitives`. Generated schemas are written under `results/generated_schemas/`, optionally ranked with a selector model, and benchmarked by top-k:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --rank-model models/schema_selector.json --benchmark-top 32 --workloads configs/workloads/volume_small_medium.json --queries 64 --csv results\alhambra_generated_search.csv --best-csv results\alhambra_generated_best.csv --no-pause
```

Schema levels may include a `condition` object. Conditions are evaluated per node, so one branch can enter a nested block while a sibling skips it and advances to the next schema block:

```json
{
  "type": "Octree",
  "numLevels": 6,
  "leafCapacity": 4096,
  "condition": {
    "minPoints": 8192,
    "minHeightRatio": 0.2
  }
}
```

Supported CPU condition fields are `minPoints`, `maxPoints`, `minDensity`, `maxDensity`, `minHeightRatio`, `maxHeightRatio`, per-axis extent bounds (`minExtentX`, `maxExtentX`, etc.), `minAnisotropy` / `maxAnisotropy`, and `minOccupancyEntropy` / `maxOccupancyEntropy`. CUDA MixedTree supports the cheap bbox-based fields plus bbox anisotropy and rejects occupancy entropy until a GPU entropy estimator exists. `configs/schemas/adaptive_quadtree_octree.json` is a hand-authored example. Generated search can sample conditions with `--generated-conditional`. Schema parsing keeps the original primitive kind (`RegularGrid`, `HGrid`, `KarrasOctree`, `LBVH`, `BIH`, etc.) separately from its current CPU fallback enum so CPU discovery and CUDA replay do not erase schema intent.

CPU point schemas can also opt into per-node adaptive leaf capacity with an `adaptiveLeafCapacity` object on a level. The base `leafCapacity` is multiplied by deterministic density, height-ratio, anisotropy, and explicit `queryMixFactor` terms, then clamped by `minCapacity` / `maxCapacity`. Positive `densityWeight` shrinks dense nodes, positive `heightRatioWeight` grows flatter-than-root nodes and shrinks taller ones, and positive `anisotropyWeight` shrinks elongated nodes. Generated CPU search can sample these rules with `--generated-adaptive-leaf-capacity` and `--generated-adaptive-leaf-probability`. Schema-search CUDA evaluators reject fixed adaptive leaf-capacity schemas; generated CUDA runs disable adaptive leaf-capacity sampling until GPU support exists.

Per-cloud condition tuning is available with `--auto-conditions`. This path estimates a deterministic shallow feature sketch from the target cloud, builds condition-threshold domains from occupancy, density, height-ratio, and extent quantiles, then writes generated schemas with concrete numeric thresholds. It evaluates many candidates with a small proxy workload and downsampled cloud, shortlists a few on the full cloud, and confirms the best candidates with the requested query count. The raw/best CSV formats keep all existing columns and append condition-summary, runtime nesting, baseline-normalized, tree-health, query-strata, and score-provenance columns. The tree-health columns include leaf occupancy quantiles, average depth, fanout, empty-child ratio, single-child count, tight-bounds volume ratio, and micro-indexed leaf counts. The provenance columns include `lambda_latency`, `lambda_build`, `lambda_memory`, `lambda_imbalance`, `score_mode`, `score_stage`, `score_is_final_latency`, `effective_queries`, `score_uses_visit_proxy`, and `visit_proxy_alpha`. Workload JSON files can define `scoreWeights`; explicit CLI score flags take precedence.

Example local threshold tuning:

```powershell
python scripts\tune_schema_for_cloud.py --input C:/Datasets/points/Alhambra_100M.las --workload-profile configs\workloads\volume_small_medium.json --queries 64 --auto-conditions --output models\alhambra_local_selector.json
```

Deep nested search is available with `--deep-nested-search`. It is CPU-first for discovery, injects the pure CPU-distinct single-DS controls (`QuadTree`, `Octree`, `KDTree`, `BVH`), runs a nested-opportunity diagnostic, generates template-guided nested candidates, stages proxy/full/confirmation/robustness measurements, and uses CUDA/Mixed only for a report-only confirmation when CUDA is available. CUDA/full confirmation can still include GPU-native controls. A candidate is reported as runtime-nested only when at least two structure types become active and non-primary structures own at least 5% of nodes or leaf points.

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --deep-nested-search --workloads configs\workloads\volume_small_medium.json --csv results\alhambra_deep_nested.csv --best-csv results\alhambra_deep_nested_best.csv --no-pause
```

Useful generator controls:

```text
--generated-min-blocks <n>
--generated-max-blocks <n>
--generated-max-depth <n>
--generated-min-leaf <n>
--generated-max-leaf <n>
--generated-conditional
--generated-condition-probability <value>
--generated-seed <seed>
--generated-schema-dir <path>
--condition-output-dir <path>
--condition-selector-output <path>
```

The score is:

```text
avg_query_latency_ms + 0.0 * build_time_ms + 0.0 * memory_mb + 0.0 * imbalance_penalty
```

where `imbalance_penalty = max_leaf_occupancy / max(1, avg_leaf_occupancy)`. Build time, memory, and imbalance are logged but ignored by default because schemas can be built offline and reused. Raw metrics and score components are also stored so later milestones can recompute labels.

With `--evaluator cuda`, schema search uploads each point cloud to the GPU and measures range/count-range/radius/KNN query batches through the selected CUDA builder. Workload queries are prepared once per dataset/workload, CUDA query buffers are reused across candidate measurements, and repeated CUDA builds on the same cloud/device reuse uploaded point buffers where the builder supports it. `lbvh` uses a Morton-sorted point order, `kdtree` and `bih` use binary linear nodes, `octree` and `quadtree` use midpoint child buckets, `karras_octree` uses Morton sorting plus prefix child ranges, `regular_grid` and `hgrid` use grid cell bins, and `mixed` follows the schema's per-depth schedule. Mixed CUDA schemas can name `QuadTree`, `Octree`, `KarrasOctree`, `KDTree`, `BIH`, `BVH`, `LBVH`, `RegularGrid`, and `HGrid` as recursive split levels. In mixed schemas, `RegularGrid` and `HGrid` are per-node grid split flavors rather than the standalone global sorted-cell evaluators. MixedTree honors bbox-anisotropy conditions approximately from node bounds and rejects occupancy-entropy predicates instead of silently ignoring them. CUDA KNN rows are labeled with `knn_backend=bruteforce_gpu_scan`: they report benchmark statistics without neighbor ids and use a parallel GPU point-buffer scan, not structure-accelerated traversal. The default CUDA workload still avoids KNN to keep measured search light.

To compare GA and non-GA outputs, run `python scripts\audit_schema_scores.py non_ga.csv ga.csv`. The audit reports best-by-score versus best-by-latency and flags mismatched score mode, stage, query count, evaluator, CUDA builder, score weights, and proxy alpha.

The Python helpers support both point benchmark sweeps and schema search:

```powershell
python scripts\run_experiments.py --schema-search --queries 64
python scripts\summarize_results.py --input results\schema_search.csv --output results\schema_search_best.csv
python scripts\train_schema_selector.py --input results\schema_search.csv
python scripts\export_model.py --model models\schema_selector.joblib --metadata models\schema_selector_meta.json --output models\schema_selector.json
python scripts\export_model.py --model models\schema_selector.joblib --metadata models\schema_selector_meta.json --skip-linear-json --onnx-output models\schema_selector.onnx --onnx-json-output models\schema_selector_onnx.json
```

The training script learns a score-ranking model over point-cloud, workload, and schema-composition features. Static schema JSON files are the initial measured candidate pool; the model is structured so later milestones can rank generated combinations too.

## Auto Selection

Point mode supports learned global selection with:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:\data\sample.xyz --schema auto --model models\schema_selector.json --workload-profile configs\workloads\mixed.json --output results\auto_run.json --no-pause
```

Runtime flow:

- load the point cloud,
- extract point-cloud features,
- load the workload profile and extract workload features,
- parse each candidate schema listed in the exported model,
- predict a score for each candidate combination using `linear_score_ranker` JSON or optional `onnx_score_ranker` ONNX Runtime inference,
- choose the lowest predicted score,
- run the normal point benchmark with the selected schema.

The JSON result includes a `schema_selection` block with the model path, workload profile, selected schema, predicted score, and full candidate ranking.

ONNX Runtime is optional at build time. Set either `OnnxRuntimeDir` to a SDK root containing `include\` and `lib\`, or set `OnnxRuntimeIncludeDir` and `OnnxRuntimeLibraryDir` explicitly. When those MSBuild properties are set, the project defines `MDSPC_ENABLE_ONNX`, includes `onnxruntime_cxx_api.h`, and links `onnxruntime.lib`. The selector wrapper JSON can request `"execution_provider": "cpu"` or `"cuda"`; CUDA requires an ONNX Runtime GPU build and matching CUDA/cuDNN runtime DLLs.

The helper scripts use the official ONNX Runtime GPU Windows NuGet package locally under `.deps`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_onnx_runtime.ps1
powershell -ExecutionPolicy Bypass -File scripts\build_with_onnx.ps1
```

For per-cloud overfitting, use the local tuner:

```powershell
python scripts\tune_schema_for_cloud.py --input C:/Datasets/points/Alhambra_100M.las --workload-profile configs\workloads\volume_small_medium.json --queries 64 --auto-conditions --output models\alhambra_local_selector.json
```

The local tuner benchmarks candidates on the target point cloud and writes a `measured_best_schema` selector. With `--auto-conditions`, it also saves the winning numeric schema JSON so later `--schema auto` runs reuse the tuned thresholds instead of re-estimating them. With that artifact, `--schema auto` reuses the measured winner instead of predicting from a global model.

## Point Queries

`PointSpatialIndex` currently supports exact CPU queries:

- `rangeQuery(AABB)`,
- `countRange(AABB)`,
- `radiusQuery(center, radius)`,
- `knnQuery(center, k)`.

The built-in tests compare these against brute force on a deterministic sparse/dense synthetic cloud. KNN results are sorted by squared distance with point-index tie-breaking.

## Legacy Triangle Code

The triangle/ray-tracing and 3D-rendering prototype has been removed from the repository and the Visual Studio project: `TriangleBenchmark.*`, `ExternalBvh.*`, `TriangleMesh.*`, `Model3D.*`, `Material.*`, `Camera*.*`, `Image.*`, `SceneContent.*`, `GeometricUtilities.*`, `ApplicationState.h`, `ChronoUtilities.h`, and the vendored `tinybvh/` and `progressbar/` libraries are gone.

The legacy `MultiDataStructure` meta-structure class and its `KdTree`/`Octree`/`QuadTree`/`Bvh` node types are intentionally retained: the point-cloud schema layer (`core/Config`) reuses the `MultiDataStructure::DataStructureLevel` enum and `LevelConfig` struct, and `tests/test_tree_cleanup.cpp` and `tests/test_level_schedule.cpp` still exercise the meta-structure. `Ray.*` and `AABB.*` are also retained because the point workload depends on `AABB` (which includes `Ray`). Fully decoupling the point schema layer from the legacy meta-structure is future work.

## Current Limitations

- Optional per-query CSV traces are available with `--query-trace` for point benchmarks and schema-search measurements. CPU point-index traces include `query_stratum` plus traversal breakdown columns for visited nodes by depth, visited nodes by structure, tested points by structure, and fully contained nodes by structure; CUDA trace rows currently leave those breakdown fields empty.
- Schema-search can write a Markdown schema explanation report with `--explain-report <path>`. The report groups measured rows by dataset/workload, shows the schema chain, active-structure node/leaf-point fractions, query-family behavior against the best baseline, and repair-style diagnoses.
- CUDA MixedTree uses the original thread-per-query traversal for small range/count/radius queries and switches to a block-per-query cooperative leaf-scan kernel for larger estimated query volumes. CUDA KNN backend labels are explicit: `cpu_tree_knn`, `gpu_bruteforce_knn`, and KDTree/BIH `gpu_tree_knn` for exact `k <= 16`.
- Evolutionary schema search can use diagnostic-guided repair mutations with `--repair-mutations`. Repairs are created only after a candidate has measured build/query counters, then re-enter the normal measurement loop as ordinary generated schemas. Current repair rules target high leaf occupancy, high tested-points per visit, high visited-node counts, high full-containment ratios, and likely single-child chains.
- Schema-search labels currently use synthetic datasets and placeholder score weights.
- The built-in PLY reader supports ASCII PLY only.
- The built-in LAS reader supports uncompressed LAS records, not LAZ.
- `PointCloud::bounds()` and generated query coordinates are local-space. Use `PointCloud::toWorldPosition()` or the JSON `dataset.coordinate_frame` metadata to reconstruct LAS/world coordinates.
- The `.mdspc` cache stores local positions plus coordinate-frame metadata and is invalidated using the source file size and last-write timestamp.
- JSON logging is intentionally lightweight and hand-written.
- The retained legacy `MultiDataStructure` meta-structure still exposes triangle/ray APIs from the original prototype (e.g. `SpatialDSNode::intersects(Ray)`); these are unused by the point path and only kept alive by the two meta-structure unit tests.

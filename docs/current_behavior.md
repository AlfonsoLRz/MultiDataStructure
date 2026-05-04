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
| `--workloads <a;b;c>` | Selects workload profile JSON files for schema-search mode. |
| `--synthetic-scale <count>` | Sets the synthetic dataset size scale for schema-search mode. |
| `--no-synthetic` | Uses only `--input` datasets in schema-search mode. |
| `--evaluator cpu\|cuda` | Selects CPU index benchmarking or a GPU evaluator for schema-search mode. |
| `--cuda-device <id>` | Selects the CUDA device for `--evaluator cuda`. |
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
- loads a schema JSON,
- builds the CPU point index,
- optionally runs generated AABB range, count-range, radius, and KNN query profiles,
- prints load/build stats,
- writes JSON metrics when `POINT_OUTPUT_PATH` or `--output` is set,
- appends CSV summary rows when `POINT_CSV_OUTPUT_PATH` or `--csv` is set.

The JSON document includes:

- `run_id`, dataset, cache, schema, and workload metadata,
- schema/point load timings,
- build metrics: build time, node/leaf/depth counts, leaf occupancy, indexed points, memory estimate,
- query metrics for mixed/range/count/radius/KNN workloads: total queries, average/median/p95 latency, throughput, visited nodes, tested points, returned points.

The CSV summary uses one row per `(dataset, schema, workload)` run and is meant for later schema comparison or learner training.

Multi-schema experiment helper:

```powershell
python scripts\run_experiments.py --input C:\data\sample.las --queries 128
```

## Schema Search

Schema-search mode creates the first selector-training table. It loads candidate schemas, workload profiles from `configs/workloads/`, and at least three deterministic synthetic point datasets by default:

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

Example:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --queries 128 --csv results\schema_search.csv --best-csv results\schema_search_best.csv --no-pause
```

Generated hyperspace search is available in the same mode. The generator samples valid nested schemas from bounded intervals over the currently implemented node families (`QuadTree`, `Octree`, `KDTree`, `BVH`) and can emit GPU-specific variants (`KarrasOctree`, `BIH`, `LBVH`, `RegularGrid`, `HGrid`) for mixed CUDA evaluation. It writes replayable schema JSON files under `results/generated_schemas/`, optionally ranks all candidates with a selector model, and benchmarks only the top-k:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 1000 --rank-model models/schema_selector_onnx.json --benchmark-top 16 --workloads configs/workloads/mixed.json --queries 64 --csv results\alhambra_generated_search.csv --best-csv results\alhambra_generated_best.csv --no-pause
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

Supported condition fields are `minPoints`, `maxPoints`, `minDensity`, `maxDensity`, `minHeightRatio`, `maxHeightRatio`, and per-axis extent bounds (`minExtentX`, `maxExtentX`, etc.). `configs/schemas/adaptive_quadtree_octree.json` is a hand-authored example. Generated search can sample conditions with `--generated-conditional`.

Useful generator controls:

```text
--generated-max-blocks <n>
--generated-max-depth <n>
--generated-min-leaf <n>
--generated-max-leaf <n>
--generated-conditional
--generated-condition-probability <value>
--generated-seed <seed>
--generated-schema-dir <path>
```

The score is:

```text
avg_query_latency_ms + 0.0 * build_time_ms + 0.0 * memory_mb + 0.0 * imbalance_penalty
```

where `imbalance_penalty = max_leaf_occupancy / max(1, avg_leaf_occupancy)`. Build time, memory, and imbalance are logged but ignored by default because schemas can be built offline and reused. Raw metrics and score components are also stored so later milestones can recompute labels.

With `--evaluator cuda`, schema search uploads each point cloud to the GPU and measures range/count-range/radius query batches through the selected CUDA builder. `lbvh` uses a Morton-sorted point order, `kdtree` and `bih` use binary linear nodes, `octree` and `quadtree` use midpoint child buckets, `karras_octree` uses Morton sorting plus prefix child ranges, `regular_grid` and `hgrid` use grid cell bins, and `mixed` follows the schema's per-depth schedule. Mixed CUDA schemas can name `QuadTree`, `Octree`, `KarrasOctree`, `KDTree`, `BIH`, `BVH`, `LBVH`, `RegularGrid`, and `HGrid` as recursive split levels. In mixed schemas, `RegularGrid` and `HGrid` are per-node grid split flavors rather than the standalone global sorted-cell evaluators. KNN remains on the CPU path for now.

The Python helpers support both point benchmark sweeps and schema search:

```powershell
python scripts\run_experiments.py --schema-search --queries 128
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
python scripts\tune_schema_for_cloud.py --input C:/Datasets/points/Alhambra_100M.las --workload-profile configs\workloads\mixed.json --queries 128 --output models\alhambra_local_selector.json
```

The local tuner benchmarks candidates on the target point cloud and writes a `measured_best_schema` selector. With that artifact, `--schema auto` reuses the measured winner instead of predicting from a global model.

## Point Queries

`PointSpatialIndex` currently supports exact CPU queries:

- `rangeQuery(AABB)`,
- `countRange(AABB)`,
- `radiusQuery(center, radius)`,
- `knnQuery(center, k)`.

The built-in tests compare these against brute force on a deterministic sparse/dense synthetic cloud. KNN results are sorted by squared distance with point-index tie-breaking.

## Legacy Triangle Code

The executable no longer exposes the triangle/ray benchmark, and `TriangleBenchmark.*` is no longer included in the Visual Studio project. Older triangle and rendering files may still exist as reference code while the project moves toward point-cloud indexing, but the maintained run path is point-first.

## Current Limitations

- Generated query profiles are aggregate-only; per-query traces are still a later extension.
- Schema-search labels currently use synthetic datasets and placeholder score weights.
- The built-in PLY reader supports ASCII PLY only.
- The built-in LAS reader supports uncompressed LAS records, not LAZ.
- The `.mdspc` cache is invalidated using the source file size and last-write timestamp.
- JSON logging is intentionally lightweight and hand-written.
- Some generic core files still expose legacy triangle/ray APIs from the original prototype.

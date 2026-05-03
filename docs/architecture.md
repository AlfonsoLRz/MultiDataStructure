# Architecture

This repository is moving from a triangle/ray prototype toward a point-cloud spatial-index framework inspired by nested LiDAR indexing.

## Current Layout

```text
MultiDataStructure/
  main.cpp
  AppConfig.*
  AABB.*
  MultiDataStructure.*
  QuadTree.*
  Octree.*
  KdTree.*
  Bvh.*
  core/
    BuildPolicy.*
    Config.*
  experiments/
    FeatureExtraction.*
    Metrics.*
    SchemaSearch.*
    SchemaSelector.*
  workloads/
    points/
      PointBenchmark.*
      PointPrimitive.*
      PointCloud.*
      PointSpatialIndex.*
      SyntheticPointClouds.*
configs/
  schemas/
    octree.json
    quadtree.json
    kdtree.json
    bvh.json
    quadtree_octree.json
    octree_kdtree.json
    urban_hybrid.json
  workloads/
    range_heavy.json
    knn_heavy.json
    mixed.json
tests/
  test_level_schedule.cpp
  test_metrics.cpp
  test_tree_cleanup.cpp
  test_pointcloud_io.cpp
  test_point_queries.cpp
  test_config_parsing.cpp
  test_schema_search.cpp
```

Legacy triangle/rendering files still exist in the repository, but `main.cpp` and the Visual Studio project no longer include the triangle benchmark workload.

## Application Entry Point

`MultiDataStructure/main.cpp` is intentionally thin and point-first. Editable defaults live in `MultiDataStructure/AppConfig.h` under `AppDefaults`:

- `DEFAULT_MODE`,
- `POINT_INPUT_PATH`,
- `POINT_SCHEMA_PATH`,
- `POINT_OUTPUT_PATH`,
- schema-search defaults for schemas, workloads, CSV outputs, and synthetic dataset scale,
- `POINT_USE_BINARY_CACHE`,
- `POINT_REBUILD_BINARY_CACHE`,
- `RUN_TESTS`,
- `PAUSE_AT_END`.

Command-line arguments still override those defaults.

Example:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:\data\sample.las --schema configs\schemas\octree.json --output results\points.json --no-pause
```

## Generic Core

The current generic spatial-index pieces are still physically in the project root to avoid a disruptive file move:

- `AABB`
- `MultiDataStructure`
- `QuadTreeNode`
- `OctreeNode`
- `KdTreeNode`
- `BvhNode`
- `NodeFactory`
- `BuildPolicy`
- `SchemaConfig`

`core/Config.*` parses JSON schema files into `SchemaConfig`. The parser currently supports `QuadTree`, `Octree`, `KDTree`, and `BVH` blocks. `Grid2D` and `Grid3D` remain future node types.

Some root-level core APIs still carry triangle/ray concepts from the original prototype. New work should prefer the point workload boundary unless a legacy API is being deliberately retired or generalized.

## Point Workload

`workloads/points/` contains the maintained workload:

- `PointPrimitive`: position plus optional intensity, classification, and id.
- `PointCloud`: vector-backed point set with private stats, private `.xyz`/`.csv`/ASCII `.ply`/uncompressed `.las` loaders, and a sibling `.mdspc` binary cache.
- `PointBenchmark`: CLI entry point that loads a point cloud, loads a schema JSON, builds a CPU point index, runs an optional generated query profile, prints statistics, and optionally writes JSON metrics.
- `PointSpatialIndex`: CPU point build/query path using schema configs and single-child point assignment. It supports exact AABB range, count-range, radius, and KNN queries with visited-node, tested-point, returned-point, and elapsed-time counters.
- `SyntheticPointClouds`: deterministic generators for terrain, facades, buildings, urban mixes, and sparse/dense mixtures.

`experiments/Metrics.*` contains the experiment-facing metric structs and summarizers for build records, query latency distributions, throughput, leaf occupancy, and memory estimates. `scripts/run_experiments.py` wraps the executable for multi-schema sweeps and writes into `results/`.

`experiments/SchemaSearch.*` runs the first supervised-label sweep. It evaluates a finite candidate schema set over deterministic synthetic datasets and workload profiles, logs one raw row per `(dataset, workload, schema)`, computes the placeholder scalar score from the plan, and writes best-schema rows per `(dataset, workload)`.

`experiments/FeatureExtraction.*` computes deterministic point-cloud and workload features for selector training. It uses a fixed-seed sample for large point clouds, covariance eigenvalues, 8x8x8 occupancy summaries, height statistics, shape heuristics, and workload query-mix features.

`scripts/tune_schema_for_cloud.py` is the local optimizer path: it runs schema-search on one target cloud with `--no-synthetic`, chooses the lowest measured score for the requested workload, and exports a `measured_best_schema` JSON. `scripts/train_schema_selector.py` is the optional global learner. It augments schema-search rows with schema-composition features, splits by dataset, trains score-ranking models and a direct classifier when possible, evaluates oracle/fixed/heuristic/learned selectors, and saves a report plus model artifact. `scripts/export_model.py` exports an explicit linear score ranker to JSON for the dependency-free C++ runtime and can optionally export an ONNX model plus `onnx_score_ranker` wrapper JSON.

`experiments/SchemaSelector.*` is the C++ selection path for `--schema auto`. It supports local measured-best artifacts, exported linear score rankers, and optional ONNX Runtime score rankers. For measured-best artifacts it reuses the tuned winner. For learned rankers it recomputes point/workload/schema features, predicts a score for each candidate combination, and hands the selected schema to the normal point benchmark. ONNX support is compiled only when the Visual Studio project receives `OnnxRuntimeDir` or explicit ONNX include/library directories.

The point path is intentionally small. It validates schema configs, point subdivision policies, and exact query correctness before later optimization work changes node layout or traversal strategy. Point insertion is single-child: points are assigned by position and are not duplicated across overlapping children.

## Schema Configs

Schema configs live in `configs/schemas/`. Each file contains:

- `name`,
- `levels`,
- per-level `type`, `numLevels`, `leafCapacity`, and `minPointsToSplit`,
- top-level `buildPolicy`.

The current point build tests load several schemas from disk and verify that different configs can build the same synthetic cloud while preserving point count.

Workload profiles live in `configs/workloads/` and specify generated query mixes for range-heavy, KNN-heavy, and mixed workloads.

## Tests

The built-in smoke tests are run through:

```powershell
.\x64\Release\MultiDataStructure.exe --run-tests
```

They currently verify:

- level schedule boundary behavior,
- empty-node removal,
- single-child collapse,
- point-cloud XYZ/CSV/PLY/LAS loading,
- point-cloud binary cache creation and reuse,
- point-cloud bounds and stats,
- deterministic synthetic point-cloud generation counts,
- schema JSON parsing,
- policy-controlled point subdivision,
- no point duplication during configured point builds,
- AABB range, count-range, radius, and KNN query correctness against brute force,
- query instrumentation on a synthetic sparse/dense cloud,
- build/query metric aggregation for experiment logs,
- workload profile parsing, schema-search score computation, and best-schema selection,
- deterministic point-cloud/workload feature extraction.

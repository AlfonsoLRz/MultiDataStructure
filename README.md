# MultiDataStructure

A workload-aware optimizer that synthesizes the best spatial index — nested or single — for a point cloud and query workload.

## What this is

This is a research framework for **workload-aware meta-optimization of spatial indexes** over large point clouds. Given a point cloud and a query workload (range, count-range, radius, kNN), a **genetic-algorithm optimizer** (the default search) explores a space of index *schemas* — each level a primitive (regular grid, hierarchical grid, quadtree, octree, kd-tree, BVH/BIH, LBVH), optionally nested across levels and conditioned per node — and synthesizes the one that minimizes query latency, build time, memory, and imbalance. The space spans **single primitives and nested combinations alike** (a schema may be a single level or several), so the optimizer returns a *tuned single primitive* when nesting does not help and a *nested schema* when it does. Candidate schemas are evaluated on a CPU reference path and, where supported, on a CUDA `MixedTree` GPU evaluator.

It is a computer-graphics / spatial-data-structures contribution — in the lineage of **meta-optimization / instance-optimized data structures** (learned index structures, automated/evolutionary structure synthesis, workload-aware autotuning) and SAH-guided / GPU BVH construction — **not** a relational database system. The structures are GPU-resident scientific data structures evaluated directly on query workloads, not through SQL; the "optimization" is over the index design space (physical structure), not query plans. See [docs/evaluation.md](docs/evaluation.md) for the measured comparison of the optimizer's synthesized index (nested or single) against single-primitive and hand-designed baselines.

## Build Requirements

- Windows + Visual Studio 2022, using the checked-in `.sln` / `.vcxproj`.
- CUDA Toolkit for GPU point evaluators.
- vcpkg packages used by the Visual Studio project, including Boost.JSON, GLFW, GLEW, and ImGui.
- Python is optional for helper scripts under `scripts/`.

## Run Tests

```powershell
.\x64\Release\MultiDataStructure.exe --run-tests --no-pause
```

## Point Benchmark

```powershell
.\x64\Release\MultiDataStructure.exe --input C:\data\cloud.las --schema configs\schemas\octree.json --queries 64 --output results\point_benchmark.json --csv results\point_benchmark.csv --query-trace results\point_queries.csv --no-pause
```

Supported point formats are `.las` (uncompressed), `.ply` (ASCII), `.xyz`, and `.csv`. LAS coordinates are stored internally in a local float frame with double-precision origin metadata.

## Schema Search

The **genetic-algorithm optimizer runs by default**. It searches single-and-nested schemas and can export the winning schema per (dataset, workload) for re-measurement:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:\data\cloud.las --no-synthetic --workloads configs\workloads\volume_small_medium.json --queries 256 --evaluator cuda --cuda-device 0 --cuda-builder mixed --optimizer-output-dir results\evolved_schemas --csv results\schema_search.csv --best-csv results\schema_search_best.csv --pareto-csv results\schema_search_pareto.csv --no-pause
```

Alternative search strategies (each opts out of the default GA): `--auto-conditions` (fast surrogate-filtered conditional tuning), `--flat-search` (one-pass scan of the candidate set), `--generated-only` (flat scan of generated candidates without the configured schema list), `--deep-nested-search` (CPU-first staged nested search). The CUDA resolver falls back to CPU with a warning when CUDA is unavailable.

## Schema Primitives

Current schema names include `QuadTree`, `Octree`, `KarrasOctree`, `KDTree`, `BIH`, `BVH`, `LBVH`, `RegularGrid`, and `HGrid`. `QuadTree` supports `axisPolicy` values `xy`, `xz`, `yz`, `ignore_shortest`, `ignore_x`, `ignore_y`, and `ignore_z`; point-cloud schemas default to `xy`. `KDTree`/`BIH` support `median_longest_axis`, `center_longest_axis`, and `round_robin`; CUDA structural comparisons are only apples-to-apples for policies the selected builder can reproduce.

## Known Limitations

- CUDA KNN is labeled `bruteforce_gpu_scan`: it scans the GPU point buffer and should not be reported as tree-accelerated KNN traversal.
- CUDA MixedTree supports bbox-anisotropy conditions but has no GPU implementation of occupancy-entropy conditions or adaptive leaf capacity. In schema search these CPU-native features no longer drop the candidate: the evaluator falls back to the CPU index for that schema (its row reads `backend=cpu`, `gpu_support_status=cpu_fallback`). Unsupported CUDA KDTree/BIH split policies similarly fall back with `gpu_support_status=unsupported_policy`; direct CUDA APIs still reject unsupported schemas.
- The built-in LAS reader does not read LAZ.
- The triangle/ray-tracing prototype (3D rendering, scene, mesh, and the vendored `tinybvh`/`progressbar` libraries) has been removed. The legacy `MultiDataStructure` meta-structure (and its `KdTree`/`Octree`/`QuadTree`/`Bvh` node types) is retained only because the point-cloud schema layer reuses its `DataStructureLevel`/`LevelConfig` types and two unit tests still cover it; the maintained path is point-cloud benchmarking and schema search.

More detailed behavior lives in `docs/current_behavior.md`; command examples live in `docs/command_reference.md`.

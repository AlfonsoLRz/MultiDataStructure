# MultiDataStructure

Experimental nested spatial data structures for point-cloud indexing and schema search.

## What this is

This is a research framework for **workload-aware automatic synthesis and tuning of nested spatial indexes** over large point clouds. Given a point cloud and a query workload (range, count-range, radius, kNN), it searches a space of nested, heterogeneous index *schemas* — each level a different primitive (regular grid, hierarchical grid, quadtree, octree, kd-tree, BVH/BIH, LBVH) with optional per-node conditions — and selects one that minimizes query latency, build time, memory, and imbalance. Candidate schemas are evaluated on a CPU reference path and, where supported, on a CUDA `MixedTree` GPU evaluator.

It is a computer-graphics / spatial-data-structures contribution — in the lineage of SAH-guided BVH construction, GPU LBVH/HLBVH/PLOC builders, and instance-optimized / learned index structures — **not** a relational database system. The structures are GPU-resident scientific data structures evaluated directly on query workloads, not through SQL; the "optimization" is over the index design space (physical structure), not query plans. See [docs/evaluation.md](docs/evaluation.md) for the measured comparison of auto-tuned nested indexes against single-primitive baselines.

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

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:\data\cloud.las --no-synthetic --generated-only --generate-schemas 256 --workloads configs\workloads\volume_small_medium.json --queries 64 --evaluator cuda --cuda-device 0 --cuda-builder mixed --csv results\schema_search.csv --best-csv results\schema_search_best.csv --pareto-csv results\schema_search_pareto.csv --no-pause
```

The CUDA schema-search resolver falls back to CPU with a warning when CUDA is unavailable. CPU discovery defaults to query-minimal primitives; CUDA confirmation can use the fuller primitive set.

## Schema Primitives

Current schema names include `QuadTree`, `Octree`, `KarrasOctree`, `KDTree`, `BIH`, `BVH`, `LBVH`, `RegularGrid`, and `HGrid`. `QuadTree` supports `axisPolicy` values `xy`, `xz`, `yz`, `ignore_shortest`, `ignore_x`, `ignore_y`, and `ignore_z`; point-cloud schemas default to `xy`. `KDTree`/`BIH` support `median_longest_axis`, `center_longest_axis`, and `round_robin`; CUDA structural comparisons are only apples-to-apples for policies the selected builder can reproduce.

## Known Limitations

- CUDA KNN is labeled `bruteforce_gpu_scan`: it scans the GPU point buffer and should not be reported as tree-accelerated KNN traversal.
- CUDA MixedTree supports bbox-anisotropy conditions but has no GPU implementation of occupancy-entropy conditions or adaptive leaf capacity. In schema search these CPU-native features no longer drop the candidate: the evaluator falls back to the CPU index for that schema (its row reads `backend=cpu`, `gpu_support_status=cpu_fallback`). Unsupported CUDA KDTree/BIH split policies similarly fall back with `gpu_support_status=unsupported_policy`; direct CUDA APIs still reject unsupported schemas.
- The built-in LAS reader does not read LAZ.
- The triangle/ray-tracing prototype (3D rendering, scene, mesh, and the vendored `tinybvh`/`progressbar` libraries) has been removed. The legacy `MultiDataStructure` meta-structure (and its `KdTree`/`Octree`/`QuadTree`/`Bvh` node types) is retained only because the point-cloud schema layer reuses its `DataStructureLevel`/`LevelConfig` types and two unit tests still cover it; the maintained path is point-cloud benchmarking and schema search.

More detailed behavior lives in `docs/current_behavior.md`; command examples live in `docs/command_reference.md`.

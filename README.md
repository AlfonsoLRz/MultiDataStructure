# MultiDataStructure

Experimental nested spatial data structures for point-cloud indexing and schema search.

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

Current schema names include `QuadTree`, `Octree`, `KarrasOctree`, `KDTree`, `BIH`, `BVH`, `LBVH`, `RegularGrid`, and `HGrid`. `QuadTree` supports `axisPolicy` values `xy`, `xz`, `yz`, `ignore_shortest`, `ignore_x`, `ignore_y`, and `ignore_z`; point-cloud schemas default to `xy`.

## Known Limitations

- CUDA KNN is labeled `bruteforce_gpu_scan`: it scans the GPU point buffer and should not be reported as tree-accelerated KNN traversal.
- CUDA MixedTree supports bbox-anisotropy conditions, but rejects occupancy-entropy conditions until a GPU entropy estimator is implemented.
- The built-in LAS reader does not read LAZ.
- Legacy triangle/ray code still exists, but the maintained path is point-cloud benchmarking and schema search.

More detailed behavior lives in `docs/current_behavior.md`; command examples live in `docs/command_reference.md`.

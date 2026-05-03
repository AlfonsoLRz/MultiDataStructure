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
| `DEFAULT_MODE` | `"points"` or `"tests"`. |
| `POINT_INPUT_PATH` | Default `.las`, `.ply`, `.xyz`, or `.csv` input path. |
| `POINT_SCHEMA_PATH` | Default schema JSON path. |
| `POINT_OUTPUT_PATH` | Default JSON output path. |
| `POINT_USE_BINARY_CACHE` | Reads/writes the sibling `.mdspc` point cache when `true`. |
| `POINT_REBUILD_BINARY_CACHE` | Forces source parsing and cache replacement when `true`. |
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
| `--output <path>` | Writes point benchmark metrics as JSON. |
| `--mode points` | Runs the point load/build path. |
| `--mode tests` | Runs the built-in smoke tests. |
| `--run-tests` | Runs the built-in smoke tests. |
| `--no-cache` | Reads source point data without reading or writing `.mdspc`. |
| `--rebuild-cache` | Re-reads source point data and replaces the `.mdspc` cache. |
| `--no-pause` | Skips the final console pause. |

## Point Workload

Point mode:

- loads the point cloud,
- reads or writes a sibling `.mdspc` binary cache next to the source point cloud,
- loads a schema JSON,
- builds the CPU point index,
- prints load/build stats,
- optionally writes JSON metrics.

The JSON document includes:

- input path,
- cache path and whether it was used,
- schema path and name,
- load/build timings,
- point count,
- node/leaf/depth counts,
- bounds and approximate density.

## Legacy Triangle Code

The executable no longer exposes the triangle/ray benchmark, and `TriangleBenchmark.*` is no longer included in the Visual Studio project. Older triangle and rendering files may still exist as reference code while the project moves toward point-cloud indexing, but the maintained run path is point-first.

## Current Limitations

- Point-cloud support is currently a load/build smoke benchmark; range, radius, and KNN query workloads are not implemented yet.
- The built-in PLY reader supports ASCII PLY only.
- The built-in LAS reader supports uncompressed LAS records, not LAZ.
- The `.mdspc` cache is invalidated using the source file size and last-write timestamp.
- JSON logging is intentionally lightweight and hand-written.
- Some generic core files still expose legacy triangle/ray APIs from the original prototype.

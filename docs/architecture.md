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
    urban_hybrid.json
tests/
  test_level_schedule.cpp
  test_tree_cleanup.cpp
  test_pointcloud_io.cpp
  test_config_parsing.cpp
```

Legacy triangle/rendering files still exist in the repository, but `main.cpp` and the Visual Studio project no longer include the triangle benchmark workload.

## Application Entry Point

`MultiDataStructure/main.cpp` is intentionally thin and point-first. Editable defaults live in `MultiDataStructure/AppConfig.h` under `AppDefaults`:

- `DEFAULT_MODE`,
- `POINT_INPUT_PATH`,
- `POINT_SCHEMA_PATH`,
- `POINT_OUTPUT_PATH`,
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
- `PointBenchmark`: CLI entry point that loads a point cloud, loads a schema JSON, builds a CPU point index, prints build statistics, and optionally writes JSON metrics.
- `PointSpatialIndex`: CPU point build path using schema configs and single-child point assignment.
- `SyntheticPointClouds`: deterministic generators for terrain, facades, buildings, urban mixes, and sparse/dense mixtures.

The point build path is intentionally small. It validates schema configs and point subdivision policies before range/KNN query workloads are added. Point insertion is single-child: points are assigned by position and are not duplicated across overlapping children.

## Schema Configs

Schema configs live in `configs/schemas/`. Each file contains:

- `name`,
- `levels`,
- per-level `type`, `numLevels`, `leafCapacity`, and `minPointsToSplit`,
- top-level `buildPolicy`.

The current point build tests load several schemas from disk and verify that different configs can build the same synthetic cloud while preserving point count.

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
- no point duplication during configured point builds.

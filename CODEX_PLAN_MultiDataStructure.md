# CODEX PLAN: MultiDataStructure as a Learnable Nested Spatial Index Framework

**Repository:** `AlfonsoLRz/MultiDataStructure`  
**Primary target:** point-cloud / LiDAR-style workloads  
**Secondary target:** preserve triangle/ray/path-tracing experiments for later reuse  
**Main research direction:** learn or auto-tune the choice of nested spatial data structures for heterogeneous spatial data

---

## 0. How Codex Should Use This Document

This document is intended to be placed in the repository root as:

```text
CODEX_PLAN.md
```

Codex should treat it as a long-running development plan. It should proceed milestone by milestone, keeping each milestone small enough to become a coherent pull request.

Codex should not attempt to implement the entire plan in one pass. Instead, it should:

1. Read the whole document once.
2. Start at **Milestone 0**.
3. Implement the requested changes for that milestone only.
4. Build and test.
5. Produce a concise summary of changes, tests run, and remaining issues.
6. Move to the next milestone only when the previous one is stable.

If a milestone reveals that the current repository differs from the assumptions in this plan, Codex should adapt the implementation while preserving the intent.

---

## 1. Project Vision

The current repository is a prototype for combining multiple spatial data structures such as quadtrees, octrees, kd-trees, and BVHs into a nested global structure.

The desired research direction is to transform this prototype into a reusable experimental framework where:

- the spatial index is configurable,
- point clouds are first-class data,
- different query workloads can be benchmarked,
- candidate nested schemas can be searched automatically,
- a learner can select a good schema from point-cloud features and workload requirements.

The first publishable research target should be:

> **Learning to choose nested spatial data structures for heterogeneous point-cloud workloads.**

This is more natural and lower risk than starting directly with path tracing, because point clouds are the natural application of nested spatial indexing and because objective functions such as range-query time, KNN time, memory, storage, node count, and leaf occupancy are easier to measure.

---

## 2. Current Repository Snapshot and Assumptions

As of this plan, the repository appears to contain:

```text
MultiDataStructure.sln
README.md
LICENSE
.gitattributes
.gitignore

MultiDataStructure/
  AABB.cpp
  AABB.h
  ApplicationState.h
  Bvh.cu
  Bvh.h
  Camera.cpp
  Camera.h
  CameraProjection.cpp
  CameraProjection.h
  ChronoUtilities.h
  CudaHelper.cpp
  CudaHelper.h
  ExternalBvh.cpp
  ExternalBvh.h
  GPUStructs.cpp
  GPUStructs.h
  GeometricUtilities.cpp
  GeometricUtilities.h
  Image.cpp
  Image.h
  KdTree.cpp
  KdTree.h
  Material.cpp
  Material.h
  Model3D.cpp
  Model3D.h
  MultiDataStructure.cpp
  MultiDataStructure.h
  MultiDataStructure.vcxproj
  MultiDataStructure.vcxproj.filters
  MultiDataStructure.vcxproj.user
  Octree.cpp
  Octree.h
  QuadTree.cpp
  QuadTree.h
  RandomUtilities.h
  Ray.cpp
  Ray.h
  SceneContent.cpp
  SceneContent.h
  TriangleMesh.cpp
  TriangleMesh.h
  bvh_kernels.cu
  bvh_kernels.cuh
  cuda_globals.cu
  cuda_globals.cuh
  main.cpp
  morton_encoding_kernels.cu
  morton_encoding_kernels.cuh
  stdafx.cpp
  stdafx.h
  timeit.hpp

Python/
  interactive_bboxes.py
```

The existing code appears to be mostly C++/CUDA, with a Visual Studio solution and a small Python folder. The current executable path is likely triangle/ray oriented. The future point-cloud learning framework should be built without destroying this existing path.

Important assumption:

- The current project probably uses a `MultiDataStructure` abstraction and node classes such as `QuadTreeNode`, `OctreeNode`, `KdTreeNode`, and `BvhNode`.
- The current code may have a level schedule such as `LevelConfig { type, numLevels }`.
- The current splitting behavior may contain hard-coded thresholds or triangle-oriented assumptions.

Codex must verify these assumptions before editing.

---

## 3. Non-Negotiable Engineering Rules

1. **Preserve the existing triangle/ray benchmark path.**
2. **Do not perform a giant rewrite.**
3. **Every milestone should build.**
4. **Every behavior-changing milestone should add tests or benchmark checks.**
5. **Prefer explicit configuration over hard-coded constants.**
6. **Prefer JSON/CSV experiment logs over console-only output.**
7. **Start with CPU point-cloud logic.** GPU point-cloud acceleration is a later step.
8. **Start with simple point formats** such as XYZ, CSV, and optionally PLY. LAS/LAZ can come later.
9. **Do not start with reinforcement learning, genetic algorithms, or differentiable training.**
10. **The first learner should be an offline Python global-schema selector.**
11. **Do not attempt node-wise adaptive learning until the global selector works.**
12. **Use small synthetic datasets first, then scale.**
13. **Keep the code readable enough for research iteration.**
14. **When uncertain, add instrumentation rather than guessing.**

---

## 4. Research Strategy

### 4.1 First research problem

Given:

- a point cloud,
- a query workload profile,
- a candidate set of nested spatial data structure schemas,

choose the schema that minimizes a cost function combining:

- query time,
- build time,
- memory/storage footprint,
- imbalance or worst-leaf occupancy.

### 4.2 Why not gradients?

The tree-building process is discrete. Structure types, split decisions, depth choices, and thresholds are not naturally differentiable. The first version should be framed as:

- black-box auto-tuning,
- supervised structured prediction,
- algorithm selection,
- or cost-model learning.

### 4.3 Why not genetic algorithms first?

Genetic algorithms are possible, but they are harder to debug, more expensive to run, and less scientifically clean as a first baseline. Before using GA, we should implement:

1. fixed schemas,
2. exhaustive or random schema search,
3. a simple heuristic selector,
4. a supervised global-schema selector.

Only then should GA or RL be considered.

### 4.4 First learner

The first learner should be a global selector:

```text
features(point cloud, workload) -> best schema
```

or:

```text
features(point cloud, workload, schema) -> predicted score
```

The second formulation is often better because it allows adding new schemas without changing the output dimension, as long as schema features are included.

---

## 5. Target Architecture

The repository should gradually move toward this structure:

```text
MultiDataStructure/
├── CODEX_PLAN.md
├── README.md
├── MultiDataStructure.sln
├── configs/
│   ├── schemas/
│   │   ├── quadtree.json
│   │   ├── octree.json
│   │   ├── kdtree.json
│   │   ├── bvh.json
│   │   ├── quadtree_octree.json
│   │   ├── grid2d_quadtree_grid3d_octree.json
│   │   └── urban_hybrid.json
│   └── workloads/
│       ├── range_heavy.json
│       ├── knn_heavy.json
│       └── mixed.json
├── docs/
│   ├── current_behavior.md
│   ├── architecture.md
│   ├── pointcloud_plan.md
│   ├── learner_plan.md
│   └── experiment_protocol.md
├── results/
│   ├── .gitkeep
│   └── README.md
├── scripts/
│   ├── run_experiments.py
│   ├── train_schema_selector.py
│   ├── export_model.py
│   ├── summarize_results.py
│   └── plot_results.py
├── tests/
│   ├── test_level_schedule.cpp
│   ├── test_tree_cleanup.cpp
│   ├── test_pointcloud_io.cpp
│   ├── test_point_queries.cpp
│   ├── test_config_parsing.cpp
│   └── test_feature_extraction.cpp
└── MultiDataStructure/
    ├── core/
    │   ├── AABB.cpp
    │   ├── AABB.h
    │   ├── BuildPolicy.cpp
    │   ├── BuildPolicy.h
    │   ├── Config.cpp
    │   ├── Config.h
    │   ├── MultiDataStructure.cpp
    │   ├── MultiDataStructure.h
    │   ├── SpatialDSNode.cpp
    │   ├── SpatialDSNode.h
    │   └── nodes/
    │       ├── Bvh.cpp / Bvh.cu
    │       ├── Bvh.h
    │       ├── KdTree.cpp
    │       ├── KdTree.h
    │       ├── Octree.cpp
    │       ├── Octree.h
    │       ├── QuadTree.cpp
    │       ├── QuadTree.h
    │       ├── Grid2D.cpp
    │       ├── Grid2D.h
    │       ├── Grid3D.cpp
    │       └── Grid3D.h
    ├── workloads/
    │   ├── triangles/
    │   │   ├── Camera.cpp
    │   │   ├── Camera.h
    │   │   ├── Ray.cpp
    │   │   ├── Ray.h
    │   │   ├── SceneContent.cpp
    │   │   ├── SceneContent.h
    │   │   ├── TriangleMesh.cpp
    │   │   ├── TriangleMesh.h
    │   │   ├── TriangleBenchmark.cpp
    │   │   └── TriangleBenchmark.h
    │   └── points/
    │       ├── PointPrimitive.cpp
    │       ├── PointPrimitive.h
    │       ├── PointCloud.cpp
    │       ├── PointCloud.h
    │       ├── PointCloudIO.cpp
    │       ├── PointCloudIO.h
    │       ├── PointQueries.cpp
    │       ├── PointQueries.h
    │       ├── SyntheticPointClouds.cpp
    │       └── SyntheticPointClouds.h
    ├── experiments/
    │   ├── ExperimentRunner.cpp
    │   ├── ExperimentRunner.h
    │   ├── Metrics.cpp
    │   ├── Metrics.h
    │   ├── FeatureExtraction.cpp
    │   ├── FeatureExtraction.h
    │   ├── SchemaSearch.cpp
    │   └── SchemaSearch.h
    └── app/
        └── main.cpp
```

The exact final layout can differ, but the separation should be clear:

- `core`: generic spatial-index logic,
- `workloads/triangles`: current ray/triangle code,
- `workloads/points`: point-cloud functionality,
- `experiments`: logging, metrics, schema search, feature extraction,
- `scripts`: Python automation and learning.

---

## 6. Milestone 0 — Baseline Freeze

### Goal

Capture the current behavior so refactors do not silently break the existing triangle/ray path.

### Codex prompt

```text
Read CODEX_PLAN.md and implement Milestone 0 only.

Freeze the current behavior of the repository before any major refactor. Add documentation, deterministic benchmark logging, and minimal tests around the existing level schedule and cleanup behavior. Do not restructure the project yet.
```

### Tasks

1. Create `docs/current_behavior.md`.
2. Document:
   - how the current executable is run,
   - what input scene it expects,
   - which node types are registered,
   - what benchmark configurations are currently hard-coded,
   - what metrics are printed,
   - current limitations.
3. Add or expose deterministic benchmark settings:
   - fixed random seed where applicable,
   - fixed number of rays/queries if applicable,
   - fixed benchmark scene path or CLI-provided scene path.
4. Add JSON result logging for the existing benchmark.
5. Create a tiny benchmark-output schema, for example:

```json
{
  "mode": "triangles",
  "scene": "...",
  "schema_name": "octree_bvh_octree_bvh",
  "timestamp": "...",
  "metrics": {
    "build_time_ms": 0.0,
    "query_time_ms": 0.0,
    "num_nodes": 0,
    "num_leaves": 0,
    "num_primitives": 0
  }
}
```

6. Add tests for level schedule semantics.
7. Add tests for empty-node removal if the function exists.
8. Add tests for single-child collapse if the function exists.
9. If no test framework exists, add a minimal one:
   - Catch2,
   - GoogleTest,
   - doctest,
   - or a lightweight internal test executable.

### Files likely touched

```text
docs/current_behavior.md
MultiDataStructure/main.cpp
MultiDataStructure/MultiDataStructure.cpp
MultiDataStructure/MultiDataStructure.h
tests/test_level_schedule.cpp
tests/test_tree_cleanup.cpp
```

### Acceptance criteria

- The current project builds.
- The current triangle benchmark still runs.
- Benchmark results can be written to JSON.
- At least one test verifies the level schedule behavior.
- `docs/current_behavior.md` is accurate enough for a new developer.

### Definition of done

A reviewer can run the current executable, obtain a JSON output, and compare future refactors against it.

---

## 7. Milestone 1 — Core / Workload Separation

### Goal

Separate generic spatial-index code from triangle/ray-specific code while preserving behavior.

### Codex prompt

```text
Implement Milestone 1 only.

Separate the generic spatial-index core from the triangle/ray benchmark code. Preserve the existing triangle benchmark and avoid algorithmic changes. Add or update documentation explaining the new architecture.
```

### Tasks

1. Identify generic code:
   - `AABB`,
   - node classes,
   - `MultiDataStructure`,
   - generic utilities.
2. Identify triangle/ray code:
   - `TriangleMesh`,
   - `SceneContent`,
   - `Ray`,
   - `Camera`,
   - rendering/benchmark setup.
3. Move or namespace generic code into a core module.
4. Move or namespace triangle code into a triangle workload module.
5. Introduce thin interfaces only where necessary.
6. Avoid deep architectural redesign at this stage.
7. Update Visual Studio project filters and build files.
8. Add `docs/architecture.md`.

### Suggested interfaces

Keep these simple. Avoid over-engineering.

```cpp
struct PrimitiveBoundsProvider {
    virtual ~PrimitiveBoundsProvider() = default;
    virtual AABB boundsOf(size_t primitiveIndex) const = 0;
    virtual size_t primitiveCount() const = 0;
};
```

```cpp
struct BenchmarkResult {
    std::string mode;
    std::string schemaName;
    double buildTimeMs = 0.0;
    double queryTimeMs = 0.0;
    size_t numNodes = 0;
    size_t numLeaves = 0;
    size_t numPrimitives = 0;
};
```

### Files likely touched

```text
MultiDataStructure/core/*
MultiDataStructure/workloads/triangles/*
MultiDataStructure/app/main.cpp
docs/architecture.md
MultiDataStructure.vcxproj
MultiDataStructure.vcxproj.filters
```

### Acceptance criteria

- Existing triangle benchmark still runs.
- Core code does not directly depend on camera setup.
- Core code does not directly depend on scene-loading logic.
- `docs/architecture.md` explains:
  - core,
  - workloads,
  - experiments,
  - application entry point.

### Definition of done

A new point-cloud workload can be added without modifying triangle-specific code.

---

## 8. Milestone 2 — Point-Cloud Data Model and I/O

### Goal

Introduce point clouds as a first-class workload.

### Codex prompt

```text
Implement Milestone 2 only.

Add a native point-cloud workload with PointPrimitive, PointCloud, simple CSV/XYZ loading, and synthetic point-cloud generation. Do not implement the learner yet. Preserve the triangle benchmark.
```

### Tasks

1. Add `PointPrimitive`.
2. Add `PointCloud`.
3. Add simple point-cloud I/O:
   - `.xyz`: x y z per line,
   - `.csv`: x,y,z and optional attributes,
   - optional `.ply` if easy.
4. Add synthetic generators:
   - flat terrain,
   - sloped terrain,
   - vertical facade,
   - block/building shell,
   - mixed urban scene,
   - sparse/dense mixture.
5. Add point-cloud statistics:
   - number of points,
   - bounding box,
   - approximate density,
   - coordinate ranges.
6. Add tests for loading and synthetic generation.

### PointPrimitive sketch

```cpp
struct PointPrimitive {
    Vec3 position;
    float intensity = 0.0f;
    uint32_t classification = 0;
    uint64_t id = 0;

    AABB bounds(float epsilon = 0.0f) const;
};
```

If there is no existing `Vec3`, reuse the current vector type or add a minimal one in core.

### PointCloud sketch

```cpp
class PointCloud {
public:
    std::vector<PointPrimitive> points;

    size_t size() const;
    bool empty() const;
    AABB bounds() const;
};
```

### Synthetic generators

```cpp
PointCloud generateFlatTerrain(size_t n, float width, float depth, float noiseZ);
PointCloud generateFacade(size_t n, float width, float height, float thickness);
PointCloud generateUrbanMixed(size_t terrainPoints, size_t buildingPoints, int numBuildings);
PointCloud generateSparseDenseMixture(size_t sparsePoints, size_t densePoints);
```

### Important rule

For point clouds, each point should initially be assigned to exactly one child during subdivision.

Do not duplicate points across overlapping children unless a later explicit radius/splat mode is added.

### Files likely touched

```text
MultiDataStructure/workloads/points/PointPrimitive.h
MultiDataStructure/workloads/points/PointPrimitive.cpp
MultiDataStructure/workloads/points/PointCloud.h
MultiDataStructure/workloads/points/PointCloud.cpp
MultiDataStructure/workloads/points/PointCloudIO.h
MultiDataStructure/workloads/points/PointCloudIO.cpp
MultiDataStructure/workloads/points/SyntheticPointClouds.h
MultiDataStructure/workloads/points/SyntheticPointClouds.cpp
tests/test_pointcloud_io.cpp
```

### Acceptance criteria

- Can load a small XYZ point cloud.
- Can generate at least three synthetic clouds.
- Can compute the point-cloud bounding box.
- Existing triangle benchmark still works.

### Definition of done

The repo has a working point-cloud data model independent from triangle meshes.

---

## 9. Milestone 3 — Generic Point-Cloud Build Path

### Goal

Build the existing nested data structures over point clouds.

### Codex prompt

```text
Implement Milestone 3 only.

Add a point-cloud build path that uses the existing nested data structure machinery. The point path should use single assignment into children and should not rely on triangle/ray code.
```

### Tasks

1. Add a point-cloud adapter to the generic core.
2. Make `MultiDataStructure` able to build from points.
3. Ensure point assignment is single-child for spatial subdivision.
4. Ensure every node can report:
   - number of contained points,
   - number of children,
   - bounding box,
   - level/depth,
   - node type.
5. Add simple configs for:
   - pure quadtree,
   - pure octree,
   - pure kd-tree if suitable,
   - quadtree -> octree.
6. Add tests:
   - build small point cloud with octree,
   - build small point cloud with quadtree,
   - ensure all points remain accounted for,
   - ensure no unintended duplication.

### Important design decision

Current triangle structures may use primitive AABB overlap to insert a triangle into multiple children. That is not appropriate for point primitives.

For points:

```text
child = locateChild(point.position)
insert point into exactly that child
```

### Files likely touched

```text
MultiDataStructure/core/MultiDataStructure.h
MultiDataStructure/core/MultiDataStructure.cpp
MultiDataStructure/core/nodes/*.h
MultiDataStructure/core/nodes/*.cpp
MultiDataStructure/workloads/points/*
configs/schemas/*.json
tests/test_point_build.cpp
```

### Acceptance criteria

- Can build a pure octree over a synthetic point cloud.
- Can build a pure quadtree over a synthetic point cloud.
- Can build at least one nested schema over a synthetic point cloud.
- Point counts are preserved.
- No hidden triangle dependencies exist in the point build path.

### Definition of done

Point clouds can be indexed by the nested structure framework.

---

## 10. Milestone 4 — Configurable Build Policies

### Goal

Make split/stop behavior explicit and configurable.

### Codex prompt

```text
Implement Milestone 4 only.

Introduce BuildPolicy and richer per-level configuration. Remove hard-coded split thresholds where possible. Add JSON schema parsing for spatial-index configurations.
```

### Tasks

1. Add `BuildPolicy`.
2. Add `SchemaConfig`.
3. Add JSON config parsing.
4. Replace hard-coded values such as split-at-2 with policy values.
5. Support per-level or per-block parameters:
   - structure type,
   - number of levels,
   - leaf capacity,
   - min points to split,
   - optional axis policy,
   - optional cleanup behavior.
6. Add config files for standard schemas.

### BuildPolicy sketch

```cpp
struct BuildPolicy {
    size_t maxDepth = 16;
    size_t leafCapacity = 1024;
    size_t minPointsToSplit = 64;
    bool collapseSingleChild = true;
    bool removeEmptyNodes = true;
    bool allowOverlapDuplication = false;
};
```

### LevelConfig sketch

```cpp
struct LevelConfig {
    std::string type;
    size_t numLevels = 1;
    size_t leafCapacity = 1024;
    size_t minPointsToSplit = 64;
};
```

### Example config: pure octree

```json
{
  "name": "octree_default",
  "levels": [
    {
      "type": "Octree",
      "numLevels": 8,
      "leafCapacity": 1024,
      "minPointsToSplit": 64
    }
  ],
  "buildPolicy": {
    "maxDepth": 8,
    "collapseSingleChild": true,
    "removeEmptyNodes": true,
    "allowOverlapDuplication": false
  }
}
```

### Example config: urban hybrid

```json
{
  "name": "urban_hybrid_grid2d_quadtree_grid3d_octree",
  "levels": [
    { "type": "Grid2D", "numLevels": 1, "leafCapacity": 0, "minPointsToSplit": 1 },
    { "type": "QuadTree", "numLevels": 4, "leafCapacity": 512, "minPointsToSplit": 128 },
    { "type": "Grid3D", "numLevels": 1, "leafCapacity": 0, "minPointsToSplit": 1 },
    { "type": "Octree", "numLevels": 3, "leafCapacity": 1024, "minPointsToSplit": 128 }
  ],
  "buildPolicy": {
    "maxDepth": 9,
    "collapseSingleChild": true,
    "removeEmptyNodes": true,
    "allowOverlapDuplication": false
  }
}
```

### Files likely touched

```text
MultiDataStructure/core/BuildPolicy.h
MultiDataStructure/core/BuildPolicy.cpp
MultiDataStructure/core/Config.h
MultiDataStructure/core/Config.cpp
configs/schemas/*.json
tests/test_config_parsing.cpp
```

### Acceptance criteria

- At least three schema configs can be parsed.
- Build policy controls point-cloud subdivision.
- Triangle path is not broken.
- Tests cover config parsing and level schedule semantics.

### Definition of done

The same point cloud can be built with different schemas loaded from disk.

---

## 11. Milestone 5 — Point-Cloud Query Workloads

### Goal

Implement the actual queries that will be optimized.

### Codex prompt

```text
Implement Milestone 5 only.

Add point-cloud query workloads: AABB range query, spherical radius query, KNN query, and count query. Instrument visited nodes, touched points, returned points, and query time.
```

### Tasks

1. Implement AABB range query.
2. Implement radius/sphere query.
3. Implement KNN query.
4. Implement count query.
5. Add query profile generation:
   - random boxes,
   - random spheres,
   - random KNN centers.
6. Add correctness tests by comparing indexed queries to brute force.
7. Add instrumentation:
   - visited nodes,
   - tested points,
   - returned points,
   - elapsed time.

### Query result sketch

```cpp
struct PointQueryStats {
    size_t visitedNodes = 0;
    size_t testedPoints = 0;
    size_t returnedPoints = 0;
    double elapsedMs = 0.0;
};
```

```cpp
struct PointQueryResult {
    std::vector<size_t> pointIndices;
    PointQueryStats stats;
};
```

### KNN correctness rule

For small test clouds, KNN results should be compared against brute force sorted by squared distance. For equal distances, use deterministic tie-breaking by point index.

### Files likely touched

```text
MultiDataStructure/workloads/points/PointQueries.h
MultiDataStructure/workloads/points/PointQueries.cpp
MultiDataStructure/workloads/points/PointBenchmark.h
MultiDataStructure/workloads/points/PointBenchmark.cpp
tests/test_point_queries.cpp
```

### Acceptance criteria

- Range query matches brute force on small clouds.
- Radius query matches brute force on small clouds.
- KNN query matches brute force on small clouds.
- Query stats are collected.
- Query workload can be run on at least one synthetic cloud.

### Implementation status

Implemented in this workspace:

- `PointSpatialIndex` exact AABB range, count-range, radius, and KNN queries.
- Generated query profiles in `PointBenchmark` for random boxes, random spheres, and random KNN centers.
- Query aggregates for elapsed time, visited nodes, tested points, and returned points.
- Brute-force correctness coverage in `tests/test_point_queries.cpp`.

### Definition of done

The framework can measure meaningful point-cloud query performance.

---

## 12. Milestone 6 — Benchmark Metrics and Experiment Logging

### Goal

Create reproducible experiment outputs.

### Codex prompt

```text
Implement Milestone 6 only.

Add a benchmark and experiment logging layer for point-cloud experiments. Results should be saved to JSON and CSV with enough information for later training.
```

### Tasks

1. Add `Metrics` structs.
2. Add experiment runner for point-cloud benchmarks.
3. Log build metrics:
   - build time,
   - total nodes,
   - leaf nodes,
   - max depth,
   - average leaf occupancy,
   - max leaf occupancy,
   - memory estimate.
4. Log query metrics:
   - total queries,
   - average latency,
   - median latency if easy,
   - p95 latency if easy,
   - average visited nodes,
   - average tested points,
   - query throughput.
5. Save JSON per run.
6. Save CSV summary across runs.
7. Add CLI options or script options to run point benchmarks.

### Experiment record sketch

```json
{
  "run_id": "...",
  "mode": "points",
  "dataset": {
    "name": "synthetic_urban_001",
    "num_points": 100000,
    "source": "synthetic"
  },
  "schema": {
    "name": "urban_hybrid",
    "config_path": "configs/schemas/urban_hybrid.json"
  },
  "workload": {
    "name": "mixed",
    "num_queries": 1000
  },
  "build_metrics": {
    "build_time_ms": 0.0,
    "num_nodes": 0,
    "num_leaves": 0,
    "max_depth": 0,
    "avg_leaf_occupancy": 0.0,
    "max_leaf_occupancy": 0,
    "memory_estimate_bytes": 0
  },
  "query_metrics": {
    "avg_latency_ms": 0.0,
    "p95_latency_ms": 0.0,
    "throughput_queries_per_sec": 0.0,
    "avg_visited_nodes": 0.0,
    "avg_tested_points": 0.0,
    "avg_returned_points": 0.0
  }
}
```

### Files likely touched

```text
MultiDataStructure/experiments/Metrics.h
MultiDataStructure/experiments/Metrics.cpp
MultiDataStructure/experiments/ExperimentRunner.h
MultiDataStructure/experiments/ExperimentRunner.cpp
scripts/run_experiments.py
results/README.md
```

### Acceptance criteria

- Running a point benchmark produces JSON.
- Running multiple schemas produces a CSV summary.
- Results include enough data for schema comparison.

### Implementation status

Implemented in this workspace:

- `Experiments::BuildMetrics` and `Experiments::QueryMetrics`.
- Build metrics for time, node/leaf/depth counts, average/max leaf occupancy, indexed points, and memory estimate.
- Query metrics for total queries, average/median/p95 latency, throughput, visited nodes, tested points, and returned points.
- Per-run JSON records from point benchmarks.
- Appendable CSV summary rows for single-schema and multi-schema runs.
- `--schemas`, `--csv`, and `--no-csv` CLI options.
- `scripts/run_experiments.py` and `results/README.md`.

### Definition of done

Experiment data is no longer ad hoc; it is machine-readable and trainable.

---

## 13. Milestone 7 — Schema Grammar and Search Harness

### Goal

Generate supervised labels for the first learner by benchmarking candidate schemas.

### Codex prompt

```text
Implement Milestone 7 only.

Add a schema-search harness that evaluates a candidate set of schemas over point-cloud datasets and workload profiles. Compute a scalar score and identify the best schema for each dataset/workload pair.
```

### Tasks

1. Define a finite candidate schema set.
2. Define workload profiles.
3. Define scoring function.
4. Implement sweep over:
   - datasets,
   - schemas,
   - workload profiles.
5. Save one row per `(dataset, workload, schema)`.
6. Save best schema per `(dataset, workload)`.

### Candidate schemas

Start small:

1. `quadtree_default`
2. `octree_default`
3. `kdtree_default`
4. `quadtree_octree`
5. `octree_kdtree`
6. `grid2d_quadtree`
7. `grid2d_quadtree_grid3d_octree`
8. `urban_hybrid`

Do not let the candidate set explode initially.

### Workload profiles

#### `range_heavy.json`

```json
{
  "name": "range_heavy",
  "queries": {
    "aabb_range": 0.8,
    "radius": 0.1,
    "knn": 0.1
  },
  "numQueries": 1000
}
```

#### `knn_heavy.json`

```json
{
  "name": "knn_heavy",
  "queries": {
    "aabb_range": 0.1,
    "radius": 0.2,
    "knn": 0.7
  },
  "knnK": 16,
  "numQueries": 1000
}
```

#### `mixed.json`

```json
{
  "name": "mixed",
  "queries": {
    "aabb_range": 0.4,
    "radius": 0.3,
    "knn": 0.3
  },
  "knnK": 16,
  "numQueries": 1000
}
```

### Score function

Start with:

```text
score = avg_query_latency_ms
      + lambda_build * build_time_ms
      + lambda_memory * memory_estimate_mb
      + lambda_imbalance * imbalance_penalty
```

where:

```text
imbalance_penalty = max_leaf_occupancy / max(1, avg_leaf_occupancy)
```

Initial weights:

```text
lambda_build = 0.001
lambda_memory = 0.01
lambda_imbalance = 0.01
```

These are placeholders. Log all raw metrics so the score can be recomputed later.

### Files likely touched

```text
MultiDataStructure/experiments/SchemaSearch.h
MultiDataStructure/experiments/SchemaSearch.cpp
configs/workloads/*.json
scripts/run_experiments.py
scripts/summarize_results.py
```

### Acceptance criteria

- Can run a sweep over at least three synthetic datasets and four schemas.
- CSV contains raw metrics and computed score.
- Best schema per dataset/workload can be extracted.

### Implementation status

Implemented in this workspace:

- `Experiments::SchemaSearch` runner with deterministic synthetic datasets, workload-profile parsing, candidate-schema sweeps, score computation, raw CSV rows, and best-schema CSV rows.
- Supported finite candidate schema set for `quadtree_default`, `octree_default`, `kdtree_default`, `quadtree_octree`, `octree_kdtree`, and `urban_hybrid_quadtree_octree_kdtree`.
- Workload profiles in `configs/workloads/` for `range_heavy`, `knn_heavy`, and `mixed`.
- `--mode schema-search`, `--workloads`, `--best-csv`, `--synthetic-scale`, and `--no-synthetic` CLI options.
- `scripts/run_experiments.py --schema-search` and `scripts/summarize_results.py`.
- Smoke coverage for workload parsing, score calculation, and best-schema selection.

### Definition of done

There is a dataset for training a schema selector.

---

## 14. Milestone 8 — Feature Extraction

### Goal

Extract useful point-cloud and workload features for learning.

### Codex prompt

```text
Implement Milestone 8 only.

Add deterministic feature extraction for point clouds and workload profiles. Export features into the schema-search result table so the learner can use them.
```

### Tasks

1. Add global point-cloud features.
2. Add occupancy-grid features.
3. Add covariance/anistropy features.
4. Add workload features.
5. Export features to CSV/JSON.
6. Add tests for deterministic output.

### Minimum point-cloud features

| Feature | Description |
|---|---|
| `num_points` | total number of points |
| `bbox_x`, `bbox_y`, `bbox_z` | bounding-box extents |
| `aspect_xy` | `bbox_x / bbox_y` |
| `aspect_xz` | `bbox_x / bbox_z` |
| `aspect_yz` | `bbox_y / bbox_z` |
| `density_bbox` | points divided by bbox volume or area fallback |
| `height_mean` | mean z |
| `height_std` | stddev z |
| `height_range` | max z - min z |
| `cov_eig_0` | largest covariance eigenvalue |
| `cov_eig_1` | middle covariance eigenvalue |
| `cov_eig_2` | smallest covariance eigenvalue |
| `linearity` | `(eig0 - eig1) / eig0` |
| `planarity` | `(eig1 - eig2) / eig0` |
| `scattering` | `eig2 / eig0` |
| `occupancy_ratio_8` | occupied cells in 8x8x8 grid divided by 512 |
| `occupancy_entropy_8` | entropy of occupancy counts |
| `density_cv_8` | coefficient of variation of occupied-cell density |
| `verticality_score` | heuristic indicator for vertical structures |
| `flatness_score` | heuristic indicator for ALS-like terrain |

### Workload features

| Feature | Description |
|---|---|
| `w_range` | fraction of range queries |
| `w_radius` | fraction of radius queries |
| `w_knn` | fraction of KNN queries |
| `knn_k` | K for KNN |
| `query_scale_mean` | average query box/radius scale |
| `query_scale_std` | stddev query scale |
| `build_weight` | score weight for build time |
| `memory_weight` | score weight for memory |

### Feature extraction notes

- Use a deterministic sample for large clouds.
- Seed the sampler.
- Log the sample size used.
- Avoid expensive all-pairs computations.

### Files likely touched

```text
MultiDataStructure/experiments/FeatureExtraction.h
MultiDataStructure/experiments/FeatureExtraction.cpp
tests/test_feature_extraction.cpp
scripts/summarize_results.py
```

### Acceptance criteria

- Every schema-search row includes feature columns.
- Feature extraction is deterministic.
- Features are documented in `docs/learner_plan.md` or `docs/experiment_protocol.md`.

### Implementation status

Implemented in this workspace:

- `Experiments::PointCloudFeatures` and `Experiments::WorkloadFeatures`.
- Deterministic point-cloud sampling, bbox/aspect/density/height features, covariance eigenvalues, shape ratios, 8x8x8 occupancy summaries, and flatness/verticality heuristics.
- Workload query-mix, KNN K, deterministic query-scale, build-weight, and memory-weight features.
- Feature columns exported in raw schema-search CSV rows and retained in best-schema CSV rows.
- `scripts/summarize_results.py` preserves feature columns when extracting best-schema rows.
- Determinism and known-plane feature coverage in `tests/test_feature_extraction.cpp`.
- Feature documentation in `docs/experiment_protocol.md`.

### Definition of done

The system can generate `(features, schema, metrics, score)` data.

---

## 15. Milestone 9 — Python Training Pipeline for Global Schema Selector

### Goal

Train the first learner offline in Python.

### Codex prompt

```text
Implement Milestone 9 only.

Add a Python training pipeline that reads schema-search results, trains a global schema selector, and evaluates it against fixed baselines and an oracle. Do not integrate the model into C++ yet.
```

### Tasks

1. Add `scripts/train_schema_selector.py`.
2. Load CSV from schema search.
3. Split by dataset, not by row, to avoid leakage.
4. Train baseline models:
   - majority/fixed schema baseline,
   - simple heuristic baseline,
   - logistic regression if possible,
   - random forest,
   - gradient boosted trees if dependencies allow.
5. Support two learning modes:
   - classification: predict best schema,
   - regression/ranking: predict score per schema.
6. Compute metrics:
   - best-schema accuracy,
   - top-2 accuracy,
   - regret vs oracle,
   - speedup vs fixed baselines,
   - memory/build-time effects.
7. Save model and report.

### Recommended training formulation

Start with score prediction:

```text
input = cloud_features + workload_features + schema_features
output = measured_score
```

At inference:

```text
for each candidate schema:
    predict score
choose schema with lowest predicted score
```

This generalizes better than direct classification.

### Schema features

Add simple schema features:

| Feature | Description |
|---|---|
| `schema_has_quadtree` | 0/1 |
| `schema_has_octree` | 0/1 |
| `schema_has_kdtree` | 0/1 |
| `schema_has_grid2d` | 0/1 |
| `schema_has_grid3d` | 0/1 |
| `schema_num_blocks` | number of structure blocks |
| `schema_total_levels` | total scheduled levels |
| `schema_max_leaf_capacity` | max leaf capacity in config |
| `schema_min_leaf_capacity` | min leaf capacity in config |

### Evaluation metrics

#### Oracle score

The best measured schema for a given dataset/workload:

```text
oracle_score = min_schema measured_score
```

#### Regret

```text
regret = selected_score - oracle_score
relative_regret = selected_score / oracle_score - 1
```

#### Fixed baseline

Compare against:

- always quadtree,
- always octree,
- always kd-tree,
- always urban hybrid.

#### Heuristic baseline

Example:

```text
if flatness_score is high:
    choose quadtree
else if scattering is high:
    choose octree
else:
    choose urban_hybrid
```

### Files likely touched

```text
scripts/train_schema_selector.py
scripts/export_model.py
scripts/plot_results.py
docs/learner_plan.md
results/*.csv
results/*.json
```

### Acceptance criteria

- Training script runs on generated experiment CSV.
- Report includes oracle, fixed baselines, heuristic baseline, and learned selector.
- Learned selector beats at least one meaningful fixed baseline on held-out data.

### Implementation status

Implemented in this workspace:

- `scripts/train_schema_selector.py` loads raw schema-search CSV rows and augments them with schema-composition features from schema JSON.
- Dataset-level train/test split to avoid row leakage.
- Score-prediction/ranking models: ridge, random forest, and gradient-boosted trees.
- Direct best-schema logistic classifier when there are enough labels.
- Oracle, fixed-schema, majority, and heuristic baselines.
- Metrics for best-schema accuracy, top-2 accuracy, regret, relative regret, score speedup, selected build time, and selected memory.
- Model artifact output via `joblib` and metadata/report JSON outputs.
- Documentation in `docs/learner_plan.md`.

### Definition of done

There is a first ML result suitable for iteration and ablation.

---

## 16. Milestone 10 — C++ Auto-Selector Integration

### Goal

Allow the C++ application to choose a schema automatically using the trained selector.

### Codex prompt

```text
Implement Milestone 10 only.

Integrate the trained global schema selector into the C++ point-cloud benchmark path. The user should be able to run with --schema auto and get a selected schema plus benchmark metrics.
```

### Tasks

1. Export the trained model to a simple format.
2. Add a small C++ inference module.
3. Add CLI mode:
   - `--schema auto`,
   - `--model models/schema_selector.json`,
   - `--workload-profile configs/workloads/mixed.json`.
4. Runtime flow:
   - load point cloud,
   - extract features,
   - evaluate candidate schemas,
   - choose best predicted schema,
   - build index,
   - run benchmark,
   - log selected schema and metrics.
5. Add tests for deterministic schema choice on synthetic features.

### Model export options

Preferred first options:

1. export a small decision tree as JSON,
2. export a linear model as JSON,
3. export a table of rules,
4. generate a C++ header from Python.

Avoid heavyweight runtime dependencies at first.

### CLI example

```bash
MultiDataStructure.exe \
  --mode points \
  --input data/cloud.xyz \
  --schema auto \
  --model models/schema_selector.json \
  --workload-profile configs/workloads/mixed.json \
  --output results/auto_run.json
```

### Files likely touched

```text
MultiDataStructure/experiments/SchemaSelector.h
MultiDataStructure/experiments/SchemaSelector.cpp
scripts/export_model.py
models/schema_selector.json
MultiDataStructure/app/main.cpp
tests/test_schema_selector.cpp
```

### Acceptance criteria

- `--schema auto` works for point clouds.
- Selected schema is printed and logged.
- Manual fixed-schema mode still works.
- Triangle benchmark still works.

### Implementation status

Implemented in this workspace:

- `scripts/export_model.py` exports a trained Ridge score ranker to lightweight JSON with feature names, coefficients, intercept, and candidate schema paths.
- `scripts/export_model.py` can also export optional ONNX Runtime artifacts: a `.onnx` score model plus an `onnx_score_ranker` wrapper JSON.
- `scripts/tune_schema_for_cloud.py` tunes one target point cloud by measured schema-search results and exports a local `measured_best_schema` selector.
- `scripts/train_schema_selector.py --model-name ridge_score_predictor` can intentionally save an exportable runtime model while still reporting other learned selectors.
- `Experiments::SchemaSelector` loads exported JSON models, supports local measured-best selection, recomputes point/workload/schema features for linear and optional ONNX score rankers, predicts one score per candidate schema combination, and returns the selected schema.
- ONNX Runtime is guarded behind `MDSPC_ENABLE_ONNX` and optional MSBuild properties (`OnnxRuntimeDir`, or explicit include/library directories), so the dependency-free build remains valid.
- `--schema auto`, `--model`, and `--workload-profile` are wired into point mode.
- Auto-selected schema, model path, workload profile, predicted score, and full candidate ranking are printed and written to point benchmark JSON under `schema_selection`.
- Deterministic C++ selector coverage in `tests/test_schema_selector.cpp`.

Note: the maintained executable is now point-cloud focused; the legacy triangle benchmark is not part of the current Visual Studio target.

### Definition of done

The repository supports automatic learned global schema selection.

---

## 17. Milestone 11 — Real Point-Cloud Dataset Support

### Goal

Move beyond synthetic clouds.

### Codex prompt

```text
Implement Milestone 11 only.

Add support for real point-cloud datasets in a lightweight way. Start with formats and preprocessing that are easy to maintain. Do not make LAS/LAZ mandatory unless dependencies are straightforward.
```

### Tasks

1. Add clear data directory conventions.
2. Support additional simple formats if useful:
   - PLY,
   - PCD,
   - converted XYZ/CSV.
3. Add a preprocessing script for converting external data to the internal simple format.
4. Add dataset metadata JSON files.
5. Add benchmarking scripts that can iterate over datasets.

### Dataset metadata example

```json
{
  "name": "sample_urban_block",
  "source": "converted",
  "format": "xyz",
  "num_points": 1000000,
  "notes": "Example real or converted point cloud",
  "recommended_workloads": ["range_heavy", "mixed"]
}
```

### Optional LAS/LAZ path

If LAS/LAZ is added, isolate it behind optional compile flags or Python preprocessing. Do not make the whole project depend on it by default.

### Files likely touched

```text
data/README.md
scripts/convert_pointcloud.py
scripts/run_real_dataset_benchmarks.py
MultiDataStructure/workloads/points/PointCloudIO.*
```

### Acceptance criteria

- At least one non-synthetic point cloud can be loaded or converted.
- Benchmark scripts can include real datasets.
- The system remains usable without proprietary or heavy dependencies.

### Definition of done

The experimental pipeline is ready for real-world validation.

---

## 18. Milestone 12 — Paper-Ready Evaluation Package

### Goal

Prepare results and artifacts for a research paper.

### Codex prompt

```text
Implement Milestone 12 only.

Add scripts and documentation for paper-ready experiments: baselines, ablations, plots, result summaries, and reproducibility instructions.
```

### Required baselines

1. pure quadtree,
2. pure octree,
3. pure kd-tree,
4. quadtree -> octree,
5. grid2D -> quadtree -> grid3D -> octree,
6. simple heuristic selector,
7. learned global selector,
8. oracle selector.

### Required datasets

1. synthetic flat ALS-like terrain,
2. synthetic volumetric TLS-like scene,
3. synthetic mixed urban scene,
4. sparse/dense mixed scene,
5. at least one real or converted point cloud.

### Required workload profiles

1. range-heavy,
2. KNN-heavy,
3. mixed.

### Required plots

1. query throughput by schema,
2. build time by schema,
3. memory estimate by schema,
4. node count by schema,
5. leaf occupancy distribution,
6. relative regret vs oracle,
7. schema choices by dataset type,
8. feature importance for learned model.

### Required tables

1. dataset statistics,
2. schema definitions,
3. average performance per workload,
4. learned selector vs baselines,
5. ablation of feature sets,
6. ablation of score weights.

### Files likely touched

```text
scripts/run_paper_experiments.py
scripts/plot_results.py
scripts/make_tables.py
docs/experiment_protocol.md
docs/paper_results_template.md
results/paper/*
```

### Acceptance criteria

- One command or documented sequence reproduces the main experiment table.
- Plots are generated from CSV/JSON, not manually.
- Experiment protocol is documented.

### Definition of done

The repo can support a first paper submission or technical report.

---

## 19. Milestone 13 — Optional Node-Wise Adaptive Learner

### Goal

Move from a global fixed schema to a dynamic local policy.

This milestone is explicitly optional and should only begin after the global selector is working.

### Codex prompt

```text
Implement Milestone 13 only if Milestones 0-12 are stable.

Prototype a node-wise adaptive policy that chooses how to refine each local region. Keep the action space small and enforce strict safety limits.
```

### Local action space

At each node, choose one of:

1. stop,
2. split as quadtree,
3. split as octree,
4. split as kd-tree,
5. split as grid2D,
6. split as grid3D,
7. choose one of a few capacity buckets.

### Local features

Use features computed from the node's local points:

- point count,
- local bounding-box extents,
- local density,
- covariance eigenvalues,
- planarity,
- scattering,
- local occupancy ratio,
- parent depth,
- remaining depth budget.

### Training label generation

For sampled nodes:

1. try each valid action,
2. build a small local subtree,
3. run local representative queries,
4. compute local score,
5. label the best action.

### Safety limits

- maximum depth,
- minimum points to split,
- maximum children,
- maximum memory growth,
- no recursive oscillation between incompatible structure types,
- deterministic behavior.

### Acceptance criteria

- Node-wise policy does not explode build time.
- Node-wise policy is compared against global selector.
- Node-wise policy improves at least one meaningful metric without major regressions.

### Definition of done

A dynamic meta-structure is available as an experimental advanced mode.

---

## 20. Milestone 14 — Optional Return to Path Tracing

### Goal

Apply lessons from point clouds to triangle/ray/path-tracing workloads.

This should only be attempted after the point-cloud framework is stable.

### Codex prompt

```text
Implement Milestone 14 only after the point-cloud learner is stable.

Extend the learned schema-selection framework back to triangle/ray workloads. Treat this as a separate research direction, not as part of the initial point-cloud paper.
```

### Tasks

1. Preserve BVH as the strongest baseline.
2. Add ray workload features:
   - primary rays,
   - shadow rays,
   - diffuse rays,
   - mixed ray distributions.
3. Add scene geometry features:
   - triangle count,
   - bounding-box anisotropy,
   - spatial clustering,
   - primitive overlap metrics.
4. Add candidate hybrid structures:
   - BVH only,
   - grid -> BVH,
   - BVH -> grid leaves,
   - octree -> BVH,
   - kd-tree -> BVH.
5. Benchmark against high-quality BVH.
6. Avoid claiming victory unless hybrid beats BVH on held-out scenes.

### Acceptance criteria

- BVH remains a baseline.
- Hybrid structures are evaluated with meaningful ray distributions.
- Results are reported separately from point-cloud results.

### Definition of done

A credible second research direction exists for rendering/path tracing.

---

## 21. First Concrete Codex Task

Use this prompt for the first agent run:

```text
You are working in the AlfonsoLRz/MultiDataStructure repository.

Read CODEX_PLAN.md. Implement Milestone 0 only.

Goal: freeze the current triangle/ray benchmark behavior before refactoring.

Tasks:
1. Create docs/current_behavior.md describing the current executable, current benchmark path, current node types, current schema configurations, and known limitations.
2. Add deterministic JSON logging for the current benchmark results.
3. Add minimal tests for level schedule semantics and cleanup behavior if those functions exist.
4. Do not restructure the repository yet.
5. Do not change algorithms unless required to expose metrics.
6. Preserve the current benchmark.

When done, report:
- files changed,
- tests added,
- build/test commands run,
- sample JSON output,
- any assumptions or limitations.
```

---

## 22. Alternative First Task If Codex Can Do a Larger Change

Use this only if the agent environment is robust and can compile the project:

```text
Read CODEX_PLAN.md. Implement Milestones 0 and 1 only.

First freeze the current behavior with docs, JSON logs, and tests. Then separate generic spatial-index code from triangle/ray-specific code while preserving current behavior.

Do not add point-cloud support yet.
```

---

## 23. Suggested Command-Line Interface Target

Eventually the executable should support commands like:

```bash
# Existing triangle benchmark
MultiDataStructure.exe \
  --mode triangles \
  --scene assets/cornell.obj \
  --benchmark default \
  --output results/triangle_baseline.json
```

```bash
# Fixed-schema point-cloud benchmark
MultiDataStructure.exe \
  --mode points \
  --input data/cloud.xyz \
  --schema configs/schemas/octree.json \
  --workload-profile configs/workloads/mixed.json \
  --output results/point_octree_mixed.json
```

```bash
# Auto-selected point-cloud schema
MultiDataStructure.exe \
  --mode points \
  --input data/cloud.xyz \
  --schema auto \
  --model models/schema_selector.json \
  --workload-profile configs/workloads/mixed.json \
  --output results/point_auto_mixed.json
```

---

## 24. Suggested Testing Matrix

### Unit tests

| Test | Purpose |
|---|---|
| `test_level_schedule` | verify level-to-node-type mapping |
| `test_tree_cleanup` | verify empty-node removal and collapse behavior |
| `test_config_parsing` | verify schema JSON loading |
| `test_pointcloud_io` | verify XYZ/CSV loading |
| `test_synthetic_clouds` | verify generated point counts and bounds |
| `test_point_build` | verify point counts after indexing |
| `test_point_queries` | compare indexed queries against brute force |
| `test_feature_extraction` | verify deterministic features |
| `test_schema_selector` | verify deterministic auto-selection |

### Integration tests

| Test | Purpose |
|---|---|
| triangle baseline run | ensure existing benchmark survives |
| point octree benchmark | validate fixed point schema |
| point hybrid benchmark | validate nested point schema |
| schema search smoke test | validate sweep output |
| training smoke test | validate learner script |

---

## 25. Suggested Dataset Progression

### Stage 1: tiny synthetic

Use 100 to 10,000 points. These are for correctness tests.

### Stage 2: medium synthetic

Use 100,000 to 1,000,000 points. These are for early performance tests.

### Stage 3: mixed synthetic

Include:

- flat terrain,
- vertical facades,
- buildings,
- sparse/dense clusters,
- overlapping density regimes.

### Stage 4: real converted data

Use real point clouds converted to XYZ/CSV/PLY.

### Stage 5: LAS/LAZ optional

Add only after the benchmark and learning loop are stable.

---

## 26. Suggested Paper Story

A possible paper structure:

1. **Problem:** single spatial indices perform poorly on heterogeneous point clouds.
2. **Background:** nested spatial structures can adapt to different distributions, but schema selection is manual.
3. **Contribution:** automatic schema selection for nested point-cloud indices.
4. **Method:** candidate schema grammar, benchmark-generated labels, point-cloud features, learned selector.
5. **Evaluation:** compare against fixed quadtree, octree, kd-tree, hand-designed hybrids, heuristic selector, and oracle.
6. **Results:** learned selector reduces regret and improves query performance across mixed workloads.
7. **Discussion:** limits, dynamic local policy as future work, extension to rendering.

Possible title:

> Learning to Select Nested Spatial Data Structures for Heterogeneous Point-Cloud Queries

---

## 27. Key Risks and Mitigations

### Risk: refactor breaks current benchmark

Mitigation:

- Milestone 0 freezes behavior.
- Keep triangle path as an integration test.

### Risk: point-cloud path inherits triangle duplication logic

Mitigation:

- Explicit single-child assignment for points.
- Add tests that count points before and after indexing.

### Risk: schema search becomes too expensive

Mitigation:

- Start with small candidate set.
- Use synthetic medium clouds.
- Log raw metrics and reuse results.

### Risk: learner overfits synthetic data

Mitigation:

- Split by dataset family.
- Include real converted datasets as soon as possible.
- Report regret, not just accuracy.

### Risk: model integration becomes too heavy

Mitigation:

- Export simple tree/rule/linear model to JSON.
- Avoid heavyweight C++ ML dependencies.

### Risk: node-wise adaptive policy explodes complexity

Mitigation:

- Defer until global selector works.
- Keep local action space small.
- Enforce hard depth and occupancy limits.

---

## 28. Final Success Criteria

The project is successful when it supports this workflow:

1. load or generate a point cloud,
2. build multiple fixed nested spatial schemas,
3. run range/radius/KNN workloads,
4. log metrics to JSON/CSV,
5. extract deterministic point-cloud features,
6. search schemas to generate labels,
7. train a global schema selector,
8. auto-select a schema for a new cloud/workload,
9. compare learned selection against fixed and heuristic baselines,
10. preserve the original triangle/ray benchmark for future rendering experiments.

At that point, the repository becomes a credible platform for research on learnable nested spatial indexing.

---

## 29. Compact Milestone Checklist

```text
[ ] M0  Baseline freeze, docs, JSON logs, tests
[ ] M1  Separate generic core from triangle workload
[ ] M2  Add point-cloud data model and simple I/O
[ ] M3  Add point-cloud build path
[ ] M4  Add configurable build policies and JSON schemas
[x] M5  Add point-cloud query workloads
[x] M6  Add benchmark metrics and experiment logging
[x] M7  Add schema grammar and search harness
[x] M8  Add feature extraction
[x] M9  Train Python global schema selector
[x] M10 Integrate auto-selector into C++
[ ] M11 Add real point-cloud dataset support
[ ] M12 Add paper-ready evaluation package
[ ] M13 Optional node-wise adaptive learner
[ ] M14 Optional return to path tracing
```

---

## 30. Minimal Non-Research Implementation Path

If time is limited, do only this subset first:

```text
M0 -> M2 -> M3 -> M5 -> M6 -> M7 -> M8 -> M9
```

This gives:

- point clouds,
- queries,
- experiments,
- schema search,
- learner.

M1 can be partially deferred if the refactor is too disruptive, but keeping the code clean will pay off.

---

## 31. Minimal Research Demonstrator

A minimum viable research demo should show:

1. synthetic flat terrain chooses quadtree or planar-heavy schema,
2. synthetic volumetric/TLS-like cloud chooses octree or 3D-heavy schema,
3. mixed urban cloud chooses hybrid schema,
4. learned selector beats always-octree and always-quadtree on held-out synthetic scenes,
5. regret vs oracle is reported.

This is enough for an early workshop-style result.

---

## 32. Notes for Future Extensions

### LAS/LAZ

Add later through:

- optional C++ dependency,
- Python preprocessing,
- or PDAL-based conversion.

Do not block the learning loop on LAS/LAZ.

### GPU acceleration

Add later after CPU correctness and metrics are stable.

### Out-of-core indexing

Relevant later, especially for LiDAR-scale data. For the first learner, in-memory indexing is acceptable.

### Path tracing

Return later with strong BVH baselines and ray-distribution-aware metrics.

---

# End of Plan

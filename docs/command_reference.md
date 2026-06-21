# MultiDataStructure Command Reference

This file collects the main command lines for building, training/exporting selectors, running point benchmarks, and searching generated schema combinations.

## Python Environment

Use the project-local environment for ML/export scripts:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

## Build

Normal dependency-free Release build:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" MultiDataStructure.sln /p:Configuration=Release /p:Platform=x64 /m
```

Download local ONNX Runtime GPU SDK under `.deps`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_onnx_runtime.ps1
```

Build with ONNX Runtime enabled:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_with_onnx.ps1
```

Helper script arguments:

```text
scripts\setup_onnx_runtime.ps1 -Version <version> -PackageId <nuget-package-id>
scripts\build_with_onnx.ps1 -Configuration Release|Debug -Platform x64 -Version <version>
```

Run tests:

```powershell
.\x64\Release\MultiDataStructure.exe --run-tests --no-pause
```

Open the ImGui optimizer interface:

```powershell
.\x64\Release\MultiDataStructure.exe --mode gui
```

With no arguments, the executable now opens the same GUI by default.

Schema-search defaults to CUDA/Mixed on device `0`, `configs/workloads/volume_small_medium.json`, 64 prepared queries, 256 generated candidates, `models/schema_selector.json`, and top-32 measured benchmarking. The GUI opens on the publication-focused path: one real cloud, CUDA/Mixed device `0`, volume workload, generated-only conditional candidates, and per-cloud auto-condition tuning with the staged 256/16/4 budget. If CUDA is unavailable, schema-search logs a warning and falls back to CPU. Point-mode benchmarking keeps its existing CPU path.

## Fixed-Schema Point Benchmark

Run one explicit schema:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:/Datasets/points/Alhambra_100M.las --schema configs/schemas/octree.json --workload-profile configs/workloads/mixed.json --queries 16 --no-pause
```

Run several static schemas on the same loaded point cloud:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:/Datasets/points/Alhambra_100M.las --schemas "configs/schemas/quadtree.json;configs/schemas/octree.json;configs/schemas/kdtree.json;configs/schemas/quadtree_octree.json;configs/schemas/octree_kdtree.json;configs/schemas/urban_hybrid.json" --queries 64 --csv results/static_schema_sweep.csv --no-pause
```

Replay a generated schema:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:/Datasets/points/Alhambra_100M.las --schema results/generated_schemas/generated_qt4l128_ot5l128.json --workload-profile configs/workloads/mixed.json --queries 64 --no-pause
```

Opt into heavy-leaf micro-indexes for CPU point benchmarks. This keeps the macro schema unchanged, but builds a leaf-local uniform grid for range/count/radius and a leaf-local KD tree for KNN only on leaves above the threshold:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:/Datasets/points/Alhambra_100M.las --schema configs/schemas/octree.json --queries 64 --leaf-micro-indexes --leaf-micro-threshold 512 --csv results/point_micro_index.csv --no-pause
```

Schema JSON can set the same opt-in through `buildPolicy`:

```json
{
  "buildPolicy": {
    "enableLeafMicroIndexes": true,
    "leafMicroIndexThreshold": 512
  }
}
```

## Local Measured Tuning

Tune one point cloud over the configured candidate list and export a measured-best selector:

```powershell
.\.venv\Scripts\python.exe scripts\tune_schema_for_cloud.py --input C:/Datasets/points/Alhambra_100M.las --workload-profile configs/workloads/volume_small_medium.json --queries 64 --auto-conditions --output models/alhambra_local_selector.json
```

Include adaptive per-node leaf capacity in generated candidates:

```powershell
.\.venv\Scripts\python.exe scripts\tune_schema_for_cloud.py --input C:/Datasets/points/Alhambra_100M.las --workload-profile configs/workloads/volume_small_medium.json --queries 64 --auto-conditions --adaptive-leaf-capacity --output models/alhambra_adaptive_selector.json
```

Use that measured winner:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:/Datasets/points/Alhambra_100M.las --schema auto --model models/alhambra_local_selector.json --workload-profile configs/workloads/volume_small_medium.json --no-pause
```

## Global Learner And ONNX Export

Train a score-ranker from schema-search rows:

```powershell
.\.venv\Scripts\python.exe scripts\train_schema_selector.py --input results/schema_search.csv --report results/schema_selector_report.json --model-output models/schema_selector.joblib --model-meta-output models/schema_selector_meta.json --model-name ridge_score_predictor
```

Export dependency-free linear JSON:

```powershell
.\.venv\Scripts\python.exe scripts\export_model.py --model models/schema_selector.joblib --metadata models/schema_selector_meta.json --output models/schema_selector.json
```

Export ONNX selector:

```powershell
.\.venv\Scripts\python.exe scripts\export_model.py --model models/schema_selector.joblib --metadata models/schema_selector_meta.json --skip-linear-json --onnx-output models/schema_selector.onnx --onnx-json-output models/schema_selector_onnx.json --onnx-execution-provider cuda
```

Run auto-selection with ONNX inference:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:/Datasets/points/Alhambra_100M.las --schema auto --model models/schema_selector_onnx.json --workload-profile configs/workloads/mixed.json --no-pause
```

## Generated Hyperspace Search

Generated-only search with lightweight JSON pruning. This generates 256 schemas, ranks them, and C++ benchmarks only the top 32:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --rank-model models/schema_selector.json --benchmark-top 32 --workloads configs/workloads/volume_small_medium.json --queries 64 --csv results/alhambra_generated_search.csv --best-csv results/alhambra_generated_best.csv --no-pause
```

Measure every generated schema without surrogate pruning. Slower, but gives the measured best over the full generated set:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --workloads configs/workloads/volume_small_medium.json --queries 64 --csv results/alhambra_generated_search_full.csv --best-csv results/alhambra_generated_best_full.csv --no-pause
```

Search generated schemas plus the static baseline schemas:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generate-schemas 256 --rank-model models/schema_selector.json --benchmark-top 32 --workloads configs/workloads/volume_small_medium.json --queries 64 --csv results/alhambra_static_plus_generated.csv --best-csv results/alhambra_static_plus_generated_best.csv --no-pause
```

Pure query-latency score:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --rank-model models/schema_selector.json --benchmark-top 32 --workloads configs/workloads/volume_small_medium.json --queries 64 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_query_only.csv --best-csv results/alhambra_query_only_best.csv --no-pause
```

Query plus memory and occupancy penalties, with build time ignored:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --rank-model models/schema_selector.json --benchmark-top 32 --workloads configs/workloads/volume_small_medium.json --queries 64 --score-build-weight 0 --score-memory-weight 0.01 --score-imbalance-weight 0.01 --csv results/alhambra_query_memory.csv --best-csv results/alhambra_query_memory_best.csv --no-pause
```

Small-to-medium 3D volume-query search. This uses AABB range queries only, with query boxes sampled from 1% to 25% of the dataset extent in each dimension:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --workloads configs/workloads/volume_small_medium.json --queries 64 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_volume_small_medium.csv --best-csv results/alhambra_volume_small_medium_best.csv --no-pause
```

Generated conditional schema search. Later generated blocks may include local node predicates, so sibling branches can skip or enter different nested blocks:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --generated-conditional --generated-condition-probability 0.5 --workloads configs/workloads/volume_small_medium.json --queries 64 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_volume_conditional.csv --best-csv results/alhambra_volume_conditional_best.csv --no-pause
```

CPU adaptive leaf-capacity search. Generated levels may include `adaptiveLeafCapacity`, which scales each node's effective leaf capacity from density, height-ratio, anisotropy, and query-mix factors. Keep this on CPU for now; CUDA generated runs disable it and fixed adaptive schemas are rejected by schema-search CUDA evaluators until GPU support exists:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --generated-conditional --generated-adaptive-leaf-capacity --generated-adaptive-leaf-probability 0.35 --evaluator cpu --workloads configs/workloads/volume_small_medium.json --queries 64 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_adaptive_leaf_cpu.csv --best-csv results/alhambra_adaptive_leaf_cpu_best.csv --no-pause
```

Per-cloud auto-condition tuning. This estimates threshold domains from the point cloud, screens generated numeric conditional schemas with a small proxy workload, confirms the shortlist on the full cloud, writes the winning schema JSON, and updates a measured selector artifact:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --auto-conditions --workloads configs/workloads/volume_small_medium.json --queries 64 --csv results/alhambra_auto_conditions.csv --best-csv results/alhambra_auto_conditions_best.csv --no-pause
```

Deep nested search for the publication target. This injects pure single-DS baselines, runs CPU discovery with template-guided nested candidates, records runtime active-structure fractions, and performs CUDA/Mixed confirmation as a report only:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --deep-nested-search --workloads configs/workloads/volume_small_medium.json --csv results/alhambra_deep_nested.csv --best-csv results/alhambra_deep_nested_best.csv --no-pause
```

Reduced deep-search smoke preset for quick validation on synthetic data:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --synthetic-scale 32 --deep-nested-search --condition-proxy-candidates 4 --condition-final-top 2 --condition-confirm-top 1 --condition-proxy-queries 1 --queries 2 --condition-output-dir $env:TEMP\mdspc_deep_smoke --condition-selector-output $env:TEMP\mdspc_deep_smoke_selector.json --no-csv --no-pause
```

Evolutionary schema optimization. This evaluates an initial population, keeps the best measured schemas as parents, adds diagnostic repair children, mutates them, adds random immigrants, and repeats:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generate-schemas 128 --generated-conditional --optimize-schemas --repair-mutations --optimizer-generations 4 --optimizer-population 64 --optimizer-elites 8 --optimizer-mutation-rate 0.65 --optimizer-random-fraction 0.20 --workloads configs/workloads/volume_small_medium.json --queries 64 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_evolution.csv --best-csv results/alhambra_evolution_best.csv --no-pause
```

LBVH GPU evaluator. This keeps the optimizer and candidate loop on CPU, but builds LBVH and measures range/count/radius/KNN queries on the selected CUDA device. CUDA KNN is reported as `knn_backend=bruteforce_gpu_scan` because it scans the GPU point buffer rather than traversing the structure:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --generated-conditional --optimize-schemas --optimizer-generations 4 --optimizer-population 64 --optimizer-elites 8 --workloads configs/workloads/volume_small_medium.json --queries 64 --evaluator cuda --cuda-device 0 --cuda-builder lbvh --cuda-query-batch 0 --cuda-memory-budget-mb 0 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_cuda_lbvh_search.csv --best-csv results/alhambra_cuda_lbvh_best.csv --no-pause
```

RegularGrid GPU evaluator. This uses CUDA cell binning plus exact range/count/radius query kernels and brute-force parallel GPU point-buffer KNN:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --generated-conditional --optimize-schemas --optimizer-generations 4 --optimizer-population 64 --optimizer-elites 8 --workloads configs/workloads/volume_small_medium.json --queries 64 --evaluator cuda --cuda-device 0 --cuda-builder regular_grid --cuda-query-batch 0 --cuda-memory-budget-mb 0 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_cuda_regular_grid_search.csv --best-csv results/alhambra_cuda_regular_grid_best.csv --no-pause
```

KDTree GPU evaluator. This builds a spatial-median KD tree on CUDA and measures exact range/count/radius queries. KNN defaults to exact tree traversal for `k <= 16`; pass `--cuda-knn-backend gpu_bruteforce_knn` to force the point-buffer scan:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --generated-conditional --optimize-schemas --optimizer-generations 4 --optimizer-population 64 --optimizer-elites 8 --workloads configs/workloads/volume_small_medium.json --queries 64 --evaluator cuda --cuda-device 0 --cuda-builder kdtree --cuda-query-batch 0 --cuda-memory-budget-mb 0 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_cuda_kdtree_search.csv --best-csv results/alhambra_cuda_kdtree_best.csv --no-pause
```

BIH GPU evaluator. This builds a binary interval hierarchy on CUDA, refits tight child bounds after partitioning, and measures exact range/count/radius queries. KNN defaults to exact tree traversal for `k <= 16`; pass `--cuda-knn-backend gpu_bruteforce_knn` to force the point-buffer scan:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --generated-conditional --optimize-schemas --optimizer-generations 4 --optimizer-population 64 --optimizer-elites 8 --workloads configs/workloads/volume_small_medium.json --queries 64 --evaluator cuda --cuda-device 0 --cuda-builder bih --cuda-query-batch 0 --cuda-memory-budget-mb 0 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_cuda_bih_search.csv --best-csv results/alhambra_cuda_bih_best.csv --no-pause
```

Octree GPU evaluator. This builds a midpoint octree on CUDA and measures exact range/count/radius queries plus brute-force parallel GPU point-buffer KNN:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --generated-conditional --optimize-schemas --optimizer-generations 4 --optimizer-population 64 --optimizer-elites 8 --workloads configs/workloads/volume_small_medium.json --queries 64 --evaluator cuda --cuda-device 0 --cuda-builder octree --cuda-query-batch 0 --cuda-memory-budget-mb 0 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_cuda_octree_search.csv --best-csv results/alhambra_cuda_octree_best.csv --no-pause
```

KarrasOctree GPU evaluator. This builds an octree from a Morton-sorted point order and prefix child ranges, which is useful when build time has a nonzero score:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --generated-conditional --optimize-schemas --optimizer-generations 4 --optimizer-population 64 --optimizer-elites 8 --workloads configs/workloads/volume_small_medium.json --queries 64 --evaluator cuda --cuda-device 0 --cuda-builder karras_octree --cuda-query-batch 0 --cuda-memory-budget-mb 0 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_cuda_karras_octree_search.csv --best-csv results/alhambra_cuda_karras_octree_best.csv --no-pause
```

QuadTree GPU evaluator. This builds a midpoint XY quadtree on CUDA and measures exact range/count/radius queries plus brute-force parallel GPU point-buffer KNN:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --generated-conditional --optimize-schemas --optimizer-generations 4 --optimizer-population 64 --optimizer-elites 8 --workloads configs/workloads/volume_small_medium.json --queries 64 --evaluator cuda --cuda-device 0 --cuda-builder quadtree --cuda-query-batch 0 --cuda-memory-budget-mb 0 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_cuda_quadtree_search.csv --best-csv results/alhambra_cuda_quadtree_best.csv --no-pause
```

HGrid GPU evaluator. This builds several CUDA RegularGrid levels and chooses a level per query to reduce over-testing across small/medium volumes:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --generated-conditional --optimize-schemas --optimizer-generations 4 --optimizer-population 64 --optimizer-elites 8 --workloads configs/workloads/volume_small_medium.json --queries 64 --evaluator cuda --cuda-device 0 --cuda-builder hgrid --cuda-query-batch 0 --cuda-memory-budget-mb 0 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_cuda_hgrid_search.csv --best-csv results/alhambra_cuda_hgrid_best.csv --no-pause
```

MixedTree GPU evaluator. This follows each schema's per-depth structure schedule, so static mixes such as QuadTree -> RegularGrid -> HGrid -> KarrasOctree -> BIH -> KDTree -> LBVH are built and queried on CUDA. Conditional generated levels are honored as split gates. Bbox anisotropy gates are supported; occupancy-entropy gates are rejected for CUDA/Mixed until implemented. In mixed schemas, `RegularGrid` and `HGrid` act as per-node grid split levels; the standalone CUDA builders still use their global grid implementations:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --generated-conditional --optimize-schemas --optimizer-generations 4 --optimizer-population 64 --optimizer-elites 8 --workloads configs/workloads/volume_small_medium.json --queries 64 --evaluator cuda --cuda-device 0 --cuda-builder mixed --cuda-query-batch 0 --cuda-memory-budget-mb 0 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_cuda_mixed_search.csv --best-csv results/alhambra_cuda_mixed_best.csv --no-pause
```

LBVH smoke test on synthetic data:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --synthetic-scale 128 --schemas configs/schemas/bvh.json --workloads configs/workloads/volume_small_medium.json --queries 16 --generate-schemas 0 --evaluator cuda --cuda-builder lbvh --no-csv --no-pause
```

RegularGrid smoke test on synthetic data:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --synthetic-scale 128 --schemas configs/schemas/bvh.json --workloads configs/workloads/volume_small_medium.json --queries 16 --generate-schemas 0 --evaluator cuda --cuda-builder regular_grid --no-csv --no-pause
```

KDTree smoke test on synthetic data:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --synthetic-scale 128 --schemas configs/schemas/kdtree.json --workloads configs/workloads/volume_small_medium.json --queries 16 --generate-schemas 0 --evaluator cuda --cuda-builder kdtree --no-csv --no-pause
```

BIH smoke test on synthetic data:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --synthetic-scale 128 --schemas configs/schemas/kdtree.json --workloads configs/workloads/volume_small_medium.json --queries 16 --generate-schemas 0 --evaluator cuda --cuda-builder bih --no-csv --no-pause
```

Octree smoke test on synthetic data:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --synthetic-scale 128 --schemas configs/schemas/octree.json --workloads configs/workloads/volume_small_medium.json --queries 16 --generate-schemas 0 --evaluator cuda --cuda-builder octree --no-csv --no-pause
```

KarrasOctree smoke test on synthetic data:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --synthetic-scale 128 --schemas configs/schemas/octree.json --workloads configs/workloads/volume_small_medium.json --queries 16 --generate-schemas 0 --evaluator cuda --cuda-builder karras_octree --no-csv --no-pause
```

QuadTree smoke test on synthetic data:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --synthetic-scale 128 --schemas configs/schemas/quadtree.json --workloads configs/workloads/volume_small_medium.json --queries 16 --generate-schemas 0 --evaluator cuda --cuda-builder quadtree --no-csv --no-pause
```

HGrid smoke test on synthetic data:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --synthetic-scale 128 --schemas configs/schemas/bvh.json --workloads configs/workloads/volume_small_medium.json --queries 16 --generate-schemas 0 --evaluator cuda --cuda-builder hgrid --no-csv --no-pause
```

MixedTree smoke test on synthetic data:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --synthetic-scale 128 --schemas configs/schemas/urban_hybrid.json --workloads configs/workloads/volume_small_medium.json --queries 16 --generate-schemas 0 --evaluator cuda --cuda-builder mixed --no-csv --no-pause
```

MixedTree smoke test that exercises all currently recursive CUDA split names:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --synthetic-scale 128 --schemas configs/schemas/gpu_mixed_all.json --workloads configs/workloads/volume_small_medium.json --queries 16 --generate-schemas 0 --evaluator cuda --cuda-builder mixed --no-csv --no-pause
```

Replay the hand-authored conditional quadtree-to-octree schema:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:/Datasets/points/Alhambra_100M.las --schema configs/schemas/adaptive_quadtree_octree.json --queries 64 --no-pause
```

Workload query-scale syntax:

```json
{
  "queryScales": {
    "aabb_range": { "min": 0.01, "max": 0.25 },
    "radius": { "min": 0.01, "max": 0.08 }
  }
}
```

Workload JSON files may also set `"stratifyQueries": true`. Schema-search then samples deterministic query strata such as `range_small`, `range_large`, `range_dense`, `radius_boundary`, `knn_dense`, and `knn_outside`, writes per-stratum metric summaries to `query_strata_summary`, and includes `query_stratum` in query traces.

## Reading The Best Schema

Inspect the best measured result:

```powershell
Import-Csv results/alhambra_generated_best.csv | Format-List dataset_name,workload_name,best_schema_name,best_schema_path,best_score,best_avg_latency_ms,best_build_time_ms,best_memory_estimate_bytes,num_candidates
```

Extract only the schema path:

```powershell
(Import-Csv results/alhambra_generated_best.csv | Select-Object -First 1).best_schema_path
```

Replay the winner:

```powershell
$best = (Import-Csv results/alhambra_generated_best.csv | Select-Object -First 1).best_schema_path
.\x64\Release\MultiDataStructure.exe --input C:/Datasets/points/Alhambra_100M.las --schema $best --workload-profile configs/workloads/volume_small_medium.json --queries 64 --no-pause
```

## Main Executable Arguments

Mode and basic IO:

```text
--mode gui|points|schema-search|tests
--gui
--input <path>
--output <path>
--csv <path>
--best-csv <path>
--no-csv
--run-tests
--no-pause
```

Point benchmark schema selection:

```text
--schema <path>
--schema auto
--schemas <a;b;c>
--model <path>
--workload-profile <path>
```

Schema-search workloads:

```text
--workloads <a;b;c>
--no-synthetic
--synthetic-scale <count>
```

Query profile:

```text
--queries <count>
--knn-k <count>
--query-seed <seed>
--query-trace <path>
```

`--query-trace` writes per-query CSV rows. CPU point-index rows include `query_stratum` plus traversal breakdowns by depth and structure; CUDA rows keep those breakdown columns empty for now.

CPU leaf micro-index options:

```text
--leaf-micro-indexes
--leaf-micro-threshold <count>
```

Leaf micro-indexes are disabled by default. When enabled, only leaves above the threshold build the extra local structures.

Point loading cache:

```text
--no-cache
--rebuild-cache
```

Generated schema search:

```text
--generate-schemas <count>
--generated-only
--deep-nested-search
--benchmark-top <count>
--rank-model <path>
--generated-min-blocks <n>
--generated-max-blocks <n>
--generated-max-depth <n>
--generated-min-leaf <n>
--generated-max-leaf <n>
--generated-conditional
--generated-condition-probability <value>
--generated-adaptive-leaf-capacity
--generated-adaptive-leaf-probability <value>
--generated-seed <seed>
--generated-schema-dir <path>
--primitive-profile <auto|query_minimal_cpu|cuda_query_full|all>
--query-minimal-primitives
--cuda-query-full-primitives
--all-primitives
```

CPU discovery defaults to `query_minimal_cpu`, which generates only `QuadTree`, `Octree`, `KDTree`, and `BVH` levels. Use `cuda_query_full` or `all` when the search itself should include CUDA-native aliases such as `KarrasOctree`, `BIH`, `LBVH`, `RegularGrid`, and `HGrid`.

Auto-condition schema tuning:

```text
--auto-conditions
--condition-proxy-candidates <n>
--condition-proxy-points <n>
--condition-proxy-queries <n>
--condition-final-top <n>
--condition-confirm-top <n>
--condition-output-dir <dir>
--condition-selector-output <path>
```

Evolutionary schema optimizer:

```text
--optimize-schemas
--optimizer-generations <n>
--optimizer-population <n>
--optimizer-elites <n>
--optimizer-mutation-rate <value>
--optimizer-random-fraction <value>
--optimizer-seed <seed>
--repair-mutations
--repair-top <n>
--repair-per-candidate <n>
```

Schema-search score weights:

```text
--score-build-weight <value>
--score-memory-weight <value>
--score-imbalance-weight <value>
```

CUDA schema-search evaluator:

```text
--evaluator cpu|cuda
--cuda-device <id>
--cuda-builder lbvh|kdtree|bih|octree|karras_octree|quadtree|regular_grid|hgrid|mixed
--cuda-query-batch <count>
--cuda-memory-budget-mb <mb>
--cuda-knn-backend auto|gpu_tree_knn|gpu_bruteforce_knn
--explain-report <path>
```

`lbvh`, `kdtree`, `bih`, `octree`, `karras_octree`, `quadtree`, `regular_grid`, `hgrid`, and static `mixed` schemas are implemented now. `karras_octree` uses Morton sorting plus prefix child ranges, `bih` is a binary interval hierarchy with tight child bounds, `hgrid` builds multiple CUDA grid levels and chooses one per query, and `mixed` follows the schema's per-depth structure schedule and treats conditional levels as GPU split gates. Mixed schema levels can currently name `QuadTree`, `Octree`, `KarrasOctree`, `KDTree`, `BIH`, `BVH`, `LBVH`, `RegularGrid`, and `HGrid`.

`QuadTree` schema levels default to `axisPolicy: "xy"` for point clouds. Other supported policies are `xz`, `yz`, `ignore_shortest`, `ignore_x`, `ignore_y`, and `ignore_z`.

Schema-search defaults to `--evaluator cuda --cuda-device 0 --cuda-builder mixed`; the resolver checks CUDA once and falls back to CPU with a warning when CUDA is unavailable. CUDA KNN has explicit backend provenance: CPU point mode remains `cpu_tree_knn`, generic GPU builders keep `gpu_bruteforce_knn`, and KDTree/BIH `auto` uses exact `gpu_tree_knn` for `k <= 16`.

Current default score:

```text
score = avg_query_latency_ms
      + 0.0  * build_time_ms
      + 0.0  * memory_mb
      + 0.0  * imbalance_penalty
```

Build time, memory, and imbalance are still logged, but they are not part of the default score.

Raw/best/pareto CSVs append score provenance columns: `lambda_latency`, `lambda_build`, `lambda_memory`, `lambda_imbalance`, `score_mode`, `score_stage`, `score_is_final_latency`, `effective_queries`, `score_uses_visit_proxy`, and `visit_proxy_alpha`. Workload JSON files may also define a `scoreWeights` object; explicit CLI score flags override those workload-local weights. To compare GA and non-GA runs:

The same CSVs now also append tree-health and query-strata diagnostics, including leaf occupancy quantiles, average depth, fanout, empty-child ratio, single-child count, tight-bounds volume ratio, micro-indexed leaf counts, and `query_strata_summary`.

```powershell
python scripts\audit_schema_scores.py results\non_ga.csv results\ga.csv --left-label non_ga --right-label ga
```

## Python Script Arguments

`scripts/export_model.py`:

```text
--model <path>
--metadata <path>
--output <path>
--skip-linear-json
--onnx-output <path>
--onnx-json-output <path>
--onnx-input-name <name>
--onnx-target-opset <int>
--onnx-execution-provider cpu|cuda
--onnx-device-id <int>
```

`scripts/tune_schema_for_cloud.py`:

```text
--exe <path>
--input <path>
--schemas <schema> [<schema> ...]
--workload-profile <path>
--queries <count>
--csv <path>
--best-csv <path>
--output <path>
--no-cache
--auto-conditions
--condition-proxy-candidates <n>
--condition-proxy-points <n>
--condition-proxy-queries <n>
--condition-final-top <n>
--condition-confirm-top <n>
```

`scripts/train_schema_selector.py`:

```text
--input <path>
--report <path>
--model-output <path>
--model-meta-output <path>
--model-name <name>
--learning-mode regression|classification|all
--test-fraction <float>
--seed <int>
```

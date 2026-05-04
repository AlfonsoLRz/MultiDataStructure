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

## Local Measured Tuning

Tune one point cloud over the configured candidate list and export a measured-best selector:

```powershell
.\.venv\Scripts\python.exe scripts\tune_schema_for_cloud.py --input C:/Datasets/points/Alhambra_100M.las --workload-profile configs/workloads/mixed.json --queries 128 --output models/alhambra_local_selector.json
```

Use that measured winner:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:/Datasets/points/Alhambra_100M.las --schema auto --model models/alhambra_local_selector.json --workload-profile configs/workloads/mixed.json --no-pause
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

Generated-only search with ONNX pruning. This generates 1000 schemas, ONNX ranks them, and C++ benchmarks only the top 16:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 1000 --rank-model models/schema_selector_onnx.json --benchmark-top 16 --workloads configs/workloads/mixed.json --queries 64 --csv results/alhambra_generated_search.csv --best-csv results/alhambra_generated_best.csv --no-pause
```

Measure every generated schema without surrogate pruning. Slower, but gives the measured best over the full generated set:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 1000 --workloads configs/workloads/mixed.json --queries 64 --csv results/alhambra_generated_search_full.csv --best-csv results/alhambra_generated_best_full.csv --no-pause
```

Search generated schemas plus the static baseline schemas:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generate-schemas 1000 --rank-model models/schema_selector_onnx.json --benchmark-top 24 --workloads configs/workloads/mixed.json --queries 64 --csv results/alhambra_static_plus_generated.csv --best-csv results/alhambra_static_plus_generated_best.csv --no-pause
```

Pure query-latency score:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 1000 --rank-model models/schema_selector_onnx.json --benchmark-top 16 --workloads configs/workloads/mixed.json --queries 64 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_query_only.csv --best-csv results/alhambra_query_only_best.csv --no-pause
```

Query plus memory and occupancy penalties, with build time ignored:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 1000 --rank-model models/schema_selector_onnx.json --benchmark-top 16 --workloads configs/workloads/mixed.json --queries 64 --score-build-weight 0 --score-memory-weight 0.01 --score-imbalance-weight 0.01 --csv results/alhambra_query_memory.csv --best-csv results/alhambra_query_memory_best.csv --no-pause
```

Small-to-medium 3D volume-query search. This uses AABB range queries only, with query boxes sampled from 1% to 25% of the dataset extent in each dimension:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 1000 --workloads configs/workloads/volume_small_medium.json --queries 128 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_volume_small_medium.csv --best-csv results/alhambra_volume_small_medium_best.csv --no-pause
```

Generated conditional schema search. Later generated blocks may include local node predicates, so sibling branches can skip or enter different nested blocks:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 1000 --generated-conditional --generated-condition-probability 0.5 --workloads configs/workloads/volume_small_medium.json --queries 128 --score-build-weight 0 --score-memory-weight 0 --score-imbalance-weight 0 --csv results/alhambra_volume_conditional.csv --best-csv results/alhambra_volume_conditional_best.csv --no-pause
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
.\x64\Release\MultiDataStructure.exe --input C:/Datasets/points/Alhambra_100M.las --schema $best --workload-profile configs/workloads/mixed.json --queries 128 --no-pause
```

## Main Executable Arguments

Mode and basic IO:

```text
--mode points|schema-search|tests
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
```

Point loading cache:

```text
--no-cache
--rebuild-cache
```

Generated schema search:

```text
--generate-schemas <count>
--generated-only
--benchmark-top <count>
--rank-model <path>
--generated-max-blocks <n>
--generated-max-depth <n>
--generated-min-leaf <n>
--generated-max-leaf <n>
--generated-conditional
--generated-condition-probability <value>
--generated-seed <seed>
--generated-schema-dir <path>
```

Schema-search score weights:

```text
--score-build-weight <value>
--score-memory-weight <value>
--score-imbalance-weight <value>
```

Current default score:

```text
score = avg_query_latency_ms
      + 0.0  * build_time_ms
      + 0.01 * memory_mb
      + 0.01 * imbalance_penalty
```

Build time is still logged, but it is not part of the default score.

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

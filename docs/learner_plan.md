# Learner Plan

The Python learner can be used in two different ways:

1. **Local tuning for one point cloud**: benchmark candidate schemas on the target cloud, choose the measured best, and export a tiny selector JSON that always uses that measured winner for the matching workload.
2. **Global score ranking**: learn a reusable score model from many measured datasets and rank candidate combinations by predicted score.

For the current project direction, local tuning is the more important path.

Current formulation:

```text
input = point_cloud_features + workload_features + schema_composition_features
target = measured schema-search score
```

At inference time, the selector scores every candidate schema combination and chooses the lowest predicted score. This is more useful than direct best-schema classification because schema composition features let later milestones evaluate generated combinations, not only the static JSON files used in the first experiments.

## What Python Extracts

Python extracts:

- a trained score-ranking model artifact,
- model metadata with feature column order and candidate schemas,
- a report comparing learned selectors against oracle, heuristic, majority, and fixed-schema baselines,
- feature importance where the model exposes it,
- regret, top-2 accuracy, score speedup, build-time, and memory effects.

The static schemas are therefore the initial measured candidate set. They are training/evaluation candidates for learning how combinations behave, not the final search space.

## Local Tuning Command

Use this when you want to overfit schema choice to one point cloud:

```powershell
python scripts/tune_schema_for_cloud.py --input C:/Datasets/points/Alhambra_100M.las --workload-profile configs/workloads/volume_small_medium.json --queries 64 --auto-conditions --output models/alhambra_local_selector.json
```

This runs schema-search with `--no-synthetic`, benchmarks the candidate schemas on that one point cloud, chooses the lowest measured score, and writes a `measured_best_schema` JSON. With `--auto-conditions`, it first estimates cheap point-cloud/node-feature domains, tunes concrete conditional thresholds through a staged proxy/full/confirmation search, and saves the winning numeric schema JSON for reuse.

For the deeper publication-oriented search, use the CPU-first nested mode:

```powershell
python scripts/tune_schema_for_cloud.py --input C:/Datasets/points/Alhambra_100M.las --workload-profile configs/workloads/volume_small_medium.json --deep-nested-search --output models/alhambra_deep_nested_selector.json
```

That mode treats pure single-structure schemas as injected baselines, searches only generated nested families during discovery, records whether the nested blocks actually became active, and reports the speedup against the best baseline in the CSV.

Then run:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:/Datasets/points/Alhambra_100M.las --schema auto --model models/alhambra_local_selector.json --workload-profile configs/workloads/volume_small_medium.json --no-pause
```

In this mode, `--schema auto` is not predicting. It is replaying the measured best schema from the local tuning run.

## Training Command

Use a project-local virtual environment for Python training/export dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

```powershell
python scripts/train_schema_selector.py --input results/schema_search.csv --report results/schema_selector_report.json --model-output models/schema_selector.joblib --model-meta-output models/schema_selector_meta.json
```

For C++ runtime use, export an explicit linear score ranker:

```powershell
python scripts/train_schema_selector.py --input results/schema_search.csv --model-output models/schema_selector.joblib --model-meta-output models/schema_selector_meta.json --model-name ridge_score_predictor
python scripts/export_model.py --model models/schema_selector.joblib --metadata models/schema_selector_meta.json --output models/schema_selector.json
```

The exported JSON contains `feature_names`, linear `coefficients`, `intercept`, and candidate schema paths. C++ recomputes the same feature vector for each candidate and picks the lowest predicted score.

For optional ONNX Runtime use, export an ONNX model plus a selector wrapper:

```powershell
python scripts/export_model.py --model models/schema_selector.joblib --metadata models/schema_selector_meta.json --skip-linear-json --onnx-output models/schema_selector.onnx --onnx-json-output models/schema_selector_onnx.json --onnx-execution-provider cuda
```

The ONNX wrapper contains `model_type: "onnx_score_ranker"`, the ONNX input/output names, execution provider, feature order, and candidate schemas. C++ uses the same feature vector and ranks candidates by the ONNX model output. This path is optional because the project still needs to build without ONNX Runtime installed.

## Generated Schema Search

The selector can now be used as a surrogate for generated candidates instead of only choosing from static schemas:

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --input C:/Datasets/points/Alhambra_100M.las --no-synthetic --generated-only --generate-schemas 256 --rank-model models/schema_selector.json --benchmark-top 32 --workloads configs/workloads/volume_small_medium.json --queries 64 --csv results\alhambra_generated_search.csv --best-csv results\alhambra_generated_best.csv --no-pause
```

This samples schema genomes from bounded intervals, writes generated schema JSON files, scores them cheaply with the exported JSON ranker, benchmarks the top candidates, and writes the measured winner. The ranker is only the pruning heuristic; the final result is still based on measured query/build/memory score.

Learning modes:

- `--learning-mode regression`: train score predictors and rank candidates by predicted score.
- `--learning-mode classification`: train a direct best-schema classifier when there are enough labels.
- `--learning-mode all`: train both and select the best held-out learned selector.

## Evaluation

Rows are split by dataset name so the same dataset cannot appear in both train and test. Evaluation groups are `(dataset_name, workload_name)` pairs.

The report includes:

- oracle measured score,
- fixed baselines such as always quadtree/octree/kd-tree/urban,
- majority train-oracle schema baseline,
- heuristic baseline,
- learned selectors,
- best-schema accuracy,
- top-2 accuracy,
- absolute and relative regret,
- score speedup against baselines,
- selected build time and memory.

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
python scripts/tune_schema_for_cloud.py --input C:/Datasets/points/Alhambra_100M.las --workload-profile configs/workloads/mixed.json --queries 128 --output models/alhambra_local_selector.json
```

This runs schema-search with `--no-synthetic`, benchmarks the candidate schemas on that one point cloud, chooses the lowest measured score, and writes a `measured_best_schema` JSON.

Then run:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:/Datasets/points/Alhambra_100M.las --schema auto --model models/alhambra_local_selector.json --workload-profile configs/workloads/mixed.json --no-pause
```

In this mode, `--schema auto` is not predicting. It is replaying the measured best schema from the local tuning run.

## Training Command

```powershell
python scripts/train_schema_selector.py --input results/schema_search.csv --report results/schema_selector_report.json --model-output models/schema_selector.joblib --model-meta-output models/schema_selector_meta.json
```

For C++ runtime use, export an explicit linear score ranker:

```powershell
python scripts/train_schema_selector.py --input results/schema_search.csv --model-output models/schema_selector.joblib --model-meta-output models/schema_selector_meta.json --model-name ridge_score_predictor
python scripts/export_model.py --model models/schema_selector.joblib --metadata models/schema_selector_meta.json --output models/schema_selector.json
```

The exported JSON contains `feature_names`, linear `coefficients`, `intercept`, and candidate schema paths. C++ recomputes the same feature vector for each candidate and picks the lowest predicted score.

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

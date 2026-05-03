# Models

Runtime selector artifacts are written here by convention.

Typical flow:

```powershell
python scripts/train_schema_selector.py --input results/schema_search.csv --model-output models/schema_selector.joblib --model-meta-output models/schema_selector_meta.json --model-name ridge_score_predictor
python scripts/export_model.py --model models/schema_selector.joblib --metadata models/schema_selector_meta.json --output models/schema_selector.json
```

`schema_selector.json` is the lightweight C++ runtime model used by `--schema auto`.

For per-cloud local tuning:

```powershell
python scripts/tune_schema_for_cloud.py --input C:/Datasets/points/Alhambra_100M.las --output models/alhambra_local_selector.json
```

That writes a `measured_best_schema` artifact. It is intentionally overfit to the tuned cloud/workload and reuses the measured best schema during `--schema auto`.

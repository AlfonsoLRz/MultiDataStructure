# Experiment Results

Point-cloud benchmark runs write machine-readable outputs here by default:

- `points_default.json`: one JSON record for a single-schema run.
- `points_summary.csv`: append-only CSV summary, one row per dataset/schema/workload run.
- `schema_search.csv`: raw schema-search rows, one row per dataset/workload/schema candidate, including feature columns.
- `schema_search_best.csv`: best schema per dataset/workload by the current score, retaining feature columns for training.

For multi-schema experiments, use:

```powershell
python scripts/run_experiments.py --input C:\data\sample.las --queries 128
```

For selector-label generation over synthetic datasets and workload profiles, use:

```powershell
python scripts/run_experiments.py --schema-search --queries 128
```

To recompute best-schema rows from an existing schema-search table:

```powershell
python scripts/summarize_results.py --input results/schema_search.csv --output results/schema_search_best.csv
```

To train the first offline selector:

```powershell
python scripts/train_schema_selector.py --input results/schema_search.csv --report results/schema_selector_report.json --model-output models/schema_selector.joblib --model-meta-output models/schema_selector_meta.json
```

To tune one point cloud and export the measured winner:

```powershell
python scripts/tune_schema_for_cloud.py --input C:/Datasets/points/Alhambra_100M.las --workload-profile configs/workloads/mixed.json --queries 128 --output models/alhambra_local_selector.json
```

To export a C++ runtime model:

```powershell
python scripts/export_model.py --model models/schema_selector.joblib --metadata models/schema_selector_meta.json --output models/schema_selector.json
```

The C++ executable also supports the same flow directly:

```powershell
.\x64\Release\MultiDataStructure.exe --input C:\data\sample.las --schemas "configs\schemas\octree.json;configs\schemas\kdtree.json" --queries 128 --csv results\points_summary.csv --no-pause
```

```powershell
.\x64\Release\MultiDataStructure.exe --mode schema-search --queries 128 --csv results\schema_search.csv --best-csv results\schema_search_best.csv --no-pause
```

```powershell
.\x64\Release\MultiDataStructure.exe --input C:\data\sample.xyz --schema auto --model models\schema_selector.json --workload-profile configs\workloads\mixed.json --output results\auto_run.json --no-pause
```

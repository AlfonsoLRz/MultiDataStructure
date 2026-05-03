# Models

Runtime selector artifacts are written here by convention.

Install Python exporter dependencies in a project-local environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

Typical flow:

```powershell
python scripts/train_schema_selector.py --input results/schema_search.csv --model-output models/schema_selector.joblib --model-meta-output models/schema_selector_meta.json --model-name ridge_score_predictor
python scripts/export_model.py --model models/schema_selector.joblib --metadata models/schema_selector_meta.json --output models/schema_selector.json
```

`schema_selector.json` is the lightweight C++ runtime model used by `--schema auto`.

For optional ONNX Runtime inference, export an ONNX model plus a selector wrapper JSON:

```powershell
python scripts/export_model.py --model models/schema_selector.joblib --metadata models/schema_selector_meta.json --skip-linear-json --onnx-output models/schema_selector.onnx --onnx-json-output models/schema_selector_onnx.json
```

The wrapper uses `model_type: "onnx_score_ranker"` and stores the ONNX input/output names, execution provider (`cpu` or `cuda`), feature order, and candidate schema paths. The C++ project only enables this path when ONNX Runtime is configured through `OnnxRuntimeDir` or explicit `OnnxRuntimeIncludeDir` / `OnnxRuntimeLibraryDir` MSBuild properties.

To build the executable with ONNX Runtime enabled:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_onnx_runtime.ps1
powershell -ExecutionPolicy Bypass -File scripts\build_with_onnx.ps1
```

For per-cloud local tuning:

```powershell
python scripts/tune_schema_for_cloud.py --input C:/Datasets/points/Alhambra_100M.las --output models/alhambra_local_selector.json
```

That writes a `measured_best_schema` artifact. It is intentionally overfit to the tuned cloud/workload and reuses the measured best schema during `--schema auto`.

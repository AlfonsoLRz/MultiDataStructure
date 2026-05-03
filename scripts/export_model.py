import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def extract_linear_model(model):
    if isinstance(model, Pipeline):
        scaler = None
        estimator = None
        for _, step in model.steps:
            if isinstance(step, StandardScaler):
                scaler = step
            elif isinstance(step, Ridge):
                estimator = step

        if estimator is None:
            raise TypeError("Only Ridge score predictors can currently be exported to C++ JSON")

        coefficients = np.asarray(estimator.coef_, dtype=float)
        intercept = float(estimator.intercept_)
        if scaler is not None:
            scale = np.asarray(scaler.scale_, dtype=float)
            mean = np.asarray(scaler.mean_, dtype=float)
            safe_scale = np.where(scale == 0.0, 1.0, scale)
            coefficients_original = coefficients / safe_scale
            intercept_original = intercept - float(np.dot(coefficients_original, mean))
            return coefficients_original.tolist(), intercept_original

        return coefficients.tolist(), intercept

    if isinstance(model, Ridge):
        return np.asarray(model.coef_, dtype=float).tolist(), float(model.intercept_)

    raise TypeError("Only Ridge score predictors can currently be exported to C++ JSON")


def candidate_schemas_from_metadata(metadata):
    candidate_schemas = metadata.get("candidate_schema_paths", [])
    if not candidate_schemas:
        candidate_schemas = [{"name": name, "path": ""} for name in metadata.get("candidate_schemas", [])]
    return candidate_schemas


def export_onnx_model(model, metadata, onnx_output, input_name, target_opset, execution_provider, device_id, json_output):
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise RuntimeError("ONNX export requires optional Python packages: pip install skl2onnx onnx") from exc

    feature_names = metadata["feature_names"]
    initial_types = [(input_name, FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=target_opset)
    if not onnx_model.graph.output:
        raise RuntimeError("Converted ONNX model has no outputs")

    output_name = onnx_model.graph.output[0].name
    onnx_path = Path(onnx_output)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_model.SerializeToString())

    if json_output:
        selector_path = Path(json_output)
    else:
        selector_path = onnx_path.with_name(f"{onnx_path.stem}_selector.json")

    payload = {
        "model_type": "onnx_score_ranker",
        "source_model": str(onnx_path),
        "training_formulation": metadata.get("training_formulation", "score_prediction_ranking"),
        "input_name": input_name,
        "output_name": output_name,
        "execution_provider": execution_provider,
        "device_id": device_id,
        "feature_names": feature_names,
        "candidate_schemas": candidate_schemas_from_metadata(metadata),
        "report": metadata.get("report", ""),
    }

    selector_path.parent.mkdir(parents=True, exist_ok=True)
    with selector_path.open("w") as file:
        json.dump(payload, file, indent=2)

    print(f"Exported ONNX model to {onnx_path}")
    print(f"Exported ONNX selector JSON to {selector_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a trained score ranker to C++ runtime artifacts.")
    parser.add_argument("--model", default="models/schema_selector.joblib", help="Joblib model from train_schema_selector.py.")
    parser.add_argument("--metadata", default="models/schema_selector_meta.json", help="Metadata JSON from train_schema_selector.py.")
    parser.add_argument("--output", default="models/schema_selector.json", help="C++ runtime model JSON.")
    parser.add_argument("--skip-linear-json", action="store_true", help="Only export ONNX artifacts; useful for non-linear sklearn models.")
    parser.add_argument("--onnx-output", help="Optional ONNX model path. Requires skl2onnx and onnx.")
    parser.add_argument("--onnx-json-output", help="Optional ONNX selector JSON path. Defaults next to --onnx-output.")
    parser.add_argument("--onnx-input-name", default="features", help="ONNX feature tensor input name.")
    parser.add_argument("--onnx-target-opset", type=int, default=17, help="ONNX target opset for skl2onnx conversion.")
    parser.add_argument("--onnx-execution-provider", default="cpu", choices=["cpu", "cuda"], help="Runtime provider to request in the ONNX selector JSON.")
    parser.add_argument("--onnx-device-id", type=int, default=0, help="CUDA device id for ONNX Runtime CUDA execution provider.")
    args = parser.parse_args()

    model = joblib.load(args.model)
    with Path(args.metadata).open() as file:
        metadata = json.load(file)

    feature_names = metadata["feature_names"]
    if args.skip_linear_json and not args.onnx_output:
        raise ValueError("--skip-linear-json requires --onnx-output")

    if not args.skip_linear_json:
        coefficients, intercept = extract_linear_model(model)
        if len(coefficients) != len(feature_names):
            raise ValueError(f"Model has {len(coefficients)} coefficients but metadata has {len(feature_names)} features")

        payload = {
            "model_type": "linear_score_ranker",
            "source_model": metadata.get("selected_model", ""),
            "training_formulation": metadata.get("training_formulation", "score_prediction_ranking"),
            "feature_names": feature_names,
            "coefficients": coefficients,
            "intercept": intercept,
            "candidate_schemas": candidate_schemas_from_metadata(metadata),
            "report": metadata.get("report", ""),
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as file:
            json.dump(payload, file, indent=2)

        print(f"Exported {payload['model_type']} with {len(feature_names)} features to {output_path}")
    if args.onnx_output:
        export_onnx_model(
            model,
            metadata,
            args.onnx_output,
            args.onnx_input_name,
            args.onnx_target_opset,
            args.onnx_execution_provider,
            args.onnx_device_id,
            args.onnx_json_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

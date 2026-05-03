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


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a trained linear score ranker to C++ JSON.")
    parser.add_argument("--model", default="models/schema_selector.joblib", help="Joblib model from train_schema_selector.py.")
    parser.add_argument("--metadata", default="models/schema_selector_meta.json", help="Metadata JSON from train_schema_selector.py.")
    parser.add_argument("--output", default="models/schema_selector.json", help="C++ runtime model JSON.")
    args = parser.parse_args()

    model = joblib.load(args.model)
    with Path(args.metadata).open() as file:
        metadata = json.load(file)

    coefficients, intercept = extract_linear_model(model)
    feature_names = metadata["feature_names"]
    if len(coefficients) != len(feature_names):
        raise ValueError(f"Model has {len(coefficients)} coefficients but metadata has {len(feature_names)} features")

    candidate_schemas = metadata.get("candidate_schema_paths", [])
    if not candidate_schemas:
        candidate_schemas = [{"name": name, "path": ""} for name in metadata.get("candidate_schemas", [])]

    payload = {
        "model_type": "linear_score_ranker",
        "source_model": metadata.get("selected_model", ""),
        "training_formulation": metadata.get("training_formulation", "score_prediction_ranking"),
        "feature_names": feature_names,
        "coefficients": coefficients,
        "intercept": intercept,
        "candidate_schemas": candidate_schemas,
        "report": metadata.get("report", ""),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(payload, file, indent=2)

    print(f"Exported {payload['model_type']} with {len(feature_names)} features to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

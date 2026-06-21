import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


POINT_FEATURE_COLUMNS = [
    "num_points",
    "feature_sample_size",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "aspect_xy",
    "aspect_xz",
    "aspect_yz",
    "density_bbox",
    "height_mean",
    "height_std",
    "height_range",
    "cov_eig_0",
    "cov_eig_1",
    "cov_eig_2",
    "linearity",
    "planarity",
    "scattering",
    "occupancy_ratio_8",
    "occupancy_entropy_8",
    "density_cv_8",
    "verticality_score",
    "flatness_score",
]

WORKLOAD_FEATURE_COLUMNS = [
    "w_range",
    "w_radius",
    "w_knn",
    "knn_k",
    "num_queries",
    "range_scale_min",
    "range_scale_max",
    "radius_scale_min",
    "radius_scale_max",
    "query_scale_mean",
    "query_scale_std",
    "build_weight",
    "memory_weight",
]

TREE_HEALTH_FEATURE_COLUMNS = [
    "leaf_occupancy_p50",
    "leaf_occupancy_p90",
    "leaf_occupancy_p99",
    "avg_depth",
    "avg_fanout",
    "max_fanout",
    "empty_child_ratio",
    "single_child_nodes",
    "mean_tight_bounds_volume_ratio",
    "micro_indexed_leaves",
    "micro_indexed_points",
]

SCHEMA_FEATURE_COLUMNS = [
    "schema_has_quadtree",
    "schema_has_octree",
    "schema_has_kdtree",
    "schema_has_grid2d",
    "schema_has_grid3d",
    "schema_num_blocks",
    "schema_total_levels",
    "schema_max_leaf_capacity",
    "schema_min_leaf_capacity",
]

DEFAULT_FIXED_BASELINES = ["quadtree", "octree", "kdtree", "urban"]


def to_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def normalize_schema_token(value):
    return "".join(ch for ch in value.lower() if ch.isalnum())


def resolve_schema_path(repo_root, schema_path):
    path = Path(schema_path)
    if path.is_absolute():
        return path
    return repo_root / path


def schema_features_from_json(repo_root, schema_path, schema_name):
    features = dict.fromkeys(SCHEMA_FEATURE_COLUMNS, 0.0)
    features["schema_min_leaf_capacity"] = 0.0

    path = resolve_schema_path(repo_root, schema_path)
    levels = []
    if path.exists():
        with path.open() as file:
            payload = json.load(file)
        levels = payload.get("levels", [])
    else:
        tokens = normalize_schema_token(f"{schema_name} {schema_path}")
        guessed = []
        for marker, typename in [
            ("quadtree", "QuadTree"),
            ("octree", "Octree"),
            ("kdtree", "KDTree"),
            ("grid2d", "Grid2D"),
            ("grid3d", "Grid3D"),
        ]:
            if marker in tokens:
                guessed.append({"type": typename, "numLevels": 1, "leafCapacity": 0})
        levels = guessed

    leaf_capacities = []
    for level in levels:
        level_type = normalize_schema_token(str(level.get("type", "")))
        if "quadtree" in level_type:
            features["schema_has_quadtree"] = 1.0
        if "octree" in level_type:
            features["schema_has_octree"] = 1.0
        if "kdtree" in level_type:
            features["schema_has_kdtree"] = 1.0
        if "grid2d" in level_type:
            features["schema_has_grid2d"] = 1.0
        if "grid3d" in level_type:
            features["schema_has_grid3d"] = 1.0

        features["schema_total_levels"] += float(level.get("numLevels", 1) or 0)
        capacity = float(level.get("leafCapacity", 0) or 0)
        if capacity > 0:
            leaf_capacities.append(capacity)

    features["schema_num_blocks"] = float(len(levels))
    if leaf_capacities:
        features["schema_max_leaf_capacity"] = max(leaf_capacities)
        features["schema_min_leaf_capacity"] = min(leaf_capacities)
    return features


def load_rows(csv_path, repo_root):
    with Path(csv_path).open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    schema_feature_cache = {}
    for row in rows:
        key = (row.get("schema_path", ""), row.get("schema_name", ""))
        if key not in schema_feature_cache:
            schema_feature_cache[key] = schema_features_from_json(repo_root, key[0], key[1])
        row["_schema_features"] = schema_feature_cache[key]
        row["_score"] = to_float(row, "score")
        row["_group"] = (row["dataset_name"], row["workload_name"])

    return rows


def available_feature_columns(rows):
    columns = []
    for column in POINT_FEATURE_COLUMNS + WORKLOAD_FEATURE_COLUMNS + TREE_HEALTH_FEATURE_COLUMNS:
        if column in rows[0]:
            columns.append(column)
    return columns


def row_features(row, data_feature_columns):
    values = [to_float(row, column) for column in data_feature_columns]
    values.extend(row["_schema_features"][column] for column in SCHEMA_FEATURE_COLUMNS)
    return values


def rows_to_matrix(rows, data_feature_columns):
    return np.array([row_features(row, data_feature_columns) for row in rows], dtype=float)


def rows_to_scores(rows):
    return np.array([row["_score"] for row in rows], dtype=float)


def grouped_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["_group"]].append(row)
    return dict(groups)


def oracle_row(group_rows):
    return min(group_rows, key=lambda row: row["_score"])


def split_by_dataset(rows, test_fraction, seed):
    datasets = sorted({row["dataset_name"] for row in rows})
    if len(datasets) <= 1:
        return rows, rows, datasets, datasets

    rng = random.Random(seed)
    shuffled = datasets[:]
    rng.shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_fraction))
    test_datasets = set(shuffled[:test_count])
    train_rows = [row for row in rows if row["dataset_name"] not in test_datasets]
    test_rows = [row for row in rows if row["dataset_name"] in test_datasets]
    return train_rows, test_rows, sorted(set(datasets) - test_datasets), sorted(test_datasets)


def schema_matches(row, baseline):
    value = normalize_schema_token(f"{row.get('schema_name', '')} {row.get('schema_path', '')}")
    if baseline == "quadtree":
        return "quadtree" in value and "octree" not in value and "kdtree" not in value
    if baseline == "octree":
        return "octree" in value and "quadtree" not in value and "kdtree" not in value
    if baseline == "kdtree":
        return "kdtree" in value and "octree" not in value and "quadtree" not in value
    if baseline == "urban":
        return "urban" in value
    return baseline in value


def select_fixed(group_rows, baseline):
    for row in group_rows:
        if schema_matches(row, baseline):
            return row
    return group_rows[0]


def majority_schema(train_rows):
    winners = [oracle_row(group)["schema_name"] for group in grouped_rows(train_rows).values()]
    if not winners:
        return ""
    return Counter(winners).most_common(1)[0][0]


def select_majority(group_rows, schema_name):
    for row in group_rows:
        if row["schema_name"] == schema_name:
            return row
    return group_rows[0]


def select_heuristic(group_rows):
    reference = group_rows[0]
    flatness = to_float(reference, "flatness_score")
    scattering = to_float(reference, "scattering")
    verticality = to_float(reference, "verticality_score")

    if flatness >= 0.45 and flatness >= verticality:
        preferences = ["quadtree", "octree", "urban", "kdtree"]
    elif scattering >= 0.18:
        preferences = ["octree", "urban", "kdtree", "quadtree"]
    elif verticality >= 0.35:
        preferences = ["urban", "octree", "kdtree", "quadtree"]
    else:
        preferences = ["kdtree", "urban", "octree", "quadtree"]

    for preference in preferences:
        for row in group_rows:
            if schema_matches(row, preference):
                return row
    return group_rows[0]


def evaluate_selector(groups, selector):
    regrets = []
    relative_regrets = []
    selected_scores = []
    oracle_scores = []
    build_times = []
    memory_bytes = []
    accuracy = 0
    top2_accuracy = 0
    count = 0

    selections = []
    for key, candidates in groups.items():
        ranked = selector(candidates)
        if not isinstance(ranked, list):
            ranked = [ranked]
        selected = ranked[0]
        oracle = oracle_row(candidates)
        oracle_score = oracle["_score"]
        selected_score = selected["_score"]
        regret = selected_score - oracle_score
        relative_regret = selected_score / oracle_score - 1.0 if abs(oracle_score) > 1e-12 else 0.0

        regrets.append(regret)
        relative_regrets.append(relative_regret)
        selected_scores.append(selected_score)
        oracle_scores.append(oracle_score)
        build_times.append(to_float(selected, "build_time_ms"))
        memory_bytes.append(to_float(selected, "memory_estimate_bytes"))
        accuracy += int(selected["schema_name"] == oracle["schema_name"])
        top2_accuracy += int(any(row["schema_name"] == oracle["schema_name"] for row in ranked[:2]))
        count += 1
        selections.append(
            {
                "dataset_name": key[0],
                "workload_name": key[1],
                "selected_schema": selected["schema_name"],
                "oracle_schema": oracle["schema_name"],
                "selected_score": selected_score,
                "oracle_score": oracle_score,
                "regret": regret,
                "relative_regret": relative_regret,
            }
        )

    if count == 0:
        return {}

    return {
        "groups": count,
        "best_schema_accuracy": accuracy / count,
        "top2_accuracy": top2_accuracy / count,
        "mean_regret": float(np.mean(regrets)),
        "mean_relative_regret": float(np.mean(relative_regrets)),
        "mean_selected_score": float(np.mean(selected_scores)),
        "mean_oracle_score": float(np.mean(oracle_scores)),
        "mean_selected_build_time_ms": float(np.mean(build_times)),
        "mean_selected_memory_bytes": float(np.mean(memory_bytes)),
        "selections": selections,
    }


def learned_selector(model, data_feature_columns):
    def select(candidates):
        x = rows_to_matrix(candidates, data_feature_columns)
        predictions = model.predict(x)
        ranked = sorted(zip(candidates, predictions), key=lambda item: item[1])
        return [row for row, _ in ranked]

    return select


def classifier_selector(model, data_feature_columns):
    def select(candidates):
        reference = candidates[0]
        features = np.array([[to_float(reference, column) for column in data_feature_columns]], dtype=float)
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            probabilities = model.predict_proba(features)[0]
            class_rank = [schema for schema, _ in sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)]
        else:
            class_rank = [model.predict(features)[0]]

        ranked = []
        remaining = candidates[:]
        for schema_name in class_rank:
            for row in list(remaining):
                if row["schema_name"] == schema_name:
                    ranked.append(row)
                    remaining.remove(row)
                    break
        ranked.extend(remaining)
        return ranked

    return select


def train_regression_models(train_rows, data_feature_columns, seed):
    x_train = rows_to_matrix(train_rows, data_feature_columns)
    y_train = rows_to_scores(train_rows)
    return {
        "ridge_score_predictor": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest_score_predictor": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=1,
            random_state=seed,
        ),
        "gradient_boosted_score_predictor": GradientBoostingRegressor(random_state=seed),
    }, x_train, y_train


def train_logistic_classifier(train_rows, data_feature_columns, seed):
    train_groups = grouped_rows(train_rows)
    group_rows = [oracle_row(rows) for rows in train_groups.values()]
    labels = [row["schema_name"] for row in group_rows]
    if len(set(labels)) < 2:
        return None
    x_train = np.array([[to_float(row, column) for column in data_feature_columns] for row in group_rows], dtype=float)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    model.fit(x_train, labels)
    return model


def feature_importance(model, feature_names):
    candidate = model
    if hasattr(model, "named_steps"):
        candidate = list(model.named_steps.values())[-1]

    if hasattr(candidate, "feature_importances_"):
        importances = candidate.feature_importances_
    elif hasattr(candidate, "coef_"):
        coef = np.asarray(candidate.coef_)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
    else:
        return []

    ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    return [{"feature": name, "importance": float(value)} for name, value in ranked[:20]]


def add_speedups(report):
    learned = report["learned_selectors"][report["selected_model"]]
    learned_score = learned["mean_selected_score"]
    for name, metrics in report["baselines"].items():
        baseline_score = metrics["mean_selected_score"]
        metrics["score_speedup_vs_selected_model"] = baseline_score / learned_score if learned_score > 0 else math.inf
    for name, metrics in report["learned_selectors"].items():
        metrics["score_speedup_vs_selected_model"] = learned_score / metrics["mean_selected_score"] if metrics["mean_selected_score"] > 0 else math.inf


def write_model_metadata(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an offline schema score predictor from schema-search CSV rows.")
    parser.add_argument("--input", default="results/schema_search.csv", help="Raw schema-search CSV.")
    parser.add_argument("--report", default="results/schema_selector_report.json", help="Training/evaluation report JSON.")
    parser.add_argument("--model-output", default="models/schema_selector.joblib", help="Learned selector model artifact.")
    parser.add_argument("--model-meta-output", default="models/schema_selector_meta.json", help="Model metadata JSON.")
    parser.add_argument("--learning-mode", choices=["all", "regression", "classification"], default="all", help="Train score-ranking models, direct best-schema classifier, or both.")
    parser.add_argument("--model-name", help="Explicit learned selector to save, e.g. ridge_score_predictor for C++ JSON export.")
    parser.add_argument("--test-fraction", type=float, default=0.34, help="Fraction of datasets held out for evaluation.")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic split/model seed.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rows = load_rows(args.input, repo_root)
    data_feature_columns = available_feature_columns(rows)
    feature_names = data_feature_columns + SCHEMA_FEATURE_COLUMNS
    train_rows, test_rows, train_datasets, test_datasets = split_by_dataset(rows, args.test_fraction, args.seed)
    test_groups = grouped_rows(test_rows)

    learned_metrics = {}
    fitted_models = {}

    if args.learning_mode in ("all", "regression"):
        regression_models, x_train, y_train = train_regression_models(train_rows, data_feature_columns, args.seed)
        for name, model in regression_models.items():
            model.fit(x_train, y_train)
            fitted_models[name] = model
            learned_metrics[name] = evaluate_selector(test_groups, learned_selector(model, data_feature_columns))
            learned_metrics[name]["feature_importance"] = feature_importance(model, feature_names)

    if args.learning_mode in ("all", "classification"):
        logistic_model = train_logistic_classifier(train_rows, data_feature_columns, args.seed)
        if logistic_model is not None:
            fitted_models["logistic_best_schema_classifier"] = logistic_model
            learned_metrics["logistic_best_schema_classifier"] = evaluate_selector(
                test_groups,
                classifier_selector(logistic_model, data_feature_columns),
            )
            learned_metrics["logistic_best_schema_classifier"]["feature_importance"] = feature_importance(logistic_model, data_feature_columns)

    if not learned_metrics:
        raise RuntimeError("No learned selector could be trained. Try --learning-mode regression or add more labeled datasets.")

    baselines = {
        "oracle": evaluate_selector(test_groups, lambda candidates: oracle_row(candidates)),
        "heuristic": evaluate_selector(test_groups, select_heuristic),
    }
    majority = majority_schema(train_rows)
    baselines["majority_train_oracle_schema"] = evaluate_selector(test_groups, lambda candidates: select_majority(candidates, majority))
    for baseline in DEFAULT_FIXED_BASELINES:
        baselines[f"fixed_{baseline}"] = evaluate_selector(test_groups, lambda candidates, name=baseline: select_fixed(candidates, name))

    if args.model_name:
        if args.model_name not in learned_metrics:
            raise RuntimeError(f"Requested --model-name {args.model_name} was not trained")
        selected_model = args.model_name
    else:
        selected_model = min(
            learned_metrics,
            key=lambda name: (
                learned_metrics[name]["mean_relative_regret"],
                learned_metrics[name]["mean_selected_score"],
            ),
        )

    report = {
        "input_csv": str(args.input),
        "training_formulation": "score_prediction_ranking",
        "requested_learning_mode": args.learning_mode,
        "concept": (
            "Static schemas are candidate measurements. The learned model predicts score from "
            "cloud/workload/schema-composition features so future generated combinations can be ranked."
        ),
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "num_rows": len(rows),
        "num_train_rows": len(train_rows),
        "num_test_rows": len(test_rows),
        "num_eval_groups": len(test_groups),
        "data_feature_columns": data_feature_columns,
        "schema_feature_columns": SCHEMA_FEATURE_COLUMNS,
        "selected_model": selected_model,
        "requested_model_name": args.model_name or "",
        "baselines": baselines,
        "learned_selectors": learned_metrics,
    }
    add_speedups(report)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as file:
        json.dump(report, file, indent=2)

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models[selected_model], model_path)

    metadata = {
        "selected_model": selected_model,
        "model_artifact": str(model_path),
        "training_formulation": report["training_formulation"],
        "data_feature_columns": data_feature_columns,
        "schema_feature_columns": SCHEMA_FEATURE_COLUMNS,
        "feature_names": feature_names,
        "candidate_schemas": sorted({row["schema_name"] for row in rows}),
        "candidate_schema_paths": [
            {"name": name, "path": path}
            for name, path in sorted({(row["schema_name"], row["schema_path"]) for row in rows})
        ],
        "report": str(report_path),
    }
    write_model_metadata(Path(args.model_meta_output), metadata)

    selected_metrics = report["learned_selectors"][selected_model]
    print(f"Rows: {len(rows)}; train rows: {len(train_rows)}; test rows: {len(test_rows)}")
    print(f"Held-out datasets: {', '.join(test_datasets)}")
    print(
        f"Selected model: {selected_model}; "
        f"accuracy={selected_metrics['best_schema_accuracy']:.3f}; "
        f"top2={selected_metrics['top2_accuracy']:.3f}; "
        f"mean_relative_regret={selected_metrics['mean_relative_regret']:.6f}"
    )
    print(f"Report: {report_path}")
    print(f"Model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

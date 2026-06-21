import argparse
import csv
import json
import subprocess
from pathlib import Path


DEFAULT_SCHEMAS = [
    "configs/schemas/quadtree.json",
    "configs/schemas/octree.json",
    "configs/schemas/kdtree.json",
    "configs/schemas/quadtree_octree.json",
    "configs/schemas/octree_kdtree.json",
    "configs/schemas/urban_hybrid.json",
]


def csv_safe_name(path):
    return Path(path).stem or "point_cloud"


def load_best_row(csv_path):
    rows = []
    with Path(csv_path).open(newline="") as file:
        for row in csv.DictReader(file):
            row["_score"] = float(row["score"])
            rows.append(row)

    if not rows:
        raise ValueError(f"No schema-search rows found in {csv_path}")

    selection_rows = [row for row in rows if "_robust_seed" not in row.get("workload_name", "")]
    if not selection_rows:
        selection_rows = rows
    selection_rows.sort(key=lambda row: (row["_score"], row["schema_name"]))
    return selection_rows[0], rows


def export_measured_selector(output_path, input_path, workload_path, best_row, rows, source_csv):
    payload = {
        "model_type": "measured_best_schema",
        "dataset_name": best_row["dataset_name"],
        "dataset_path": input_path,
        "workload_name": best_row["workload_name"],
        "workload_profile_path": workload_path,
        "selected_schema": {
            "name": best_row["schema_name"],
            "path": best_row["schema_path"],
            "score": best_row["_score"],
            "score_mode": best_row.get("score_mode", "unknown"),
            "score_stage": best_row.get("score_stage", "unknown"),
            "score_is_final_latency": best_row.get("score_is_final_latency", "unknown"),
            "avg_latency_ms": float(best_row.get("avg_latency_ms", 0.0)),
            "build_time_ms": float(best_row.get("build_time_ms", 0.0)),
            "memory_estimate_bytes": float(best_row.get("memory_estimate_bytes", 0.0)),
        },
        "candidate_scores": [
            {
                "name": row["schema_name"],
                "path": row["schema_path"],
                "score": row["_score"],
                "score_mode": row.get("score_mode", "unknown"),
                "score_stage": row.get("score_stage", "unknown"),
                "score_is_final_latency": row.get("score_is_final_latency", "unknown"),
                "avg_latency_ms": float(row.get("avg_latency_ms", 0.0)),
                "build_time_ms": float(row.get("build_time_ms", 0.0)),
                "memory_estimate_bytes": float(row.get("memory_estimate_bytes", 0.0)),
            }
            for row in rows
        ],
        "source_csv": str(source_csv),
        "note": "Local measured selector: overfit to this point cloud and workload by benchmarking candidate schemas.",
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(payload, file, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune schema choice for one point cloud by measured candidate benchmarking.")
    parser.add_argument("--exe", default="x64/Release/MultiDataStructure.exe", help="Path to the built executable.")
    parser.add_argument("--input", required=True, help="Point cloud path to tune for.")
    parser.add_argument("--schemas", nargs="*", default=DEFAULT_SCHEMAS, help="Candidate schema JSON files.")
    parser.add_argument("--workload-profile", default="configs/workloads/volume_small_medium.json", help="Single workload profile to optimize.")
    parser.add_argument("--queries", type=int, default=64, help="Queries for measured schema-search tuning.")
    parser.add_argument("--csv", help="Raw local tuning CSV path.")
    parser.add_argument("--best-csv", help="Best-row CSV path from schema-search.")
    parser.add_argument("--output", default="models/local_schema_selector.json", help="Local measured selector JSON for --schema auto.")
    parser.add_argument("--no-cache", action="store_true", help="Disable point-cloud cache use during tuning.")
    parser.add_argument("--auto-conditions", action="store_true", help="Tune conditional schema thresholds with staged proxy/full measurements.")
    parser.add_argument("--deep-nested-search", action="store_true", help="Run CPU-first deep nested search against single-DS baselines, with CUDA confirmation when available.")
    parser.add_argument("--adaptive-leaf-capacity", action="store_true", help="Allow generated schemas to tune per-node adaptive leaf capacity.")
    parser.add_argument("--adaptive-leaf-probability", type=float, help="Probability that a generated level receives adaptive leaf-capacity rules.")
    parser.add_argument("--condition-proxy-candidates", type=int, help="Candidates for the auto-condition proxy stage.")
    parser.add_argument("--condition-proxy-points", type=int, help="Point cap for the auto-condition proxy stage.")
    parser.add_argument("--condition-proxy-queries", type=int, help="Queries for the auto-condition proxy stage.")
    parser.add_argument("--condition-final-top", type=int, help="Full-cloud candidates after proxy pruning.")
    parser.add_argument("--condition-confirm-top", type=int, help="Confirmation candidates after short full-cloud pruning.")
    parser.add_argument("--condition-output-dir", help="Directory for tuned schema JSON artifacts.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    exe = Path(args.exe)
    if not exe.is_absolute():
        exe = repo_root / exe

    dataset_name = csv_safe_name(args.input)
    csv_path = Path(args.csv or f"results/schema_search_{dataset_name}_local.csv")
    best_csv_path = Path(args.best_csv or f"results/schema_search_best_{dataset_name}_local.csv")

    command = [
        str(exe),
        "--mode",
        "schema-search",
        "--input",
        args.input,
        "--schemas",
        ";".join(args.schemas),
        "--workloads",
        args.workload_profile,
        "--queries",
        str(args.queries),
        "--csv",
        str(csv_path),
        "--best-csv",
        str(best_csv_path),
        "--no-synthetic",
        "--no-pause",
    ]
    if args.no_cache:
        command.append("--no-cache")
    if args.deep_nested_search:
        command.append("--deep-nested-search")
    if args.adaptive_leaf_capacity:
        command.append("--generated-adaptive-leaf-capacity")
    if args.adaptive_leaf_probability is not None:
        command.extend(["--generated-adaptive-leaf-probability", str(args.adaptive_leaf_probability)])
    if args.auto_conditions or args.deep_nested_search:
        command.append("--auto-conditions")
        optional_condition_args = [
            ("--condition-proxy-candidates", args.condition_proxy_candidates),
            ("--condition-proxy-points", args.condition_proxy_points),
            ("--condition-proxy-queries", args.condition_proxy_queries),
            ("--condition-final-top", args.condition_final_top),
            ("--condition-confirm-top", args.condition_confirm_top),
            ("--condition-output-dir", args.condition_output_dir),
        ]
        for flag, value in optional_condition_args:
            if value is not None:
                command.extend([flag, str(value)])
        command.extend(["--condition-selector-output", args.output])

    subprocess.run(command, cwd=repo_root, check=True)

    best_row, rows = load_best_row(repo_root / csv_path if not csv_path.is_absolute() else csv_path)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    export_measured_selector(output_path, args.input, args.workload_profile, best_row, rows, csv_path)

    print(f"Selected measured schema: {best_row['schema_name']} score={best_row['_score']:.6f}")
    print(f"Local selector: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

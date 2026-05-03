import argparse
import csv
from pathlib import Path


FEATURE_COLUMNS = [
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
    "w_range",
    "w_radius",
    "w_knn",
    "knn_k",
    "num_queries",
    "query_scale_mean",
    "query_scale_std",
    "build_weight",
    "memory_weight",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract best schema rows from a schema-search CSV.")
    parser.add_argument("--input", default="results/schema_search.csv", help="Schema-search CSV produced by --mode schema-search.")
    parser.add_argument("--output", default="results/schema_search_best.csv", help="Best-schema CSV to write.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    best_by_pair = {}
    candidate_counts = {}
    with input_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = (row["dataset_name"], row["workload_name"])
            candidate_counts[key] = candidate_counts.get(key, 0) + 1
            score = float(row["score"])
            current = best_by_pair.get(key)
            if current is None or score < float(current["score"]):
                best_by_pair[key] = row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    present_feature_columns = [
        column for column in FEATURE_COLUMNS if best_by_pair and column in next(iter(best_by_pair.values()))
    ]

    with output_path.open("w", newline="") as file:
        fieldnames = [
            "dataset_name",
            "workload_name",
            *present_feature_columns,
            "best_schema_name",
            "best_schema_path",
            "best_score",
            "best_avg_latency_ms",
            "best_build_time_ms",
            "best_memory_estimate_bytes",
            "num_candidates",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for key, row in best_by_pair.items():
            writer.writerow(
                {
                    "dataset_name": row["dataset_name"],
                    "workload_name": row["workload_name"],
                    **{column: row.get(column, "") for column in present_feature_columns},
                    "best_schema_name": row["schema_name"],
                    "best_schema_path": row["schema_path"],
                    "best_score": row["score"],
                    "best_avg_latency_ms": row["avg_latency_ms"],
                    "best_build_time_ms": row["build_time_ms"],
                    "best_memory_estimate_bytes": row["memory_estimate_bytes"],
                    "num_candidates": candidate_counts[key],
                }
            )

    print(f"Wrote {len(best_by_pair)} best-schema rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

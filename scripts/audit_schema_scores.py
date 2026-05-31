import argparse
import csv
from collections import defaultdict
from pathlib import Path


UNKNOWN = "unknown"


def first_present(row, names, default=""):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def as_float(value, default=float("inf")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int_text(value):
    if value in (None, ""):
        return UNKNOWN
    try:
        return str(int(float(value)))
    except ValueError:
        return str(value)


def infer_score_mode(row):
    mode = first_present(row, ["score_mode"], "")
    if mode:
        return mode
    uses_proxy = first_present(row, ["score_uses_visit_proxy"], "")
    if uses_proxy in ("1", "true", "True"):
        return "visit_proxy"
    if uses_proxy in ("0", "false", "False"):
        return "latency"
    return UNKNOWN


def normalize_row(row, source_label):
    schema_name = first_present(row, ["schema_name", "best_schema_name"], UNKNOWN)
    schema_path = first_present(row, ["schema_path", "best_schema_path"], "")
    score = first_present(row, ["score", "best_score"], "")
    latency = first_present(row, ["avg_latency_ms", "best_avg_latency_ms", "latency_mean_ms"], "")
    build = first_present(row, ["build_time_ms", "best_build_time_ms"], "")
    memory = first_present(row, ["memory_estimate_bytes", "best_memory_estimate_bytes"], "")
    total_queries = first_present(row, ["effective_queries", "total_queries", "num_queries"], "")
    evaluator = first_present(row, ["backend", "evaluator"], UNKNOWN)

    return {
        "source": source_label,
        "dataset": first_present(row, ["dataset_name"], UNKNOWN),
        "workload": first_present(row, ["workload_name"], UNKNOWN),
        "schema_name": schema_name,
        "schema_path": schema_path,
        "schema_key": schema_path or schema_name,
        "score": as_float(score),
        "latency": as_float(latency),
        "build": as_float(build),
        "memory": as_float(memory),
        "score_mode": infer_score_mode(row),
        "score_stage": first_present(row, ["score_stage"], UNKNOWN),
        "score_is_final_latency": first_present(row, ["score_is_final_latency"], UNKNOWN),
        "effective_queries": as_int_text(total_queries),
        "query_seed": as_int_text(first_present(row, ["query_seed"], "")),
        "evaluator": evaluator,
        "cuda_builder": first_present(row, ["cuda_builder"], ""),
        "lambda_latency": first_present(row, ["lambda_latency", "score_latency_weight"], UNKNOWN),
        "lambda_build": first_present(row, ["lambda_build", "score_build_weight"], UNKNOWN),
        "lambda_memory": first_present(row, ["lambda_memory", "score_memory_weight"], UNKNOWN),
        "lambda_imbalance": first_present(row, ["lambda_imbalance", "score_imbalance_weight"], UNKNOWN),
        "visit_proxy_alpha": first_present(row, ["visit_proxy_alpha"], UNKNOWN),
    }


def load_rows(path, label):
    with Path(path).open(newline="") as file:
        return [normalize_row(row, label) for row in csv.DictReader(file)]


def best_by(rows, metric):
    finite = [row for row in rows if row[metric] != float("inf")]
    if not finite:
        return None
    return min(finite, key=lambda row: (row[metric], row["schema_name"]))


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["workload"])].append(row)
    return groups


def comparable_fingerprint(row):
    return {
        "score_mode": row["score_mode"],
        "score_stage": row["score_stage"],
        "score_is_final_latency": row["score_is_final_latency"],
        "effective_queries": row["effective_queries"],
        "query_seed": row["query_seed"],
        "evaluator": row["evaluator"],
        "cuda_builder": row["cuda_builder"],
        "lambda_latency": row["lambda_latency"],
        "lambda_build": row["lambda_build"],
        "lambda_memory": row["lambda_memory"],
        "lambda_imbalance": row["lambda_imbalance"],
        "visit_proxy_alpha": row["visit_proxy_alpha"],
    }


def compare_best(left_groups, right_groups):
    mismatches = []
    reports = []
    all_keys = sorted(set(left_groups) | set(right_groups))
    for key in all_keys:
        left_rows = left_groups.get(key, [])
        right_rows = right_groups.get(key, [])
        left_score = best_by(left_rows, "score")
        right_score = best_by(right_rows, "score")
        left_latency = best_by(left_rows, "latency")
        right_latency = best_by(right_rows, "latency")

        reports.append((key, left_score, right_score, left_latency, right_latency))

        if not left_score or not right_score:
            mismatches.append((key, "missing rows in one input"))
            continue

        left_fp = comparable_fingerprint(left_score)
        right_fp = comparable_fingerprint(right_score)
        for field in sorted(left_fp):
            if UNKNOWN in (left_fp[field], right_fp[field]):
                continue
            if left_fp[field] != right_fp[field]:
                mismatches.append((key, f"{field}: {left_fp[field]} vs {right_fp[field]}"))

    return reports, mismatches


def format_best(row, metric):
    if row is None:
        return "missing"
    value = row[metric]
    value_text = "unknown" if value == float("inf") else f"{value:.6f}"
    return (
        f"{row['schema_name']} {metric}={value_text} "
        f"score_mode={row['score_mode']} stage={row['score_stage']} "
        f"queries={row['effective_queries']} evaluator={row['evaluator']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Audit whether two schema-search CSVs are comparable and summarize score-vs-latency winners."
    )
    parser.add_argument("left_csv", help="First schema-search CSV, for example non-GA.")
    parser.add_argument("right_csv", help="Second schema-search CSV, for example GA.")
    parser.add_argument("--left-label", default="left", help="Label for the first CSV.")
    parser.add_argument("--right-label", default="right", help="Label for the second CSV.")
    parser.add_argument("--fail-on-mismatch", action="store_true", help="Exit with code 2 when comparable settings differ.")
    args = parser.parse_args()

    left_rows = load_rows(args.left_csv, args.left_label)
    right_rows = load_rows(args.right_csv, args.right_label)
    reports, mismatches = compare_best(group_rows(left_rows), group_rows(right_rows))

    print(f"Loaded {len(left_rows)} rows from {args.left_label}: {args.left_csv}")
    print(f"Loaded {len(right_rows)} rows from {args.right_label}: {args.right_csv}")
    print()

    for (dataset, workload), left_score, right_score, left_latency, right_latency in reports:
        print(f"[{dataset} / {workload}]")
        print(f"  {args.left_label} best score:   {format_best(left_score, 'score')}")
        print(f"  {args.left_label} best latency: {format_best(left_latency, 'latency')}")
        print(f"  {args.right_label} best score:   {format_best(right_score, 'score')}")
        print(f"  {args.right_label} best latency: {format_best(right_latency, 'latency')}")
        print()

    if mismatches:
        print("Potential apples-to-oranges comparisons:")
        for (dataset, workload), message in mismatches:
            print(f"  - {dataset} / {workload}: {message}")
    else:
        print("No concrete provenance mismatches detected. Unknown historical fields were ignored.")

    return 2 if args.fail_on_mismatch and mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

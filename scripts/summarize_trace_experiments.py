#!/usr/bin/env python3
"""Summarize the trace-driven experiments (results/eval_traces) into one report.

Reads the schema-search CSVs of the pipeline-replay runs and the batch-search CSVs of
the DL runs, and prints per dataset: the best searched schema, the best single-primitive
baseline evaluated under identical queries, the delta, and the amortization estimate
(GA search wall cost vs projected savings per full application pass, using the
full_query_count recorded in each trace sidecar).

Usage: python scripts/summarize_trace_experiments.py [--dir results/eval_traces]
"""

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def summarize_pipeline(csv_path: Path, sidecar_path: Path):
    rows = read_csv(csv_path)
    if not rows:
        return None

    def as_float(row, key, default=float("inf")):
        try:
            return float(row.get(key, ""))
        except ValueError:
            return default

    # Keep final latency measurements only (skip proxy-stage rows).
    final_rows = [r for r in rows if r.get("score_is_final_latency", "1") in ("1", "true", "True", "")]
    if not final_rows:
        final_rows = rows

    ranked = sorted(final_rows, key=lambda r: as_float(r, "score"))
    best = ranked[0]
    baselines = [r for r in final_rows if r.get("is_baseline", "0") in ("1", "true", "True")]
    best_baseline = min(baselines, key=lambda r: as_float(r, "score")) if baselines else None

    summary = {
        "csv": csv_path.name,
        "rows": len(rows),
        "final_rows": len(final_rows),
        "best_schema": best.get("schema_name", "?"),
        "best_score": as_float(best, "score"),
        "best_avg_latency_ms": as_float(best, "avg_latency_ms", None),
        "best_build_ms": as_float(best, "build_time_ms", None),
    }
    if best_baseline is not None:
        summary["best_baseline"] = best_baseline.get("schema_name", "?")
        summary["baseline_score"] = as_float(best_baseline, "score")
        summary["baseline_avg_latency_ms"] = as_float(best_baseline, "avg_latency_ms", None)
        if summary["baseline_score"] > 0 and summary["best_score"] < float("inf"):
            summary["latency_delta_pct"] = 100.0 * (summary["baseline_score"] - summary["best_score"]) / summary["baseline_score"]

    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        summary["full_queries_per_pass"] = sum(s["full_query_count"] for s in sidecar.get("stages", []))
        summary["mean_spacing"] = sidecar.get("mean_spacing")
        # Amortization: projected per-pass saving from measured per-query latency delta.
        if summary.get("baseline_avg_latency_ms") and summary.get("best_avg_latency_ms") is not None:
            per_query_saving_ms = summary["baseline_avg_latency_ms"] - summary["best_avg_latency_ms"]
            summary["projected_saving_per_pass_s"] = per_query_saving_ms * summary["full_queries_per_pass"] / 1000.0
    return summary


def summarize_batch(csv_path: Path):
    rows = read_csv(csv_path)
    if not rows:
        return None
    for row in rows:
        row["pass_ms"] = float(row["pass_ms"])
    searched = [r for r in rows if r.get("tag") == "searched"]
    baselines = [r for r in rows if r.get("tag") == "baseline"]
    best = min(rows, key=lambda r: r["pass_ms"])
    summary = {
        "csv": csv_path.name,
        "rows": len(rows),
        "best_schema": best["schema"],
        "best_pass_ms": best["pass_ms"],
        "best_tag": best.get("tag"),
    }
    if baselines:
        bb = min(baselines, key=lambda r: r["pass_ms"])
        summary["best_baseline"] = bb["schema"]
        summary["baseline_pass_ms"] = bb["pass_ms"]
        summary["delta_pct"] = 100.0 * (bb["pass_ms"] - best["pass_ms"]) / bb["pass_ms"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("results/eval_traces"))
    parser.add_argument("--traces", type=Path, default=Path("results/traces"))
    args = parser.parse_args()

    print("=== Pipeline-replay schema search (one static index, dense self-queries) ===")
    for name in ("sanandreas_5m", "sanandreas_50m", "alhambra_100m"):
        csv_path = args.dir / f"{name}.csv"
        if not csv_path.exists():
            print(f"  {name}: (missing)")
            continue
        summary = summarize_pipeline(csv_path, args.traces / name / "pipeline_workload.json")
        print(f"  {name}: {json.dumps(summary, indent=4, default=str)}")

    print("\n=== Batch-distribution search (DL dataloader regime, per-pair rebuilds) ===")
    for name in ("dl_sanandreas_batch", "dl_alhambra_batch"):
        csv_path = args.dir / f"{name}.csv"
        if not csv_path.exists():
            print(f"  {name}: (missing)")
            continue
        print(f"  {name}: {json.dumps(summarize_batch(csv_path), indent=4, default=str)}")


if __name__ == "__main__":
    main()

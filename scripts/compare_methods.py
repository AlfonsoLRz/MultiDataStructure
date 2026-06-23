#!/usr/bin/env python3
"""Consolidate schema-search CSVs into one comparable evaluation.

Compares, per (dataset, workload): the best single primitive (engine baseline
controls), the best hand-designed nested config, and the best auto-tuned schema
(produced by --auto-conditions), against the per-cell oracle. Emits a markdown
table, optional plots, and a machine-readable summary.

Behavior-preserving analysis only: reads CSVs the executable already wrote; never
launches the binary. Tolerates both the rich live header and the older stale
header (missing columns degrade gracefully). matplotlib is optional.
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
from collections import defaultdict


# Default single-primitive configs (the "best single primitive" floor), by basename.
DEFAULT_SINGLES = [
    "quadtree.json", "octree.json", "kdtree.json", "bvh.json", "bih.json",
    "hgrid.json", "lbvh.json", "regular_grid.json", "karras_octree.json",
]

# Default hand-designed nested configs passed via --schemas (basenames).
DEFAULT_HANDDESIGNED = [
    "quadtree_octree.json",
    "octree_kdtree.json",
    "urban_hybrid.json",
    "grid2d_quadtree_grid3d_octree.json",
    "urban_grid_hybrid.json",
    "adaptive_quadtree_octree.json",
]

# Auto-tuned schemas (exported by --auto-conditions) live under a dir with this marker.
DEFAULT_AUTO_MARKER = "auto_schemas"

# Provenance fields that must be constant for rows to be apples-to-apples.
PROVENANCE_FIELDS = [
    "score_objective", "score_mode", "effective_queries", "query_seed",
    "cuda_builder", "lambda_latency", "lambda_build", "lambda_memory",
    "lambda_imbalance",
]

SINGLE = "single"
HANDDESIGNED = "nested_handdesigned"
AUTO = "auto_tuned"
OTHER = "other"


def to_float(value, default=math.inf):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def expand_inputs(patterns):
    """Glob the patterns, dropping *_best.csv / *_pareto.csv summary files."""
    paths = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            base = os.path.basename(path).lower()
            if base.endswith("_best.csv") or base.endswith("_pareto.csv"):
                continue
            if path not in paths:
                paths.append(path)
    return paths


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = os.path.basename(path)
                rows.append(row)
    return rows


def is_final_latency(row):
    """Keep only rows safe to compare as a final latency measurement."""
    if row.get("score_is_final_latency") not in (None, ""):
        if str(row["score_is_final_latency"]).strip() not in ("1", "true", "True"):
            return False
    if str(row.get("score_mode", "")).strip() == "visit_proxy":
        return False
    if str(row.get("score_uses_visit_proxy", "")).strip() in ("1", "true", "True"):
        return False
    return True


def latency_of(row):
    """Seed-averaged mean when confirmed across >=2 seeds, else single-run avg."""
    seeds = to_int(row.get("confirm_seeds_used"), 0)
    mean = to_float(row.get("latency_mean_ms"), 0.0)
    if seeds >= 2 and mean > 0.0:
        return mean
    return to_float(row.get("avg_latency_ms"))


def memory_bytes(row):
    return to_float(row.get("memory_estimate_bytes"), 0.0)


def gpu_status(row):
    return str(row.get("gpu_support_status", "")).strip() or "unknown"


def is_nested(row):
    return (to_float(row.get("active_structure_types"), 0.0) >= 2.0
            and to_float(row.get("nested_active_fraction"), 0.0) >= 0.05)


def is_conditional(row):
    return (to_int(row.get("condition_fields"), 0) > 0
            or to_int(row.get("conditional_levels"), 0) > 0)


def classify(row, handdesigned_set, single_set, auto_marker):
    """Bucket a row by provenance: single primitive, hand-authored nested, or tuner output."""
    if str(row.get("is_baseline", "")).strip() in ("1", "true", "True"):
        return SINGLE
    path = str(row.get("schema_path", "")).replace("\\", "/")
    base = os.path.basename(path).lower()
    if base in single_set:
        return SINGLE
    if base in handdesigned_set:
        return HANDDESIGNED
    if auto_marker and auto_marker in path:
        return AUTO
    if "best_schema" in base:
        return AUTO
    # Fall back to runtime structure: single-structure -> SINGLE, multi-structure -> AUTO.
    if to_float(row.get("active_structure_types"), 0.0) <= 1.0 and not is_conditional(row):
        return SINGLE
    if is_nested(row) or is_conditional(row):
        return AUTO
    return OTHER


def group_key(row):
    return (row.get("dataset_name", "?"), row.get("workload_name", "?"))


def best_by_latency(rows):
    finite = [r for r in rows if latency_of(r) != math.inf]
    if not finite:
        return None
    return min(finite, key=lambda r: (latency_of(r), r.get("schema_name", "")))


def schema_label(row):
    if row is None:
        return "-"
    name = row.get("schema_name", "?")
    tags = []
    if is_nested(row):
        tags.append("nested")
    if is_conditional(row):
        tags.append("cond")
    suffix = f" ({'+'.join(tags)})" if tags else ""
    return f"{name}{suffix}"


def fmt_ms(value):
    if value in (None, math.inf) or value != value:
        return "n/a"
    return f"{value:.4f}"


def check_provenance(rows):
    """Return the set of distinct provenance tuples among kept rows."""
    tuples = set()
    for row in rows:
        tuples.add(tuple(str(row.get(f, "")).strip() for f in PROVENANCE_FIELDS))
    return tuples


def analyze(rows, handdesigned_set, single_set, auto_marker, gpu_full_only):
    groups = defaultdict(list)
    fallback = []
    for row in rows:
        if gpu_full_only and gpu_status(row) not in ("full",):
            fallback.append(row)
            continue
        groups[group_key(row)].append(row)

    cells = []
    for key in sorted(groups):
        members = groups[key]
        by_class = defaultdict(list)
        for row in members:
            by_class[classify(row, handdesigned_set, single_set, auto_marker)].append(row)

        best_single = best_by_latency(by_class.get(SINGLE, []))
        best_hand = best_by_latency(by_class.get(HANDDESIGNED, []))
        # Prefer the auto schema actually tuned for THIS dataset (its name is in the export path).
        auto_rows = by_class.get(AUTO, [])
        matched = [r for r in auto_rows if key[0] in str(r.get("schema_path", ""))]
        best_auto = best_by_latency(matched or auto_rows)
        oracle = best_by_latency(members)

        single_ms = latency_of(best_single) if best_single else math.inf
        auto_ms = latency_of(best_auto) if best_auto else math.inf
        oracle_ms = latency_of(oracle) if oracle else math.inf

        speedup = (single_ms / auto_ms) if (best_single and best_auto and auto_ms > 0) else None
        rel_regret = (auto_ms / oracle_ms - 1.0) if (best_auto and oracle and oracle_ms > 0) else None
        confident = str(best_auto.get("ranking_confident", "")).strip() in ("1", "true", "True") if best_auto else False

        cells.append({
            "dataset": key[0],
            "workload": key[1],
            "best_single": best_single,
            "best_handdesigned": best_hand,
            "best_auto": best_auto,
            "oracle": oracle,
            "single_ms": single_ms,
            "hand_ms": latency_of(best_hand) if best_hand else math.inf,
            "auto_ms": auto_ms,
            "oracle_ms": oracle_ms,
            "speedup_auto_vs_single": speedup,
            "relative_regret_auto": rel_regret,
            "auto_confident": confident,
        })
    return cells, fallback


def geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def write_comparison_md(cells, fallback, provenance, path):
    lines = ["# Method comparison — auto-tuned nested vs best single primitive", ""]

    if len(provenance) > 1:
        lines += ["> **WARNING: rows span multiple provenance settings** "
                  "(score objective / mode / queries / seed / weights differ). "
                  "Comparison may be apples-to-oranges; run audit_schema_scores.py.", ""]

    lines += ["## Headline (GPU `full` rows only)", "",
              "| Dataset | Workload | Best single | Hand-nested | Auto-tuned | Oracle | "
              "Speedup auto/single | Rel. regret auto | Confident |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        speedup = f"{c['speedup_auto_vs_single']:.2f}x" if c["speedup_auto_vs_single"] else "n/a"
        regret = f"{c['relative_regret_auto']*100:.1f}%" if c["relative_regret_auto"] is not None else "n/a"
        lines.append(
            f"| {c['dataset']} | {c['workload']} | "
            f"{schema_label(c['best_single'])} {fmt_ms(c['single_ms'])}ms | "
            f"{schema_label(c['best_handdesigned'])} {fmt_ms(c['hand_ms'])}ms | "
            f"{schema_label(c['best_auto'])} {fmt_ms(c['auto_ms'])}ms | "
            f"{fmt_ms(c['oracle_ms'])}ms | {speedup} | {regret} | "
            f"{'yes' if c['auto_confident'] else 'no'} |")

    speedups = [c["speedup_auto_vs_single"] for c in cells]
    regrets = [c["relative_regret_auto"] for c in cells if c["relative_regret_auto"] is not None]
    wins = sum(1 for s in speedups if s and s > 1.0)
    counted = sum(1 for s in speedups if s)
    gm = geomean(speedups)
    lines += ["",
              f"**Summary:** auto-tuned beats best single in **{wins}/{counted}** cells; "
              f"geomean speedup **{gm:.2f}x**" if gm else "**Summary:** no comparable cells.",
              (f"; mean relative regret vs oracle **{statistics.fmean(regrets)*100:.1f}%**."
               if regrets else "."), ""]

    lines += ["## Excluded — GPU fallback rows (not in headline)", ""]
    if fallback:
        lines += ["| Dataset | Workload | Schema | gpu_support_status | backend |",
                  "|---|---|---|---|---|"]
        seen = set()
        for row in fallback:
            sig = (row.get("dataset_name"), row.get("workload_name"),
                   row.get("schema_name"), gpu_status(row))
            if sig in seen:
                continue
            seen.add(sig)
            lines.append(f"| {row.get('dataset_name','?')} | {row.get('workload_name','?')} | "
                         f"{row.get('schema_name','?')} | {gpu_status(row)} | "
                         f"{row.get('backend','?')} |")
    else:
        lines.append("_None — all measured rows ran on GPU._")
    lines.append("")

    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def write_summary_json(cells, fallback, provenance, path):
    def row_brief(row):
        if row is None:
            return None
        return {
            "schema_name": row.get("schema_name"),
            "schema_path": row.get("schema_path"),
            "latency_ms": latency_of(row),
            "nested": is_nested(row),
            "conditional": is_conditional(row),
            "gpu_support_status": gpu_status(row),
        }

    payload = {
        "provenance_distinct": len(provenance),
        "cells": [{
            "dataset": c["dataset"], "workload": c["workload"],
            "best_single": row_brief(c["best_single"]),
            "best_handdesigned": row_brief(c["best_handdesigned"]),
            "best_auto": row_brief(c["best_auto"]),
            "oracle": row_brief(c["oracle"]),
            "speedup_auto_vs_single": c["speedup_auto_vs_single"],
            "relative_regret_auto": c["relative_regret_auto"],
            "auto_confident": c["auto_confident"],
        } for c in cells],
        "fallback_count": len(fallback),
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def write_combined_raw(rows, path):
    """Concatenate all input rows (drop helper cols) for train_schema_selector.py."""
    fields = []
    for row in rows:
        for key in row:
            if key != "_source" and key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def make_plots(cells, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        lines = ["# Plots unavailable (matplotlib not installed)", "",
                 "Per-cell latency (ms): single / hand-nested / auto-tuned / oracle", ""]
        for c in cells:
            lines.append(f"- {c['dataset']}/{c['workload']}: "
                         f"single={fmt_ms(c['single_ms'])} hand={fmt_ms(c['hand_ms'])} "
                         f"auto={fmt_ms(c['auto_ms'])} oracle={fmt_ms(c['oracle_ms'])}")
        with open(os.path.join(out_dir, "plots_fallback.md"), "w") as handle:
            handle.write("\n".join(lines))
        return ["plots_fallback.md"]

    written = []
    labels = [f"{c['dataset'][:10]}\n{c['workload'][:10]}" for c in cells]
    x = range(len(cells))

    # Latency by method (grouped bars, log scale).
    fig, ax = plt.subplots(figsize=(max(6, len(cells) * 1.6), 4))
    width = 0.2
    for i, (name, key) in enumerate([("single", "single_ms"), ("hand-nested", "hand_ms"),
                                     ("auto-tuned", "auto_ms"), ("oracle", "oracle_ms")]):
        vals = [c[key] if c[key] != math.inf else 0 for c in cells]
        ax.bar([xi + (i - 1.5) * width for xi in x], vals, width, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("avg latency (ms)")
    ax.set_yscale("log")
    ax.set_title("Query latency by method")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_by_method.png"), dpi=120)
    plt.close(fig)
    written.append("latency_by_method.png")

    # Relative regret vs oracle (auto-tuned).
    fig, ax = plt.subplots(figsize=(max(6, len(cells) * 1.4), 4))
    regrets = [(c["relative_regret_auto"] or 0.0) * 100 for c in cells]
    ax.bar(list(x), regrets, color="#c0504d")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("relative regret vs oracle (%)")
    ax.set_title("Auto-tuned regret vs per-cell oracle")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regret_vs_oracle.png"), dpi=120)
    plt.close(fig)
    written.append("regret_vs_oracle.png")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=["results/eval/*.csv"],
                        help="Glob(s) for raw schema-search CSVs (*_best/*_pareto excluded).")
    parser.add_argument("--out-dir", default="results/eval/report")
    parser.add_argument("--handdesigned", default=",".join(DEFAULT_HANDDESIGNED),
                        help="Comma-separated basenames treated as hand-designed nested configs.")
    parser.add_argument("--singles", default=",".join(DEFAULT_SINGLES),
                        help="Comma-separated basenames treated as single-primitive configs.")
    parser.add_argument("--auto-marker", default=DEFAULT_AUTO_MARKER,
                        help="Path substring marking auto-tuned (exported) schemas.")
    parser.add_argument("--include-fallback", action="store_true",
                        help="Include GPU-fallback rows in the headline (default: separate table).")
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit(f"No input CSVs matched: {args.inputs}")
    rows = load_rows(paths)
    if not rows:
        raise SystemExit("No rows loaded.")

    kept = [r for r in rows if is_final_latency(r)]
    if not kept:
        raise SystemExit("No final-latency rows after provenance filtering.")
    provenance = check_provenance(kept)

    handdesigned_set = {b.strip().lower() for b in args.handdesigned.split(",") if b.strip()}
    single_set = {b.strip().lower() for b in args.singles.split(",") if b.strip()}
    cells, fallback = analyze(kept, handdesigned_set, single_set, args.auto_marker,
                              gpu_full_only=not args.include_fallback)

    os.makedirs(args.out_dir, exist_ok=True)
    write_comparison_md(cells, fallback, provenance, os.path.join(args.out_dir, "comparison.md"))
    write_summary_json(cells, fallback, provenance, os.path.join(args.out_dir, "summary.json"))
    write_combined_raw(kept, os.path.join(args.out_dir, "combined_raw.csv"))
    plots = make_plots(cells, args.out_dir)

    print(f"Loaded {len(rows)} rows from {len(paths)} file(s); {len(kept)} final-latency rows.")
    print(f"Distinct provenance settings: {len(provenance)}"
          + ("  <-- WARNING: not apples-to-apples" if len(provenance) > 1 else ""))
    print(f"Cells: {len(cells)}; GPU-fallback rows excluded from headline: {len(fallback)}")
    speedups = [c["speedup_auto_vs_single"] for c in cells if c["speedup_auto_vs_single"]]
    if speedups:
        gm = geomean(speedups)
        wins = sum(1 for s in speedups if s > 1.0)
        print(f"Auto-tuned beats best single in {wins}/{len(speedups)} cells; geomean speedup {gm:.2f}x")
    print(f"Wrote: {os.path.join(args.out_dir, 'comparison.md')}, summary.json, "
          f"combined_raw.csv, {', '.join(plots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

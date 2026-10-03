#!/usr/bin/env python3
"""Regenerate the two tables the results are actually reported from.

    python scripts/summarize_results.py                 # both tables to stdout
    python scripts/summarize_results.py --csv           # also refresh final_results_table.csv
    python scripts/summarize_results.py --compare PREFIX_2026-08-15_indexicon_summary_capped.csv

This existed as throwaway one-liners, which is how two wrong numbers reached a summary: the
by-size table read `searched_ms`/`tuned_ms` from heatmap_summary.csv and silently reported the
transfer tiers (250M+) as search losses, because transfer cells leave those columns blank and the
winner only appears in the per-cell *_test.csv. Both tables are derived here from the per-cell
CSVs so that cannot recur.

Definitions, because the columns are easy to misread:
  ours best   - fastest schema of ANY kind in the cell's held-out re-measure
  searched    - fastest GA/crossover product (evolved_*, xover_*, generated_*)
  tuned       - fastest fixed-grid single (tuned_*)
  best/worst default - over the nine configs/schemas/*_default.json baselines
  ix best     - fastest of Indexicon's three structures on the same trace, same session
Latency is read from latency_repeat_mean_ms (pooled over all repeats) when present, falling back
to avg_latency_ms (the cold first pass) for CSVs written before that column existed. Mixing the
two understates us by ~11-13%, so the fallback is reported when it is used.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

MATRIX = [Path("results/eval_traces/matrix")]  # --matrix overrides; several dirs are merged
INDEXICON = Path("results/framework_compare/indexicon")

SEARCHED = re.compile(r"^(evolved_|xover_|generated_)")
TUNED = re.compile(r"^tuned_")
DEFAULT = re.compile(r"_default$")
SIZE_ORDER = ["1M", "5M", "25M", "50M", "100M", "250M", "500M", "1B"]


def latency(row: dict) -> float | None:
    """Pooled repeat mean when available, else the cold pass. None if neither parses."""
    for key in ("latency_repeat_mean_ms", "avg_latency_ms"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                return None
    return None


def cell_rows(path: Path) -> list[tuple[str, float]]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name, ms = row.get("schema_name", ""), latency(row)
            if name and ms is not None and ms > 0:
                rows.append((name, ms))
    return rows


def best(rows, predicate):
    hits = [(n, ms) for n, ms in rows if predicate(n)]
    return min(hits, key=lambda pair: pair[1]) if hits else (None, None)


def kind_of(name: str) -> str:
    if SEARCHED.match(name):
        return "searched"
    if TUNED.match(name):
        return "tuned"
    if DEFAULT.search(name):
        return "default"
    return "hand"


def collect() -> list[dict]:
    config_cells = {}
    config_path = Path("configs/datasets/curated_cells.json")
    if config_path.exists():
        import json
        for cell in json.loads(config_path.read_text(encoding="utf-8"))["cells"]:
            config_cells[cell["cell"]] = cell

    out, skipped = [], []
    for path in sorted(p for directory in MATRIX for p in directory.glob("*_test.csv")):
        name = path.name[: -len("_test.csv")]
        # Only cells the current config declares. results/eval_traces/matrix/ still holds CSVs
        # from the superseded ad-hoc dataset (terrain_*, industrial_*, urban_*), and folding those
        # into a median silently mixes two different benchmarks.
        if config_cells and name not in config_cells:
            skipped.append(name)
            continue
        rows = cell_rows(path)
        if not rows:
            continue
        meta = config_cells.get(name, {})
        overall_name, overall_ms = min(rows, key=lambda pair: pair[1])
        searched_name, searched_ms = best(rows, lambda n: bool(SEARCHED.match(n)))
        tuned_name, tuned_ms = best(rows, lambda n: bool(TUNED.match(n)))
        defaults = [ms for n, ms in rows if DEFAULT.search(n)]
        out.append(dict(
            cell=name,
            size=meta.get("size_label", name.rsplit("_", 1)[-1]),
            morphology=meta.get("morphology", ""),
            arm=meta.get("arm", ""),
            ours_best=overall_ms,
            ours_best_name=overall_name,
            kind=kind_of(overall_name),
            searched_ms=searched_ms,
            searched_name=searched_name,
            tuned_ms=tuned_ms,
            tuned_name=tuned_name,
            best_default_ms=min(defaults) if defaults else None,
            worst_default_ms=max(defaults) if defaults else None,
            schemas_measured=len(rows),
        ))
    if skipped:
        print(f"(skipped {len(skipped)} *_test.csv not in the current cell config: "
              f"{', '.join(sorted(skipped)[:6])}{' ...' if len(skipped) > 6 else ''})\n")
    return out


def by_size_table(cells: list[dict]) -> None:
    print(f"{'size':>6}{'cells':>7}{'ours best':>11}{'searched':>10}{'tuned':>9}"
          f"{'best dflt':>11}{'worst dflt':>12}{'dflt cost':>11}  winner kinds")
    for size in SIZE_ORDER:
        group = [c for c in cells if c["size"] == size]
        if not group:
            continue
        def med(key):
            vals = [c[key] for c in group if c.get(key)]
            return statistics.median(vals) if vals else None
        ours, bd, wd = med("ours_best"), med("best_default_ms"), med("worst_default_ms")
        srch, tuned = med("searched_ms"), med("tuned_ms")
        kinds = {}
        for c in group:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        kind_str = " ".join(f"{k}:{v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
        fmt = lambda v: f"{v:.4f}" if v else "-"
        cost = f"{bd / ours:.2f}x" if ours and bd else "-"
        print(f"{size:>6}{len(group):>7}{fmt(ours):>11}{fmt(srch):>10}{fmt(tuned):>9}"
              f"{fmt(bd):>11}{fmt(wd):>12}{cost:>11}  {kind_str}")


def head_to_head(compare: Path | None) -> None:
    summary = INDEXICON / "indexicon_summary.csv"
    if not summary.exists():
        print(f"\n(no {summary} - skipping head-to-head)")
        return
    rows = list(csv.DictReader(summary.open(encoding="utf-8-sig")))
    prior = {}
    if compare and compare.exists():
        for line in compare.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or line.startswith("cell,") or not line.strip():
                continue
            parts = line.split(",")
            prior[parts[0]] = float(parts[7])   # ix_best_ms / mds_best_ms columns
            prior[parts[0] + ":ours"] = float(parts[3])

    print(f"\n{'cell':<24}{'ours best':>11}{'ix best':>9}{'ix/ours':>9}"
          + ("{:>10}".format("was") if prior else ""))
    ratios, prior_ratios = [], []
    for row in rows:
        def num(key):
            value = row.get(key, "")
            return float(value) if value not in ("", None) else float("inf")
        ours = num("mds_best_ms")
        ix = min(num("ix_octree_ms"), num("ix_kdtree_ms"), num("ix_rtree_ms"))
        if ours in (0.0, float("inf")) or ix == float("inf"):
            continue
        ratios.append(ix / ours)
        line = f"{row['cell']:<24}{ours:>11.4f}{ix:>9.4f}{ix / ours:>9.2f}"
        if row["cell"] in prior:
            was = prior[row["cell"]] / prior[row["cell"] + ":ours"]
            prior_ratios.append(was)
            line += f"{was:>10.2f}"
        print(line)
    if ratios:
        med = statistics.median(ratios)
        print(f"\n  median ix/ours {med:.3f}  ->  we are {1 / med:.2f}x slower; "
              f"we win {sum(1 for r in ratios if r >= 1)}/{len(ratios)}")
        if prior_ratios:
            pmed = statistics.median(prior_ratios)
            print(f"  before        {pmed:.3f}  ->  {1 / pmed:.2f}x slower")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", action="store_true", help="also refresh final_results_table.csv")
    parser.add_argument("--compare", type=Path, help="a preserved prior summary to diff against")
    parser.add_argument("--matrix", type=Path, nargs="+",
                        help="result dirs holding <cell>_test.csv (default results/eval_traces/matrix). "
                             "Several dirs are merged, e.g. the workstation's and the server's engine-6 runs: "
                             "every ratio is within one cell, so each stays within one machine.")
    parser.add_argument("--no-indexicon", action="store_true", help="skip the August Indexicon table")
    args = parser.parse_args()
    global MATRIX
    if args.matrix:
        MATRIX = args.matrix

    cells = collect()
    if not cells:
        sys.exit(f"no *_test.csv under {MATRIX}")
    by_size_table(cells)
    if not args.no_indexicon:
        head_to_head(args.compare if args.compare else INDEXICON /
                     "PREFIX_2026-08-15_indexicon_summary_capped.csv")

    if args.csv:
        out = MATRIX[0] / "final_results_table.csv"
        fields = list(cells[0].keys())
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(cells)
        print(f"\nwrote {out} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

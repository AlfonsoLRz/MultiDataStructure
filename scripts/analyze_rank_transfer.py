#!/usr/bin/env python3
"""Measure how well low-fidelity (subsampled-cloud) schema rankings transfer to full scale.

Joins the flat-evaluation CSVs of the same candidate set at different fidelities on
schema_name and reports: Spearman rank correlation of the scores, whether the
full-scale winner appears in the low-fidelity top-K, and top-K set overlap. This is
the evidence behind the multi-fidelity search-cost argument: search cheap, confirm
only the top-K at full scale.

Usage:
  python scripts/analyze_rank_transfer.py --full results/eval_traces/alhambra_100m.csv \
      --low results/eval_traces/alhambra_sub2m.csv results/eval_traces/alhambra_sub5m.csv
"""

import argparse
import csv
from pathlib import Path

from scipy.stats import spearmanr


def load_scores(path: Path):
    scores = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            name = row.get("schema_name", "")
            try:
                score = float(row.get("score", ""))
            except ValueError:
                continue
            if name and (name not in scores or score < scores[name]):
                scores[name] = score
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--low", type=Path, nargs="+", required=True)
    parser.add_argument("--topk", type=int, nargs="*", default=[3, 5, 10])
    args = parser.parse_args()

    full = load_scores(args.full)
    full_ranked = sorted(full, key=full.get)
    print(f"full scale: {args.full.name} ({len(full)} candidates); winner = {full_ranked[0]}")

    for low_path in args.low:
        low = load_scores(low_path)
        shared = [n for n in full_ranked if n in low]
        if len(shared) < 3:
            print(f"\n{low_path.name}: only {len(shared)} shared candidates, skipping")
            continue

        rho, pvalue = spearmanr([full[n] for n in shared], [low[n] for n in shared])
        low_ranked = sorted(shared, key=low.get)
        print(f"\n{low_path.name}: {len(shared)} shared candidates")
        print(f"  Spearman rho = {rho:.3f} (p = {pvalue:.2e})")
        print(f"  low-fidelity winner = {low_ranked[0]}")
        for k in args.topk:
            in_topk = full_ranked[0] in low_ranked[:k]
            overlap = len(set(full_ranked[:k]) & set(low_ranked[:k]))
            print(f"  full winner in low top-{k}: {in_topk} | top-{k} set overlap: {overlap}/{k}")


if __name__ == "__main__":
    main()

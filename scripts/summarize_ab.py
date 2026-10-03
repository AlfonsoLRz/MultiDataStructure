"""Summarise an A/B engine run (scripts/ab_engine.ps1): per cell and schema, B relative to A.

    python scripts/summarize_ab.py D:\\MDS_ab\\runs\\<tag> [--noise D:\\MDS_ab\\runs\\aa]

Reports B/A latency ratios for radius, kNN and all queries, plus build time, per round and as a
geometric mean. A ratio is marked as a real effect only when every round agrees on its direction
and, if an A/A run is given, the change exceeds twice the A/A spread for that metric.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

METRICS = ("radius_mean_ms", "knn_mean_ms", "all_mean_ms", "build_ms_mean")


def load_runs(folder):
    runs = defaultdict(dict)  # (arm, round) -> {(cloud, schema): row}
    for path in sorted(Path(folder).glob("[AB]_r*.csv")):
        arm, round_ = path.stem.split("_r")
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                runs[(arm, int(round_))][(Path(row["cloud"]).name, row["schema_name"])] = row
    return runs


def ratios(runs):
    rounds = sorted({r for (_, r) in runs})
    table = defaultdict(lambda: defaultdict(list))  # key -> metric -> [B/A per round]
    for r in rounds:
        a, b = runs.get(("A", r), {}), runs.get(("B", r), {})
        for key in a.keys() & b.keys():
            for metric in METRICS:
                va, vb = float(a[key][metric]), float(b[key][metric])
                if va > 0 and vb > 0:
                    table[key][metric].append(vb / va)
    return table


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--noise", type=Path, help="an A/A run of the same cells, for the noise floor")
    args = parser.parse_args()

    table = ratios(load_runs(args.run))
    noise = {}
    if args.noise:
        aa = ratios(load_runs(args.noise))
        for metric in METRICS:
            spreads = [abs(math.log(r)) for m in aa.values() for r in m.get(metric, [])]
            noise[metric] = max(spreads) if spreads else 0.0

    print(f"{'cloud':38s} {'schema':42s} " + " ".join(f"{m.replace('_mean_ms','').replace('_ms_mean',''):>14s}" for m in METRICS))
    overall = defaultdict(list)
    for key in sorted(table):
        cells = []
        for metric in METRICS:
            values = table[key][metric]
            if not values:
                cells.append(f"{'-':>14s}")
                continue
            g = geomean(values)
            overall[metric].append(g)
            consistent = all(v < 1 for v in values) or all(v > 1 for v in values)
            beyond = abs(math.log(g)) > 2 * noise.get(metric, 0.0)
            mark = "*" if consistent and beyond else " "
            cells.append(f"{g:13.3f}{mark}")
        print(f"{key[0][:38]:38s} {key[1][:42]:42s} " + " ".join(cells))
    print(f"{'GEOMEAN':81s} " + " ".join(f"{geomean(overall[m]):13.3f} " for m in METRICS))
    if noise:
        print("A/A noise (max |log ratio|): " + ", ".join(f"{m}={math.exp(v):.3f}x" for m, v in noise.items()))
    print("B/A < 1 means B is faster. * = every round agrees and exceeds 2x the A/A spread (if given).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report of the cache accounting (scripts/cache_accounting.py, testing.md 2s).

Per cell: each schema's latency under every arm as a ratio to the base arm, its visits per query, and
where its time goes by depth band (top: depths 0-7; bottom: the deepest four levels of that tree;
middle: the rest), priced from the instrumented pass as visits x ns per visit. Then nested/single per
arm, and for e9's nested finalist against e9's pick, the visits and nanoseconds per band, which says
whether the visits nesting saves are the cheap ones. Over cells: the geomean effect of each arm on
singles and on nested schemas, and nested/single per arm.

    python scripts/cache_accounting_report.py --results results/eval_traces/cache_accounting_makinon2
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from cache_accounting import ARMS
from out_of_core import E9
from out_of_core_report import geomean, is_single

TOP_LEVELS = 8
BOTTOM_LEVELS = 4


def load(folder):
    """arm -> schema -> rows (one per process)."""
    data = {}
    for arm in ARMS:
        path = folder / f"{arm}.csv"
        if not path.exists():
            continue
        by_schema = defaultdict(list)
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                by_schema[row["schema_name"]].append(row)
        data[arm] = by_schema
    return data


def floats(text):
    return [float(v) for v in text.split(";") if v != ""]


def bands(row):
    """(visits per query, ns per query) per band from the instrumented pass. The deepest four
    levels are the bottom whatever the tree's depth, so a shallow grid is split too."""
    visits = floats(row["visits_by_depth"])
    ns = floats(row["visit_ns_by_depth"])
    deepest = len(visits) - 1
    out = {"top": [0.0, 0.0], "middle": [0.0, 0.0], "bottom": [0.0, 0.0]}
    for depth, (v, n) in enumerate(zip(visits, ns)):
        band = "bottom" if depth > deepest - BOTTOM_LEVELS else "top" if depth < TOP_LEVELS else "middle"
        out[band][0] += v
        out[band][1] += v * n
    return out


def median_latency(rows):
    return statistics.median(float(r["all_mean_ms"]) for r in rows) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    over = defaultdict(list)
    for cell_dir in sorted(p for p in Path(args.results).iterdir() if p.is_dir()):
        data = load(cell_dir)
        if "base" not in data:
            continue
        cell = cell_dir.name
        scene = cell.split("_")[0]
        arms = [arm for arm in ARMS if arm in data]
        print(f"\n## {cell}")
        huge = {arm: statistics.median(float(r["anon_huge_mb"]) for rows in data[arm].values() for r in rows) for arm in arms}
        print("  huge pages backing the process (MB, median over rows): " + ", ".join(f"{arm} {huge[arm]:.0f}" for arm in arms))
        print(f"  {'schema':<22} {'nodes':>9} {'pool MB':>8} {'base us':>9} " + " ".join(f"{arm:>9}" for arm in arms if arm != "base")
              + f" {'visits/q':>9} {'top':>14} {'middle':>14} {'bottom':>14} {'instr us':>9}")
        print(f"  {'':<22} {'':>9} {'':>8} {'':>9} " + " ".join(f"{'/ base':>9}" for arm in arms if arm != "base")
              + f" {'':>9} {'visits  us/q':>14} {'visits  us/q':>14} {'visits  us/q':>14} {'(sum)':>9}")
        base = data["base"]
        latency = {arm: {name: median_latency(rows) for name, rows in data[arm].items()} for arm in arms}
        for name in sorted(base, key=lambda n: latency["base"][n]):
            row = base[name][0]
            b = bands(row)
            ratios = " ".join(f"{latency[arm][name] / latency['base'][name]:9.3f}" if name in latency[arm] else f"{'-':>9}"
                              for arm in arms if arm != "base")
            # The band costs come from the instrumented pass, which also pays the telemetry and the
            # per-visit timestamp reads; their sum against the timed latency shows that overhead.
            print(f"  {name:<22} {int(row['num_nodes']):9d} {int(row['num_nodes']) * 32 / 1e6:8.1f} {latency['base'][name]:9.1f} {ratios} "
                  f"{float(row['mean_visited_nodes']):9.2f} "
                  + " ".join(f"{b[band][0]:6.2f} {b[band][1] / 1e3:7.2f}" for band in ("top", "middle", "bottom"))
                  + f" {sum(b[band][1] for band in b) / 1e3:9.2f}")
            for arm in arms:
                if arm != "base" and name in latency[arm]:
                    over[(arm, "single" if is_single(name) else "nested")].append(latency[arm][name] / latency["base"][name])

        for arm in arms:
            singles = {n: t for n, t in latency[arm].items() if is_single(n)}
            nested = {n: t for n, t in latency[arm].items() if not is_single(n)}
            if singles and nested:
                best_single = min(singles, key=singles.get)
                best_nested = min(nested, key=nested.get)
                ratio = nested[best_nested] / singles[best_single]
                over[("nested/single", arm)].append(ratio)
                print(f"  {arm}: best single {best_single} {singles[best_single]:.1f} us, best nested {best_nested} {nested[best_nested]:.1f} us, "
                      f"nested/single {ratio:.3f}")

        # e9's nested finalist against e9's pick: where the saved visits are, and what they cost.
        key = f"{scene}_100M"
        if key in E9:
            pick, nested = (E9[key][0]["name"], E9[key][1]["name"])
            if pick in base and nested in base:
                bp, bn = bands(base[pick][0]), bands(base[nested][0])
                parts = []
                for band in ("top", "middle", "bottom"):
                    parts.append(f"{band} {bp[band][0]:.2f} -> {bn[band][0]:.2f} visits, {bp[band][1] / 1e3:.2f} -> {bn[band][1] / 1e3:.2f} us")
                print(f"  {pick} -> {nested}: " + "; ".join(parts)
                      + f"; latency {latency['base'][pick]:.1f} -> {latency['base'][nested]:.1f} us")

    print("\n## Over cells")
    for arm in ARMS:
        if arm == "base":
            continue
        for kind in ("single", "nested"):
            values = over[(arm, kind)]
            if values:
                print(f"  {arm} / base, {kind}: geomean {geomean(values):.3f} over {len(values)} rows, range {min(values):.3f}-{max(values):.3f}")
    for arm in ARMS:
        values = over[("nested/single", arm)]
        if values:
            print(f"  nested/single, {arm}: geomean {geomean(values):.3f} over {len(values)} cells, range {min(values):.3f}-{max(values):.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report of the out-of-core battery (scripts/out_of_core.py, testing.md 2q).

Per cell, workload and storage regime: the best single and best nested schema of stage A and their
ratio, the regret of e9's pick, the flatness rule's pick and the library defaults against the best
single, the ISPRS schemas against their leaf-matched singles, and the reads behind the best single.
Then stage B (nested prefixes over each cold regime's best single; a win needs >= 3% in every
process), the checks (every regime returns the in-memory answers; reads by other processes on the
device), and geomeans over cells.

    python scripts/out_of_core_report.py --results results/eval_traces/out_of_core_makinon2 \\
        --e9 results/eval_traces/matrix_e9_makinon2
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from out_of_core import E9

REGIME_ORDER = ["memory", "0:memory", "0:paged", "0.01:memory", "0.01:paged"]
ISPRS = [("grid1_qt3_grid1_ot5l4096", "qt10l4096", "ot10l4096"), ("grid1_qt3_grid1_ot5l16384", "qt10l16384", "ot10l16384")]
FLAT_THRESHOLD = 0.60  # 2p's split


def is_single(name):
    return "_" not in name or name.endswith("_default")


def geomean(values):
    values = [v for v in values if v and v > 0]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else float("nan")


def load(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def medians(rows, column):
    """(schema, regime) -> (median over processes, {run: value})."""
    runs = defaultdict(dict)
    for row in rows:
        runs[(row["schema_name"], row["store_regime"])][int(row["test_run"])] = float(row[column])
    return {key: (statistics.median(v.values()), v) for key, v in runs.items()}


def flatness(e9, cell):
    path = Path(e9, f"{cell}_singles.csv") if e9 else None
    if not path or not path.exists():
        return None
    rows = load(path)
    return float(rows[0]["flatness_score"]) if rows and rows[0].get("flatness_score") else None


def checks(rows, label):
    """Every regime must return the in-memory answers; report reads by other processes."""
    sums = defaultdict(set)
    worst_extra = 0.0
    for row in rows:
        sums[row["schema_name"]].add(row["query_set_checksum"])
        if row["store_regime"] != "memory":
            worst_extra = max(worst_extra, float(row.get("store_device_extra_mb") or 0))
        if row["store_regime"] != "memory" and row.get("store_direct") != "1":
            print(f"    {label}: {row['schema_name']} {row['store_regime']} was NOT read with O_DIRECT")
    bad = [name for name, values in sums.items() if len(values) > 1]
    print(f"    {label}: answers identical across regimes for {len(sums) - len(bad)}/{len(sums)} schemas"
          + (f"; DIFFER: {', '.join(bad)}" if bad else "") + f"; most other reads on the device in one pass: {worst_extra:.1f} MB")


def regime_table(rows, column, cell, rule_pick, summary, workload):
    timing = medians(rows, column)
    total = medians(rows, "all_mean_ms")
    io = medians(rows, "store_io_ms")
    reads = medians(rows, "store_reads")
    kib = medians(rows, "store_bytes")
    pick = E9.get(cell, (None, None))[0]
    regimes = [r for r in REGIME_ORDER if any(k[1] == r for k in timing)]
    print(f"    {'regime':<12} {'best single':<22} {'us':>10} {'best nested':<26} {'nested/single':>13} {'e9 pick':>8} {'rule':>6} "
          f"{'ot def':>7} {'kd def':>7} {'reads/q':>8} {'KiB/q':>8} {'I/O %':>6}")
    for regime in regimes:
        entries = {name: t for (name, r), (t, _) in timing.items() if r == regime and t > 0}
        singles = {n: t for n, t in entries.items() if is_single(n)}
        nested = {n: t for n, t in entries.items() if not is_single(n)}
        best_single = min(singles, key=singles.get)
        best_nested = min(nested, key=nested.get) if nested else None
        ratio = entries[best_nested] / singles[best_single] if best_nested else float("nan")
        rel = lambda name: entries[name] / singles[best_single] if name in entries else float("nan")
        key = (best_single, regime)
        # Reads, bytes and I/O are per query over the whole trace (range and count boxes together).
        io_share = 100.0 * io[key][0] / total[key][0] if key in io and total[key][0] > 0 else 0.0
        print(f"    {regime:<12} {best_single:<22} {singles[best_single] * 1e3:10.1f} {best_nested or '-':<26} {ratio:13.3f} "
              f"{rel(pick['name']) if pick else float('nan'):8.3f} {rel(rule_pick):6.3f} {rel('octree_default'):7.3f} {rel('kdtree_default'):7.3f} "
              f"{reads[key][0] if key in reads else 0:8.2f} {kib[key][0] / 1024 if key in kib else 0:8.1f} {io_share:6.1f}")
        summary[(workload, column, regime)].append({"nested": ratio, "pick": rel(pick["name"]) if pick else float("nan"),
                                                    "rule": rel(rule_pick), "ot_default": rel("octree_default"),
                                                    "kd_default": rel("kdtree_default"), "best": best_single})
        for paper, qt, ot in ISPRS:
            if paper in entries:
                summary[(workload, column, regime, "isprs")].append((entries[paper] / min(entries.get(qt, math.inf), entries.get(ot, math.inf)),
                                                                     entries[paper] / singles[best_single]))
    for paper, qt, ot in ISPRS:
        line = [f"{paper}:"]
        for regime in regimes:
            entries = {name: t for (name, r), (t, _) in timing.items() if r == regime}
            if paper in entries and qt in entries and ot in entries:
                line.append(f"{regime} {entries[paper] / entries[qt]:.2f}x qt10, {entries[paper] / entries[ot]:.2f}x ot10")
        if len(line) > 1:
            print("      " + "; ".join(line))


def stage_b(path, column):
    rows = load(path)
    runs = defaultdict(dict)
    for row in rows:
        runs[row["schema_name"]][int(row["test_run"])] = float(row[column])
    single = next(name for name in runs if not name.startswith("B"))
    base = runs[single]
    best, adopted = None, []
    for name, values in runs.items():
        if name == single:
            continue
        ratios = [values[k] / base[k] for k in base if k in values]
        median = statistics.median(ratios)
        if best is None or median < best[1]:
            best = (name, median)
        if ratios and all(r <= 0.97 for r in ratios):
            adopted.append((name, max(ratios)))
    return single, base, best, adopted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--e9", default="", help="e9 results, for each cloud's flatness_score (the rule's pick)")
    args = parser.parse_args()

    summary = defaultdict(list)
    for cell_dir in sorted(p for p in Path(args.results).iterdir() if p.is_dir()):
        cell = cell_dir.name
        flat = flatness(args.e9, cell)
        rule_pick = "qt16l128tc" if flat is not None and flat > FLAT_THRESHOLD else "kd48l128sm"
        print(f"\n## {cell} (flatness {flat if flat is None else round(flat, 2)}, rule pick {rule_pick}, e9 pick "
              f"{E9[cell][0]['name'] if cell in E9 else '-'})")
        for workload, columns in (("trace", ["all_mean_ms"]), ("boxes", ["range_mean_ms", "count_mean_ms"])):
            path = cell_dir / f"{workload}.csv"
            if not path.exists():
                continue
            rows = load(path)
            checks(rows, workload)
            for column in columns:
                print(f"  {workload}, {column} (latency per query in us; ratios over the regime's best single)")
                regime_table(rows, column, cell, rule_pick, summary, workload)
            for b_path in sorted(cell_dir.glob(f"stageB_{workload}_*.csv")):
                if b_path.stem.endswith(tuple(f"_r{k}" for k in range(10))):
                    continue
                single, base, best, adopted = stage_b(b_path, columns[0])
                regime = b_path.stem.split(f"stageB_{workload}_")[1].replace("-", ":")
                print(f"  stage B {workload} {regime}: {single} {statistics.median(base.values()) * 1e3:.1f} us; best prefix {best[0]} "
                      f"{best[1]:.3f} (median over processes); >= 3% in every process: "
                      + (", ".join(f"{n} (worst {r:.3f})" for n, r in adopted) if adopted else "none"))
                summary[(workload, "stageB", regime)].append(best[1])

    print("\n## Over cells (geomeans of ratios to each regime's best single)")
    for key, items in summary.items():
        if key[1] == "stageB":
            print(f"  {key[0]} stage B {key[2]}: best prefix / single {geomean(items):.3f} over {len(items)} cells, "
                  f"range {min(items):.3f}-{max(items):.3f}")
        elif len(key) == 4:
            print(f"  {key[0]} {key[1]} {key[2]} ISPRS nested: / best leaf-matched single {geomean([a for a, _ in items]):.3f}, "
                  f"/ best single {geomean([b for _, b in items]):.3f}")
        else:
            cols = {c: geomean([item[c] for item in items]) for c in ("nested", "pick", "rule", "ot_default", "kd_default")}
            bests = ", ".join(sorted({item["best"] for item in items}))
            print(f"  {key[0]} {key[1]} {key[2]}: nested/single {cols['nested']:.3f}, e9 pick {cols['pick']:.3f}, rule {cols['rule']:.3f}, "
                  f"octree default {cols['ot_default']:.3f}, kd default {cols['kd_default']:.3f}; best singles: {bests}")


if __name__ == "__main__":
    main()

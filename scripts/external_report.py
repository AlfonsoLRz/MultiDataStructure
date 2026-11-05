#!/usr/bin/env python3
"""Report of the external-library battery (scripts/external_baselines.py, testing.md 2r).

Per cell and query kind (all = the trace's query mix, radius, kNN): our library defaults at leaf 32,
the flatness rule's pick and the e9 pick, and each library at its default and at its best
parameter, as latency per query (median over rounds) and relative to the e9 pick. Then each
library's answers checked against the engine's (per-query counts and id checksums), and geomeans
over cells and scan groups:

  - default / e9 pick: how far a library's out-of-the-box setting is from our tuned pick;
  - best / e9 pick: the library tuned on its own parameter, against the same pick;
  - default / best: the library's own leaf lever, the external counterpart of 2m's tuning lever.

    python scripts/external_report.py --results results/eval_traces/external_makinon2 --e9 results/eval_traces/matrix_e9 results/eval_traces/matrix_e9_makinon2
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from run_cells import REPO
from summarize_nesting import GROUPS

# (library label, tool CSV, configuration-name prefix, default configuration or None)
LIBRARIES = [
    ("nanoflann kd", "nanoflann", "nanoflann_kd_", "nanoflann_kd_leaf10"),
    ("PCL KdTreeFLANN", "pcl", "pcl_kdtree", "pcl_kdtree_flann"),
    ("PCL octree", "pcl", "pcl_octree", None),
    ("Open3D KDTreeFlann", "open3d", "open3d_", "open3d_kdtreeflann"),
    ("unibn octree", "unibn", "unibn_", "unibn_octree_bucket32"),
    ("Boost R-tree", "boost_rtree", "boost_rtree_", "boost_rtree_rstar16_str"),
]
METRICS = [("all", "all_mean_ms"), ("radius", "radius_mean_ms"), ("knn", "knn_mean_ms")]
FLAT_THRESHOLD = 0.60


def geomean(values):
    values = [v for v in values if v and v > 0 and math.isfinite(v)]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else float("nan")


def load(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def medians(rows):
    """configuration -> {column: median over rounds}."""
    runs = defaultdict(lambda: defaultdict(list))
    for row in rows:
        partial = float(row.get("unsupported_queries") or 0) > 0
        for _, column in METRICS:
            # A library that skips a query kind has no mean over the trace's whole mix.
            runs[row["schema_name"]][column].append(0.0 if partial and column == "all_mean_ms" else float(row[column]))
    return {name: {c: statistics.median(v) for c, v in cols.items()} for name, cols in runs.items()}


def flatness(e9_dirs, cell):
    for d in e9_dirs:
        path = Path(d, f"{cell}_singles.csv")
        if path.exists():
            rows = load(path)
            if rows and rows[0].get("flatness_score"):
                return float(rows[0]["flatness_score"])
    return None


def answers(path):
    """configuration -> {query_id: (returned, checksum)}."""
    out = defaultdict(dict)
    for row in load(path):
        out[row["schema_name"]][int(row["query_id"])] = (int(row["returned_points"]), row["result_checksum"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--e9", nargs="*", default=[], help="e9 result folders, for each cloud's flatness_score")
    parser.add_argument("--cells-config", default="configs/datasets/curated_cells.json")
    args = parser.parse_args()
    morphology = {c["cell"]: c["morphology"] for c in json.loads((REPO / args.cells_config).read_text())["cells"]}
    group_of = {m: g for g, ms in GROUPS.items() for m in ms}

    summary = defaultdict(list)
    for cell_dir in sorted(p for p in Path(args.results).iterdir() if p.is_dir() and (p / "mds.csv").exists()):
        cell = cell_dir.name
        group = group_of.get(morphology.get(cell, ""), "other")
        ours = medians(load(cell_dir / "mds.csv"))
        pick = next(name for name in ours if name.startswith("e9pick_"))
        flat = flatness(args.e9, cell)
        rule = "qt16l128tc" if flat is not None and flat > FLAT_THRESHOLD else "kd48l128sm"
        print(f"\n## {cell} ({group}, flatness {flat if flat is None else round(flat, 2)}): e9 pick {pick}, rule pick {rule}")
        print(f"  {'':<34}" + "".join(f"{label:>22}" for label, _ in METRICS))
        line = lambda label, values: print(f"  {label:<34}" + "".join(f"{v:>22}" for v in values))
        base = ours[pick]
        for name in ("octree_default", "kdtree_default", rule, pick):
            line(f"ours {name}", [f"{ours[name][c] * 1e3:8.2f} us {ours[name][c] / base[c]:6.3f}x" if base[c] > 0 else "-" for _, c in METRICS])
            if name != pick:
                for metric, c in METRICS:
                    summary[("ours", name if "default" in name else "rule", metric, group)].append(ours[name][c] / base[c] if base[c] > 0 else float("nan"))

        reference = answers(cell_dir / "mds_queries.csv").get(pick, {})
        best_external = {metric: math.inf for metric, _ in METRICS}
        for label, tool, prefix, default in LIBRARIES:
            path = cell_dir / f"{tool}.csv"
            if not path.exists():
                continue
            configs = {n: v for n, v in medians(load(path)).items() if n.startswith(prefix)}
            if not configs:
                continue
            for metric, c in METRICS:
                valid = {n: v[c] for n, v in configs.items() if v[c] > 0}
                if not valid or base[c] <= 0:
                    continue
                best = min(valid, key=valid.get)
                best_external[metric] = min(best_external[metric], valid[best])
                summary[(label, "best", metric, group)].append(valid[best] / base[c])
                if default and default in valid:
                    summary[(label, "default", metric, group)].append(valid[default] / base[c])
                    summary[(label, "lever", metric, group)].append(valid[default] / valid[best])
            cells_text = []
            for metric, c in METRICS:
                valid = {n: v[c] for n, v in configs.items() if v[c] > 0}
                if not valid or base[c] <= 0:
                    cells_text.append("-")
                    continue
                best = min(valid, key=valid.get)
                text = f"{valid[best] * 1e3:7.2f} {valid[best] / base[c]:5.2f}x"
                if default and default in valid and default != best:
                    text += f" (def {valid[default] / base[c]:.2f}x)"
                cells_text.append(text)
            line(f"{label} best", cells_text)
            best_names = {metric: min((n for n in configs if configs[n][c] > 0), key=lambda n: configs[n][c], default="-") for metric, c in METRICS}
            print(f"  {'':<34}  best configuration: {best_names['all'] if best_names['all'] != '-' else best_names['radius']}")

            queries_path = cell_dir / f"{tool}_queries.csv"
            if queries_path.exists() and reference:
                checks = []
                for name, got in answers(queries_path).items():
                    if not name.startswith(prefix):
                        continue
                    counted = [q for q in got if q in reference]
                    count_bad = sum(1 for q in counted if got[q][0] != reference[q][0])
                    id_bad = sum(1 for q in counted if got[q][0] == reference[q][0] and got[q][1] != reference[q][1])
                    checks.append((count_bad + id_bad) / max(1, len(counted)))
                    if count_bad or id_bad:
                        summary[("check", label, "", "")].append((name, cell, count_bad, id_bad, len(counted)))
                print(f"  {'':<34}  answers vs engine: worst configuration {100 * max(checks):.2f}% of queries differ")
        for metric, _ in METRICS:
            if base[[c for m, c in METRICS if m == metric][0]] > 0 and math.isfinite(best_external[metric]):
                summary[("best external", "best", metric, group)].append(best_external[metric] / base[[c for m, c in METRICS if m == metric][0]])

    print("\n## Over cells: geomeans relative to the e9 pick (>1 = slower than our tuned pick), and each library's lever")
    groups = ["all cells"] + list(GROUPS) + ["other"]
    keys = sorted({k[:3] for k in summary if k[0] != "check"}, key=lambda k: (k[0] != "ours", k[0], k[1], k[2]))
    print(f"  {'':<46}" + "".join(f"{g:>20}" for g in groups))
    for key in keys:
        values = []
        for g in groups:
            items = [v for k, vs in summary.items() if k[:3] == key and (g == "all cells" or k[3] == g) for v in vs]
            values.append(f"{geomean(items):.3f} ({len(items)})" if items else "-")
        print(f"  {key[0] + ' ' + key[1] + ' ' + key[2]:<46}" + "".join(f"{v:>20}" for v in values))
    disagreements = [v for k, vs in summary.items() if k[0] == "check" for v in vs]
    print(f"\n## Answers differing from the engine's: {len(disagreements)} configuration-cells")
    for name, cell, count_bad, id_bad, total in disagreements:
        print(f"  {cell} {name}: {count_bad} counts and {id_bad} id sets differ of {total} queries")


if __name__ == "__main__":
    main()

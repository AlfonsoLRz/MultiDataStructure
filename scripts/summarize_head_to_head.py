"""Summarise a pinned MDS-vs-Indexicon run (scripts/head_to_head.ps1).

    python scripts/summarize_head_to_head.py D:\\MDS_ab\\h2h\\<tag>

Per cell: radius and kNN mean latency (microseconds, averaged over rounds) and build time for
each Indexicon structure and each MDS schema, then the ratios that matter:
MDS / Indexicon for octree vs octree, kd vs kd, the same-bucket kd yardstick, and MDS's best
schema (including the searched winner) vs Indexicon's best structure. Ratio < 1 = MDS faster.
Writes summary.csv next to the inputs.
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main():
    run = Path(sys.argv[1])
    ix = defaultdict(lambda: defaultdict(list))   # (cell, structure) -> metric -> values
    mds = defaultdict(lambda: defaultdict(list))  # (cell, schema) -> metric -> values

    for path in sorted(run.glob("ix_*_r*.json")):
        cell = path.stem[len("ix_"):path.stem.rindex("_r")].lower()
        for result in json.loads(path.read_text())["results"]:
            if result.get("status") != "ok":
                continue
            key = (cell, result["method"].replace("indexicon_", ""))
            for kind in ("radius", "knn"):
                if kind in result["by_type"]:
                    ix[key][kind].append(1000.0 * result["by_type"][kind]["avg_latency_ms"])
            ix[key]["build"].append(result["build_ms"])

    for path in sorted(run.glob("mds_r*.csv")):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                cell = Path(row["cloud"]).name.split(".")[0].lower()
                key = (cell, row["schema_name"])
                mds[key]["radius"].append(1000.0 * float(row["radius_mean_ms"]))
                mds[key]["knn"].append(1000.0 * float(row["knn_mean_ms"]))
                mds[key]["build"].append(float(row["build_ms_mean"]))

    def mean(table, key, metric):
        values = table.get(key, {}).get(metric, [])
        return statistics.mean(values) if values else float("nan")

    def mixed(table, key):
        # Equal-weight mean of radius and kNN, the two kinds the pipeline traces issue.
        return (mean(table, key, "radius") + mean(table, key, "knn")) / 2.0

    cells = sorted({c for c, _ in ix} | {c for c, _ in mds})
    rows = []
    for cell in cells:
        print(f"\n== {cell}")
        print(f"   {'method':44s} {'radius us':>10s} {'knn us':>10s} {'build ms':>10s}")
        for key in sorted(k for k in ix if k[0] == cell):
            print(f"   {'indexicon ' + key[1]:44s} {mean(ix, key, 'radius'):10.3f} {mean(ix, key, 'knn'):10.3f} {mean(ix, key, 'build'):10.0f}")
        for key in sorted(k for k in mds if k[0] == cell):
            print(f"   {'mds ' + key[1]:44s} {mean(mds, key, 'radius'):10.3f} {mean(mds, key, 'knn'):10.3f} {mean(mds, key, 'build'):10.0f}")

        pairs = [
            ("octree vs octree", (cell, "octree_default"), (cell, "octree")),
            ("kd vs kd", (cell, "kdtree_default"), (cell, "kdtree")),
            ("same-bucket kd (128)", (cell, "tuned_kd19l128"), (cell, "kdtree")),
        ]
        mds_keys = [k for k in mds if k[0] == cell]
        ix_keys = [k for k in ix if k[0] == cell]
        if mds_keys and ix_keys:
            best_mds = min(mds_keys, key=lambda k: mixed(mds, k))
            best_ix = min(ix_keys, key=lambda k: mixed(ix, k))
            pairs.append((f"best ({best_mds[1]} vs {best_ix[1]})", best_mds, best_ix))
        for label, mk, ik in pairs:
            r = mean(mds, mk, "radius") / mean(ix, ik, "radius")
            k = mean(mds, mk, "knn") / mean(ix, ik, "knn")
            b = mean(mds, mk, "build") / mean(ix, ik, "build")
            m = mixed(mds, mk) / mixed(ix, ik)
            print(f"   MDS/IX {label:40s} radius {r:5.2f}  knn {k:5.2f}  mixed {m:5.2f}  build {b:5.2f}")
            rows.append({"cell": cell, "pair": label, "mds": mk[1], "indexicon": ik[1],
                         "radius_ratio": round(r, 4), "knn_ratio": round(k, 4), "mixed_ratio": round(m, 4),
                         "build_ratio": round(b, 4),
                         "mds_radius_us": round(mean(mds, mk, "radius"), 4), "ix_radius_us": round(mean(ix, ik, "radius"), 4),
                         "mds_knn_us": round(mean(mds, mk, "knn"), 4), "ix_knn_us": round(mean(ix, ik, "knn"), 4)})

    if rows:
        with open(run / "summary.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("\nratio < 1 = MDS faster. summary.csv written.")


if __name__ == "__main__":
    main()

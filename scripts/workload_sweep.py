#!/usr/bin/env python3
"""Does nesting win under other workloads? A fixed schema panel timed on workload variants of one cell.

The curated pipeline traces are small local queries (a radius of a few centimetres, kNN with
k = 16). On them a deep sliding-midpoint kd single ties every nested schema (e9, testing.md 2j).
This sweep keeps each trace's query centres and changes what is asked there:

  base       the trace as recorded
  radius4    radius queries at 4x the trace radius
  radius16   radius queries at 16x
  knn64      kNN with k = 64
  range8     axis-aligned boxes of half-extent 8x the trace radius
  count64    count queries over boxes of half-extent 64x
  mixed      one query in six of each: radius 1x and 16x, kNN 16 and 64, range 8x, count 64x

and times a panel of singles and nested schemas, all with open-ended depth, with engine-check
--interleave in --runs separate processes (order rotated), pinned like run_cells.py. Per workload it
reports the best single, the best nested schema and their ratio.

    python3 scripts/workload_sweep.py --cell domfountain_100M --exe ../build_e9/MultiDataStructure \\
        --data-root /home2/alfonsolr/mds/data --trace-root /home2/alfonsolr/mds/traces/curated \\
        --cpu 6 --numa-node 0 --out ../results/workload_sweep
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

from run_cells import REPO, launch, merge_runs, remap_cloud, remap_trace_dir

WORKLOADS = ["base", "radius4", "radius16", "knn64", "range8", "count64", "mixed"]


def schema(name, leaf, levels):
    """levels: [(type, numLevels, extra fields)]."""
    out = []
    for kind, depth, extra in levels:
        level = {"type": kind, "numLevels": depth, "leafCapacity": leaf, "minPointsToSplit": max(2, leaf // 4)}
        level.update(extra)
        out.append(level)
    return {"name": name, "levels": out, "buildPolicy": {
        "maxDepth": sum(depth for _, depth, _ in levels), "leafCapacity": leaf, "minPointsToSplit": max(2, leaf // 4),
        "collapseSingleChild": True, "removeEmptyNodes": True, "allowOverlapDuplication": False}}


SM = {"axisPolicy": "sliding_midpoint"}
XY = {"axisPolicy": "xy"}
PANEL = [
    # singles
    schema("kd48l128sm", 128, [("KDTree", 48, SM)]),
    schema("kd48l512sm", 512, [("KDTree", 48, SM)]),
    schema("qt24l128", 128, [("QuadTree", 24, XY)]),
    schema("ot24l128", 128, [("Octree", 24, {})]),
    # nested
    schema("qt3_kd48l128sm", 128, [("QuadTree", 3, XY), ("KDTree", 48, SM)]),
    schema("qt6_kd48l128sm", 128, [("QuadTree", 6, XY), ("KDTree", 48, SM)]),
    schema("qt6_kd48l512sm", 512, [("QuadTree", 6, XY), ("KDTree", 48, SM)]),
    schema("qt9_ot24l128", 128, [("QuadTree", 9, XY), ("Octree", 24, {})]),
    schema("str64_kd48l128sm", 128, [("STR", 1, {"fanout": 64}), ("KDTree", 48, SM)]),
]


def variant_rows(rows, workload, r0):
    """Trace rows (dicts, world frame) rewritten for one workload."""
    out = []
    for i, row in enumerate(rows):
        c = [float(row["center_x"]), float(row["center_y"]), float(row["center_z"])]
        if row["query_type"] in ("range", "aabb_range", "count_range") and not any(c):
            c = [(float(row[f"bounds_min_{a}"]) + float(row[f"bounds_max_{a}"])) / 2 for a in "xyz"]
        kind, radius, k, half = row["query_type"], float(row["radius"] or 0), int(float(row["k"] or 0)), 0.0
        if workload == "radius4":
            kind, radius = "radius", 4 * r0
        elif workload == "radius16":
            kind, radius = "radius", 16 * r0
        elif workload == "knn64":
            kind, k = "knn", 64
        elif workload == "range8":
            kind, half = "range", 8 * r0
        elif workload == "count64":
            kind, half = "count_range", 64 * r0
        elif workload == "mixed":
            kind, radius, k, half = [("radius", r0, 0, 0), ("radius", 16 * r0, 0, 0), ("knn", 0, 16, 0),
                                     ("knn", 0, 64, 0), ("range", 0, 0, 8 * r0), ("count_range", 0, 0, 64 * r0)][i % 6]
        lo = [x - half for x in c] if half else [float(row[f"bounds_min_{a}"]) for a in "xyz"]
        hi = [x + half for x in c] if half else [float(row[f"bounds_max_{a}"]) for a in "xyz"]
        out.append({"query_id": i, "query_type": kind, "bounds_min_x": lo[0], "bounds_min_y": lo[1], "bounds_min_z": lo[2],
                    "bounds_max_x": hi[0], "bounds_max_y": hi[1], "bounds_max_z": hi[2],
                    "center_x": c[0], "center_y": c[1], "center_z": c[2], "radius": radius, "k": k,
                    "returned_points": -1, "latency_ms": 0})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--cells", default="configs/datasets/curated_cells.json")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--trace-root", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--queries", type=int, default=3000)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--cpu", type=int, default=-1)
    parser.add_argument("--numa-node", type=int, default=-1)
    args = parser.parse_args()

    cell = {c["cell"]: c for c in json.loads((REPO / args.cells).read_text())["cells"]}[args.cell]
    cloud = remap_cloud(cell["mdspc_path"], args.data_root)
    trace_dir = Path(remap_trace_dir(cell["trace_dir"], args.cell, args.trace_root))
    trace_dir = trace_dir if trace_dir.is_absolute() else REPO / trace_dir
    out = Path(args.out).resolve() / args.cell
    (out / "schemas").mkdir(parents=True, exist_ok=True)

    schemas = []
    for s in PANEL:
        path = out / "schemas" / f"{s['name']}.json"
        path.write_text(json.dumps(s, indent=2))
        schemas.append(str(path))

    rows = list(csv.DictReader(open(trace_dir / "pipeline_trace_test.csv", newline="", encoding="utf-8-sig")))
    rows = [rows[(i * len(rows)) // args.queries] for i in range(min(args.queries, len(rows)))]
    radii = [float(r["radius"]) for r in rows if r["query_type"] == "radius" and float(r["radius"]) > 0]
    r0 = statistics.median(radii) if radii else 0.05
    print(f"{args.cell}: {len(rows)} queries, trace radius {r0:.4f}, {len(schemas)} schemas", flush=True)

    for workload in args.workloads.split(","):
        merged = out / f"{workload}.csv"
        if merged.exists():
            continue
        trace = out / f"trace_{workload}.csv"
        variant = variant_rows(rows, workload, r0)
        with open(trace, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(variant[0]))
            writer.writeheader()
            writer.writerows(variant)
        runs = []
        for k in range(args.runs):
            shift = k * len(schemas) // args.runs
            run_csv = out / f"{workload}_r{k}.csv"
            run_csv.unlink(missing_ok=True)
            argv = [args.exe, "--mode", "engine-check", "--input", cloud, "--input-trace", str(trace),
                    "--queries", str(len(rows)), "--schemas", ";".join(schemas[shift:] + schemas[:shift]),
                    "--warmup", "1", "--measure-repeats", str(args.repeats), "--interleave",
                    "--summary-csv", str(run_csv), "--no-pause"]
            print(f"\n##### {args.cell} {workload} run {k + 1}/{args.runs}", flush=True)
            if launch(argv, args.cpu, args.numa_node, cwd=REPO).wait() == 0:
                runs.append(str(run_csv))
        if runs:
            merge_runs(runs, str(merged))

    report(out, [s["name"] for s in PANEL])


def report(out, names):
    singles = {n for n in names if "_" not in n}
    lines = [f"{'workload':10}{'best single':>16}{'us':>9}{'best nested':>20}{'us':>9}{'nested/single':>15}"]
    for workload in WORKLOADS:
        merged = out / f"{workload}.csv"
        if not merged.exists():
            continue
        lat = {}
        for r in csv.DictReader(open(merged, newline="", encoding="utf-8-sig")):
            lat.setdefault(r["schema_name"], []).append(float(r["all_mean_ms"]) * 1000)
        lat = {n: statistics.median(v) for n, v in lat.items()}
        s = min((n for n in lat if n in singles), key=lat.get)
        n = min((n for n in lat if n not in singles), key=lat.get)
        lines.append(f"{workload:10}{s:>16}{lat[s]:9.3f}{n:>20}{lat[n]:9.3f}{lat[n] / lat[s]:15.3f}")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()

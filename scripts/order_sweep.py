#!/usr/bin/env python3
"""Why does nesting never win? Three diagnostics on one cell, each a schema panel timed on a query variant.

e9 (testing.md 2k) picks a single structure in every cell, although the best nested finalist visits
fewer nodes (0.976x over 73 cells, 0.80-0.92 on TLS scans). This script tests the ways that could
still be our fault, and reproduces the regime of Ogayar-Anguita et al. (ISPRS 2023) in our engine.

Panels (--panel):
  base    the workload_sweep panel: 4 deep singles, 5 nested schemas (leaf 128/512)
  grid    the paper's grid top level as far as the engine allows (box-tested 27/64-cell grids)
          over kd/qt/ot tails
  paper   the paper's regime: quadtree, octree, qt->ot and grid->qt->grid->ot, depth <= 10,
          leaves of 4096 and 16384 points

Variants (--workloads):
  trace        the test trace as recorded (scan order: consecutive centres are metres apart)
  shuffled     the same queries in a seeded random order (the top of the tree goes cold)
  descent      count_range with a zero-extent box at each centre, shuffled: a root-to-leaf
               descent plus one leaf scan, which isolates the structure cost from leaf work
  shuffled_r4  shuffled radius queries at 4x the trace radius
  bigbox       100 range queries over boxes of 20-40% of the extent per axis (the ISPRS benchmark)
  bigbox_count the same boxes as count_range queries

Timing: engine-check --interleave, --runs processes with the schema order rotated, pinned as in
run_cells.py. Per variant the report gives the best single, the best nested schema, their ratio,
and the visited-node and tested-point ratios from the same CSVs.

    python3 scripts/order_sweep.py --cell domfountain_100M --exe ../build_e9/MultiDataStructure \\
        --data-root /home2/alfonsolr/mds/data --trace-root /home2/alfonsolr/mds/traces/curated \\
        --cpu 6 --numa-node 0 --out ../results/order_sweep --panel base,grid \\
        --workloads trace,shuffled,descent,shuffled_r4
"""
import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from run_cells import REPO, launch, merge_runs, remap_cloud, remap_trace_dir
from workload_sweep import PANEL as BASE_PANEL, SM, XY, schema

WORKLOADS = ["trace", "shuffled", "descent", "shuffled_r4", "bigbox", "bigbox_count"]

GRID_PANEL = [
    schema("grid1_kd48l128sm", 128, [("RegularGrid", 1, {}), ("KDTree", 48, SM)]),
    schema("hgrid1_kd48l128sm", 128, [("HGrid", 1, {}), ("KDTree", 48, SM)]),
    schema("hgrid2_kd48l128sm", 128, [("HGrid", 2, {}), ("KDTree", 48, SM)]),
    schema("hgrid1_ot24l128", 128, [("HGrid", 1, {}), ("Octree", 24, {})]),
    schema("hgrid2_qt24l128", 128, [("HGrid", 2, {}), ("QuadTree", 24, XY)]),
]

PAPER_PANEL = []
for leaf in (4096, 16384):
    PAPER_PANEL += [
        schema(f"qt10l{leaf}", leaf, [("QuadTree", 10, XY)]),
        schema(f"ot10l{leaf}", leaf, [("Octree", 10, {})]),
        schema(f"qt5_ot5l{leaf}", leaf, [("QuadTree", 5, XY), ("Octree", 5, {})]),
        schema(f"grid1_qt3_grid1_ot5l{leaf}", leaf,
               [("RegularGrid", 1, {}), ("QuadTree", 3, XY), ("RegularGrid", 1, {}), ("Octree", 5, {})]),
    ]

PANELS = {"base": BASE_PANEL, "grid": GRID_PANEL, "paper": PAPER_PANEL}


def is_single(name):
    return "_" not in name


def centre_of(row):
    c = [float(row["center_x"]), float(row["center_y"]), float(row["center_z"])]
    if row["query_type"] in ("range", "aabb_range", "count_range") and not any(c):
        c = [(float(row[f"bounds_min_{a}"]) + float(row[f"bounds_max_{a}"])) / 2 for a in "xyz"]
    return c


def make_row(i, kind, lo, hi, c, radius=0.0, k=0):
    return {"query_id": i, "query_type": kind, "bounds_min_x": lo[0], "bounds_min_y": lo[1], "bounds_min_z": lo[2],
            "bounds_max_x": hi[0], "bounds_max_y": hi[1], "bounds_max_z": hi[2],
            "center_x": c[0], "center_y": c[1], "center_z": c[2], "radius": radius, "k": k,
            "returned_points": -1, "latency_ms": 0}


def variant_rows(rows, workload, r0, seed, boxes):
    """Trace rows (dicts, world frame) rewritten for one variant."""
    rng = random.Random(seed)
    if workload in ("bigbox", "bigbox_count"):
        kind = "range" if workload == "bigbox" else "count_range"
        centres = [centre_of(r) for r in rows]
        lo_all = [min(c[a] for c in centres) for a in range(3)]
        hi_all = [max(c[a] for c in centres) for a in range(3)]
        out = []
        for i in range(boxes):
            lo, hi = [], []
            for a in range(3):
                extent = hi_all[a] - lo_all[a]
                size = extent * rng.uniform(0.2, 0.4)
                start = lo_all[a] + rng.uniform(0.0, max(0.0, extent - size))
                lo.append(start)
                hi.append(start + size)
            c = [(lo[a] + hi[a]) / 2 for a in range(3)]
            out.append(make_row(i, kind, lo, hi, c))
        return out
    order = list(range(len(rows)))
    if workload != "trace":
        rng.shuffle(order)
    out = []
    for i, j in enumerate(order):
        row = rows[j]
        c = centre_of(row)
        kind, radius, k = row["query_type"], float(row["radius"] or 0), int(float(row["k"] or 0))
        lo = [float(row[f"bounds_min_{a}"]) for a in "xyz"]
        hi = [float(row[f"bounds_max_{a}"]) for a in "xyz"]
        if workload == "descent":
            kind, radius, k, lo, hi = "count_range", 0.0, 0, list(c), list(c)
        elif workload == "shuffled_r4":
            kind, radius, k = "radius", 4 * r0, 0
        out.append(make_row(i, kind, lo, hi, c, radius, k))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--cells", default="configs/datasets/curated_cells.json")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--trace-root", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--panel", default="base,grid")
    parser.add_argument("--queries", type=int, default=3000)
    parser.add_argument("--boxes", type=int, default=100, help="queries per bigbox variant")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--workloads", default="trace,shuffled,descent,shuffled_r4")
    parser.add_argument("--cpu", type=int, default=-1)
    parser.add_argument("--numa-node", type=int, default=-1)
    args = parser.parse_args()

    cell = {c["cell"]: c for c in json.loads((REPO / args.cells).read_text())["cells"]}[args.cell]
    cloud = remap_cloud(cell["mdspc_path"], args.data_root)
    trace_dir = Path(remap_trace_dir(cell["trace_dir"], args.cell, args.trace_root))
    trace_dir = trace_dir if trace_dir.is_absolute() else REPO / trace_dir
    out = Path(args.out).resolve() / args.cell
    (out / "schemas").mkdir(parents=True, exist_ok=True)

    panel = [s for name in args.panel.split(",") for s in PANELS[name]]
    schemas = []
    for s in panel:
        path = out / "schemas" / f"{s['name']}.json"
        path.write_text(json.dumps(s, indent=2))
        schemas.append(str(path))
    tag = "+".join(args.panel.split(","))

    rows = list(csv.DictReader(open(trace_dir / "pipeline_trace_test.csv", newline="", encoding="utf-8-sig")))
    rows = [rows[(i * len(rows)) // args.queries] for i in range(min(args.queries, len(rows)))]
    radii = [float(r["radius"]) for r in rows if r["query_type"] == "radius" and float(r["radius"]) > 0]
    r0 = statistics.median(radii) if radii else 0.05
    print(f"{args.cell}: {len(rows)} queries, trace radius {r0:.4f}, panel {tag}: {len(schemas)} schemas", flush=True)

    for workload in args.workloads.split(","):
        merged = out / f"{tag}_{workload}.csv"
        if merged.exists():
            continue
        trace = out / f"trace_{workload}.csv"
        variant = variant_rows(rows, workload, r0, args.seed, args.boxes)
        with open(trace, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(variant[0]))
            writer.writeheader()
            writer.writerows(variant)
        runs = []
        for k in range(args.runs):
            shift = k * len(schemas) // args.runs
            run_csv = out / f"{tag}_{workload}_r{k}.csv"
            run_csv.unlink(missing_ok=True)
            argv = [args.exe, "--mode", "engine-check", "--input", cloud, "--input-trace", str(trace),
                    "--queries", str(len(variant)), "--schemas", ";".join(schemas[shift:] + schemas[:shift]),
                    "--warmup", "1", "--measure-repeats", str(args.repeats), "--interleave",
                    "--summary-csv", str(run_csv), "--no-pause"]
            print(f"\n##### {args.cell} {tag} {workload} run {k + 1}/{args.runs}", flush=True)
            if launch(argv, args.cpu, args.numa_node, cwd=REPO).wait() == 0:
                runs.append(str(run_csv))
        if runs:
            merge_runs(runs, str(merged))

    report(out, tag)


def report(out, tag):
    header = (f"{'variant':14}{'best single':>22}{'us':>10}{'best nested':>26}{'us':>10}"
              f"{'nested/single':>14}{'visited':>9}{'tested':>8}")
    lines = [f"panel {tag}", header]
    for workload in WORKLOADS:
        merged = out / f"{tag}_{workload}.csv"
        if not merged.exists():
            continue
        lat, vis, tst = {}, {}, {}
        for r in csv.DictReader(open(merged, newline="", encoding="utf-8-sig")):
            lat.setdefault(r["schema_name"], []).append(float(r["all_mean_ms"]) * 1000)
            vis.setdefault(r["schema_name"], []).append(float(r["mean_visited_nodes"]))
            tst.setdefault(r["schema_name"], []).append(float(r["mean_tested_points"]))
        lat = {n: statistics.median(v) for n, v in lat.items()}
        vis = {n: statistics.median(v) for n, v in vis.items()}
        tst = {n: statistics.median(v) for n, v in tst.items()}
        singles = [n for n in lat if is_single(n)]
        nested = [n for n in lat if not is_single(n)]
        if not singles or not nested:
            continue
        s = min(singles, key=lat.get)
        n = min(nested, key=lat.get)
        lines.append(f"{workload:14}{s:>22}{lat[s]:10.3f}{n:>26}{lat[n]:10.3f}{lat[n] / lat[s]:14.3f}"
                     f"{vis[n] / vis[s]:9.2f}{tst[n] / tst[s]:8.2f}")
        for name in sorted(lat, key=lat.get):
            lines.append(f"    {name:32}{lat[name]:10.3f}  visited {vis[name]:8.1f}  tested {tst[name]:9.0f}")
    text = "\n".join(lines)
    (out / f"report_{tag}.txt").write_text(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()

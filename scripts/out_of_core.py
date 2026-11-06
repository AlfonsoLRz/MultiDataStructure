#!/usr/bin/env python3
"""Does composition pay out of core? A schema panel timed with its leaves on disk (testing.md 2q).

E2 (2o) emulated an expensive node by spinning. Here engine-check --leaf-store writes each tree to a
file, and every query reads the leaves it scans with O_DIRECT: one synchronous read per leaf, the
per-node load of a file-per-node store (ISPRS 2023) or of a COPC chunk. Regimes, each timed on the
same built trees (--store-regimes):

  memory        in memory, the control
  0:memory      no cache, hierarchy in memory: every leaf is a device read
  0:paged       no cache, and every child group a query tests is a page read too (nesting's best case)
  0.01:memory   an LRU cache of 1% of the store
  0.01:paged    the same with a paged hierarchy

Workloads:
  trace   the cell's held-out pipeline trace, every regime
  boxes   the ISPRS benchmark (order_sweep.variant_rows): --range-boxes range queries and --count-boxes
          count queries over boxes of 20-40% of the extent per axis, in memory and cold; leaves of
          at least 2048 points, plus e9's pick (a leaf of 128 already takes ~26 s per box)

Stage A times a fixed panel (every primitive across the leaf range, the ISPRS schemas and their
leaf-matched singles, other nested schemas, e9's pick and best nested schema, the library defaults).
Stage B gives the best single of each cold regime quadtree, octree and HGrid prefixes, and counts a
nested schema as a win only if it is >= 3% faster in every process (the e9 confirm rule).

Disk: stage A runs groups of at most --group schemas per process, shuffled per run, and deletes the
group's stores when the process ends, so at most --group stores (1.6 GB each at 100M) exist at once;
stage B holds its 10 at once.

    python3 scripts/out_of_core.py --cell domfountain_100M --exe ../build_ooc/MultiDataStructure \\
        --data-root ../data --trace-root ../traces/curated --store-root ../store_tmp \\
        --out ../results/out_of_core --cpu 2 --numa-node 0 --device sdg

On makinon2's QLC SSD the measured read time of a store varied up to 3.5x between processes with the
same reads (2q, store_settle_check.py), so stage B can be re-run on its own (--stage-b-only, over the stage A results) with the best single
chosen on a device model (--io-model C0/MBPS, as out_of_core_report.py models it). Reads, bytes and
CPU time are all the model needs, so those stores can live in RAM (--buffered --store-root /dev/shm/...).
--leaves, --nested-leaves and --trace-regimes re-time one regime on a finer leaf grid, singles and nested
alike (2q: the 0.01:paged follow-up, where stage A's grid of 4x steps left nested schemas ahead).
--panel grid-check swaps stage A's panel for the wide fan-outs and quadtree-over-HGrid schemas at
--leaves (2q: whether stage B's one win on the settled drive's model is a missing single).
"""
import argparse
import csv
import json
import random
import shutil
import statistics
import time
from pathlib import Path

from order_sweep import PAPER_PANEL, variant_rows
from run_cells import REPO, launch, merge_runs, remap_cloud, remap_trace_dir
from workload_sweep import SM, XY, schema

LEAVES = (128, 512, 2048, 8192, 32768)
BOX_MIN_LEAF = 2048
TC = {"splitAt": "tight"}
QT = dict(XY, **TC)
REGIMES = {"trace": "memory,0:memory,0:paged,0.01:memory,0.01:paged", "boxes": "memory,0:memory"}
STAGE_B_REGIMES = {"trace": ["0:memory", "0:paged"], "boxes": ["0:memory"]}
DEFAULTS = ["configs/schemas/octree.json", "configs/schemas/kdtree.json"]
BOX_SEED = 20261101  # order_sweep's: the first boxes are those of the in-memory ISPRS regime (2l)

KD48 = lambda leaf: schema(f"kd48l{leaf}sm", leaf, [("KDTree", 48, SM)])
# e9's pick and best nested finalist per 100M cell (results/eval_traces/matrix_e9_makinon2/<cell>_pick.json),
# rebuilt from their names; the generated schemas themselves were deleted with the server results.
E9 = {
    "domfountain_100M": (KD48(128), schema("ot2_kd48l128sm", 128, [("Octree", 2, {}), ("KDTree", 48, SM)])),
    "lille_100M": (KD48(64), schema("ot2_kd48l128sm", 128, [("Octree", 2, {}), ("KDTree", 48, SM)])),
    "sanandreas_100M": (schema("qt24l32", 32, [("QuadTree", 24, {})]),
                        schema("qt9_ot24l32", 32, [("QuadTree", 9, {}), ("Octree", 24, {})])),
    "waikato_hamilton_100M": (schema("qt24l64", 64, [("QuadTree", 24, {})]),
                              schema("kd6_qt24l128tc", 128, [("KDTree", 6, {}), ("QuadTree", 24, TC)])),
    "galiciaforest_100M": (KD48(128), schema("qt6_kd48l128sm", 128, [("QuadTree", 6, {}), ("KDTree", 48, SM)])),
}
# The flatness rule's 2.5D pick (2p); its 3D pick, kd48l128sm, is in the grid.
RULE_FLAT = schema("qt16l128tc", 128, [("QuadTree", 16, QT)])


def singles(leaves):
    out = []
    for leaf in leaves:
        out += [KD48(leaf),
                schema(f"qt24l{leaf}tc", leaf, [("QuadTree", 24, QT)]),
                schema(f"ot24l{leaf}tc", leaf, [("Octree", 24, TC)]),
                schema(f"str64l{leaf}", leaf, [("STR", 11, {"fanout": 64})]),
                schema(f"hg12l{leaf}", leaf, [("HGrid", 12, {})])]
    return out


def panel(cell, workload, leaves=LEAVES, nested_leaves=(2048, 8192)):
    """(name, schema dict or repo-relative path) of stage A."""
    leaves = [leaf for leaf in leaves if workload == "trace" or leaf >= BOX_MIN_LEAF]
    schemas = singles(leaves) + list(PAPER_PANEL)
    for leaf in nested_leaves:
        schemas += [schema(f"qt6_kd48l{leaf}sm", leaf, [("QuadTree", 6, XY), ("KDTree", 48, SM)]),
                    schema(f"qt9_ot24l{leaf}", leaf, [("QuadTree", 9, XY), ("Octree", 24, {})])]
    if cell in E9:
        pick, nested = E9[cell]
        schemas.append(pick)
        if workload == "trace":
            schemas.append(nested)
    if workload == "trace":
        schemas.append(RULE_FLAT)
    unique = {}
    for s in schemas:
        unique.setdefault(s["name"], s)
    out = [(name, s) for name, s in unique.items()]
    if workload == "trace":
        out += [(Path(path).stem + "_default", path) for path in DEFAULTS]
    return out


def grid_check_panel(leaves):
    """2q: is stage B's quadtree-over-HGrid win (Lille, 0:paged, the settled drive's model) a missing
    single? The wide fan-outs (HGrid 64-way, RegularGrid 27-way, STR 64- and 27-way) at more leaves,
    and the winning prefixes over HGrid at each."""
    out = []
    for leaf in leaves:
        out += [schema(f"hg12l{leaf}", leaf, [("HGrid", 12, {})]),
                schema(f"rg12l{leaf}", leaf, [("RegularGrid", 12, {})]),
                schema(f"str64l{leaf}", leaf, [("STR", 11, {"fanout": 64})]),
                schema(f"str27l{leaf}", leaf, [("STR", 11, {"fanout": 27})])]
        out += [schema(f"qu{depth}_hg12l{leaf}", leaf, [("QuadTree", depth, XY), ("HGrid", 12, {})]) for depth in (2, 4)]
    return [(s["name"], s) for s in out]


def write_schemas(entries, folder):
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, s in entries:
        if isinstance(s, str):
            paths[name] = str(REPO / s)
        else:
            path = folder / f"{name}.json"
            path.write_text(json.dumps(s, indent=2))
            paths[name] = str(path)
    return paths


def write_trace(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_engine(args, cloud, trace, queries, schema_paths, regimes, store_dir, summary, warmup, repeats):
    argv = [args.exe, "--mode", "engine-check", "--input", cloud, "--input-trace", trace, "--schemas", ";".join(schema_paths),
            "--warmup", str(warmup), "--measure-repeats", str(repeats), "--interleave", "--summary-csv", str(summary),
            "--leaf-store", str(store_dir), "--store-regimes", regimes, "--no-pause"]
    if queries:
        argv += ["--queries", str(queries)]
    if args.buffered:
        argv.append("--store-allow-buffered")
    elif args.device:
        argv += ["--store-device", args.device]
    started = time.time()
    code = launch(argv, args.cpu, args.numa_node, cwd=REPO).wait()
    # The stores are this process's temporary files.
    shutil.rmtree(store_dir, ignore_errors=True)
    print(f"    {len(schema_paths)} schemas, {regimes}: {(time.time() - started) / 60:.1f} min, exit {code}", flush=True)
    return code


def medians(csv_path, column, io_model=None):
    """(schema_name, store_regime) -> median over processes of `column`, plus the per-process values.
    With io_model (C0 us, MB/s), the latency of the whole query mix with its I/O modeled instead."""
    values = {}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            value = float(row[column])
            if io_model:
                io_us = float(row["store_reads"] or 0) * io_model[0] + float(row["store_bytes"] or 0) / io_model[1]
                value = float(row["all_mean_ms"]) - float(row["store_io_ms"] or 0) + io_us * 1e-3
            values.setdefault((row["schema_name"], row["store_regime"]), []).append((int(row["test_run"]), value))
    return {key: (statistics.median(v for _, v in runs), dict(runs)) for key, runs in values.items()}


def is_single(s):
    return isinstance(s, str) or len(s["levels"]) == 1


def stage_b_schemas(best):
    """Nested schemas over the best single: a quadtree, octree or HGrid prefix on top of it."""
    leaf = best["buildPolicy"]["leafCapacity"]
    out = []
    for kind, extra, depths in (("QuadTree", XY, (2, 4, 8)), ("Octree", {}, (2, 4, 8)), ("HGrid", {}, (1, 2, 3))):
        for depth in depths:
            prefix = {"type": kind, "numLevels": depth, "leafCapacity": leaf, "minPointsToSplit": max(2, leaf // 4)}
            prefix.update(extra)
            nested = json.loads(json.dumps(best))
            nested["name"] = f"B{kind[:2].lower()}{depth}_{best['name']}"
            nested["levels"] = [prefix] + nested["levels"]
            nested["buildPolicy"]["maxDepth"] = sum(level["numLevels"] for level in nested["levels"])
            out.append((nested["name"], nested))
    return out


def run_workload(args, cell, workload, cloud, trace, queries, out):
    leaves = [int(v) for v in args.leaves.split(",")]
    entries = grid_check_panel(leaves) if args.panel == "grid-check" else \
        panel(cell, workload, leaves, [int(v) for v in args.nested_leaves.split(",")])
    paths = write_schemas(entries, out / "schemas")
    regimes = args.trace_regimes if workload == "trace" and args.trace_regimes else REGIMES[workload]
    b_regimes = [r for r in regimes.split(",") if r != "memory"] if workload == "trace" and args.trace_regimes else STAGE_B_REGIMES[workload]
    runs = args.box_runs if workload == "boxes" else args.runs
    # Boxes are slow out of core, so their one pass is also the instrumented one; the trace keeps a
    # warm-up pass, which the in-memory control needs.
    warmup, repeats = (0, 1) if workload == "boxes" else (1, args.repeats)
    if not args.stage_b_only:
        run_csvs = []
        print(f"  {workload}: {len(entries)} schemas, {runs} runs, groups of {args.group}", flush=True)
        for run in range(runs):
            summary = out / f"{workload}_r{run}.csv"
            summary.unlink(missing_ok=True)
            names = [name for name, _ in entries]
            random.Random(1000 + run).shuffle(names)
            for g in range(0, len(names), args.group):
                group = [paths[name] for name in names[g:g + args.group]]
                run_engine(args, cloud, trace, queries, group, regimes,
                           Path(args.store_root) / cell / f"{workload}_r{run}_g{g}", summary, warmup, repeats)
            run_csvs.append(summary)
        merge_runs(run_csvs, out / f"{workload}.csv")

    # Stage B: nested schemas over each cold regime's best single (measured, or on the device model).
    column = "range_mean_ms" if workload == "boxes" else "all_mean_ms"
    io_model = tuple(map(float, args.io_model.split("/"))) if args.io_model else None
    by_name = dict(entries)
    timing = medians(out / f"{workload}.csv", column, io_model)
    for regime in b_regimes:
        candidates = [(t, name) for (name, r), (t, _) in timing.items()
                      if r == regime and name in by_name and not isinstance(by_name[name], str) and is_single(by_name[name])]
        if not candidates:
            continue
        best_name = min(candidates)[1]
        nested = stage_b_schemas(by_name[best_name])
        tag = f"stageB_{workload}_{regime.replace(':', '-')}{args.stage_b_tag}"
        b_paths = write_schemas([(best_name, by_name[best_name])] + nested, out / "schemas")
        order = [b_paths[best_name]] + [b_paths[name] for name, _ in nested]
        print(f"  stage B, {workload} {regime}: {best_name} against {len(nested)} prefixes", flush=True)
        b_csvs = []
        for run in range(args.stage_b_runs):
            summary = out / f"{tag}_r{run}.csv"
            summary.unlink(missing_ok=True)
            shift = run * len(order) // args.stage_b_runs
            run_engine(args, cloud, trace, queries, order[shift:] + order[:shift], regime,
                       Path(args.store_root) / cell / f"{tag}_r{run}", summary, warmup, repeats)
            b_csvs.append(summary)
        merge_runs(b_csvs, out / f"{tag}.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True, help="';'-separated cells")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--cells", default="configs/datasets/curated_cells.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--trace-root", default="")
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workloads", default="trace,boxes")
    parser.add_argument("--runs", type=int, default=2, help="stage A processes per schema, trace")
    parser.add_argument("--box-runs", type=int, default=2, help="stage A processes per schema, boxes")
    parser.add_argument("--stage-b-runs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2, help="timed passes per process, trace")
    parser.add_argument("--range-boxes", type=int, default=10)
    parser.add_argument("--count-boxes", type=int, default=20)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--queries", type=int, default=0, help="trace queries (default: the cell's)")
    parser.add_argument("--device", default="", help="block device of --store-root (e.g. sdg), to flag other reads")
    parser.add_argument("--stage-b-only", action="store_true", help="skip stage A; stage B over the existing <workload>.csv")
    parser.add_argument("--io-model", default="", help="C0/MBPS: choose stage B's single on reads * C0 us + bytes / MBPS")
    parser.add_argument("--stage-b-tag", default="", help="suffix of the stage B files (e.g. _sata)")
    parser.add_argument("--buffered", action="store_true", help="buffered store reads (a RAM-backed --store-root)")
    parser.add_argument("--leaves", default=",".join(map(str, LEAVES)), help="leaf sizes of the singles")
    parser.add_argument("--nested-leaves", default="2048,8192", help="leaf sizes of qt6_kd48 and qt9_ot24")
    parser.add_argument("--trace-regimes", default="", help="the trace's regimes instead (stage B over the cold ones)")
    parser.add_argument("--panel", default="default", choices=["default", "grid-check"], help="grid-check: grid_check_panel at --leaves")
    parser.add_argument("--cpu", type=int, default=-1)
    parser.add_argument("--numa-node", type=int, default=-1)
    args = parser.parse_args()
    args.exe = str(Path(args.exe).resolve())
    args.store_root = str(Path(args.store_root).resolve())

    cells = {c["cell"]: c for c in json.loads((REPO / args.cells).read_text())["cells"]}
    for name in args.cell.split(";"):
        cell = cells[name]
        out = Path(args.out).resolve() / name
        out.mkdir(parents=True, exist_ok=True)
        cloud = remap_cloud(cell["mdspc_path"], args.data_root)
        trace = f"{remap_trace_dir(cell['trace_dir'], name, args.trace_root)}/pipeline_trace_test.csv"
        print(f"\n### {name} [{time.strftime('%H:%M:%S')}]", flush=True)
        for workload in args.workloads.split(","):
            if workload == "trace":
                run_workload(args, name, "trace", cloud, trace, args.queries or cell["queries"], out)
            elif workload == "boxes":
                with open(trace, newline="") as handle:
                    rows = list(csv.DictReader(handle))
                boxes = (variant_rows(rows, "bigbox", 0.0, BOX_SEED, args.range_boxes) +
                         variant_rows(rows, "bigbox_count", 0.0, BOX_SEED, args.count_boxes))
                for i, row in enumerate(boxes):
                    row["query_id"] = i
                box_trace = out / "boxes_trace.csv"
                write_trace(boxes, box_trace)
                run_workload(args, name, "boxes", cloud, str(box_trace), 0, out)
        print(f"### {name} done [{time.strftime('%H:%M:%S')}]", flush=True)


if __name__ == "__main__":
    main()

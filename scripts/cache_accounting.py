#!/usr/bin/env python3
"""Where a query's time goes: a cache accounting of node visits (testing.md 2s).

Nesting saves node visits (2l: 0.976 over the e9 cells, 0.80-0.92 on TLS/MLS) but not time, and the
ledger's reading is that the visits it saves are the root-to-mid-level ones, in cache for every query,
while the time goes to the cold bottom levels and the leaf scans. This battery measures that and tries
the two layout fixes that target the cold visits instead:

  arms   base      the engine as it is
         huge      --pool-huge-pages: node pool and point arrays on 2 MB pages (no TLB miss per cold visit)
         prefetch  --prefetch-children: a pushed child's child group or leaf arrays are prefetched
         both

Per cell the panel is e9's pick and best nested finalist, the flatness rule's pick, a wide single
(ot24l128tc, hg12l2048), the library defaults, and 2q's stage B winner where the paged regimes had
one. Each process is one arm over a group of at most --group schemas, interleaved (engine-check
--interleave), and the arms rotate across runs so a slow stretch of the machine hits them all. The
instrumented pass of every process also records visits per depth and the nanoseconds charged to each
depth (engine-check's visits_by_depth / visit_ns_by_depth columns), so the report can price the
levels nesting touches.

    python3 scripts/cache_accounting.py --cells "domfountain_100M;sanandreas_100M;lille_100M" \\
        --exe ../build_cache/MultiDataStructure --data-root ../data --trace-root ../traces/curated \\
        --out ../results/cache_accounting --cpu 2 --numa-node 0

The 25M cells take the 100M cell's e9 picks of the same scene (e9 ran the 25M tier under the
earlier engine); their clouds live in a scratch copy, hence a separate --data-root/--trace-root.
"""
import argparse
import json
import random
import time
from pathlib import Path

from out_of_core import DEFAULTS, E9, KD48, TC, write_schemas
from run_cells import REPO, launch, merge_runs, remap_cloud, remap_trace_dir
from workload_sweep import SM, XY, schema

ARMS = {"base": [], "huge": ["--pool-huge-pages"], "prefetch": ["--prefetch-children"],
        "both": ["--pool-huge-pages", "--prefetch-children"]}
# e9's flatness_score per scene (2p); all three are under the rule's 0.60 split, so its pick is kd48l128sm.
FLATNESS = {"domfountain": 0.41, "sanandreas": 0.53, "lille": 0.48}
RULE_FLAT = schema("qt16l128tc", 128, [("QuadTree", 16, dict(XY, **TC))])
# 2q's stage B winners with the hierarchy paged (there were none with it in memory): Lille on the
# settled drive without a cache and on SATA with the 1% cache, DomFountain on SATA with the cache.
STAGE_B = {"lille": [schema("qu4_hg12l2048", 2048, [("QuadTree", 4, XY), ("HGrid", 12, {})]),
                     schema("qu8_kd48l2048sm", 2048, [("QuadTree", 8, XY), ("KDTree", 48, SM)])],
           "domfountain": [schema("qu8_kd48l4096sm", 4096, [("QuadTree", 8, XY), ("KDTree", 48, SM)])]}


def panel(cell):
    scene = cell.split("_")[0]
    pick, nested = E9[f"{scene}_100M"]
    flat = FLATNESS.get(scene, 0.0)
    schemas = [pick, nested, RULE_FLAT if flat > 0.60 else KD48(128),
               schema("ot24l128tc", 128, [("Octree", 24, TC)]),
               schema("hg12l2048", 2048, [("HGrid", 12, {})])]
    schemas += STAGE_B.get(scene, [])
    unique = {}
    for s in schemas:
        unique.setdefault(s["name"], s)
    return [(name, s) for name, s in unique.items()] + [(Path(path).stem + "_default", path) for path in DEFAULTS]


def run_engine(args, cloud, trace, queries, schema_paths, arm, summary):
    argv = [args.exe, "--mode", "engine-check", "--input", cloud, "--input-trace", trace, "--schemas", ";".join(schema_paths),
            "--queries", str(queries), "--warmup", "1", "--measure-repeats", str(args.repeats), "--interleave",
            "--summary-csv", str(summary), "--no-pause"] + ARMS[arm]
    started = time.time()
    code = launch(argv, args.cpu, args.numa_node, cwd=REPO).wait()
    print(f"    {arm}: {len(schema_paths)} schemas, {(time.time() - started) / 60:.1f} min, exit {code}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", required=True, help="';'-separated cells")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--cell-config", default="configs/datasets/curated_cells.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--trace-root", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--runs", type=int, default=3, help="processes per arm and schema group")
    parser.add_argument("--repeats", type=int, default=2, help="timed passes per process")
    parser.add_argument("--group", type=int, default=5, help="schemas per process")
    parser.add_argument("--queries", type=int, default=0, help="trace queries (default: the cell's)")
    parser.add_argument("--cpu", type=int, default=-1)
    parser.add_argument("--numa-node", type=int, default=-1)
    args = parser.parse_args()
    args.exe = str(Path(args.exe).resolve())
    arms = args.arms.split(",")

    cells = {c["cell"]: c for c in json.loads((REPO / args.cell_config).read_text())["cells"]}
    for name in args.cells.split(";"):
        cell = cells[name]
        out = Path(args.out).resolve() / name
        out.mkdir(parents=True, exist_ok=True)
        cloud = remap_cloud(cell["mdspc_path"], args.data_root)
        trace = f"{remap_trace_dir(cell['trace_dir'], name, args.trace_root)}/pipeline_trace_test.csv"
        entries = panel(name)
        paths = write_schemas(entries, out / "schemas")
        print(f"\n### {name}: {len(entries)} schemas, arms {', '.join(arms)}, {args.runs} runs [{time.strftime('%H:%M:%S')}]", flush=True)
        run_csvs = {arm: [] for arm in arms}
        for run in range(args.runs):
            names = [n for n, _ in entries]
            random.Random(2000 + run).shuffle(names)
            groups = [names[g:g + args.group] for g in range(0, len(names), args.group)]
            for arm in arms[run % len(arms):] + arms[:run % len(arms)]:
                summary = out / f"{arm}_r{run}.csv"
                summary.unlink(missing_ok=True)
                for group in groups:
                    run_engine(args, cloud, trace, args.queries or cell["queries"], [paths[n] for n in group], arm, summary)
                run_csvs[arm].append(summary)
        for arm in arms:
            merge_runs(run_csvs[arm], out / f"{arm}.csv")
        print(f"### {name} done [{time.strftime('%H:%M:%S')}]", flush=True)


if __name__ == "__main__":
    main()

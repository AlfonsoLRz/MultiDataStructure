#!/usr/bin/env python3
"""The leaf-size lever in both page modes, and the headline panel timed sequentially (testing.md 2u).

Every timing before this battery was taken on 4 KiB pages, and 2s found that 2 MB pages speed the large
pools up by 8-30% and move the optimum towards smaller leaves. Two panels per cell, each timed under the
engine's base arm and under --pool-huge-pages:

  leaf    kd48l{16..1024}sm and qt24l{16..1024}tc (the rule's two primitives, open-ended depth, seven leaf
          sizes each) plus the two library defaults: the leaf lever of each primitive in both page modes.
  panel   the paper's headline schemas: the library defaults, the rule's two picks, e9's pick and best
          nested finalist (from --pick-dir/<cell>/e9pick_*.json and --nested-dir/<cell>/*.json), and 2q's
          stage B winners where the paged regimes had one.

Protocol: engine-check sequential (no --interleave: each schema is built, warmed up and timed before the
next is built, which 2r found within 1-4% of timing it alone), --warmup 1, --repeats timed passes, the
cell's held-out trace at its query budget, --rounds processes per arm with the schema order rotated, arms
rotated across rounds, pinned like run_cells.py. Readers take the median over rounds.

    python3 scripts/leaf_pages.py --mode leaf --cells "domfountain_100M;lille_100M" --exe ../build_paper/MultiDataStructure \\
        --data-root ../data --trace-root ../traces/curated --out ../results/leaf_pages --cpu 20 --numa-node 1
    python3 scripts/leaf_pages.py --mode panel --pick-dir ../ext_schemas --nested-dir ../paper_nested ...
"""
import argparse
import json
import time
from pathlib import Path

from cache_accounting import ARMS, STAGE_B
from out_of_core import DEFAULTS, KD48, TC, write_schemas
from run_cells import REPO, launch, merge_runs, remap_cloud, remap_trace_dir
from workload_sweep import XY, schema

LEAVES = [16, 32, 64, 128, 256, 512, 1024]
QT = dict(XY, **TC)


def leaf_panel():
    entries = [(f"kd48l{leaf}sm", KD48(leaf)) for leaf in LEAVES]
    entries += [(f"qt24l{leaf}tc", schema(f"qt24l{leaf}tc", leaf, [("QuadTree", 24, QT)])) for leaf in LEAVES]
    return entries + [(Path(p).stem + "_default", p) for p in DEFAULTS]


def headline_panel(cell, pick_dir, nested_dir):
    scene = cell.split("_")[0]
    entries = [(Path(p).stem + "_default", p) for p in DEFAULTS]
    entries += [("kd48l128sm", KD48(128)), ("qt16l128tc", schema("qt16l128tc", 128, [("QuadTree", 16, QT)]))]
    for folder, pattern in ((pick_dir, "e9pick_*.json"), (nested_dir, "*.json")):
        if folder:
            for path in sorted(Path(folder, cell).glob(pattern)):
                s = json.loads(path.read_text())
                entries.append((s["name"], s))
    entries += [(s["name"], s) for s in STAGE_B.get(scene, [])]
    unique = {}
    for name, s in entries:
        unique.setdefault(name, s)
    return list(unique.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["leaf", "panel"], required=True)
    parser.add_argument("--cells", required=True, help="';'-separated cells")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--cell-config", default="configs/datasets/curated_cells.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--trace-root", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pick-dir", default="", help="panel mode: folder with <cell>/e9pick_*.json (external_baselines.py prepare)")
    parser.add_argument("--nested-dir", default="", help="panel mode: folder with <cell>/<best nested finalist>.json")
    parser.add_argument("--arms", default="base,huge")
    parser.add_argument("--rounds", type=int, default=3, help="processes per arm")
    parser.add_argument("--repeats", type=int, default=10, help="timed passes per schema")
    parser.add_argument("--queries", type=int, default=0, help="trace queries (default: the cell's budget)")
    parser.add_argument("--cpu", type=int, default=-1)
    parser.add_argument("--numa-node", type=int, default=-1)
    args = parser.parse_args()
    args.exe = str(Path(args.exe).resolve())
    arms = args.arms.split(",")

    cells = {c["cell"]: c for c in json.loads((REPO / args.cell_config).read_text())["cells"]}
    for name in args.cells.split(";"):
        cell = cells[name]
        out = Path(args.out).resolve() / args.mode / name
        out.mkdir(parents=True, exist_ok=True)
        cloud = remap_cloud(cell["mdspc_path"], args.data_root)
        trace = f"{remap_trace_dir(cell['trace_dir'], name, args.trace_root)}/pipeline_trace_test.csv"
        entries = leaf_panel() if args.mode == "leaf" else headline_panel(name, args.pick_dir, args.nested_dir)
        paths = write_schemas(entries, out / "schemas")
        names = [n for n, _ in entries]
        print(f"\n### {args.mode} {name}: {len(entries)} schemas ({', '.join(names)}), arms {', '.join(arms)}, "
              f"{args.rounds} rounds [{time.strftime('%H:%M:%S')}]", flush=True)
        run_csvs = {arm: [] for arm in arms}
        for run in range(args.rounds):
            shift = (run * len(names)) // args.rounds
            order = names[shift:] + names[:shift]
            for arm in arms[run % len(arms):] + arms[:run % len(arms)]:
                summary = out / f"{arm}_r{run}.csv"
                summary.unlink(missing_ok=True)
                argv = [args.exe, "--mode", "engine-check", "--input", cloud, "--input-trace", trace,
                        "--schemas", ";".join(paths[n] for n in order), "--queries", str(args.queries or cell["queries"]),
                        "--warmup", "1", "--measure-repeats", str(args.repeats), "--summary-csv", str(summary), "--no-pause"] + ARMS[arm]
                started = time.time()
                code = launch(argv, args.cpu, args.numa_node, cwd=REPO).wait()
                print(f"  round {run} {arm}: {(time.time() - started) / 60:.1f} min, exit {code}", flush=True)
                run_csvs[arm].append(summary)
        for arm in arms:
            merge_runs([p for p in run_csvs[arm] if p.exists()], out / f"{arm}.csv")
        print(f"### {name} done [{time.strftime('%H:%M:%S')}]", flush=True)


if __name__ == "__main__":
    main()

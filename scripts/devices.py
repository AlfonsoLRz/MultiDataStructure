#!/usr/bin/env python3
"""Measured storage devices, hierarchy placement, cache size and node layout (testing.md 2v, 2w).

2q modelled its devices because makinon2's QLC drive timed erratically. This battery measures them:
the store of every schema is written to a directory on the device under test and read back with
direct (unbuffered) I/O, O_DIRECT on Linux and FILE_FLAG_NO_BUFFERING on Windows. One run of this
script measures one device on one machine:

  makinon2     --device-name sata --store-root /home/alfonsolr/mds_store  --block-device sdb   (Crucial MX500)
               --device-name qlc  --store-root /home2/alfonsolr/mds/store_tmp --block-device sdg (PNY CS900)
  workstation  --device-name nvme --store-root C:/mds_store   (WD SN850X)
               --device-name hdd  --store-root D:/mds_store   (WD Red Plus)

Phases, each with the depth-first and the page-blocked pool (--pool-layout, one process per layout):

  placement  the full panel in three regimes: in memory, leaves on the device with the hierarchy in
             memory (0:memory), leaves and hierarchy on the device (0:paged)
  cache      a smaller panel on a dense pipeline segment (scripts/dense_trace.py: every point of
             --dense-windows runs of --dense-window consecutive points, stage by stage, which is how a
             pipeline reuses leaves), with no cache and an LRU cache of 0.001, 0.01, 0.1 and 1% of the store (the smoke test on Lille
             100M found 0.1% already removing 99% of the reads: the working set of a dense segment is small),
             hierarchy in memory or paged (leave it out of --phases on the hard disk)

Panel (placement): kd-tree (sliding midpoint), quadtree, octree, hierarchical grid and STR-64 at leaves
128 to 32768, the two library defaults, the flatness rule's 2.5D pick, the searched best single and its
best nested finalist (out_of_core.E9). Cache phase: kd-tree and quadtree at 512, 2048 and 8192, the
grid at 2048, the defaults, the searched best single and the nested finalist.

Protocol: sequential engine-check (each schema built, its store written, every regime timed, then the
next schema), warm-up 1 and --repeats timed passes per regime, the cell's held-out trace at --queries,
--rounds rounds with the schema order shuffled per round. Groups of --group schemas per process; the
process's stores are deleted when it exits. Each group writes its own CSV through a .partial file, so
an interrupted battery resumes where it stopped, and --stop-at HH:MM stops starting new processes
after that time (the workstation runs only outside working hours).

    python scripts/devices.py --cells "lille_100M;sanandreas_100M" --exe x64/Release/MultiDataStructure.exe \\
        --device-name nvme --store-root C:/mds_store --out results/devices --cpu 4 --stop-at 07:30
"""
import argparse
import datetime
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from out_of_core import DEFAULTS, E9, KD48, QT, RULE_FLAT, TC, write_schemas
from run_cells import REPO, launch, remap_cloud, remap_trace_dir
from workload_sweep import schema

PLACEMENT_REGIMES = "memory,0:memory,0:paged"
CACHE_REGIMES = ",".join(f"{f}:{h}" for h in ("memory", "paged") for f in ("0", "0.00001", "0.0001", "0.001", "0.01"))
LEAVES = (128, 512, 2048, 8192, 32768)


def kinds(leaf):
    return {"kd": KD48(leaf), "qt": schema(f"qt24l{leaf}tc", leaf, [("QuadTree", 24, QT)]),
            "ot": schema(f"ot24l{leaf}tc", leaf, [("Octree", 24, TC)]), "hg": schema(f"hg12l{leaf}", leaf, [("HGrid", 12, {})]),
            "str": schema(f"str64l{leaf}", leaf, [("STR", 11, {"fanout": 64})])}


def placement_panel(cell):
    entries = [(s["name"], s) for leaf in LEAVES for s in kinds(leaf).values()]
    return entries + common(cell)


def cache_panel(cell):
    entries = [(kinds(leaf)[k]["name"], kinds(leaf)[k]) for leaf in (512, 2048, 8192) for k in ("kd", "qt")]
    entries.append((kinds(2048)["hg"]["name"], kinds(2048)["hg"]))
    return entries + common(cell)


def common(cell):
    out = [(RULE_FLAT["name"], RULE_FLAT)]
    if cell in E9:
        pick, nested = E9[cell]
        out += [(pick["name"], pick), (nested["name"], nested)]
    out += [(Path(p).stem + "_default", p) for p in DEFAULTS]
    unique = {}
    for name, s in out:
        unique.setdefault(name, s)
    return list(unique.items())


def past(stop_at):
    """True once the wall clock is past HH:MM of the night the battery started in (a stop time before
    noon belongs to the next morning)."""
    if not stop_at:
        return False
    hour, minute = map(int, stop_at.split(":"))
    now = datetime.datetime.now()
    limit = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if hour < 12 and now.hour >= 12:
        return False
    return now >= limit


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", required=True, help="';'-separated cells")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--cell-config", default="configs/datasets/curated_cells.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--trace-root", default="")
    parser.add_argument("--device-name", required=True, help="nvme, sata, qlc or hdd: the folder and the label")
    parser.add_argument("--store-root", required=True, help="a directory on the device under test")
    parser.add_argument("--block-device", default="", help="Linux block device of --store-root (e.g. sdb): flags other reads")
    parser.add_argument("--out", required=True)
    parser.add_argument("--layouts", default="dfs,paged")
    parser.add_argument("--phases", default="placement,cache")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--queries", type=int, default=0, help="trace queries (default: the cell's budget)")
    parser.add_argument("--stop-at", default="", help="HH:MM: start no new process after this time")
    parser.add_argument("--only", default="", help="','-separated schema names to keep (smoke tests)")
    parser.add_argument("--dense-window", type=int, default=1000, help="consecutive points per dense run (cache phase)")
    parser.add_argument("--dense-windows", type=int, default=3)
    parser.add_argument("--cache-repeats", type=int, default=1, help="timed passes per regime in the cache phase")
    parser.add_argument("--cpu", type=int, default=-1)
    parser.add_argument("--numa-node", type=int, default=-1)
    args = parser.parse_args()
    args.exe = str(Path(args.exe).resolve())

    cells = {c["cell"]: c for c in json.loads((REPO / args.cell_config).read_text())["cells"]}
    root = Path(args.out).resolve() / args.device_name
    panels = {"placement": (placement_panel, PLACEMENT_REGIMES), "cache": (cache_panel, CACHE_REGIMES)}
    for run in range(args.rounds):
        for name in args.cells.split(";"):
            cell = cells[name]
            out = root / name
            cloud = remap_cloud(cell["mdspc_path"], args.data_root)
            trace_dir = remap_trace_dir(cell['trace_dir'], name, args.trace_root)
            trace = f"{trace_dir}/pipeline_trace_test.csv"
            dense = out / "dense_trace.csv"
            if "cache" in args.phases and not dense.exists():
                out.mkdir(parents=True, exist_ok=True)
                subprocess.run([sys.executable, str(Path(__file__).with_name("dense_trace.py")), "--cloud", cloud,
                                "--workload", f"{trace_dir}/pipeline_workload.json", "--out", str(dense),
                                "--window", str(args.dense_window), "--windows", str(args.dense_windows)], check=True)
            for phase in args.phases.split(","):
                make, regimes = panels[phase]
                entries = make(name)
                if args.only:
                    entries = [(n, s) for n, s in entries if n in args.only.split(",")]
                paths = write_schemas(entries, out / "schemas")
                names = [n for n, _ in entries]
                random.Random(3000 + run).shuffle(names)
                for layout in args.layouts.split(","):
                    for g in range(0, len(names), args.group):
                        summary = out / f"{phase}_{layout}_r{run}_g{g // args.group}.csv"
                        if summary.exists():
                            continue
                        if past(args.stop_at):
                            print(f"stop time {args.stop_at} reached; rerun to resume", flush=True)
                            return
                        partial = summary.with_suffix(".partial.csv")
                        partial.unlink(missing_ok=True)
                        store_dir = Path(args.store_root) / f"{name}_{phase}_{layout}_r{run}_g{g}"
                        dense_phase = phase == "cache"
                        argv = [args.exe, "--mode", "engine-check", "--input", cloud, "--input-trace", str(dense) if dense_phase else trace,
                                "--schemas", ";".join(paths[n] for n in names[g:g + args.group]),
                                "--warmup", "1", "--measure-repeats", str(args.cache_repeats if dense_phase else args.repeats),
                                "--summary-csv", str(partial), "--leaf-store", str(store_dir), "--store-regimes", regimes,
                                "--pool-layout", layout, "--no-pause"]
                        # The dense segment is replayed whole, in order; the curated trace at the cell's stride.
                        dense_count = sum(1 for _ in open(dense)) - 1 if dense_phase else 0
                        argv += ["--queries", str(dense_count if dense_phase else (args.queries or cell["queries"]))]
                        if args.block_device:
                            argv += ["--store-device", args.block_device]
                        started = time.time()
                        code = launch(argv, args.cpu, args.numa_node, cwd=REPO).wait()
                        shutil.rmtree(store_dir, ignore_errors=True)
                        print(f"[{time.strftime('%H:%M:%S')}] {args.device_name} {name} {phase} {layout} r{run} g{g // args.group}: "
                              f"{(time.time() - started) / 60:.1f} min, exit {code}", flush=True)
                        if code == 0 and partial.exists():
                            partial.rename(summary)
    (root / "done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Throughput with many threads, local or remote memory, small or huge pages (testing.md 2x).

Every other battery times one query at a time on one core. A pipeline stage runs on all cores, where
memory bandwidth, not latency, may set the pace and change which structure wins. Here the held-out
trace is replayed in order (batch-check --replay-trace, cycled to --batch queries) by N OpenMP threads
over one index (backend cpuN, schedule dynamic 64), for:

  threads   1, 2, 4, 9, 18 (one per physical core of the socket), 36 (with hyperthreads)
  memory    on the socket's own node (local) or on the other socket's (remote)
  pages     4 KiB, or 2 MB for pools above 4 MB (--pool-huge-pages)
  layout    depth-first or page-blocked pool (--pool-layout)

over the library defaults, the flatness rule's pick, the searched best single, the hierarchical grid
and the nested finalist; and nanoflann (its own --threads mode) at its default leaf and its best leaf
of 2r under the same thread counts, cpus and memory nodes. One process per (configuration, round);
--rounds rounds rotate the order. Run only when the socket is idle: the battery checks the load
average before each process and waits (up to --max-wait minutes) while it is above --max-load.

    python3 scripts/threads.py --cells "lille_100M;sanandreas_100M" --exe ../build_v2/MultiDataStructure \\
        --nanoflann ../baselines_bin/nanoflann_point_baseline --data-root ../data --trace-root ../traces/curated \\
        --out ../results/threads --socket 0
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from out_of_core import DEFAULTS, E9, KD48, RULE_FLAT, write_schemas
from run_cells import REPO, remap_cloud, remap_trace_dir
from workload_sweep import schema

THREADS = (1, 2, 4, 9, 18, 36)
# makinon2: socket 0 = physical cores 0-17 with hyperthreads 36-53, node 0; socket 1 = 18-35 / 54-71, node 1.
SOCKETS = {0: (list(range(0, 18)), list(range(36, 54)), 0), 1: (list(range(18, 36)), list(range(54, 72)), 1)}
NANOFLANN_LEAVES = "10,32"


def panel(cell):
    entries = [(Path(p).stem + "_default", p) for p in DEFAULTS]
    entries += [(RULE_FLAT["name"], RULE_FLAT), (KD48(128)["name"], KD48(128)),
                ("hg12l2048", schema("hg12l2048", 2048, [("HGrid", 12, {})]))]
    if cell in E9:
        pick, nested = E9[cell]
        entries += [(pick["name"], pick), (nested["name"], nested)]
    unique = {}
    for name, s in entries:
        unique.setdefault(name, s)
    return list(unique.items())


def cpus(socket, threads):
    physical, hyper, node = SOCKETS[socket]
    order = physical + hyper
    return ",".join(map(str, order[:threads])), node


def wait_for_idle(max_load, max_wait_min):
    waited = 0
    while os.getloadavg()[0] > max_load and waited < max_wait_min * 60:
        time.sleep(60)
        waited += 60
    return os.getloadavg()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--exe", required=True)
    parser.add_argument("--nanoflann", default="", help="nanoflann_point_baseline built with OpenMP (empty: skip)")
    parser.add_argument("--cell-config", default="configs/datasets/curated_cells.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--trace-root", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--socket", type=int, default=0)
    parser.add_argument("--threads", default=",".join(map(str, THREADS)))
    parser.add_argument("--memory", default="local,remote")
    parser.add_argument("--pages", default="small,huge")
    parser.add_argument("--layouts", default="dfs,paged")
    parser.add_argument("--batch", type=int, default=300000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-load", type=float, default=8.0, help="wait while the 1-minute load average is above this")
    parser.add_argument("--max-wait", type=float, default=120.0, help="minutes to wait for an idle machine before running anyway (logged)")
    args = parser.parse_args()
    args.exe = str(Path(args.exe).resolve())

    cells = {c["cell"]: c for c in json.loads((REPO / args.cell_config).read_text())["cells"]}
    root = Path(args.out).resolve()
    configs = [(t, m, p, l) for t in map(int, args.threads.split(",")) for m in args.memory.split(",")
               for p in args.pages.split(",") for l in args.layouts.split(",")]
    for run in range(args.rounds):
        shift = (run * len(configs)) // args.rounds
        for name in args.cells.split(";"):
            cell = cells[name]
            out = root / name
            cloud = remap_cloud(cell["mdspc_path"], args.data_root)
            trace = f"{remap_trace_dir(cell['trace_dir'], name, args.trace_root)}/pipeline_trace_test.csv"
            paths = write_schemas(panel(name), out / "schemas")
            for threads, memory, pages, layout in configs[shift:] + configs[:shift]:
                cpu_list, node = cpus(args.socket, threads)
                mem_node = node if memory == "local" else 1 - node
                tag = f"t{threads}_{memory}_{pages}_{layout}_r{run}"
                jobs = [("mds", [args.exe, "--mode", "batch-check", "--input", cloud, "--input-trace", trace, "--replay-trace",
                                 "--schemas", ";".join(paths.values()), "--backends", "cpuN", "--threads", str(threads),
                                 "--batch-sizes", str(args.batch), "--batch-repeats", str(args.repeats), "--pool-layout", layout,
                                 "--no-pause"] + (["--pool-huge-pages"] if pages == "huge" else []))]
                if args.nanoflann and layout == "dfs" and pages == "small":
                    jobs.append(("nanoflann", [str(Path(args.nanoflann).resolve()), "--input", cloud, "--query-trace", trace,
                                               "--threads", str(threads), "--batch", str(args.batch), "--repeats", str(args.repeats),
                                               "--leaf-sizes", NANOFLANN_LEAVES]))
                for tool, argv in jobs:
                    summary = out / f"{tool}_{tag}.csv"
                    if summary.exists():
                        continue
                    partial = summary.with_suffix(".partial.csv")
                    partial.unlink(missing_ok=True)
                    load = wait_for_idle(args.max_load, args.max_wait)
                    argv = argv + ["--summary-csv", str(partial)]
                    started = time.time()
                    code = subprocess.run(["numactl", f"--physcpubind={cpu_list}", f"--membind={mem_node}"] + argv, cwd=REPO).returncode
                    print(f"[{time.strftime('%H:%M:%S')}] {name} {tool} {tag}: {(time.time() - started) / 60:.1f} min, exit {code}, "
                          f"load before {load:.1f}", flush=True)
                    if code == 0 and partial.exists():
                        partial.rename(summary)
    (root / "done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


if __name__ == "__main__":
    main()

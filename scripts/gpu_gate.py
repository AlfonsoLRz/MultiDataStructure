#!/usr/bin/env python3
"""Gate table of the GPU pilot (plan 2026-11-02): does the schema matter on the GPU, and does nesting?

    python scripts/gpu_gate.py results/eval_traces/gpu_pilot_makinon2

Reads <cell>/<workload>.csv and <cell>/<workload>_depth.csv written by gpu_pilot.py (batch-check
summaries). Per cell, workload and batch size it reports, from end-to-end throughput:

  gpu spread     best / worst schema on the GPU (warp kernels, and thread kernels)
  cpu pick       throughput on the GPU of the schema the CPU would pick (best on cpu1, or cpuN where
                 cpu1 was not timed), as a fraction of the GPU's best: < 1 means hardware-aware
                 selection pays
  mixed/uniform  best nested (mixed fan-out) schema / best single schema on the GPU: > 1 means
                 composition pays
  warp/thread    best warp-kernel throughput / best thread-kernel throughput
  gpu/cpuN       best GPU / best all-core CPU, end to end and kernel-only
  depth          best depth-capped single / best other single on the GPU

Gate: spread >= 1.3, cpu pick < 1 on at least 2 of 3 morphologies, mixed/uniform >= 1.15 somewhere.
"""
import argparse
import csv
import math
from pathlib import Path

WORKLOADS = ["base", "radius4", "range8"]
DEPTH = {"kd16l32sm", "kd20l32sm", "qt12l32"}


def load(cell_dir, workload):
    rows = []
    for suffix in ("", "_depth"):
        path = cell_dir / f"{workload}{suffix}.csv"
        if path.exists():
            rows += list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    return rows


def table(rows, backend, batch, kernel=False):
    out = {}
    for r in rows:
        if r["backend"] != backend or int(r["batch"]) != batch:
            continue
        if kernel:
            ms = float(r["kernel_ms"])
            if ms > 0:
                out[r["schema_name"]] = batch / (ms * 1e-3)
        else:
            out[r["schema_name"]] = float(r["throughput_qps"])
    return out


def nested(name):
    return "_" in name


def geomean(values):
    values = [v for v in values if v and v > 0]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("--batches", default="100000,1000000")
    args = parser.parse_args()
    root = Path(args.root)
    batches = [int(b) for b in args.batches.split(",")]

    header = (f"{'cell':18}{'workload':9}{'batch':>8}{'spread w':>9}{'spread t':>9}{'cpu pick':>9}"
              f"{'mix/uni':>8}{'warp/thr':>9}{'gpu/cpuN':>9}{'kern/cpuN':>10}{'depth':>7}  gpu best (warp) | cpu best | skipped")
    print(header)
    summary = {k: [] for k in ("spread", "pick", "mixed", "warp", "gpucpu", "kerncpu", "depth")}
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for workload in WORKLOADS:
            rows = load(cell_dir, workload)
            if not rows:
                continue
            mismatches = sum(int(r["mismatches"]) for r in rows)
            if mismatches:
                print(f"{cell_dir.name} {workload}: {mismatches} MISMATCHES")
            for batch in batches:
                warp = table(rows, "gpuwarp", batch)
                thread = table(rows, "gpu", batch)
                cpu1 = table(rows, "cpu1", batch)
                cpun = table(rows, "cpuN", batch)
                if not warp and not thread:
                    print(f"{cell_dir.name:18}{workload:9}{batch:>8}  no GPU rows (skipped: out of memory)")
                    continue
                gpu_all = {n: max(warp.get(n, 0), thread.get(n, 0)) for n in set(warp) | set(thread)}
                cpu_ref = cpu1 or cpun or table(rows, "cpu1", min(int(r["batch"]) for r in rows))
                cpu_best = max(cpu_ref, key=cpu_ref.get)
                gbest = max(gpu_all, key=gpu_all.get)
                spread_w = max(warp.values()) / min(warp.values()) if warp else float("nan")
                spread_t = max(thread.values()) / min(thread.values()) if thread else float("nan")
                pick = gpu_all.get(cpu_best, float("nan")) / gpu_all[gbest]
                singles = {n: v for n, v in gpu_all.items() if not nested(n)}
                mixed = {n: v for n, v in gpu_all.items() if nested(n)}
                mix = max(mixed.values()) / max(singles.values()) if mixed and singles else float("nan")
                wt = max(warp.values()) / max(thread.values()) if warp and thread else float("nan")
                gc = gpu_all[gbest] / max(cpun.values()) if cpun else float("nan")
                kern = {**table(rows, "gpu", batch, True)}
                for n, v in table(rows, "gpuwarp", batch, True).items():
                    kern[n] = max(kern.get(n, 0), v)
                kc = max(kern.values()) / max(cpun.values()) if kern and cpun else float("nan")
                depth_s = {n: v for n, v in singles.items() if n in DEPTH}
                other_s = {n: v for n, v in singles.items() if n not in DEPTH}
                dep = max(depth_s.values()) / max(other_s.values()) if depth_s and other_s else float("nan")
                wbest = max(warp, key=warp.get) if warp else "-"
                skipped = sum(1 for b in ("gpu", "gpuwarp") for n in cpun if n not in table(rows, b, batch))
                print(f"{cell_dir.name:18}{workload:9}{batch:>8}{spread_w:9.2f}{spread_t:9.2f}{pick:9.3f}{mix:8.3f}"
                      f"{wt:9.2f}{gc:9.2f}{kc:10.2f}{dep:7.3f}  {wbest} | {cpu_best} | {skipped}")
                for key, value in (("spread", spread_w), ("pick", pick), ("mixed", mix), ("warp", wt),
                                   ("gpucpu", gc), ("kerncpu", kc), ("depth", dep)):
                    if not math.isnan(value):
                        summary[key].append(value)
    print()
    print("geomeans over rows: " + ", ".join(f"{k} {geomean(v):.3f} (min {min(v):.3f}, max {max(v):.3f})"
                                             for k, v in summary.items() if v))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GPU pilot: does the schema (fan-out and leaf size per level) matter for batched GPU queries?

On the CPU a node visit is a hot cache line and nesting is worth ~1% (testing.md 2l). On the GPU a
node visit is a memory transaction per thread, or one round of 32 lanes testing 32 children in the
warp-per-query kernels, so fan-out becomes a design choice. This pilot times a fan-out panel with
`--mode batch-check` on one cell: singles spanning 2- to 64-way nodes at three leaf sizes, plus
nested schemas that put a wide level over a narrow tail and the reverse. Backends: cpu1, cpuN,
gpu (thread per query) and gpuwarp (warp per query); batch sizes 10^4 to 10^6; workloads derived
from the cell's test trace as in workload_sweep.py (base, radius4, range8).

Gate for the HPG paper (plan 2026-11-02): GPU best/worst schema >= 1.3x, the GPU-best schema differs
from the CPU-best, and some mixed-fan-out schema beats every uniform one by >= 1.15x somewhere.

    python3 scripts/gpu_pilot.py --cell domfountain_100M --exe ../build_gpu/MultiDataStructure \\
        --data-root /home2/alfonsolr/mds/data --trace-root /home2/alfonsolr/mds/traces/curated \\
        --out ../results/gpu_pilot --cuda-device 1
"""
import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path

from run_cells import REPO, remap_cloud, remap_trace_dir
from workload_sweep import SM, XY, schema, variant_rows

WORKLOADS = ["base", "radius4", "range8"]
LEAVES = [32, 128, 512]


def panel():
    out = []
    for leaf in LEAVES:
        out += [
            schema(f"kd48l{leaf}sm", leaf, [("KDTree", 48, SM)]),                      # 2-way
            schema(f"qt24l{leaf}", leaf, [("QuadTree", 24, XY)]),                      # 4-way
            schema(f"ot24l{leaf}", leaf, [("Octree", 24, {})]),                        # 8-way
            schema(f"str27l{leaf}", leaf, [("STR", 12, {"fanout": 27})]),              # 27-way
            schema(f"grid27l{leaf}", leaf, [("RegularGrid", 12, {})]),                 # 27-way, regular cells
            schema(f"str64l{leaf}", leaf, [("STR", 12, {"fanout": 64})]),              # 64-way
            schema(f"hgrid64l{leaf}", leaf, [("HGrid", 12, {})]),                      # 64-way, regular cells
        ]
    for leaf in (128, 512):
        out += [
            # wide top, narrow tail
            schema(f"hgrid1_kd48l{leaf}sm", leaf, [("HGrid", 1, {}), ("KDTree", 48, SM)]),
            schema(f"str64x2_kd48l{leaf}sm", leaf, [("STR", 2, {"fanout": 64}), ("KDTree", 48, SM)]),
            schema(f"str27_ot24l{leaf}", leaf, [("STR", 1, {"fanout": 27}), ("Octree", 24, {})]),
            schema(f"hgrid2_ot24l{leaf}", leaf, [("HGrid", 2, {}), ("Octree", 24, {})]),
            # narrow top, wide tail
            schema(f"qt3_str64l{leaf}", leaf, [("QuadTree", 3, XY), ("STR", 12, {"fanout": 64})]),
            schema(f"kd6_hgrid64l{leaf}sm", leaf, [("KDTree", 6, SM), ("HGrid", 12, {})]),
            schema(f"qt6_kd48l{leaf}sm", leaf, [("QuadTree", 6, XY), ("KDTree", 48, SM)]),   # e9's best nested shape
        ]
    return out


def is_mixed(name):
    return "_" in name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--cells", default="configs/datasets/curated_cells.json")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--trace-root", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--batch-sizes", default="10000,100000,1000000")
    parser.add_argument("--backends", default="cpu1,cpuN,gpu,gpuwarp")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda-device", type=int, default=-1)
    parser.add_argument("--queries", type=int, default=3000, help="trace rows used as query templates")
    args = parser.parse_args()

    cell = {c["cell"]: c for c in json.loads((REPO / args.cells).read_text())["cells"]}[args.cell]
    cloud = remap_cloud(cell["mdspc_path"], args.data_root)
    trace_dir = Path(remap_trace_dir(cell["trace_dir"], args.cell, args.trace_root))
    trace_dir = trace_dir if trace_dir.is_absolute() else REPO / trace_dir
    out = Path(args.out).resolve() / args.cell
    (out / "schemas").mkdir(parents=True, exist_ok=True)

    schemas = []
    for s in panel():
        path = out / "schemas" / f"{s['name']}.json"
        path.write_text(json.dumps(s, indent=2))
        schemas.append(str(path))

    rows = list(csv.DictReader(open(trace_dir / "pipeline_trace_test.csv", newline="", encoding="utf-8-sig")))
    rows = [rows[(i * len(rows)) // args.queries] for i in range(min(args.queries, len(rows)))]
    radii = [float(r["radius"]) for r in rows if r["query_type"] == "radius" and float(r["radius"]) > 0]
    r0 = statistics.median(radii) if radii else 0.05
    print(f"{args.cell}: {len(schemas)} schemas, trace radius {r0:.4f}", flush=True)

    for workload in args.workloads.split(","):
        summary = out / f"{workload}.csv"
        if summary.exists():
            continue
        trace = out / f"trace_{workload}.csv"
        variant = variant_rows(rows, workload, r0)
        with open(trace, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(variant[0]))
            writer.writeheader()
            writer.writerows(variant)
        partial = out / f"{workload}.partial.csv"
        partial.unlink(missing_ok=True)
        argv = [args.exe, "--mode", "batch-check", "--input", cloud, "--input-trace", str(trace),
                "--schemas", ";".join(schemas), "--batch-sizes", args.batch_sizes, "--backends", args.backends,
                "--batch-repeats", str(args.repeats), "--summary-csv", str(partial), "--no-pause"]
        if args.cuda_device >= 0:
            argv += ["--cuda-device", str(args.cuda_device)]
        print(f"\n##### {args.cell} {workload}", flush=True)
        status = subprocess.run(argv, cwd=REPO).returncode
        if status not in (0, 2):
            print(f"batch-check exited {status}", flush=True)
        if partial.exists():
            partial.rename(summary)

    report(out)


def report(out):
    lines = []
    for workload in WORKLOADS:
        summary = out / f"{workload}.csv"
        if not summary.exists():
            continue
        rows = list(csv.DictReader(open(summary, newline="", encoding="utf-8-sig")))
        mismatches = sum(int(r["mismatches"]) for r in rows)
        lines.append(f"== {workload}: {len(rows)} rows, mismatches {mismatches}")
        batches = sorted({int(r["batch"]) for r in rows})
        backends = [b for b in ("cpu1", "cpuN", "gpu", "gpuwarp") if any(r["backend"] == b for r in rows)]
        for batch in batches:
            qps = {b: {r["schema_name"]: float(r["throughput_qps"]) for r in rows if r["backend"] == b and int(r["batch"]) == batch}
                   for b in backends}
            lines.append(f"-- batch {batch}")
            cpu_best = max(qps["cpu1"], key=qps["cpu1"].get) if qps.get("cpu1") else None
            for b in backends:
                table = qps[b]
                if not table:
                    continue
                ranked = sorted(table, key=table.get, reverse=True)
                best, worst = ranked[0], ranked[-1]
                uniform = [n for n in ranked if not is_mixed(n)]
                mixed = [n for n in ranked if is_mixed(n)]
                line = (f"{b:8} best {best:24} {table[best]:12.0f} q/s   best/worst {table[best] / table[worst]:5.2f}"
                        f"   top5 {', '.join(ranked[:5])}")
                if uniform and mixed:
                    line += f"   mixed/uniform {table[mixed[0]] / table[uniform[0]]:5.3f} ({mixed[0]} vs {uniform[0]})"
                if cpu_best and cpu_best in table and b != "cpu1":
                    line += f"   cpu1-best on {b}: {table[cpu_best] / table[best]:5.3f} of best"
                lines.append(line)
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()

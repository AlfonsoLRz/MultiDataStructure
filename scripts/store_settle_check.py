#!/usr/bin/env python3
"""Is the out-of-core read time steady once the stores are at rest? (testing.md 2q)

In the out-of-core battery every process wrote its stores just before timing them, and on
makinon2's QLC SSD (PNY CS900 4TB, 93% full) the same tree, with the same reads, then took 120-290 us
per read in different processes, a store keeping its speed across a process's passes (the stores
written last tended to be the fast ones). Here the stores of a few schemas are written once,
the device is left idle for --settle-minutes, and --runs processes then time them (order rotated),
reusing the files without writing; --batches repeats the settle and the runs. If the read time is a
property of the device at rest, it agrees across processes, batches and files, and the report's
--io-model is checked against real reads. A process whose median time per read is more than 10%
over each schema's fastest is set aside (the drive's own background work), C0 and MBPS are fitted
on the rest, and per regime the measured latency is ranked next to the modeled one on each device
class and on the fit.

    python3 scripts/store_settle_check.py --cell sanandreas_100M --schema-dir ../results/out_of_core/sanandreas_100M/schemas \\
        --schemas "kd48l128sm;kd48l2048sm;qt24l2048tc;qt9_ot24l2048;qt6_kd48l2048sm;ot10l4096;grid1_qt3_grid1_ot5l4096;hg12l8192" \\
        --exe ../build_ooc/MultiDataStructure --data-root ../data --trace-root ../traces/curated \\
        --store-root ../store_tmp/settle --out ../results/store_settle --device sdg --cpu 2 --numa-node 0
"""
import argparse
import csv
import json
import shutil
import statistics
import time
from pathlib import Path

from run_cells import REPO, launch, merge_runs, remap_cloud, remap_trace_dir


def engine(args, cloud, trace, queries, schemas, regimes, summary, warmup, repeats):
    argv = [args.exe, "--mode", "engine-check", "--input", cloud, "--input-trace", trace, "--queries", str(queries),
            "--schemas", ";".join(schemas), "--warmup", str(warmup), "--measure-repeats", str(repeats), "--interleave",
            "--summary-csv", str(summary), "--leaf-store", str(args.store_dir), "--store-regimes", regimes, "--no-pause"]
    if args.device:
        argv += ["--store-device", args.device]
    started = time.time()
    code = launch(argv, args.cpu, args.numa_node, cwd=REPO).wait()
    print(f"  {summary.name}: {(time.time() - started) / 60:.1f} min, exit {code}", flush=True)
    if code != 0:
        raise SystemExit(f"engine-check failed ({code})")


def fit(rows):
    """Least squares of I/O per query on (reads, KiB) per query, through the origin: (us per read, MB/s)."""
    sxx = sxy = syy = sx = sy = 0.0
    for r in rows:
        x, y, t = float(r["store_reads"]), float(r["store_bytes"]), float(r["store_io_ms"]) * 1e3
        sxx, sxy, syy, sx, sy = sxx + x * x, sxy + x * y, syy + y * y, sx + x * t, sy + y * t
    det = sxx * syy - sxy * sxy
    c0 = (sx * syy - sy * sxy) / det
    per_byte = (sy * sxx - sx * sxy) / det
    return c0, (1.0 / per_byte if per_byte > 0 else float("inf"))


CLASSES = {"NVMe": (60, 3000), "SATA": (120, 500), "HDD": (8000, 150)}


def report(csv_path):
    rows = [r for r in csv.DictReader(open(csv_path, newline="")) if float(r["store_reads"] or 0) > 0]
    per_read = lambda r: float(r["store_io_ms"]) * 1e3 / float(r["store_reads"])
    fastest = {}
    for r in rows:
        key = (r["schema_name"], r["store_regime"])
        fastest[key] = min(fastest.get(key, float("inf")), per_read(r))
    # A process is steady when its median time per read is within 10% of each schema's fastest.
    slowdown = {}
    for run in sorted({r["test_run"] for r in rows}, key=int):
        slowdown[run] = statistics.median(per_read(r) / fastest[(r["schema_name"], r["store_regime"])] for r in rows if r["test_run"] == run)
    steady = [run for run, s in slowdown.items() if s < 1.1]
    print("processes (batch-major): median us per read over each schema's fastest: "
          + ", ".join(f"{run}: {s:.2f}" for run, s in slowdown.items()) + f"; steady: {', '.join(steady)}")
    c0_all, mbps_all = fit(rows)
    total_rows = len(rows)
    rows = [r for r in rows if r["test_run"] in steady]
    c0, mbps = fit(rows)
    print(f"fit over all {total_rows} rows: {c0_all:.1f} us per read + bytes at {mbps_all:.0f} MB/s; "
          f"over the {len(rows)} steady rows: {c0:.1f} us + {mbps:.0f} MB/s")
    classes = dict(CLASSES, fit=(c0, mbps))
    by = {}
    for r in rows:
        by.setdefault((r["schema_name"], r["store_regime"]), []).append(r)
    spreads = []
    for regime in sorted({k[1] for k in by}):
        print(f"\n  {regime}, steady processes; modeled us per query and rank per class")
        print(f"  {'schema':<26} {'reads/q':>8} {'KiB/q':>7} {'us/read':>8} {'spread':>7} {'us/query':>9} "
              + " ".join(f"{name:>12}" for name in classes))
        table = []
        for (name, rg), rs in by.items():
            if rg != regime:
                continue
            reads, nbytes = float(rs[0]["store_reads"]), float(rs[0]["store_bytes"])
            reads_t = [per_read(r) for r in rs]
            spreads.append(max(reads_t) / min(reads_t))
            total = statistics.median(float(r["all_mean_ms"]) * 1e3 for r in rs)
            cpu = statistics.median((float(r["all_mean_ms"]) - float(r["store_io_ms"])) * 1e3 for r in rs)
            modeled = {k: cpu + reads * a + nbytes / m for k, (a, m) in classes.items()}
            table.append((total, name, reads, nbytes / 1024, statistics.median(reads_t), spreads[-1], modeled))
        table.sort()
        ranks = {k: {t[1]: i + 1 for i, t in enumerate(sorted(table, key=lambda t: t[6][k]))} for k in classes}
        for total, name, reads, kib, us_read, spread, modeled in table:
            print(f"  {name:<26} {reads:8.2f} {kib:7.1f} {us_read:8.1f} {spread:7.3f} {total:9.1f} "
                  + " ".join(f"{modeled[k]:9.0f} #{ranks[k][name]}" for k in classes))
    print(f"\n  max/min us per read over the steady processes, per schema: median {statistics.median(spreads):.3f}, worst {max(spreads):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--schema-dir", required=True)
    parser.add_argument("--schemas", required=True, help="';'-separated schema names in --schema-dir")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--cells", default="configs/datasets/curated_cells.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--trace-root", default="")
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--regimes", default="0:memory,0:paged")
    parser.add_argument("--settle-minutes", type=float, default=30)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument("--cpu", type=int, default=-1)
    parser.add_argument("--numa-node", type=int, default=-1)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).resolve() / args.cell
    if args.report_only:
        report(out / "settle.csv")
        return
    args.exe = str(Path(args.exe).resolve())
    args.store_dir = Path(args.store_root).resolve() / args.cell
    cell = {c["cell"]: c for c in json.loads((REPO / args.cells).read_text())["cells"]}[args.cell]
    cloud = remap_cloud(cell["mdspc_path"], args.data_root)
    trace = f"{remap_trace_dir(cell['trace_dir'], args.cell, args.trace_root)}/pipeline_trace_test.csv"
    schemas = [str(Path(args.schema_dir).resolve() / f"{name}.json") for name in args.schemas.split(";")]
    out.mkdir(parents=True, exist_ok=True)
    args.store_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Writes every store (a one-query pass in memory is all that is timed).
        engine(args, cloud, trace, 1, schemas, "memory", out / "write.csv", 0, 1)
        csvs = []
        for batch in range(args.batches):
            print(f"batch {batch}: settling {args.settle_minutes:g} min [{time.strftime('%H:%M:%S')}]", flush=True)
            time.sleep(args.settle_minutes * 60)
            for run in range(args.runs):
                shift = run * len(schemas) // args.runs
                summary = out / f"settle_b{batch}_r{run}.csv"
                engine(args, cloud, trace, cell["queries"], schemas[shift:] + schemas[:shift], args.regimes, summary, 1, args.repeats)
                csvs.append(summary)
        merge_runs(csvs, out / "settle.csv")
    finally:
        shutil.rmtree(args.store_dir, ignore_errors=True)
    report(out / "settle.csv")


if __name__ == "__main__":
    main()

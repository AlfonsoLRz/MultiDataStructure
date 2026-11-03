#!/usr/bin/env python3
"""All-GPU normals pilot: does the schema, and nesting, matter when the results never leave the GPU?

The fan-out pilot (gpu_pilot.py, testing.md 2m) found nesting ahead only in kernel time, hidden end to
end by the download of result indices. Here the workload is a normal-estimation pass: one kNN query
at evenly spaced points of the cloud (`--self-knn`), and the PCA normal of each neighbourhood
computed in the same kernel (`--fused-normals`), so only normals, counts and checksums come back.
Backends: cpuN (every core), gpu and gpuwarp on each schema's tree, and two schema-free baselines
built on the GPU: gpugrid (uniform grid, several cell sizes) and gpucudakd (Wald's cudaKDTree).

    python3 scripts/gpu_normals_pilot.py --cell domfountain_100M --exe ../build_gpu/MultiDataStructure \\
        --data-root /home2/alfonsolr/mds/data --out ../results/gpu_normals --cuda-device 1

Gate (plan 2026-11-03): (a) the best schema on the GPU beats cudaKDTree and the best grid by >= 1.3x
on 2 of 6 cells and is not slower on average: there is a GPU paper; (b) the best nested schema beats
the best single by >= 1.1x in kernel time on 2 of 6 cells: nesting is part of it; (c) the CPU's pick
against the GPU's best, as in the fan-out pilot.
"""
import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

from gpu_pilot import depth_panel, panel
from run_cells import REPO, remap_cloud
from workload_sweep import SM, XY, schema

KS = [16, 32]


def wide_panel():
    """Mixed fan-out schemas around the shapes that won kernel time in the fan-out pilot."""
    out = []
    for leaf in (32, 128):
        out += [
            schema(f"kd6_hgrid64l{leaf}sm", leaf, [("KDTree", 6, SM), ("HGrid", 12, {})]),
            schema(f"kd9_hgrid64l{leaf}sm", leaf, [("KDTree", 9, SM), ("HGrid", 12, {})]),
            schema(f"hgrid2_ot24l{leaf}", leaf, [("HGrid", 2, {}), ("Octree", 24, {})]),
            schema(f"str64_kd48l{leaf}sm", leaf, [("STR", 1, {"fanout": 64}), ("KDTree", 48, SM)]),
        ]
    out.append(schema("kd48l16sm", 16, [("KDTree", 48, SM)]))
    out.append(schema("kd48l8sm", 8, [("KDTree", 48, SM)]))
    return out


def schemas_for(out_dir):
    seen, paths = set(), []
    for s in panel() + depth_panel() + wide_panel():
        if s["name"] in seen:
            continue
        seen.add(s["name"])
        path = out_dir / "schemas" / f"{s['name']}.json"
        path.write_text(json.dumps(s, indent=2))
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--cells", default="configs/datasets/curated_cells.json")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ks", default=",".join(map(str, KS)))
    parser.add_argument("--batch-sizes", default="1000000,10000000")
    parser.add_argument("--backends", default="cpuN,gpu,gpuwarp,gpugrid,gpucudakd")
    parser.add_argument("--grid-cell-factors", default="2,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda-device", type=int, default=-1)
    args = parser.parse_args()

    cell = {c["cell"]: c for c in json.loads((REPO / args.cells).read_text())["cells"]}[args.cell]
    cloud = remap_cloud(cell["mdspc_path"], args.data_root)
    out = Path(args.out).resolve() / args.cell
    (out / "schemas").mkdir(parents=True, exist_ok=True)
    schemas = schemas_for(out)
    print(f"{args.cell}: {len(schemas)} schemas", flush=True)

    for k in map(int, args.ks.split(",")):
        summary = out / f"k{k}.csv"
        if summary.exists():
            continue
        partial = out / f"k{k}.partial.csv"
        partial.unlink(missing_ok=True)
        argv = [args.exe, "--mode", "batch-check", "--input", cloud, "--schemas", ";".join(schemas),
                "--self-knn", str(k), "--fused-normals", "--gpu-count-only", "--batch-sizes", args.batch_sizes,
                "--backends", args.backends, "--grid-cell-factors", args.grid_cell_factors,
                "--batch-repeats", str(args.repeats), "--summary-csv", str(partial), "--no-pause"]
        if args.cuda_device >= 0:
            argv += ["--cuda-device", str(args.cuda_device)]
        print(f"\n##### {args.cell} k={k}", flush=True)
        status = subprocess.run(argv, cwd=REPO).returncode
        # 2 = finished with mismatches (kept and reported); anything else is a crash and reruns.
        if status in (0, 2) and partial.exists():
            partial.rename(summary)
        else:
            print(f"batch-check exited {status}; {partial.name} kept, k={k} reruns next time", flush=True)
    print("\n" + report(out.parent, [args.cell]), flush=True)


def load(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8-sig"))) if path.exists() else []


def geomean(values):
    values = [v for v in values if v and v > 0 and math.isfinite(v)]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else float("nan")


def report(root, cells=None):
    """Gate table over <root>/<cell>/k<k>.csv: throughput in queries per second, end to end and kernel."""
    lines = [f"{'cell':20}{'k':>3}{'batch':>9}  {'warp/thr':>8}{'best/cukd':>10}{'best/grid':>10}{'kern/cukd':>10}"
             f"{'mix/uni':>8}{'k mix':>7}{'cpu pick':>9}{'gpu/cpuN':>9}  best schema (end to end) | mismatches"]
    rows_out = {k: [] for k in ("cukd", "grid", "kcukd", "mix", "kmix", "pick", "gpucpu", "warp")}
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir() and (cells is None or p.name in cells)):
        for k in KS:
            rows = load(cell_dir / f"k{k}.csv")
            if not rows:
                continue
            bad = sum(int(r["mismatches"]) + int(r.get("normal_mismatches") or 0) for r in rows if r["backend"] != "gpucudakd")
            for batch in sorted({int(r["batch"]) for r in rows}):
                sel = [r for r in rows if int(r["batch"]) == batch]

                def qps(backends, kernel=False, names=None):
                    t = {}
                    for r in sel:
                        if r["backend"] not in backends or (names and not names(r["schema_name"])):
                            continue
                        v = batch / (float(r["kernel_ms"]) * 1e-3) if kernel and float(r["kernel_ms"]) > 0 else float(r["throughput_qps"])
                        t[r["schema_name"]] = max(t.get(r["schema_name"], 0.0), v)
                    return t

                ours = qps(("gpu", "gpuwarp"))
                ours_k = qps(("gpu", "gpuwarp"), kernel=True)
                warp, thread = qps(("gpuwarp",)), qps(("gpu",))
                cukd, cukd_k = qps(("gpucudakd",)), qps(("gpucudakd",), kernel=True)
                grid = qps(("gpugrid",))
                cpun = qps(("cpuN",))
                if not ours:
                    continue
                best = max(ours, key=ours.get)
                nested = lambda n: "_" in n
                mix = max([v for n, v in ours.items() if nested(n)] or [0]) / max(v for n, v in ours.items() if not nested(n))
                kmix = max([v for n, v in ours_k.items() if nested(n)] or [0]) / max(v for n, v in ours_k.items() if not nested(n))
                cpu_best = max(cpun, key=cpun.get) if cpun else None
                pick = ours.get(cpu_best, float("nan")) / ours[best] if cpu_best else float("nan")
                values = dict(
                    warp=max(warp.values()) / max(thread.values()) if warp and thread else float("nan"),
                    cukd=ours[best] / max(cukd.values()) if cukd else float("nan"),
                    grid=ours[best] / max(grid.values()) if grid else float("nan"),
                    kcukd=max(ours_k.values()) / max(cukd_k.values()) if cukd_k else float("nan"),
                    mix=mix, kmix=kmix, pick=pick,
                    gpucpu=ours[best] / max(cpun.values()) if cpun else float("nan"))
                for key, v in values.items():
                    if math.isfinite(v):
                        rows_out[key].append(v)
                lines.append(f"{cell_dir.name:20}{k:>3}{batch:>9}  {values['warp']:8.2f}{values['cukd']:10.2f}{values['grid']:10.2f}"
                             f"{values['kcukd']:10.2f}{mix:8.3f}{kmix:7.3f}{pick:9.3f}{values['gpucpu']:9.2f}  {best} | {bad}")
    lines.append("")
    lines.append("geomeans: " + ", ".join(f"{k} {geomean(v):.3f} (min {min(v):.3f}, max {max(v):.3f})" for k, v in rows_out.items() if v))
    return "\n".join(lines)


if __name__ == "__main__":
    main()

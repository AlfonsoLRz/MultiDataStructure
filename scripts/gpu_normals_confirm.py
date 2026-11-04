#!/usr/bin/env python3
"""Re-time the normals pilot's finalists in rotating rounds, so a job sharing the GPU cannot pick the winner.

The pilot (gpu_normals_pilot.py) times each schema once, one after another; on a shared GPU a stretch
of another user's work makes every schema in it about 2x slower (testing.md 2n). Here, per cell and k,
the finalists are the pilot's fastest singles and nested schemas on gpuwarp at the largest batch, plus
the schema the CPU picks (fastest on cpuN). They are timed against cudaKDTree in several rounds, each
round in a rotated order, and every schema keeps its fastest round:

    python3 scripts/gpu_normals_confirm.py --pilot ../results/gpu_normals --out ../results/gpu_normals_confirm \\
        --exe ../build_gpu/MultiDataStructure --data-root /home2/alfonsolr/mds/data --cuda-device 1

Rounds already written (<cell>/k<k>_r<round>.csv) are kept; --report-only prints the table. The spread
column, slowest over fastest round of cudaKDTree and of the best schema, shows how much contention
there was. --grid-cell-factors times the uniform grid instead (<cell>/grid_k<k>_r<round>.csv), against
the CPU reference of one schema; the report then adds it, best factor and fastest round.
"""
import argparse
import json
import subprocess
from pathlib import Path

from gpu_normals_pilot import KS, geomean, load
from run_cells import REPO, remap_cloud


def nested(name):
    return "_" in name


def finalists(pilot_cell, top):
    """The fastest `top` singles and nested schemas per k on gpuwarp (kernel time), plus the CPU's pick."""
    names = []
    for k in KS:
        rows = load(pilot_cell / f"k{k}.csv")
        if not rows:
            continue
        batch = max(int(r["batch"]) for r in rows)
        sel = [r for r in rows if int(r["batch"]) == batch]
        warp = sorted((float(r["kernel_ms"]), r["schema_name"]) for r in sel if r["backend"] == "gpuwarp")
        names += [n for _, n in warp if not nested(n)][:top] + [n for _, n in warp if nested(n)][:top]
        cpu = sorted((float(r["median_ms"]), r["schema_name"]) for r in sel if r["backend"] == "cpuN")
        names += [n for _, n in cpu[:1]]
    return list(dict.fromkeys(names))


def best_rounds(cell_dir, k, prefix=""):
    """Per (schema, backend): the fastest kernel and end-to-end time over the rounds, and the slowest kernel."""
    out = {}
    for path in sorted(cell_dir.glob(f"{prefix}k{k}_r*.csv")):
        rows = load(path)
        batch = max((int(r["batch"]) for r in rows), default=0)
        for r in (r for r in rows if int(r["batch"]) == batch):
            key = (r["schema_name"], r["backend"])
            kernel, total = float(r["kernel_ms"]), float(r["median_ms"])
            entry = out.setdefault(key, dict(kernel=kernel, total=total, slowest=kernel, rounds=0,
                                             normal_mismatches=int(r.get("normal_mismatches") or 0)))
            entry["kernel"], entry["total"] = min(entry["kernel"], kernel), min(entry["total"], total)
            entry["slowest"] = max(entry["slowest"], kernel)
            entry["rounds"] += 1
    return out


def report(root):
    lines = [f"{'cell':20}{'k':>3}{'rounds':>7}  {'best/cukd':>9}{'kern/cukd':>10}{'best/grid':>10}{'kern/grid':>10}{'mix/uni':>8}"
             f"{'k mix':>7}{'cpu pick':>9}{'spread cukd':>12}{'spread best':>12}  best single ; best nested (kernel) ; grid"
             f" | warp normal mismatches"]
    columns = {k: [] for k in ("cukd", "kcukd", "grid", "kgrid", "mix", "kmix", "pick")}
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for k in KS:
            t = best_rounds(cell_dir, k)
            warp = {s: v for (s, b), v in t.items() if b == "gpuwarp"}
            cukd = t.get(("cudakdtree", "gpucudakd"))
            singles = {s: v for s, v in warp.items() if not nested(s)}
            mixes = {s: v for s, v in warp.items() if nested(s)}
            if not singles or not cukd:
                continue
            fastest = lambda d, f: min(d, key=lambda s: d[s][f]) if d else None
            best, best_k = fastest(warp, "total"), fastest(warp, "kernel")
            single_k, nested_k = fastest(singles, "kernel"), fastest(mixes, "kernel")
            cpu = {s: v for (s, b), v in t.items() if b == "cpuN"}
            pick = fastest(cpu, "total")
            grid = {s: v for (s, b), v in best_rounds(cell_dir, k, "grid_").items() if b == "gpugrid"}
            grid_best = fastest(grid, "total")
            values = dict(
                cukd=cukd["total"] / warp[best]["total"],
                kcukd=cukd["kernel"] / warp[best_k]["kernel"],
                grid=grid[grid_best]["total"] / warp[best]["total"] if grid else float("nan"),
                kgrid=min(v["kernel"] for v in grid.values()) / warp[best_k]["kernel"] if grid else float("nan"),
                mix=min(singles[s]["total"] for s in singles) / min(mixes[s]["total"] for s in mixes) if mixes else float("nan"),
                kmix=singles[single_k]["kernel"] / mixes[nested_k]["kernel"] if mixes else float("nan"),
                pick=warp[best]["total"] / warp[pick]["total"] if pick in warp else float("nan"))
            for key, v in values.items():
                if v == v:
                    columns[key].append(v)
            rounds = min(v["rounds"] for v in warp.values())
            lines.append(f"{cell_dir.name:20}{k:>3}{rounds:>7}  {values['cukd']:9.2f}{values['kcukd']:10.2f}{values['grid']:10.2f}"
                         f"{values['kgrid']:10.2f}{values['mix']:8.3f}{values['kmix']:7.3f}{values['pick']:9.3f}"
                         f"{cukd['slowest'] / cukd['kernel']:12.2f}{warp[best_k]['slowest'] / warp[best_k]['kernel']:12.2f}"
                         f"  {single_k} ; {nested_k} ; {grid_best or '-'} | "
                         f"{warp[best_k]['normal_mismatches']}")
    lines.append("")
    lines.append("geomeans (above 1: ours, or nested, faster; pick is the CPU's pick over the GPU's best): " +
                 ", ".join(f"{k} {geomean(v):.3f} (min {min(v):.3f}, max {max(v):.3f})" for k, v in columns.items() if v))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", required=True, help="gpu_normals_pilot.py output root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--exe", default="")
    parser.add_argument("--cells", default="", help="comma list; default every cell of --pilot")
    parser.add_argument("--catalog", default="configs/datasets/curated_cells.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--top", type=int, default=4, help="singles and nested schemas kept per k")
    parser.add_argument("--batch", type=int, default=10000000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda-device", type=int, default=-1)
    parser.add_argument("--grid-cell-factors", default="", help="time the uniform grid instead, at these cell factors")
    parser.add_argument("--grid-reference", default="kd48l32sm", help="the schema whose CPU answers check the grid")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    pilot, out = Path(args.pilot).resolve(), Path(args.out).resolve()

    if not args.report_only:
        catalog = {c["cell"]: c for c in json.loads((REPO / args.catalog).read_text())["cells"]}
        cells = args.cells.split(",") if args.cells else sorted(p.name for p in pilot.iterdir() if p.is_dir())
        grid = bool(args.grid_cell_factors)
        plan = {c: [args.grid_reference] if grid else finalists(pilot / c, args.top) for c in cells}
        for c, names in plan.items():
            print(f"{c}: {len(names)} finalists: {' '.join(names)}", flush=True)
        for r in range(args.rounds):
            for c, names in plan.items():
                cloud = remap_cloud(catalog[c]["mdspc_path"], args.data_root)
                (out / c).mkdir(parents=True, exist_ok=True)
                order = names[r % len(names):] + names[: r % len(names)]
                schemas = [str(pilot / c / "schemas" / f"{n}.json") for n in order]
                for k in KS:
                    summary = out / c / f"{'grid_' if grid else ''}k{k}_r{r}.csv"
                    if summary.exists():
                        continue
                    partial = summary.with_suffix(".partial.csv")
                    partial.unlink(missing_ok=True)
                    argv = [args.exe, "--mode", "batch-check", "--input", cloud, "--schemas", ";".join(schemas),
                            "--self-knn", str(k), "--fused-normals", "--gpu-count-only", "--batch-sizes", str(args.batch),
                            "--backends", "gpugrid" if grid else "cpuN,gpuwarp,gpucudakd", "--batch-repeats", str(args.repeats),
                            "--summary-csv", str(partial), "--no-pause"]
                    if grid:
                        argv += ["--grid-cell-factors", args.grid_cell_factors]
                    if args.cuda_device >= 0:
                        argv += ["--cuda-device", str(args.cuda_device)]
                    print(f"\n##### round {r} {c} k={k}", flush=True)
                    status = subprocess.run(argv, cwd=REPO).returncode
                    if status in (0, 2) and partial.exists():
                        partial.rename(summary)
                    else:
                        print(f"batch-check exited {status}; round {r} {c} k={k} reruns next time", flush=True)
    print("\n" + report(out), flush=True)


if __name__ == "__main__":
    main()

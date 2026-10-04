#!/usr/bin/env python3
"""Does nesting beat the best single structure once nested schemas are not depth-capped?

The search's schema generator caps every candidate at 12 levels in total (testing.md 2f), so a
searched nested schema could never resolve a 25M+ cloud down to its leaf capacity. This builds the
comparison the search should have made, per cell of a battery folder:

  nested grid    prefix structure (qt 3/6/9, ot 2/4/6 or kd 6/12 levels), then a tail of another
                 structure that runs until leafCapacity binds (16 levels for qt/ot, 22 for kd)
  single grid    the same tails alone (qt16, ot16, kd22), plus the cell's tuned-singles winner and
                 the quadtree/kdtree/octree defaults
  extras         the GA winner and the two hand-nested schemas with their last level uncapped

Everything is timed serially on one pinned core on the trace's _opt half; the two best nested and
two best singles by opt latency, the GA winner and the controls then go to a held-out re-measure on
_test (5 repeats), repeated in --test-runs separate processes with the order rotated, because at
100M+ the same schema can differ by ~30% between processes. Selection never sees _test.

    python scripts/nest_diagnostic.py --cells toronto_5M,toronto_50M
    python scripts/nest_diagnostic.py --report
    python3 scripts/nest_diagnostic.py --exe ../build/MultiDataStructure --cpu 20 --numa-node 1 \\
        --data-root /home2/alfonsolr/mds/data --trace-root /home2/alfonsolr/mds/traces/curated \\
        --matrix /home2/alfonsolr/mds/results/matrix_e6 --out /home2/alfonsolr/mds/results/nest_diag \\
        --queries 3000 --cells sanandreas_250M
"""
import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from run_cells import best_schema_path, launch, remap_cloud, remap_trace_dir  # noqa: E402

TYPES = {"qt": ("QuadTree", 16, {"axisPolicy": "xy"}), "ot": ("Octree", 16, {}),
         "kd": ("KDTree", 22, {"axisPolicy": "median_longest_axis"})}
TAIL_DEPTH = {"QuadTree": 16, "Octree": 16, "KDTree": 22}
PREFIXES = (("qt", (3, 6, 9)), ("ot", (2, 4, 6)), ("kd", (6, 12)))


def level(kind, depth, leaf):
    t, _, extra = TYPES[kind]
    return {"type": t, "numLevels": depth, "leafCapacity": leaf, "minPointsToSplit": 8, **extra}


def schema(name, levels):
    return {"name": name, "levels": levels, "buildPolicy": {
        "maxDepth": sum(l["numLevels"] for l in levels), "leafCapacity": levels[0]["leafCapacity"],
        "minPointsToSplit": 8, "collapseSingleChild": True, "removeEmptyNodes": True,
        "allowOverlapDuplication": False}}


def uncap(path, name):
    s = json.loads(Path(path).read_text())
    last = s["levels"][-1]
    last["numLevels"] = max(last["numLevels"], TAIL_DEPTH.get(last["type"], 22))
    s["buildPolicy"]["maxDepth"] = sum(l["numLevels"] for l in s["levels"])
    s["name"] = name
    return s


def repo_path(p):
    return Path(p) if Path(p).is_absolute() else REPO / p


def run(args, argv, log):
    with open(log, "a") as handle:
        return launch([args.exe] + argv, args.cpu, args.numa_node, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT).wait()


def latencies(*paths):
    """Per schema: the median over runs of each run's repeat mean, and the widest CI over runs."""
    runs = {}
    for path in paths:
        if not Path(path).exists():
            continue
        with open(path, encoding="utf-8-sig") as handle:
            seen = set()
            for r in csv.DictReader(handle):
                if r["schema_name"] in seen:
                    continue
                seen.add(r["schema_name"])
                runs.setdefault(r["schema_name"], []).append(dict(
                    ms=float(r["latency_repeat_mean_ms"]) * 1000, lo=float(r["latency_repeat_ci_low_ms"]) * 1000,
                    hi=float(r["latency_repeat_ci_high_ms"]) * 1000, path=r["schema_path"]))
    return {n: dict(ms=median(x["ms"] for x in v), lo=min(x["lo"] for x in v), hi=max(x["hi"] for x in v),
                    path=v[0]["path"], runs=len(v)) for n, v in runs.items()}


def test_files(d):
    return sorted(d.glob("test_r*.csv")) or [d / "test.csv"]


def is_single(name):
    return name.startswith(("sd_", "tuned_")) or name.endswith("_default")


def run_cell(args, cell):
    name = cell["cell"]
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    matrix = repo_path(args.matrix)
    tuned = best_schema_path(str(matrix / f"{name}_singles.csv"))
    ga = best_schema_path(str(matrix / f"{name}_ga.csv"))
    tuned_leaf = json.loads(repo_path(tuned).read_text())["levels"][-1]["leafCapacity"]
    leaves = sorted({32, 128, tuned_leaf})

    grid = {}
    for prefix, depths in PREFIXES:
        for depth in depths:
            for tail in TYPES:
                if tail != prefix:
                    for leaf in leaves:
                        n = f"nd_{prefix}{depth}_{tail}l{leaf}"
                        grid[n] = schema(n, [level(prefix, depth, leaf), level(tail, TYPES[tail][1], leaf)])
    for tail in TYPES:
        for leaf in leaves:
            n = f"sd_{tail}{TYPES[tail][1]}l{leaf}"
            grid[n] = schema(n, [level(tail, TYPES[tail][1], leaf)])
    for src in ("octree_kdtree", "quadtree_octree"):
        grid[f"nd_hand_{src}_uncapped"] = uncap(REPO / f"configs/schemas/{src}.json", f"nd_hand_{src}_uncapped")
    if ga:
        grid["nd_ga_uncapped"] = uncap(repo_path(ga), "nd_ga_uncapped")
    paths = []
    for n, s in grid.items():
        (out / f"{n}.json").write_text(json.dumps(s, indent=2))
        paths.append(str(out / f"{n}.json"))

    controls = [tuned] + [f"configs/schemas/{x}.json" for x in ("quadtree", "kdtree", "octree")]
    cloud = remap_cloud(cell["mdspc_path"], args.data_root)
    common = ["--mode", "schema-search", "--input", cloud, "--no-synthetic",
              "--workloads", "configs/workloads/pipeline_replay.json", "--queries", str(args.queries or cell["queries"]),
              "--evaluator", "cpu", "--benchmark-top", "0", "--no-score-cache", "--no-pause",
              "--flat-search", "--generate-schemas", "0", "--no-baselines"]
    trace = remap_trace_dir(cell["trace_dir"], name, args.trace_root)
    for stale in [out / "opt.csv", out / "test.csv"] + list(out.glob("test_r*.csv")):
        stale.unlink(missing_ok=True)
    steps = [("opt", "opt.csv", 3)] + [("test", f"test_r{k}.csv", 5) for k in range(args.test_runs)]
    schemas = controls + paths
    for k, (half, csv_name, repeats) in enumerate(steps):
        if half == "test":
            if k == 1:
                opt = latencies(out / "opt.csv")
                nested = sorted((v["ms"], v["path"]) for n, v in opt.items() if n.startswith("nd_"))
                singles = sorted((v["ms"], v["path"]) for n, v in opt.items() if is_single(n))
                schemas = list(dict.fromkeys([p for _, p in nested[:2]] + [p for _, p in singles[:2]] + [tuned] +
                                             ([ga, str(out / "nd_ga_uncapped.json")] if ga else []) + controls[1:]))
            else:
                schemas = schemas[1:] + schemas[:1]  # rotate, so no schema always runs first
        started = time.time()
        code = run(args, common + ["--input-trace", f"{trace}/pipeline_trace_{half}.csv", "--schemas", ";".join(schemas),
                                   "--measure-repeats", str(repeats), "--csv", str(out / csv_name)], out / "log.txt")
        print(f"{name}: {csv_name} {len(schemas)} schemas, exit {code}, {time.time() - started:.0f} s", flush=True)
        if code != 0:
            return


def report(args):
    ratios, ga_ratios = [], []
    for d in sorted(Path(args.out).iterdir()):
        if not any(f.exists() for f in test_files(d)):
            continue
        opt, test = latencies(d / "opt.csv"), latencies(*test_files(d))
        nested = min((n for n in test if n.startswith("nd_") and n != "nd_ga_uncapped"), key=lambda n: opt[n]["ms"])
        single = min((n for n in test if is_single(n)), key=lambda n: opt[n]["ms"])
        a, b = test[nested], test[single]
        verdict = "WIN" if a["hi"] < b["lo"] else "LOSS" if a["lo"] > b["hi"] else "tie"
        ga = next((n for n in test if n.startswith(("evolved_", "xover_", "generated_"))), None)
        ratios.append(a["ms"] / b["ms"])
        if ga:
            ga_ratios.append(test[ga]["ms"] / b["ms"])
        print(f"{d.name:22} {nested:22} {a['ms']:6.2f} us  vs {single:22} {b['ms']:6.2f} us  "
              f"{a['ms'] / b['ms']:.2f} {verdict:4}  | GA winner {test[ga]['ms'] / b['ms'] if ga else float('nan'):.2f}")
    geo = lambda v: math.exp(sum(map(math.log, v)) / len(v))
    if ratios:
        print(f"\nnested/single  geomean {geo(ratios):.3f}  median {median(ratios):.3f}  ({len(ratios)} cells)")
    if ga_ratios:
        print(f"GA winner/single  geomean {geo(ga_ratios):.3f}  median {median(ga_ratios):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", default="", help="comma list of cell names")
    parser.add_argument("--matrix", default="results/eval_traces/matrix_e6", help="battery folder with the GA and singles arms")
    parser.add_argument("--out", default="D:/MDS_ab/nest_diag")
    default_exe = REPO / ("x64/Release/MultiDataStructure.exe" if platform.system() == "Windows" else "../build/MultiDataStructure")
    parser.add_argument("--exe", default=str(default_exe))
    parser.add_argument("--cpu", type=int, default=4, help="core to pin to; -1 = no pinning")
    parser.add_argument("--numa-node", type=int, default=-1, help="Linux: bind memory to this NUMA node (numactl)")
    parser.add_argument("--data-root", default="", help="see run_cells.py")
    parser.add_argument("--trace-root", default="", help="see run_cells.py")
    parser.add_argument("--queries", type=int, default=0, help="override the cell's query count")
    parser.add_argument("--test-runs", type=int, default=3, help="separate held-out processes, order rotated")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        return report(args)
    cells = {c["cell"]: c for c in json.loads((REPO / "configs/datasets/curated_cells.json").read_text())["cells"]}
    for name in filter(None, args.cells.split(",")):
        run_cell(args, cells[name])


if __name__ == "__main__":
    main()

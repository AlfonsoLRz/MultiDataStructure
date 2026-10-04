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
_test (5 repeats). Selection never sees _test.

    python scripts/nest_diagnostic.py --cells toronto_5M,toronto_50M
    python scripts/nest_diagnostic.py --report
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
from run_cells import best_schema_path  # noqa: E402

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


def run(exe, argv, log, cpu):
    with open(log, "a") as handle:
        p = subprocess.Popen([exe] + argv, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
        if platform.system() == "Windows" and cpu >= 0:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetProcessAffinityMask(int(p._handle), 1 << cpu)
            k32.SetPriorityClass(int(p._handle), 0x80)  # HIGH_PRIORITY_CLASS
        return p.wait()


def latencies(path):
    out = {}
    with open(path, encoding="utf-8-sig") as handle:
        for r in csv.DictReader(handle):
            out.setdefault(r["schema_name"], dict(
                ms=float(r["latency_repeat_mean_ms"]) * 1000, lo=float(r["latency_repeat_ci_low_ms"]) * 1000,
                hi=float(r["latency_repeat_ci_high_ms"]) * 1000, path=r["schema_path"]))
    return out


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
    common = ["--mode", "schema-search", "--input", cell["mdspc_path"], "--no-synthetic",
              "--workloads", "configs/workloads/pipeline_replay.json", "--queries", str(cell["queries"]),
              "--evaluator", "cpu", "--benchmark-top", "0", "--no-score-cache", "--no-pause",
              "--flat-search", "--generate-schemas", "0", "--no-baselines"]
    trace = cell["trace_dir"]
    for half, schemas, repeats in (("opt", controls + paths, 3), ("test", None, 5)):
        if half == "test":
            opt = latencies(out / "opt.csv")
            nested = sorted((v["ms"], v["path"]) for n, v in opt.items() if n.startswith("nd_"))
            singles = sorted((v["ms"], v["path"]) for n, v in opt.items() if is_single(n))
            schemas = list(dict.fromkeys([p for _, p in nested[:2]] + [p for _, p in singles[:2]] + [tuned] +
                                         ([ga, str(out / "nd_ga_uncapped.json")] if ga else []) + controls[1:]))
        (out / f"{half}.csv").unlink(missing_ok=True)
        started = time.time()
        code = run(args.exe, common + ["--input-trace", f"{trace}/pipeline_trace_{half}.csv", "--schemas", ";".join(schemas),
                                       "--measure-repeats", str(repeats), "--csv", str(out / f"{half}.csv")],
                   out / "log.txt", args.cpu)
        print(f"{name}: {half} {len(schemas)} schemas, exit {code}, {time.time() - started:.0f} s", flush=True)
        if code != 0:
            return


def report(args):
    ratios, ga_ratios = [], []
    for d in sorted(Path(args.out).iterdir()):
        if not (d / "test.csv").exists():
            continue
        opt, test = latencies(d / "opt.csv"), latencies(d / "test.csv")
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
    parser.add_argument("--cpu", type=int, default=4, help="core to pin to (Windows); -1 = no pinning")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        return report(args)
    cells = {c["cell"]: c for c in json.loads((REPO / "configs/datasets/curated_cells.json").read_text())["cells"]}
    for name in filter(None, args.cells.split(",")):
        run_cell(args, cells[name])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Is a nested win composition, or a single nobody offered? (testing.md 2o)

A nested winner is timed against singles built from each of its blocks, alone and open-ended, at
its leaf capacity and at half and twice it, plus the singles given with --against. Timing is the
held-out interleave of run_cells.py (engine-check --interleave in --test-runs processes, order
rotated, median over processes), on the cell's test trace, at the same emulated node cost:

    python3 scripts/strip_check.py --cell galiciaforest_25M --nested <winner.json> --against <single.json> \\
        --node-cost-ns 10000 --exe ../build_e2/MultiDataStructure --data-root ../data --trace-root ../traces/curated \\
        --out ../results/strip_check --cpu 4 --numa-node 0

If some block's single is within a few percent of the winner, the win came from that primitive's
configuration (often its leaf size), not from stacking blocks.
"""
import argparse
import copy
import json
from pathlib import Path

from run_cells import REPO, launch, merge_runs, remap_cloud, remap_trace_dir
from summarize_nesting import timing

# Levels that let a block run to the leaf capacity on its own (a prefix block is depth-limited).
OPEN_LEVELS = {"Octree": 24, "QuadTree": 24, "KDTree": 48, "BVH": 48, "BIH": 48, "STR": 11, "HGrid": 12}


def block_singles(nested, out_dir):
    schema = json.loads(Path(nested).read_text())
    leaf = schema["buildPolicy"]["leafCapacity"]
    paths = []
    for i, block in enumerate(schema["levels"]):
        for capacity in (leaf // 2, leaf, leaf * 2):
            single = copy.deepcopy(schema)
            level = dict(block, leafCapacity=capacity, numLevels=max(block["numLevels"], OPEN_LEVELS.get(block["type"], 0)))
            single["levels"] = [level]
            single["buildPolicy"]["leafCapacity"] = capacity
            single["name"] = f"strip{i}_{block['type'].lower()}{block.get('fanout', '')}l{capacity}"
            path = out_dir / f"{Path(nested).stem}_{single['name']}.json"
            path.write_text(json.dumps(single, indent=2))
            paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--nested", required=True)
    parser.add_argument("--against", default="", help="';'-separated singles to include (the run's best single)")
    parser.add_argument("--node-cost-ns", type=float, default=0.0)
    parser.add_argument("--exe", required=True)
    parser.add_argument("--cells", default="configs/datasets/curated_cells.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--trace-root", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="", help="appended to the output folder name")
    parser.add_argument("--test-runs", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=-1)
    parser.add_argument("--numa-node", type=int, default=-1)
    args = parser.parse_args()

    cell = {c["cell"]: c for c in json.loads((REPO / args.cells).read_text())["cells"]}[args.cell]
    cloud = remap_cloud(cell["mdspc_path"], args.data_root)
    trace = f"{remap_trace_dir(cell['trace_dir'], args.cell, args.trace_root)}/pipeline_trace_test.csv"
    tag = f"{args.cell}_d{args.node_cost_ns:g}_{Path(args.nested).stem}{args.label}"
    out = Path(args.out).resolve() / tag
    out.mkdir(parents=True, exist_ok=True)
    schemas = [str(Path(args.nested).resolve())] + [s for s in args.against.split(";") if s] + block_singles(args.nested, out)

    runs = []
    for k in range(args.test_runs):
        shift = k * len(schemas) // args.test_runs
        run_csv = str(out / f"timing_r{k}.csv")
        argv = [args.exe, "--mode", "engine-check", "--input", cloud, "--input-trace", trace, "--queries", str(cell["queries"]),
                "--schemas", ";".join(schemas[shift:] + schemas[:shift]), "--warmup", "1", "--measure-repeats", "10",
                "--interleave", "--summary-csv", run_csv, "--no-pause"]
        if args.node_cost_ns > 0:
            argv += ["--node-cost-ns", str(args.node_cost_ns)]
        if launch(argv, args.cpu, args.numa_node, cwd=REPO).wait() == 0:
            runs.append(run_csv)
    merge_runs(runs, str(out / "timing.csv"))
    latency = timing(out / "timing.csv")
    reference = latency.get(schemas[0].replace("\\", "/"))
    print(f"\n{tag}: latency per query, relative to the nested winner (below 1 = faster than it)")
    for path in sorted(latency, key=latency.get):
        mark = "  <- nested winner" if path == schemas[0].replace("\\", "/") else ""
        print(f"  {latency[path] * 1e3:9.2f} us  {latency[path] / reference:6.3f}  {Path(path).stem}{mark}")


if __name__ == "__main__":
    main()

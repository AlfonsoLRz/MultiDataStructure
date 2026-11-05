#!/usr/bin/env python3
"""External libraries on the held-out traces, next to our engine (testing.md 2r).

Is the 1.66x library-default gap of 2m/2p specific to our engine? Each library is timed with one
protocol (tools/baseline_common.h: the .mdspc cloud in its local float frame, engine-check's trace
stride, one untimed warm-up and --repeats timed passes, each search call timed alone, one thread),
at its default and across its main parameter, on the cell's held-out pipeline trace:

  nanoflann   kd-tree, leaf 10 (default), 16, 32, 64, 128, 256
  PCL         KdTreeFLANN (PCL's default search), OctreePointCloudSearch at extent / 128 ... / 4096
  Open3D      geometry::KDTreeFlann (no parameters)
  unibn       Behley's octree, bucket 32 (default), 8 ... 512; radius only
  Boost       Boost.Geometry R-tree, STR bulk load, 16 (the documented example), 8, 32, 64 per node

and our engine (engine-check --interleave) with the library defaults at leaf 32 (octree, median kd),
the flatness rule's two picks (kd48l128sm, qt16l128tc) and the cell's e9 pick. Every process is
pinned like run_cells.py, and --rounds rounds rotate the order of the processes; readers take the
median over rounds. Round 0 also writes per-query counts and checksums, so the report can check
every library's answers against the engine's.

Two steps. `prepare` runs where the e9 results are (the workstation) and writes our schema panel per
cell; `run` runs on the measuring machine:

    python scripts/external_baselines.py prepare --cells "domfountain_100M;sanandreas_25M" \\
        --e9 results/eval_traces/matrix_e9 results/eval_traces/matrix_e9_makinon2 --out results/external/schemas
    python3 scripts/external_baselines.py run --cells "domfountain_100M;sanandreas_25M" --schemas ../ext_schemas \\
        --exe ../build_ooc/MultiDataStructure --tools ../baselines_bin --data-root ../data \\
        --trace-root ../traces/curated --out ../results/external --cpu 20 --numa-node 1
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

from run_cells import REPO, launch, merge_runs, remap_cloud, remap_trace_dir
from workload_sweep import SM, XY, schema

TOOLS = ["nanoflann", "pcl", "open3d", "unibn", "boost_rtree"]
DEFAULTS = ["configs/schemas/octree.json", "configs/schemas/kdtree.json"]
TC = {"splitAt": "tight"}
RULE = [schema("kd48l128sm", 128, [("KDTree", 48, SM)]), schema("qt16l128tc", 128, [("QuadTree", 16, dict(XY, **TC))])]


def schema_from_name(name):
    """A tuned or generated single rebuilt from its name (tuned_kd48l64sm, generated_qt24l32, ...)."""
    match = re.search(r"(kd|qt|ot)(\d+)l(\d+)(\S*)$", name)
    if not match:
        raise ValueError(f"cannot rebuild a schema from {name}")
    kind, depth, leaf, suffix = match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)
    extra = dict(SM) if "sm" in suffix else {}
    if "tc" in suffix:
        extra.update(TC)
    primitive = {"kd": "KDTree", "qt": "QuadTree", "ot": "Octree"}[kind]
    return schema(f"e9pick_{kind}{depth}l{leaf}{suffix.replace('ap=', '')}", leaf, [(primitive, depth, extra)])


def prepare(args):
    for cell in args.cells.split(";"):
        pick_file = next((Path(d, f"{cell}_pick.json") for d in args.e9 if Path(d, f"{cell}_pick.json").exists()), None)
        if pick_file is None:
            raise SystemExit(f"no e9 pick for {cell}")
        pick = json.loads(pick_file.read_text())["pick"].replace("\\", "/").split("/")[-1]
        local = pick_file.parent / "best_schemas" / f"{cell}_ga" / pick
        if local.exists():
            chosen = json.loads(local.read_text())
            chosen["name"] = "e9pick_" + chosen["name"].replace("generated_", "").replace("ap=", "")
        else:
            chosen = schema_from_name(Path(pick).stem)
        folder = Path(args.out, cell)
        folder.mkdir(parents=True, exist_ok=True)
        for s in RULE + [chosen]:
            (folder / f"{s['name']}.json").write_text(json.dumps(s, indent=2))
        print(cell, "e9 pick", chosen["name"])


def run(args):
    cells = {c["cell"]: c for c in json.loads((REPO / args.cells_config).read_text())["cells"]}
    tools = Path(args.tools).resolve()
    for name in args.cells.split(";"):
        cell = cells[name]
        out = Path(args.out).resolve() / name
        out.mkdir(parents=True, exist_ok=True)
        cloud = remap_cloud(cell["mdspc_path"], args.data_root)
        trace = f"{remap_trace_dir(cell['trace_dir'], name, args.trace_root)}/pipeline_trace_test.csv"
        ours = [str(REPO / path) for path in DEFAULTS] + sorted(str(p.resolve()) for p in Path(args.schemas, name).glob("*.json"))
        print(f"\n### {name} [{time.strftime('%H:%M:%S')}]: {len(ours)} schemas of ours, {', '.join(TOOLS)}", flush=True)
        processes = ["mds"] + TOOLS
        for k in range(args.rounds):
            shift = k % len(processes)
            for tool in processes[shift:] + processes[:shift]:
                summary = out / f"{tool}_r{k}.csv"
                summary.unlink(missing_ok=True)
                if tool == "mds":
                    argv = [args.exe, "--mode", "engine-check", "--input", cloud, "--input-trace", trace, "--queries", str(cell["queries"]),
                            "--schemas", ";".join(ours), "--warmup", "1", "--measure-repeats", str(args.repeats), "--interleave",
                            "--summary-csv", str(summary), "--no-pause"]
                    if k == 0:
                        argv += ["--query-csv", str(out / "mds_queries.csv")]
                else:
                    argv = [str(tools / f"{tool}_point_baseline"), "--input", cloud, "--query-trace", trace, "--max-queries", str(cell["queries"]),
                            "--warmup", "1", "--repeats", str(args.repeats), "--summary-csv", str(summary)]
                    if k == 0:
                        queries_csv = out / f"{tool}_queries.csv"
                        queries_csv.unlink(missing_ok=True)
                        argv += ["--query-csv", str(queries_csv)]
                started = time.time()
                code = launch(argv, args.cpu, args.numa_node, cwd=REPO).wait()
                print(f"  round {k} {tool}: {(time.time() - started) / 60:.1f} min, exit {code}", flush=True)
        for tool in processes:
            runs = [out / f"{tool}_r{k}.csv" for k in range(args.rounds) if (out / f"{tool}_r{k}.csv").exists()]
            merge_runs(runs, out / f"{tool}.csv")
        print(f"### {name} done [{time.strftime('%H:%M:%S')}]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cells", required=True, help="';'-separated cells")
    p.add_argument("--e9", nargs="+", required=True, help="e9 result folders holding <cell>_pick.json")
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cells", required=True, help="';'-separated cells")
    r.add_argument("--cells-config", default="configs/datasets/curated_cells.json")
    r.add_argument("--schemas", required=True, help="the prepare step's output")
    r.add_argument("--exe", required=True)
    r.add_argument("--tools", required=True, help="folder of the *_point_baseline drivers")
    r.add_argument("--data-root", default="")
    r.add_argument("--trace-root", default="")
    r.add_argument("--out", required=True)
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--repeats", type=int, default=10)
    r.add_argument("--cpu", type=int, default=-1)
    r.add_argument("--numa-node", type=int, default=-1)
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args)
    else:
        args.exe = str(Path(args.exe).resolve())
        run(args)


if __name__ == "__main__":
    main()

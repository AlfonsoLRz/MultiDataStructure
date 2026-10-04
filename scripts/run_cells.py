"""Portable cell battery: the protocol of scripts/run_matrix_heatmap.ps1, runnable on Windows and Linux.

Per cell of a cell config (scripts/make_cell_config.py):
  search cells    GA arm and tuned-singles arm on the trace's _opt half, then a held-out TEST
                  re-measure on the _test half: the two winners, every default single structure
                  and the hand-designed nested schemas, 5 timing repeats in each of --test-runs
                  separate processes (schema order rotated), merged into <cell>_test.csv.
  transfer cells  no search; the TEST re-measure uses the winners of the largest searched cell
                  of the same scene (the donor), as the PowerShell battery does.

Data and traces can live elsewhere than the config says, which is what lets one config drive the
Windows workstation and the Linux server:
    --data-root /home2/alfonsolr/mds/data     <Scene>/<tier>/<file> under this root
    --trace-root /home2/alfonsolr/mds/traces/curated     <cell>/ under this root

Examples:
    python scripts/run_cells.py --only sanandreas_100M --out results/eval_traces/matrix_e6
    python3 scripts/run_cells.py --exe ../build/MultiDataStructure --data-root ... --trace-root ... \
        --only sanandreas_100M,sanandreas_250M --parallel 16 --tier-c-queries 3000
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULTS = ["quadtree", "octree", "kdtree", "bvh", "bih", "hgrid", "lbvh", "regular_grid", "karras_octree"]
HAND_NESTED = ["configs/schemas/octree_kdtree.json", "configs/schemas/quadtree_octree.json"]


def remap_cloud(path, data_root):
    if not data_root:
        return path
    parts = Path(path.replace("\\", "/")).parts
    # .../<Scene>/<tier>/<file>
    return str(Path(data_root, *parts[-3:]))


def remap_trace_dir(trace_dir, cell_name, trace_root):
    return str(Path(trace_root, cell_name)) if trace_root else trace_dir


def schema_blocks(path):
    """Number of blocks (levels entries) of a schema file; 0 if unreadable."""
    try:
        return len(json.loads((Path(path) if Path(path).is_absolute() else REPO / path).read_text())["levels"])
    except (OSError, ValueError, KeyError):
        return 0


def best_schema_path(csv_path, nested_only=False):
    """Winner of an arm CSV: lowest score among final-latency rows whose schema file exists.
    nested_only: the best schema with two or more blocks."""
    if not Path(csv_path).exists():
        return None
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "score_is_final_latency" in rows[0]:
        final = [r for r in rows if r.get("score_is_final_latency") == "1"]
        rows = final or rows
    for row in sorted(rows, key=lambda r: float(r.get("score") or "inf")):
        path = (row.get("schema_path") or "").replace("\\", "/")
        # Relative schema paths are relative to the repo, where the executable runs.
        if path and (Path(path) if Path(path).is_absolute() else REPO / path).exists():
            if not nested_only or schema_blocks(path) > 1:
                return path
    return None


def merge_runs(run_csvs, out_csv):
    """Concatenate per-process TEST CSVs, tagging rows with test_run; readers take the median."""
    rows, fields = [], None
    for k, path in enumerate(run_csvs):
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = fields or list(reader.fieldnames) + ["test_run"]
            rows += [dict(row, test_run=k) for row in reader]
    if not rows:
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def launch(argv, cpu=-1, numa_node=-1, **popen_args):
    """Start a measurement process pinned to one core (cpu >= 0) at high priority.

    Unpinned, the workstation's hybrid CPU let a run land on an efficiency core: the same schema
    then differed by up to 1.44x between held-out runs. Linux binds memory to numa_node too."""
    prefix = []
    if platform.system() != "Windows" and cpu >= 0:
        prefix = (["numactl", f"--physcpubind={cpu}", f"--membind={numa_node}"] if numa_node >= 0
                  else ["taskset", "-c", str(cpu)])
    process = subprocess.Popen(prefix + argv, **popen_args)
    if platform.system() == "Windows" and cpu >= 0:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetProcessAffinityMask(int(process._handle), 1 << cpu)
        kernel32.SetPriorityClass(int(process._handle), 0x80)  # HIGH_PRIORITY_CLASS
    return process


class Battery:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "best_schemas").mkdir(exist_ok=True)
        self.failed = []
        index = REPO / "configs/schemas/tuned_singles_small/index.txt"
        self.tuned = ";".join(line.strip() for line in index.read_text().splitlines() if line.strip())
        self.defaults = [f"configs/schemas/{name}.json" for name in DEFAULTS]

    def run(self, title, argv):
        print(f"\n########## {title}  [{time.strftime('%H:%M:%S')}] ##########", flush=True)
        started = time.time()
        code = launch([self.args.exe] + argv, self.args.cpu, self.args.numa_node, cwd=REPO).wait()
        print(f"########## {title} done in {(time.time() - started) / 60:.1f} min (exit {code}) ##########", flush=True)
        if code != 0:
            self.failed.append(f"{title} (exit {code})")
        return code == 0

    def common(self, cloud, trace, queries):
        # --benchmark-top 0: measure every listed schema. The default (32) lets the surrogate
        # rank model silently drop candidates before measurement - with the 36 tuned singles,
        # 4 of them, possibly including the best one. The PowerShell battery had that flaw.
        return ["--mode", "schema-search", "--input", cloud, "--no-synthetic",
                "--workloads", "configs/workloads/pipeline_replay.json", "--input-trace", trace,
                "--queries", str(queries), "--evaluator", "cpu", "--benchmark-top", "0",
                "--no-score-cache", "--no-pause"]

    def cell(self, cell, cells):
        name = cell["cell"]
        cloud = remap_cloud(cell["mdspc_path"], self.args.data_root)
        trace_dir = remap_trace_dir(cell["trace_dir"], name, self.args.trace_root)
        opt, test = f"{trace_dir}/pipeline_trace_opt.csv", f"{trace_dir}/pipeline_trace_test.csv"
        for label, path in (("cloud", cloud), ("opt trace", opt), ("test trace", test)):
            if not Path(path).exists():
                print(f"{name}: {label} missing ({path}) - skipping")
                self.failed.append(f"{name} ({label} missing)")
                return

        searched = cell["arm"] == "search" or name in self.args.search_transfer
        queries = cell["queries"]
        if cell["tier"] == "C" and self.args.tier_c_queries:
            queries = self.args.tier_c_queries
        ga_csv, singles_csv, test_csv = (str(self.out / f"{name}_{arm}.csv") for arm in ("ga", "singles", "test"))

        if searched and not self.args.test_only:
            for stale in (ga_csv, singles_csv):
                Path(stale).unlink(missing_ok=True)
            gen, generations, population = cell["gen_schemas"], cell["generations"], cell["population"]
            if gen == 0:  # a transfer cell promoted to search: use the largest searched budget
                gen, generations, population = 32, 3, 20
            ok_ga = self.run(f"{name} GA (opt)", self.common(cloud, opt, queries) + [
                "--generate-schemas", str(gen), "--optimizer-generations", str(generations),
                "--optimizer-population", str(population), "--parallel", str(self.args.parallel),
                "--generated-schema-dir", str(self.out / "generated_schemas"),
                "--optimizer-output-dir", str(self.out / "best_schemas" / f"{name}_ga"), "--csv", ga_csv])
            ok_singles = self.run(f"{name} tuned singles (opt)", ["--flat-search"] + self.common(cloud, opt, queries) + [
                "--schemas", self.tuned, "--generate-schemas", "0", "--no-baselines", "--csv", singles_csv])
            if not (ok_ga or ok_singles):
                print(f"{name}: all search arms failed - refusing the TEST re-measure")
                return

        if searched:
            # The GA's best nested schema goes to TEST even when its overall winner is a single,
            # so every searched cell answers "best nested vs best single" on held-out queries.
            winners = [w for w in (best_schema_path(ga_csv), best_schema_path(ga_csv, nested_only=True),
                                   best_schema_path(singles_csv)) if w]
        else:
            winners = self.donor_winners(cell, cells)
        winners = list(dict.fromkeys(winners))
        Path(test_csv).unlink(missing_ok=True)
        # Several processes, not just more repeats: the same schema measured 2.7 and 3.6 us in two
        # processes at 100M (memory placement), which repeats inside one process cannot see. The
        # order is rotated so no schema always pays for running first.
        order = winners + self.defaults + HAND_NESTED
        runs = []
        for k in range(self.args.test_runs):
            shift = k * len(order) // self.args.test_runs
            run_csv = str(self.out / f"{name}_test_r{k}.csv")
            Path(run_csv).unlink(missing_ok=True)
            if self.run(f"{name} TEST re-measure {k + 1}/{self.args.test_runs}", ["--flat-search"] + self.common(cloud, test, queries) + [
                    "--schemas", ";".join(order[shift:] + order[:shift]), "--generate-schemas", "0", "--no-baselines",
                    "--measure-repeats", "5", "--csv", run_csv]):
                runs.append(run_csv)
        merge_runs(runs, test_csv)

    def donor_winners(self, cell, cells):
        donors = sorted((c for c in cells if c["scene"] == cell["scene"] and
                         (c["arm"] == "search" or c["cell"] in self.args.search_transfer) and c["cell"] != cell["cell"]),
                        key=lambda c: c["points"], reverse=True)
        for donor in donors:
            arms = [str(self.out / f"{donor['cell']}_{arm}.csv") for arm in ("ga", "singles")]
            winners = [w for w in (best_schema_path(arms[0]), best_schema_path(arms[0], nested_only=True),
                                   best_schema_path(arms[1])) if w]
            if winners:
                print(f"{cell['cell']}: transferring winners from {donor['cell']}")
                return winners
        print(f"{cell['cell']}: no searched donor has winners in {self.out}")
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", default="configs/datasets/curated_cells.json")
    parser.add_argument("--only", default="", help="comma list of cell names")
    parser.add_argument("--out", default="results/eval_traces/matrix_e6")
    default_exe = REPO / ("x64/Release/MultiDataStructure.exe" if platform.system() == "Windows" else "../build/MultiDataStructure")
    parser.add_argument("--exe", default=str(default_exe))
    parser.add_argument("--data-root", default="")
    parser.add_argument("--trace-root", default="")
    # 1: time candidates one at a time. Eight at a time roughly doubled the GA's latencies and made
    # its ranking noisy enough to pick the wrong winner (testing.md 2e).
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--test-runs", type=int, default=3, help="separate processes for the held-out re-measure")
    parser.add_argument("--cpu", type=int, default=-1, help="pin every run to this core (workstation: 4, a P-core)")
    parser.add_argument("--numa-node", type=int, default=-1, help="Linux: bind memory to this NUMA node")
    parser.add_argument("--tier-c-queries", type=int, default=0, help="query count for tier C cells (config: 300-600)")
    parser.add_argument("--search-transfer", default="", help="comma list of transfer cells to SEARCH instead")
    parser.add_argument("--test-only", action="store_true", help="only the held-out re-measure, reusing arm CSVs")
    args = parser.parse_args()
    args.search_transfer = {c for c in args.search_transfer.split(",") if c}

    config = json.loads((REPO / args.cells).read_text())
    cells = config["cells"]
    only = [c for c in args.only.split(",") if c]
    selected = [c for c in cells if not only or c["cell"] in only]
    missing = set(only) - {c["cell"] for c in selected}
    if missing:
        sys.exit(f"unknown cells: {', '.join(sorted(missing))}")

    battery = Battery(args)
    print(f"{len(selected)} cell(s), exe {args.exe}, out {args.out}", flush=True)
    # Search cells first, so transfer cells find their donors' winners.
    for cell in sorted(selected, key=lambda c: (c["arm"] != "search" and c["cell"] not in args.search_transfer, c["points"])):
        battery.cell(cell, cells)

    print(f"\n########## battery complete [{time.strftime('%H:%M:%S')}] ##########")
    if battery.failed:
        print(f"{len(battery.failed)} step(s) failed:\n  " + "\n  ".join(battery.failed))
        sys.exit(1)


if __name__ == "__main__":
    main()

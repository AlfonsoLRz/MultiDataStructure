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
import statistics
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
        # Rank candidates on telemetry-off passes (the score reads them when repeats >= 2), each
        # timed for at least 30 ms; pass 0 alone - instrumented, cold - ranked them until e7.
        timed = ["--measure-repeats", "3", "--min-timed-ms", "30"]

        if searched and not self.args.test_only:
            for stale in (ga_csv, singles_csv):
                Path(stale).unlink(missing_ok=True)
            gen, generations, population = cell["gen_schemas"], cell["generations"], cell["population"]
            if gen == 0:  # a transfer cell promoted to search: use the largest searched budget
                gen, generations, population = 32, 3, 20
            ok_ga = self.run(f"{name} GA (opt)", self.common(cloud, opt, queries) + timed + [
                "--generate-schemas", str(gen), "--optimizer-generations", str(generations),
                "--optimizer-population", str(population), "--parallel", str(self.args.parallel),
                "--generated-schema-dir", str(self.out / "generated_schemas"),
                "--optimizer-output-dir", str(self.out / "best_schemas" / f"{name}_ga"), "--csv", ga_csv])
            ok_singles = self.run(f"{name} tuned singles (opt)", ["--flat-search"] + self.common(cloud, opt, queries) + timed + [
                "--schemas", self.tuned, "--generate-schemas", "0", "--no-baselines", "--csv", singles_csv])
            if not (ok_ga or ok_singles):
                print(f"{name}: all search arms failed - refusing the TEST re-measure")
                return

        if searched:
            pick = self.confirm(name, cloud, opt, queries, ga_csv, singles_csv)
            winners = [pick.get("pick"), pick.get("best_nested"), pick.get("best_single"),
                       best_schema_path(ga_csv), best_schema_path(singles_csv)]
        else:
            winners = self.donor_winners(cell, cells)
            (self.out / f"{name}_pick.json").write_text(json.dumps(self.donor_pick or {}, indent=2))
        winners = list(dict.fromkeys(w for w in winners if w))

        # Held-out: one sequential run for the structure metrics and the far-off schemas, then the
        # contenders (within 2x of the best) timed interleaved in --test-runs processes.
        Path(test_csv).unlink(missing_ok=True)
        order = winners + self.defaults + HAND_NESTED
        if not self.run(f"{name} TEST structure", ["--flat-search"] + self.common(cloud, test, queries) + [
                "--schemas", ";".join(order), "--generate-schemas", "0", "--no-baselines",
                "--measure-repeats", "3", "--csv", test_csv]):
            return
        latency = {}
        with open(test_csv, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                latency.setdefault(row["schema_path"].replace("\\", "/"), float(row["latency_repeat_mean_ms"] or row["avg_latency_ms"]))
        best = min(latency.values())
        contenders = list(dict.fromkeys([w for w in winners if w in latency] +
                                        sorted((p for p in latency if latency[p] <= 2.0 * best), key=latency.get)))
        contenders = contenders[:self.args.max_interleave]
        self.interleaved(f"{name} TEST timing", cloud, test, queries, contenders, str(self.out / f"{name}_test_timing.csv"))

    def interleaved(self, title, cloud, trace, queries, schemas, out_csv):
        """engine-check --interleave over `schemas` in --test-runs processes, order rotated; merged."""
        runs = []
        for k in range(self.args.test_runs):
            shift = k * len(schemas) // self.args.test_runs
            run_csv = f"{out_csv[:-4]}_r{k}.csv"
            Path(run_csv).unlink(missing_ok=True)
            if self.run(f"{title} {k + 1}/{self.args.test_runs}", [
                    "--mode", "engine-check", "--input", cloud, "--input-trace", trace, "--queries", str(queries),
                    "--schemas", ";".join(schemas[shift:] + schemas[:shift]), "--warmup", "1", "--measure-repeats", "10",
                    "--interleave", "--summary-csv", run_csv, "--no-pause"]):
                runs.append(run_csv)
        Path(out_csv).unlink(missing_ok=True)
        merge_runs(runs, out_csv)
        return out_csv

    def confirm(self, name, cloud, trace, queries, ga_csv, singles_csv):
        """Re-time the top nested and single finalists interleaved on the opt half; safe pick.

        The GA ranks candidates one at a time, each in its own few milliseconds of a shared
        machine, so its top few are within noise of each other: in e7 a nested schema that looked
        3% better on the opt half lost on held-out queries in 9 of 16 losses. Here the finalists
        are timed side by side, and a nested schema is adopted only if it beats the best single by
        --nested-margin in the median over processes AND in every process; otherwise the single
        is the pick. Schemas that build the same tree (equal structure digest) count as one, the
        one with fewer blocks."""
        rows = []
        for path in (ga_csv, singles_csv):
            if Path(path).exists():
                with open(path, newline="", encoding="utf-8-sig") as handle:
                    all_rows = list(csv.DictReader(handle))
                final = [r for r in all_rows if r.get("score_is_final_latency") == "1"] or all_rows
                rows += final
        candidates = {}
        for row in sorted(rows, key=lambda r: float(r.get("score") or "inf")):
            path = (row.get("schema_path") or "").replace("\\", "/")
            if path and path not in candidates and (Path(path) if Path(path).is_absolute() else REPO / path).exists():
                candidates[path] = dict(blocks=schema_blocks(path), digest=row.get("structure_digest", ""))
        # One candidate per tree: a single beats any nested schema with its digest (fewer blocks),
        # otherwise the best-ranked holder of a digest stands for it.
        single_digests = {c["digest"] for c in candidates.values() if c["blocks"] == 1 and c["digest"]}
        seen, nested, singles = set(), [], []
        for path, c in candidates.items():
            if c["digest"] and (c["digest"] in seen or (c["blocks"] > 1 and c["digest"] in single_digests)):
                continue
            seen.add(c["digest"])
            (nested if c["blocks"] > 1 else singles).append(path)
        nested, singles = nested[:3], singles[:3]
        result = dict(nested_finalists=nested, single_finalists=singles)
        if not singles:
            result.update(pick=nested[0] if nested else None, adopted="nested" if nested else None)
        elif not nested:
            result.update(pick=singles[0], best_single=singles[0], adopted="single")
        else:
            out_csv = self.interleaved(f"{name} confirm (opt)", cloud, trace, queries, nested + singles,
                                       str(self.out / f"{name}_confirm.csv"))
            per_run = {}
            if Path(out_csv).exists():
                with open(out_csv, newline="", encoding="utf-8-sig") as handle:
                    for row in csv.DictReader(handle):
                        per_run.setdefault(row["test_run"], {})[row["schema_path"].replace("\\", "/")] = float(row["all_mean_ms"])
            runs = [r for r in per_run.values() if all(p in r for p in nested + singles)]
            if not runs:
                result.update(pick=singles[0], best_single=singles[0], best_nested=nested[0], adopted="single (confirm failed)")
            else:
                med = lambda p: statistics.median(r[p] for r in runs)
                best_nested, best_single = min(nested, key=med), min(singles, key=med)
                ratios = [r[best_nested] / r[best_single] for r in runs]
                ratio = statistics.median(ratios)
                adopt = ratio <= 1.0 - self.args.nested_margin and max(ratios) < 1.0
                result.update(best_nested=best_nested, best_single=best_single, ratio_runs=ratios, ratio=ratio,
                              pick=best_nested if adopt else best_single, adopted="nested" if adopt else "single")
        (self.out / f"{name}_pick.json").write_text(json.dumps(result, indent=2))
        print(f"{name}: pick {result.get('adopted')} {result.get('pick')} (nested/single {result.get('ratio', float('nan')):.3f})", flush=True)
        return result

    def donor_winners(self, cell, cells):
        self.donor_pick = None
        donors = sorted((c for c in cells if c["scene"] == cell["scene"] and
                         (c["arm"] == "search" or c["cell"] in self.args.search_transfer) and c["cell"] != cell["cell"]),
                        key=lambda c: c["points"], reverse=True)
        for donor in donors:
            pick_json = self.out / f"{donor['cell']}_pick.json"
            if pick_json.exists():
                pick = json.loads(pick_json.read_text())
                self.donor_pick = dict(pick, donor=donor["cell"])
                winners = [w for w in (pick.get("pick"), pick.get("best_nested"), pick.get("best_single")) if w]
                if winners:
                    print(f"{cell['cell']}: transferring the safe pick and finalists of {donor['cell']}")
                    return winners
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
    parser.add_argument("--test-runs", type=int, default=3, help="separate processes for each interleaved timing step")
    parser.add_argument("--nested-margin", type=float, default=0.03,
                        help="a nested finalist must beat the best single by this fraction to be the pick")
    parser.add_argument("--max-interleave", type=int, default=8,
                        help="most schemas timed side by side (each holds a full index in memory)")
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

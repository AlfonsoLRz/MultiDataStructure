"""Get every cell in a cell config ready to run: cache, trace, split, check.

    python scripts/prepare_curated_cells.py
    python scripts/prepare_curated_cells.py --only alhambra_1M alhambra_5M
    python scripts/prepare_curated_cells.py --check-only

Three steps per cell, each skipped when its output is already current:

1. convert the tier's `.laz` to `.mdspc` (scripts/laz_to_mdspc.py) - the C++ cannot
   read LAZ, and the cache is what the executable, the Indexicon driver and the
   framework harness all consume, so they index identical positions;
2. capture a pipeline trace from that same `.mdspc`
   (scripts/capture_pipeline_workload.py);
3. split it into the `_opt` half the search sees and the `_test` half every
   headline number comes from (scripts/split_trace.py).

Then it checks the thing that went wrong last time. Mean spacing must fall as a
scene's tiers grow: for a surface-like cloud, spacing ~ N^-0.5. The estimator bug
fixed on 2026-08-06 produced a nearly flat curve, which silently made query radii
cover a bigger neighbourhood at every rung and confounded the scale comparison.
The fitted exponent per scene is reported so a regression is visible rather than
buried in a radius column.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# A surface sampled at uniform density has spacing proportional to N^-0.5; a fully
# volumetric one, N^-0.33. Anything flatter than -0.25 means the spacing is not
# tracking density and the derived radii are not comparable across the ladder.
FLAT_EXPONENT_LIMIT = -0.25


def run(command: list[str]) -> bool:
    print(f"    $ {' '.join(str(c) for c in command[1:])}", flush=True)
    return subprocess.run(command, cwd=REPO).returncode == 0


def cache_is_current(cell: dict) -> bool:
    import laz_to_mdspc

    destination = Path(cell["mdspc_path"])
    source = Path(cell["laz_path"])
    if not destination.exists():
        return False
    try:
        header = laz_to_mdspc.read_header(destination)
    except (OSError, ValueError):
        return False
    size, write_time = laz_to_mdspc.source_stamp(source)
    return (header["magic"] == laz_to_mdspc.CACHE_MAGIC
            and header["version"] == laz_to_mdspc.CACHE_VERSION
            and header["source_size"] == size
            and header["source_write_time"] == write_time)


def trace_is_current(cell: dict) -> bool:
    """A trace counts as current only if it came from the cache now on disk.

    The sidecar records the cloud path and the estimator version. A trace captured
    before its tier was rebuilt, or by the old estimator, is not comparable with one
    captured after, and reusing it is exactly the mistake this checks for.
    """
    import capture_pipeline_workload as capture

    trace_dir = REPO / cell["trace_dir"]
    sidecar = trace_dir / "pipeline_workload.json"
    for required in (sidecar, trace_dir / "pipeline_trace_opt.csv", trace_dir / "pipeline_trace_test.csv"):
        if not required.exists():
            return False

    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if meta.get("spacing_estimator_version") != capture.SPACING_ESTIMATOR_VERSION:
        return False
    if meta.get("points_loaded") != cell["points"]:
        return False
    return sidecar.stat().st_mtime >= Path(cell["mdspc_path"]).stat().st_mtime


def spacing_report(cells: list[dict]) -> list[str]:
    """Fitted spacing exponent per scene, and the problems worth stopping for."""
    by_scene: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for cell in cells:
        sidecar = REPO / cell["trace_dir"] / "pipeline_workload.json"
        if not sidecar.exists():
            continue
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        by_scene[cell["scene"]].append((cell["points"], meta["mean_spacing"]))

    problems = []
    print(f"\n{'scene':<20} {'tiers':>5} {'spacing at smallest':>20} {'at largest':>12} {'exponent':>9}")
    for scene in sorted(by_scene):
        series = sorted(by_scene[scene])
        if len(series) < 2:
            counts, spacings = zip(*series)
            print(f"{scene:<20} {len(series):>5} {spacings[0]:>20.6f} {'-':>12} {'-':>9}")
            continue
        counts = np.array([c for c, _ in series], dtype=float)
        spacings = np.array([s for _, s in series], dtype=float)
        exponent = float(np.polyfit(np.log(counts), np.log(spacings), 1)[0])
        flag = "" if exponent <= FLAT_EXPONENT_LIMIT else "  <-- FLAT"
        if flag:
            problems.append(f"{scene}: spacing exponent {exponent:.3f} is flatter than "
                            f"{FLAT_EXPONENT_LIMIT}; radii are not tracking density")
        print(f"{scene:<20} {len(series):>5} {spacings[0]:>20.6f} {spacings[-1]:>12.6f} "
              f"{exponent:>9.3f}{flag}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs/datasets/curated_cells.json")
    parser.add_argument("--only", nargs="*", help="cell names to prepare (default: all)")
    parser.add_argument("--check-only", action="store_true", help="report state, convert and capture nothing")
    parser.add_argument("--max-queries-per-stage", type=int, default=20_000)
    parser.add_argument("--force", action="store_true", help="reconvert and recapture even when current")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO / "scripts"))

    if not args.config.exists():
        sys.exit(f"cell config not found: {args.config} - run scripts/make_cell_config.py first")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cells = [c for c in config["cells"] if not args.only or c["cell"] in args.only]
    if not cells:
        sys.exit("no cells selected")

    print(f"{len(cells)} cell(s) from {args.config}")
    failed, missing = [], []

    for index, cell in enumerate(cells, 1):
        name = cell["cell"]
        if not Path(cell["laz_path"]).exists():
            print(f"[{index}/{len(cells)}] {name}: source missing, skipping")
            missing.append(name)
            continue

        needs_cache = args.force or not cache_is_current(cell)
        needs_trace = args.force or needs_cache or not trace_is_current(cell)
        state = "cache+trace" if needs_cache else ("trace" if needs_trace else "current")
        print(f"[{index}/{len(cells)}] {name} ({cell['points']:,} pts, {cell['morphology']}): {state}")
        if args.check_only or state == "current":
            continue

        if needs_cache and not run([PYTHON, "scripts/laz_to_mdspc.py", cell["laz_path"]]):
            failed.append(f"{name} (conversion)")
            continue

        if needs_trace:
            trace_dir = cell["trace_dir"]
            ok = run([PYTHON, "scripts/capture_pipeline_workload.py", cell["mdspc_path"],
                      "--out", trace_dir,
                      "--max-queries-per-stage", str(args.max_queries_per_stage)])
            if not ok:
                failed.append(f"{name} (capture)")
                continue
            if not run([PYTHON, "scripts/split_trace.py", f"{trace_dir}/pipeline_trace.csv"]):
                failed.append(f"{name} (split)")

    problems = spacing_report(cells)

    if missing:
        print(f"\n{len(missing)} cell(s) had no source file: {', '.join(missing)}")
    if failed:
        print(f"\n{len(failed)} step(s) failed:")
        for line in failed:
            print(f"  {line}")
    if problems:
        print("\nspacing problems:")
        for line in problems:
            print(f"  {line}")

    return 1 if (failed or problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a uniform (shape x size) point-cloud matrix for controlled schema-search studies.

The evaluation datasets so far are ad-hoc sizes of whatever each source provides
(5M/50M/200M terrain, 100M architecture, 700M industrial), which confounds shape
with scale. This script subsamples every source cloud to the SAME size ladder so
"winner vs shape" and "winner vs scale" can be studied independently:

    shapes  = the distinct source distributions (terrain, architecture, industrial,
              indoor, ...)
    sizes   = 1M, 5M, 25M, 100M (each shape truncated at its native point count)

Each cell is written as a .las preserving world coordinates (offsets = min), so the
existing capture scripts and trace replay work unchanged. A manifest JSON lists the
grid for the experiment runner.

Example:
  python scripts/make_dataset_matrix.py --out "D:/Datasets/Point Clouds/matrix" \
      --sizes 1 5 25 100 \
      terrain="D:/Datasets/Point Clouds/SanAndreas/200M.las" \
      architecture="D:/Datasets/Point Clouds/Alhambra/Alhambra_100M.las" \
      industrial="D:/Datasets/Point Clouds/SolarPanels/SolarPanels700M.las" \
      indoor="D:/Datasets/Point Clouds/Sketchfab/hintze-hall-lo - Cloud.las"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def write_las(dst: Path, pts: np.ndarray, scale: float = 0.001) -> None:
    import laspy

    header = laspy.LasHeader(point_format=0, version="1.2")
    header.scales = np.array([scale, scale, scale])
    header.offsets = pts.min(axis=0)
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    las.write(str(dst))


def sample_indices(count: int, target: int, seed: int) -> np.ndarray:
    """Deterministic uniform sample of `target` indices out of `count`, without replacement.

    `pts[::stride]` is wrong here for two reasons. A LAS file is usually ordered by scan line
    or by tile, so a stride correlates with position and thins anisotropically instead of
    uniformly. Worse, `stride = count // target` collapses to 1 whenever
    `target <= count < 2*target`, making `pts[::1][:target]` a raw PREFIX -- one flight line
    or one corner of the survey, not a subsample of the shape at all.

    Sorted output preserves the source's memory/file locality, which keeps the LAS write and
    any later sequential read fast. Memory stays modest: one float64 key array per chunk
    rather than a permutation of `count`, so this is usable at 700M points.
    """
    rng = np.random.default_rng(seed)
    if target >= count:
        return np.arange(count)

    # Bernoulli oversample then trim. Drawing p slightly above target/count makes a short draw
    # vanishingly unlikely; the loop covers the tail case rather than trusting the margin.
    picked: list[np.ndarray] = []
    chosen = 0
    chunk = 1 << 24
    p = min(1.0, (target / count) * 1.02 + 1e-6)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        mask = rng.random(stop - start) < p
        idx = np.nonzero(mask)[0]
        if len(idx):
            picked.append(idx + start)
            chosen += len(idx)
    pool = np.concatenate(picked) if picked else np.empty(0, dtype=np.int64)

    if len(pool) < target:  # unlucky draw: top up from the complement
        remaining = np.setdiff1d(np.arange(count), pool, assume_unique=False)
        extra = rng.choice(remaining, size=target - len(pool), replace=False)
        pool = np.concatenate([pool, extra])
    if len(pool) > target:
        pool = rng.choice(pool, size=target, replace=False)
    pool.sort()
    return pool


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from capture_pipeline_workload import load_points

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shapes", nargs="+", help="name=path pairs, one per shape/distribution")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 25, 100], help="sizes in millions of points")
    parser.add_argument("--seed", type=int, default=7919,
                        help="Base seed for the subsample. Each rung uses seed+size_millions, so "
                             "rungs are independent draws but the whole matrix is reproducible.")
    parser.add_argument("--size-tolerance", type=float, default=0.01,
                        help="Accept a source within this fraction of a rung (default 1%%). A "
                             "cloud 0.2%% short of 100M still fills the 100M rung, which keeps "
                             "the matrix rectangular; the manifest records the true count.")
    parser.add_argument("--las-scale", type=float, default=0.001,
                        help="LAS coordinate scale for the written cells (default 1mm). Raise the "
                             "precision for sub-millimetre sources, which would otherwise be "
                             "quantised on write.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"sizes_millions": args.sizes, "cells": []}

    for pair in args.shapes:
        name, _, path = pair.partition("=")
        if not path:
            raise SystemExit(f"expected name=path, got: {pair}")
        source = Path(path)
        print(f"[{name}] loading {source} ...")
        pts = load_points(source, 0)
        print(f"[{name}] {len(pts):,} points")

        for size_m in args.sizes:
            target = size_m * 1_000_000
            # Tolerance: a source 0.2% short of a rung (e.g. Alhambra at 99,999,776 for the
            # 100M rung) used to drop the cell silently, which is what left the matrix ragged
            # and made cross-shape comparisons at the top rung compare different sizes.
            floor = int(target * (1.0 - args.size_tolerance))
            if len(pts) < floor:
                print(f"[{name}] skip {size_m}M (source has {len(pts):,}, "
                      f"below the {floor:,} tolerance floor)")
                continue
            take = min(target, len(pts))
            idx = sample_indices(len(pts), take, args.seed + size_m)
            sub = pts[idx]
            dst = args.out / f"{name}_{size_m}M.las"
            write_las(dst, sub, args.las_scale)
            short = "" if len(sub) == target else f"  [{target - len(sub):,} short of nominal, within tolerance]"
            manifest["cells"].append({"shape": name, "size_millions": size_m,
                                      "points": len(sub), "nominal_points": target,
                                      "path": str(dst), "source": str(source),
                                      "source_points": int(len(pts)),
                                      "sample_seed": args.seed + size_m,
                                      "sample_method": "uniform_without_replacement"})
            print(f"[{name}] wrote {dst.name} ({len(sub):,} points){short}")

    manifest_path = args.out / "matrix_manifest.json"
    with open(manifest_path, "w", newline="\n") as out:
        json.dump(manifest, out, indent=2)
    print(f"manifest: {manifest_path} ({len(manifest['cells'])} cells)")


if __name__ == "__main__":
    main()

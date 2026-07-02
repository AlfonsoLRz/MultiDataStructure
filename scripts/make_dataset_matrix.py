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


def write_las(dst: Path, pts: np.ndarray) -> None:
    import laspy

    header = laspy.LasHeader(point_format=0, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = pts.min(axis=0)
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    las.write(str(dst))


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from capture_pipeline_workload import load_points

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shapes", nargs="+", help="name=path pairs, one per shape/distribution")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 25, 100], help="sizes in millions of points")
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
            if target > len(pts):
                print(f"[{name}] skip {size_m}M (source has only {len(pts):,})")
                continue
            stride = max(1, len(pts) // target)
            sub = pts[::stride][:target]
            dst = args.out / f"{name}_{size_m}M.las"
            write_las(dst, sub)
            manifest["cells"].append({"shape": name, "size_millions": size_m,
                                      "points": len(sub), "path": str(dst), "source": str(source)})
            print(f"[{name}] wrote {dst.name} ({len(sub):,} points)")

    manifest_path = args.out / "matrix_manifest.json"
    with open(manifest_path, "w", newline="\n") as out:
        json.dump(manifest, out, indent=2)
    print(f"manifest: {manifest_path} ({len(manifest['cells'])} cells)")


if __name__ == "__main__":
    main()

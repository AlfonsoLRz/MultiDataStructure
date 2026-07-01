#!/usr/bin/env python3
"""Emit the neighborhood-query workload of a PointNet++-style network as replayable traces.

Deep point-cloud networks issue dense, machine-generated neighborhood queries at every
set-abstraction (SA) and feature-propagation (FP) layer, for every block, every batch,
every epoch. This script derives that exact query stream from the published layer
configuration (PointNet++ SSG defaults) over blocks sampled from a real cloud:

  per block (npoints, normalized to the unit sphere):
    SA1: FPS -> n1 centroids, ball query (radius r1, cap k1)   [targets the block]
    SA2: FPS -> n2 centroids, ball query (radius r2, cap k2)   [targets SA1 centroids]
    FP1: 3-NN interpolation, one query per block point          [targets SA1 centroids]
    FP2: 3-NN interpolation, one query per SA1 centroid         [targets SA2 centroids]

Each layer's queries are recorded against the point set that layer actually searches,
so the output is a list of (cloud file, trace file) pairs plus a batch manifest JSON —
the batch-distribution evaluation regime consumes the manifest to tune ONE schema for
the whole distribution (build time is first-class: real dataloaders rebuild per block).

Example:
  python scripts/capture_dl_workload.py "D:/Datasets/Point Clouds/SanAndreas/5M.las" \
      --out results/traces/dl_sanandreas --blocks 32 --npoints 4096
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TRACE_HEADER = (
    "query_id,query_type,bounds_min_x,bounds_min_y,bounds_min_z,"
    "bounds_max_x,bounds_max_y,bounds_max_z,center_x,center_y,center_z,"
    "radius,k,returned_points,latency_ms"
)

# PointNet++ SSG segmentation-style defaults (radii in unit-sphere-normalized coords).
SA_LAYERS = [
    {"name": "sa1", "npoint": 512, "radius": 0.2, "k": 32},
    {"name": "sa2", "npoint": 128, "radius": 0.4, "k": 64},
]
FP_K = 3


def load_points(path: Path, cap: int = 2_000_000) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).parent))
    from capture_pipeline_workload import load_points as load

    return load(path, cap)


def farthest_point_sampling(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(points)
    count = min(count, n)
    chosen = np.empty(count, dtype=np.int64)
    chosen[0] = rng.integers(n)
    distances = np.linalg.norm(points - points[chosen[0]], axis=1)
    for i in range(1, count):
        chosen[i] = int(np.argmax(distances))
        distances = np.minimum(distances, np.linalg.norm(points - points[chosen[i]], axis=1))
    return chosen


def write_cloud_xyz(path: Path, points: np.ndarray) -> None:
    np.savetxt(path, points, fmt="%.6f")


def write_trace(path: Path, rows: list) -> None:
    with open(path, "w", newline="\n") as out:
        out.write(TRACE_HEADER + "\n")
        for i, (kind, center, radius, k) in enumerate(rows):
            out.write(f"{i},{kind},0,0,0,0,0,0,{center[0]:.6f},{center[1]:.6f},{center[2]:.6f},{radius:.6f},{k},-1,0\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cloud", type=Path, help="source point cloud (.las/.laz/.ply/.xyz/.csv)")
    parser.add_argument("--out", type=Path, required=True, help="output directory (blocks, traces, manifest)")
    parser.add_argument("--blocks", type=int, default=32, help="number of sampled blocks (batch size analog)")
    parser.add_argument("--npoints", type=int, default=4096, help="points per block (PointNet++ input size)")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    from scipy.spatial import cKDTree

    points = load_points(args.cloud)
    print(f"working set: {len(points):,} points from {args.cloud}")
    tree = cKDTree(points)
    rng = np.random.default_rng(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(args.cloud), "blocks": [], "layers": SA_LAYERS, "fp_k": FP_K,
                "npoints": args.npoints, "seed": args.seed}

    for block_index in range(args.blocks):
        seed_point = points[rng.integers(len(points))]
        _, neighbor_idx = tree.query(seed_point, k=min(args.npoints, len(points)))
        block = points[np.atleast_1d(neighbor_idx)]

        # Normalize to the unit sphere (PointNet++ convention; the SA radii assume it).
        centroid = block.mean(axis=0)
        block = block - centroid
        scale = np.max(np.linalg.norm(block, axis=1))
        if scale > 0:
            block = block / scale

        block_name = f"block_{block_index:03d}"
        entries = []

        # SA1: FPS centroids over the block, ball queries against the BLOCK.
        sa1 = SA_LAYERS[0]
        sa1_idx = farthest_point_sampling(block, sa1["npoint"], args.seed + block_index)
        sa1_centroids = block[sa1_idx]
        block_file = args.out / f"{block_name}.xyz"
        write_cloud_xyz(block_file, block)
        sa1_rows = [("radius", c, sa1["radius"], sa1["k"]) for c in sa1_centroids]
        # FP1 also targets SA1 centroids; SA2 runs on them too -> next pair.
        sa1_trace = args.out / f"{block_name}_sa1.csv"
        write_trace(sa1_trace, sa1_rows)
        entries.append({"cloud": block_file.name, "trace": sa1_trace.name, "layer": "sa1",
                        "index_points": len(block), "queries": len(sa1_rows)})

        # SA2 ball queries + FP1 3-NN interpolation, both against the SA1 CENTROIDS.
        sa2 = SA_LAYERS[1]
        sa2_idx = farthest_point_sampling(sa1_centroids, sa2["npoint"], args.seed + 7 * block_index + 1)
        sa2_centroids = sa1_centroids[sa2_idx]
        sa1_file = args.out / f"{block_name}_sa1pts.xyz"
        write_cloud_xyz(sa1_file, sa1_centroids)
        sa2_rows = [("radius", c, sa2["radius"], sa2["k"]) for c in sa2_centroids]
        fp1_rows = [("knn", p, 0.0, FP_K) for p in block]
        sa2_trace = args.out / f"{block_name}_sa2_fp1.csv"
        write_trace(sa2_trace, sa2_rows + fp1_rows)
        entries.append({"cloud": sa1_file.name, "trace": sa2_trace.name, "layer": "sa2+fp1",
                        "index_points": len(sa1_centroids), "queries": len(sa2_rows) + len(fp1_rows)})

        # FP2 3-NN interpolation against the SA2 CENTROIDS.
        sa2_file = args.out / f"{block_name}_sa2pts.xyz"
        write_cloud_xyz(sa2_file, sa2_centroids)
        fp2_rows = [("knn", c, 0.0, FP_K) for c in sa1_centroids]
        fp2_trace = args.out / f"{block_name}_fp2.csv"
        write_trace(fp2_trace, fp2_rows)
        entries.append({"cloud": sa2_file.name, "trace": fp2_trace.name, "layer": "fp2",
                        "index_points": len(sa2_centroids), "queries": len(fp2_rows)})

        manifest["blocks"].append({"name": block_name, "pairs": entries})
        if (block_index + 1) % 8 == 0:
            print(f"  {block_index + 1}/{args.blocks} blocks")

    manifest_path = args.out / "batch_manifest.json"
    with open(manifest_path, "w", newline="\n") as out:
        json.dump(manifest, out, indent=2)

    pairs = sum(len(b["pairs"]) for b in manifest["blocks"])
    queries = sum(e["queries"] for b in manifest["blocks"] for e in b["pairs"])
    print(f"manifest: {manifest_path} ({args.blocks} blocks, {pairs} cloud/trace pairs, {queries:,} queries)")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""A dense pipeline segment: every point of a contiguous run of the cloud, queried stage by stage.

The curated traces sample each stage with a stride of N/20000 points (5000 at 100M), so consecutive
queries are metres apart and a leaf is almost never read twice: the worst case for a cache. A real
pipeline stage queries every point in scan order, and the curated clouds are stored in scan order
(2000 consecutive points of Lille 100M span about 8 m), so consecutive queries fall in the same few
leaves. This writes that case: --windows runs of --window consecutive points spread over the file, each
run answered by the three stages of the curated trace (kNN k=16, radius outlier removal and Euclidean
clustering, with the cell's radii from pipeline_workload.json), stage after stage within a run, in
world coordinates and the trace CSV format the engine reads.

    python scripts/dense_trace.py --cloud D:/.../Lille_100M.laz.mdspc --workload results/traces/curated/lille_100M/pipeline_workload.json \\
        --out results/traces/curated/lille_100M/dense_trace.csv --window 5000 --windows 3
"""
import argparse
import json
import struct

import numpy as np

FIELDS = ("query_id,query_type,bounds_min_x,bounds_min_y,bounds_min_z,bounds_max_x,bounds_max_y,bounds_max_z,"
          "center_x,center_y,center_z,radius,k,returned_points,latency_ms")


def read_cloud(path):
    """Points of an .mdspc cache (float32, local frame) and its frame: world = origin + local * scale."""
    with open(path, "rb") as handle:
        header = handle.read(88)
    if header[:7] != b"MDSPC01":
        raise SystemExit(f"not an .mdspc cache: {path}")
    version = struct.unpack_from("<I", header, 8)[0]
    count = struct.unpack_from("<Q", header, 32)[0]
    if version == 3:
        origin = np.array(struct.unpack_from("<3d", header, 40))
        scale = np.array(struct.unpack_from("<3d", header, 64))
        offset = 88
    elif version == 2:
        origin, scale, offset = np.zeros(3), np.ones(3), 40
    else:
        raise SystemExit(f"unsupported .mdspc version {version}")
    points = np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=(count, 3))
    return points, origin, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--workload", required=True, help="the cell's pipeline_workload.json (stage radii and k)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=5000, help="consecutive points per run")
    parser.add_argument("--windows", type=int, default=3, help="runs, spread evenly over the file")
    args = parser.parse_args()

    points, origin, scale = read_cloud(args.cloud)
    stages = json.loads(open(args.workload).read())["stages"]
    n = len(points)
    starts = [int((i + 0.5) * n / args.windows) - args.window // 2 for i in range(args.windows)]
    lines, qid = [FIELDS], 0
    for start in starts:
        start = max(0, min(start, n - args.window))
        world = origin + np.asarray(points[start:start + args.window], dtype=np.float64) * scale
        for stage in stages:
            kind, radius, k = stage["query_type"], float(stage["radius"]), int(stage["k"])
            for x, y, z in world:
                lines.append(f"{qid},{kind},0,0,0,0,0,0,{x:.6f},{y:.6f},{z:.6f},{radius:.6f},{k},-1,0")
                qid += 1
    with open(args.out, "w", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"{args.out}: {qid} queries, {args.windows} runs of {args.window} points, {len(stages)} stages")


if __name__ == "__main__":
    main()

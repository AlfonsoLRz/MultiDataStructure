#!/usr/bin/env python3
"""Derive the query trace a standard LiDAR processing pipeline issues over a cloud.

The classic geometry-processing stages are *dense self-queries*: every point queries
the cloud with fixed stage parameters, so the exact query stream is derivable from
the cloud + the stage configuration without instrumenting any library:

  - normal estimation        -> one kNN(k) query per point
  - radius outlier removal   -> one radius(r, count) query per point
  - Euclidean clustering     -> one radius(eps) query per point (DBSCAN-style expansion
                                lower bound; real runs revisit points, so this is
                                conservative)

The output is the project's standard query-trace CSV (--query-trace column format),
one file per stage plus a combined pipeline trace, and a JSON sidecar recording the
stage parameters and FULL query counts (for end-to-end projections when the emitted
trace is subsampled with --max-queries-per-stage).

Radius parameters default to multiples of the cloud's mean point spacing, estimated
from a sample (reported in the sidecar for reproducibility).

Example:
  python scripts/capture_pipeline_workload.py "D:/Datasets/Point Clouds/SanAndreas/5M.las" \
      --out results/traces/sanandreas_5m --max-queries-per-stage 20000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TRACE_HEADER = (
    "query_id,query_type,bounds_min_x,bounds_min_y,bounds_min_z,"
    "bounds_max_x,bounds_max_y,bounds_max_z,center_x,center_y,center_z,"
    "radius,k,returned_points,latency_ms"
)


def load_points(path: Path, max_points: int = 0) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in (".las", ".laz"):
        import laspy

        with laspy.open(str(path)) as reader:
            total = reader.header.point_count
            stride = max(1, total // max_points) if max_points and total > max_points else 1
            chunks = []
            for chunk in reader.chunk_iterator(2_000_000):
                xyz = np.column_stack((np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)))
                chunks.append(xyz[::stride].astype(np.float64))
            return np.concatenate(chunks, axis=0)
    if suffix == ".ply":
        return load_ply(path, max_points)
    if suffix in (".xyz", ".txt", ".csv"):
        data = np.loadtxt(str(path), delimiter="," if suffix == ".csv" else None, usecols=(0, 1, 2))
        if max_points and len(data) > max_points:
            data = data[:: max(1, len(data) // max_points)]
        return data.astype(np.float64)
    raise SystemExit(f"unsupported cloud format: {path}")


def load_ply(path: Path, max_points: int = 0) -> np.ndarray:
    # Minimal PLY reader (binary_little_endian / ascii, x/y/z properties).
    with open(path, "rb") as handle:
        fmt, count, props, header_end = None, 0, [], 0
        while True:
            line = handle.readline()
            if not line:
                raise SystemExit("PLY: unexpected EOF in header")
            text = line.decode("ascii", "replace").strip()
            if text.startswith("format"):
                fmt = text.split()[1]
            elif text.startswith("element vertex"):
                count = int(text.split()[-1])
            elif text.startswith("property") and count and not props_done(props):
                parts = text.split()
                props.append((parts[1], parts[2]))
            elif text.startswith("element") and count:
                props.append(("__END__", "__END__"))
            elif text == "end_header":
                header_end = handle.tell()
                break
        props = [p for p in props if p[0] != "__END__"]
        type_map = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
                    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
                    "short": "i2", "ushort": "u2", "int": "i4", "int32": "i4",
                    "uint": "u4", "uint32": "u4"}
        if fmt == "ascii":
            data = np.loadtxt(str(path), skiprows=0, max_rows=count,
                              usecols=tuple(i for i, p in enumerate(props) if p[1] in ("x", "y", "z")),
                              comments=None, encoding="ascii", dtype=np.float64,
                              converters=None)
            return data if not max_points or len(data) <= max_points else data[:: max(1, len(data) // max_points)]
        endian = "<" if "little" in (fmt or "") else ">"
        dtype = np.dtype([(name, endian + type_map[t]) for t, name in props])
        handle.seek(header_end)
        raw = np.fromfile(handle, dtype=dtype, count=count)
        pts = np.column_stack((raw["x"], raw["y"], raw["z"])).astype(np.float64)
        if max_points and len(pts) > max_points:
            pts = pts[:: max(1, len(pts) // max_points)]
        return pts


def props_done(props):
    return any(p[0] == "__END__" for p in props)


def mean_spacing(points: np.ndarray, sample: int = 100_000, seed: int = 1337) -> float:
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=min(sample, len(points)), replace=False)
    sampled = points[idx]
    tree = cKDTree(points[rng.choice(len(points), size=min(len(points), 2_000_000), replace=False)]
                   if len(points) > 2_000_000 else points)
    dists, _ = tree.query(sampled, k=2)
    return float(np.mean(dists[:, 1]))


def write_stage_trace(path: Path, centers: np.ndarray, kind: str, radius: float, k: int) -> int:
    with open(path, "w", newline="\n") as out:
        out.write(TRACE_HEADER + "\n")
        for i, c in enumerate(centers):
            out.write(f"{i},{kind},0,0,0,0,0,0,{c[0]:.6f},{c[1]:.6f},{c[2]:.6f},{radius:.6f},{k},-1,0\n")
    return len(centers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cloud", type=Path, help="input point cloud (.las/.laz/.ply/.xyz/.csv)")
    parser.add_argument("--out", type=Path, required=True, help="output directory for traces + sidecar")
    parser.add_argument("--max-queries-per-stage", type=int, default=50_000,
                        help="deterministic stride subsample of query points per stage (0 = all points)")
    parser.add_argument("--max-load-points", type=int, default=0,
                        help="stride-subsample the cloud on load (0 = all); traces stay valid for the subsampled cloud")
    parser.add_argument("--normal-k", type=int, default=16, help="kNN k for normal estimation (PCL default-ish 16-30)")
    parser.add_argument("--outlier-radius-mult", type=float, default=4.0,
                        help="radius outlier removal radius, in mean-spacing multiples")
    parser.add_argument("--cluster-eps-mult", type=float, default=2.5,
                        help="clustering eps, in mean-spacing multiples")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    started = time.perf_counter()
    points = load_points(args.cloud, args.max_load_points)
    load_seconds = time.perf_counter() - started
    print(f"loaded {len(points):,} points from {args.cloud} in {load_seconds:.1f}s")

    spacing = mean_spacing(points, seed=args.seed)
    print(f"mean point spacing (sampled): {spacing:.6f}")

    args.out.mkdir(parents=True, exist_ok=True)

    total = len(points)
    step = max(1, total // args.max_queries_per_stage) if args.max_queries_per_stage else 1
    query_points = points[::step]

    stages = [
        ("normal_estimation", "knn", 0.0, args.normal_k),
        ("radius_outlier_removal", "radius", spacing * args.outlier_radius_mult, 0),
        ("euclidean_clustering", "radius", spacing * args.cluster_eps_mult, 0),
    ]

    sidecar = {
        "cloud": str(args.cloud),
        "points_loaded": total,
        "mean_spacing": spacing,
        "query_stride": step,
        "stages": [],
    }

    combined = args.out / "pipeline_trace.csv"
    with open(combined, "w", newline="\n") as out:
        out.write(TRACE_HEADER + "\n")
        query_id = 0
        for name, kind, radius, k in stages:
            stage_path = args.out / f"{name}_trace.csv"
            emitted = write_stage_trace(stage_path, query_points, kind, radius, k)
            for c in query_points:
                out.write(f"{query_id},{kind},0,0,0,0,0,0,{c[0]:.6f},{c[1]:.6f},{c[2]:.6f},{radius:.6f},{k},-1,0\n")
                query_id += 1
            sidecar["stages"].append({
                "name": name,
                "query_type": kind,
                "radius": radius,
                "k": k,
                "emitted_queries": emitted,
                "full_query_count": total,   # the real stage queries EVERY point
                "trace": stage_path.name,
            })
            print(f"stage {name}: {emitted:,} queries emitted (full stage = {total:,}) -> {stage_path.name}")

    with open(args.out / "pipeline_workload.json", "w", newline="\n") as out:
        json.dump(sidecar, out, indent=2)
    print(f"combined trace: {combined} | sidecar: {args.out / 'pipeline_workload.json'}")


if __name__ == "__main__":
    sys.exit(main())

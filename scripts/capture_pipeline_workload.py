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


def subsample_to(points: np.ndarray, max_points: int) -> np.ndarray:
    """Stride-subsample to at most `max_points`, honouring the cap exactly.

    `points[::len//max]` alone is wrong in two ways that both bite silently. Integer division
    yields stride 1 for any max_points <= len < 2*max_points, so the cap does nothing and the
    caller gets the whole cloud; and even with stride >= 2 the result overshoots the cap by up
    to one stride. Both are fixed here: the stride rounds up, and the result is truncated.
    """
    if not max_points or len(points) <= max_points:
        return points
    stride = -(-len(points) // max_points)  # ceil, so the result never exceeds max_points
    return points[::stride][:max_points]


def load_mdspc(path: Path, max_points: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Memory-map a `.mdspc` cache, returning (local float32 positions, origin).

    Materialising a billion points as float64 world coordinates is 24 GB, so the
    top of the scale ladder cannot be captured the way the smaller cells are. The
    cache is 12 B/point and already on disk, and mapping it means the trace is
    generated from exactly the positions the executable indexes.

    Positions stay in the cache's local frame; the caller adds the origin when it
    writes coordinates out. Spacing is translation-invariant, so the estimator does
    not care, and the frame convention (traces are world coordinates) is preserved
    at the one place it matters.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import laz_to_mdspc

    header = laz_to_mdspc.read_header(path)
    if header["magic"] != laz_to_mdspc.CACHE_MAGIC:
        raise SystemExit(f"not a point-cloud cache: {path}")
    points = np.memmap(path, dtype=np.float32, mode="r",
                       offset=laz_to_mdspc.HEADER_BYTES, shape=(header["points"], 3))
    return subsample_to(points, max_points), np.asarray(header["origin"], dtype=np.float64)


def load_points(path: Path, max_points: int = 0) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in (".las", ".laz"):
        import laspy

        with laspy.open(str(path)) as reader:
            total = reader.header.point_count
            # Stride must run over the WHOLE file, not restart per chunk, or the sample is
            # biased toward each chunk's leading points. Accumulate a global point counter and
            # slice each chunk at the right phase.
            stride = -(-total // max_points) if max_points and total > max_points else 1
            chunks = []
            seen = 0
            for chunk in reader.chunk_iterator(2_000_000):
                xyz = np.column_stack((np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)))
                offset = (-seen) % stride if stride > 1 else 0
                if offset < len(xyz):
                    chunks.append(xyz[offset::stride].astype(np.float64))
                seen += len(xyz)
            points = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3))
            return points[:max_points] if max_points and len(points) > max_points else points
    if suffix == ".ply":
        return load_ply(path, max_points)
    if suffix in (".xyz", ".txt", ".csv"):
        data = np.loadtxt(str(path), delimiter="," if suffix == ".csv" else None, usecols=(0, 1, 2))
        return subsample_to(data.astype(np.float64), max_points)
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
            return subsample_to(data, max_points)
        endian = "<" if "little" in (fmt or "") else ">"
        dtype = np.dtype([(name, endian + type_map[t]) for t, name in props])
        handle.seek(header_end)
        raw = np.fromfile(handle, dtype=dtype, count=count)
        pts = np.column_stack((raw["x"], raw["y"], raw["z"])).astype(np.float64)
        return subsample_to(pts, max_points)


def props_done(props):
    return any(p[0] == "__END__" for p in props)


# Bumped whenever the spacing estimator changes in a way that moves derived radii. Recorded
# in the sidecar so traces captured with an older estimator are identifiable rather than
# silently mixed into a comparison.
SPACING_ESTIMATOR_VERSION = 2

BLOCK_TARGET_POINTS = 2_000_000


def mean_spacing(points: np.ndarray, sample: int = 100_000, seed: int = 1337,
                 blocks: int = 6) -> float:
    """Mean nearest-neighbour distance, estimated from local spatial blocks.

    Nearest-neighbour spacing is a property of LOCAL density, so it must be measured
    against a neighbourhood at the cloud's true density. Version 1 of this function built
    the KD-tree from an independent global random subsample of at most 2M points and then
    queried it with points drawn separately from the full cloud. That was wrong twice over:
    the query points were usually absent from the tree, so `dists[:, 1]` returned the
    SECOND nearest neighbour rather than the first; and, far worse, a uniform global
    subsample of an N-point cloud is (N / 2M) times sparser than the cloud itself, which
    inflates every distance in it. Measured against an exact full-cloud computation the
    error grew with cloud size -- 1.00x at 1M points, 2.03x at 5M, 5.03x at 25M -- so the
    derived radii (4x spacing for outlier removal, 2.5x for clustering) were nearly
    CONSTANT in world units across a size ladder instead of shrinking with density. Since
    those radii define the workload, that silently changed what was being measured as a
    function of cloud size, confounding exactly the scale comparisons the matrix exists to
    make.

    This version instead samples several axis-aligned blocks sized to hold roughly
    BLOCK_TARGET_POINTS points, builds a tree per block, and queries only points in the
    block's interior -- points near a face would otherwise report an inflated distance
    because their true neighbour lies outside the block. Density inside a block is the
    cloud's real local density, so the estimate is unbiased, and the cost stays O(block)
    regardless of total cloud size. Clouds that already fit in one block are measured
    exactly.

    Blocks are anchored uniformly within the bounding box and the per-block means are
    combined weighted by how many points each contributed, which makes this an unbiased
    estimator of the same point-weighted mean the exact computation produces. Anchoring on
    randomly chosen POINTS instead would be more robust but biases the result toward dense
    regions -- measured 0.88-0.95x of truth on a bimodal industrial cloud -- so point
    anchoring is used only as a fallback when uniform anchors keep landing in empty space,
    as they do on sparse or L-shaped clouds.

    Measured against exact full-cloud computation over nine matrix cells (1M-25M, four
    morphologies, three seeds each): within 1% on homogeneous clouds, worst case 1.14x on
    industrial_25M, whose density is strongly bimodal (dense panels, sparse ground). Crucially
    the error does NOT grow with cloud size, which is what made the old estimator's bias
    corrosive. Smaller-but-more-numerous blocks were tried and are worse (up to 1.29x) because
    the interior margin discards proportionally more of a small block.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)

    if len(points) <= BLOCK_TARGET_POINTS:
        idx = rng.choice(len(points), size=min(sample, len(points)), replace=False)
        tree = cKDTree(points)
        dists, _ = tree.query(points[idx], k=2, workers=-1)
        return float(np.mean(dists[:, 1]))

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    extent = np.maximum(hi - lo, 1e-12)

    # Side length of a cube expected to hold BLOCK_TARGET_POINTS, assuming points are spread
    # over the bounding box. Real clouds are far from uniform, so the count is corrected by
    # measurement below rather than trusted.
    side = extent * (min(1.0, BLOCK_TARGET_POINTS / len(points)) ** (1.0 / 3.0))

    distance_total = 0.0
    sampled_total = 0
    accepted = 0
    empty_streak = 0
    attempts = 0
    max_attempts = blocks * 10

    while accepted < blocks and attempts < max_attempts:
        attempts += 1
        if empty_streak >= 4:
            anchor = points[rng.integers(len(points))]  # fallback: land on real data
        else:
            anchor = lo + rng.random(3) * extent

        block_lo = np.clip(anchor - side / 2.0, lo, hi)
        block_hi = np.minimum(block_lo + side, hi)
        block_lo = np.maximum(block_hi - side, lo)

        block = points_in_box(points, block_lo, block_hi)
        if len(block) < 1000:
            side = side * 1.6  # too sparse here: widen and retry
            empty_streak += 1
            continue
        if len(block) > BLOCK_TARGET_POINTS * 4:
            side = side * 0.7  # denser than assumed: shrink so the tree stays cheap
            continue
        empty_streak = 0

        # Query only the interior. The margin is a first-pass spacing estimate, so it adapts
        # to whatever density this block actually has.
        tree = cKDTree(block)
        probe = block[rng.choice(len(block), size=min(4096, len(block)), replace=False)]
        probe_dists, _ = tree.query(probe, k=2, workers=-1)
        margin = float(np.mean(probe_dists[:, 1])) * 4.0

        interior = np.all((block >= block_lo + margin) & (block <= block_hi - margin), axis=1)
        candidates = block[interior]
        if len(candidates) < 500:
            side = side * 1.6  # block too thin to have an interior; widen
            continue

        idx = rng.choice(len(candidates), size=min(sample, len(candidates)), replace=False)
        dists, _ = tree.query(candidates[idx], k=2, workers=-1)
        distance_total += float(np.sum(dists[:, 1]))
        sampled_total += len(idx)
        accepted += 1

    if sampled_total == 0:
        # Degenerate geometry (e.g. a plane, or one dense cluster): fall back to an exact
        # measurement over a capped prefix rather than returning a silently wrong number.
        capped = points[:BLOCK_TARGET_POINTS]
        tree = cKDTree(capped)
        idx = rng.choice(len(capped), size=min(sample, len(capped)), replace=False)
        dists, _ = tree.query(capped[idx], k=2, workers=-1)
        return float(np.mean(dists[:, 1]))

    return distance_total / sampled_total


def write_stage_trace(path: Path, centers: np.ndarray, kind: str, radius: float, k: int) -> int:
    with open(path, "w", newline="\n") as out:
        out.write(TRACE_HEADER + "\n")
        for i, c in enumerate(centers):
            out.write(f"{i},{kind},0,0,0,0,0,0,{c[0]:.6f},{c[1]:.6f},{c[2]:.6f},{radius:.6f},{k},-1,0\n")
    return len(centers)


def points_in_box(points: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  chunk: int = 20_000_000) -> np.ndarray:
    """Points inside an axis-aligned box, filtered in chunks.

    The obvious `points[np.all((points >= lo) & (points <= hi), axis=1)]` allocates
    two boolean arrays the size of the cloud. At a billion points that is 4 GB of
    temporaries per attempt, and the estimator makes up to sixty attempts. Chunking
    bounds the temporaries instead, and works the same on a memmap as on an array.
    """
    if len(points) <= chunk:
        return np.asarray(points[np.all((points >= lo) & (points <= hi), axis=1)])

    kept = []
    for start in range(0, len(points), chunk):
        part = np.asarray(points[start:start + chunk])
        hit = part[np.all((part >= lo) & (part <= hi), axis=1)]
        if len(hit):
            kept.append(hit)
    return np.concatenate(kept, axis=0) if kept else np.empty((0, 3), dtype=points.dtype)


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
    if args.cloud.suffix.lower() == ".mdspc":
        points, origin = load_mdspc(args.cloud, args.max_load_points)
    else:
        points, origin = load_points(args.cloud, args.max_load_points), np.zeros(3)
    load_seconds = time.perf_counter() - started
    print(f"loaded {len(points):,} points from {args.cloud} in {load_seconds:.1f}s")

    spacing = mean_spacing(points, seed=args.seed)
    print(f"mean point spacing (sampled): {spacing:.6f}")

    args.out.mkdir(parents=True, exist_ok=True)

    total = len(points)
    step = max(1, total // args.max_queries_per_stage) if args.max_queries_per_stage else 1
    # Only the strided subset is materialised, and only here does the frame matter:
    # trace coordinates are always world, whatever frame the source was read in.
    query_points = np.asarray(points[::step], dtype=np.float64) + origin

    stages = [
        ("normal_estimation", "knn", 0.0, args.normal_k),
        ("radius_outlier_removal", "radius", spacing * args.outlier_radius_mult, 0),
        ("euclidean_clustering", "radius", spacing * args.cluster_eps_mult, 0),
    ]

    sidecar = {
        "cloud": str(args.cloud),
        "points_loaded": total,
        "mean_spacing": spacing,
        # Traces captured with estimator version 1 have radii inflated by up to ~5x on
        # clouds above 2M points and are not comparable with version 2 -- see mean_spacing().
        "spacing_estimator_version": SPACING_ESTIMATOR_VERSION,
        "query_stride": step,
        # Non-zero only for .mdspc inputs, where positions are stored relative to the
        # cloud's origin. Recorded so a trace can be traced back to the exact frame.
        "source_origin": [float(v) for v in origin],
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

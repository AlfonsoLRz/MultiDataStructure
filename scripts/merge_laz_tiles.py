"""Merge a spatially contiguous subset of LAZ tiles into one world-coordinate LAS.

The tiled surveys (Waikato_Hamilton: 1240 tiles / ~11B pts; SanSimeon: 2336 tiles)
are far larger than any matrix cell needs, and random tile subsets would create
artificial density seams. This selects a contiguous blob instead: it reads only
LAZ *headers* (fast, no decompression), starts from the tile nearest the survey
centroid, greedily grows by the tile nearest the blob's running centroid until
--target-points is reached, then decompresses just those tiles into a single LAS
(point format 0, xyz only, world coordinates -- the convention every trace and
loader in this repo expects).

Usage:
  python scripts/merge_laz_tiles.py "D:/Datasets/Point Clouds/Waikato_Hamilton" \
      --out "D:/Datasets/Point Clouds/Waikato_Hamilton/hamilton_120M.las" --target-points 120000000
"""

import argparse
from pathlib import Path

import numpy as np
import laspy


def scan_headers(tiles: list) -> list:
    infos = []
    for tile in tiles:
        with laspy.open(tile) as reader:
            header = reader.header
            center = (
                (header.mins[0] + header.maxs[0]) / 2.0,
                (header.mins[1] + header.maxs[1]) / 2.0,
            )
            infos.append({"path": tile, "count": header.point_count, "center": center})
    return infos


def pick_contiguous(infos: list, target: int) -> list:
    centers = np.array([t["center"] for t in infos])
    centroid = centers.mean(axis=0)
    remaining = list(range(len(infos)))
    seed = min(remaining, key=lambda i: np.sum((centers[i] - centroid) ** 2))

    picked, total = [seed], infos[seed]["count"]
    remaining.remove(seed)
    blob = centers[seed].astype(float).copy()
    while remaining and total < target:
        nxt = min(remaining, key=lambda i: np.sum((centers[i] - blob) ** 2))
        remaining.remove(nxt)
        picked.append(nxt)
        total += infos[nxt]["count"]
        blob = centers[picked].mean(axis=0)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tile_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-points", type=int, default=120_000_000)
    parser.add_argument("--pattern", default="*.laz")
    parser.add_argument("--keep-noise", action="store_true",
                        help="keep LAS classification 7/18 (noise) points; dropped by default "
                             "because stray z outliers inflate every index's root AABB")
    args = parser.parse_args()

    tiles = sorted(args.tile_dir.glob(args.pattern))
    if not tiles:
        raise SystemExit(f"no {args.pattern} tiles under {args.tile_dir}")
    print(f"scanning {len(tiles)} tile headers ...")
    infos = scan_headers(tiles)
    total_available = sum(t["count"] for t in infos)
    print(f"survey total: {total_available:,} points")

    picked = pick_contiguous(infos, args.target_points)
    picked_total = sum(infos[i]["count"] for i in picked)
    print(f"picked {len(picked)} contiguous tiles = {picked_total:,} points "
          f"(target {args.target_points:,})")

    xs, ys, zs = [], [], []
    dropped = 0
    for n, i in enumerate(picked):
        with laspy.open(infos[i]["path"]) as reader:
            las = reader.read()
        tile_x = np.asarray(las.x, dtype=np.float64)
        tile_y = np.asarray(las.y, dtype=np.float64)
        tile_z = np.asarray(las.z, dtype=np.float64)
        if not args.keep_noise:
            classification = np.asarray(las.classification)
            keep = (classification != 7) & (classification != 18)
            dropped += int(len(keep) - keep.sum())
            tile_x, tile_y, tile_z = tile_x[keep], tile_y[keep], tile_z[keep]
        xs.append(tile_x)
        ys.append(tile_y)
        zs.append(tile_z)
        print(f"  [{n + 1}/{len(picked)}] {Path(infos[i]['path']).name}: {len(tile_x):,} pts")
    if dropped:
        print(f"dropped {dropped:,} noise-classified points (7/18)")
    x = np.concatenate(xs); del xs
    y = np.concatenate(ys); del ys
    z = np.concatenate(zs); del zs

    header = laspy.LasHeader(point_format=0, version="1.2")
    header.offsets = [x.min(), y.min(), z.min()]
    header.scales = [0.001, 0.001, 0.001]
    out = laspy.LasData(header)
    out.x, out.y, out.z = x, y, z
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(args.out))
    print(f"wrote {len(x):,} points -> {args.out}")
    print(f"bounds x[{x.min():.1f}, {x.max():.1f}] y[{y.min():.1f}, {y.max():.1f}] "
          f"z[{z.min():.1f}, {z.max():.1f}]")


if __name__ == "__main__":
    main()

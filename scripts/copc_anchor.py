#!/usr/bin/env python3
"""A production out-of-core format on the out-of-core battery's boxes (testing.md 2q).

Our leaf store is a research layout. To anchor its absolute numbers, the same cloud goes into a COPC
file (Cloud-Optimized Point Cloud: an octree of LAZ chunks with level-of-detail points in every
node, built by untwine), and PDAL's readers.copc reads each of the battery's range boxes from it,
one thread, with the file evicted from the page cache before every box (posix_fadvise DONTNEED,
no root needed). COPC also pays for LAZ decompression and carries every point's attributes; our
store reads raw floats, so the comparison bounds the store against a real format, not like for like.

    python scripts/copc_anchor.py --cell domfountain_100M --boxes ../results/out_of_core/domfountain_100M/boxes_trace.csv \\
        --data-root ../data --work ../store_tmp/copc --out ../results/out_of_core/domfountain_100M/copc.csv --rounds 2
Run inside the mds-copc conda env (pdal, python-pdal, untwine, laspy). The LAS and COPC files are
written under --work and deleted at the end.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import laspy
import numpy as np
import pdal

from run_cells import REPO, remap_cloud


def read_mdspc(path):
    with open(path, "rb") as handle:
        header = handle.read(88)
        if header[:7] != b"MDSPC01":
            raise SystemExit(f"not an .mdspc cache: {path}")
        version = int.from_bytes(header[8:12], "little")
        count = int.from_bytes(header[32:40], "little")
        origin = np.frombuffer(header[40:64], dtype="<f8") if version == 3 else np.zeros(3)
        scale = np.frombuffer(header[64:88], dtype="<f8") if version == 3 else np.ones(3)
        handle.seek(88 if version == 3 else 40)
        xyz = np.fromfile(handle, dtype="<f4", count=3 * count).reshape(-1, 3)
    return xyz, origin, scale


def write_las(xyz, origin, scale, path):
    """World coordinates at 0.1 mm, as the traces' boxes are in world coordinates."""
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.offsets = origin
    header.scales = np.array([1e-4, 1e-4, 1e-4])
    las = laspy.LasData(header)
    world = xyz.astype(np.float64) * scale + origin
    las.x, las.y, las.z = world[:, 0], world[:, 1], world[:, 2]
    las.write(path)


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--cells", default="configs/datasets/curated_cells.json")
    parser.add_argument("--boxes", required=True, help="the battery's boxes_trace.csv; its range rows are read")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--keep", action="store_true", help="keep the LAS and COPC files")
    args = parser.parse_args()

    cell = {c["cell"]: c for c in json.loads((REPO / args.cells).read_text())["cells"]}[args.cell]
    work = Path(args.work).resolve() / args.cell
    work.mkdir(parents=True, exist_ok=True)
    las_path, copc_path = work / "cloud.las", work / "cloud.copc.laz"

    started = time.time()
    xyz, origin, scale = read_mdspc(remap_cloud(cell["mdspc_path"], args.data_root))
    write_las(xyz, origin, scale, str(las_path))
    del xyz
    subprocess.run(["untwine", "-i", str(las_path), "-o", str(copc_path)], check=True)
    las_path.unlink()
    print(f"{args.cell}: COPC {copc_path.stat().st_size / 1e9:.2f} GB built in {(time.time() - started) / 60:.1f} min", flush=True)

    with open(args.boxes, newline="") as handle:
        boxes = [r for r in csv.DictReader(handle) if r["query_type"] == "range"]
    rows = []
    for round_ in range(args.rounds):
        for i, box in enumerate(boxes):
            lo = [float(box[f"bounds_min_{a}"]) for a in "xyz"]
            hi = [float(box[f"bounds_max_{a}"]) for a in "xyz"]
            bounds = f"([{lo[0]},{hi[0]}],[{lo[1]},{hi[1]}],[{lo[2]},{hi[2]}])"
            pipeline = pdal.Pipeline(json.dumps([{"type": "readers.copc", "filename": str(copc_path), "bounds": bounds, "threads": 1}]))
            evict(str(copc_path))
            start = time.perf_counter()
            points = pipeline.execute()
            seconds = time.perf_counter() - start
            rows.append({"cell": args.cell, "round": round_, "box": i, "points": points, "seconds": seconds,
                         "mpoints_per_s": points / seconds / 1e6, "copc_bytes": copc_path.stat().st_size})
            print(f"  round {round_} box {i}: {points} points in {seconds:.2f} s ({points / seconds / 1e6:.2f} Mpoints/s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

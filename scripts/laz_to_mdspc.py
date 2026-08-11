"""Convert LAS/LAZ point clouds into the project's own `.mdspc` position cache.

The C++ reader is a hand-rolled LAS parser with no LASzip, so it cannot open a
`.laz` at all. Rather than decompressing to LAS (three times the bytes, and a
slow re-parse on every cold load), this writes the cache format directly and the
cache becomes the input: `PointCloud::loadFromSource` accepts a `.mdspc` path,
and `tools/indexicon_point_baseline.cpp` already reads the same file, so both
sides of an external comparison consume byte-identical positions.

    python scripts/laz_to_mdspc.py cloud.laz
    python scripts/laz_to_mdspc.py --verify 100000 cloud.laz other.laz
    python scripts/laz_to_mdspc.py --cell-config configs/datasets/curated_cells.json

The default output is `<input>.mdspc`, which is also where `PointCloud::load`
looks for the sidecar of `<input>`, so passing either path works.

Coordinate frame: `PointCloud::loadLAS` puts the origin at the header's bounding
box minimum and keeps a unit scale, so positions are float32 metres relative to
that corner. This reproduces that exactly - a different frame would silently
shift every world coordinate relative to a cloud loaded from LAS.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    import laspy
except ImportError:  # pragma: no cover - environment problem, not a code path
    sys.exit("laspy is required: pip install \"laspy[lazrs]\"")

# PointCloud.cpp: CACHE_MAGIC, CACHE_VERSION.
CACHE_MAGIC = b"MDSPC01\0"
CACHE_VERSION = 3

# struct BinaryHeaderPrefix { char[8]; uint32; <4 pad>; uint64; int64; uint64; }
# followed by struct BinaryHeaderMetadata { double origin[3]; double scale[3]; }.
# Default MSVC alignment, no packing pragma: 40 + 48 = 88 bytes.
PREFIX_FORMAT = "<8sI4xQqQ"
METADATA_FORMAT = "<6d"
HEADER_BYTES = struct.calcsize(PREFIX_FORMAT) + struct.calcsize(METADATA_FORMAT)

# std::filesystem::file_time_type on MSVC counts 100 ns ticks from 1601-01-01,
# the same epoch as FILETIME. Python reports nanoseconds from the Unix epoch.
FILETIME_EPOCH_OFFSET_TICKS = 116_444_736_000_000_000

CHUNK_POINTS = 4_000_000


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def source_stamp(path: Path) -> tuple[int, int]:
    """(size, write time) as `headerMatchesSource` would read them."""
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns // 100 + FILETIME_EPOCH_OFFSET_TICKS


def write_header(handle, source: Path, num_points: int, origin, scale) -> None:
    size, write_time = source_stamp(source)
    handle.write(struct.pack(PREFIX_FORMAT, CACHE_MAGIC, CACHE_VERSION,
                             size, write_time, num_points))
    handle.write(struct.pack(METADATA_FORMAT, *origin, *scale))


def read_header(path: Path) -> dict:
    with path.open("rb") as handle:
        prefix = struct.unpack(PREFIX_FORMAT, handle.read(struct.calcsize(PREFIX_FORMAT)))
        metadata = struct.unpack(METADATA_FORMAT, handle.read(struct.calcsize(METADATA_FORMAT)))
    return dict(magic=prefix[0], version=prefix[1], source_size=prefix[2],
                source_write_time=prefix[3], points=prefix[4],
                origin=list(metadata[:3]), scale=list(metadata[3:]))


def header_bounds_look_valid(mins, maxs) -> bool:
    """The same test `loadLAS` applies before trusting the header's bbox."""
    finite = all(np.isfinite(v) for v in (*mins, *maxs))
    ordered = all(hi >= lo for lo, hi in zip(mins, maxs))
    return finite and ordered and any(hi > lo for lo, hi in zip(mins, maxs))


def convert(source: Path, destination: Path) -> dict:
    with laspy.open(str(source)) as reader:
        header = reader.header
        num_points = int(header.point_count)
        mins = [float(v) for v in header.mins]
        maxs = [float(v) for v in header.maxs]

        if not header_bounds_look_valid(mins, maxs):
            raise SystemExit(f"{source.name}: header bounding box is unusable, and the "
                             f"first-point origin fallback is not reproduced here")

        origin = mins
        scale = [1.0, 1.0, 1.0]
        log(f"{source.name}: {num_points:,} points, origin "
            f"({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f})")

        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with temporary.open("wb") as out:
            write_header(out, source, num_points, origin, scale)
            for chunk in reader.chunk_iterator(CHUNK_POINTS):
                local = np.empty((len(chunk), 3), dtype=np.float32)
                # chunk.x/y/z are the scaled doubles; subtract in double, store float32.
                local[:, 0] = np.asarray(chunk.x) - origin[0]
                local[:, 1] = np.asarray(chunk.y) - origin[1]
                local[:, 2] = np.asarray(chunk.z) - origin[2]
                out.write(local.tobytes())
                written += len(chunk)
                if written % (CHUNK_POINTS * 5) == 0:
                    log(f"    {written:,} / {num_points:,}")

    if written != num_points:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"{source.name}: wrote {written:,} points, header declared {num_points:,}")

    expected_bytes = HEADER_BYTES + num_points * 12
    actual_bytes = temporary.stat().st_size
    if actual_bytes != expected_bytes:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"{source.name}: cache is {actual_bytes:,} bytes, expected {expected_bytes:,}")

    destination.unlink(missing_ok=True)
    temporary.rename(destination)

    extent = max(hi - lo for lo, hi in zip(mins, maxs))
    # float32 keeps 24 significand bits, so the coarsest spacing anywhere in the
    # cloud is one ulp at the far corner. Reported because it is the floor on any
    # exactness comparison against a double-precision baseline.
    resolution = np.spacing(np.float32(extent))
    log(f"{source.name}: wrote {destination.name} ({actual_bytes / 2**30:.2f} GiB), "
        f"extent {extent:,.1f} m, float32 resolution {resolution * 1000:.3f} mm")

    return dict(source=str(source), destination=str(destination), points=num_points,
                origin=origin, scale=scale, extent=extent,
                float32_resolution_m=float(resolution))


def verify(source: Path, destination: Path, sample: int, seed: int = 7919) -> float:
    """Compare `sample` random cache positions against the LAS-decoded world ones."""
    header = read_header(destination)
    if header["magic"] != CACHE_MAGIC or header["version"] != CACHE_VERSION:
        raise SystemExit(f"{destination.name}: not a version-{CACHE_VERSION} cache")

    las = laspy.read(str(source))
    count = header["points"]
    if len(las.points) != count:
        raise SystemExit(f"{destination.name}: {count:,} points, source has {len(las.points):,}")

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(count, size=min(sample, count), replace=False))

    cache = np.memmap(destination, dtype=np.float32, mode="r",
                      offset=HEADER_BYTES, shape=(count, 3))
    origin = np.array(header["origin"])
    world_from_cache = np.asarray(cache[indices], dtype=np.float64) + origin
    world_from_las = np.stack([np.asarray(las.x)[indices],
                               np.asarray(las.y)[indices],
                               np.asarray(las.z)[indices]], axis=1)
    error = float(np.abs(world_from_cache - world_from_las).max())

    extent = float(np.max(np.asarray(las.header.maxs) - np.asarray(las.header.mins)))
    floor = float(np.spacing(np.float32(extent)))
    verdict = "OK" if error <= floor else "FAIL"
    log(f"{destination.name}: max |world error| {error * 1000:.4f} mm over "
        f"{len(indices):,} samples (float32 floor {floor * 1000:.4f} mm) {verdict}")
    if verdict == "FAIL":
        raise SystemExit(f"{destination.name}: round-trip error exceeds the float32 floor")

    return error


def sources_from_cell_config(path: Path) -> list[Path]:
    config = json.loads(path.read_text(encoding="utf-8"))
    return [Path(cell["laz_path"]) for cell in config["cells"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", type=Path, help="LAS/LAZ files to convert")
    parser.add_argument("--cell-config", type=Path,
                        help="convert every cell listed in a cell config instead")
    parser.add_argument("--out-dir", type=Path,
                        help="write caches here instead of beside each input")
    parser.add_argument("--verify", type=int, default=0, metavar="N",
                        help="after converting, compare N random points against the source")
    parser.add_argument("--force", action="store_true",
                        help="reconvert even when an up-to-date cache is already present")
    args = parser.parse_args()

    sources = list(args.inputs)
    if args.cell_config:
        sources += sources_from_cell_config(args.cell_config)
    if not sources:
        parser.error("give at least one input, or --cell-config")

    report = []
    for source in sources:
        if not source.exists():
            raise SystemExit(f"missing input: {source}")

        if args.out_dir:
            destination = args.out_dir / (source.name + ".mdspc")
        else:
            destination = source.with_name(source.name + ".mdspc")

        if destination.exists() and not args.force:
            existing = read_header(destination)
            size, write_time = source_stamp(source)
            if (existing["magic"] == CACHE_MAGIC and existing["version"] == CACHE_VERSION
                    and existing["source_size"] == size
                    and existing["source_write_time"] == write_time):
                log(f"{source.name}: cache is current, skipping")
                continue
            log(f"{source.name}: cache is stale, rebuilding")

        report.append(convert(source, destination))
        if args.verify:
            verify(source, destination, args.verify)

    if report:
        total = sum(entry["points"] for entry in report)
        log(f"converted {len(report)} cloud(s), {total:,} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

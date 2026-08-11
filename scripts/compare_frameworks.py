#!/usr/bin/env python3
"""Compare one MultiDataStructure schema against optional point-cloud frameworks.

The script uses the C++ executable as the source of truth for query generation:

1. Resolve a schema JSON, or a measured selector JSON that points to one.
2. Run schema-search in CPU/flat mode for exactly that schema and write a query trace.
3. Replay the same trace through optional baselines:
   - Open3D KDTreeFlann for radius/KNN, Open3D AABB crop for range.
   - PDAL filters.crop for AABB range queries.
   - A standalone PCL helper executable, if provided.

Missing frameworks are reported as skipped instead of making this script fail.
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class Query:
    query_id: int
    kind: str
    bounds_min: Optional[Tuple[float, float, float]]
    bounds_max: Optional[Tuple[float, float, float]]
    center: Optional[Tuple[float, float, float]]
    radius: float
    k: int
    expected_count: Optional[int]
    mds_latency_ms: Optional[float]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def maybe_path(path: str, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def resolve_schema(schema_or_selector: str, root: Path) -> Tuple[Path, Dict[str, object]]:
    """Return the concrete schema JSON path.

    Accepted inputs:
    - A normal schema JSON with a top-level "levels" array.
    - A measured selector JSON with selected_schema.path.
    - A selector JSON with schema_selection.selected_schema_path.
    """
    path = maybe_path(schema_or_selector, root)
    if not path.exists():
        raise FileNotFoundError(f"Schema/selector does not exist: {path}")

    with path.open() as handle:
        payload = json.load(handle)

    selected_path = None
    if isinstance(payload, dict):
        selected = payload.get("selected_schema")
        if isinstance(selected, dict):
            selected_path = selected.get("path")

        if not selected_path:
            schema_selection = payload.get("schema_selection")
            if isinstance(schema_selection, dict):
                selected_path = schema_selection.get("selected_schema_path")

        if selected_path:
            concrete = maybe_path(str(selected_path), root)
            if not concrete.exists():
                raise FileNotFoundError(f"Selector points to missing schema: {concrete}")
            return concrete, {
                "input_kind": "selector",
                "selector_path": str(path),
                "selected_schema_path": str(concrete),
                "selected_schema_name": (
                    selected.get("name") if isinstance(selected, dict) else payload.get("selected_schema_name")
                ),
            }

        if "levels" in payload:
            return path, {"input_kind": "schema", "selected_schema_path": str(path), "selected_schema_name": payload.get("name")}

    raise ValueError(f"Unrecognized schema/selector JSON format: {path}")


def run_mds_trace(args: argparse.Namespace, schema_path: Path, out_dir: Path, root: Path) -> Tuple[Dict[str, object], Path]:
    exe = maybe_path(args.exe, root)
    if not exe.exists():
        raise FileNotFoundError(f"Built executable was not found: {exe}")

    raw_csv = out_dir / "mds_raw.csv"
    best_csv = out_dir / "mds_best.csv"
    pareto_csv = out_dir / "mds_pareto.csv"
    trace_csv = out_dir / "query_trace.csv"

    command = [
        str(exe),
        "--mode", "schema-search",
        "--flat-search",
        "--evaluator", "cpu",
        "--input", args.input,
        "--schemas", str(schema_path),
        "--workloads", args.workload_profile,
        "--queries", str(args.queries),
        "--knn-k", str(args.knn_k),
        "--query-seed", str(args.query_seed),
        "--csv", str(raw_csv),
        "--best-csv", str(best_csv),
        "--pareto-csv", str(pareto_csv),
        "--query-trace", str(trace_csv),
        "--generate-schemas", "0",
        "--benchmark-top", "0",
        "--no-baselines",
        "--no-synthetic",
        "--no-score-cache",
        "--no-pause",
    ]
    if args.input_trace:
        # Replay a recorded application trace as the workload; the emitted
        # query_trace.csv then carries those exact queries to every framework.
        command.extend(["--input-trace", args.input_trace])
    if args.no_cache:
        command.append("--no-cache")

    started = time.perf_counter()
    subprocess.run(command, cwd=root, check=True)
    wall_ms = (time.perf_counter() - started) * 1000.0

    rows = []
    with raw_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"MultiDataStructure wrote no rows to {raw_csv}")
    row = rows[0]

    result = {
        "method": "multidatastructure_cpu",
        "status": "ok",
        "schema_name": row.get("schema_name", ""),
        "schema_path": row.get("schema_path", str(schema_path)),
        "build_ms": to_float(row.get("build_time_ms")),
        "avg_latency_ms": to_float(row.get("avg_latency_ms")),
        "p95_latency_ms": to_float(row.get("p95_latency_ms")),
        "range_queries": to_int(row.get("range_queries")),
        "radius_queries": to_int(row.get("radius_queries")),
        "knn_queries": to_int(row.get("knn_queries")),
        "wall_ms": wall_ms,
        "raw_csv": str(raw_csv),
        "query_trace": str(trace_csv),
    }
    return result, trace_csv


def read_queries(trace_csv: Path, query_limit: int = 0) -> List[Query]:
    queries: List[Query] = []
    with trace_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            kind = str(row.get("query_type", "")).strip().lower()
            if kind not in {"range", "count_range", "radius", "knn"}:
                continue
            if kind == "count_range":
                kind = "range"

            bounds_min = None
            bounds_max = None
            center = None
            if kind == "range":
                bounds_min = (
                    to_float(row.get("bounds_min_x")),
                    to_float(row.get("bounds_min_y")),
                    to_float(row.get("bounds_min_z")),
                )
                bounds_max = (
                    to_float(row.get("bounds_max_x")),
                    to_float(row.get("bounds_max_y")),
                    to_float(row.get("bounds_max_z")),
                )
            else:
                center = (
                    to_float(row.get("center_x")),
                    to_float(row.get("center_y")),
                    to_float(row.get("center_z")),
                )

            queries.append(Query(
                query_id=to_int(row.get("query_id"), len(queries)),
                kind=kind,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                center=center,
                radius=to_float(row.get("radius")),
                k=to_int(row.get("k")),
                expected_count=to_int(row.get("returned_points")) if row.get("returned_points") not in (None, "") else None,
                mds_latency_ms=to_float(row.get("latency_ms"), math.nan),
            ))
            if query_limit > 0 and len(queries) >= query_limit:
                break
    return queries


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_samples(method: str, samples: List[Dict[str, object]], build_ms: float = math.nan, note: str = "") -> Dict[str, object]:
    latencies = [float(sample["latency_ms"]) for sample in samples]
    mismatches = []
    by_type: Dict[str, Dict[str, object]] = {}
    for kind in sorted({str(sample["kind"]) for sample in samples}):
        kind_samples = [sample for sample in samples if sample["kind"] == kind]
        kind_latencies = [float(sample["latency_ms"]) for sample in kind_samples]
        by_type[kind] = {
            "queries": len(kind_samples),
            "avg_latency_ms": sum(kind_latencies) / len(kind_latencies) if kind_latencies else math.nan,
            "p95_latency_ms": percentile(kind_latencies, 0.95),
        }

    for sample in samples:
        expected = sample.get("expected_count")
        returned = sample.get("returned_count")
        if expected is not None and returned is not None and int(expected) != int(returned):
            mismatches.append(abs(int(expected) - int(returned)))

    return {
        "method": method,
        "status": "ok",
        "build_ms": build_ms,
        "queries": len(samples),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else math.nan,
        "p95_latency_ms": percentile(latencies, 0.95),
        "by_type": by_type,
        "count_mismatches": len(mismatches),
        "max_count_delta": max(mismatches) if mismatches else 0,
        "note": note,
    }


def skipped(method: str, reason: str) -> Dict[str, object]:
    return {"method": method, "status": "skipped", "reason": reason}


def load_points_numpy(input_path: Path):
    import numpy as np

    suffix = input_path.suffix.lower()
    if suffix in {".xyz", ".txt"}:
        rows = []
        with input_path.open() as handle:
            for line in handle:
                parts = line.replace(",", " ").split()
                if len(parts) < 3:
                    continue
                try:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue
        return np.asarray(rows, dtype=np.float64), (0.0, 0.0, 0.0)

    if suffix == ".csv":
        rows = []
        with input_path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            indices = None
            if header:
                lowered = [cell.strip().lower() for cell in header]
                if {"x", "y", "z"}.issubset(set(lowered)):
                    indices = (lowered.index("x"), lowered.index("y"), lowered.index("z"))
                else:
                    try:
                        rows.append((float(header[0]), float(header[1]), float(header[2])))
                    except (ValueError, IndexError):
                        pass
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    if indices:
                        rows.append((float(row[indices[0]]), float(row[indices[1]]), float(row[indices[2]])))
                    else:
                        rows.append((float(row[0]), float(row[1]), float(row[2])))
                except (ValueError, IndexError):
                    continue
        return np.asarray(rows, dtype=np.float64), (0.0, 0.0, 0.0)

    if suffix == ".ply":
        return load_ascii_ply_numpy(input_path)

    if suffix in {".las", ".laz"}:
        try:
            import laspy
        except ImportError as exc:
            raise RuntimeError("laspy is required for Python/Open3D LAS baselines") from exc
        las = laspy.read(str(input_path))
        # Trace CSVs (and the MDS query_trace export) are ALWAYS world coordinates
        # (frame convention since 076b226), so the baseline cloud must stay in world
        # frame too. The old origin-subtracted load made every Open3D radius count
        # wrong while kNN silently returned k plausible-looking neighbors.
        points = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
        return points, (0.0, 0.0, 0.0)

    if suffix == ".mdspc":
        # The curated benchmark ships .laz, which the C++ cannot decode, so both sides
        # consume the .mdspc cache instead. Reading it here rather than the source keeps
        # the baseline on exactly the same float32 positions the executable indexes -
        # otherwise a "framework is faster" result could come from a different cloud.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import laz_to_mdspc

        header = laz_to_mdspc.read_header(input_path)
        cache = np.memmap(input_path, dtype=np.float32, mode="r",
                          offset=laz_to_mdspc.HEADER_BYTES, shape=(header["points"], 3))
        return np.asarray(cache, dtype=np.float64) + np.asarray(header["origin"]), (0.0, 0.0, 0.0)

    raise RuntimeError(f"Python point loader does not support {input_path.suffix}")


def load_ascii_ply_numpy(input_path: Path):
    import numpy as np

    with input_path.open() as handle:
        vertex_count = None
        properties = []
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError("Invalid PLY: missing end_header")
            stripped = line.strip()
            if stripped == "end_header":
                break
            if stripped.startswith("format ") and "ascii" not in stripped:
                raise RuntimeError("Only ASCII PLY is supported by this lightweight loader")
            if stripped.startswith("element "):
                parts = stripped.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if in_vertex and stripped.startswith("property "):
                parts = stripped.split()
                if len(parts) >= 3:
                    properties.append(parts[-1])

        if vertex_count is None:
            raise RuntimeError("Invalid PLY: missing vertex element")
        try:
            x_i, y_i, z_i = properties.index("x"), properties.index("y"), properties.index("z")
        except ValueError as exc:
            raise RuntimeError("PLY must contain x/y/z vertex properties") from exc

        rows = []
        for _ in range(vertex_count):
            parts = handle.readline().split()
            if len(parts) <= max(x_i, y_i, z_i):
                continue
            rows.append((float(parts[x_i]), float(parts[y_i]), float(parts[z_i])))
    return np.asarray(rows, dtype=np.float64), (0.0, 0.0, 0.0)


def run_open3d(points, queries: Sequence[Query]) -> Dict[str, object]:
    try:
        import numpy as np
        import open3d as o3d
    except ImportError as exc:
        return skipped("open3d", f"missing Python package: {exc.name}")

    if points.size == 0:
        return skipped("open3d", "point cloud is empty")

    started = time.perf_counter()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    build_ms = (time.perf_counter() - started) * 1000.0

    samples = []
    for query in queries:
        started = time.perf_counter()
        if query.kind == "range" and query.bounds_min is not None and query.bounds_max is not None:
            bbox = o3d.geometry.AxisAlignedBoundingBox(query.bounds_min, query.bounds_max)
            cropped = pcd.crop(bbox)
            returned = len(cropped.points)
        elif query.kind == "radius" and query.center is not None:
            returned, _, _ = kdtree.search_radius_vector_3d(query.center, query.radius)
        elif query.kind == "knn" and query.center is not None:
            returned, _, _ = kdtree.search_knn_vector_3d(query.center, query.k)
        else:
            continue
        latency_ms = (time.perf_counter() - started) * 1000.0
        samples.append({
            "kind": query.kind,
            "latency_ms": latency_ms,
            "returned_count": returned,
            "expected_count": query.expected_count,
        })

    return summarize_samples(
        "open3d",
        samples,
        build_ms=build_ms,
        note="KDTreeFlann for radius/KNN; AxisAlignedBoundingBox crop for range.",
    )


def read_las_origin(input_path: Path) -> Tuple[float, float, float]:
    with input_path.open("rb") as handle:
        header = handle.read(227)
    if len(header) < 227 or header[:4] != b"LASF":
        return (0.0, 0.0, 0.0)
    import struct
    min_x = struct.unpack_from("<d", header, 187)[0]
    min_y = struct.unpack_from("<d", header, 203)[0]
    min_z = struct.unpack_from("<d", header, 219)[0]
    return (min_x, min_y, min_z)


def source_origin_for_pdal(input_path: Path, numpy_origin: Tuple[float, float, float]) -> Tuple[float, float, float]:
    # Queries are world coordinates (see load_points_numpy) and PDAL reads the source
    # file in world coordinates itself, so no origin shift is ever needed anymore.
    return (0.0, 0.0, 0.0)


def pdal_reader_type(input_path: Path) -> Optional[str]:
    suffix = input_path.suffix.lower()
    if suffix in {".las", ".laz"}:
        return "readers.las"
    if suffix == ".ply":
        return "readers.ply"
    return None


def run_pdal(input_path: Path, queries: Sequence[Query], origin: Tuple[float, float, float], query_limit: int) -> Dict[str, object]:
    try:
        import pdal
    except ImportError:
        return skipped("pdal_crop", "missing Python package: pdal")

    reader = pdal_reader_type(input_path)
    if not reader:
        return skipped("pdal_crop", f"unsupported input extension for this harness: {input_path.suffix}")

    range_queries = [q for q in queries if q.kind == "range" and q.bounds_min is not None and q.bounds_max is not None]
    if query_limit > 0:
        range_queries = range_queries[:query_limit]
    if not range_queries:
        return skipped("pdal_crop", "workload has no AABB range queries")

    samples = []
    for query in range_queries:
        mn = tuple(query.bounds_min[i] + origin[i] for i in range(3))
        mx = tuple(query.bounds_max[i] + origin[i] for i in range(3))
        bounds = f"([{mn[0]},{mx[0]}],[{mn[1]},{mx[1]}],[{mn[2]},{mx[2]}])"
        pipeline_json = [
            {"type": reader, "filename": str(input_path)},
            {"type": "filters.crop", "bounds": bounds},
        ]
        started = time.perf_counter()
        pipe = pdal.Pipeline(json.dumps(pipeline_json))
        returned = int(pipe.execute())
        latency_ms = (time.perf_counter() - started) * 1000.0
        samples.append({
            "kind": "range",
            "latency_ms": latency_ms,
            "returned_count": returned,
            "expected_count": query.expected_count,
        })

    note = "filters.crop AABB only; each query runs a PDAL pipeline."
    if query_limit > 0:
        note += f" Limited to first {len(range_queries)} range queries."
    return summarize_samples("pdal_crop", samples, build_ms=math.nan, note=note)


def run_pcl_helper(pcl_exe: Optional[str], input_path: Path, trace_csv: Path, out_dir: Path) -> List[Dict[str, object]]:
    if not pcl_exe:
        found = shutil.which("pcl_point_baseline")
        if not found:
            return [skipped("pcl", "provide --pcl-exe path/to/pcl_point_baseline")]
        pcl_exe = found

    exe = Path(pcl_exe)
    if not exe.exists() and shutil.which(pcl_exe) is None:
        return [skipped("pcl", f"PCL helper was not found: {pcl_exe}")]

    if input_path.suffix.lower() in {".las", ".laz"}:
        return [skipped("pcl", "the optional PCL helper reads PCD/PLY/XYZ/CSV, not LAS/LAZ")]

    output_json = out_dir / "pcl_baseline.json"
    command = [
        str(exe),
        "--input", str(input_path),
        "--query-trace", str(trace_csv),
        "--output", str(output_json),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        return [skipped("pcl", f"PCL helper failed with exit code {exc.returncode}")]

    with output_json.open() as handle:
        payload = json.load(handle)
    results = payload.get("results")
    if isinstance(results, list):
        return results
    return [skipped("pcl", f"PCL helper wrote an unexpected payload: {output_json}")]


def write_markdown(payload: Dict[str, object], path: Path) -> None:
    results = payload["results"]
    lines = [
        "# Framework Baseline Comparison",
        "",
        f"- Input: `{payload['input']}`",
        f"- Workload: `{payload['workload_profile']}`",
        f"- Schema: `{payload['schema']['selected_schema_path']}`",
        f"- Query trace: `{payload['query_trace']}`",
        "",
        "| Method | Status | Queries | Build ms | Avg query ms | P95 query ms | Mismatches | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        status = result.get("status", "unknown")
        if status != "ok":
            lines.append(
                f"| {result.get('method', '?')} | {status} | 0 |  |  |  |  | {result.get('reason', '')} |"
            )
            continue
        build = result.get("build_ms")
        lines.append(
            f"| {result.get('method', '?')} | ok | {result.get('queries', '')} | "
            f"{format_float(build)} | {format_float(result.get('avg_latency_ms'))} | "
            f"{format_float(result.get('p95_latency_ms'))} | "
            f"{result.get('count_mismatches', 0)} | {result.get('note', '')} |"
        )

    lines += ["", "## Per-Type Detail", ""]
    for result in results:
        if result.get("status") != "ok":
            continue
        lines.append(f"### {result.get('method')}")
        by_type = result.get("by_type", {})
        if not by_type:
            lines.append("_No per-type breakdown available._")
            lines.append("")
            continue
        lines += [
            "",
            "| Query type | Queries | Avg ms | P95 ms |",
            "|---|---:|---:|---:|",
        ]
        for kind, stats in sorted(by_type.items()):
            lines.append(
                f"| {kind} | {stats.get('queries', 0)} | "
                f"{format_float(stats.get('avg_latency_ms'))} | "
                f"{format_float(stats.get('p95_latency_ms'))} |"
            )
        lines.append("")

    path.write_text("\n".join(lines))


def format_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    return f"{number:.6f}"


def json_ready(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def normalize_mds_for_table(result: Dict[str, object], queries: Sequence[Query]) -> Dict[str, object]:
    by_type = {}
    for kind in ["range", "radius", "knn"]:
        latencies = [
            float(query.mds_latency_ms)
            for query in queries
            if query.kind == kind and query.mds_latency_ms is not None and math.isfinite(float(query.mds_latency_ms))
        ]
        if latencies:
            by_type[kind] = {
                "queries": len(latencies),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "p95_latency_ms": percentile(latencies, 0.95),
            }
    return {
        "method": result["method"],
        "status": "ok",
        "queries": len(queries),
        "build_ms": result.get("build_ms", math.nan),
        "avg_latency_ms": result.get("avg_latency_ms", math.nan),
        "p95_latency_ms": result.get("p95_latency_ms", math.nan),
        "by_type": by_type,
        "count_mismatches": 0,
        "max_count_delta": 0,
        "note": "Project CPU evaluator using the resolved schema.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="x64/Release/MultiDataStructure.exe", help="Path to the built project executable.")
    parser.add_argument("--input", required=True, help="Point cloud input path.")
    parser.add_argument("--schema", required=True, help="Schema JSON or measured selector JSON.")
    parser.add_argument("--workload-profile", default="configs/workloads/volume_small_medium.json")
    parser.add_argument("--queries", type=int, default=64, help="Workload queries passed to schema-search.")
    parser.add_argument("--input-trace", help="Recorded application query trace CSV replayed as the workload "
                        "(passed through to the executable's --input-trace).")
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--query-seed", type=int, default=1337)
    parser.add_argument("--frameworks", nargs="*", default=["open3d", "pdal", "pcl"],
                        choices=["open3d", "pdal", "pcl"], help="Framework baselines to attempt.")
    parser.add_argument("--pcl-exe", help="Path to the optional PCL helper built from tools/pcl_point_baseline.cpp.")
    parser.add_argument("--pdal-query-limit", type=int, default=32,
                        help="Limit PDAL crop queries because it runs one pipeline per query. Use 0 for no limit.")
    parser.add_argument("--framework-query-limit", type=int, default=0,
                        help="Replay only the first N trace queries through external frameworks. 0 means all.")
    parser.add_argument("--out-dir", help="Output directory. Defaults under results/framework_compare/.")
    parser.add_argument("--no-cache", action="store_true", help="Pass --no-cache to the project executable.")
    args = parser.parse_args()

    root = repo_root()
    input_path = maybe_path(args.input, root)
    if not input_path.exists():
        raise FileNotFoundError(f"Point cloud input does not exist: {input_path}")

    out_dir = Path(args.out_dir) if args.out_dir else root / "results" / "framework_compare" / f"{input_path.stem}_{now_stamp()}"
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_path, schema_info = resolve_schema(args.schema, root)
    mds_result, trace_csv = run_mds_trace(args, schema_path, out_dir, root)
    queries = read_queries(trace_csv, args.framework_query_limit)
    if not queries:
        raise RuntimeError(f"No supported queries found in trace: {trace_csv}")

    results = [normalize_mds_for_table(mds_result, queries)]

    points = None
    numpy_origin = (0.0, 0.0, 0.0)
    if "open3d" in args.frameworks:
        try:
            points, numpy_origin = load_points_numpy(input_path)
            results.append(run_open3d(points, queries))
        except Exception as exc:
            results.append(skipped("open3d", str(exc)))

    if "pdal" in args.frameworks:
        origin = source_origin_for_pdal(input_path, numpy_origin)
        results.append(run_pdal(input_path, queries, origin, args.pdal_query_limit))

    if "pcl" in args.frameworks:
        results.extend(run_pcl_helper(args.pcl_exe, input_path, trace_csv, out_dir))

    payload = {
        "input": str(input_path),
        "workload_profile": args.workload_profile,
        "schema": schema_info,
        "query_trace": str(trace_csv),
        "mds_raw": mds_result,
        "results": results,
    }
    json_path = out_dir / "framework_comparison.json"
    md_path = out_dir / "framework_comparison.md"
    json_path.write_text(json.dumps(json_ready(payload), indent=2, allow_nan=False))
    write_markdown(payload, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for result in results:
        if result.get("status") == "ok":
            print(f"{result['method']}: avg={format_float(result.get('avg_latency_ms'))} ms, queries={result.get('queries')}")
        else:
            print(f"{result.get('method')}: skipped ({result.get('reason')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

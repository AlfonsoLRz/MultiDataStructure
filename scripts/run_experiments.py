import argparse
import subprocess
from pathlib import Path


DEFAULT_SCHEMAS = [
    "configs/schemas/octree.json",
    "configs/schemas/quadtree.json",
    "configs/schemas/kdtree.json",
    "configs/schemas/quadtree_octree.json",
    "configs/schemas/octree_kdtree.json",
    "configs/schemas/urban_hybrid.json",
]

DEFAULT_WORKLOADS = [
    "configs/workloads/range_heavy.json",
    "configs/workloads/knn_heavy.json",
    "configs/workloads/mixed.json",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run point-cloud schema benchmark experiments.")
    parser.add_argument("--exe", default="x64/Release/MultiDataStructure.exe", help="Path to the built benchmark executable.")
    parser.add_argument("--schema-search", action="store_true", help="Run the Milestone 7 schema-search sweep.")
    parser.add_argument("--input", help="Point cloud input path (.las, .ply, .xyz, .csv). Optional for schema search when synthetic datasets are enabled.")
    parser.add_argument("--schemas", nargs="*", default=DEFAULT_SCHEMAS, help="Schema JSON files to evaluate.")
    parser.add_argument("--workloads", nargs="*", default=DEFAULT_WORKLOADS, help="Workload JSON files for schema-search mode.")
    parser.add_argument("--queries", type=int, default=128, help="Generated queries per query type.")
    parser.add_argument("--knn-k", type=int, default=8, help="K for generated KNN queries.")
    parser.add_argument("--query-seed", type=int, default=1337, help="Generated query profile seed.")
    parser.add_argument("--output", default="results/points_experiment.json", help="JSON output path. Multiple schemas get suffixed filenames.")
    parser.add_argument("--csv", default="results/points_summary.csv", help="CSV summary path.")
    parser.add_argument("--best-csv", default="results/schema_search_best.csv", help="Best-schema CSV path for schema-search mode.")
    parser.add_argument("--synthetic-scale", type=int, default=512, help="Synthetic dataset size scale for schema-search mode.")
    parser.add_argument("--no-synthetic", action="store_true", help="Use only --input datasets in schema-search mode.")
    parser.add_argument("--no-cache", action="store_true", help="Disable .mdspc cache use.")
    args = parser.parse_args()

    if not args.schema_search and not args.input:
        parser.error("--input is required unless --schema-search is used")

    repo_root = Path(__file__).resolve().parents[1]
    exe = Path(args.exe)
    if not exe.is_absolute():
        exe = repo_root / exe

    if args.schema_search:
        command = [
            str(exe),
            "--mode",
            "schema-search",
            "--schemas",
            ";".join(args.schemas),
            "--workloads",
            ";".join(args.workloads),
            "--queries",
            str(args.queries),
            "--knn-k",
            str(args.knn_k),
            "--query-seed",
            str(args.query_seed),
            "--csv",
            args.csv,
            "--best-csv",
            args.best_csv,
            "--synthetic-scale",
            str(args.synthetic_scale),
            "--no-pause",
        ]
        if args.input:
            command.extend(["--input", args.input])
        if args.no_synthetic:
            command.append("--no-synthetic")
    else:
        command = [
            str(exe),
            "--input",
            args.input,
            "--schemas",
            ";".join(args.schemas),
            "--queries",
            str(args.queries),
            "--knn-k",
            str(args.knn_k),
            "--query-seed",
            str(args.query_seed),
            "--output",
            args.output,
            "--csv",
            args.csv,
            "--no-pause",
        ]

    if args.no_cache:
        command.append("--no-cache")

    subprocess.run(command, cwd=repo_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

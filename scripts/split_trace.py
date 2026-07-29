"""Split recorded application traces into disjoint optimize/test halves (rigor R1).

Every schema search must consume the *_opt half; final re-measures, framework
comparisons and paper tables consume the *_test half only. The best-single baseline
per cell is likewise selected on _opt and reported on _test.

Pipeline traces (pipeline_trace.csv): rows are stratified by stage signature
(query_type, k, radius) so both halves keep the exact stage mix, then split by a
seeded permutation inside each stratum. Original row order is preserved in the
outputs so the replayer's deterministic stride subsampling (--queries N) stays
spatially uniform. For dense self-query traces this guards against instance-level
overfitting -- the optimizer never measures the reported queries -- but it is NOT a
distribution-shift test; use cross-cloud replay (e.g. SanAndreas -> SanSimeon) for that.

DL batch manifests (batch_manifest.json): the block is the atomic unit (per-block
rebuild semantics), so whole blocks are permuted with the same seed and dealt
alternately into the two output manifests.

Usage:
  python scripts/split_trace.py results/traces/sanandreas_5m/pipeline_trace.csv
  python scripts/split_trace.py results/traces/dl_sanandreas/batch_manifest.json
  python scripts/split_trace.py --all results/traces

Outputs <stem>_opt.<ext> and <stem>_test.<ext> next to each input, plus a
<stem>_split.json sidecar recording seed, ratio and per-stratum counts.
"""

import argparse
import csv
import json
import random
from pathlib import Path

DEFAULT_SEED = 7919  # same prime the C++ confirm-seed offset uses
DEFAULT_RATIO = 0.5  # fraction of each stratum that goes to the optimize half


def split_pipeline_trace(path: Path, seed: int, ratio: float) -> dict:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path}: empty trace")

    strata = {}
    for i, row in enumerate(rows):
        key = (row.get("query_type", ""), row.get("k", ""), row.get("radius", ""))
        strata.setdefault(key, []).append(i)

    rng = random.Random(seed)
    opt_idx, test_idx = [], []
    stratum_counts = {}
    for key in sorted(strata):
        idx = strata[key][:]
        rng.shuffle(idx)
        cut = round(len(idx) * ratio)
        opt_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
        stratum_counts["|".join(key)] = {"total": len(idx), "opt": cut, "test": len(idx) - cut}

    def write_half(indices: list, out_path: Path) -> None:
        indices.sort()  # keep original (spatial) order for stride subsampling
        with open(out_path, "w", newline="\n") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for new_id, i in enumerate(indices):
                row = dict(rows[i])
                if "query_id" in row:
                    row["query_id"] = str(new_id)
                writer.writerow(row)

    opt_path = path.with_name(path.stem + "_opt.csv")
    test_path = path.with_name(path.stem + "_test.csv")
    write_half(opt_idx, opt_path)
    write_half(test_idx, test_path)

    summary = {
        "input": path.name, "kind": "pipeline", "seed": seed, "ratio": ratio,
        "total": len(rows), "opt": len(opt_idx), "test": len(test_idx),
        "strata": stratum_counts,
        "outputs": [opt_path.name, test_path.name],
    }
    return summary


def split_batch_manifest(path: Path, seed: int, ratio: float) -> dict:
    manifest = json.loads(path.read_text())
    blocks = manifest["blocks"]
    order = list(range(len(blocks)))
    random.Random(seed).shuffle(order)
    cut = round(len(order) * ratio)
    halves = {"opt": sorted(order[:cut]), "test": sorted(order[cut:])}

    outputs = []
    for half, indices in halves.items():
        out = {k: v for k, v in manifest.items() if k != "blocks"}
        out["split"] = {"half": half, "seed": seed, "ratio": ratio, "source": path.name}
        out["blocks"] = [blocks[i] for i in indices]
        out_path = path.with_name(path.stem + f"_{half}.json")
        out_path.write_text(json.dumps(out, indent=2))
        outputs.append(out_path.name)

    summary = {
        "input": path.name, "kind": "batch", "seed": seed, "ratio": ratio,
        "total": len(blocks), "opt": len(halves["opt"]), "test": len(halves["test"]),
        "opt_blocks": [blocks[i]["name"] for i in halves["opt"]],
        "outputs": outputs,
    }
    return summary


def split_one(path: Path, seed: int, ratio: float) -> dict:
    if path.suffix == ".json":
        summary = split_batch_manifest(path, seed, ratio)
    else:
        summary = split_pipeline_trace(path, seed, ratio)
    sidecar = path.with_name(path.stem + "_split.json")
    sidecar.write_text(json.dumps(summary, indent=2))
    print(f"{path}: {summary['total']} {'blocks' if summary['kind'] == 'batch' else 'queries'}"
          f" -> opt {summary['opt']} / test {summary['test']}  ({', '.join(summary['outputs'])})")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+",
                        help="pipeline_trace.csv / batch_manifest.json paths, or a root dir with --all")
    parser.add_argument("--all", action="store_true",
                        help="treat inputs as directories; split every pipeline_trace.csv and batch_manifest.json under them")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO,
                        help="fraction of queries/blocks assigned to the optimize half")
    args = parser.parse_args()

    targets = []
    for raw in args.inputs:
        p = Path(raw)
        if args.all:
            targets.extend(sorted(p.rglob("pipeline_trace.csv")))
            targets.extend(sorted(p.rglob("batch_manifest.json")))
        else:
            targets.append(p)

    if not targets:
        raise SystemExit("no trace files found")
    for target in targets:
        split_one(target, args.seed, args.ratio)


if __name__ == "__main__":
    main()

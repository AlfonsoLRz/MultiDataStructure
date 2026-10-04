#!/usr/bin/env python3
"""Does the search's pick beat the best single structure? Held-out, from run_cells.py (e8+) output.

    python scripts/summarize_nesting.py results/eval_traces/matrix_e8 [more dirs...]

Per cell it reads <cell>_pick.json (the confirmation step's safe pick and its best nested and best
single finalists, all chosen on the opt half) and <cell>_test_timing.csv (the held-out contenders
timed interleaved, one row per schema per process). A schema's latency is the median over
processes of its all-queries mean. Reported ratios, below 1 = the first is faster:

  pick/single     what the method delivers against the single it would otherwise use
  nested/single   the best nested finalist against the best single finalist
  pick/oracle     the pick against the best single measured on the test half itself (an oracle,
                  biased towards the singles: it is the minimum of several noisy measurements)

Cells are grouped by size and by scan type; a ratio within 3% of 1 counts as a tie.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GROUPS = {
    "ground scans": ("tls_street", "mls_street", "heritage"),
    "aerial": ("aerial_terrain", "aerial_urban", "industrial"),
    "forest/vegetation": ("forest_archipelago", "forest_plot", "vegetated_urban"),
}
SIZES = ["1M", "5M", "25M", "50M", "100M", "250M", "500M", "1B"]


def blocks(path):
    p = Path(path)
    p = p if p.is_absolute() else REPO / p
    try:
        return len(json.loads(p.read_text())["levels"])
    except (OSError, ValueError, KeyError):
        return 0


def timing(path):
    runs = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            runs.setdefault(row["schema_path"].replace("\\", "/"), []).append(float(row["all_mean_ms"]))
    return {p: statistics.median(v) for p, v in runs.items()}


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values)) if values else float("nan")


def verdicts(values):
    return (f"wins {sum(v < 0.97 for v in values)}, ties {sum(0.97 <= v <= 1.03 for v in values)}, "
            f"losses {sum(v > 1.03 for v in values)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+")
    args = parser.parse_args()
    meta = {c["cell"]: c for c in json.loads((REPO / "configs/datasets/curated_cells.json").read_text())["cells"]}

    cells = []
    for directory in map(Path, args.dirs):
        for pick_json in sorted(directory.glob("*_pick.json")):
            name = pick_json.name[: -len("_pick.json")]
            timing_csv = directory / f"{name}_test_timing.csv"
            pick = json.loads(pick_json.read_text())
            if not timing_csv.exists() or not pick.get("pick"):
                continue
            lat = timing(timing_csv)
            p, n, s = pick.get("pick"), pick.get("best_nested"), pick.get("best_single")
            if p not in lat or (s and s not in lat):
                continue
            singles = [v for path, v in lat.items() if blocks(path) == 1]
            cells.append(dict(
                cell=name, size=meta.get(name, {}).get("size_label", ""), morph=meta.get(name, {}).get("morphology", ""),
                adopted=pick.get("adopted", ""), donor=pick.get("donor", ""),
                pick_single=lat[p] / lat[s] if s else float("nan"),
                nested_single=lat[n] / lat[s] if n in lat and s else float("nan"),
                pick_oracle=lat[p] / min(singles) if singles else float("nan"),
                pick_name=Path(p).stem))

    print(f"{'cell':24}{'scan':16}{'pick':8}{'pick/single':>12}{'nested/single':>14}{'pick/oracle':>12}  pick")
    for c in cells:
        print(f"{c['cell']:24}{c['morph'][:15]:16}{c['adopted'][:7]:8}{c['pick_single']:12.3f}{c['nested_single']:14.3f}"
              f"{c['pick_oracle']:12.3f}  {c['pick_name'][:48]}{'  (from ' + c['donor'] + ')' if c['donor'] else ''}")

    def block(title, rows):
        if not rows:
            return
        ps = [r["pick_single"] for r in rows if math.isfinite(r["pick_single"])]
        ns = [r["nested_single"] for r in rows if math.isfinite(r["nested_single"])]
        po = [r["pick_oracle"] for r in rows if math.isfinite(r["pick_oracle"])]
        print(f"{title:22} cells {len(rows):3}  pick/single {geomean(ps):.3f} ({verdicts(ps)})  "
              f"nested/single {geomean(ns):.3f}  pick/oracle {geomean(po):.3f}  nested picked {sum(r['adopted'] == 'nested' for r in rows)}")

    print()
    for size in SIZES:
        block(size, [c for c in cells if c["size"] == size])
    print()
    for group, morphs in GROUPS.items():
        block(group, [c for c in cells if c["morph"] in morphs])
    print()
    block("all", cells)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Node-cost dial (testing.md 2o): does nesting start to pay when entering a node gets expensive?

run_cells.py with --node-cost-ns d makes every query wait d ns per node it enters, in the search and
in the held-out timing alike, so each cost gets its own search, tuned singles and pick:

    python scripts/node_cost_dial.py results/eval_traces/node_cost_makinon2

reads <root>/d<ns>/ (one run_cells.py output per cost, d0 the uncharged control) and prints, per
cell and cost, below 1 = nested faster:

  nested/single  held-out, the best nested finalist against the best single finalist of that
                 cost's own search (summarize_nesting.py's definition)
  adopted        what the confirmation step picked at that cost
  visits         nested/single nodes entered per query (instrumented, <cell>_test.csv)
  d0 model       the d0 finalists with d ns added per visit: (t_n + d V_n) / (t_s + d V_s). Where
                 the measured ratio follows it, the cost is additive; where the re-searched ratio
                 beats it, the search found schemas that suit expensive nodes.
"""
import argparse
import csv
import json
import re
from pathlib import Path

from summarize_nesting import geomean, structure_rows, timing


def cell_result(directory, name):
    pick_json, timing_csv = directory / f"{name}_pick.json", directory / f"{name}_test_timing.csv"
    if not pick_json.exists() or not timing_csv.exists():
        return None
    pick, lat = json.loads(pick_json.read_text()), timing(timing_csv)
    n, s = pick.get("best_nested"), pick.get("best_single")
    rows = structure_rows(directory / f"{name}_test.csv")
    visits = lambda p: float(rows[p]["avg_visited_nodes"]) if p in rows else float("nan")
    out = dict(adopted=pick.get("adopted", ""), nested=n, single=s,
               t_n=lat.get(n, float("nan")), t_s=lat.get(s, float("nan")), v_n=visits(n), v_s=visits(s))
    out["ratio"] = out["t_n"] / out["t_s"] if n in lat and s in lat else float("nan")
    out["visits"] = out["v_n"] / out["v_s"] if out["v_s"] else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    costs = sorted((int(m.group(1)), p) for p in root.iterdir() if (m := re.fullmatch(r"d(\d+)", p.name)))
    cells = sorted({p.name[: -len("_pick.json")] for _, d in costs for p in d.glob("*_pick.json")})
    base = {c: cell_result(dict(costs).get(0, root / "d0"), c) for c in cells}

    print(f"{'cell':24}{'cost ns':>9}{'nested/single':>15}{'visits':>8}{'d0 model':>10}  adopted | best nested ; best single")
    per_cost = {}
    for cost, directory in costs:
        for c in cells:
            r = cell_result(directory, c)
            if not r:
                continue
            b = base.get(c)
            # Milliseconds per query: the timing CSVs are per-query means in ms, visits per query.
            model = ((b["t_n"] + cost * 1e-6 * b["v_n"]) / (b["t_s"] + cost * 1e-6 * b["v_s"])) if b else float("nan")
            per_cost.setdefault(cost, []).append((r["ratio"], model, r["adopted"] == "nested"))
            print(f"{c:24}{cost:>9}{r['ratio']:15.3f}{r['visits']:8.3f}{model:10.3f}  {r['adopted']} | "
                  f"{Path(r['nested'] or '-').stem} ; {Path(r['single'] or '-').stem}")
    print()
    for cost, values in per_cost.items():
        ratios = [v[0] for v in values if v[0] == v[0]]
        models = [v[1] for v in values if v[1] == v[1]]
        print(f"cost {cost:>6} ns: nested/single geomean {geomean(ratios):.3f} (min {min(ratios, default=float('nan')):.3f}), "
              f"d0 model {geomean(models):.3f}, nested adopted in {sum(v[2] for v in values)} of {len(values)} cells")


if __name__ == "__main__":
    main()

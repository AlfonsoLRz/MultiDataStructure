#!/usr/bin/env python3
"""Can fewer visited nodes buy latency? A per-cell linear cost model from the instrumented pass.

    python scripts/cost_model.py results/eval_traces/matrix_e9 [more dirs...] [--out cost_model.md]

Per cell, over every schema in <cell>_test.csv (finalists, library defaults, hand-made nested), least
squares (numpy lstsq) of

    latency_us = c_node * avg_visited_nodes + c_point * avg_tested_points + c_out * avg_returned_points [+ b]

reported both without and with the intercept b: coefficients in ns, R^2 and the number of schemas. The
model is the usual argument for nesting: a composite that visits fewer nodes should be faster. Then for
the finalists of <cell>_pick.json (best nested vs best single):

    dvisited*c_node / t_single   the latency the nested finalist's node saving (or cost) is worth under the
                                 no-intercept fit, as a fraction of the single's held-out latency (median of
                                 all_mean_ms over processes in <cell>_test_timing.csv, else instrumented)
    c_node break-even            the per-node cost at which a 20% visited-node saving would be a 10% latency
                                 saving: 0.2*visited_single*c = 0.1*t_single

avg_returned_points is the same for every schema of a cell (same queries, same answers), so c_out is in
effect a per-query constant and the intercept fit is collinear with it (lstsq returns the minimum-norm
solution; the two fits agree). regular_grid_default tests ~100x the points of any tree and pins c_point;
--drop regular_grid_default refits without such leverage points.

With --arms the same fit is also run over the opt-half arms (<cell>_singles.csv + <cell>_ga.csv, ~180
schemas where present) and reported alongside, as a better-conditioned check on the ~14-schema fit.
Medians are aggregated by size (cell-name suffix) and by morphology group (summarize_nesting.py).
"""
import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_nesting import GROUPS, REPO, SIZES, timing  # noqa: E402

COLS = ("avg_visited_nodes", "avg_tested_points", "avg_returned_points")


def norm(path):
    return path.replace("\\", "/")


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def cell_size(name, meta):
    m = re.search(r"_(\d+[MB])(?:_|$)", name)
    return m.group(1) if m else meta.get("size_label", "")


def fit(rows, intercept):
    """Least-squares cost model over rows; coefficients in ns per node/point, intercept in us."""
    X = np.array([[float(r[c]) for c in COLS] for r in rows])
    y = np.array([float(r["avg_latency_ms"]) * 1000.0 for r in rows])
    A = np.hstack([X, np.ones((len(rows), 1))]) if intercept else X
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(c_node=coef[0] * 1000, c_point=coef[1] * 1000, c_out=coef[2] * 1000,
                b=coef[3] if intercept else 0.0, r2=1 - ss_res / ss_tot if ss_tot else float("nan"), n=len(rows))


def analyse(directory, name, meta, arms, drop, notes):
    rows = [r for r in read_rows(directory / f"{name}_test.csv") if r["schema_name"] not in drop]
    if len(rows) < 4:
        notes.append(f"{name}: {len(rows)} rows in _test.csv, skipped")
        return None
    pick = json.loads((directory / f"{name}_pick.json").read_text())
    timing_csv = directory / f"{name}_test_timing.csv"
    timed = timing(timing_csv) if timing_csv.exists() else {}
    out = dict(cell=name, morph=meta.get("morphology", ""), size=cell_size(name, meta),
               plain=fit(rows, False), with_b=fit(rows, True), arms=None)
    if arms:
        arm_rows = [r for r in read_rows(directory / f"{name}_singles.csv") + read_rows(directory / f"{name}_ga.csv")
                    if r["schema_name"] not in drop]
        if len(arm_rows) >= 4:
            out["arms"] = fit(arm_rows, False)
        else:
            notes.append(f"{name}: no opt-half arms (transfer cell)")

    by_path = {norm(r["schema_path"]): r for r in rows}
    n, s = by_path.get(pick.get("best_nested") or ""), by_path.get(pick.get("best_single") or "")
    out.update(dvis_frac=float("nan"), c_break=float("nan"), vis_ratio=float("nan"), t_single=float("nan"))
    if n and s:
        t_single = timed.get(pick["best_single"], float(s["avg_latency_ms"])) * 1000.0  # us
        vis_s, vis_n = float(s["avg_visited_nodes"]), float(n["avg_visited_nodes"])
        out.update(t_single=t_single, vis_ratio=vis_n / vis_s if vis_s else float("nan"),
                   dvis_frac=(vis_n - vis_s) * out["plain"]["c_node"] / 1000.0 / t_single,
                   c_break=0.1 * t_single / (0.2 * vis_s) * 1000.0 if vis_s else float("nan"))
    else:
        notes.append(f"{name}: finalists not both in _test.csv, finalist columns are nan")
    return out


def fmt(v, digits=1):
    return f"{v:.{digits}f}" if v is not None and math.isfinite(v) else "nan"


def table(cells, arms, lines):
    head = ("| cell | morphology | size | n | c_node ns | c_point ns | c_out ns | R2 | +b: c_node | c_point | c_out | b us | R2 "
            "| vis n/s | dvis*c/t_s | c_node break-even ns | t_single us |")
    if arms:
        head += " arms n | c_node | c_point | c_out | R2 |"
    lines.append(head)
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" * 5 if arms else ""))
    for c in cells:
        p, w = c["plain"], c["with_b"]
        line = (f"| {c['cell']} | {c['morph']} | {c['size']} | {p['n']} | {fmt(p['c_node'])} | {fmt(p['c_point'], 2)} | {fmt(p['c_out'], 2)} "
                f"| {fmt(p['r2'], 3)} | {fmt(w['c_node'])} | {fmt(w['c_point'], 2)} | {fmt(w['c_out'], 2)} | {fmt(w['b'], 2)} | {fmt(w['r2'], 3)} "
                f"| {fmt(c['vis_ratio'], 3)} | {fmt(c['dvis_frac'], 3)} | {fmt(c['c_break'])} | {fmt(c['t_single'], 2)} |")
        if arms:
            a = c["arms"]
            line += (f" {a['n']} | {fmt(a['c_node'])} | {fmt(a['c_point'], 2)} | {fmt(a['c_out'], 2)} | {fmt(a['r2'], 3)} |"
                     if a else " - | - | - | - | - |")
        lines.append(line)


def aggregate(cells, arms, lines):
    def med(vals):
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        return statistics.median(vals) if vals else float("nan")

    def block(title, rows):
        if not rows:
            return
        p = [c["plain"] for c in rows]
        w = [c["with_b"] for c in rows]
        line = (f"| {title} | {len(rows)} | {fmt(med(x['c_node'] for x in p))} | {fmt(med(x['c_point'] for x in p), 2)} "
                f"| {fmt(med(x['c_out'] for x in p), 2)} | {fmt(med(x['r2'] for x in p), 3)} | {fmt(med(x['c_node'] for x in w))} "
                f"| {fmt(med(x['r2'] for x in w), 3)} | {fmt(med(c['vis_ratio'] for c in rows), 3)} "
                f"| {fmt(med(c['dvis_frac'] for c in rows), 3)} | {fmt(med(c['c_break'] for c in rows))} |")
        if arms:
            a = [c["arms"] for c in rows if c["arms"]]
            line += (f" {len(a)} | {fmt(med(x['c_node'] for x in a))} | {fmt(med(x['c_point'] for x in a), 2)} "
                     f"| {fmt(med(x['r2'] for x in a), 3)} |")
        lines.append(line)

    head = "| group | cells | c_node ns | c_point ns | c_out ns | R2 | +b: c_node | R2 | vis n/s | dvis*c/t_s | c_node break-even ns |"
    if arms:
        head += " arms cells | c_node | c_point | R2 |"
    lines.append(head)
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" * 4 if arms else ""))
    for size in SIZES:
        block(size, [c for c in cells if c["size"] == size])
    for group, morphs in GROUPS.items():
        block(group, [c for c in cells if c["morph"] in morphs])
    block("all", cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--arms", action="store_true", help="also fit over the opt-half singles+GA arms")
    parser.add_argument("--drop", nargs="*", default=[], metavar="SCHEMA", help="schema names left out of the fits")
    parser.add_argument("--out", help="also write the markdown tables here")
    args = parser.parse_args()
    meta = {c["cell"]: c for c in json.loads((REPO / "configs/datasets/curated_cells.json").read_text())["cells"]}

    cells, notes = [], []
    for directory in map(Path, args.dirs):
        if not directory.is_dir():
            notes.append(f"{directory}: not a directory, skipped")
            continue
        for pick_json in sorted(directory.glob("*_pick.json")):
            name = pick_json.name[: -len("_pick.json")]
            result = analyse(directory, name, meta.get(name, {}), args.arms, set(args.drop), notes)
            if result:
                cells.append(result)
    order = {s: i for i, s in enumerate(SIZES)}
    cells.sort(key=lambda c: (order.get(c["size"], 99), c["cell"]))

    lines = [f"# Cost model ({len(cells)} cells)", "",
             "latency_us = c_node*visited + c_point*tested + c_out*returned (+ b), least squares over the schemas of "
             "<cell>_test.csv; coefficients in ns. vis n/s = visited nodes of the best nested over the best single "
             "finalist; dvis*c/t_s = that difference priced at c_node (no-intercept fit) over the single's held-out "
             "latency; break-even = c_node at which a 20% visited saving is a 10% latency saving.", ""]
    if args.drop:
        lines += [f"Dropped from every fit: {', '.join(args.drop)}.", ""]
    if args.arms:
        lines += ["The `arms` columns refit (no intercept) over the opt-half tuned grid + GA schemas.", ""]
    lines += ["## Per cell", ""]
    table(cells, args.arms, lines)
    lines += ["", "## Medians by size and morphology", ""]
    aggregate(cells, args.arms, lines)
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fixed rules against the per-cell oracle (testing.md 2p; Paper A, Table 6 and Fig. 4).

    python scripts/rule_fit.py [--latency avg_latency_ms|latency_repeat_mean_ms] [--out report.txt]

Opt half, instrumented (<cell>_singles.csv: the 55 tuned singles, the same names in every searched
cell; <cell>_ga.csv: the GA's singles and the library defaults at leaf 32). Two oracles are named:

  oracle_opt   a cell's fastest single-level schema on the opt half (both files; the regret base here)
  oracle_test  the fastest single timed interleaved on the held-out half (the held-out check below)

A rule picks one of the 55 grid schemas without looking at the held-out cloud: every cell of that
cloud (all sizes) is left out of the fit (leave-one-cloud-out). Regret = the rule's schema latency /
oracle_opt. The last two sections price the nested finalists' node saving and the default -> oracle
lever with the per-cell cost model (counts explain under half of it, 2p).
"""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize_nesting import GROUPS, timing  # noqa: E402
from lever_decomposition import describe, norm, read_rows  # noqa: E402
from cost_model import fit  # noqa: E402

FEATURES = ["num_points", "density_bbox", "aspect_xy", "aspect_xz", "height_std", "linearity", "planarity",
            "scattering", "occupancy_ratio_8", "occupancy_entropy_8", "density_cv_8", "verticality_score",
            "flatness_score"]
MIN_LEAF = 6


def rel_fit(rows):
    """Cost model fitted on relative error (rows weighted by 1/latency), so the 1-5 us schemas count as
    much as the 600 us big-leaf arms; coefficients in ns."""
    X = np.array([[float(r[c]) for c in ("avg_visited_nodes", "avg_tested_points", "avg_returned_points")] for r in rows])
    y = np.array([float(r["avg_latency_ms"]) * 1000.0 for r in rows])
    coef, *_ = np.linalg.lstsq(X / y[:, None], np.ones(len(y)), rcond=None)
    pred = X @ coef
    return dict(c_node=coef[0] * 1000, c_point=coef[1] * 1000, c_out=coef[2] * 1000,
                r2=1 - float(np.sum(((y - pred) / y) ** 2)) / float(np.sum(((y - y.mean()) / y) ** 2)),
                mape=float(np.median(np.abs(pred / y - 1))))


FIT = fit


def group_of(morph):
    return next((g for g, ms in GROUPS.items() if morph in ms), "?")


def cloud_of(cell):
    return cell.rsplit("_", 1)[0].replace("_es05", "")


def size_class(cell):
    s = cell.rsplit("_", 1)[1]
    n = float(s[:-1]) * (1000 if s.endswith("B") else 1)
    return "<=5M" if n <= 5 else ">=25M"


def key(shape):
    return (shape["type"], shape["leaf"], shape["depth"], shape["policy"] or "", shape["tight"])


def load(repo, dirs, latency_col):
    """One record per searched cell: opt-half latencies of every single-level schema, oracle_opt, features."""
    meta = {c["cell"]: c for c in json.loads((repo / "configs/datasets/curated_cells.json").read_text())["cells"]}
    cells = []
    for directory in dirs:
        for pick_json in sorted(directory.glob("*_pick.json")):
            name = pick_json.name[: -len("_pick.json")]
            grid_rows = read_rows(directory / f"{name}_singles.csv")
            if not grid_rows:
                continue
            ga_rows = read_rows(directory / f"{name}_ga.csv")
            lat, shapes, rows_by = {}, {}, {}
            for row in grid_rows + ga_rows:
                shape = describe(row, directory)
                if shape["levels"] != 1:
                    continue
                v = float(row[latency_col] or row["avg_latency_ms"])
                if row["schema_name"] not in lat or v < lat[row["schema_name"]]:
                    lat[row["schema_name"]] = v
                    shapes[row["schema_name"]] = shape
                    rows_by[row["schema_name"]] = row
            oracle = min(lat, key=lat.get)
            morph = meta.get(name, {}).get("morphology", "")
            cells.append(dict(
                cell=name, dir=directory, cloud=cloud_of(name), group=group_of(morph), size=size_class(name),
                lat=lat, shapes=shapes, rows=rows_by, oracle=oracle, t_oracle=lat[oracle],
                feats={f: float(grid_rows[0][f]) for f in FEATURES}, grid_rows=grid_rows, ga_rows=ga_rows,
                pick=json.loads(pick_json.read_text())))
    return cells


def stats(values):
    v = sorted(values)
    return dict(n=len(v), geo=math.exp(sum(map(math.log, v)) / len(v)), med=statistics.median(v),
                p90=v[min(len(v) - 1, int(round(0.9 * (len(v) - 1))))], max=v[-1],
                over5=sum(x > 1.05 for x in v), over15=sum(x > 1.15 for x in v))


def line(title, values):
    s = stats(values)
    return (f"  {title:34} n={s['n']:2}  geo {s['geo']:.3f}  med {s['med']:.3f}  p90 {s['p90']:.3f}  "
            f"max {s['max']:.3f}  >1.05: {s['over5']:2}  >1.15: {s['over15']:2}")


def best_schema(L, idx, fallback=None):
    """Index of the grid schema with the lowest mean log regret over the cells idx (fallback if empty)."""
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        idx = np.asarray(fallback, dtype=int)
    return int(np.argmin(L[idx].sum(axis=0)))


def tree_fit(L, X, idx, depth):
    """Exhaustive regression tree of the given depth over feature thresholds; leaves assign one schema.
    Returns (cost, tree) where tree is ('leaf', s) or ('split', f, thr, left, right)."""
    s = best_schema(L, idx)
    best = (float(L[idx, s].sum()), ("leaf", s))
    if depth == 0 or len(idx) < 2 * MIN_LEAF:
        return best
    for f in range(X.shape[1]):
        order = idx[np.argsort(X[idx, f], kind="stable")]
        vals = X[order, f]
        cum = np.cumsum(L[order], axis=0)
        total = cum[-1]
        for i in range(MIN_LEAF, len(order) - MIN_LEAF + 1):
            if vals[i - 1] == vals[i]:
                continue
            if depth == 1:
                cost = float(cum[i - 1].min() + (total - cum[i - 1]).min())
                if cost < best[0] - 1e-12:
                    thr = 0.5 * (vals[i - 1] + vals[i])
                    best = (cost, ("split", f, thr, ("leaf", int(np.argmin(cum[i - 1]))),
                                   ("leaf", int(np.argmin(total - cum[i - 1])))))
            else:
                left = tree_fit(L, X, order[:i], depth - 1)
                right = tree_fit(L, X, order[i:], depth - 1)
                if left[0] + right[0] < best[0] - 1e-12:
                    best = (left[0] + right[0], ("split", f, 0.5 * (vals[i - 1] + vals[i]), left[1], right[1]))
    return best


def tree_apply(tree, x):
    while tree[0] == "split":
        _, f, thr, left, right = tree
        tree = left if x[f] <= thr else right
    return tree[1]


def tree_text(tree, names, indent="    "):
    if tree[0] == "leaf":
        return f"{indent}-> {names[tree[1]]}\n"
    _, f, thr, left, right = tree
    return (f"{indent}{FEATURES[f]} <= {thr:.4g}\n" + tree_text(left, names, indent + "  ") +
            f"{indent}{FEATURES[f]} >  {thr:.4g}\n" + tree_text(right, names, indent + "  "))


def matrices(cells):
    """Grid schema names, the log-regret matrix L (cells x schemas) and the feature matrix X."""
    names = sorted(n for n in cells[0]["lat"] if n.startswith("tuned_"))
    assert all(all(n in c["lat"] for n in names) for c in cells), "grid schema missing in a cell"
    L = np.array([[math.log(c["lat"][n] / c["t_oracle"]) for n in names] for c in cells])
    X = np.array([[c["feats"][f] for f in FEATURES] for c in cells])
    X[:, 0] = np.log10(X[:, 0])
    X[:, 1] = np.log10(np.maximum(X[:, 1], 1e-12))
    return names, L, X


def regrets(cells, names, L, X, depth2=False):
    """rule -> cell index -> regret, and rule -> picked schema -> number of leave-one-cloud-out folds."""
    clouds = sorted({c["cloud"] for c in cells})
    reg = defaultdict(dict)
    for i, c in enumerate(cells):
        for d in ("octree_default", "kdtree_default", "quadtree_default"):
            if d in c["lat"]:
                reg[f"R0 {d}"][i] = c["lat"][d] / c["t_oracle"]
        reg["floor: best grid schema per cell"][i] = math.exp(L[i].min())
        reg["in-sample: tuned_kd48l128sm everywhere"][i] = math.exp(L[i, names.index("tuned_kd48l128sm")])

    picks = defaultdict(lambda: defaultdict(int))
    for cloud in clouds:
        test = np.array([i for i, c in enumerate(cells) if c["cloud"] == cloud])
        train = np.array([i for i, c in enumerate(cells) if c["cloud"] != cloud])
        s1 = best_schema(L, train)
        for i in test:
            reg["R1 one global schema"][i] = math.exp(L[i, s1])
        picks["R1"][names[s1]] += 1
        for i in test:
            g = cells[i]["group"]
            tr = np.array([j for j in train if cells[j]["group"] == g])
            s2 = best_schema(L, tr, train)
            reg["R2 per acquisition group"][i] = math.exp(L[i, s2])
            picks["R2 " + g][names[s2]] += 1
            tr = np.array([j for j in train if cells[j]["group"] == g and cells[j]["size"] == cells[i]["size"]])
            if len(tr) >= 3:
                s2s = best_schema(L, tr)
                reg["R2s per group x size (<=5M, >=25M)"][i] = math.exp(L[i, s2s])
                picks[f"R2s {g} {cells[i]['size']}"][names[s2s]] += 1
        t1 = tree_fit(L, X, train, 1)[1]
        for i in test:
            reg["R3a feature tree, depth 1"][i] = math.exp(L[i, tree_apply(t1, X[i])])
        if depth2:
            t2 = tree_fit(L, X, train, 2)[1]
            for i in test:
                reg["R3b feature tree, depth 2"][i] = math.exp(L[i, tree_apply(t2, X[i])])
    return reg, picks


def oracle_cv(cells):
    """Median and p90 of the repeat coefficient of variation of the oracle_opt rows: the noise floor of the regret."""
    cvs = sorted(float(c["rows"][c["oracle"]].get("latency_cv") or "nan") for c in cells)
    cvs = [v for v in cvs if math.isfinite(v)]
    return (statistics.median(cvs), cvs[int(0.9 * (len(cvs) - 1))]) if cvs else (float("nan"), float("nan"))


def per_cell_table(cells, reg, names, L, X):
    """Rows for Fig. 4: cell, group, flatness, oracle_opt schema, the flatness rule's pick and regret."""
    t1 = tree_fit(L, X, np.arange(len(cells)), 1)[1]
    rows = []
    for i, c in enumerate(cells):
        rows.append(dict(cell=c["cell"], group=c["group"], flatness=c["feats"]["flatness_score"],
                         oracle_opt=c["oracle"], rule_pick=names[tree_apply(t1, X[i])],
                         regret_rule=reg["R3a feature tree, depth 1"].get(i, float("nan")),
                         regret_octree_default=reg["R0 octree_default"].get(i, float("nan")),
                         regret_kd_default=reg["R0 kdtree_default"].get(i, float("nan"))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=str(HERE.parent))
    parser.add_argument("--dirs", nargs="+", default=["results/eval_traces/matrix_e9", "results/eval_traces/matrix_e9_makinon2"],
                        help="e9 result folders, relative to --repo")
    parser.add_argument("--latency", default="avg_latency_ms")
    parser.add_argument("--relfit", action="store_true", help="cost model on relative error")
    parser.add_argument("--depth2", action="store_true", help="also the depth-2 feature tree (slow)")
    parser.add_argument("--out", default="", help="write the report here (default: print only)")
    parser.add_argument("--cells-csv", default="", help="write the per-cell table (Fig. 4) here")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    global FIT
    FIT = rel_fit if args.relfit else (lambda rows: fit(rows, False))
    cells = load(repo, [repo / d for d in args.dirs], args.latency)
    names, L, X = matrices(cells)
    clouds = sorted({c["cloud"] for c in cells})
    groups = defaultdict(list)
    for i, c in enumerate(cells):
        groups[c["group"]].append(i)

    out = [f"# Fixed rules against oracle_opt ({len(cells)} searched cells, {len(clouds)} clouds, "
           f"latency column {args.latency})", ""]
    out.append("Regret = rule's schema latency / oracle_opt (the cell's fastest single on the opt half, instrumented). "
               "Rules are fitted leave-one-cloud-out.")
    out.append("")

    reg, picks = regrets(cells, names, L, X, args.depth2)

    out.append("## Regret, all cells and by group")
    for rule, by_cell in reg.items():
        out.append(line(rule, list(by_cell.values())))
        for g, idx in groups.items():
            vals = [by_cell[i] for i in idx if i in by_cell]
            if vals:
                out.append(line(f"    {g}", vals))
    out.append("")
    out.append("## What the leave-one-cloud-out rules pick (count of folds)")
    for rule, counter in sorted(picks.items()):
        out.append(f"  {rule:32} " + ", ".join(f"{n} x{k}" for n, k in sorted(counter.items(), key=lambda t: -t[1])))
    out.append("")
    out.append("## Full-data trees (for the rule's wording)")
    out.append("  depth 1:")
    out.append(tree_text(tree_fit(L, X, np.arange(len(cells)), 1)[1], names))
    if args.depth2:
        out.append("  depth 2:")
        out.append(tree_text(tree_fit(L, X, np.arange(len(cells)), 2)[1], names))

    out.append("## oracle_opt per cell (cell: schema, its regret under R2)")
    for i, c in enumerate(cells):
        out.append(f"  {c['cell']:28} {c['group'][:8]:9} {c['oracle']:44} R2 {reg['R2 per acquisition group'][i]:.3f}")
    out.append("")

    # Noise floor: repeat CV of the oracle rows
    cvs = [float(c["rows"][c["oracle"]].get("latency_cv") or "nan") for c in cells]
    cvs = [v for v in cvs if math.isfinite(v)]
    if cvs:
        out.append(f"oracle_opt rows' repeat CV: median {statistics.median(cvs):.3f}, p90 {sorted(cvs)[int(0.9 * (len(cvs) - 1))]:.3f}")
        out.append("")

    # Held-out confirmation against oracle_test: structural twins of the R2 pick among the timed test schemas
    out.append("## Held-out check against oracle_test: R2 pick's structural twin in <cell>_test_timing.csv (interleaved)")
    held = []
    for i, c in enumerate(cells):
        g = c["group"]
        train = np.array([j for j, d in enumerate(cells) if d["cloud"] != c["cloud"] and d["group"] == g])
        s2 = names[best_schema(L, train, [j for j, d in enumerate(cells) if d['cloud'] != c['cloud']])]
        k2 = key(c["shapes"][s2])
        timing_csv = c["dir"] / f"{c['cell']}_test_timing.csv"
        if not timing_csv.exists():
            continue
        timed = timing(timing_csv)
        test_rows = read_rows(c["dir"] / f"{c['cell']}_test.csv")
        singles = {}
        twin = None
        for row in test_rows:
            p = norm(row["schema_path"])
            if p not in timed:
                continue
            shape = describe(row, c["dir"])
            if shape["levels"] != 1:
                continue
            singles[p] = timed[p]
            if key(shape) == k2:
                twin = p
        if twin and singles:
            held.append((c["cell"], s2, Path(twin).stem, timed[twin] / min(singles.values())))
    for cell, s2, twin, r in held:
        out.append(f"  {cell:28} {s2:22} twin {twin:34} / oracle_test {r:.3f}")
    if held:
        out.append(line("held-out R2 twin / oracle_test", [h[3] for h in held]))
    out.append("")

    # Where the nested finalists' node saving goes (cost model fitted on the opt-half arms)
    out.append("## Nested vs single finalists: node and point terms (cost model on opt-half arms)")
    out.append("  per cell: visits n/s, points n/s, predicted n/s = 1 + (c_node dV + c_point dP)/t_s, "
               "measured n/s instrumented, held-out")
    rows2 = []
    for c in cells:
        test_rows = {norm(r["schema_path"]): r for r in read_rows(c["dir"] / f"{c['cell']}_test.csv")}
        n, s = test_rows.get(c["pick"].get("best_nested") or ""), test_rows.get(c["pick"].get("best_single") or "")
        if not n or not s:
            continue
        m = FIT(c["grid_rows"] + c["ga_rows"])
        t_s = float(s["avg_latency_ms"]) * 1000.0
        dv = float(n["avg_visited_nodes"]) - float(s["avg_visited_nodes"])
        dp = float(n["avg_tested_points"]) - float(s["avg_tested_points"])
        node_term, point_term = m["c_node"] * dv / 1000.0 / t_s, m["c_point"] * dp / 1000.0 / t_s
        timing_csv = c["dir"] / f"{c['cell']}_test_timing.csv"
        timed = timing(timing_csv) if timing_csv.exists() else {}
        held_ratio = (timed[c["pick"]["best_nested"]] / timed[c["pick"]["best_single"]]
                      if c["pick"].get("best_nested") in timed and c["pick"].get("best_single") in timed else float("nan"))
        rows2.append(dict(cell=c["cell"], group=c["group"],
                          vis=float(n["avg_visited_nodes"]) / float(s["avg_visited_nodes"]),
                          pts=float(n["avg_tested_points"]) / max(float(s["avg_tested_points"]), 1e-9),
                          node=node_term, point=point_term, pred=1 + node_term + point_term,
                          inst=float(n["avg_latency_ms"]) / float(s["avg_latency_ms"]), held=held_ratio,
                          r2=m["r2"], c_node=m["c_node"], c_point=m["c_point"]))
    for title, sel in [("all", rows2)] + [(g, [r for r in rows2 if r["group"] == g]) for g in GROUPS]:
        if not sel:
            continue
        geo = lambda k: math.exp(sum(math.log(r[k]) for r in sel if math.isfinite(r[k])) /
                                 max(1, sum(math.isfinite(r[k]) for r in sel)))
        med = lambda k: statistics.median(r[k] for r in sel)
        out.append(f"  {title:18} n={len(sel):2}  visits {geo('vis'):.3f}  points {geo('pts'):.3f}  "
                   f"node term {med('node'):+.3f}  point term {med('point'):+.3f}  predicted {geo('pred'):.3f}  "
                   f"measured inst {geo('inst'):.3f}  held-out {geo('held'):.3f}  (median R2 {med('r2'):.3f}, "
                   f"c_node {med('c_node'):.1f} ns, c_point {med('c_point'):.2f} ns)")
    out.append("")

    # Same decomposition for the leaf/primitive lever: library octree default -> oracle_opt
    out.append("## Library octree default -> oracle_opt: node and point terms (same fits)")
    rows3 = []
    for c in cells:
        d = c["rows"].get("octree_default")
        o = c["rows"][c["oracle"]]
        if not d:
            continue
        m = FIT(c["grid_rows"] + c["ga_rows"])
        t_d = float(d["avg_latency_ms"]) * 1000.0
        dv = float(o["avg_visited_nodes"]) - float(d["avg_visited_nodes"])
        dp = float(o["avg_tested_points"]) - float(d["avg_tested_points"])
        rows3.append(dict(group=c["group"], vis=float(o["avg_visited_nodes"]) / float(d["avg_visited_nodes"]),
                          pts=float(o["avg_tested_points"]) / float(d["avg_tested_points"]),
                          node=m["c_node"] * dv / 1000.0 / t_d, point=m["c_point"] * dp / 1000.0 / t_d,
                          meas=float(o["avg_latency_ms"]) / float(d["avg_latency_ms"])))
    for title, sel in [("all", rows3)] + [(g, [r for r in rows3 if r["group"] == g]) for g in GROUPS]:
        if not sel:
            continue
        geo = lambda k: math.exp(sum(math.log(r[k]) for r in sel) / len(sel))
        med = lambda k: statistics.median(r[k] for r in sel)
        out.append(f"  {title:18} n={len(sel):2}  visits {geo('vis'):.3f}  points {geo('pts'):.3f}  "
                   f"node term {med('node'):+.3f}  point term {med('point'):+.3f}  measured oracle_opt/default {geo('meas'):.3f}")
    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if args.cells_csv:
        rows = per_cell_table(cells, reg, names, L, X)
        Path(args.cells_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.cells_csv, "w", encoding="utf-8") as f:
            f.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")


if __name__ == "__main__":
    main()

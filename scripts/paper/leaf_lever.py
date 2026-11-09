"""The sequential batteries of scripts/leaf_pages.py for the paper (testing.md 2u): the headline panel at 100M
under both page modes (tab_panel), and the leaf-size lever of the rule's two primitives in both page modes
(tab_leaf_lever). Reads results/eval_traces/leaf_pages_makinon2/{panel,leaf}/<cell>/{base,huge}.csv, one row per
schema and round (merge_runs adds test_run); a latency is the median over rounds of all_mean_ms.
"""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import RESULTS, Table, cells_config, fmt, fmt_range, geomean, group_of, read_csv  # noqa: E402

ROOT = RESULTS / "leaf_pages_makinon2"
ARMS = ("base", "huge")
FLAT_THRESHOLD = 0.60
LEAVES = [16, 32, 64, 128, 256, 512, 1024]


def latencies(path):
    """schema -> (median over rounds of all_mean_ms in us, rounds, median anon_huge_mb, num_nodes)."""
    by = defaultdict(list)
    for row in read_csv(path):
        by[row["schema_name"]].append(row)
    return {name: dict(us=statistics.median(float(r["all_mean_ms"]) for r in rows) * 1e3, rounds=len(rows),
                       huge_mb=statistics.median(float(r.get("anon_huge_mb") or 0) for r in rows),
                       nodes=int(rows[0]["num_nodes"]), visits=float(rows[0]["mean_visited_nodes"]))
            for name, rows in by.items()}


def flatness(cell):
    for d in ("matrix_e9_makinon2", "matrix_e9"):
        path = RESULTS / d / f"{cell}_singles.csv"
        if path.exists():
            rows = read_csv(path)
            if rows and rows[0].get("flatness_score"):
                return float(rows[0]["flatness_score"])
    return None


def is_single(name):
    return "_" not in name.replace("e9pick_", "").replace("ap=", "") or name.endswith("_default")


# ---------------------------------------------------------------- the headline panel (1C)
def panel_cells():
    """Per cell: per arm the latencies, plus the roles (pick, nested, rule, defaults, stage B winner)."""
    meta = cells_config()
    out = []
    for cell_dir in sorted(p for p in (ROOT / "panel").iterdir() if p.is_dir()) if (ROOT / "panel").is_dir() else []:
        arms = {arm: latencies(cell_dir / f"{arm}.csv") for arm in ARMS if (cell_dir / f"{arm}.csv").exists()}
        if "base" not in arms:
            continue
        names = list(arms["base"])
        pick = next(n for n in names if n.startswith("e9pick_"))
        nested = [n for n in names if not is_single(n) and not n.startswith("qu")]
        stage_b = [n for n in names if n.startswith("qu")]
        flat = flatness(cell_dir.name)
        rule = "qt16l128tc" if flat is not None and flat > FLAT_THRESHOLD else "kd48l128sm"
        out.append(dict(cell=cell_dir.name, group=group_of(meta.get(cell_dir.name, {}).get("morphology", "")), arms=arms, pick=pick,
                        nested=nested[0] if nested else None, stage_b=stage_b, rule=rule, flatness=flat))
    return out


def panel_stats():
    """Geomeans over cells per arm: defaults / pick, rule / pick, nested / best single, stage B / pick, huge / base per role."""
    cells = panel_cells()
    ratios = defaultdict(list)
    for c in cells:
        for arm, lat in c["arms"].items():
            pick = lat[c["pick"]]["us"]
            singles = {n: v["us"] for n, v in lat.items() if is_single(n)}
            best_single = min(singles.values())
            ratios[(arm, "octree default / pick")].append(lat["octree_default"]["us"] / pick)
            ratios[(arm, "kd default / pick")].append(lat["kdtree_default"]["us"] / pick)
            ratios[(arm, "rule / pick")].append(lat[c["rule"]]["us"] / pick)
            ratios[(arm, "pick / best single")].append(pick / best_single)
            if c["nested"]:
                ratios[(arm, "nested / best single")].append(lat[c["nested"]]["us"] / best_single)
                ratios[(arm, "nested / pick")].append(lat[c["nested"]]["us"] / pick)
            for s in c["stage_b"]:
                ratios[(arm, "stage B winner / pick")].append(lat[s]["us"] / pick)
        if "huge" in c["arms"]:
            for name, v in c["arms"]["base"].items():
                if name in c["arms"]["huge"]:
                    role = "nested" if not is_single(name) else "single"
                    ratios[("huge / base", role)].append(c["arms"]["huge"][name]["us"] / v["us"])
                    ratios[("huge / base", "all")].append(c["arms"]["huge"][name]["us"] / v["us"])
    return cells, dict(ratios)


def tab_panel():
    cells, ratios = panel_stats()
    table = Table("tab_panel", ["Ratio", "base: 4 KiB pages", "huge pages"], align="lrr")
    for key in ("octree default / pick", "kd default / pick", "rule / pick", "pick / best single", "nested / best single", "stage B winner / pick"):
        row = [key]
        for arm in ARMS:
            values = ratios.get((arm, key), [])
            row.append(f"{fmt(geomean(values))} ({fmt_range(values)}, $n$={len(values)})" if values else "--")
        table.add(*row)
    table.midrule()
    for role in ("single", "nested"):
        values = ratios.get(("huge / base", role), [])
        if values:
            table.add(f"huge / base, {role} schemas", "", f"{fmt(geomean(values))} ({fmt_range(values)}, $n$={len(values)})")
    path = table.write()
    # per-cell detail for the supplement and the ledger
    detail = Table("tab_panel_cells", ["Cell", "pick", "nested", "octree def.", "kd def.", "rule", "nested\\,/\\,single", "huge\\,/\\,base (pick)"], align="llrrrrrr")
    for c in cells:
        b = c["arms"]["base"]
        pick = b[c["pick"]]["us"]
        best_single = min(v["us"] for n, v in b.items() if is_single(n))
        huge = c["arms"].get("huge", {})
        detail.add(c["cell"].replace("_", "\\_"), c["pick"].replace("e9pick_", "").replace("_", "\\_"),
                   f"{b[c['nested']]['us']:.2f}" if c["nested"] else "--", f"{b['octree_default']['us']:.2f}", f"{b['kdtree_default']['us']:.2f}",
                   f"{b[c['rule']]['us']:.2f}", fmt(b[c["nested"]]["us"] / best_single) if c["nested"] else "--",
                   fmt(huge[c["pick"]]["us"] / pick) if c["pick"] in huge else "--")
    detail.write()
    return path


# ---------------------------------------------------------------- the leaf lever (1B)
def leaf_cells():
    """Per cell and arm: latency per leaf for kd48sm and qt24tc, and the defaults."""
    out = []
    folder = ROOT / "leaf"
    if not folder.is_dir():
        return out
    for cell_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        arms = {arm: latencies(cell_dir / f"{arm}.csv") for arm in ARMS if (cell_dir / f"{arm}.csv").exists()}
        if "base" not in arms:
            continue
        curves = {}
        for arm, lat in arms.items():
            for prim, pattern in (("kd48sm", r"kd48l(\d+)sm"), ("qt24tc", r"qt24l(\d+)tc")):
                curve = {int(m.group(1)): v["us"] for n, v in lat.items() if (m := re.fullmatch(pattern, n))}
                if curve:
                    curves[(arm, prim)] = curve
        out.append(dict(cell=cell_dir.name, arms=arms, curves=curves, flatness=flatness(cell_dir.name)))
    return out


def leaf_stats():
    """Per primitive and arm: the best leaf per cell, and the lever as best / leaf 32 and best / worst."""
    cells = leaf_cells()
    stats = defaultdict(lambda: dict(best=[], over32=[], worst=[], over128=[]))
    shift = []
    for c in cells:
        for (arm, prim), curve in c["curves"].items():
            best_leaf = min(curve, key=curve.get)
            s = stats[(arm, prim)]
            s["best"].append(best_leaf)
            s["over32"].append(curve[32] / curve[best_leaf] if 32 in curve else float("nan"))
            s["over128"].append(curve[128] / curve[best_leaf] if 128 in curve else float("nan"))
            s["worst"].append(max(curve.values()) / curve[best_leaf])
        for prim in ("kd48sm", "qt24tc"):
            b, h = c["curves"].get(("base", prim)), c["curves"].get(("huge", prim))
            if b and h:
                shift.append((c["cell"], prim, min(b, key=b.get), min(h, key=h.get)))
    return cells, dict(stats), shift


def tab_leaf_lever():
    cells, stats, shift = leaf_stats()
    if not cells:
        raise FileNotFoundError(f"{ROOT / 'leaf'} has no finished cells yet")
    table = Table("tab_leaf_lever", ["Primitive", "Pages", "best leaf (cells)", "leaf 32\\,/\\,best", "leaf 128\\,/\\,best", "worst\\,/\\,best"], align="llllrr")
    for prim, label in (("kd48sm", "kd-tree, sliding midpoint"), ("qt24tc", "quadtree, points' centre")):
        for arm, pages in (("base", "4 KiB"), ("huge", "2 MB")):
            s = stats.get((arm, prim))
            if not s:
                continue
            counts = defaultdict(int)
            for leaf in s["best"]:
                counts[leaf] += 1
            best = ", ".join(f"{leaf} ({n})" for leaf, n in sorted(counts.items()))
            table.add(label if arm == "base" else "", pages, best, f"{fmt(geomean(s['over32']))} ({fmt_range(s['over32'])})",
                      f"{fmt(geomean(s['over128']))} ({fmt_range(s['over128'])})", f"{fmt(geomean(s['worst']))} ({fmt_range(s['worst'])})")
    path = table.write()
    moved = sum(1 for _, _, b, h in shift if h < b)
    print(f"  leaf lever: {len(cells)} cells; the best leaf moved to a smaller leaf on huge pages in {moved} of {len(shift)} (cell, primitive) pairs")
    return path


if __name__ == "__main__":
    print(tab_panel())
    try:
        print(tab_leaf_lever())
    except FileNotFoundError as error:
        print(error)

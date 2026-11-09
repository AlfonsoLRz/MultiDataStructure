#!/usr/bin/env python3
"""The paper's tables, written to paper/generated/tab_*.tex from the local result copies.

    python scripts/paper/tables.py [--only scenes,budgets,levers,composition,ooc,rule,libraries,layout]

  tab_scenes              T1  the scenes: group, acquisition, source, native points, flatness, cells used
  tab_budgets             T2  per size: cells, queries, search budget
  tab_levers              T3  the lever chain (held-out): library default -> primitive -> tuned single -> nested
  tab_composition         T4  best nested / best single by size and group (held-out, interleaved)
  tab_composition_regimes T4b the same on other workloads, on the GPU and under emulated node costs
  tab_ooc                 T5  out of core: stage B adoption per regime and device class (2q), from ooc.py
  tab_rule                T6  fixed rules against the per-cell oracle (2p)
  tab_libraries           T7  external libraries relative to the e9 pick (2r), plus the 1B anchor when present
  tab_layout              T8  huge pages and child prefetch per row class (2s), plus the leaf lever in both page modes

The caption and label of each table live in the section that inputs it. The *_stats functions return
the aggregates the tables print, so numbers.py can register the same values as macros.
"""
import argparse
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (E9_DIRS, GROUP_ORDER, REPO, RESULTS, SIZES, Table, cells_config, finite, fmt, fmt_range,  # noqa: E402
                    geomean, read_csv, scene_sources, size_points, tex, thousands)
import e9 as e9mod  # noqa: E402

MICRO = "$\\mu$s"


def normalise(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def blank(text):
    return tex(text) if text else "--"


# ---------------------------------------------------------------- T1 scenes
def scene_rows(cells):
    sources = scene_sources()
    by_key = {normalise(k): (k, v) for k, v in sources.items()}
    per_scene = defaultdict(list)
    for c in cells:
        per_scene[c["scene"]].append(c)
    rows = []
    for scene, cs in per_scene.items():
        key, info = by_key.get(normalise(scene), (scene, {}))
        sizes = sorted({c["size"] for c in cs}, key=size_points)
        flat = finite([c["flatness"] for c in cs])
        rows.append(dict(scene=key.replace("_", " "), group=cs[0]["group"], info=info, sizes=sizes, n=len(cs),
                         flatness=statistics.median(flat) if flat else float("nan"),
                         flat_min=min(flat) if flat else float("nan"), flat_max=max(flat) if flat else float("nan")))
    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    return sorted(rows, key=lambda r: (order.get(r["group"], 9), r["scene"]))


def tab_scenes(cells):
    table = Table("tab_scenes", ["Scene", "Group", "Acquisition", "Source", "Native points", "Flatness", "Cells"], align="llllrrl")
    group = None
    for r in scene_rows(cells):
        if group is not None and r["group"] != group:
            table.midrule()
        group = r["group"]
        info = r["info"]
        flat = f"{r['flat_min']:.2f}--{r['flat_max']:.2f}" if r["flat_max"] - r["flat_min"] > 0.05 else fmt(r["flatness"], 2)
        cells_text = f"{r['n']} ({r['sizes'][0]}--{r['sizes'][-1]})" if len(r["sizes"]) > 1 else f"{r['n']} ({r['sizes'][0]})"
        table.add(tex(r["scene"]), r["group"], blank(info.get("sensor") or info.get("platform")),
                  blank(info.get("source_short") or info.get("source", "").split(",")[0]),
                  thousands(info["native_points"]) if info.get("native_points") else "--", flat, cells_text)
    return table.write()


# ---------------------------------------------------------------- T2 budgets
def tab_budgets(cells):
    config = cells_config()
    used = {c["cell"] for c in cells}
    table = Table("tab_budgets", ["Points", "Cells", "Queries", "Arm", "GA budget"], align="lrrll")
    for size in SIZES:
        entries = [c for name, c in config.items() if c["size_label"] == size and name in used]
        if entries:
            e = entries[0]
            budget = f"{e['gen_schemas']} $\\times$ {e['generations']} $\\times$ {e['population']}" if e["gen_schemas"] else "--"
            table.add(size, len(entries), e["queries"], e["arm"], budget)
    return table.write()


# ---------------------------------------------------------------- T3 levers
def lever_stats(cells):
    """block title -> dict of geomeans over the held-out chain."""
    out = {}

    def block(title, rows):
        rows = [c for c in rows if c.get("held")]
        if not rows:
            return
        g = lambda key: geomean([c["held"][key] for c in rows if math.isfinite(c["held"][key])])
        out[title] = dict(n=len(rows), ot_l1=g("ot_l1"), kd_l1=g("kd_l1"), l1_l2=g("l1_l2"), ot_best=g("ot_best"), kd_best=g("kd_best"),
                          nested_single=geomean([c["nested_single"] for c in rows if math.isfinite(c["nested_single"])]))

    for size in SIZES:
        block(size, e9mod.by(cells, "size", size))
    for group in GROUP_ORDER:
        block(group, e9mod.by(cells, "group", group))
    block("all", cells)
    return out


def stack(*lines):
    """A header cell broken over several lines, so that a table of word-named ratios fits its column."""
    return "\\begin{tabular}[b]{@{}c@{}}" + "\\\\".join(lines) + "\\end{tabular}"


def tab_levers(cells):
    stats = lever_stats(cells)
    table = Table("tab_levers", ["Size or group", "clouds", stack("octree default", "/ best primitive"), stack("kd-tree default", "/ best primitive"),
                                 stack("best primitive", "/ tuned"), stack("octree default", "/ tuned"), stack("kd-tree default", "/ tuned")])
    for i, title in enumerate(list(SIZES) + GROUP_ORDER + ["all"]):
        if title not in stats:
            continue
        if title in (GROUP_ORDER[0], "all"):
            table.midrule()
        s = stats[title]
        table.add(title, s["n"], fmt(s["ot_l1"]), fmt(s["kd_l1"]), fmt(s["l1_l2"]), fmt(s["ot_best"]), fmt(s["kd_best"]))
    return table.write()


# ---------------------------------------------------------------- T4 composition
def composition_stats(cells):
    out = {}

    def block(title, rows):
        if not rows:
            return
        ns = [c["nested_single"] for c in rows if math.isfinite(c["nested_single"])]
        po = [c["pick_oracle"] for c in rows if math.isfinite(c["pick_oracle"])]
        out[title] = dict(n=len(rows), picked=sum(c["adopted"] == "nested" for c in rows), geo=geomean(ns), lo=min(ns), hi=max(ns),
                          verdicts=e9mod.verdict_counts(ns), pick_oracle=geomean(po))

    for size in SIZES:
        block(size, e9mod.by(cells, "size", size))
    for group in GROUP_ORDER:
        block(group, e9mod.by(cells, "group", group))
    block("all", cells)
    return out


def tab_composition(cells):
    stats = composition_stats(cells)
    table = Table("tab_composition", ["Size or group", "clouds", stack("nested", "adopted"), stack("best nested", "/ best single"), "range", stack("wins / ties", "/ losses"),
                                       stack("searched best", "/ fastest single")])
    for title in list(SIZES) + GROUP_ORDER + ["all"]:
        if title not in stats:
            continue
        if title in (GROUP_ORDER[0], "all"):
            table.midrule()
        s = stats[title]
        w, t, l = s["verdicts"]
        table.add(title, s["n"], s["picked"], fmt(s["geo"]), f"{s['lo']:.2f}--{s['hi']:.2f}", f"{w}\\,/\\,{t}\\,/\\,{l}", fmt(s["pick_oracle"]))
    return table.write()


def workload_stats():
    """workload -> nested/single per cloud, from workload_sweep_makinon2/<cell>/report.txt (6 clouds, 100M)."""
    out = defaultdict(list)
    for report in sorted(RESULTS.glob("workload_sweep_makinon2/*/report.txt")):
        for line in report.read_text().splitlines()[1:]:
            parts = line.split()
            if len(parts) == 6:
                out[parts[0]].append(float(parts[5]))
    return dict(out)


def gpu_stats():
    """nested/single on the GPU (2n confirm rounds, fastest round kept): end to end and kernel only, per cell and k."""
    from gpu_normals_confirm import best_rounds, nested
    from gpu_normals_pilot import KS
    total, kernel, cukd = [], [], []
    for cell_dir in sorted(p for p in (RESULTS / "gpu_normals_makinon2" / "confirm").iterdir() if p.is_dir()):
        for k in KS:
            t = best_rounds(cell_dir, k)
            warp = {s: v for (s, b), v in t.items() if b == "gpuwarp"}
            singles = {s: v for s, v in warp.items() if not nested(s)}
            mixes = {s: v for s, v in warp.items() if nested(s)}
            reference = t.get(("cudakdtree", "gpucudakd"))
            if singles and mixes:
                total.append(min(v["total"] for v in mixes.values()) / min(v["total"] for v in singles.values()))
                kernel.append(min(v["kernel"] for v in mixes.values()) / min(v["kernel"] for v in singles.values()))
            if warp and reference:
                cukd.append(min(v["total"] for v in warp.values()) / reference["total"])
    return dict(total=total, kernel=kernel, ours_over_cudakdtree=cukd)


def node_cost_stats():
    """cost ns -> (nested/single per cell, nested picks, cells); 1 and 10 us from the leaf-8192 rerun (2o)."""
    from node_cost_dial import cell_result
    roots = {0: RESULTS / "node_cost_makinon2" / "d0", 100: RESULTS / "node_cost_makinon2" / "d100",
             1000: RESULTS / "node_cost_leaf8192_makinon2" / "d1000", 10000: RESULTS / "node_cost_leaf8192_makinon2" / "d10000"}
    out = {}
    for cost, directory in roots.items():
        if directory.is_dir():
            names = sorted(p.name[: -len("_pick.json")] for p in directory.glob("*_pick.json"))
            results = [r for r in (cell_result(directory, n) for n in names) if r]
            out[cost] = ([r["ratio"] for r in results], sum(r["adopted"] == "nested" for r in results), len(results))
    return out


def tab_composition_regimes():
    table = Table("tab_composition_regimes", ["Regime", "$n$", "nested\\,/\\,single", "range", "picked"])
    for workload, values in workload_stats().items():
        table.add(f"workload {workload}", len(values), fmt(geomean(values)), fmt_range(values), "--")
    table.midrule()
    gpu = gpu_stats()
    if gpu["total"]:
        table.add("GPU, end to end", len(gpu["total"]), fmt(geomean(gpu["total"])), fmt_range(gpu["total"]), "--")
        table.add("GPU, kernel only", len(gpu["kernel"]), fmt(geomean(gpu["kernel"])), fmt_range(gpu["kernel"]), "--")
    table.midrule()
    for cost, (ratios, adopted, n) in node_cost_stats().items():
        label = f"node cost {cost} ns" if cost < 1000 else f"node cost {cost // 1000} {MICRO}"
        table.add(label, n, fmt(geomean(ratios)), fmt_range(ratios), f"{adopted} of {n}")
    return table.write()


# ---------------------------------------------------------------- T5 out of core
def tab_ooc():
    import ooc
    return ooc.tab_ooc()


# ---------------------------------------------------------------- T6 rule
RULES = [("R0 octree_default", "library octree default, leaf 32"), ("R0 kdtree_default", "library kd-tree default, leaf 32"),
         ("floor: best grid schema per cell", "floor: best of the 55 tuned singles per cell"),
         ("R1 one global schema", "one global schema"), ("R2 per acquisition group", "one schema per scan group"),
         ("R3a feature tree, depth 1", "one split on the flatness score")]
LATENCY_COLUMNS = {"avg_latency_ms": "instrumented", "latency_repeat_mean_ms": "timed passes"}


def rule_results():
    """latency column -> (cells, regrets, picks) from rule_fit (2p)."""
    import rule_fit
    results = {}
    for column in LATENCY_COLUMNS:
        cells = rule_fit.load(REPO, E9_DIRS, column)
        names, L, X = rule_fit.matrices(cells)
        reg, picks = rule_fit.regrets(cells, names, L, X)
        results[column] = (cells, reg, picks)
    return results


def rule_stats(results):
    out = {}
    for key, label in RULES:
        out[key] = {}
        for column in LATENCY_COLUMNS:
            values = sorted(results[column][1][key].values())
            out[key][column] = dict(geo=geomean(values), p90=values[int(round(0.9 * (len(values) - 1)))], over=sum(v > 1.15 for v in values), n=len(values))
    return out


def tab_rule():
    results = rule_results()
    stats = rule_stats(results)
    table = Table("tab_rule", ["Rule", "regret", "p90", "cells $>1.15$", "regret", "p90", "cells $>1.15$"])
    for key, label in RULES:
        row = [label]
        for column in LATENCY_COLUMNS:
            s = stats[key][column]
            row += [fmt(s["geo"]), fmt(s["p90"], 2), str(s["over"])]
        table.add(*row)
    return table.write()


# ---------------------------------------------------------------- T7 libraries
OURS = [("ours, octree default", "octree_default"), ("ours, kd-tree default", "kdtree_default"), ("ours, rule pick", None)]
LIBRARY_FOLDERS = [RESULTS / "external_makinon2"]
ANCHOR_FOLDERS = [RESULTS / "external_1B_makinon2"]


def scene_flatness(cell):
    """The flatness score of a cell, or of the largest searched cell of its scene when the cell was a transfer
    cell without search arms (the 1B cells): the score varies little with size above 25M."""
    from external_report import flatness
    from common import cell_scene
    e9_dirs = [str(d) for d in E9_DIRS]
    value = flatness(e9_dirs, cell)
    if value is None:
        for size in ("100M", "250M", "50M", "25M"):
            value = flatness(e9_dirs, f"{cell_scene(cell)}_{size}")
            if value is not None:
                break
    return value


def library_per_cell(folders=None):
    """(label, setting, metric) -> {cell: ratio to the e9 pick}, over the external batteries present."""
    from external_report import LIBRARIES, METRICS, medians
    folders = folders or LIBRARY_FOLDERS
    per_cell = defaultdict(dict)
    for folder in folders:
        if not folder.is_dir():
            continue
        for cell_dir in sorted(p for p in folder.iterdir() if p.is_dir() and (p / "mds.csv").exists()):
            cell = cell_dir.name
            ours = medians(read_csv(cell_dir / "mds.csv"))
            pick = next(name for name in ours if name.startswith("e9pick_"))
            flat = scene_flatness(cell)
            rule = "qt16l128tc" if flat is not None and flat > 0.60 else "kd48l128sm"
            base = ours[pick]
            for label, name in OURS:
                name = name or rule
                for metric, c in METRICS:
                    if base[c] > 0 and name in ours:
                        per_cell[(label, "default", metric)][cell] = ours[name][c] / base[c]
            for label, tool, prefix, default in LIBRARIES:
                path = cell_dir / f"{tool}.csv"
                if not path.exists():
                    continue
                configs = {n: v for n, v in medians(read_csv(path)).items() if n.startswith(prefix)}
                has_parameter = len(configs) > 1
                for metric, c in METRICS:
                    valid = {n: v[c] for n, v in configs.items() if v[c] > 0}
                    if not valid or base[c] <= 0:
                        continue
                    best = min(valid.values())
                    per_cell[(label, "best", metric)][cell] = best / base[c]
                    if default and default in valid:
                        per_cell[(label, "default", metric)][cell] = valid[default] / base[c]
                        if has_parameter:
                            per_cell[(label, "lever", metric)][cell] = valid[default] / best
    return per_cell


def library_summary(folders=None):
    """(label, setting, metric) -> list of ratios to the e9 pick, and the cells covered."""
    per_cell = library_per_cell(folders)
    cells = sorted({cell for values in per_cell.values() for cell in values})
    return {key: list(values.values()) for key, values in per_cell.items()}, cells


def tab_libraries():
    from external_report import LIBRARIES, METRICS
    summary, cells = library_summary()
    table = Table("tab_libraries", ["Library, setting", "all", "radius", "kNN", "own lever"])
    for label in [l for l, _ in OURS] + [l for l, *_ in LIBRARIES]:
        for setting in ("default", "best"):
            vals = {m: summary.get((label, setting, m), []) for m, _ in METRICS}
            if not any(vals.values()):
                continue
            lever = summary.get((label, "lever", "all"), []) or summary.get((label, "lever", "radius"), [])
            if setting == "best" and not lever:
                continue  # the library exposes no parameter: its best is its default
            name = tex(label) if label.startswith("ours") else f"{tex(label)}, {'def.' if setting == 'default' else 'best'}"
            table.add(name, *[fmt(geomean(vals[m]), 2) if vals[m] else "--" for m, _ in METRICS],
                      fmt(geomean(lever), 2) if setting == "default" and lever else "")
    table.note = f"{len(cells)} cells"
    return table.write()


# ---------------------------------------------------------------- T8 layout
def layout_stats():
    """arm -> row class -> ratios to base, from cache_accounting_makinon2 (2s); plus per-depth bands per cell."""
    from cache_accounting_report import ARMS, bands, load, median_latency
    from out_of_core_report import is_single
    rows, depth = defaultdict(list), {}
    for cell_dir in sorted(p for p in (RESULTS / "cache_accounting_makinon2").iterdir() if p.is_dir()):
        data = load(cell_dir)
        if "base" not in data:
            continue
        size = cell_dir.name.rsplit("_", 1)[1]
        latency = {arm: {name: median_latency(rs) for name, rs in data[arm].items()} for arm in data}
        for name, base_rows in data["base"].items():
            pool_mb = int(base_rows[0]["num_nodes"]) * 32 / 1e6
            classes = []
            if is_single(name):
                classes.append(f"{size} singles")
                if size == "100M":
                    classes.append("100M, pool $\\geq$ 64 MB" if pool_mb >= 64 else "100M, pool $<$ 64 MB")
            else:
                classes.append(f"{size} nested")
            if "hg12" in name:
                classes.append("HGrid rows")
            ratios = {}
            for arm in data:
                if arm != "base" and name in latency[arm]:
                    ratios[arm] = latency[arm][name] / latency["base"][name]
                    for cls in classes:
                        rows[(arm, cls)].append(ratios[arm])
            depth[(cell_dir.name, name)] = dict(bands=bands(base_rows[0]), latency_us=latency["base"][name], pool_mb=pool_mb,
                                               single=is_single(name), size=size, ratios=ratios,
                                               visits=float(base_rows[0]["mean_visited_nodes"]),
                                               visits_by_depth=[float(v) for v in base_rows[0]["visits_by_depth"].split(";") if v],
                                               ns_by_depth=[float(v) for v in base_rows[0]["visit_ns_by_depth"].split(";") if v])
    return dict(rows), depth


LAYOUT_CLASSES = ["100M singles", "100M, pool $\\geq$ 64 MB", "100M, pool $<$ 64 MB", "25M singles", "HGrid rows"]
LAYOUT_NAMES = {"100M singles": "100M points, all structures", "100M, pool $\\geq$ 64 MB": "100M points, node pool $\\geq$ 64 MB",
                "100M, pool $<$ 64 MB": "100M points, node pool $<$ 64 MB", "25M singles": "25M points, all structures",
                "HGrid rows": "hierarchical grid, 64-way"}


def tab_layout():
    rows, _ = layout_stats()
    arms = [a for a in ("huge", "prefetch", "both") if any(k[0] == a for k in rows)]
    labels = {"huge": "huge pages", "prefetch": "child prefetch", "both": "both"}
    table = Table("tab_layout", ["Structures (count)"] + [f"{labels[a]}\\,/\\,base" for a in arms], align="l" + "r" * len(arms))
    for cls in LAYOUT_CLASSES:
        n = len(rows.get((arms[0], cls), []))
        if n:
            table.add(f"{LAYOUT_NAMES[cls]} ({n})", *[f"{fmt(geomean(rows[(a, cls)]))} ({fmt_range(rows[(a, cls)])})" if rows.get((a, cls)) else "--" for a in arms])
    path = table.write()
    try:
        import leaf_lever
        leaf_lever.tab_panel()
        leaf_lever.tab_leaf_lever()
    except (ImportError, FileNotFoundError) as error:
        print(f"  leaf lever table skipped: {error}")
    return path


def interleave_stats():
    """2s follow-up: each schema timed alone over timed interleaved with the panel, per arm (DomFountain 100M), and
    huge / base for the lone SanAndreas 100M schemas, from cache_accounting_makinon2/interleave."""
    folder = RESULTS / "cache_accounting_makinon2" / "interleave"
    med = lambda path: {name: statistics.median(float(r["all_mean_ms"]) for r in rows)
                        for name, rows in _group(read_csv(path), "schema_name").items()} if path.exists() else {}
    out = {"alone_over_interleaved": {}, "sanandreas_huge_over_base": {}}
    for arm in ("base", "huge"):
        alone, inter = med(folder / f"alone_{arm}.csv"), med(folder / f"interleaved_{arm}.csv")
        for name in alone:
            if name in inter:
                out["alone_over_interleaved"][(arm, name)] = alone[name] / inter[name]
    base, huge = med(folder / "sanandreas_alone_base.csv"), med(folder / "sanandreas_alone_huge.csv")
    for name in base:
        if name in huge:
            out["sanandreas_huge_over_base"][name] = huge[name] / base[name]
    return out


def _group(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def null_pair(cell="sanandreas_100M", single="qt24l32", nested="qt9_ot24l32"):
    """Two schemas that build the same tree (identical visits at every depth) timed in shuffled groups: their
    latency ratio per arm is the co-residence noise of a schema-against-schema ratio in the interleaved protocol."""
    _, depth = layout_stats()
    s, n = depth.get((cell, single)), depth.get((cell, nested))
    if not s or not n:
        return {}
    out = {"base": n["latency_us"] / s["latency_us"]}
    for arm in s["ratios"]:
        if arm in n["ratios"]:
            out[arm] = (n["latency_us"] * n["ratios"][arm]) / (s["latency_us"] * s["ratios"][arm])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma list of table names (without tab_)")
    args = parser.parse_args()
    wanted = set(args.only.split(",")) if args.only else None
    cells, notes = e9mod.load_cells()
    print(f"{len(cells)} held-out e9 cells" + (f"; {len(notes)} notes" if notes else ""))
    for note in notes:
        print("  -", note)
    makers = {"scenes": lambda: tab_scenes(cells), "budgets": lambda: tab_budgets(cells), "levers": lambda: tab_levers(cells),
              "composition": lambda: (tab_composition(cells), tab_composition_regimes())[1], "ooc": tab_ooc, "rule": tab_rule,
              "libraries": tab_libraries, "layout": tab_layout}
    for name, make in makers.items():
        if wanted and name not in wanted:
            continue
        path = make()
        print(f"wrote {Path(path).relative_to(REPO)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The paper's data figures as vector PDFs in paper/figures/, from the same readers as the tables.

    python scripts/paper/figures.py [--only levers,composition,rule,depth,scale] [--png DIR]

  fig_levers        Fig. 1  the lever ladder: library default -> best primitive -> tuned single -> best nested, per cell
  fig_composition   Fig. 3  best nested / best single per cell against size; the other regimes as strips
  fig_rule          Fig. 4  flatness score against the oracle's primitive and the rule's regret
  fig_depth         Fig. 5  visits and nanoseconds per visit by depth for three trees, nested finalist overlaid
  fig_scale         Fig. 6  (a) nanoflann's default over our pick from 1M to 1B; (b) the huge-page gain against pool size

Palette: the three scan groups take the first three categorical slots of the dataviz reference palette
(validated all-pairs on white: blue #2a78d6, orange #eb6834, green #199e70), de-emphasis gray #898781;
magnitude by depth uses one blue ramp. Marks are thin, grids are hairlines, text never wears a series colour.
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import E9_DIRS, FIGURES, GROUP_ORDER, GROUP_SHORT, REPO, cell_scene, cell_size, geomean, size_points  # noqa: E402
import e9 as e9mod  # noqa: E402
import labels  # noqa: E402
import tables  # noqa: E402

COLORS = {"ground": "#2a78d6", "aerial": "#eb6834", "forest": "#199e70", "other": "#898781"}
GRAY, INK, MUTED, GRID = "#898781", "#0b0b0b", "#c3c2b7", "#e1e0d9"
RAMP = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
SINGLE, DOUBLE = 3.35, 6.9  # inches: one and two columns of the cas-dc layout
SIZE_TICKS = ["1M", "5M", "25M", "50M", "100M", "250M", "500M", "1B"]

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": INK, "axes.labelcolor": INK, "legend.frameon": False, "lines.linewidth": 1.2, "lines.markersize": 4,
    "axes.titlelocation": "left", "axes.titleweight": "normal", "figure.dpi": 150,
})


def group_legend(ax, groups=GROUP_ORDER, extra=(), **kw):
    """Scan-group legend in one row just above the axes (below the title), never over the data."""
    handles = [Line2D([], [], marker="o", color=COLORS[g], linestyle="", markersize=4, label=g) for g in groups] + list(extra)
    kw = dict(dict(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles), columnspacing=1.0, borderaxespad=0.2), **kw)
    ax.legend(handles=handles, handletextpad=0.3, **kw)


def log_ratio_axis(ax, lo, hi, ticks):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()


def save(fig, name, png_dir):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    if png_dir:
        Path(png_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(png_dir) / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- Fig. 1 levers
def fig_levers(cells, png_dir):
    stages = ["library default\n(octree, leaf 32)", "best primitive\n(default parameters)", "tuned single\n(any leaf, split, depth)"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    per_group = {g: [] for g in GROUP_ORDER}
    for c in cells:
        h = c.get("held")
        if not h or not all(math.isfinite(v) for v in (h["ot_best"], h["l1_l2"])):
            continue
        y = [h["ot_best"], h["l1_l2"], 1.0]
        ax.plot(range(3), y, color=COLORS[c["group"]], alpha=0.25, linewidth=0.6, zorder=2)
        per_group.setdefault(c["group"], []).append(y)
    for g in GROUP_ORDER:
        ys = per_group.get(g)
        if not ys:
            continue
        gm = [geomean([y[i] for y in ys]) for i in range(3)]
        ax.plot(range(3), gm, color=COLORS[g], linewidth=2, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=1, zorder=4,
                label=f"{g} ({len(ys)})")
    ax.axhline(1.0, color=MUTED, linewidth=0.6, zorder=1)
    log_ratio_axis(ax, 0.9, 3.2, [1, 1.25, 1.5, 2, 3])
    ax.set_xticks(range(3))
    ax.set_xticklabels(stages)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylabel("latency / tuned single")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, columnspacing=1.0, handlelength=1.6, borderaxespad=0,
              title="scan group (cells)", title_fontsize=6.5, alignment="left")
    return save(fig, "fig_levers", png_dir)


WORKLOAD_NAMES = {"base": "pipeline trace", "radius4": "radius x4", "radius16": "radius x16", "knn64": "k = 64",
                  "range8": "range boxes", "count64": "count boxes", "mixed": "mixed"}


# ---------------------------------------------------------------- Fig. 3 composition
def fig_composition(cells, png_dir):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), gridspec_kw=dict(width_ratios=[3, 2.2], wspace=0.62))
    ax.axhspan(0.97, 1.03, color=GRID, zorder=0)
    ax.axhline(1.0, color=MUTED, linewidth=0.6, zorder=1)
    for g in GROUP_ORDER:
        rows = [c for c in cells if c["group"] == g and math.isfinite(c["nested_single"])]
        ax.scatter([size_points(c["size"]) for c in rows], [c["nested_single"] for c in rows], s=14, color=COLORS[g],
                   edgecolors="white", linewidths=0.5, zorder=3, label=g)
    ax.set_xscale("log")
    ax.set_xticks([size_points(s) for s in SIZE_TICKS])
    ax.set_xticklabels([s if s in ("1M", "5M", "25M", "100M", "1B") else "" for s in SIZE_TICKS])
    ax.minorticks_off()
    log_ratio_axis(ax, 0.85, 1.2, [0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.2])
    ax.set_xlabel("points")
    ax.set_ylabel("best nested / best single")
    ax.set_title("(a) in memory, held-out half", pad=14)
    group_legend(ax)
    ax.text(0.99, 0.03, "shaded: within 3%", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#52514e")

    strips = []
    for workload, values in tables.workload_stats().items():
        strips.append((WORKLOAD_NAMES.get(workload, workload), values))
    gpu = tables.gpu_stats()
    strips += [("GPU, end to end", gpu["total"]), ("GPU, kernel only", gpu["kernel"])]
    for cost, (ratios, adopted, n) in tables.node_cost_stats().items():
        label = f"node cost {cost} ns" if cost < 1000 else f"node cost {cost // 1000} µs"
        strips.append((label, ratios))
    bx.axvspan(0.97, 1.03, color=GRID, zorder=0)
    bx.axvline(1.0, color=MUTED, linewidth=0.6, zorder=1)
    for i, (label, values) in enumerate(strips):
        y = len(strips) - 1 - i
        bx.scatter(values, [y] * len(values), s=10, color=GRAY, alpha=0.6, edgecolors="none", zorder=2)
        bx.plot([geomean(values)], [y], marker="|", color=INK, markersize=9, markeredgewidth=1.4, zorder=3)
    bx.set_yticks(range(len(strips)))
    bx.set_yticklabels([s[0] for s in reversed(strips)])
    bx.set_xscale("log")
    bx.set_xticks([0.6, 0.8, 1, 1.25, 1.5])
    bx.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.5"])
    bx.minorticks_off()
    bx.set_xlim(0.55, 1.6)
    bx.set_xlabel("best nested / best single")
    bx.set_title("(b) other workloads, GPU, emulated node costs", pad=14)
    bx.grid(axis="y", visible=False)
    bx.tick_params(axis="y", length=0)
    return save(fig, "fig_composition", png_dir)


# ---------------------------------------------------------------- Fig. 4 rule
def fig_rule(png_dir):
    import rule_fit
    cells = rule_fit.load(REPO, E9_DIRS, "avg_latency_ms")
    names, L, X = rule_fit.matrices(cells)
    reg, _ = rule_fit.regrets(cells, names, L, X)
    rows = rule_fit.per_cell_table(cells, reg, names, L, X)
    kind = lambda schema: "kd-tree" if schema.startswith("tuned_kd") else "quadtree" if schema.startswith("tuned_qt") else "other"
    markers = {"kd-tree": "o", "quadtree": "s", "other": "^"}
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(SINGLE, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2], hspace=0.35))
    for r in rows:
        k = kind(r["oracle_opt"])
        colour = COLORS[GROUP_SHORT.get(r["group"], "other")]  # rule_fit keeps summarize_nesting's long group names
        ax.scatter(r["flatness"], ["other", "quadtree", "kd-tree"].index(k), marker=markers[k], s=16, color=colour,
                   edgecolors="white", linewidths=0.5, zorder=3)
        bx.scatter(r["flatness"], r["regret_rule"], marker=markers[k], s=16, color=colour, edgecolors="white", linewidths=0.5, zorder=3)
        bx.scatter(r["flatness"], r["regret_octree_default"], marker=markers[k], s=9, color=GRAY, alpha=0.6, edgecolors="none", zorder=2)
    for a in (ax, bx):
        a.axvline(0.60, color=MUTED, linewidth=0.8, zorder=1)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["other", "quadtree", "kd-tree"])
    ax.set_xlim(0.2, 0.95)
    ax.set_ylim(-0.5, 2.5)
    ax.set_title("(a) primitive of the fastest single per cell", pad=12, fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="y", length=0)
    log_ratio_axis(bx, 0.95, 3.2, [1, 1.25, 1.5, 2, 3])
    bx.set_xlabel("flatness score")
    bx.set_ylabel("latency / fastest single")
    bx.set_title("(b) regret of the rule (colour) and of the octree default (gray)", pad=4, fontsize=7)
    ax.text(0.59, 1.02, "rule: kd-tree", transform=ax.get_xaxis_transform(), fontsize=6, color="#52514e", ha="right", va="bottom")
    ax.text(0.61, 1.02, "quadtree", transform=ax.get_xaxis_transform(), fontsize=6, color="#52514e", ha="left", va="bottom")
    # One legend below the x label: colour = scan group, shape = primitive of the fastest single.
    handles = [Line2D([], [], marker="o", color=COLORS[g], linestyle="", markersize=4, label=g) for g in GROUP_ORDER]
    handles += [Line2D([], [], marker=m, color=GRAY, linestyle="", markersize=3.5, label=k) for k, m in markers.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.0), ncol=3, columnspacing=1.2, handletextpad=0.3)
    return save(fig, "fig_rule", png_dir)


# ---------------------------------------------------------------- Fig. 5 depth
DEPTH_CELLS = [("domfountain_100M", "kd48l128sm", "DomFountain, 100M (terrestrial)"),
               ("lille_100M", "kd48l64sm", "Lille, 100M (mobile)"),
               ("sanandreas_100M", "qt24l32", "SanAndreas, 100M (aerial)")]


def fig_depth(png_dir):
    _, depth = tables.layout_stats()
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 3.1), sharex="col", gridspec_kw=dict(hspace=0.12, wspace=0.3))
    for col, (cell, pick, title) in enumerate(DEPTH_CELLS):
        p = depth.get((cell, pick))
        if not p:
            continue
        top, bottom = axes[0, col], axes[1, col]
        depths = range(len(p["visits_by_depth"]))
        top.bar(depths, p["visits_by_depth"], width=0.8, color=COLORS["ground"] if "kd" in pick else COLORS["aerial"], zorder=3)
        top.set_title(f"{title}\n{labels.short(pick)}", fontsize=7)
        top.set_ylim(0, 1.15 * max(p["visits_by_depth"]))
        ns = [v if visits > 0.005 else float("nan") for v, visits in zip(p["ns_by_depth"], p["visits_by_depth"])]
        bottom.plot(depths, ns, color=INK, marker="o", markersize=2.5, linewidth=1)
        bottom.set_yscale("log")
        bottom.set_ylim(20, 6000)
        bottom.set_yticks([30, 100, 300, 1000, 3000])
        bottom.set_yticklabels(["30", "100", "300", "1000", "3000"])
        bottom.minorticks_off()
        bottom.set_xlabel("depth")
        bottom.axvspan(-0.5, 7.5, color=GRID, zorder=0)
        top.axvspan(-0.5, 7.5, color=GRID, zorder=0)
        if col == 0:
            top.set_ylabel("visits per query")
            bottom.set_ylabel("ns per visit")
    axes[1, 0].text(0.97, 0.05, "shaded: depths 0-7", transform=axes[1, 0].transAxes, fontsize=6, color="#52514e", va="bottom", ha="right")
    return save(fig, "fig_depth", png_dir)


# ---------------------------------------------------------------- Fig. 6 scale and pools
def fig_scale(cells, png_dir):
    per_cell = tables.library_per_cell(tables.LIBRARY_FOLDERS + tables.ANCHOR_FOLDERS)
    ratios = per_cell.get(("nanoflann kd", "default", "all"), {})
    group_of_scene = {c["scene"]: c["group"] for c in cells}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.3))
    by_scene = {}
    for cell, r in ratios.items():
        by_scene.setdefault(cell_scene(cell), []).append((size_points(cell_size(cell)), r))
    ends = []
    for scene, pts in sorted(by_scene.items()):
        pts.sort()
        g = group_of_scene.get(scene, "other")
        ax.plot([x for x, _ in pts], [r for _, r in pts], color=COLORS[g], linewidth=0.9, alpha=0.8, marker="o", markersize=3.5,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ends.append([labels.scene(scene), pts[-1][0], math.log10(pts[-1][1]), math.log10(pts[-1][1])])
    # Spread the end labels so that none overlaps: at least 0.022 decades apart, nudged from the middle out.
    ends.sort(key=lambda e: e[2])
    gap = 0.036
    for _ in range(50):
        moved = False
        for a, b in zip(ends, ends[1:]):
            if b[3] - a[3] < gap:
                shift = (gap - (b[3] - a[3])) / 2
                a[3] -= shift
                b[3] += shift
                moved = True
        if not moved:
            break
    for name, x, y, y_label in ends:
        ax.annotate(name, (x, 10 ** y), xytext=(1.9e9, 10 ** y_label), fontsize=5.5, color="#52514e", va="center",
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.5) )
    ax.axhline(1.0, color=MUTED, linewidth=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(7e5, 9e9)
    ax.set_xticks([size_points(s) for s in SIZE_TICKS])
    ax.set_xticklabels([s if s in ("1M", "5M", "25M", "100M", "1B") else "" for s in SIZE_TICKS])
    ax.minorticks_off()
    log_ratio_axis(ax, 0.7, 1.5, [0.7, 0.8, 0.9, 1, 1.2, 1.4])
    ax.set_xlabel("points")
    ax.set_ylabel("nanoflann default / our pick")
    ax.set_title("(a) nanoflann at its default leaf, all queries", pad=14)
    group_legend(ax)

    _, depth = tables.layout_stats()
    for (cell, name), d in depth.items():
        if "huge" not in d["ratios"] or not d["single"]:
            continue
        bx.scatter(d["pool_mb"], d["ratios"]["huge"], s=16, color=COLORS["ground"] if d["size"] == "100M" else GRAY,
                   edgecolors="white", linewidths=0.5, zorder=3)
    bx.axhline(1.0, color=MUTED, linewidth=0.6, zorder=1)
    bx.set_xscale("log")
    bx.set_xticks([3, 10, 30, 100, 300])
    bx.set_xticklabels(["3", "10", "30", "100", "300"])
    bx.minorticks_off()
    log_ratio_axis(bx, 0.65, 1.05, [0.7, 0.8, 0.9, 1])
    bx.set_xlabel("node pool, MB (32 B per node)")
    bx.set_ylabel("huge pages / base")
    bx.set_title("(b) 2 MB over 4 KiB pages, by node pool size", pad=14)
    bx.legend(handles=[Line2D([], [], marker="o", color=COLORS["ground"], linestyle="", markersize=4, label="100M"),
                       Line2D([], [], marker="o", color=GRAY, linestyle="", markersize=4, label="25M")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    return save(fig, "fig_scale", png_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="")
    parser.add_argument("--png", default="", help="also write PNG previews to this folder")
    args = parser.parse_args()
    wanted = set(args.only.split(",")) if args.only else None
    cells, _ = e9mod.load_cells()
    makers = {"levers": lambda: fig_levers(cells, args.png), "composition": lambda: fig_composition(cells, args.png),
              "rule": lambda: fig_rule(args.png), "depth": lambda: fig_depth(args.png), "scale": lambda: fig_scale(cells, args.png)}
    for name, make in makers.items():
        if wanted and name not in wanted:
            continue
        print(f"wrote {make().relative_to(REPO)}")


if __name__ == "__main__":
    main()

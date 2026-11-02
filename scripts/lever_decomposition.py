#!/usr/bin/env python3
"""Where does the latency go: primitive choice, parameter tuning, or composition? From e9 output.

    python scripts/lever_decomposition.py results/eval_traces/matrix_e9 [more dirs...] [--out table.md]

A schema search that rarely picks a nested schema (summarize_nesting.py) still improves on what a user
would otherwise run. This splits that improvement into three levers, each a latency ratio (>1 = the
later lever is faster, geomeans over cells):

  default/lever1   choosing the primitive: the library default (an octree, and a median kd-tree) against
                   the best primitive type at default-ish parameters
  lever1/lever2    tuning the parameters: that against the best single of any leaf, policy and depth
  nested/single    composing: the best nested finalist against the best single finalist (same numbers
                   as summarize_nesting.py; overall geomean ~1.01 in e9, i.e. no gain)
  default/best     the whole single-structure gain

Two chains are reported, because the battery measures singles in two places:
  held-out  <cell>_test.csv (test half): the finalists plus every library default (leaf 32). The default
            is octree_default / kdtree_default, lever1 the best `*_default`, lever2 the best single there.
            Latencies come from <cell>_test_timing.csv (median over processes of all_mean_ms) when both
            schemas of a ratio were timed interleaved, else from the instrumented avg_latency_ms.
  opt half  <cell>_singles.csv (the tuned grid) plus the singles of <cell>_ga.csv, instrumented. The
            default is tuned_ot16l128 (octree, leaf 128) and tuned_kd20l32 (median kd, leaf 32); lever1
            the best per primitive type of the grid schemas with default split policy and leaf nearest
            --leaf (128); lever2 the best single of grid and GA. Transfer cells (500M, 1B) have no search
            arms and are skipped here.

Single vs nested is the schema file's level count (load_schema resolves server paths), falling back to
the active_structure_types column. Cells are grouped by morphology (as summarize_nesting.py) and by the
size suffix of the cell name.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_nesting import GROUPS, REPO, SIZES, geomean, load_schema, timing  # noqa: E402

DEFAULT_POLICIES = {"KDTree": ("median_longest_axis",), "BVH": ("center_longest_axis",)}


def norm(path):
    return path.replace("\\", "/")


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def describe(row, directory):
    """Shape of a schema row: levels, primitive type and parameters of its first level."""
    schema = load_schema(norm(row["schema_path"]), directory)
    if schema and schema.get("levels"):
        first = schema["levels"][0]
        tight = bool(first.get("tightChildren")) or first.get("splitAt") == "tight" or row["schema_name"].endswith("tc")
        return dict(levels=len(schema["levels"]), type=first.get("type", "?"), leaf=first.get("leafCapacity"),
                    policy=first.get("axisPolicy", ""), depth=first.get("numLevels"), tight=tight)
    return dict(levels=int(row.get("active_structure_types") or 0), type="?", leaf=None, policy="",
                depth=None, tight=False)


def default_policy(shape):
    return shape["policy"] in ("",) + DEFAULT_POLICIES.get(shape["type"], ()) and not shape["tight"]


def cell_size(name, meta):
    m = re.search(r"_(\d+[MB])(?:_|$)", name)
    return m.group(1) if m else meta.get("size_label", "")


class Chain:
    """Singles of one measurement (one csv), with a latency per schema and a ratio helper."""

    def __init__(self, rows, directory, timed=None):
        self.timed = timed or {}
        self.rows = []
        for row in rows:
            shape = describe(row, directory)
            if shape["levels"] != 1:
                continue
            self.rows.append(dict(name=row["schema_name"], path=norm(row["schema_path"]),
                                  lat=float(row["avg_latency_ms"]), **shape))

    def ratio(self, a, b):
        """lat(a)/lat(b); interleaved held-out timing when both were timed, else instrumented."""
        if not a or not b:
            return float("nan")
        if a["path"] in self.timed and b["path"] in self.timed:
            return self.timed[a["path"]] / self.timed[b["path"]]
        return a["lat"] / b["lat"]

    def best(self, rows=None):
        rows = self.rows if rows is None else rows
        return min(rows, key=lambda r: r["lat"]) if rows else None

    def find(self, name):
        return next((r for r in self.rows if r["name"] == name), None)

    def lever1(self, candidates, leaf):
        """Best over primitive types of the default-policy schema with leaf nearest `leaf` per type."""
        per_type = {}
        for r in candidates:
            if not default_policy(r) or not r["leaf"]:
                continue
            key = (abs(math.log(r["leaf"] / leaf)), r["leaf"], -(r["depth"] or 0))
            if r["type"] not in per_type or key < per_type[r["type"]][0]:
                per_type[r["type"]] = (key, r)
        return self.best([v[1] for v in per_type.values()])


def analyse(directory, name, meta, leaf, notes):
    pick = json.loads((directory / f"{name}_pick.json").read_text())
    test_rows = read_rows(directory / f"{name}_test.csv")
    if not test_rows:
        notes.append(f"{name}: no _test.csv, skipped")
        return None
    timing_csv = directory / f"{name}_test_timing.csv"
    timed = timing(timing_csv) if timing_csv.exists() else {}
    if not timed:
        notes.append(f"{name}: no _test_timing.csv, held-out ratios use the instrumented pass")
    out = dict(cell=name, morph=meta.get("morphology", ""), size=cell_size(name, meta))

    # nested/single exactly as summarize_nesting.py
    n, s = pick.get("best_nested"), pick.get("best_single")
    out["nested_single"] = timed[n] / timed[s] if n in timed and s in timed else float("nan")
    if not math.isfinite(out["nested_single"]):
        notes.append(f"{name}: finalists not both in test_timing, nested/single is nan")

    # held-out chain
    held = Chain(test_rows, directory, timed)
    defaults = [r for r in held.rows if r["name"].endswith("_default")]
    d_ot, d_kd = held.find("octree_default"), held.find("kdtree_default")
    l1, l2 = held.lever1(defaults, leaf=32), held.best()
    out["held"] = dict(d_ot=d_ot, d_kd=d_kd, l1=l1, l2=l2,
                       ot_l1=held.ratio(d_ot, l1), kd_l1=held.ratio(d_kd, l1), l1_l2=held.ratio(l1, l2),
                       ot_best=held.ratio(d_ot, l2), kd_best=held.ratio(d_kd, l2))
    if not d_ot or not d_kd:
        notes.append(f"{name}: library default missing in _test.csv ({'octree' if not d_ot else 'kdtree'})")

    # opt-half chain
    grid_rows = read_rows(directory / f"{name}_singles.csv")
    ga_rows = read_rows(directory / f"{name}_ga.csv")
    if grid_rows:
        grid = Chain(grid_rows, directory)
        both = Chain(grid_rows + ga_rows, directory)
        d_ot, d_kd = grid.find("tuned_ot16l128"), grid.find("tuned_kd20l32")
        if not d_ot:  # nearest octree leaf 128 without tc
            d_ot = grid.lever1([r for r in grid.rows if r["type"] == "Octree"], leaf)
            notes.append(f"{name}: tuned_ot16l128 absent, default octree is {d_ot['name'] if d_ot else 'none'}")
        if not d_kd:
            notes.append(f"{name}: tuned_kd20l32 absent in the grid")
        l1, l2 = grid.lever1(grid.rows, leaf), both.best()
        out["opt"] = dict(d_ot=d_ot, d_kd=d_kd, l1=l1, l2=l2,
                          ot_l1=both.ratio(d_ot, l1), kd_l1=both.ratio(d_kd, l1), l1_l2=both.ratio(l1, l2),
                          ot_best=both.ratio(d_ot, l2), kd_best=both.ratio(d_kd, l2))
    else:
        out["opt"] = None
        notes.append(f"{name}: no _singles.csv (transfer cell), opt-half chain skipped")
    return out


def label(r):
    return r["name"] if r else "-"


def fmt(v):
    return f"{v:.3f}" if v is not None and math.isfinite(v) else "nan"


def table(cells, chain, lines):
    lines.append("| cell | morphology | size | ot_def/l1 | kd_def/l1 | l1/l2 | nested/single | ot_def/best | kd_def/best "
                 "| lever1 | lever2 (best single) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for c in cells:
        h = c[chain]
        if not h:
            continue
        lines.append(f"| {c['cell']} | {c['morph']} | {c['size']} | {fmt(h['ot_l1'])} | {fmt(h['kd_l1'])} | {fmt(h['l1_l2'])} | "
                     f"{fmt(c['nested_single'])} | {fmt(h['ot_best'])} | {fmt(h['kd_best'])} | {label(h['l1'])} | {label(h['l2'])} |")


def aggregate(cells, chain, lines):
    def block(title, rows):
        rows = [c for c in rows if c[chain]]
        if not rows:
            return

        def g(key):
            vals = [c[chain][key] for c in rows if math.isfinite(c[chain][key])]
            return f"{geomean(vals):.3f}" if vals else "nan"

        ns = [c["nested_single"] for c in rows if math.isfinite(c["nested_single"])]
        lines.append(f"| {title} | {len(rows)} | {g('ot_l1')} | {g('kd_l1')} | {g('l1_l2')} | "
                     f"{fmt(geomean(ns))} ({len(ns)}) | {g('ot_best')} | {g('kd_best')} |")

    lines.append("| group | cells | ot_def/l1 | kd_def/l1 | l1/l2 | nested/single (n) | ot_def/best | kd_def/best |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for size in SIZES:
        block(size, [c for c in cells if c["size"] == size])
    for group, morphs in GROUPS.items():
        block(group, [c for c in cells if c["morph"] in morphs])
    block("all", cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--leaf", type=int, default=128, help="target leaf capacity of the lever-1 (default-ish) grid schemas")
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
            result = analyse(directory, name, meta.get(name, {}), args.leaf, notes)
            if result:
                cells.append(result)
    order = {s: i for i, s in enumerate(SIZES)}
    cells.sort(key=lambda c: (order.get(c["size"], 99), c["cell"]))

    lines = [f"# Lever decomposition ({len(cells)} cells)", "",
             "Ratios are latency(first)/latency(second): >1 means the second lever is faster. "
             "nested/single is the best nested finalist over the best single finalist (held-out, interleaved).", "",
             "## Held-out chain (test half: library defaults leaf 32, finalists)", ""]
    table(cells, "held", lines)
    lines += ["", "### Geomeans, held-out chain", ""]
    aggregate(cells, "held", lines)
    lines += ["", f"## Opt-half chain (tuned grid + GA singles, instrumented; default tuned_ot16l128 / tuned_kd20l32, "
              f"lever1 leaf nearest {args.leaf})", ""]
    table(cells, "opt", lines)
    lines += ["", "### Geomeans, opt-half chain", ""]
    aggregate(cells, "opt", lines)
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

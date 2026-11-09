"""Out of core (testing.md 2q) for the paper: stage B adoption of a nested prefix over each regime's best
single, per device class, with the hierarchy in RAM and with it paged; and the price of paging the
hierarchy. Reads the stage B CSVs of the three out-of-core batteries with the device models of 2q:

  out_of_core_model_makinon2       stage B on the SATA, HDD and settled-QLC models, regimes 0:memory and 0:paged
  out_of_core_gridcheck_makinon2   the wider 0:paged panel on the settled-QLC model
  out_of_core_paged_makinon2       regime 0.01:paged (1% LRU cache) on the SATA and NVMe models, finer leaf grid
  out_of_core_makinon2             stage A (trace.csv): the regimes' best singles per device model

A prefix is adopted when it is at least 3% faster than the single in every process (out_of_core_report.stage_b).
"""
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import RESULTS, Table, fmt, fmt_range, geomean  # noqa: E402
from out_of_core_report import is_single, load, medians, stage_b  # noqa: E402

MODELS = {"nvme": (60.0, 3000.0), "sata": (120.0, 500.0), "hdd": (8000.0, 150.0), "qlc": (107.0, 47.0)}
DEVICE_LABEL = {"nvme": "NVMe", "sata": "SATA SSD", "hdd": "HDD", "qlc": "QLC SSD, at rest"}
CELL_LABEL = {"domfountain_100M": "DomFountain", "galiciaforest_100M": "GaliciaForest", "lille_100M": "Lille",
              "sanandreas_100M": "SanAndreas", "waikato_hamilton_100M": "Waikato"}

# (hierarchy, cache, workload, regime file stem, folder, device tags)
ROWS = [
    ("in RAM", "none", "trace", "0-memory", "out_of_core_model_makinon2", ["sata", "hdd", "qlc"]),
    ("in RAM", "none", "boxes", "0-memory", "out_of_core_model_makinon2", ["sata", "hdd", "qlc"]),
    ("paged", "none", "trace", "0-paged", "out_of_core_model_makinon2", ["sata", "hdd"]),
    ("paged", "none", "trace", "0-paged", "out_of_core_gridcheck_makinon2", ["qlc"]),
    ("paged", "1\\% LRU", "trace", "0.01-paged", "out_of_core_paged_makinon2", ["nvme", "sata"]),
]


def boxes_hit(cell):
    """The box workload returns points on this cell (GaliciaForest is an archipelago of plots, and boxes drawn
    over the trace's extent miss them all; out_of_core_report skips it the same way)."""
    path = RESULTS / "out_of_core_makinon2" / cell / "boxes.csv"
    return path.exists() and any(float(row["mean_returned_points"] or 0) > 0 for row in load(path))


def stage_b_rows():
    """One record per (hierarchy, cache, workload, device): per-cell best prefix / single and the adoptions."""
    out = []
    for hierarchy, cache, workload, regime, folder, devices in ROWS:
        for device in devices:
            ratios, adopted = {}, {}
            for cell_dir in sorted(p for p in (RESULTS / folder).iterdir() if p.is_dir()):
                path = cell_dir / f"stageB_{workload}_{regime}_{device}.csv"
                if not path.exists() or (workload == "boxes" and not boxes_hit(cell_dir.name)):
                    continue
                single, base, best, taken = stage_b(path, "model_ms", MODELS[device])
                ratios[cell_dir.name] = best[1]
                if taken:
                    adopted[cell_dir.name] = min(r for _, r in taken)
            if ratios:
                out.append(dict(hierarchy=hierarchy, cache=cache, workload=workload, regime=regime.replace("-", ":"), device=device,
                                ratios=ratios, adopted=adopted))
    return out


PAIRS = {"no cache": ("0:paged", "0:memory"), "1% LRU": ("0.01:paged", "0.01:memory")}


def paging_price():
    """Per device model: the best schema with the hierarchy paged over the best schema with it pinned in RAM,
    with and without the 1% cache (nested or single, whichever is fastest in each regime), and the best
    singles' leaf sizes per regime, from stage A (trace.csv) of out_of_core_makinon2."""
    out = {}
    for device, model in MODELS.items():
        prices, leaves = defaultdict(list), defaultdict(set)
        for cell_dir in sorted(p for p in (RESULTS / "out_of_core_makinon2").iterdir() if p.is_dir()):
            path = cell_dir / "trace.csv"
            if not path.exists():
                continue
            timing = medians(load(path, model), "model_ms")
            best_any, best_single = {}, {}
            for (name, regime), (t, _) in timing.items():
                if t <= 0:
                    continue
                if regime not in best_any or t < best_any[regime][1]:
                    best_any[regime] = (name, t)
                if is_single(name) and (regime not in best_single or t < best_single[regime][1]):
                    best_single[regime] = (name, t)
            for label, (paged, pinned) in PAIRS.items():
                if paged in best_any and pinned in best_any:
                    prices[label].append(best_any[paged][1] / best_any[pinned][1])
            for regime, (name, _) in best_single.items():
                leaves[regime].add(int(name.split("l")[-1].rstrip("smtc")))
        out[device] = dict(prices=dict(prices), leaves={r: sorted(v) for r, v in leaves.items()})
    return out


ISPRS = [("grid1_qt3_grid1_ot5l4096", "qt10l4096", "ot10l4096"), ("grid1_qt3_grid1_ot5l16384", "qt10l16384", "ot10l16384")]


def isprs_stats(device="sata", regimes=("0:memory", "0:paged", "0.01:paged")):
    """The ISPRS 2023 nested schemas against their leaf-matched depth-10 singles and against the regime's best
    single, per regime on one device model (stage A trace.csv): geomeans over cells."""
    out = {}
    model = MODELS[device]
    for regime in regimes:
        matched, best = [], []
        for cell_dir in sorted(p for p in (RESULTS / "out_of_core_makinon2").iterdir() if p.is_dir()):
            path = cell_dir / "trace.csv"
            if not path.exists():
                continue
            timing = {name: t for (name, r), (t, _) in medians(load(path, model), "model_ms").items() if r == regime and t > 0}
            singles = {n: t for n, t in timing.items() if is_single(n)}
            if not singles:
                continue
            for paper, qt, ot in ISPRS:
                if paper in timing and (qt in timing or ot in timing):
                    matched.append(timing[paper] / min(timing.get(qt, float("inf")), timing.get(ot, float("inf"))))
                    best.append(timing[paper] / min(singles.values()))
        if matched:
            out[regime] = dict(matched=geomean(matched), best=geomean(best), n=len(matched))
    return out


def hierarchy_mb(regime="0:memory"):
    """Node pool of each cell's best single with the leaves on the device, per device model, in MB (32 B per
    node): what pinning the hierarchy in RAM costs."""
    sizes = []
    for device, model in MODELS.items():
        for cell_dir in sorted(p for p in (RESULTS / "out_of_core_makinon2").iterdir() if p.is_dir()):
            path = cell_dir / "trace.csv"
            if not path.exists():
                continue
            rows = load(path, model)
            timing = {name: t for (name, r), (t, _) in medians(rows, "model_ms").items() if r == regime and is_single(name) and t > 0}
            if not timing:
                continue
            best = min(timing, key=timing.get)
            nodes = next((int(row["num_nodes"]) for row in rows if row["schema_name"] == best and row["store_regime"] == regime), 0)
            sizes.append(nodes * 32 / 1e6)
    return sizes


def ooc_stats():
    rows = stage_b_rows()
    in_ram = [r for r in rows if r["hierarchy"] == "in RAM"]
    paged = [r for r in rows if r["hierarchy"] == "paged"]
    count = lambda rs: (sum(len(r["adopted"]) for r in rs), sum(len(r["ratios"]) for r in rs))
    return dict(rows=rows, in_ram=count(in_ram), paged=count(paged), price=paging_price())


def tab_ooc():
    stats = ooc_stats()
    table = Table("tab_ooc", ["Hierarchy", "Cache", "Workload", "Device model", "$n$", "best prefix\\,/\\,single", "adopted"], align="lllllrl")
    last = None
    for r in stats["rows"]:
        if last is not None and r["hierarchy"] != last:
            table.midrule()
        last = r["hierarchy"]
        adopted = ", ".join(f"{CELL_LABEL.get(c, c)} {v:.2f}" for c, v in sorted(r["adopted"].items(), key=lambda kv: kv[1])) or "none"
        values = list(r["ratios"].values())
        table.add(r["hierarchy"], r["cache"], r["workload"], DEVICE_LABEL[r["device"]], len(values),
                  f"{fmt(geomean(values))} ({fmt_range(values)})", adopted)
    path = table.write()
    a, n = stats["in_ram"]
    b, m = stats["paged"]
    print(f"  out of core: hierarchy in RAM {a} of {n} adopted, paged {b} of {m}")
    for device, p in stats["price"].items():
        prices = ", ".join(f"{label} {fmt_range(v)}" for label, v in p["prices"].items())
        print(f"  {device}: paged / pinned best schema: {prices}; best single leaves {p['leaves']}")
    return path


if __name__ == "__main__":
    tab_ooc()

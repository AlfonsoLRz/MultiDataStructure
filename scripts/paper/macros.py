#!/usr/bin/env python3
"""Every measured number the manuscript quotes, as LaTeX macros in paper/generated/numbers.tex, with its
source in numbers.md. The prose uses the macros only, so a regenerated battery cannot leave a stale
number behind. Run from the repo root:

    python scripts/paper/macros.py
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import GROUP_ORDER, Macros, finite, geomean  # noqa: E402
import e9 as e9mod  # noqa: E402
import ooc  # noqa: E402
import tables  # noqa: E402

E9 = "e9 held-out cells (matrix_e9, matrix_e9_makinon2), summarize_nesting/lever_decomposition"


def pct(ratio):
    return (ratio - 1) * 100


def main():
    m = Macros()
    cells, _ = e9mod.load_cells()

    # ---- the study
    scenes = tables.scene_rows(cells)
    m.add("nCells", len(cells), E9)
    m.add("nScenes", len(scenes), E9)
    m.add("nSearchedCells", sum(not c["transfer"] for c in cells), E9)
    m.add("nTransferCells", sum(c["transfer"] for c in cells), E9)
    for group in GROUP_ORDER:
        m.add("nCells" + group.capitalize(), len(e9mod.by(cells, "group", group)), E9)

    # ---- composition in memory (T4)
    comp = tables.composition_stats(cells)
    m.add("nestedSingleAll", comp["all"]["geo"], E9)
    m.add("nestedPickedAll", comp["all"]["picked"], E9)
    m.add("pickOracleAll", comp["all"]["pick_oracle"], E9)
    m.add("nestedSingleLo", comp["all"]["lo"], E9, 2)
    m.add("nestedSingleHi", comp["all"]["hi"], E9, 2)
    wins, ties, losses = comp["all"]["verdicts"]
    m.add("nestedWinsAll", wins, E9)
    m.add("nestedTiesAll", ties, E9)
    m.add("nestedLossesAll", losses, E9)
    for group in GROUP_ORDER:
        m.add("nestedSingle" + group.capitalize(), comp[group]["geo"], E9)
    m.add("nestedSingleOneB", comp["1B"]["geo"], E9)
    m.add("nestedSingleHundredM", comp["100M"]["geo"], E9)

    # ---- levers (T3)
    lev = tables.lever_stats(cells)["all"]
    m.add("leverOctreeDefaultTuned", lev["ot_best"], E9, 2)
    m.add("leverKdDefaultTuned", lev["kd_best"], E9, 2)
    m.add("leverOctreeDefaultPrimitive", lev["ot_l1"], E9, 2)
    m.add("leverKdDefaultPrimitive", lev["kd_l1"], E9, 2)
    m.add("leverPrimitiveTuned", lev["l1_l2"], E9, 2)

    # ---- other regimes (T4b)
    work = tables.workload_stats()
    m.add("nWorkloads", len(work), "workload_sweep_makinon2")
    m.add("workloadNestedSingleMax", max(geomean(v) for v in work.values()), "workload_sweep_makinon2")
    m.add("workloadNestedSingleMin", min(geomean(v) for v in work.values()), "workload_sweep_makinon2")
    gpu = tables.gpu_stats()
    m.add("gpuNestedSingleTotal", geomean(gpu["total"]), "gpu_normals_makinon2/confirm, fastest round per schema")
    m.add("gpuNestedSingleKernel", geomean(gpu["kernel"]), "gpu_normals_makinon2/confirm")
    m.add("oursOverCudaKdTree", geomean(gpu["ours_over_cudakdtree"]), "gpu_normals_makinon2/confirm: our best schema / cudaKDTree, end to end", 2)
    m.add("nGpuRows", len(gpu["total"]), "gpu_normals_makinon2/confirm")
    costs = tables.node_cost_stats()
    for cost, name in ((0, "Zero"), (100, "HundredNs"), (1000, "OneUs"), (10000, "TenUs")):
        if cost in costs:
            ratios, adopted, n = costs[cost]
            m.add("nodeCostNestedSingle" + name, geomean(ratios), f"node_cost*_makinon2/d{cost}")
            m.add("nodeCostNestedPicked" + name, adopted, f"node_cost*_makinon2/d{cost}")

    # ---- the rule (T6)
    rule = tables.rule_stats(tables.rule_results())
    inst, timed = "avg_latency_ms", "latency_repeat_mean_ms"
    m.add("ruleRegret", rule["R3a feature tree, depth 1"][inst]["geo"], "rule_fit, instrumented column")
    m.add("ruleRegretTimed", rule["R3a feature tree, depth 1"][timed]["geo"], "rule_fit, timed-passes column")
    m.add("ruleRegretP", rule["R3a feature tree, depth 1"][inst]["p90"], "rule_fit, p90 instrumented", 2)
    m.add("ruleCellsOver", rule["R3a feature tree, depth 1"][inst]["over"], "rule_fit: cells with regret > 1.15, instrumented")
    m.add("globalSchemaRegret", rule["R1 one global schema"][inst]["geo"], "rule_fit")
    m.add("groupSchemaRegret", rule["R2 per acquisition group"][inst]["geo"], "rule_fit")
    m.add("octreeDefaultRegret", rule["R0 octree_default"][inst]["geo"], "rule_fit")
    m.add("octreeDefaultRegretTimed", rule["R0 octree_default"][timed]["geo"], "rule_fit")
    m.add("kdDefaultRegret", rule["R0 kdtree_default"][inst]["geo"], "rule_fit")
    m.add("floorRegret", rule["floor: best grid schema per cell"][inst]["geo"], "rule_fit")
    capture = [1 - math.log(rule["R3a feature tree, depth 1"][c]["geo"]) / math.log(rule["R0 octree_default"][c]["geo"]) for c in (inst, timed)]
    m.add("ruleCapturePct", round(min(capture) * 100), "rule_fit: share of log(default -> oracle) closed by the rule, the lower of the two columns", 0)
    m.add("nRuleCells", rule["R3a feature tree, depth 1"][inst]["n"], "rule_fit: searched cells with the 55 tuned singles on the opt half")

    # ---- libraries (T7)
    summary, lib_cells = tables.library_summary()
    m.add("nLibraryCells", len(lib_cells), "external_makinon2 (+ external_1B_makinon2)")
    g = lambda label, setting, metric: geomean(summary.get((label, setting, metric), []))
    m.add("oursOctreeDefaultOverPick", g("ours, octree default", "default", "all"), "external_report medians", 2)
    m.add("oursKdDefaultOverPick", g("ours, kd-tree default", "default", "all"), "external_report medians", 2)
    m.add("ruleOverPick", g("ours, rule pick", "default", "all"), "external_report medians", 2)
    m.add("nanoflannDefaultAll", g("nanoflann kd", "default", "all"), "external_makinon2", 2)
    m.add("nanoflannDefaultRadius", g("nanoflann kd", "default", "radius"), "external_makinon2", 2)
    m.add("nanoflannDefaultKnn", g("nanoflann kd", "default", "knn"), "external_makinon2", 2)
    m.add("nanoflannBestAll", g("nanoflann kd", "best", "all"), "external_makinon2", 2)
    m.add("nanoflannBestKnn", g("nanoflann kd", "best", "knn"), "external_makinon2", 2)
    m.add("nanoflannBestRadius", g("nanoflann kd", "best", "radius"), "external_makinon2", 2)
    m.add("pclKdAll", g("PCL KdTreeFLANN", "default", "all"), "external_makinon2", 2)
    m.add("pclOctreeAll", g("PCL octree", "best", "all"), "external_makinon2", 2)
    m.add("openThreeDAll", g("Open3D KDTreeFlann", "default", "all"), "external_makinon2", 2)
    m.add("unibnRadius", g("unibn octree", "default", "radius"), "external_makinon2", 2)
    m.add("boostAll", g("Boost R-tree", "default", "all"), "external_makinon2", 2)
    levers = [g(l, "lever", "all") or g(l, "lever", "radius") for l in ("nanoflann kd", "unibn octree", "Boost R-tree")]
    m.add("libraryLeverMin", min(finite(levers)), "external_makinon2: each library's default / its best setting", 2)
    m.add("libraryLeverMax", max(finite(levers)), "external_makinon2", 2)

    # ---- layout and depth cost (T8, Fig. 5)
    rows, depth = tables.layout_stats()
    L = "cache_accounting_makinon2"
    m.add("hugeSinglesHundredM", geomean(rows[("huge", "100M singles")]), L)
    m.add("hugeBigPools", geomean(rows[("huge", "100M, pool $\\geq$ 64 MB")]), L)
    m.add("hugeSmallPools", geomean(rows[("huge", "100M, pool $<$ 64 MB")]), L)
    m.add("hugeSinglesTwentyFiveM", geomean(rows[("huge", "25M singles")]), L)
    m.add("hugeMin", min(rows[("huge", "100M singles")]), L)
    m.add("prefetchSinglesHundredM", geomean(rows[("prefetch", "100M singles")]), L)
    m.add("prefetchHGrid", geomean(rows[("prefetch", "HGrid rows")]), L)
    m.add("prefetchMin", min(rows[("prefetch", "100M singles")]), L)
    m.add("bothSinglesHundredM", geomean(rows[("both", "100M singles")]), L)
    m.add("hugeNestedHundredM", geomean(rows[("huge", "100M nested")]), L)
    pick, nested = depth[("domfountain_100M", "kd48l128sm")], depth[("domfountain_100M", "ot2_kd48l128sm")]
    band_ns = lambda d, band: d["bands"][band][1] / d["bands"][band][0] if d["bands"][band][0] else float("nan")
    m.add("visitCostTopNs", round(band_ns(pick, "top")), f"{L}: DomFountain 100M kd48l128sm, depths 0-7, instrumented pass", 0)
    m.add("visitCostMiddleNs", round(band_ns(pick, "middle")), f"{L}: DomFountain 100M kd48l128sm, middle depths", 0)
    m.add("visitCostLeafNs", round(band_ns(pick, "bottom")), f"{L}: DomFountain 100M kd48l128sm, deepest four levels", 0)
    m.add("topLevelsShareDomFountain", round(100 * pick["bands"]["top"][1] / sum(b[1] for b in pick["bands"].values())), f"{L}: share of the instrumented query time in depths 0-7", 0)
    lille_p, lille_n = depth[("lille_100M", "kd48l64sm")], depth[("lille_100M", "ot2_kd48l128sm")]
    saved = []
    for p, n in ((pick, nested), (lille_p, lille_n)):
        dv = p["visits"] - n["visits"]
        dt = sum(b[1] for b in p["bands"].values()) - sum(b[1] for b in n["bands"].values())
        if dv > 0:
            saved.append(dt / dv)
    m.add("savedVisitCostNs", round(geomean(saved)), f"{L}: (instrumented ns of the pick - of the nested finalist) / visits saved, DomFountain and Lille 100M", 0)
    m.add("domFountainPoolMb", round(pick["pool_mb"]), f"{L}: kd48l128sm nodes x 32 B", 0)

    # ---- out of core (T5)
    o = ooc.ooc_stats()
    m.add("oocInRamAdopted", o["in_ram"][0], "ooc.py: stage B adoptions with the hierarchy in RAM")
    m.add("oocInRamCells", o["in_ram"][1], "ooc.py")
    m.add("oocPagedAdopted", o["paged"][0], "ooc.py: stage B adoptions with the hierarchy paged")
    m.add("oocPagedCells", o["paged"][1], "ooc.py")
    for label, name in (("no cache", "NoCache"), ("1% LRU", "Cache")):
        prices = [v for p in o["price"].values() for v in p["prices"].get(label, [])]
        if prices:
            m.add("pagedOverPinnedMin" + name, min(prices), f"ooc.py: best schema with the hierarchy paged / pinned, {label}, four device models", 1)
            m.add("pagedOverPinnedMax" + name, max(prices), "ooc.py", 1)

    # ---- composition side effects: build time and node count of the nested finalists
    m.add("nestedBuildRatio", geomean([c["build_ratio"] for c in cells]), E9 + ": build time of the best nested finalist / best single (held-out structure run)")
    m.add("nestedNodesRatio", geomean([c["nodes_ratio"] for c in cells]), E9 + ": nodes of the best nested finalist / best single")

    # ---- noise floor of the rule's oracle, and the counts-do-not-predict-time example (2p)
    import rule_fit
    rcells = tables.rule_results()["avg_latency_ms"][0]
    cv_med, cv_p90 = rule_fit.oracle_cv(rcells)
    m.add("oracleRepeatCvPct", round(100 * cv_med, 1), "rule_fit: median repeat CV of the oracle_opt rows", 1)
    example = next((c for c in rcells if c["cell"] == "domfountain_25M"), None)
    if example:
        octree, oracle = example["rows"].get("octree_default"), example["rows"][example["oracle"]]
        if octree:
            m.add("exampleOracleName", example["oracle"].replace("tuned_", "").replace("_", "\\_"), "rule_fit: DomFountain 25M oracle_opt")
            for label, row in (("Octree", octree), ("Oracle", oracle)):
                m.add(f"example{label}Visits", round(float(row["avg_visited_nodes"]), 1), "e9 opt half, DomFountain 25M, instrumented", 1)
                m.add(f"example{label}Points", round(float(row["avg_tested_points"])), "e9 opt half, DomFountain 25M", 0)
                m.add(f"example{label}Us", round(float(row["avg_latency_ms"]) * 1e3, 2), "e9 opt half, DomFountain 25M", 2)
                m.add(f"example{label}Nodes", round(int(row["num_nodes"]) / 1e6, 2), "e9 opt half, DomFountain 25M, millions of nodes", 2)

    # ---- interleaving is TLB reach; the null pair gauges schema-against-schema noise (2s follow-ups)
    inter = tables.interleave_stats()
    for (arm, name), v in inter["alone_over_interleaved"].items():
        m.add("aloneOverInterleaved" + ("Kd" if name.startswith("kd") else "Ot") + arm.capitalize(), v, f"{L}/interleave: {name} alone / interleaved, {arm} arm")
    for name, v in inter["sanandreas_huge_over_base"].items():
        m.add("hugeAlone" + ("Qt" if name.startswith("qt") else "Kd"), v, f"{L}/interleave: SanAndreas 100M {name} alone, huge / base")
    pair = tables.null_pair()
    if pair:
        m.add("nullPairMin", min(pair.values()), f"{L}: SanAndreas 100M qt9_ot24l32 / qt24l32 (the same tree) across arms", 2)
        m.add("nullPairMax", max(pair.values()), f"{L}: the same", 2)

    # ---- out of core: the ISPRS schemas and the hierarchy's size
    isprs = ooc.isprs_stats()
    if isprs:
        m.add("isprsOverLeafMatchedMin", min(v["matched"] for v in isprs.values()), "ooc.py: ISPRS nested schemas / leaf-matched depth-10 singles, SATA model, geomean per regime", 2)
        m.add("isprsOverLeafMatchedMax", max(v["matched"] for v in isprs.values()), "ooc.py", 2)
        m.add("isprsOverBestSingleMin", min(v["best"] for v in isprs.values()), "ooc.py: ISPRS nested schemas / the regime's best single, SATA model", 1)
        m.add("isprsOverBestSingleMax", max(v["best"] for v in isprs.values()), "ooc.py", 1)
    sizes = ooc.hierarchy_mb()
    if sizes:
        m.add("hierarchyMbMin", round(min(sizes), 1), "ooc.py: node pool of the best single with leaves on the device (0:memory), over the four device models, MB", 1)
        m.add("hierarchyMbMax", round(max(sizes)), "ooc.py", 0)

    # ---- the 1B anchor against nanoflann (battery 1A, 2t)
    anchor, anchor_cells = tables.library_summary(tables.ANCHOR_FOLDERS)
    if anchor_cells:
        A = "external_1B_makinon2: three 1B clouds, sequential protocol, 1000 queries"
        a = lambda label, setting, metric: geomean(anchor.get((label, setting, metric), []))
        m.add("nAnchorCells", len(anchor_cells), A)
        m.add("anchorNanoflannDefaultAll", a("nanoflann kd", "default", "all"), A + ": nanoflann leaf 10 / the selected schema", 2)
        m.add("anchorNanoflannDefaultMin", min(anchor[("nanoflann kd", "default", "all")]), A, 2)
        m.add("anchorNanoflannDefaultMax", max(anchor[("nanoflann kd", "default", "all")]), A, 2)
        m.add("anchorNanoflannBestAll", a("nanoflann kd", "best", "all"), A, 2)
        m.add("anchorNanoflannDefaultKnn", a("nanoflann kd", "default", "knn"), A, 2)
        m.add("anchorNanoflannDefaultRadius", a("nanoflann kd", "default", "radius"), A, 2)
        m.add("anchorOursOctreeDefault", a("ours, octree default", "default", "all"), A, 2)
        m.add("anchorOursKdDefault", a("ours, kd-tree default", "default", "all"), A, 2)
        m.add("anchorRuleOverPick", a("ours, rule pick", "default", "all"), A, 2)

    # ---- the headline panel at 100M, sequential, both page modes (battery 1C, 2u)
    import leaf_lever
    panel_cells, ratios = leaf_lever.panel_stats()
    if panel_cells:
        P = "leaf_pages_makinon2/panel: nine 100M cells, sequential protocol, median of 3 rounds"
        m.add("nPanelCells", len(panel_cells), P)
        for key, name in (("octree default / pick", "OctreeDefault"), ("kd default / pick", "KdDefault"), ("rule / pick", "Rule"),
                          ("pick / best single", "PickOracle"), ("nested / best single", "NestedSingle"), ("stage B winner / pick", "StageB")):
            for arm in ("base", "huge"):
                values = ratios.get((arm, key), [])
                if values:
                    m.add(f"panel{name}{arm.capitalize()}", geomean(values), f"{P}, {arm} arm: {key}")
        m.add("panelNestedSingleBaseMax", max(ratios[("base", "nested / best single")]), P, 2)
        m.add("panelHugeOverBaseSingles", geomean(ratios[("huge / base", "single")]), P + ": huge / base over the single schemas")
        m.add("panelHugeOverBaseSinglesMin", min(ratios[("huge / base", "single")]), P, 2)
        m.add("panelHugeOverBaseSinglesMax", max(ratios[("huge / base", "single")]), P, 2)
        m.add("panelHugeOverBaseNested", geomean(ratios[("huge / base", "nested")]), P)

    # ---- the leaf lever in both page modes (battery 1B, 2u)
    leaf_cells, leaf, shift = leaf_lever.leaf_stats()
    if leaf_cells:
        Lf = "leaf_pages_makinon2/leaf: nine 100M cells, leaves 16-1024, sequential, median of 3 rounds"
        words = {"base": "Base", "huge": "Huge"}
        for prim, pname in (("kd48sm", "Kd"), ("qt24tc", "Qt")):
            for arm in ("base", "huge"):
                s = leaf.get((arm, prim))
                if s:
                    m.add(f"leaf{pname}OverThirtyTwo{words[arm]}", geomean(s["over32"]), f"{Lf}: {prim} leaf 32 / best leaf, {arm}")
                    m.add(f"leaf{pname}OverHundredTwentyEight{words[arm]}", geomean(s["over128"]), f"{Lf}: {prim} leaf 128 / best leaf, {arm}")
        worst = [v for s in leaf.values() for v in s["worst"]]
        m.add("leafWorstMin", min(worst), Lf + ": worst leaf / best leaf", 2)
        m.add("leafWorstMax", max(worst), Lf, 2)
        m.add("leafPairs", len(shift), Lf + ": (cell, primitive) pairs with both page modes")
        m.add("leafShiftPairs", sum(1 for _, _, b, h in shift if h < b), Lf + ": pairs whose best leaf is smaller on huge pages")
        m.add("leafKdShiftPairs", sum(1 for _, p, b, h in shift if p == "kd48sm" and h < b), Lf + ": kd-tree pairs whose best leaf is smaller on huge pages")

    n = m.write()
    print(f"{n} macros written to paper/generated/numbers.tex")


if __name__ == "__main__":
    main()

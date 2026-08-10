"""Generate tuned single-primitive schema grids (rigor R2).

Emits single-block schema JSONs per primitive so the "best single" baseline gets a
real tuning budget instead of the hardcoded `*_default` configs.

Depth is DERIVED from leaf capacity and cloud size, not drawn independently. The builder
stops splitting on depth before it consults capacity (PointSpatialIndex.cpp:888 precedes
:894), so a primitive with branching factor b holds at most b**depth leaves regardless of
what `leafCapacity` asks for. The previous version of this script drew both axes freely with
depth <= 12, which is ample for an 8-ary octree but leaves a BINARY kd-tree with only 4096
leaves: on a 5M cloud that is 1221 points per leaf against a requested capacity of 32, and
every capacity value below ~1024 produces the byte-identical tree. Nine of twelve emitted kd
schemas were inert, and in the small grid `tuned_kd12l32` was indistinguishable from
`kdtree_default`, so the kd arm of the tuned-singles baseline was not tuning anything.

Pass --points for the largest cell the grid will be evaluated on (default 25M).

Per primitive it deterministically samples --per-primitive configs (seed 7919) and
always includes the primitive's default parameters, so the tuned-singles arm
strictly dominates the old default baseline. Evaluate the emitted index with:

  --mode schema-search --flat-search --schemas <;-joined index> --generate-schemas 0
  --no-baselines --input-trace <trace_opt.csv> --no-score-cache

Budget note: the corrected 5M battery GA evaluated 111 candidates
(results/eval_traces/sanandreas_5m.csv: 48 generated + 57 evolved + 6 baselines), so
the default 12 per primitive (~107 total; RegularGrid caps at its 11 combos) is
budget-matched to that cell. Adjust --per-primitive for differently sized cells.

Usage:
  python scripts/make_single_grids.py                       # all primitives, 12 each
  python scripts/make_single_grids.py --per-primitive 6 --primitives KDTree Octree
"""

import argparse
import json
import math
import random
from pathlib import Path

SEED = 7919
CAPACITIES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# type -> (abbrev, UNUSED, axis-policy choices or None, default (depth, capacity, axis),
#          branching factor)
#
# The branching factor matters because the builder stops on depth BEFORE it consults leaf
# capacity (PointSpatialIndex.cpp:888 precedes :894). A primitive that splits into `b` children
# can hold at most b**depth leaves, so on a cloud of N points the shallowest depth that lets
# `leafCapacity` actually bind is ceil(log_b(N / capacity)). Drawing depth and capacity as
# independent axes silently produces schemas whose capacity is inert: the historical grid capped
# every primitive at depth 12, which is ample for an 8-ary octree (8**12 leaves) but leaves a
# BINARY kd-tree with only 4096 leaves -- 1221 points per leaf on a 5M cloud against a requested
# capacity of 32, a 38x miss, and identical trees for every capacity value. See testing.md.
#
# Branching factors are taken from the C++ builder, NOT from intuition about the primitive:
# gridSubdivisionsForLevel (PointSpatialIndex.cpp:130-136) splits RegularGrid 3x3x3 = 27 ways
# and HGrid 4x4x4 = 64 ways, and childBounds/locateChild (:906, :963) recurse on grids exactly
# like the trees. Treating RegularGrid as a single-level 1-way primitive reproduced the same
# capacity-is-inert bug for every tuned_rg* schema.
#
# The second tuple element is vestigial: depth is now derived, never drawn, so the old per-
# primitive depth lists are unused. Kept as None to preserve the tuple shape.
PRIMITIVES = {
    "QuadTree": ("qt", None, ["xy", "xz", "yz", "ignore_shortest"], (8, 32, "xy"), 4),
    "Octree": ("ot", None, None, (8, 32, None), 8),
    "KDTree": ("kd", None, ["median_longest_axis", "round_robin", "center_longest_axis"], (12, 32, "median_longest_axis"), 2),
    "BVH": ("bvh", None, None, (12, 32, "center_longest_axis"), 2),
    "LBVH": ("lbvh", None, None, (12, 32, "center_longest_axis"), 2),
    "BIH": ("bih", None, ["median_longest_axis", "round_robin", "center_longest_axis"], (12, 32, "median_longest_axis"), 2),
    "KarrasOctree": ("kot", None, None, (8, 32, None), 8),
    "RegularGrid": ("rg", None, None, (1, 32, None), 27),
    "HGrid": ("hg", None, None, (4, 32, None), 64),
}

# Depth ceiling per primitive. Derived depth is clamped here, and a schema whose capacity
# cannot bind within the ceiling is reported rather than emitted silently. Grids branch so
# widely that an unclamped derivation asks for millions of cells; trees are cheap by
# comparison, so they get the generous default.
MAX_DEPTH_BY_PRIMITIVE = {
    "RegularGrid": 5,
    "HGrid": 4,
}

AXIS_SHORT = {
    "xy": "xy", "xz": "xz", "yz": "yz", "ignore_shortest": "is",
    "median_longest_axis": "ml", "round_robin": "rr", "center_longest_axis": "cl",
}


def required_depth(prim: str, capacity: int, points: int) -> int:
    """Shallowest depth at which `capacity` can bind, given the primitive's branching factor.

    Returns 1 for RegularGrid (single level by construction) and for cases where the cloud
    already fits in one leaf.
    """
    branching = PRIMITIVES[prim][4]
    if branching <= 1 or points <= capacity:
        return 1
    return max(1, math.ceil(math.log(points / capacity, branching)))


def schema_json(prim: str, depth: int, capacity: int, axis: str | None) -> tuple[str, dict]:
    abbrev, _, _, default, _ = PRIMITIVES[prim]
    min_split = max(2, capacity // 4)
    name = f"tuned_{abbrev}{depth}l{capacity}"
    if axis is not None and axis != default[2]:
        name += f"ap{AXIS_SHORT[axis]}"
    level = {
        "type": prim,
        "numLevels": depth,
        "leafCapacity": capacity,
        "minPointsToSplit": min_split,
    }
    if axis is not None:
        level["axisPolicy"] = axis
    schema = {
        "name": name,
        "levels": [level],
        "buildPolicy": {
            "maxDepth": depth,
            "leafCapacity": capacity,
            "minPointsToSplit": min_split,
            "collapseSingleChild": True,
            "removeEmptyNodes": True,
            "allowOverlapDuplication": False,
        },
    }
    return name, schema


def grid_for(prim: str, per_primitive: int, rng: random.Random, points: int, max_depth: int,
             report: list | None = None) -> list:
    """Sample (depth, capacity, axis) combinations whose capacity actually binds.

    Depth is no longer an independent axis. For each capacity we take the depth that just
    achieves it on a cloud of `points`, plus one deeper rung where that fits under `max_depth`,
    so the arm still explores over-splitting without wasting evaluations on schemas that are
    duplicates of each other. Capacities needing more depth than the primitive's ceiling are
    dropped and appended to `report` so the caller can say so out loud -- silently emitting
    them is what produced inert schemas before.
    """
    _, _, axes, default, _ = PRIMITIVES[prim]
    axis_choices = axes if axes else [default[2]]

    full = []
    for capacity in CAPACITIES:
        needed = required_depth(prim, capacity, points)
        if needed > max_depth:
            if report is not None:
                report.append((capacity, needed))
            continue  # capacity cannot bind within the ceiling; emitting it would be inert
        for depth in {needed, min(needed + 1, max_depth)}:
            for axis in axis_choices:
                full.append((depth, capacity, axis))
    if not full:
        # Every capacity needs more depth than allowed (grids on huge clouds). Emit the
        # deepest legal tree at the largest capacity it CAN honour, so the arm still has a
        # representative for this primitive instead of vanishing from the baseline.
        best = max((c for c in CAPACITIES if required_depth(prim, c, points) <= max_depth),
                   default=max(CAPACITIES))
        full = [(max_depth, best, axis) for axis in axis_choices]

    # The hardcoded default is kept so the tuned arm strictly dominates the old baseline, but its
    # depth is raised to whatever its capacity actually needs -- otherwise the arm reintroduces
    # exactly the depth-capped schema this function exists to avoid.
    default_depth = max(default[0], required_depth(prim, default[1], points))
    default_combo = (min(default_depth, max_depth), default[1], default[2])
    if default_combo not in full:
        full.append(default_combo)

    picked = rng.sample(full, min(per_primitive, len(full)))
    if default_combo not in picked:
        picked[0] = default_combo  # tuned arm always contains the old default
    # dedupe while keeping order (sample can't repeat, but the swap above could)
    seen, combos = set(), []
    for combo in picked:
        if combo not in seen:
            seen.add(combo)
            combos.append(combo)
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("configs/schemas/tuned_singles"))
    parser.add_argument("--per-primitive", type=int, default=12)
    parser.add_argument("--primitives", nargs="*", default=list(PRIMITIVES),
                        help=f"subset of: {', '.join(PRIMITIVES)}")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--points", type=int, default=25_000_000,
                        help="Cloud size the grid is being generated for. Depth is derived from "
                             "this and the leaf capacity, because the builder stops on depth "
                             "before it consults capacity. Use the LARGEST cell the grid will be "
                             "evaluated on; a depth sufficient for N is sufficient for anything "
                             "smaller. Default 25M (the matrix's largest complete rung).")
    parser.add_argument("--max-depth", type=int, default=24,
                        help="Hard ceiling on derived depth. The historical value was 12, which "
                             "silently capped every binary primitive; 24 lets a kd-tree reach "
                             "capacity 32 at ~500M points.")
    args = parser.parse_args()

    unknown = [p for p in args.primitives if p not in PRIMITIVES]
    if unknown:
        raise SystemExit(f"unknown primitives: {unknown}")

    rng = random.Random(args.seed)
    for prim in args.primitives:
        prim_dir = args.out / PRIMITIVES[prim][0]
        prim_dir.mkdir(parents=True, exist_ok=True)
        # Regenerating with different parameters emits different names, so stale schemas from a
        # previous run would otherwise accumulate beside the new ones and a stray --schemas glob
        # could pick them up.
        for stale in prim_dir.glob("tuned_*.json"):
            stale.unlink()

        ceiling = min(args.max_depth, MAX_DEPTH_BY_PRIMITIVE.get(prim, args.max_depth))
        unreachable: list[tuple[int, int]] = []
        combos = grid_for(prim, args.per_primitive, rng, args.points, ceiling, unreachable)
        for depth, capacity, axis in combos:
            name, schema = schema_json(prim, depth, capacity, axis)
            (prim_dir / f"{name}.json").write_text(json.dumps(schema, indent=2) + "\n")

        emitted = len(list(prim_dir.glob("tuned_*.json")))
        note = f" (depth ceiling {ceiling})" if ceiling != args.max_depth else ""
        print(f"{prim}: {emitted} schemas -> {prim_dir.as_posix()}{note}")
        if unreachable:
            caps = ", ".join(str(c) for c, _ in sorted(unreachable))
            deepest = max(d for _, d in unreachable)
            print(f"  skipped capacities {caps}: would need depth up to {deepest} at "
                  f"{args.points:,} points, above this primitive's ceiling of {ceiling}. "
                  f"Emitting them would produce schemas whose leafCapacity never binds.")

    # Built from what is actually on disk, not from this invocation's --primitives. Running with
    # a subset (which the usage examples encourage) used to overwrite index.txt with only that
    # subset while leaving the other primitives' JSON in place, silently shrinking the baseline
    # arm the battery consumes.
    index = sorted(p.as_posix() for p in args.out.glob("*/tuned_*.json"))
    index_path = args.out / "index.txt"
    index_path.write_text("\n".join(index) + "\n")
    print(f"\n{len(index)} schemas on disk; index: {index_path.as_posix()}")
    print("PowerShell:  $tuned = (Get-Content " + index_path.as_posix() + ") -join ';'")


if __name__ == "__main__":
    main()

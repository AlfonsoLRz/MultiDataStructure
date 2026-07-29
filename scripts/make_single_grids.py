"""Generate tuned single-primitive schema grids (rigor R2).

Emits single-block schema JSONs per primitive so the "best single" baseline gets a
real tuning budget instead of the hardcoded `*_default` configs. The grid mirrors
the GA generator's design space (SchemaGenerationOptions: power-of-two leaf
capacities in [32, 32768], depth <= 12, minPointsToSplit = leafCapacity/4), so
"tuned single" and "nested" search the same space restricted to one block.

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
import random
from pathlib import Path

SEED = 7919
CAPACITIES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# type -> (abbrev, depth choices, axis-policy choices or None, default (depth, capacity, axis))
PRIMITIVES = {
    "QuadTree": ("qt", [4, 6, 8, 10, 12], ["xy", "xz", "yz", "ignore_shortest"], (8, 32, "xy")),
    "Octree": ("ot", [4, 6, 8, 10, 12], None, (8, 32, None)),
    "KDTree": ("kd", [6, 8, 10, 12], ["median_longest_axis", "round_robin", "center_longest_axis"], (12, 32, "median_longest_axis")),
    "BVH": ("bvh", [6, 8, 10, 12], None, (12, 32, "center_longest_axis")),
    "LBVH": ("lbvh", [6, 8, 10, 12], None, (12, 32, "center_longest_axis")),
    "BIH": ("bih", [6, 8, 10, 12], ["median_longest_axis", "round_robin", "center_longest_axis"], (12, 32, "median_longest_axis")),
    "KarrasOctree": ("kot", [4, 6, 8, 10, 12], None, (8, 32, None)),
    "RegularGrid": ("rg", [1], None, (1, 32, None)),
    "HGrid": ("hg", [2, 3, 4, 5, 6], None, (4, 32, None)),
}

AXIS_SHORT = {
    "xy": "xy", "xz": "xz", "yz": "yz", "ignore_shortest": "is",
    "median_longest_axis": "ml", "round_robin": "rr", "center_longest_axis": "cl",
}


def schema_json(prim: str, depth: int, capacity: int, axis: str | None) -> tuple[str, dict]:
    abbrev, _, _, default = PRIMITIVES[prim]
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


def grid_for(prim: str, per_primitive: int, rng: random.Random) -> list:
    _, depths, axes, default = PRIMITIVES[prim]
    axis_choices = axes if axes else [default[2]]
    full = [(d, c, a) for d in depths for c in CAPACITIES for a in axis_choices]
    default_combo = (default[0], default[1], default[2])
    picked = rng.sample(full, min(per_primitive, len(full)))
    if default_combo in full and default_combo not in picked:
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
    args = parser.parse_args()

    unknown = [p for p in args.primitives if p not in PRIMITIVES]
    if unknown:
        raise SystemExit(f"unknown primitives: {unknown}")

    rng = random.Random(args.seed)
    index = []
    for prim in args.primitives:
        prim_dir = args.out / PRIMITIVES[prim][0]
        prim_dir.mkdir(parents=True, exist_ok=True)
        for depth, capacity, axis in grid_for(prim, args.per_primitive, rng):
            name, schema = schema_json(prim, depth, capacity, axis)
            path = prim_dir / f"{name}.json"
            path.write_text(json.dumps(schema, indent=2) + "\n")
            index.append(path.as_posix())
        print(f"{prim}: {len(list(prim_dir.glob('tuned_*.json')))} schemas -> {prim_dir.as_posix()}")

    index_path = args.out / "index.txt"
    index_path.write_text("\n".join(index) + "\n")
    print(f"\n{len(index)} schemas total; index: {index_path.as_posix()}")
    print("PowerShell:  $tuned = (Get-Content " + index_path.as_posix() + ") -join ';'")


if __name__ == "__main__":
    main()

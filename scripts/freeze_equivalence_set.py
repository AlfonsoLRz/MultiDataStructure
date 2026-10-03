"""Freeze the schema set the engine-equivalence gate builds (configs/schemas/equivalence_set/).

The gate has to cover the schemas that actually matter (every searched winner) and the grammar's
corners (conditional blocks, adaptive leaf capacity, empty-node retention, every split policy).
Those live in results/, which is gitignored and pruned between batteries, so they are copied here
once, deduplicated by content, and committed. Re-running is deterministic given the same inputs;
it is not meant to be re-run casually, because changing the set changes what the goldens cover.

    python scripts/freeze_equivalence_set.py
"""

import hashlib
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "schemas" / "equivalence_set"
GENERATED = REPO / "results" / "generated_schemas"
BEST = REPO / "results" / "eval_traces" / "matrix" / "best_schemas"
TUNED = REPO / "configs" / "schemas" / "tuned_singles_small"


def edge_schemas():
    """Hand-written corners the searched sets rarely reach."""
    def single(name, level, **policy):
        build = {"maxDepth": level["numLevels"], "leafCapacity": level["leafCapacity"],
                 "minPointsToSplit": level.get("minPointsToSplit", 2), "collapseSingleChild": True,
                 "removeEmptyNodes": True, "allowOverlapDuplication": False}
        build.update(policy)
        return name, {"name": name, "levels": [level], "buildPolicy": build}

    return [
        single("edge_octree_keep_empty", {"type": "Octree", "numLevels": 6, "leafCapacity": 16},
               removeEmptyNodes=False, collapseSingleChild=False),
        single("edge_kd_round_robin", {"type": "KDTree", "numLevels": 18, "leafCapacity": 64, "axisPolicy": "round_robin"}),
        single("edge_kd_center", {"type": "KDTree", "numLevels": 18, "leafCapacity": 64, "axisPolicy": "center_longest_axis"}),
        single("edge_kd_median", {"type": "KDTree", "numLevels": 18, "leafCapacity": 64, "axisPolicy": "median_longest_axis"}),
        single("edge_quadtree_ignore_shortest", {"type": "QuadTree", "numLevels": 10, "leafCapacity": 32, "axisPolicy": "ignore_shortest"}),
        single("edge_quadtree_yz", {"type": "QuadTree", "numLevels": 10, "leafCapacity": 32, "axisPolicy": "yz"}),
        single("edge_regular_grid_keep_empty", {"type": "RegularGrid", "numLevels": 3, "leafCapacity": 32},
               removeEmptyNodes=False, collapseSingleChild=False),
        single("edge_hgrid", {"type": "HGrid", "numLevels": 3, "leafCapacity": 32}),
        single("edge_leaf_capacity_one", {"type": "Octree", "numLevels": 12, "leafCapacity": 1, "minPointsToSplit": 2}),
        single("edge_depth_capped", {"type": "Octree", "numLevels": 3, "leafCapacity": 8}),
    ]


def content_key(schema):
    stripped = {k: v for k, v in schema.items() if k not in ("name", "_comment")}
    return hashlib.sha1(json.dumps(stripped, sort_keys=True).encode()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()

    chosen = {}  # content key -> (name, schema, provenance)

    def add(name, schema, provenance):
        key = content_key(schema)
        if key not in chosen:
            schema = dict(schema)
            schema["name"] = name
            chosen[key] = (name, schema, provenance)

    # 1. Every searched winner, one per cell. Skip the mangled "<cell> GA (opt)_ga" directories
    #    left by the pre-1d6d1a4 PowerShell scoping bug; their clean twins carry the same schema.
    for cell_dir in sorted(p for p in BEST.iterdir() if p.is_dir() and " " not in p.name):
        for path in sorted(cell_dir.glob("*_best_schema.json")):
            cell = cell_dir.name.removesuffix("_ga")
            add(f"winner_{cell}", json.loads(path.read_text()), str(path.relative_to(REPO)))

    # 2. A seeded sample of the grammar's conditional and adaptive corners.
    rng = random.Random(20261003)
    generated = sorted(GENERATED.glob("*.json"))
    conditional = [p for p in generated if '"condition"' in p.read_text()]
    adaptive = [p for p in generated if '"adaptiveLeafCapacity"' in p.read_text()]
    for path in rng.sample(conditional, min(16, len(conditional))) + rng.sample(adaptive, min(4, len(adaptive))):
        add(f"gen_{path.stem}", json.loads(path.read_text()), str(path.relative_to(REPO)))

    # 3. Tuned singles: two per primitive family (smallest and largest leaf by name order).
    for family in sorted(p for p in TUNED.iterdir() if p.is_dir()):
        files = sorted(family.glob("*.json"))
        for path in {files[0], files[-1]} if files else ():
            add(path.stem, json.loads(path.read_text()), str(path.relative_to(REPO)))
    for name in ("tuned_kd19l128", "tuned_kd16l1024", "tuned_qt10l32", "tuned_ot7l32"):
        hits = list(TUNED.rglob(f"{name}.json"))
        if hits:
            add(name, json.loads(hits[0].read_text()), str(hits[0].relative_to(REPO)))

    # 4. Edge cases.
    for name, schema in edge_schemas():
        add(name, schema, "scripts/freeze_equivalence_set.py")

    index = []
    for name, schema, provenance in sorted(chosen.values(), key=lambda item: item[0]):
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        (OUT / f"{safe}.json").write_text(json.dumps(schema, indent=2) + "\n")
        index.append(f"{safe}.json\t{provenance}")
    (OUT / "index.txt").write_text("\n".join(index) + "\n")
    print(f"froze {len(index)} schemas into {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

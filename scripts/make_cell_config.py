"""Turn the curated benchmark's manifest into the cell list the battery runs.

The battery scripts used to carry one hardcoded absolute cloud path per cell,
which is why re-pointing them at a new dataset meant editing PowerShell. This
reads `curated_manifest.json` and writes `configs/datasets/curated_cells.json`,
the single place that says which (scene, size) pairs are in the experiment and
what budget each one gets.

    python scripts/make_cell_config.py                    # tiers A and B
    python scripts/make_cell_config.py --profile all
    python scripts/make_cell_config.py --profile C --out configs/datasets/scale_ladder.json

Cells whose points are not all measured are refused unless --allow-synthetic is
passed, and every refusal is printed. Ten of the thirteen scenes are upsampled
above their native count, and a tier drawn from an upsampled master is not the
same object as a tier drawn from the scan: publishing a morphology comparison
over interpolated points would be measuring the interpolator.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_MANIFEST = Path(r"D:\Datasets\Point Clouds\CuratedDataset\curated_manifest.json")

# Scene -> morphology. The manifest names scenes, not shapes; this is the axis the
# heatmap's rows are labelled with, so it is a deliberate editorial choice and not
# something to infer from the data.
MORPHOLOGY = {
    "SanAndreas": "aerial_terrain",
    "Aneto": "aerial_terrain",
    "Waikato_Hamilton": "aerial_urban",
    "Barcelona": "aerial_urban",
    "Toronto": "mls_street",
    "Lille": "mls_street",
    "ParisLux": "mls_street",
    "GaliciaForest": "forest_archipelago",
    "GaliciaForest/plots/ES_05": "forest_plot",
    "Alhambra": "heritage",
    "Cathedral": "heritage",
    "DomFountain": "tls_street",
    "OttobrunnCampus": "tls_street",
    "Neugasse": "tls_street",
    "Retiro": "vegetated_urban",
    "SolarPlantation": "industrial",
    # Museum is the old `indoor_1M` stand-in (a 2.4M-point Sketchfab model). There
    # is no real indoor scan in the set, so it is labelled for what it is.
    "Museum": "small_object",
}

# Tier -> size labels. A is the morphology grid, B extends the scenes big enough
# to support it, C is the scale ladder on the three scenes that are natively
# billion-point.
TIERS = {
    "A": ["1M", "5M", "25M"],
    "B": ["50M", "100M"],
    "C": ["250M", "500M", "1B"],
}
TIER_C_SCENES = ["SanAndreas", "Waikato_Hamilton", "GaliciaForest"]

# size label -> (queries, generated schemas, GA generations, GA population).
# Above 100M the GA is not run at all: the winner is transferred from the largest
# searched rung and only re-measured, which is what the scale claim needs.
BUDGET = {
    "1M":   (3000, 24, 2, 16),
    "5M":   (3000, 24, 2, 16),
    "25M":  (2000, 24, 2, 12),
    "50M":  (1500, 20, 2, 12),
    "100M": (1000, 16, 2, 10),
    "250M": (600, 0, 0, 0),
    "500M": (400, 0, 0, 0),
    "1B":   (300, 0, 0, 0),
}

SEARCHED_TIERS = {"A", "B"}


def cell_name(scene: str, size_label: str) -> str:
    """`Galicia Forest/plots/ES_05` + `25M` -> `galicia_es05_25M`."""
    stem = scene
    plot = re.match(r"(.+)/plots/(.+)$", scene)
    if plot:
        stem = f"{plot.group(1)} {plot.group(2)}"
    stem = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_").lower()
    stem = stem.replace("galicia_forest", "galicia").replace("_es_", "_es")
    return f"{stem}_{size_label}"


def tier_of(size_label: str) -> str | None:
    for tier, labels in TIERS.items():
        if size_label in labels:
            return tier
    return None


def build_cells(manifest: dict, profile: list[str], allow_synthetic: bool,
                scenes: list[str] | None) -> tuple[list[dict], list[str]]:
    cells, refused = [], []
    by_size = {entry["label"]: entry["points"] for entry in manifest["sizes"]}

    for raw in manifest["cells"]:
        scene, size_label = raw["scene"], raw["size_label"]
        tier = tier_of(size_label)
        if tier is None or tier not in profile:
            continue
        if scene not in MORPHOLOGY:
            continue
        if scenes and scene not in scenes:
            continue
        if tier == "C" and scene not in TIER_C_SCENES:
            continue

        fraction = raw.get("synthetic_fraction")
        if fraction is None:
            refused.append(f"{cell_name(scene, size_label)}: manifest predates per-tier "
                           f"synthetic counts - rebuild with _tools/build.py")
            continue
        # Not exactly zero: Alhambra's native count is 224 points short of the 100M
        # rung, and refusing a 2e-6 fraction would silently drop the only heritage
        # scene from Tier B. One in ten thousand is far below measurement noise.
        if fraction > 1e-4 and not allow_synthetic:
            refused.append(f"{cell_name(scene, size_label)}: {fraction:.1%} interpolated "
                           f"(native {raw['native_points']:,})")
            continue

        queries, gen_schemas, generations, population = BUDGET[size_label]
        laz = Path(raw["path"])
        cells.append(dict(
            cell=cell_name(scene, size_label),
            scene=scene,
            morphology=MORPHOLOGY[scene],
            size_label=size_label,
            tier=tier,
            arm="search" if tier in SEARCHED_TIERS else "transfer",
            points=raw["points"],
            native_points=raw["native_points"],
            synthetic_fraction=fraction,
            extent_m=raw.get("extent_m"),
            bbox_density=raw.get("bbox_density"),
            laz_path=str(laz),
            mdspc_path=str(laz) + ".mdspc",
            trace_dir=f"results/traces/curated/{cell_name(scene, size_label)}",
            queries=queries,
            gen_schemas=gen_schemas,
            generations=generations,
            population=population,
        ))

    # Cheapest first, so a run that is cut short still yields complete small cells.
    cells.sort(key=lambda c: (c["points"], c["scene"]))
    _ = by_size
    return cells, refused


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=Path("configs/datasets/curated_cells.json"))
    parser.add_argument("--profile", default="AB",
                        help="tiers to include: A (morphology grid), B (mid-scale), "
                             "C (scale ladder), or any combination, or 'all'")
    parser.add_argument("--scene", action="append", dest="scenes",
                        help="restrict to these scenes (repeatable)")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="include tiers containing interpolated points")
    args = parser.parse_args()

    if not args.manifest.exists():
        sys.exit(f"manifest not found: {args.manifest}")

    profile = list(TIERS) if args.profile.lower() == "all" else list(args.profile.upper())
    unknown = [tier for tier in profile if tier not in TIERS]
    if unknown:
        sys.exit(f"unknown tier(s): {', '.join(unknown)}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cells, refused = build_cells(manifest, profile, args.allow_synthetic, args.scenes)

    if not cells:
        sys.exit("no cells selected - check --profile, --scene, and the synthetic filter")

    missing = [cell["cell"] for cell in cells if not Path(cell["laz_path"]).exists()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        source_manifest=str(args.manifest),
        profile="".join(profile),
        allow_synthetic=bool(args.allow_synthetic),
        morphologies=sorted({cell["morphology"] for cell in cells}),
        cell_count=len(cells),
        total_points=sum(cell["points"] for cell in cells),
        refused=refused,
        cells=cells,
    ), indent=1), encoding="utf-8")

    print(f"wrote {args.out} - {len(cells)} cells, "
          f"{sum(c['points'] for c in cells):,} points, "
          f"{len({c['morphology'] for c in cells})} morphologies")
    for tier in profile:
        in_tier = [cell for cell in cells if cell["tier"] == tier]
        if in_tier:
            print(f"  tier {tier}: {len(in_tier):>3} cells  "
                  f"({', '.join(sorted({c['size_label'] for c in in_tier}))})")

    if refused:
        print(f"\nrefused {len(refused)} cell(s) - not all points are measured:")
        for line in refused:
            print(f"  {line}")
        print("  pass --allow-synthetic to include them anyway")

    if missing:
        print(f"\nWARNING: {len(missing)} selected cell(s) have no file on disk:")
        for name in missing:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

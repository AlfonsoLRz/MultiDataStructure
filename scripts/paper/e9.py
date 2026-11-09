"""The held-out e9 cells (testing.md 2k-2m): one record per cell with the composition ratios of
summarize_nesting.py and the lever chain of lever_decomposition.py, read from the two e9 result folders
(workstation <= 50M, makinon2 >= 100M; never mixed in one ratio, every ratio is within a cell)."""
import json
import math
from pathlib import Path

from common import E9_DIRS, cell_scene, cell_size, cells_config, group_of
from lever_decomposition import analyse, read_rows
from summarize_nesting import blocks, structure_rows, starved, timing


def load_cells(dirs=E9_DIRS, leaf=128):
    meta = cells_config()
    cells, notes = [], []
    for directory in map(Path, dirs):
        for pick_json in sorted(directory.glob("*_pick.json")):
            name = pick_json.name[: -len("_pick.json")]
            timing_csv = directory / f"{name}_test_timing.csv"
            pick = json.loads(pick_json.read_text())
            if not timing_csv.exists() or not pick.get("pick"):
                notes.append(f"{name}: no held-out timing, skipped")
                continue
            lat = timing(timing_csv)
            p, n, s = pick.get("pick"), pick.get("best_nested"), pick.get("best_single")
            if p not in lat or (s and s not in lat):
                notes.append(f"{name}: pick or best single not timed, skipped")
                continue
            singles = [v for path, v in lat.items() if blocks(path, directory) == 1]
            rows = structure_rows(directory / f"{name}_test.csv")
            levers = analyse(directory, name, meta.get(name, {}), leaf, notes)
            morph = meta.get(name, {}).get("morphology", "")
            first = next(iter(rows.values()), {})
            ratio = lambda column: (float(rows[n][column]) / float(rows[s][column])
                                    if n in rows and s in rows and float(rows[s][column]) > 0 else float("nan"))
            cells.append(dict(
                build_ratio=ratio("build_time_ms"), nodes_ratio=ratio("num_nodes"),
                cell=name, scene=cell_scene(name), size=cell_size(name), morph=morph, group=group_of(morph),
                adopted=pick.get("adopted", ""), donor=pick.get("donor", ""), transfer=bool(pick.get("donor")),
                pick_single=lat[p] / lat[s] if s else float("nan"),
                nested_single=lat[n] / lat[s] if n in lat and s else float("nan"),
                pick_oracle=lat[p] / min(singles) if singles else float("nan"),
                pick_name=Path(p).stem, nested_name=Path(n).stem if n else "", single_name=Path(s).stem if s else "",
                starved=("N" if n and starved(n, directory, rows) else "") + ("S" if s and starved(s, directory, rows) else ""),
                flatness=float(first.get("flatness_score") or "nan"),
                held=levers["held"] if levers else None, opt=(levers or {}).get("opt"),
                pick_latency_us=lat[p] * 1e3))
    return cells, notes


def by(cells, key, value):
    return [c for c in cells if c[key] == value]


def verdict_counts(values, tie=0.03):
    values = [v for v in values if math.isfinite(v)]
    return (sum(v < 1 - tie for v in values), sum(1 - tie <= v <= 1 + tie for v in values), sum(v > 1 + tie for v in values))

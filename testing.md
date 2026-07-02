# Testing plan — application-workload evaluation for the paper

Status ledger + exact commands for everything still to measure and write. Context:
the paper (Computers & Graphics draft, `paper_elsevier/`) is reframed around
**instance-optimized spatial index schemas for dense, machine-generated workloads**
(pipeline self-queries + DL dataloader neighborhood queries), with trace-driven
replay as the evaluation mechanism. See `docs/application_workloads.md` for the
protocol and `paper_elsevier/sections/evaluation.tex` for the tables to fill
(placeholders are marked `\todonum{}`).

## 1. What is DONE and VALID (do not rerun)

| result | where | numbers |
|---|---|---|
| **DL batch regime** (unaffected by the frame bug: identity-frame XYZ blocks) | `results/eval_traces/invalid_frame_bug/dl_*.csv` (valid despite the folder name — move them back) | SanAndreas blocks: searched `bvh1l256+ot5l512+kd4l8192` cuts a full pass **17.5%** vs bvh; Alhambra blocks: `kd3l256(rr)` cuts **13.4%** vs octree_kdtree |
| **Boundary study (ray tracing)** | branch `triangle-meshes`, in paper §applications + evaluation table | synthetic box: grid +85%; scanned dragon: single global CWBVH wins; nested worst |
| **Corrected 5M pipeline GA** (post frame-fix) | `results/eval_traces/sanandreas_5m*.csv` | winner `qt4l2048+qt6l256` @0.003560 vs quadtree_default 0.003603 (**−1.2%**, parity-class); serial 9.4 min / `--parallel 8` **1.5 min** / cuda 1.3 min |
| Trace infra, capture scripts, batch-search mode, binary-PLY loader, frame fix + guard | commits `ae6e236..076b226` | all tests green |

**Key honest finding from the corrected 5M run:** with the frame bug fixed, the search
is *parity-class* against the best canonical structure on easy terrain at small scale.
The paper's claims should rest on (a) the DL regime (clear 13–17% wins), (b) the
*insurance* argument (wrong default = 4–35× in the invalid round; re-verify the ratio
in the corrected runs), (c) framework speedups (re-measure), and (d) whether larger /
denser / more varied datasets restore bigger deltas — which is what the dataset
matrix below is for.

## 2. What was INVALIDATED and must be re-measured

The coordinate-frame bug (fixed in `076b226`; traces = world coords, LAS loads =
local frame → radius queries returned 0 points) invalidated the first-round pipeline
and Open3D numbers. Quarantined in `results/eval_traces/invalid_frame_bug/`.
**Anything quoting per-query latency on a LAS cloud from before the fix is wrong.**

## 3. Pending experiments (exact commands)

The whole battery is scripted: `scripts/run_experiment_battery.ps1`
(steps are independent; comment out what you don't want; each appends CSVs under
`results/eval_traces/`). Already-done steps to skip: quarantine + the three 5M GAs.
Remaining, in order:

1. **GA 50M** (was killed mid-run — delete the partial `sanandreas_50m.csv` first).
2. **GA Alhambra 100M** serial — the search-cost anchor (~50 min).
3. **Rank transfer** — flat re-eval of the Alhambra candidate set on
   `alhambra_sub2m/sub5m.las`, then
   `python scripts/analyze_rank_transfer.py --full results/eval_traces/alhambra_100m.csv --low results/eval_traces/alhambra_sub2m.csv results/eval_traces/alhambra_sub5m.csv`
   → report Spearman + top-K regret (fill §search-cost table).
4. **CI ×3** — `--measure-repeats 5` on each winner (fills ± intervals).
5. **Open3D comparisons ×2** — `compare_frameworks.py --input-trace ...` from
   `.venv-mds` (Python 3.12; the main venv has no Open3D wheel).
6. **700M multi-fidelity** — capture + subsample already exist
   (`results/traces/solarpanels_700m/`): subsample GA (~2 min with `--parallel 8`)
   then top-10 confirm at full 700M (~10 builds; budget 1–2 h).
7. **PCL column** — `vcpkg install pcl[core]:x64-windows` (long compile), build
   `tools/pcl_point_baseline.cpp` (see battery script step), convert clouds to
   binary PLY (`scratchpad to_binary_ply.py` logic is embedded in the battery
   script), run compare_frameworks with `--frameworks open3d pcl`.
   PDAL stays omitted (crop-only harness support; traces are kNN/radius).

Gotchas (hard-won):
- **Always `--no-score-cache` for trace runs** (cache key ignores the trace path).
- **Check `returned_points > 0`** in any new trace replay (frame-mismatch guard
  also warns on stderr).
- GPU evaluator: schemas with unsupported policies fall back to CPU
  (`gpu_support_status` column) → don't mix backends in one ranking.

## 4. NEW: uniform (shape × size) dataset matrix

Current datasets confound shape with scale (terrain@5/50/200M, architecture@100M,
industrial@700M). Build a controlled grid so "winner vs shape" and "winner vs scale"
separate cleanly:

```
python scripts/make_dataset_matrix.py --out "D:/Datasets/Point Clouds/matrix" --sizes 1 5 25 100 ^
  terrain="D:/Datasets/Point Clouds/SanAndreas/200M.las" ^
  architecture="D:/Datasets/Point Clouds/Alhambra/Alhambra_100M.las" ^
  industrial="D:/Datasets/Point Clouds/SolarPanels/SolarPanels700M.las" ^
  indoor="D:/Datasets/Point Clouds/Sketchfab/hintze-hall-lo - Cloud.las"
```

→ up to 4 shapes × {1,5,25,100}M cells (+ optionally synthetic shapes from the C++
`SyntheticPointClouds` generators — facade / building shell / urban mixed /
sparse-dense — via `--mode schema-search` synthetic datasets at matched
`--synthetic-scale`, giving controlled-anisotropy rows real scans can't).

Then per cell: capture the pipeline trace
(`capture_pipeline_workload.py <cell.las> --out results/traces/matrix/<shape>_<size>`)
and run the GA (use `--parallel 8`; low-fidelity+confirm for the 100M cells). The
paper figure this feeds: a **shape × size heatmap of winning schema families** —
the strongest possible visualization of the instance-optimization thesis (and it
subsumes today's ad-hoc pipeline table).

## 5. Paper writing checklist (`paper_elsevier/`)

- [ ] Fill `sections/evaluation.tex` `\todonum{}` placeholders: pipeline table
      (from corrected + matrix runs), framework table, search-cost table
      (serial anchor / par8 / subsample+confirm regret+rho / GPU caveat).
- [ ] Replace the pipeline table with (or add) the **shape × size matrix heatmap**
      once §4 runs.
- [ ] Soften per-dataset claims to match corrected numbers (5M is parity-class;
      lead with DL wins + insurance + framework speedups).
- [ ] Methodology: add a paragraph on the world-coordinate trace convention and the
      frame-mismatch guard (already in threats-to-validity; cross-reference).
- [ ] Search-cost subsection (`sections/applications.tex`): refresh ρ / regret /
      wall-times from corrected runs.
- [ ] Figures: pipeline overview (placeholder exists), matrix heatmap, amortization
      plot (search cost vs passes-to-payback per dataset).
- [ ] Update `docs/application_workloads.md` tables in lockstep (its header
      currently carries the re-measurement notice — remove when refreshed).
- [ ] Author/affiliation block, acknowledgements, final bibtex pass
      (2 "empty pages" warnings are benign).

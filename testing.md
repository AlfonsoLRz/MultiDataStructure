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

## 1b. Battery v2 held-out results (2026-07-29 — THE numbers for the paper)

All searches on `_opt` halves; every number below from `*_test.csv` / `*_testeval.csv`
(untouched queries/blocks, `--measure-repeats 5` warm CIs). Full CSVs: `results/eval_traces/v2/`.

| cell | best searched (held-out) | best tuned single | verdict |
|---|---|---|---|
| 5M terrain | GA selected `quadtree_default` itself | `tuned_qt8l32` ≡ default params | parity (control) |
| 50M terrain | `qt8l256+ot2l128+bvh2l4096` 0.00814 | `tuned_qt8l32` 0.01070 | **nested −24%, CI-separated** |
| Alhambra 100M | `bvh7l128+ot5l1024` 0.01301 | `tuned_ot8l1024` 0.01334 | marginal −2.5%, CIs overlap |
| Solar 700M | `qt4l2048+ot6l256` 0.0377 | `octree_default` 0.0803 (no full tuned sweep at 700M — limitation) | **nested 2.13×, top-9 all nested** |
| DL blocks ×2 | within 1–3% of best | `tuned_ot4l256` wins both clouds | tuned shallow single is block-optimal; old −17.5%/−13.4% headlines DO NOT survive held-out + tuned singles |

Cross-cutting:
- **Scale, not (yet) morphology, drives separation**: same terrain parity→−24% from 5M→50M; 700M = 2.13×. Morphology axis pending matrix GA runs.
- **Insurance**: wrong single = 4× (qt@Alhambra), 12–25× (kd/bvh@700M), 250–720× (regular grid).
- **Gate B: random ≥ GA in every cell** (budget-matched, same grammar). Value = grammar + measured selection + multi-fidelity, not evolutionary traversal → simplify optimizer claims.
- **Rank transfer (corrected)**: ρ=0.884 (sub2M) / 0.954 (sub5M); full-scale winner always in subsample top-3 → search cheap + confirm top-3, zero regret.
- **Cross-cloud (SanAndreas-50M winner → SanSimeon 25M test half)**: transfers *safely* (+9% vs that cloud's best-known = quadtree; wrong singles 1.4–6.8×), not optimally. `results/eval_traces/v2/crosscloud_sansimeon.csv`.
- **Frameworks (frame-fixed, counts verified ≤1-pt boundary deltas)**: MDS vs Open3D KDTreeFlann **3.7×** at both 5M (0.00363 vs 0.0133) and Alhambra 100M (0.0129 vs 0.0479): `results/framework_compare/v2_*_fixed/`.
- **PCL columns (native FLANN, exact parity 0 mismatches)**: at 5M **pcl_kdtree BEATS MDS** (0.00285 vs 0.00337, 1.18×) — small clouds are FLANN territory (and parity cells anyway). At Alhambra 100M **MDS nested is 2.76× faster than pcl_kdtree** (0.01293 vs 0.03565) and 4.6× vs pcl_octree. Build: FLANN 12.4s vs MDS 67.5s → break-even ≈ 2.4M queries; one normal-estimation pass at 100M = 100M queries → amortizes ~40× within a single stage. `results/framework_compare/v2_{sanandreas_5m,alhambra_100m}_pcl/`. Helper needs vcpkg DLLs beside it (`tools/bin/*.dll`, copied); reads PLY not LAS (binary PLY converters in scratchpad, world frame).
- **Indexicon sanity check (2026-07-29, `tools/indexicon_point_baseline.cpp`, clone at
  `external/Indexicon`, MIT)**: identical 2000-query mixed trace on 5M, world frame,
  **0 count mismatches in all four structure/op combinations** (independent correctness
  cross-validation). Per-type ms — range: our octree 0.00242 ≈ theirs 0.00231, our kd
  0.00387 vs theirs 0.02482 (10×); kNN: our octree 0.01207 vs theirs 0.01994 (1.65×),
  our kd 0.01371 vs theirs 0.07186 (5.2×). **Our in-house primitives are not weak — the
  nested wins cannot be attributed to baseline implementation quality.** Their builds are
  5-7× faster (819/900 ms vs 4.9-6.3 s; lean nodes, no telemetry) — worth citing as their
  strength alongside portability/dynamism. Radius unsupported in Indexicon (586 rows
  skipped); comparison is range+kNN only. Raw: `results/eval_traces/v2/indexicon_cmp/`.
- **Matrix heatmap DONE (2026-07-29 afternoon, `scripts/run_matrix_heatmap.ps1`)** — 16 cells,
  GA + tuned-singles arms on `_opt`, held-out `_test` re-measures; summary table:
  `results/eval_traces/matrix/heatmap_summary.csv` (speedup of searched winner vs best
  tuned/default single, warm 5-repeat means):

  | shape \ size | 1M | 5M | 25M | 100M |
  |---|---|---|---|---|
  | terrain | 0.88 | 0.96 | **1.09** | **1.27** (off-grid single qt9l32!) |
  | industrial | 0.95 | 0.93 | **1.07** | **1.23** (nested) |
  | urban | 1.06 (noise) | 0.91 | 1.03 | **1.07** (nested) |
  | architecture | 0.99 | 1.04 | 1.04 | — (no 100M rung; full-cloud Alhambra = 1.03) |
  | indoor | 0.96 | — | — | — |

  Readings: (a) **crossover ≈ 10–25M for every shape** — scale is the first-order driver,
  confirmed with shape held constant; (b) shape modulates the 100M magnitude
  (terrain 1.27 / industrial 1.23 / urban 1.07 / architecture flat ~1.04); (c) the search
  wins via TWO mechanisms — nesting (industrial/urban 100M) and **off-grid single tuning**
  (terrain_100M winner qt9l32 = depth-9 single quadtree outside the tuned-singles grid) —
  reinforcing the Gate B conclusion that the grammar+measurement, not the traversal, is
  the contribution. Caveat for the paper: matrix cells are uniform subsamples (N and
  density co-vary); the native-density 5M→50M ladder (−24%) and 700M (2.13×) corroborate
  the scale trend on real densities.

## 2. What was INVALIDATED and must be re-measured

The coordinate-frame bug (fixed in `076b226`; traces = world coords, LAS loads =
local frame → radius queries returned 0 points) invalidated the first-round pipeline
and Open3D numbers. Quarantined in `results/eval_traces/invalid_frame_bug/`.
**Anything quoting per-query latency on a LAS cloud from before the fix is wrong.**

## 3. Pending experiments (exact commands)

> **SUPERSEDED for the result-bearing cells by `scripts/run_experiment_battery_v2.ps1`**
> (rigor retrofits R1-R3, see `plans/research_direction_v2.md` §C and §3b below).
> v1 stays valid for provenance and for the PCL/vcpkg chain (step 7), which v2 omits.

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

## 3b. Battery v2 — rigor retrofits (READY TO RUN, 2026-07-28)

`scripts/run_experiment_battery_v2.ps1` re-runs the result-bearing cells with the
three rigor retrofits in place. All infrastructure is built, smoke-tested, and the
traces are already split:

- **R1 splits done**: `scripts/split_trace.py` produced `pipeline_trace_{opt,test}.csv`
  for all 4 pipeline traces (30k/30k, stratified per stage, zero query overlap) and
  `batch_manifest_{opt,test}.json` for both DL sets (16/16 blocks). Searches consume
  `_opt`; every headline number comes from `*_test.csv` / `*_testeval.csv` ONLY.
- **R2 tuned singles**: `scripts/make_single_grids.py` → `configs/schemas/tuned_singles/`
  (107 schemas, budget-matched to the 5M GA's 111 evals) and `tuned_singles_small/`
  (36, for 100M+/batch cells). Same design space as the GA generator, always includes
  the old `*_default` params.
- **R3 arms per cell**: GA vs GA+`--repair-mutations` vs pure random
  (`--generated-only --benchmark-top 0` — verified: no surrogate prune) vs tuned
  singles; random budget auto-matched to the cell's GA CSV row count. 5M + Alhambra
  run all arms serial (identical measurement noise for the arm comparison); 50M uses
  `--parallel 8` for wall time.
- Batch quirk (verified): batch mode floors `--generate-schemas` at 1, so eval-only
  runs pass 1 and analysis must drop the lone `generated_*` row in `*_testeval.csv`.
- **Consequence for §1**: the DL 17.5%/13.4% numbers were selected and reported on
  the same 32 blocks; they stay as the no-split reference, but the paper should quote
  the v2 held-out-block numbers once `dl_*_testeval.csv` exists.

Rough budget: 5M cell ~1 h (all-serial ×4 arms + test), 50M ~1.5 h, Alhambra ~3-4 h
(serial anchor + 36-schema singles arm), DL cells ~30 min, 700M ~2 h. Steps are
independent — `-Only <cell,...>` runs a subset (cell names in the script header).

Fixes from the first 5M run (2026-07-28 evening):
- **GA/repair winner clobber**: both arms exported their best-schema JSON to the same
  dataset+workload-derived filename; the repair run overwrote the GA winner before the
  test step read it. Fixed with per-arm `--optimizer-output-dir`; the first run's
  `sanandreas_5m_{ga,test}_invalid_clobber.csv` are quarantined, GA arm re-run.
- **Batch winners were unreplayable**: batch-search never wrote searched schemas to
  disk (winner existed only as a CSV name). `BatchSearch.cpp` now respects
  `--generated-schema-dir` for the pool AND repair children (2-line change,
  **exe rebuilt 2026-07-28 21:08** — root `x64/Release` output copied over
  `MultiDataStructure/x64/Release`). Battery batch cells pass it and resolve winners
  by name -> `<dir>/<name>.json`.
- Batch CSVs have their own format (`schema,tag,pass_ms,...`) — the battery uses
  dedicated batch helpers; don't reuse the pipeline-CSV ones.
- **Frame bug survivor found in `compare_frameworks.py` (fixed 2026-07-29 morning)**:
  `load_points_numpy` loaded LAS clouds origin-subtracted (LOCAL frame) while traces
  and the MDS query_trace export are WORLD coords since `076b226` → every Open3D
  radius count was wrong (2000/2000 mismatches, deltas up to 7,585 points at
  Alhambra); kNN masked it by always returning k. Fixed: world-frame load, PDAL
  origin shift removed. **The overnight `v2_sanandreas_5m` / `v2_alhambra_100m`
  Open3D numbers are INVALID — re-run both after the chain finishes.** The kNN-only
  columns of older Open3D comparisons are also suspect (degenerate far-query
  traversals). `tools/pcl_point_baseline.cpp` verified clean (no frame shifts).
- Latent (not yet hit): warm `--measure-repeats` loop uses the workload's dominant
  kNN k instead of per-query k (`SchemaSearch.cpp:3406`) — harmless for uniform-k
  pipeline traces; fix before ever putting repeats on a mixed-k trace.

## 4. NEW: uniform (shape × size) dataset matrix

Current datasets confound shape with scale (terrain@5/50/200M, architecture@100M,
industrial@700M). Build a controlled grid so "winner vs shape" and "winner vs scale"
separate cleanly:

```
# one-time: merge a contiguous ~120M-pt block of the Hamilton urban-ALS tiles
# (headers-only tile selection, noise classes 7/18 dropped; ~13 tiles of 1240)
python scripts/merge_laz_tiles.py "D:/Datasets/Point Clouds/Waikato_Hamilton" ^
  --out "D:/Datasets/Point Clouds/Waikato_Hamilton/hamilton_120M.las" --target-points 120000000

# NOTE: shapes must come BEFORE --sizes (nargs='+' would swallow the name=path args)
python scripts/make_dataset_matrix.py ^
  terrain="D:/Datasets/Point Clouds/SanAndreas/200M.las" ^
  architecture="D:/Datasets/Point Clouds/Alhambra/Alhambra_100M.las" ^
  urban="D:/Datasets/Point Clouds/Waikato_Hamilton/hamilton_120M.las" ^
  indoor="D:/Datasets/Point Clouds/Sketchfab/hintze.las" ^
  industrial="D:/Datasets/Point Clouds/SolarPanels/SolarPanels700M.las" ^
  --out "D:/Datasets/Point Clouds/matrix" --sizes 1 5 25 100
```

→ up to 5 shapes × {1,5,25,100}M cells (+ optionally synthetic shapes from the C++
`SyntheticPointClouds` generators — facade / building shell / urban mixed /
sparse-dense — via `--mode schema-search` synthetic datasets at matched
`--synthetic-scale`, giving controlled-anisotropy rows real scans can't).

Dataset roles beyond the matrix (`plans/research_direction_v2.md` §G):
- **SanSimeon** (2336 OpenTopography LAZ tiles on disk) is NOT another scale rung —
  SanAndreas covers plain terrain. Its role is the **cross-cloud generalization
  test**: tune the terrain winner on SanAndreas, replay it on a SanSimeon merge
  (same morphology, different survey) — the strong form of held-out that R1's
  same-cloud split cannot give. Merge with
  `merge_laz_tiles.py "D:/Datasets/Point Clouds/SanSimeon" --out .../sansimeon_25M.las --target-points 25000000`.
- Plain-terrain cells are the *control* (5M corrected result = parity expected);
  the thesis needs the heterogeneous cells (urban / architecture / industrial) to diverge.
- Still to acquire (free): **Paris-Lille-3D** (urban street MLS, 143M labeled — feeds
  the structure-activation-vs-class figure) and **FOR-Instance** (forest UAV-LS —
  the volumetric octree-favoring morphology, currently absent).

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

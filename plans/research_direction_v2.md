# Research direction v2 — merged working plan

**28 July 2026. Supersedes `MultiDataStructure_research_direction_plan.md` (the ChatGPT
draft, kept for reference). Backbone: `testing.md` (the battery ledger stays the source of
truth for what is DONE / INVALIDATED / PENDING). This document adds the direction-level
decisions, the rigor retrofits that must land before the battery, and the answers to the
open strategic questions (database, Indexicon, R-tree, datasets).**

---

## A. Corrections to the previous plan

The ChatGPT draft is directionally sound — its three core demands (equal-budget single
tuning, disjoint query splits, optimizer ablation arms) are real gaps and are adopted
below (§C). But it was written against `docs/evaluation.md` (last touched 2026-06-26),
predating the 2026-07-02 pivot to trace-driven application workloads, and needs these
corrections before anyone quotes it in a meeting:

1. **"Only 3/12 cells ranking-confident" is wrong twice.** The artifact says 1/12
   (`results/eval_ga/report/summary.json`, only `synthetic_facade/range_heavy`), and the
   flag measures whether the *top candidate beats the runner-up* (usually a near-identical
   evolved sibling), not whether the optimizer beats the baselines. The medium-volume wins
   (1.70×/1.76×) **are** CI-separated from the best single; the small-query/kNN cells are
   pure noise. Both halves matter: the wins are more solid than the draft claims, the
   noise-floor concern is real but mislocated.
2. **The Alhambra 1.26× should not be quoted as evidence.** 3 schemas, 64 queries,
   `confirm_seeds=2`, CIs overlap almost completely (322.99 [195, 451] vs octree 407.84
   [293, 523]), the winner is hand-designed (not synthesized), GPU-only, and the same doc
   section later shows the radius-heavy workload is faster on CPU-resident schemas anyway.
3. **"Optimization and re-measurement use the same query set" is half-right.** The
   synthetic headline re-measures on unseen confirm seeds (weak-but-real held-out). But
   **all trace-driven runs — the current paper pipeline — search and re-measure on the
   identical trace** (`run_experiment_battery.ps1`). The criticism lands exactly where the
   paper now lives → retrofit R1.
4. **Singles-at-defaults is understated.** Every baseline row is `*_default`
   (`scripts/compare_methods.py` `DEFAULT_SINGLES`; hardcoded numLevels/leafCapacity).
   Zero tuning budget on the single arm → retrofit R2.
5. **Most machinery it proposes to build already exists**: conditional per-node schemas
   (`SchemaLevelCondition`: density, anisotropy, occupancy entropy, height ratio, extents;
   `--auto-conditions`), telemetry-guided repair (`--repair-mutations`, 6 diagnosis rules,
   off by default), successive-halving rungs (`--rungs`), parallel dispatch (`--parallel`),
   bootstrap CIs (`--confirm-seeds`, `--measure-repeats`), NSGA-II GA. What is missing are
   the **comparison arms**, not the mechanisms → retrofit R3.
6. **It doesn't know the frame-bug invalidation** (`076b226`): first-round pipeline,
   Open3D (5.4×/3.7×), rank-transfer ρ and CI numbers are quarantined in
   `results/eval_traces/invalid_frame_bug/`. Nothing pre-fix on a LAS cloud is quotable.
7. **It doesn't know the two strongest post-fix facts** (see §B): the DL batch wins are
   final, and the corrected 5M pipeline is parity-class — which together reframe the whole
   fail-fast question.
8. **Indexicon is oversold** (see §F): real, MIT-licensed, complementary — but its
   evaluation has **no radius queries** and its datasets are geographic, not LiDAR.

## B. Current evidence base (echoes `testing.md` §1–2; do not duplicate, do not re-derive)

- **Final and valid — DL batch regime** (full-pass wall cost incl. per-block rebuilds):
  searched `bvh1l256+ot5l512+kd4l8192` cuts a SanAndreas pass **−17.5%** vs bvh;
  `kd3l256(rr)` cuts an Alhambra pass **−13.4%** vs `octree_kdtree`. This is the
  strongest current result and lives in the regime the ChatGPT draft demoted to
  "high-risk optional".
- **Final and valid — boundary/ray study** (`triangle-meshes` branch): nesting *loses* on
  scanned geometry (global CWBVH wins); honest negative already in the paper.
- **Final and valid — corrected 5M pipeline GA: parity-class** (−1.2% vs
  `quadtree_default`). The repo's own "key honest finding".
- **Invalidated**: everything pipeline/Open3D/rank-transfer/CI measured before `076b226`.

**Consequence — the central open question of the paper is now precisely:**
*does the nested/conditional advantage appear with scale (25–700M) and morphological
heterogeneity, given that it is parity on plain terrain at small scale?* That is exactly
what the shape × size matrix (testing.md §4) answers, and why the battery must not run
until the rigor retrofits below make its numbers defensible.

## C. Rigor retrofits (land BEFORE the battery — user-confirmed ordering)

### R1. Disjoint trace splits (small; Python-only)
- New `scripts/split_trace.py`: split every captured trace CSV into `<name>_opt.csv` /
  `<name>_test.csv` (50/50, deterministic seed, stratified by the stage/query-type column
  the capture scripts already emit; keep the JSON sidecar's full-count projections intact).
- Battery wiring: every search step consumes `_opt`; the final re-measure step and all
  framework comparisons consume `_test` only. The "best single" per cell is also selected
  on `_opt` and reported on `_test` (kills max-selection bias).
- Gotchas that already bit us and apply here: `--no-score-cache` on all trace runs (cache
  key ignores trace path); assert `returned_points > 0` in every new replay.

### R2. Equal-budget tuned singles (moderate; no C++ needed)
- New `scripts/make_single_grids.py`: emit single-block schema JSONs per primitive
  (numLevels × leafCapacity × minPointsToSplit × axisPolicy grid) into
  `configs/schemas/tuned_singles/<primitive>/`.
- Evaluate via existing `--flat-search --schemas <list> --no-baselines`, capped to the
  same number of full-fidelity evaluations as the nested arm gets in that cell.
  (Do **not** combine with `--auto-conditions` — it shadows `--schemas`.)
- Every table's "best single" column becomes "best *tuned* single". Prediction to test:
  the one existing data point (facade/knn_heavy: GA's tuned 12-level BVH *lost* to
  `bvh_default`, 0.95×) suggests tuning won't erase the wins — but it's n=1.

### R3. Optimizer ablation arms (ride along the battery)
Budget-matched per cell (same full-fidelity evaluation count, same rung schedule):
1. **Random**: flat `--generated-only` sampling of the same grammar with surrogate pruning
   disabled (verify the disable flag in `docs/command_reference.md`; if none exists, set
   benchmark top-K = candidate count — the silent default is generate 256 → prune to 32,
   which is *not* a random arm).
2. **GA** (current default, NSGA-II on).
3. **GA + `--repair-mutations`**.
Record best-so-far-vs-evaluations curves and final held-out latency. Decides Gate B.

## D. Battery + matrix (defer to `testing.md` §3–§4, amended)

Run once, after R1–R2, with R3 arms attached to the result-bearing cells. Amendments to
testing.md §4: add `waikato_hamilton` as an urban-ALS shape (already on disk, missing from
the matrix command) and the two new morphologies from §G once downloaded. GA on the
interesting subset; pipeline traces everywhere; low-fidelity + confirm for ≥100M cells.

## E. Decision gates (rewritten against the real evidence base)

- **Gate A — structural advantage.** After R1+R2: nested/conditional beats the best
  *tuned* single on *held-out* test splits with non-overlapping CIs in multiple real
  heterogeneous or large cells (urban MLS, forest, architecture, industrial, ≥25M).
  Parity on plain terrain is *expected* and reported as scoping, not failure. If parity
  holds everywhere except DL batch → the paper narrows to the batch/rebuild regime +
  insurance argument (still publishable, different framing).
- **Gate B — optimizer contribution.** GA+repair reaches equal regret with ≥2× fewer
  evaluations than random, or clearly better held-out schemas at fixed budget. If random
  matches GA+repair → drop the optimizer-novelty claim, keep the simplest search that
  works, and center the paper on conditional composition (the ChatGPT draft's own advice,
  which is right).
- **Gate C — conditionality.** Conditional schemas vs fixed sequences vs tuned single on
  the matrix subset; require repeatable gains plus an interpretable activation story —
  the structure-activation-vs-semantic-class figure on Paris-Lille-3D labels (§G) is the
  target visualization (machinery exists: `visitedByStructure`, `activeStructureSummary`).
- **Gate D — application.** Already half-passed: DL full-pass cost incl. rebuilds is an
  honest end-to-end-adjacent metric with final wins. The `pcl::search::Search<PointT>`
  adapter is a **conditional stretch** (user-confirmed): build only if Gate A passes on a
  large/heterogeneous cell; until then the paper leans on DL wins + limitations paragraph.

## F. Indexicon + R-tree (external baselines, not centerpiece)

Indexicon (arXiv:2606.04676, Jun 2026; **code MIT**, header-only C++17): R-tree, quadtree
variants, octree, kd-tree; range + kNN; six geographic datasets (3D: MARINE 25M,
TORONTO 21.6M, MIAMI 3.5M; 2D: OSM 103.5M, TAXIS 112.8M, TIGER 17.9M). **No radius
queries** — our traces are radius/kNN-heavy, so it is not a drop-in benchmark.

1. Cite in `related_work.tex`: it solves portable *individual* indexes; we solve
   *composition and conditional selection* — complementary, and it strengthens the
   "this space is active in 2026" motivation.
2. Cheap sanity check (1 day): our octree/kd vs Indexicon's on one dataset, range+kNN,
   identical traces + returned-count parity — shows wins don't stem from weak in-house
   primitives.
3. **R-tree question (resolved): we have none** (CPU BVH/LBVH are fallback-mapped onto
   kd/octree splits). Do **not** add R-tree as an internal schema primitive — overlapping
   MBRs don't fit the per-level disjoint-split grammar, and bulk-loaded R-trees on static
   in-memory points is their weakest regime. Instead add **Indexicon's R-tree as an
   external baseline column** in the trace replay: range/kNN native; radius = small
   distance-to-MBR pruned traversal in their header-only interface (~50–100 lines).
   Revisit promoting an R/BVH-family primitive only if that baseline ever wins a cell —
   that would be data telling us the grammar misses a partition family.
4. Optional generalization cells: MARINE + TORONTO as non-LiDAR 3D externals (range+kNN
   only), labeled as such — not mixed into LiDAR headlines.

## G. Datasets (resolved): don't buy better terrain — buy missing morphologies

On disk (`D:\Datasets\Point Clouds`): SanAndreas 5/50/200M + SanSimeon tiles (terrain,
OpenTopography), Alhambra 100M (architecture TLS), SolarPanels 700M (industrial),
Sketchfab hintze/lion (indoor/object), **Waikato_Hamilton NZ 2023 urban-aerial tiles
(downloaded Jul 2, not yet in the matrix — add it)**.

"OpenTopography is plain" is fine: plain terrain is the **control cell** where parity is
expected (the corrected 5M result *is* that cell). SanSimeon's role is therefore not
another scale rung but the **cross-cloud generalization test**: tune the terrain winner
on SanAndreas, replay it on a SanSimeon merge (same morphology, different survey) — the
strong form of held-out that a same-cloud split cannot provide (`scripts/merge_laz_tiles.py`
builds contiguous subsets from the tile sets). The thesis needs heterogeneous cells to
diverge; two morphologies are genuinely missing, both free:

1. **Urban street-level MLS** — the strongest heterogeneity showcase (façades + ground +
   poles + trees + density falloff from trajectory): **Paris-Lille-3D** (143.1M labeled
   points, ~2 km, CC BY-NC-ND → internal benchmarking OK, don't redistribute derived
   clouds; labels feed the Gate C activation figure). Fallback: Toronto-3D (78M).
2. **Forest/vegetation** — the volumetric, octree-favoring regime, currently absent
   entirely: **FOR-Instance** (dense UAV-LS over five countries, Zenodo
   doi:10.5281/zenodo.8287792; verify license at download). Alternatives: Semantic3D
   rural scans, NEON tiles.

Target matrix: ≤6 shapes × {1, 5, 25, 100}M. Synthetic generators stay for
controlled-anisotropy ablations only.

## H. Database path (the applicability question, answered)

**Premise correction.** "No database lets you implement custom nested structures" is not
quite true: PostgreSQL has custom index access methods, and GiST/SP-GiST are explicitly
generalized-tree frameworks; DuckDB supports extension-defined indexes (its spatial
extension ships an R-tree); SQLite has an R-tree module. What **no** DBMS provides is
*composition* of index primitives with per-node conditional choice — and a production
access method (pages, WAL, concurrency, vacuum, planner costing, operators) is a separate
systems project that would smother the spatial claim.

**Reframe.** Applicability ≠ DBMS residency. The instance-optimized-index literature this
work belongs to (Data Calculator; Flood/Tsunami; learned multidimensional indexes) was
published on standalone systems evaluated on workload traces — the methodology this repo
already runs. A C&G paper needs application grounding (pipelines, DL dataloaders), not a
database.

**The pragmatic future path** (follow-up systems paper, e.g. SIGSPATIAL/SSTD/EDBT):
point-cloud "databases" in practice are patch/tile systems — pgPointcloud patches,
COPC/EPT tiles, Potree octrees. Per-patch in-memory schema synthesis is **structurally
identical to the batch-search regime already implemented** (`--batch-manifest`: many small
(cloud, trace) pairs, rebuild cost first-class — the DL machinery). Demonstrator sketch:
coarse SQL/tile filter (pgPointcloud GiST over patch extents, or COPC hierarchy) →
synthesized per-patch/per-morphology-cluster schema answers exact in-patch radius/kNN →
compare end-to-end against pgPointcloud `PC_*` + explode and DuckDB-spatial R-tree.
The DL batch result is the bridge: it already proves the "many small indexes, rebuild
cost counted" regime. **Defer until after the C&G submission; write one future-work
paragraph in the paper.**

## I. Deferred + repo hygiene

Deferred (with repo evidence, not just caution): GPU headline (MixedTree has no tree kNN;
entropy conditions and adaptive leaf capacity are CPU-only → `cpu_fallback`); database
integration (§H); real PointNet++ in-training integration (trace-replay limitation goes in
the paper's limitations; full-pass cost incl. rebuilds is already honest).

Hygiene:
- Fix `docs/evaluation.md` self-contradictions (1.13× vs 1.12×; "1/12" vs "3/12") or stamp
  the doc "superseded by trace-driven evaluation, see testing.md".
- `results/` is gitignored — every headline number lives only in this working copy.
  Archive corrected battery CSVs durably (results branch or release zip).
- Move the valid DL CSVs out of `results/eval_traces/invalid_frame_bug/` (testing.md §1).
- Post-paper, optional: minimal CMake for the CLI core (external reproducibility;
  currently VS-only, no CI, no Linux).

## Questions worth bringing to the supervisor meeting

1. If the matrix shows parity everywhere except the DL batch regime, are we happy to
   reframe the paper around "instance-optimized schemas for machine-generated block
   workloads" (where the wins are final) rather than general point-cloud indexing?
2. Is −13% to −17% full-pass cost, CI-confirmed, enough of a headline for C&G, or do we
   require a Gate A win on a large heterogeneous cloud too?
3. Random vs GA+repair (Gate B): if random matches, do we keep any optimizer-novelty
   claim or fully pivot to conditional composition?
4. Paris-Lille-3D is CC BY-NC-ND — acceptable for the evaluation set given we only
   publish numbers and figures, not derived clouds?
5. Do we invest the ~1 day in the Indexicon R-tree baseline column now, or only if a
   reviewer asks? (Recommendation: now — it closes the most predictable review question.)

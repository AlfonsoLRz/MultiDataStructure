# Testing plan — application-workload evaluation for the paper

Status ledger + exact commands for everything still to measure and write. Context:
the paper (Computers & Graphics draft, `paper/`) is reframed around
**instance-optimized spatial index schemas for dense, machine-generated workloads**
(pipeline self-queries + DL dataloader neighborhood queries), with trace-driven
replay as the evaluation mechanism. See `docs/application_workloads.md` for the
protocol and `paper/sections/evaluation.tex` for the tables to fill
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
- **Indexicon external baseline COMPLETE (2026-08-06, plan §F.3 closed)** —
  `tools/indexicon_point_baseline.cpp` rewritten, built by
  `tools/build_indexicon_baseline.ps1` (pins the gitignored clone at `c9f9b1d`, MIT), run by
  `scripts/run_external_baselines.ps1`. Now covers **three** structures (octree, kd-tree,
  **packed R-tree**) and **all** query types; reads `.las` via the `.mdspc` sidecar; emits
  JSON. 13 matrix cells (4 shapes × 1/5/25M + indoor_1M), held-out `_test` halves, 3000
  queries × 3 repeats, **both sides measured in the same session**.
  Raw: `results/framework_compare/indexicon/` (+ `indexicon_summary.csv`).
  - **Radius is no longer skipped.** Implemented per structure with the same
    min-distance-to-node pruning each one's own kNN uses; `--verify-bruteforce` validates
    against brute force and refuses to report on disagreement. The old run skipped 586/2000
    rows — and pipeline traces are 2/3 radius, so the previous comparison covered a
    *minority* of the real workload. `native_mbr_prune` is 1.39–1.59× faster than the naive
    `aabb_filter` emulation, so reporting the naive one would have flattered us ~1.5×.
  - **Exactness: 0 count mismatches over 351,000 measured samples.** This is the headline
    and it is clean. (Only disagreement anywhere: 1 point at 3e-8 relative distance from a
    query radius on the old 5M synthetic trace — `float` vs `double` rounding.)
  - **⚠ The old claim "our primitives match or beat theirs" is WITHDRAWN.** Their octree and
    kd-tree are faster in all 13 cells. Two separate causes, and the split matters:
    - *Per-node traversal cost (octree) — NOT telemetry.* Per-cell fit of
      latency = a·visited + b·tested gives **a ≈ 51–131 ns/node** vs **b ≈ 1.1–2.8 ns/point**,
      stable across all 13 cells. Node counts are ~flat in scale while tested points grow
      ~10×, so the fixed cost dilutes and the octree gap closes: 0.48→0.57→**0.94** (arch),
      0.40→0.76→**1.01** (industrial), 0.52→0.76→**0.94** (terrain), 0.51→0.63→**0.84**
      (urban). **Parity by 25M.**
      ⚠ **Correction (2026-08-06, second revision — the first correction was itself wrong):**
      the per-node telemetry (`recordNodeVisit` building a `std::string` by value and probing
      an `unordered_map` per node, inside the timed region) was rewritten to arrays indexed by
      a build-time `Node::_structureIndex`. An initial A/B said "no improvement" — that ran a
      **stale binary** (see the `x64\Release` path bug below) which lacked the change. Redone
      against the correct binaries, with **identical node/point counters** proving the
      traversal is unchanged: **octree 1.21×, kd-tree 1.03×** (66.7 vs 28.7 visited
      nodes/query) = **6–14 ns/node** of the ~106 ns/node total. The remaining ~90 ns/node is
      real traversal work, chiefly node-struct cache misses (a 524k-node kd-tree spans ~50 MB).
      Consequence: MDS was paying ~10% more per node than needed, so the octree ratios above
      are pessimistic by ~1.2× and our octree likely **beats** Indexicon's at 25M on re-measure.
    - *A kd grid configuration bug (real, but cheap).* Our kd-tree tests **8–19× more points
      than our own octree** (15,008/query at industrial_25M vs octree 1,110). Cause is
      arithmetic: kd is **binary**, so `numLevels: 12` caps it at 2¹²=4096 leaves, and the
      builder stops on depth before consulting capacity (`PointSpatialIndex.cpp:888` before
      `:894`). Measured on terrain_5M: `kdtree_default` = 8191 nodes, **avg leaf occupancy
      1220.7 vs requested `leafCapacity` 32** (38× miss, p50/p90/p99 = 1221/1222/1224).
      Reaching capacity 32 at 25M needs depth ≈20; grid ceiling is 12 → **every** kd schema
      in `configs/schemas/tuned_singles*/kd/` is depth-capped at ≥1M, `leafCapacity` inert in
      9/12. In `tuned_singles_small` (used by the whole heatmap battery) all 4 kd schemas are
      depth-capped and **`tuned_kd12l32` builds a byte-identical tree to `kdtree_default`** —
      the kd tuning arm was largely inert. Root cause: `make_single_grids.py:32-45` draws
      depth and capacity as independent axes with depth ≤12, which is coherent for an 8-ary
      octree but not for a binary kd-tree.
    - **FIXED 2026-08-06.** `make_single_grids.py` now derives depth from capacity and cloud
      size via a per-primitive branching factor (`--points`, default 25M; ceiling 12→24) and
      clears stale schemas from its output dir (93 stale files were found beside the live
      ones). Both grids regenerated at original sizes (108 / 36): **0 inert schemas**, kd arm
      now spans depths 13–20 instead of all-12.
    - **Baseline got 1.22× stronger** (terrain_5M, held-out, 3 repeats): best fixed-grid kd is
      `tuned_kd16l1024` 0.00487 ms (16,383 nodes, occ 610) vs `kdtree_default` 0.00592 ms
      (8,191 nodes, occ 1221). This *reduces* our reported margins wherever a kd-tree was the
      best single. Grid now spans over-splitting too: `tuned_kd20l32` hits occ 19.1 as asked
      and is 2nd slowest, so depth is a real trade-off rather than a cap.
      **⚠ Every matrix/battery number with a kd baseline predates this and needs re-measuring.**
      Note the depth sweep still shows only **1.03×** from cap-lifting alone on the *default*
      capacity — the win comes from capacity/depth pairs the old grid could not express.
  - **§F.3 gate answered: R-tree slowest of Indexicon's three in 8/13 cells, fastest in 0.**
    Within-library comparison, immune to both problems above. The plan's condition ("revisit
    an R/BVH primitive only if that baseline ever wins a cell") is not met — keep it out of
    the grammar, now on external evidence rather than our own omission.
  - Their builds remain 5–7× faster (lean nodes, no telemetry) — their genuine strength,
    alongside portability and dynamism.
  - Open3D/PCL wins (§ above) were earned *while* paying this instrumentation → conservative.
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

### 2b. Audit of 2026-08-06 — ten defects fixed, two of them invalidating

A read-only sweep of the C++ core, the capture/prep scripts and the orchestration found ten
real defects. Two change what past numbers *mean*; the rest corrupt re-runs or baselines.

- **⚠ INVALIDATING — trace radii were wrong above 2M points.** `mean_spacing()` built its
  KD-tree from an independent 2M random subsample, so it measured the 2nd-nearest neighbour
  in a cloud ~(N/2M) times sparser than the real one. Measured inflation vs exact: **1.00× at
  1M, 2.03× at 5M, 5.03× at 25M**. Radius-outlier-removal (4× spacing) and clustering (2.5×)
  derive from it, so the query radius was nearly *constant in world units* across a size
  ladder instead of shrinking with density — larger cells got disproportionately larger
  neighbourhoods. **This confounds the scale-crossover claim** and explains most of the ~5×
  growth in tested-points from 5M→25M. Fixed with a local-block estimator (uniform anchors,
  count-weighted, interior-only queries); verified 0.99–1.14× of exact across nine cells with
  no size-dependent bias. Sidecar now records `spacing_estimator_version`. **Every trace must
  be re-captured; version-1 traces are not comparable with version-2 above 2M points.**

  *2026-08-11 follow-up (estimator v3):* the v2 block estimator itself failed on the curated
  benchmark's terrestrial scans. TLS density spans two orders of magnitude across a scene
  (occupancy p99/p50 = 186× on DomFountain vs 6× aerial), and v2's bbox-uniform block anchors
  landed in a different density regime every seed: 3.4× seed spread on Cathedral, 1.48× bias
  on DomFountain, and per-scene spacing curves that *rose* with point count — the
  `prepare_curated_cells.py` exponent gate caught it before the battery ran. v3 measures
  clouds up to 130M points **exactly** (full tree, probes drawn from the points; 13 s at 25M)
  and, above that, anchors blocks on random *points* with fixed equal-weight probes per block,
  which is the density-correct sampling. Verified on the two worst scenes: Cathedral exponent
  −0.464, DomFountain −0.479 (theory −0.5), monotone, seed-stable to 2.5%. v2 traces on
  non-uniform clouds are not comparable with v3; the whole battery recaptured under v3.
- **⚠ INVALIDATING — scripts ran a stale binary.** MSBuild links to `<repo>\x64\Release`, but
  all four runner scripts used `<repo>\MultiDataStructure\x64\Release`, a legacy path nothing
  writes to; the copy there was from **2026-07-28**. Any C++-dependent measurement since then
  tested code that did not include the change under test — this is how the telemetry A/B got
  the wrong answer. Fixed: `scripts/resolve_exe.ps1` resolves the real path and **throws if
  the binary is older than the newest source**; the stale copy was deleted.
- **Score cache ignored the replayed trace.** `makeEvaluationCacheKey` never hashed
  `_tracePath`, and the bypass guard tested `_queryTracePath` (the `--query-trace` *output*
  flag) instead of `--input-trace`. Two runs over different traces collided — reproduced: the
  second returned the first's rows exactly (visited 65.7225 both), vs 66.16 with the cache
  off. Fixed both; verified 0 hits / 0 misses on trace runs now.
- **RegularGrid/HGrid branching factors were wrong** in the tuned-singles generator (1/8 vs
  the real 27/64 from `gridSubdivisionsForLevel`), pinning every `tuned_rg*` to depth 1 with
  an inert capacity. Fixed; `rg` now spans depths 3–5 and `tuned_rg5l32` reaches occupancy
  36.5 against capacity 32. Side-effect worth noting: a *tuned* regular grid runs 0.0076 ms
  vs `regular_grid_default` 0.508 ms — **67× stronger**, so the "regular grid 148–777×"
  insurance figure in METHOD.md §3.5 will shrink substantially on re-measure.
- **Warm `--measure-repeats` loop used the dominant k**, not per-query k, so on mixed-k traces
  the CI could exclude `avg_latency_ms`. Fixed to mirror the cold pass.
- **`--csv` writers append**, and re-runs selected the *oldest* row (`Select-Object -First 1`)
  — so re-running a cell reported its first-ever numbers forever. All runner scripts now clear
  their target CSV first.
- **Battery cells failed silently**: no exit-code gating, no cloud/trace existence check, and
  the TEST re-measure still ran with defaults-only after every search arm failed — producing a
  headline CSV indistinguishable from a genuine negative result. Now guarded, gated, and
  summarised, with a non-zero exit on any failure.
- **Matrix subsampling degenerated to prefix truncation** when `target ≤ len < 2·target`
  (`stride = len//target` → 1), taking the first N points — one flight line, not a subsample.
  Replaced with a seeded uniform sample (verified spread 1.000 vs 0.556 before), plus a size
  tolerance so a source 0.2% short still fills its rung, and a configurable LAS scale.
- **`--max-load-points` was ignored** in the same band, and its stride restarted per 2M chunk.
  Fixed globally with exact truncation.
- **`make_single_grids.py --primitives <subset>`** overwrote `index.txt` with only that
  subset while leaving other primitives' JSON on disk. Index is now built from disk.

### 2c. 2026-10-03 — replayed trace centers were rounded to float32

`loadQueryTrace` parsed world coordinates with `std::stof` into `glm::vec3`. Float32 steps are
0.25 m at UTM northings (~4e6) and 0.0625 m at eastings (~6e5), so every replayed query center
on a UTM-framed scene moved before the world→local transform. The external drivers
(`tools/indexicon_point_baseline.cpp`, PCL, Open3D) parse the same CSV as doubles, so **they
answered the recorded queries and MDS answered displaced ones.** Fixed: `TraceQuery` holds
`glm::dvec3`, parsed with `stod`; regression test in `tests/test_query_trace.cpp`.

Mean center shift on the held-out traces, vs the median query radius where known:

| Scenes | Mean shift | Effect |
|---|---|---|
| Aneto, Barcelona, GaliciaForest (+ES_05), OttobrunnCampus, Retiro, Toronto, Waikato_Hamilton | 12.4–13.3 cm | Toronto: radius median 8.1 cm, so most radius spheres missed the surface they were captured on |
| SanAndreas | 6.6 cm | radius median 58 cm, 11% displacement |
| Alhambra, Cathedral, DomFountain, Lille, Museum, Neugasse, ParisLux, SolarPlantation | 0 | small local frames, unaffected |

What this means for past numbers:
- **MDS-internal comparisons stay self-consistent.** Searched vs tuned singles, the heatmap and transfer arms all saw the same displaced queries, so their rankings are rankings for a plausible workload, just not the recorded one. Re-run before quoting magnitudes on the affected scenes.
- **External comparisons on affected cells were apples to oranges.** That's 8 of the 17 Indexicon cells (aneto, barcelona, galiciaforest, galiciaforest_es05, ottobrunncampus, retiro, sanandreas, toronto, waikato_hamilton) and the SanAndreas PCL/Open3D rows. Re-measure all of them.

### 2c'. 2026-10-04 — the tuned-singles arm never measured all 36 singles

Flat searches default to `--benchmark-top 32`: with more listed schemas than that, the surrogate
rank model (`models/schema_selector.json`) keeps 32 and drops the rest **unmeasured**. The tuned-
singles list has 36 entries and the battery never passed `--benchmark-top`, so every `_singles.csv`
has 32 rows. In 72 of 85 cells the dropped four were the 32768/8192-point-leaf singles
(`tuned_bih11l32768apcl`, `tuned_bvh10l32768`, `tuned_rg3l32768`, `tuned_rg3l8192`), which are
not plausible winners. In 13 cells the model also dropped competitive ones (`tuned_bih20l32`,
`tuned_bih20l32apcl`, `tuned_bvh16l1024`), so the "best tuned single" there may be understated.
Fixed for new runs: `scripts/run_cells.py` and the GUI pass `--benchmark-top 0`, and the search no
longer loads the rank model when pruning is off. The model file was also gitignored (blanket
`*.json`), so a fresh clone could not run a default flat search; it is now tracked.

**Repeat statistics (fixed 2026-10-04, `2325c5e`).** From engine 2 to that commit,
`latency_repeat_*` pooled the instrumented first pass with the uninstrumented timed passes - two
different measurements in one mean and CI. Now `latency_repeat_*` covers only the uninstrumented
passes; `avg_latency_ms` remains the instrumented pass. Only held-out TEST re-measures use repeats;
the engine-6 batteries were restarted (PC) or rebuilt mid-run with the affected cell re-tested
(makinon2: sanandreas_100M).

### 2d. Engine rewrite ledger (from 2026-10-03)

Every engine change is gated by `scripts/engine_equivalence.ps1` + `scripts/diff_engine_equivalence.py`
against golden snapshots in `D:\MDS_equivalence\` (golden_v1 = binary of `8349c52`, engine
`1-node-tree`). `PointSpatialIndex::EngineVersion` names the engine in every engine-check row and
in the score-cache key, so cached scores never cross engines.

**Counter definitions changed in engine 2 (`2-telemetry-off`).** `visited_nodes` now counts nodes
*entered* (bounds passed the prune) for every query type. Before, range/count/radius also counted
nodes that were tested and pruned; that number is now `tested_boxes`. Verified on 196,500
SanAndreas_1M queries: old `visited` == new `tested_boxes` exactly, except on the two schemas
that keep empty nodes (the old count included empty nodes, `tested_boxes` does not). kNN
`visited` is unchanged (it already counted entered nodes). **Old and new `visited_nodes` are not
comparable for range/radius; compare old `visited` with new `tested_boxes`.** Timed passes
(`--measure-repeats` > 1 in search, every timed pass in engine-check) now run with telemetry
compiled out; the first pass stays instrumented and feeds scoring.

| Step | Change | Equivalence | Effect |
|---|---|---|---|
| 1 | kNN sort and range-containment copy moved out of / simplified in the timer | results + structure identical | (A/B pending) |
| 2 | telemetry-off timed path, `tested_boxes` | results + structure identical; counters remapped as above | (A/B pending) |
| 3a | split rules resolved once per build instead of string-parsed per point per level | 131/131 schemas bit-identical trees on SanAndreas_1M | **build 16.2x faster** (95.0 s -> 5.9 s summed over 131 schemas) |
| 4 | node pool: 32 B hot nodes in contiguous 64 B-aligned sibling groups, cold data split off, explicit-stack traversals, `NodeView` API | results + structure identical to golden_v1 (393k queries, 2 clouds); counters identical to step 3a | 1M unpinned: radius 0.83x, kNN 0.81x, range 0.77x (geomean over 131 schemas); 25M A/B pending |
| 5b | AVX2 leaf scans for radius/range/count and kNN candidate filtering; explicit mul+add (no FMA), ordered compares, scalar tail | results + structure + counters identical to step 4 (393k queries) | 1M unpinned vs step 4: count 0.50-0.55x, range 0.83-0.89x, kNN 0.89-0.91x, radius 0.93-0.97x |
| 7 | depth-first nearest-child kNN for k <= 64 (best-first kept above) | results + structure identical; kNN counters change (v4) | **pinned 25M A/B vs step 5: kNN 0.872x geomean, faster in 15/15 cell x schema pairs, both rounds agree**; radius (identical code, control) 0.966x |

**Reading an A/B: binary layout noise.** In the step 5 -> 7 A/B the radius code is byte-identical,
yet single (cell, schema) radius ratios ranged 0.78-1.07 across two pinned rounds. Relinking
moves code alignment enough to shift one measurement by up to ~20%. Per-case ratios are not
evidence; quote geomeans over many cases and net them against an unchanged-code control column.

**Pinned head-to-head vs Indexicon (2026-10-04, `scripts/head_to_head.ps1 -Tag engine6`, engine
`6-dfs-knn`, Indexicon rebuilt with the same `/O2 /arch:AVX2 /GL`, 1 warm-up + 3 timed passes,
3000 held-out trace queries, P-core 4).** MDS / Indexicon, mixed = equal-weight mean of radius and
kNN latency; < 1 = MDS faster. `D:\MDS_ab\h2h\engine6\summary.csv`.

| Pair | Alhambra 25M | SanAndreas 25M | Toronto 25M |
|---|---|---|---|
| octree vs octree | 0.58 | 0.46 | 0.65 |
| kd vs kd | 0.69 | 0.86 | 0.84 |
| same-bucket kd (128) vs kd | 0.69 | 0.80 | 0.73 |
| build, octree vs octree | 0.44 | 0.42 | 0.38 |

**Overall engine A/B (step 0 = engine 1 vs engine 6, pinned, 2 rounds, 3 cells x 5 schemas, every
case consistent):** query latency 0.508x geomean (radius 0.466x, kNN 0.557x), build 0.070x.
**Equivalence of engine 6 vs golden_v1:** 1,260,000 queries over all tiers (131 schemas x five 1M
clouds, micro-indexes, three 25M traces): 0 result and 0 structure mismatches.

**nanoflann (1.10.1, header-only, same flags, `D:\MDS_ab\h2h\engine6_nf`), pinned.** MDS / nanoflann:

| Pair | Alhambra 25M | SanAndreas 25M | Toronto 25M |
|---|---|---|---|
| MDS kd19l128 vs nanoflann leaf 128: radius | 0.78 | 0.82 | 0.82 |
| ... kNN | 1.13 | 1.05 | 1.25 |
| ... mixed | 1.00 | 0.97 | 0.99 |
| ... build | 0.81 | 0.81 | 0.63 |
| MDS kdtree_default vs nanoflann leaf 10: mixed | 0.89 | 0.91 | 0.92 |

nanoflann is the strongest single-structure baseline: parity on the mixed trace, MDS ahead on
radius and build, behind on kNN. Likely source of the kNN gap: nanoflann's depth-first search
updates the cell distance incrementally along the split axis; ours recomputes a full 3D distance
to each child's tight box. Candidate next engine step.

August (engine 1, float-shifted queries on SanAndreas/Toronto): Indexicon was faster on every
one of these pairs (kd ~2x). The searched winners in `equivalence_set/winner_*` were selected by
engine 1 on shifted queries and are now stale (best radius, poor kNN: winner_toronto_25M kNN
9.05 us vs 2.98 us for kdtree_default). **Re-run the searches before quoting searched-vs-single.**

### 2e. Engine-6 battery, 1M-100M on the workstation (2026-10-04, `results/eval_traces/matrix_e6`)

69 cells (all searched cells except the three scenes the server runs at >= 100M), protocol of
`scripts/run_cells.py`: GA (8 parallel workers) and all 36 tuned singles on the `_opt` trace half,
held-out TEST of winners + 9 defaults + 2 hand-nested on `_test`, 5 uninstrumented repeats.
207/207 steps exit 0, 3.5 h.

**Searched nested schemas no longer beat the best single structure at <= 100M.** Best single
(tuned or default) / best searched, held-out:

| size | cells with a searched row | median | CI-separated searched wins | losses |
|---|---|---|---|---|
| 1M | 17 | 0.87 | 1 | 11 |
| 5M | 12 | 0.89 | 2 | 7 |
| 25M | 13 | 0.97 | 0 | 4 |
| 50M | 13 | 0.89 | 1 | 3 |
| 100M | 6 | 0.85 | 0 | 3 |

(Cells without a searched row: the GA's own winner was a single structure.) In August searched
won ~60% of 5M-25M cells. Two things changed under it: node visits became cheap (engine 6), which
removes what nesting was buying, and the defaults stopped being depth-capped (`d72c117`).

**Measured selection still pays; nesting adds little.** Regret of always using one structure vs
the per-cell best (69 cells):

| always use | geomean | median | p90 | worst |
|---|---|---|---|---|
| bih_default | 1.38 | 1.28 | 1.86 | 3.85 |
| kdtree_default | 1.42 | 1.30 | 1.99 | 3.74 |
| quadtree_default | 1.47 | 1.35 | 2.70 | 3.30 |
| octree_default | 1.62 | 1.58 | 2.32 | 2.82 |
| bvh_default / lbvh_default | 2.77 | 1.57 | 24.2 | 285 |
| best default per cell (oracle) | 1.10 | 1.08 | 1.26 | 1.36 |

**Serial-GA diagnostic (8 cells at 25M, `results/eval_traces/matrix_e6_serial`, `--parallel 1`).**
Timing candidates 8 at a time makes the GA's selection noisier (aneto 0.73 -> 1.04, sanandreas
0.75 -> 0.95 once timed serially), but even serial the searched winner is at parity with the best
single: single/searched 0.77-1.07, median ~1.0. The "nested" winners that do win are a deep single
structure with a one-level tail (`ot11l32_qt1l32`, `qt11l32_ot1l32`) or a single deeper than the
tuned grid reaches (`qt12l64`). **Prefer `--parallel 1` (or a serial re-rank of the top
candidates) when the search's ranking is the result.**

**Server, 100M-500M so far (`results/eval_traces/matrix_e6_makinon2`, held out):** SanAndreas -
`quadtree_default` best at 100M/250M/500M, searched nested 1.12-1.22x slower. Waikato - at 250M and
500M the GA's `evolved_g3_i15_qt12l32` (a single quadtree, deeper than the tuned grid's qt10) wins
by 6-15% over quadtree_default, CIs overlapping. Same pattern: the search's value at scale is
off-grid single tuning, not nesting. 1B and GaliciaForest still running.

**Superseded by 2f: the nested schemas were depth-capped, so this was not a fair test of nesting.**

### 2f. 2026-10-04 — every searched schema was capped at 12 levels

`SchemaGenerationOptions::_maxDepth = 12` (`SchemaSearch.h`) caps the TOTAL depth of every
generated, mutated and crossed-over schema, and `buildPolicy.maxDepth` = the sum of the level
depths. That is the defect `d72c117` fixed for the default singles, never applied to the search
space. A nested kd schema can have at most 4096 leaves whatever the cloud size.

Evidence from `matrix_e6` (held-out rows): every nested GA winner sits at max depth 11-12 with p99
leaves of 200-2000 points against a requested 32-128, testing 2-4x more points per query than the
tuned single; the GA's winners keep pushing against the ceiling (`ot12`, `qt12`, `qt11l32_ot1`).
Of the 4456 distinct schemas the GA evaluated, 69% hit the cap with p99 leaves > 4x capacity and 60%
had a block with leafCapacity >= 2048 (the generator samples 32..32768, and an upper block's
leafCapacity ends the branch, it does not hand off to the next block). The hand-nested baselines
are capped at 8/10 levels (DomFountain 100M: 286 us, p99 leaf ~900k points). Server rows
(250M-1B) show the same: searched winners at depth 12, quadtree_default at 14-16. No cloud feature
predicts the old GA-vs-tuned outcome (|Spearman rho| <= 0.2 for size, planarity, density CV,
aspect, query mix), consistent with a handicapped search rather than a shape effect.

**Diagnostic, `scripts/nest_diagnostic.py` (15 cells, `D:\MDS_ab\nest_diag`).** Two-block schemas
(prefix qt 3/6/9, ot 2/4/6 or kd 6/12 levels, then a tail that runs until leafCapacity binds)
against uncapped singles (qt16, ot16, kd22 at the same leaves) plus the tuned winner and defaults.
All serial on one pinned core; best nested and best single picked on `_opt`, compared on `_test`.

| cell | best nested | best single (by opt) | nested/single | old GA winner/single |
|---|---|---|---|---|
| toronto_5M | qt6 -> ot, leaf 128 | qt16 l128 | **0.85** (CI-separated) | 1.25 |
| aneto_5M | qt9 -> ot, leaf 128 | qt16 l128 | **0.62** (0.82 vs best default) | 0.69 |
| toronto_50M | qt6 -> ot, leaf 128 | qt16 l128 | **0.91** (CI-separated) | 1.18 |
| ottobrunncampus_50M | qt3 -> ot, leaf 128 | ot16 l128 | **0.85** (CI-separated) | 1.15 |
| alhambra_50M | ot6 -> kd, leaf 128 | tuned kd19l128 | 0.81 (CIs overlap) | 1.10 |
| cathedral_25M, neugasse_5M, museum_1M | | | 0.96-0.98 | 1.06-1.32 |
| galiciaforest_es05_5M, alhambra_1M, domfountain_100M* | | | 1.01-1.03 | 1.17-1.48 |
| sanandreas_25M, waikato_hamilton_25M, lille_1M, solarplantation_25M | | | 1.06-1.09 | 1.08-1.45 |

Geomean nested/single 0.943 (median 0.98, range 0.62-1.09); the old capped GA winners against the
same singles: 1.19. \*DomFountain 100M first measured 1.87 for `qt6_kd` alone (5.07 us, same tested
points as on opt); a re-run in another order put all four finalists at 3.57-3.67 us, a tie. At
100M run-to-run drift reached ~30% for identical schemas, so 100M+ comparisons need interleaved
repeats, not one run per schema.

Reading: nesting wins where the cloud changes character with scale - 2.5D at large scale, 3D
detail at small scale (street MLS/TLS with facades, a campus, terrain with vegetation): a
quadtree over the plan, then an octree. Purely aerial clouds (SanAndreas, Waikato, SolarPlantation)
are 2.5D at every scale, so qt -> ot equals a quadtree and it ties. Nothing loses. To fix in the
search: an open-ended last block, one leaf capacity per schema (or upper blocks hand off), leaf
range ~16-512, uncapped hand-nested and tuned-grid (qt/ot) schemas, serial timing for the ranking;
the two-block grid above (~45 candidates, 30 s at 5M, 4 min at 50M) is itself a strong seed.

### 2g. Battery e7 — the search with the 2f fixes (started 2026-10-04, `5b4f9dc`)

What changed in the search (`normalizeSchemaForGeneration`, `twoBlockSeedCandidates`):
- the last block of every generated/mutated/crossed/repaired schema is open-ended (16 levels for
  quad/oct, 22 for binary splits); `--generated-max-depth` (12) now caps the prefix only;
- one leaf capacity per schema, drawn from 16..512 (was per block, 32..32768);
- the GA's initial population gets the deterministic two-block grid of 2f (38 schemas) on top of
  the random ones (`--no-seed-grid` turns it off);
- hand-nested baselines and the tuned grid gained uncapped versions (`octree_kdtree` 4+22,
  `quadtree_octree` 4+16; `tuned_{qt16,ot16,kd22}l{64,128}`). Equivalence goldens captured before
  this hold the capped hand-nested schemas under the same names (config change, not engine);
- `run_cells.py`: `--parallel 1` by default; `--cpu`/`--numa-node` pin every run (the workstation's
  hybrid CPU let unpinned runs land on E-cores: up to 1.44x between runs of one schema; pinned
  median 1.06x); TEST is 3 separate processes with rotated order, merged into `<cell>_test.csv`
  (readers take the per-schema median); TEST also gets the GA's best NESTED schema, so every cell
  answers nested vs single on held-out queries.

Where it runs:
- workstation, 63 cells 1M-50M, core 4: `results/eval_traces/matrix_e7`, log `D:\MDS_ab\battery_pc_e7.log`
- makinon2 (`src_e7`, `build_e7`), `results/matrix_e7`, `run_battery_e7_streams.sh`: stream a
  (core 2, node 0) the nine 100M searches incl. the six workstation 100M scenes, then SanAndreas
  500M/1B; stream b (core 20, node 1) the three 250M searches, then GaliciaForest/Waikato 500M/1B.
  So <=50M is measured on the workstation and >=100M on the server; never mix them in one table.
- makinon2 also runs `scripts/nest_diagnostic.py` on the three 250M cells (core 35, node 1),
  `results/nest_diag`.

### 2h. 2026-10-04 — e7's losses were selection noise; selection and timing fixed (battery e8)

e7 (49 workstation cells, 1M-25M) answered nested vs single at held-out geomean 0.994 (25M 0.964;
ground scans 0.960, aerial 1.025, forest 1.036), but the GA's final pick against the tuned single
was a coin flip (1.003; 19 wins, 18 losses). The 16 losses were **not depth** (no nested p99 leaf
above 2x capacity). Two causes:
- **Selection on the wrong measure.** The GA and singles arms ranked on `avg_latency_ms`, the
  instrumented cold pass 0 (`--measure-repeats 1`); TEST reported telemetry-off passes. Rank
  correlation of nested/single between opt and test: 0.27. In 9 of the 16 losses the nested schema
  had looked >= 3% better on opt.
- **Shared machines.** The workstation also runs other jobs (an NLOS simulation on ~4 cores, a
  python job on 1); makinon2 had another user on ~70 of 72 threads (load 77). For whole
  measurement windows a core runs 40-50% slow (SMT co-runner or frequency): the per-query median
  shifts, the CI stays tight. Identical trees (same digest; e.g. waikato_5M `qt9 -> ot16` vs `qt16`)
  timed 0.81-1.47 us across processes, 1.03-1.30x apart sequentially.

Fixes (`496025d`):
- the score reads the telemetry-off passes when `--measure-repeats >= 2` (arms run 3 repeats,
  `--min-timed-ms 30`);
- `engine-check --interleave` builds all schemas, then rotates one pass per schema with an untimed
  warm sweep first. Identical trees then time 0.971-1.028 apart, under the same load that gave
  0.74-1.05 sequentially;
- `run_cells.py` confirmation step: the top 3 nested and top 3 singles (deduplicated by a
  geometry-only structure digest) are timed interleaved on the opt half in 3 processes. Nested is
  the pick only if it is >= 3% faster in the median AND in every process;
- held-out timing is interleaved too (`<cell>_test_timing.csv`, `<cell>_pick.json`), summarized by
  `scripts/summarize_nesting.py`.

**Validation on 21 e7 cells (the 16 losses + 5 wins, e7's arms re-used,
`D:\MDS_ab\e7_confirm`):**
- pick/best single held-out geomean 0.978: **0 losses**, 4 wins (Toronto 25M 0.88, Neugasse 5M
  0.89, OttobrunnCampus 25M 0.91, Cathedral 25M 0.94), 17 ties;
- by scan type: ground 0.954, aerial 0.999, forest 1.000;
- price of the rule: Alhambra 25M fell back to the single although nested measured 0.961 held out.

New structures (`eb52a0e`, CPU only, equivalence vs golden_v1: 0 result mismatches, only the two
uncapped hand-nested configs change structure), first numbers at 5M (interleaved, vs `qt16l128`):
- `sliding_midpoint` kd (nanoflann's rule, midpoint of the points' extent): Toronto 1.01,
  SanAndreas 1.02; 10-20% faster than our median kd, best radius of all (Toronto 1.83 vs 2.33 us).
  Cutting the cell's midpoint instead (the first attempt) peeled points off edges and hit the
  depth cap: 10x tested points;
- `splitAt: tight` octree/quadtree: quadtree 0.97-0.99 (a tie), octree worse;
- `STR` (fanout 27/64): 2.3-2.9x slower. Wide nodes need SIMD child-box tests to pay off; the
  Boost R-tree (`17fadc1`) is slow here for the same reason.

Battery e8 = e7's cells with all of the above: workstation `results/eval_traces/matrix_e8` (log
`D:\MDS_ab\battery_pc_e8.log`); makinon2 after its e7 streams.

### 2i. 2026-10-05 — first e8 results: nesting wins on ground scans, never loses on aerial

**Workstation, all 63 cells (1M-50M; the last six 50M cells were re-queued after the session
running the battery ended), `scripts/summarize_nesting.py results/eval_traces/matrix_e8`.**
pick/best single (both chosen on opt), held-out, interleaved:

| | cells | pick/single | wins / ties / losses |
|---|---|---|---|
| 1M | 17 | 0.976 | 3 / 14 / 0 |
| 5M | 16 | 0.955 | 4 / 12 / 0 |
| 25M | 16 | 0.947 | 5 / 11 / 0 |
| 50M | 14 | 0.936 | 7 / 6 / 1 |
| ground scans (TLS/MLS street, heritage) | 32 | **0.911** | 19 / 13 / 0 |
| aerial (terrain, urban, industrial) | 19 | 1.000 | 0 / 19 / 0 |
| forest / vegetation | 11 | 1.004 | 0 / 10 / 1 |
| all | 63 | 0.954 | 19 / 43 / 1 |

Against the best single measured on the test half itself (an oracle that favours singles): 0.957.

The winning nested schemas share one shape: a plan-view quadtree (or octree) for 2-9 levels, then a
sliding-midpoint kd tail at leaf 128 (`qt6l128_kd22l128ap=sm` and neighbours):
- Neugasse 0.78-0.81 at every size, Toronto 0.79-0.85, Cathedral 0.83-0.86, DomFountain
  0.85-0.90, OttobrunnCampus 0.85-0.86 at 25M/50M;
- the single sliding-midpoint kd is in the tuned grid and the seeds, so the gain is the nesting;
- aerial cells pick a single quadtree every time;
- the one loss is GaliciaForest 50M, 1.046 (the archipelago).

**makinon2 e7 (old selection, mostly under the other user's load; `results/eval_traces/matrix_e7_makinon2`),**
GA best nested / tuned single, held-out:
- ground scans at 100M: OttobrunnCampus 0.87, Lille 0.89, DomFountain 0.96, ES05 plot 0.97; Alhambra 1.07;
- aerial (SanAndreas, Waikato 100M-250M): 0.98-1.01, with 500M transfers 1.12-1.14;
- GaliciaForest archipelago 100M-500M: 1.34-1.77, poor nested picks of the kind the e8 confirmation
  step rejects.

**Uncapped 250M diagnostic (`results/eval_traces/nest_diag_makinon2`):** GaliciaForest 1.03,
Waikato 1.00, SanAndreas 1.09. All three are aerial, matching the aerial ties above.

makinon2 e8 runs from 2026-10-05 (`src_e8`, `build_e8`, `run_battery_e8_streams.sh`,
`results/matrix_e8`, copied to `results/eval_traces/matrix_e8_makinon2`); the server is quiet again
(load ~1). First cells, pick/best single held-out: DomFountain 100M 0.893 (qt9 -> kd3 -> kd22, all
sliding-midpoint kd, leaf 128) and Alhambra 100M 0.967 (qt3 -> sliding-midpoint kd22, leaf 64; it
lost 1.07 in e7). Same shape as the workstation winners.

### 2j. 2026-10-06 — the ties are real, depth starved the singles, the GPU path never measured batches

**Why 43 of e8's 63 workstation cells are ties.** The safe pick hides nothing: across the ties the
best nested finalist vs the best single is 1.013 on opt and 1.011 held-out, 34 of the 43 nested
finalists were already no faster on opt, and held out only Lille 25M would have won (0.929). On
aerial clouds nesting rebuilds the quadtree (`qt6_ot16` vs `qt16`: same tested points, same visited
nodes). Nesting pays where a scene has distinct regimes (ground and facades, dense near the scanner,
sparse far away), not on 2.5D terrain.

**Depth starved the singles (fixed in da34c56).** The generator's open-ended last block stopped at
22 levels (kd/BVH) or 16 (quadtree/octree), and the tuned grid topped out at the same depths.
Sliding-midpoint kd halves space, not points, so dense TLS/MLS clusters need more:
- 41 of the 105 e8 finalists ended at their cap, several with the p99 leaf over 2x capacity
  (OttobrunnCampus 25M `kd22l128sm`: p99 779, max 30,781, capacity 128);
- nested schemas got prefix + 22 levels, i.e. extra depth for free.

Pilot (`D:\MDS_ab\depth_pilot`, TEST-identical interleaved timing, core 4): each cell's e8 best
nested and best single, plus copies with the last block at 32/48 (kd) or 20/24 (quadtree/octree).

| cell | nested/single at e8 depth | at equal depth | deeper single |
|---|---|---|---|
| OttobrunnCampus 25M | 0.858 | 0.992 | 0.864x |
| ParisLux 50M | 0.967 | 1.028 | 0.915x |
| Cathedral 1M | 0.961 | 0.987 | 0.973x |
| Lille 50M / Alhambra 50M | 0.987 / 0.985 | 0.989 / 0.978 | 0.98-0.99x |
| Neugasse 5M | 0.783 | 0.762 | 1.000x |
| Toronto 25M | 0.791 | 0.791 | 1.000x (qt16 -> qt20/24: slower) |
| SanAndreas 25M / Waikato 50M | 1.012 / 1.008 | 0.980 / 1.004 | 1.000x |
| GaliciaForest 50M | 1.078 | 1.055 | 0.969x |

- The wins against a starved single (0.86-0.97) were depth artefacts; the big wins (Neugasse,
  Toronto) are not, and grow slightly.
- kd trees end at 27-34 levels at 25M-50M; quadtrees at 11-18. 48 and 24 leave room for 1B.
- A shallow single can win on purpose: `qt10l32` deepened to 20 levels gets slower (1.39 -> 1.54 us),
  so the shallow tuned variants stay.
- Identical trees (equal digest) timed up to 4-6% apart within one interleaved process: the noise
  floor sits near the 3% nesting margin.

The fix: `openEndedDepth` 48 (kd/BVH) and 24 (quadtree/octree); deep tuned singles
`tuned_kd48l{64,128}sm`, `tuned_{qt,ot}24l{64,128}`, `tuned_qt24l128tc` (55 in the grid);
`octree_kdtree` 4 + 48, `quadtree_octree` 4 + 24; a node whose points all coincide stays a leaf
(no split can separate them; with deep caps they chained down to the cap). Equivalence vs the
previous snapshot (Tier S): 0 result and 0 counter mismatches; trees change only for
`octree_kdtree` (deepened) and `edge_leaf_capacity_one` (duplicates). `summarize_nesting.py` gained
a `starved` column.

**The GPU was never in the battery, and the legacy GPU path cannot measure batches.** run_cells
passes `--evaluator cpu`; confirm and TEST use engine-check, which is CPU-only. The legacy CUDA path
(`--evaluator cuda`) is not usable for a batch comparison:
- `avg_latency_ms` is each query's in-kernel `clock64` span while thousands run concurrently, so
  each span is about the whole kernel: that is where docs/evaluation.md's "1.5-4.5 s per query" at
  Alhambra 100M and "25x slower than CPU" come from. Do not cite them.
- A node with one non-empty child becomes a leaf (KDTree.cu:353, MixedTree.cu:1036, Octree.cu,
  QuadTree.cu): the 25M GPU kd-tree has 2,350 leaves, max 1.3M points.
- kNN is a brute-force scan of all N points except on kdtree/bih with K <= 16; range and radius only
  count; only `mixed` honours nesting (depth <= 12); stacks of 128/256 drop children on overflow;
  nothing checks GPU results against the CPU.
Replacement in progress: build on the CPU, upload that exact tree, query it with new batch kernels
(`PointGpuIndex`), next to an all-core CPU batch baseline (plan Part B).

**Battery e9** (e8 protocol, the depth fix, 55 tuned singles) started 2026-10-06: workstation 63
cells -> `results/eval_traces/matrix_e9` (`D:\MDS_ab\bin_e9`, log `D:\MDS_ab\battery_pc_e9.log`);
makinon2 `src_e9`/`build_e9` from 7a3a425 -> `results/matrix_e9`. Server e8 was stopped with seven
100M cells done (`results/eval_traces/matrix_e8_makinon2`; SanAndreas 100M incomplete): nested won
all six complete cells (0.82-0.97), two of them against a starved single (Alhambra, ES05).

### 2k. 2026-10/11 — e9: with fair depth, nesting never wins

**Correction to 2j.** The depth pilot deepened only each cell's e8 best single (a median kd on
Neugasse, a quadtree on Toronto) and concluded that the big wins (Neugasse 0.76, Toronto 0.79)
survive. It never deepened the sliding-midpoint kd, which e8 capped at 22 levels. At 48 levels that
single ties the nested quadtree -> sliding-midpoint kd everywhere: the e8 nesting advantage was the
extra depth the quadtree prefix bought the kd tail.

**e9** (e8 protocol, open-ended depth 48/24, 55 tuned singles): makinon2 complete (18 cells, 100M-1B,
`results/eval_traces/matrix_e9_makinon2`); workstation 55 of 63 (`results/eval_traces/matrix_e9`; the
run stopped at lille_50M when the Claude session hosting it ended, 8 x 50M cells missing).

| | cells | nested picked | best nested / best single (held-out) | pick / oracle |
|---|---|---|---|---|
| 1M / 5M / 25M / 50M | 17 / 16 / 16 / 6 | 0 | 1.015 / 1.013 / 1.006 / 1.017 | 1.001-1.009 |
| 100M / 250M / 500M / 1B | 9 / 3 / 3 / 3 | 0 | 1.012 / 1.006 / 1.006 / 1.004 | 1.000-1.008 |
| ground / aerial / forest | 31 / 25 / 16 | 0 | 1.020 / 1.005 / 1.002 | 1.002-1.004 |
| all | 73 | 0 | 1.011 | 1.003 |

Winners: a deep sliding-midpoint kd (`kd48l64sm`/`kd48l128sm`) on ground and forest scans, a quadtree
(`qt16`/`qt24`, `qt10l32`) on aerial ones. Selection still works (pick within 0.3% of the held-out
oracle); what it selects is a single, tuned structure.

**Build and size** (e9 finalists): the best nested schema builds 8% faster than the best single
(geomean 0.922; >10% faster in 32 of 57 cells, >10% slower in 10) with 6% fewer nodes, at 1.3% higher
latency. A build-weighted objective could favour nesting slightly; latency alone does not.

**Other workloads** (`scripts/workload_sweep.py`, makinon2, 100M clouds, panel of 4 singles and 5
nested schemas all with open-ended depth, 7 workloads built on each test trace's centres, 2 rotated
processes; `results/eval_traces/workload_sweep_makinon2`). Best nested / best single:

| workload | geomean (6 clouds) | range |
|---|---|---|
| base trace | 1.005 | 0.996-1.016 |
| radius 4x / 16x | 1.022 / 1.018 | 1.003-1.096 / 0.996-1.093 |
| kNN k=64 | 1.002 | 0.989-1.008 |
| range boxes 8x | 1.010 | 0.989-1.093 |
| count boxes 64x | 1.020 | 0.992-1.109 |
| mixed | 1.011 | 1.000-1.040 |

No workload where nesting pays; on Alhambra it loses 9-11% on larger queries. Leaf 512 wins the
larger queries for singles and nested alike.

**GPU batches** (`--mode batch-check`, RTX 4080 SUPER, same panel, Toronto and Neugasse 25M, half of
the planned sweep; 0 mismatches against the CPU in every run): best nested / best single ranges
0.78-1.05 with no consistent sign. Heavy queries are the thread-per-query kernels' weak spot: the
mixed workload (with 8x range and 64x count boxes) took 1,025 ms per 100k queries on Toronto against
3.4 ms for the base trace, one thread scanning a large box alone.

### 2l. 2026-11-01 — why nesting never wins: expected, not a defect

The user asked whether the e9 tie is expected or a failure on our side; the design follows
Ogayar-Anguita, López-Ruiz, Rueda-Ruiz, Segura-Sánchez, "Nested spatial data structures for optimal
indexing of LiDAR data", ISPRS J. 195 (2023) 287-297 (PDF read in full on 2026-11-01).

**1. In this engine a nested schema is not a different data structure.** Every level is built into
one pool of identical 32-byte nodes (tight box, first child or first point, count) with one SoA
point array in tree order, the same AVX2 leaf scans and the same depth-first layout
(`PointSpatialIndex.cpp` flattenTree/buildNode). The level boundary only changes the split rule
applied while a node is built (`locateInPlan`); the four searches are one generic stack walk that
tests every child's tight box and never branches on primitive type. A grid node is 27 (RegularGrid)
or 64 (HGrid) box-tested children with no direct addressing, and the CPU generator never emits grids.

**2. Nesting does the only thing it can: fewer node visits.** Over the 73 e9 cells, best nested vs
best single finalist (instrumented pass of `<cell>_test.csv`):

| | geomean | TLS/MLS scans |
|---|---|---|
| nodes visited per query | 0.976 | 0.80-0.92 |
| points tested per query | 1.061 | ~1.00 |
| nodes in the tree | 0.915 | |
| build time | 0.941 | |
| latency | 0.994 | |

**3. The levels where nesting acts are free.** The traces are stride-subsampled from scan order
(DomFountain 100M: consecutive centres 0.12 / 0.43 / 2.6 m apart at p10/p50/p90 in a 325 x 445 m
scene), engine-check replays them in that order, and the queries are tiny (k = 16; radii sized to
return ~13-280 points; 12-45 node visits; 1-3 leaves). The root-to-mid-level path is in cache for
every query; the time goes to the cold bottom levels and the leaf scans, which are identical for
every schema. Build and memory, where nesting is 6-8% better, carry zero weight in the objective.

**4. What the ISPRS paper measured.** It is an out-of-core storage framework: every node is a LAZ
file or a database record, global depth 10, and leaves hold thousands to millions of points
(Table 3: 2,876-154,719 leaf nodes for 48M-1.14B points, 787-33,015 points per leaf on average).
The benchmark is 100 random boxes of 20-40% of the dataset per axis, scored as nodes loaded (Fig. 7)
and as throughput in Mpoints/s including opening and LAZ-decompressing the node files (Fig. 8).
The multi-structure (2D grid -> quadtree -> 3D grid -> octree) loads far fewer nodes (Manhattan
~1,500 vs ~8,600 for the quadtree and octree), has fewer nodes and builds "somewhat faster"; the
paper itself calls the throughput differences "not very significant" (about 0.9-1.2x). The singles
are the same framework with one structure, not external libraries.

So the paper's effect is exactly the one e9 reproduces: fewer, better-fitted coarse nodes. There it
became time because a node cost a file open plus decompression; here a node costs one hot cache
line and a query touches 1-3 leaves of 128 points, so the same saving is worth about 1%. The
unified pool gives every single structure the locality and flat leaves that nesting supplied in the
paper, and a 48-level sliding-midpoint kd already is a grid of well-fitted subtrees.

**Diagnostics** (`scripts/order_sweep.py`, makinon2, build_e9, 100M cells, 2 rotated processes,
3000 queries; `results/eval_traces/order_sweep_makinon2`):

*Query order* (panel of 4 deep singles, 5 nested, 5 grid-topped schemas; the test trace as recorded,
the same queries shuffled, shuffled zero-extent count queries = a pure root-to-leaf descent, and
shuffled radius queries at 4x the trace radius). Best nested / best single:

| cell | trace | shuffled | descent | shuffled, radius 4x | shuffle cost, best single |
|---|---|---|---|---|---|
| DomFountain 100M (TLS) | 0.980 | 0.993 | 1.040 | 1.018 | +9% |
| Lille 100M (MLS) | 1.023 | 1.003 | 1.039 | 1.027 | +12% |
| SanAndreas 100M (ALS) | 1.000 | 1.004 | 0.996 | 0.997 | +5% |

Shuffling makes the top of the tree cold and slows every schema by the same 5-12%; the nested
prefix still visits 6-14% fewer nodes (kd tail) and still buys nothing. On the pure descent the
best structure is the plain quadtree (1.36 / 1.15 / 1.06 us), and the nested qt9 -> ot24 that
visits the same number of nodes is 4% slower on the ground scans. A 1M smoke test had shown the
nested kd 11% faster on cold descents (0.21 vs 0.24 us): at 1M the whole tree sits in L2/L3, so a
saved node visit is a saved cache hit; at 100M the saved visits are the hot top levels and the cost
is the cold bottom, which both trees share.

*Grid top level* (RegularGrid 27 cells and HGrid 64 cells, 1-2 levels, box-tested, over the same
tails). Best grid-topped / best single, recorded order: DomFountain 1.018, Lille 1.031,
SanAndreas 1.151 (`grid1_kd48l128sm`); the two-level HGrid (4,096 cells) 1.05 / 1.18 / 1.49. A grid
root never wins; a direct-addressed grid could at most remove the 27-64 hot box tests at the root,
which the descent variant bounds at well under 0.1 us.

*The ISPRS regime* (depth <= 10, leaves 4,096 and 16,384 points, 100 boxes of 20-40% of the extent
per axis, `range` and `count_range`):

| cell | best single | nested (2D grid -> qt3 -> 3D grid -> ot5) / octree: range, count, nodes visited | qt5 -> ot5 / quadtree |
|---|---|---|---|
| DomFountain 100M | `ot10l4096` 16.8 ms | 1.101, 2.38, 2.09x | 0.866 |
| Lille 100M | `ot10l4096` 19.5 ms | 1.032, 1.54, 1.51x | 0.868 |
| SanAndreas 100M | `ot10l4096` 24.1 ms | 1.017, 1.32, 1.14x | 0.997 |

The paper's comparison reproduces in part: nesting an octree under a quadtree beats the quadtree
(0.87 on the ground scans, as the paper's multi-structure beat its quadtree), but the plain octree
beats both, and the paper's full 2D grid -> quadtree -> 3D grid -> octree visits 1.1-2.1x more nodes
than the octree. In the paper the multi-structure loaded fewer nodes than the octree because its
octree cells were implicit (loose) and its grid and quadtree trimmed the empty space above and
beside the buildings. Our engine stores tight bounds on every node and never creates empty children,
so a single octree already skips that space, and the extra levels only add nodes.

**Verdict.** Expected, not a defect. (a) Nesting in this engine is a split rule, not a storage
scheme; the unified pool, tight boxes, empty-node removal and flat SoA leaves give every single
structure what nesting supplied in the paper. (b) Its one measurable effect, fewer coarse nodes,
is reproduced (0.915x nodes, 0.976x visits, 0.94x build) but is worth ~1% of an in-memory query
whose cost is the cold bottom levels and the leaf scans; in the paper it was worth file opens and
LAZ decompression. (c) Cold query order, grid roots and the paper's own regime do not change the
sign. (d) The search works as designed: it is allowed to return a single primitive and does so
within 0.3% of the held-out oracle. Nesting could still pay where a node visit is expensive
(out-of-core nodes, network storage, a GPU traversal counted in divergent warps) or when build
time and memory carry weight in the objective; neither holds for the in-memory latency battery.

Raw outputs: `results/eval_traces/order_sweep_makinon2/<cell>/{base+grid,paper}_<variant>.csv`,
reports `report_base+grid.txt`, `report_paper.txt`; server copies deleted after download.

### 2m. 2026-11-02 — two papers: the levers and the cost model (Paper A), the GPU pilot (Paper B)

Decision (plan file, 2026-11-02): Paper A for C&G, the measured study of what matters in point-cloud
indexing with a per-node vs per-point cost model; Paper B for HPG 2027, GPU batch queries with the
schema chosen for the hardware, gated by a fan-out pilot on makinon2.

**Levers** (`scripts/lever_decomposition.py`, e9 held-out chain over 73 cells; latency ratios, >1 =
the later lever is faster; `<cell>_test.csv` holds the library defaults at leaf 32 and the finalists,
timed interleaved where both sides were contenders):

| group | cells | octree default / best primitive at defaults | kd default / same | best primitive / best tuned single | nested / single | octree default / best single |
|---|---|---|---|---|---|---|
| ground scans | 31 | 1.175 | 1.124 | 1.391 | 1.020 | 1.610 |
| aerial | 25 | 1.533 | 1.571 | 1.067 | 1.005 | 1.703 |
| forest / vegetation | 16 | 1.504 | 1.103 | 1.127 | 1.002 | 1.707 |
| all | 73 | 1.360 | 1.255 | 1.210 | 1.011 | 1.659 |

The library octree is 1.66x slower than the best single, the median kd 1.53x. Choosing the primitive
is worth 1.26-1.36x (nearly all of the aerial gain), tuning leaf, split rule and depth another 1.21x
(nearly all of the ground-scan gain), composition nothing. With a leaf-128 octree as the default
(opt-half chain, instrumented) the whole gain is 1.41x: half of the "default" penalty is leaf 32.

**Cost model** (`scripts/cost_model.py`): t = c_node * visited + c_point * tested + c_out * returned,
fitted over the 184-220 opt-half single arms per cell (median R^2 0.98): c_node ~ 20-25 ns per visited
node, c_point ~ 0.4 ns per tested point on the workstation (~1 ns on makinon2), c_out ~ 100 ns per
returned point acting as the per-query constant. A query visits 20-100 nodes, so traversal is a third
to a half of its 1-5 us at best. The nested finalists visit 5% fewer nodes at the median (more on
aerial cells), worth under 4% of latency in nearly every cell; the c_node at which a 20% visit saving
would give a 10% speed-up is 22-30 ns at <= 50M and 65-95 ns at >= 100M, at or above the measured
value. The 13-15-schema `_test.csv` fits are ill-conditioned (regular_grid_default alone pins
c_point) and are not used.

**GPU pilot** (`scripts/gpu_pilot.py`, `--mode batch-check` with the new warp-per-query kernels,
backend `gpuwarp`, commit 067c009; GPU==CPU and warp==thread checks pass in `--run-tests` on
makinon2). Panel: 2/4/8/27/64-way singles at leaf 32/128/512 (kd sm, quadtree, octree, STR 27/64,
RegularGrid, HGrid) plus wide-top/narrow-tail and narrow-top/wide-tail nested schemas at leaf
128/512, 35 schemas; workloads base, radius4, range8; batches 10^4-10^6; backends cpu1, cpuN (72
threads), gpu, gpuwarp; Quadro RTX 6000. Cells: DomFountain, SanAndreas, GaliciaForest at 25M and
100M (25M clouds copied to makinon2 for this). Smoke on DomFountain 1M (tree resident in cache, not
the regime of interest): 0 mismatches; gpuwarp 2.7x gpu on the base trace and 1.5x on range8 at 10k
queries; GPU best/worst schema 2.5-5x; no mixed schema beat the best uniform on the GPU there.
**GPU pilot results (2026-11-03, all 6 cells; Quadro RTX 6000 via CUDA_DEVICE_ORDER=PCI_BUS_ID;
`python scripts/gpu_gate.py results/eval_traces/gpu_pilot_makinon2`).** The first attempt ran on the
2080 Ti by mistake (CUDA numbers GPUs fastest-first) and lost workloads to out-of-memory crashes; it
is kept apart in `results/gpu_pilot_2080ti` on makinon2. 0 mismatches in every row; GPU batches whose
results exceed device memory (1M range8 queries on DomFountain) are skipped. Batches 100k and 1M (10k
left out: 72 threads on 10k queries is mostly fork cost), geomean over the 34 cell x workload x batch
rows:

| measure | geomean | range | reading |
|---|---|---|---|
| GPU best / worst schema (warp kernels) | 8.5x | 1.3-178x | a bad schema is very bad on the GPU |
| GPU throughput of the CPU's pick / GPU best | 0.89 | 0.47-1.00 | 25M cells 0.83 (min 0.47), 100M cells 0.96 (min 0.82) |
| best nested / best single on the GPU | 0.935 | 0.76-1.084 | composition does not pay on the GPU either |
| warp / thread kernels (best of each) | 1.45x | 1.05-2.14x | |
| best GPU / best 72-thread CPU, end to end | 1.23x | 0.25-5.3x | below 1 on most radius4/range8 rows |
| same, kernel time only | 5.3x | 1.5-14x | |
| best depth-capped single / best other single | 0.61 | 0.04-2.1 | wins only on SanAndreas 25M (2.1x) and GaliciaForest 25M radius4 (1.2x) |

End to end includes the query upload, the sort, both kernels and the download of the result indices;
the tree upload is excluded. On the GPU-best schema at the largest batch, everything but the kernels
is 22-30% of the time on the base traces (12-184 indices per query) and 77-95% on radius4/range8
(145-27,575 per query; DomFountain 100M radius4 moves 17.6 GB for 1M queries, ~6.5 GB/s). Kernel plus
sort alone is 1.4-3x the 72-thread CPU on the base traces and 4.4-11.8x on radius4/range8.

The CPU's pick loses most at 25M: on SanAndreas 25M the 12-level quadtree is 1.8-2.1x faster on the
GPU than the CPU's `qt24l32`. At 100M the CPU picks the same depth-capped tree itself, and the CPU
and GPU choices agree to within 4% on average (the larger tree no longer fits the CPU caches either:
a hypothesis, not measured). The best mixed result is `kd6_hgrid64l128sm` on SanAndreas 100M radius4,
1.084x at 100k and 1.022x at 1M. Gate: (ii) holds; (i) holds at 25M, but is within 4% on average at
100M; (iii) fails. A GPU paper built on schema selection is not supported. What stands is the kernel
margin over the CPU, the warp kernels, and the result volume as the end-to-end bottleneck.



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

## 5. Paper writing checklist (`paper/`)

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

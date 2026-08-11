# Evaluation: the optimizer's synthesized index (nested or single) vs. baselines

> **SUPERSEDED — do not quote any number in this file.** This is the 12-cell synthetic
> study. It has been replaced by the trace-driven evaluation (`testing.md`, and the
> matrix battery in `results/eval_traces/matrix/`), which uses recorded application
> traces, a held-out `_test` half, and tuned rather than default single-primitive
> baselines. Its numbers also predate the 2026-08-06 audit: the query radii were
> computed by the biased spacing estimator, the kd and grid baselines were crippled by
> depth-capped tuning grids, and the binary it measured was stale. The file is kept for
> the method description and for the record of what was claimed when. See METHOD.md §7.

This is the focused evaluation of the project's central claim. It compares, on a common
query workload, three ways of choosing a point-cloud spatial index:

1. **Best single primitive** — the fastest of nine standard single-structure indexes
   (quadtree, octree, kd-tree, BVH, BIH, LBVH, regular grid, hierarchical grid, Karras octree).
2. **Hand-designed nested** — six fixed multi-level schemas authored by hand
   (`quadtree_octree`, `octree_kdtree`, `urban_hybrid`, `grid2d_quadtree_grid3d_octree`,
   `urban_grid_hybrid`, `adaptive_quadtree_octree`).
3. **Optimizer-synthesized** — the schema the **genetic-algorithm optimizer** evolves per
   (dataset, workload) (`--optimize-schemas`, the default search: NSGA-II Pareto selection +
   crossover + successive-halving rungs + CMA-ES refinement). Its search space spans single
   primitives *and* nested combinations, so it returns whichever wins — a tuned single
   primitive or a nested schema.

against the per-cell **oracle** (the best of everything measured).

## Claim

> A workload-aware optimizer synthesizes the best spatial index per cell — a tuned single
> primitive when nesting does not help, a nested schema when it does — landing near the
> per-cell oracle and yielding a Pareto front over latency / build / memory / imbalance.
> The advantage of nesting concentrates on heterogeneous data and higher-volume query
> workloads, and shrinks to a tie on small, homogeneous ones.

## Result (headline)

Measured on CPU over **3 synthetic dataset families × 4 workload profiles = 12 cells**.
Latency is the seed-averaged mean over `--confirm-seeds 3` where confirmed, else the
single-run average; lower is better.

| Dataset | Workload | Best single | Hand-nested | Optimizer (GA) | Oracle | Speedup opt/single | Regret vs oracle |
|---|---|---|---|---|---|---|---|
| synthetic_facade | knn_heavy | bvh 0.0046 | octree_kdtree 0.0124 | evolved bvh 0.0048 | 0.0046 | 0.95× | 5.1% |
| synthetic_facade | mixed | bih 0.0051 | octree_kdtree 0.0115 | xover bvh+ot+kd (nested) 0.0056 | 0.0051 | 0.92× | 8.5% |
| synthetic_facade | range_heavy | bvh 0.0047 | octree_kdtree 0.0059 | evolved bvh+qt+bvh (nested) 0.0043 | 0.0043 | **1.10×** | 0.0% |
| synthetic_facade | volume_small_medium | octree 0.0446 | octree_kdtree 0.0367 | **evolved ot10+kd+kd (nested) 0.0263** | 0.0263 | **1.70×** | 0.0% |
| synthetic_flat_terrain | knn_heavy | bvh 0.0042 | quadtree_octree 0.0062 | gen kd9+qt (nested) 0.0045 | 0.0042 | 0.93× | 7.2% |
| synthetic_flat_terrain | mixed | bvh 0.0048 | quadtree_octree 0.0055 | gen bvh11+qt (nested) 0.0052 | 0.0048 | 0.92× | 8.5% |
| synthetic_flat_terrain | range_heavy | lbvh 0.0047 | quadtree_octree 0.0040 | xover bvh+ot+bvh (nested) 0.0040 | 0.0040 | 1.17× | 1.6% |
| synthetic_flat_terrain | volume_small_medium | karras_octree 0.0456 | octree_kdtree 0.0356 | **evolved ot11+kd (nested) 0.0259** | 0.0259 | **1.76×** | 0.0% |
| synthetic_urban_mixed | knn_heavy | karras_octree 0.0045 | quadtree_octree 0.0045 | gen ot11+bvh (nested) 0.0043 | 0.0043 | 1.05× | 0.0% |
| synthetic_urban_mixed | mixed | octree 0.0028 | quadtree_octree 0.0028 | gen ot11+bvh (nested) 0.0024 | 0.0024 | 1.14× | 0.0% |
| synthetic_urban_mixed | range_heavy | karras_octree 0.0015 | octree_kdtree 0.0015 | gen ot11+bvh (nested) 0.0014 | 0.0014 | 1.07× | 0.0% |
| synthetic_urban_mixed | volume_small_medium | lbvh 0.0192 | urban_hybrid 0.0256 | evolved ot5+kd (nested) 0.0164 | 0.0164 | 1.17× | 0.0% |

(latencies in ms; full machine-readable data in [results/eval_ga/report/](../results/eval_ga/report/) — `comparison.md`, `summary.json`. The **Optimizer (GA)** arm is the genetic algorithm's exported best schema per cell, re-measured uniformly next to the baselines.)

**Summary:** the optimizer beats the best single primitive in **8 of 12 cells**, geomean
speedup **1.13×**, mean relative regret vs. oracle **2.6%** — it lands within ~2–3% of the
per-cell oracle and *finds the oracle outright in 7 of 12 cells*. (For reference, the
`--auto-conditions` surrogate-filter that produced the previous headline scored 9/12, 1.12×,
3.2% regret; the GA tracks the oracle more closely at a matched evaluation fidelity.)

### What the optimizer selects

- On **`volume_small_medium`** (medium-extent range queries) the GA evolves **deep nested**
  schemas — a 10–11-level octree feeding kd-tree leaves — and wins clearly (**1.70–1.76×**),
  beating both the best single primitive and the hand-nested baselines.
- On **heterogeneous `synthetic_urban_mixed`** the GA evolves a *single nested schema*
  (`ot11l256_bvh1l32`, octree→BVH) that wins **all four workloads** (1.05–1.17×) and is the
  per-cell oracle each time — nesting pays off, and one synthesized schema generalizes across
  the cloud's workloads.
- On **small-query homogeneous cells** (`knn_heavy`/`mixed` over facade/terrain) a single
  primitive is already near-optimal, so the optimizer **ties or slightly loses** (0.92–0.95×).
  This is the honest case: there is little for nesting to exploit, and the optimizer does not
  manufacture a win.

Most winners are genuinely **nested multi-level schemas the GA evolved** (`evolved_*` from
mutation, `xover_*` from crossover) — so the optimizer is exercising the nested part of its
search space exactly where it helps, and a tuned single primitive only where nesting cannot.
Per-cell latency gaps are small at this µs scale (only `facade/range_heavy` is flagged
`ranking_confident`); the `volume_small_medium` (1.7×+) and `urban_mixed` (full-sweep) wins
are the substantive ones.

### Learned selector (illustrative)

Training `scripts/train_schema_selector.py` on the 216 measured rows (held-out family:
`synthetic_facade`) yields a feature-based score-ranker with best-schema accuracy 0.50 (top-2
0.75) and **mean relative regret 2.8%** vs. oracle — cloud/workload/schema features predict a
near-best schema, though the GA's more diverse nested winners are harder to rank than the
earlier surrogate-filter's tidier picks. This is illustrative only: with **3 dataset families**
the train/test split is tiny, so treat it as a sanity check, not a generalization claim.

## Comparability protocol

Every compared row shares one provenance: `--evaluator cpu`, `--score-objective latency`
(weights latency/build/memory/imbalance = 1/0/0/0), `score_is_final_latency=1`,
`score_mode=latency`, `--queries 1024`, `--query-seed 1337`, `--knn-k 16`,
`--synthetic-scale 300000`. `scripts/compare_methods.py` verifies this and reports
**"Distinct provenance settings: 1"** — the comparison is apples-to-apples.

The optimizer is run as a **separate discovery stage** (`--optimize-schemas`, the default
search; it exports a schema per cell via `--optimizer-output-dir`) at the **same 1024-query
fidelity** the headline re-measures at, so its exported pick and the baselines are scored on
the identical query set — no discovery-vs-measurement fidelity gap. This matters: an earlier
pass discovered at 256 queries and exported a 256-query pick that mis-ranked against the
1024-query re-measurement (costing ~3 points of regret and several cells);
`scripts/audit_schema_scores.py` flags exactly that kind of `effective_queries` mismatch,
which is why discovery and measurement are kept at matched fidelity here.

## GPU appendix — build vs. query acceleration

The same 18-schema set was re-measured with `--evaluator cuda --cuda-builder mixed`.
Of 216 rows, **144 ran GPU-`full`**; **72 fell back to CPU** (`unsupported_policy`/`cpu_fallback`)
because the CUDA `MixedTree` does not implement median/round-robin kd-tree split policies,
BIH, or adaptive leaf capacity. The fallback set: `kdtree`, `bih`, `octree_kdtree`,
`urban_hybrid`, `urban_grid_hybrid`, and two auto-tuned winners that use round-robin kd levels.

For the 144 GPU-runnable rows, GPU vs. CPU (geomean over all cells):

| Phase | GPU vs CPU | Reading |
|---|---|---|
| Index **build** | **~70× faster** on GPU | GPU is a strong bulk-build accelerator |
| **Query** latency | **~0.04× (≈25× slower)** on GPU | overhead-bound at this scale |

At 300k points with ~µs of work per query, GPU kernel-launch and host↔device transfer
overhead dominate the query path, so the CPU index is faster for these latency-bound
interactive queries; the GPU's clear, consistent win is **build time**. A GPU *query*
advantage would require much larger clouds and/or large batched query streams — left as
future work. This is why the **headline comparison is on CPU**: it is the only backend where
all schemas (including median/round-robin kd, BIH, adaptive, and entropy-conditioned ones)
run natively and are therefore mutually comparable.

## Real-data spot-check: Alhambra (100M points, GPU)

A spot-check on the real `Alhambra_100M.las` cloud (99,999,776 points; heterogeneous
architectural TLS scan), `mixed` workload. CPU is infeasible at this scale — a single
octree CPU build is ~53 s — so this run is **GPU-only** (`--evaluator cuda --cuda-builder
mixed`) and necessarily smaller (64 queries, `--confirm-seeds 2`, one workload). It is a
spot-check, not a full real-data evaluation.

1. **GPU build speedup holds at scale.** Octree builds in ~0.8 s on GPU vs ~53 s on CPU
   (~65×) — the synthetic appendix's "GPU is a build accelerator" result reproduces on real
   data.
2. **Best single primitive = octree, by a wide margin.** Mean per-query latency by single
   primitive (the `mixed` workload is radius-dominated, ~70k points returned per query over
   100M): octree ≈ 285 ms, karras_octree 546, lbvh 868, hgrid 894, quadtree 1420, bvh 7759,
   regular_grid 15215 ms.
3. **Nesting beats the best single on real data.** Measured apples-to-apples (same 64
   queries/seed, all returning 70,569 pts), a hand-designed GPU-native nested schema
   **`grid2d_quadtree_grid3d_octree` reaches 323 ms vs octree's 408 ms — 1.26×**, *larger*
   than the synthetic 1.12× geomean. `quadtree_octree` (464 ms) does **not** beat octree, so
   the nesting recipe matters. Alhambra's heterogeneity is consistent with the synthetic
   finding that nesting pays off on mixed data.
4. **A measurement bug surfaced at scale — found and fixed.** GPU-unsupported candidates
   (BIH, median/round-robin kd) fall back to the CPU evaluator, but under `--evaluator cuda`
   the workload preparation only built GPU queries, so the CPU-fallback path ran an *empty*
   query set and reported **0 points / 0.000 ms** (e.g. `kdtree` *built* in 90 s but answered
   nothing). `--auto-conditions` mistook those `0 ms` artifacts for the fastest and exported a
   **degenerate** schema (a 1,048,576-capacity BVH leaf). **Fixed** in two parts:
   (A) the CUDA query-prep now mirrors each GPU query into a CPU query so fallback schemas
   measure real latencies; (B) the search generates `center_longest_axis` kd/BIH under CUDA so
   generated candidates stay GPU-native. After the fix the re-run exports a real nested schema
   (`urban_hybrid`: quadtree→octree→kdtree) and **no row reports 0 ms**.
5. **What the corrected run reveals: at 100M, CPU traversal beats the GPU query path.** With
   honest measurement, the radius-heavy `mixed` workload is far faster on CPU-resident schemas
   (urban_hybrid ~0.53, octree_kdtree ~0.59, bih ~0.64, kdtree ~0.69 ms/query) than on the GPU
   builders (lbvh ~1.5 s, octree ~4.5 s/query; GPU per-query latency at 100M is high and
   run-variable). The GPU range/radius traversal over 100M points is overhead/memory-bound, so
   the tuner now correctly selects a CPU-resident schema (which, using median kd, runs on CPU
   anyway). This re-confirms **GPU is a build accelerator, not a query accelerator** at this
   scale/workload; a GPU query win would need much larger batched query streams or a faster
   GPU traversal.
6. **Single-query GPU latency fixed (commit `dbe8d2a`).** The slow GPU query numbers above came
   from the *one-thread-per-query* kernel, which the dispatcher picked unless a query was large
   — fine for saturating the GPU with a big batch, catastrophic for one query (a single thread
   serially traverses 100M points while its block's other 255 threads idle). Making the
   dispatcher batch-size-aware (a ≤32-query batch now uses the cooperative *block-per-query*
   kernel) drops a single Alhambra range query from **70.56 ms → 0.37 ms (188×, identical
   returned counts)** — now competitive with the ~0.5 ms CPU path. So for *interactive single
   queries* (the realistic access pattern) GPU is viable; the batched "GPU query is slow"
   result above is large batches of small queries still on the one-thread path. Open
   follow-ups: multi-block per *large* single query, and a tree-guided kNN to replace the
   brute-force scan.

(First-schema GPU build times absorb warmup/upload — octree's 110 s build in the earlier
Stage-B run is a warmup artifact; discovery shows ~0.8 s. Data in `results/eval_alhambra/`.)

## Honest caveats

- **Headline is synthetic (3 families); one real-data spot-check.** `synthetic_flat_terrain`,
  `synthetic_facade`, `synthetic_urban_mixed` (the `building`/`sparse-dense` generators exist
  but are not wired into schema-search). The Alhambra section above is a single GPU-only,
  one-workload spot-check; a full real-data evaluation (multiple real clouds and workloads,
  with the auto-tuner reliability fix) is the main outstanding follow-up.
- **Medium scale (300k points).** Large enough for ms-scale builds and resolvable
  `volume_small_medium` latencies, but small enough that the other workloads' per-query
  latencies (~µs) are near the measurement floor — hence only 3/12 cells are
  `ranking_confident`. Bigger clouds would sharpen the small-query cells.
- **12-cell sample.** 3 families × 4 workloads is a small grid; the 1.12× geomean is a modest,
  honestly-bounded effect, not a sweeping win.
- **KNN is `bruteforce_gpu_scan`** on most GPU builders (tree KNN only for kd/BIH at k≤16), so
  `knn_heavy` GPU numbers reflect brute force, not tree-accelerated traversal.
- **Score objective fixed to latency.** Build time and memory enter only via the Pareto
  view (`results/eval/*_pareto.csv`), never the headline score; a `balanced` objective would
  reorder some winners.

## Reproduce

Built binary `x64/Release/MultiDataStructure.exe` (no source changes for this evaluation).

```powershell
# Stage A — the optimizer (GA, the DEFAULT search) synthesizes + exports one schema per
#   (dataset, workload), CPU, at full 1024-query fidelity. For w in range_heavy knn_heavy mixed volume_small_medium:
x64\Release\MultiDataStructure.exe --mode schema-search --optimize-schemas `
  --optimizer-generations 6 --optimizer-population 16 --optimizer-seed 1337 --evaluator cpu `
  --workloads configs/workloads/<w>.json --query-seed 1337 --queries 1024 --knn-k 16 `
  --synthetic-scale 300000 --score-objective latency --no-score-cache `
  --optimizer-output-dir results/eval_ga/evolved_<w> --csv results/eval_ga/ga_discover_<w>.csv --no-pause

# Stage B — measure 9 singles + 6 hand-nested + the GA exports UNIFORMLY, CPU. --flat-search
#   opts out of the (now default) GA so this is a one-pass scan of exactly the listed schemas:
#   $all = the 9 single configs ; the 6 nested configs ; results/eval_ga/evolved_<w>/*_best_schema.json
x64\Release\MultiDataStructure.exe --mode schema-search --flat-search --evaluator cpu `
  --workloads configs/workloads/<w>.json --schemas "$all" `
  --generate-schemas 0 --no-baselines --benchmark-top 100 `
  --query-seed 1337 --queries 1024 --knn-k 16 --synthetic-scale 300000 --score-objective latency `
  --confirm-seeds 3 --confirm-top 24 --no-score-cache `
  --csv results/eval_ga/measure_<w>.csv --no-pause

# Analyze (the GA exports are tagged the optimizer arm via the `evolved` path marker):
python scripts/compare_methods.py --inputs results/eval_ga/measure_*.csv `
  --auto-marker evolved --out-dir results/eval_ga/report
python scripts/train_schema_selector.py --input results/eval_ga/report/combined_raw.csv `
  --report results/eval_ga/report/schema_selector_report.json
```

Two flags matter: `--flat-search` opts the re-measurement out of the now-default GA so it
scans exactly the provided schemas; `--generate-schemas 0 --no-baselines` keeps that flat
scan from adding generated/baseline candidates that would contaminate the controlled
comparison. (The GPU appendix below was measured separately on the GPU evaluator and is a
build/query characterization independent of which search produced the schemas.)

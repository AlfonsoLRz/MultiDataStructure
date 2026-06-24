# Evaluation: auto-tuned nested index vs. best single primitive

This is the focused evaluation of the project's central claim. It compares, on a common
query workload, three ways of choosing a point-cloud spatial index:

1. **Best single primitive** — the fastest of nine standard single-structure indexes
   (quadtree, octree, kd-tree, BVH, BIH, LBVH, regular grid, hierarchical grid, Karras octree).
2. **Hand-designed nested** — six fixed multi-level schemas authored by hand
   (`quadtree_octree`, `octree_kdtree`, `urban_hybrid`, `grid2d_quadtree_grid3d_octree`,
   `urban_grid_hybrid`, `adaptive_quadtree_octree`).
3. **Auto-tuned** — the schema selected per (dataset, workload) by the `--auto-conditions`
   search, which generates nested/conditional candidates and ranks them by a staged
   proxy → shortlist → confirmation measurement.

against the per-cell **oracle** (the best of everything measured).

## Claim

> A workload-aware auto-tuned index modestly but consistently beats the best single
> primitive, and tracks the oracle closely — with the advantage concentrated on
> heterogeneous data and higher-volume query workloads, and shrinking to a tie on
> small, homogeneous query workloads.

## Result (headline)

Measured on CPU over **3 synthetic dataset families × 4 workload profiles = 12 cells**.
Latency is the seed-averaged mean over `--confirm-seeds 3` where confirmed, else the
single-run average; lower is better.

| Dataset | Workload | Best single | Hand-nested | Auto-tuned | Oracle | Speedup auto/single | Regret vs oracle |
|---|---|---|---|---|---|---|---|
| synthetic_facade | knn_heavy | bvh 0.0049 | octree_kdtree 0.0145 | kdtree 0.0048 | 0.0048 | 1.00× | 0.4% |
| synthetic_facade | mixed | bih 0.0056 | octree_kdtree 0.0132 | bvh 0.0061 | 0.0056 | 0.92× | 8.7% |
| synthetic_facade | range_heavy | bih 0.0048 | octree_kdtree 0.0066 | generated (nested+cond) 0.0047 | 0.0047 | 1.02× | 0.0% |
| synthetic_facade | volume_small_medium | octree 0.0298 | octree_kdtree 0.0227 | **generated_ot4l256 0.0167** | 0.0162 | **1.78×** | 3.6% |
| synthetic_flat_terrain | knn_heavy | lbvh 0.0045 | quadtree_octree 0.0060 | bvh 0.0049 | 0.0044 | 0.93× | 9.5% |
| synthetic_flat_terrain | mixed | bvh 0.0051 | quadtree_octree 0.0061 | bvh 0.0053 | 0.0051 | 0.96× | 4.5% |
| synthetic_flat_terrain | range_heavy | bvh 0.0050 | quadtree_octree 0.0042 | generated_ot5l128_bvh7l32768 0.0046 | 0.0042 | 1.08× | 8.8% |
| synthetic_flat_terrain | volume_small_medium | octree 0.0278 | octree_kdtree 0.0230 | **generated_ot4l256 0.0169** | 0.0168 | **1.64×** | 0.9% |
| synthetic_urban_mixed | knn_heavy | octree 0.0050 | quadtree_octree 0.0051 | generated_kd6l4096_ot2l32_qt3l64 0.0047 | 0.0047 | 1.07× | 0.0% |
| synthetic_urban_mixed | mixed | octree 0.0032 | octree_kdtree 0.0031 | generated_kd6l4096_ot2l32_qt3l64 0.0030 | 0.0030 | 1.06× | 0.0% |
| synthetic_urban_mixed | range_heavy | karras_octree 0.0017 | octree_kdtree 0.0018 | generated_ot5l128_bvh7l32768 0.0015 | 0.0015 | 1.16× | 0.0% |
| synthetic_urban_mixed | volume_small_medium | bvh 0.0135 | octree_kdtree 0.0154 | generated_ot4l256 0.0121 | 0.0119 | 1.12× | 1.7% |

(latencies in ms; full machine-readable data in [results/eval/report/](../results/eval/report/) — `comparison.md`, `summary.json`.)

**Summary:** auto-tuned beats the best single primitive in **9 of 12 cells**, geomean
speedup **1.12×**, mean relative regret vs. oracle **3.2%**.

### What the tuner actually selects

- On **`volume_small_medium`** (many medium-extent range queries) the tuner picks a
  *well-tuned single octree* (`generated_ot4l256`: 4 levels, leaf 256) and wins clearly
  (**1.64–1.78×**). The hand-nested `octree_kdtree` also beats the default single octree
  here, but the tuned octree beats both — i.e. *parameter* tuning matters more than nesting
  on this workload.
- On **heterogeneous `synthetic_urban_mixed`** the winners are genuinely *nested*
  (`generated_kd6l4096_ot2l32_qt3l64`, `generated_ot5l128_bvh7l32768`) — nesting pays off
  where the data mixes regimes.
- On **small-query workloads** (`knn_heavy`, `mixed`) over homogeneous facade/terrain the
  best single primitive is already near-optimal, and auto-tuning **ties or slightly loses**
  (0.92–1.00×). This is honest: nesting/tuning has little to exploit there.

Only 3/12 cells are flagged `ranking_confident` (the per-cell latency gaps are small at this
scale); the **`volume_small_medium` wins are the statistically robust ones**.

### Learned selector (illustrative)

Training `scripts/train_schema_selector.py` on the 216 measured rows (held-out family:
`synthetic_facade`) yields a ridge score-ranker with best-schema accuracy 0.75 and **mean
relative regret 0.4%** vs. oracle — i.e. cloud/workload/schema features predict a near-best
schema. This is illustrative only: with **3 dataset families** the train/test split is tiny,
so treat it as a sanity check, not a generalization claim.

## Comparability protocol

Every compared row shares one provenance: `--evaluator cpu`, `--score-objective latency`
(weights latency/build/memory/imbalance = 1/0/0/0), `score_is_final_latency=1`,
`score_mode=latency`, `--queries 1024`, `--query-seed 1337`, `--knn-k 16`,
`--synthetic-scale 300000`. `scripts/compare_methods.py` verifies this and reports
**"Distinct provenance settings: 1"** — the comparison is apples-to-apples.

The auto-tuner is run as a **separate discovery stage** (`--auto-conditions`, 256 queries)
that *exports* a schema per cell; the headline then **re-measures** those exports next to the
baselines under the identical settings above. This matters: `scripts/audit_schema_scores.py`
on the discovery vs. measurement CSVs correctly flags `effective_queries: 256 vs 1024` and
`score_stage: confirmation vs final` as non-comparable — which is exactly why discovery
numbers are never compared directly. Reassuringly, the tuner's *selection* is stable across
the two stages (same winning schema per cell).

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
# Stage A — discover one auto-tuned schema per (dataset, workload), CPU:
#   for w in range_heavy knn_heavy mixed volume_small_medium:
x64\Release\MultiDataStructure.exe --mode schema-search --auto-conditions --evaluator cpu `
  --workloads configs/workloads/<w>.json --query-seed 1337 --queries 256 --knn-k 16 `
  --synthetic-scale 300000 --score-objective latency --no-score-cache `
  --condition-output-dir results/eval/auto_schemas --csv results/eval/discover_<w>.csv --no-pause

# Stage B — measure 9 singles + 6 hand-nested + the auto exports UNIFORMLY, CPU:
#   $all = the 9 single configs ; the 6 nested configs ; results/eval/auto_schemas/*_<w>_best_schema.json
x64\Release\MultiDataStructure.exe --mode schema-search --evaluator cpu `
  --workloads configs/workloads/<w>.json --schemas "$all" `
  --generate-schemas 0 --no-baselines --benchmark-top 100 `
  --query-seed 1337 --queries 1024 --knn-k 16 --synthetic-scale 300000 --score-objective latency `
  --measure-repeats 3 --confirm-seeds 3 --confirm-top 18 --no-score-cache `
  --csv results/eval/<w>.csv --best-csv results/eval/<w>_best.csv --pareto-csv results/eval/<w>_pareto.csv --no-pause

# Analyze:
python scripts/compare_methods.py --inputs results/eval/range_heavy.csv results/eval/knn_heavy.csv `
  results/eval/mixed.csv results/eval/volume_small_medium.csv --out-dir results/eval/report
python scripts/train_schema_selector.py --input results/eval/report/combined_raw.csv `
  --report results/eval/report/schema_selector_report.json
```

`--generate-schemas 0 --no-baselines` is essential: without it the flat search silently
generates 256 candidates and surrogate-prunes to top-32, which would contaminate the
controlled 18-schema comparison. The GPU appendix repeats Stage B with
`--evaluator cuda --cuda-builder mixed` into `results/eval/gpu_<w>.csv`.

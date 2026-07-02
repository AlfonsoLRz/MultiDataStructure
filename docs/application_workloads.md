# Application-workload experiments (trace-driven schema search)

This documents the first full run of the application-workload evaluation (2026-07-02
overnight): schema search driven by **recorded application query traces** instead of
synthetic mixes, in the two regimes described in the paper's Applications section.

## Protocol

1. **Capture** the workload a real application issues:
   ```
   python scripts/capture_pipeline_workload.py "D:/.../cloud.las" --out results/traces/<name> --max-queries-per-stage 20000
   python scripts/capture_dl_workload.py       "D:/.../cloud.las" --out results/traces/dl_<name> --blocks 32 --npoints 4096
   ```
   Pipeline stages are dense self-queries (normal estimation kNN k=16; radius outlier
   removal at 4x mean spacing; Euclidean clustering at 2.5x), so the trace is derivable
   and the sidecar records the full per-stage query count (= N points per stage).
   DL blocks record PointNet++ SSG layer queries against the point set each layer
   actually searches (block -> SA1 centroids -> SA2 centroids).

2. **Search** against the replayed trace:
   ```
   MultiDataStructure --mode schema-search --input <cloud> --no-synthetic \
     --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/<name>/pipeline_trace.csv \
     --queries 1500..3000 --evaluator cpu --no-score-cache ...
   MultiDataStructure --batch-manifest results/traces/dl_<name>/batch_manifest.json --generate-schemas 64 ...
   ```
   `--no-score-cache` is required for trace runs: the score-cache key does not include
   the trace path, so synthetic and trace runs of the same workload name would collide.

## Results (single seed, CPU evaluator, RTX-4080-class workstation)

### Pipeline replay (one static index, dense self-queries; score = avg query latency)

| dataset | winner (searched) | ms/query | best canonical | ms/query | GA vs best | worst canonical |
|---|---|---|---|---|---|---|
| SanAndreas 5M (UAV terrain) | `qt8l32 + qt1l512 + qt3l32` (nested quadtree) | 0.00356 | quadtree | 0.00386 | **-7.8%** | bvh: 0.0145 (4.1x) |
| SanAndreas 50M (UAV terrain) | `kd1l8192 + qt11l512` (kd over quadtree) | 0.00630 | octree | 0.00703 | **-10.3%** | bvh: 0.120 (19x) |
| Alhambra 100M (heritage architecture) | `ot8l32 + ot2l8192` (nested octree) | 0.00468 | octree | 0.00481 | **-2.7%** | quadtree: 0.167 (**35x**) |

### Batch distribution (DL dataloader; score = full-pass build + query wall cost)

| dataset | winner (searched) | ms/pass | best baseline | ms/pass | delta |
|---|---|---|---|---|---|
| SanAndreas blocks (96 pairs, 168k queries) | `bvh1l256 + ot5l512 + kd4l8192` | 226.8 | bvh | 274.9 | **-17.5%** |
| Alhambra blocks (96 pairs, 168k queries) | `kd3l256` (round-robin axes) | 237.8 | octree_kdtree | 274.7 | **-13.4%** |

### Reading

- **The winning family flips per dataset, per scale, and per regime**: pure nested
  quadtree at 5M terrain, kd-over-quadtree at 50M terrain, nested octree for full-3D
  architecture, a three-family nest for terrain dataloaders, a shallow kd-tree for
  architecture dataloaders. Even within one dataset family, scale alone changes the
  winner — and the best canonical primitive changes with it (quadtree at 5M, octree
  at 50M).
- **Every dataset's winner is catastrophic somewhere else** (quadtree: best on
  terrain, 35x worst on architecture). The search's first value is insurance against
  a wrongly chosen default; its second value is the consistent single-digit-to-17%
  win over the best hand-picked structure under identical queries.
- **Amortization** (full-stage counts from the sidecars): on Alhambra the tuned
  schema saves ~0.00013 ms/query x 3x10^8 queries/pass ~= 38 s per pipeline pass vs
  the best canonical choice (the 50-minute search pays back in ~80 passes) — and
  ~13.5 hours per pass vs the worst canonical choice (pays back in ONE pass). In the
  DL regime the tuned schema saves ~17% of every epoch's neighborhood-structuring
  time, which profiling literature places at the majority of end-to-end runtime.

### External framework baselines (identical replayed queries)

`scripts/compare_frameworks.py --input-trace ...` (new passthrough) replays the same
application trace through external libraries. Run from `.venv-mds` (Python 3.12 +
Open3D 0.19; the main `.venv` is Python 3.14, for which Open3D has no wheel):

| dataset | GA-tuned schema (CPU) | Open3D KDTreeFlann | speedup |
|---|---|---|---|
| SanAndreas 5M, 3000 pipeline queries | 0.00421 ms/query | 0.02271 ms/query | **5.4x** |
| Alhambra 100M, 3000 pipeline queries | 0.00476 ms/query | 0.01752 ms/query | **3.7x** |

PDAL/PCL columns pending (PDAL CLI and the PCL helper binary are not installed on
this machine); the harness already supports both.

## Search cost and multi-fidelity (the "isn't the search slow?" defense)

Measured on Alhambra_100M, whose full-scale GA took 50.4 min (~95% of it = ~43 full
100M-point index builds; queries are negligible):

- **Rank transfer**: re-evaluating the same 43 candidates on subsampled clouds gives
  Spearman rho = 0.70 (2M) / 0.81 (5M) vs full scale. The top of the full ranking is
  nearly flat (2.7% spread), so exact top-3 order does not survive subsampling — but
  **confirming the low-fidelity top-10 at full scale recovers the true winner exactly
  (0% regret) at both fidelities**; top-5 confirmation lands within 18.9%.
  Low-fidelity evaluation of all 43 candidates costs 0.8 min (2M) / 2.0 min (5M).
- **Recipe**: search at 2-5M fidelity, confirm top-10 at full scale (~10 builds):
  **~13-15 min instead of 50, zero regret** — and the confirms parallelize too.
- **Parallel dispatch**: the same 5M GA with `--parallel 8` ran in **1.4 min vs 9 min**
  (6.4x); the parallel run's best schema was within ~10% of the serial run's (GA paths
  diverge under reordering; same budget).
- **Repeat-measurement CIs** (5 repeats, winners on full clouds): latency CV 3.4-6.4%.
  The DL-regime (-13 to -17%) and 50M (-10.3%) wins comfortably exceed measurement
  noise; the Alhambra pipeline win over the best canonical octree (-2.7%) is within
  ~1 CV and should be reported as parity-or-better, with the 35x wrong-default and
  3.7x-vs-Open3D results carrying that dataset's argument.
- **Framing**: this is a one-time offline cost per (dataset-class, workload) — the
  regimes that motivate the paper reuse the tuned schema across hundreds of pipeline
  passes or training epochs.

## Caveats / next steps

- Single seed, single machine, CPU evaluator; rerun the winners with
  `--confirm-seeds`/`--measure-repeats` before quoting CIs in the paper.
- Traces are derived (correct by construction for self-query stages), not captured
  from a live library; the Open3D-native timing column requires a Python 3.11/3.12
  env (open3d has no 3.14 wheel yet) and `scripts/compare_frameworks.py`.
- `50M.ply` is binary PLY; the C++ loader supports ASCII PLY only, so the 50M run
  uses `results/traces/sanandreas_50m/50M_converted.las`. A binary-PLY loader is a
  pending fix.
- SolarPanels700M (700M points) is untested in this round; the C++ side loads LAS
  fine but budget several hours for a GA at that scale.

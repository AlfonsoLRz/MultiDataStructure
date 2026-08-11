# The method, in plain terms

**What this document is.** A single reference explaining what the system actually does,
what the evidence says, what to claim out loud, and what to ask others for. Written
2026-07-29, after the held-out evaluation battery. If a number here disagrees with an
older doc, this one and `testing.md` §1b win.

Reading order if you are in a hurry: §1 (the one-paragraph pitch), §6 (what to sell),
§7 (what not to claim), §8 (questions you will be asked).

---

## 1. The one-paragraph version

Point-cloud applications don't issue a few interactive queries — they issue *one query per
point, per centroid, or per network layer*, so a single pipeline stage over a 100M-point
scan fires 100M neighborhood queries. The index is therefore an inner loop, its query
distribution is known in advance, and any one-off cost spent choosing a better index is
amortized millions of times over. Our system exploits exactly that: it **records the query
stream a real application issues, treats the index design itself as a measured variable
(single, nested, or conditionally nested, all in one replayable JSON format), builds
candidates over the actual target cloud, and picks the winner by replaying the recorded
workload.** Above roughly 25 million points this beats the best *tuned* conventional index
by 7–113%, and beats what practitioners actually use (Open3D, PCL/FLANN) by 2.8–3.7× at
100M. Below ~10 million points it correctly returns a plain single structure, because there
nothing better exists.

---

## 2. What the system is made of

Four components. The first is the contribution; the rest are machinery in service of it.

### 2.1 The schema: one format for every index design

A **schema** is an ordered list of levels. Each level says: which primitive
(quadtree / octree / kd-tree / BVH / BIH / LBVH / Karras octree / regular grid /
hierarchical grid), how many tree levels it owns, the target leaf capacity, the minimum
points needed to keep splitting, and a split policy (e.g. which axes a quadtree ignores,
whether a kd-tree splits at the median or the box centre, round-robin axes).

Three kinds of design fall out of the same format:

| Design | Schema shape | Example |
|---|---|---|
| **Single** (a conventional index) | one level | `octree, depth 8, leaf 32` |
| **Nested** (fixed by depth) | several levels in sequence | `quadtree depth 4 → octree depth 6 → kd-tree depth 2` |
| **Conditional** (per-node choice) | levels carrying predicates | `octree; then kd-tree only in nodes whose anisotropy > 0.7` |

The conditional predicates can test a node's point count, density, extent per axis, height
ratio, bounding-box anisotropy, and occupancy entropy (Shannon entropy over a 4×4×4
sub-grid). At build time each node either enters the current level or skips forward to the
next one — so the structure genuinely varies by location within one cloud.

**Why this matters more than it sounds:** because a conventional index is just a
one-level schema, tuned single structures and elaborate nested ones compete *in the same
candidate pool, measured the same way*. There is no separate "baseline path" that could be
accidentally handicapped. And every schema is a JSON file, so a winner can be replayed
months later without re-running any search.

### 2.2 The workload: recorded, not invented

Rather than inventing a query mix, we record what real applications issue and replay it:

- **Geometry-processing pipelines** (`scripts/capture_pipeline_workload.py`): normal
  estimation (kNN, k=16), radius outlier removal (radius = 4× mean spacing), Euclidean
  clustering (2.5× mean spacing). These stages are *dense self-queries* — every point
  queries the cloud — so the exact query stream is derivable from the cloud plus the stage
  parameters, no library instrumentation needed.
- **Deep-learning dataloaders** (`scripts/capture_dl_workload.py`): PointNet++-style
  per-block queries — farthest-point sampling then ball queries at set-abstraction layers,
  3-NN interpolation at feature-propagation layers. Crucially, each block gets its **own**
  index, so build cost counts as much as query cost.

Traces are CSV files with one row per query (type, bounds/centre, radius, k) in **world
coordinates**. Every compared system — ours, Open3D, PCL, Indexicon — replays the identical
file, and we verify returned result counts match.

### 2.3 The search: generate, measure, keep the best

Candidates come from a bounded random generator over the schema grammar (number of blocks,
primitives, depths, leaf capacities, policies, optional predicates whose thresholds are
derived from quantiles of a quick sketch of the target cloud). They are then **measured on
the real cloud with the real trace**, and the best measured candidate wins.

Cost control, because building candidates over 100M+ points is the expense (>95% of search
time is candidate construction, not the optimizer):

- **Multi-fidelity / successive halving**: screen many candidates cheaply (few queries, or
  a visit-count proxy instead of wall clock), promote a shortlist to full measurement,
  confirm the top few at full scale and full query count.
- **Subsample-then-confirm**: rank candidates on a 2–5M-point subsample, then confirm only
  the top handful at full scale. Measured Spearman ρ = 0.88–0.95 against full-scale
  ranking, and the true winner sat in the subsample top-3 in every case → **zero regret**.
- **Parallel dispatch**: candidates are embarrassingly parallel; 8 workers took a
  representative search from 9.4 min to 1.5 min.

There is also an evolutionary layer (elites, mutation, crossover, NSGA-II ranking, plus
"repair" mutations driven by measured traversal counters). **The honest finding is that it
is not needed** — see §5.

### 2.4 The measurement discipline

This is what makes the numbers defensible, and it is worth selling as part of the
contribution:

- **Exactness.** All four query types (range, count-range, radius, kNN) are exact and
  unit-tested against brute force. Nothing approximate, no learned predictor in the query
  path.
- **Telemetry.** Per query: visited nodes, tested points, returned points, fully-contained
  nodes, plus breakdowns by depth and by active structure type. This is how a win gets
  *explained* rather than just reported.
- **Provenance.** Every measurement row carries dataset, cloud/workload features, schema
  identity, backend, GPU support status (full / cpu_fallback / unsupported_policy), score
  stage (proxy vs final), and weights. Cheap proxy rows can never be compared against
  full-fidelity rows by accident.
- **Discovery ≠ reporting.** Searching happens on one half of the trace; the reported
  numbers come from replaying exported winners on the untouched other half, under one
  uniform provenance, with five repeat measurements and bootstrap confidence intervals.

---

## 3. What the evidence says

All numbers held-out (searched on one trace half, measured on the other), against
**tuned** single structures (a budget-matched grid over depth × leaf capacity × policy per
primitive), with verified result counts. Sources: `testing.md` §1b,
`results/eval_traces/{v2,matrix}`.

### 3.1 The headline: a scale crossover

| Cloud | Points | Searched winner | vs best tuned single |
|---|---|---|---|
| SanAndreas terrain | 5M | *returns the plain quadtree* | parity (by design) |
| SanAndreas terrain | 50M | qt8/256 + ot2/128 + bvh2/4k | **−24%**, CIs separated |
| Alhambra (TLS architecture) | 100M | bvh7/128 + ot5/1k | −2.5% (not significant) |
| SolarPanels (industrial) | 700M | qt4/2k + ot6/256 | **2.13×**, top-9 all nested |

### 3.2 Shape × size matrix (16 controlled cells)

Speedup over best tuned single; >1 means the search wins:

| shape \ size | 1M | 5M | 25M | 100M |
|---|---|---|---|---|
| terrain | 0.88 | 0.96 | **1.09** | **1.27** |
| industrial | 0.95 | 0.93 | **1.07** | **1.23** |
| urban (aerial) | 1.06 | 0.91 | 1.03 | **1.07** |
| architecture (TLS) | 0.99 | 1.04 | 1.04 | — |
| indoor | 0.96 | — | — | — |

**The crossover happens between 10M and 25M points for every shape.** Scale is the
first-order driver; shape modulates the magnitude (terrain and industrial steepest,
architecture flattest).

### 3.3 Two mechanisms, not one

At scale the search wins in two different ways, and doesn't need to be told which applies:

1. **Composition** — genuinely nested schemas win (industrial and urban at 100M, 700M).
2. **Off-grid single tuning** — the terrain-100M winner is a *single* quadtree at **depth
   9**, a configuration that fell between the grid points of the tuned-singles sweep
   (which tried depths 4/6/8/10/12). Hand-tuning grids miss it; measurement finds it.

### 3.4 Against real libraries

Identical held-out queries, verified counts (residual disagreements ≤1 point on ≤10 of 3000
queries, boundary rounding):

| Cloud | Ours | Open3D KDTreeFlann | PCL kd-tree (FLANN) | PCL octree |
|---|---|---|---|---|
| 5M | 0.0036 | 0.0133 (3.7×) | **0.0028** (0.85×) | 0.0147 |
| 100M | **0.0129** | 0.0479 (3.7×) | 0.0356 (2.76×) | 0.0590 |

The same crossover appears against external libraries: **FLANN genuinely wins at 5M**
(and 5M is a parity cell anyway), we win 2.76× at 100M. FLANN builds 5.4× faster (12.4 s
vs 67.5 s at 100M), so our crossover amortizes after ~2.4M queries — 2.4% of *one*
per-point pipeline stage on that cloud.

### 3.5 The insurance argument (often the most persuasive one)

Picking the wrong fixed structure is far more expensive than the margin between good ones:

- Worst tree-based default vs best default: **4.3× (5M), 10.5× (50M), 12.2× (100M),
  12.8× (700M)**.
- Regular grid: 148× to 777×.
- The winning single *changes identity* across clouds: the quadtree that's best on terrain
  is 4× off on Alhambra; kd-tree wins architecture at 1M, BIH wins indoor.
- DL blocks: worst default costs 1.9× per pass — every batch, every epoch.

So even where nesting gains nothing, *measured selection* protects you from a
catastrophically wrong default that a human would plausibly have picked.

### 3.6 Cross-cloud transfer

Replaying the SanAndreas-50M winner on an independent 25M coastal survey (San Simeon,
different campaign and density): ranks 2nd, +9% over that cloud's best structure, while
wrongly fixed singles cost 1.4–6.8×. **Transfer is safe but not optimal** — reuse a
same-morphology schema as a default, re-tune to recover the last ~10%.

### 3.7 Primitive-quality control (Indexicon)

*Rewritten 2026-08-06 after the full §F.3 baseline landed. The earlier version of this
section claimed our primitives "match or beat theirs on every comparable operation"; that
claim was measured on range+kNN only, on one synthetic trace, and it does not survive the
radius-inclusive pipeline traces. It is withdrawn — see below.*

Against the independent Indexicon library (arXiv:2606.04676, MIT), now covering all three
of its structures (octree, kd-tree, **packed R-tree**) and all query types:

**What holds — exactness.** **Zero count mismatches across 351,000 measured samples**
(13 matrix cells × 3 structures × 3000 held-out queries × 3 repeats, radius + kNN). The
single disagreement anywhere is on the older 5M synthetic trace: one point lying 3×10⁻⁸
relative distance from a query radius, where our `float` distance test and the driver's
`double` one round to opposite sides. Since Indexicon has no radius primitive,
radius is implemented in our driver over each of their structures using the same
min-distance-to-node pruning their own kNN uses, and every traversal is validated against
brute force before it is allowed to report (`--verify-bruteforce`). This is an exactness
cross-validation against independently written code, and it is the claim to lean on: the
composition results cannot be an artifact of broken primitives.

**What does not hold — the old latency claim.** On the radius-inclusive pipeline traces
their octree and kd-tree answer queries *faster* than ours in all 13 matrix cells measured
(`results/framework_compare/indexicon/indexicon_summary.csv`, both sides measured in the
same session on the same held-out trace). One cause dominates — per-node instrumentation —
and a separate, cheaper configuration bug in the kd grid changes how it shows up.

**(a) A large per-visited-node cost, ~10% of it instrumentation.** Solving
latency = a·visited + b·tested per cell yields **a ≈ 51–131 ns per visited node** against
**b ≈ 1.1–2.8 ns per point tested**, stable across all 13 cells (independent 6-schema fit on
terrain_5M: 106 ns/node vs 1.17 ns/point). Node counts barely move with scale (59→89) while
tested points grow ~10×, so the fixed cost dilutes and the octree gap closes —
0.48 → 0.57 → 0.94 (architecture), 0.40 → 0.76 → 1.01 (industrial), 0.52 → 0.76 → 0.94
(terrain). **At 25M our octree is already at parity.**

Part of that per-node cost was telemetry: `recordNodeVisit` built a `std::string` by value
and probed an `unordered_map` on every visited node, inside the timed region. Those counters
were rewritten to arrays indexed by a build-time `Node::_structureIndex` (2026-08-06). A/B
on terrain_5M with **identical node and point counters** — so the traversal is provably
unchanged — the rewrite is worth **1.21× on the octree** (66.7 nodes/query) and **1.03× on
the kd-tree** (28.7 nodes/query): about **6–14 ns/node** of the ~106 ns/node total. The
remaining ~90 ns/node is genuine traversal work, dominated by cache misses over node structs
(a 524k-node kd-tree spans ~50 MB).

*Two earlier readings are withdrawn.* The first blamed the whole 106 ns/node on telemetry.
The second concluded telemetry was negligible because removing it "changed nothing" — that
measurement ran a **stale binary** which did not contain the change (see §10: MSBuild links
to `x64\Release`, every script pointed at a legacy `MultiDataStructure\x64\Release` copy
frozen at 2026-07-28). The numbers above come from an explicit old-vs-new binary comparison
made after that was found. **Consequence for the Indexicon column:** MDS was paying ~10%
more per node than it needs to, so the octree ratios above are pessimistic by roughly 1.2×
and our octree likely *beats* Indexicon's at 25M once re-measured.

**(b) A configuration bug in the kd grid — real, and now fixed.**
Our kd-tree visits only ~30 nodes at every scale but tests **8–19× more points than our own
octree** (15,008 per query at industrial_25M vs the octree's 1,110). The cause is arithmetic:
a kd-tree is binary, so `numLevels: 12` caps it at 2¹² = 4096 leaves, and the builder stops
on depth before it ever consults capacity (`PointSpatialIndex.cpp:888` precedes `:894`).
`kdtree_default` therefore builds 8191 nodes with **avg leaf occupancy 1220.7 against a
requested `leafCapacity` of 32** — a 38× miss, uniform across leaves. Reaching capacity 32
at 25M needs depth ≈ 20; the grid's ceiling is 12, so *every* kd schema in
`configs/schemas/tuned_singles*/kd/` is depth-capped at ≥1M points and `leafCapacity` is
inert in 9 of 12. In `tuned_singles_small` — the grid the whole heatmap battery used — all
4 kd schemas are depth-capped and `tuned_kd12l32` builds a **byte-identical tree to
`kdtree_default`**, so the kd tuning arm was largely inert.

**Fixed 2026-08-06.** `make_single_grids.py` now *derives* depth from capacity and cloud size
via each primitive's branching factor (`--points`, default 25M; ceiling raised 12 → 24), and
clears stale schemas from its output directory. Both grids were regenerated at their original
sizes (108 / 36 schemas): **zero inert schemas remain**, and the kd arm now spans depths
13–20 rather than sitting entirely at 12.

**Effect on the baseline: the tuned-singles kd arm gets 1.22× stronger** (terrain_5M,
held-out, 3 repeats): the best fixed-grid kd is `tuned_kd16l1024` at 0.00487 ms
(16,383 nodes, occupancy 610) versus `kdtree_default` at 0.00592 ms (8,191 nodes, occupancy
1221). That strengthens the baseline our searched schemas are measured against, i.e. it
*reduces* our reported margins in any cell where a kd-tree was the best single. Note the
grid now also spans over-splitting: `tuned_kd20l32` reaches occupancy 19.1 exactly as asked
and is the second slowest of the arm, so depth is a genuine trade-off rather than a cap.
**All matrix/battery numbers involving a kd-tree baseline predate this fix and must be
re-measured.**

**What this implies elsewhere.** Schema-vs-schema comparisons penalise node-heavy designs
relative to point-heavy ones, since the per-node cost is ~90× the per-point cost. That is a
property of the traversal, not of the measurement, so it is a legitimate part of what the
search optimises — but it is worth stating, because it means our results favour shallow,
wide structures and would shift on an implementation with better node locality.

**The R-tree question is answered, and cleanly.** Comparing Indexicon's three structures
*against each other* is immune to both problems above, since all three are their code. On
these workloads their packed R-tree is **the slowest of the three in 8 of 13 cells and the
fastest in none**. That is external evidence for the decision not to admit an R-tree into
the grammar (overlapping MBRs don't fit per-level disjoint splits; packed R-trees over
static in-memory points are that family's weakest regime) — rather than our own omission
standing in for the result. Per `plans/research_direction_v2.md` §F.3 the gate was "revisit
if that baseline ever wins a cell": it never does.

Their builds remain 5–7× faster than ours, which is their genuine strength (lean,
portable, dynamic).

---

## 4. Where it does *not* help (state this proactively)

Being explicit here buys credibility and costs nothing, since these are all consistent with
the same scale law:

- **Small clouds (≤10M).** Parity at best. A tuned single structure — often the plain
  default — is the right answer, and the search returns it.
- **Tiny per-block indexes (DL dataloaders, ~4k points/block).** A tuned shallow octree
  (depth 4, leaf 256) wins both DL datasets outright; searched schemas land within 1–3%.
  *Note: earlier drafts claimed 13–17% wins here — that was an artifact of training and
  reporting on the same blocks against default baselines. It did not survive the held-out
  protocol. The honest story is selection + insurance (1.9× worst-default penalty).*
- **Against a hyper-engineered incumbent.** In the ray-tracing boundary study, a single
  global SAH-built compressed wide BVH beats every nested wrapper on a scanned model, and
  the search converges to the incumbent. Correct behaviour, negative result, worth keeping
  in the paper.
- **Architecture / TLS morphology.** Flattest gains (~1.04). A well-tuned octree handles
  station-scan geometry unusually well.
- **GPU.** Not part of any claim: nested GPU evaluation has no tree-accelerated kNN
  (brute-force scan), entropy predicates and adaptive leaf capacity are CPU-only. GPU build
  is ~70× faster than CPU, which is a separate (future) story.

---

## 5. The uncomfortable finding, and why it makes the paper better

Under **budget-matched** evaluations (same number of full-fidelity candidate measurements),
plain uniform random sampling of the schema grammar matched or beat the evolutionary
search — with or without the telemetry-guided repair mutations — on the held-out half of
*every* cell. At 50M the random arm's pick was the overall winner with separated CIs; the
700M full-scale confirmation is led by uniformly sampled candidates.

Conclusion: **the value is in (a) the schema grammar spanning single/nested/conditional
designs at all parameterizations, and (b) measuring candidates on the real recorded
workload — not in how the space is traversed.**

Why this is good news:
- It's a *simplification*: practitioners can reproduce the wins with sampling + measurement.
- It kills a whole class of reviewer objection ("your GA is just an expensive random
  search") by conceding it up front with the experiment that proves it.
- The mechanism story (§3.3) explains it: the win comes from covering designs that
  hand-tuning grids don't cover, and coverage is what sampling provides.

Keep the evolutionary layer in the code as a convenience; do not claim it as a contribution.

---

## 6. What to sell

In priority order. Lead with 1–3.

1. **The regime insight.** "Point-cloud queries are dense and machine-generated, so the
   index is an inner loop whose workload is known in advance — which makes per-instance
   index synthesis both possible and economically obvious." This reframes the problem and
   is what makes everything else follow.
2. **The scale crossover, quantified.** "Above ~25M points, instance-optimized schemas beat
   the best *tuned* conventional index; below ~10M they don't, and our system tells you so."
   A method that knows its own boundary is more trustworthy than one claiming universal
   wins.
3. **The 700M anchor + the library crossover.** 2.13× over the best single at 700M;
   2.76× over FLANN and 3.7× over Open3D at 100M, with verified identical results and a
   2.4M-query break-even. This is the "it matters in practice" evidence.
4. **The insurance argument.** Wrong fixed default = 4–13× (up to 777×). Even at parity,
   measured selection is worth having. This is the most robust claim in the whole paper —
   it holds in every single cell.
5. **Do-no-harm behaviour.** The optimizer returns a plain single structure when that's
   best (verified: at 5M it returned the plain quadtree; in the ray study it returned the
   incumbent BVH). Trustworthiness is a feature.
6. **The measurement protocol as a contribution.** Held-out trace splits, budget-matched
   tuned baselines, verified result counts across libraries, provenance columns,
   discovery/reporting separation. Reviewers in this area have been burned by unfair
   comparisons; showing you engineered against that is a selling point, not boilerplate.
7. **Search cost is solved, not hand-waved.** ρ = 0.88–0.95 subsample→full transfer, top-3
   confirm at zero regret, 8× parallel dispatch. The 700M result is *only* reachable this
   way — which turns the cost story into a capability story.

---

## 7. What not to claim

- Don't say "nested structures are faster." Say "instance-optimized schemas — sometimes
  nested, sometimes an off-grid single — win above ~25M points."
- Don't say "our optimizer finds the optimum." Say "lowest measured cost in the evaluated
  design space" / "near the measured oracle."
- Don't claim the evolutionary search is a contribution (§5).
- Don't quote the old DL numbers (−17.5% / −13.4%), the pre-fix Open3D numbers (5.4×/3.7×
  from before the frame fix), or anything from `docs/evaluation.md`'s 12-cell synthetic
  study. All superseded.
- Don't claim GPU advantages (§4).
- Don't claim our primitives are faster than Indexicon's (§3.7). The exactness agreement is
  the claim. Their octree/kd-tree are faster on these traces; our octree reaches parity by
  25M, and the remaining gap is traversal cost (node locality), not instrumentation.
- Don't claim generalization across clouds beyond §3.6's "safe but not optimal."
- Don't call it a database or claim DBMS integration. The future path is per-patch/tile
  synthesis (pgPointcloud, COPC) — a follow-up systems paper, sketched in
  `plans/research_direction_v2.md` §H.

---

## 8. Questions you will be asked (and the answers)

**"Isn't this just parameter tuning?"** Partly, and we measure exactly that: the
tuned-singles arm *is* thorough parameter tuning of every primitive, budget-matched. Above
25M the schema search still beats it (1.07–2.13×). Also note both winning mechanisms
(§3.3) — one of them is off-grid single tuning, which is an argument that hand-tuning grids
are the weaker practice, not that tuning is irrelevant.

**"Why not just use a kd-tree? FLANN is very fast."** It is — and at 5M it beats us. At
100M we're 2.76× faster than FLANN with identical results, and its build advantage
amortizes after 2.4M queries, i.e. 2.4% of one pipeline stage. Scale decides.

**"Your baselines were probably untuned."** Three defenses: budget-matched tuned grids per
primitive (not defaults); external libraries (Open3D, PCL/FLANN), which we beat at 100M
*while* paying instrumentation they don't; and an independent library (Indexicon) whose
results agree with ours exactly on every query type, which is what rules out broken or
weak primitives. Note we do *not* claim to be faster than Indexicon — see §3.7 for why
that particular comparison is not currently measurable.

**"Did you overfit to the queries you tuned on?"** Every reported number is on a disjoint
held-out trace half (whole held-out blocks in the DL regime), five repeats, bootstrap CIs.

**"How expensive is the search?"** ~43 min serial for a 100M cloud, >95% of it candidate
construction. With subsample-then-confirm and 8-way parallelism: minutes plus a few
full-scale builds, at zero measured regret. Amortized over ≥1 pipeline pass (100M queries),
it's noise.

**"Does the winner transfer to a new cloud?"** Same morphology: safe but ~9% off optimal
(§3.6). Different morphology: no — that's the whole point, and the insurance numbers show
what a mismatched choice costs.

**"What about dynamic clouds / insertions?"** Out of scope; the target regime is static or
infrequently-changing clouds queried densely. Indexicon is the reference for dynamic ops.

---

## 9. What to ask for (concrete requests)

**Ask a collaborator / student for:**
- A **PCL `pcl::search::Search<PointT>` adapter** so a real PCL application (normal
  estimation, clustering, MLS) runs with our index as its backend, reporting end-to-end
  wall time, not just query latency. This is the biggest remaining gap: we have full-pass
  trace cost, not application wall time. Scoped as a conditional stretch in the plan.
- **Datasets to close the morphology story**: Paris-Lille-3D (urban street MLS, 143M,
  labeled — enables a "which structure activates where, by semantic class" figure) and
  FOR-Instance (forest UAV-LS — the volumetric morphology we have no cell for).
- A **second machine** to repeat the key cells (single-machine measurement is a stated
  threat to validity).

**Ask reviewers / readers to judge us on:**
- Held-out numbers only (`*_test.csv` / `*_testeval.csv`).
- Comparisons where result counts are verified identical.
- The scale-conditional claim, not a universal one.

**Ask yourself before submitting** (the honest gaps):
- A visited node costs ~90× a tested point in our CPU traversal (§3.7). ~10% of that was
  per-node telemetry and is now removed (1.21× on node-heavy schemas); the rest is real work,
  chiefly node-struct cache misses. Node locality is the highest-value *optimization* target
  outstanding; until it changes, our numbers favour shallow, wide structures.
- **Every C++-side number measured before 2026-08-06 came from a binary at
  `MultiDataStructure\x64\Release`, which MSBuild stopped writing to.** It was frozen at
  2026-07-28. Re-measure anything C++-dependent; `scripts/resolve_exe.ps1` now refuses a
  binary older than the sources so this cannot recur silently.
- No end-to-end application wall time yet (see PCL adapter above).
- One sampling seed per search arm; winner *families* are stable, exact schemas are not.
- Matrix cells subsample their sources, so N and density co-vary (mitigated by the
  native-density ladder).
- 700M lacks a full-scale tuned-singles sweep (cost); we report vs the best available
  single at full scale and say so.
- Discussion / conclusions / limitations section files don't exist yet in the manuscript.

---

## 10. Where everything lives

| What | Where |
|---|---|
| Battery ledger + all held-out numbers | `testing.md` §1b (results), §3b (protocol) |
| Strategy, gates, DB path, dataset plan | `plans/research_direction_v2.md` |
| Result CSVs (committed) | `results/eval_traces/v2`, `results/eval_traces/matrix` |
| Heatmap summary table | `results/eval_traces/matrix/heatmap_summary.csv` |
| Framework comparisons | `results/framework_compare/v2_*_fixed`, `v2_*_pcl`, `v2/indexicon_cmp` |
| Durable archive | `D:\MDS_results_archive\2026-07-29_battery_v2_heatmap.zip` |
| Manuscript | `nested-data-structures/` (cas-dc double column; separate git remote, gitignored here) |
| Reproduce the battery | `scripts/run_experiment_battery_v2.ps1` (`-Only <cells>`) |
| Reproduce the heatmap | `scripts/run_matrix_heatmap.ps1` |
| Rigor tooling | `scripts/split_trace.py`, `scripts/make_single_grids.py` |
| External baselines | `tools/pcl_point_baseline.cpp`, `tools/indexicon_point_baseline.cpp` |
| Indexicon build (pins the clone) | `tools/build_indexicon_baseline.ps1` |
| Indexicon baseline runs | `scripts/run_external_baselines.ps1` → `results/framework_compare/indexicon/` |
| External-baseline caveats | `docs/framework_baselines.md` |
| **Executable resolution (and the stale-binary guard)** | `scripts/resolve_exe.ps1` |

**Build output path — read this before measuring anything.** MSBuild links the solution to
`<repo>\x64\Release\MultiDataStructure.exe`. The path every runner script used until
2026-08-06, `<repo>\MultiDataStructure\x64\Release\MultiDataStructure.exe`, is a legacy
location from building the project standalone; nothing writes there any more, and the copy
sitting there was from **2026-07-28**. Scripts therefore measured a stale binary, and C++
changes appeared to have no effect. All scripts now go through `Resolve-MdsExe`, which uses
the correct path and throws if the binary predates the newest source file.

**Known code debts**: a visited node costs ~90× a tested point, dominated by node-struct
cache misses, so the CPU traversal favours shallow wide structures (§3.7; the per-node
telemetry component was removed 2026-08-06, worth 1.21× on node-heavy schemas);
`avg_latency_ms` is the *first* (cold) pass while the repeat CI covers the warm ones
(`SchemaSearch.cpp:3380` vs `:3410`), a few percent apart on the cells checked but worth
knowing before quoting either; positional shape args must precede `--sizes` in
`make_dataset_matrix.py`.

Two debts listed here before 2026-08-06 are now fixed and are no longer limitations: the
warm `--measure-repeats` loop uses per-query *k* (`SchemaSearch.cpp:3403`), and
`make_dataset_matrix.py` has a size tolerance so 99.99M fills the 100M rung.

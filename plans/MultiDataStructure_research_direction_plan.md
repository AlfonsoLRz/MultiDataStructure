# MultiDataStructure: Research Direction and Fail-Fast Evaluation Plan

> **SUPERSEDED by [`research_direction_v2.md`](research_direction_v2.md)** (28 Jul 2026),
> which fact-checks this document against the repo (several §1 claims are stale or wrong —
> see v2 §A), merges it with `testing.md`, and records the confirmed decisions.
> Kept for reference; do not quote its numbers.

**Working document — 28 July 2026**

Repository: [AlfonsoLRz/MultiDataStructure, `optimization` branch](https://github.com/AlfonsoLRz/MultiDataStructure/tree/optimization)

## 1. Executive recommendation

The project should continue, but its primary objective should **not** be to become a database. The most defensible direction is:

> **A model-free, workload-aware and telemetry-guided system that automatically synthesizes exact, spatially conditional compositions of classical indexes for heterogeneous point clouds, and demonstrably accelerates real point-cloud processing pipelines.**

This is a natural follow-up to *Nested spatial data structures for optimal indexing of LiDAR data*: the previous work introduced the ability to construct arbitrary nested schemas manually; the new project should explain **how to choose them automatically, efficiently and for a concrete workload**.

The central application should initially be a CPU point-cloud library or pipeline, preferably through a PCL-compatible search backend. Database integration should remain a possible later demonstrator. GPU schema synthesis and deep-learning neighbourhood workloads should be pursued only if the CPU contribution survives the initial fail-fast evaluation.

The project currently contains promising signals:

- The documented CPU evaluation reports a 1.13× geometric-mean improvement over the reported best single primitive across 12 synthetic dataset/workload cells.
- The largest wins, around 1.70–1.76×, occur for medium-volume queries.
- A nested schema reportedly beats an octree by approximately 1.26× on the 100M-point Alhambra cloud.
- The optimizer sometimes correctly selects a single structure when nesting is not beneficial.
- The implementation already records unusually rich traversal telemetry that could support a principled, domain-specific optimizer.

However, the present evidence is not sufficient for publication:

- only three synthetic dataset families form the main evaluation;
- most timings are close to the measurement floor, with only 3/12 cells marked as ranking-confident;
- the optimization and reported re-measurement use the same query set;
- it is not yet demonstrated that every single primitive received an equivalent tuning budget;
- the real-data evidence is one cloud and one main workload;
- optimizer cost and amortization are not part of the headline;
- the GPU path contains partial feature support, CPU fallbacks and a brute-force kNN path;
- end-to-end application improvements have not yet been shown.

The immediate goal is therefore not to add more structures. It is to determine whether the current advantage survives a rigorous, held-out, equal-budget and application-grounded evaluation.

---

## 2. Proposed scientific positioning

### 2.1 Primary claim

Given:

- a static or infrequently changing point cloud;
- a representative distribution or trace of exact spatial queries;
- an optimization budget and optional build/memory constraints;

the system synthesizes either:

- a tuned single primitive;
- a fixed nested sequence; or
- a spatially conditional nested schema whose structure varies according to local geometry;

and produces lower held-out workload cost than carefully tuned conventional indexes.

The result should remain an **exact** spatial index. No neural predictor or approximate neighbour search is required.

### 2.2 Stronger research hypothesis

The motivating hypothesis should be more specific than “different structures perform differently”:

> **Heterogeneous point clouds contain regions of different effective dimensionality, density, anisotropy and occupancy. A single global partitioning rule is consequently suboptimal, while a workload-aware system can assign different classical structures to the regions in which their partitioning behaviour is appropriate.**

Examples:

- ground and roof surfaces may favour 2D/quadtree-like subdivision;
- volumetric vegetation or interiors may favour octrees;
- elongated or anisotropic regions may favour kd-tree subdivision;
- very dense leaves may justify a leaf-local micro-index;
- different query radii and selectivities change the desirable leaf capacity and tree depth.

This hypothesis makes the method interpretable and differentiates it from generic parameter search.

### 2.3 Suggested paper sentence

> We present a model-free, telemetry-guided optimizer that synthesizes exact, spatially conditional compositions of classical indexes for heterogeneous point clouds. Given representative queries, the optimizer diagnoses where traversal work is wasted and modifies local partitioning strategies, outperforming equally tuned single-structure indexes on held-out workloads and accelerating complete point-cloud processing pipelines.

### 2.4 Claims that should be avoided

Avoid claiming:

- that the optimizer finds the globally optimal structure;
- that every heterogeneous cloud benefits from nesting;
- that GPU querying is generally faster than CPU querying;
- that the current implementation is a point-cloud database;
- that an evolved schema generalizes without held-out evidence;
- that comparison against default primitive configurations constitutes comparison against tuned single structures.

Use “best measured candidate,” “near the measured oracle,” or “lowest cost in the evaluated design space.”

---

## 3. Candidate directions, ranked

| Direction | Recommendation | Scientific opportunity | Main risk |
|---|---|---|---|
| **A. Telemetry-guided point-cloud search backend** | **Primary direction** | Automatic conditional index synthesis plus end-to-end PCL acceleration | Tuned single primitives may eliminate the gain |
| **B. Standalone self-designing spatial-index library** | Strong secondary framing | Connects to instance-optimized data systems and the new Indexicon ecosystem | Microbenchmarks alone may appear incremental |
| **C. Database integration** | Future demonstrator only | PostgreSQL/DuckDB can host custom indexes | Persistence, updates, concurrency and planner work become a separate project |
| **D. GPU schema synthesis** | Defer until CPU claim is proven | Large batches and large clouds may provide a GPU regime | Incomplete parity, fallbacks and kernel strategy currently obscure the structural result |
| **E. Deep-learning neighbourhood workloads** | High-risk optional case study | Huge repeated radius/kNN workloads offer natural amortization | Modern point networks normally use specialised GPU kernels, so a CPU trace replay is not an end-to-end result |

### 3.1 Why a PCL backend is the preferred application

The current index interface already provides the necessary basic operations:

- build;
- AABB range and count-range;
- radius;
- kNN;
- returned point indices and traversal statistics.

PCL provides a generic search abstraction with kd-tree, octree, organised-neighbour and brute-force implementations. Multiple algorithms accept a supplied search object. A `pcl::search::Search<PointT>` adapter would therefore permit controlled replacement of the search backend without inventing an artificial application.

Initial applications:

1. **Normal estimation**
   - One radius or kNN query per point.
   - Very large query counts offer clear build/tuning amortization.
   - Output equivalence is straightforward to verify.

2. **Euclidean or region-growing clustering**
   - Repeated neighbourhood queries.
   - Heterogeneous density is directly relevant.
   - Likely to be more query-bound than a full reconstruction pipeline.

3. **Moving Least Squares or related surface processing**
   - Repeated radius queries with meaningful local geometry.
   - Provides a non-trivial end-to-end application.

4. **ICP/GICP**
   - Defer initially: it is kNN-dominated, while the current experiments show little evidence that nested schemas dominate strong kNN structures.

For every application, report both search-only and full application time. A faster search backend that does not change end-to-end time is useful engineering, but a weaker application result.

---

## 4. The important new finding: Indexicon

The June 2026 preprint **Indexicon: A Spatial Indexing Library** is unusually relevant:

- Paper: [arXiv:2606.04676](https://arxiv.org/abs/2606.04676)
- Code: [psimatis/Indexicon-Spatial-Library](https://github.com/psimatis/Indexicon-Spatial-Library)

Indexicon supplies:

- a common, extensible C++ interface;
- R-tree, quadtree, octree and kd-tree implementations;
- range and kNN queries;
- bulk construction, insertion and deletion;
- structural statistics;
- six real spatial datasets and query workloads;
- comparisons against PCL, Boost Geometry and Nanoflann;
- 2D and 3D point benchmarks at meaningful scales.

### 4.1 Why this helps rather than invalidates the project

Indexicon addresses **portable, comparable implementations of individual indexes**. MultiDataStructure addresses **automatic composition and conditional selection of structures**. These are complementary contributions.

Its appearance is helpful because it provides:

- independent implementations, reducing the risk that results arise from uneven quality among in-house primitives;
- external datasets and workloads;
- a reproducible benchmark environment;
- dynamic-operation baselines if the project later expands beyond static clouds;
- a possible host interface for a nested/composite index.

### 4.2 Concrete actions

1. Check the repository licence before incorporating code.
2. Reproduce a subset of the published Indexicon results on the available CPU.
3. Add an adapter so MultiDataStructure and Indexicon can replay identical queries and return identical result counts.
4. Compare against Indexicon’s best tuned individual structure, not only its defaults.
5. Reuse at least its 3D point datasets/workloads as external spatial-index tests.
6. Keep LiDAR/TLS datasets as a separate application set: some Indexicon datasets are spatial or spatiotemporal points rather than point-cloud scans.
7. Explore, but do not initially commit to, implementing a composite/nested wrapper in the Indexicon interface.
8. Consider contacting the authors only after a clean, reproducible comparison identifies a clear complementary result.

### 4.3 Fail-fast implication

If an Indexicon single structure substantially outperforms the current MultiDataStructure implementation, first determine whether this is:

- an algorithmic limitation of nesting;
- an implementation/layout difference;
- a precision or result-materialisation difference;
- a tuning difference.

Do not hide this result. It may motivate using Indexicon primitives inside the synthesis framework. The research contribution can be the optimizer and composition semantics rather than ownership of every primitive implementation.

---

## 5. Core research questions and hypotheses

### RQ1 — Does nesting survive fair single-structure tuning?

**Hypothesis:** Nested or conditional schemas improve held-out latency on heterogeneous clouds and medium/large spatial queries after every single primitive receives an equal tuning budget.

Essential comparison:

- default single;
- tuned single;
- hand-designed nested;
- automatically synthesized nested;
- measured oracle across all evaluated candidates.

This is the first and most important fail-fast question.

### RQ2 — Is local conditionality more useful than a global depth schedule?

**Hypothesis:** Per-node structure selection based on local geometry outperforms a fixed sequence such as octree→kd-tree applied uniformly by depth.

Compare:

- single primitive;
- fixed nested sequence;
- nested sequence with only capacity/depth tuning;
- per-node conditional gates;
- conditional gates plus adaptive leaf micro-indexes.

Candidate conditions:

- bounding-box anisotropy;
- height ratio;
- density;
- occupancy entropy;
- point count;
- local extent;
- optionally PCA/eigenvalue-based linearity, planarity and scattering, if inexpensive enough.

The paper should visualise where each structure becomes active within at least one real cloud.

### RQ3 — Does telemetry-guided search outperform generic search?

**Hypothesis:** Traversal diagnostics produce better mutations and lower sample complexity than random mutation.

Compare under identical candidate-evaluation budgets:

- random search;
- vanilla evolutionary search;
- telemetry-guided repair;
- optionally Bayesian/Optuna-style black-box optimisation;
- oracle from a larger offline candidate pool where feasible.

Measure:

- best held-out latency after N evaluations;
- regret versus measured oracle;
- wall-clock tuning time;
- number of invalid, duplicated or inactive schemas;
- stability over optimizer seeds.

Potential repair logic:

| Observed bottleneck | Candidate response |
|---|---|
| Many tested points per leaf | Reduce capacity or add leaf-local index |
| Many visited nodes but few tested points | Reduce depth or increase fanout/capacity |
| Low tight-bounds ratio | Switch split rule or primitive |
| High planar anisotropy | Try quadtree/ignored-axis subdivision |
| High volumetric occupancy entropy | Try octree |
| Strong single-axis elongation | Try kd-tree split |
| Excessive single-child chains | Remove or replace unproductive levels |
| High output size | Distinguish unavoidable materialisation cost from traversal cost |

### RQ4 — Does the selected schema generalize to unseen queries?

**Hypothesis:** Optimisation over a representative trace yields low regret on independent queries from the same distribution.

Required split for every dataset/workload:

- optimisation/training trace;
- validation trace for model and search decisions;
- untouched final test trace.

A possible split is 50/20/30, with independent seeds and stratification by query type, scale, location and selectivity. The final paper must report only held-out test results as its headline.

Also test controlled workload shift:

- small queries → medium queries;
- uniform locations → density-biased locations;
- radius-heavy → mixed;
- k=8 → k=32 or k=64;
- one pipeline stage → a full pipeline mixture.

### RQ5 — When does tuning amortize?

Let:

- \(T_o\) be optimizer time;
- \(T_{b,n}\) and \(T_{b,s}\) be nested and single build times;
- \(L_n\) and \(L_s\) be average query latency.

The approximate break-even number of queries is:

\[
N_{\mathrm{break-even}} =
\frac{T_o + T_{b,n} - T_{b,s}}
     {L_s - L_n}
\]

when \(L_n < L_s\).

Report this value rather than assuming offline tuning is free. Normal estimation over millions of points may amortize even a long search; interactive desktop cropping may not.

### RQ6 — Does lower query latency improve a real application?

**Hypothesis:** At least one PCL application achieves a statistically reliable end-to-end improvement without changing its output.

Report:

- index build;
- search;
- non-search application work;
- total wall time;
- memory;
- output agreement or numerical deviation.

---

## 6. Evaluation design

### 6.1 Dataset matrix

Use datasets that deliberately vary morphology:

| Family | Desired morphology | Candidate source |
|---|---|---|
| Airborne/terrain | Predominantly 2.5D, broad and relatively flat | DALES or another open ALS cloud |
| Urban mobile mapping | Roads, façades, vehicles, poles and density changes | Paris–Lille-3D or comparable |
| Static terrestrial scan | Dense façades/interiors and occlusion patterns | Alhambra plus another TLS dataset |
| Indoor | Walls, floors, clutter and limited volume | Semantic3D/indoor alternative as licence permits |
| Vegetation/natural | Volumetric and irregular | Open forestry or natural-scene LiDAR |
| Mixed-resolution composite | ALS + TLS or deliberately merged densities | Existing synthetic and a real/composed case |
| External spatial benchmark | Non-LiDAR spatial point distributions | Indexicon 3D datasets |

Target at least:

- five real point clouds;
- three morphology classes;
- approximately 1M, 10M and 100M-point regimes where feasible;
- synthetic datasets only for controlled ablations.

### 6.2 Workload matrix

1. **AABB range retrieval**
   - small, medium and large extent;
   - report returned cardinality.

2. **Count-range**
   - separates traversal efficiency from result vector materialisation.

3. **Radius**
   - multiple radii;
   - uniform and density-biased centres.

4. **kNN**
   - at least k = 8, 16, 32 and 64;
   - inside, boundary and outside-cloud centres.

5. **Mixed synthetic profiles**
   - maintained mainly for continuity with current results.

6. **Recorded application traces**
   - PCL normal estimation;
   - clustering;
   - MLS or another selected pipeline.

Every query trace should record query parameters, expected result count and stratum. Optimizer and test traces must remain disjoint.

### 6.3 Baselines

#### Internal baselines

- brute-force scan;
- default quadtree, octree, kd-tree, BVH/BIH, LBVH, regular grid and hierarchical grid as compatible;
- each single primitive tuned with the same evaluation budget;
- original hand-designed nested schemas;
- fixed nested sequences with tuned depth/capacity;
- conditional nested schemas;
- leaf micro-index variants.

#### Search baselines

- random search;
- vanilla GA;
- GA plus telemetry repair;
- optional black-box Bayesian optimisation;
- measured oracle from the union of candidates.

#### External baselines

- Indexicon;
- PCL kd-tree and octree;
- Nanoflann for kNN/radius where compatible;
- Boost Geometry R-tree for appropriate range workloads;
- Open3D only as an application/library reference, with care because some operations use different internal paths.

External methods must execute the same exact semantics. Do not aggregate incompatible operations into one headline.

### 6.4 Fairness protocol

Control:

- identical point coordinates and precision;
- identical query traces;
- exact returned counts;
- result materialisation versus count-only semantics;
- single-threaded versus parallel execution;
- compiler, optimisation flags and architecture flags;
- warmup;
- CPU affinity where possible;
- allocator behaviour;
- cache state;
- index memory ownership;
- inclusion/exclusion of upload, build and conversion time;
- number of queries per timing batch.

Because many current measurements are in microseconds, time a full query batch and repeat it rather than relying only on per-query clocks.

### 6.5 Metrics and statistics

Primary:

- held-out average, median and p95 latency;
- geometric-mean speedup versus best tuned single;
- paired per-query latency difference;
- 95% confidence interval;
- end-to-end application time.

Secondary:

- build time;
- peak and estimated memory;
- optimizer wall time;
- candidate evaluations;
- break-even query count;
- node visits and tested points;
- returned points;
- tree depth, occupancy and fanout;
- active structure fraction;
- Pareto fronts for latency/build/memory.

Use multiple:

- query seeds;
- optimizer seeds;
- timing repetitions.

Avoid declaring a winner when confidence intervals overlap or the absolute difference is at the timer/noise floor.

---

## 7. Database direction: technically possible, strategically deferred

The assumption that no database can host a custom index is not strictly correct:

- PostgreSQL supports custom index access methods and operator classes; GiST is explicitly intended as a general framework for arbitrary tree indexing schemes.
- DuckDB supports extension-defined indexes and already exposes an R-tree through its spatial extension.

What databases do **not** provide is automatic composition of their existing indexes into arbitrary nested trees. A proper integration would require:

- persistent on-disk/page representation;
- build and scan methods;
- insertion and deletion;
- transaction and concurrency behaviour;
- recovery/WAL as applicable;
- vacuum/rebuild behaviour;
- planner selectivity and cost estimation;
- SQL operators for range, radius and kNN;
- mapping returned points to tuples or patches.

This is a separate systems project. The current implementation is a static in-memory index, so a database-first strategy would expand the scope before the central spatial claim is proven.

### Possible future demonstrator

A pragmatic two-level architecture would be:

1. PostgreSQL/PostGIS indexes coarse point-cloud tiles or patches.
2. Each selected patch carries or builds a synthesized in-memory schema.
3. SQL performs coarse filtering; MultiDataStructure performs exact in-patch queries.

This matches large point-cloud storage more naturally than one tuple per point. Pursue it only after the held-out CPU and application results are strong.

---

## 8. GPU and deep-learning branches

### 8.1 GPU

Do not use GPU results in the principal claim until:

- all compared schemas execute natively on GPU;
- CPU fallbacks are excluded from GPU comparisons;
- kNN uses an exact tree-guided implementation rather than a brute-force scan;
- query dispatch is correct across single-query and batched regimes;
- build/upload/warmup costs are consistently separated;
- parity tests validate result counts against CPU.

The GPU may ultimately have two valid regimes:

- rapid construction of very large static indexes;
- high-throughput batches or sufficiently large individual queries.

That is potentially publishable, but it currently obscures the cleaner CPU structure-synthesis question.

### 8.2 Deep-learning neighbourhood workloads

The existing PointNet++-style trace generator identifies a plausible high-query-count regime, and build time is correctly treated as first-class when indexes are rebuilt per block.

However, trace replay alone is not sufficient. Modern point networks generally execute neighbourhood grouping using specialised GPU kernels. A CPU index that accelerates a simulated trace may not improve training or inference.

Required evidence before including this direction:

- integration into a real PointNet++/PyTorch or comparable pipeline;
- full training/inference step timing;
- inclusion of data transfer and index construction;
- comparison against the framework’s actual ball-query/kNN kernels;
- identical neighbourhood semantics.

If this cannot be implemented, keep the trace experiment as an exploratory appendix, not the application headline.

---

## 9. Fail-fast gates

### Gate A — Fair structural advantage

Run CPU-only held-out tests on at least five real clouds, giving single primitives and nested candidates equal tuning budgets.

**Continue if:**

- conditional/nested schemas retain roughly ≥15% geometric-mean held-out improvement;
- multiple real dataset/workload cells have CI-separated wins;
- no single external implementation trivially dominates every candidate.

**Reconsider if:**

- the gain falls below approximately 5–10%;
- wins occur only on synthetic data;
- gains arise only from untuned baseline parameters.

### Gate B — Contribution beyond generic tuning

Compare telemetry-guided repair against random and vanilla GA under equal budgets.

**Continue as an optimizer paper if:**

- telemetry guidance reaches the same regret with at least approximately 2× fewer evaluations; or
- produces clearly better held-out schemas under a fixed realistic budget.

**Simplify if:**

- random search performs equivalently;
- NSGA-II/CMA-ES/surrogate components do not add measurable value.

In that case, retain the simplest optimizer that works and focus the paper on conditional spatial composition.

### Gate C — Conditionality

Compare per-node conditional structures against globally fixed sequences.

**Continue with the heterogeneity claim if:**

- conditionality provides repeatable gains on real mixed-morphology clouds;
- active regions are interpretable;
- the gain is not solely due to extra depth or smaller leaves.

**Pivot if:**

- a globally tuned octree/kd-tree or fixed hybrid performs equivalently.

### Gate D — Application

Integrate the most promising CPU schema into two PCL workloads.

**Continue with an application-grounded paper if:**

- at least one application improves end-to-end wall time by approximately ≥10%;
- output remains identical or numerically equivalent;
- tuning/build cost amortizes within a credible number of runs.

### Gate E — Publication decision

Proceed to a full paper only if Gates A and C pass, plus either B or D strongly passes.

If they fail, possible honest pivots are:

- an automatic selector of tuned single structures;
- a reproducible benchmark/software contribution;
- an Indexicon-compatible composite extension;
- a short/negative-result paper explaining when nesting does and does not help;
- stopping the project before further GPU/database engineering.

---

## 10. Staged implementation and experimentation roadmap

### Phase 0 — Freeze scope and preserve reproducibility

Deliverables:

- CPU-only reproducible build for the core index/search executable;
- preferably CMake/Linux support for external comparisons, while preserving the current Windows build;
- immutable experiment manifests;
- recorded hardware, compiler and flags;
- exact result-count validation;
- separated optimisation, validation and test traces.

Do not add new primitives during this phase.

### Phase 1 — Equal-budget fail-fast benchmark

Tasks:

- define tunable parameter spaces for every single primitive;
- cap every method by the same number of full-fidelity evaluations or equivalent wall budget;
- run random search and the current optimizer;
- use at least three real clouds before expanding to the full dataset matrix;
- report held-out results only.

Decision:

- if tuned singles remove the nested advantage, stop or redefine the contribution;
- if the strongest existing 1.7× and 1.26× signals persist, proceed.

### Phase 2 — Indexicon and external baselines

Tasks:

- build Indexicon and reproduce representative published cases;
- implement trace and result adapters;
- compare identical range/radius/kNN workloads;
- investigate performance gaps using visits, tested points, layout and precision;
- decide whether to use Indexicon primitives, treat them only as competitors, or implement a composite interface.

Deliverable:

- an external-baseline report with correctness parity and timing protocol.

### Phase 3 — Conditional morphology study

Tasks:

- enable CPU-native anisotropy, entropy, density and extent conditions;
- add local intrinsic-dimensionality features only if justified;
- visualise active primitive regions;
- ablate fixed versus conditional schemas;
- quantify added build/memory cost.

Deliverable:

- a mechanistic explanation of why nesting helps, not only a latency table.

### Phase 4 — Telemetry-guided optimizer

Tasks:

- formalise diagnostic categories;
- map diagnostics to constrained repair mutations;
- remove redundant generic optimisation machinery unless it adds value;
- compare best-so-far curves under equal budgets;
- analyse stability over optimizer seeds.

Deliverable:

- sample-efficiency and regret curves.

### Phase 5 — PCL application integration

Tasks:

- implement `pcl::search::Search<PointT>` adapter;
- validate radius and kNN semantics;
- integrate normal estimation and clustering/MLS;
- capture real application traces;
- tune on one trace and evaluate on a disjoint run or cloud region;
- measure search-only and total application time.

Deliverable:

- at least one credible end-to-end application result.

### Phase 6 — Scale, drift and final evaluation

Tasks:

- expand to the full real-data matrix;
- test 1M/10M/100M regimes;
- test workload shift;
- report build, memory, tuning and break-even;
- repeat key comparisons on a second CPU if available;
- freeze the final test traces before paper-wide tuning.

### Phase 7 — Optional branches

Only after the CPU paper passes:

- GPU-native parity and batching study;
- database tile/patch demonstrator;
- real deep-learning pipeline integration;
- dynamic insertion/deletion, potentially using Indexicon as a reference.

---

## 11. Paper structure if the gates pass

1. **Introduction**
   - Manual nested schemas are expressive but hard to select.
   - Local morphology and workload jointly determine index quality.
   - Contributions: conditional schema model, telemetry-guided optimizer, real application evidence.

2. **Related work**
   - Classical point-cloud/spatial indexes.
   - Original nested spatial data structures.
   - Indexicon and spatial-index libraries.
   - Data Calculator and instance-optimized systems.
   - Flood/Tsunami and workload-aware multidimensional indexes.
   - Automated/evolutionary design, while distinguishing generic black-box search.

3. **Schema design space**
   - Primitives, levels and conditions.
   - Exactness and supported operations.
   - Static versus local conditional composition.

4. **Telemetry-guided optimisation**
   - Workload traces.
   - Diagnostics.
   - Repair operators.
   - Multi-objective constraints and stopping budget.

5. **Experimental protocol**
   - Real datasets.
   - Disjoint query splits.
   - Equal-budget tuning.
   - External baselines.
   - Statistical and fairness protocol.

6. **Index-level results**
   - Held-out latency, memory, build and amortization.
   - When a single primitive wins.
   - Fixed versus conditional nesting.
   - Optimizer sample efficiency.

7. **Application results**
   - PCL integration.
   - End-to-end normal estimation/clustering/MLS.

8. **Limitations**
   - Static/in-memory focus.
   - Optimisation cost.
   - Workload drift.
   - CPU scope unless GPU parity is complete.

---

## 12. Immediate next actions

1. Freeze the GPU and database branches for the main evaluation.
2. Inspect whether the current “best single” baselines are fully tuned; implement equal-budget tuning if not.
3. Introduce strict optimisation/validation/test query separation.
4. Select three real clouds for the first fail-fast run.
5. Add random search as a budget-matched optimizer baseline.
6. Re-run the strongest `volume_small_medium` and Alhambra results on held-out traces.
7. Download/build Indexicon, verify its licence and reproduce a small benchmark.
8. Prototype the PCL search adapter only if the held-out structural gain survives.
9. Promote telemetry-guided repair to a first-class experimental arm.
10. Decide after Gates A–C whether the project merits full application integration.

---

## 13. Questions for the next research meeting

1. What exact new claim goes beyond automatic parameter tuning of the previous framework?
2. Are all single primitives currently tuned with the same budget as nested schemas?
3. Which current wins remain when optimisation and test queries are disjoint?
4. Is the strongest gain caused by mixing primitive types, conditional local choice, increased depth, or leaf-capacity tuning?
5. Can we predict beforehand which clouds contain enough local dimensional heterogeneity to benefit?
6. Which PCL application is most query-bound and easiest to instrument without semantic differences?
7. What optimisation time is acceptable, and how many queries must execute before it amortizes?
8. Should the optimizer target one workload, a distribution of workloads, or a robust schema under workload drift?
9. Can Indexicon provide our external implementations and datasets, or should it remain only a baseline?
10. If an Indexicon primitive is faster than ours, are we willing to make the synthesis method implementation-agnostic?
11. What is the minimum result that justifies another full paper rather than a software/benchmark contribution?
12. Which fail-fast outcome would persuade us to stop rather than continue adding engineering?

---

## 14. Final perspective

There is a plausible paper in this project, but its value is not that arbitrary trees can be nested—the previous work already established that. Its value would be demonstrating that:

1. local spatial heterogeneity creates predictable weaknesses in global indexes;
2. traversal telemetry can identify those weaknesses;
3. an automatic, interpretable and model-free process can repair them;
4. the resulting structure generalizes to unseen queries;
5. the improvement matters in a real point-cloud task.

Indexicon makes the timing particularly interesting: it supplies the independent index implementations and datasets that the project was missing, while leaving conditional composition and automated synthesis largely open. The next work should therefore be experimental and discriminating, not expansive. A small number of rigorous held-out tests can now determine whether the many months already invested contain a strong follow-up contribution or whether it is time for a narrower pivot.

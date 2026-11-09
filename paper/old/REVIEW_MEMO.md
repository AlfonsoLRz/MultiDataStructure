# Review memo — *Workload-Aware Synthesis of Nested Spatial Indexes for Large Point Clouds*

Target: **Computers & Graphics** (Elsevier, `cas-dc`).
Reviewed: `paper/` at 2026-08-11, after the restore described in §0.
Build at time of review: **16 pages, 0 errors, 0 undefined citations or references, 6 figures, 6 tables.**

Everything below is a proposal. Nothing in `sections/*.tex` has been reworded — the only
edits applied so far are the restore (§0) and the two new figures (§8). Each finding carries
the exact replacement text so approval is a yes/no per item.

---

## 0. What was restored before review (already applied)

The manuscript on disk this morning was three commits behind the version last built. The
`paper/` directory committed in `4f7585b` contained a pre-figures, pre-Discussion draft:
`discussion.tex` was absent, all four TikZ figures were placeholder frames, and `main.tex`
lacked the `tikz` preamble. The commits carrying that work (`042a887`, `eec19d9`, `105342b`)
existed only in a local clone that was replaced, and are not recoverable from the GitHub
remote (which is at `28b61e5`).

Restored from the version read earlier in the session, verbatim:

| Item | State now |
|---|---|
| `sections/discussion.tex` | recreated — 6 subsections + Conclusions, 11.7 KB |
| Fig. 1 pipeline overview | TikZ restored (introduction) |
| Figs. 2–4 schema / features / staged search | TikZ restored (methodology) |
| `main.tex` preamble | `tikz` + libraries, 6 `\definecolor`, `\tikzset`, `\pendingnum`, `\input{discussion}` |
| `\figureplaceholder` macro | removed (no remaining callers) |
| Stale `../paper_elsevier` header comment | corrected |

**Caveat, and it is the one thing in this memo you should personally check:** this was
reconstructed from a reading, not from git objects. Page count returned to 15 (before the
new figures) and all 27 section labels resolve, which is good corroboration, but no diff
against the lost original is possible. Please read the rendered Discussion and Conclusions
once.

---

## 1. Readiness verdict

**Not submission-ready. Four high-severity blockers, all about evidence status rather than
writing.**

| # | Blocker | Severity |
|---|---|---|
| A1 | Tables 1–2 and the abstract's central claim rest on traces the repo marks invalid above 2M points; the manuscript does not disclose this | **high** |
| A2 | Table 3's Alhambra row reports a `baseline`-tagged schema as the searched result; the real searched arm is 8.9% off, not 1.3% | **high** |
| A3 | Every table except the Indexicon results predates the 2026-08-06 stale-binary fix | **high** |
| A6 | Table 6 (boundary study) has no supporting artifact anywhere in `results/` | **high** |
| A4 | Abstract and Highlights claim wins "on every morphology" above 25M; Table 2 shows two parity cells there | medium |
| A5 | `\pendingnum` markers disagree between §5.6 and §6.2 on the same quantity | medium |
| B1–B5 | Structure, scoping, and submission-package items | medium–low |
| C1–C4 | Citation hygiene and prose | low |

The manuscript's candor is its strongest asset — it publishes its own retractions (the
withdrawn 13–17% DL headline, the kd grid bug that *reduces* its own margins, random ≥ GA).
Every proposal below preserves that voice. The fix for A1 is to extend the disclosure habit
the paper already has, not to soften any claim.

---

## 2. Provenance map

Verified by re-deriving each number from its source. "Stale-binary risk" marks results
produced before `resolve_exe.ps1` landed (2026-08-06) — per `METHOD.md` §9, every
C++-dependent measurement before that date ran a binary frozen at 2026-07-28.

| Manuscript claim | Source | Verified | Trace | Binary |
|---|---|---|---|---|
| **Table 2**, all 16 cells | `eval_traces/matrix/heatmap_summary.csv` | ✅ exact, all 16 | v1 | 07-29 ⚠ |
| **Table 4**, both rows | `framework_compare/v2_*_pcl/framework_comparison.md` | ✅ exact | v1 | 07-29 ⚠ |
| **Table 1** verdicts (CI-separated / n.s.) | `eval_traces/v2/*_test.csv` | ✅ CIs support each verdict | v1 | 07-28/29 ⚠ |
| **Table 3** SanAndreas row | `dl_sanandreas_testeval.csv` | ✅ 116.29 / 118.66 / 223.43 → 1.92× | v1 | 07-29 ⚠ |
| **Table 3** Alhambra row | `dl_alhambra_testeval.csv` | ❌ **see A2** | v1 | 07-29 ⚠ |
| Indexicon exactness (351,000 samples, 0 mismatches) | `indexicon/indexicon_summary.csv` | ✅ 13×3×3000×3 = 351,000; all mismatch cols 0 | v1 | **08-06 ✅** |
| R-tree "slowest in 8 of 13, fastest in none" | same | ✅ exactly 8 / 0 | v1 | **08-06 ✅** |
| Indexicon octree ratio progression | same | ✅ 0.48→0.568→0.94 arch; 0.399→0.761→1.007 ind | v1 | **08-06 ✅** |
| §5.7 break-even ≈ 2.4M queries | `v2_alhambra_100m_pcl` | ✅ 2,424,322 exactly | v1 | 07-29 ⚠ |
| §5.6 per-node 51–131 ns / per-point 1.1–2.8 ns | `METHOD.md` §3.7 | ⚠ **see A5** | — | mixed |

**The spacing invalidation, quantified.** `testing.md` §2b predicts v1 radii were inflated
2.03× at 5M and 5.03× at 25M. Comparing the recorded spacing directly confirms it:

| Cell | v1 spacing | v2 spacing (`estimator_version: 2`) | ratio |
|---|---|---|---|
| SanAndreas 5M | 0.8031 | 0.3932 | **2.04×** |

The v2 re-capture began 2026-08-11 and currently covers 6 cells
(`results/traces/curated/`: alhambra and sanandreas at 1M/5M/25M).

**On the 700M headline.** `configs/datasets/curated_cells.json` refuses every
`solarplantation_*` cell as 27.1% interpolated. Those refusals are for the **subsampled
tiers**; the native cloud is 728.8M and the paper uses it at native density, so the 2.13×
result is *not* refused by the new curation standard. Worth stating in the response letter
if a reviewer inspects the artifact repo — it looks alarming until the tier suffix is read.

---

## 3. Severity A — evidence integrity

### A1. The evidence status is not disclosed

Tables 1 and 2, the abstract's scale-crossover claim, and three of five Highlights derive
from v1 traces. `testing.md` §2b states plainly: *"This confounds the scale-crossover
claim… Every trace must be re-captured; version-1 traces are not comparable with version-2
above 2M points."*

The crossover claim may well survive — the bias inflates radii *more* at larger sizes, and
a reviewer who finds the notice unaided will assume the worst. Disclosing it costs little
and is consistent with how the paper already handles the frame bug.

**Proposal 1 — add to the Evaluation preamble**, after the first paragraph of §5:

> A measurement note. The pipeline traces used here set each stage's neighbourhood radius
> from an estimate of mean point spacing. An audit of that estimator found it biased at
> scale: because it sampled a fixed-size subsample rather than the full cloud, it
> overestimated spacing by roughly a factor of two at five million points and a factor of
> five at twenty-five million, which inflated query radii on the larger cells of a size
> ladder. The estimator was replaced with a local-block variant, verified within 1--14\% of
> exact across nine cells with no size-dependent bias, and traces are being re-captured
> under it. The cells re-measured with the corrected estimator are marked in Table~X; cells
> not yet re-measured are reported with the original traces and flagged, because the bias
> acts along the size axis and is therefore most relevant to the scale-crossover claim of
> Section~\ref{sec:eval-matrix}.

**Proposal 2 — wrap every affected value in `\pendingnum{}`**, the macro already in
`main.tex`. `\grep pendingnum` then enumerates exactly what the re-run must refresh. Applies
to: Table 1 (all four rows), Table 2 (all 16 cells), Table 3 (both rows), Table 4 (both
rows), Table 5, and the break-even figure in §5.7.

**Alternative, if you prefer a cleaner-looking submission:** hold the paper until the
battery re-runs and submit with v2 numbers throughout. This is the stronger option
scientifically. Under it, everything below still applies except the `\pendingnum` wrapping.

### A2. Table 3's Alhambra row reports the wrong schema — confirmed defect

`dl_alhambra_testeval.csv` has exactly one row tagged `searched`:

| schema | tag | pass_ms |
|---|---|---|
| `tuned_ot4l256` | baseline | 115.01 |
| `generated_ot11l256_kd1l32768` | **baseline** | **116.51** ← the paper reports this as "searched" |
| `generated_qt3l256` | **searched** | **125.29** ← the actual searched arm |
| … | | |
| `hgrid_default` | baseline | 241.23 |

The manuscript reports `116.5 ($+1.3\%$)` and `241.2 ($1.9\times$)`. Correct values are
**125.29 (+8.9%)** and **2.10×**. The SanAndreas row is correct (116.29 / 118.66 / 1.92×),
so this is a single-row transcription error, not a systematic one.

This matters beyond the cell: §5.3 claims searched schemas land "within 1--3\%" of block-
optimal, which holds for SanAndreas (+2.0%) but not Alhambra (+8.9%).

**Proposed Table 3 row:**

```latex
Alhambra blocks   & octree$_{4}$/256 & \textbf{115.0} & 125.3 ($+8.9\%$) & 241.2 ($2.10\times$) \\
```

**Proposed §5.3 sentence** (replacing "with the searched schemas within 1--3\% of it"):

> …a tuned shallow octree (depth 4, leaf capacity 256) wins both datasets outright, with the
> searched schema landing 2\% behind it on one dataset and 9\% behind on the other
> (Table~\ref{tab:batch}).

And the following sentence, which currently reads "the search converges to near
block-optimal without prior knowledge, while a wrongly fixed structure costs up to
$1.9\times$":

> …the search converges to within a few percent of block-optimal without prior knowledge,
> while a wrongly fixed structure costs up to $2.1\times$ per pass, every batch, every epoch.

### A3. Stale-binary exposure

`METHOD.md` §9: *"Every C++-side number measured before 2026-08-06 came from a binary at
`MultiDataStructure\x64\Release`, which MSBuild stopped writing to. It was frozen at
2026-07-28. Re-measure anything C++-dependent."*

Per the provenance map, that is every table except the Indexicon results. The kd-baseline
correction compounds this: `METHOD.md` §3.7 ends *"All matrix/battery numbers involving a
kd-tree baseline predate this fix and must be re-measured,"* and the fix makes the
tuned-single kd baseline **1.22× stronger** — it *reduces* the paper's margins wherever a
kd-tree is the best single.

The manuscript discloses the kd bug well in §5.6 but presents the corrected margins as
though already applied. Recommend folding this into the A1 measurement note, or adding after
the §5.6 kd paragraph:

> The corrected grid has not yet been propagated through the matrix and pipeline cells
> reported here; those results use the affected grid, so the margins reported against a
> kd-tree baseline are, if anything, optimistic by up to the factor above.

### A4. Abstract and Highlights overstate what Table 2 shows

The abstract says *"above twenty-five million the synthesized schema wins on every
morphology."* At 25M, Table 2 gives urban **1.03** and architecture **1.04** — parity by the
paper's own CI-separation standard, and §5.2 correctly calls architecture "flat". The claim
holds at 100M, not at 25M.

**Proposed abstract clause:**

> …whereas above twenty-five million the synthesized schema is never worse and wins on
> every morphology by one hundred million, reaching $2.13\times$ over the strongest single
> structure on a 700-million-point industrial scan…

**Proposed Highlight 3:**

> \item Held-out gains appear from 25M points and reach every morphology by 100M; below 10M a single structure wins.

Also check Highlight 5: the `4$--$13\times$` insurance figure is one `testing.md` warns will
*shrink* once the RegularGrid/HGrid branching fix propagates (a tuned grid runs 67× faster
than the default). Consider `\pendingnum` on it.

### A5. `\pendingnum` markers are inconsistent between sections

`discussion.tex:34-35` marks the per-node cost as pending — `\pendingnum{$50$--$130$~ns}`
and `\pendingnum{$1$--$3$~ns}` — while `evaluation.tex:99` states the same quantities
unbracketed as `51$--$131`\,ns and `1.1$--$2.8`\,ns. One of the two is wrong.

`METHOD.md` §3.7 reports a firmer figure from the 2026-08-06 A/B: **106 ns/node vs 1.17
ns/point** on terrain_5M, of which 6–14 ns/node was telemetry now removed (worth 1.21× on
node-heavy schemas). If those supersede the range, both sections should quote them and the
markers come off. If not, both should be marked pending. The regression is load-bearing —
§6.2, and the Conclusions' "clearest next step" both rest on it.

### A6. Table 6 (boundary study) has no artifact in the repository

Every other table in the manuscript traces to a file under `results/`. Table 6 does not. A
recursive search of `results/` for ray-tracing evidence returns nothing: no file mentions
`dragon` or `Mrays`, and the 1,137 filename matches for `bvh`/`ray` are all point-cloud
schema JSONs (`generated_bvh7l128_ot5l1024.json` and similar), not mesh or throughput
results.

The boundary study carries real argumentative weight — it is one of the three "durable
findings" in the Conclusions and appears in the abstract — so a reviewer requesting the
artifact is likely.

Three possibilities, and you know which applies: the results live outside `results/` (a
different working copy, or `D:\MDS_results_archive\`); they were produced by the separate
triangle-mesh/ray-query branch recorded in project notes and never copied back; or the
numbers predate the current tree. If the artifact exists, add it to the archive and reference
it in the data statement. If it cannot be located and re-run, the honest options are to
report the study qualitatively without Table 6, or to drop it and rely on the small-cloud and
DL-block negatives, which are fully backed.

**This is the finding most likely to be raised in review and least likely to be answerable
under time pressure — worth resolving early.**

---

## 4. Severity B — scope, structure, packaging

### B1. Conclusions live inside `discussion.tex`

`\section{Conclusions}` sits at `discussion.tex:150`. Recommend splitting to
`sections/conclusions.tex` with its own `\input` in `main.tex`. Cosmetic for the PDF, but
file structure should match document structure in a source package a copy-editor will open.

### B2. Conditional schemas are described but never evaluated

The grammar's conditional predicates appear in §3.2, are drawn in Fig. 2, and are claimed in
contribution 1 ("single, nested, conditional, and adaptive"). No conditional-vs-fixed-vs-
single experiment exists anywhere in `results/`, and `plans/research_direction_v2.md` lists
Gate C as unrun. A reviewer will ask what conditionality buys.

Do not invent a result. Two honest options — scope it explicitly, or move it to future work.
Proposed sentence for the end of §3.2:

> The conditional form is part of the representation and of the search space, but the
> evaluation below does not isolate its contribution: no cell in our battery is won by a
> schema whose advantage depends on a condition predicate rather than on its primitive
> sequence and parameters. We therefore present conditionality as a capability of the
> grammar rather than as a demonstrated source of gain, and an experiment that varies it in
> isolation --- ideally on a semantically labelled cloud, where activation can be read
> against surface class --- is left to future work.

### B3. Threats to validity — mostly addressed already

The rewritten §5.9 now covers the interpolated-tier exclusion and the nested-ladder
confound. The spacing correction is the remaining gap; A1's note covers it. No separate
change needed if A1 is adopted.

### B4. Submission-package items

- `\section*{Acknowledgements}` is "To be completed." — yours to supply, or delete.
- Affiliation has `addressline`, `city`, `postcode` commented out. Elsevier wants a full postal address.
- Single-author CRediT is present and complete.
- **Data availability statement is absent.** This matters more than usual here: `results/`
  is gitignored, and `research_direction_v2.md` §I notes every headline number lives only in
  a working copy, with a durable archive at `D:\MDS_results_archive\`. C&G requires a
  statement. The dataset-licensing limitation in §6.6 gives you the honest version — that
  two mobile-mapping scenes are non-commercial and one prohibits derivative distribution,
  so the derived tiers cannot be redistributed.
- **Dead template assets**: `figs/Fig1.pdf`, `Fig2.pdf`, `Fig3.pdf`, `grabs.pdf`,
  `pic1.pdf`, and all six `thumbnails/*.jpeg` are unreferenced Overleaf-import boilerplate
  (`\includegraphics` count across the manuscript: **0**). Delete before packaging.
- Highlights: C&G asks for ≤ 85 characters each. Measured lengths (markup stripped) are
  80 / 84 / **94** / **86** / 81 — **items 3 and 4 are over** and need trimming regardless
  of A4. Measured replacements that both fit and carry the A4 correction:

  ```latex
  \item Held-out gains from 25M points, on every morphology by 100M; below 10M a single wins.   % 85
  \item $2.13\times$ over the best single at 700M; $2.76\times$ over a FLANN kd-tree at 100M.   % 71
  ```

  If you would rather keep item 3 shorter still, `Instance-optimized schemas win from 25M
  points, on every morphology by 100M.` is 76 and pairs with moving the small-cloud claim
  into item 5.

### B5. Figure count and placement

Now six figures: four conceptual diagrams and the two data figures added in §8. That is a
reasonable balance for a measurement paper. Both new figures are full-width `figure*` floats
and land adjacent to the material they support (Fig. 5 above §5.2; Fig. 6 in §5.7).

---

## 5. Severity C — citations and prose

### C1. Bibliography

- 37 keys cited, all resolve. **2 uncited entries**: `liu2023nlos`, `schutz2026curast` —
  remove or cite.
- **7 cited entries lack a DOI**: `bikker2024tinybvh`, `chaudhuri1997indexselection`,
  `idreos2007databasecracking`, `indexicon2026`, `meagher1980octree`, `qi2017pointnetpp`,
  `samet2006foundations`. Elsevier prefers DOIs where they exist. `qi2017pointnetpp` and
  `chaudhuri1997indexselection` certainly have them.
- **11 cited works are arXiv/preprint.** Several have published versions worth citing
  instead: `nathan2020flood` (SIGMOD'20), `pandey2020learnedspatial`, `zhou2018open3d`,
  `li2022psnet`.

### C2. Terminology

Consistent throughout — *schema*, *primitive*, *tuned single*, *searched*, and *trace* each
hold one meaning. No changes proposed. One note: "morphology" and "shape" are used
interchangeably (Table 2's axis says "shape", the abstract says "morphology"). Harmless, but
picking one would tighten it.

### C3. Prose

No stock or AI-like phrasing found in a scan of all six sections: zero instances of
"delve into", "it is important to note", "plays a crucial role", "leverage", "underscore",
"paradigm shift", or the other flagged patterns. No duplicated sentences across sections. No
sentence-opening repetition above threshold. Dash style is consistent (`---` throughout, no
Unicode em-dashes). Prose quality is high and needs no editing pass.

### C4. Overfull boxes

Nine, all minor: one 123pt box at `\maketitle` (the CAS title block, pre-existing and not
introduced by any edit here), and eight paragraph-level boxes of 0.5–21pt. The pre-restore
baseline had six, so the restore and figures added three. None affect the printed page.
Worth a `\sloppy` pass on the 21pt one at `evaluation.tex:29-30` before submission.

---

## 6. What I could not check

- **Fidelity of the restored Discussion/Conclusions** — reconstructed from a reading; no
  diff against the lost commits is possible (§0).
- **Whether the corrected kd grid has been propagated** into any reported cell. I inferred
  "no" from file dates and `METHOD.md` §3.7; you can confirm directly.
- **Whether the v2 re-capture changes any conclusion** — only 6 cells exist so far, and no
  battery has been run against them.
- **Boundary-study numbers (Table 6)** — see A6. Every other table traces to a CSV; this
  one has no artifact under `results/` at all.
- **Search-cost numbers (Table 5)** — the wall-times and ρ values appear in `testing.md`
  §1b but I did not locate primary logs for the 42.9-minute anchor.

---

## 7. Functional-completeness retrospective

1. **Lifecycle stage / artifacts covered?** Yes for the manuscript sources and its evidence base. Not covered: the response-letter and cover-letter artifacts (not yet written).
2. **Locked meaning preserved?** Yes — no wording changed. All proposals are quoted, not applied.
3. **Question → method → result → interpretation → summary aligned?** Yes, except conditionality (B2), which is claimed in the contributions and grammar but has no result.
4. **Adapters applied?** Computational/simulation-study adapter. Held-out protocol, budget-matched baselines, ablation, and exactness cross-validation all present and verified.
5. **Change-impact scan?** Ran on both directions for A2 and A4 — each finding lists every downstream sentence affected, not only the table cell.
6. **Unknowns / blockers?** Listed in §6. Two high-severity blockers (A1, A2) remain open by design; this memo is diagnosis, not application.
7. **Deterministic checks used, limits stated?** Yes: bibliography, prose-pattern, and provenance checks were scripted and re-derived from source CSVs. The prose scan detects observable patterns only — it is not authorship detection. The `academic-writing-skills` bundled auditors require a project-state file this project does not have, so equivalent checks were scripted directly; that substitution is a stated limit.
8. **Exact deliverable inspected?** Yes — findings are against `paper/` as built at 16 pages, including your edits through 13:03.
9. **Post-edit candidate gate?** Not applicable this round: no manuscript wording was changed. It applies when the §3–§5 proposals are accepted, and will be run on the exact revised passages then.

---

## 8. Figures added this round (applied, verifiable)

Two data figures were built from source CSVs, per the approved plan. Both use the
manuscript's existing accent pair (`mdsaccent` blue, `mdsout` orange), which was validated
as a diverging/emphasis pair: CVD ΔE 19.7, normal-vision ΔE 24.7, contrast ≥ 3:1 — passing
every gate.

**Fig. 5 — shape × size heatmap** (`sections/fig_matrix_heatmap.tex`), from
`heatmap_summary.csv`. Diverging blue↔orange centred at 1.0 so the parity boundary is the
visual centre, with the 10–25M crossover marked. Every cell also prints its value, so the
encoding never rests on colour alone — it survives greyscale printing and CVD.

**Fig. 6 — amortization** (`sections/fig_amortization.tex`), from
`v2_alhambra_100m_pcl/framework_comparison.md`. Cumulative wall cost vs. query volume,
break-even marked, one-pipeline-pass reference line. Both series directly labelled.

**Verification** (`scratchpad/verify_figs.py`, `verify_curves.py`): all 16 heatmap cells
re-derived from the CSV and matched exactly; both amortization curves checked point-by-point
against their own cost formulas; break-even re-derived independently as **2,424,322
queries**, matching the paper's "~2.4M". Values were re-derived from source, never
transcribed.

Both figures inherit the A1 caveat — they visualize v1-trace data and should carry the same
pending marking as the tables they illustrate.

# Experiment battery v2 — rigor retrofits R1-R3 (plans/research_direction_v2.md §C).
#
# Differences vs run_experiment_battery.ps1 (v1, kept for provenance):
#   R1  every SEARCH consumes the *_opt trace half; every HEADLINE number comes from a
#       final re-measure on the untouched *_test half (scripts/split_trace.py outputs).
#   R2  a tuned-singles arm (configs/schemas/tuned_singles*, scripts/make_single_grids.py)
#       replaces "best default single" with "best tuned single" in every cell.
#   R3  budget-matched optimizer arms per cell: GA vs GA+--repair-mutations vs pure
#       random (--generated-only --benchmark-top 0, no surrogate prune). The random
#       arm's budget = the evaluation count of the cell's GA CSV, so arms are matched
#       on full-fidelity evaluations, not wall time.
#
# Analysis rules (enforced downstream, not here):
#   - report only *_test.csv / *_testeval.csv rows in paper tables;
#   - in batch test evals ignore the single stray generated_* row (batch mode floors
#     --generate-schemas at 1; BatchSearch.cpp:181 turns 0 into 64);
#   - never mix gpu_support_status backends in one ranking.
#
# Gotchas inherited from v1: --no-score-cache on every trace run; check
# returned_points > 0 in any new replay; delete partial CSVs before reruns.
#
# Run everything:            .\scripts\run_experiment_battery_v2.ps1
# Run selected cells only:   .\scripts\run_experiment_battery_v2.ps1 -Only sanandreas_5m,dl_alhambra
# Cell names: sanandreas_5m sanandreas_50m alhambra_100m dl_sanandreas dl_alhambra
#             solar_700m rank_transfer frameworks
param([string[]]$Only)
function Test-CellSelected($name) { return (-not $Only) -or ($Only -contains $name) }

$ErrorActionPreference = 'Continue'
$repo = 'c:\Github\MultiDataStructure'
$py = Join-Path $repo '.venv\Scripts\python.exe'
$pyMds = Join-Path $repo '.venv-mds\Scripts\python.exe'
. "$PSScriptRoot\resolve_exe.ps1"
# MSBuild links to <repo>\x64\Release; MultiDataStructure\x64\Release is a stale legacy copy
# that nothing writes any more. Resolve-MdsExe also refuses a binary older than the sources.
$exe = Resolve-MdsExe -Repo $repo
Set-Location $repo

$outDir = 'results/eval_traces/v2'
New-Item -ItemType Directory -Force -Path $outDir, "$outDir/best_schemas", "$outDir/generated" | Out-Null

$tuned = (Get-Content configs/schemas/tuned_singles/index.txt) -join ';'          # 107 schemas (5M/50M cells)
$tunedSmall = (Get-Content configs/schemas/tuned_singles_small/index.txt) -join ';' # 36 schemas (100M+/batch cells)
$defaults = @('quadtree', 'octree', 'kdtree', 'bvh', 'bih', 'hgrid', 'lbvh', 'regular_grid', 'karras_octree' |
  ForEach-Object { "configs/schemas/$_.json" }) -join ';'
$handNested = 'configs/schemas/octree_kdtree.json;configs/schemas/quadtree_octree.json'

# Steps that failed, reported together at the end. A cell whose search arms all failed still
# produced a plausible-looking *_test.csv (defaults only, no winners) and read as a legitimate
# negative result, so failures must be impossible to miss.
$script:FailedSteps = @()

function Step($name, $script) {
  Write-Output "`n########## $name  [$(Get-Date -Format HH:mm:ss)] ##########"
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $global:LASTEXITCODE = 0
  & $script
  $code = $LASTEXITCODE
  $sw.Stop()
  if ($code -ne 0) { $script:FailedSteps += "$name (exit $code)" }
  Write-Output "########## $name done in $([math]::Round($sw.Elapsed.TotalMinutes,1)) min (exit $code) ##########"
  return $code
}

# The C++ --csv writers APPEND (PointBenchmark.cpp appendCsvSummary, header-only-if-empty), so
# a re-run leaves the previous run's rows in place. Get-BestSchemaPath sorts over the whole
# file and would happily return a winner from an older run with different parameters, and
# Get-EvalBudget counts unique schema names across all of it, inflating the budget-matched
# random arm. Every step clears its own target first.
function Reset-Csv([string[]]$paths) {
  foreach ($p in $paths) { if ($p -and (Test-Path $p)) { Remove-Item -Force $p } }
}

# Lowest-score schema_path per unique schema in a results CSV (winner of an arm).
function Get-BestSchemaPath($csv) {
  foreach ($r in (Import-Csv $csv | Sort-Object { [double]$_.score })) {
    if ($r.schema_path -and (Test-Path $r.schema_path)) { return $r.schema_path }
  }
  return $null
}

function Get-EvalBudget($csv) {
  return @(Import-Csv $csv | ForEach-Object { $_.schema_name } | Select-Object -Unique).Count
}

# One pipeline cell: GA / GA+repair / random / tuned-singles arms on the _opt trace,
# then a single --measure-repeats 5 re-measure of all winners + defaults on _test.
function Invoke-PipelineCell($cell, $cloud, $traceDir, $queries, $genSchemas, $generations, $population, $tunedList, $extraGaArgs) {
  $opt = "$traceDir/pipeline_trace_opt.csv"
  $test = "$traceDir/pipeline_trace_test.csv"

  # Guard before launching anything (run_matrix_heatmap.ps1 already did this; this script did
  # not). Without it a missing trace lets all five steps fail and the cell still emits a
  # *_test.csv containing only the default schemas.
  foreach ($required in @($cloud, $opt, $test)) {
    if (-not (Test-Path $required)) {
      Write-Output "$cell : missing $required - skipping cell"
      $script:FailedSteps += "$cell (missing $required)"
      return
    }
  }

  Reset-Csv @("$outDir/${cell}_ga.csv", "$outDir/${cell}_repair.csv",
              "$outDir/${cell}_random.csv", "$outDir/${cell}_singles.csv",
              "$outDir/${cell}_test.csv")

  # Separate output dirs per arm: the exported best-schema JSON is named from
  # dataset+workload only, so a shared dir would let the repair run overwrite the
  # GA winner before the test step reads it (bit us on the first 5M run).
  Step "$cell GA (opt)" {
    & $exe --mode schema-search --input $cloud --no-synthetic `
      --workloads configs/workloads/pipeline_replay.json --input-trace $opt `
      --queries $queries --evaluator cpu --generate-schemas $genSchemas `
      --optimizer-generations $generations --optimizer-population $population `
      --optimizer-output-dir "$outDir/best_schemas/${cell}_ga" --no-score-cache --no-pause `
      @extraGaArgs --csv "$outDir/${cell}_ga.csv"
  }
  Step "$cell GA+repair (opt)" {
    & $exe --mode schema-search --input $cloud --no-synthetic `
      --workloads configs/workloads/pipeline_replay.json --input-trace $opt `
      --queries $queries --evaluator cpu --generate-schemas $genSchemas `
      --optimizer-generations $generations --optimizer-population $population --repair-mutations `
      --optimizer-output-dir "$outDir/best_schemas/${cell}_repair" --no-score-cache --no-pause `
      @extraGaArgs --csv "$outDir/${cell}_repair.csv"
  }
  Step "$cell random (opt, budget-matched)" {
    $budget = Get-EvalBudget "$outDir/${cell}_ga.csv"
    Write-Output "  random budget = $budget (from ${cell}_ga.csv)"
    & $exe --mode schema-search --input $cloud --no-synthetic --generated-only `
      --generate-schemas $budget --benchmark-top 0 --no-baselines --generated-seed 7919 `
      --generated-schema-dir "$outDir/generated/$cell" `
      --workloads configs/workloads/pipeline_replay.json --input-trace $opt `
      --queries $queries --evaluator cpu --no-score-cache --no-pause `
      --csv "$outDir/${cell}_random.csv"
  }
  Step "$cell tuned singles (opt)" {
    & $exe --mode schema-search --flat-search --evaluator cpu --input $cloud --no-synthetic `
      --schemas $tunedList --generate-schemas 0 --no-baselines `
      --workloads configs/workloads/pipeline_replay.json --input-trace $opt `
      --queries $queries --no-score-cache --no-pause --csv "$outDir/${cell}_singles.csv"
  }
  $winners = @("$outDir/${cell}_ga.csv", "$outDir/${cell}_repair.csv",
               "$outDir/${cell}_random.csv", "$outDir/${cell}_singles.csv" |
    Where-Object { Test-Path $_ } | ForEach-Object { Get-BestSchemaPath $_ } |
    Where-Object { $_ }) | Select-Object -Unique
  if (-not $winners) {
    # Every search arm produced nothing. Re-measuring here would emit a headline CSV holding
    # only the default schemas, which is indistinguishable from "the search lost to the
    # defaults" - the most damaging way this script can fail.
    Write-Output "$cell : all search arms produced no winner - refusing to write a headline CSV"
    $script:FailedSteps += "$cell (no winners from any arm; TEST skipped)"
    return
  }

  Step "$cell TEST re-measure (headline source)" {
    $list = (@($winners) + ($defaults -split ';') + ($handNested -split ';')) -join ';'
    & $exe --mode schema-search --flat-search --evaluator cpu --input $cloud --no-synthetic `
      --schemas $list --generate-schemas 0 --no-baselines `
      --workloads configs/workloads/pipeline_replay.json --input-trace $test `
      --queries $queries --measure-repeats 5 --no-score-cache --no-pause `
      --csv "$outDir/${cell}_test.csv"
  } | Out-Null
}

# Batch CSVs have their own format (schema,tag,pass_ms,... — no schema_path/score),
# so the batch cells use dedicated helpers. Searched candidates only exist on disk
# because the search passes --generated-schema-dir (BatchSearch persists the pool and
# repair children there); winners resolve by name -> <dir>/<name>.json.
function Get-BatchWinnerPath($csv, $nameLike, [string[]]$searchDirs) {
  foreach ($r in (Import-Csv $csv | Sort-Object { [double]$_.pass_ms })) {
    if ($r.schema -notlike $nameLike) { continue }
    foreach ($dir in $searchDirs) {
      $candidate = Join-Path $dir "$($r.schema).json"
      if (Test-Path $candidate) { return $candidate }
    }
  }
  return $null
}

function Get-BatchEvalBudget($csv) {
  return @(Import-Csv $csv | ForEach-Object { $_.schema } | Select-Object -Unique).Count
}

# One DL batch cell: repair-search / random / tuned-singles arms on the _opt manifest,
# then all winners + canonical baselines evaluated on the _test manifest.
# Batch mode floors --generate-schemas at 1 (a 0 request becomes 64), so eval-only
# runs pass 1 and analysis drops the lone generated_* row.
function Invoke-BatchCell($cell, $manifestDir, $generations) {
  $opt = "$manifestDir/batch_manifest_opt.json"
  $test = "$manifestDir/batch_manifest_test.json"
  $genDir = "$outDir/generated/$cell"
  $tunedDirs = @(Get-ChildItem -Directory configs/schemas/tuned_singles_small | ForEach-Object { $_.FullName })

  Step "$cell batch repair-search (opt)" {
    & $exe --mode batch-search --batch-manifest $opt `
      --generate-schemas 64 --optimizer-generations $generations --generated-schema-dir $genDir `
      --no-score-cache --no-pause --csv "$outDir/${cell}_search.csv"
  }
  Step "$cell batch random (opt, budget-matched)" {
    $budget = Get-BatchEvalBudget "$outDir/${cell}_search.csv"
    Write-Output "  random budget = $budget (from ${cell}_search.csv)"
    & $exe --mode batch-search --batch-manifest $opt `
      --generate-schemas $budget --optimizer-generations 1 --generated-schema-dir "$genDir-random" `
      --no-score-cache --no-pause --csv "$outDir/${cell}_random.csv"
  }
  Step "$cell batch tuned singles (opt)" {
    & $exe --mode batch-search --batch-manifest $opt `
      --schemas $tunedSmall --generate-schemas 1 --optimizer-generations 1 `
      --no-score-cache --no-pause --csv "$outDir/${cell}_singles.csv"
  }
  Step "$cell batch TEST eval (headline source)" {
    $winners = @(
      (Get-BatchWinnerPath "$outDir/${cell}_search.csv" '*' @($genDir)),
      (Get-BatchWinnerPath "$outDir/${cell}_random.csv" 'generated_*' @("$genDir-random")),
      (Get-BatchWinnerPath "$outDir/${cell}_singles.csv" 'tuned_*' $tunedDirs)
    ) | Where-Object { $_ } | Select-Object -Unique
    $list = (@($winners) + ($defaults -split ';') + ($handNested -split ';')) -join ';'
    & $exe --mode batch-search --batch-manifest $test `
      --schemas $list --generate-schemas 1 --optimizer-generations 1 `
      --no-score-cache --no-pause --csv "$outDir/${cell}_testeval.csv"
  }
}

# ---------------- pipeline cells ----------------
# 5M and Alhambra run every arm SERIAL so the R3 arm comparison (GA vs repair vs
# random vs singles) selects under identical measurement noise; --parallel candidate
# dispatch makes concurrent builds/queries compete for memory bandwidth and would
# give the GA arms a different noise regime than the (always-serial) flat arms.
# 50M keeps --parallel 8 for wall time; draw R3 conclusions from 5M + Alhambra.
if (Test-CellSelected 'sanandreas_5m') {
  Invoke-PipelineCell 'sanandreas_5m'  "D:\Datasets\Point Clouds\SanAndreas\5M.las" `
    'results/traces/sanandreas_5m'  3000 32 3 24 $tuned @()
}
if (Test-CellSelected 'sanandreas_50m') {
  Invoke-PipelineCell 'sanandreas_50m' 'results/traces/sanandreas_50m/50M_converted.las' `
    'results/traces/sanandreas_50m' 1500 24 2 12 $tuned @('--parallel', '8')
}
if (Test-CellSelected 'alhambra_100m') {
  Invoke-PipelineCell 'alhambra_100m'  "D:\Datasets\Point Clouds\Alhambra\Alhambra_100M.las" `
    'results/traces/alhambra_100m'  1000 16 2 10 $tunedSmall @()   # serial: the search-cost anchor
}

# ---------------- DL batch cells ----------------
if (Test-CellSelected 'dl_sanandreas') { Invoke-BatchCell 'dl_sanandreas' 'results/traces/dl_sanandreas' 3 }
if (Test-CellSelected 'dl_alhambra')   { Invoke-BatchCell 'dl_alhambra'   'results/traces/dl_alhambra'   3 }

# ---------------- 700M multi-fidelity (search on subsample _opt, confirm at full, test at full) ----------------
if (Test-CellSelected 'solar_700m') {
Step '700M GA on subsample (opt)' {
  & $exe --mode schema-search --input 'results/traces/solarpanels_700m/solar_sub5m.las' --no-synthetic `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/solarpanels_700m/pipeline_trace_opt.csv `
    --queries 1500 --evaluator cpu --generate-schemas 32 --optimizer-generations 3 --optimizer-population 24 --parallel 8 `
    --optimizer-output-dir "$outDir/best_schemas_700m" --no-score-cache --no-pause `
    --csv "$outDir/solar_sub5m.csv"
}
Step '700M tuned singles on subsample (opt)' {
  & $exe --mode schema-search --flat-search --evaluator cpu `
    --input 'results/traces/solarpanels_700m/solar_sub5m.las' --no-synthetic `
    --schemas $tunedSmall --generate-schemas 0 --no-baselines `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/solarpanels_700m/pipeline_trace_opt.csv `
    --queries 1500 --no-score-cache --no-pause --csv "$outDir/solar_sub5m_singles.csv"
}
Step '700M confirm top-10 at full scale (opt)' {
  $unique = [ordered]@{}
  foreach ($r in (Import-Csv "$outDir/solar_sub5m.csv" | Sort-Object { [double]$_.score })) {
    if (-not $unique.Contains($r.schema_name) -and $r.schema_path -and (Test-Path $r.schema_path)) { $unique[$r.schema_name] = $r.schema_path }
    if ($unique.Count -ge 10) { break }
  }
  $best = Get-BestSchemaPath "$outDir/solar_sub5m_singles.csv"
  if ($best) { $unique['best_tuned_single'] = $best }
  $schemaList = (@($unique.Values) | Select-Object -Unique) -join ';'
  Write-Output "confirming $((($schemaList -split ';')).Count) schemas at 700M"
  & $exe --mode schema-search --flat-search --evaluator cpu `
    --input "D:\Datasets\Point Clouds\SolarPanels\SolarPanels700M.las" --no-synthetic `
    --schemas $schemaList --generate-schemas 0 `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/solarpanels_700m/pipeline_trace_opt.csv `
    --queries 1000 --no-score-cache --no-pause --csv "$outDir/solar_700m_confirm.csv"
}
Step '700M TEST re-measure (headline source)' {
  $winner = Get-BestSchemaPath "$outDir/solar_700m_confirm.csv"
  $list = (@($winner) + ($defaults -split ';') + ($handNested -split ';') | Where-Object { $_ }) -join ';'
  & $exe --mode schema-search --flat-search --evaluator cpu `
    --input "D:\Datasets\Point Clouds\SolarPanels\SolarPanels700M.las" --no-synthetic `
    --schemas $list --generate-schemas 0 --no-baselines `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/solarpanels_700m/pipeline_trace_test.csv `
    --queries 1000 --measure-repeats 5 --no-score-cache --no-pause --csv "$outDir/solar_700m_test.csv"
}
}

# ---------------- search-cost artifacts (opt side; unchanged science from v1) ----------------
if (Test-CellSelected 'rank_transfer') {
Step 'rank transfer sub2m + sub5m (opt)' {
  $schemaList = (Import-Csv "$outDir/alhambra_100m_ga.csv" | ForEach-Object { $_.schema_path } |
    Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique) -join ';'
  foreach ($sub in @('sub2m', 'sub5m')) {
    & $exe --mode schema-search --flat-search --evaluator cpu `
      --input "results/traces/alhambra_100m/alhambra_$sub.las" --no-synthetic `
      --schemas $schemaList --generate-schemas 0 --no-baselines `
      --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/alhambra_100m/pipeline_trace_opt.csv `
      --queries 1000 --no-score-cache --no-pause --csv "$outDir/alhambra_$sub.csv"
  }
  & $py scripts/analyze_rank_transfer.py --full "$outDir/alhambra_100m_ga.csv" `
    --low "$outDir/alhambra_sub2m.csv" "$outDir/alhambra_sub5m.csv"
}
}

# ---------------- framework comparisons (TEST half only) ----------------
if (Test-CellSelected 'frameworks') {
Step 'framework compare 5M + Alhambra (test)' {
  & $pyMds scripts/compare_frameworks.py --exe $exe `
    --input "D:\Datasets\Point Clouds\SanAndreas\5M.las" `
    --schema (Get-BestSchemaPath "$outDir/sanandreas_5m_ga.csv") `
    --workload-profile configs/workloads/pipeline_replay.json `
    --input-trace results/traces/sanandreas_5m/pipeline_trace_test.csv `
    --queries 3000 --frameworks open3d --out-dir results/framework_compare/v2_sanandreas_5m
  & $pyMds scripts/compare_frameworks.py --exe $exe `
    --input "D:\Datasets\Point Clouds\Alhambra\Alhambra_100M.las" `
    --schema (Get-BestSchemaPath "$outDir/alhambra_100m_ga.csv") `
    --workload-profile configs/workloads/pipeline_replay.json `
    --input-trace results/traces/alhambra_100m/pipeline_trace_test.csv `
    --queries 3000 --frameworks open3d --out-dir results/framework_compare/v2_alhambra_100m
}
}

Write-Output "`n########## battery v2 complete [$(Get-Date -Format HH:mm:ss)] ##########"
Write-Output 'Headline sources: results/eval_traces/v2/*_test.csv and *_testeval.csv ONLY.'

if ($script:FailedSteps.Count -gt 0) {
  Write-Output "`n########## $($script:FailedSteps.Count) STEP(S) FAILED ##########"
  $script:FailedSteps | ForEach-Object { Write-Output "  FAILED: $_" }
  Write-Output 'Results above are INCOMPLETE. Do not quote a cell whose steps appear here.'
  exit 1
}
Write-Output 'All steps exited 0.'

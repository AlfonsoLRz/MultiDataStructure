# Morphology x scale heatmap battery (testing.md section 4): per cell run a GA arm and
# the tuned-singles arm on the _opt trace half, then re-measure winners + all default
# singles + hand-designed nested on the _test half (serial, 5 repeats).
# Gate B is settled (random >= GA), so no repair/random arms here - the heatmap needs
# the winner FAMILY per cell, not another optimizer comparison.
#
# Cells come from a config written by scripts/make_cell_config.py, not from literals:
# the dataset root, the cell list and the per-cell budgets all live in one JSON so that
# re-pointing at a different dataset is not a PowerShell edit.
#
# Output: results/eval_traces/matrix/<cell>_{ga,singles,test}.csv
param(
  [string[]]$Only,
  [string]$CellConfig = 'configs/datasets/curated_cells.json',
  [switch]$SkipCapture
)
function Test-CellSelected($name) { return (-not $Only) -or ($Only -contains $name) }

$ErrorActionPreference = 'Continue'
$repo = 'c:\Github\MultiDataStructure'
$py = Join-Path $repo '.venv\Scripts\python.exe'
. "$PSScriptRoot\resolve_exe.ps1"
# MSBuild links to <repo>\x64\Release; MultiDataStructure\x64\Release is a stale legacy copy.
$exe = Resolve-MdsExe -Repo $repo
Set-Location $repo

if (-not (Test-Path $CellConfig)) {
  throw "cell config not found: $CellConfig - run scripts/make_cell_config.py first"
}
$config = Get-Content $CellConfig -Raw | ConvertFrom-Json
$cells = $config.cells
Write-Output "cell config: $CellConfig ($($cells.Count) cells, profile $($config.profile))"
if ($config.refused.Count) {
  Write-Output "  $($config.refused.Count) cell(s) refused by the config as not fully measured"
}

$outDir = 'results/eval_traces/matrix'
New-Item -ItemType Directory -Force -Path $outDir, "$outDir/best_schemas" | Out-Null

$tunedSmall = (Get-Content configs/schemas/tuned_singles_small/index.txt) -join ';'
$defaults = @('quadtree', 'octree', 'kdtree', 'bvh', 'bih', 'hgrid', 'lbvh', 'regular_grid', 'karras_octree' |
  ForEach-Object { "configs/schemas/$_.json" }) -join ';'
$handNested = 'configs/schemas/octree_kdtree.json;configs/schemas/quadtree_octree.json'

$script:FailedSteps = @()
# The parameter names are deliberately obscure. PowerShell scoping is dynamic: a
# scriptblock passed in here executes inside this function's scope, so any variable
# it references that happens to share a name with a parameter of Step resolves to
# Step's parameter, not the caller's variable. With a parameter called $name, every
# "${name}_ga.csv" inside a step silently became "<cell> GA (opt)_ga.csv" - the GA
# results went to mangled filenames, the winner lookup found nothing, and the test
# re-measure quietly ran with defaults only. An entire battery passed with exit 0
# while measuring the wrong thing.
function Step($stepTitle, $stepBody) {
  Write-Output "`n########## $stepTitle  [$(Get-Date -Format HH:mm:ss)] ##########"
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $global:LASTEXITCODE = 0
  $failure = $null
  try { & $stepBody } catch { $failure = $_.Exception.Message }
  $code = $LASTEXITCODE
  $sw.Stop()
  if ($failure) { Write-Output "STEP FAILED: $failure" }
  if ($failure -or $code -ne 0) { $script:FailedSteps += "$stepTitle (exit $code)" }
  Write-Output "########## $stepTitle done in $([math]::Round($sw.Elapsed.TotalMinutes,1)) min (exit $code) ##########"
  return ($null -eq $failure -and $code -eq 0)
}
function Get-BestSchemaPath($csv) {
  if (-not (Test-Path $csv)) { return $null }
  foreach ($r in (Import-Csv $csv | Sort-Object { [double]$_.score })) {
    if ($r.schema_path -and (Test-Path $r.schema_path)) { return $r.schema_path }
  }
  return $null
}

# Tier C is not searched: at 250M and above the GA would cost days for a claim that
# only needs the winner to hold up as the cloud grows. The donor is the largest
# searched rung of the same scene.
function Get-TransferWinners($cell) {
  $donors = $cells |
    Where-Object { $_.scene -eq $cell.scene -and $_.arm -eq 'search' } |
    Sort-Object -Property points -Descending
  foreach ($donor in $donors) {
    $winners = @("$outDir/$($donor.cell)_ga.csv", "$outDir/$($donor.cell)_singles.csv" |
      ForEach-Object { Get-BestSchemaPath $_ } | Where-Object { $_ }) | Select-Object -Unique
    if ($winners) {
      Write-Output "$($cell.cell): transferring winners from $($donor.cell)"
      return $winners
    }
  }
  Write-Output "$($cell.cell): no searched donor rung has a winner yet"
  return @()
}

function Invoke-Cell($cell) {
  $name = $cell.cell
  $cloud = $cell.mdspc_path
  $opt = "$($cell.trace_dir)/pipeline_trace_opt.csv"
  $test = "$($cell.trace_dir)/pipeline_trace_test.csv"

  if (-not (Test-Path $cell.laz_path)) { Write-Output "$name : cloud missing - skipping"; $script:FailedSteps += "$name (cloud missing)"; return }
  if (-not (Test-Path $cloud)) { Write-Output "$name : .mdspc missing - run scripts/laz_to_mdspc.py"; $script:FailedSteps += "$name (mdspc missing)"; return }
  if (-not (Test-Path $opt))   { Write-Output "$name : opt trace missing - skipping"; $script:FailedSteps += "$name (opt trace missing)"; return }
  if (-not (Test-Path $test))  { Write-Output "$name : test trace missing - skipping"; $script:FailedSteps += "$name (test trace missing)"; return }

  # The C++ --csv writer appends, so a re-run would leave the previous run's rows in place and
  # Get-BestSchemaPath (which sorts the whole file) could return a winner from an older run.
  foreach ($stale in @("$outDir/${name}_ga.csv", "$outDir/${name}_singles.csv", "$outDir/${name}_test.csv")) {
    if (Test-Path $stale) { Remove-Item -Force $stale }
  }

  $searched = $false
  if ($cell.arm -eq 'search') {
    $ga = Step "$name GA (opt)" {
      & $exe --mode schema-search --input $cloud --no-synthetic `
        --workloads configs/workloads/pipeline_replay.json --input-trace $opt `
        --queries $cell.queries --evaluator cpu --generate-schemas $cell.gen_schemas `
        --optimizer-generations $cell.generations --optimizer-population $cell.population --parallel 8 `
        --optimizer-output-dir "$outDir/best_schemas/${name}_ga" --no-score-cache --no-pause `
        --csv "$outDir/${name}_ga.csv"
    }
    $singles = Step "$name tuned singles (opt)" {
      & $exe --mode schema-search --flat-search --evaluator cpu --input $cloud --no-synthetic `
        --schemas $tunedSmall --generate-schemas 0 --no-baselines `
        --workloads configs/workloads/pipeline_replay.json --input-trace $opt `
        --queries $cell.queries --no-score-cache --no-pause --csv "$outDir/${name}_singles.csv"
    }
    $searched = $ga -or $singles
    if (-not $searched) {
      # Every search arm failed, so the only schemas left would be the defaults. That
      # re-measure looks like a legitimate "nothing beats the baseline" result.
      Write-Output "$name : all search arms failed - refusing the TEST re-measure"
      return
    }
  }

  Step "$name TEST re-measure" {
    if ($cell.arm -eq 'search') {
      $winners = @("$outDir/${name}_ga.csv", "$outDir/${name}_singles.csv" |
        ForEach-Object { Get-BestSchemaPath $_ } | Where-Object { $_ }) | Select-Object -Unique
    } else {
      $winners = Get-TransferWinners $cell
    }
    $list = (@($winners) + ($defaults -split ';') + ($handNested -split ';')) -join ';'
    & $exe --mode schema-search --flat-search --evaluator cpu --input $cloud --no-synthetic `
      --schemas $list --generate-schemas 0 --no-baselines `
      --workloads configs/workloads/pipeline_replay.json --input-trace $test `
      --queries $cell.queries --measure-repeats 5 --no-score-cache --no-pause `
      --csv "$outDir/${name}_test.csv"
  } | Out-Null
}

if (-not $SkipCapture) {
  Step 'capture missing traces' {
    foreach ($cell in $cells) {
      if (-not (Test-CellSelected $cell.cell)) { continue }
      if (Test-Path "$($cell.trace_dir)/pipeline_trace_opt.csv") { continue }
      if (-not (Test-Path $cell.laz_path)) { Write-Output "$($cell.cell): cloud missing, cannot capture"; continue }
      Write-Output "capturing $($cell.cell) -> $($cell.trace_dir)"
      & $py scripts/capture_pipeline_workload.py $cell.laz_path --out $cell.trace_dir --max-queries-per-stage 20000
      & $py scripts/split_trace.py "$($cell.trace_dir)/pipeline_trace.csv"
    }
  } | Out-Null
}

foreach ($cell in $cells) {
  if (Test-CellSelected $cell.cell) { Invoke-Cell $cell }
}

Write-Output "`n########## heatmap battery complete [$(Get-Date -Format HH:mm:ss)] ##########"
Write-Output 'Headline sources: results/eval_traces/matrix/*_test.csv'
if ($script:FailedSteps.Count) {
  Write-Output "`n$($script:FailedSteps.Count) step(s) failed:"
  $script:FailedSteps | ForEach-Object { Write-Output "  $_" }
  exit 1
}

# Shape x size heatmap battery (testing.md section 4): per matrix cell run a GA arm
# and the tuned-singles arm on the _opt trace half, then re-measure winners + all
# default singles + hand-designed nested on the _test half (serial, 5 repeats).
# Gate B is settled (random >= GA), so no repair/random arms here - the heatmap
# needs the winner FAMILY per cell, not another optimizer comparison.
# Output: results/eval_traces/matrix/<cell>_{ga,singles,test}.csv
param([string[]]$Only)
function Test-CellSelected($name) { return (-not $Only) -or ($Only -contains $name) }

$ErrorActionPreference = 'Continue'
$repo = 'c:\Github\MultiDataStructure'
$py = Join-Path $repo '.venv\Scripts\python.exe'
$exe = Join-Path $repo 'MultiDataStructure\x64\Release\MultiDataStructure.exe'
Set-Location $repo

$outDir = 'results/eval_traces/matrix'
New-Item -ItemType Directory -Force -Path $outDir, "$outDir/best_schemas" | Out-Null

$tunedSmall = (Get-Content configs/schemas/tuned_singles_small/index.txt) -join ';'
$defaults = @('quadtree', 'octree', 'kdtree', 'bvh', 'bih', 'hgrid', 'lbvh', 'regular_grid', 'karras_octree' |
  ForEach-Object { "configs/schemas/$_.json" }) -join ';'
$handNested = 'configs/schemas/octree_kdtree.json;configs/schemas/quadtree_octree.json'

function Step($name, $script) {
  Write-Output "`n########## $name  [$(Get-Date -Format HH:mm:ss)] ##########"
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  try { & $script } catch { Write-Output "STEP FAILED: $($_.Exception.Message)" }
  $sw.Stop()
  Write-Output "########## $name done in $([math]::Round($sw.Elapsed.TotalMinutes,1)) min (exit $LASTEXITCODE) ##########"
}
function Get-BestSchemaPath($csv) {
  foreach ($r in (Import-Csv $csv | Sort-Object { [double]$_.score })) {
    if ($r.schema_path -and (Test-Path $r.schema_path)) { return $r.schema_path }
  }
  return $null
}

# Capture traces for the 100M cells (only <=25M were captured this morning).
Step 'capture 100M matrix traces' {
  Get-ChildItem "D:\Datasets\Point Clouds\matrix\*_100M.las" -ErrorAction SilentlyContinue | ForEach-Object {
    $name = $_.BaseName.ToLower()
    $traceDir = "results/traces/matrix/$name"
    if (Test-Path "$traceDir/pipeline_trace_opt.csv") { return }
    Write-Output "capturing $($_.Name) -> $traceDir"
    & $py scripts/capture_pipeline_workload.py $_.FullName --out $traceDir --max-queries-per-stage 20000
    & $py scripts/split_trace.py "$traceDir/pipeline_trace.csv"
  }
}

function Invoke-MatrixCell($cell, $queries, $genSchemas, $generations, $population) {
  $cloud = "D:\Datasets\Point Clouds\matrix\$cell.las"
  $opt = "results/traces/matrix/$($cell.ToLower())/pipeline_trace_opt.csv"
  $test = "results/traces/matrix/$($cell.ToLower())/pipeline_trace_test.csv"
  if (-not (Test-Path $cloud)) { Write-Output "$cell : cloud missing - skipping"; return }
  if (-not (Test-Path $opt)) { Write-Output "$cell : opt trace missing - skipping"; return }

  Step "$cell GA (opt)" {
    & $exe --mode schema-search --input $cloud --no-synthetic `
      --workloads configs/workloads/pipeline_replay.json --input-trace $opt `
      --queries $queries --evaluator cpu --generate-schemas $genSchemas `
      --optimizer-generations $generations --optimizer-population $population --parallel 8 `
      --optimizer-output-dir "$outDir/best_schemas/${cell}_ga" --no-score-cache --no-pause `
      --csv "$outDir/${cell}_ga.csv"
  }
  Step "$cell tuned singles (opt)" {
    & $exe --mode schema-search --flat-search --evaluator cpu --input $cloud --no-synthetic `
      --schemas $tunedSmall --generate-schemas 0 --no-baselines `
      --workloads configs/workloads/pipeline_replay.json --input-trace $opt `
      --queries $queries --no-score-cache --no-pause --csv "$outDir/${cell}_singles.csv"
  }
  Step "$cell TEST re-measure" {
    $winners = @("$outDir/${cell}_ga.csv", "$outDir/${cell}_singles.csv" |
      ForEach-Object { Get-BestSchemaPath $_ } | Where-Object { $_ }) | Select-Object -Unique
    $list = (@($winners) + ($defaults -split ';') + ($handNested -split ';')) -join ';'
    & $exe --mode schema-search --flat-search --evaluator cpu --input $cloud --no-synthetic `
      --schemas $list --generate-schemas 0 --no-baselines `
      --workloads configs/workloads/pipeline_replay.json --input-trace $test `
      --queries $queries --measure-repeats 5 --no-score-cache --no-pause `
      --csv "$outDir/${cell}_test.csv"
  }
}

# Cheap cells first so partial results are useful early; 100M giants last.
$cells = @(
  @('indoor_1M',       3000, 24, 2, 16), @('terrain_1M',    3000, 24, 2, 16),
  @('architecture_1M', 3000, 24, 2, 16), @('urban_1M',      3000, 24, 2, 16),
  @('industrial_1M',   3000, 24, 2, 16),
  @('terrain_5M',      3000, 24, 2, 16), @('architecture_5M', 3000, 24, 2, 16),
  @('urban_5M',        3000, 24, 2, 16), @('industrial_5M', 3000, 24, 2, 16),
  @('terrain_25M',     2000, 24, 2, 12), @('architecture_25M', 2000, 24, 2, 12),
  @('urban_25M',       2000, 24, 2, 12), @('industrial_25M', 2000, 24, 2, 12),
  @('terrain_100M',    1000, 16, 2, 10), @('urban_100M',    1000, 16, 2, 10),
  @('industrial_100M', 1000, 16, 2, 10)
)
foreach ($c in $cells) {
  if (Test-CellSelected $c[0]) { Invoke-MatrixCell $c[0] $c[1] $c[2] $c[3] $c[4] }
}

Write-Output "`n########## heatmap battery complete [$(Get-Date -Format HH:mm:ss)] ##########"
Write-Output 'Headline sources: results/eval_traces/matrix/*_test.csv'

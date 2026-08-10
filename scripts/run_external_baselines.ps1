# Replay held-out trace halves through the external Indexicon baseline (plan section F.3).
#
# Both sides are measured in THIS run, on the identical cloud, the identical held-out trace, the
# same query count and the same repeat count. That matters: reading the MDS side out of a
# previous battery's CSVs compares two different machine sessions, which is not a baseline, it is
# an anecdote. The like-for-like pairs are MDS octree_default vs indexicon_octree and MDS
# kdtree_default vs indexicon_kdtree; the cell's best known MDS schema rides along as context.
#
# Indexicon has no radius primitive; tools/indexicon_point_baseline.cpp emulates it with a
# traversal pruned by each structure's own min-distance-to-node helper (--radius-mode
# native_mbr_prune, the default). -RadiusMode aabb_filter measures the naive
# bounding-box-plus-filter alternative instead; the two are NOT comparable and the mode is
# recorded in every JSON payload and every summary row.
#
# CAVEAT that the numbers cannot settle on their own: MDS records per-query telemetry (visited
# nodes, tested points, per-depth and per-structure breakdowns) that Indexicon does not. Any MDS
# deficit here is a joint measurement of index quality AND instrumentation cost. Treat a loss as
# a prompt to measure a telemetry-free MDS path, not as a settled result.
#
#   .\scripts\run_external_baselines.ps1                          # every matrix cell with a cache
#   .\scripts\run_external_baselines.ps1 -Only indoor_1M,terrain_25M
#   .\scripts\run_external_baselines.ps1 -RadiusMode aabb_filter -Suffix _aabb
#
# Output: results/framework_compare/indexicon/<cell>.json, <cell>_mds.csv, indexicon_summary.csv
param(
  [string[]]$Only,
  [ValidateSet('native_mbr_prune', 'aabb_filter')][string]$RadiusMode = 'native_mbr_prune',
  [int]$Repeats = 3,
  [int]$Queries = 3000,
  [int]$VerifyBruteforce = 16,
  [switch]$SkipMds,
  [string]$Suffix = ''
)

$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$driver = Join-Path $repo 'tools\bin\indexicon_point_baseline.exe'
if (-not (Test-Path $driver)) { throw "driver missing - run .\tools\build_indexicon_baseline.ps1 first" }
. "$PSScriptRoot\resolve_exe.ps1"
# MSBuild links to <repo>\x64\Release; MultiDataStructure\x64\Release is a stale legacy copy.
$exe = Resolve-MdsExe -Repo $repo
if (-not $SkipMds -and -not (Test-Path $exe)) { throw "MultiDataStructure.exe missing - build it or pass -SkipMds" }

$matrixDir = 'D:\Datasets\Point Clouds\matrix'
$outDir = 'results/framework_compare/indexicon'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$inv = [cultureinfo]::InvariantCulture
# Export-Csv joins fields with commas while this machine formats decimals with commas, which
# silently produces an unparseable file. Every numeric column is stringified invariantly.
function Fmt($value, $decimals = 6) {
  if ($null -eq $value -or $value -eq '') { return '' }
  return [math]::Round([double]$value, $decimals).ToString($inv)
}

# Overall latency is the query-count-weighted mean over types, so it is directly comparable to
# the MDS avg_latency_ms over the same trace rather than to an unweighted per-type mean.
function Get-WeightedAvg($result) {
  $total = 0.0; $n = 0
  foreach ($prop in $result.by_type.PSObject.Properties) {
    $total += $prop.Value.avg_latency_ms * $prop.Value.queries
    $n += $prop.Value.queries
  }
  if ($n -eq 0) { return $null }
  return $total / $n
}

# The cell's best schema from the heatmap battery, replayed here so the context row is measured
# in this session too rather than quoted from July.
function Get-WinnerSchema($cell) {
  $csv = "results/eval_traces/matrix/${cell}_test.csv"
  if (-not (Test-Path $csv)) { return $null }
  foreach ($row in (Import-Csv $csv | Sort-Object { [double]$_.avg_latency_ms })) {
    if ($row.schema_path -and (Test-Path $row.schema_path)) { return $row.schema_path }
  }
  return $null
}

$cells = Get-ChildItem "$matrixDir\*.las" -ErrorAction SilentlyContinue |
  ForEach-Object { $_.BaseName } | Sort-Object
if ($Only) { $cells = $cells | Where-Object { $Only -contains $_ } }
if (-not $cells) { throw "no matrix cells found under $matrixDir" }

$summary = @()
foreach ($cell in $cells) {
  $cloud = "$matrixDir\$cell.las"
  $trace = "results/traces/matrix/$cell/pipeline_trace_test.csv"
  if (-not (Test-Path $trace)) { Write-Output "$cell : held-out trace missing - skipping"; continue }
  if (-not (Test-Path "$cloud.mdspc")) {
    Write-Output "$cell : no .mdspc cache beside the cloud - run MultiDataStructure on it once; skipping"
    continue
  }

  Write-Output "`n########## $cell  [$(Get-Date -Format HH:mm:ss)] ##########"

  # Verification builds all three structures a second time, so it is skipped on the giant cells
  # where that would not fit in memory. Correctness is a property of the traversal, not of the
  # cloud, so verifying it on the small cells covers the large ones too.
  $cellPoints = [int64](((Get-Item "$cloud.mdspc").Length - 88) / 12)
  $verify = if ($cellPoints -gt 30000000) { 0 } else { $VerifyBruteforce }
  if ($verify -eq 0 -and $VerifyBruteforce -gt 0) {
    Write-Output "$cell : $cellPoints points - skipping brute-force verification (memory)"
  }

  $json = "$outDir/$cell$Suffix.json"
  & $driver --input $cloud --query-trace $trace --output $json `
    --max-queries $Queries --repeats $Repeats --radius-mode $RadiusMode `
    --verify-bruteforce $verify
  if ($LASTEXITCODE -ne 0) { Write-Output "$cell : Indexicon driver failed (exit $LASTEXITCODE)"; continue }
  $payload = Get-Content $json -Raw | ConvertFrom-Json

  $mdsCsv = "$outDir/${cell}_mds$Suffix.csv"
  # The C++ --csv writer appends (PointBenchmark.cpp appendCsvSummary). Left in place, a
  # re-run's rows land AFTER the previous run's, and the Select-Object -First 1 lookups below
  # would keep returning the oldest numbers for this cell forever.
  if (Test-Path $mdsCsv) { Remove-Item -Force $mdsCsv }
  if (-not $SkipMds) {
    $schemas = @('configs/schemas/octree.json', 'configs/schemas/kdtree.json')
    $winner = Get-WinnerSchema $cell
    if ($winner) { $schemas += $winner }
    & $exe --mode schema-search --flat-search --evaluator cpu --input $cloud --no-synthetic `
      --schemas ($schemas -join ';') --generate-schemas 0 --no-baselines `
      --workloads configs/workloads/pipeline_replay.json --input-trace $trace `
      --queries $Queries --measure-repeats $Repeats --no-score-cache --no-pause `
      --csv $mdsCsv | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Output "$cell : MDS re-measure failed (exit $LASTEXITCODE)" }
  }

  $mdsRows = @()
  if (Test-Path $mdsCsv) { $mdsRows = Import-Csv $mdsCsv | Where-Object { $_.avg_latency_ms } }
  function MdsMs($name) {
    $row = $mdsRows | Where-Object { $_.schema_name -eq $name } | Select-Object -First 1
    if ($row) { return [double]$row.avg_latency_ms }
    return $null
  }

  $row = [ordered]@{
    cell        = $cell
    points      = $payload.points
    radius_mode = $RadiusMode
    repeats     = $Repeats
    queries     = $Queries
  }
  foreach ($result in $payload.results) {
    $short = $result.method -replace '^indexicon_', ''
    $row["ix_${short}_ms"] = Fmt (Get-WeightedAvg $result)
    $row["ix_${short}_build_ms"] = Fmt $result.build_ms 1
    $row["ix_${short}_mismatch"] = $result.count_mismatches
  }
  $mdsOctree = MdsMs 'octree_default'
  $mdsKdTree = MdsMs 'kdtree_default'
  $mdsBestRow = $mdsRows | Sort-Object { [double]$_.avg_latency_ms } | Select-Object -First 1
  $row['mds_octree_ms'] = Fmt $mdsOctree
  $row['mds_kdtree_ms'] = Fmt $mdsKdTree
  $row['mds_best'] = if ($mdsBestRow) { $mdsBestRow.schema_name } else { '' }
  $row['mds_best_ms'] = Fmt ($mdsBestRow.avg_latency_ms)
  # Node/point counts make the per-visited-node cost recoverable: regressing MDS latency on
  # (avg_visited_nodes, avg_tested_points) across the schemas in a cell separates fixed per-node
  # cost - which is where the per-node telemetry lands - from the actual distance tests.
  function MdsCol($name, $col) {
    $r = $mdsRows | Where-Object { $_.schema_name -eq $name } | Select-Object -First 1
    if ($r) { return $r.$col }
    return ''
  }
  $row['mds_octree_visited'] = Fmt (MdsCol 'octree_default' 'avg_visited_nodes') 2
  $row['mds_octree_tested'] = Fmt (MdsCol 'octree_default' 'avg_tested_points') 2
  $row['mds_kdtree_visited'] = Fmt (MdsCol 'kdtree_default' 'avg_visited_nodes') 2
  $row['mds_kdtree_tested'] = Fmt (MdsCol 'kdtree_default' 'avg_tested_points') 2
  # Cold/warm spread: avg_latency_ms is the first pass (SchemaSearch.cpp:3380), while the repeat
  # CI covers the warm passes. Reported so the cold-pass bias stays visible rather than assumed.
  $row['mds_octree_ci_low'] = Fmt (MdsCol 'octree_default' 'latency_repeat_ci_low_ms')
  $row['mds_octree_ci_high'] = Fmt (MdsCol 'octree_default' 'latency_repeat_ci_high_ms')

  # Like-for-like primitive ratios: > 1 means our implementation of that primitive is faster.
  $ixOctree = ($payload.results | Where-Object { $_.method -eq 'indexicon_octree' } | ForEach-Object { Get-WeightedAvg $_ })
  $ixKdTree = ($payload.results | Where-Object { $_.method -eq 'indexicon_kdtree' } | ForEach-Object { Get-WeightedAvg $_ })
  if ($mdsOctree -and $ixOctree) { $row['octree_ix_over_mds'] = Fmt ($ixOctree / $mdsOctree) 3 }
  if ($mdsKdTree -and $ixKdTree) { $row['kdtree_ix_over_mds'] = Fmt ($ixKdTree / $mdsKdTree) 3 }

  $summary += [pscustomobject]$row
}

$summaryPath = "$outDir/indexicon_summary$Suffix.csv"
$summary | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding utf8
Write-Output "`n########## external baselines complete [$(Get-Date -Format HH:mm:ss)] ##########"
Write-Output "wrote $summaryPath"
$summary | Format-Table -AutoSize
Write-Output 'octree_ix_over_mds / kdtree_ix_over_mds > 1 means OUR primitive is faster than Indexicon''s.'
Write-Output 'MDS rows carry per-query telemetry that Indexicon does not - see the header caveat.'

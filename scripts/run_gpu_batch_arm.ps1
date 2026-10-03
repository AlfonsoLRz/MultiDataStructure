# Measure the CUDA arm as BATCH THROUGHPUT on the held-out pipeline trace.
#
# Why this exists, and why it is not just another column in the head-to-head.
#
# Every number in the curated battery was produced with --evaluator cpu, so nothing published so
# far measures the GPU at query time. That was a defensible default: under the mixed builder the
# CUDA kNN path is a brute-force full-cloud scan that does not even return neighbour indices, and
# it lost by ~3500x at 5M (results/eval_traces/sanandreas_5m_cuda.csv). It is not a defensible
# permanent state, because a genuine GPU tree traversal does exist - KDTree.cu treeKnnKernel -
# reachable only through --cuda-builder kdtree and gated at k <= 16, and the captured pipeline
# workload is exactly k = 16.
#
# The metric here is deliberately NOT per-query latency. Our CPU per-query latency is 4-7 us at
# every rung from 1M to 1B, and a single kernel launch costs about 5 us, so a per-query GPU
# dispatch cannot win at any scale - that is arithmetic, not a tuning problem. What the workload
# actually is, though, is a batch: scripts/capture_pipeline_workload.py derives dense self-queries,
# one per point, and the sidecar records full_query_count = N per stage (25 million at the 25M
# rung). We replay a few thousand of them serially. Batch throughput is the honest way to measure a
# batch, and it is the only framing in which the GPU arm can legitimately win.
#
# Reported per cell:
#   gpu_batch_ms_per_query = gpu_query_ms / queries
#     gpu_query_ms is a CUDA-event measurement wrapping the whole batch loop INCLUDING the host
#     to device and device to host copies (MixedTree.cu startTimer/stopTimer). It is NOT
#     avg_latency_ms, which on the CUDA arm is in-kernel clock64() and is comparable to neither
#     the CPU column nor the external baseline's wall clock. Reading the wrong one here would
#     manufacture a GPU win out of excluded transfer cost.
#   stage_projection_s = gpu_batch_ms_per_query * full_query_count / 1000
#     what one full pipeline stage over the whole cloud would cost at this throughput, next to the
#     same projection for the CPU arm. This is the number a practitioner actually experiences.
#
#   .\scripts\run_gpu_batch_arm.ps1
#   .\scripts\run_gpu_batch_arm.ps1 -Only sanandreas_25M,alhambra_25M
#
# Output: results/framework_compare/gpu_batch/<cell>_gpu.csv, gpu_batch_summary.csv
param(
  [string[]]$Only,
  [int]$Queries = 3000,
  [int]$Repeats = 3,
  [string]$CellConfig = 'configs/datasets/curated_cells.json',
  [string]$Suffix = ''
)

$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

. "$PSScriptRoot\resolve_exe.ps1"
$exe = Resolve-MdsExe -Repo $repo
if (-not (Test-Path $CellConfig)) { throw "cell config not found: $CellConfig" }

$outDir = 'results/framework_compare/gpu_batch'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$inv = [cultureinfo]::InvariantCulture
function Fmt($value, $decimals = 6) {
  if ($null -eq $value -or $value -eq '') { return '' }
  return [math]::Round([double]$value, $decimals).ToString($inv)
}

$config = Get-Content $CellConfig -Raw | ConvertFrom-Json
$cells = $config.cells | Where-Object { $_.size_label -eq '25M' -or $_.cell -eq 'museum_1M' }
if ($Only) { $cells = $config.cells | Where-Object { $Only -contains $_.cell } }
if (-not $cells) { throw "no cells selected from $CellConfig" }

$summary = @()
foreach ($entry in $cells) {
  $cell = $entry.cell
  $cloud = $entry.mdspc_path
  $trace = "$($entry.trace_dir)/pipeline_trace_test.csv"
  $sidecar = "$($entry.trace_dir)/pipeline_workload.json"
  if (-not (Test-Path $cloud)) { Write-Host "$cell : .mdspc missing - skipping"; continue }
  if (-not (Test-Path $trace)) { Write-Host "$cell : held-out trace missing - skipping"; continue }

  Write-Host "`n########## $cell  [$(Get-Date -Format HH:mm:ss)] ##########"

  # Both arms measured in THIS run on the identical cloud and trace, so the CPU column is a
  # control for the GPU column rather than a quote from an earlier session.
  $rows = @{}
  foreach ($arm in @('cpu', 'cuda')) {
    $csv = "$outDir/${cell}_$arm$Suffix.csv"
    if (Test-Path $csv) { Remove-Item -Force $csv }
    $schema = if ($arm -eq 'cuda') { 'configs/schemas/kdtree_gpu.json' } else { 'configs/schemas/kdtree.json' }
    $extra = @()
    if ($arm -eq 'cuda') { $extra = @('--cuda-builder', 'kdtree', '--cuda-knn-backend', 'gpu_tree_knn', '--cuda-query-batch', '0') }

    & $exe --mode schema-search --flat-search --evaluator $arm --input $cloud --no-synthetic `
      --schemas $schema --generate-schemas 0 --no-baselines `
      --workloads configs/workloads/pipeline_replay.json --input-trace $trace `
      --queries $Queries --measure-repeats $Repeats --no-score-cache --no-pause `
      --csv $csv @extra | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "$cell : $arm arm failed (exit $LASTEXITCODE)" }
    if (Test-Path $csv) { $rows[$arm] = Import-Csv $csv | Select-Object -First 1 }
  }

  $cpu = $rows['cpu']
  $gpu = $rows['cuda']
  if (-not $gpu) { Write-Host "$cell : no CUDA row produced - skipping"; continue }

  # Guards. A cuda_failed row scores infinity and a cpu_fallback row is a CPU measurement wearing
  # a GPU label; either one silently turns this table into fiction, so both are recorded, not
  # assumed away. bruteforce_gpu_scan in particular returns no neighbours at all.
  $backend = $gpu.backend
  $support = $gpu.gpu_support_status
  $knnBackend = $gpu.knn_backend
  $trustworthy = ($backend -eq 'cuda') -and ($support -eq 'full') -and ($knnBackend -eq 'gpu_tree_knn')
  if (-not $trustworthy) {
    Write-Host "$cell : UNTRUSTWORTHY GPU ROW - backend=$backend support=$support knn_backend=$knnBackend"
  }

  $effQueries = if ($gpu.total_queries) { [double]$gpu.total_queries } else { [double]$Queries }
  $gpuBatchMs = if ($gpu.gpu_query_ms -and $effQueries -gt 0) { [double]$gpu.gpu_query_ms / $effQueries } else { $null }
  $cpuMs = if ($cpu -and $cpu.latency_repeat_mean_ms) { [double]$cpu.latency_repeat_mean_ms }
           elseif ($cpu) { [double]$cpu.avg_latency_ms } else { $null }

  # Full-stage projection uses the FULL query count the capture recorded, not the replayed subset.
  $fullQueries = $null
  if (Test-Path $sidecar) {
    $side = Get-Content $sidecar -Raw | ConvertFrom-Json
    $fullQueries = ($side.stages | Measure-Object -Property full_query_count -Sum).Sum
  }

  $row = [ordered]@{
    cell                   = $cell
    points                 = $entry.points
    queries                = $effQueries
    repeats                = $Repeats
    gpu_backend            = $backend
    gpu_support_status     = $support
    knn_backend            = $knnBackend
    gpu_row_trustworthy    = [int]$trustworthy
    gpu_upload_ms          = Fmt $gpu.gpu_upload_ms 2
    gpu_build_ms           = Fmt $gpu.gpu_build_ms 2
    gpu_query_ms_batch     = Fmt $gpu.gpu_query_ms 3
    gpu_batch_ms_per_query = Fmt $gpuBatchMs
    gpu_memory_bytes       = $gpu.gpu_memory_bytes
    cpu_ms_per_query       = Fmt $cpuMs
    speedup_gpu_over_cpu   = if ($gpuBatchMs -and $cpuMs) { Fmt ($cpuMs / $gpuBatchMs) 3 } else { '' }
    full_query_count       = $fullQueries
    gpu_stage_projection_s = if ($gpuBatchMs -and $fullQueries) { Fmt ($gpuBatchMs * $fullQueries / 1000.0) 2 } else { '' }
    cpu_stage_projection_s = if ($cpuMs -and $fullQueries) { Fmt ($cpuMs * $fullQueries / 1000.0) 2 } else { '' }
  }
  $summary += [pscustomobject]$row
  Write-Host ("  gpu batch {0} ms/query vs cpu {1} ms/query  ->  {2}x" -f `
    (Fmt $gpuBatchMs), (Fmt $cpuMs), $(if ($gpuBatchMs -and $cpuMs) { Fmt ($cpuMs / $gpuBatchMs) 2 } else { 'n/a' }))
}

$summaryPath = "$outDir/gpu_batch_summary$Suffix.csv"
$summary | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding utf8
Write-Host "`n########## gpu batch arm complete [$(Get-Date -Format HH:mm:ss)] ##########"
Write-Host "wrote $summaryPath"
$summary | Format-Table -AutoSize
Write-Host 'gpu_batch_ms_per_query is batch wall clock INCLUDING transfers, divided by query count.'
Write-Host 'It is NOT comparable to a per-query latency column - see the header.'
$bad = $summary | Where-Object { $_.gpu_row_trustworthy -ne 1 }
if ($bad) { Write-Host "WARNING: $($bad.Count) cell(s) produced a fallback or failed GPU row - do not quote them." }

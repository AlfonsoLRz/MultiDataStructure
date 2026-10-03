# MDS vs Indexicon, same session, same cells, same queries, same protocol, pinned to one P-core.
#
#   scripts/head_to_head.ps1 -Tag engine6
#   scripts/head_to_head.ps1 -Tag engine6 -CellNames toronto_25M -Rounds 1
#
# Both sides replay the held-out pipeline trace of each cell, subset to -Queries with the same
# integer stride, 1 untimed warm-up pass then 3 timed passes pooled into one mean per query kind,
# each query timed individually around the traversal alone (MDS: --mode engine-check with
# telemetry off in timed passes; Indexicon: tools/indexicon_point_baseline.cpp, rebuilt with the
# same /O2 /arch:AVX2 /GL flags). Rounds alternate which library runs first.
#
# Pairs that compare like with like: MDS octree_default vs Indexicon octree, MDS kdtree_default vs
# Indexicon kdtree, and the same-bucket yardstick MDS tuned_kd19l128 (128-point leaves) vs
# Indexicon kdtree (bucket 128). The cell's searched winner rides along: it is what the method
# would actually deploy.
#
# Disclosed asymmetries: Indexicon's kNN allocates its heap and result vector inside its timed
# region (MDS reuses thread-local scratch); MDS answers in float local coordinates, Indexicon in
# double world coordinates. Indexicon has no radius primitive; the driver's native_mbr_prune
# traversal (verified against brute force) stands in for one.

param(
    [string]$Tag = "h2h_$(Get-Date -Format yyyyMMdd_HHmmss)",
    [string]$Mds,
    [string[]]$CellNames = @('toronto_25M', 'sanandreas_25M', 'alhambra_25M'),
    [int]$Rounds = 2,
    [int]$Queries = 3000,
    [int]$Cpu = 4,
    [string]$OutRoot = 'D:\MDS_ab\h2h',
    [string]$Cells = (Join-Path (Split-Path -Parent $PSScriptRoot) 'configs\datasets\curated_cells.json')
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. "$PSScriptRoot\resolve_exe.ps1"
if (-not $Mds) { $Mds = Resolve-MdsExe }
$driver = Join-Path $repo 'tools\bin\indexicon_point_baseline.exe'
if (-not (Test-Path $driver)) { throw "driver missing - run tools\build_indexicon_baseline.ps1" }
$out = Join-Path $OutRoot $Tag
New-Item -ItemType Directory -Force $out | Out-Null

$cellIndex = @{}
foreach ($cell in (Get-Content $Cells -Raw | ConvertFrom-Json).cells) { $cellIndex[$cell.cell] = $cell }
$winners = @{
    'toronto_25M'    = 'configs\schemas\equivalence_set\winner_toronto_25M.json'
    'sanandreas_25M' = 'configs\schemas\equivalence_set\winner_aneto_5M.json'
    'alhambra_25M'   = 'configs\schemas\equivalence_set\winner_alhambra_25M.json'
}

function Invoke-Pinned([string]$exe, [string[]]$arguments, [string]$log) {
    $p = Start-Process -FilePath $exe -ArgumentList $arguments -NoNewWindow -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    # Touching Handle now is what makes ExitCode readable later: without it, a Start-Process
    # object with redirected streams reports a null exit code once the process has ended.
    $null = $p.Handle
    try {
        $p.ProcessorAffinity = [IntPtr](1L -shl $Cpu)
        $p.PriorityClass = 'High'
    } catch { Write-Warning "could not pin/prioritise: $_" }
    $p.WaitForExit()
    return $p.ExitCode
}

for ($round = 1; $round -le $Rounds; $round++) {
    foreach ($name in $CellNames) {
        $cell = $cellIndex[$name]
        $trace = "$($cell.trace_dir)/pipeline_trace_test.csv"
        $schemas = @('configs\schemas\octree.json', 'configs\schemas\kdtree.json',
                     'configs\schemas\equivalence_set\tuned_kd19l128.json')
        if ($winners.ContainsKey($name)) { $schemas += $winners[$name] }

        $runs = @(
            @{ side = 'ix'; exe = $driver; args = @('--input', "`"$($cell.mdspc_path)`"", '--query-trace', "`"$trace`"",
                '--structures', 'octree,kdtree', '--max-queries', $Queries, '--warmup', '1', '--repeats', '3',
                '--output', "`"$(Join-Path $out "ix_${name}_r${round}.json")`"") },
            @{ side = 'mds'; exe = (Resolve-Path $Mds).Path; args = @('--mode', 'engine-check', '--input', "`"$($cell.mdspc_path)`"",
                '--schemas', "`"$($schemas -join ';')`"", '--input-trace', "`"$trace`"", '--queries', $Queries,
                '--warmup', '1', '--measure-repeats', '3', '--summary-csv', "`"$(Join-Path $out "mds_r${round}.csv")`"") }
        )
        if ($round % 2 -eq 0) { [array]::Reverse($runs) }
        foreach ($run in $runs) {
            $sw = [Diagnostics.Stopwatch]::StartNew()
            $code = Invoke-Pinned $run.exe $run.args (Join-Path $out "$($run.side)_${name}_r${round}.log")
            Write-Host ("round {0} {1,-4} {2}: exit {3} in {4:N0} s" -f $round, $run.side, $name, $code, $sw.Elapsed.TotalSeconds)
            if ($code -ne 0) { throw "$($run.side) failed on $name (see logs in $out)" }
        }
    }
}

@{ mds = (Resolve-Path $Mds).Path; mds_time = (Get-Item $Mds).LastWriteTime.ToString('s');
   driver_time = (Get-Item $driver).LastWriteTime.ToString('s'); cpu = $Cpu; rounds = $Rounds;
   queries = $Queries; git = (git rev-parse --short HEAD) } | ConvertTo-Json | Set-Content (Join-Path $out 'h2h.json')

& (Join-Path $repo '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'summarize_head_to_head.py') $out

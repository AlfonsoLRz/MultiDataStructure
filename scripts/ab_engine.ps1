# A/B two engine builds on the same cells, schemas and queries, pinned to one P-core.
#
#   scripts/ab_engine.ps1 -A D:\MDS_ab\bin\step0.exe -B x64\Release\MultiDataStructure.exe -Tag step1
#   scripts/ab_engine.ps1 -A D:\MDS_ab\bin\step0.exe -B D:\MDS_ab\bin\step0.exe -Tag aa   # noise floor
#
# Protocol (why each piece is there):
#   - --mode engine-check, 1 warm-up + 3 timed passes of the held-out trace, the same pooled-mean
#     protocol as tools/indexicon_point_baseline.cpp, with nothing but the traversal on the clock.
#   - Pinned to one logical CPU of a P-core at High priority. The i7-14700KF is hybrid; an
#     unpinned run migrates between P and E cores and the noise swamps 5-10% effects.
#   - Binaries alternate A,B then B,A across rounds, so slow drift (thermals, background work)
#     cancels instead of favouring whichever ran first.
#   - A gain is only claimed when every round agrees on its sign and it exceeds twice the A/A
#     spread; scripts/summarize_ab.py does that arithmetic.

param(
    [Parameter(Mandatory)][string]$A,
    [Parameter(Mandatory)][string]$B,
    [string]$Tag = "ab_$(Get-Date -Format yyyyMMdd_HHmmss)",
    [string[]]$CellNames = @('toronto_25M', 'sanandreas_25M', 'alhambra_25M'),
    [int]$Rounds = 2,
    [int]$Queries = 3000,
    [int]$Cpu = 4,
    [string]$OutRoot = 'D:\MDS_ab\runs',
    [string]$Cells = (Join-Path (Split-Path -Parent $PSScriptRoot) 'configs\datasets\curated_cells.json')
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$out = Join-Path $OutRoot $Tag
New-Item -ItemType Directory -Force $out | Out-Null

$cellIndex = @{}
foreach ($cell in (Get-Content $Cells -Raw | ConvertFrom-Json).cells) { $cellIndex[$cell.cell] = $cell }

# The schemas that decide the comparison: the four defaults, the same-bucket kd yardstick, and
# each cell's searched winner from the frozen equivalence set.
$common = @('octree', 'kdtree', 'quadtree') | ForEach-Object { "configs\schemas\$_.json" }
$common += 'configs\schemas\equivalence_set\tuned_kd19l128.json'
$winners = @{
    'toronto_25M'    = 'configs\schemas\equivalence_set\winner_toronto_25M.json'
    # sanandreas_25M's searched winner is byte-identical to aneto_5M's, frozen under that name.
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

$binaries = @{ A = (Resolve-Path $A).Path; B = (Resolve-Path $B).Path }
for ($round = 1; $round -le $Rounds; $round++) {
    $order = if ($round % 2 -eq 1) { 'A', 'B' } else { 'B', 'A' }
    foreach ($arm in $order) {
        foreach ($name in $CellNames) {
            $cell = $cellIndex[$name]
            $schemas = @($common)
            if ($winners.ContainsKey($name)) { $schemas += $winners[$name] }
            $arguments = @('--mode', 'engine-check', '--input', "`"$($cell.mdspc_path)`"",
                           '--schemas', "`"$($schemas -join ';')`"",
                           '--input-trace', "`"$($cell.trace_dir)/pipeline_trace_test.csv`"",
                           '--queries', $Queries, '--warmup', '1', '--measure-repeats', '3',
                           '--summary-csv', "`"$(Join-Path $out "${arm}_r${round}.csv")`"")
            $log = Join-Path $out "${arm}_r${round}_${name}.log"
            $sw = [Diagnostics.Stopwatch]::StartNew()
            $code = Invoke-Pinned $binaries[$arm] $arguments $log
            Write-Host ("round {0} {1} {2}: exit {3} in {4:N0} s" -f $round, $arm, $name, $code, $sw.Elapsed.TotalSeconds)
            if ($code -ne 0) { throw "engine-check failed: $log" }
        }
    }
}

@{ A = $binaries.A; B = $binaries.B; a_time = (Get-Item $binaries.A).LastWriteTime.ToString('s');
   b_time = (Get-Item $binaries.B).LastWriteTime.ToString('s'); cpu = $Cpu; rounds = $Rounds;
   queries = $Queries; git = (git rev-parse --short HEAD) } | ConvertTo-Json | Set-Content (Join-Path $out 'ab.json')

$py = Join-Path $repo '.venv\Scripts\python.exe'
& $py (Join-Path $PSScriptRoot 'summarize_ab.py') $out

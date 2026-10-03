# Capture an engine-equivalence snapshot: what one binary's engine returns, query by query.
#
# Run it once with the pre-change binary (the golden) and once with the candidate, then compare:
#   scripts/engine_equivalence.ps1 -Exe bin_ab\step0.exe -Out D:\MDS_equivalence\golden_v1
#   scripts/engine_equivalence.ps1 -Out D:\MDS_equivalence\step3a
#   python scripts/diff_engine_equivalence.py D:\MDS_equivalence\golden_v1 D:\MDS_equivalence\step3a
#
# Two tiers, both driven by --mode engine-check:
#   S  small clouds x every shipped schema + configs/schemas/equivalence_set, on GENERATED queries
#      (every kind, including range and count, which application traces never issue), plus one
#      leaf-micro-index pass. This is where grammar corners get exercised.
#   L  the three 25M cells the A/B runs on, replaying their held-out pipeline traces, on the
#      schemas that decide the Indexicon comparison.
# Snapshots are tens of MB per tier; keep them out of the repo (D:\MDS_equivalence by default).

param(
    [string]$Exe,
    [string]$Out = "D:\MDS_equivalence\snapshot_$(Get-Date -Format yyyyMMdd_HHmmss)",
    [ValidateSet('S', 'L', 'All')][string]$Tier = 'All',
    [string]$Cells = (Join-Path (Split-Path -Parent $PSScriptRoot) 'configs\datasets\curated_cells.json')
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. "$PSScriptRoot\resolve_exe.ps1"
if (-not $Exe) { $Exe = Resolve-MdsExe }
New-Item -ItemType Directory -Force $Out | Out-Null

$cellIndex = @{}
foreach ($cell in (Get-Content $Cells -Raw | ConvertFrom-Json).cells) { $cellIndex[$cell.cell] = $cell }

$shipped = Get-ChildItem configs\schemas\*.json | ForEach-Object { $_.FullName }
$frozen = Get-ChildItem configs\schemas\equivalence_set\*.json | ForEach-Object { $_.FullName }
$tierS = @($shipped) + @($frozen)
$tierL = @('octree', 'kdtree', 'quadtree', 'bvh') | ForEach-Object { "configs\schemas\$_.json" }
$tierL += @('tuned_kd19l128', 'tuned_kd16l1024', 'tuned_qt10l32', 'tuned_ot7l32',
            'winner_alhambra_25M', 'winner_galiciaforest_25M', 'winner_parislux_25M',
            'gen_generated_ot6l1024_kd3l16384ap_rrcp32_ot1l1024cp128h750') |
    ForEach-Object { "configs\schemas\equivalence_set\$_.json" }

$failures = @()
function Invoke-Check([string]$label, [string]$cloud, [string[]]$schemas, [string[]]$extra) {
    # Lists go through a file-free ';' join; the exe splits on ';' and ','.
    $joined = $schemas -join ';'
    $cliArgs = @('--mode', 'engine-check', '--input', $cloud, '--schemas', $joined,
              '--query-csv', (Join-Path $Out "${label}_queries.csv"),
              '--summary-csv', (Join-Path $Out "${label}_summary.csv"),
              '--warmup', '1', '--measure-repeats', '1') + $extra
    Write-Host "== $label ($($schemas.Count) schemas)"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $Exe @cliArgs | Out-Host
    if ($LASTEXITCODE -ne 0) { $script:failures += "$label (exit $LASTEXITCODE)" }
    Write-Host "   done in $([math]::Round($sw.Elapsed.TotalSeconds, 1)) s"
}

if ($Tier -in 'S', 'All') {
    foreach ($name in 'sanandreas_1M', 'toronto_1M', 'museum_1M', 'cathedral_1M', 'galiciaforest_1M') {
        Invoke-Check "S_$name" $cellIndex[$name].mdspc_path $tierS @('--queries', '1500', '--query-seed', '1337')
    }
    Invoke-Check 'S_sanandreas_1M_micro' $cellIndex['sanandreas_1M'].mdspc_path $frozen `
        @('--queries', '1500', '--query-seed', '7', '--leaf-micro-indexes', '--leaf-micro-threshold', '64')
}

if ($Tier -in 'L', 'All') {
    foreach ($name in 'toronto_25M', 'sanandreas_25M', 'alhambra_25M') {
        $cell = $cellIndex[$name]
        Invoke-Check "L_$name" $cell.mdspc_path $tierL @('--queries', '3000', '--input-trace', "$($cell.trace_dir)/pipeline_trace_test.csv")
    }
}

@{ exe = (Resolve-Path $Exe).Path; exe_time = (Get-Item $Exe).LastWriteTime.ToString('s');
   git = (git rev-parse --short HEAD); dirty = [bool](git status --porcelain --untracked-files=no);
   tier = $Tier } | ConvertTo-Json | Set-Content (Join-Path $Out 'snapshot.json')

if ($failures) { Write-Host "FAILED: $($failures -join ', ')"; exit 1 }
Write-Host "snapshot written to $Out"

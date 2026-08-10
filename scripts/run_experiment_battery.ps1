# Corrected experiment battery (post coordinate-frame fix). Reruns everything the
# frame bug poisoned, in dependency order, against the fixed binary.
$ErrorActionPreference = 'Continue'
$repo = 'c:\Github\MultiDataStructure'
$py = Join-Path $repo '.venv\Scripts\python.exe'
$pyMds = Join-Path $repo '.venv-mds\Scripts\python.exe'
. "$PSScriptRoot\resolve_exe.ps1"
# MSBuild links to <repo>\x64\Release; MultiDataStructure\x64\Release is a stale legacy copy.
$exe = Resolve-MdsExe -Repo $repo
Set-Location $repo

function Step($name, $script) {
  Write-Output "`n########## $name  [$(Get-Date -Format HH:mm:ss)] ##########"
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & $script
  $sw.Stop()
  Write-Output "########## $name done in $([math]::Round($sw.Elapsed.TotalMinutes,1)) min (exit $LASTEXITCODE) ##########"
}

# 0. Quarantine the frame-bug results for provenance.
Step 'quarantine invalid results' {
  New-Item -ItemType Directory -Force -Path results/eval_traces/invalid_frame_bug | Out-Null
  Get-ChildItem results/eval_traces/*.csv | Move-Item -Destination results/eval_traces/invalid_frame_bug/ -Force
}

# 1. Pipeline GAs (serial Alhambra stays the search-cost anchor).
Step 'GA 5M (serial)' {
  & $exe --mode schema-search --input "D:\Datasets\Point Clouds\SanAndreas\5M.las" --no-synthetic `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/sanandreas_5m/pipeline_trace.csv `
    --queries 3000 --evaluator cpu --generate-schemas 32 --optimizer-generations 3 --optimizer-population 24 `
    --optimizer-output-dir results/eval_traces/best_schemas --no-score-cache --no-pause `
    --csv results/eval_traces/sanandreas_5m.csv --best-csv results/eval_traces/sanandreas_5m_best.csv
}
Step 'GA 5M (--parallel 8)' {
  & $exe --mode schema-search --input "D:\Datasets\Point Clouds\SanAndreas\5M.las" --no-synthetic `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/sanandreas_5m/pipeline_trace.csv `
    --queries 3000 --evaluator cpu --generate-schemas 32 --optimizer-generations 3 --optimizer-population 24 `
    --parallel 8 --no-score-cache --no-pause --csv results/eval_traces/sanandreas_5m_par8.csv
}
Step 'GA 5M (cuda evaluator)' {
  & $exe --mode schema-search --input "D:\Datasets\Point Clouds\SanAndreas\5M.las" --no-synthetic `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/sanandreas_5m/pipeline_trace.csv `
    --queries 3000 --evaluator cuda --generate-schemas 32 --optimizer-generations 3 --optimizer-population 24 `
    --no-score-cache --no-pause --csv results/eval_traces/sanandreas_5m_cuda.csv
}
Step 'GA 50M (converted LAS)' {
  & $exe --mode schema-search --input "results/traces/sanandreas_50m/50M_converted.las" --no-synthetic `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/sanandreas_50m/pipeline_trace.csv `
    --queries 1500 --evaluator cpu --generate-schemas 24 --optimizer-generations 2 --optimizer-population 12 `
    --optimizer-output-dir results/eval_traces/best_schemas --no-score-cache --no-pause `
    --csv results/eval_traces/sanandreas_50m.csv --best-csv results/eval_traces/sanandreas_50m_best.csv
}
Step 'GA Alhambra 100M (serial anchor)' {
  & $exe --mode schema-search --input "D:\Datasets\Point Clouds\Alhambra\Alhambra_100M.las" --no-synthetic `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/alhambra_100m/pipeline_trace.csv `
    --queries 1000 --evaluator cpu --generate-schemas 16 --optimizer-generations 2 --optimizer-population 10 `
    --optimizer-output-dir results/eval_traces/best_schemas --no-score-cache --no-pause `
    --csv results/eval_traces/alhambra_100m.csv --best-csv results/eval_traces/alhambra_100m_best.csv
}

# 2. Rank transfer: the NEW full-scale candidate set on the subsamples.
Step 'rank transfer sub2m + sub5m' {
  $schemaList = (Import-Csv results/eval_traces/alhambra_100m.csv | ForEach-Object { $_.schema_path } | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique) -join ';'
  foreach ($sub in @('sub2m','sub5m')) {
    & $exe --mode schema-search --flat-search --evaluator cpu `
      --input "results/traces/alhambra_100m/alhambra_$sub.las" --no-synthetic `
      --schemas $schemaList --generate-schemas 0 --no-baselines `
      --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/alhambra_100m/pipeline_trace.csv `
      --queries 1000 --no-score-cache --no-pause --csv "results/eval_traces/alhambra_$sub.csv"
  }
}

# 3. CI re-measures (winner JSONs were re-exported by the GAs above).
Step 'CI x3' {
  & $exe --mode schema-search --flat-search --evaluator cpu --input "D:\Datasets\Point Clouds\SanAndreas\5M.las" --no-synthetic `
    --schemas results/eval_traces/best_schemas/5M_pipeline_replay_best_schema.json --generate-schemas 0 --no-baselines `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/sanandreas_5m/pipeline_trace.csv `
    --queries 3000 --measure-repeats 5 --no-score-cache --no-pause --csv results/eval_traces/ci_5m.csv
  & $exe --mode schema-search --flat-search --evaluator cpu --input "results/traces/sanandreas_50m/50M_converted.las" --no-synthetic `
    --schemas results/eval_traces/best_schemas/50M_converted_pipeline_replay_best_schema.json --generate-schemas 0 --no-baselines `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/sanandreas_50m/pipeline_trace.csv `
    --queries 1500 --measure-repeats 5 --no-score-cache --no-pause --csv results/eval_traces/ci_50m.csv
  & $exe --mode schema-search --flat-search --evaluator cpu --input "D:\Datasets\Point Clouds\Alhambra\Alhambra_100M.las" --no-synthetic `
    --schemas results/eval_traces/best_schemas/Alhambra_100M_pipeline_replay_best_schema.json --generate-schemas 0 --no-baselines `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/alhambra_100m/pipeline_trace.csv `
    --queries 1000 --measure-repeats 5 --no-score-cache --no-pause --csv results/eval_traces/ci_alhambra.csv
}

# 4. Framework comparisons (Open3D) on the corrected replay.
Step 'framework compare 5M + Alhambra' {
  & $pyMds scripts/compare_frameworks.py --exe $exe `
    --input "D:\Datasets\Point Clouds\SanAndreas\5M.las" `
    --schema results/eval_traces/best_schemas/5M_pipeline_replay_best_schema.json `
    --workload-profile configs/workloads/pipeline_replay.json `
    --input-trace results/traces/sanandreas_5m/pipeline_trace.csv `
    --queries 3000 --frameworks open3d --out-dir results/framework_compare/sanandreas_5m_trace
  & $pyMds scripts/compare_frameworks.py --exe $exe `
    --input "D:\Datasets\Point Clouds\Alhambra\Alhambra_100M.las" `
    --schema results/eval_traces/best_schemas/Alhambra_100M_pipeline_replay_best_schema.json `
    --workload-profile configs/workloads/pipeline_replay.json `
    --input-trace results/traces/alhambra_100m/pipeline_trace.csv `
    --queries 3000 --frameworks open3d --out-dir results/framework_compare/alhambra_100m_trace
}

# 5. 700M multi-fidelity (capture + subsample already exist and are world-coord valid).
Step 'GA on 700M subsample' {
  & $exe --mode schema-search --input "results/traces/solarpanels_700m/solar_sub5m.las" --no-synthetic `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/solarpanels_700m/pipeline_trace.csv `
    --queries 1500 --evaluator cpu --generate-schemas 32 --optimizer-generations 3 --optimizer-population 24 --parallel 8 `
    --optimizer-output-dir results/eval_traces/best_schemas_700m --no-score-cache --no-pause `
    --csv results/eval_traces/solar_sub5m.csv
}
Step 'confirm top-10 at full 700M' {
  $unique = [ordered]@{}
  foreach ($r in (Import-Csv results/eval_traces/solar_sub5m.csv | Sort-Object { [double]$_.score })) {
    if (-not $unique.Contains($r.schema_name) -and $r.schema_path -and (Test-Path $r.schema_path)) { $unique[$r.schema_name] = $r.schema_path }
    if ($unique.Count -ge 10) { break }
  }
  $schemaList = (@($unique.Values) | Select-Object -Unique) -join ';'
  Write-Output "confirming $((($schemaList -split ';')).Count) schemas at 700M"
  & $exe --mode schema-search --flat-search --evaluator cpu `
    --input "D:\Datasets\Point Clouds\SolarPanels\SolarPanels700M.las" --no-synthetic `
    --schemas $schemaList --generate-schemas 0 `
    --workloads configs/workloads/pipeline_replay.json --input-trace results/traces/solarpanels_700m/pipeline_trace.csv `
    --queries 1000 --no-score-cache --no-pause --csv results/eval_traces/solar_700m_confirm.csv
}

# 6. PCL chain last (vcpkg compile is CPU-heavy; keep it off the timed steps).
Step 'vcpkg install pcl' {
  & C:\vcpkg\vcpkg.exe install "pcl[core]:x64-windows" --recurse
}
Step 'build pcl helper' {
  Import-Module "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Microsoft.VisualStudio.DevShell.dll" -ErrorAction SilentlyContinue
  Enter-VsDevShell -VsInstallPath "C:\Program Files\Microsoft Visual Studio\18\Community" -DevCmdArguments '-arch=x64' -SkipAutomaticLocation
  Set-Location $repo
  $inc = 'C:\vcpkg\installed\x64-windows\include'
  $lib = 'C:\vcpkg\installed\x64-windows\lib'
  New-Item -ItemType Directory -Force -Path "$repo\tools\bin" | Out-Null
  $libs = (Get-ChildItem "$lib\pcl_*.lib" | ForEach-Object { $_.FullName }) + @("$lib\flann.lib", "$lib\flann_cpp.lib" | Where-Object { Test-Path $_ })
  cl /nologo /std:c++17 /EHsc /O2 /MD /DNOMINMAX /I $inc /I "$inc\eigen3" `
    "$repo\tools\pcl_point_baseline.cpp" /Fe:"$repo\tools\bin\pcl_point_baseline.exe" /link $libs
}
Step 'PCL column on 5M trace' {
  & $py "C:\Users\PC\AppData\Local\Temp\claude\c--Github-MultiDataStructure\c5d468df-b0e4-4d44-aac6-68617e591b4c\scratchpad\to_binary_ply.py" "D:\Datasets\Point Clouds\SanAndreas\5M.las" "$repo\results\traces\sanandreas_5m\5M.ply"
  & $pyMds scripts/compare_frameworks.py --exe $exe `
    --input "results/traces/sanandreas_5m/5M.ply" `
    --schema results/eval_traces/best_schemas/5M_pipeline_replay_best_schema.json `
    --workload-profile configs/workloads/pipeline_replay.json `
    --input-trace results/traces/sanandreas_5m/pipeline_trace.csv `
    --queries 3000 --frameworks open3d pcl --pcl-exe tools/bin/pcl_point_baseline.exe `
    --out-dir results/framework_compare/sanandreas_5m_pcl
}

Write-Output "`n########## corrected battery complete [$(Get-Date -Format HH:mm:ss)] ##########"

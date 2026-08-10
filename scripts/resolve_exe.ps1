# Resolve the MultiDataStructure executable, and refuse to return a stale one.
#
# Why this exists. MSBuild links the solution to <repo>\x64\Release\MultiDataStructure.exe, but
# every runner script used to hard-code <repo>\MultiDataStructure\x64\Release\MultiDataStructure.exe
# -- a legacy path left over from building the project standalone rather than through the .sln.
# Nothing writes there any more, so that file sat frozen at a build from 2026-07-28 while the
# scripts happily measured it. C++ changes appeared to have "no effect" because the binary under
# test did not contain them, and a whole conclusion was drawn from that before the mismatch was
# noticed. A silently stale binary is the worst kind of measurement bug: everything succeeds.
#
# Dot-source this and call Resolve-MdsExe:
#   . "$PSScriptRoot\resolve_exe.ps1"
#   $exe = Resolve-MdsExe

function Resolve-MdsExe {
    param(
        [string]$Repo = (Split-Path -Parent $PSScriptRoot),
        # Set to skip the freshness check (e.g. deliberately measuring an older build A/B).
        [switch]$AllowStale
    )

    $exe = Join-Path $Repo 'x64\Release\MultiDataStructure.exe'
    $legacy = Join-Path $Repo 'MultiDataStructure\x64\Release\MultiDataStructure.exe'

    if (-not (Test-Path $exe)) {
        if (Test-Path $legacy) {
            throw "Built executable not found at $exe, but a legacy copy exists at $legacy. " +
                  "That path is NOT a build output any more - rebuild the solution " +
                  "(msbuild MultiDataStructure.sln /p:Configuration=Release /p:Platform=x64)."
        }
        throw "Built executable not found at $exe. Build the solution first."
    }

    if (-not $AllowStale) {
        $exeTime = (Get-Item $exe).LastWriteTime
        $newest = Get-ChildItem (Join-Path $Repo 'MultiDataStructure') -Recurse -File `
            -Include *.cpp, *.h, *.cu, *.cuh -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($newest -and $newest.LastWriteTime -gt $exeTime) {
            throw "$exe was built $exeTime but $($newest.Name) changed $($newest.LastWriteTime). " +
                  "Rebuild before measuring, or pass -AllowStale if this is deliberate."
        }
    }

    return $exe
}

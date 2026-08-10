# Build the Indexicon external replay baseline (tools/indexicon_point_baseline.cpp).
#
# Indexicon is a third-party header-only library (MIT). It is NOT vendored into this repo -
# external/ is gitignored - so this script fetches it at a pinned commit and compiles against it,
# which is what makes the baseline numbers reproducible on another machine.
#
#   .\tools\build_indexicon_baseline.ps1
#   .\tools\build_indexicon_baseline.ps1 -Clean          # discard and re-clone the pinned tree
#
# Output: tools/bin/indexicon_point_baseline.exe
param(
  [string]$Commit = 'c9f9b1da7a55c44fcb30435bcfe101f1f699784f',
  [string]$Repo = 'https://github.com/psimatis/Indexicon-Spatial-Library',
  [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$clone = Join-Path $root 'external\Indexicon'
$source = Join-Path $root 'tools\indexicon_point_baseline.cpp'
$binDir = Join-Path $root 'tools\bin'
$exe = Join-Path $binDir 'indexicon_point_baseline.exe'

if ($Clean -and (Test-Path $clone)) {
  Write-Output "removing $clone"
  Remove-Item -Recurse -Force $clone
}

if (-not (Test-Path (Join-Path $clone '.git'))) {
  Write-Output "cloning $Repo -> $clone"
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $clone) | Out-Null
  git clone --quiet $Repo $clone
  if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

$head = (git -C $clone rev-parse HEAD).Trim()
if ($head -ne $Commit) {
  Write-Output "checking out pinned commit $Commit (was $head)"
  git -C $clone fetch --quiet origin
  git -C $clone checkout --quiet $Commit
  if ($LASTEXITCODE -ne 0) { throw "could not check out $Commit" }
}
Write-Output "Indexicon pinned at $((git -C $clone rev-parse --short HEAD).Trim()) (MIT, see external/Indexicon/LICENSE)"

# Locate the MSVC toolchain. cl.exe is normally not on PATH outside a developer prompt.
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($cl) {
  $vcvars = $null
} else {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found; run this from a Developer PowerShell, or install VS Build Tools" }
  $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (-not $vsPath) { throw "no Visual Studio installation with the C++ toolset was found" }
  $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
  if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found under $vsPath" }
}

New-Item -ItemType Directory -Force -Path $binDir | Out-Null
$objDir = Join-Path $binDir 'obj'
New-Item -ItemType Directory -Force -Path $objDir | Out-Null

# /Fo keeps the .obj out of the repo root (an earlier build left one stranded there).
$clArgs = "/nologo /std:c++17 /EHsc /O2 /MD /DNOMINMAX /I `"$clone`" `"$source`" /Fo`"$objDir\\`" /Fe`"$exe`""
if ($vcvars) {
  # vcvars64.bat shells out to vswhere.exe by bare name; if the Installer directory is not on
  # PATH it prints a spurious "not recognized" line to stderr before carrying on. Putting the
  # directory on PATH for the child keeps the build output clean.
  $installerDir = Split-Path -Parent $vswhere
  cmd /c "set `"PATH=$installerDir;%PATH%`" && `"$vcvars`" >nul && cl $clArgs"
} else {
  cmd /c "cl $clArgs"
}
if ($LASTEXITCODE -ne 0) { throw "compilation failed (exit $LASTEXITCODE)" }

Write-Output "built $exe"

# Build the nanoflann external replay baseline (tools/nanoflann_point_baseline.cpp).
# nanoflann is header-only; the header comes from vcpkg (installed with PCL). Same optimisation
# flags as MultiDataStructure.vcxproj, so neither side is compiled more aggressively.
#
#   .\tools\build_nanoflann_baseline.ps1
#
# Output: tools/bin/nanoflann_point_baseline.exe
param([string]$VcpkgInclude = 'C:\vcpkg\installed\x64-windows\include')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $root 'tools\nanoflann_point_baseline.cpp'
$binDir = Join-Path $root 'tools\bin'
$objDir = Join-Path $binDir 'obj'
$exe = Join-Path $binDir 'nanoflann_point_baseline.exe'
if (-not (Test-Path (Join-Path $VcpkgInclude 'nanoflann.hpp'))) { throw "nanoflann.hpp not found under $VcpkgInclude (vcpkg install nanoflann)" }
New-Item -ItemType Directory -Force -Path $objDir | Out-Null

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
$clArgs = "/nologo /std:c++17 /EHsc /O2 /arch:AVX2 /GL /MD /DNOMINMAX /I `"$VcpkgInclude`" `"$source`" /Fo`"$objDir\\`" /Fe`"$exe`""
$installerDir = Split-Path -Parent $vswhere
cmd /c "set `"PATH=$installerDir;%PATH%`" && `"$vcvars`" >nul && cl $clArgs"
if ($LASTEXITCODE -ne 0) { throw "compilation failed (exit $LASTEXITCODE)" }
Write-Output "built $exe"

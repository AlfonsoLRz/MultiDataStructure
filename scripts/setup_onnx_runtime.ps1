param(
    [string]$Version = "1.25.1",
    [string]$PackageId = "microsoft.ml.onnxruntime.gpu.windows"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$depsDir = Join-Path $root ".deps"
$nugetDir = Join-Path $depsDir "nuget"
$packageDir = Join-Path $depsDir "onnxruntime-gpu-windows-$Version"
$nupkg = Join-Path $nugetDir "$PackageId.$Version.nupkg"
$zip = Join-Path $nugetDir "$PackageId.$Version.zip"
$url = "https://api.nuget.org/v3-flatcontainer/$PackageId/$Version/$PackageId.$Version.nupkg"

New-Item -ItemType Directory -Force -Path $nugetDir | Out-Null

if (-not (Test-Path $nupkg)) {
    Write-Host "Downloading $PackageId $Version..."
    Invoke-WebRequest -Uri $url -OutFile $nupkg
}

Copy-Item -LiteralPath $nupkg -Destination $zip -Force
Expand-Archive -LiteralPath $zip -DestinationPath $packageDir -Force

$includeDir = Join-Path $packageDir "buildTransitive\native\include"
$nativeDir = Join-Path $packageDir "runtimes\win-x64\native"

if (-not (Test-Path (Join-Path $includeDir "onnxruntime_cxx_api.h"))) {
    throw "ONNX Runtime header was not found after extraction: $includeDir"
}
if (-not (Test-Path (Join-Path $nativeDir "onnxruntime.lib"))) {
    throw "ONNX Runtime import library was not found after extraction: $nativeDir"
}

Write-Host "ONNX Runtime SDK is ready:"
Write-Host "  Include: $includeDir"
Write-Host "  Native:  $nativeDir"

param(
    [string]$Configuration = "Release",
    [string]$Platform = "x64",
    [string]$Version = "1.25.1"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$packageDir = Join-Path $root ".deps\onnxruntime-gpu-windows-$Version"

if (-not (Test-Path $packageDir)) {
    & (Join-Path $PSScriptRoot "setup_onnx_runtime.ps1") -Version $Version
}

$includeDir = Join-Path $packageDir "buildTransitive\native\include"
$nativeDir = Join-Path $packageDir "runtimes\win-x64\native"
$msbuild = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
if (-not (Test-Path $msbuild)) {
    throw "MSBuild was not found at $msbuild"
}

& $msbuild (Join-Path $root "MultiDataStructure.sln") `
    /p:Configuration=$Configuration `
    /p:Platform=$Platform `
    /p:OnnxRuntimeIncludeDir="$includeDir" `
    /p:OnnxRuntimeLibraryDir="$nativeDir" `
    /p:OnnxRuntimeBinDir="$nativeDir" `
    /m

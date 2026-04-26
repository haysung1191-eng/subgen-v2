param(
    [string]$Preset = "filmora-ko",
    [string]$Device = "cuda",
    [string]$AlignDevice = "cuda",
    [string]$ComputeType = "float16"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms

$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = "Select a video or audio file"
$dialog.Filter = "Media Files|*.mp4;*.mkv;*.avi;*.mov;*.wmv;*.m4v;*.webm;*.mp3;*.wav;*.m4a;*.flac;*.aac;*.ogg|All Files|*.*"
$dialog.Multiselect = $false

if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
    Write-Host "No file selected."
    exit 0
}

$inputPath = $dialog.FileName
$inputItem = Get-Item $inputPath
$outputPath = Join-Path $inputItem.DirectoryName ($inputItem.BaseName + ".v2.srt")
$debugDir = Join-Path $inputItem.DirectoryName ($inputItem.BaseName + "-v2-debug")

Set-Location $PSScriptRoot

Write-Host "Input : $inputPath"
Write-Host "Output: $outputPath"
Write-Host "Debug : $debugDir"
Write-Host "Preset: $Preset"

python -m subgen_v2.cli `
    $inputPath `
    -o $outputPath `
    --preset $Preset `
    --device $Device `
    --align-device $AlignDevice `
    --compute-type $ComputeType `
    --debug-dir $debugDir

if ($LASTEXITCODE -ne 0) {
    throw "subgen_v2 failed with exit code $LASTEXITCODE"
}

python -m subgen_v2.review `
    $debugDir `
    --top 40

if ($LASTEXITCODE -ne 0) {
    throw "subgen_v2 review failed with exit code $LASTEXITCODE"
}

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

python -m subgen_v2.cli `
    $inputPath `
    -o $outputPath `
    --device cuda `
    --align-device cuda `
    --align-utterance-padding-ms 180 `
    --end-fallback-threshold-ms 320 `
    --debug-dir $debugDir

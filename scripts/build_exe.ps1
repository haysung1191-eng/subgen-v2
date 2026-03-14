param(
    [switch]$OneFile
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$entry = 'src/subgen/gui.py'
if (!(Test-Path $entry)) {
    throw "Entry file not found: $entry"
}

$args = @(
    '-m', 'PyInstaller',
    '--noconfirm',
    '--clean',
    '--name', 'subgen-gui',
    '--windowed',
    '--paths', 'src',
    '--collect-all', 'ctranslate2',
    '--collect-all', 'faster_whisper',
    '--collect-all', 'silero_vad',
    '--collect-all', 'numpy',
    '--collect-all', 'whisperx',
    '--hidden-import', 'tkinter',
    '--hidden-import', 'torch',
    $entry
)

if ($OneFile) {
    $args += '--onefile'
}

python @args

Write-Host "Build completed. Check dist\\subgen-gui\\subgen-gui.exe (onedir) or dist\\subgen-gui.exe (onefile)."

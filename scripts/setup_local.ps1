Param(
    [switch]$ForceEnv
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$envSource = Join-Path $repoRoot ".env.local.example"
$envTarget = Join-Path $repoRoot ".env"

if (-not (Test-Path $envSource)) {
    throw "Missing template: $envSource"
}

if ($ForceEnv -or -not (Test-Path $envTarget)) {
    Copy-Item -Path $envSource -Destination $envTarget -Force
    Write-Host "Wrote local environment file to $envTarget"
} else {
    Write-Host ".env already exists. Use -ForceEnv to overwrite it from .env.local.example"
}

if (-not (Test-Path $pythonExe)) {
    py -3.11 -m venv $venvPath
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install ".[pipeline,rag-local,advanced-analytics,finance,healthcare]"

Write-Host ""
Write-Host "Local full profile setup complete."
Write-Host "Next: .\scripts\run_local.ps1"

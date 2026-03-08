$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$envPath = Join-Path $repoRoot ".env"

if (-not (Test-Path $pythonExe)) {
    throw "Missing virtual environment. Run .\scripts\setup_local.ps1 first."
}

if (-not (Test-Path $envPath)) {
    throw "Missing .env file. Run .\scripts\setup_local.ps1 first."
}

Get-Content -Path $envPath | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#")) {
        return
    }

    $parts = $line.Split("=", 2)
    if ($parts.Count -ne 2) {
        return
    }

    [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1])
}

Set-Location $repoRoot
& $pythonExe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

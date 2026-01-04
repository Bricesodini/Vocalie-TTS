#!/usr/bin/env pwsh
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RunDir = Join-Path $RootDir '.run'
$FrontendDir = Join-Path $RootDir 'frontend'

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Error "node >= 20 not found. Install Node.js and rerun."
    exit 1
}

if (-not (Test-Path $RunDir)) {
    New-Item -ItemType Directory -Path $RunDir | Out-Null
}

function Install-Frontend {
    if (-not (Test-Path (Join-Path $FrontendDir 'node_modules'))) {
        Write-Host "Installing frontend dependencies (Windows workflow)"
        Push-Location $FrontendDir
        npm install --include=optional --no-audit --progress=false
        Pop-Location
    }
}

function Start-Proc {
    param($Name, $Cmd)
    Write-Host "Starting $Name..."
    $netCmd = if ($Cmd -is [scriptblock]) { $Cmd } else { { Invoke-Expression $Cmd } }
    $job = Start-Process -FilePath pwsh -ArgumentList '-NoLogo', '-NoProfile', '-Command', $Cmd -PassThru
    $job.Id | Out-File -Encoding ascii -FilePath (Join-Path $RunDir "$Name.pid")
}

Install-Frontend

Start-Proc -Name 'backend' -Cmd "bash scripts/dev-backend.sh"
Start-Proc -Name 'frontend' -Cmd "cd $FrontendDir; npm run dev"

Write-Host "Backend: http://127.0.0.1:8000"
Write-Host "Frontend: http://localhost:3000"
Write-Host "PIDs stored in $RunDir"
Write-Host "Press Ctrl+C to stop all; use scripts/stop.sh from a Git Bash shell."

trap {
    & "$RootDir/scripts/stop.sh" | Out-Null
    exit
} SIGINT, SIGTERM

while ($true) {
    Start-Sleep -Seconds 5
}

# Start AI Studio: web app on http://127.0.0.1:8000, MCP on http://127.0.0.1:8000/mcp
# Runs on Windows (not WSL): the GPU engines are Windows programs and WSL can't reach the AMD card.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "No .venv yet. See README: Setup." }
$hostAddr = if ($env:STUDIO_HOST) { $env:STUDIO_HOST } else { '127.0.0.1' }
$port = if ($env:STUDIO_PORT) { $env:STUDIO_PORT } else { '8000' }
& $py -m uvicorn studio.app:app --host $hostAddr --port $port

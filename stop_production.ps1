# Stop Production API Server (Windows PowerShell)

Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Stopping Production API Server                          ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "api.pid") {
    $PID = Get-Content "api.pid"
    $process = Get-Process -Id $PID -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id $PID -Force
        Write-Host "✓ Server stopped (PID: $PID)" -ForegroundColor Green
        Remove-Item "api.pid"
    } else {
        Write-Host "⚠ Process not found (PID: $PID)" -ForegroundColor Yellow
        Remove-Item "api.pid"
    }
} else {
    # Fallback: kill by process name
    $processes = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*start_api_production.py*" }
    if ($processes) {
        $processes | Stop-Process -Force
        Write-Host "✓ Server stopped" -ForegroundColor Green
    } else {
        Write-Host "ℹ No running server found" -ForegroundColor Yellow
    }
}

Write-Host ""



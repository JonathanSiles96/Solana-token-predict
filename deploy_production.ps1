# Production Deployment Script for Windows
# PowerShell version of deploy_production.sh

Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Production Deployment - Solana Token API                ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop current service if running
Write-Host "[1/7] Stopping current API service..." -ForegroundColor Yellow
$process = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*start_api_production.py*" }
if ($process) {
    Stop-Process -Id $process.Id -Force
    Write-Host "✓ Service stopped" -ForegroundColor Green
    Start-Sleep -Seconds 2
} else {
    Write-Host "ℹ No running service found" -ForegroundColor Yellow
}

# Step 2: Backup current database
Write-Host "[2/7] Backing up database..." -ForegroundColor Yellow
if (Test-Path "data\signals.db") {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupFile = "data\signals_backup_$timestamp.db"
    Copy-Item "data\signals.db" $backupFile
    Write-Host "✓ Database backed up to $backupFile" -ForegroundColor Green
}

# Step 3: Pull latest changes (if using git)
Write-Host "[3/7] Pulling latest changes..." -ForegroundColor Yellow
if (Test-Path ".git") {
    git pull
    Write-Host "✓ Code updated" -ForegroundColor Green
} else {
    Write-Host "ℹ Not a git repository, skipping" -ForegroundColor Yellow
}

# Step 4: Install/Update dependencies
Write-Host "[4/7] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --upgrade
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Step 5: Verify model exists
Write-Host "[5/7] Verifying model file..." -ForegroundColor Yellow
if (-not (Test-Path "outputs\models\token_scorer.pkl")) {
    Write-Host "✗ Model file not found! Please train the model first." -ForegroundColor Red
    exit 1
}
Write-Host "✓ Model file found" -ForegroundColor Green

# Step 6: Create logs directory if not exists
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# Step 7: Start production server
Write-Host "[6/7] Starting production server..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs\api_$timestamp.log"

Start-Process -FilePath "python" -ArgumentList "start_api_production.py" -NoNewWindow -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.error"
Start-Sleep -Seconds 3

# Check if server started
$serverProcess = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*start_api_production.py*" }
if ($serverProcess) {
    Write-Host "✓ Production server started (PID: $($serverProcess.Id))" -ForegroundColor Green
    $serverProcess.Id | Out-File "api.pid"
} else {
    Write-Host "✗ Failed to start server. Check logs: $logFile" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "    Deployment completed successfully!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "📡 API is running at: http://localhost:8000"
Write-Host "📡 External access: http://185.8.107.12:8000"
Write-Host "📚 Docs: http://localhost:8000/docs"
Write-Host "📝 Logs: $logFile"
Write-Host ""
Write-Host "To stop: Stop-Process -Id $($serverProcess.Id)" -ForegroundColor Cyan
Write-Host ""



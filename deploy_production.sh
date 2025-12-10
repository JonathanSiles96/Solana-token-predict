#!/bin/bash
# Production Deployment Script for Linux

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Production Deployment - Solana Token API                ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Stop current service if running
echo -e "${YELLOW}[1/7] Stopping current API service...${NC}"
if pgrep -f "start_api_production.py" > /dev/null; then
    pkill -f "start_api_production.py"
    echo -e "${GREEN}✓ Service stopped${NC}"
else
    echo -e "${YELLOW}ℹ No running service found${NC}"
fi
sleep 2

# Step 2: Backup current database
echo -e "${YELLOW}[2/7] Backing up database...${NC}"
if [ -f "data/signals.db" ]; then
    BACKUP_FILE="data/signals_backup_$(date +%Y%m%d_%H%M%S).db"
    cp data/signals.db "$BACKUP_FILE"
    echo -e "${GREEN}✓ Database backed up to $BACKUP_FILE${NC}"
fi

# Step 3: Pull latest changes (if using git)
echo -e "${YELLOW}[3/7] Pulling latest changes...${NC}"
if [ -d ".git" ]; then
    git pull
    echo -e "${GREEN}✓ Code updated${NC}"
else
    echo -e "${YELLOW}ℹ Not a git repository, skipping${NC}"
fi

# Step 4: Install/Update dependencies
echo -e "${YELLOW}[4/7] Installing dependencies...${NC}"
pip install -r requirements.txt --upgrade
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 5: Verify model exists
echo -e "${YELLOW}[5/7] Verifying model file...${NC}"
if [ ! -f "outputs/models/token_scorer.pkl" ]; then
    echo -e "${RED}✗ Model file not found! Please train the model first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Model file found${NC}"

# Step 6: Test API startup
echo -e "${YELLOW}[6/7] Testing API startup...${NC}"
timeout 5 python start_api_production.py &
PID=$!
sleep 3
if ps -p $PID > /dev/null; then
    kill $PID
    echo -e "${GREEN}✓ API startup test passed${NC}"
else
    echo -e "${RED}✗ API failed to start${NC}"
    exit 1
fi

# Step 7: Start production server
echo -e "${YELLOW}[7/7] Starting production server...${NC}"
nohup python start_api_production.py > logs/api_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > api.pid
echo -e "${GREEN}✓ Production server started (PID: $(cat api.pid))${NC}"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}    Deployment completed successfully!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "📡 API is running at: http://$(hostname -I | awk '{print $1}'):8000"
echo "📚 Docs: http://$(hostname -I | awk '{print $1}'):8000/docs"
echo "📝 Logs: logs/api_*.log"
echo "🔍 Status: curl http://localhost:8000/health"
echo ""
echo "To stop the server: ./stop_production.sh"
echo ""


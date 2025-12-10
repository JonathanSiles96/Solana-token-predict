#!/bin/bash
# Stop Production API Server

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Stopping Production API Server                          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

if [ -f "api.pid" ]; then
    PID=$(cat api.pid)
    if ps -p $PID > /dev/null; then
        kill $PID
        echo "✓ Server stopped (PID: $PID)"
        rm api.pid
    else
        echo "⚠ Process not found (PID: $PID)"
        rm api.pid
    fi
else
    # Fallback: kill by process name
    if pgrep -f "start_api_production.py" > /dev/null; then
        pkill -f "start_api_production.py"
        echo "✓ Server stopped"
    else
        echo "ℹ No running server found"
    fi
fi

echo ""


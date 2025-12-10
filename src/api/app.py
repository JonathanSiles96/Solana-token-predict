"""
FastAPI application for Token Filtering & Gain Drivers API
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.models import (
    TokenSignal, EntrySignal, TokenScoreRequest, TokenScoreResponse,
    TokenRankingResponse, BulkSignalRequest, HealthResponse,
    StatsResponse, TakeProfitLevel, TradeOutcome, RetrainingStatus,
    MaxReturnUpdate, PriceUpdate, MaxReturnStats,
    TraceTradeRegistration, TraceTradeFetch, TraceTradeResponse, TraceTradeStats
)
from src.trading.dynamic_orders import DynamicOrderManager
from src.api.database import SignalDatabase
from src.models.token_scorer import TokenScorer
from src.data_processing.data_loader import FeatureExtractor
from src.models.auto_retrain import AutoRetrainer, BackgroundRetrainingTask
from src.analysis.trace_fetcher import TraceDataFetcher
from src.analysis.prediction_analyzer import PredictionAnalyzer
from src.strategy.config import get_strategy

# Initialize FastAPI app
app = FastAPI(
    title="Solana Token Filtering & Gain Drivers API",
    description="Real-time token scoring and signal management for Solana trading",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
db = SignalDatabase()
scorer: Optional[TokenScorer] = None
feature_extractor = FeatureExtractor()

# Dynamic order manager (trailing stop loss, adaptive TP/SL)
order_manager = DynamicOrderManager(
    initial_capital=10000,  # Adjust to your capital
    max_position_size=0.10,  # 10% max per position
    min_position_size=0.02   # 2% min per position
)

# Auto-retraining (retrain when 50 new outcomes, minimum 100 total samples)
auto_retrainer = AutoRetrainer(
    retrain_threshold=50,
    min_total_samples=100,
    model_type='gradient_boosting'
)
retraining_task: Optional[BackgroundRetrainingTask] = None

# Trace_24H integration
trace_fetcher = TraceDataFetcher()
prediction_analyzer = PredictionAnalyzer()

# Strategy configuration (used for filtering)
strategy = get_strategy()

# Load model on startup
@app.on_event("startup")
async def load_model():
    """Load trained model on startup"""
    global scorer, retraining_task
    try:
        model_path = "outputs/models/token_scorer.pkl"
        if Path(model_path).exists():
            scorer = TokenScorer.load(model_path)
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"⚠ Model not found at {model_path}. Score endpoints will be unavailable.")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
    
    # Start background retraining task (check every 5 minutes)
    try:
        retraining_task = BackgroundRetrainingTask(
            auto_retrainer=auto_retrainer,
            check_interval=300  # 5 minutes
        )
        retraining_task.start()
        print(f"✓ Background retraining task started")
    except Exception as e:
        print(f"⚠️ Failed to start background retraining: {e}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global retraining_task
    if retraining_task:
        retraining_task.stop()


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API homepage with data summary"""
    stats = db.get_statistics()
    
    # Get top 10 tokens by market cap
    signals = db.get_latest_signals(limit=50)
    
    import pandas as pd
    df = pd.DataFrame(signals) if signals else pd.DataFrame()
    
    top_tokens = []
    if not df.empty and 'signal_mc' in df.columns:
        df_sorted = df.sort_values(by='signal_mc', ascending=False).head(10)
        top_tokens = df_sorted[['mint_key', 'signal_mc', 'signal_liquidity', 
                                'signal_volume_1h', 'signal_holders']].to_dict('records')
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": scorer is not None and scorer.trained,
        "database_connected": True,
        "statistics": {
            "total_signals": stats['total_signals'],
            "total_entry_signals": stats['total_entry_signals'],
            "avg_gain": f"{stats['avg_gain']:.2%}" if stats['avg_gain'] else "N/A",
            "median_gain": f"{stats['median_gain']:.2%}" if stats['median_gain'] else "N/A"
        },
        "top_10_tokens_by_mc": top_tokens,
        "endpoints": {
            "docs": "http://185.8.107.12:8000/docs",
            "stats": "http://185.8.107.12:8000/stats",
            "rank": "http://185.8.107.12:8000/tokens/rank?top_n=10",
            "signals": "http://185.8.107.12:8000/signals/tokens?limit=20"
        },
        "timestamp": datetime.utcnow()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    stats = db.get_statistics()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=scorer is not None and scorer.trained,
        database_connected=True,
        total_signals=stats['total_signals']
    )


@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get database and performance statistics"""
    stats = db.get_statistics()
    return StatsResponse(
        total_signals=stats['total_signals'],
        total_entry_signals=stats['total_entry_signals'],
        date_range=stats['date_range'],
        avg_gain=stats['avg_gain'],
        median_gain=stats['median_gain'],
        top_performers=stats['top_performers']
    )


@app.get("/debug/signals")
async def debug_signals():
    """
    Debug endpoint to check signal storage
    
    Shows raw signal data to help diagnose issues
    """
    try:
        # Get latest 10 signals
        signals = db.get_latest_signals(limit=10)
        
        if not signals:
            return {
                "status": "no_signals",
                "message": "No signals in database",
                "total_count": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get total count
        stats = db.get_statistics()
        
        # Format signals for display
        formatted_signals = []
        for sig in signals:
            formatted_signals.append({
                "mint_key": sig.get('mint_key', 'N/A')[:12] + "...",
                "signal_at": sig.get('signal_at'),
                "signal_mc": sig.get('signal_mc'),
                "signal_liquidity": sig.get('signal_liquidity'),
                "signal_source": sig.get('signal_source'),
                "created_at": sig.get('created_at')
            })
        
        return {
            "status": "ok",
            "total_signals": stats['total_signals'],
            "latest_10_signals": formatted_signals,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/monitor/activity")
async def monitor_activity():
    """
    Monitor recent signal activity (last 5 minutes)
    
    Shows if your system is sending signals successfully
    """
    try:
        import pandas as pd
        from datetime import timedelta
        
        # Get recent signals
        recent_signals = db.get_latest_signals(limit=100)
        recent_entries = db.query_entry_signals(limit=100, offset=0)
        
        # DEBUG: Log what we got from database
        print(f"DEBUG - Retrieved {len(recent_signals)} token signals, {len(recent_entries)} entry signals")
        if recent_signals:
            print(f"DEBUG - Sample token signal signal_at: {recent_signals[0].get('signal_at')}")
        
        if not recent_signals and not recent_entries:
            return {
                "status": "no_data",
                "message": "No signals received yet",
                "token_signals": 0,
                "entry_signals": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Count signals by time window
        # Make timezone-aware for comparison
        from datetime import timezone
        now = datetime.now(timezone.utc)
        five_min_ago = now - timedelta(minutes=5)
        one_hour_ago = now - timedelta(hours=1)
        
        # DEBUG
        print(f"DEBUG - Now: {now}, Five min ago: {five_min_ago}")
        
        # Count token signals
        token_last_5min = 0
        token_last_hour = 0
        latest_token_signal = None
        sources = []
        
        if recent_signals:
            try:
                token_df = pd.DataFrame(recent_signals)
                if 'signal_at' in token_df.columns:
                    # Convert to timezone-aware datetime
                    token_df['signal_at'] = pd.to_datetime(token_df['signal_at'], errors='coerce', utc=True)
                    token_df = token_df.dropna(subset=['signal_at'])
                    
                    # DEBUG
                    if not token_df.empty:
                        print(f"DEBUG - Latest signal timestamp: {token_df['signal_at'].max()}")
                        print(f"DEBUG - Time diff: {now - token_df['signal_at'].max()}")
                    
                    if not token_df.empty:
                        token_last_5min = len(token_df[token_df['signal_at'] >= five_min_ago])
                        token_last_hour = len(token_df[token_df['signal_at'] >= one_hour_ago])
                        latest_token_signal = str(token_df['signal_at'].max())
                        
                        # DEBUG
                        print(f"DEBUG - Signals in last 5min: {token_last_5min}, last hour: {token_last_hour}")
                        
                        if 'signal_source' in token_df.columns:
                            sources = [s for s in token_df['signal_source'].unique() if s is not None]
            except Exception as e:
                print(f"Error processing token signals: {e}")
                import traceback
                traceback.print_exc()
        
        # Count entry signals
        entry_last_5min = 0
        entry_last_hour = 0
        latest_entry_signal = None
        
        if recent_entries:
            try:
                entry_df = pd.DataFrame(recent_entries)
                if 'signal_at' in entry_df.columns:
                    # Convert to timezone-aware datetime
                    entry_df['signal_at'] = pd.to_datetime(entry_df['signal_at'], errors='coerce', utc=True)
                    entry_df = entry_df.dropna(subset=['signal_at'])
                    
                    if not entry_df.empty:
                        entry_last_5min = len(entry_df[entry_df['signal_at'] >= five_min_ago])
                        entry_last_hour = len(entry_df[entry_df['signal_at'] >= one_hour_ago])
                        latest_entry_signal = str(entry_df['signal_at'].max())
            except Exception as e:
                print(f"Error processing entry signals: {e}")
                import traceback
                traceback.print_exc()
        
        return {
            "status": "active" if (token_last_5min > 0 or entry_last_5min > 0) else "idle",
            "token_signals": {
                "last_5_minutes": token_last_5min,
                "last_hour": token_last_hour,
                "latest_signal": latest_token_signal,
                "sources": sources
            },
            "entry_signals": {
                "last_5_minutes": entry_last_5min,
                "last_hour": entry_last_hour,
                "latest_signal": latest_entry_signal
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        print(f"✗ Monitor activity error: {e}")
        raise HTTPException(status_code=500, detail=f"Error monitoring activity: {str(e)}")


# ============================================================================
# SIGNAL INGESTION ENDPOINTS (for team to push data)
# ============================================================================

@app.post("/signals/token", status_code=201)
async def receive_token_signal(signal: TokenSignal):
    """
    Receive a new token signal from trading system
    
    This endpoint accepts real-time token signals with all relevant metrics.
    """
    try:
        signal_dict = signal.model_dump()
        
        # Ensure signal_at is properly formatted
        if isinstance(signal_dict.get('signal_at'), datetime):
            # Convert to ISO format string for storage
            signal_dict['signal_at'] = signal_dict['signal_at'].isoformat()
        
        row_id = db.insert_token_signal(signal_dict)
        
        # Log incoming signal with timestamp
        print(f"✓ Token signal received: {signal.mint_key[:8]}... | MC: {signal.signal_mc} | Source: {signal.signal_source} | Time: {signal_dict['signal_at']}")
        
        return {
            "status": "success",
            "message": "Token signal received",
            "id": row_id,
            "mint_key": signal.mint_key,
            "signal_at": signal_dict['signal_at'],
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        print(f"✗ Error receiving token signal: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error storing signal: {str(e)}")


@app.post("/signals/token/bulk", status_code=201)
async def receive_bulk_signals(request: BulkSignalRequest):
    """
    Receive multiple token signals at once
    
    Useful for backfilling historical data or batch updates.
    """
    try:
        inserted_ids = []
        for signal in request.signals:
            signal_dict = signal.model_dump()
            row_id = db.insert_token_signal(signal_dict)
            inserted_ids.append(row_id)
        
        return {
            "status": "success",
            "message": f"Inserted {len(inserted_ids)} signals",
            "count": len(inserted_ids),
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing signals: {str(e)}")


@app.post("/signals/entry", status_code=201)
async def receive_entry_signal(entry: EntrySignal):
    """
    Receive an entry signal with gain tracking
    
    Use this to track trade entries and subsequent performance.
    """
    try:
        entry_dict = entry.model_dump()
        
        # Ensure signal_at is properly formatted
        if isinstance(entry_dict.get('signal_at'), datetime):
            entry_dict['signal_at'] = entry_dict['signal_at'].isoformat()
        
        row_id = db.insert_entry_signal(entry_dict)
        
        # Log incoming entry signal with timestamp
        gain_str = f"{entry.gain_pct*100:.1f}%" if entry.gain_pct else "0%"
        print(f"✓ Entry signal received: {entry.ticker} | Entry MC: {entry.entry_mc} | Gain: {gain_str} | Time: {entry_dict['signal_at']}")
        
        return {
            "status": "success",
            "message": "Entry signal received",
            "id": row_id,
            "ticker": entry.ticker,
            "signal_at": entry_dict['signal_at'],
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        print(f"✗ Error receiving entry signal: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error storing entry signal: {str(e)}")


@app.put("/signals/entry/{ticker}/update")
async def update_entry_gain(
    ticker: str,
    current_mc: float = Query(..., description="Current market cap"),
    gain_pct: float = Query(..., description="Current gain percentage"),
    signal_at: datetime = Query(..., description="Original signal timestamp")
):
    """
    Update entry signal with current gain
    
    Use this to update ongoing trades with current performance.
    """
    try:
        db.update_entry_signal_gain(ticker, signal_at, current_mc, gain_pct)
        
        # Log update
        print(f"✓ Entry updated: {ticker} | Current MC: {current_mc} | Gain: {gain_pct*100:.1f}%")
        
        return {
            "status": "success",
            "message": "Entry signal updated",
            "ticker": ticker,
            "gain_pct": gain_pct,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        print(f"✗ Error updating entry signal: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating entry: {str(e)}")


# ============================================================================
# TRADE OUTCOME ENDPOINTS (for retraining)
# ============================================================================

@app.post("/trades/outcome", status_code=201)
async def record_trade_outcome(outcome: TradeOutcome, background_tasks: BackgroundTasks):
    """
    Record a trade outcome for model retraining
    
    When you exit a trade (TP or SL), send the outcome here.
    The system will automatically retrain when enough new outcomes accumulate.
    
    **NEW**: Now accepts max_return data! Include the highest return the token achieved.
    """
    try:
        outcome_dict = outcome.model_dump()
        
        # Ensure signal_at is properly formatted
        if isinstance(outcome_dict.get('signal_at'), datetime):
            outcome_dict['signal_at'] = outcome_dict['signal_at'].isoformat()
        
        # Store outcome
        row_id = db.insert_trade_outcome(outcome_dict)
        
        # If max_return provided, also update the signal
        if outcome.max_return is not None:
            try:
                db.update_max_return(
                    outcome.mint_key,
                    outcome.signal_at,
                    outcome.max_return,
                    outcome.max_return_mc or 0,
                    outcome.max_return_timestamp or datetime.utcnow()
                )
            except Exception as e:
                print(f"⚠️  Could not update signal max_return: {e}")
        
        # Log outcome
        gain_pct = outcome.actual_gain * 100
        max_return_str = f" | Max: {outcome.max_return*100:.1f}%" if outcome.max_return else ""
        is_winner = "WIN" if outcome.actual_gain >= 0.3 else "LOSS"
        print(f"✓ Trade outcome: {outcome.mint_key[:8]}... | Gain: {gain_pct:.1f}%{max_return_str} | {is_winner} | Reason: {outcome.exit_reason}")
        
        # Check if retraining needed (run in background)
        background_tasks.add_task(check_retraining_trigger)
        
        return {
            "status": "success",
            "message": "Trade outcome recorded",
            "id": row_id,
            "mint_key": outcome.mint_key,
            "is_winner": outcome.actual_gain >= 0.3,
            "max_return": outcome.max_return,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        print(f"✗ Error recording trade outcome: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error storing outcome: {str(e)}")


@app.get("/retraining/status", response_model=RetrainingStatus)
async def get_retraining_status():
    """
    Get current model retraining status
    
    Shows:
    - How many new outcomes waiting for retraining
    - Total training samples available
    - Current model version and performance
    - When last retrain happened
    """
    try:
        status = auto_retrainer.get_status()
        return RetrainingStatus(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")


@app.post("/retraining/trigger")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Manually trigger model retraining
    
    Normally retraining happens automatically when enough outcomes accumulate.
    Use this to force retraining immediately.
    """
    try:
        # Run retraining in background
        background_tasks.add_task(run_retraining)
        
        return {
            "status": "triggered",
            "message": "Retraining started in background",
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering retraining: {str(e)}")


# ============================================================================
# MAX RETURN TRACKING ENDPOINTS
# ============================================================================

@app.post("/tracking/max-return", status_code=201)
async def update_max_return(update: MaxReturnUpdate):
    """
    Update max return for a signal
    
    Use this to record the maximum return a token achieved after signaling.
    This is critical data for model evaluation and retraining.
    
    Example:
    ```python
    {
        "mint_key": "8K4VbtJ6...",
        "signal_at": "2025-11-05T18:00:00Z",
        "max_return": 2.45,  # 245% gain
        "max_return_mc": 350000,
        "max_return_timestamp": "2025-11-05T20:15:00Z"
    }
    ```
    """
    try:
        db.update_max_return(
            update.mint_key,
            update.signal_at,
            update.max_return,
            update.max_return_mc,
            update.max_return_timestamp
        )
        
        print(f"✓ Max return updated: {update.mint_key[:8]}... | {update.max_return*100:.1f}%")
        
        return {
            "status": "success",
            "message": "Max return updated",
            "mint_key": update.mint_key,
            "max_return_pct": update.max_return * 100,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        print(f"✗ Error updating max return: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating max return: {str(e)}")


@app.post("/tracking/price-update", status_code=200)
async def update_price_tracking(update: PriceUpdate):
    """
    Update current price for automatic max_return tracking
    
    Send current price updates and the system will automatically track
    if this represents a new maximum for the signal.
    
    Example:
    ```python
    {
        "mint_key": "8K4VbtJ6...",
        "signal_at": "2025-11-05T18:00:00Z",
        "current_price": 0.00025,
        "current_mc": 180000
    }
    ```
    """
    try:
        db.update_price_tracking(
            update.mint_key,
            update.signal_at,
            update.current_price,
            update.current_mc
        )
        
        return {
            "status": "success",
            "message": "Price tracking updated",
            "mint_key": update.mint_key,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating price: {str(e)}")


@app.get("/tracking/active-signals")
async def get_active_signals(limit: int = Query(100, ge=1, le=500)):
    """
    Get signals currently being tracked (active status)
    
    Returns signals that need ongoing price monitoring.
    """
    try:
        signals = db.get_signals_for_tracking(limit)
        
        return {
            "status": "success",
            "count": len(signals),
            "signals": signals,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting active signals: {str(e)}")


@app.post("/tracking/stop/{mint_key}")
async def stop_tracking(
    mint_key: str,
    signal_at: datetime = Query(..., description="Signal timestamp"),
    reason: str = Query("completed", description="Reason for stopping tracking")
):
    """
    Stop tracking a signal
    
    Use this when you're done monitoring a token (e.g., after exit or timeout).
    """
    try:
        db.stop_tracking_signal(mint_key, signal_at, reason)
        
        return {
            "status": "success",
            "message": "Tracking stopped",
            "mint_key": mint_key,
            "reason": reason,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping tracking: {str(e)}")


@app.get("/reports/max-returns")
async def get_max_returns_report(
    limit: int = Query(100, ge=1, le=1000),
    min_max_return: Optional[float] = Query(None, description="Filter by minimum max return (e.g., 0.5 for 50%)")
):
    """
    Get report of signals with their max_return data
    
    This shows the maximum return each signaled token achieved.
    Critical for evaluating model performance!
    
    Example response:
    ```json
    {
        "signals": [
            {
                "mint_key": "8K4VbtJ6...",
                "signal_at": "2025-11-05T18:00:00Z",
                "signal_mc": 100000,
                "max_return": 2.45,
                "max_return_mc": 345000,
                "max_return_timestamp": "2025-11-05T20:15:00Z"
            }
        ]
    }
    ```
    """
    try:
        signals = db.get_signals_with_max_return(limit, min_max_return)
        
        # Format for readability
        for signal in signals:
            if signal.get('max_return') is not None:
                signal['max_return_pct'] = signal['max_return'] * 100
        
        return {
            "status": "success",
            "count": len(signals),
            "signals": signals,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting max returns: {str(e)}")


@app.get("/reports/max-return-stats", response_model=MaxReturnStats)
async def get_max_return_statistics():
    """
    Get comprehensive statistics on max returns
    
    Shows:
    - How many signals have max_return data
    - Average, best, and worst max returns
    - Distribution (how many achieved 50%+, 100%+, 200%+)
    - Top 10 performers
    
    This is essential for understanding if your model is identifying good opportunities!
    """
    try:
        stats = db.get_max_return_statistics()
        return MaxReturnStats(**stats)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")


# ============================================================================
# DYNAMIC ORDER MANAGEMENT ENDPOINTS (Trailing Stop Loss & Adaptive TP/SL)
# ============================================================================

@app.post("/orders/create")
async def create_dynamic_orders(signal: TokenSignal, entry_price: float = Query(..., description="Entry price")):
    """
    🎯 Create dynamic order parameters with trailing stop loss
    
    This endpoint generates adaptive trading parameters that:
    - Calculate optimal position size based on confidence
    - Set dynamic stop loss (tighter for risky tokens)
    - Enable trailing stop loss after profit threshold
    - Generate multi-level take-profit strategy
    - Adapt to token risk profile
    
    Example:
    ```python
    response = requests.post(
        "http://localhost:8000/orders/create?entry_price=0.000123",
        json={
            "mint_key": "TOKEN_ABC",
            "signal_at": "2025-11-20T10:00:00Z",
            "signal_mc": 85000,
            "signal_liquidity": 28000,
            "signal_volume_1h": 45000,
            "signal_holders": 387,
            "signal_bundled_pct": 4.8,
            "signal_snipers_pct": 28.3
        }
    )
    ```
    """
    if scorer is None or not scorer.trained:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get model prediction
        token_data = signal.model_dump(exclude_none=True)
        
        import pandas as pd
        df = pd.DataFrame([token_data])
        df = feature_extractor.extract_numeric_features(df)
        df = feature_extractor.add_derived_features(df)
        token_data = df.iloc[0].to_dict()
        
        # Ensure features
        for feature in scorer.feature_names:
            if feature not in token_data or pd.isna(token_data[feature]):
                token_data[feature] = 0
        
        predicted_gain, confidence = scorer.score_token(token_data)
        
        model_prediction = {
            'predicted_gain': predicted_gain,
            'confidence': confidence
        }
        
        # Create dynamic orders
        orders = order_manager.create_initial_orders(
            token_data=token_data,
            model_prediction=model_prediction,
            entry_price=entry_price,
            token_id=signal.mint_key
        )
        
        # Get summary
        summary = order_manager.get_order_summary(orders)
        
        print(f"📝 Orders created: {signal.mint_key[:8]}... | Entry: ${entry_price:.6f}")
        
        return {
            "status": "success",
            "mint_key": signal.mint_key,
            "orders": summary,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error creating orders: {str(e)}")


@app.post("/orders/update/{mint_key}")
async def update_dynamic_orders(
    mint_key: str,
    current_price: float = Query(..., description="Current price"),
    current_volume: Optional[float] = Query(None, description="Current volume"),
    time_elapsed_minutes: Optional[int] = Query(None, description="Minutes since entry")
):
    """
    🔄 Update dynamic orders based on current price
    
    This endpoint:
    - Updates trailing stop loss if activated
    - Checks if TP levels hit
    - Detects weakness for aggressive exit
    - Extends TP levels on strong momentum
    - Provides exit signals
    
    Example:
    ```python
    response = requests.post(
        "http://localhost:8000/orders/update/TOKEN_ABC",
        params={
            "current_price": 0.000156,
            "time_elapsed_minutes": 30
        }
    )
    
    if response.json()['should_exit']:
        print("EXIT SIGNAL:", response.json()['reason'])
    ```
    """
    # Get active orders
    if mint_key not in order_manager.active_orders:
        raise HTTPException(status_code=404, detail=f"No active orders for {mint_key}")
    
    orders = order_manager.active_orders[mint_key]
    
    try:
        # Update orders
        updates = order_manager.update_orders(
            orders=orders,
            current_price=current_price,
            current_volume=current_volume,
            time_elapsed_minutes=time_elapsed_minutes,
            token_id=mint_key
        )
        
        # Log updates
        if updates['actions']:
            print(f"🔔 Order updates for {mint_key[:8]}...")
            for action in updates['actions']:
                print(f"  - {action['type']}: {action.get('reason', '')}")
        
        if updates['should_exit']:
            print(f"🚪 EXIT SIGNAL: {mint_key[:8]}... | Reason: {updates['reason']}")
        
        return {
            "status": "success",
            "mint_key": mint_key,
            "current_price": current_price,
            "current_gain_pct": updates['current_gain_pct'],
            "should_exit": updates['should_exit'],
            "exit_reason": updates['reason'],
            "new_stop_loss": updates['new_stop_loss'],
            "actions": updates['actions'],
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error updating orders: {str(e)}")


@app.get("/orders/active")
async def get_active_orders():
    """
    📋 Get all active orders being managed
    
    Returns list of all active positions with their current parameters
    """
    try:
        active_orders = {}
        
        for token_id, orders in order_manager.active_orders.items():
            active_orders[token_id] = order_manager.get_order_summary(orders)
        
        return {
            "status": "success",
            "count": len(active_orders),
            "orders": active_orders,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting orders: {str(e)}")


@app.delete("/orders/{mint_key}")
async def close_order(mint_key: str, reason: str = Query("manual_close", description="Close reason")):
    """
    ❌ Close an active order
    
    Remove order from active management (e.g., after exit)
    """
    if mint_key not in order_manager.active_orders:
        raise HTTPException(status_code=404, detail=f"No active orders for {mint_key}")
    
    try:
        # Remove from active
        orders = order_manager.active_orders.pop(mint_key)
        
        print(f"❌ Order closed: {mint_key[:8]}... | Reason: {reason}")
        
        return {
            "status": "success",
            "message": "Order closed",
            "mint_key": mint_key,
            "reason": reason,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error closing order: {str(e)}")


# ============================================================================
# SIGNAL QUERY ENDPOINTS (for retrieving data)
# ============================================================================

@app.get("/signals/tokens")
async def get_token_signals(
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    min_mc: Optional[float] = Query(None, description="Minimum market cap"),
    max_mc: Optional[float] = Query(None, description="Maximum market cap"),
    min_liquidity: Optional[float] = Query(None, description="Minimum liquidity"),
    security: Optional[str] = Query(None, description="Security status (✅, ⚠️, 🚨)"),
    source: Optional[str] = Query(None, description="Signal source"),
    from_date: Optional[datetime] = Query(None, description="Start date"),
    to_date: Optional[datetime] = Query(None, description="End date")
):
    """
    Query token signals with filters
    
    Returns paginated list of token signals matching criteria.
    """
    try:
        filters = {
            'limit': limit,
            'offset': offset,
            'min_mc': min_mc,
            'max_mc': max_mc,
            'min_liquidity': min_liquidity,
            'security': security,
            'source': source,
            'from_date': from_date,
            'to_date': to_date
        }
        
        signals = db.query_signals(filters)
        
        return {
            "status": "success",
            "count": len(signals),
            "limit": limit,
            "offset": offset,
            "signals": signals,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying signals: {str(e)}")


@app.get("/signals/tokens/{mint_key}")
async def get_token_signal(mint_key: str):
    """
    Get latest signal for a specific token
    
    Returns the most recent signal data for the given mint address.
    """
    signal = db.get_signal_by_mint(mint_key)
    
    if not signal:
        raise HTTPException(status_code=404, detail=f"No signal found for mint {mint_key}")
    
    return {
        "status": "success",
        "signal": signal,
        "timestamp": datetime.utcnow()
    }


@app.get("/signals/tokens/latest")
async def get_latest_signals(limit: int = Query(10, ge=1, le=100)):
    """Get latest token signals"""
    signals = db.get_latest_signals(limit)
    
    return {
        "status": "success",
        "count": len(signals),
        "signals": signals,
        "timestamp": datetime.utcnow()
    }


@app.get("/signals/entries")
async def get_entry_signals(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    Query entry signals with gain tracking
    
    Returns list of trade entries and their performance.
    """
    try:
        entries = db.query_entry_signals(limit, offset)
        
        return {
            "status": "success",
            "count": len(entries),
            "limit": limit,
            "offset": offset,
            "entries": entries,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying entries: {str(e)}")


# ============================================================================
# TOKEN SCORING ENDPOINTS (ML predictions)
# ============================================================================

@app.post("/predict", response_model=TokenScoreResponse)
async def predict_token(signal: TokenSignal):
    """
    🚀 RECOMMENDED: Single-request prediction endpoint
    
    Send your signal data and get instant prediction with trading recommendations.
    No need to store the signal first - this is a one-step process!
    
    **NEW**: Integrated Trace_24H registration! 
    - Include `trace_trade_id` and `entry_price` to automatically register the trade
    - System will track outcome and improve model over time
    
    Returns:
    - Predicted gain and confidence
    - GO/NO-GO decision
    - Recommended TP levels (with sell amounts)
    - Recommended SL
    - Position sizing
    - Trading notes
    - Trace_24H registration confirmation (if provided)
    
    Example with Trace_24H:
    ```python
    import requests
    
    signal = {
        "mint_key": "8K4VbtJ6...",
        "signal_at": "2025-12-01T10:00:00Z",
        "signal_mc": 100000,
        "signal_liquidity": 30000,
        "signal_volume_1h": 50000,
        "signal_holders": 500,
        "signal_security": "✅",
        
        # Trace_24H integration (optional)
        "trace_trade_id": "12345",  # From Trace_24H system
        "entry_price": 0.000123     # Entry price
    }
    
    response = requests.post("http://your-api:8000/predict", json=signal)
    prediction = response.json()
    
    if prediction['go_decision']:
        print(f"GO! Expected gain: {prediction['predicted_gain_pct']:.1f}%")
        print(f"Trace registered: {prediction['trace_registered']}")
        print(f"Trade ID: {prediction['trace_trade_id']}")
    ```
    """
    if scorer is None or not scorer.trained:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is trained and available."
        )
    
    try:
        # Convert signal to dict
        token_data = signal.model_dump(exclude_none=True)
        
        # Convert to DataFrame and add derived features
        import pandas as pd
        df = pd.DataFrame([token_data])
        df = feature_extractor.extract_numeric_features(df)
        df = feature_extractor.extract_categorical_features(df)
        df = feature_extractor.add_derived_features(df)
        token_data = df.iloc[0].to_dict()
        
        # Ensure all model features exist with defaults
        for feature in scorer.feature_names:
            if feature not in token_data or pd.isna(token_data[feature]):
                token_data[feature] = 0
        
        # Get prediction
        predicted_gain, confidence = scorer.score_token(token_data)
        
        # Get trading recommendations
        recommendations = scorer.recommend_parameters(token_data)
        
        # Convert TP levels to TakeProfitLevel objects
        tp_levels = [
            TakeProfitLevel(
                gain_pct=tp['gain_pct'],
                sell_amount_pct=tp['sell_amount_pct']
            )
            for tp in recommendations['recommended_tp_levels']
        ]
        
        # Log prediction
        go_status = "GO ✓" if recommendations['go_decision'] else "SKIP ✗"
        print(f"🎯 Prediction: {signal.mint_key[:8]}... | {go_status} | Gain: {predicted_gain*100:.1f}% | Confidence: {confidence:.2f}")
        
        # Trace_24H integration: Auto-register if trade_id provided
        trace_registered = False
        trace_trade_id = None
        
        if signal.trace_trade_id and recommendations['go_decision']:
            try:
                # Register with Trace_24H tracking
                trace_data = {
                    'trade_id': signal.trace_trade_id,
                    'mint_key': signal.mint_key,
                    'signal_at': signal.signal_at,
                    'entry_time': datetime.utcnow(),
                    'entry_price': signal.entry_price,
                    'entry_mc': signal.signal_mc,
                    'predicted_gain': predicted_gain,
                    'predicted_confidence': confidence
                }
                
                db.insert_trace_trade(trace_data)
                trace_registered = True
                trace_trade_id = signal.trace_trade_id
                
                print(f"✓ Trace registered: {signal.trace_trade_id} | {signal.mint_key[:8]}... | Predicted: {predicted_gain*100:.1f}%")
                
            except Exception as e:
                print(f"⚠️  Failed to register Trace trade: {e}")
                # Don't fail the prediction if trace registration fails
        
        return TokenScoreResponse(
            mint_key=signal.mint_key,
            predicted_gain=predicted_gain,
            predicted_gain_pct=predicted_gain * 100,
            confidence=confidence,
            risk_adjusted_score=predicted_gain * confidence,
            go_decision=recommendations['go_decision'],
            recommended_tp_levels=tp_levels,
            recommended_sl=recommendations['recommended_sl'],
            position_size_factor=recommendations['position_size_factor'],
            notes=recommendations['notes'],
            trace_registered=trace_registered,
            trace_trade_id=trace_trade_id
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error predicting token: {str(e)}")


@app.post("/tokens/score", response_model=TokenScoreResponse)
async def score_token(request: TokenScoreRequest):
    """
    Score a token using the trained ML model (legacy endpoint)
    
    ⚠️ For better experience, use /predict endpoint instead!
    
    Provides predicted gain, confidence, and trading recommendations.
    """
    if scorer is None or not scorer.trained:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is trained and available."
        )
    
    try:
        # Prepare token data
        token_data = request.model_dump(exclude_none=True)
        
        # Convert to DataFrame and add derived features
        import pandas as pd
        df = pd.DataFrame([token_data])
        df = feature_extractor.extract_numeric_features(df)
        df = feature_extractor.extract_categorical_features(df)
        df = feature_extractor.add_derived_features(df)
        token_data = df.iloc[0].to_dict()
        
        # Ensure all model features exist with defaults
        for feature in scorer.feature_names:
            if feature not in token_data or pd.isna(token_data[feature]):
                token_data[feature] = 0
        
        # Get prediction
        predicted_gain, confidence = scorer.score_token(token_data)
        
        # Get trading recommendations
        recommendations = scorer.recommend_parameters(token_data)
        
        # Convert TP levels to TakeProfitLevel objects
        tp_levels = [
            TakeProfitLevel(
                gain_pct=tp['gain_pct'],
                sell_amount_pct=tp['sell_amount_pct']
            )
            for tp in recommendations['recommended_tp_levels']
        ]
        
        # Cache the score
        if request.mint_key:
            db.cache_token_score(
                request.mint_key,
                predicted_gain,
                confidence,
                recommendations['predicted_gain'] * confidence,
                token_data
            )
        
        return TokenScoreResponse(
            mint_key=request.mint_key,
            predicted_gain=predicted_gain,
            predicted_gain_pct=predicted_gain * 100,
            confidence=confidence,
            risk_adjusted_score=predicted_gain * confidence,
            go_decision=recommendations['go_decision'],
            recommended_tp_levels=tp_levels,
            recommended_sl=recommendations['recommended_sl'],
            position_size_factor=recommendations['position_size_factor'],
            notes=recommendations['notes']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring token: {str(e)}")


@app.post("/tokens/score/{mint_key}", response_model=TokenScoreResponse)
async def score_token_by_mint(mint_key: str):
    """
    Score a token by its mint address
    
    Looks up the latest signal data and scores it.
    """
    if scorer is None or not scorer.trained:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get latest signal for this mint
    signal = db.get_signal_by_mint(mint_key)
    
    if not signal:
        raise HTTPException(status_code=404, detail=f"No signal found for mint {mint_key}")
    
    try:
        # Convert to DataFrame and add derived features
        import pandas as pd
        df = pd.DataFrame([signal])
        df = feature_extractor.extract_numeric_features(df)
        df = feature_extractor.extract_categorical_features(df)
        df = feature_extractor.add_derived_features(df)
        signal_data = df.iloc[0].to_dict()
        
        # Score the token
        predicted_gain, confidence = scorer.score_token(signal_data)
        recommendations = scorer.recommend_parameters(signal_data)
        
        # Convert TP levels to TakeProfitLevel objects
        tp_levels = [
            TakeProfitLevel(
                gain_pct=tp['gain_pct'],
                sell_amount_pct=tp['sell_amount_pct']
            )
            for tp in recommendations['recommended_tp_levels']
        ]
        
        return TokenScoreResponse(
            mint_key=mint_key,
            predicted_gain=predicted_gain,
            predicted_gain_pct=predicted_gain * 100,
            confidence=confidence,
            risk_adjusted_score=predicted_gain * confidence,
            go_decision=recommendations['go_decision'],
            recommended_tp_levels=tp_levels,
            recommended_sl=recommendations['recommended_sl'],
            position_size_factor=recommendations['position_size_factor'],
            notes=recommendations['notes']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring token: {str(e)}")


@app.get("/tokens/rank")
async def rank_tokens(
    top_n: int = Query(20, ge=1, le=100, description="Number of top tokens to return"),
    sort_by: str = Query("signal_mc", description="Sort by field (signal_mc, signal_liquidity, signal_volume_1h)")
):
    """
    Get ranked list of tokens from CSV data
    
    Returns top N tokens sorted by selected metric (no ML required).
    """
    try:
        # Get recent signals
        signals = db.get_latest_signals(limit=top_n * 2)
        
        if not signals:
            return {
                "status": "success",
                "total_tokens": 0,
                "top_tokens": [],
                "timestamp": datetime.utcnow()
            }
        
        # Convert to list and sort
        import pandas as pd
        df = pd.DataFrame(signals)
        
        # Add derived features using FeatureExtractor
        df = feature_extractor.extract_numeric_features(df)
        df = feature_extractor.extract_categorical_features(df)
        df = feature_extractor.add_derived_features(df)
        
        # Sort by the selected field (descending)
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=False)
        
        # Get top N
        top_tokens = df.head(top_n)
        
        # Select key fields
        result_cols = ['mint_key', 'signal_mc', 'signal_liquidity', 'signal_volume_1h', 
                      'signal_holders', 'signal_security', 'signal_at']
        result_cols = [col for col in result_cols if col in top_tokens.columns]
        
        result = top_tokens[result_cols].to_dict('records')
        
        return {
            "status": "success",
            "total_tokens": len(signals),
            "top_tokens": result,
            "sorted_by": sort_by,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ranking tokens: {str(e)}")


# ============================================================================
# WEBHOOK ENDPOINT (for streaming data)
# ============================================================================

@app.post("/webhook/signal")
async def webhook_receive_signal(signal: TokenSignal, background_tasks: BackgroundTasks):
    """
    Webhook endpoint for real-time signal streaming
    
    Your team can configure their system to POST signals here in real-time.
    Signals are stored and optionally scored in the background.
    """
    try:
        # Store signal immediately
        signal_dict = signal.model_dump()
        row_id = db.insert_token_signal(signal_dict)
        
        # Log webhook signal
        print(f"⚡ Webhook signal: {signal.mint_key[:8]}... | MC: {signal.signal_mc} | Source: {signal.signal_source}")
        
        # Optionally score in background
        if scorer and scorer.trained:
            background_tasks.add_task(
                score_and_cache,
                signal.mint_key,
                signal_dict
            )
        
        return {
            "status": "success",
            "message": "Signal received and stored",
            "id": row_id,
            "mint_key": signal.mint_key,
            "will_score": scorer is not None,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        print(f"✗ Webhook error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")


# ============================================================================
# TRACE_24H INTEGRATION ENDPOINTS
# ============================================================================

@app.post("/trace/register", status_code=201)
async def register_trace_trade(trade: TraceTradeRegistration):
    """
    Register a new Trace_24H trade
    
    When Trace_24H system buys a token based on our /predict endpoint,
    it should call this endpoint to register the trade for tracking.
    
    This allows us to:
    - Track which predictions result in actual trades
    - Automatically fetch results after 24 hours
    - Analyze prediction accuracy
    - Retrain model with ground-truth outcomes
    
    Example:
    ```python
    {
        "trade_id": "12345",
        "mint_key": "8K4VbtJ6...",
        "entry_time": "2025-12-01T10:00:00Z",
        "entry_price": 0.000123,
        "entry_mc": 100000,
        "predicted_gain": 0.85,
        "predicted_confidence": 0.72
    }
    ```
    """
    try:
        trade_dict = trade.model_dump()
        row_id = db.insert_trace_trade(trade_dict)
        
        print(f"✓ Trace trade registered: {trade.trade_id} | {trade.mint_key[:8]}... | "
              f"Entry MC: ${trade.entry_mc:,.0f} | Predicted: {trade.predicted_gain*100:.1f}%")
        
        return {
            "status": "success",
            "message": "Trace trade registered",
            "id": row_id,
            "trade_id": trade.trade_id,
            "mint_key": trade.mint_key,
            "will_auto_fetch": True,
            "fetch_after": "24 hours",
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        print(f"✗ Error registering trace trade: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error registering trade: {str(e)}")


@app.post("/trace/fetch/{trade_id}")
async def fetch_trace_trade(trade_id: str, background_tasks: BackgroundTasks):
    """
    Fetch completed Trace_24H trade data
    
    Fetches the full price-tick history from Trace_24H system,
    analyzes the trade, and stores results for model retraining.
    
    This is typically called automatically 24 hours after trade entry,
    but can also be called manually.
    
    Returns:
    - Trade outcome metrics
    - Prediction accuracy
    - Stored in database for retraining
    """
    try:
        # Check if trade exists in database
        trade_record = db.get_trace_trade_by_id(trade_id)
        
        if not trade_record:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found. Register it first with /trace/register")
        
        # Fetch trade data in background
        background_tasks.add_task(
            fetch_and_process_trace_trade,
            trade_id,
            trade_record
        )
        
        return {
            "status": "processing",
            "message": "Trade data fetch started in background",
            "trade_id": trade_id,
            "mint_key": trade_record.get('mint_key'),
            "timestamp": datetime.utcnow()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error fetching trace trade: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching trade: {str(e)}")


@app.get("/trace/pending")
async def get_pending_trace_trades(limit: int = Query(100, ge=1, le=500)):
    """
    Get Trace_24H trades that need to be fetched
    
    Returns trades that:
    - Are still in 'open' status
    - Are older than 24 hours (ready for fetching)
    
    Use this to identify trades that need data collection.
    """
    try:
        pending_trades = db.get_pending_trace_trades(limit)
        
        return {
            "status": "success",
            "count": len(pending_trades),
            "trades": pending_trades,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting pending trades: {str(e)}")


@app.post("/trace/fetch-all-pending")
async def fetch_all_pending_trades(background_tasks: BackgroundTasks):
    """
    Fetch all pending Trace_24H trades
    
    Automatically fetches data for all trades that are:
    - Older than 24 hours
    - Still in 'open' status
    
    This is useful for batch processing or catching up on missed trades.
    """
    try:
        pending_trades = db.get_pending_trace_trades(limit=100)
        
        if not pending_trades:
            return {
                "status": "success",
                "message": "No pending trades to fetch",
                "count": 0,
                "timestamp": datetime.utcnow()
            }
        
        # Fetch all in background
        background_tasks.add_task(
            fetch_multiple_trace_trades,
            pending_trades
        )
        
        return {
            "status": "processing",
            "message": f"Fetching {len(pending_trades)} pending trades in background",
            "count": len(pending_trades),
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        print(f"✗ Error fetching pending trades: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/trace/{trade_id}", response_model=TraceTradeResponse)
async def get_trace_trade(trade_id: str):
    """
    Get details of a specific Trace_24H trade
    
    Returns:
    - Trade entry/exit details
    - Prediction vs actual outcome
    - Prediction accuracy metrics
    """
    try:
        trade = db.get_trace_trade_by_id(trade_id)
        
        if not trade:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
        
        # Calculate prediction error if available
        prediction_error = None
        if trade.get('predicted_gain') is not None and trade.get('max_return') is not None:
            prediction_error = abs(trade['predicted_gain'] - trade['max_return'])
        
        return TraceTradeResponse(
            trade_id=trade['trade_id'],
            mint_key=trade['mint_key'],
            entry_time=trade['entry_time'],
            entry_mc=trade['entry_mc'],
            exit_time=trade.get('exit_time'),
            exit_mc=trade.get('exit_mc'),
            final_gain=trade.get('final_gain'),
            max_return=trade.get('max_return'),
            max_return_mc=trade.get('max_return_mc'),
            max_return_timestamp=trade.get('max_return_timestamp'),
            predicted_gain=trade.get('predicted_gain'),
            predicted_confidence=trade.get('predicted_confidence'),
            prediction_error=prediction_error,
            status=trade['status'],
            price_updates_count=trade.get('price_updates_count')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting trade: {str(e)}")


@app.get("/trace/stats", response_model=TraceTradeStats)
async def get_trace_trade_stats():
    """
    Get comprehensive statistics on Trace_24H trades
    
    Shows:
    - Total trades tracked
    - Win rate and performance metrics
    - Prediction accuracy
    - Distribution of outcomes
    """
    try:
        stats = db.get_trace_trade_statistics()
        
        # Calculate average prediction error if we have data
        # This would require querying completed trades with predictions
        avg_prediction_error = None
        
        return TraceTradeStats(
            total_trades=stats['total_trades'],
            open_trades=stats['open_trades'],
            completed_trades=stats['completed_trades'],
            avg_final_gain=stats['avg_final_gain'],
            avg_max_return=stats['avg_max_return'],
            best_max_return=stats['best_max_return'],
            win_rate=stats['win_rate'],
            winners=stats['winners'],
            losers=stats['losers'],
            avg_prediction_error=avg_prediction_error
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.get("/trace/analysis/accuracy")
async def get_prediction_accuracy_analysis():
    """
    Get detailed prediction accuracy analysis
    
    Compares model predictions against actual Trace_24H outcomes.
    
    Returns:
    - Overall accuracy metrics
    - Error patterns by token characteristics
    - Recommendations for model improvement
    """
    try:
        report = trace_fetcher.get_prediction_accuracy_report()
        
        # Get improvement recommendations
        recommendations = prediction_analyzer.generate_improvement_recommendations()
        
        # Get pattern analysis
        patterns = prediction_analyzer.identify_prediction_patterns()
        
        return {
            "status": "success",
            "accuracy_report": report,
            "patterns": patterns,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        print(f"✗ Error generating analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Background tasks for Trace_24H processing

async def fetch_and_process_trace_trade(trade_id: str, trade_record: Dict):
    """Background task to fetch and process a single Trace_24H trade"""
    try:
        print(f"\n🔄 Fetching Trace_24H trade: {trade_id}")
        
        # Get prediction data from record
        prediction_data = None
        if trade_record.get('predicted_gain') is not None:
            prediction_data = {
                'predicted_gain': trade_record['predicted_gain'],
                'confidence': trade_record.get('predicted_confidence')
            }
        
        # Fetch and process
        result = trace_fetcher.process_trade(trade_id, prediction_data)
        
        if result:
            # Update database with results
            metrics = result['calculated_metrics']
            
            db.update_trace_trade_completion(
                trade_id,
                result.get('exit_time'),
                result.get('exit_price'),
                metrics.get('final_mc'),
                metrics.get('final_return'),
                metrics.get('max_return'),
                metrics.get('max_return_mc'),
                metrics.get('max_return_timestamp'),
                metrics.get('total_updates', 0)
            )
            
            print(f"✓ Trace trade processed: {trade_id}")
            
            # Trigger retraining check
            check_retraining_trigger()
        else:
            print(f"✗ Failed to process trace trade: {trade_id}")
    
    except Exception as e:
        print(f"✗ Error processing trace trade {trade_id}: {e}")
        import traceback
        traceback.print_exc()


async def fetch_multiple_trace_trades(trade_records: List[Dict]):
    """Background task to fetch multiple Trace_24H trades"""
    try:
        print(f"\n🔄 Fetching {len(trade_records)} Trace_24H trades...")
        
        trade_ids = [t['trade_id'] for t in trade_records]
        
        # Process with rate limiting
        results = trace_fetcher.process_multiple_trades(trade_ids, delay_seconds=1.0)
        
        # Update all in database
        for result in results:
            try:
                trade_id = result.get('trade_id')
                metrics = result['calculated_metrics']
                
                db.update_trace_trade_completion(
                    trade_id,
                    result.get('exit_time'),
                    result.get('exit_price'),
                    metrics.get('final_mc'),
                    metrics.get('final_return'),
                    metrics.get('max_return'),
                    metrics.get('max_return_mc'),
                    metrics.get('max_return_timestamp'),
                    metrics.get('total_updates', 0)
                )
            except Exception as e:
                print(f"✗ Error updating trade {trade_id}: {e}")
        
        print(f"✓ Processed {len(results)} / {len(trade_ids)} trades")
        
        # Trigger retraining check
        if results:
            check_retraining_trigger()
    
    except Exception as e:
        print(f"✗ Error processing multiple trades: {e}")
        import traceback
        traceback.print_exc()


async def score_and_cache(mint_key: str, signal_data: dict):
    """Background task to score and cache token"""
    try:
        # Convert to DataFrame and add derived features
        import pandas as pd
        df = pd.DataFrame([signal_data])
        df = feature_extractor.extract_numeric_features(df)
        df = feature_extractor.extract_categorical_features(df)
        df = feature_extractor.add_derived_features(df)
        enriched_data = df.iloc[0].to_dict()
        
        predicted_gain, confidence = scorer.score_token(enriched_data)
        score = predicted_gain * confidence
        
        db.cache_token_score(
            mint_key,
            predicted_gain,
            confidence,
            score,
            enriched_data
        )
    except Exception as e:
        print(f"Error scoring token {mint_key}: {e}")


def check_retraining_trigger():
    """Background task to check if retraining should be triggered"""
    try:
        result = auto_retrainer.check_and_retrain()
        
        if result['status'] == 'success':
            print(f"\n🎉 Automatic retraining completed!")
            print(f"   Version: {result['version']}")
            print(f"   Test R²: {result['metrics']['test_r2']:.3f}")
            print(f"   Training samples: {result['training_samples']}")
            print(f"   ⚠️  Restart API to load new model: ./stop_production.sh && ./deploy_production.sh\n")
            
        elif result['status'] in ['not_needed', 'insufficient_data']:
            # Normal, no action needed
            pass
        elif result['status'] == 'quality_low':
            print(f"⚠️  New model quality too low, keeping current model")
        else:
            print(f"ℹ️  Retraining check: {result.get('message', 'Unknown status')}")
            
    except Exception as e:
        print(f"✗ Retraining check failed: {e}")
        import traceback
        traceback.print_exc()


def run_retraining():
    """Background task to run retraining"""
    try:
        print(f"\n🔄 Manual retraining triggered...")
        result = auto_retrainer.check_and_retrain()
        
        if result['status'] == 'success':
            print(f"\n✅ Manual retraining completed!")
            print(f"   Version: {result['version']}")
            print(f"   Test R²: {result['metrics']['test_r2']:.3f}")
            print(f"   Training samples: {result['training_samples']}")
            print(f"   ⚠️  Restart API to load new model\n")
        else:
            print(f"❌ Retraining failed: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Manual retraining failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class TokenSignal(BaseModel):
    """Token signal data from trading system"""
    mint_key: str = Field(..., description="Unique token identifier")
    signal_at: datetime = Field(..., description="Signal timestamp")
    signal_mc: float = Field(..., description="Market cap at signal")
    signal_liquidity: float = Field(..., description="Liquidity pool size")
    signal_volume_1h: float = Field(..., description="1-hour volume")
    signal_holders: int = Field(..., description="Number of holders")
    signal_age: Optional[str] = Field(None, description="Token age")
    signal_dev_sol: Optional[float] = Field(None, description="Dev SOL balance")
    signal_security: Optional[str] = Field(None, description="Security status")
    signal_bundled_pct: Optional[float] = Field(None, description="Bundled tx %")
    signal_snipers_pct: Optional[float] = Field(None, description="Sniper tx %")
    signal_sold_pct: Optional[float] = Field(None, description="Sold %")
    signal_first_20_pct: Optional[float] = Field(None, description="Top 20 holder %")
    signal_fish_pct: Optional[float] = Field(None, description="Fish holder %")
    signal_fish_count: Optional[int] = Field(None, description="Fish count")
    signal_top_mc: Optional[float] = Field(None, description="Peak MC")
    signal_best_mc: Optional[float] = Field(None, description="Best MC ever")
    signal_source: Optional[str] = Field(None, description="Signal source")
    signal_bond: Optional[int] = Field(None, description="Bond status")
    signal_made: Optional[int] = Field(None, description="Successful exits")
    
    # Trace_24H integration (optional)
    trace_trade_id: Optional[str] = Field(None, description="Trace_24H trade ID (if executing trade)")
    entry_price: Optional[float] = Field(None, description="Entry price (if executing trade)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "mint_key": "8K4VbtJ6cWVxUfxK1DoFP9g2H1V4cWFF2RrQTqEwpump",
                "signal_at": "2025-06-18T09:33:39+00:00",
                "signal_mc": 73305,
                "signal_liquidity": 27600,
                "signal_volume_1h": 10500,
                "signal_holders": 395,
                "signal_bundled_pct": 4.6,
                "signal_sold_pct": 6.3,
                "signal_snipers_pct": 35.49
            }
        }


class EntrySignal(BaseModel):
    """Entry signal with gain tracking"""
    ticker: str = Field(..., description="Token ticker")
    signal_at: datetime = Field(..., description="Entry signal timestamp")
    entry_mc: float = Field(..., description="Entry market cap")
    current_mc: Optional[float] = Field(None, description="Current market cap")
    gain_pct: Optional[float] = Field(None, description="Current gain %")
    source: str = Field(..., description="Signal source (early/whale)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "P4D",
                "signal_at": "2025-09-26T00:27:29Z",
                "entry_mc": 93237,
                "current_mc": 148000,
                "gain_pct": 0.58,
                "source": "early_trending"
            }
        }


class TradeOutcome(BaseModel):
    """Trade outcome for model retraining"""
    mint_key: str = Field(..., description="Token mint key")
    signal_at: datetime = Field(..., description="Signal timestamp")
    entry_price: Optional[float] = Field(None, description="Entry price")
    exit_price: Optional[float] = Field(None, description="Exit price")
    actual_gain: float = Field(..., description="Actual gain percentage (e.g., 0.5 = 50%)")
    max_return: Optional[float] = Field(None, description="Maximum return achieved (e.g., 1.5 = 150%)")
    max_return_mc: Optional[float] = Field(None, description="Market cap at max return")
    max_return_timestamp: Optional[datetime] = Field(None, description="When max return was achieved")
    exit_reason: Optional[str] = Field(None, description="Exit reason (tp1, tp2, tp3, sl, manual)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "mint_key": "8K4VbtJ6cWVxUfxK1DoFP9g2H1V4cWFF2RrQTqEwpump",
                "signal_at": "2025-11-05T18:00:00Z",
                "entry_price": 0.000123,
                "exit_price": 0.000185,
                "actual_gain": 0.50,
                "max_return": 1.50,
                "max_return_mc": 250000,
                "max_return_timestamp": "2025-11-05T19:30:00Z",
                "exit_reason": "tp2"
            }
        }


class TokenScoreRequest(BaseModel):
    """Request to score a token"""
    mint_key: Optional[str] = Field(None, description="Token mint key")
    signal_mc: float
    signal_liquidity: float
    signal_volume_1h: float
    signal_holders: int
    signal_dev_sol: Optional[float] = None
    signal_security: Optional[str] = None
    signal_bundled_pct: Optional[float] = None
    signal_snipers_pct: Optional[float] = None
    signal_sold_pct: Optional[float] = None
    signal_first_20_pct: Optional[float] = None
    liq_to_mc_ratio: Optional[float] = None
    vol_to_liq_ratio: Optional[float] = None
    age_minutes: Optional[float] = None
    risk_score: Optional[float] = None


class TakeProfitLevel(BaseModel):
    """Take-profit level with gain and sell amount"""
    gain_pct: float = Field(..., description="Gain percentage at this level")
    sell_amount_pct: float = Field(..., description="Percentage of remaining position to sell")


class TokenScoreResponse(BaseModel):
    """Response with token score"""
    mint_key: Optional[str]
    predicted_gain: float = Field(..., description="Predicted gain (decimal)")
    predicted_gain_pct: float = Field(..., description="Predicted gain (%)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    risk_adjusted_score: float = Field(..., description="Risk-adjusted score")
    go_decision: bool = Field(..., description="Trade or skip")
    recommended_tp_levels: List[TakeProfitLevel] = Field(..., description="Take-profit levels with sell amounts")
    recommended_sl: float = Field(..., description="Stop-loss level (negative value, e.g., -0.55 for -55%)")
    position_size_factor: float = Field(..., description="Position sizing as % of wallet (0.05-0.10 for 5-10%)")
    notes: str = Field(..., description="Trading notes")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Trace_24H integration (if trade registered)
    trace_registered: bool = Field(default=False, description="Whether trade was registered with Trace_24H")
    trace_trade_id: Optional[str] = Field(None, description="Trace_24H trade ID")


class TokenRankingResponse(BaseModel):
    """Response with ranked tokens"""
    total_tokens: int
    top_tokens: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BulkSignalRequest(BaseModel):
    """Bulk signal upload"""
    signals: List[TokenSignal]


class SignalQueryParams(BaseModel):
    """Query parameters for signals"""
    limit: Optional[int] = Field(100, description="Max results", ge=1, le=1000)
    offset: Optional[int] = Field(0, description="Offset for pagination", ge=0)
    min_mc: Optional[float] = Field(None, description="Minimum market cap")
    max_mc: Optional[float] = Field(None, description="Maximum market cap")
    min_liquidity: Optional[float] = Field(None, description="Minimum liquidity")
    security: Optional[str] = Field(None, description="Security status filter")
    source: Optional[str] = Field(None, description="Signal source filter")
    from_date: Optional[datetime] = Field(None, description="Start date")
    to_date: Optional[datetime] = Field(None, description="End date")


class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_loaded: bool
    database_connected: bool
    total_signals: int


class StatsResponse(BaseModel):
    """Statistics response"""
    total_signals: int
    total_entry_signals: int
    date_range: Dict[str, Optional[str]]
    avg_gain: Optional[float]
    median_gain: Optional[float]
    top_performers: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RetrainingStatus(BaseModel):
    """Model retraining status"""
    retraining_active: bool
    new_outcomes_count: int
    total_training_samples: int
    last_retrain_at: Optional[datetime]
    next_retrain_trigger: int
    current_model_version: Optional[str]
    current_model_accuracy: Optional[float]
    current_model_win_rate: Optional[float]


class MaxReturnUpdate(BaseModel):
    """Update max return for a signal"""
    mint_key: str = Field(..., description="Token mint key")
    signal_at: datetime = Field(..., description="Signal timestamp")
    max_return: float = Field(..., description="Maximum return as decimal (e.g., 1.5 = 150%)")
    max_return_mc: float = Field(..., description="Market cap at max return")
    max_return_timestamp: datetime = Field(..., description="When max return occurred")
    
    class Config:
        json_schema_extra = {
            "example": {
                "mint_key": "8K4VbtJ6cWVxUfxK1DoFP9g2H1V4cWFF2RrQTqEwpump",
                "signal_at": "2025-11-05T18:00:00Z",
                "max_return": 2.45,
                "max_return_mc": 350000,
                "max_return_timestamp": "2025-11-05T20:15:00Z"
            }
        }


class PriceUpdate(BaseModel):
    """Update current price for tracking"""
    mint_key: str = Field(..., description="Token mint key")
    signal_at: datetime = Field(..., description="Signal timestamp")
    current_price: float = Field(..., description="Current price")
    current_mc: float = Field(..., description="Current market cap")
    
    class Config:
        json_schema_extra = {
            "example": {
                "mint_key": "8K4VbtJ6cWVxUfxK1DoFP9g2H1V4cWFF2RrQTqEwpump",
                "signal_at": "2025-11-05T18:00:00Z",
                "current_price": 0.00025,
                "current_mc": 180000
            }
        }


class MaxReturnStats(BaseModel):
    """Max return statistics"""
    total_tracked: int
    with_max_return: int
    avg_max_return: Optional[float]
    best_max_return: Optional[float]
    worst_max_return: Optional[float]
    above_50_pct: int
    above_100_pct: int
    above_200_pct: int
    top_performers: List[Dict[str, Any]]


class TraceTradeRegistration(BaseModel):
    """Register a new Trace_24H trade"""
    trade_id: str = Field(..., description="Unique trade ID from Trace_24H system")
    mint_key: str = Field(..., description="Token mint key")
    entry_time: datetime = Field(..., description="Entry timestamp")
    entry_price: Optional[float] = Field(None, description="Entry price")
    entry_mc: Optional[float] = Field(None, description="Entry market cap")
    predicted_gain: Optional[float] = Field(None, description="Model's predicted gain")
    predicted_confidence: Optional[float] = Field(None, description="Model's confidence")
    
    class Config:
        json_schema_extra = {
            "example": {
                "trade_id": "12345",
                "mint_key": "8K4VbtJ6cWVxUfxK1DoFP9g2H1V4cWFF2RrQTqEwpump",
                "entry_time": "2025-12-01T10:00:00Z",
                "entry_price": 0.000123,
                "entry_mc": 100000,
                "predicted_gain": 0.85,
                "predicted_confidence": 0.72
            }
        }


class TraceTradeFetch(BaseModel):
    """Request to fetch completed Trace_24H trade data"""
    trade_id: str = Field(..., description="Trade ID to fetch")


class TraceTradeResponse(BaseModel):
    """Response with Trace_24H trade data"""
    trade_id: str
    mint_key: str
    entry_time: datetime
    entry_mc: float
    exit_time: Optional[datetime]
    exit_mc: Optional[float]
    final_gain: Optional[float]
    max_return: Optional[float]
    max_return_mc: Optional[float]
    max_return_timestamp: Optional[datetime]
    predicted_gain: Optional[float]
    predicted_confidence: Optional[float]
    prediction_error: Optional[float]
    status: str
    price_updates_count: Optional[int]


class TraceTradeStats(BaseModel):
    """Statistics on Trace_24H trades"""
    total_trades: int
    open_trades: int
    completed_trades: int
    avg_final_gain: Optional[float]
    avg_max_return: Optional[float]
    best_max_return: Optional[float]
    win_rate: Optional[float]
    winners: int
    losers: int
    avg_prediction_error: Optional[float]

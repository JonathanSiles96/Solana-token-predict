"""
Trading Strategy Configuration

This file defines the trading strategy rules based on:
- Risk Management
- Entry Strategy  
- Exit Strategy
- Portfolio Management

All thresholds and rules are configurable here.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class SignalSource(Enum):
    """Signal sources with priority levels"""
    PRIMAL = "primal"           # Highest priority (71% confidence)
    WHALE = "whale"             # Good for liquidity insights
    SOLANA_TRACKER = "solana_tracker"
    TELEGRAM_EARLY = "telegram_early"  # Speculative, secondary
    EARLY_TRENDING = "early_trending"
    WHALE_TRENDING = "whale_trending"
    UNKNOWN = "unknown"


@dataclass
class RiskManagementConfig:
    """Risk management rules - UPDATED Dec 24"""
    
    # Position sizing
    min_position_pct: float = 0.05      # 5% minimum
    max_position_pct: float = 0.10      # 10% maximum
    high_risk_position_pct: float = 0.05  # Reduce for risky trades
    
    # Stop-loss levels - UPDATED Dec 24
    # CRITICAL: Tighter SLs to reduce drawdown on losers
    # Data shows losers dump fast - cut losses early
    default_sl: float = -0.20           # -20% default (was -28%)
    tight_sl: float = -0.15             # -15% for lower tiers (was -22%)
    loose_sl: float = -0.22             # -22% for TIER0/TIER1 (was -30%)
    
    # Risk thresholds that trigger tighter SL
    high_risk_bundled_pct: float = 50.0  # Raised to allow more bundled
    high_risk_snipers_pct: float = 50.0  # Raised from 35%
    high_risk_sold_pct: float = 40.0     # Raised from 20%
    
    # Maximum concurrent positions
    max_active_trades: int = 10          # Raised from 7 to capture more opps
    
    # Diversification: max % of capital in single source
    max_per_source_pct: float = 0.50    # 50% max in one source (was 40%)


@dataclass
class EntryConfig:
    """Entry strategy rules - CORRECTED Dec 16 based on max_return data"""
    
    # CORRECTED Analysis of 1473 signals with max_return data:
    # 
    # Tokens that NEVER go positive (max_return < 1.0):
    # - primal: 3.2% (BEST - 97% reach break-even)
    # - solana_tracker: 2.3% (BEST - 98% reach break-even)
    # - tg_early_trending: 38.9% (BAD!)
    # - whale: 41.0% (BAD!)
    #
    # Tokens reaching 1.3x+ profit:
    # - solana_tracker: 70.9%
    # - primal: 69.5%
    # - tg_early_trending: 61.1%
    # - whale: 59.0%
    
    # Minimum thresholds - UPDATED Dec 24
    # CRITICAL: Let tier logic in token_scorer.py handle GO/SKIP
    # These are just absolute minimums - very permissive
    min_confidence: float = 0.0         # No min - let tier logic decide
    min_risk_adjusted_score: float = 0.0  # Let tier logic decide
    min_volume_1h: float = 0            # No min - tier logic handles this
    min_holders: int = 0                # No min - tier logic handles this
    min_liquidity: float = 0            # No min
    min_mc: float = 0                   # No min
    
    # Red flags - VERY PERMISSIVE Dec 24
    # Data shows high bundled/snipers tokens still hit 2-10x!
    # Better to trade and manage with TP/SL than miss opportunities
    max_bundled_pct: float = 100.0      # No block - just warn
    max_sold_pct: float = 100.0         # No block - just warn
    max_snipers_pct: float = 100.0      # No block - just warn
    
    # Warning thresholds - just for logging, DON'T BLOCK
    warn_bundled_pct: float = 70.0      # Warning if bundled > 70%
    warn_sold_pct: float = 50.0         # Warning if sold > 50%
    warn_snipers_pct: float = 60.0      # Warning if snipers > 60%
    
    # Entry timing
    max_initial_pump_pct: float = 40.0  # Avoid if already pumped > 40% (was 50%)
    ideal_entry_pump_range: tuple = (5.0, 25.0)  # Enter after 5-25% pump
    
    # Liquidity ratio
    min_liq_to_mc_ratio: float = 0.20   # Minimum 20% liq/mc (was 15%)
    
    # Security status filter - DATA DRIVEN
    # ✅: 93.7% win rate, 🚨: 86.5% win rate, ⚠️: 74.3% win rate
    # Surprisingly 🚨 performs well! Allow all except "danger" (60.3%)
    require_green_security: bool = False  # Don't require green - data shows 🚨 is good!
    allowed_security_statuses: List[str] = field(default_factory=lambda: ["✅", "white_check_mark", "⚠️", "warning", "🚨"])
    
    # Token age filter (in minutes) - PERMISSIVE
    min_token_age: int = 0              # Allow new tokens
    max_token_age: int = 1440           # Allow up to 24 hours (was 2 hours)
    
    # First 20 holders concentration - PERMISSIVE
    max_first_20_pct: float = 90.0      # Allow up to 90% (was 50%)
    
    # Signal source priority - CORRECTED Dec 22 based on MAX_RETURN data
    # CRITICAL FIX: Previous logic blocked primal/whale but their tokens hit 1.5-3.2x!
    # The issue was exit timing, not the signals themselves.
    # 
    # NEW APPROACH: Enable all sources, rely on fundamentals & TP/SL
    source_priority: Dict[str, int] = field(default_factory=lambda: {
        "primal": 80,                   # ENABLED - max_return data shows 1.5-3x winners
        "whale": 80,                    # ENABLED - data shows 2x+ potential
        "solana_tracker": 75,           # Good data quality
        "tg_early_trending": 75,        # Good for early entry
        "whale_trending": 60,           # Good whale signals
        "early_trending": 60,           # Early opportunities
        "telegram_early": 50,           # Telegram signals
        "unknown": 30                   # Unknown sources - cautious
    })
    
    # Minimum source priority to trade
    min_source_priority: int = 20       # Allow most sources, filter by other metrics


@dataclass
class ExitConfig:
    """Exit strategy rules - UPDATED Dec 24"""
    
    # Take profit levels - UPDATED Dec 24
    # CRITICAL FIXES:
    # - TP1: Lower to 15% to actually capture profits (was 20%)
    # - TP2: 35% (was 50%)
    # - TP3: 70% (was 100%)
    # - TP4: 150% for runners
    # - Sell more at early TPs to lock in gains
    tp_levels: List[Dict] = field(default_factory=lambda: [
        {"gain_pct": 15, "sell_pct": 35, "label": "TP1"},   # 35% at +15%
        {"gain_pct": 35, "sell_pct": 30, "label": "TP2"},   # 30% at +35%
        {"gain_pct": 70, "sell_pct": 25, "label": "TP3"},   # 25% at +70%
        {"gain_pct": 150, "sell_pct": 10, "label": "TP4"}   # 10% runner at +150%
    ])
    
    # Trailing stop - UPDATED Dec 24
    # Activate earlier, trail tighter to protect gains
    trailing_stop_activation_pct: float = 15.0  # Activate after TP1 (+15% gain)
    trailing_stop_distance_pct: float = 12.0    # Trail 12% behind peak (was 20%)
    
    # Post-TP SL rules (move SL up after hitting TPs)
    post_tp1_sl_pct: float = 0.0    # After TP1, SL moves to break-even
    post_tp2_sl_pct: float = 15.0   # After TP2, SL moves to +15% (guaranteed profit)
    post_tp3_sl_pct: float = 35.0   # After TP3, SL moves to +35%
    
    # Early exit triggers
    low_momentum_threshold: float = 0.1   # Exit if momentum drops below this
    whale_exit_snipers_pct: float = 50.0  # Exit if snipers spike above 50%
    whale_exit_sold_pct: float = 60.0     # Exit if sold spikes above 60%
    
    # Time-based rules
    stagnation_minutes: int = 20          # If no +10% in 20min, tighten SL
    max_hold_minutes: int = 240           # Max hold 4 hours (was 6)


@dataclass
class PortfolioConfig:
    """Portfolio management rules"""
    
    max_active_positions: int = 7        # Max 5-7 coins per hour
    rebalance_interval_minutes: int = 60  # Reassess every hour
    min_trade_interval_seconds: int = 30  # Wait between trades
    
    # Close underperformers
    underperformer_threshold_pct: float = -15.0  # Close if down 15%
    underperformer_time_minutes: int = 30        # After 30 minutes


@dataclass
class TrainingConfig:
    """Model training configuration"""
    
    # Target definition
    # A "successful" trade is one that achieves this return
    success_threshold_pct: float = 30.0   # 30% gain = success
    
    # Training data requirements
    min_training_samples: int = 100
    min_samples_per_class: int = 20
    
    # Feature importance thresholds
    min_feature_importance: float = 0.01
    
    # Model parameters
    model_type: str = "gradient_boosting"
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Retraining triggers
    retrain_after_new_outcomes: int = 50
    min_accuracy_threshold: float = 0.60


@dataclass
class TradingStrategy:
    """Complete trading strategy configuration"""
    
    name: str = "Solana Token Strategy v1"
    version: str = "1.0.0"
    
    risk: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def get_source_priority(self, source: str) -> int:
        """Get priority for a signal source"""
        source_lower = source.lower() if source else "unknown"
        
        # Map common variations
        source_map = {
            "primal": "primal",
            "whale": "whale",
            "whale_trending": "whale_trending",
            "whaletrending": "whale_trending",
            "early_trending": "early_trending",
            "earlytrending": "early_trending",
            "solana_tracker": "solana_tracker",
            "solanatracker": "solana_tracker",
            "telegram": "telegram_early",
            "telegram_early": "telegram_early",
        }
        
        normalized = source_map.get(source_lower, "unknown")
        return self.entry.source_priority.get(normalized, 20)
    
    def is_high_risk(self, signal_data: dict) -> bool:
        """Check if signal is high risk"""
        bundled = signal_data.get("signal_bundled_pct", 0) or 0
        snipers = signal_data.get("signal_snipers_pct", 0) or 0
        sold = signal_data.get("signal_sold_pct", 0) or 0
        
        return (
            bundled > self.risk.high_risk_bundled_pct or
            snipers > self.risk.high_risk_snipers_pct or
            sold > self.risk.high_risk_sold_pct
        )
    
    def passes_entry_filters(self, signal_data: dict, prediction: dict) -> Tuple[bool, list, float]:
        """
        Check if signal passes all entry filters - ULTRA STRICT for 80%+ win rate
        
        Returns:
            (passes: bool, reasons: list, score: float)
        """
        reasons = []
        score = 0.0
        
        # Get values with defaults
        confidence = prediction.get("confidence", 0)
        risk_adj_score = prediction.get("risk_adjusted_score", 0)
        volume_1h = signal_data.get("signal_volume_1h", 0) or 0
        holders = signal_data.get("signal_holders", 0) or 0
        liquidity = signal_data.get("signal_liquidity", 0) or 0
        mc = signal_data.get("signal_mc", 0) or 0
        bundled = signal_data.get("signal_bundled_pct", 0) or 0
        sold = signal_data.get("signal_sold_pct", 0) or 0
        snipers = signal_data.get("signal_snipers_pct", 0) or 0
        security = signal_data.get("signal_security", "") or ""
        first_20_pct = signal_data.get("signal_first_20_pct", 0) or 0
        token_age = signal_data.get("age_minutes", 0) or 0
        source = signal_data.get("signal_source", "unknown") or "unknown"
        
        # Calculate liq ratio
        liq_ratio = liquidity / mc if mc > 0 else 0
        
        # === CRITICAL: SECURITY STATUS FILTER ===
        if self.entry.require_green_security:
            is_green = any(s in str(security) for s in self.entry.allowed_security_statuses)
            if not is_green:
                reasons.append(f"🚨 Security not ✅: '{security}' - Only green security tokens allowed")
                return False, reasons, 0
        
        # === SOURCE PRIORITY FILTER ===
        source_priority = self.get_source_priority(source)
        if source_priority < self.entry.min_source_priority:
            reasons.append(f"🚨 Source priority too low: {source} ({source_priority}) < {self.entry.min_source_priority}")
            return False, reasons, 0
        
        # === HARD REJECTIONS (red flags) - STRICTER THRESHOLDS ===
        
        if bundled > self.entry.max_bundled_pct:
            reasons.append(f"🚨 Bundled too high: {bundled:.1f}% > {self.entry.max_bundled_pct}%")
            return False, reasons, 0
        
        if sold > self.entry.max_sold_pct:
            reasons.append(f"🚨 Sold too high: {sold:.1f}% > {self.entry.max_sold_pct}%")
            return False, reasons, 0
        
        if snipers > self.entry.max_snipers_pct:
            reasons.append(f"🚨 Snipers too high: {snipers:.1f}% > {self.entry.max_snipers_pct}%")
            return False, reasons, 0
        
        # === NEW: First 20 holders concentration ===
        if first_20_pct > self.entry.max_first_20_pct:
            reasons.append(f"🚨 Top 20 hold too much: {first_20_pct:.1f}% > {self.entry.max_first_20_pct}%")
            return False, reasons, 0
        
        # === NEW: Token age filter ===
        if token_age > 0:  # Only apply if we have age data
            if token_age < self.entry.min_token_age:
                reasons.append(f"🚨 Token too new: {token_age:.0f}m < {self.entry.min_token_age}m")
                return False, reasons, 0
            if token_age > self.entry.max_token_age:
                reasons.append(f"🚨 Token too old: {token_age:.0f}m > {self.entry.max_token_age}m")
                return False, reasons, 0
        
        # === MINIMUM REQUIREMENTS ===
        
        passes = True
        
        # Market cap check
        if mc >= self.entry.min_mc:
            score += 10
        else:
            passes = False
            reasons.append(f"❌ Low MC: ${mc:,.0f} < ${self.entry.min_mc:,.0f}")
        
        # Confidence check
        if confidence >= self.entry.min_confidence:
            score += 20
            reasons.append(f"✅ Confidence: {confidence:.2f}")
        else:
            passes = False
            reasons.append(f"❌ Low confidence: {confidence:.2f} < {self.entry.min_confidence}")
        
        # Risk-adjusted score check - CRITICAL
        if risk_adj_score >= self.entry.min_risk_adjusted_score:
            score += 30
            reasons.append(f"✅ Risk-adj score: {risk_adj_score:.2f}")
        else:
            passes = False
            reasons.append(f"❌ risk-adj {risk_adj_score:.2f} < {self.entry.min_risk_adjusted_score}")
        
        # Volume check
        if volume_1h >= self.entry.min_volume_1h:
            score += 15
            reasons.append(f"✅ Volume 1h: ${volume_1h:,.0f}")
        else:
            passes = False
            reasons.append(f"❌ Low volume: ${volume_1h:,.0f} < ${self.entry.min_volume_1h:,.0f}")
        
        # Holders check
        if holders >= self.entry.min_holders:
            score += 15
            reasons.append(f"✅ Holders: {holders}")
        else:
            passes = False
            reasons.append(f"❌ Few holders: {holders} < {self.entry.min_holders}")
        
        # Liquidity check
        if liquidity >= self.entry.min_liquidity:
            score += 10
        else:
            passes = False
            reasons.append(f"❌ Low liquidity: ${liquidity:,.0f} < ${self.entry.min_liquidity:,.0f}")
        
        # Liq ratio check
        if liq_ratio >= self.entry.min_liq_to_mc_ratio:
            score += 10
        else:
            passes = False
            reasons.append(f"❌ Low liq ratio: {liq_ratio:.1%} < {self.entry.min_liq_to_mc_ratio:.0%}")
        
        # === WARNINGS (reduce score) ===
        
        if bundled > self.entry.warn_bundled_pct:
            score -= 15
            reasons.append(f"⚠️ Bundled: {bundled:.1f}%")
        
        if sold > self.entry.warn_sold_pct:
            score -= 15
            reasons.append(f"⚠️ Sold: {sold:.1f}%")
        
        if snipers > self.entry.warn_snipers_pct:
            score -= 15
            reasons.append(f"⚠️ Snipers: {snipers:.1f}%")
        
        # === SOURCE PRIORITY BONUS ===
        score += source_priority / 10  # Add up to 10 points for source
        
        return passes, reasons, max(0, score)
    
    def calculate_position_size(self, signal_data: dict, prediction: dict) -> float:
        """Calculate recommended position size based on risk"""
        
        base_size = prediction.get("position_size_factor", self.risk.min_position_pct)
        
        # Reduce for high risk
        if self.is_high_risk(signal_data):
            return min(base_size, self.risk.high_risk_position_pct)
        
        # Cap at max
        return min(base_size, self.risk.max_position_pct)
    
    def get_stop_loss(self, signal_data: dict) -> float:
        """Get appropriate stop-loss level"""
        
        if self.is_high_risk(signal_data):
            return self.risk.tight_sl
        
        # Check for low-risk indicators
        bundled = signal_data.get("signal_bundled_pct", 0) or 0
        snipers = signal_data.get("signal_snipers_pct", 0) or 0
        holders = signal_data.get("signal_holders", 0) or 0
        
        if bundled < 5 and snipers < 15 and holders > 200:
            return self.risk.loose_sl
        
        return self.risk.default_sl
    
    def get_take_profit_levels(self, predicted_gain: float) -> List[Dict]:
        """Get take profit levels adjusted for prediction"""
        
        # If predicted gain is very high, extend TP levels
        if predicted_gain > 2.0:  # > 200% predicted
            return [
                {"gain_pct": 50, "sell_pct": 15, "label": "TP1"},
                {"gain_pct": 100, "sell_pct": 20, "label": "TP2"},
                {"gain_pct": 200, "sell_pct": 25, "label": "TP3"},
                {"gain_pct": 400, "sell_pct": 30, "label": "TP4"},
                {"gain_pct": 800, "sell_pct": 100, "label": "MOON"}
            ]
        elif predicted_gain > 1.0:  # > 100% predicted
            return [
                {"gain_pct": 30, "sell_pct": 20, "label": "TP1"},
                {"gain_pct": 70, "sell_pct": 25, "label": "TP2"},
                {"gain_pct": 150, "sell_pct": 30, "label": "TP3"},
                {"gain_pct": 300, "sell_pct": 100, "label": "TP4"}
            ]
        else:
            return self.exit.tp_levels


# Global strategy instance
STRATEGY = TradingStrategy()


def get_strategy() -> TradingStrategy:
    """Get the current trading strategy"""
    return STRATEGY


def update_strategy(**kwargs) -> TradingStrategy:
    """Update strategy parameters"""
    global STRATEGY
    # This would allow runtime updates
    return STRATEGY


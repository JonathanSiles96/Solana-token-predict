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
from typing import List, Dict, Optional
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
    """Risk management rules"""
    
    # Position sizing
    min_position_pct: float = 0.05      # 5% minimum
    max_position_pct: float = 0.10      # 10% maximum
    high_risk_position_pct: float = 0.05  # Reduce for risky trades
    
    # Stop-loss levels
    default_sl: float = -0.35           # -35% default
    tight_sl: float = -0.45             # -45% for risky trades
    loose_sl: float = -0.25             # -25% for low-risk trades
    
    # Risk thresholds that trigger tighter SL
    high_risk_bundled_pct: float = 15.0
    high_risk_snipers_pct: float = 35.0
    high_risk_sold_pct: float = 20.0
    
    # Maximum concurrent positions
    max_active_trades: int = 7
    
    # Diversification: max % of capital in single source
    max_per_source_pct: float = 0.40    # 40% max in one source


@dataclass
class EntryConfig:
    """Entry strategy rules"""
    
    # Minimum thresholds for entry (your criteria)
    min_confidence: float = 0.80        # Confidence ≥ 0.8
    min_risk_adjusted_score: float = 1.0  # Risk-adjusted score ≥ 1.0
    min_volume_1h: float = 10000        # Volume > 10k
    min_holders: int = 50               # Holders > 50
    min_liquidity: float = 5000         # Minimum liquidity
    min_mc: float = 10000               # Minimum market cap
    
    # Red flags - AVOID these
    max_bundled_pct: float = 60.0       # Avoid if bundled > 60%
    max_sold_pct: float = 60.0          # Avoid if sold > 60%
    max_snipers_pct: float = 50.0       # Avoid if snipers > 50%
    
    # Warning thresholds (reduce position size)
    warn_bundled_pct: float = 15.0      # Warning if bundled > 15%
    warn_sold_pct: float = 25.0         # Warning if sold > 25%
    warn_snipers_pct: float = 35.0      # Warning if snipers > 35%
    
    # Entry timing
    max_initial_pump_pct: float = 50.0  # Avoid if already pumped > 50%
    ideal_entry_pump_range: tuple = (5.0, 30.0)  # Enter after 5-30% pump
    
    # Liquidity ratio
    min_liq_to_mc_ratio: float = 0.15   # Minimum 15% liq/mc
    
    # Signal source priority (higher = better)
    source_priority: Dict[str, int] = field(default_factory=lambda: {
        "primal": 100,
        "whale": 80,
        "solana_tracker": 70,
        "whale_trending": 60,
        "early_trending": 50,
        "telegram_early": 40,
        "unknown": 20
    })


@dataclass
class ExitConfig:
    """Exit strategy rules"""
    
    # Take profit levels: (gain_pct, sell_pct)
    # Example: At 30% gain, sell 20% of position
    tp_levels: List[Dict] = field(default_factory=lambda: [
        {"gain_pct": 30, "sell_pct": 20, "label": "TP1"},
        {"gain_pct": 50, "sell_pct": 25, "label": "TP2"},
        {"gain_pct": 100, "sell_pct": 30, "label": "TP3"},
        {"gain_pct": 200, "sell_pct": 35, "label": "TP4"},
        {"gain_pct": 500, "sell_pct": 100, "label": "MOON"}
    ])
    
    # Trailing stop activation
    trailing_stop_activation_pct: float = 30.0  # Activate after 30% gain
    trailing_stop_distance_pct: float = 15.0    # Trail 15% behind peak
    
    # Early exit triggers
    low_momentum_threshold: float = 0.1   # Exit if momentum drops below this
    whale_exit_snipers_pct: float = 40.0  # Exit if snipers spike
    whale_exit_sold_pct: float = 50.0     # Exit if sold spikes


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
    
    def passes_entry_filters(self, signal_data: dict, prediction: dict) -> tuple:
        """
        Check if signal passes all entry filters
        
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
        
        # Calculate liq ratio
        liq_ratio = liquidity / mc if mc > 0 else 0
        
        # === HARD REJECTIONS (red flags) ===
        
        if bundled > self.entry.max_bundled_pct:
            reasons.append(f"🚨 Bundled too high: {bundled:.1f}% > {self.entry.max_bundled_pct}%")
            return False, reasons, 0
        
        if sold > self.entry.max_sold_pct:
            reasons.append(f"🚨 Sold too high: {sold:.1f}% > {self.entry.max_sold_pct}%")
            return False, reasons, 0
        
        if snipers > self.entry.max_snipers_pct:
            reasons.append(f"🚨 Snipers too high: {snipers:.1f}% > {self.entry.max_snipers_pct}%")
            return False, reasons, 0
        
        # === MINIMUM REQUIREMENTS ===
        
        passes = True
        
        # Confidence check
        if confidence >= self.entry.min_confidence:
            score += 25
            reasons.append(f"✅ Confidence: {confidence:.2f}")
        else:
            passes = False
            reasons.append(f"❌ Low confidence: {confidence:.2f} < {self.entry.min_confidence}")
        
        # Risk-adjusted score check
        if risk_adj_score >= self.entry.min_risk_adjusted_score:
            score += 25
            reasons.append(f"✅ Risk-adj score: {risk_adj_score:.2f}")
        else:
            passes = False
            reasons.append(f"❌ Low risk-adj: {risk_adj_score:.2f} < {self.entry.min_risk_adjusted_score}")
        
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
            reasons.append(f"❌ Low liquidity: ${liquidity:,.0f}")
        
        # Liq ratio check
        if liq_ratio >= self.entry.min_liq_to_mc_ratio:
            score += 10
        else:
            reasons.append(f"⚠️ Low liq ratio: {liq_ratio:.1%}")
        
        # === WARNINGS (reduce score but don't reject) ===
        
        if bundled > self.entry.warn_bundled_pct:
            score -= 10
            reasons.append(f"⚠️ High bundled: {bundled:.1f}%")
        
        if sold > self.entry.warn_sold_pct:
            score -= 10
            reasons.append(f"⚠️ High sold: {sold:.1f}%")
        
        if snipers > self.entry.warn_snipers_pct:
            score -= 10
            reasons.append(f"⚠️ High snipers: {snipers:.1f}%")
        
        # === SOURCE PRIORITY BONUS ===
        source = signal_data.get("signal_source", "unknown")
        source_priority = self.get_source_priority(source)
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


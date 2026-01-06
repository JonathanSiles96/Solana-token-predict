"""
Adaptive Filter Strength Model

This module provides a dynamic filter strength system that continuously adapts
based on market conditions and recent trading performance.

The model:
1. Tracks recent signal performance (win rate, avg returns)
2. Adjusts filter strength based on market conditions
3. Updates thresholds dynamically to catch strong signals
4. Adapts to changing market environments
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from collections import deque


@dataclass
class FilterStrengthConfig:
    """Configuration for adaptive filter strength"""
    
    # Base filter thresholds
    base_min_confidence: float = 0.0
    base_min_risk_score: float = 0.0
    base_min_volume: float = 0.0
    base_min_holders: int = 0
    
    # Adaptation parameters
    adaptation_window_hours: int = 24  # Look back 24 hours for performance
    min_samples_for_adaptation: int = 10  # Need at least 10 signals to adapt
    adaptation_rate: float = 0.1  # How quickly to adapt (0-1)
    
    # Performance targets
    target_win_rate: float = 0.70  # Target 70% win rate
    target_avg_return: float = 0.30  # Target 30% avg return
    
    # Market condition indicators
    volatility_window: int = 60  # Minutes to calculate volatility
    volume_trend_window: int = 120  # Minutes to calculate volume trend


@dataclass
class MarketCondition:
    """Current market condition snapshot"""
    timestamp: datetime
    volatility: float  # Price volatility (0-1)
    volume_trend: float  # Volume trend (-1 to 1, negative = declining)
    win_rate_recent: float  # Recent win rate (0-1)
    avg_return_recent: float  # Recent avg return
    signal_frequency: float  # Signals per hour
    market_phase: str  # 'bullish', 'bearish', 'neutral', 'volatile'


class AdaptiveFilterStrength:
    """
    Adaptive filter strength model that adjusts thresholds based on:
    - Recent trading performance
    - Market conditions
    - Signal quality trends
    """
    
    def __init__(self, config: Optional[FilterStrengthConfig] = None):
        self.config = config or FilterStrengthConfig()
        
        # Performance tracking
        self.recent_signals: deque = deque(maxlen=100)  # Last 100 signals
        self.recent_outcomes: deque = deque(maxlen=100)  # Last 100 outcomes
        
        # Current filter strength (0-1, higher = stricter)
        self.current_filter_strength: float = 0.5  # Start at medium
        
        # Market condition tracking
        self.market_conditions: deque = deque(maxlen=50)  # Last 50 market snapshots
        
        # Adaptation history
        self.adaptation_history: List[Dict] = []
    
    def record_signal(self, signal_data: Dict, prediction: Dict, timestamp: datetime):
        """Record a signal for later performance tracking"""
        self.recent_signals.append({
            'timestamp': timestamp,
            'signal_data': signal_data,
            'prediction': prediction,
            'filter_strength': self.current_filter_strength
        })
    
    def record_outcome(self, signal_data: Dict, outcome: Dict, timestamp: datetime):
        """Record trading outcome for performance tracking"""
        self.recent_outcomes.append({
            'timestamp': timestamp,
            'signal_data': signal_data,
            'outcome': outcome,
            'max_return': outcome.get('max_return', 0),
            'is_winner': outcome.get('max_return', 0) >= 1.3,  # 30%+ = winner
            'return_pct': outcome.get('return_pct', 0)
        })
        
        # Trigger adaptation check
        self._check_and_adapt()
    
    def _check_and_adapt(self):
        """Check if we should adapt filter strength based on recent performance"""
        if len(self.recent_outcomes) < self.config.min_samples_for_adaptation:
            return
        
        # Calculate recent performance
        recent_window = datetime.now() - timedelta(hours=self.config.adaptation_window_hours)
        recent_outcomes = [
            o for o in self.recent_outcomes
            if o['timestamp'] >= recent_window
        ]
        
        if len(recent_outcomes) < self.config.min_samples_for_adaptation:
            return
        
        # Calculate metrics
        win_rate = sum(1 for o in recent_outcomes if o['is_winner']) / len(recent_outcomes)
        avg_return = np.mean([o['return_pct'] for o in recent_outcomes]) / 100
        
        # Determine if we need to adjust
        win_rate_diff = win_rate - self.config.target_win_rate
        return_diff = avg_return - self.config.target_avg_return
        
        # If win rate too low, increase filter strength (be stricter)
        # If win rate too high, decrease filter strength (be more permissive)
        strength_adjustment = 0.0
        
        if win_rate < self.config.target_win_rate - 0.10:  # 10% below target
            strength_adjustment = 0.15  # Increase strictness
        elif win_rate < self.config.target_win_rate - 0.05:  # 5% below target
            strength_adjustment = 0.08  # Slight increase
        elif win_rate > self.config.target_win_rate + 0.10:  # 10% above target
            strength_adjustment = -0.10  # Decrease strictness (catch more)
        elif win_rate > self.config.target_win_rate + 0.05:  # 5% above target
            strength_adjustment = -0.05  # Slight decrease
        
        # Adjust for return performance
        if avg_return < self.config.target_avg_return - 0.10:
            strength_adjustment += 0.10  # Increase strictness if returns low
        elif avg_return > self.config.target_avg_return + 0.10:
            strength_adjustment -= 0.05  # Slight decrease if returns high
        
        # Apply adaptation
        if abs(strength_adjustment) > 0.01:  # Only adapt if meaningful change
            old_strength = self.current_filter_strength
            self.current_filter_strength = np.clip(
                self.current_filter_strength + strength_adjustment * self.config.adaptation_rate,
                0.0, 1.0
            )
            
            # Record adaptation
            self.adaptation_history.append({
                'timestamp': datetime.now(),
                'old_strength': old_strength,
                'new_strength': self.current_filter_strength,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'reason': f"WR: {win_rate:.2%}, Return: {avg_return:.2%}"
            })
    
    def get_adjusted_thresholds(self, base_thresholds: Dict) -> Dict:
        """
        Get adjusted filter thresholds based on current filter strength
        
        Args:
            base_thresholds: Base threshold values
            
        Returns:
            Adjusted thresholds based on current filter strength
        """
        # Higher filter strength = stricter = higher thresholds
        # Lower filter strength = more permissive = lower thresholds
        
        strength_multiplier = 1.0 + (self.current_filter_strength - 0.5) * 0.5  # 0.75 to 1.25
        
        adjusted = {}
        
        # Adjust confidence threshold
        if 'min_confidence' in base_thresholds:
            base_conf = base_thresholds['min_confidence']
            adjusted['min_confidence'] = base_conf * strength_multiplier
        
        # Adjust risk score threshold
        if 'min_risk_score' in base_thresholds:
            base_score = base_thresholds['min_risk_score']
            adjusted['min_risk_score'] = base_score * strength_multiplier
        
        # Adjust volume threshold
        if 'min_volume' in base_thresholds:
            base_vol = base_thresholds['min_volume']
            adjusted['min_volume'] = base_vol * strength_multiplier
        
        # Adjust holders threshold
        if 'min_holders' in base_thresholds:
            base_holders = base_thresholds['min_holders']
            adjusted['min_holders'] = int(base_holders * strength_multiplier)
        
        # Adjust source priority threshold
        if 'min_source_priority' in base_thresholds:
            base_priority = base_thresholds['min_source_priority']
            adjusted['min_source_priority'] = int(base_priority * strength_multiplier)
        
        return adjusted
    
    def assess_market_condition(self, recent_price_data: List[Dict]) -> MarketCondition:
        """
        Assess current market condition based on recent price/volume data
        
        Args:
            recent_price_data: List of price snapshots with keys: timestamp, price, volume, mcap
            
        Returns:
            MarketCondition object
        """
        if len(recent_price_data) < 10:
            # Default to neutral if not enough data
            return MarketCondition(
                timestamp=datetime.now(),
                volatility=0.5,
                volume_trend=0.0,
                win_rate_recent=0.0,
                avg_return_recent=0.0,
                signal_frequency=0.0,
                market_phase='neutral'
            )
        
        # Calculate volatility
        prices = [d['price'] for d in recent_price_data[-self.config.volatility_window:]]
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.0
        else:
            volatility = 0.0
        
        # Calculate volume trend
        volumes = [d.get('volume', 0) for d in recent_price_data[-self.config.volume_trend_window:]]
        if len(volumes) > 1:
            volume_trend = (volumes[-1] - volumes[0]) / (volumes[0] + 1e-6)
            volume_trend = np.clip(volume_trend, -1.0, 1.0)
        else:
            volume_trend = 0.0
        
        # Calculate recent win rate
        recent_window = datetime.now() - timedelta(hours=self.config.adaptation_window_hours)
        recent_outcomes = [
            o for o in self.recent_outcomes
            if o['timestamp'] >= recent_window
        ]
        win_rate = sum(1 for o in recent_outcomes if o['is_winner']) / len(recent_outcomes) if recent_outcomes else 0.0
        avg_return = np.mean([o['return_pct'] for o in recent_outcomes]) / 100 if recent_outcomes else 0.0
        
        # Calculate signal frequency
        recent_signals = [
            s for s in self.recent_signals
            if s['timestamp'] >= recent_window
        ]
        hours_covered = self.config.adaptation_window_hours
        signal_frequency = len(recent_signals) / max(hours_covered, 1)
        
        # Determine market phase
        if volatility > 0.3:
            market_phase = 'volatile'
        elif volume_trend > 0.2 and win_rate > 0.6:
            market_phase = 'bullish'
        elif volume_trend < -0.2 or win_rate < 0.4:
            market_phase = 'bearish'
        else:
            market_phase = 'neutral'
        
        condition = MarketCondition(
            timestamp=datetime.now(),
            volatility=volatility,
            volume_trend=volume_trend,
            win_rate_recent=win_rate,
            avg_return_recent=avg_return,
            signal_frequency=signal_frequency,
            market_phase=market_phase
        )
        
        self.market_conditions.append(condition)
        return condition
    
    def get_filter_strength_for_market(self, market_condition: MarketCondition) -> float:
        """
        Adjust filter strength based on market condition
        
        In volatile/bearish markets: be stricter (higher strength)
        In bullish markets: be more permissive (lower strength) to catch opportunities
        """
        base_strength = self.current_filter_strength
        
        # Adjust based on market phase
        if market_condition.market_phase == 'volatile':
            adjustment = 0.15  # Stricter in volatile markets
        elif market_condition.market_phase == 'bearish':
            adjustment = 0.10  # Stricter in bearish markets
        elif market_condition.market_phase == 'bullish':
            adjustment = -0.05  # Slightly more permissive in bullish
        else:
            adjustment = 0.0
        
        # Adjust based on recent performance
        if market_condition.win_rate_recent < 0.5:
            adjustment += 0.10  # Stricter if win rate low
        elif market_condition.win_rate_recent > 0.8:
            adjustment -= 0.05  # More permissive if win rate high
        
        return np.clip(base_strength + adjustment, 0.0, 1.0)
    
    def get_status(self) -> Dict:
        """Get current status of the adaptive filter"""
        recent_window = datetime.now() - timedelta(hours=self.config.adaptation_window_hours)
        recent_outcomes = [
            o for o in self.recent_outcomes
            if o['timestamp'] >= recent_window
        ]
        
        win_rate = sum(1 for o in recent_outcomes if o['is_winner']) / len(recent_outcomes) if recent_outcomes else 0.0
        avg_return = np.mean([o['return_pct'] for o in recent_outcomes]) / 100 if recent_outcomes else 0.0
        
        return {
            'current_filter_strength': self.current_filter_strength,
            'recent_win_rate': win_rate,
            'recent_avg_return': avg_return,
            'total_signals_tracked': len(self.recent_signals),
            'total_outcomes_tracked': len(self.recent_outcomes),
            'adaptation_count': len(self.adaptation_history),
            'last_adaptation': self.adaptation_history[-1] if self.adaptation_history else None
        }


# Global instance
_adaptive_filter: Optional[AdaptiveFilterStrength] = None


def get_adaptive_filter() -> AdaptiveFilterStrength:
    """Get or create the global adaptive filter instance"""
    global _adaptive_filter
    if _adaptive_filter is None:
        _adaptive_filter = AdaptiveFilterStrength()
    return _adaptive_filter



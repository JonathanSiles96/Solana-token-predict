"""
24-Hour Data Analyzer

This module analyzes 24-hour streams of price, market cap, and liquidity data
to stabilize profit before full AI integration.

The analyzer:
1. Tracks price movements over 24 hours
2. Analyzes market cap trends
3. Monitors liquidity changes
4. Identifies patterns that lead to profitable trades
5. Provides signals for entry/exit timing
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class PriceSnapshot:
    """Single price snapshot"""
    timestamp: datetime
    price: float
    market_cap: float
    liquidity: float
    volume: float
    holders: Optional[int] = None


@dataclass
class PricePattern:
    """Identified price pattern"""
    pattern_type: str  # 'pump', 'dump', 'consolidation', 'breakout', 'reversal'
    start_time: datetime
    end_time: datetime
    price_change_pct: float
    volume_change_pct: float
    liquidity_change_pct: float
    confidence: float  # 0-1, how confident we are in this pattern


@dataclass
class TradingSignal:
    """Trading signal from 24h analysis"""
    signal_type: str  # 'entry', 'exit', 'hold', 'avoid'
    timestamp: datetime
    confidence: float  # 0-1
    reason: str
    expected_gain_pct: float
    risk_level: str  # 'low', 'medium', 'high'
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None


class TwentyFourHourAnalyzer:
    """
    Analyzes 24-hour data streams to identify profitable patterns
    and generate trading signals
    """
    
    def __init__(self):
        self.price_history: List[PriceSnapshot] = []
        self.max_history_hours: int = 24
        self.patterns: List[PricePattern] = []
        self.signals: List[TradingSignal] = []
    
    def add_price_snapshot(self, snapshot: PriceSnapshot):
        """Add a new price snapshot to the history"""
        self.price_history.append(snapshot)
        
        # Keep only last 24 hours
        cutoff = datetime.now() - timedelta(hours=self.max_history_hours)
        self.price_history = [
            s for s in self.price_history
            if s.timestamp >= cutoff
        ]
        
        # Sort by timestamp
        self.price_history.sort(key=lambda x: x.timestamp)
    
    def analyze_patterns(self) -> List[PricePattern]:
        """
        Analyze price history to identify patterns
        
        Returns:
            List of identified patterns
        """
        if len(self.price_history) < 10:
            return []
        
        patterns = []
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([
            {
                'timestamp': s.timestamp,
                'price': s.price,
                'mcap': s.market_cap,
                'liquidity': s.liquidity,
                'volume': s.volume
            }
            for s in self.price_history
        ])
        
        # Calculate returns
        df['price_return'] = df['price'].pct_change()
        df['mcap_return'] = df['mcap'].pct_change()
        df['liq_return'] = df['liquidity'].pct_change()
        df['volume_return'] = df['volume'].pct_change()
        
        # Identify pump patterns (sustained price increase with volume)
        pump_windows = self._identify_pumps(df)
        patterns.extend(pump_windows)
        
        # Identify dump patterns (sustained price decrease)
        dump_windows = self._identify_dumps(df)
        patterns.extend(dump_windows)
        
        # Identify consolidation (sideways movement)
        consolidation_windows = self._identify_consolidation(df)
        patterns.extend(consolidation_windows)
        
        # Identify breakouts (price breaks above resistance)
        breakout_windows = self._identify_breakouts(df)
        patterns.extend(breakout_windows)
        
        self.patterns = patterns
        return patterns
    
    def _identify_pumps(self, df: pd.DataFrame, min_gain: float = 0.20) -> List[PricePattern]:
        """Identify pump patterns (20%+ gain with volume)"""
        patterns = []
        
        window_size = 5  # 5 snapshots
        for i in range(len(df) - window_size):
            window = df.iloc[i:i+window_size]
            
            price_change = (window['price'].iloc[-1] - window['price'].iloc[0]) / window['price'].iloc[0]
            volume_change = (window['volume'].iloc[-1] - window['volume'].iloc[0]) / (window['volume'].iloc[0] + 1e-6)
            liq_change = (window['liquidity'].iloc[-1] - window['liquidity'].iloc[0]) / (window['liquidity'].iloc[0] + 1e-6)
            
            if price_change >= min_gain and volume_change > 0.1:  # 20%+ price, 10%+ volume
                confidence = min(1.0, price_change / 0.50)  # Higher confidence for bigger pumps
                
                patterns.append(PricePattern(
                    pattern_type='pump',
                    start_time=window['timestamp'].iloc[0],
                    end_time=window['timestamp'].iloc[-1],
                    price_change_pct=price_change * 100,
                    volume_change_pct=volume_change * 100,
                    liquidity_change_pct=liq_change * 100,
                    confidence=confidence
                ))
        
        return patterns
    
    def _identify_dumps(self, df: pd.DataFrame, min_loss: float = -0.15) -> List[PricePattern]:
        """Identify dump patterns (15%+ loss)"""
        patterns = []
        
        window_size = 5
        for i in range(len(df) - window_size):
            window = df.iloc[i:i+window_size]
            
            price_change = (window['price'].iloc[-1] - window['price'].iloc[0]) / window['price'].iloc[0]
            volume_change = (window['volume'].iloc[-1] - window['volume'].iloc[0]) / (window['volume'].iloc[0] + 1e-6)
            liq_change = (window['liquidity'].iloc[-1] - window['liquidity'].iloc[0]) / (window['liquidity'].iloc[0] + 1e-6)
            
            if price_change <= min_loss:  # 15%+ loss
                confidence = min(1.0, abs(price_change) / 0.30)  # Higher confidence for bigger dumps
                
                patterns.append(PricePattern(
                    pattern_type='dump',
                    start_time=window['timestamp'].iloc[0],
                    end_time=window['timestamp'].iloc[-1],
                    price_change_pct=price_change * 100,
                    volume_change_pct=volume_change * 100,
                    liquidity_change_pct=liq_change * 100,
                    confidence=confidence
                ))
        
        return patterns
    
    def _identify_consolidation(self, df: pd.DataFrame, max_volatility: float = 0.05) -> List[PricePattern]:
        """Identify consolidation patterns (low volatility, sideways movement)"""
        patterns = []
        
        window_size = 10
        for i in range(len(df) - window_size):
            window = df.iloc[i:i+window_size]
            
            price_std = window['price'].std() / window['price'].mean()
            
            if price_std <= max_volatility:  # Low volatility
                price_change = abs((window['price'].iloc[-1] - window['price'].iloc[0]) / window['price'].iloc[0])
                
                if price_change < 0.10:  # Less than 10% change
                    patterns.append(PricePattern(
                        pattern_type='consolidation',
                        start_time=window['timestamp'].iloc[0],
                        end_time=window['timestamp'].iloc[-1],
                        price_change_pct=price_change * 100,
                        volume_change_pct=0.0,
                        liquidity_change_pct=0.0,
                        confidence=0.7
                    ))
        
        return patterns
    
    def _identify_breakouts(self, df: pd.DataFrame) -> List[PricePattern]:
        """Identify breakout patterns (price breaks above recent high)"""
        patterns = []
        
        window_size = 10
        for i in range(window_size, len(df)):
            lookback = df.iloc[i-window_size:i]
            current = df.iloc[i]
            
            recent_high = lookback['price'].max()
            current_price = current['price']
            
            if current_price > recent_high * 1.05:  # 5% above recent high
                volume_increase = current['volume'] > lookback['volume'].mean() * 1.2
                
                if volume_increase:
                    price_change = (current_price - recent_high) / recent_high
                    confidence = min(1.0, price_change / 0.20)  # Higher confidence for bigger breakouts
                    
                    patterns.append(PricePattern(
                        pattern_type='breakout',
                        start_time=lookback['timestamp'].iloc[0],
                        end_time=current['timestamp'],
                        price_change_pct=price_change * 100,
                        volume_change_pct=((current['volume'] - lookback['volume'].mean()) / lookback['volume'].mean()) * 100,
                        liquidity_change_pct=0.0,
                        confidence=confidence
                    ))
        
        return patterns
    
    def generate_trading_signal(self, current_snapshot: PriceSnapshot) -> TradingSignal:
        """
        Generate trading signal based on 24h analysis
        
        Args:
            current_snapshot: Current price snapshot
            
        Returns:
            TradingSignal object
        """
        # Analyze patterns first
        patterns = self.analyze_patterns()
        
        if not patterns:
            return TradingSignal(
                signal_type='hold',
                timestamp=current_snapshot.timestamp,
                confidence=0.5,
                reason='Insufficient data for analysis',
                expected_gain_pct=0.0,
                risk_level='medium'
            )
        
        # Get most recent pattern
        recent_pattern = max(patterns, key=lambda p: p.end_time)
        
        # Determine signal based on pattern
        if recent_pattern.pattern_type == 'pump':
            # If pump just started, might be entry opportunity
            if recent_pattern.confidence > 0.7 and recent_pattern.price_change_pct < 50:
                return TradingSignal(
                    signal_type='entry',
                    timestamp=current_snapshot.timestamp,
                    confidence=recent_pattern.confidence,
                    reason=f'Pump pattern detected: {recent_pattern.price_change_pct:.1f}% gain',
                    expected_gain_pct=min(100, recent_pattern.price_change_pct * 1.5),
                    risk_level='medium',
                    price_target=current_snapshot.price * (1 + recent_pattern.price_change_pct / 100 * 1.5),
                    stop_loss=current_snapshot.price * 0.85
                )
            else:
                return TradingSignal(
                    signal_type='avoid',
                    timestamp=current_snapshot.timestamp,
                    confidence=0.8,
                    reason='Pump already advanced, risk of dump',
                    expected_gain_pct=0.0,
                    risk_level='high'
                )
        
        elif recent_pattern.pattern_type == 'dump':
            return TradingSignal(
                signal_type='avoid',
                timestamp=current_snapshot.timestamp,
                confidence=0.9,
                reason=f'Dump pattern detected: {recent_pattern.price_change_pct:.1f}% loss',
                expected_gain_pct=0.0,
                risk_level='high'
            )
        
        elif recent_pattern.pattern_type == 'breakout':
            return TradingSignal(
                signal_type='entry',
                timestamp=current_snapshot.timestamp,
                confidence=recent_pattern.confidence,
                reason=f'Breakout pattern detected: {recent_pattern.price_change_pct:.1f}% above recent high',
                expected_gain_pct=min(80, recent_pattern.price_change_pct * 2),
                risk_level='low',
                price_target=current_snapshot.price * (1 + recent_pattern.price_change_pct / 100 * 2),
                stop_loss=current_snapshot.price * 0.90
            )
        
        elif recent_pattern.pattern_type == 'consolidation':
            return TradingSignal(
                signal_type='hold',
                timestamp=current_snapshot.timestamp,
                confidence=0.6,
                reason='Consolidation pattern - waiting for breakout',
                expected_gain_pct=0.0,
                risk_level='low'
            )
        
        # Default: hold
        return TradingSignal(
            signal_type='hold',
            timestamp=current_snapshot.timestamp,
            confidence=0.5,
            reason='No clear pattern identified',
            expected_gain_pct=0.0,
            risk_level='medium'
        )
    
    def get_24h_summary(self) -> Dict:
        """Get summary of 24h analysis"""
        if not self.price_history:
            return {
                'data_points': 0,
                'time_span_hours': 0,
                'patterns_found': 0,
                'price_change_pct': 0.0,
                'volume_change_pct': 0.0,
                'liquidity_change_pct': 0.0
            }
        
        first = self.price_history[0]
        last = self.price_history[-1]
        
        time_span = (last.timestamp - first.timestamp).total_seconds() / 3600
        
        price_change = (last.price - first.price) / first.price * 100
        volume_change = (last.volume - first.volume) / (first.volume + 1e-6) * 100
        liq_change = (last.liquidity - first.liquidity) / (first.liquidity + 1e-6) * 100
        
        return {
            'data_points': len(self.price_history),
            'time_span_hours': time_span,
            'patterns_found': len(self.patterns),
            'price_change_pct': price_change,
            'volume_change_pct': volume_change,
            'liquidity_change_pct': liq_change,
            'recent_patterns': [
                {
                    'type': p.pattern_type,
                    'confidence': p.confidence,
                    'price_change': p.price_change_pct
                }
                for p in self.patterns[-5:]  # Last 5 patterns
            ]
        }


# Global instance
_analyzer: Optional[TwentyFourHourAnalyzer] = None


def get_24h_analyzer() -> TwentyFourHourAnalyzer:
    """Get or create the global 24h analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = TwentyFourHourAnalyzer()
    return _analyzer



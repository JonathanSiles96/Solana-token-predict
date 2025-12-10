"""
Dynamic Order Management System with Trailing Stop Loss

This module provides adaptive trading parameters that adjust based on:
- Real-time price action
- Volatility
- Volume patterns
- Token behavior
- Model confidence

Features:
- Trailing stop loss
- Dynamic take-profit levels
- Position sizing based on risk
- Automatic order adjustment
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass


@dataclass
class OrderParameters:
    """Trading order parameters"""
    entry_price: float
    position_size: float  # % of portfolio
    
    # Stop Loss
    stop_loss: float  # Price level
    stop_loss_pct: float  # % from entry
    trailing_stop_enabled: bool
    trailing_stop_activation: float  # Activate trailing after this % gain
    trailing_stop_distance: float  # Trail distance in %
    
    # Take Profit Levels
    tp_levels: List[Dict[str, float]]  # [{"price": X, "size_pct": Y}]
    
    # Risk Management
    max_loss_amount: float  # $ amount
    max_position_duration: int  # minutes
    
    # Behavioral Flags
    aggressive_exit: bool  # Exit faster on weakness
    hold_for_momentum: bool  # Hold longer on strength
    
    # Metadata
    confidence: float
    predicted_gain: float
    created_at: datetime


class DynamicOrderManager:
    """
    Manages dynamic orders with trailing stop loss and adaptive behavior
    
    Usage:
    ```python
    manager = DynamicOrderManager(
        initial_capital=10000,
        max_position_size=0.10,  # 10% max
        min_position_size=0.02,  # 2% min
    )
    
    # Create initial orders
    orders = manager.create_initial_orders(
        token_data=token_features,
        model_prediction=prediction,
        entry_price=0.000123
    )
    
    # Update based on price action
    updates = manager.update_orders(
        orders=orders,
        current_price=0.000145,
        current_volume=50000,
        time_elapsed_minutes=15
    )
    ```
    """
    
    def __init__(self,
                 initial_capital: float = 10000,
                 max_position_size: float = 0.10,
                 min_position_size: float = 0.02,
                 base_stop_loss: float = -0.35,  # -35%
                 aggressive_stop_loss: float = -0.25,  # -25% for risky tokens
                 trailing_stop_default: float = 0.15):  # 15% trailing distance
        """
        Initialize order manager
        
        Args:
            initial_capital: Starting capital ($)
            max_position_size: Maximum position size (% of capital)
            min_position_size: Minimum position size (% of capital)
            base_stop_loss: Default stop loss (%)
            aggressive_stop_loss: Tighter stop loss for risky tokens (%)
            trailing_stop_default: Default trailing stop distance (%)
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.base_stop_loss = base_stop_loss
        self.aggressive_stop_loss = aggressive_stop_loss
        self.trailing_stop_default = trailing_stop_default
        
        # Track active positions
        self.active_orders: Dict[str, OrderParameters] = {}
        self.order_history: List[Dict] = []
    
    def create_initial_orders(self,
                            token_data: Dict,
                            model_prediction: Dict,
                            entry_price: float,
                            token_id: Optional[str] = None) -> OrderParameters:
        """
        Create initial order parameters based on model prediction
        
        Args:
            token_data: Token features
            model_prediction: Model output (predicted_gain, confidence, etc.)
            entry_price: Entry price
            token_id: Unique token identifier
        
        Returns:
            OrderParameters object
        """
        predicted_gain = model_prediction['predicted_gain']
        confidence = model_prediction['confidence']
        
        # Calculate position size based on confidence and risk
        position_size = self._calculate_position_size(
            predicted_gain=predicted_gain,
            confidence=confidence,
            token_data=token_data
        )
        
        # Determine stop loss strategy
        stop_loss_pct, trailing_config = self._calculate_stop_loss_strategy(
            predicted_gain=predicted_gain,
            confidence=confidence,
            token_data=token_data
        )
        
        stop_loss_price = entry_price * (1 + stop_loss_pct)
        
        # Generate dynamic take-profit levels
        tp_levels = self._generate_take_profit_levels(
            entry_price=entry_price,
            predicted_gain=predicted_gain,
            confidence=confidence,
            token_data=token_data
        )
        
        # Behavioral flags
        aggressive_exit = self._should_use_aggressive_exit(token_data)
        hold_for_momentum = self._should_hold_for_momentum(predicted_gain, confidence)
        
        # Calculate max loss
        max_loss_amount = self.initial_capital * position_size * abs(stop_loss_pct)
        
        # Position duration based on confidence
        max_duration = self._calculate_max_duration(confidence, predicted_gain)
        
        orders = OrderParameters(
            entry_price=entry_price,
            position_size=position_size,
            stop_loss=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            trailing_stop_enabled=trailing_config['enabled'],
            trailing_stop_activation=trailing_config['activation'],
            trailing_stop_distance=trailing_config['distance'],
            tp_levels=tp_levels,
            max_loss_amount=max_loss_amount,
            max_position_duration=max_duration,
            aggressive_exit=aggressive_exit,
            hold_for_momentum=hold_for_momentum,
            confidence=confidence,
            predicted_gain=predicted_gain,
            created_at=datetime.now()
        )
        
        # Store active order
        if token_id:
            self.active_orders[token_id] = orders
        
        return orders
    
    def update_orders(self,
                     orders: OrderParameters,
                     current_price: float,
                     current_volume: Optional[float] = None,
                     time_elapsed_minutes: Optional[int] = None,
                     token_id: Optional[str] = None) -> Dict:
        """
        Update orders based on current market conditions
        
        Returns order updates and actions to take
        
        Args:
            orders: Current order parameters
            current_price: Current token price
            current_volume: Current trading volume
            time_elapsed_minutes: Time since entry
            token_id: Token identifier
        
        Returns:
            Dict with updates and recommended actions
        """
        current_gain = (current_price - orders.entry_price) / orders.entry_price
        
        updates = {
            'current_price': current_price,
            'current_gain': current_gain,
            'current_gain_pct': current_gain * 100,
            'actions': [],
            'new_stop_loss': orders.stop_loss,
            'new_tp_levels': orders.tp_levels,
            'reason': None,
            'should_exit': False
        }
        
        # Check trailing stop loss
        if orders.trailing_stop_enabled:
            trailing_update = self._update_trailing_stop(
                orders=orders,
                current_price=current_price,
                current_gain=current_gain
            )
            
            if trailing_update['updated']:
                updates['new_stop_loss'] = trailing_update['new_stop_loss']
                updates['actions'].append({
                    'type': 'update_stop_loss',
                    'old_value': orders.stop_loss,
                    'new_value': trailing_update['new_stop_loss'],
                    'reason': 'trailing_stop_activated'
                })
                
                # Update the order
                orders.stop_loss = trailing_update['new_stop_loss']
        
        # Check if stop loss hit
        if current_price <= orders.stop_loss:
            updates['should_exit'] = True
            updates['reason'] = 'stop_loss_hit'
            updates['actions'].append({
                'type': 'exit',
                'reason': 'stop_loss_hit',
                'price': current_price,
                'gain': current_gain
            })
            return updates
        
        # Check take profit levels
        tp_hit = self._check_take_profit_hit(orders, current_price)
        if tp_hit:
            updates['actions'].append(tp_hit)
            
            # Partial exit
            if tp_hit['size_pct'] < 1.0:
                updates['reason'] = f"partial_exit_tp{tp_hit['level']}"
            else:
                updates['should_exit'] = True
                updates['reason'] = f"take_profit_{tp_hit['level']}_hit"
        
        # Check max duration
        if time_elapsed_minutes and time_elapsed_minutes >= orders.max_position_duration:
            updates['should_exit'] = True
            updates['reason'] = 'max_duration_reached'
            updates['actions'].append({
                'type': 'exit',
                'reason': 'max_duration',
                'time_elapsed': time_elapsed_minutes
            })
        
        # Behavioral adjustments
        if orders.aggressive_exit:
            weakness_detected = self._detect_weakness(
                current_gain=current_gain,
                current_volume=current_volume,
                time_elapsed=time_elapsed_minutes
            )
            
            if weakness_detected:
                updates['should_exit'] = True
                updates['reason'] = 'weakness_detected'
                updates['actions'].append({
                    'type': 'exit',
                    'reason': 'aggressive_exit_weakness'
                })
        
        # Momentum holding
        if orders.hold_for_momentum and current_gain > orders.predicted_gain:
            momentum_strong = self._detect_momentum(
                current_gain=current_gain,
                current_volume=current_volume,
                predicted_gain=orders.predicted_gain
            )
            
            if momentum_strong:
                # Extend TP levels
                extended_tp = self._extend_take_profit_levels(
                    orders=orders,
                    current_gain=current_gain
                )
                
                if extended_tp:
                    updates['new_tp_levels'] = extended_tp
                    updates['actions'].append({
                        'type': 'extend_tp_levels',
                        'reason': 'strong_momentum',
                        'new_levels': extended_tp
                    })
        
        # Store history
        self.order_history.append({
            'timestamp': datetime.now(),
            'token_id': token_id,
            'price': current_price,
            'gain': current_gain,
            'updates': updates
        })
        
        return updates
    
    def _calculate_position_size(self,
                                 predicted_gain: float,
                                 confidence: float,
                                 token_data: Dict) -> float:
        """Calculate position size based on risk/reward"""
        
        # Base position size from confidence
        base_size = self.min_position_size + (
            confidence * (self.max_position_size - self.min_position_size)
        )
        
        # Adjust for predicted gain
        if predicted_gain > 2.0:  # 200%+
            size_multiplier = 1.2
        elif predicted_gain > 1.0:  # 100%+
            size_multiplier = 1.1
        elif predicted_gain < 0.3:  # < 30%
            size_multiplier = 0.8
        else:
            size_multiplier = 1.0
        
        # Adjust for risk factors
        risk_score = token_data.get('risk_score', 0.5)
        if risk_score > 0.7:  # High risk
            size_multiplier *= 0.7
        elif risk_score < 0.3:  # Low risk
            size_multiplier *= 1.2
        
        position_size = base_size * size_multiplier
        
        # Clamp to limits
        return max(self.min_position_size, min(self.max_position_size, position_size))
    
    def _calculate_stop_loss_strategy(self,
                                      predicted_gain: float,
                                      confidence: float,
                                      token_data: Dict) -> Tuple[float, Dict]:
        """
        Calculate stop loss and trailing stop configuration
        
        Returns:
            (stop_loss_pct, trailing_config)
        """
        # Base stop loss
        bundled_pct = token_data.get('signal_bundled_pct', 0)
        snipers_pct = token_data.get('signal_snipers_pct', 0)
        
        # Tighter stop for risky tokens
        if bundled_pct > 15 or snipers_pct > 40:
            stop_loss_pct = self.aggressive_stop_loss
        else:
            stop_loss_pct = self.base_stop_loss
        
        # Adjust based on confidence
        if confidence > 0.8:
            stop_loss_pct *= 0.9  # Slightly tighter (more confident)
        elif confidence < 0.5:
            stop_loss_pct *= 1.1  # Slightly wider (less confident)
        
        # Trailing stop configuration
        trailing_enabled = confidence > 0.6  # Only use trailing if confident
        
        # Activate trailing after reaching certain gain
        if predicted_gain > 2.0:
            trailing_activation = 0.5  # 50% gain
        elif predicted_gain > 1.0:
            trailing_activation = 0.3  # 30% gain
        else:
            trailing_activation = 0.2  # 20% gain
        
        # Trailing distance based on volatility
        liq_to_mc = token_data.get('liq_to_mc_ratio', 0.2)
        if liq_to_mc < 0.15:  # Low liquidity = higher volatility
            trailing_distance = 0.20  # 20% trail
        elif liq_to_mc > 0.30:  # High liquidity = lower volatility
            trailing_distance = 0.10  # 10% trail
        else:
            trailing_distance = self.trailing_stop_default  # 15% trail
        
        trailing_config = {
            'enabled': trailing_enabled,
            'activation': trailing_activation,
            'distance': trailing_distance
        }
        
        return stop_loss_pct, trailing_config
    
    def _generate_take_profit_levels(self,
                                     entry_price: float,
                                     predicted_gain: float,
                                     confidence: float,
                                     token_data: Dict) -> List[Dict]:
        """
        Generate dynamic take-profit levels
        
        Strategy:
        - TP1: Conservative (30-50% of predicted)
        - TP2: Target (70-90% of predicted)
        - TP3: Stretch (110-150% of predicted)
        """
        tp_levels = []
        
        # TP1: Early profit taking (25% of position)
        tp1_gain = predicted_gain * 0.35
        tp1_price = entry_price * (1 + tp1_gain)
        tp_levels.append({
            'level': 1,
            'price': tp1_price,
            'gain_pct': tp1_gain,
            'size_pct': 0.25  # Sell 25% of position
        })
        
        # TP2: Main target (50% of remaining position = 37.5% total)
        tp2_gain = predicted_gain * 0.80
        tp2_price = entry_price * (1 + tp2_gain)
        tp_levels.append({
            'level': 2,
            'price': tp2_price,
            'gain_pct': tp2_gain,
            'size_pct': 0.50  # Sell 50% of remaining (37.5% total)
        })
        
        # TP3: Stretch goal (remaining 37.5%)
        if confidence > 0.7:
            tp3_gain = predicted_gain * 1.30  # 130% of predicted
        else:
            tp3_gain = predicted_gain * 1.10  # 110% of predicted
        
        tp3_price = entry_price * (1 + tp3_gain)
        tp_levels.append({
            'level': 3,
            'price': tp3_price,
            'gain_pct': tp3_gain,
            'size_pct': 1.0  # Sell remaining
        })
        
        return tp_levels
    
    def _update_trailing_stop(self,
                             orders: OrderParameters,
                             current_price: float,
                             current_gain: float) -> Dict:
        """Update trailing stop loss if conditions met"""
        
        # Check if trailing should activate
        if current_gain < orders.trailing_stop_activation:
            return {'updated': False}
        
        # Calculate new trailing stop
        trailing_stop_price = current_price * (1 - orders.trailing_stop_distance)
        
        # Only update if it's higher than current stop
        if trailing_stop_price > orders.stop_loss:
            return {
                'updated': True,
                'new_stop_loss': trailing_stop_price,
                'old_stop_loss': orders.stop_loss,
                'trailing_from_price': current_price
            }
        
        return {'updated': False}
    
    def _check_take_profit_hit(self,
                               orders: OrderParameters,
                               current_price: float) -> Optional[Dict]:
        """Check if any TP level hit"""
        
        for tp in orders.tp_levels:
            if current_price >= tp['price']:
                return {
                    'type': 'take_profit_hit',
                    'level': tp['level'],
                    'price': tp['price'],
                    'gain_pct': tp['gain_pct'],
                    'size_pct': tp['size_pct']
                }
        
        return None
    
    def _should_use_aggressive_exit(self, token_data: Dict) -> bool:
        """Determine if aggressive exit strategy should be used"""
        
        # Use aggressive exit for riskier tokens
        bundled_pct = token_data.get('signal_bundled_pct', 0)
        snipers_pct = token_data.get('signal_snipers_pct', 0)
        sold_pct = token_data.get('signal_sold_pct', 0)
        
        risk_flags = (
            (bundled_pct > 12) +
            (snipers_pct > 35) +
            (sold_pct > 15)
        )
        
        return risk_flags >= 2
    
    def _should_hold_for_momentum(self,
                                  predicted_gain: float,
                                  confidence: float) -> bool:
        """Determine if should hold for momentum"""
        
        # Hold for momentum if high confidence and high predicted gain
        return confidence > 0.75 and predicted_gain > 1.0
    
    def _calculate_max_duration(self,
                                confidence: float,
                                predicted_gain: float) -> int:
        """Calculate maximum position duration in minutes"""
        
        # Base duration
        if predicted_gain > 2.0:
            base_duration = 180  # 3 hours for moon shots
        elif predicted_gain > 1.0:
            base_duration = 120  # 2 hours for 2x+
        else:
            base_duration = 60  # 1 hour for modest gains
        
        # Adjust for confidence
        if confidence < 0.5:
            base_duration = int(base_duration * 0.7)  # Exit faster if uncertain
        elif confidence > 0.8:
            base_duration = int(base_duration * 1.3)  # Hold longer if confident
        
        return base_duration
    
    def _detect_weakness(self,
                        current_gain: float,
                        current_volume: Optional[float],
                        time_elapsed: Optional[int]) -> bool:
        """Detect signs of weakness"""
        
        # Price declining after initial gain
        if current_gain > 0.2 and current_gain < 0.1:
            return True
        
        # Been in position long time with minimal gain
        if time_elapsed and time_elapsed > 30 and current_gain < 0.15:
            return True
        
        return False
    
    def _detect_momentum(self,
                        current_gain: float,
                        current_volume: Optional[float],
                        predicted_gain: float) -> bool:
        """Detect strong momentum"""
        
        # Exceeded prediction significantly
        if current_gain > predicted_gain * 1.2:
            return True
        
        # Strong consistent gains
        if current_gain > 1.0:  # 100%+
            return True
        
        return False
    
    def _extend_take_profit_levels(self,
                                   orders: OrderParameters,
                                   current_gain: float) -> Optional[List[Dict]]:
        """Extend TP levels for strong momentum"""
        
        # Only extend if past TP2
        if current_gain <= orders.tp_levels[1]['gain_pct']:
            return None
        
        # Add TP4 - moon shot level
        extended_levels = orders.tp_levels.copy()
        
        tp4_gain = current_gain * 1.5  # 150% of current
        tp4_price = orders.entry_price * (1 + tp4_gain)
        
        extended_levels.append({
            'level': 4,
            'price': tp4_price,
            'gain_pct': tp4_gain,
            'size_pct': 1.0
        })
        
        return extended_levels
    
    def get_order_summary(self, orders: OrderParameters) -> Dict:
        """Get human-readable order summary"""
        
        return {
            'entry_price': f"${orders.entry_price:.6f}",
            'position_size': f"{orders.position_size*100:.1f}% of portfolio",
            'stop_loss': {
                'price': f"${orders.stop_loss:.6f}",
                'pct': f"{orders.stop_loss_pct*100:.1f}%",
                'trailing_enabled': orders.trailing_stop_enabled,
                'trailing_activation': f"{orders.trailing_stop_activation*100:.0f}%",
                'trailing_distance': f"{orders.trailing_stop_distance*100:.0f}%"
            },
            'take_profit_levels': [
                {
                    'level': tp['level'],
                    'price': f"${tp['price']:.6f}",
                    'gain': f"+{tp['gain_pct']*100:.1f}%",
                    'sell': f"{tp['size_pct']*100:.0f}% of position"
                }
                for tp in orders.tp_levels
            ],
            'max_duration': f"{orders.max_position_duration} minutes",
            'max_loss': f"${orders.max_loss_amount:.2f}",
            'strategy': {
                'aggressive_exit': orders.aggressive_exit,
                'hold_for_momentum': orders.hold_for_momentum
            },
            'model_data': {
                'predicted_gain': f"+{orders.predicted_gain*100:.1f}%",
                'confidence': f"{orders.confidence:.2f}"
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = DynamicOrderManager(
        initial_capital=10000,
        max_position_size=0.10,
        min_position_size=0.02
    )
    
    # Example token data
    token_data = {
        'signal_mc': 85000,
        'signal_liquidity': 28000,
        'signal_holders': 387,
        'signal_bundled_pct': 4.8,
        'signal_snipers_pct': 28.3,
        'signal_sold_pct': 6.2,
        'liq_to_mc_ratio': 0.33,
        'risk_score': 0.25
    }
    
    # Example model prediction
    model_prediction = {
        'predicted_gain': 1.50,  # 150%
        'confidence': 0.75
    }
    
    # Create initial orders
    orders = manager.create_initial_orders(
        token_data=token_data,
        model_prediction=model_prediction,
        entry_price=0.000123,
        token_id="TOKEN_123"
    )
    
    # Print summary
    print("\n=== INITIAL ORDER PARAMETERS ===")
    summary = manager.get_order_summary(orders)
    
    import json
    print(json.dumps(summary, indent=2))
    
    # Simulate price movements
    print("\n=== PRICE MOVEMENT SIMULATION ===")
    
    prices = [
        (0.000135, 5, "Early gain +9.7%"),
        (0.000156, 15, "Good momentum +26.8%"),
        (0.000189, 30, "Strong move +53.7%"),
        (0.000245, 60, "Moon shot +99.2%"),
    ]
    
    for price, time_elapsed, description in prices:
        print(f"\n{description}")
        print(f"  Price: ${price:.6f} | Time: {time_elapsed}min")
        
        updates = manager.update_orders(
            orders=orders,
            current_price=price,
            time_elapsed_minutes=time_elapsed,
            token_id="TOKEN_123"
        )
        
        print(f"  Gain: {updates['current_gain_pct']:.1f}%")
        print(f"  Stop Loss: ${updates['new_stop_loss']:.6f}")
        
        if updates['actions']:
            for action in updates['actions']:
                print(f"  🔔 Action: {action['type']} - {action.get('reason', '')}")
        
        if updates['should_exit']:
            print(f"  🚪 EXIT: {updates['reason']}")
            break


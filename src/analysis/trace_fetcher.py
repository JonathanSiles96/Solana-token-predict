"""
Trace_24H-Phillip Integration - External Trade Data Fetcher

This module fetches ground-truth price history from the Trace_24H-Phillip system,
which automatically trades tokens marked as BUY by our model and tracks their
24-hour performance for analysis and model improvement.
"""

import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sys
from pathlib import Path
import time
import traceback

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.database import SignalDatabase


class TraceDataFetcher:
    """
    Fetches trade data from Trace_24H-Phillip system
    
    The Trace_24H-Phillip system:
    - Buys every token marked as BUY by our /predict endpoint
    - Holds for 24 hours
    - Auto-closes for data collection
    - Provides full price-tick history via API
    
    This allows us to:
    - Get ground-truth outcomes for every prediction
    - Analyze prediction accuracy
    - Retrain model with real-world results
    - Identify model weaknesses
    """
    
    BASE_URL = "https://op.xcapitala.com/api"
    API_KEY = "dfhgdfrh45yery3463"
    
    def __init__(self, db_path: str = "data/signals.db"):
        """
        Initialize trace data fetcher
        
        Args:
            db_path: Path to signals database
        """
        self.db = SignalDatabase(db_path)
        self.headers = {
            "api-key": self.API_KEY,
            "Content-Type": "application/json"
        }
    
    def fetch_trade_price_history(self, trade_id: str) -> Optional[Dict]:
        """
        Fetch price-tick history for a specific trade
        
        Args:
            trade_id: Trade ID from Trace_24H system
            
        Returns:
            Dict with trade data and price updates, or None if error
            
        Example response:
        {
            "trade_id": "12345",
            "mint_key": "8K4VbtJ6...",
            "entry_time": "2025-12-01T10:00:00Z",
            "entry_price": 0.000123,
            "entry_mc": 100000,
            "exit_time": "2025-12-02T10:00:00Z",
            "exit_price": 0.000185,
            "exit_mc": 150000,
            "duration_hours": 24,
            "final_gain": 0.50,
            "price_updates": [
                {
                    "timestamp": "2025-12-01T10:00:00Z",
                    "price": 0.000123,
                    "mc": 100000,
                    "liquidity": 30000
                },
                {
                    "timestamp": "2025-12-01T10:05:00Z",
                    "price": 0.000135,
                    "mc": 110000,
                    "liquidity": 31000
                },
                ...
            ]
        }
        """
        try:
            url = f"{self.BASE_URL}/trades/{trade_id}/price-updates-external"
            
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response structure
            if not data or 'price_updates' not in data:
                print(f"⚠️  Invalid response structure for trade {trade_id}")
                return None
            
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"⚠️  Trade {trade_id} not found")
            else:
                print(f"✗ HTTP error fetching trade {trade_id}: {e}")
            return None
            
        except requests.exceptions.Timeout:
            print(f"✗ Timeout fetching trade {trade_id}")
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Request error fetching trade {trade_id}: {e}")
            return None
            
        except Exception as e:
            print(f"✗ Unexpected error fetching trade {trade_id}: {e}")
            traceback.print_exc()
            return None
    
    def calculate_trade_metrics(self, trade_data: Dict) -> Dict:
        """
        Calculate comprehensive metrics from trade data
        
        Args:
            trade_data: Trade data with price updates
            
        Returns:
            Dict with calculated metrics
        """
        price_updates = trade_data.get('price_updates', [])
        entry_mc = trade_data.get('entry_mc', 0)
        
        if not price_updates or entry_mc == 0:
            return {}
        
        # Extract market cap values
        mc_values = [update.get('mc', 0) for update in price_updates if update.get('mc', 0) > 0]
        
        if not mc_values:
            return {}
        
        # Calculate metrics
        max_mc = max(mc_values)
        min_mc = min(mc_values)
        final_mc = mc_values[-1]
        
        max_return = (max_mc - entry_mc) / entry_mc
        min_return = (min_mc - entry_mc) / entry_mc
        final_return = (final_mc - entry_mc) / entry_mc
        
        # Find when max return occurred
        max_mc_index = mc_values.index(max_mc)
        max_return_timestamp = price_updates[max_mc_index].get('timestamp')
        
        # Calculate volatility
        if len(mc_values) > 1:
            import numpy as np
            returns = [(mc_values[i] - mc_values[i-1]) / mc_values[i-1] 
                      for i in range(1, len(mc_values))]
            volatility = np.std(returns) if returns else 0
        else:
            volatility = 0
        
        # Time to max return (in hours)
        entry_time = datetime.fromisoformat(trade_data.get('entry_time', '').replace('Z', '+00:00'))
        max_time = datetime.fromisoformat(max_return_timestamp.replace('Z', '+00:00'))
        time_to_max_hours = (max_time - entry_time).total_seconds() / 3600
        
        # Calculate liquidity metrics
        liq_values = [update.get('liquidity', 0) for update in price_updates if update.get('liquidity', 0) > 0]
        avg_liquidity = sum(liq_values) / len(liq_values) if liq_values else 0
        max_liquidity = max(liq_values) if liq_values else 0
        min_liquidity = min(liq_values) if liq_values else 0
        
        return {
            'max_return': max_return,
            'max_return_mc': max_mc,
            'max_return_timestamp': max_return_timestamp,
            'min_return': min_return,
            'min_return_mc': min_mc,
            'final_return': final_return,
            'final_mc': final_mc,
            'time_to_max_hours': time_to_max_hours,
            'volatility': volatility,
            'avg_liquidity': avg_liquidity,
            'max_liquidity': max_liquidity,
            'min_liquidity': min_liquidity,
            'total_updates': len(price_updates),
            'duration_hours': trade_data.get('duration_hours', 24)
        }
    
    def store_trade_outcome(self, trade_data: Dict, metrics: Dict, 
                           prediction_data: Optional[Dict] = None) -> bool:
        """
        Store trade outcome in database for model retraining
        
        Args:
            trade_data: Raw trade data from API
            metrics: Calculated metrics
            prediction_data: Original prediction from model (if available)
            
        Returns:
            True if stored successfully
        """
        try:
            mint_key = trade_data.get('mint_key')
            entry_time = trade_data.get('entry_time')
            
            if not mint_key or not entry_time:
                print(f"⚠️  Missing mint_key or entry_time in trade data")
                return False
            
            # Parse entry time
            signal_at = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            
            # Prepare outcome data
            outcome = {
                'mint_key': mint_key,
                'signal_at': signal_at,
                'entry_price': trade_data.get('entry_price'),
                'exit_price': trade_data.get('exit_price'),
                'actual_gain': metrics.get('final_return', trade_data.get('final_gain', 0)),
                'max_return': metrics.get('max_return'),
                'max_return_mc': metrics.get('max_return_mc'),
                'max_return_timestamp': metrics.get('max_return_timestamp'),
                'exit_reason': 'trace_24h_auto_close'
            }
            
            # Store in trade_outcomes table
            self.db.insert_trade_outcome(outcome)
            
            # Also update max_return in token_signals table
            if metrics.get('max_return') is not None:
                self.db.update_max_return(
                    mint_key,
                    signal_at,
                    metrics['max_return'],
                    metrics['max_return_mc'],
                    metrics['max_return_timestamp']
                )
            
            print(f"✓ Stored outcome: {mint_key[:8]}... | "
                  f"Final: {metrics.get('final_return', 0)*100:.1f}% | "
                  f"Max: {metrics.get('max_return', 0)*100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"✗ Error storing trade outcome: {e}")
            traceback.print_exc()
            return False
    
    def process_trade(self, trade_id: str, 
                     prediction_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Fetch, analyze, and store a single trade
        
        Args:
            trade_id: Trade ID from Trace_24H system
            prediction_data: Original prediction from model
            
        Returns:
            Dict with processed trade data and metrics
        """
        print(f"\n{'='*80}")
        print(f"Processing Trade: {trade_id}")
        print(f"{'='*80}")
        
        # Fetch trade data
        trade_data = self.fetch_trade_price_history(trade_id)
        
        if not trade_data:
            print(f"✗ Failed to fetch trade data")
            return None
        
        # Calculate metrics
        metrics = self.calculate_trade_metrics(trade_data)
        
        if not metrics:
            print(f"✗ Failed to calculate metrics")
            return None
        
        # Store outcome
        success = self.store_trade_outcome(trade_data, metrics, prediction_data)
        
        if not success:
            print(f"✗ Failed to store outcome")
            return None
        
        # Combine for return
        result = {
            **trade_data,
            'calculated_metrics': metrics,
            'prediction': prediction_data
        }
        
        # Print summary
        print(f"\n📊 Trade Summary:")
        print(f"   Mint: {trade_data.get('mint_key', 'N/A')[:12]}...")
        print(f"   Entry MC: ${trade_data.get('entry_mc', 0):,.0f}")
        print(f"   Final Return: {metrics.get('final_return', 0)*100:.1f}%")
        print(f"   Max Return: {metrics.get('max_return', 0)*100:.1f}%")
        print(f"   Time to Max: {metrics.get('time_to_max_hours', 0):.1f}h")
        print(f"   Volatility: {metrics.get('volatility', 0):.4f}")
        print(f"   Price Updates: {metrics.get('total_updates', 0)}")
        
        if prediction_data:
            predicted_gain = prediction_data.get('predicted_gain', 0)
            actual_gain = metrics.get('max_return', 0)
            error = abs(predicted_gain - actual_gain)
            
            print(f"\n🎯 Prediction Accuracy:")
            print(f"   Predicted: {predicted_gain*100:.1f}%")
            print(f"   Actual Max: {actual_gain*100:.1f}%")
            print(f"   Error: {error*100:.1f}%")
            
            if actual_gain > predicted_gain:
                print(f"   ⬆️  Model underestimated by {(actual_gain - predicted_gain)*100:.1f}%")
            else:
                print(f"   ⬇️  Model overestimated by {(predicted_gain - actual_gain)*100:.1f}%")
        
        print(f"\n{'='*80}\n")
        
        return result
    
    def process_multiple_trades(self, trade_ids: List[str], 
                               delay_seconds: float = 1.0) -> List[Dict]:
        """
        Process multiple trades with rate limiting
        
        Args:
            trade_ids: List of trade IDs to process
            delay_seconds: Delay between requests (rate limiting)
            
        Returns:
            List of processed trade results
        """
        results = []
        
        print(f"\n🔄 Processing {len(trade_ids)} trades...")
        print(f"   Rate limit: {delay_seconds}s between requests\n")
        
        for i, trade_id in enumerate(trade_ids, 1):
            print(f"[{i}/{len(trade_ids)}] Processing trade {trade_id}...")
            
            result = self.process_trade(trade_id)
            
            if result:
                results.append(result)
            
            # Rate limiting (except for last request)
            if i < len(trade_ids):
                time.sleep(delay_seconds)
        
        print(f"\n✓ Processed {len(results)} / {len(trade_ids)} trades successfully")
        
        return results
    
    def get_prediction_accuracy_report(self) -> Dict:
        """
        Generate prediction accuracy report
        
        Compares model predictions to actual outcomes from Trace_24H trades
        
        Returns:
            Dict with accuracy metrics and analysis
        """
        try:
            # Get training data (signals with outcomes)
            df = self.db.get_training_data()
            
            if df.empty:
                return {
                    'status': 'no_data',
                    'message': 'No trade outcomes available yet'
                }
            
            import pandas as pd
            import numpy as np
            
            # Filter for trades with max_return data
            df_with_max = df[df['max_return'].notna()].copy()
            
            if len(df_with_max) == 0:
                return {
                    'status': 'no_max_return_data',
                    'message': 'No max_return data available yet'
                }
            
            # Calculate prediction errors
            # Note: We'd need to store predictions separately to compare
            # For now, calculate statistics on actual outcomes
            
            stats = {
                'total_trades': len(df),
                'trades_with_max_return': len(df_with_max),
                'avg_final_gain': df['final_gain'].mean(),
                'avg_max_return': df_with_max['max_return'].mean(),
                'median_max_return': df_with_max['max_return'].median(),
                'best_max_return': df_with_max['max_return'].max(),
                'worst_max_return': df_with_max['max_return'].min(),
                'win_rate': (df['final_gain'] >= 0.3).sum() / len(df) if len(df) > 0 else 0,
                'winners': (df['final_gain'] >= 0.3).sum(),
                'losers': (df['final_gain'] < 0.3).sum()
            }
            
            # Distribution
            stats['distribution'] = {
                'above_200_pct': (df_with_max['max_return'] >= 2.0).sum(),
                'above_100_pct': (df_with_max['max_return'] >= 1.0).sum(),
                'above_50_pct': (df_with_max['max_return'] >= 0.5).sum(),
                'above_30_pct': (df_with_max['max_return'] >= 0.3).sum(),
                'below_30_pct': (df_with_max['max_return'] < 0.3).sum()
            }
            
            return {
                'status': 'success',
                **stats
            }
            
        except Exception as e:
            print(f"✗ Error generating accuracy report: {e}")
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e)
            }


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("🚀 Trace_24H-Phillip Data Fetcher")
    print("="*80)
    
    # Initialize fetcher
    fetcher = TraceDataFetcher()
    
    # Example: Process a single trade
    # Replace with actual trade ID from Trace_24H system
    test_trade_id = "12345"
    
    print(f"\nTest: Fetching trade {test_trade_id}...")
    result = fetcher.process_trade(test_trade_id)
    
    if result:
        print("\n✅ Success! Trade data fetched and stored.")
    else:
        print("\n⚠️  Test trade not found (expected if using example ID)")
        print("    Replace test_trade_id with actual trade ID from Trace_24H system")
    
    # Example: Get accuracy report
    print("\n" + "="*80)
    print("Prediction Accuracy Report")
    print("="*80)
    
    report = fetcher.get_prediction_accuracy_report()
    
    if report['status'] == 'success':
        print(f"\n📊 Statistics:")
        print(f"   Total Trades: {report['total_trades']}")
        print(f"   With Max Return: {report['trades_with_max_return']}")
        print(f"   Win Rate: {report['win_rate']*100:.1f}%")
        print(f"   Avg Final Gain: {report['avg_final_gain']*100:.1f}%")
        print(f"   Avg Max Return: {report['avg_max_return']*100:.1f}%")
        print(f"   Best Max Return: {report['best_max_return']*100:.1f}%")
        
        print(f"\n📈 Distribution:")
        dist = report['distribution']
        print(f"   > 200%: {dist['above_200_pct']}")
        print(f"   > 100%: {dist['above_100_pct']}")
        print(f"   > 50%:  {dist['above_50_pct']}")
        print(f"   > 30%:  {dist['above_30_pct']}")
        print(f"   < 30%:  {dist['below_30_pct']}")
    else:
        print(f"\n⚠️  {report.get('message', 'No data available')}")
    
    print("\n" + "="*80)
    print("✓ Test complete!")


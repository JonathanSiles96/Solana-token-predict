"""
Price tracking service for monitoring max returns on signals

This service continuously monitors token prices after signals are generated
and automatically updates their max_return values.
"""

import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
import threading
import traceback
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.database import SignalDatabase


class PriceTracker:
    """
    Service to track token prices and update max returns
    
    Usage:
    ```python
    # Create tracker with your price fetching function
    def fetch_price(mint_key: str) -> Optional[Dict]:
        # Your logic to fetch current price and MC from API/blockchain
        return {"price": 0.00025, "mc": 180000}
    
    tracker = PriceTracker(
        fetch_price_func=fetch_price,
        check_interval=60,  # Check every 60 seconds
        max_age_hours=48     # Stop tracking after 48 hours
    )
    
    # Start tracking
    tracker.start()
    
    # Stop when done
    tracker.stop()
    ```
    """
    
    def __init__(self, 
                 fetch_price_func: Callable[[str], Optional[Dict]],
                 check_interval: int = 60,
                 max_age_hours: int = 48,
                 db_path: str = "data/signals.db"):
        """
        Initialize price tracker
        
        Args:
            fetch_price_func: Function that fetches price data for a mint_key
                             Should return dict with 'price' and 'mc' keys
            check_interval: Seconds between price checks (default 60)
            max_age_hours: Stop tracking signals older than this (default 48)
            db_path: Path to signals database
        """
        self.fetch_price = fetch_price_func
        self.check_interval = check_interval
        self.max_age_hours = max_age_hours
        self.db = SignalDatabase(db_path)
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self.stats = {
            'total_checks': 0,
            'successful_updates': 0,
            'errors': 0,
            'new_max_returns': 0
        }
    
    def start(self):
        """Start the price tracking service"""
        if self._running:
            print("⚠️  Price tracker already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()
        print(f"✓ Price tracker started (checking every {self.check_interval}s)")
    
    def stop(self):
        """Stop the price tracking service"""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        
        print("✓ Price tracker stopped")
        print(f"  Stats: {self.stats['total_checks']} checks, "
              f"{self.stats['successful_updates']} updates, "
              f"{self.stats['new_max_returns']} new max returns")
    
    def _tracking_loop(self):
        """Main tracking loop"""
        print(f"🔄 Starting price tracking loop...")
        
        while self._running:
            try:
                self._check_all_signals()
                time.sleep(self.check_interval)
            
            except Exception as e:
                print(f"✗ Error in tracking loop: {e}")
                traceback.print_exc()
                time.sleep(self.check_interval)
    
    def _check_all_signals(self):
        """Check prices for all active signals"""
        try:
            # Get active signals
            signals = self.db.get_signals_for_tracking(limit=500)
            
            if not signals:
                return
            
            print(f"\n📊 Checking {len(signals)} active signals...")
            
            for signal in signals:
                if not self._running:
                    break
                
                try:
                    self._check_signal(signal)
                except Exception as e:
                    print(f"✗ Error checking {signal['mint_key'][:8]}...: {e}")
                    self.stats['errors'] += 1
            
            print(f"✓ Batch complete. Stats: {self.stats}")
        
        except Exception as e:
            print(f"✗ Error checking signals: {e}")
            traceback.print_exc()
    
    def _check_signal(self, signal: Dict):
        """Check price for a single signal"""
        mint_key = signal['mint_key']
        signal_at = signal['signal_at']
        signal_mc = signal.get('signal_mc', 0)
        
        # Check if signal is too old
        signal_time = datetime.fromisoformat(signal_at.replace('Z', '+00:00'))
        age_hours = (datetime.now() - signal_time).total_seconds() / 3600
        
        if age_hours > self.max_age_hours:
            print(f"  ⏰ Signal too old ({age_hours:.1f}h), stopping tracking: {mint_key[:8]}...")
            self.db.stop_tracking_signal(mint_key, signal_at, reason='timeout')
            return
        
        # Fetch current price
        self.stats['total_checks'] += 1
        price_data = self.fetch_price(mint_key)
        
        if not price_data or not price_data.get('mc'):
            return
        
        current_mc = price_data['mc']
        current_price = price_data.get('price', 0)
        
        # Calculate current return
        if signal_mc and signal_mc > 0:
            current_return = (current_mc - signal_mc) / signal_mc
            
            # Get existing max_return
            current_max = signal.get('max_return')
            
            # Update tracking
            self.db.update_price_tracking(
                mint_key,
                signal_at,
                current_price,
                current_mc
            )
            self.stats['successful_updates'] += 1
            
            # Log if new max
            if current_max is None or current_return > current_max:
                self.stats['new_max_returns'] += 1
                print(f"  🎉 New max return: {mint_key[:8]}... | "
                      f"{current_return*100:.1f}% (was {current_max*100 if current_max else 0:.1f}%)")
    
    def check_signal_once(self, mint_key: str, signal_at: datetime) -> Dict:
        """
        Manually check a single signal (for testing)
        
        Returns:
            Dict with update status and current values
        """
        try:
            signal = self.db.get_signal_by_mint(mint_key)
            if not signal:
                return {"status": "error", "message": "Signal not found"}
            
            price_data = self.fetch_price(mint_key)
            if not price_data:
                return {"status": "error", "message": "Could not fetch price"}
            
            signal_mc = signal.get('signal_mc', 0)
            current_mc = price_data['mc']
            current_return = (current_mc - signal_mc) / signal_mc if signal_mc > 0 else 0
            
            self.db.update_price_tracking(
                mint_key,
                signal_at,
                price_data.get('price', 0),
                current_mc
            )
            
            return {
                "status": "success",
                "mint_key": mint_key,
                "signal_mc": signal_mc,
                "current_mc": current_mc,
                "current_return": current_return,
                "current_return_pct": current_return * 100
            }
        
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Example price fetcher (you'll need to implement your actual price API)
def example_price_fetcher(mint_key: str) -> Optional[Dict]:
    """
    Example price fetcher - replace with your actual implementation
    
    This should call your price API (e.g., DexScreener, Jupiter, etc.)
    and return current price and market cap.
    """
    try:
        # TODO: Implement your actual price fetching logic
        # Example using DexScreener API:
        # import requests
        # response = requests.get(f"https://api.dexscreener.com/latest/dex/tokens/{mint_key}")
        # data = response.json()
        # if data and 'pairs' in data and len(data['pairs']) > 0:
        #     pair = data['pairs'][0]
        #     return {
        #         'price': float(pair.get('priceUsd', 0)),
        #         'mc': float(pair.get('fdv', 0) or pair.get('marketCap', 0))
        #     }
        
        # Placeholder - returns None (no price available)
        return None
    
    except Exception as e:
        print(f"Error fetching price for {mint_key}: {e}")
        return None


if __name__ == "__main__":
    """
    Example usage - run this as a standalone service
    """
    print("🚀 Starting Price Tracker Service")
    print("=" * 60)
    print("NOTE: You need to implement your price fetching logic first!")
    print("      Edit example_price_fetcher() to use your price API")
    print("=" * 60)
    
    # Create tracker
    tracker = PriceTracker(
        fetch_price_func=example_price_fetcher,
        check_interval=60,      # Check every 60 seconds
        max_age_hours=48        # Track for 48 hours
    )
    
    # Start tracking
    try:
        tracker.start()
        
        # Keep running until interrupted
        while True:
            time.sleep(10)
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Shutting down...")
        tracker.stop()
        print("✓ Shutdown complete")


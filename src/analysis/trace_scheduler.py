"""
Background Scheduler for Automatic Trace_24H Trade Fetching

This service automatically fetches completed Trace_24H trades every hour
and processes them for model retraining.
"""

import time
import threading
import traceback
from datetime import datetime
from typing import Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.trace_fetcher import TraceDataFetcher
from src.api.database import SignalDatabase


class TraceScheduler:
    """
    Background scheduler for automatic Trace_24H trade fetching
    
    This service:
    - Runs in background thread
    - Checks for completed trades every hour
    - Automatically fetches and processes them
    - Triggers model retraining when thresholds met
    
    Usage:
    ```python
    scheduler = TraceScheduler(check_interval=3600)  # 1 hour
    scheduler.start()
    
    # Later...
    scheduler.stop()
    ```
    """
    
    def __init__(self, 
                 check_interval: int = 3600,
                 db_path: str = "data/signals.db"):
        """
        Initialize scheduler
        
        Args:
            check_interval: Seconds between checks (default 3600 = 1 hour)
            db_path: Path to signals database
        """
        self.check_interval = check_interval
        self.db = SignalDatabase(db_path)
        self.fetcher = TraceDataFetcher(db_path)
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self.stats = {
            'total_checks': 0,
            'trades_processed': 0,
            'trades_failed': 0,
            'last_check': None
        }
    
    def start(self):
        """Start the scheduler"""
        if self._running:
            print("⚠️  Trace scheduler already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        print(f"✓ Trace scheduler started (checking every {self.check_interval}s)")
    
    def stop(self):
        """Stop the scheduler"""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        
        print("✓ Trace scheduler stopped")
        print(f"  Stats: {self.stats['total_checks']} checks, "
              f"{self.stats['trades_processed']} processed, "
              f"{self.stats['trades_failed']} failed")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        print(f"🔄 Starting Trace_24H scheduler loop...")
        
        while self._running:
            try:
                self._check_and_fetch_trades()
                self.stats['last_check'] = datetime.now()
                
                # Sleep with periodic wake-up checks
                sleep_remaining = self.check_interval
                while sleep_remaining > 0 and self._running:
                    sleep_time = min(60, sleep_remaining)  # Wake every minute to check if stopped
                    time.sleep(sleep_time)
                    sleep_remaining -= sleep_time
            
            except Exception as e:
                print(f"✗ Error in scheduler loop: {e}")
                traceback.print_exc()
                time.sleep(60)  # Wait a minute before retrying
    
    def _check_and_fetch_trades(self):
        """Check for pending trades and fetch them"""
        try:
            self.stats['total_checks'] += 1
            
            # Get pending trades
            pending = self.db.get_pending_trace_trades(limit=50)
            
            if not pending:
                print(f"🔍 Scheduler check: No pending trades (check #{self.stats['total_checks']})")
                return
            
            print(f"\n{'='*80}")
            print(f"🔄 Trace Scheduler: Found {len(pending)} pending trades")
            print(f"{'='*80}\n")
            
            # Process each trade
            trade_ids = [t['trade_id'] for t in pending]
            results = self.fetcher.process_multiple_trades(trade_ids, delay_seconds=2.0)
            
            # Update stats
            self.stats['trades_processed'] += len(results)
            self.stats['trades_failed'] += len(pending) - len(results)
            
            # Update database for each successful trade
            for result in results:
                try:
                    trade_id = result.get('trade_id')
                    metrics = result['calculated_metrics']
                    
                    self.db.update_trace_trade_completion(
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
                    
                    print(f"✓ Trade {trade_id} completed and stored")
                    
                except Exception as e:
                    print(f"✗ Error updating trade {trade_id}: {e}")
                    self.stats['trades_failed'] += 1
            
            print(f"\n{'='*80}")
            print(f"✓ Scheduler: Processed {len(results)} / {len(pending)} trades")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"✗ Error checking trades: {e}")
            traceback.print_exc()
    
    def check_once(self):
        """Manually trigger a single check (for testing)"""
        print("\n🔄 Manual scheduler check triggered...")
        self._check_and_fetch_trades()
        print("✓ Manual check complete")
    
    def get_stats(self) -> dict:
        """Get scheduler statistics"""
        return {
            **self.stats,
            'running': self._running,
            'check_interval_seconds': self.check_interval
        }


if __name__ == "__main__":
    """
    Run as standalone service
    """
    print("🚀 Starting Trace_24H Background Scheduler")
    print("="*80)
    print("This service automatically fetches completed Trace_24H trades")
    print("Press Ctrl+C to stop")
    print("="*80)
    
    # Create scheduler (check every hour)
    scheduler = TraceScheduler(check_interval=3600)
    
    try:
        # Start scheduler
        scheduler.start()
        
        # Keep running
        while True:
            time.sleep(10)
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Shutting down...")
        scheduler.stop()
        print("✓ Shutdown complete")


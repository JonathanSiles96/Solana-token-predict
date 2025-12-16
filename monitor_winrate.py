# -*- coding: utf-8 -*-
"""
Real-time Win Rate Monitor
Shows live statistics and alerts
"""
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thresholds for alerts
ALERT_WIN_RATE_LOW = 75  # Alert if win rate drops below this
ALERT_CONSECUTIVE_LOSSES = 3  # Alert after this many consecutive losses

def load_trades():
    """Load tracked trades"""
    trades_file = "data/tracked_trades.json"
    if Path(trades_file).exists():
        with open(trades_file, 'r') as f:
            return json.load(f)
    return []

def get_recent_trades(trades, hours=24):
    """Get trades from last N hours"""
    cutoff = datetime.now() - timedelta(hours=hours)
    recent = []
    for t in trades:
        try:
            trade_time = datetime.fromisoformat(t['timestamp'])
            if trade_time > cutoff:
                recent.append(t)
        except:
            pass
    return recent

def calculate_live_stats(trades):
    """Calculate live statistics"""
    stats = {
        'total_signals': len(trades),
        'go_signals': 0,
        'skip_signals': 0,
        'completed': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0,
        'consecutive_losses': 0,
        'by_tier': {},
        'by_source': {},
    }
    
    go_trades = [t for t in trades if t.get('go_decision')]
    stats['go_signals'] = len(go_trades)
    stats['skip_signals'] = len(trades) - len(go_trades)
    
    completed = [t for t in go_trades if t.get('outcome')]
    stats['completed'] = len(completed)
    
    if completed:
        wins = [t for t in completed if t['outcome'] == 'win']
        losses = [t for t in completed if t['outcome'] == 'loss']
        stats['wins'] = len(wins)
        stats['losses'] = len(losses)
        stats['win_rate'] = len(wins) / len(completed) * 100 if completed else 0
        
        # Count consecutive losses from end
        consecutive = 0
        for t in reversed(completed):
            if t['outcome'] == 'loss':
                consecutive += 1
            else:
                break
        stats['consecutive_losses'] = consecutive
        
        # By tier
        for tier in ['TIER1', 'TIER2', 'TIER3', 'TIER4']:
            tier_completed = [t for t in completed if t.get('tier') == tier]
            if tier_completed:
                tier_wins = [t for t in tier_completed if t['outcome'] == 'win']
                stats['by_tier'][tier] = {
                    'total': len(tier_completed),
                    'wins': len(tier_wins),
                    'win_rate': len(tier_wins) / len(tier_completed) * 100
                }
        
        # By source
        sources = set(t.get('source', 'unknown') for t in completed)
        for source in sources:
            source_completed = [t for t in completed if t.get('source') == source]
            if source_completed:
                source_wins = [t for t in source_completed if t['outcome'] == 'win']
                stats['by_source'][source] = {
                    'total': len(source_completed),
                    'wins': len(source_wins),
                    'win_rate': len(source_wins) / len(source_completed) * 100
                }
    
    return stats

def display_dashboard(stats, hours=24):
    """Display live dashboard"""
    # Clear screen (works on most terminals)
    print("\033c", end="")
    
    print("="*60)
    print(f"  WIN RATE MONITOR - Last {hours} hours")
    print(f"  Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Main stats
    print(f"""
  Total Signals:    {stats['total_signals']}
  GO Decisions:     {stats['go_signals']}
  SKIP Decisions:   {stats['skip_signals']}
  
  Completed Trades: {stats['completed']}
  Wins:            {stats['wins']}
  Losses:          {stats['losses']}
  
  WIN RATE:        {stats['win_rate']:.1f}%  {'OK' if stats['win_rate'] >= 80 else 'LOW!'}
""")
    
    # Alerts
    alerts = []
    if stats['win_rate'] > 0 and stats['win_rate'] < ALERT_WIN_RATE_LOW:
        alerts.append(f"Win rate below {ALERT_WIN_RATE_LOW}%!")
    if stats['consecutive_losses'] >= ALERT_CONSECUTIVE_LOSSES:
        alerts.append(f"{stats['consecutive_losses']} consecutive losses!")
    
    if alerts:
        print("  " + "!"*50)
        print("  ALERTS:")
        for alert in alerts:
            print(f"    >> {alert}")
        print("  " + "!"*50)
    
    # By tier
    if stats['by_tier']:
        print("\n  BY TIER:")
        print("  " + "-"*40)
        for tier, data in sorted(stats['by_tier'].items()):
            wr = data['win_rate']
            status = "OK" if wr >= 80 else "LOW"
            print(f"    {tier}: {wr:.1f}% ({data['wins']}/{data['total']}) [{status}]")
    
    # By source
    if stats['by_source']:
        print("\n  BY SOURCE:")
        print("  " + "-"*40)
        for source, data in sorted(stats['by_source'].items(), key=lambda x: -x[1]['win_rate']):
            wr = data['win_rate']
            status = "OK" if wr >= 80 else "LOW"
            print(f"    {source}: {wr:.1f}% ({data['wins']}/{data['total']}) [{status}]")
    
    print("\n" + "="*60)
    print("  Press Ctrl+C to exit")
    print("="*60)

def run_monitor(hours=24, refresh=30):
    """Run live monitor"""
    print("Starting Win Rate Monitor...")
    print(f"Monitoring last {hours} hours, refreshing every {refresh} seconds")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            trades = load_trades()
            recent = get_recent_trades(trades, hours)
            stats = calculate_live_stats(recent)
            display_dashboard(stats, hours)
            time.sleep(refresh)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=int, default=24, help='Hours to look back')
    parser.add_argument('--refresh', type=int, default=30, help='Refresh interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    
    args = parser.parse_args()
    
    if args.once:
        trades = load_trades()
        recent = get_recent_trades(trades, args.hours)
        stats = calculate_live_stats(recent)
        display_dashboard(stats, args.hours)
    else:
        run_monitor(args.hours, args.refresh)


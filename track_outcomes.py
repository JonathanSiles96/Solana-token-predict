# -*- coding: utf-8 -*-
"""
Track trade outcomes and calculate real-time win rate
"""
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TRADES_FILE = "data/tracked_trades.json"

def load_trades():
    """Load tracked trades"""
    if Path(TRADES_FILE).exists():
        with open(TRADES_FILE, 'r') as f:
            return json.load(f)
    return []

def save_trades(trades):
    """Save tracked trades"""
    Path("data").mkdir(exist_ok=True)
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2, default=str)

def add_trade(mint_key, source, security, holders, tier, go_decision):
    """Add a new trade to track"""
    trades = load_trades()
    
    trade = {
        'id': len(trades) + 1,
        'mint_key': mint_key,
        'source': source,
        'security': security,
        'holders': holders,
        'tier': tier,
        'go_decision': go_decision,
        'timestamp': datetime.now().isoformat(),
        'outcome': None,  # Will be updated later
        'max_return': None,
        'notes': ''
    }
    
    trades.append(trade)
    save_trades(trades)
    print(f"Trade #{trade['id']} added: {mint_key} ({tier})")
    return trade['id']

def update_outcome(trade_id, outcome, max_return=None, notes=''):
    """Update trade outcome"""
    trades = load_trades()
    
    for trade in trades:
        if trade['id'] == trade_id:
            trade['outcome'] = outcome  # 'win', 'loss', 'breakeven'
            trade['max_return'] = max_return
            trade['notes'] = notes
            trade['outcome_time'] = datetime.now().isoformat()
            save_trades(trades)
            print(f"Trade #{trade_id} updated: {outcome}")
            return
    
    print(f"Trade #{trade_id} not found!")

def calculate_stats():
    """Calculate win rate statistics"""
    trades = load_trades()
    
    if not trades:
        print("No trades recorded yet!")
        return
    
    print("="*60)
    print("TRADE STATISTICS")
    print("="*60)
    
    # Total trades
    total = len(trades)
    go_trades = [t for t in trades if t['go_decision']]
    skip_trades = [t for t in trades if not t['go_decision']]
    
    print(f"\nTotal signals: {total}")
    print(f"GO decisions: {len(go_trades)}")
    print(f"SKIP decisions: {len(skip_trades)}")
    
    # Trades with outcomes
    completed = [t for t in trades if t['outcome'] is not None]
    
    if completed:
        go_completed = [t for t in completed if t['go_decision']]
        wins = [t for t in go_completed if t['outcome'] == 'win']
        losses = [t for t in go_completed if t['outcome'] == 'loss']
        
        if go_completed:
            win_rate = len(wins) / len(go_completed) * 100
            print(f"\n--- GO TRADES ---")
            print(f"Completed: {len(go_completed)}")
            print(f"Wins: {len(wins)}")
            print(f"Losses: {len(losses)}")
            print(f"WIN RATE: {win_rate:.1f}%")
            
            if wins:
                avg_win = sum(t['max_return'] or 1.5 for t in wins) / len(wins)
                print(f"Avg win return: {avg_win:.2f}x")
            if losses:
                avg_loss = sum(t['max_return'] or 0.7 for t in losses) / len(losses)
                print(f"Avg loss return: {avg_loss:.2f}x")
        
        # By tier
        print("\n--- BY TIER ---")
        for tier in ['TIER1', 'TIER2', 'TIER3', 'TIER4']:
            tier_trades = [t for t in go_completed if t['tier'] == tier]
            if tier_trades:
                tier_wins = [t for t in tier_trades if t['outcome'] == 'win']
                tier_wr = len(tier_wins) / len(tier_trades) * 100
                print(f"{tier}: {tier_wr:.1f}% ({len(tier_wins)}/{len(tier_trades)})")
        
        # Skip analysis
        skip_completed = [t for t in completed if not t['go_decision']]
        if skip_completed:
            skip_would_win = [t for t in skip_completed if t['outcome'] == 'win']
            print(f"\n--- SKIP ANALYSIS ---")
            print(f"Skipped signals that would have won: {len(skip_would_win)}/{len(skip_completed)}")
            if skip_would_win:
                print("(These are missed opportunities - consider adjusting filters)")
    else:
        print("\nNo completed trades yet. Update outcomes with:")
        print("  python track_outcomes.py --update <trade_id> --outcome win/loss")
    
    # Pending
    pending = [t for t in trades if t['outcome'] is None]
    if pending:
        print(f"\n--- PENDING ({len(pending)}) ---")
        for t in pending[-5:]:  # Show last 5
            print(f"  #{t['id']}: {t['mint_key'][:20]}... ({t['tier']})")
    
    print("\n" + "="*60)

def show_recent(n=10):
    """Show recent trades"""
    trades = load_trades()
    
    print(f"\nLast {n} trades:")
    print("-"*80)
    
    for trade in trades[-n:]:
        status = trade['outcome'] or 'PENDING'
        print(f"#{trade['id']} | {trade['tier']} | {trade['source']} | {status} | {trade['mint_key'][:20]}...")
    
    print("-"*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--add', nargs=6, metavar=('MINT', 'SOURCE', 'SECURITY', 'HOLDERS', 'TIER', 'GO'),
                       help='Add trade: mint source security holders tier go(true/false)')
    parser.add_argument('--update', type=int, help='Trade ID to update')
    parser.add_argument('--outcome', choices=['win', 'loss', 'breakeven'], help='Trade outcome')
    parser.add_argument('--return', type=float, dest='max_return', help='Max return (e.g., 1.5 for 50% gain)')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--recent', type=int, default=10, help='Show recent trades')
    
    args = parser.parse_args()
    
    if args.add:
        mint, source, security, holders, tier, go = args.add
        add_trade(mint, source, security, int(holders), tier, go.lower() == 'true')
    elif args.update and args.outcome:
        update_outcome(args.update, args.outcome, args.max_return)
    elif args.stats:
        calculate_stats()
    else:
        show_recent(args.recent)
        calculate_stats()


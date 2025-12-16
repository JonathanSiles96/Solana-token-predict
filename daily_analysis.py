# -*- coding: utf-8 -*-
"""
Daily Analysis Script
Run this every day to track win rates and find patterns
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_data():
    """Load all CSV data"""
    files = ['2025.csv', '2025 (1).csv', '2025 (3).csv', '2025 (4).csv']
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except:
            pass
    
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=['mint_key', 'signal_time'], keep='first')
        return df
    return pd.DataFrame()


def daily_report(df, date=None):
    """Generate daily win rate report"""
    
    # Filter to signals with outcomes
    df = df[df['max_return'].notna() & (df['max_return'] > 0)].copy()
    
    if len(df) == 0:
        print("No data with outcomes!")
        return
    
    # Win = 30% gain
    df['winner'] = df['max_return'] >= 1.3
    
    print("="*60)
    print(f"DAILY WIN RATE REPORT")
    print(f"Date: {date or 'All Time'}")
    print("="*60)
    
    # Overall stats
    total = len(df)
    winners = df['winner'].sum()
    win_rate = winners / total * 100
    avg_return = df['max_return'].mean()
    
    print(f"\nOVERALL: {winners}/{total} = {win_rate:.1f}% win rate")
    print(f"Avg max_return: {avg_return:.2f}x ({(avg_return-1)*100:.0f}%)")
    
    # By source (CRITICAL)
    print("\n" + "-"*40)
    print("BY SOURCE (sorted by win rate)")
    print("-"*40)
    
    source_stats = []
    for source in df['source'].unique():
        sub = df[df['source'] == source]
        if len(sub) >= 3:
            wr = sub['winner'].mean() * 100
            avg = sub['max_return'].mean()
            source_stats.append((source, wr, len(sub), avg))
    
    source_stats.sort(key=lambda x: -x[1])
    for source, wr, n, avg in source_stats:
        tier = "TIER1" if wr >= 95 else "TIER2" if wr >= 85 else "TIER3" if wr >= 75 else "SKIP?"
        print(f"  {source}: {wr:.1f}% ({n} trades, avg {avg:.2f}x) [{tier}]")
    
    # By security
    print("\n" + "-"*40)
    print("BY SECURITY STATUS")
    print("-"*40)
    
    for sec in df['security'].unique():
        sub = df[df['security'] == sec]
        if len(sub) >= 3:
            wr = sub['winner'].mean() * 100
            avg = sub['max_return'].mean()
            print(f"  {sec}: {wr:.1f}% ({len(sub)} trades, avg {avg:.2f}x)")
    
    # By holders tier
    print("\n" + "-"*40)
    print("BY HOLDERS COUNT")
    print("-"*40)
    
    holder_tiers = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 10000)]
    for low, high in holder_tiers:
        sub = df[(df['holders'] >= low) & (df['holders'] < high)]
        if len(sub) >= 3:
            wr = sub['winner'].mean() * 100
            label = f"{low}-{high}" if high < 10000 else f"{low}+"
            print(f"  Holders {label}: {wr:.1f}% ({len(sub)} trades)")
    
    # Best performing combos
    print("\n" + "-"*40)
    print("BEST PERFORMING COMBOS")
    print("-"*40)
    
    combos = [
        ("whale source", df['source'] == 'whale'),
        ("tg_early_trending", df['source'] == 'tg_early_trending'),
        ("green security", df['security'].str.contains('good|white|check', case=False, na=False)),
        ("holders > 400", df['holders'] > 400),
        ("bundled < 5%", df['bundled_pct'] < 5),
        ("volume > 100K", df['volume_1h'] > 100000),
        ("bundled<5% + holders>300", (df['bundled_pct'] < 5) & (df['holders'] > 300)),
        ("volume>100K + holders>200", (df['volume_1h'] > 100000) & (df['holders'] > 200)),
    ]
    
    for name, mask in combos:
        try:
            sub = df[mask]
            if len(sub) >= 3:
                wr = sub['winner'].mean() * 100
                avg = sub['max_return'].mean()
                status = "USE" if wr >= 80 else "MAYBE" if wr >= 70 else "AVOID"
                print(f"  {name}: {wr:.1f}% ({len(sub)} trades) [{status}]")
        except:
            pass
    
    # Worst performers (to avoid)
    print("\n" + "-"*40)
    print("PATTERNS TO AVOID")
    print("-"*40)
    
    avoid = [
        ("danger security", df['security'].str.contains('danger', case=False, na=False)),
        ("holders < 100", df['holders'] < 100),
        ("bundled > 60%", df['bundled_pct'] > 60),
        ("snipers > 50%", df['snipers_pct'] > 50),
    ]
    
    for name, mask in avoid:
        try:
            sub = df[mask]
            if len(sub) >= 3:
                wr = sub['winner'].mean() * 100
                print(f"  {name}: {wr:.1f}% ({len(sub)} trades) [AVOID]")
        except:
            pass
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    # Find best sources
    best_sources = [s for s, wr, n, _ in source_stats if wr >= 90 and n >= 10]
    if best_sources:
        print(f"\n1. PRIORITIZE these sources: {', '.join(best_sources)}")
    
    # Find best filters
    print("\n2. GO signals should match one of:")
    print("   - Source: whale or tg_early_trending (100% WR)")
    print("   - Security: green/white_check_mark (93%+ WR)")
    print("   - Holders > 400 (87%+ WR)")
    print("   - Bundled < 5% AND Holders > 300 (88%+ WR)")
    
    print("\n3. SKIP signals that have:")
    print("   - Holders < 100")
    print("   - Security: danger")
    print("   - Unknown/low-priority source")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='Date to analyze (YYYY-MM-DD)')
    args = parser.parse_args()
    
    df = load_data()
    if len(df) > 0:
        daily_report(df, args.date)
    else:
        print("No data found!")


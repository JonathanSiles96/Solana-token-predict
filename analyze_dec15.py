# -*- coding: utf-8 -*-
"""Analyze Dec 15th trades to understand losses"""
import pandas as pd
import numpy as np
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load all data
files = ['2025 (5).csv', '2025 (6).csv']
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f'Loaded {f}: {len(df)} rows')
    except Exception as e:
        print(f'Error loading {f}: {e}')

if not dfs:
    print("No data!")
    exit()

df = pd.concat(dfs, ignore_index=True)
print(f'\nTotal rows: {len(df)}')

# Parse dates
df['signal_dt'] = pd.to_datetime(df['signal_time'], errors='coerce')
df['date'] = df['signal_dt'].dt.date

print(f'Date range: {df["date"].min()} to {df["date"].max()}')

# Filter to Dec 15
dec15 = df[df['date'] == pd.to_datetime('2025-12-15').date()].copy()
print(f'\nDec 15 signals: {len(dec15)}')

if len(dec15) == 0:
    # Try Dec 14
    dec15 = df[df['date'] == pd.to_datetime('2025-12-14').date()].copy()
    print(f'Trying Dec 14: {len(dec15)} signals')

if len(dec15) == 0:
    print("No data for Dec 15 or 14!")
    # Show available dates
    print("\nAvailable dates:")
    print(df['date'].value_counts().head(10))
    exit()

# Analyze
print("\n" + "="*60)
print("DECEMBER 15 ANALYSIS")
print("="*60)

# With outcomes
with_outcome = dec15[dec15['max_return'].notna() & (dec15['max_return'] > 0)]
print(f'\nSignals with outcomes: {len(with_outcome)}')

if len(with_outcome) > 0:
    # Define winner
    with_outcome = with_outcome.copy()
    with_outcome['winner'] = with_outcome['max_return'] >= 1.3  # 30% gain
    
    winners = with_outcome['winner'].sum()
    total = len(with_outcome)
    print(f'Winners (>=30% gain): {winners}/{total} = {winners/total*100:.1f}%')
    print(f'Avg max_return: {with_outcome["max_return"].mean():.2f}x')
    
    # BUT - this is max_return, not actual trading result!
    print("\n" + "-"*40)
    print("CRITICAL: max_return vs ACTUAL result")
    print("-"*40)
    
    # Look at the actual trading_score JSON to see go_decision
    print("\nAnalyzing GO decisions...")
    
    go_signals = []
    skip_signals = []
    
    for idx, row in with_outcome.iterrows():
        try:
            if pd.notna(row.get('trading_score')):
                import json
                scores = json.loads(row['trading_score'])
                if scores and len(scores) > 0:
                    go = scores[0].get('go_decision', False)
                    if go:
                        go_signals.append(row)
                    else:
                        skip_signals.append(row)
        except:
            pass
    
    print(f"GO signals: {len(go_signals)}")
    print(f"SKIP signals: {len(skip_signals)}")
    
    if go_signals:
        go_df = pd.DataFrame(go_signals)
        go_winners = go_df['winner'].sum()
        print(f"\nGO signals win rate: {go_winners}/{len(go_df)} = {go_winners/len(go_df)*100:.1f}%")
        print(f"GO avg max_return: {go_df['max_return'].mean():.2f}x")
        
        # Show GO signals that LOST
        go_losers = go_df[~go_df['winner']]
        print(f"\nGO signals that LOST (max_return < 1.3): {len(go_losers)}")
        
        if len(go_losers) > 0:
            print("\nLOSING GO SIGNALS:")
            print("-"*60)
            for idx, row in go_losers.head(10).iterrows():
                print(f"  {row['name']}: max_return={row['max_return']:.2f}x, "
                      f"source={row['source']}, security={row['security']}, "
                      f"holders={row['holders']}, bundled={row.get('bundled_pct', 0):.0f}%")
    
    if skip_signals:
        skip_df = pd.DataFrame(skip_signals)
        skip_winners = skip_df['winner'].sum()
        print(f"\nSKIP signals that would have WON: {skip_winners}/{len(skip_df)}")
    
    # The REAL issue: What about signals that went DOWN before reaching max?
    print("\n" + "-"*40)
    print("REAL ISSUE: Stop Loss Hit Before Max Return")
    print("-"*40)
    
    # If max_return is 2.0x but price dropped 40% first, you'd be stopped out!
    # We need to check if there's drawdown data
    
    print("\nmax_return distribution:")
    print(f"  < 0.7x (lost >30%): {(with_outcome['max_return'] < 0.7).sum()}")
    print(f"  0.7-0.9x (small loss): {((with_outcome['max_return'] >= 0.7) & (with_outcome['max_return'] < 0.9)).sum()}")
    print(f"  0.9-1.1x (breakeven): {((with_outcome['max_return'] >= 0.9) & (with_outcome['max_return'] < 1.1)).sum()}")
    print(f"  1.1-1.3x (small win): {((with_outcome['max_return'] >= 1.1) & (with_outcome['max_return'] < 1.3)).sum()}")
    print(f"  1.3-2.0x (good win): {((with_outcome['max_return'] >= 1.3) & (with_outcome['max_return'] < 2.0)).sum()}")
    print(f"  > 2.0x (great win): {(with_outcome['max_return'] >= 2.0).sum()}")

# By source on Dec 15
print("\n" + "-"*40)
print("BY SOURCE (Dec 15)")
print("-"*40)

for source in with_outcome['source'].unique():
    sub = with_outcome[with_outcome['source'] == source]
    if len(sub) >= 1:
        wins = sub['winner'].sum()
        avg = sub['max_return'].mean()
        print(f"  {source}: {wins}/{len(sub)} wins ({wins/len(sub)*100:.0f}%), avg return {avg:.2f}x")

print("\n" + "="*60)


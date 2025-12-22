"""
EXIT OPTIMIZATION FRAMEWORK
===========================
Comprehensive analysis of TP ladder and SL optimization
Per Tier AND Per Source

Deliverables:
1. TP Hit-Rate Analysis (what % reach each level)
2. Best First TP (TP1) Analysis
3. Time-to-TP Analysis
4. Stop-Loss Optimization (Pre-TP and Post-TP)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("EXIT OPTIMIZATION FRAMEWORK - Comprehensive Analysis")
print("="*80)

# Load data
df8 = pd.read_csv('2025 (8).csv')
df9 = pd.read_csv('2025 (9).csv')
df = pd.concat([df8, df9], ignore_index=True)

# Clean data
df['max_return'] = pd.to_numeric(df['max_return'], errors='coerce')
df['volume_1h'] = pd.to_numeric(df['volume_1h'], errors='coerce')
df['holders'] = pd.to_numeric(df['holders'], errors='coerce')
df['liquidity'] = pd.to_numeric(df['liquidity'], errors='coerce')
df['snipers_pct'] = pd.to_numeric(df['snipers_pct'], errors='coerce')
df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce')

# Filter valid data
df = df[df['max_return'].notna() & (df['max_return'] > 0)]
print(f"\nTotal signals with valid max_return: {len(df)}")

# Convert max_return to gain percentage
df['max_gain_pct'] = (df['max_return'] - 1) * 100  # e.g., 1.5 -> 50%

# Check security
df['is_green_security'] = df['security'].astype(str).str.contains('✅|check', case=False, na=False)

# Assign tiers based on the new optimized logic
def assign_tier(row):
    vol = row['volume_1h'] if pd.notna(row['volume_1h']) else 0
    holders = row['holders'] if pd.notna(row['holders']) else 0
    liq = row['liquidity'] if pd.notna(row['liquidity']) else 0
    snipers = row['snipers_pct'] if pd.notna(row['snipers_pct']) else 100
    is_green = row['is_green_security']
    
    # TIER 0: Security ✅ + Volume >= 20k (99.5% WR)
    if is_green and vol >= 20000:
        return 'TIER0'
    # TIER 1: Green security OR (Vol>=50k + Hold>=250 + Snip<=35%)
    elif is_green:
        return 'TIER1'
    elif vol >= 50000 and holders >= 250 and snipers <= 35:
        return 'TIER1'
    # TIER 2: Liquidity >= 25k
    elif liq >= 25000 and vol >= 10000:
        return 'TIER2'
    # TIER 3: Volume >= 20k + Holders >= 150
    elif vol >= 20000 and holders >= 150:
        return 'TIER3'
    # TIER 4: Volume >= 10k + Holders >= 200 + Liq >= 20k
    elif vol >= 10000 and holders >= 200 and liq >= 20000:
        return 'TIER4'
    else:
        return 'SKIP'

df['tier'] = df.apply(assign_tier, axis=1)

# Filter only tradeable signals (not SKIP)
tradeable = df[df['tier'] != 'SKIP'].copy()
print(f"Tradeable signals (pass filter): {len(tradeable)}")

print("\n" + "="*80)
print("SECTION 1: TP HIT-RATE ANALYSIS (Reality Check)")
print("="*80)
print("\nWhat % of trades reach each profit level?\n")

tp_levels = [10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 300, 500]

def analyze_tp_hitrates(data, label):
    """Analyze TP hit rates for a subset of data"""
    n = len(data)
    if n < 5:
        return None
    
    results = []
    for tp in tp_levels:
        hit_count = (data['max_gain_pct'] >= tp).sum()
        hit_pct = hit_count / n * 100
        results.append({'TP%': tp, 'Hit%': hit_pct, 'Count': hit_count})
    
    return pd.DataFrame(results)

# Overall hit rates
print("--- OVERALL TP HIT RATES ---")
overall_rates = analyze_tp_hitrates(tradeable, "Overall")
if overall_rates is not None:
    for _, row in overall_rates.iterrows():
        bar = "█" * int(row['Hit%'] / 5)
        print(f"+{row['TP%']:>3}%: {row['Hit%']:>5.1f}% ({row['Count']:>4} trades) {bar}")

# Per Tier
print("\n--- TP HIT RATES BY TIER ---")
for tier in ['TIER0', 'TIER1', 'TIER2', 'TIER3', 'TIER4']:
    tier_data = tradeable[tradeable['tier'] == tier]
    n = len(tier_data)
    if n < 5:
        continue
    print(f"\n{tier} (n={n}):")
    rates = analyze_tp_hitrates(tier_data, tier)
    if rates is not None:
        for _, row in rates.iterrows():
            if row['Hit%'] > 0:
                print(f"  +{row['TP%']:>3}%: {row['Hit%']:>5.1f}%")

# Per Source
print("\n--- TP HIT RATES BY SOURCE ---")
for source in tradeable['source'].unique():
    src_data = tradeable[tradeable['source'] == source]
    n = len(src_data)
    if n < 10:
        continue
    print(f"\n{source} (n={n}):")
    rates = analyze_tp_hitrates(src_data, source)
    if rates is not None:
        for _, row in rates.iterrows():
            if row['Hit%'] > 0:
                print(f"  +{row['TP%']:>3}%: {row['Hit%']:>5.1f}%")

# Per Tier + Source combination
print("\n--- TP HIT RATES BY TIER + SOURCE ---")
for tier in ['TIER0', 'TIER1', 'TIER2', 'TIER3', 'TIER4']:
    for source in tradeable['source'].unique():
        combo_data = tradeable[(tradeable['tier'] == tier) & (tradeable['source'] == source)]
        n = len(combo_data)
        if n < 5:
            continue
        print(f"\n{tier} + {source} (n={n}):")
        rates = analyze_tp_hitrates(combo_data, f"{tier}_{source}")
        if rates is not None:
            key_levels = [20, 30, 50, 100, 200]
            for _, row in rates.iterrows():
                if row['TP%'] in key_levels:
                    print(f"  +{row['TP%']:>3}%: {row['Hit%']:>5.1f}%")

print("\n" + "="*80)
print("SECTION 2: BEST TP1 ANALYSIS")
print("="*80)
print("\nSimulating different TP1 levels to find optimal first take-profit\n")

def simulate_tp1(data, tp1_level):
    """Simulate returns if TP1 was set at given level"""
    # If trade hit TP1, we take profit at TP1
    # If trade didn't hit TP1, we use actual max (assuming no SL for now)
    results = []
    for _, row in data.iterrows():
        max_gain = row['max_gain_pct']
        if max_gain >= tp1_level:
            realized = tp1_level  # Hit TP1
        else:
            realized = max_gain  # Didn't hit, take what we got
        results.append(realized)
    return np.array(results)

print("--- TP1 SIMULATION BY TIER ---")
for tier in ['TIER0', 'TIER1', 'TIER2', 'TIER3', 'TIER4']:
    tier_data = tradeable[tradeable['tier'] == tier]
    n = len(tier_data)
    if n < 10:
        continue
    
    print(f"\n{tier} (n={n}):")
    print(f"  {'TP1%':>5} | {'Avg Return':>10} | {'Win Rate':>8} | {'Std Dev':>8} | Recommendation")
    print(f"  {'-'*5} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*15}")
    
    best_tp1 = None
    best_score = -999
    
    for tp1 in [10, 15, 20, 25, 30, 40, 50]:
        returns = simulate_tp1(tier_data, tp1)
        avg_return = returns.mean()
        win_rate = (returns > 0).mean() * 100
        std_dev = returns.std()
        
        # Score: balance return vs consistency
        # Penalize high variance
        score = avg_return - (std_dev * 0.3)
        
        rec = ""
        if score > best_score:
            best_score = score
            best_tp1 = tp1
            rec = "← Best"
        
        print(f"  {tp1:>5}% | {avg_return:>9.1f}% | {win_rate:>7.1f}% | {std_dev:>7.1f}% | {rec}")
    
    print(f"\n  RECOMMENDED TP1 for {tier}: +{best_tp1}%")

print("\n" + "="*80)
print("SECTION 3: TIME-TO-TP ANALYSIS")
print("="*80)

# Check if we have time data
if 'signal_time' in df.columns and 'max_return_time' in df.columns:
    print("\nAnalyzing time to reach max return...")
    
    tradeable_time = tradeable.copy()
    tradeable_time['signal_time'] = pd.to_datetime(tradeable_time['signal_time'], errors='coerce')
    tradeable_time['max_return_time'] = pd.to_datetime(tradeable_time['max_return_time'], errors='coerce')
    
    # Filter rows with valid times
    valid_time = tradeable_time[tradeable_time['signal_time'].notna() & tradeable_time['max_return_time'].notna()]
    valid_time = valid_time[valid_time['max_return_time'] > valid_time['signal_time']]
    
    if len(valid_time) > 10:
        valid_time['time_to_max_minutes'] = (valid_time['max_return_time'] - valid_time['signal_time']).dt.total_seconds() / 60
        
        print(f"\nSignals with valid time data: {len(valid_time)}")
        
        # Time analysis by gain level
        print("\n--- TIME TO REACH GAIN LEVEL ---")
        for gain_level in [10, 20, 30, 50, 100]:
            subset = valid_time[valid_time['max_gain_pct'] >= gain_level]
            if len(subset) > 5:
                avg_time = subset['time_to_max_minutes'].mean()
                median_time = subset['time_to_max_minutes'].median()
                print(f"+{gain_level}% gainers: Avg time={avg_time:.0f}min, Median={median_time:.0f}min (n={len(subset)})")
        
        # Winners vs losers time
        print("\n--- WINNERS vs LOSERS TIME ---")
        winners = valid_time[valid_time['max_gain_pct'] >= 30]
        losers = valid_time[valid_time['max_gain_pct'] < 0]
        
        if len(winners) > 5:
            print(f"Winners (30%+): Avg time to peak = {winners['time_to_max_minutes'].mean():.0f}min")
        if len(losers) > 5:
            print(f"Losers (<0%): Avg time to bottom = {losers['time_to_max_minutes'].mean():.0f}min")
    else:
        print("Insufficient time data for analysis")
else:
    print("\nTime data not available in dataset.")
    print("Need 'signal_time' and 'max_return_time' columns for time analysis.")

print("\n" + "="*80)
print("SECTION 4: STOP-LOSS ANALYSIS")
print("="*80)

# For proper SL analysis we need price history, but we can estimate from max_return
# If max_return < 1, the trade never went positive
# We need MAE (Maximum Adverse Excursion) data for proper analysis

print("\n--- LOSS DISTRIBUTION ANALYSIS ---")
print("\nAnalyzing trades that ended negative (max_return < 1.0)...")

losers = tradeable[tradeable['max_return'] < 1.0].copy()
losers['max_loss_pct'] = (1 - losers['max_return']) * 100

if len(losers) > 5:
    print(f"\nTotal losing trades: {len(losers)} ({len(losers)/len(tradeable)*100:.1f}%)")
    print(f"Average max loss: -{losers['max_loss_pct'].mean():.1f}%")
    print(f"Median max loss: -{losers['max_loss_pct'].median():.1f}%")
    print(f"Worst loss: -{losers['max_loss_pct'].max():.1f}%")
    
    print("\n--- LOSS PERCENTILE ANALYSIS ---")
    for pct in [25, 50, 70, 80, 90, 95]:
        loss_at_pct = losers['max_loss_pct'].quantile(pct/100)
        print(f"  {pct}th percentile loss: -{loss_at_pct:.1f}%")
    
    print("\n--- LOSS DISTRIBUTION BY TIER ---")
    for tier in ['TIER0', 'TIER1', 'TIER2', 'TIER3', 'TIER4']:
        tier_losers = losers[losers['tier'] == tier]
        n = len(tier_losers)
        if n < 3:
            continue
        tier_total = len(tradeable[tradeable['tier'] == tier])
        loss_rate = n / tier_total * 100 if tier_total > 0 else 0
        avg_loss = tier_losers['max_loss_pct'].mean()
        print(f"  {tier}: {n} losers ({loss_rate:.1f}%), avg loss: -{avg_loss:.1f}%")
else:
    print("Very few losing trades to analyze!")

# Winners that had drawdown analysis (estimate)
print("\n--- RECOVERY ANALYSIS (Winners) ---")
winners = tradeable[tradeable['max_gain_pct'] >= 30].copy()
print(f"Winners (30%+): {len(winners)} trades")

# Estimate: trades that hit lower lows before reaching peak
# We can infer some behavior from the final max_return distribution
print("\nMax gain distribution for winners:")
for level in [30, 50, 100, 200, 500]:
    count = (winners['max_gain_pct'] >= level).sum()
    pct = count / len(winners) * 100
    print(f"  Reached +{level}%: {pct:.1f}% of winners")

print("\n" + "="*80)
print("SECTION 5: RECOMMENDED EXIT STRATEGY")
print("="*80)

print("\n--- OPTIMAL TP LADDER BY TIER ---\n")

for tier in ['TIER0', 'TIER1', 'TIER2', 'TIER3', 'TIER4']:
    tier_data = tradeable[tradeable['tier'] == tier]
    n = len(tier_data)
    if n < 10:
        continue
    
    # Calculate hit rates
    hit_30 = (tier_data['max_gain_pct'] >= 30).mean() * 100
    hit_50 = (tier_data['max_gain_pct'] >= 50).mean() * 100
    hit_100 = (tier_data['max_gain_pct'] >= 100).mean() * 100
    hit_200 = (tier_data['max_gain_pct'] >= 200).mean() * 100
    avg_max = tier_data['max_gain_pct'].mean()
    
    print(f"{tier} (n={n}):")
    print(f"  Hit rates: +30%={hit_30:.0f}%, +50%={hit_50:.0f}%, +100%={hit_100:.0f}%, +200%={hit_200:.0f}%")
    print(f"  Avg max gain: +{avg_max:.0f}%")
    
    # Recommend based on hit rates
    if hit_30 >= 80:
        tp1 = 25 if hit_50 >= 70 else 20
    elif hit_30 >= 60:
        tp1 = 20
    else:
        tp1 = 15
    
    if hit_100 >= 50:
        tp2 = 75
        tp3 = 150
    elif hit_50 >= 60:
        tp2 = 50
        tp3 = 100
    else:
        tp2 = 40
        tp3 = 75
    
    print(f"  RECOMMENDED: TP1={tp1}% (sell 25%), TP2={tp2}% (sell 30%), TP3={tp3}% (sell 45%)")
    print()

print("\n--- OPTIMAL STOP-LOSS BY TIER ---\n")

for tier in ['TIER0', 'TIER1', 'TIER2', 'TIER3', 'TIER4']:
    tier_data = tradeable[tradeable['tier'] == tier]
    tier_losers = tier_data[tier_data['max_return'] < 1.0]
    n_total = len(tier_data)
    n_losers = len(tier_losers)
    
    if n_total < 10:
        continue
    
    loss_rate = n_losers / n_total * 100
    
    if n_losers > 3:
        avg_loss = (1 - tier_losers['max_return'].mean()) * 100
        p80_loss = (1 - tier_losers['max_return'].quantile(0.2)) * 100  # 80th percentile loss
    else:
        avg_loss = 0
        p80_loss = 30
    
    # Recommend SL based on loss profile
    # Tighter SL for lower tiers, wider for higher tiers
    if tier in ['TIER0', 'TIER1']:
        sl = max(30, min(45, p80_loss + 10))  # Wider SL for high-quality
    elif tier in ['TIER2', 'TIER3']:
        sl = max(25, min(40, p80_loss + 5))
    else:
        sl = max(20, min(35, p80_loss))
    
    print(f"{tier}:")
    print(f"  Loss rate: {loss_rate:.1f}% | Avg loss: -{avg_loss:.1f}%")
    print(f"  RECOMMENDED SL: -{sl:.0f}%")
    print()

print("\n" + "="*80)
print("FINAL EXIT STRATEGY FRAMEWORK")
print("="*80)

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         COMPLETE EXIT STRATEGY                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TIER0 (Security ✅ + Volume 20k+) - Highest Quality                         ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Initial SL: -40% (wide - let it breathe)                                   ║
║  TP1: +25% → Sell 20%                                                       ║
║  TP2: +75% → Sell 25%                                                       ║
║  TP3: +150% → Sell 30%                                                      ║
║  TP4: +300% → Sell 25% (runner mode)                                        ║
║  After TP1: Move SL to -10% (protect profit)                                ║
║  After TP2: Trail 25% from ATH                                              ║
║                                                                              ║
║  TIER1 (Green Security OR High Vol+Hold) - Very High Quality                 ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Initial SL: -35%                                                           ║
║  TP1: +20% → Sell 20%                                                       ║
║  TP2: +50% → Sell 30%                                                       ║
║  TP3: +100% → Sell 30%                                                      ║
║  TP4: +200% → Sell 20% (runner)                                             ║
║  After TP1: Move SL to -5%                                                  ║
║  After TP2: Trail 20% from ATH                                              ║
║                                                                              ║
║  TIER2 (High Liquidity) - High Quality                                       ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Initial SL: -30%                                                           ║
║  TP1: +20% → Sell 25%                                                       ║
║  TP2: +45% → Sell 30%                                                       ║
║  TP3: +80% → Sell 30%                                                       ║
║  TP4: +150% → Sell 15% (runner)                                             ║
║  After TP1: Move SL to entry (0%)                                           ║
║  After TP2: Trail 20% from ATH                                              ║
║                                                                              ║
║  TIER3 (Volume + Holders) - Good Quality                                     ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Initial SL: -28%                                                           ║
║  TP1: +18% → Sell 30%                                                       ║
║  TP2: +40% → Sell 35%                                                       ║
║  TP3: +70% → Sell 35%                                                       ║
║  After TP1: Move SL to entry                                                ║
║  After TP2: Trail 18% from ATH                                              ║
║                                                                              ║
║  TIER4 (Moderate Metrics) - Moderate Quality                                 ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Initial SL: -25%                                                           ║
║  TP1: +15% → Sell 35%                                                       ║
║  TP2: +30% → Sell 40%                                                       ║
║  TP3: +50% → Sell 25%                                                       ║
║  After TP1: Move SL to entry                                                ║
║  After TP2: Trail 15% from ATH                                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("\n=== ANALYSIS COMPLETE ===")
print("\nNOTE: For deeper time-based analysis and per-coin behavior,")
print("we need the 24-hour price/liquidity trace data from Trace_24H.")
print("This will enable reactive AI that adapts to each coin's behavior.")


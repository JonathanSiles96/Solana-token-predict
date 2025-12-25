"""
BUILD PROPER TP LADDER & SL ANALYSIS
=====================================

Using 2025(12).csv data to:
1. Analyze signals that PASSED filter - TP hit rates
2. Build optimal TP ladder (10%, 20%, 30%... 400%)
3. Analyze drawdowns for SL optimization
4. Post-TP SL updates for profit protection

"""
import pandas as pd
import numpy as np
import sys
import io

# Fix unicode output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load data
df = pd.read_csv('2025 (12).csv')

print('='*80)
print('TP LADDER & SL ANALYSIS - DEC 24 DATA')
print('='*80)
print(f'Total signals: {len(df)}')
print(f'Columns: {df.columns.tolist()}')

# Check for max_return data
df['max_return'] = pd.to_numeric(df['max_return'], errors='coerce')
has_return = df['max_return'].notna()
print(f'\nSignals with max_return data: {has_return.sum()}')

# Filter for valid signals
df_valid = df[has_return].copy()

# Convert max_return to percentage gain
# If max_return is already a multiplier (e.g., 1.5 = +50%), convert to percentage
if df_valid['max_return'].median() > 0 and df_valid['max_return'].median() < 10:
    # It's a multiplier, convert to percentage
    df_valid['max_return_pct'] = (df_valid['max_return'] - 1) * 100
else:
    # It's already a percentage
    df_valid['max_return_pct'] = df_valid['max_return']

print(f'\nMax return range: {df_valid["max_return"].min():.2f} to {df_valid["max_return"].max():.2f}')
print(f'Max return pct range: {df_valid["max_return_pct"].min():.1f}% to {df_valid["max_return_pct"].max():.1f}%')

# ============================================================================
# SECTION 1: IDENTIFY "PASSED FILTER" SIGNALS
# ============================================================================
print('\n' + '='*80)
print('SECTION 1: SIGNALS BY FILTER STATUS')
print('='*80)

# Check trading_score for GO/SKIP decisions
if 'trading_score' in df_valid.columns:
    # Look for GO signals
    df_valid['is_go'] = df_valid['trading_score'].str.contains('GO|TIER', case=False, na=False)
    df_valid['is_skip'] = df_valid['trading_score'].str.contains('SKIP|BLOCKED', case=False, na=False)
    
    go_count = df_valid['is_go'].sum()
    skip_count = df_valid['is_skip'].sum()
    
    print(f'GO signals: {go_count}')
    print(f'SKIP signals: {skip_count}')
    print(f'Unknown: {len(df_valid) - go_count - skip_count}')
    
    # Analyze GO signals only
    df_go = df_valid[df_valid['is_go']].copy()
    if len(df_go) > 0:
        print(f'\n--- GO SIGNALS ANALYSIS (n={len(df_go)}) ---')
    else:
        print('\nNo GO signals found, analyzing ALL signals')
        df_go = df_valid.copy()
else:
    print('No trading_score column, analyzing ALL signals')
    df_go = df_valid.copy()

# ============================================================================
# SECTION 2: TP LADDER - HIT RATES AT EACH LEVEL
# ============================================================================
print('\n' + '='*80)
print('SECTION 2: TP LADDER - HIT RATES')
print('='*80)

# TP levels to analyze
tp_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 
             120, 150, 180, 200, 250, 300, 350, 400]

print(f'\n{"TP Level":<12} {"Hit Count":<12} {"Hit Rate":<12} {"Recommendation"}')
print('-'*60)

for tp in tp_levels:
    hit = (df_go['max_return_pct'] >= tp).sum()
    rate = hit / len(df_go) * 100
    
    # Recommendation based on hit rate
    if rate >= 80:
        rec = "STRONG TP"
    elif rate >= 60:
        rec = "Good TP"
    elif rate >= 40:
        rec = "Moderate"
    elif rate >= 20:
        rec = "Stretch"
    else:
        rec = "Runner only"
    
    print(f'+{tp}%'.ljust(12) + f'{hit}'.ljust(12) + f'{rate:.1f}%'.ljust(12) + rec)

# ============================================================================
# SECTION 3: OPTIMAL TP LADDER RECOMMENDATION
# ============================================================================
print('\n' + '='*80)
print('SECTION 3: OPTIMAL TP LADDER')
print('='*80)

# Find optimal TP levels based on diminishing returns
print('\nRecommended TP Ladder:')

# TP1: Target ~80% hit rate
for tp in [10, 12, 15, 18, 20]:
    hit_rate = (df_go['max_return_pct'] >= tp).mean() * 100
    if hit_rate >= 75:
        tp1 = tp
        tp1_rate = hit_rate
        break
else:
    tp1 = 10
    tp1_rate = (df_go['max_return_pct'] >= 10).mean() * 100

# TP2: Target ~60% hit rate
for tp in [25, 30, 35, 40]:
    hit_rate = (df_go['max_return_pct'] >= tp).mean() * 100
    if hit_rate >= 55:
        tp2 = tp
        tp2_rate = hit_rate
        break
else:
    tp2 = 30
    tp2_rate = (df_go['max_return_pct'] >= 30).mean() * 100

# TP3: Target ~40% hit rate
for tp in [50, 60, 70, 80]:
    hit_rate = (df_go['max_return_pct'] >= tp).mean() * 100
    if hit_rate >= 35:
        tp3 = tp
        tp3_rate = hit_rate
        break
else:
    tp3 = 60
    tp3_rate = (df_go['max_return_pct'] >= 60).mean() * 100

# TP4: Target ~25% hit rate (runners)
for tp in [100, 120, 150, 200]:
    hit_rate = (df_go['max_return_pct'] >= tp).mean() * 100
    if hit_rate >= 20:
        tp4 = tp
        tp4_rate = hit_rate
        break
else:
    tp4 = 100
    tp4_rate = (df_go['max_return_pct'] >= 100).mean() * 100

print(f'''
  TP1: +{tp1}% ({tp1_rate:.1f}% hit rate) - Sell 35% of position
  TP2: +{tp2}% ({tp2_rate:.1f}% hit rate) - Sell 30% of position
  TP3: +{tp3}% ({tp3_rate:.1f}% hit rate) - Sell 25% of position
  TP4: +{tp4}% ({tp4_rate:.1f}% hit rate) - Sell 10% runner
''')

# ============================================================================
# SECTION 4: STOP LOSS ANALYSIS
# ============================================================================
print('\n' + '='*80)
print('SECTION 4: STOP LOSS ANALYSIS')
print('='*80)

# For SL analysis, we need min_return (drawdown) data
# Check if we have it
if 'min_return' in df_valid.columns:
    df_valid['min_return_pct'] = (df_valid['min_return'] - 1) * 100
    has_min = True
else:
    print('\nWARNING: No min_return column in data!')
    print('We need drawdown data to properly set SL levels.')
    print('Will estimate based on max_return distribution instead.')
    has_min = False

# Analyze signals that WOULD HAVE been stopped out at various SL levels
# but ended up being winners
print('\nSL ANALYSIS: What if we had used different SL levels?')
print('(This shows signals we would have KEPT at each SL)')
print()

# Assuming we'll get min_return data soon, for now analyze based on final outcome
# Signals that hit 30%+ are "winners"
winners = df_go[df_go['max_return_pct'] >= 30]
losers = df_go[df_go['max_return_pct'] < 30]

print(f'Winners (hit +30%+): {len(winners)} ({len(winners)/len(df_go)*100:.1f}%)')
print(f'Losers (never hit +30%): {len(losers)} ({len(losers)/len(df_go)*100:.1f}%)')

# For losers, what was their max return?
if len(losers) > 0:
    print(f'\nLoser distribution:')
    print(f'  Complete rugs (max_return <= 0): {(losers["max_return_pct"] <= 0).sum()}')
    print(f'  Slight gain (0-10%): {((losers["max_return_pct"] > 0) & (losers["max_return_pct"] < 10)).sum()}')
    print(f'  Good but not winner (10-30%): {((losers["max_return_pct"] >= 10) & (losers["max_return_pct"] < 30)).sum()}')

# ============================================================================
# SECTION 5: POST-TP SL UPDATES
# ============================================================================
print('\n' + '='*80)
print('SECTION 5: POST-TP STOP LOSS UPDATES')
print('='*80)

print('''
RECOMMENDED POST-TP SL LADDER:

After TP1 (+{tp1}%): Move SL to BREAK-EVEN (0%)
  - Rationale: Lock in no-loss, let winners run
  
After TP2 (+{tp2}%): Move SL to +{sl2}%
  - Rationale: Already took 65% profits, protect remaining
  
After TP3 (+{tp3}%): Move SL to +{sl3}%
  - Rationale: Only runner left, trail it closely

TRAILING STOP:
  - Activate after +15%
  - Trail at 12% below peak
  - Tighten to 8% after +50%
'''.format(tp1=tp1, tp2=tp2, tp3=tp3, sl2=max(10, tp1), sl3=max(25, tp2//2)))

# ============================================================================
# SECTION 6: BY SOURCE ANALYSIS
# ============================================================================
print('\n' + '='*80)
print('SECTION 6: TP LADDER BY SOURCE')
print('='*80)

for source in df_go['source'].dropna().unique():
    source_data = df_go[df_go['source'] == source]
    if len(source_data) < 5:
        continue
    
    print(f'\n{source.upper()} (n={len(source_data)})')
    
    for tp in [15, 30, 50, 100]:
        hit_rate = (source_data['max_return_pct'] >= tp).mean() * 100
        print(f'  +{tp}%: {hit_rate:.1f}%')

# ============================================================================
# SECTION 7: SUMMARY & CODE RECOMMENDATIONS
# ============================================================================
print('\n' + '='*80)
print('SECTION 7: CODE RECOMMENDATIONS')
print('='*80)

print(f'''
UPDATE token_scorer.py WITH:

tp_levels = [
    {{'gain_pct': {tp1}, 'sell_amount_pct': 35}},   # TP1: {tp1_rate:.0f}% hit
    {{'gain_pct': {tp2}, 'sell_amount_pct': 30}},   # TP2: {tp2_rate:.0f}% hit
    {{'gain_pct': {tp3}, 'sell_amount_pct': 25}},   # TP3: {tp3_rate:.0f}% hit
    {{'gain_pct': {tp4}, 'sell_amount_pct': 10}}    # TP4: {tp4_rate:.0f}% hit (runner)
]

trailing_activation = 0.15  # Activate at +15%
trailing_distance = 0.12    # Trail 12% below peak

post_tp1_sl = 0.0   # Break-even after TP1
post_tp2_sl = 0.{max(10, tp1)}  # +{max(10, tp1)}% after TP2
post_tp3_sl = 0.{max(25, tp2//2)}  # +{max(25, tp2//2)}% after TP3

# Initial SL (NEED MIN_RETURN DATA TO OPTIMIZE)
sl_level = -0.18  # -18% default until we have drawdown data
''')

print('\n' + '='*80)
print('NEXT STEP: GET MIN_RETURN DATA FOR PROPER SL ANALYSIS')
print('='*80)
print('''
Once we have min_return (drawdown) data, we can answer:
1. At what % drawdown do winners typically bottom out?
2. What SL level maximizes keeping winners while cutting losers?
3. How tight can we make trailing stops without getting stopped out?

Ask Dima for the 24h price data with LOW prices, not just HIGH!
''')


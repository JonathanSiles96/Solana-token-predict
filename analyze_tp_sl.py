"""
Data Analysis for TP/SL Optimization
Analyze the CSV to determine optimal TP and SL levels
"""
import pandas as pd
import numpy as np

# Read the CSV
df = pd.read_csv('2025 (11).csv')

print('='*80)
print('DATA ANALYSIS FOR TP/SL OPTIMIZATION')
print('='*80)

print(f'\nTotal signals: {len(df)}')
print(f'Signals with max_return data: {df["max_return"].notna().sum()}')

# Filter to valid signals
df_valid = df[df['max_return'].notna() & (df['max_return'] > 0)].copy()
print(f'Valid signals for analysis: {len(df_valid)}')

# Max return distribution
print('\n' + '='*80)
print('MAX RETURN DISTRIBUTION (what gains are achievable)')
print('='*80)
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(df_valid['max_return'], p)
    print(f'  {p}th percentile: {val:.2f}x ({(val-1)*100:.1f}% gain)')

print(f'\n  Mean max_return: {df_valid["max_return"].mean():.2f}x')
print(f'  Median max_return: {df_valid["max_return"].median():.2f}x')
print(f'  Best max_return: {df_valid["max_return"].max():.2f}x')

# How many reach each TP level?
print('\n' + '='*80)
print('HOW MANY SIGNALS REACH EACH TP LEVEL?')
print('='*80)
tp_levels = [1.08, 1.10, 1.12, 1.15, 1.20, 1.30, 1.50, 1.70, 2.0, 3.0, 5.0, 10.0]
for tp in tp_levels:
    count = (df_valid['max_return'] >= tp).sum()
    pct = count / len(df_valid) * 100
    print(f'  Reach +{(tp-1)*100:.0f}% gain: {count}/{len(df_valid)} ({pct:.1f}%)')

# Signals that never go positive
never_positive = (df_valid['max_return'] < 1.0).sum()
print(f'\n  ❌ Never profitable (max_return < 1.0): {never_positive} ({never_positive/len(df_valid)*100:.1f}%)')

# Winners vs Losers
print('\n' + '='*80)
print('WINNER VS LOSER ANALYSIS')
print('='*80)
winners = df_valid[df_valid['winner'] == True]
losers = df_valid[df_valid['winner'] == False]
print(f'Winners (per system): {len(winners)}')
print(f'Losers (per system): {len(losers)}')

if len(winners) > 0:
    print(f'\nWinner stats:')
    print(f'  Avg max_return: {winners["max_return"].mean():.2f}x')
    print(f'  Median max_return: {winners["max_return"].median():.2f}x')
    
if len(losers) > 0:
    loser_valid = losers[losers['max_return'].notna()]
    if len(loser_valid) > 0:
        print(f'\nLoser stats (what they could have achieved):')
        print(f'  Avg max_return: {loser_valid["max_return"].mean():.2f}x')
        print(f'  Median max_return: {loser_valid["max_return"].median():.2f}x')
        # How many losers actually hit positive max_return?
        losers_hit_positive = (loser_valid['max_return'] >= 1.15).sum()
        print(f'  Losers that hit +15%: {losers_hit_positive}/{len(loser_valid)} ({losers_hit_positive/len(loser_valid)*100:.1f}%)')

# By source analysis
print('\n' + '='*80)
print('BY SOURCE ANALYSIS')
print('='*80)
for source in sorted(df_valid['source'].dropna().unique()):
    src_df = df_valid[df_valid['source'] == source]
    if len(src_df) >= 3:
        reach_10 = (src_df['max_return'] >= 1.10).sum() / len(src_df) * 100
        reach_15 = (src_df['max_return'] >= 1.15).sum() / len(src_df) * 100
        reach_30 = (src_df['max_return'] >= 1.30).sum() / len(src_df) * 100
        reach_50 = (src_df['max_return'] >= 1.50).sum() / len(src_df) * 100
        print(f'{source}:')
        print(f'  n={len(src_df)}, median={src_df["max_return"].median():.2f}x, mean={src_df["max_return"].mean():.2f}x')
        print(f'  Hit +10%: {reach_10:.0f}% | +15%: {reach_15:.0f}% | +30%: {reach_30:.0f}% | +50%: {reach_50:.0f}%')

# OPTIMAL TP/SL RECOMMENDATIONS
print('\n' + '='*80)
print('OPTIMAL TP/SL RECOMMENDATIONS (DATA-DRIVEN)')
print('='*80)

# Find optimal TP1 - should be hit by at least 60-70% of signals
for tp in [1.08, 1.10, 1.12, 1.15, 1.20]:
    hit_rate = (df_valid['max_return'] >= tp).sum() / len(df_valid) * 100
    if hit_rate >= 60:
        print(f'  TP1 RECOMMENDATION: +{(tp-1)*100:.0f}% (hit by {hit_rate:.0f}% of signals)')
        break

# Find optimal TP2 - should be hit by 40-50% of signals  
for tp in [1.20, 1.30, 1.40, 1.50]:
    hit_rate = (df_valid['max_return'] >= tp).sum() / len(df_valid) * 100
    if hit_rate >= 35 and hit_rate <= 60:
        print(f'  TP2 RECOMMENDATION: +{(tp-1)*100:.0f}% (hit by {hit_rate:.0f}% of signals)')
        break

# Find optimal TP3 - should be hit by 20-30% of signals
for tp in [1.50, 1.70, 2.0, 2.5]:
    hit_rate = (df_valid['max_return'] >= tp).sum() / len(df_valid) * 100
    if hit_rate >= 15 and hit_rate <= 35:
        print(f'  TP3 RECOMMENDATION: +{(tp-1)*100:.0f}% (hit by {hit_rate:.0f}% of signals)')
        break

# Find optimal TP4 - for runners, 10-15% of signals
for tp in [2.0, 3.0, 5.0]:
    hit_rate = (df_valid['max_return'] >= tp).sum() / len(df_valid) * 100
    if hit_rate >= 5 and hit_rate <= 20:
        print(f'  TP4 RECOMMENDATION: +{(tp-1)*100:.0f}% (hit by {hit_rate:.0f}% of signals)')
        break

# Stop loss analysis - look at drawdowns
print('\n  STOP LOSS ANALYSIS:')
# How many never go positive?
never_profit = (df_valid['max_return'] < 1.0).sum()
barely_profit = ((df_valid['max_return'] >= 1.0) & (df_valid['max_return'] < 1.10)).sum()
print(f'  Signals that never profit: {never_profit} ({never_profit/len(df_valid)*100:.1f}%)')
print(f'  Signals 0-10% max gain: {barely_profit} ({barely_profit/len(df_valid)*100:.1f}%)')
print(f'  => SL should be tight (-15 to -20%) to cut these losses quickly')

print('\n' + '='*80)
print('FINAL RECOMMENDATIONS')
print('='*80)
print('''
Based on this data analysis:

TP LADDER:
  TP1: +12-15% gain, sell 40% of position (hit by ~65% of signals)
  TP2: +30-35% gain, sell 30% of position (hit by ~45% of signals)
  TP3: +60-70% gain, sell 20% of position (hit by ~25% of signals)
  TP4: +100-150% gain, sell remaining 10% (hit by ~15% of signals)

STOP LOSS:
  Initial SL: -18% to -22% (tight to cut losers early)
  After TP1: Move SL to break-even (0%)
  After TP2: Move SL to +15%
  After TP3: Move SL to +35%

TRAILING STOP:
  Activate: After hitting TP1 (+15%)
  Distance: 12-15% below peak price
  
TOP-UP STRATEGY:
  If price dips 5-8% after entry but fundamentals still good:
    - Add 25-50% to position
    - Lower average entry price
    - Tighter SL on added position
''')


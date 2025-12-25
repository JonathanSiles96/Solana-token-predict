"""
PROPER STRUCTURED ANALYSIS
What we CAN analyze: max_return (peak price relative to entry)
What we CANNOT analyze without more data: drawdowns, SL hits, time-to-target

Let's be honest about limitations and provide actionable insights.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('2025 (11).csv')

# Get valid signals with outcome data
df_valid = df[df['max_return'].notna()].copy()
df_valid['max_return'] = df_valid['max_return'].astype(float)

print('='*80)
print('REALITY CHECK - WHAT WE ACTUALLY KNOW')
print('='*80)
print(f'Total signals in file: {len(df)}')
print(f'Signals with max_return data: {len(df_valid)}')
print(f'Signals missing max_return: {len(df) - len(df_valid)}')

# Check for losers (max_return < 1 means price went down from entry)
losers = df_valid[df_valid['max_return'] < 1]
winners = df_valid[df_valid['max_return'] >= 1]

print(f'\nLoser signals (max_return < 1.0): {len(losers)} ({len(losers)/len(df_valid)*100:.1f}%)')
print(f'Winner signals (max_return >= 1.0): {len(winners)} ({len(winners)/len(df_valid)*100:.1f}%)')

# Distribution of max_return
print('\n' + '='*80)
print('MAX RETURN DISTRIBUTION (what peak price signals reached)')
print('='*80)

# For losers (rugged/dumped tokens)
print('\nLOSERS (never went above entry):')
if len(losers) > 0:
    print(f'  Worst: {(losers["max_return"].min()-1)*100:.1f}%')
    print(f'  Median: {(losers["max_return"].median()-1)*100:.1f}%')
    print(f'  Count: {len(losers)}')

# TP ladder hit rates (among ALL signals)
print('\n' + '='*80)
print('TP LADDER - % OF ALL SIGNALS REACHING EACH LEVEL')
print('='*80)

tp_levels = [
    (5, '+5%'),
    (10, '+10%'),
    (15, '+15%'),
    (20, '+20%'),
    (30, '+30%'),
    (40, '+40%'),
    (50, '+50%'),
    (75, '+75%'),
    (100, '+100%'),
    (150, '+150%'),
    (200, '+200%'),
    (300, '+300%'),
]

print(f'\n{"Level":<10} {"Hit Count":<12} {"Hit Rate":<12} {"Cumulative Value"}')
print('-'*60)

for pct, label in tp_levels:
    threshold = 1 + (pct/100)
    hit = (df_valid['max_return'] >= threshold).sum()
    rate = hit / len(df_valid) * 100
    print(f'{label:<10} {hit:<12} {rate:.1f}%')

# The CRITICAL question: What about signals that NEVER recovered?
print('\n' + '='*80)
print('CRITICAL MISSING DATA - DRAWDOWN ANALYSIS')
print('='*80)
print('''
WE DON'T HAVE min_return DATA!

Without knowing the MIN price each token reached, we CANNOT answer:
- How many signals hit -20% SL before recovering to +30%?
- What % of "winners" had scary drawdowns first?
- What's the optimal SL level?

THIS IS A MAJOR GAP. We need to:
1. Track min_return in the database going forward
2. Or fetch historical price data for past signals

The "100% WR" for whale/tg_early_trending is based on max_return only.
In reality, some of those probably hit -30% before pumping to +50%.
''')

# Let's at least analyze by source with more detail
print('\n' + '='*80)
print('BY SOURCE - DETAILED TP LADDER')
print('='*80)

for source in ['whale', 'tg_early_trending', 'primal', 'solana_tracker']:
    source_data = df_valid[df_valid['source'] == source]
    if len(source_data) == 0:
        continue
    
    print(f'\n{source.upper()} (n={len(source_data)})')
    print(f'  Never profit (max<1.0): {(source_data["max_return"] < 1).sum()} ({(source_data["max_return"] < 1).mean()*100:.1f}%)')
    
    print(f'  TP Ladder:')
    for pct, label in [(10, '+10%'), (20, '+20%'), (30, '+30%'), (50, '+50%'), (100, '+100%')]:
        threshold = 1 + (pct/100)
        hit_rate = (source_data['max_return'] >= threshold).mean() * 100
        print(f'    {label}: {hit_rate:.1f}%')
    
    print(f'  Avg max return: {source_data["max_return"].mean():.2f}x')
    print(f'  Median max return: {source_data["max_return"].median():.2f}x')

# Analyze by filters
print('\n' + '='*80)
print('FILTER COMBINATIONS - TP LADDER')
print('='*80)

def analyze_filter_detailed(name, mask):
    filtered = df_valid[mask]
    if len(filtered) < 10:
        return
    
    print(f'\n{name} (n={len(filtered)})')
    print(f'  Never profit: {(filtered["max_return"] < 1).sum()} ({(filtered["max_return"] < 1).mean()*100:.1f}%)')
    
    for pct, label in [(10, '+10%'), (20, '+20%'), (30, '+30%'), (50, '+50%'), (100, '+100%')]:
        threshold = 1 + (pct/100)
        hit_rate = (filtered['max_return'] >= threshold).mean() * 100
        print(f'  {label}: {hit_rate:.1f}%')

# Best filters from previous analysis
analyze_filter_detailed('whale + Hold>=150', 
    (df_valid['source'] == 'whale') & (df_valid['holders'] >= 150))

analyze_filter_detailed('tg_early_trending + Hold>=150', 
    (df_valid['source'] == 'tg_early_trending') & (df_valid['holders'] >= 150))

analyze_filter_detailed('Holders >= 500', 
    df_valid['holders'] >= 500)

analyze_filter_detailed('Volume >= 25k + Holders >= 200', 
    (df_valid['volume_1h'] >= 25000) & (df_valid['holders'] >= 200))

analyze_filter_detailed('Liquidity >= 25k', 
    df_valid['liquidity'] >= 25000)

# What about the "skipped" signals - would they have been bad?
print('\n' + '='*80)
print('SIGNALS THAT WOULD BE SKIPPED BY NEW LOGIC')
print('='*80)

# My current skip logic for primal/solana_tracker
skip_mask = (
    (df_valid['source'].isin(['primal', 'solana_tracker'])) & 
    (df_valid['holders'] < 300) & 
    (df_valid['liquidity'] < 25000) & 
    (df_valid['volume_1h'] < 50000)
)

skipped = df_valid[skip_mask]
traded = df_valid[~skip_mask]

print(f'\nWould SKIP: {len(skipped)} signals')
print(f'Would TRADE: {len(traded)} signals')

if len(skipped) > 0:
    print(f'\nSKIPPED signals performance:')
    print(f'  Never profit: {(skipped["max_return"] < 1).sum()} ({(skipped["max_return"] < 1).mean()*100:.1f}%)')
    for pct, label in [(10, '+10%'), (30, '+30%'), (50, '+50%')]:
        threshold = 1 + (pct/100)
        hit_rate = (skipped['max_return'] >= threshold).mean() * 100
        print(f'  {label}: {hit_rate:.1f}%')
    print(f'  Avg max return: {skipped["max_return"].mean():.2f}x')

if len(traded) > 0:
    print(f'\nTRADED signals performance:')
    print(f'  Never profit: {(traded["max_return"] < 1).sum()} ({(traded["max_return"] < 1).mean()*100:.1f}%)')
    for pct, label in [(10, '+10%'), (30, '+30%'), (50, '+50%')]:
        threshold = 1 + (pct/100)
        hit_rate = (traded['max_return'] >= threshold).mean() * 100
        print(f'  {label}: {hit_rate:.1f}%')
    print(f'  Avg max return: {traded["max_return"].mean():.2f}x')

print('\n' + '='*80)
print('HONEST SUMMARY')
print('='*80)
print('''
WHAT WE KNOW:
- max_return tells us the PEAK price each token reached
- It does NOT tell us if SL was hit first
- Without min_return/drawdown data, our SL analysis is pure guesswork

WHAT WE NEED:
1. Add min_return tracking to the database
2. Track actual trade outcomes (was SL hit? which TPs hit?)
3. Time-series data for proper backtesting

WHAT WE CAN DO NOW:
- Use max_return to set OPTIMISTIC TP targets
- Be CONSERVATIVE with SL (we don't know drawdown risk)
- Prioritize sources/filters with higher max_return hit rates
- Start collecting better data going forward
''')


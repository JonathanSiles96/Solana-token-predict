# -*- coding: utf-8 -*-
"""
Verify the CORRECTED strategy - primal/solana_tracker should GO, whale/tg risky
"""
import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df1 = pd.read_csv('2025 (5).csv')
df2 = pd.read_csv('2025 (6).csv')
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['max_return'].notna()]

print('='*60)
print('CORRECTED STRATEGY VERIFICATION')
print('='*60)

# Simulate the corrected strategy
def simulate_corrected_strategy(row):
    """Apply corrected strategy rules"""
    source = str(row.get('source', 'unknown')).lower()
    holders = row.get('holders', 0) or 0
    volume = row.get('volume_1h', 0) or 0
    bundled = row.get('bundled_pct', 0) or 0
    snipers = row.get('snipers_pct', 0) or 0
    
    # TIER 1: Best sources - primal, solana_tracker
    if source in ['primal', 'solana_tracker']:
        # Quality filters
        if holders >= 100 and volume >= 15000 and bundled <= 20 and snipers <= 30:
            return 'GO_TIER1'
        else:
            return 'SKIP_QUALITY'
    
    # TIER 2: Risky sources - whale, tg_early_trending
    elif source in ['whale', 'tg_early_trending']:
        # Much stricter filters
        if holders >= 200 and volume >= 25000 and bundled <= 10 and snipers <= 15:
            return 'GO_TIER2'
        else:
            return 'SKIP_RISKY_SOURCE'
    
    else:
        return 'SKIP_UNKNOWN'

# Apply strategy
df['decision'] = df.apply(simulate_corrected_strategy, axis=1)

print('\n--- Decision Distribution ---')
print(df['decision'].value_counts())

# Calculate results for each decision type
print('\n--- Results by Decision ---')
for decision in df['decision'].unique():
    subset = df[df['decision'] == decision]
    total = len(subset)
    never_positive = (subset['max_return'] < 1.0).sum()
    hit_130 = (subset['max_return'] >= 1.3).sum()
    hit_5x = (subset['winner'] == True).sum()
    avg_return = subset['max_return'].mean()
    
    print(f'\n{decision} ({total} signals):')
    print(f'  Never positive: {never_positive} ({never_positive/total*100:.1f}%)')
    print(f'  Reached 1.3x+: {hit_130} ({hit_130/total*100:.1f}%)')
    print(f'  Reached 5x+: {hit_5x} ({hit_5x/total*100:.1f}%)')
    print(f'  Avg max_return: {avg_return:.2f}x')

# Summary for GO decisions only
print('\n' + '='*60)
print('SUMMARY: GO DECISIONS ONLY')
print('='*60)
go_signals = df[df['decision'].isin(['GO_TIER1', 'GO_TIER2'])]
if len(go_signals) > 0:
    total = len(go_signals)
    never_positive = (go_signals['max_return'] < 1.0).sum()
    hit_130 = (go_signals['max_return'] >= 1.3).sum()
    avg_return = go_signals['max_return'].mean()
    
    print(f'Total GO signals: {total}')
    print(f'Never positive: {never_positive} ({never_positive/total*100:.1f}%)')
    print(f'Reached 1.3x+: {hit_130} ({hit_130/total*100:.1f}%)')
    print(f'Avg max_return: {avg_return:.2f}x')
    print(f'\nBreak-even rate: {(total-never_positive)/total*100:.1f}%')
    print(f'Profit rate (1.3x+): {hit_130/total*100:.1f}%')

print('\n' + '='*60)
print('IMPORTANT CAVEAT')
print('='*60)
print('''
max_return is the PEAK price, not actual trade result.

If your stop loss is -30% and the price drops 35% before 
recovering to 3x, you still LOSE because SL hit first.

To know the ACTUAL win rate, we need:
1. Drawdown before max (how low did it go first?)
2. Time sequence (when did it hit each price point?)
3. Your actual SL setting

The numbers above are THEORETICAL MAXIMUM.
Real results depend on SL and entry timing.
''')


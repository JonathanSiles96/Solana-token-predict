# -*- coding: utf-8 -*-
"""
Check what ACTUALLY happened on Dec 15
"""
import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df1 = pd.read_csv('2025 (5).csv')
df2 = pd.read_csv('2025 (6).csv')
df = pd.concat([df1, df2], ignore_index=True)

print('='*60)
print('Dec 14-15 Reality Check')
print('='*60)

# Parse signal_time
df['signal_time'] = pd.to_datetime(df['signal_time'])
df['date'] = df['signal_time'].dt.date.astype(str)

print('\nDates in data:')
print(df['date'].value_counts().sort_index())

# Focus on Dec 14-15
dec15 = df[df['date'].isin(['2025-12-14', '2025-12-15'])]
print(f'\nTotal signals Dec 14-15: {len(dec15)}')

print('\n--- BY SOURCE on Dec 14-15 ---')
for source in sorted(dec15['source'].unique()):
    source_df = dec15[dec15['source'] == source]
    total = len(source_df)
    never_profit = (source_df['max_return'] < 1.0).sum()
    hit_130 = (source_df['max_return'] >= 1.3).sum()
    hit_5x = (source_df['winner'] == True).sum()
    avg_return = source_df['max_return'].mean()
    
    print(f'\n{source} ({total} signals):')
    print(f'  Never positive: {never_profit} ({never_profit/total*100:.1f}%)')
    print(f'  Reached 1.3x+: {hit_130} ({hit_130/total*100:.1f}%)')
    print(f'  Reached 5x+ (winner): {hit_5x} ({hit_5x/total*100:.1f}%)')
    print(f'  Avg max_return: {avg_return:.2f}x')

print('\n' + '='*60)
print('THE REAL ISSUE')
print('='*60)
print('''
Looking at the data, here's what we know:
1. max_return is the PEAK price, not actual trade result
2. We DON'T have actual trade entry/exit data
3. We DON'T have stop-loss hit information

The "13.1% loss" you experienced could be:
1. Stop loss hitting before price recovered
2. Entry slippage (buying higher than signal price)
3. Exit timing (selling before the peak)
4. Position sizing issues

WITHOUT actual trade logs, I cannot verify what's happening.
''')

print('\n' + '='*60)
print('WHAT I NEED FROM YOU')
print('='*60)
print('''
To properly diagnose:
1. Share your actual trade logs (entry price, exit price, SL hit?)
2. What was the stop loss setting?
3. How long after the signal do you enter?

The max_return data shows tokens DO go up, but if you're
getting stopped out before, that's a TIMING/SL issue, not a signal issue.
''')


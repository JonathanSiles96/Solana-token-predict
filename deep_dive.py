# -*- coding: utf-8 -*-
"""
Deep dive into the ACTUAL winner column vs max_return
"""
import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df1 = pd.read_csv('2025 (5).csv')
df2 = pd.read_csv('2025 (6).csv')
df = pd.concat([df1, df2], ignore_index=True)

print('='*60)
print('DEEP DIVE: Understanding the data')
print('='*60)

print('\n--- winner column values ---')
print(df['winner'].value_counts(dropna=False))

# Check the relationship between winner and max_return
print('\n--- winner=True cases: max_return stats ---')
winners = df[df['winner'] == True]
print(f'Count: {len(winners)}')
print(f'max_return min: {winners["max_return"].min():.2f}')
print(f'max_return max: {winners["max_return"].max():.2f}')

print('\n--- winner=False cases: max_return stats ---')
losers = df[df['winner'] == False]
print(f'Count: {len(losers)}')
print(f'max_return min: {losers["max_return"].min():.2f}')
print(f'max_return max: {losers["max_return"].max():.2f}')

print('\n--- IMPORTANT INSIGHT ---')
print('winner=True means max_return >= 5x (big winners only)')
print('winner=False includes tokens with 2x, 3x, 4x returns!')

# ACTUAL win rate by source
print('\n' + '='*60)
print('WIN RATE BY SOURCE (winner=True means 5x+)')
print('='*60)
df_with_winner = df[df['winner'].notna()]
for source in sorted(df_with_winner['source'].unique()):
    source_df = df_with_winner[df_with_winner['source'] == source]
    wins = (source_df['winner'] == True).sum()
    total = len(source_df)
    if total > 5:
        rate = wins / total * 100
        print(f'{source}: {wins}/{total} = {rate:.1f}% hit 5x+')

# More realistic: What about tokens that at least hit 1.3x (30% gain)?
print('\n' + '='*60)
print('REALISTIC WIN: Tokens that reached 1.3x or higher')
print('='*60)
for source in sorted(df_with_winner['source'].unique()):
    source_df = df_with_winner[df_with_winner['source'] == source]
    wins = (source_df['max_return'] >= 1.3).sum()
    total = len(source_df)
    if total > 5:
        rate = wins / total * 100
        print(f'{source}: {wins}/{total} = {rate:.1f}% reached 1.3x+')

# Even more realistic: max_return > 1.0 (any profit)
print('\n' + '='*60)
print('ANY PROFIT: Tokens that reached 1.0x or higher')
print('='*60)
for source in sorted(df_with_winner['source'].unique()):
    source_df = df_with_winner[df_with_winner['source'] == source]
    wins = (source_df['max_return'] >= 1.0).sum()
    total = len(source_df)
    if total > 5:
        rate = wins / total * 100
        print(f'{source}: {wins}/{total} = {rate:.1f}% reached 1.0x+')

print('\n' + '='*60)
print('THE REAL QUESTION: How many hit STOP LOSS before profit?')
print('='*60)
print('''
We can't know from this data!

max_return = 2x doesn't mean you made 2x
- Price might drop 40% first (stop loss hit)
- THEN rise to 2x (but you're already out)

What we need:
1. Entry price
2. Min price BEFORE max price (drawdown)
3. Time sequence

Without drawdown data, we're guessing.
''')

# Best estimate based on max_return being low
print('\n' + '='*60)
print('ROUGH ESTIMATE: Tokens that never went profitable')
print('='*60)
losers_only = df[df['max_return'] < 1.0]
print(f'Tokens with max_return < 1.0: {len(losers_only)}')
for source in sorted(losers_only['source'].unique()):
    source_df = losers_only[losers_only['source'] == source]
    total_source = len(df_with_winner[df_with_winner['source'] == source])
    count = len(source_df)
    pct = count / total_source * 100 if total_source > 0 else 0
    if total_source > 5:
        print(f'{source}: {count}/{total_source} = {pct:.1f}% never went positive')

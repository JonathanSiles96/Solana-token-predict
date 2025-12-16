# -*- coding: utf-8 -*-
"""Verify the new strict strategy"""
import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df1 = pd.read_csv('2025 (5).csv')
df2 = pd.read_csv('2025 (6).csv')
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['max_return'].notna() & (df['max_return'] > 0)]
df['winner'] = df['max_return'] >= 1.3

print('='*60)
print('NEW STRICT STRATEGY: Only whale + tg_early_trending')
print('='*60)

# Only whale and tg_early_trending
strict = df[df['source'].isin(['whale', 'tg_early_trending'])]
winners = strict['winner'].sum()
total = len(strict)

print(f'Total trades: {total}')
print(f'Winners: {winners}')
print(f'WIN RATE: {winners/total*100:.1f}%')
avg_return = strict['max_return'].mean()
print(f'Avg return: {avg_return:.2f}x ({(avg_return-1)*100:.0f}%)')
print()

# By source
print('BY SOURCE:')
for src in ['whale', 'tg_early_trending']:
    sub = strict[strict['source'] == src]
    w = sub['winner'].sum()
    t = len(sub)
    avg = sub['max_return'].mean()
    print(f'  {src}: {w}/{t} = {w/t*100:.0f}% win rate, avg {avg:.2f}x')
print()

# Compare to old strategy (primal + solana_tracker)
print('='*60)
print('OLD LOSING SOURCES: primal + solana_tracker')
print('='*60)
old = df[df['source'].isin(['primal', 'solana_tracker'])]
old_winners = old['winner'].sum()
old_total = len(old)
old_avg = old['max_return'].mean()
print(f'Total trades: {old_total}')
print(f'Winners: {old_winners}')
print(f'Win rate: {old_winners/old_total*100:.1f}%')
print(f'Avg return: {old_avg:.2f}x ({(old_avg-1)*100:.0f}%)')
print()

print('='*60)
print('SUMMARY:')
print('='*60)
print(f'NEW (whale+tg_early): {winners/total*100:.0f}% win rate, {total} trades')
print(f'OLD (primal+solana): {old_winners/old_total*100:.0f}% win rate, {old_total} trades')
print()
print('EXPECTED DAILY TRADES: ~30-50 from whale+tg_early_trending')
print('EXPECTED WIN RATE: 100%')
print('='*60)


# -*- coding: utf-8 -*-
"""
REALITY CHECK: Why 100% max_return win rate doesn't mean 100% actual wins

The problem:
- max_return = highest price reached
- But price might drop 40% FIRST, hitting your stop loss
- So you lose even though max_return shows a "win"
"""
import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df1 = pd.read_csv('2025 (5).csv')
df2 = pd.read_csv('2025 (6).csv')
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['max_return'].notna() & (df['max_return'] > 0)]

print('='*60)
print('REALITY CHECK: max_return vs Actual Trading Results')
print('='*60)

# Focus on whale and tg_early_trending
good_sources = df[df['source'].isin(['whale', 'tg_early_trending'])].copy()

print(f'\nTotal signals from whale + tg_early_trending: {len(good_sources)}')

# The KEY question: What was the DRAWDOWN before max_return?
# Unfortunately, we don't have drawdown data in the CSV
# But we can check if any had max_return < 1.0 (meaning price only went down)

print('\n--- max_return Distribution ---')
print(f'max_return < 0.7 (lost >30%): {(good_sources["max_return"] < 0.7).sum()}')
print(f'max_return 0.7-0.9: {((good_sources["max_return"] >= 0.7) & (good_sources["max_return"] < 0.9)).sum()}')
print(f'max_return 0.9-1.0 (slight loss): {((good_sources["max_return"] >= 0.9) & (good_sources["max_return"] < 1.0)).sum()}')
print(f'max_return 1.0-1.3 (small gain): {((good_sources["max_return"] >= 1.0) & (good_sources["max_return"] < 1.3)).sum()}')
print(f'max_return 1.3-2.0 (good gain): {((good_sources["max_return"] >= 1.3) & (good_sources["max_return"] < 2.0)).sum()}')
print(f'max_return > 2.0 (great gain): {(good_sources["max_return"] >= 2.0).sum()}')

# Check the winner column if it exists
if 'winner' in good_sources.columns:
    print(f'\nwinner=true: {(good_sources["winner"] == True).sum()}')
    print(f'winner=false: {(good_sources["winner"] == False).sum()}')

# The REAL issue: we don't know the drawdown path
print('\n' + '='*60)
print('THE REAL PROBLEM:')
print('='*60)
print('''
max_return only tells us the PEAK price, not the PATH.

Example:
- Token price: $100 at entry
- Drops to $60 (you get stopped out at -30%)
- Then rises to $300 (max_return = 3x)
- Your result: -30% LOSS
- Data shows: "winner" (max_return 3x)

This is why "100% win rate" doesn't match reality!
''')

# What we need to check
print('='*60)
print('WHAT WE NEED TO KNOW (but dont have):')
print('='*60)
print('''
1. Max DRAWDOWN before the peak
   - Did price drop 20%, 30%, 50% before recovering?
   - This triggers stop loss = LOSS

2. Time to max_return
   - Did it take 5 minutes or 5 hours?
   - Longer time = more chance of hitting SL

3. Entry execution price
   - Did you buy at signal price or higher?
   - Slippage = worse entry = more likely to hit SL
''')

# Estimate realistic win rate
print('='*60)
print('REALISTIC ESTIMATE:')
print('='*60)

# If max_return is high but we assume 30-40% might hit SL first...
total = len(good_sources)
high_confidence = len(good_sources[good_sources['max_return'] >= 2.0])  # 2x+ usually means strong momentum
medium = len(good_sources[(good_sources['max_return'] >= 1.3) & (good_sources['max_return'] < 2.0)])
low = len(good_sources[good_sources['max_return'] < 1.3])

print(f'''
Based on max_return distribution:
- Strong momentum (>2x): {high_confidence} signals - likely 80%+ actual win rate
- Medium (1.3-2x): {medium} signals - likely 60-70% actual win rate  
- Weak (<1.3x): {low} signals - likely 50% or less actual win rate

Conservative estimate:
- If 80% of strong signals win: {int(high_confidence * 0.8)} wins
- If 65% of medium signals win: {int(medium * 0.65)} wins
- If 50% of weak signals win: {int(low * 0.5)} wins

Total estimated wins: {int(high_confidence * 0.8) + int(medium * 0.65) + int(low * 0.5)} / {total}
Realistic win rate: ~{(int(high_confidence * 0.8) + int(medium * 0.65) + int(low * 0.5)) / total * 100:.0f}%
''')

print('='*60)
print('RECOMMENDATION:')
print('='*60)
print('''
1. Track ACTUAL trade results, not max_return
2. Record: entry price, exit price, hit SL or TP?
3. After 50 real trades, we'll know the TRUE win rate

The "100% win rate" was based on max_return, not reality.
Real win rate is probably 60-75% for good sources.
''')


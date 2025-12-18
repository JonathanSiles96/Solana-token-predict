# -*- coding: utf-8 -*-
"""
Analyze REAL trade data from 2025 (7).csv
"""
import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('2025 (7).csv')

print('='*70)
print('REAL TRADE ANALYSIS - 2025 (7).csv')
print('='*70)

print(f'\nTotal trades: {len(df)}')

# Basic P&L stats
print('\n--- OVERALL PERFORMANCE ---')
total_invested = df['total_invested_usd'].sum()
total_profit = df['profit_usd'].sum()
print(f'Total invested: ${total_invested:.2f}')
print(f'Total profit: ${total_profit:.2f}')
print(f'Overall return: {total_profit/total_invested*100:.2f}%')

# Count wins and losses
df['is_winner'] = df['return_pct'] > 0
wins = df['is_winner'].sum()
losses = len(df) - wins
print(f'\nWins: {wins} ({wins/len(df)*100:.1f}%)')
print(f'Losses: {losses} ({losses/len(df)*100:.1f}%)')

# Breakdown by source
print('\n' + '='*70)
print('PERFORMANCE BY SOURCE')
print('='*70)

for source in df['source'].unique():
    source_df = df[df['source'] == source]
    n = len(source_df)
    wins = source_df['is_winner'].sum()
    win_rate = wins / n * 100 if n > 0 else 0
    total_pnl = source_df['profit_usd'].sum()
    avg_return = source_df['return_pct'].mean()
    
    print(f'\n{source.upper()} ({n} trades):')
    print(f'  Win Rate: {wins}/{n} = {win_rate:.1f}%')
    print(f'  Total P&L: ${total_pnl:.2f}')
    print(f'  Avg Return: {avg_return:.1f}%')

# Analyze what makes winners vs losers
print('\n' + '='*70)
print('WINNER vs LOSER ANALYSIS')
print('='*70)

winners = df[df['is_winner'] == True]
losers = df[df['is_winner'] == False]

print(f'\nWINNERS ({len(winners)} trades):')
print(f'  Avg return: {winners["return_pct"].mean():.1f}%')
print(f'  Total profit: ${winners["profit_usd"].sum():.2f}')
print(f'  Sources: {winners["source"].value_counts().to_dict()}')

print(f'\nLOSERS ({len(losers)} trades):')
print(f'  Avg return: {losers["return_pct"].mean():.1f}%')
print(f'  Total loss: ${losers["profit_usd"].sum():.2f}')
print(f'  Sources: {losers["source"].value_counts().to_dict()}')

# Analyze by task (AI-Phillip vs Trace_24H-Phillip)
print('\n' + '='*70)
print('PERFORMANCE BY TASK TYPE')
print('='*70)

for task in df['task'].unique():
    task_df = df[df['task'] == task]
    n = len(task_df)
    wins = task_df['is_winner'].sum()
    win_rate = wins / n * 100 if n > 0 else 0
    total_pnl = task_df['profit_usd'].sum()
    
    print(f'\n{task} ({n} trades):')
    print(f'  Win Rate: {wins}/{n} = {win_rate:.1f}%')
    print(f'  Total P&L: ${total_pnl:.2f}')

# Look at TP/SL patterns
print('\n' + '='*70)
print('TAKE PROFIT vs STOP LOSS ANALYSIS')
print('='*70)

# Convert to boolean properly
def str_to_bool(val):
    if pd.isna(val):
        return False
    return str(val).lower() == 'true'

tp1_hit = df['take_profit_1'].apply(str_to_bool)
tp2_hit = df['take_profit_2'].apply(str_to_bool)
tp3_hit = df['take_profit_3'].apply(str_to_bool)
tp4_hit = df['take_profit_4'].apply(str_to_bool)
sl_hit = df['stop_loss_1'].apply(str_to_bool)

print(f'TP1 hit: {tp1_hit.sum()} ({tp1_hit.sum()/len(df)*100:.1f}%)')
print(f'TP2 hit: {tp2_hit.sum()} ({tp2_hit.sum()/len(df)*100:.1f}%)')
print(f'TP3 hit: {tp3_hit.sum()} ({tp3_hit.sum()/len(df)*100:.1f}%)')
print(f'TP4 hit: {tp4_hit.sum()} ({tp4_hit.sum()/len(df)*100:.1f}%)')
print(f'SL hit: {sl_hit.sum()} ({sl_hit.sum()/len(df)*100:.1f}%)')

# Trades that hit TP but then SL (partial profit then loss)
tp_then_sl = (tp1_hit | tp2_hit) & sl_hit
print(f'\nHit TP then SL (partial): {tp_then_sl.sum()} trades')

# Pure SL (no TP hit)
pure_sl = sl_hit & ~tp1_hit & ~tp2_hit & ~tp3_hit & ~tp4_hit
print(f'Pure SL loss (no TP): {pure_sl.sum()} trades')

# Full TP run (all 4 TPs)
full_tp = tp1_hit & tp2_hit & tp3_hit & tp4_hit
print(f'Full TP run (all 4): {full_tp.sum()} trades')

# Winners breakdown
print('\n' + '='*70)
print('WINNERS DETAIL')
print('='*70)
for idx, row in winners.iterrows():
    symbol = row.get('coin_id', 'N/A')
    source = row['source']
    ret = row['return_pct']
    profit = row['profit_usd']
    print(f'{symbol:15} | {source:20} | {ret:>8.1f}% | ${profit:>7.4f}')

# Analyze why some are 100% winners and some 0%
print('\n' + '='*70)
print('WHY SOME ARE 100% AND SOME 0%?')
print('='*70)

# Check market cap differences
print('\n--- Market Cap at Signal vs Bought ---')
winners_mc_diff = (winners['bought_market_cap'] - winners['signal_market_cap']) / winners['signal_market_cap'] * 100
losers_mc_diff = (losers['bought_market_cap'] - losers['signal_market_cap']) / losers['signal_market_cap'] * 100

print(f'Winners: avg MC change from signal to buy: {winners_mc_diff.mean():.1f}%')
print(f'Losers: avg MC change from signal to buy: {losers_mc_diff.mean():.1f}%')

# Check slippage (bought higher than signal = bad)
print('\n--- Entry Slippage ---')
print(f'Winners avg entry slippage: {winners_mc_diff.mean():.1f}%')
print(f'Losers avg entry slippage: {losers_mc_diff.mean():.1f}%')

# Check if we're buying at peak
print('\n--- Buying at Peak? ---')
winners_vs_peak = (winners['highest_mc'] - winners['bought_market_cap']) / winners['bought_market_cap'] * 100
losers_vs_peak = (losers['highest_mc'] - losers['bought_market_cap']) / losers['bought_market_cap'] * 100

print(f'Winners: room to grow after buy: {winners_vs_peak.mean():.1f}%')
print(f'Losers: room to grow after buy: {losers_vs_peak.mean():.1f}%')

# Key recommendation
print('\n' + '='*70)
print('RECOMMENDATIONS')
print('='*70)
print('''
Based on REAL trade data:

1. WHALE source: 7.7% win rate - TERRIBLE. Should be blocked!

2. TG_EARLY_TRENDING: 27.3% - Not good but better

3. SOLANA_TRACKER: 40% on small sample - Monitor

4. PRIMAL: 0% (5 trades) - Sample too small

KEY ISSUES:
- We're entering too late (MC already pumped)
- Stop loss hitting before any profit
- Only 10/58 trades profitable!

MUST FIX:
1. BLOCK whale source immediately
2. Better entry timing (dont buy pumped tokens)
3. Wider stop loss or different SL strategy
''')

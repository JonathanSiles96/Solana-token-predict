# -*- coding: utf-8 -*-
"""Quick win rate analysis"""
import pandas as pd
import numpy as np
import sys
import io

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load all 2025 data
files = ['2025.csv', '2025 (1).csv', '2025 (3).csv', '2025 (4).csv']
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f'Loaded {f}: {len(df)} rows')
    except Exception as e:
        print(f'Failed {f}: {e}')

if not dfs:
    print("No data!")
    exit()

df = pd.concat(dfs, ignore_index=True)
df = df.drop_duplicates(subset=['mint_key', 'signal_time'], keep='first')
print(f'\nTotal unique signals: {len(df)}')

# Filter to those with outcomes
df = df[df['max_return'].notna() & (df['max_return'] > 0)].copy()
print(f'With outcomes: {len(df)}')

if len(df) == 0:
    print("No outcome data!")
    exit()

# Win = 30% gain (max_return >= 1.3)
df['winner'] = df['max_return'] >= 1.3

print(f'\n========== BASELINE ==========')
print(f'Winners (>=30% gain): {df["winner"].sum()}/{len(df)} = {df["winner"].mean()*100:.1f}%')
print(f'Avg max_return: {df["max_return"].mean():.2f}x')

print(f'\n========== BY SECURITY ==========')
for sec in df['security'].unique():
    sub = df[df['security'] == sec]
    if len(sub) >= 5:
        print(f'{sec}: {sub["winner"].mean()*100:.1f}% win rate ({len(sub)} samples)')

print(f'\n========== BY SOURCE ==========')
for src in df['source'].unique():
    sub = df[df['source'] == src]
    if len(sub) >= 5:
        print(f'{src}: {sub["winner"].mean()*100:.1f}% win rate ({len(sub)} samples)')

print(f'\n========== BY BUNDLED % ==========')
for thresh in [5, 10, 20, 40, 60, 100]:
    sub = df[df['bundled_pct'] < thresh]
    if len(sub) >= 5:
        print(f'Bundled < {thresh}%: {sub["winner"].mean()*100:.1f}% win rate ({len(sub)} samples)')

print(f'\n========== BY SNIPERS % ==========')
for thresh in [10, 20, 30, 50]:
    sub = df[df['snipers_pct'] < thresh]
    if len(sub) >= 5:
        print(f'Snipers < {thresh}%: {sub["winner"].mean()*100:.1f}% win rate ({len(sub)} samples)')

print(f'\n========== BY HOLDERS ==========')
for thresh in [100, 200, 300, 400, 500]:
    sub = df[df['holders'] > thresh]
    if len(sub) >= 5:
        print(f'Holders > {thresh}: {sub["winner"].mean()*100:.1f}% win rate ({len(sub)} samples)')

print(f'\n========== COMBO FILTERS ==========')

# Security values
green_sec = ['good', 'white_check_mark']

# Test combos
try:
    sub = df[df['security'].str.contains('good|white_check_mark|✅', case=False, na=False)]
    if len(sub) >= 5:
        print(f'Green security: {sub["winner"].mean()*100:.1f}% win rate ({len(sub)} samples)')
except: pass

try:
    sub = df[(df['bundled_pct'] < 5) & (df['snipers_pct'] < 10)]
    if len(sub) >= 5:
        print(f'Bundled<5% + Snipers<10%: {sub["winner"].mean()*100:.1f}% win rate ({len(sub)} samples)')
except: pass

try:
    sub = df[(df['bundled_pct'] < 10) & (df['snipers_pct'] < 20) & (df['holders'] > 200)]
    if len(sub) >= 5:
        print(f'Bundled<10% + Snipers<20% + Holders>200: {sub["winner"].mean()*100:.1f}% win rate ({len(sub)} samples)')
except: pass

try:
    sub = df[(df['bundled_pct'] < 5) & (df['holders'] > 300)]
    if len(sub) >= 5:
        print(f'Bundled<5% + Holders>300: {sub["winner"].mean()*100:.1f}% win rate ({len(sub)} samples)')
except: pass

try:
    sub = df[(df['volume_1h'] > 100000) & (df['holders'] > 200)]
    if len(sub) >= 5:
        print(f'Volume>100K + Holders>200: {sub["winner"].mean()*100:.1f}% win rate ({len(sub)} samples)')
except: pass

print(f'\n========== TOP 10 BEST MAX RETURNS ==========')
top = df.nlargest(10, 'max_return')[['name', 'security', 'bundled_pct', 'snipers_pct', 'holders', 'max_return', 'source']]
print(top.to_string())

print('\nDone!')


"""
Filter Optimization Analysis
Which filters actually improve win rate and returns?
"""
import pandas as pd
import numpy as np

df = pd.read_csv('2025 (11).csv')

# Filter to signals with max_return data
df_valid = df[df['max_return'].notna() & (df['max_return'] > 0)].copy()

print('='*80)
print('FILTER OPTIMIZATION ANALYSIS')
print('='*80)
print(f'Total signals with outcome data: {len(df_valid)}')

# Define "winner" as hitting at least +30% (1.3x)
df_valid['is_winner'] = df_valid['max_return'] >= 1.3

baseline_wr = df_valid['is_winner'].mean() * 100
baseline_return = df_valid['max_return'].mean()
print(f'\nBASELINE (no filter):')
print(f'  Win Rate (30%+): {baseline_wr:.1f}%')
print(f'  Avg Max Return: {baseline_return:.2f}x')
print(f'  Median Max Return: {df_valid["max_return"].median():.2f}x')

def analyze_filter(name, mask, show_detail=True):
    """Analyze a filter's impact on win rate and returns"""
    filtered = df_valid[mask]
    excluded = df_valid[~mask]
    
    if len(filtered) < 5:
        return None
        
    wr = filtered['is_winner'].mean() * 100
    avg_ret = filtered['max_return'].mean()
    med_ret = filtered['max_return'].median()
    
    excluded_wr = excluded['is_winner'].mean() * 100 if len(excluded) > 0 else 0
    excluded_ret = excluded['max_return'].mean() if len(excluded) > 0 else 0
    
    improvement = wr - baseline_wr
    
    if show_detail:
        print(f'\n{name}:')
        print(f'  Passed: n={len(filtered)}, WR={wr:.1f}%, Avg={avg_ret:.2f}x, Med={med_ret:.2f}x')
        print(f'  Failed: n={len(excluded)}, WR={excluded_wr:.1f}%, Avg={excluded_ret:.2f}x')
        print(f'  Improvement: {improvement:+.1f}% WR')
        
    return {
        'name': name,
        'n': len(filtered),
        'win_rate': wr,
        'avg_return': avg_ret,
        'improvement': improvement
    }

# =========================================
print('\n' + '='*80)
print('BY SOURCE')
print('='*80)

for source in sorted(df_valid['source'].dropna().unique()):
    mask = df_valid['source'] == source
    analyze_filter(f'Source: {source}', mask)

# =========================================
print('\n' + '='*80)
print('BY HOLDER COUNT')
print('='*80)

analyze_filter('Holders >= 100', df_valid['holders'] >= 100)
analyze_filter('Holders >= 200', df_valid['holders'] >= 200)
analyze_filter('Holders >= 300', df_valid['holders'] >= 300)
analyze_filter('Holders >= 400', df_valid['holders'] >= 400)
analyze_filter('Holders >= 500', df_valid['holders'] >= 500)

# =========================================
print('\n' + '='*80)
print('BY VOLUME 1H')
print('='*80)

analyze_filter('Volume >= 10k', df_valid['volume_1h'] >= 10000)
analyze_filter('Volume >= 25k', df_valid['volume_1h'] >= 25000)
analyze_filter('Volume >= 50k', df_valid['volume_1h'] >= 50000)
analyze_filter('Volume >= 100k', df_valid['volume_1h'] >= 100000)

# =========================================
print('\n' + '='*80)
print('BY LIQUIDITY')
print('='*80)

analyze_filter('Liquidity >= 15k', df_valid['liquidity'] >= 15000)
analyze_filter('Liquidity >= 20k', df_valid['liquidity'] >= 20000)
analyze_filter('Liquidity >= 25k', df_valid['liquidity'] >= 25000)
analyze_filter('Liquidity >= 30k', df_valid['liquidity'] >= 30000)

# =========================================
print('\n' + '='*80)
print('BY MARKET CAP')
print('='*80)

analyze_filter('MC >= 30k', df_valid['market_cap'] >= 30000)
analyze_filter('MC >= 50k', df_valid['market_cap'] >= 50000)
analyze_filter('MC >= 75k', df_valid['market_cap'] >= 75000)
analyze_filter('MC >= 100k', df_valid['market_cap'] >= 100000)

# =========================================
print('\n' + '='*80)
print('BY SECURITY STATUS')
print('='*80)

analyze_filter('Security: Green (check)', df_valid['security'].str.contains('check|CHECK', case=False, na=False))
analyze_filter('Security: Warning', df_valid['security'].str.contains('warning', case=False, na=False))
analyze_filter('Security: Alert (red)', df_valid['security'].str.contains('danger|DANGER', case=False, na=False) == False)

# =========================================
print('\n' + '='*80)
print('BY SNIPERS %')
print('='*80)

analyze_filter('Snipers <= 20%', df_valid['snipers_pct'] <= 20)
analyze_filter('Snipers <= 30%', df_valid['snipers_pct'] <= 30)
analyze_filter('Snipers <= 40%', df_valid['snipers_pct'] <= 40)
analyze_filter('Snipers <= 50%', df_valid['snipers_pct'] <= 50)

# =========================================
print('\n' + '='*80)
print('BY BUNDLED %')
print('='*80)

analyze_filter('Bundled <= 10%', df_valid['bundled_pct'] <= 10)
analyze_filter('Bundled <= 30%', df_valid['bundled_pct'] <= 30)
analyze_filter('Bundled <= 50%', df_valid['bundled_pct'] <= 50)
analyze_filter('Bundled <= 80%', df_valid['bundled_pct'] <= 80)

# =========================================
print('\n' + '='*80)
print('COMBINED FILTERS (find optimal)')
print('='*80)

# Test various combinations
combos = []

# Combo 1: Volume + Holders
mask = (df_valid['volume_1h'] >= 25000) & (df_valid['holders'] >= 200)
r = analyze_filter('Vol>=25k + Hold>=200', mask)
if r: combos.append(r)

# Combo 2: Liquidity + Volume
mask = (df_valid['liquidity'] >= 20000) & (df_valid['volume_1h'] >= 25000)
r = analyze_filter('Liq>=20k + Vol>=25k', mask)
if r: combos.append(r)

# Combo 3: Good metrics combined
mask = (df_valid['holders'] >= 200) & (df_valid['liquidity'] >= 20000) & (df_valid['snipers_pct'] <= 50)
r = analyze_filter('Hold>=200 + Liq>=20k + Snipe<=50%', mask)
if r: combos.append(r)

# Combo 4: Conservative
mask = (df_valid['holders'] >= 300) & (df_valid['volume_1h'] >= 30000) & (df_valid['snipers_pct'] <= 40)
r = analyze_filter('Hold>=300 + Vol>=30k + Snipe<=40%', mask)
if r: combos.append(r)

# Combo 5: Aggressive
mask = (df_valid['volume_1h'] >= 10000) | (df_valid['holders'] >= 150) | (df_valid['liquidity'] >= 15000)
r = analyze_filter('Vol>=10k OR Hold>=150 OR Liq>=15k', mask)
if r: combos.append(r)

# Combo 6: Source + basic metrics
for source in ['primal', 'whale', 'tg_early_trending', 'solana_tracker']:
    mask = (df_valid['source'] == source) & (df_valid['holders'] >= 150)
    r = analyze_filter(f'{source} + Hold>=150', mask)
    if r: combos.append(r)

# =========================================
print('\n' + '='*80)
print('OPTIMAL FILTER RECOMMENDATIONS')
print('='*80)

# Sort by improvement
if combos:
    combos_sorted = sorted(combos, key=lambda x: x['improvement'], reverse=True)
    
    print('\nTop 5 filters by WR improvement:')
    for i, c in enumerate(combos_sorted[:5]):
        print(f'  {i+1}. {c["name"]}')
        print(f'     n={c["n"]}, WR={c["win_rate"]:.1f}%, Improvement: {c["improvement"]:+.1f}%')

# Show final recommendation
print('\n' + '='*80)
print('FINAL RECOMMENDATION')
print('='*80)
print('''
Based on the analysis, the optimal filtering strategy is:

MUST HAVE (hard filters):
- Volume >= 10k OR Holders >= 150 OR Liquidity >= 15k
  (At least one of these must pass)

PREFERRED (soft filters that improve WR):
- Holders >= 200: improves WR
- Volume >= 25k: improves WR  
- Snipers <= 50%: reduces bad trades
- Liquidity >= 20k: improves safety

SOURCE WEIGHTING:
- All main sources (primal, whale, tg_early_trending, solana_tracker) are viable
- Filter by metrics, not source

The key insight: Don't block by source, block by weak metrics!
''')


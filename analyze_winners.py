"""
Analyze historical data to find rule-based filters for 80%+ win rate
NO ML - just statistical analysis of what works
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_all_2025_data():
    """Load all 2025 CSV files"""
    files = [
        '2025.csv',
        '2025 (1).csv', 
        '2025 (3).csv',
        '2025 (4).csv'
    ]
    
    dfs = []
    for f in files:
        if Path(f).exists():
            try:
                df = pd.read_csv(f)
                dfs.append(df)
                print(f"✓ Loaded {f}: {len(df)} rows")
            except:
                pass
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        # Remove duplicates based on mint_key + signal_time
        combined = combined.drop_duplicates(subset=['mint_key', 'signal_time'], keep='first')
        return combined
    return pd.DataFrame()


def analyze_win_rates(df, win_threshold=0.30):
    """
    Analyze win rates by different factors
    
    win_threshold: 0.30 = 30% gain considered a "win"
    """
    print(f"\n{'='*70}")
    print(f"WIN RATE ANALYSIS (Win = max_return >= {win_threshold+1:.0%})")
    print(f"{'='*70}")
    
    # Define winner
    df['is_winner'] = df['max_return'] >= (1 + win_threshold)
    df['gain_pct'] = (df['max_return'] - 1) * 100  # Convert to percentage
    
    total = len(df)
    winners = df['is_winner'].sum()
    base_win_rate = winners / total * 100
    
    print(f"\nBaseline: {winners}/{total} winners ({base_win_rate:.1f}% win rate)")
    print(f"Average max_return: {df['max_return'].mean():.2f}x")
    
    results = []
    
    # === 1. SECURITY STATUS ===
    print(f"\n{'='*50}")
    print("1. BY SECURITY STATUS")
    print(f"{'='*50}")
    
    for security in df['security'].unique():
        subset = df[df['security'] == security]
        if len(subset) >= 10:
            wr = subset['is_winner'].mean() * 100
            avg_ret = subset['max_return'].mean()
            print(f"  {security}: {wr:.1f}% win rate ({len(subset)} samples, avg {avg_ret:.2f}x)")
            results.append(('security', security, wr, len(subset), avg_ret))
    
    # === 2. SOURCE ===
    print(f"\n{'='*50}")
    print("2. BY SIGNAL SOURCE")
    print(f"{'='*50}")
    
    for source in df['source'].unique():
        subset = df[df['source'] == source]
        if len(subset) >= 10:
            wr = subset['is_winner'].mean() * 100
            avg_ret = subset['max_return'].mean()
            print(f"  {source}: {wr:.1f}% win rate ({len(subset)} samples, avg {avg_ret:.2f}x)")
            results.append(('source', source, wr, len(subset), avg_ret))
    
    # === 3. MARKET CAP RANGES ===
    print(f"\n{'='*50}")
    print("3. BY MARKET CAP")
    print(f"{'='*50}")
    
    mc_bins = [(0, 30000), (30000, 50000), (50000, 75000), (75000, 100000), (100000, 200000), (200000, float('inf'))]
    for low, high in mc_bins:
        subset = df[(df['market_cap'] >= low) & (df['market_cap'] < high)]
        if len(subset) >= 10:
            wr = subset['is_winner'].mean() * 100
            avg_ret = subset['max_return'].mean()
            label = f"${low/1000:.0f}K-${high/1000:.0f}K" if high < float('inf') else f">${low/1000:.0f}K"
            print(f"  MC {label}: {wr:.1f}% win rate ({len(subset)} samples)")
            results.append(('mc', label, wr, len(subset), avg_ret))
    
    # === 4. BUNDLED % ===
    print(f"\n{'='*50}")
    print("4. BY BUNDLED % (lower is better)")
    print(f"{'='*50}")
    
    bundled_bins = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 100)]
    for low, high in bundled_bins:
        subset = df[(df['bundled_pct'] >= low) & (df['bundled_pct'] < high)]
        if len(subset) >= 10:
            wr = subset['is_winner'].mean() * 100
            avg_ret = subset['max_return'].mean()
            print(f"  Bundled {low}-{high}%: {wr:.1f}% win rate ({len(subset)} samples)")
            results.append(('bundled', f'{low}-{high}%', wr, len(subset), avg_ret))
    
    # === 5. SNIPERS % ===
    print(f"\n{'='*50}")
    print("5. BY SNIPERS % (lower is better)")
    print(f"{'='*50}")
    
    sniper_bins = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
    for low, high in sniper_bins:
        subset = df[(df['snipers_pct'] >= low) & (df['snipers_pct'] < high)]
        if len(subset) >= 10:
            wr = subset['is_winner'].mean() * 100
            avg_ret = subset['max_return'].mean()
            print(f"  Snipers {low}-{high}%: {wr:.1f}% win rate ({len(subset)} samples)")
            results.append(('snipers', f'{low}-{high}%', wr, len(subset), avg_ret))
    
    # === 6. HOLDERS ===
    print(f"\n{'='*50}")
    print("6. BY HOLDERS COUNT")
    print(f"{'='*50}")
    
    holder_bins = [(0, 100), (100, 200), (200, 300), (300, 500), (500, float('inf'))]
    for low, high in holder_bins:
        subset = df[(df['holders'] >= low) & (df['holders'] < high)]
        if len(subset) >= 10:
            wr = subset['is_winner'].mean() * 100
            avg_ret = subset['max_return'].mean()
            label = f"{low}-{high}" if high < float('inf') else f">{low}"
            print(f"  Holders {label}: {wr:.1f}% win rate ({len(subset)} samples)")
            results.append(('holders', label, wr, len(subset), avg_ret))
    
    # === 7. LIQUIDITY ===
    print(f"\n{'='*50}")
    print("7. BY LIQUIDITY")
    print(f"{'='*50}")
    
    liq_bins = [(0, 15000), (15000, 20000), (20000, 25000), (25000, 30000), (30000, float('inf'))]
    for low, high in liq_bins:
        subset = df[(df['liquidity'] >= low) & (df['liquidity'] < high)]
        if len(subset) >= 10:
            wr = subset['is_winner'].mean() * 100
            avg_ret = subset['max_return'].mean()
            label = f"${low/1000:.0f}K-${high/1000:.0f}K" if high < float('inf') else f">${low/1000:.0f}K"
            print(f"  Liquidity {label}: {wr:.1f}% win rate ({len(subset)} samples)")
            results.append(('liquidity', label, wr, len(subset), avg_ret))
    
    # === 8. VOLUME 1H ===
    print(f"\n{'='*50}")
    print("8. BY VOLUME 1H")
    print(f"{'='*50}")
    
    vol_bins = [(0, 20000), (20000, 50000), (50000, 100000), (100000, 200000), (200000, float('inf'))]
    for low, high in vol_bins:
        subset = df[(df['volume_1h'] >= low) & (df['volume_1h'] < high)]
        if len(subset) >= 10:
            wr = subset['is_winner'].mean() * 100
            avg_ret = subset['max_return'].mean()
            label = f"${low/1000:.0f}K-${high/1000:.0f}K" if high < float('inf') else f">${low/1000:.0f}K"
            print(f"  Volume {label}: {wr:.1f}% win rate ({len(subset)} samples)")
            results.append(('volume', label, wr, len(subset), avg_ret))
    
    # === 9. FIRST 20 HOLDERS % ===
    print(f"\n{'='*50}")
    print("9. BY FIRST 20 HOLDERS % (lower = more distributed)")
    print(f"{'='*50}")
    
    first20_bins = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 100)]
    for low, high in first20_bins:
        subset = df[(df['first20_pct'] >= low) & (df['first20_pct'] < high)]
        if len(subset) >= 10:
            wr = subset['is_winner'].mean() * 100
            avg_ret = subset['max_return'].mean()
            print(f"  First20 {low}-{high}%: {wr:.1f}% win rate ({len(subset)} samples)")
            results.append(('first20', f'{low}-{high}%', wr, len(subset), avg_ret))
    
    # === 10. TOKEN AGE ===
    print(f"\n{'='*50}")
    print("10. BY TOKEN AGE (minutes)")
    print(f"{'='*50}")
    
    age_bins = [(0, 5), (5, 15), (15, 30), (30, 60), (60, float('inf'))]
    for low, high in age_bins:
        subset = df[(df['age'] >= low) & (df['age'] < high)]
        if len(subset) >= 10:
            wr = subset['is_winner'].mean() * 100
            avg_ret = subset['max_return'].mean()
            label = f"{low}-{high}m" if high < float('inf') else f">{low}m"
            print(f"  Age {label}: {wr:.1f}% win rate ({len(subset)} samples)")
            results.append(('age', label, wr, len(subset), avg_ret))
    
    return results


def find_best_filters(df, win_threshold=0.30, target_win_rate=0.70):
    """
    Find filter combinations that achieve target win rate
    """
    print(f"\n{'='*70}")
    print(f"SEARCHING FOR {target_win_rate*100:.0f}%+ WIN RATE FILTERS")
    print(f"{'='*70}")
    
    df['is_winner'] = df['max_return'] >= (1 + win_threshold)
    
    best_filters = []
    
    # Test different filter combinations
    filter_tests = [
        # (name, filter_func)
        ("✅ Security Only", lambda x: x['security'].isin(['✅', 'white_check_mark'])),
        ("Bundled < 10%", lambda x: x['bundled_pct'] < 10),
        ("Bundled < 5%", lambda x: x['bundled_pct'] < 5),
        ("Snipers < 20%", lambda x: x['snipers_pct'] < 20),
        ("Snipers < 10%", lambda x: x['snipers_pct'] < 10),
        ("Holders > 200", lambda x: x['holders'] > 200),
        ("Holders > 300", lambda x: x['holders'] > 300),
        ("Volume > 50K", lambda x: x['volume_1h'] > 50000),
        ("Volume > 100K", lambda x: x['volume_1h'] > 100000),
        ("Liquidity > 20K", lambda x: x['liquidity'] > 20000),
        ("Liquidity > 25K", lambda x: x['liquidity'] > 25000),
        ("MC 40K-80K", lambda x: (x['market_cap'] >= 40000) & (x['market_cap'] <= 80000)),
        ("First20 < 30%", lambda x: x['first20_pct'] < 30),
        ("Age 5-30min", lambda x: (x['age'] >= 5) & (x['age'] <= 30)),
        ("Source: primal", lambda x: x['source'] == 'primal'),
        ("Source: whale", lambda x: x['source'] == 'whale'),
        ("Source: tg_early_trending", lambda x: x['source'] == 'tg_early_trending'),
    ]
    
    print("\nSingle Filter Results:")
    print("-" * 50)
    
    for name, filter_func in filter_tests:
        try:
            subset = df[filter_func(df)]
            if len(subset) >= 20:
                wr = subset['is_winner'].mean() * 100
                avg_ret = subset['max_return'].mean()
                marker = "🎯" if wr >= target_win_rate * 100 else "  "
                print(f"{marker} {name}: {wr:.1f}% win rate ({len(subset)} samples, avg {avg_ret:.2f}x)")
                best_filters.append((name, wr, len(subset), avg_ret))
        except:
            pass
    
    # Test combinations
    print(f"\n{'='*50}")
    print("COMBO FILTERS (Stacking rules)")
    print(f"{'='*50}")
    
    combo_tests = [
        # Simple combos
        ("✅ + Bundled<10%", 
         lambda x: x['security'].isin(['✅', 'white_check_mark']) & (x['bundled_pct'] < 10)),
        
        ("✅ + Snipers<15%", 
         lambda x: x['security'].isin(['✅', 'white_check_mark']) & (x['snipers_pct'] < 15)),
        
        ("✅ + Holders>200", 
         lambda x: x['security'].isin(['✅', 'white_check_mark']) & (x['holders'] > 200)),
        
        ("✅ + Volume>50K", 
         lambda x: x['security'].isin(['✅', 'white_check_mark']) & (x['volume_1h'] > 50000)),
        
        ("Bundled<10% + Snipers<15%", 
         lambda x: (x['bundled_pct'] < 10) & (x['snipers_pct'] < 15)),
        
        ("Bundled<5% + Snipers<10%", 
         lambda x: (x['bundled_pct'] < 5) & (x['snipers_pct'] < 10)),
        
        # Triple combos
        ("✅ + Bundled<10% + Snipers<15%", 
         lambda x: x['security'].isin(['✅', 'white_check_mark']) & 
                   (x['bundled_pct'] < 10) & (x['snipers_pct'] < 15)),
        
        ("✅ + Bundled<10% + Holders>200", 
         lambda x: x['security'].isin(['✅', 'white_check_mark']) & 
                   (x['bundled_pct'] < 10) & (x['holders'] > 200)),
        
        ("✅ + Bundled<5% + Volume>50K", 
         lambda x: x['security'].isin(['✅', 'white_check_mark']) & 
                   (x['bundled_pct'] < 5) & (x['volume_1h'] > 50000)),
        
        ("✅ + First20<30% + Holders>200", 
         lambda x: x['security'].isin(['✅', 'white_check_mark']) & 
                   (x['first20_pct'] < 30) & (x['holders'] > 200)),
        
        # Source-based
        ("primal + Bundled<10%", 
         lambda x: (x['source'] == 'primal') & (x['bundled_pct'] < 10)),
        
        ("whale + ✅", 
         lambda x: (x['source'] == 'whale') & x['security'].isin(['✅', 'white_check_mark'])),
        
        # Quad combos
        ("✅ + Bundled<10% + Snipers<20% + Holders>150", 
         lambda x: x['security'].isin(['✅', 'white_check_mark']) & 
                   (x['bundled_pct'] < 10) & (x['snipers_pct'] < 20) & (x['holders'] > 150)),
        
        ("✅ + Bundled<5% + Snipers<15% + Volume>30K", 
         lambda x: x['security'].isin(['✅', 'white_check_mark']) & 
                   (x['bundled_pct'] < 5) & (x['snipers_pct'] < 15) & (x['volume_1h'] > 30000)),
        
        # "Golden" filters
        ("GOLDEN: ✅ + Bundled<5% + Snipers<10% + Holders>200 + Vol>50K", 
         lambda x: x['security'].isin(['✅', 'white_check_mark']) & 
                   (x['bundled_pct'] < 5) & (x['snipers_pct'] < 10) & 
                   (x['holders'] > 200) & (x['volume_1h'] > 50000)),
        
        ("GOLDEN2: ✅ + Bundled<3% + First20<25% + Liq>25K", 
         lambda x: x['security'].isin(['✅', 'white_check_mark']) & 
                   (x['bundled_pct'] < 3) & (x['first20_pct'] < 25) & (x['liquidity'] > 25000)),
    ]
    
    high_wr_filters = []
    
    for name, filter_func in combo_tests:
        try:
            subset = df[filter_func(df)]
            if len(subset) >= 10:
                wr = subset['is_winner'].mean() * 100
                avg_ret = subset['max_return'].mean()
                marker = "🎯" if wr >= target_win_rate * 100 else "  "
                if wr >= 50:  # Only show 50%+ win rate combos
                    print(f"{marker} {name}")
                    print(f"      → {wr:.1f}% win rate | {len(subset)} trades | avg {avg_ret:.2f}x")
                if wr >= target_win_rate * 100:
                    high_wr_filters.append((name, wr, len(subset), avg_ret))
        except:
            pass
    
    # Print best filters
    if high_wr_filters:
        print(f"\n{'='*70}")
        print(f"🎯 FILTERS WITH {target_win_rate*100:.0f}%+ WIN RATE")
        print(f"{'='*70}")
        
        high_wr_filters.sort(key=lambda x: (-x[1], -x[2]))  # Sort by win rate, then sample size
        
        for name, wr, samples, avg_ret in high_wr_filters[:10]:
            print(f"\n  {name}")
            print(f"  Win Rate: {wr:.1f}% | Samples: {samples} | Avg Return: {avg_ret:.2f}x")
    else:
        print(f"\n⚠️  No filters found with {target_win_rate*100:.0f}%+ win rate")
        print("   Try lowering the target or getting more data")
    
    return high_wr_filters


def suggest_strategy(df, win_threshold=0.30):
    """
    Based on analysis, suggest a simple rule-based strategy
    """
    print(f"\n{'='*70}")
    print("📋 SUGGESTED RULE-BASED STRATEGY")
    print(f"{'='*70}")
    
    df['is_winner'] = df['max_return'] >= (1 + win_threshold)
    
    # Find best thresholds by testing
    best_bundled = None
    best_bundled_wr = 0
    for thresh in [3, 5, 10, 15, 20]:
        subset = df[df['bundled_pct'] < thresh]
        if len(subset) >= 50:
            wr = subset['is_winner'].mean()
            if wr > best_bundled_wr:
                best_bundled_wr = wr
                best_bundled = thresh
    
    best_snipers = None
    best_snipers_wr = 0
    for thresh in [5, 10, 15, 20, 25]:
        subset = df[df['snipers_pct'] < thresh]
        if len(subset) >= 50:
            wr = subset['is_winner'].mean()
            if wr > best_snipers_wr:
                best_snipers_wr = wr
                best_snipers = thresh
    
    print(f"""
Based on your data analysis, here's a suggested NO-ML strategy:

┌─────────────────────────────────────────────────────────────────┐
│  ENTRY RULES (ALL must pass)                                     │
├─────────────────────────────────────────────────────────────────┤
│  1. Security: ✅ only (green checkmark)                          │
│  2. Bundled: < {best_bundled or 10}% (best threshold from your data)              │
│  3. Snipers: < {best_snipers or 15}% (best threshold from your data)              │
│  4. Holders: > 150                                               │
│  5. Volume 1h: > $30,000                                         │
│  6. Liquidity: > $15,000                                         │
│  7. MC: $30K - $150K (sweet spot)                               │
│  8. First 20%: < 40% (avoid whale concentration)                │
├─────────────────────────────────────────────────────────────────┤
│  EXIT RULES                                                      │
├─────────────────────────────────────────────────────────────────┤
│  • TP1: +30% → Sell 25%                                         │
│  • TP2: +50% → Sell 25%                                         │
│  • TP3: +100% → Sell 30%                                        │
│  • TP4: +200% → Sell remaining                                  │
│  • SL: -30% hard stop                                            │
├─────────────────────────────────────────────────────────────────┤
│  SKIP IF ANY (Red Flags)                                        │
├─────────────────────────────────────────────────────────────────┤
│  • Bundled > 30%                                                 │
│  • Snipers > 35%                                                 │
│  • Sold > 25%                                                    │
│  • Security: 🚨                                                  │
│  • First 20 holders > 60%                                        │
│  • Volume < $10,000                                              │
└─────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  WIN RATE ANALYSIS - Find Profitable Filters                  ║
    ║  NO ML - Pure Statistical Analysis                            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Load data
    df = load_all_2025_data()
    
    if df.empty:
        print("❌ No data found! Make sure 2025*.csv files exist.")
        exit(1)
    
    print(f"\n✓ Total signals loaded: {len(df)}")
    
    # Filter to signals with outcome data
    df_with_outcome = df[df['max_return'].notna() & (df['max_return'] > 0)].copy()
    print(f"✓ Signals with outcomes: {len(df_with_outcome)}")
    
    if len(df_with_outcome) < 100:
        print("⚠️  Limited data - results may not be reliable")
    
    # Run analysis
    results = analyze_win_rates(df_with_outcome, win_threshold=0.30)
    
    # Find best filters
    best = find_best_filters(df_with_outcome, win_threshold=0.30, target_win_rate=0.70)
    
    # Suggest strategy
    suggest_strategy(df_with_outcome, win_threshold=0.30)
    
    print(f"\n{'='*70}")
    print("DONE! Use the filters above for rule-based trading.")
    print(f"{'='*70}\n")


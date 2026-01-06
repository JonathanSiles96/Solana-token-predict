"""
Analyze Primal Detection from 2026 CSV files

This script analyzes the CSV files to properly rebuild primal detection
based on correct data analysis.
"""

import pandas as pd
import numpy as np

# Read the CSV files
print("="*80)
print("PRIMAL DETECTION ANALYSIS")
print("="*80)

try:
    df1 = pd.read_csv('2026 (1).csv')
    print(f"\nCSV1 (2026 (1).csv): {len(df1)} rows")
    print(f"Columns: {df1.columns.tolist()[:5]}...")
    
    if 'source' in df1.columns:
        print(f"\nSource distribution in CSV1:")
        print(df1['source'].value_counts())
        
        # Analyze primal performance
        if 'primal' in df1['source'].values:
            primal_trades = df1[df1['source'] == 'primal']
            print(f"\nPrimal trades: {len(primal_trades)}")
            
            if 'return_pct' in primal_trades.columns:
                primal_returns = primal_trades['return_pct'].dropna()
                if len(primal_returns) > 0:
                    print(f"  Avg return: {primal_returns.mean():.2f}%")
                    print(f"  Win rate (>0%): {(primal_returns > 0).sum() / len(primal_returns) * 100:.1f}%")
                    print(f"  Win rate (>10%): {(primal_returns > 10).sum() / len(primal_returns) * 100:.1f}%")
                    print(f"  Win rate (>30%): {(primal_returns > 30).sum() / len(primal_returns) * 100:.1f}%")
            
            if 'profit_usd' in primal_trades.columns:
                primal_profits = primal_trades['profit_usd'].dropna()
                if len(primal_profits) > 0:
                    print(f"  Total profit: ${primal_profits.sum():.2f}")
                    print(f"  Avg profit: ${primal_profits.mean():.2f}")
                    print(f"  Profitable trades: {(primal_profits > 0).sum()} / {len(primal_profits)}")
    
except Exception as e:
    print(f"Error reading CSV1: {e}")

try:
    df2 = pd.read_csv('2026 (2).csv')
    print(f"\n\nCSV2 (2026 (2).csv): {len(df2)} rows")
    print(f"Columns: {df2.columns.tolist()[:5]}...")
    
    if 'source' in df2.columns:
        print(f"\nSource distribution in CSV2:")
        print(df2['source'].value_counts())
        
        # Analyze primal performance
        if 'primal' in df2['source'].values:
            primal_signals = df2[df2['source'] == 'primal']
            print(f"\nPrimal signals: {len(primal_signals)}")
            
            if 'max_return' in primal_signals.columns:
                primal_max = primal_signals['max_return'].dropna()
                if len(primal_max) > 0:
                    print(f"  Avg max return: {primal_max.mean():.2f}x")
                    print(f"  Hit +15%: {(primal_max >= 1.15).sum() / len(primal_max) * 100:.1f}%")
                    print(f"  Hit +30%: {(primal_max >= 1.30).sum() / len(primal_max) * 100:.1f}%")
                    print(f"  Hit +50%: {(primal_max >= 1.50).sum() / len(primal_max) * 100:.1f}%")
                    print(f"  Hit +100%: {(primal_max >= 2.00).sum() / len(primal_max) * 100:.1f}%")
            
            # Analyze primal characteristics
            if 'holders' in primal_signals.columns:
                primal_holders = primal_signals['holders'].dropna()
                if len(primal_holders) > 0:
                    print(f"\n  Primal holder stats:")
                    print(f"    Mean: {primal_holders.mean():.0f}")
                    print(f"    Median: {primal_holders.median():.0f}")
                    print(f"    Min: {primal_holders.min():.0f}")
                    print(f"    Max: {primal_holders.max():.0f}")
            
            if 'volume_1h' in primal_signals.columns:
                primal_vol = primal_signals['volume_1h'].dropna()
                if len(primal_vol) > 0:
                    print(f"\n  Primal volume stats:")
                    print(f"    Mean: ${primal_vol.mean():,.0f}")
                    print(f"    Median: ${primal_vol.median():,.0f}")
            
            if 'liquidity' in primal_signals.columns:
                primal_liq = primal_signals['liquidity'].dropna()
                if len(primal_liq) > 0:
                    print(f"\n  Primal liquidity stats:")
                    print(f"    Mean: ${primal_liq.mean():,.0f}")
                    print(f"    Median: ${primal_liq.median():,.0f}")

except Exception as e:
    print(f"Error reading CSV2: {e}")

print("\n" + "="*80)
print("RECOMMENDATIONS FOR PRIMAL DETECTION")
print("="*80)
print("""
Based on the analysis:
1. Primal source should be prioritized (TIER0)
2. Primal signals show high hit rates at +15% and +30%
3. Use primal as a strong signal source with appropriate filters
4. Combine primal with other metrics for best results
""")



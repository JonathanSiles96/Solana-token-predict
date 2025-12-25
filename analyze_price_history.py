"""
PRICE HISTORY ANALYSIS FOR OPTIMAL SL/TP
=========================================

Uses Philip's API to fetch price updates and calculate:
1. min_return (max drawdown before recovery)
2. Optimal SL levels that keep winners while cutting losers
3. Post-TP trailing behavior

API: https://op.xcapitala.com/api/trades/{trade_id}/price-updates-external
Header: api-key = dfhgdfrh45yery3463
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import sys
import io

# Fix unicode output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# API Configuration
API_BASE = "https://op.xcapitala.com/api"
API_KEY = "dfhgdfrh45yery3463"
HEADERS = {"api-key": API_KEY}


def fetch_price_updates(trade_id: str, date: str = None, limit: int = 1000) -> list:
    """
    Fetch price update history for a trade
    
    Args:
        trade_id: The trade ID to fetch
        date: Start date (format: "2021-01-01 00:00:00")
        limit: Max records (max 10000)
    
    Returns:
        List of price updates
    """
    url = f"{API_BASE}/trades/{trade_id}/price-updates-external"
    
    params = {"limit": min(limit, 10000)}
    if date:
        params["date"] = date
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"Request error: {e}")
        return None


def calculate_returns(price_updates: list, entry_price: float = None) -> dict:
    """
    Calculate min/max returns from price updates
    
    Returns:
        dict with min_return, max_return, timestamps, and analysis
    """
    if not price_updates or len(price_updates) == 0:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(price_updates)
    
    # Try to find price column (could be 'price', 'mc', 'market_cap', etc.)
    price_col = None
    for col in ['price', 'mc', 'market_cap', 'value', 'usd_price']:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Convert to numeric
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.dropna(subset=[price_col])
    
    if len(df) == 0:
        return None
    
    # Use first price as entry if not provided
    if entry_price is None:
        entry_price = df[price_col].iloc[0]
    
    if entry_price == 0:
        return None
    
    # Calculate returns
    df['return_pct'] = ((df[price_col] - entry_price) / entry_price) * 100
    
    # Find max and min
    max_return = df['return_pct'].max()
    min_return = df['return_pct'].min()
    
    max_idx = df['return_pct'].idxmax()
    min_idx = df['return_pct'].idxmin()
    
    # Check if min came before max (typical dip before pump pattern)
    min_before_max = min_idx < max_idx if pd.notna(min_idx) and pd.notna(max_idx) else None
    
    return {
        'entry_price': entry_price,
        'max_return': max_return,
        'min_return': min_return,
        'max_price': df.loc[max_idx, price_col] if pd.notna(max_idx) else None,
        'min_price': df.loc[min_idx, price_col] if pd.notna(min_idx) else None,
        'min_before_max': min_before_max,
        'num_updates': len(df),
        'price_column': price_col
    }


def test_api():
    """Test the API connection"""
    print("="*60)
    print("TESTING API CONNECTION")
    print("="*60)
    
    # Try with a sample trade_id format
    test_ids = [
        "test123",  # Simple test
    ]
    
    for test_id in test_ids:
        print(f"\nTrying trade_id: {test_id}")
        result = fetch_price_updates(test_id, limit=5)
        
        if result:
            print(f"SUCCESS! Got {len(result)} records")
            print(f"Sample: {result[:2]}")
            return True
        else:
            print("No data or error")
    
    return False


def analyze_from_csv(csv_path: str, max_trades: int = 50):
    """
    Analyze trades from CSV file
    
    Will try different trade_id formats based on available columns
    """
    print("="*60)
    print("ANALYZING TRADES FROM CSV")
    print("="*60)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} signals")
    print(f"Columns: {df.columns.tolist()}")
    
    # Potential trade ID columns
    id_candidates = ['trade_id', 'pair_key', 'mint_key']
    
    trade_id_col = None
    for col in id_candidates:
        if col in df.columns:
            trade_id_col = col
            print(f"\nUsing '{col}' as trade_id")
            break
    
    if trade_id_col is None:
        print("ERROR: No suitable trade ID column found!")
        return None
    
    # Sample unique IDs
    unique_ids = df[trade_id_col].dropna().unique()[:max_trades]
    print(f"Processing {len(unique_ids)} unique trades...")
    
    results = []
    
    for i, trade_id in enumerate(unique_ids):
        print(f"\n[{i+1}/{len(unique_ids)}] Trade: {str(trade_id)[:20]}...")
        
        # Fetch price history
        price_data = fetch_price_updates(str(trade_id))
        
        if price_data:
            returns = calculate_returns(price_data)
            if returns:
                results.append({
                    'trade_id': trade_id,
                    **returns
                })
                print(f"  Max: {returns['max_return']:.1f}%, Min: {returns['min_return']:.1f}%")
        
        # Rate limit
        time.sleep(0.5)
    
    if not results:
        print("\nNo valid results!")
        return None
    
    # Analyze results
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nTotal trades analyzed: {len(results_df)}")
    print(f"\nMax Return Distribution:")
    print(f"  Mean: {results_df['max_return'].mean():.1f}%")
    print(f"  Median: {results_df['max_return'].median():.1f}%")
    print(f"  Min: {results_df['max_return'].min():.1f}%")
    print(f"  Max: {results_df['max_return'].max():.1f}%")
    
    print(f"\nMin Return (Drawdown) Distribution:")
    print(f"  Mean: {results_df['min_return'].mean():.1f}%")
    print(f"  Median: {results_df['min_return'].median():.1f}%")
    print(f"  Worst: {results_df['min_return'].min():.1f}%")
    
    # Optimal SL Analysis
    print("\n" + "="*60)
    print("OPTIMAL SL ANALYSIS")
    print("="*60)
    
    # For each potential SL level, count how many trades would survive
    sl_levels = [-5, -8, -10, -12, -15, -18, -20, -25, -30, -35, -40]
    
    print(f"\n{'SL Level':<12} {'Survived':<12} {'Stopped Out':<15} {'Winners Lost'}")
    print("-"*60)
    
    for sl in sl_levels:
        # Would survive = min_return > sl
        survived = (results_df['min_return'] > sl).sum()
        stopped = len(results_df) - survived
        
        # Winners lost = hit SL but would have been profitable
        winners_lost = ((results_df['min_return'] <= sl) & (results_df['max_return'] >= 30)).sum()
        
        print(f"{sl}%".ljust(12) + f"{survived}".ljust(12) + f"{stopped}".ljust(15) + f"{winners_lost}")
    
    # Find optimal SL
    print("\n" + "="*60)
    print("OPTIMAL SL RECOMMENDATION")
    print("="*60)
    
    # Optimal = SL that keeps most winners while cutting most losers
    best_sl = None
    best_score = -999
    
    for sl in sl_levels:
        winners_kept = ((results_df['min_return'] > sl) & (results_df['max_return'] >= 30)).sum()
        losers_cut = ((results_df['min_return'] <= sl) & (results_df['max_return'] < 30)).sum()
        winners_lost = ((results_df['min_return'] <= sl) & (results_df['max_return'] >= 30)).sum()
        losers_kept = ((results_df['min_return'] > sl) & (results_df['max_return'] < 30)).sum()
        
        # Score: winners_kept + losers_cut - 2*winners_lost
        score = winners_kept + losers_cut - 2 * winners_lost
        
        if score > best_score:
            best_score = score
            best_sl = sl
    
    print(f"\nRecommended Initial SL: {best_sl}%")
    print(f"(Maximizes keeping winners while cutting losers)")
    
    return results_df


def test_with_sample_id(trade_id: str):
    """Test API with a specific trade ID"""
    print(f"\nTesting with trade_id: {trade_id}")
    
    result = fetch_price_updates(trade_id, limit=100)
    
    if result:
        print(f"SUCCESS! Got {len(result)} price updates")
        
        if len(result) > 0:
            print(f"\nSample data structure:")
            print(f"Keys: {result[0].keys() if isinstance(result[0], dict) else 'N/A'}")
            print(f"First record: {result[0]}")
            
            # Try to calculate returns
            returns = calculate_returns(result)
            if returns:
                print(f"\nReturns analysis:")
                print(f"  Max return: {returns['max_return']:.1f}%")
                print(f"  Min return (drawdown): {returns['min_return']:.1f}%")
                print(f"  Updates count: {returns['num_updates']}")
        
        return result
    else:
        print("No data returned")
        return None


def analyze_with_trade_ids(trade_ids: list):
    """
    Analyze a list of numeric trade IDs
    
    Args:
        trade_ids: List of numeric trade IDs from Philip's system
    """
    print("="*60)
    print(f"ANALYZING {len(trade_ids)} TRADES")
    print("="*60)
    
    results = []
    
    for i, trade_id in enumerate(trade_ids):
        print(f"\n[{i+1}/{len(trade_ids)}] Trade ID: {trade_id}")
        
        price_data = fetch_price_updates(str(trade_id))
        
        if price_data and len(price_data) > 0:
            returns = calculate_returns(price_data)
            if returns:
                results.append({
                    'trade_id': trade_id,
                    **returns
                })
                print(f"  Max: {returns['max_return']:.1f}%, Min: {returns['min_return']:.1f}%")
        
        time.sleep(0.3)  # Rate limit
    
    if not results:
        print("\nNo valid results!")
        return None
    
    # Full analysis
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("DRAWDOWN ANALYSIS FOR SL OPTIMIZATION")
    print("="*60)
    
    print(f"\nTotal trades analyzed: {len(results_df)}")
    
    # Separate winners and losers
    winners = results_df[results_df['max_return'] >= 30]
    losers = results_df[results_df['max_return'] < 30]
    
    print(f"\nWinners (max_return >= 30%): {len(winners)}")
    print(f"Losers (max_return < 30%): {len(losers)}")
    
    if len(winners) > 0:
        print(f"\nWINNERS DRAWDOWN ANALYSIS:")
        print(f"  Avg drawdown before recovery: {winners['min_return'].mean():.1f}%")
        print(f"  Median drawdown: {winners['min_return'].median():.1f}%")
        print(f"  Worst drawdown: {winners['min_return'].min():.1f}%")
        print(f"  Best (smallest) drawdown: {winners['min_return'].max():.1f}%")
        
        # What SL would keep 80% of winners?
        percentiles = [50, 60, 70, 80, 90, 95]
        print(f"\n  Drawdown percentiles (for SL setting):")
        for p in percentiles:
            val = winners['min_return'].quantile((100-p)/100)
            print(f"    {p}% of winners stayed above: {val:.1f}%")
    
    if len(losers) > 0:
        print(f"\nLOSERS DRAWDOWN ANALYSIS:")
        print(f"  Avg drawdown: {losers['min_return'].mean():.1f}%")
        print(f"  Most went to: {losers['min_return'].min():.1f}%")
    
    # Optimal SL recommendation
    print("\n" + "="*60)
    print("OPTIMAL SL RECOMMENDATION")
    print("="*60)
    
    sl_levels = [-8, -10, -12, -15, -18, -20, -25, -30]
    
    print(f"\n{'SL':<8} {'Winners Kept':<15} {'Winners Lost':<15} {'Losers Cut':<12} {'Score'}")
    print("-"*70)
    
    best_sl = None
    best_score = -999
    
    for sl in sl_levels:
        winners_kept = (winners['min_return'] > sl).sum() if len(winners) > 0 else 0
        winners_lost = (winners['min_return'] <= sl).sum() if len(winners) > 0 else 0
        losers_cut = (losers['min_return'] <= sl).sum() if len(losers) > 0 else 0
        losers_kept = (losers['min_return'] > sl).sum() if len(losers) > 0 else 0
        
        # Score = winners_kept + losers_cut - 2*winners_lost
        score = winners_kept + losers_cut - 2 * winners_lost
        
        print(f"{sl}%".ljust(8) + 
              f"{winners_kept}/{len(winners)}".ljust(15) + 
              f"{winners_lost}".ljust(15) + 
              f"{losers_cut}/{len(losers)}".ljust(12) + 
              f"{score}")
        
        if score > best_score:
            best_score = score
            best_sl = sl
    
    print(f"\n*** RECOMMENDED INITIAL SL: {best_sl}% ***")
    
    # Save results
    results_df.to_csv('drawdown_analysis.csv', index=False)
    print(f"\nResults saved to drawdown_analysis.csv")
    
    return results_df


def main():
    print("="*60)
    print("PRICE HISTORY ANALYZER")
    print("="*60)
    print(f"API: {API_BASE}")
    print(f"API Key: {API_KEY[:10]}...")
    print("\nNOTE: API requires NUMERIC trade IDs from Philip's system")
    print("      The CSV pair_key/mint_key (Solana addresses) won't work")
    
    # OPTION 1: Test with a specific trade ID
    # Uncomment and replace with actual trade ID to test:
    # test_with_sample_id("12345")
    
    # OPTION 2: Analyze a list of trade IDs
    # Uncomment and provide actual trade IDs:
    # trade_ids = [12345, 12346, 12347, ...]  # Get these from Philip
    # analyze_with_trade_ids(trade_ids)
    
    print("\n" + "="*60)
    print("TO USE THIS SCRIPT:")
    print("="*60)
    print("""
1. Get numeric trade IDs from Philip's system
   - Either in the CSV export
   - Or via another API endpoint

2. Then run one of:
   
   # Test single trade:
   test_with_sample_id("12345")
   
   # Analyze multiple trades:
   trade_ids = [12345, 12346, ...]
   analyze_with_trade_ids(trade_ids)

3. The analysis will show:
   - Optimal SL level based on drawdowns
   - Winner vs Loser drawdown patterns
   - Recommendations for post-TP SL updates
""")


if __name__ == "__main__":
    main()


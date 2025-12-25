"""
Backfill Historical Analytics Data
===================================

This script:
1. Adds min_return columns to existing database
2. Fetches historical OHLCV data from GeckoTerminal (FREE API)
3. Calculates min_return (max drawdown) for existing signals

Usage:
    python backfill_analytics.py
"""

import sqlite3
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

DB_PATH = "data/signals.db"

# GeckoTerminal API (FREE, no key needed)
GECKO_BASE_URL = "https://api.geckoterminal.com/api/v2"


def migrate_database():
    """Add min_return columns if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if columns exist
    cursor.execute("PRAGMA table_info(token_signals)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'min_return' not in columns:
        print("Adding min_return columns...")
        cursor.execute("ALTER TABLE token_signals ADD COLUMN min_return REAL")
        cursor.execute("ALTER TABLE token_signals ADD COLUMN min_return_mc REAL")
        cursor.execute("ALTER TABLE token_signals ADD COLUMN min_return_timestamp TIMESTAMP")
        conn.commit()
        print("Columns added successfully!")
    else:
        print("min_return columns already exist")
    
    conn.close()


def get_pool_address(mint_address: str) -> str:
    """Get the main pool address for a token from GeckoTerminal"""
    try:
        url = f"{GECKO_BASE_URL}/networks/solana/tokens/{mint_address}"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        # Get top pool
        pools = data.get('data', {}).get('relationships', {}).get('top_pools', {}).get('data', [])
        if pools:
            return pools[0].get('id', '').replace('solana_', '')
        
        return None
        
    except Exception as e:
        print(f"Error getting pool: {e}")
        return None


def get_ohlcv_data(pool_address: str, timeframe: str = "minute", aggregate: int = 15) -> pd.DataFrame:
    """
    Get OHLCV candle data from GeckoTerminal
    
    Args:
        pool_address: The pool address
        timeframe: "minute", "hour", or "day"
        aggregate: Number of timeframe units per candle (e.g., 15 for 15-minute candles)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        url = f"{GECKO_BASE_URL}/networks/solana/pools/{pool_address}/ohlcv/{timeframe}"
        params = {"aggregate": aggregate, "limit": 1000}  # Max 1000 candles
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        candles = data.get('data', {}).get('attributes', {}).get('ohlcv_list', [])
        
        if not candles:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
        
    except Exception as e:
        print(f"Error getting OHLCV: {e}")
        return None


def calculate_returns(df: pd.DataFrame, signal_time: datetime, signal_price: float) -> dict:
    """
    Calculate max and min returns from OHLCV data after signal time
    
    Returns:
        dict with max_return, min_return, and timestamps
    """
    # Filter to candles after signal time
    df_after = df[df['timestamp'] >= signal_time].copy()
    
    if df_after.empty:
        return None
    
    # Calculate returns
    df_after['high_return'] = (df_after['high'] - signal_price) / signal_price
    df_after['low_return'] = (df_after['low'] - signal_price) / signal_price
    
    # Find max and min
    max_idx = df_after['high_return'].idxmax()
    min_idx = df_after['low_return'].idxmin()
    
    return {
        'max_return': df_after.loc[max_idx, 'high_return'],
        'max_return_timestamp': df_after.loc[max_idx, 'timestamp'],
        'min_return': df_after.loc[min_idx, 'low_return'],
        'min_return_timestamp': df_after.loc[min_idx, 'timestamp']
    }


def backfill_signal(cursor, signal: dict) -> bool:
    """Backfill min_return for a single signal"""
    mint_key = signal['mint_key']
    signal_at = signal['signal_at']
    signal_mc = signal['signal_mc']
    
    print(f"  Processing {mint_key[:12]}... ", end="")
    
    # Get pool address
    pool_address = get_pool_address(mint_key)
    if not pool_address:
        print("No pool found")
        return False
    
    # Get OHLCV data
    df = get_ohlcv_data(pool_address)
    if df is None or df.empty:
        print("No OHLCV data")
        return False
    
    # Parse signal time
    try:
        signal_time = datetime.fromisoformat(signal_at.replace('Z', '+00:00'))
    except:
        signal_time = datetime.strptime(signal_at, '%Y-%m-%d %H:%M:%S')
    
    # We need signal price, not MC. Estimate from first available price
    first_candle = df[df['timestamp'] >= signal_time].head(1)
    if first_candle.empty:
        print("No data after signal")
        return False
    
    signal_price = first_candle['open'].values[0]
    
    # Calculate returns
    returns = calculate_returns(df, signal_time, signal_price)
    if not returns:
        print("Could not calculate returns")
        return False
    
    # Update database
    cursor.execute("""
        UPDATE token_signals 
        SET min_return = ?, min_return_timestamp = ?
        WHERE mint_key = ? AND signal_at = ?
    """, (returns['min_return'], returns['min_return_timestamp'], mint_key, signal_at))
    
    print(f"max={returns['max_return']*100:.1f}%, min={returns['min_return']*100:.1f}%")
    return True


def main():
    print("="*60)
    print("BACKFILL HISTORICAL ANALYTICS")
    print("="*60)
    
    # Step 1: Migrate database
    print("\n1. Checking database schema...")
    migrate_database()
    
    # Step 2: Get signals to backfill
    print("\n2. Finding signals to backfill...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get signals with max_return but no min_return
    cursor.execute("""
        SELECT mint_key, signal_at, signal_mc, max_return
        FROM token_signals 
        WHERE max_return IS NOT NULL 
        AND min_return IS NULL
        ORDER BY signal_at DESC
        LIMIT 50
    """)
    
    signals = [dict(row) for row in cursor.fetchall()]
    print(f"   Found {len(signals)} signals to backfill")
    
    if not signals:
        print("\n   No signals need backfilling!")
        conn.close()
        return
    
    # Step 3: Backfill each signal
    print("\n3. Backfilling min_return data...")
    print("   (Using GeckoTerminal API - FREE, no key needed)")
    print()
    
    success = 0
    for i, signal in enumerate(signals):
        print(f"[{i+1}/{len(signals)}]", end=" ")
        if backfill_signal(cursor, signal):
            success += 1
        
        # Rate limit: GeckoTerminal allows ~30 req/min
        time.sleep(2)
        
        # Commit every 10
        if (i + 1) % 10 == 0:
            conn.commit()
            print(f"   Committed {i+1} records")
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {success}/{len(signals)} signals backfilled")
    print("="*60)


if __name__ == "__main__":
    main()


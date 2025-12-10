"""
Helper utility functions
"""

import json
import re
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime


def parse_signal_params(signal_params_str: str) -> Dict[str, Any]:
    """
    Parse signal_params JSON string into dictionary
    
    Args:
        signal_params_str: JSON string containing signal parameters
        
    Returns:
        Dictionary of parsed parameters
    """
    if pd.isna(signal_params_str) or signal_params_str == '':
        return {}
    
    try:
        # Handle double-quoted JSON strings
        params = json.loads(signal_params_str)
        return params
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_gain_from_text(text: str) -> Optional[float]:
    """
    Extract gain/return percentage from entry signal text
    
    Examples:
        "📈 ‎P4D is up 58% 📈" -> 0.58
        "📈 ‎ON is up 2.1X 📈" -> 1.1
        "📈 ‎CLASH is up 75X 📈" -> 74.0
        
    Args:
        text: Message text containing gain information
        
    Returns:
        Gain as decimal (e.g., 0.58 for 58% gain) or None
    """
    if pd.isna(text):
        return None
    
    # Pattern for percentage gains: "up XX%"
    pct_match = re.search(r'up\s+(\d+(?:\.\d+)?)%', text)
    if pct_match:
        return float(pct_match.group(1)) / 100.0
    
    # Pattern for X multipliers: "up XXX"
    x_match = re.search(r'up\s+(\d+(?:\.\d+)?)X', text)
    if x_match:
        multiplier = float(x_match.group(1))
        return multiplier - 1.0  # Convert to gain (5X = 4.0 gain)
    
    return None


def extract_ticker_from_text(text: str) -> Optional[str]:
    """
    Extract ticker symbol from message text
    
    Example:
        "📈 ‎P4D is up 58% 📈" -> "P4D"
        
    Args:
        text: Message text
        
    Returns:
        Ticker symbol or None
    """
    if pd.isna(text):
        return None
    
    # Pattern: ticker followed by "is up"
    match = re.search(r'‎(\w+)\s+is\s+up', text)
    if match:
        return match.group(1)
    
    return None


def extract_mc_from_text(text: str) -> tuple[Optional[float], Optional[float]]:
    """
    Extract market cap values from text
    
    Example:
        "$93.2K —> $148K" -> (93200, 148000)
        
    Args:
        text: Message text
        
    Returns:
        Tuple of (entry_mc, exit_mc) or (None, None)
    """
    if pd.isna(text):
        return None, None
    
    # Pattern: $XX.XK —> $XX.XK
    match = re.search(r'\$(\d+(?:\.\d+)?)K\s*—>\s*\$(\d+(?:\.\d+)?)(K|M)', text)
    if match:
        entry_mc = float(match.group(1)) * 1000
        exit_value = float(match.group(2))
        unit = match.group(3)
        
        if unit == 'K':
            exit_mc = exit_value * 1000
        elif unit == 'M':
            exit_mc = exit_value * 1000000
        else:
            exit_mc = exit_value
            
        return entry_mc, exit_mc
    
    return None, None


def calculate_volatility(top_mc: float, signal_mc: float, best_mc: float = None) -> float:
    """
    Calculate volatility metric from market cap range
    
    Args:
        top_mc: Top market cap observed
        signal_mc: Market cap at signal time
        best_mc: Best market cap ever (optional)
        
    Returns:
        Volatility score
    """
    if pd.isna(signal_mc) or signal_mc == 0:
        return 0.0
    
    if pd.isna(top_mc):
        top_mc = signal_mc
        
    # Range as percentage of signal MC
    volatility = (top_mc - signal_mc) / signal_mc
    
    # If best_mc available, factor it in
    if best_mc and not pd.isna(best_mc) and best_mc > top_mc:
        volatility = max(volatility, (best_mc - signal_mc) / signal_mc)
    
    return volatility


def calculate_liquidity_score(liquidity: float, mc: float, volume_1h: float) -> float:
    """
    Calculate comprehensive liquidity score
    
    Args:
        liquidity: Liquidity pool size
        mc: Market cap
        volume_1h: 1-hour volume
        
    Returns:
        Liquidity score (0-1 range, higher is better)
    """
    if pd.isna(mc) or mc == 0:
        return 0.0
    
    # Liquidity to MC ratio (healthy: 30-40%)
    liq_ratio = liquidity / mc if not pd.isna(liquidity) and liquidity > 0 else 0
    liq_score = min(liq_ratio / 0.4, 1.0)  # Normalize to 0-1
    
    # Volume to liquidity ratio (good: 1-3x)
    vol_ratio = volume_1h / liquidity if not pd.isna(volume_1h) and not pd.isna(liquidity) and liquidity > 0 else 0
    vol_score = min(vol_ratio / 3.0, 1.0)
    
    # Combined score
    return 0.6 * liq_score + 0.4 * vol_score


def calculate_risk_score(bundled_pct: float, snipers_pct: float, 
                         sold_pct: float, security: str) -> float:
    """
    Calculate risk score (lower is better/safer)
    
    Args:
        bundled_pct: Percentage of bundled transactions
        snipers_pct: Percentage of sniper transactions
        sold_pct: Percentage sold
        security: Security status (✅, ⚠️, 🚨)
        
    Returns:
        Risk score (0-1, where 0 is safest)
    """
    risk = 0.0
    
    # Bundled percentage risk (bad if >20%)
    if not pd.isna(bundled_pct):
        risk += min(bundled_pct / 100.0 / 0.2, 1.0) * 0.3
    
    # Snipers percentage risk (bad if >30%)
    if not pd.isna(snipers_pct):
        risk += min(snipers_pct / 100.0 / 0.3, 1.0) * 0.3
    
    # Sold percentage risk (bad if >10%)
    if not pd.isna(sold_pct):
        risk += min(sold_pct / 100.0 / 0.1, 1.0) * 0.2
    
    # Security status
    if pd.isna(security):
        risk += 0.1
    elif security == '✅':
        risk += 0.0
    elif security == '⚠️':
        risk += 0.1
    elif security == '🚨':
        risk += 0.2
    
    return min(risk, 1.0)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling NaN and zero division
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
        
    Returns:
        Division result or default
    """
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return default
    return numerator / denominator


def parse_age_to_minutes(age_str: str) -> Optional[float]:
    """
    Parse age string to minutes
    
    Examples:
        "1m" -> 1
        "15m" -> 15
        "1h" -> 60
        "3d" -> 4320
        
    Args:
        age_str: Age string
        
    Returns:
        Age in minutes or None
    """
    if pd.isna(age_str):
        return None
    
    age_str = str(age_str).lower().strip()
    
    # Match patterns like "1m", "15m", "1h", "3d", "55s"
    match = re.match(r'(\d+)([smhd])', age_str)
    if not match:
        return None
    
    value = float(match.group(1))
    unit = match.group(2)
    
    if unit == 's':
        return value / 60.0
    elif unit == 'm':
        return value
    elif unit == 'h':
        return value * 60
    elif unit == 'd':
        return value * 1440
    
    return None


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format number with K/M/B suffixes
    
    Args:
        num: Number to format
        decimals: Decimal places
        
    Returns:
        Formatted string
    """
    if pd.isna(num):
        return "N/A"
    
    if abs(num) >= 1e9:
        return f"${num/1e9:.{decimals}f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.{decimals}f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.{decimals}f}K"
    else:
        return f"${num:.{decimals}f}"


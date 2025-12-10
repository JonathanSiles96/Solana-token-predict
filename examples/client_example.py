"""
Example Python client for Token Signal API

This script demonstrates how to:
1. Send token signals to the API
2. Score tokens
3. Get ranked token lists
4. Stream signals via webhook
"""

import requests
from datetime import datetime
from typing import Dict, List, Optional
import time


class TokenSignalClient:
    """Client for Token Signal API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client
        
        Args:
            base_url: API base URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def health_check(self) -> Dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def send_signal(self, signal_data: Dict) -> Dict:
        """
        Send a token signal
        
        Args:
            signal_data: Token signal data
            
        Returns:
            Response dict
        """
        response = self.session.post(
            f"{self.base_url}/signals/token",
            json=signal_data
        )
        response.raise_for_status()
        return response.json()
    
    def send_bulk_signals(self, signals: List[Dict]) -> Dict:
        """
        Send multiple signals at once
        
        Args:
            signals: List of signal dicts
            
        Returns:
            Response dict
        """
        response = self.session.post(
            f"{self.base_url}/signals/token/bulk",
            json={"signals": signals}
        )
        response.raise_for_status()
        return response.json()
    
    def send_entry_signal(self, ticker: str, entry_mc: float, 
                         source: str = "early_trending",
                         current_mc: Optional[float] = None,
                         gain_pct: Optional[float] = None) -> Dict:
        """
        Send an entry signal (trade tracking)
        
        Args:
            ticker: Token ticker
            entry_mc: Entry market cap
            source: Signal source
            current_mc: Current market cap (optional)
            gain_pct: Current gain percentage (optional)
            
        Returns:
            Response dict
        """
        entry_data = {
            "ticker": ticker,
            "signal_at": datetime.utcnow().isoformat() + "Z",
            "entry_mc": entry_mc,
            "source": source
        }
        
        if current_mc:
            entry_data["current_mc"] = current_mc
        if gain_pct is not None:
            entry_data["gain_pct"] = gain_pct
        
        response = self.session.post(
            f"{self.base_url}/signals/entry",
            json=entry_data
        )
        response.raise_for_status()
        return response.json()
    
    def update_entry_gain(self, ticker: str, signal_at: str,
                         current_mc: float, gain_pct: float) -> Dict:
        """
        Update entry signal with current gain
        
        Args:
            ticker: Token ticker
            signal_at: Original signal timestamp
            current_mc: Current market cap
            gain_pct: Current gain percentage
            
        Returns:
            Response dict
        """
        params = {
            "current_mc": current_mc,
            "gain_pct": gain_pct,
            "signal_at": signal_at
        }
        
        response = self.session.put(
            f"{self.base_url}/signals/entry/{ticker}/update",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def query_signals(self, limit: int = 100, offset: int = 0,
                     min_mc: Optional[float] = None,
                     max_mc: Optional[float] = None,
                     security: Optional[str] = None) -> Dict:
        """
        Query token signals
        
        Args:
            limit: Max results
            offset: Pagination offset
            min_mc: Minimum market cap filter
            max_mc: Maximum market cap filter
            security: Security status filter
            
        Returns:
            Response dict with signals
        """
        params = {"limit": limit, "offset": offset}
        
        if min_mc:
            params["min_mc"] = min_mc
        if max_mc:
            params["max_mc"] = max_mc
        if security:
            params["security"] = security
        
        response = self.session.get(
            f"{self.base_url}/signals/tokens",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def get_signal_by_mint(self, mint_key: str) -> Dict:
        """
        Get latest signal for a specific token
        
        Args:
            mint_key: Token mint address
            
        Returns:
            Signal data
        """
        response = self.session.get(
            f"{self.base_url}/signals/tokens/{mint_key}"
        )
        response.raise_for_status()
        return response.json()
    
    def score_token(self, token_data: Dict) -> Dict:
        """
        Score a token using ML model
        
        Args:
            token_data: Token features
            
        Returns:
            Score response with predictions and recommendations
        """
        response = self.session.post(
            f"{self.base_url}/tokens/score",
            json=token_data
        )
        response.raise_for_status()
        return response.json()
    
    def score_token_by_mint(self, mint_key: str) -> Dict:
        """
        Score a token by its mint address
        
        Args:
            mint_key: Token mint address
            
        Returns:
            Score response
        """
        response = self.session.post(
            f"{self.base_url}/tokens/score/{mint_key}"
        )
        response.raise_for_status()
        return response.json()
    
    def get_ranked_tokens(self, top_n: int = 20, 
                         min_confidence: float = 0.5) -> Dict:
        """
        Get ranked list of top tokens
        
        Args:
            top_n: Number of top tokens to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            Ranked tokens list
        """
        params = {
            "top_n": top_n,
            "min_confidence": min_confidence
        }
        
        response = self.session.get(
            f"{self.base_url}/tokens/rank",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict:
        """Get API statistics"""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()


def example_usage():
    """Example usage of the client"""
    
    # Initialize client
    client = TokenSignalClient("http://localhost:8000")
    
    print("=" * 80)
    print("Token Signal API Client - Example Usage")
    print("=" * 80)
    
    # 1. Health check
    print("\n1. Health Check")
    print("-" * 80)
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Model loaded: {health['model_loaded']}")
    print(f"Total signals: {health['total_signals']}")
    
    # 2. Send a token signal
    print("\n2. Sending Token Signal")
    print("-" * 80)
    
    signal = {
        "mint_key": "ExampleMint123456789ABCDEFGH",
        "signal_at": datetime.utcnow().isoformat() + "Z",
        "signal_mc": 75000,
        "signal_liquidity": 28000,
        "signal_volume_1h": 35000,
        "signal_holders": 450,
        "signal_dev_sol": 15,
        "signal_security": "✅",
        "signal_bundled_pct": 5.0,
        "signal_snipers_pct": None,
        "signal_sold_pct": 7.0,
        "signal_first_20_pct": 25,
        "signal_source": "example_client"
    }
    
    result = client.send_signal(signal)
    print(f"Signal sent: ID={result['id']}")
    
    # 3. Score the token
    print("\n3. Scoring Token")
    print("-" * 80)
    
    score = client.score_token(signal)
    print(f"Predicted Gain: {score['predicted_gain_pct']:.1f}%")
    print(f"Confidence: {score['confidence']:.2f}")
    print(f"Risk-Adjusted Score: {score['risk_adjusted_score']:.2f}")
    print(f"Recommendation: {'🟢 GO' if score['go_decision'] else '🔴 SKIP'}")
    print(f"TP Levels: {score['recommended_tp_levels']}")
    print(f"SL Level: {score['recommended_sl']:.2f}")
    print(f"Position Size Factor: {score['position_size_factor']:.2f}")
    print(f"Notes: {score['notes']}")
    
    # 4. Send entry signal
    print("\n4. Tracking Trade Entry")
    print("-" * 80)
    
    entry = client.send_entry_signal(
        ticker="EXAMPLE",
        entry_mc=75000,
        source="example_client"
    )
    print(f"Entry signal created: ID={entry['id']}")
    
    # Simulate gain update after some time
    time.sleep(1)
    
    update = client.update_entry_gain(
        ticker="EXAMPLE",
        signal_at=signal['signal_at'],
        current_mc=95000,
        gain_pct=0.267  # 26.7% gain
    )
    print(f"Entry updated: {update['message']}")
    
    # 5. Query signals
    print("\n5. Querying Signals")
    print("-" * 80)
    
    query_result = client.query_signals(
        limit=5,
        min_mc=50000,
        security="✅"
    )
    print(f"Found {query_result['count']} signals matching criteria")
    
    # 6. Get ranked tokens
    print("\n6. Getting Top Ranked Tokens")
    print("-" * 80)
    
    ranked = client.get_ranked_tokens(top_n=10, min_confidence=0.5)
    print(f"Total tokens evaluated: {ranked['total_tokens']}")
    print(f"\nTop 5 Tokens:")
    for i, token in enumerate(ranked['top_tokens'][:5], 1):
        print(f"  {i}. {token['mint_key'][:16]}... | "
              f"Gain: {token['predicted_gain']:.2f} | "
              f"Confidence: {token['confidence']:.2f} | "
              f"Score: {token['score']:.2f}")
    
    # 7. Get statistics
    print("\n7. API Statistics")
    print("-" * 80)
    
    stats = client.get_stats()
    print(f"Total signals: {stats['total_signals']}")
    print(f"Total entries: {stats['total_entry_signals']}")
    if stats.get('avg_gain'):
        print(f"Average gain: {stats['avg_gain']*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()


# -*- coding: utf-8 -*-
"""Test the new data-driven GO/SKIP logic"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.models.token_scorer import TokenScorer

# Test the new logic
scorer = TokenScorer()
scorer.trained = True
scorer.model = None
scorer.scaler = None
scorer.feature_names = []

# Mock score_token
def mock_score(data):
    return 1.5, 0.8
scorer.score_token = mock_score

print('='*60)
print('Testing NEW Data-Driven GO/SKIP Logic')
print('='*60)
print()

tests = [
    {'name': 'TIER1 - whale source', 'signal_source': 'whale', 'holders': 100, 'liquidity': 20000, 'bundled_pct': 50},
    {'name': 'TIER1 - tg_early_trending', 'signal_source': 'tg_early_trending', 'holders': 50, 'liquidity': 10000},
    {'name': 'TIER2 - green security', 'signal_security': 'white_check_mark', 'holders': 100, 'liquidity': 20000},
    {'name': 'TIER2 - holders>400', 'holders': 450, 'liquidity': 20000, 'signal_source': 'primal'},
    {'name': 'TIER2 - bundled<5 holders>300', 'bundled_pct': 3, 'holders': 350, 'liquidity': 20000},
    {'name': 'TIER2 - volume>100K holders>200', 'volume_1h': 150000, 'holders': 250, 'liquidity': 20000},
    {'name': 'TIER3 - bundled<10 snipers<20 holders>200', 'bundled_pct': 8, 'snipers_pct': 15, 'holders': 250, 'liquidity': 20000},
    {'name': 'TIER4 - baseline', 'holders': 250, 'liquidity': 25000, 'bundled_pct': 15, 'signal_source': 'solana_tracker'},
    {'name': 'SKIP - low holders', 'holders': 50, 'liquidity': 20000},
    {'name': 'SKIP - danger security', 'signal_security': 'danger', 'holders': 200, 'liquidity': 20000},
]

for test in tests:
    data = {
        'signal_mc': 50000,
        'signal_liquidity': test.get('liquidity', 20000),
        'signal_volume_1h': test.get('volume_1h', 50000),
        'signal_holders': test.get('holders', 200),
        'signal_bundled_pct': test.get('bundled_pct', 10),
        'signal_snipers_pct': test.get('snipers_pct', 10),
        'signal_sold_pct': test.get('sold_pct', 5),
        'signal_security': test.get('signal_security', 'warning'),
        'signal_source': test.get('signal_source', 'unknown'),
        'signal_first_20_pct': 25,
        'age_minutes': 10,
    }
    
    result = scorer.recommend_parameters(data)
    status = 'GO  ' if result['go_decision'] else 'SKIP'
    print(f"{status} | {test['name']}")
    print(f"      {result['notes'][:70]}")
    print(f"      Position: {result['position_size_factor']*100:.0f}% | SL: {result['recommended_sl']*100:.0f}%")
    print()

print('='*60)
print('All tests completed!')
print('='*60)


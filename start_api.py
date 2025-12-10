"""
Start the API server in DEVELOPMENT mode

Lightweight dev version:
- Uses existing model (no retraining)
- Auto-reload enabled
- Debug logging

For production with training: python start_api_production.py
"""

import uvicorn
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    model_path = Path("outputs/models/token_scorer.pkl")
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  Solana Token Filtering & Gain Drivers API               ║
    ║  Real-time Signal Processing & Token Scoring             ║
    ║  DEVELOPMENT MODE                                         ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    if not model_path.exists():
        print("""
    ⚠️  No model found!
    
    Train the model first:
      python start_api_production.py
    
    API will start but predictions won't work.
    """)
    else:
        print(f"    ✅ Model loaded: {model_path}")
    
    print("""
    📡 API: http://localhost:8000
    📚 Docs: http://localhost:8000/docs
    
    🎯 Features: Predictions • Dynamic Orders • Trailing Stop
    🔄 Auto-reload enabled
    
    💡 Use start_api_production.py for training + production
    """)
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


"""
Start the API server in PRODUCTION mode

This script:
1. Loads training data from CSV reports
2. Trains/updates the model with real data
3. Validates model performance
4. Starts the production API server with:
   - Model predictions
   - Dynamic orders with trailing stop loss
   - Signal management
   - All endpoints

Usage:
    python start_api_production.py
"""

import uvicorn
import sys
import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))


def parse_token_data(text):
    """Extract token features from Slack message"""
    data = {}
    
    # MC
    mc_match = re.search(r'💰 MC: \$([0-9,.]+[KM]?)', text)
    if mc_match:
        mc_str = mc_match.group(1).replace(',', '')
        if 'K' in mc_str:
            data['signal_mc'] = float(mc_str.replace('K', '')) * 1000
        elif 'M' in mc_str:
            data['signal_mc'] = float(mc_str.replace('M', '')) * 1000000
        else:
            data['signal_mc'] = float(mc_str)
    
    # Liquidity
    liq_match = re.search(r'💧 Liq: \$([0-9,.]+[KM]?)', text)
    if liq_match:
        liq_str = liq_match.group(1).replace(',', '')
        if 'K' in liq_str:
            data['signal_liquidity'] = float(liq_str.replace('K', '')) * 1000
        elif 'M' in liq_str:
            data['signal_liquidity'] = float(liq_str.replace('M', '')) * 1000000
        else:
            data['signal_liquidity'] = float(liq_str)
    
    # Volume
    vol_match = re.search(r'Vol: \$([0-9,.]+[KM]?) \[1h\]', text)
    if vol_match:
        vol_str = vol_match.group(1).replace(',', '')
        if 'K' in vol_str:
            data['signal_volume_1h'] = float(vol_str.replace('K', '')) * 1000
        elif 'M' in vol_str:
            data['signal_volume_1h'] = float(vol_str.replace('M', '')) * 1000000
        else:
            data['signal_volume_1h'] = float(vol_str)
    
    # Holders
    hold_match = re.search(r'👥 Hodls: ([0-9,]+)', text)
    if hold_match:
        data['signal_holders'] = int(hold_match.group(1).replace(',', ''))
    
    # Bundled %
    bundled_match = re.search(r'Bundled: ([0-9.]+)%', text)
    if bundled_match:
        data['signal_bundled_pct'] = float(bundled_match.group(1))
    
    # Snipers %
    sniper_match = re.search(r'Snipers: \d+ • ([0-9.]+)%', text)
    if sniper_match:
        data['signal_snipers_pct'] = float(sniper_match.group(1))
    
    # Sold %
    sold_match = re.search(r'Sold: ([0-9.]+)%', text)
    if sold_match:
        data['signal_sold_pct'] = float(sold_match.group(1))
    
    # Security
    if '✅' in text:
        data['signal_security'] = 'good'
    elif '🚨' in text:
        data['signal_security'] = 'bad'
    elif '⚠️' in text:
        data['signal_security'] = 'warning'
    
    # First 20%
    first20_match = re.search(r'First 20: ([0-9]+)%', text)
    if first20_match:
        data['signal_first_20_pct'] = float(first20_match.group(1))
    
    # Dev SOL
    dev_match = re.search(r'Dev: ([0-9]+) SOL', text)
    if dev_match:
        data['signal_dev_sol'] = float(dev_match.group(1))
    
    return data


def parse_gain_from_entry_signal(text):
    """Extract gain from entry signal"""
    # Percentage
    pct_match = re.search(r'is up ([0-9.]+)% 📈 from', text)
    if pct_match:
        return float(pct_match.group(1)) / 100
    
    # X multiplier
    x_match = re.search(r'is up ([0-9.]+)X 📈 from', text)
    if x_match:
        return float(x_match.group(1)) - 1
    
    return None


def extract_ticker(text):
    """Extract ticker from message"""
    ticker_match = re.search(r'‎([A-Z0-9]+) is up', text)
    if ticker_match:
        return ticker_match.group(1)
    
    ticker_match2 = re.search(r'\$([A-Z0-9]+)', text)
    if ticker_match2:
        return ticker_match2.group(1)
    
    return None


def load_and_train_model():
    """Load CSV data and train model"""
    
    print("\n" + "="*70)
    print("STEP 1: LOADING TRAINING DATA")
    print("="*70)
    
    csv_files = ['solanaearlytrending_messages.csv', 'whaletrending_messages.csv']
    
    # Check if CSV files exist
    csv_exists = all(Path(f).exists() for f in csv_files)
    model_exists = Path("outputs/models/token_scorer.pkl").exists()
    
    if not csv_exists:
        if model_exists:
            print("⚠️  CSV files not found, using existing model")
            print(f"   Model: outputs/models/token_scorer.pkl")
            return True
        else:
            print("❌ No CSV files and no existing model found")
            print("\nRequired files:")
            print("  - solanaearlytrending_messages.csv")
            print("  - whaletrending_messages.csv")
            print("\nOR existing model at:")
            print("  - outputs/models/token_scorer.pkl")
            return False
    
    print("✓ Found CSV files")
    
    # Load CSVs
    early_df = pd.read_csv('solanaearlytrending_messages.csv')
    whale_df = pd.read_csv('whaletrending_messages.csv')
    
    print(f"✓ Loaded {len(early_df):,} early messages")
    print(f"✓ Loaded {len(whale_df):,} whale messages")
    
    # Combine
    all_messages = pd.concat([
        early_df.assign(source='early_trending'),
        whale_df.assign(source='whale_trending')
    ], ignore_index=True)
    
    # Extract gains
    entry_signals = all_messages[all_messages['type'] == 'entry_signal'].copy()
    
    gains_data = []
    for _, row in entry_signals.iterrows():
        ticker = extract_ticker(row['text'])
        gain = parse_gain_from_entry_signal(row['text'])
        
        if ticker and gain is not None:
            gains_data.append({
                'ticker': ticker,
                'gain': gain,
                'date_time': row['date_time'],
                'source': row['source']
            })
    
    gains_df = pd.DataFrame(gains_data)
    print(f"✓ Extracted {len(gains_df):,} gains")
    
    # Extract token features
    new_signals = all_messages[
        (all_messages['type'] == 'new_trending') | 
        (all_messages['type'] == 'new_whale')
    ].copy()
    
    tokens_data = []
    for _, row in new_signals.iterrows():
        ticker = extract_ticker(row['text'])
        if not ticker:
            continue
        
        features = parse_token_data(row['text'])
        if not features:
            continue
        
        features['ticker'] = ticker
        features['date_time'] = row['date_time']
        features['source'] = row['source']
        features['signal_at'] = row['date_time']
        features['mint_key'] = f"{ticker}_mint"
        
        tokens_data.append(features)
    
    tokens_df = pd.DataFrame(tokens_data)
    print(f"✓ Extracted {len(tokens_df):,} tokens")
    
    # Match tokens with gains
    max_gains = gains_df.groupby('ticker')['gain'].max().reset_index()
    max_gains.columns = ['ticker', 'max_return']
    
    training_data = tokens_df.merge(max_gains, on='ticker', how='inner')
    
    print(f"✓ Matched {len(training_data):,} tokens with gains")
    
    if len(training_data) < 50:
        print(f"\n⚠️  Only {len(training_data)} training samples")
        print("   Minimum 100 recommended for good performance")
        
        if model_exists:
            print("   Using existing model instead")
            return True
        
        print("   Training anyway...")
    
    print(f"\n📊 Training Data:")
    print(f"  Mean gain: {training_data['max_return'].mean()*100:.1f}%")
    print(f"  Median: {training_data['max_return'].median()*100:.1f}%")
    print(f"  > 50%: {(training_data['max_return'] > 0.5).sum()} ({(training_data['max_return'] > 0.5).sum()/len(training_data)*100:.1f}%)")
    print(f"  > 100%: {(training_data['max_return'] > 1.0).sum()} ({(training_data['max_return'] > 1.0).sum()/len(training_data)*100:.1f}%)")
    
    # Train model
    print("\n" + "="*70)
    print("STEP 2: TRAINING MODEL")
    print("="*70)
    
    from src.models.token_scorer import TokenScorer
    from src.data_processing.data_loader import FeatureExtractor
    
    # Prepare features
    feature_extractor = FeatureExtractor()
    training_data = feature_extractor.extract_numeric_features(training_data)
    training_data = feature_extractor.add_derived_features(training_data)
    
    # Create output directories
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    
    # Train
    print(f"\nTraining gradient boosting model...")
    scorer = TokenScorer(model_type='gradient_boosting')
    
    try:
        metrics = scorer.train(training_data, target_col='max_return')
        
        print(f"\n✅ MODEL TRAINED:")
        print(f"  Test R²: {metrics['test']['r2']:.3f}")
        print(f"  Test RMSE: {metrics['test']['rmse']:.3f}")
        print(f"  Test MAE: {metrics['test']['mae']:.3f}")
        
        if metrics['test']['r2'] > 0.5:
            print(f"  🎯 Excellent quality!")
        elif metrics['test']['r2'] > 0.3:
            print(f"  ✅ Good quality")
        else:
            print(f"  ⚠️  Fair quality - more data recommended")
        
        # Save model
        model_path = "outputs/models/token_scorer.pkl"
        scorer.save(model_path)
        print(f"\n✓ Model saved: {model_path}")
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'metric': ['train_r2', 'test_r2', 'test_rmse', 'test_mae'],
            'value': [
                metrics['train']['r2'],
                metrics['test']['r2'],
                metrics['test']['rmse'],
                metrics['test']['mae']
            ]
        })
        metrics_df.to_csv("outputs/reports/model_metrics.csv", index=False)
        
        if 'feature_importance' in metrics:
            metrics['feature_importance'].to_csv("outputs/reports/feature_importance.csv", index=False)
            
            print(f"\n🎯 Top 5 Features:")
            for idx, row in metrics['feature_importance'].head(5).iterrows():
                print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        
        if model_exists:
            print("   Using existing model instead")
            return True
        
        return False


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  Solana Token Filtering & Gain Drivers API               ║
    ║  Real-time Signal Processing & Token Scoring             ║
    ║  PRODUCTION MODE                                          ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Train/load model
    success = load_and_train_model()
    
    if not success:
        print("\n❌ Cannot start: No model available")
        print("\nPlease provide either:")
        print("  1. CSV files for training, OR")
        print("  2. Pre-trained model at outputs/models/token_scorer.pkl")
        sys.exit(1)
    
    # Start API
    print("\n" + "="*70)
    print("STEP 3: STARTING PRODUCTION API")
    print("="*70)
    
    print("""
    ✅ API SERVER READY!
    
    📡 Endpoints:
      • http://0.0.0.0:8000/docs - API Documentation
      • http://0.0.0.0:8000/predict - Get predictions
      • http://0.0.0.0:8000/orders/create - Create dynamic orders
      • http://0.0.0.0:8000/orders/update/{mint_key} - Update orders
      • http://0.0.0.0:8000/reports/max-return-stats - Statistics
    
    🎯 Features Enabled:
      ✅ Model predictions with confidence
      ✅ Dynamic order management
      ✅ Trailing stop loss
      ✅ Adaptive position sizing
      ✅ Multi-level take profits
      ✅ Behavioral strategies
      ✅ Signal management
      ✅ Auto-retraining
    
    🚀 Starting production server with 4 workers...
    """)
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled for production
        workers=4,     # Multiple workers for production
        log_level="info",
        access_log=True
    )


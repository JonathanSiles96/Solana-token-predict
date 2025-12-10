"""
Strategy-Based Token Scorer

This module combines ML predictions with strategy-based filtering
to make trading decisions based on your defined rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.strategy.config import TradingStrategy, get_strategy
from src.models.token_scorer import TokenScorer
from src.data_processing.data_loader import FeatureExtractor
from src.api.database import SignalDatabase


class StrategyScorer:
    """
    Combines ML model predictions with strategy-based filtering
    
    This scorer:
    1. Gets ML prediction (predicted gain, confidence)
    2. Applies strategy filters (your rules)
    3. Returns GO/SKIP decision with full reasoning
    4. Stores prediction for outcome tracking
    """
    
    def __init__(self, 
                 model_path: str = "outputs/models/token_scorer.pkl",
                 db_path: str = "data/signals.db",
                 strategy: Optional[TradingStrategy] = None):
        """
        Initialize strategy scorer
        
        Args:
            model_path: Path to trained model
            db_path: Path to signals database
            strategy: Trading strategy config (uses default if None)
        """
        self.model_path = Path(model_path)
        self.strategy = strategy or get_strategy()
        self.db = SignalDatabase(db_path)
        self.feature_extractor = FeatureExtractor()
        
        # Load model
        self.model: Optional[TokenScorer] = None
        self.load_model()
    
    def load_model(self):
        """Load the ML model"""
        if self.model_path.exists():
            try:
                self.model = TokenScorer.load(str(self.model_path))
                print(f"✓ Strategy scorer loaded model from {self.model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load model: {e}")
                self.model = None
        else:
            print(f"⚠️  Model not found at {self.model_path}")
            self.model = None
    
    def score_signal(self, signal_data: Dict, store_prediction: bool = True) -> Dict:
        """
        Score a signal using ML model + strategy filters
        
        Args:
            signal_data: Signal features
            store_prediction: Whether to store prediction in database
            
        Returns:
            Complete scoring result with decision and reasoning
        """
        result = {
            'mint_key': signal_data.get('mint_key', 'unknown'),
            'signal_at': signal_data.get('signal_at'),
            'timestamp': datetime.utcnow().isoformat(),
            
            # ML Prediction
            'predicted_gain': 0.0,
            'predicted_gain_pct': 0.0,
            'confidence': 0.0,
            'risk_adjusted_score': 0.0,
            
            # Strategy Filters
            'passed_filters': False,
            'filter_score': 0.0,
            'filter_reasons': [],
            
            # Final Decision
            'go_decision': False,
            'decision_reasons': [],
            
            # Trading Parameters
            'position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit_levels': [],
            
            # Source Analysis
            'signal_source': signal_data.get('signal_source', 'unknown'),
            'source_priority': 0
        }
        
        # === STEP 1: Get ML Prediction ===
        if self.model and self.model.trained:
            try:
                # Prepare features
                df = pd.DataFrame([signal_data])
                df = self.feature_extractor.extract_numeric_features(df)
                df = self.feature_extractor.extract_categorical_features(df)
                df = self.feature_extractor.add_derived_features(df)
                token_data = df.iloc[0].to_dict()
                
                # Get prediction
                predicted_gain, confidence = self.model.score_token(token_data)
                
                result['predicted_gain'] = predicted_gain
                result['predicted_gain_pct'] = predicted_gain * 100
                result['confidence'] = confidence
                result['risk_adjusted_score'] = predicted_gain * confidence
                
            except Exception as e:
                print(f"⚠️  Model prediction error: {e}")
                # Use conservative defaults
                result['predicted_gain'] = 0.0
                result['confidence'] = 0.0
        
        # === STEP 2: Apply Strategy Filters ===
        prediction = {
            'confidence': result['confidence'],
            'risk_adjusted_score': result['risk_adjusted_score']
        }
        
        passed, filter_reasons, filter_score = self.strategy.passes_entry_filters(
            signal_data, prediction
        )
        
        result['passed_filters'] = passed
        result['filter_score'] = filter_score
        result['filter_reasons'] = filter_reasons
        
        # === STEP 3: Make GO/SKIP Decision ===
        decision_reasons = []
        
        # Source priority
        source_priority = self.strategy.get_source_priority(
            signal_data.get('signal_source', 'unknown')
        )
        result['source_priority'] = source_priority
        
        if source_priority >= 70:
            decision_reasons.append(f"✅ Good source priority: {source_priority}")
        elif source_priority < 50:
            decision_reasons.append(f"⚠️ Low source priority: {source_priority}")
        
        # Final decision logic
        if not passed:
            result['go_decision'] = False
            decision_reasons.append("❌ Did not pass strategy filters")
        elif result['risk_adjusted_score'] < self.strategy.entry.min_risk_adjusted_score:
            result['go_decision'] = False
            decision_reasons.append(f"❌ Risk-adjusted score too low: {result['risk_adjusted_score']:.2f}")
        elif result['confidence'] < self.strategy.entry.min_confidence:
            result['go_decision'] = False
            decision_reasons.append(f"❌ Confidence too low: {result['confidence']:.2f}")
        else:
            result['go_decision'] = True
            decision_reasons.append("✅ Passed all checks - GO!")
        
        result['decision_reasons'] = decision_reasons
        
        # === STEP 4: Calculate Trading Parameters ===
        if result['go_decision']:
            result['position_size'] = self.strategy.calculate_position_size(
                signal_data, prediction
            )
            result['stop_loss'] = self.strategy.get_stop_loss(signal_data)
            result['take_profit_levels'] = self.strategy.get_take_profit_levels(
                result['predicted_gain']
            )
        else:
            result['position_size'] = 0.0
            result['stop_loss'] = self.strategy.risk.default_sl
            result['take_profit_levels'] = []
        
        # === STEP 5: Store Prediction for Tracking ===
        if store_prediction and signal_data.get('mint_key') and signal_data.get('signal_at'):
            try:
                self.db.update_signal_prediction(
                    mint_key=signal_data['mint_key'],
                    signal_at=signal_data['signal_at'],
                    predicted_gain=result['predicted_gain'],
                    predicted_confidence=result['confidence'],
                    risk_adjusted_score=result['risk_adjusted_score'],
                    go_decision=result['go_decision'],
                    passed_filters=result['passed_filters'],
                    filter_score=result['filter_score'],
                    filter_reasons=json.dumps(result['filter_reasons']),
                    position_size=result['position_size'],
                    stop_loss=result['stop_loss']
                )
            except Exception as e:
                print(f"⚠️  Failed to store prediction: {e}")
        
        return result
    
    def score_batch(self, signals: List[Dict], store_predictions: bool = True) -> List[Dict]:
        """
        Score multiple signals
        
        Args:
            signals: List of signal data
            store_predictions: Whether to store predictions
            
        Returns:
            List of scoring results
        """
        results = []
        
        for signal in signals:
            result = self.score_signal(signal, store_predictions)
            results.append(result)
        
        return results
    
    def filter_signals(self, signals: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter signals into GO and SKIP categories
        
        Args:
            signals: List of signal data
            
        Returns:
            Tuple of (go_signals, skip_signals)
        """
        go_signals = []
        skip_signals = []
        
        for signal in signals:
            result = self.score_signal(signal, store_predictions=True)
            
            if result['go_decision']:
                go_signals.append({**signal, **result})
            else:
                skip_signals.append({**signal, **result})
        
        # Sort GO signals by filter score (best first)
        go_signals.sort(key=lambda x: x.get('filter_score', 0), reverse=True)
        
        return go_signals, skip_signals
    
    def get_top_opportunities(self, limit: int = 10) -> List[Dict]:
        """
        Get top trading opportunities from recent signals
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of top opportunities with scoring
        """
        # Get recent signals
        signals = self.db.get_latest_signals(limit=limit * 3)  # Get more to filter
        
        if not signals:
            return []
        
        # Score and filter
        go_signals, _ = self.filter_signals(signals)
        
        # Return top N
        return go_signals[:limit]
    
    def analyze_prediction_accuracy(self) -> Dict:
        """
        Analyze how well predictions match outcomes
        
        Returns:
            Analysis report with accuracy metrics
        """
        # Get signals with both predictions and outcomes
        df = self.db.get_signals_with_outcomes_for_training()
        
        if df.empty:
            return {
                'status': 'no_data',
                'message': 'No signals with both predictions and outcomes'
            }
        
        # Calculate accuracy metrics
        report = {
            'status': 'success',
            'total_samples': len(df),
            'date_range': {
                'min': df['signal_at'].min(),
                'max': df['signal_at'].max()
            }
        }
        
        # Prediction accuracy
        if 'predicted_gain' in df.columns and 'max_return' in df.columns:
            valid = df[df['predicted_gain'].notna() & df['max_return'].notna()]
            
            if len(valid) > 0:
                errors = valid['max_return'] - valid['predicted_gain']
                
                report['prediction_accuracy'] = {
                    'mean_error': errors.mean(),
                    'mean_abs_error': errors.abs().mean(),
                    'rmse': np.sqrt((errors ** 2).mean()),
                    'correlation': valid['predicted_gain'].corr(valid['max_return'])
                }
        
        # Go decision accuracy
        if 'go_decision' in df.columns and 'is_winner' in df.columns:
            valid = df[df['go_decision'].notna() & df['is_winner'].notna()]
            
            if len(valid) > 0:
                go_df = valid[valid['go_decision'] == 1]
                skip_df = valid[valid['go_decision'] == 0]
                
                report['decision_accuracy'] = {
                    'go_count': len(go_df),
                    'skip_count': len(skip_df),
                    'go_win_rate': go_df['is_winner'].mean() if len(go_df) > 0 else None,
                    'skip_win_rate': skip_df['is_winner'].mean() if len(skip_df) > 0 else None,
                    'go_avg_return': go_df['max_return'].mean() if len(go_df) > 0 else None,
                    'skip_avg_return': skip_df['max_return'].mean() if len(skip_df) > 0 else None
                }
                
                # Calculate improvement from filtering
                if report['decision_accuracy']['go_win_rate'] and report['decision_accuracy']['skip_win_rate']:
                    improvement = (
                        report['decision_accuracy']['go_win_rate'] - 
                        report['decision_accuracy']['skip_win_rate']
                    )
                    report['decision_accuracy']['filter_improvement'] = improvement
                    report['decision_accuracy']['filter_works'] = improvement > 0
        
        # By source analysis
        if 'signal_source' in df.columns:
            source_stats = df.groupby('signal_source').agg({
                'max_return': ['mean', 'count'],
                'is_winner': 'mean'
            }).round(4)
            
            report['by_source'] = source_stats.to_dict()
        
        return report
    
    def get_performance_summary(self) -> Dict:
        """Get overall strategy performance summary"""
        return self.db.get_strategy_performance_stats()


class StrategyTrainer:
    """
    Train the ML model using strategy outcomes
    
    This trainer:
    1. Loads signals with outcomes
    2. Applies feature engineering
    3. Trains model to predict max_return
    4. Evaluates using strategy-based metrics
    """
    
    def __init__(self, 
                 db_path: str = "data/signals.db",
                 model_output_path: str = "outputs/models/token_scorer.pkl",
                 strategy: Optional[TradingStrategy] = None):
        """Initialize trainer"""
        self.db = SignalDatabase(db_path)
        self.model_output_path = Path(model_output_path)
        self.strategy = strategy or get_strategy()
        self.feature_extractor = FeatureExtractor()
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, int, int]:
        """
        Prepare training data from signals with outcomes
        
        Returns:
            Tuple of (dataframe, winner_count, loser_count)
        """
        # Get signals with outcomes
        df = self.db.get_signals_with_outcomes_for_training()
        
        if df.empty:
            print("⚠️  No training data available")
            return df, 0, 0
        
        print(f"\n📊 Training Data Preparation")
        print(f"   Total signals with outcomes: {len(df)}")
        
        # Apply feature engineering
        df = self.feature_extractor.extract_numeric_features(df)
        df = self.feature_extractor.extract_categorical_features(df)
        df = self.feature_extractor.add_derived_features(df)
        
        # Define winners/losers based on strategy threshold
        threshold = self.strategy.training.success_threshold_pct / 100
        df['is_winner'] = (df['max_return'] >= threshold).astype(int)
        
        winners = df['is_winner'].sum()
        losers = len(df) - winners
        
        print(f"   Winners (>= {threshold*100}%): {winners} ({winners/len(df)*100:.1f}%)")
        print(f"   Losers (< {threshold*100}%): {losers} ({losers/len(df)*100:.1f}%)")
        
        return df, winners, losers
    
    def train(self, force: bool = False) -> Dict:
        """
        Train the model on strategy outcomes
        
        Args:
            force: Force training even with few samples
            
        Returns:
            Training results
        """
        print("\n" + "="*80)
        print("STRATEGY-BASED MODEL TRAINING")
        print("="*80)
        
        # Prepare data
        df, winners, losers = self.prepare_training_data()
        
        if df.empty:
            return {'status': 'no_data', 'message': 'No training data available'}
        
        # Check minimum samples
        min_samples = self.strategy.training.min_training_samples
        min_per_class = self.strategy.training.min_samples_per_class
        
        if len(df) < min_samples and not force:
            return {
                'status': 'insufficient_data',
                'message': f'Need {min_samples} samples, have {len(df)}',
                'current_samples': len(df)
            }
        
        if (winners < min_per_class or losers < min_per_class) and not force:
            return {
                'status': 'imbalanced_data',
                'message': f'Need {min_per_class} per class. Winners: {winners}, Losers: {losers}'
            }
        
        # Train model
        print(f"\n🎯 Training on {len(df)} samples...")
        
        model = TokenScorer(model_type=self.strategy.training.model_type)
        
        try:
            metrics = model.train(
                df, 
                target_col='max_return',
                test_size=self.strategy.training.test_size,
                cv_folds=self.strategy.training.cv_folds
            )
            
            # Save model
            self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(self.model_output_path))
            
            # Calculate strategy-specific metrics
            strategy_metrics = self._calculate_strategy_metrics(model, df)
            
            # Save to database
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")
            self.db.save_model_version(
                version=version,
                filepath=str(self.model_output_path),
                accuracy=metrics['test']['r2'],
                win_rate=strategy_metrics.get('predicted_win_rate', 0),
                training_samples=len(df),
                notes=f"Strategy training with {winners} winners, {losers} losers"
            )
            
            print(f"\n✅ Model trained and saved to {self.model_output_path}")
            
            return {
                'status': 'success',
                'version': version,
                'samples': len(df),
                'winners': winners,
                'losers': losers,
                'metrics': metrics,
                'strategy_metrics': strategy_metrics
            }
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
    
    def _calculate_strategy_metrics(self, model: TokenScorer, df: pd.DataFrame) -> Dict:
        """Calculate strategy-specific performance metrics"""
        
        # Get predictions for all samples
        predictions = []
        for _, row in df.iterrows():
            try:
                pred, conf = model.score_token(row.to_dict())
                predictions.append({
                    'predicted_gain': pred,
                    'confidence': conf,
                    'actual_return': row['max_return'],
                    'is_winner': row['is_winner']
                })
            except:
                continue
        
        if not predictions:
            return {}
        
        pred_df = pd.DataFrame(predictions)
        
        # Calculate strategy metrics
        threshold = self.strategy.training.success_threshold_pct / 100
        
        # Predicted winners (based on prediction + confidence)
        pred_df['predicted_go'] = (
            (pred_df['predicted_gain'] >= threshold) & 
            (pred_df['confidence'] >= self.strategy.entry.min_confidence)
        )
        
        go_df = pred_df[pred_df['predicted_go']]
        skip_df = pred_df[~pred_df['predicted_go']]
        
        metrics = {
            'predicted_go_count': len(go_df),
            'predicted_skip_count': len(skip_df),
            'predicted_win_rate': go_df['is_winner'].mean() if len(go_df) > 0 else 0,
            'skip_actual_win_rate': skip_df['is_winner'].mean() if len(skip_df) > 0 else 0,
            'go_avg_return': go_df['actual_return'].mean() if len(go_df) > 0 else 0,
            'skip_avg_return': skip_df['actual_return'].mean() if len(skip_df) > 0 else 0
        }
        
        # Filter improvement
        if len(go_df) > 0 and len(skip_df) > 0:
            metrics['filter_improvement'] = metrics['predicted_win_rate'] - metrics['skip_actual_win_rate']
            metrics['return_improvement'] = metrics['go_avg_return'] - metrics['skip_avg_return']
        
        return metrics


if __name__ == "__main__":
    """Test strategy scorer"""
    print("🧪 Testing Strategy Scorer")
    print("="*80)
    
    # Initialize scorer
    scorer = StrategyScorer()
    
    # Test signal
    test_signal = {
        'mint_key': 'TEST123',
        'signal_at': datetime.utcnow().isoformat(),
        'signal_mc': 100000,
        'signal_liquidity': 30000,
        'signal_volume_1h': 50000,
        'signal_holders': 500,
        'signal_bundled_pct': 5.0,
        'signal_snipers_pct': 15.0,
        'signal_sold_pct': 8.0,
        'signal_source': 'primal'
    }
    
    # Score it
    result = scorer.score_signal(test_signal, store_prediction=False)
    
    print(f"\n📊 Scoring Result:")
    print(f"   Predicted Gain: {result['predicted_gain_pct']:.1f}%")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Risk-Adjusted: {result['risk_adjusted_score']:.2f}")
    print(f"   Passed Filters: {result['passed_filters']}")
    print(f"   Filter Score: {result['filter_score']:.1f}")
    print(f"   GO Decision: {'✅ GO' if result['go_decision'] else '❌ SKIP'}")
    
    print(f"\n📝 Filter Reasons:")
    for reason in result['filter_reasons']:
        print(f"   {reason}")
    
    print(f"\n📝 Decision Reasons:")
    for reason in result['decision_reasons']:
        print(f"   {reason}")
    
    if result['go_decision']:
        print(f"\n💰 Trading Parameters:")
        print(f"   Position Size: {result['position_size']*100:.1f}%")
        print(f"   Stop Loss: {result['stop_loss']*100:.1f}%")
        print(f"   Take Profits:")
        for tp in result['take_profit_levels']:
            print(f"      {tp['label']}: +{tp['gain_pct']}% → Sell {tp['sell_pct']}%")
    
    print("\n" + "="*80)
    print("✓ Test complete!")


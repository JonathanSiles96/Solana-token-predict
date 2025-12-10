"""
Token scoring and ranking system
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.helpers import (
    calculate_liquidity_score,
    calculate_risk_score,
    safe_divide
)


class TokenScorer:
    """
    ML-based token scoring system for ranking and filtering tokens
    """
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Initialize TokenScorer
        
        Args:
            model_type: Type of model ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_names = []
        self.feature_importance = {}
        self.trained = False
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for modeling
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with prepared features
        """
        features = df.copy()
        
        # Core features (required)
        core_features = [
            'signal_mc', 'signal_liquidity', 'signal_volume_1h',
            'signal_holders', 'signal_dev_sol'
        ]
        
        # Ratio features (computed)
        ratio_features = [
            'liq_to_mc_ratio', 'vol_to_liq_ratio', 'mc_volatility'
        ]
        
        # Risk features
        risk_features = [
            'signal_bundled_pct', 'signal_snipers_pct', 'signal_sold_pct',
            'signal_first_20_pct', 'security_encoded', 'risk_score'
        ]
        
        # Additional features
        other_features = [
            'age_minutes', 'signal_fish_count', 'signal_bond', 'signal_made'
        ]
        
        all_features = core_features + ratio_features + risk_features + other_features
        
        # If training, use only available features
        if not self.trained:
            available_features = [f for f in all_features if f in features.columns]
            self.feature_names = available_features
        else:
            # If predicting, ensure all training features exist (fill missing with 0)
            for feat in self.feature_names:
                if feat not in features.columns:
                    features[feat] = 0
            available_features = self.feature_names
        
        # Handle inf values
        for col in available_features:
            if col in features.columns:
                features[col] = features[col].replace([np.inf, -np.inf], np.nan)
        
        # Log transform skewed features (only if not already done)
        log_transform_features = ['signal_mc', 'signal_liquidity', 'signal_volume_1h']
        for col in log_transform_features:
            if col in features.columns and f'{col}_log' in available_features:
                if f'{col}_log' not in features.columns:
                    features[f'{col}_log'] = np.log1p(features[col].fillna(0))
        
        return features[available_features]
    
    def train(self, 
             df: pd.DataFrame,
             target_col: str = 'final_gain',
             test_size: float = 0.2,
             cv_folds: int = 5) -> Dict:
        """
        Train the scoring model
        
        Args:
            df: Training dataframe with features and target
            target_col: Name of target column
            test_size: Test set size
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training metrics
        """
        print(f"\n{'='*80}")
        print(f"TRAINING TOKEN SCORER ({self.model_type.upper()})")
        print(f"{'='*80}\n")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Get target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        y = df[target_col]
        
        # Remove rows with missing target or features
        valid_idx = y.notna() & X.notna().all(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Target: {target_col}")
        print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"Target mean: {y.mean():.4f}, median: {y.median():.4f}\n")
        
        if len(X) < 50:
            raise ValueError("Insufficient training data (need at least 50 samples)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Fill remaining NaNs with median
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())  # Use train median for test
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Metrics
        metrics = {
            'train': {
                'mse': mean_squared_error(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'mae': mean_absolute_error(y_train, y_train_pred),
                'r2': r2_score(y_train, y_train_pred)
            },
            'test': {
                'mse': mean_squared_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'mae': mean_absolute_error(y_test, y_test_pred),
                'r2': r2_score(y_test, y_test_pred)
            }
        }
        
        # Cross-validation
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=cv_folds, scoring='r2', n_jobs=-1
        )
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance_df
            metrics['feature_importance'] = importance_df
        
        # Print results
        print(f"\n{'='*80}")
        print("TRAINING RESULTS")
        print(f"{'='*80}")
        print(f"\nTrain Set:")
        print(f"  RMSE: {metrics['train']['rmse']:.4f}")
        print(f"  MAE:  {metrics['train']['mae']:.4f}")
        print(f"  R²:   {metrics['train']['r2']:.4f}")
        
        print(f"\nTest Set:")
        print(f"  RMSE: {metrics['test']['rmse']:.4f}")
        print(f"  MAE:  {metrics['test']['mae']:.4f}")
        print(f"  R²:   {metrics['test']['r2']:.4f}")
        
        print(f"\nCross-Validation R²: {metrics['cv_r2_mean']:.4f} (+/- {metrics['cv_r2_std']:.4f})")
        
        if 'feature_importance' in metrics:
            print(f"\nTop 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False))
        
        print(f"\n{'='*80}\n")
        
        self.trained = True
        
        return metrics
    
    def score_token(self, token_data: Dict) -> Tuple[float, float]:
        """
        Score a single token
        
        Args:
            token_data: Dictionary with token features
            
        Returns:
            Tuple of (predicted_gain, confidence)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to dataframe
        df = pd.DataFrame([token_data])
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Fill missing with 0 (conservative)
        X = X.fillna(0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        # Calculate confidence based on feature completeness
        completeness = (1 - X.isna().sum().sum() / len(X.columns))
        
        # Adjust confidence based on prediction uncertainty
        # For tree-based models, use prediction variance across stages (if available)
        if hasattr(self.model, 'estimators_') and self.model_type == 'random_forest':
            # Random Forest has estimators with predict method
            tree_predictions = np.array([
                tree.predict(X_scaled)[0] 
                for tree in self.model.estimators_
            ])
            prediction_std = tree_predictions.std()
            uncertainty = min(prediction_std / (abs(prediction) + 0.1), 1.0)
            confidence = completeness * (1 - uncertainty)
        else:
            # For Gradient Boosting or other models, use feature completeness only
            confidence = completeness * 0.8
        
        return prediction, confidence
    
    def rank_tokens(self, df: pd.DataFrame, top_n: int = None) -> pd.DataFrame:
        """
        Rank multiple tokens by predicted gain
        
        Args:
            df: DataFrame with token features
            top_n: Return only top N tokens (None for all)
            
        Returns:
            DataFrame with tokens ranked by score
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Fill missing
        X_filled = X.fillna(0)
        
        # Scale
        X_scaled = self.scaler.transform(X_filled)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        # Calculate confidence
        completeness = (1 - X.isna().sum(axis=1) / len(X.columns))
        
        # Create result dataframe
        result_df = df.copy()
        result_df['predicted_gain'] = predictions
        result_df['confidence'] = completeness * 0.8
        result_df['score'] = predictions * completeness  # Risk-adjusted score
        
        # Rank
        result_df = result_df.sort_values('score', ascending=False)
        
        if top_n:
            result_df = result_df.head(top_n)
        
        return result_df
    
    def recommend_parameters(self, 
                            token_data: Dict,
                            base_sl: float = -0.35,
                            min_position_pct: float = 0.05,
                            max_position_pct: float = 0.10) -> Dict:
        """
        Recommend trading parameters based on token score and strategy rules
        
        Uses your strategy configuration from src/strategy/config.py:
        - Entry: Confidence ≥ 0.8, Risk-adj score ≥ 1.0, Volume > 10k, Holders > 50
        - Red flags: Bundled > 60%, Sold > 60%, Snipers > 50%
        - Position sizing: 5-10% based on confidence
        - Stop loss: -35% default, -45% for risky tokens
        
        Args:
            token_data: Token features
            base_sl: Base stop-loss level (default from strategy)
            min_position_pct: Minimum position size
            max_position_pct: Maximum position size
            
        Returns:
            Dictionary with recommended parameters
        """
        # Load strategy configuration
        try:
            from src.strategy.config import get_strategy
            strategy = get_strategy()
        except:
            strategy = None
        
        predicted_gain, confidence = self.score_token(token_data)
        
        # Calculate risk-adjusted score
        risk_adjusted_score = predicted_gain * confidence
        
        # Calculate risk_score
        if 'risk_score' not in token_data or pd.isna(token_data.get('risk_score')):
            from src.utils.helpers import calculate_risk_score
            risk = calculate_risk_score(
                token_data.get('signal_bundled_pct', 0),
                token_data.get('signal_snipers_pct', 0),
                token_data.get('signal_sold_pct', 0),
                token_data.get('signal_security', None)
            )
        else:
            risk = token_data.get('risk_score', 0.5)
        
        # Get key metrics
        bundled_pct = token_data.get('signal_bundled_pct') or 0
        sold_pct = token_data.get('signal_sold_pct') or 0
        snipers_pct = token_data.get('signal_snipers_pct') or 0
        liquidity = token_data.get('signal_liquidity') or 0
        mc = token_data.get('signal_mc') or 0
        volume_1h = token_data.get('signal_volume_1h') or 0
        holders = token_data.get('signal_holders') or 0
        liq_ratio = (liquidity / mc) if mc > 0 else 0
        
        # Use strategy thresholds if available, else defaults
        if strategy:
            # Entry thresholds from your strategy
            min_confidence = strategy.entry.min_confidence  # 0.80
            min_risk_adj = strategy.entry.min_risk_adjusted_score  # 1.0
            min_volume = strategy.entry.min_volume_1h  # 10000
            min_holders = strategy.entry.min_holders  # 50
            min_liquidity = strategy.entry.min_liquidity  # 5000
            min_liq_ratio = strategy.entry.min_liq_to_mc_ratio  # 0.15
            
            # Red flags from your strategy
            max_bundled = strategy.entry.max_bundled_pct  # 60
            max_sold = strategy.entry.max_sold_pct  # 60
            max_snipers = strategy.entry.max_snipers_pct  # 50
            
            # Warning thresholds
            warn_bundled = strategy.entry.warn_bundled_pct  # 15
            warn_sold = strategy.entry.warn_sold_pct  # 25
            warn_snipers = strategy.entry.warn_snipers_pct  # 35
            
            # Position sizing
            min_pos = strategy.risk.min_position_pct
            max_pos = strategy.risk.max_position_pct
            high_risk_pos = strategy.risk.high_risk_position_pct
            
            # Stop loss levels
            default_sl = strategy.risk.default_sl
            tight_sl = strategy.risk.tight_sl
            loose_sl = strategy.risk.loose_sl
        else:
            # Fallback defaults
            min_confidence = 0.80
            min_risk_adj = 1.0
            min_volume = 10000
            min_holders = 50
            min_liquidity = 5000
            min_liq_ratio = 0.15
            max_bundled = 60
            max_sold = 60
            max_snipers = 50
            warn_bundled = 15
            warn_sold = 25
            warn_snipers = 35
            min_pos = 0.05
            max_pos = 0.10
            high_risk_pos = 0.05
            default_sl = -0.35
            tight_sl = -0.45
            loose_sl = -0.25
        
        # === DETERMINE IF HIGH RISK ===
        is_high_risk = (
            bundled_pct > warn_bundled or
            sold_pct > warn_sold or
            snipers_pct > warn_snipers
        )
        
        # === STOP LOSS ===
        if is_high_risk:
            sl_level = tight_sl
        elif risk < 0.3 and holders > 200:
            sl_level = loose_sl
        else:
            sl_level = default_sl
        
        # === TAKE PROFIT LEVELS ===
        if predicted_gain > 2.0:
            tp_levels = [
                {'gain_pct': 50, 'sell_amount_pct': 15},
                {'gain_pct': 100, 'sell_amount_pct': 25},
                {'gain_pct': 200, 'sell_amount_pct': 35},
                {'gain_pct': 500, 'sell_amount_pct': 100}
            ]
        elif predicted_gain > 1.0:
            tp_levels = [
                {'gain_pct': 30, 'sell_amount_pct': 20},
                {'gain_pct': 70, 'sell_amount_pct': 25},
                {'gain_pct': 150, 'sell_amount_pct': 35},
                {'gain_pct': 300, 'sell_amount_pct': 100}
            ]
        elif predicted_gain > 0.5:
            tp_levels = [
                {'gain_pct': 20, 'sell_amount_pct': 25},
                {'gain_pct': 50, 'sell_amount_pct': 30},
                {'gain_pct': 100, 'sell_amount_pct': 35},
                {'gain_pct': 200, 'sell_amount_pct': 100}
            ]
        else:
            tp_levels = [
                {'gain_pct': 10, 'sell_amount_pct': 30},
                {'gain_pct': 30, 'sell_amount_pct': 35},
                {'gain_pct': 50, 'sell_amount_pct': 35},
                {'gain_pct': 100, 'sell_amount_pct': 100}
            ]
        
        # === POSITION SIZING ===
        if is_high_risk:
            position_size = high_risk_pos
        else:
            risk_factor = 1.0 - (risk * 0.5)
            position_size = min_pos + (confidence * risk_factor * (max_pos - min_pos))
            position_size = max(min_pos, min(max_pos, position_size))
        
        # === GO/SKIP DECISION (Your Strategy Rules) ===
        
        # Check for RED FLAGS (auto-SKIP)
        has_red_flags = (
            bundled_pct > max_bundled or
            sold_pct > max_sold or
            snipers_pct > max_snipers
        )
        
        # Check minimum requirements
        has_minimum_data = mc > 0 and liquidity > 0 and holders > 0
        
        passes_entry_criteria = (
            confidence >= min_confidence and
            risk_adjusted_score >= min_risk_adj and
            volume_1h >= min_volume and
            holders >= min_holders and
            liquidity >= min_liquidity
        )
        
        # Decision logic
        if has_red_flags:
            go_decision = False
            decision_reason = f"🚨 Red flag: bundled={bundled_pct:.0f}%, sold={sold_pct:.0f}%, snipers={snipers_pct:.0f}%"
        elif not has_minimum_data:
            go_decision = False
            decision_reason = "❌ Missing required data (MC, liquidity, or holders)"
        elif not passes_entry_criteria:
            go_decision = False
            reasons = []
            if confidence < min_confidence:
                reasons.append(f"confidence {confidence:.2f} < {min_confidence}")
            if risk_adjusted_score < min_risk_adj:
                reasons.append(f"risk-adj {risk_adjusted_score:.2f} < {min_risk_adj}")
            if volume_1h < min_volume:
                reasons.append(f"volume ${volume_1h:,.0f} < ${min_volume:,.0f}")
            if holders < min_holders:
                reasons.append(f"holders {holders} < {min_holders}")
            decision_reason = "❌ " + ", ".join(reasons)
        else:
            go_decision = True
            decision_reason = f"✅ Passed all filters: conf={confidence:.2f}, risk-adj={risk_adjusted_score:.2f}"
        
        # Add warnings to notes
        warnings = []
        if bundled_pct > warn_bundled:
            warnings.append(f"bundled:{bundled_pct:.0f}%")
        if sold_pct > warn_sold:
            warnings.append(f"sold:{sold_pct:.0f}%")
        if snipers_pct > warn_snipers:
            warnings.append(f"snipers:{snipers_pct:.0f}%")
        if liq_ratio < min_liq_ratio:
            warnings.append(f"low_liq_ratio:{liq_ratio:.1%}")
        
        notes = decision_reason
        if warnings:
            notes += f" | ⚠️ {', '.join(warnings)}"
        
        return {
            'predicted_gain': predicted_gain,
            'confidence': confidence,
            'go_decision': go_decision,
            'recommended_tp_levels': tp_levels,
            'recommended_sl': sl_level,
            'position_size_factor': position_size,
            'notes': notes
        }
    
    def save(self, filepath: str):
        """Save model to disk"""
        if not self.trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TokenScorer':
        """Load model from disk"""
        model_data = joblib.load(filepath)
        
        scorer = cls(model_type=model_data['model_type'])
        scorer.model = model_data['model']
        scorer.scaler = model_data['scaler']
        scorer.feature_names = model_data['feature_names']
        scorer.feature_importance = model_data['feature_importance']
        scorer.trained = True
        
        print(f"Model loaded from {filepath}")
        
        return scorer


if __name__ == "__main__":
    # Test with sample data
    print("Testing TokenScorer...")
    
    np.random.seed(42)
    n_samples = 500
    
    # Create synthetic data
    sample_df = pd.DataFrame({
        'signal_mc': np.random.lognormal(11, 1, n_samples),
        'signal_liquidity': np.random.lognormal(10, 0.8, n_samples),
        'signal_volume_1h': np.random.lognormal(10, 1.2, n_samples),
        'signal_holders': np.random.poisson(500, n_samples),
        'signal_dev_sol': np.random.uniform(0, 50, n_samples),
        'liq_to_mc_ratio': np.random.uniform(0.1, 0.5, n_samples),
        'vol_to_liq_ratio': np.random.uniform(0.5, 3.0, n_samples),
        'mc_volatility': np.random.uniform(0, 1, n_samples),
        'risk_score': np.random.uniform(0, 1, n_samples),
        'security_encoded': np.random.choice([0, 1, 2], n_samples),
        'signal_bundled_pct': np.random.uniform(0, 30, n_samples),
        'signal_snipers_pct': np.random.uniform(0, 40, n_samples),
        'signal_sold_pct': np.random.uniform(0, 20, n_samples),
        'signal_first_20_pct': np.random.uniform(10, 80, n_samples),
        'age_minutes': np.random.uniform(1, 120, n_samples),
        'final_gain': np.random.lognormal(-0.5, 1, n_samples)
    })
    
    # Train model
    scorer = TokenScorer(model_type='gradient_boosting')
    metrics = scorer.train(sample_df)
    
    # Test scoring
    test_token = sample_df.iloc[0].to_dict()
    gain, conf = scorer.score_token(test_token)
    print(f"\nTest token score: {gain:.4f} (confidence: {conf:.2f})")
    
    # Test recommendations
    recommendations = scorer.recommend_parameters(test_token)
    print(f"\nRecommendations: {recommendations}")
    
    print("\nTest complete!")


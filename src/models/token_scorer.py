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
        
        # === MODEL QUALITY VALIDATION ===
        test_r2 = metrics['test']['r2']
        cv_r2 = metrics['cv_r2_mean']
        
        print(f"\n{'='*80}")
        print("MODEL QUALITY ASSESSMENT")
        print(f"{'='*80}")
        
        self.model_quality = "unknown"
        
        if test_r2 < 0:
            print(f"\n🚨 CRITICAL: Test R² is NEGATIVE ({test_r2:.4f})")
            print("   Model is WORSE than predicting the mean!")
            print("   This model should NOT be used for trading.")
            print("   Possible causes:")
            print("   - Insufficient training data")
            print("   - Features don't predict the target")
            print("   - Severe overfitting")
            self.model_quality = "unusable"
            metrics['model_quality'] = "unusable"
        elif test_r2 < 0.1:
            print(f"\n⚠️  WARNING: Test R² is very low ({test_r2:.4f})")
            print("   Model has weak predictive power.")
            print("   Use with extreme caution.")
            self.model_quality = "poor"
            metrics['model_quality'] = "poor"
        elif test_r2 < 0.3:
            print(f"\n⚠️  CAUTION: Test R² is moderate ({test_r2:.4f})")
            print("   Model has limited predictive power.")
            self.model_quality = "fair"
            metrics['model_quality'] = "fair"
        elif test_r2 < 0.5:
            print(f"\n✅ GOOD: Test R² is acceptable ({test_r2:.4f})")
            self.model_quality = "good"
            metrics['model_quality'] = "good"
        else:
            print(f"\n🎯 EXCELLENT: Test R² is strong ({test_r2:.4f})")
            self.model_quality = "excellent"
            metrics['model_quality'] = "excellent"
        
        if cv_r2 < 0:
            print(f"\n🚨 Cross-validation R² is NEGATIVE ({cv_r2:.4f})")
            print("   Model does not generalize to new data!")
            if self.model_quality != "unusable":
                self.model_quality = "unusable"
                metrics['model_quality'] = "unusable"
        
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
        raw_prediction = self.model.predict(X_scaled)[0]
        
        # SANITY CHECK: Cap predictions at realistic values
        # No token realistically gains more than 10x (1000%) or loses more than 100%
        MAX_GAIN = 10.0   # 1000% max gain
        MIN_GAIN = -0.99  # -99% max loss
        prediction = max(MIN_GAIN, min(MAX_GAIN, raw_prediction))
        
        # Log warning if prediction was clamped (indicates model issues)
        if raw_prediction != prediction:
            import warnings
            warnings.warn(f"Prediction clamped: {raw_prediction:.2f} -> {prediction:.2f} (model may need retraining)")
        
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
        Recommend trading parameters based on token score and ULTRA-STRICT strategy rules
        
        ULTRA-STRICT for 80%+ Win Rate:
        - Entry: Confidence ≥ 0.8, Risk-adj score ≥ 100, Volume > 20k, Holders > 150
        - Red flags: Bundled > 15%, Sold > 15%, Snipers > 25%
        - Security: Only ✅ tokens allowed
        - Source: Only primal, whale, solana_tracker
        - Position sizing: 5-10% based on confidence
        - Stop loss: -35% default, -45% for any warnings
        
        Args:
            token_data: Token features
            base_sl: Base stop-loss level
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
        security = str(token_data.get('signal_security', '') or '')
        first_20_pct = token_data.get('signal_first_20_pct') or 0
        source = str(token_data.get('signal_source', 'unknown') or 'unknown').lower()
        token_age = token_data.get('age_minutes') or 0
        
        # Use strategy thresholds - ULTRA STRICT
        if strategy:
            # Entry thresholds - STRICTER
            min_confidence = strategy.entry.min_confidence  # 0.80
            min_risk_adj = strategy.entry.min_risk_adjusted_score  # 100.0 (was 1.0)
            min_volume = strategy.entry.min_volume_1h  # 20000 (was 10000)
            min_holders = strategy.entry.min_holders  # 150 (was 50)
            min_liquidity = strategy.entry.min_liquidity  # 15000 (was 5000)
            min_liq_ratio = strategy.entry.min_liq_to_mc_ratio  # 0.20
            min_mc = strategy.entry.min_mc  # 30000
            
            # Red flags - MUCH STRICTER
            max_bundled = strategy.entry.max_bundled_pct  # 15 (was 60)
            max_sold = strategy.entry.max_sold_pct  # 15 (was 60)
            max_snipers = strategy.entry.max_snipers_pct  # 25 (was 50)
            max_first_20 = strategy.entry.max_first_20_pct  # 50
            
            # Warning thresholds - STRICTER
            warn_bundled = strategy.entry.warn_bundled_pct  # 5 (was 15)
            warn_sold = strategy.entry.warn_sold_pct  # 8 (was 25)
            warn_snipers = strategy.entry.warn_snipers_pct  # 15 (was 35)
            
            # Security & source
            require_green = strategy.entry.require_green_security
            allowed_security = strategy.entry.allowed_security_statuses
            min_source_priority = strategy.entry.min_source_priority  # 70
            
            # Token age
            min_age = strategy.entry.min_token_age  # 1
            max_age = strategy.entry.max_token_age  # 60
            
            # Position sizing
            min_pos = strategy.risk.min_position_pct
            max_pos = strategy.risk.max_position_pct
            high_risk_pos = strategy.risk.high_risk_position_pct
            
            # Stop loss levels
            default_sl = strategy.risk.default_sl
            tight_sl = strategy.risk.tight_sl
            loose_sl = strategy.risk.loose_sl
            
            # Get source priority
            source_priority = strategy.get_source_priority(source)
        else:
            # Fallback defaults - CORRECTED Dec 22 to allow more opportunities
            min_confidence = 0.4
            min_risk_adj = 0.2  # predicted_gain * confidence must be >= 0.2
            min_volume = 5000
            min_holders = 50
            min_liquidity = 10000
            min_liq_ratio = 0.15
            min_mc = 15000
            max_bundled = 95    # LOOSENED - high bundled tokens still win!
            max_sold = 60
            max_snipers = 60
            max_first_20 = 60
            warn_bundled = 50
            warn_sold = 30
            warn_snipers = 40
            require_green = False  # Allow warning tokens too
            allowed_security = ["✅", "white_check_mark", "⚠️", "warning", "🚨"]
            min_source_priority = 30
            min_age = 0
            max_age = 180
            min_pos = 0.05
            max_pos = 0.10
            high_risk_pos = 0.05
            default_sl = -0.35
            tight_sl = -0.45
            loose_sl = -0.25
            source_priority = 60  # Default for unknown sources
        
        # === DETERMINE IF HIGH RISK ===
        is_high_risk = (
            bundled_pct > warn_bundled or
            sold_pct > warn_sold or
            snipers_pct > warn_snipers
        )
        
        # === STOP LOSS - Stricter ===
        if is_high_risk:
            sl_level = tight_sl
        elif risk < 0.2 and holders > 250 and bundled_pct < 3 and snipers_pct < 10:
            sl_level = loose_sl
        else:
            sl_level = default_sl
        
        # === GO/SKIP DECISION - SIMPLE DATA-PROVEN STRATEGY Dec 23 ===
        # 
        # DATA ANALYSIS (858 signals from Dec 21-22):
        # - whale: 100% WR (n=62) - BEST SOURCE!
        # - tg_early_trending: 100% WR (n=135) - BEST SOURCE!
        # - solana_tracker: 66% WR (n=220)
        # - primal: 69% WR (n=277)
        #
        # FILTER COMBINATIONS:
        # - ✅ security: 98% WR (n=56)
        # - Liq>=25k: 89% WR (n=182)
        # - Vol>=30k + Hold>=250: 89% WR (n=192)
        # - Hold>=500: 96% WR (n=47)
        # - Hold>=300: 86% WR (n=208)
        # - Vol>=10k: 82% WR (n=481)
        
        go_decision = False
        decision_reason = ""
        win_tier = "none"
        
        # Check security
        security_str = str(security) if security else ""
        is_green = "✅" in security_str or "check" in security_str.lower()
        is_alert = "🚨" in security_str
        
        # ============================================================
        # DATA-DRIVEN TIER SYSTEM - Dec 24/25, 2025 Analysis
        # ============================================================
        # Dec 25 data (675 GO signals):
        # - primal: 82% hit +15%, 70% hit +30% (BEST SOURCE!)
        # - solana_tracker: 79% hit +15%, 68% hit +30%
        # - whale: 58% hit +15%, 58% hit +30%
        # - tg_early_trending: 52% hit +15%, 52% hit +30%
        #
        # Overall: 72% hit +10%, 62.5% hit +30%, 39% hit +100%
        
        # === TIER 0: BEST SOURCES (primal/solana_tracker 80%+ at +15%) ===
        if source in ['primal', 'solana_tracker']:
            go_decision = True
            win_tier = "TIER0"
            decision_reason = f"🏆 TIER0: {source} = 80%+ hit rate"
        
        # === TIER 1: WHALE/TG_EARLY with good metrics ===
        elif source in ['whale', 'tg_early_trending'] and (holders >= 150 or volume_1h >= 15000):
            go_decision = True
            win_tier = "TIER1"
            decision_reason = f"✅ TIER1: {source} + metrics = 58% hit rate"
        
        # === TIER 2: HIGH HOLDERS (historically reliable) ===
        elif holders >= 400:
            go_decision = True
            win_tier = "TIER2"
            decision_reason = f"✅ TIER2: Hold400+ = high confidence"
        
        # === TIER 3: HIGH LIQUIDITY/MC ===
        elif liquidity >= 25000 or mc >= 75000:
            go_decision = True
            win_tier = "TIER3"
            decision_reason = f"✅ TIER3: Liq>=25k or MC>=75k"
        
        # === TIER 4: GOOD VOLUME + HOLDERS ===
        elif volume_1h >= 20000 and holders >= 150:
            go_decision = True
            win_tier = "TIER4"
            decision_reason = f"✅ TIER4: Vol20k+Hold150"
        
        # === TIER 5: WHALE/TG_EARLY without strict metrics ===
        elif source in ['whale', 'tg_early_trending']:
            go_decision = True
            win_tier = "TIER5"
            decision_reason = f"✅ TIER5: {source} base"
        
        # === TIER 6: DECENT SINGLE METRIC ===
        elif volume_1h >= 15000 or holders >= 250 or liquidity >= 20000:
            go_decision = True
            win_tier = "TIER6"
            decision_reason = f"✅ TIER6: Decent metrics"
        
        # === TIER 7: MINIMUM VIABLE ===
        elif volume_1h >= 8000 or holders >= 150 or liquidity >= 15000:
            go_decision = True
            win_tier = "TIER7"
            decision_reason = f"✅ TIER7: Minimum viable"
        
        # === SKIP: Weak signals ===
        else:
            go_decision = False
            win_tier = "none"
            reasons = []
            if volume_1h < 8000:
                reasons.append(f"vol {volume_1h/1000:.1f}k<8k")
            if holders < 150:
                reasons.append(f"hold {holders}<150")
            if liquidity < 15000:
                reasons.append(f"liq {liquidity/1000:.1f}k<15k")
            decision_reason = f"⚠️ SKIP: {', '.join(reasons) if reasons else 'weak metrics'}"
        
        # === TAKE PROFIT & STOP LOSS - DATA-DRIVEN Dec 24 ===
        # 
        # ANALYSIS OF 683 SIGNALS WITH MAX_RETURN DATA:
        # - 92.7% reach +8% gain
        # - 87.0% reach +15% gain  → TP1 target
        # - 78.5% reach +30% gain  → TP2 target
        # - 54.6% reach +70% gain  → TP3 target
        # - 31.2% reach +200% gain → TP4 for runners
        # - Median max_return: 1.91x (91% gain)
        # - Mean max_return: 3.38x (238% gain)
        #
        # STOP LOSS: Tight to cut losers early
        # TRAILING: Activate after TP1, trail 12% behind peak
        # TOP-UP: Add 50% position if dips 5-8% with good fundamentals
        
        # Initialize defaults
        trailing_activation = 0.15  # Activate after +15% gain
        trailing_distance = 0.12    # Trail 12% behind peak
        post_tp1_sl = 0.0           # Move SL to break-even after TP1
        post_tp2_sl = 0.15          # Move SL to +15% after TP2
        
        # ============================================================
        # DATA-DRIVEN TP LADDER (Dec 24, 2025 - 675 GO signals)
        # ============================================================
        # Hit rates: +10%=72%, +25%=65%, +50%=57%, +100%=39%
        # By source: primal/solana_tracker best (82%@15%), whale/tg (58%@15%)
        
        if win_tier in ['TIER0', 'TIER1', 'TIER2']:
            # High confidence tiers - let winners run more
            tp_levels = [
                {'gain_pct': 12, 'sell_amount_pct': 30},   # TP1: 30% at +12% (~70% hit)
                {'gain_pct': 30, 'sell_amount_pct': 30},   # TP2: 30% at +30% (~62% hit)
                {'gain_pct': 60, 'sell_amount_pct': 25},   # TP3: 25% at +60% (~48% hit)
                {'gain_pct': 120, 'sell_amount_pct': 15}   # TP4: 15% runner (~31% hit)
            ]
            sl_level = -0.18  # -18% SL (NEED MIN_RETURN DATA TO OPTIMIZE)
            trailing_activation = 0.15
            trailing_distance = 0.12
            post_tp1_sl = 0.0    # Break-even after TP1
            post_tp2_sl = 0.12   # +12% after TP2
            post_tp3_sl = 0.30   # +30% after TP3
            
        elif win_tier in ['TIER3', 'TIER4']:
            # Good metrics - standard ladder
            tp_levels = [
                {'gain_pct': 10, 'sell_amount_pct': 35},   # TP1: 35% at +10% (72% hit)
                {'gain_pct': 25, 'sell_amount_pct': 30},   # TP2: 30% at +25% (65% hit)
                {'gain_pct': 50, 'sell_amount_pct': 25},   # TP3: 25% at +50% (57% hit)
                {'gain_pct': 100, 'sell_amount_pct': 10}   # TP4: 10% runner (39% hit)
            ]
            sl_level = -0.15  # -15% SL
            trailing_activation = 0.12
            trailing_distance = 0.10
            post_tp1_sl = 0.0    # Break-even after TP1
            post_tp2_sl = 0.10   # +10% after TP2
            post_tp3_sl = 0.25   # +25% after TP3
            
        elif win_tier in ['TIER5', 'TIER6']:
            # Lower confidence - take profits earlier
            tp_levels = [
                {'gain_pct': 10, 'sell_amount_pct': 40},   # TP1: 40% at +10%
                {'gain_pct': 22, 'sell_amount_pct': 35},   # TP2: 35% at +22%
                {'gain_pct': 45, 'sell_amount_pct': 25}    # TP3: 25% at +45%
            ]
            sl_level = -0.12  # -12% SL - tighter
            trailing_activation = 0.10
            trailing_distance = 0.08
            post_tp1_sl = 0.0
            post_tp2_sl = 0.08
            post_tp3_sl = 0.20
            
        elif win_tier == 'TIER7':
            # Baseline - be conservative
            tp_levels = [
                {'gain_pct': 8, 'sell_amount_pct': 45},    # TP1: 45% at +8%
                {'gain_pct': 18, 'sell_amount_pct': 35},   # TP2: 35% at +18%
                {'gain_pct': 35, 'sell_amount_pct': 20}    # TP3: 20% at +35%
            ]
            sl_level = -0.10  # -10% SL
            trailing_activation = 0.08
            trailing_distance = 0.06
            post_tp1_sl = 0.0
            post_tp2_sl = 0.05
            post_tp3_sl = 0.15
            
        else:
            # Skip tier - if somehow traded, be very tight
            tp_levels = [
                {'gain_pct': 6, 'sell_amount_pct': 50},
                {'gain_pct': 12, 'sell_amount_pct': 35},
                {'gain_pct': 25, 'sell_amount_pct': 15}
            ]
            sl_level = -0.08  # -8% SL
            trailing_activation = 0.06
            trailing_distance = 0.05
            post_tp1_sl = 0.0
            post_tp2_sl = 0.03
            post_tp3_sl = 0.10
        
        # === POSITION SIZING - Dec 24 Update ===
        # Position sizing based on tier WR (higher WR = bigger position)
        if win_tier == 'TIER0':
            position_size = max_pos  # 10% for 100% WR signals
        elif win_tier == 'TIER1':
            position_size = max_pos * 0.90  # 9% for 94% WR
        elif win_tier == 'TIER2':
            position_size = max_pos * 0.85  # 8.5% for 91% WR
        elif win_tier == 'TIER3':
            position_size = max_pos * 0.80  # 8% for 90% WR
        elif win_tier == 'TIER4':
            position_size = max_pos * 0.70  # 7% for 86% WR
        elif win_tier == 'TIER5':
            position_size = max_pos * 0.60  # 6% for 82% WR
        elif win_tier == 'TIER6':
            position_size = max_pos * 0.55  # 5.5% for 82% WR
        elif win_tier == 'TIER7':
            position_size = max_pos * 0.50  # 5% for 78% WR
        else:
            position_size = min_pos  # minimum for skips
        
        # Add warnings to notes
        warnings = []
        if bundled_pct > warn_bundled:
            warnings.append(f"bundled:{bundled_pct:.0f}%")
        if sold_pct > warn_sold:
            warnings.append(f"sold:{sold_pct:.0f}%")
        if snipers_pct > warn_snipers:
            warnings.append(f"snipers:{snipers_pct:.0f}%")
        
        notes = decision_reason
        if warnings:
            notes += f" | ⚠️ {', '.join(warnings)}"
        
        # Include trailing stop info
        trailing_info = {
            'enabled': go_decision,  # Enable trailing for all GO decisions
            'activation_pct': trailing_activation,
            'distance_pct': trailing_distance,
            'post_tp1_sl': post_tp1_sl,
            'post_tp2_sl': post_tp2_sl
        }
        
        # Top-up configuration
        topup_info = {
            'enabled': go_decision and win_tier in ['TIER0', 'TIER1', 'TIER2', 'TIER3'],
            'trigger_dip_pct': -0.05,   # Top up if dips 5%
            'max_dip_pct': -0.12,       # Don't top up if dipped > 12%
            'add_size_pct': 0.50,       # Add 50% of original position
            'topup_sl_pct': -0.10,      # Tight -10% SL on top-up
            'max_topups': 1
        }
        
        return {
            'predicted_gain': predicted_gain,
            'confidence': confidence,
            'go_decision': go_decision,
            'win_tier': win_tier,
            'recommended_tp_levels': tp_levels,
            'recommended_sl': sl_level,
            'trailing_stop': trailing_info,
            'topup': topup_info,
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


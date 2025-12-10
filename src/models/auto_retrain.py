"""
Automatic Model Retraining Module
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import threading
import time

from src.api.database import SignalDatabase
from src.models.token_scorer import TokenScorer
from src.data_processing.data_loader import FeatureExtractor


class AutoRetrainer:
    """
    Automatic model retraining system
    
    Retrains model when enough new outcome data is available
    """
    
    def __init__(self, 
                 retrain_threshold: int = 50,
                 min_total_samples: int = 100,
                 model_type: str = 'gradient_boosting'):
        """
        Initialize AutoRetrainer
        
        Args:
            retrain_threshold: Number of new outcomes to trigger retraining
            min_total_samples: Minimum total samples needed to retrain
            model_type: Type of model to train
        """
        self.retrain_threshold = retrain_threshold
        self.min_total_samples = min_total_samples
        self.model_type = model_type
        self.db = SignalDatabase()
        self.feature_extractor = FeatureExtractor()
        self.retraining_active = False
        self.last_retrain_at = None
        
    def check_and_retrain(self) -> Dict:
        """
        Check if retraining is needed and retrain if necessary
        
        Returns:
            Dictionary with retraining status and results
        """
        # Check if already retraining
        if self.retraining_active:
            return {
                'status': 'already_running',
                'message': 'Retraining already in progress'
            }
        
        # Get new outcomes count
        new_outcomes = self.db.get_new_outcomes_count()
        
        # Check if we have enough new data
        if new_outcomes < self.retrain_threshold:
            return {
                'status': 'not_needed',
                'message': f'Not enough new outcomes ({new_outcomes}/{self.retrain_threshold})',
                'new_outcomes': new_outcomes,
                'threshold': self.retrain_threshold
            }
        
        # Get all training data
        training_df = self.db.get_training_data()
        
        if len(training_df) < self.min_total_samples:
            return {
                'status': 'insufficient_data',
                'message': f'Not enough total samples ({len(training_df)}/{self.min_total_samples})',
                'total_samples': len(training_df),
                'min_required': self.min_total_samples
            }
        
        # Start retraining in background
        print(f"\n🔄 Auto-retraining triggered: {new_outcomes} new outcomes, {len(training_df)} total samples")
        
        try:
            self.retraining_active = True
            result = self._retrain_model(training_df)
            self.last_retrain_at = datetime.utcnow()
            return result
        finally:
            self.retraining_active = False
    
    def _retrain_model(self, df: pd.DataFrame) -> Dict:
        """
        Retrain the model with new data
        
        Args:
            df: Training dataframe
            
        Returns:
            Dictionary with retraining results
        """
        try:
            print(f"🔧 Preparing training data...")
            
            # Add derived features
            df = self.feature_extractor.extract_numeric_features(df)
            df = self.feature_extractor.extract_categorical_features(df)
            df = self.feature_extractor.add_derived_features(df)
            
            # Remove rows with missing target
            df_clean = df[df['final_gain'].notna()].copy()
            
            if len(df_clean) < self.min_total_samples:
                return {
                    'status': 'failed',
                    'message': f'Insufficient valid samples after cleaning ({len(df_clean)})'
                }
            
            print(f"✓ Training data prepared: {len(df_clean)} valid samples")
            
            # Initialize new scorer
            scorer = TokenScorer(model_type=self.model_type)
            
            # Train model
            print(f"🚀 Training {self.model_type} model...")
            metrics = scorer.train(df_clean, target_col='final_gain')
            
            # Check if model is good enough
            test_r2 = metrics['test']['r2']
            cv_r2_mean = metrics['cv_r2_mean']
            
            # Calculate win rate (if we can predict winners correctly)
            # For now, we'll use R2 as proxy for quality
            acceptable = test_r2 > 0.1 and cv_r2_mean > 0.0
            
            if not acceptable:
                print(f"⚠️ New model quality too low (Test R²: {test_r2:.3f}, CV R²: {cv_r2_mean:.3f})")
                return {
                    'status': 'quality_low',
                    'message': 'New model did not meet quality threshold',
                    'test_r2': test_r2,
                    'cv_r2': cv_r2_mean,
                    'threshold_met': False
                }
            
            # Generate version name
            version = f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            filepath = f"outputs/models/token_scorer_{version}.pkl"
            
            # Ensure directory exists
            Path("outputs/models").mkdir(parents=True, exist_ok=True)
            
            # Save new model
            scorer.save(filepath)
            print(f"💾 Model saved: {filepath}")
            
            # Update symlink to latest model
            symlink_path = "outputs/models/token_scorer.pkl"
            if Path(symlink_path).exists():
                Path(symlink_path).unlink()
            
            # Copy instead of symlink (Windows compatible)
            import shutil
            shutil.copy(filepath, symlink_path)
            print(f"🔗 Updated active model: {symlink_path}")
            
            # Save version to database
            self.db.save_model_version(
                version=version,
                filepath=filepath,
                accuracy=test_r2,  # Using R2 as proxy
                win_rate=0.0,  # Will be calculated later
                training_samples=len(df_clean),
                notes=f"Auto-retrained. Test R²: {test_r2:.3f}, CV R²: {cv_r2_mean:.3f}"
            )
            
            # Mark outcomes as used
            self.db.mark_outcomes_as_used()
            print(f"✓ Marked outcomes as used for training")
            
            return {
                'status': 'success',
                'message': 'Model successfully retrained',
                'version': version,
                'filepath': filepath,
                'metrics': {
                    'test_r2': test_r2,
                    'test_rmse': metrics['test']['rmse'],
                    'test_mae': metrics['test']['mae'],
                    'cv_r2_mean': cv_r2_mean,
                    'cv_r2_std': metrics['cv_r2_std']
                },
                'training_samples': len(df_clean),
                'restart_required': True
            }
            
        except Exception as e:
            print(f"✗ Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'failed',
                'message': str(e),
                'error': traceback.format_exc()
            }
    
    def get_status(self) -> Dict:
        """Get current retraining status"""
        new_outcomes = self.db.get_new_outcomes_count()
        
        # Get total training samples
        try:
            training_df = self.db.get_training_data()
            total_samples = len(training_df)
        except:
            total_samples = 0
        
        # Get active model version
        active_version = self.db.get_active_model_version()
        
        return {
            'retraining_active': self.retraining_active,
            'new_outcomes_count': new_outcomes,
            'total_training_samples': total_samples,
            'last_retrain_at': self.last_retrain_at,
            'next_retrain_trigger': self.retrain_threshold,
            'current_model_version': active_version['version'] if active_version else None,
            'current_model_accuracy': active_version['accuracy'] if active_version else None,
            'current_model_win_rate': active_version['win_rate'] if active_version else None
        }


class BackgroundRetrainingTask:
    """Background task that periodically checks for retraining"""
    
    def __init__(self, auto_retrainer: AutoRetrainer, check_interval: int = 300):
        """
        Initialize background task
        
        Args:
            auto_retrainer: AutoRetrainer instance
            check_interval: Check interval in seconds (default: 5 minutes)
        """
        self.auto_retrainer = auto_retrainer
        self.check_interval = check_interval
        self.running = False
        self.thread = None
    
    def start(self):
        """Start background retraining task"""
        if self.running:
            print("⚠️ Background retraining task already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"✓ Background retraining task started (check every {self.check_interval}s)")
    
    def stop(self):
        """Stop background retraining task"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("✓ Background retraining task stopped")
    
    def _run(self):
        """Background task loop"""
        while self.running:
            try:
                # Check and retrain if needed
                result = self.auto_retrainer.check_and_retrain()
                
                if result['status'] == 'success':
                    print(f"\n✅ Auto-retraining completed successfully!")
                    print(f"   Version: {result['version']}")
                    print(f"   Samples: {result['training_samples']}")
                    print(f"   Test R²: {result['metrics']['test_r2']:.3f}")
                    print(f"   ⚠️  Please restart API to load new model\n")
                
            except Exception as e:
                print(f"✗ Background retraining check failed: {e}")
            
            # Sleep until next check
            time.sleep(self.check_interval)


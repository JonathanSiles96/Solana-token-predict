"""
Data loading and preprocessing for Solana token data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.helpers import (
    parse_signal_params, 
    extract_gain_from_text,
    extract_ticker_from_text,
    extract_mc_from_text,
    parse_age_to_minutes
)


class DataLoader:
    """
    Load and preprocess Solana token trading data
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data files
        
        Returns:
            Tuple of (mints_df, early_trending_df, whale_trending_df)
        """
        # Load distinct mints
        mints_df = self.load_distinct_mints()
        
        # Load message data
        early_trending_df = self.load_early_trending()
        whale_trending_df = self.load_whale_trending()
        
        return mints_df, early_trending_df, whale_trending_df
    
    def load_distinct_mints(self) -> pd.DataFrame:
        """
        Load and parse distinct mints data
        
        Returns:
            DataFrame with parsed signal parameters
        """
        file_path = self.data_dir / "distinct_mints_with_max_return_null.csv"
        
        print(f"Loading distinct mints from {file_path}...")
        df = pd.read_csv(file_path)
        
        print(f"Loaded {len(df)} mints")
        
        # Parse signal_params JSON
        print("Parsing signal parameters...")
        signal_params_list = df['signal_params'].apply(parse_signal_params)
        
        # Extract individual fields from signal_params
        params_df = pd.DataFrame(signal_params_list.tolist())
        
        # Combine with original dataframe
        result_df = pd.concat([df[['mint_key', 'max_return']], params_df], axis=1)
        
        print(f"Extracted {len(params_df.columns)} features from signal_params")
        
        return result_df
    
    def load_early_trending(self) -> pd.DataFrame:
        """
        Load early trending messages
        
        Returns:
            DataFrame with early trending messages
        """
        file_path = self.data_dir / "solanaearlytrending_messages.csv"
        
        print(f"Loading early trending messages from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Convert date_time to datetime
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        print(f"Loaded {len(df)} early trending messages")
        print(f"Message types: {df['type'].value_counts().to_dict()}")
        
        return df
    
    def load_whale_trending(self) -> pd.DataFrame:
        """
        Load whale trending messages
        
        Returns:
            DataFrame with whale trending messages
        """
        file_path = self.data_dir / "whaletrending_messages.csv"
        
        print(f"Loading whale trending messages from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Convert date_time to datetime
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        print(f"Loaded {len(df)} whale trending messages")
        print(f"Message types: {df['type'].value_counts().to_dict()}")
        
        return df
    
    def extract_entry_signals(self, messages_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract entry signal data with gains
        
        Args:
            messages_df: Messages dataframe (early or whale trending)
            
        Returns:
            DataFrame with entry signals and extracted gains
        """
        # Filter to entry_signal type
        entry_signals = messages_df[messages_df['type'] == 'entry_signal'].copy()
        
        print(f"Found {len(entry_signals)} entry signals")
        
        # Extract gain information
        entry_signals['gain_pct'] = entry_signals['text'].apply(extract_gain_from_text)
        entry_signals['ticker'] = entry_signals['text'].apply(extract_ticker_from_text)
        
        # Extract MC values
        mc_data = entry_signals['text'].apply(extract_mc_from_text)
        entry_signals['entry_mc'] = mc_data.apply(lambda x: x[0])
        entry_signals['exit_mc'] = mc_data.apply(lambda x: x[1])
        
        # Calculate realized gain from MC if available
        entry_signals['mc_gain'] = (
            (entry_signals['exit_mc'] - entry_signals['entry_mc']) / entry_signals['entry_mc']
        ).fillna(entry_signals['gain_pct'])
        
        # Use mc_gain if available, otherwise gain_pct
        entry_signals['final_gain'] = entry_signals['mc_gain'].fillna(entry_signals['gain_pct'])
        
        # Filter out rows with no gain data
        valid_signals = entry_signals[entry_signals['final_gain'].notna()].copy()
        
        print(f"Extracted {len(valid_signals)} entry signals with gain data")
        print(f"Gain range: {valid_signals['final_gain'].min():.2f} to {valid_signals['final_gain'].max():.2f}")
        print(f"Mean gain: {valid_signals['final_gain'].mean():.2f} ({valid_signals['final_gain'].mean()*100:.1f}%)")
        print(f"Median gain: {valid_signals['final_gain'].median():.2f} ({valid_signals['final_gain'].median()*100:.1f}%)")
        
        return valid_signals
    
    def create_analysis_dataset(self, 
                                mints_df: pd.DataFrame,
                                entry_signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create combined dataset for analysis
        
        This links entry signals back to original mint data when possible
        
        Args:
            mints_df: Mints dataframe
            entry_signals_df: Entry signals dataframe
            
        Returns:
            Combined analysis dataset
        """
        # For now, return mints_df with additional computed features
        # In a production system, we'd match signals to mints via mint addresses
        
        print("Creating analysis dataset...")
        
        analysis_df = mints_df.copy()
        
        # Add gain statistics as a reference
        gain_stats = {
            'mean_observed_gain': entry_signals_df['final_gain'].mean(),
            'median_observed_gain': entry_signals_df['final_gain'].median(),
            'p75_observed_gain': entry_signals_df['final_gain'].quantile(0.75),
            'p90_observed_gain': entry_signals_df['final_gain'].quantile(0.90),
            'total_signals': len(entry_signals_df)
        }
        
        print(f"Gain statistics for reference:")
        for key, value in gain_stats.items():
            print(f"  {key}: {value:.4f}")
        
        return analysis_df
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for a dataframe
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary of summary statistics
        """
        stats = {
            'total_records': len(df),
            'columns': len(df.columns),
            'missing_pct': (df.isna().sum() / len(df) * 100).to_dict()
        }
        
        return stats


class FeatureExtractor:
    """
    Extract features from raw data for modeling
    """
    
    @staticmethod
    def extract_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and clean numeric features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with cleaned numeric features
        """
        numeric_cols = [
            'signal_mc', 'signal_top_mc', 'signal_best_mc',
            'signal_liquidity', 'signal_volume_1h',
            'signal_holders', 'signal_dev_sol',
            'signal_fish_pct', 'signal_sold_pct', 'signal_bundled_pct',
            'signal_snipers_pct', 'signal_first_20_pct',
            'signal_fish_count', 'signal_bond', 'signal_made'
        ]
        
        result_df = df.copy()
        
        # Ensure numeric types
        for col in numeric_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        
        return result_df
    
    @staticmethod
    def extract_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and encode categorical features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with encoded categorical features
        """
        result_df = df.copy()
        
        # Security status encoding
        if 'signal_security' in result_df.columns:
            security_map = {'✅': 0, '⚠️': 1, '🚨': 2}
            result_df['security_encoded'] = result_df['signal_security'].map(security_map).fillna(1)
        
        # Age parsing
        if 'signal_age_mark' in result_df.columns:
            result_df['age_minutes'] = result_df['signal_age_mark'].apply(parse_age_to_minutes)
        
        return result_df
    
    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived/calculated features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with derived features
        """
        result_df = df.copy()
        
        # Liquidity to MC ratio
        if 'signal_liquidity' in result_df.columns and 'signal_mc' in result_df.columns:
            result_df['liq_to_mc_ratio'] = (
                result_df['signal_liquidity'] / result_df['signal_mc']
            ).replace([np.inf, -np.inf], np.nan)
        
        # Volume to liquidity ratio
        if 'signal_volume_1h' in result_df.columns and 'signal_liquidity' in result_df.columns:
            result_df['vol_to_liq_ratio'] = (
                result_df['signal_volume_1h'] / result_df['signal_liquidity']
            ).replace([np.inf, -np.inf], np.nan)
        
        # MC volatility
        if 'signal_top_mc' in result_df.columns and 'signal_mc' in result_df.columns:
            result_df['mc_volatility'] = (
                (result_df['signal_top_mc'] - result_df['signal_mc']) / result_df['signal_mc']
            ).replace([np.inf, -np.inf], np.nan)
        
        # Best to signal MC ratio (potential)
        if 'signal_best_mc' in result_df.columns and 'signal_mc' in result_df.columns:
            result_df['best_to_signal_ratio'] = (
                result_df['signal_best_mc'] / result_df['signal_mc']
            ).replace([np.inf, -np.inf], np.nan)
        
        # Holder concentration (first 20%)
        if 'signal_first_20_pct' in result_df.columns:
            result_df['holder_concentration'] = result_df['signal_first_20_pct']
        
        # Risk score composite
        risk_components = []
        if 'signal_bundled_pct' in result_df.columns:
            risk_components.append(result_df['signal_bundled_pct'].fillna(0) / 100)
        if 'signal_snipers_pct' in result_df.columns:
            risk_components.append(result_df['signal_snipers_pct'].fillna(0) / 100)
        if 'signal_sold_pct' in result_df.columns:
            risk_components.append(result_df['signal_sold_pct'].fillna(0) / 100)
        
        if risk_components:
            result_df['risk_score'] = pd.concat(risk_components, axis=1).mean(axis=1)
        
        return result_df


def load_and_prepare_data(data_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and prepare all data
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (prepared_mints_df, early_signals_df, whale_signals_df)
    """
    loader = DataLoader(data_dir)
    
    # Load raw data
    mints_df, early_df, whale_df = loader.load_all_data()
    
    # Extract entry signals
    early_signals = loader.extract_entry_signals(early_df)
    whale_signals = loader.extract_entry_signals(whale_df)
    
    # Extract features
    extractor = FeatureExtractor()
    mints_df = extractor.extract_numeric_features(mints_df)
    mints_df = extractor.extract_categorical_features(mints_df)
    mints_df = extractor.add_derived_features(mints_df)
    
    return mints_df, early_signals, whale_signals


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")
    mints_df, early_signals, whale_signals = load_and_prepare_data(".")
    
    print(f"\nMints DataFrame shape: {mints_df.shape}")
    print(f"Early signals shape: {early_signals.shape}")
    print(f"Whale signals shape: {whale_signals.shape}")
    
    print("\nSample features:")
    print(mints_df.head())


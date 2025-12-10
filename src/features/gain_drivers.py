"""
Feature engineering focused on identifying gain drivers
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class GainDriverAnalyzer:
    """
    Analyze and identify key drivers of token gains
    """
    
    def __init__(self):
        """Initialize the gain driver analyzer"""
        self.feature_importance = {}
        self.correlations = {}
        self.scaler = StandardScaler()
        
    def identify_key_features(self, df: pd.DataFrame) -> List[str]:
        """
        Identify key features available for analysis
        
        Args:
            df: Input dataframe
            
        Returns:
            List of feature column names
        """
        # Core market metrics
        market_features = [
            'signal_mc', 'signal_top_mc', 'signal_liquidity',
            'signal_volume_1h', 'signal_holders'
        ]
        
        # Risk indicators
        risk_features = [
            'signal_bundled_pct', 'signal_snipers_pct',
            'signal_sold_pct', 'signal_fish_pct',
            'signal_first_20_pct', 'security_encoded'
        ]
        
        # Derived features
        derived_features = [
            'liq_to_mc_ratio', 'vol_to_liq_ratio',
            'mc_volatility', 'risk_score', 
            'age_minutes', 'holder_concentration'
        ]
        
        # Additional features
        other_features = [
            'signal_dev_sol', 'signal_fish_count',
            'signal_bond', 'signal_made'
        ]
        
        all_features = market_features + risk_features + derived_features + other_features
        
        # Filter to only features that exist in dataframe
        available_features = [f for f in all_features if f in df.columns]
        
        print(f"Identified {len(available_features)} available features")
        
        return available_features
    
    def calculate_correlation_with_gain(self, 
                                       df: pd.DataFrame,
                                       features: List[str],
                                       gain_col: str = 'final_gain') -> pd.DataFrame:
        """
        Calculate correlation between features and gains
        
        Args:
            df: Input dataframe with features and gain column
            features: List of feature columns
            gain_col: Name of gain column
            
        Returns:
            DataFrame with correlation results
        """
        correlations = []
        
        for feature in features:
            if feature in df.columns and gain_col in df.columns:
                # Remove NaN values
                valid_data = df[[feature, gain_col]].dropna()
                
                if len(valid_data) > 10:  # Need sufficient data
                    # Pearson correlation
                    pearson_corr, pearson_p = stats.pearsonr(
                        valid_data[feature], 
                        valid_data[gain_col]
                    )
                    
                    # Spearman correlation (rank-based, more robust)
                    spearman_corr, spearman_p = stats.spearmanr(
                        valid_data[feature],
                        valid_data[gain_col]
                    )
                    
                    correlations.append({
                        'feature': feature,
                        'pearson_corr': pearson_corr,
                        'pearson_pval': pearson_p,
                        'spearman_corr': spearman_corr,
                        'spearman_pval': spearman_p,
                        'valid_samples': len(valid_data),
                        'abs_corr': abs(spearman_corr)  # For sorting
                    })
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_corr', ascending=False)
        
        self.correlations = corr_df
        
        return corr_df
    
    def feature_importance_rf(self,
                             df: pd.DataFrame,
                             features: List[str],
                             target_col: str = 'final_gain',
                             n_estimators: int = 100) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest
        
        Args:
            df: Input dataframe
            features: List of feature columns
            target_col: Target column name
            n_estimators: Number of trees
            
        Returns:
            DataFrame with feature importance
        """
        # Prepare data
        valid_features = [f for f in features if f in df.columns]
        data = df[valid_features + [target_col]].dropna()
        
        if len(data) < 20:
            print("Insufficient data for feature importance analysis")
            return pd.DataFrame()
        
        X = data[valid_features]
        y = data[target_col]
        
        # Train Random Forest
        print(f"Training Random Forest with {len(data)} samples...")
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Extract feature importance
        importance_df = pd.DataFrame({
            'feature': valid_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['random_forest'] = importance_df
        
        return importance_df
    
    def compare_high_vs_low_gainers(self,
                                    df: pd.DataFrame,
                                    features: List[str],
                                    gain_col: str = 'final_gain',
                                    threshold: float = 0.5) -> pd.DataFrame:
        """
        Compare features between high and low gainers
        
        Args:
            df: Input dataframe
            features: List of features to compare
            gain_col: Gain column name
            threshold: Threshold for high gainers (50% = 0.5)
            
        Returns:
            DataFrame with comparison statistics
        """
        if gain_col not in df.columns:
            print(f"Gain column '{gain_col}' not found")
            return pd.DataFrame()
        
        # Split into high and low gainers
        high_gainers = df[df[gain_col] >= threshold]
        low_gainers = df[df[gain_col] < threshold]
        
        print(f"High gainers (>={threshold*100}%): {len(high_gainers)}")
        print(f"Low gainers (<{threshold*100}%): {len(low_gainers)}")
        
        comparisons = []
        
        for feature in features:
            if feature in df.columns:
                high_vals = high_gainers[feature].dropna()
                low_vals = low_gainers[feature].dropna()
                
                if len(high_vals) > 5 and len(low_vals) > 5:
                    # T-test
                    t_stat, t_pval = stats.ttest_ind(high_vals, low_vals, equal_var=False)
                    
                    # Mann-Whitney U test (non-parametric)
                    u_stat, u_pval = stats.mannwhitneyu(high_vals, low_vals, alternative='two-sided')
                    
                    comparisons.append({
                        'feature': feature,
                        'high_mean': high_vals.mean(),
                        'low_mean': low_vals.mean(),
                        'high_median': high_vals.median(),
                        'low_median': low_vals.median(),
                        'mean_diff_pct': ((high_vals.mean() - low_vals.mean()) / low_vals.mean() * 100) if low_vals.mean() != 0 else np.nan,
                        't_statistic': t_stat,
                        't_pvalue': t_pval,
                        'u_pvalue': u_pval,
                        'significant': u_pval < 0.05
                    })
        
        comp_df = pd.DataFrame(comparisons)
        comp_df = comp_df.sort_values('u_pvalue')
        
        return comp_df
    
    def analyze_regime_based_performance(self,
                                        df: pd.DataFrame,
                                        features: List[str],
                                        gain_col: str = 'final_gain') -> Dict:
        """
        Analyze performance across different market regimes
        
        Regimes based on:
        - Market cap levels (small, medium, large)
        - Liquidity levels
        - Volume levels
        
        Args:
            df: Input dataframe
            features: List of features
            gain_col: Gain column name
            
        Returns:
            Dictionary of regime-based analysis
        """
        results = {}
        
        if 'signal_mc' in df.columns and gain_col in df.columns:
            # MC-based regimes
            df_with_gain = df[df[gain_col].notna()].copy()
            
            if len(df_with_gain) > 0:
                # Define MC quantiles
                mc_quantiles = df_with_gain['signal_mc'].quantile([0.33, 0.67])
                
                df_with_gain['mc_regime'] = pd.cut(
                    df_with_gain['signal_mc'],
                    bins=[0, mc_quantiles[0.33], mc_quantiles[0.67], np.inf],
                    labels=['small', 'medium', 'large']
                )
                
                # Analyze by regime
                regime_stats = df_with_gain.groupby('mc_regime')[gain_col].agg([
                    'count', 'mean', 'median', 'std',
                    ('p25', lambda x: x.quantile(0.25)),
                    ('p75', lambda x: x.quantile(0.75))
                ])
                
                results['mc_regimes'] = regime_stats
        
        if 'liq_to_mc_ratio' in df.columns and gain_col in df.columns:
            # Liquidity ratio regimes
            df_with_gain = df[df[gain_col].notna() & df['liq_to_mc_ratio'].notna()].copy()
            
            if len(df_with_gain) > 0:
                liq_quantiles = df_with_gain['liq_to_mc_ratio'].quantile([0.33, 0.67])
                
                df_with_gain['liq_regime'] = pd.cut(
                    df_with_gain['liq_to_mc_ratio'],
                    bins=[0, liq_quantiles[0.33], liq_quantiles[0.67], np.inf],
                    labels=['low_liq', 'med_liq', 'high_liq']
                )
                
                regime_stats = df_with_gain.groupby('liq_regime')[gain_col].agg([
                    'count', 'mean', 'median', 'std'
                ])
                
                results['liquidity_regimes'] = regime_stats
        
        return results
    
    def create_feature_summary(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Create summary statistics for all features
        
        Args:
            df: Input dataframe
            features: List of features
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for feature in features:
            if feature in df.columns:
                feature_data = df[feature].dropna()
                
                if len(feature_data) > 0:
                    summary_data.append({
                        'feature': feature,
                        'count': len(feature_data),
                        'missing_pct': (df[feature].isna().sum() / len(df)) * 100,
                        'mean': feature_data.mean(),
                        'std': feature_data.std(),
                        'min': feature_data.min(),
                        'p25': feature_data.quantile(0.25),
                        'median': feature_data.median(),
                        'p75': feature_data.quantile(0.75),
                        'max': feature_data.max()
                    })
        
        return pd.DataFrame(summary_data)
    
    def generate_comprehensive_report(self,
                                     df: pd.DataFrame,
                                     gain_col: str = 'final_gain') -> Dict:
        """
        Generate comprehensive gain driver analysis report
        
        Args:
            df: Input dataframe
            gain_col: Gain column name
            
        Returns:
            Dictionary containing all analysis results
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE GAIN DRIVER ANALYSIS")
        print("="*80)
        
        report = {}
        
        # Identify features
        features = self.identify_key_features(df)
        report['features'] = features
        
        # Feature summary
        print("\n1. Feature Summary Statistics")
        print("-" * 80)
        feature_summary = self.create_feature_summary(df, features)
        report['feature_summary'] = feature_summary
        print(feature_summary.to_string())
        
        # Correlation analysis
        print("\n2. Correlation with Gains")
        print("-" * 80)
        correlations = self.calculate_correlation_with_gain(df, features, gain_col)
        report['correlations'] = correlations
        print(correlations.head(15).to_string())
        
        # High vs Low gainers
        print("\n3. High vs Low Gainers Comparison")
        print("-" * 80)
        comparison = self.compare_high_vs_low_gainers(df, features, gain_col)
        report['high_low_comparison'] = comparison
        if not comparison.empty:
            print(comparison.head(15).to_string())
        
        # Regime analysis
        print("\n4. Regime-Based Analysis")
        print("-" * 80)
        regime_analysis = self.analyze_regime_based_performance(df, features, gain_col)
        report['regime_analysis'] = regime_analysis
        for regime_type, stats in regime_analysis.items():
            print(f"\n{regime_type}:")
            print(stats.to_string())
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80 + "\n")
        
        return report


if __name__ == "__main__":
    # Test with sample data
    print("Testing gain driver analyzer...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_df = pd.DataFrame({
        'signal_mc': np.random.lognormal(11, 1, n_samples),
        'signal_liquidity': np.random.lognormal(10, 0.8, n_samples),
        'signal_volume_1h': np.random.lognormal(10, 1.2, n_samples),
        'signal_holders': np.random.poisson(500, n_samples),
        'liq_to_mc_ratio': np.random.uniform(0.1, 0.5, n_samples),
        'final_gain': np.random.lognormal(-1, 1.5, n_samples)
    })
    
    analyzer = GainDriverAnalyzer()
    report = analyzer.generate_comprehensive_report(sample_df)
    
    print("Test complete!")


"""
Prediction Analyzer - Compare model predictions against actual outcomes

This module analyzes prediction accuracy by comparing what the model predicted
against what actually happened (from Trace_24H data).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.database import SignalDatabase


class PredictionAnalyzer:
    """
    Analyzes prediction accuracy and identifies model weaknesses
    
    This helps understand:
    - How accurate are predictions?
    - Does model over/under estimate gains?
    - Which token types does model struggle with?
    - What features drive prediction errors?
    """
    
    def __init__(self, db_path: str = "data/signals.db"):
        """Initialize analyzer with database connection"""
        self.db = SignalDatabase(db_path)
    
    def get_prediction_vs_actual(self, min_samples: int = 10) -> pd.DataFrame:
        """
        Get predictions vs actual outcomes
        
        Args:
            min_samples: Minimum samples required
            
        Returns:
            DataFrame with predictions and actuals
        """
        # Get training data (signals with outcomes)
        df = self.db.get_training_data()
        
        if df.empty or len(df) < min_samples:
            print(f"⚠️  Insufficient data: {len(df)} samples (need {min_samples})")
            return pd.DataFrame()
        
        # For now, we don't have stored predictions per signal
        # So we'll analyze actual outcomes and their characteristics
        # Once we implement prediction storage, we can do true pred vs actual
        
        return df
    
    def analyze_accuracy_by_feature(self, feature: str = 'signal_mc') -> Dict:
        """
        Analyze accuracy broken down by feature value
        
        Args:
            feature: Feature to segment by (e.g., 'signal_mc', 'signal_liquidity')
            
        Returns:
            Dict with accuracy metrics by feature bins
        """
        df = self.get_prediction_vs_actual()
        
        if df.empty:
            return {'status': 'insufficient_data'}
        
        # Create bins for the feature
        try:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                df['feature_bin'] = pd.qcut(df[feature], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                
                # Calculate metrics by bin
                results = {}
                for bin_name in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
                    bin_data = df[df['feature_bin'] == bin_name]
                    
                    if len(bin_data) > 0:
                        results[bin_name] = {
                            'count': len(bin_data),
                            'avg_final_gain': bin_data['final_gain'].mean(),
                            'avg_max_return': bin_data['max_return'].mean() if 'max_return' in bin_data else None,
                            'win_rate': (bin_data['final_gain'] >= 0.3).sum() / len(bin_data)
                        }
                
                return {
                    'status': 'success',
                    'feature': feature,
                    'bins': results
                }
            else:
                return {'status': 'feature_not_found', 'feature': feature}
                
        except Exception as e:
            print(f"✗ Error analyzing by {feature}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def identify_prediction_patterns(self) -> Dict:
        """
        Identify common patterns in prediction errors
        
        Returns:
            Dict with identified patterns
        """
        df = self.get_prediction_vs_actual()
        
        if df.empty:
            return {'status': 'insufficient_data'}
        
        patterns = {}
        
        # Pattern 1: High MC tokens
        if 'signal_mc' in df.columns:
            high_mc = df[df['signal_mc'] > 100000]
            if len(high_mc) > 5:
                patterns['high_mc_tokens'] = {
                    'count': len(high_mc),
                    'avg_gain': high_mc['final_gain'].mean(),
                    'win_rate': (high_mc['final_gain'] >= 0.3).sum() / len(high_mc)
                }
        
        # Pattern 2: Low liquidity tokens
        if 'signal_liquidity' in df.columns:
            low_liq = df[df['signal_liquidity'] < 10000]
            if len(low_liq) > 5:
                patterns['low_liquidity_tokens'] = {
                    'count': len(low_liq),
                    'avg_gain': low_liq['final_gain'].mean(),
                    'win_rate': (low_liq['final_gain'] >= 0.3).sum() / len(low_liq)
                }
        
        # Pattern 3: High risk tokens
        if 'signal_bundled_pct' in df.columns:
            high_risk = df[df['signal_bundled_pct'] > 10]
            if len(high_risk) > 5:
                patterns['high_bundled_tokens'] = {
                    'count': len(high_risk),
                    'avg_gain': high_risk['final_gain'].mean(),
                    'win_rate': (high_risk['final_gain'] >= 0.3).sum() / len(high_risk)
                }
        
        # Pattern 4: By source
        if 'signal_source' in df.columns:
            for source in df['signal_source'].unique():
                if pd.notna(source):
                    source_data = df[df['signal_source'] == source]
                    if len(source_data) > 5:
                        patterns[f'source_{source}'] = {
                            'count': len(source_data),
                            'avg_gain': source_data['final_gain'].mean(),
                            'win_rate': (source_data['final_gain'] >= 0.3).sum() / len(source_data)
                        }
        
        return {
            'status': 'success',
            'total_samples': len(df),
            'patterns': patterns
        }
    
    def generate_improvement_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations for model improvement
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Get patterns
        patterns = self.identify_prediction_patterns()
        
        if patterns['status'] != 'success':
            return ["Insufficient data for recommendations"]
        
        pattern_data = patterns.get('patterns', {})
        
        # Analyze each pattern
        for pattern_name, data in pattern_data.items():
            win_rate = data.get('win_rate', 0)
            avg_gain = data.get('avg_gain', 0)
            count = data.get('count', 0)
            
            if count < 5:
                continue
            
            # Low win rate patterns
            if win_rate < 0.3:
                recommendations.append(
                    f"⚠️  {pattern_name}: Low win rate ({win_rate*100:.1f}%). "
                    f"Consider filtering these tokens or adjusting predictions."
                )
            
            # High win rate patterns
            elif win_rate > 0.7:
                recommendations.append(
                    f"✓ {pattern_name}: High win rate ({win_rate*100:.1f}%). "
                    f"Model performs well here - prioritize these tokens."
                )
            
            # Negative average gain
            if avg_gain < 0:
                recommendations.append(
                    f"🚨 {pattern_name}: Negative avg gain ({avg_gain*100:.1f}%). "
                    f"Model should filter these out more aggressively."
                )
        
        # Get feature analysis
        for feature in ['signal_mc', 'signal_liquidity', 'signal_volume_1h']:
            feature_analysis = self.analyze_accuracy_by_feature(feature)
            
            if feature_analysis.get('status') == 'success':
                bins = feature_analysis.get('bins', {})
                
                # Find best and worst performing bins
                if bins:
                    bin_performance = {k: v.get('win_rate', 0) for k, v in bins.items() if v.get('count', 0) >= 3}
                    
                    if bin_performance:
                        best_bin = max(bin_performance, key=bin_performance.get)
                        worst_bin = min(bin_performance, key=bin_performance.get)
                        
                        if bin_performance[worst_bin] < 0.3:
                            recommendations.append(
                                f"💡 {feature}: {worst_bin} range underperforms "
                                f"({bin_performance[worst_bin]*100:.1f}% win rate). "
                                f"Consider adjusting filters."
                            )
                        
                        if bin_performance[best_bin] > 0.6:
                            recommendations.append(
                                f"✨ {feature}: {best_bin} range performs well "
                                f"({bin_performance[best_bin]*100:.1f}% win rate). "
                                f"Prioritize these values."
                            )
        
        if not recommendations:
            recommendations.append("✓ Model performing reasonably across all segments. Continue monitoring.")
        
        return recommendations
    
    def export_analysis_report(self, output_path: str = "outputs/reports/prediction_analysis.txt"):
        """
        Export comprehensive analysis report
        
        Args:
            output_path: Path to save report
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PREDICTION ACCURACY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall stats
            df = self.get_prediction_vs_actual()
            if not df.empty:
                f.write(f"Total Samples: {len(df)}\n")
                f.write(f"Avg Final Gain: {df['final_gain'].mean()*100:.1f}%\n")
                
                if 'max_return' in df.columns:
                    max_ret = df['max_return'].dropna()
                    if len(max_ret) > 0:
                        f.write(f"Avg Max Return: {max_ret.mean()*100:.1f}%\n")
                
                win_rate = (df['final_gain'] >= 0.3).sum() / len(df)
                f.write(f"Win Rate: {win_rate*100:.1f}%\n\n")
            
            # Patterns
            f.write("-"*80 + "\n")
            f.write("IDENTIFIED PATTERNS\n")
            f.write("-"*80 + "\n\n")
            
            patterns = self.identify_prediction_patterns()
            if patterns['status'] == 'success':
                for pattern_name, data in patterns.get('patterns', {}).items():
                    f.write(f"{pattern_name}:\n")
                    f.write(f"  Count: {data['count']}\n")
                    f.write(f"  Avg Gain: {data['avg_gain']*100:.1f}%\n")
                    f.write(f"  Win Rate: {data['win_rate']*100:.1f}%\n\n")
            
            # Recommendations
            f.write("-"*80 + "\n")
            f.write("IMPROVEMENT RECOMMENDATIONS\n")
            f.write("-"*80 + "\n\n")
            
            recommendations = self.generate_improvement_recommendations()
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Analysis report exported to {output_file}")


if __name__ == "__main__":
    """Example usage"""
    print("🔍 Prediction Analyzer")
    print("="*80)
    
    analyzer = PredictionAnalyzer()
    
    # Get patterns
    print("\n📊 Analyzing prediction patterns...")
    patterns = analyzer.identify_prediction_patterns()
    
    if patterns['status'] == 'success':
        print(f"\nTotal samples: {patterns['total_samples']}")
        print(f"\nIdentified {len(patterns.get('patterns', {}))} patterns:")
        
        for name, data in patterns.get('patterns', {}).items():
            print(f"\n  {name}:")
            print(f"    Count: {data['count']}")
            print(f"    Avg Gain: {data['avg_gain']*100:.1f}%")
            print(f"    Win Rate: {data['win_rate']*100:.1f}%")
    else:
        print(f"\n⚠️  {patterns.get('message', 'Insufficient data')}")
    
    # Get recommendations
    print("\n" + "="*80)
    print("💡 IMPROVEMENT RECOMMENDATIONS")
    print("="*80)
    
    recommendations = analyzer.generate_improvement_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")
    
    # Export report
    print("\n" + "="*80)
    analyzer.export_analysis_report()
    print("\n✓ Analysis complete!")


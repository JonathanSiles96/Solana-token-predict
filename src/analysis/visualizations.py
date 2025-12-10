"""
Visualization utilities for analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class AnalysisVisualizer:
    """
    Create visualizations for token analysis
    """
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_feature_importance(self, 
                               importance_df: pd.DataFrame,
                               top_n: int = 20,
                               save_name: str = "feature_importance.png"):
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to show
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance_df.head(top_n)
        
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.invert_yaxis()
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_correlation_matrix(self,
                               df: pd.DataFrame,
                               features: List[str],
                               save_name: str = "correlation_matrix.png"):
        """
        Plot correlation matrix
        
        Args:
            df: Input dataframe
            features: List of features
            save_name: Filename to save
        """
        # Select available features
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < 2:
            print("Not enough features for correlation matrix")
            return
        
        # Calculate correlation
        corr_data = df[available_features].corr()
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr_data, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        ax.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_gain_distribution(self,
                              gains: pd.Series,
                              save_name: str = "gain_distribution.png"):
        """
        Plot gain distribution
        
        Args:
            gains: Series of gains
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(gains, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Gain')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Gain Distribution')
        axes[0].axvline(gains.mean(), color='red', linestyle='--', label=f'Mean: {gains.mean():.2f}')
        axes[0].axvline(gains.median(), color='green', linestyle='--', label=f'Median: {gains.median():.2f}')
        axes[0].legend()
        
        # Log scale
        axes[1].hist(gains[gains > 0], bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Gain')
        axes[1].set_ylabel('Frequency (Log Scale)')
        axes[1].set_title('Gain Distribution (Positive Gains, Log Scale)')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_feature_vs_gain(self,
                            df: pd.DataFrame,
                            feature: str,
                            gain_col: str = 'final_gain',
                            save_name: Optional[str] = None):
        """
        Plot feature vs gain scatter plot
        
        Args:
            df: Input dataframe
            feature: Feature column name
            gain_col: Gain column name
            save_name: Filename to save (auto-generated if None)
        """
        if feature not in df.columns or gain_col not in df.columns:
            print(f"Feature '{feature}' or gain column not found")
            return
        
        # Remove NaN
        plot_data = df[[feature, gain_col]].dropna()
        
        if len(plot_data) < 10:
            print(f"Insufficient data for {feature}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(plot_data[feature], plot_data[gain_col], alpha=0.5, s=20)
        ax.set_xlabel(feature)
        ax.set_ylabel('Gain')
        ax.set_title(f'{feature} vs Gain')
        
        # Add trend line
        z = np.polyfit(plot_data[feature], plot_data[gain_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(plot_data[feature].min(), plot_data[feature].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend')
        ax.legend()
        
        plt.tight_layout()
        
        if save_name is None:
            save_name = f"feature_vs_gain_{feature}.png"
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_model_performance(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              save_name: str = "model_performance.png"):
        """
        Plot model performance (predicted vs actual)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Gain')
        axes[0].set_ylabel('Predicted Gain')
        axes[0].set_title('Predicted vs Actual Gain')
        axes[0].legend()
        
        # Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Gain')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_regime_analysis(self,
                            regime_stats: pd.DataFrame,
                            regime_name: str,
                            save_name: Optional[str] = None):
        """
        Plot regime-based analysis
        
        Args:
            regime_stats: DataFrame with regime statistics
            regime_name: Name of regime
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Mean gain by regime
        axes[0].bar(range(len(regime_stats)), regime_stats['mean'])
        axes[0].set_xticks(range(len(regime_stats)))
        axes[0].set_xticklabels(regime_stats.index)
        axes[0].set_ylabel('Mean Gain')
        axes[0].set_title(f'Mean Gain by {regime_name}')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Sample count by regime
        axes[1].bar(range(len(regime_stats)), regime_stats['count'], color='orange')
        axes[1].set_xticks(range(len(regime_stats)))
        axes[1].set_xticklabels(regime_stats.index)
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Sample Count by {regime_name}')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_name is None:
            save_name = f"regime_analysis_{regime_name}.png"
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def create_summary_dashboard(self,
                                metrics: Dict,
                                save_name: str = "summary_dashboard.png"):
        """
        Create a summary dashboard with key metrics
        
        Args:
            metrics: Dictionary with various metrics
            save_name: Filename to save
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Token Analysis Summary Dashboard', fontsize=16, fontweight='bold')
        
        # Add text summaries and plots based on available metrics
        # This is a placeholder - customize based on your metrics structure
        
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        summary_text = f"""
        Analysis Summary:
        - Total Tokens: {metrics.get('total_tokens', 'N/A')}
        - Mean Gain: {metrics.get('mean_gain', 'N/A')}
        - Model R²: {metrics.get('r2_score', 'N/A')}
        """
        ax1.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualizations...")
    
    np.random.seed(42)
    n = 500
    
    test_df = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'final_gain': np.random.lognormal(0, 1, n)
    })
    
    viz = AnalysisVisualizer(output_dir="test_outputs")
    viz.plot_gain_distribution(test_df['final_gain'])
    
    print("Test complete!")


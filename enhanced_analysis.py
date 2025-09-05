import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import our models
from models import (
    LinearRegressionModel, RandomForestModel, XGBoostModel, SVRModel,
    ModelComparison, load_and_prepare_data
)

class SneakerPriceAnalysis:
    """Enhanced analysis class for sneaker price prediction"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.X = None
        self.y = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.model_comparison = ModelComparison()
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading and preparing data...")
        self.X, self.y, self.feature_names = load_and_prepare_data(self.data_path)
        
        if self.X is not None:
            print(f"✓ Data loaded successfully")
            print(f"  Features: {self.X.shape[1]}")
            print(f"  Samples: {self.X.shape[0]}")
            print(f"  Target range: ${self.y.min():.2f} - ${self.y.max():.2f}")
        else:
            print("✗ Failed to load data")
        
        return self.X is not None
    
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        if self.X is None:
            raise ValueError("Data must be loaded first")
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        print(f"✓ Data split completed")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Validation set: {self.X_val.shape[0]} samples")
        print(f"  Test set: {self.X_test.shape[0]} samples")
    
    def initialize_models(self):
        """Initialize all models for comparison"""
        print("Initializing models...")
        
        # Create model instances
        models = {
            'Linear Regression': LinearRegressionModel(),
            'Random Forest': RandomForestModel(n_estimators=100, random_state=42),
            'XGBoost': XGBoostModel(n_estimators=100, random_state=42),
            'SVR': SVRModel(kernel='rbf', C=1.0)
        }
        
        # Add models to comparison
        for name, model in models.items():
            self.model_comparison.add_model(name, model)
        
        print(f"✓ {len(models)} models initialized")
    
    def train_all_models(self):
        """Train all models"""
        if self.X_train is None:
            raise ValueError("Data must be split first")
        
        self.model_comparison.train_all_models(
            self.X_train, self.y_train, self.X_val, self.y_val
        )
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        if self.X_test is None:
            raise ValueError("Data must be split first")
        
        self.model_comparison.evaluate_all_models(self.X_test, self.y_test)
        return self.model_comparison.get_results_dataframe()
    
    def plot_model_comparison(self, save_path=None):
        """Create comprehensive model comparison plots"""
        results_df = self.model_comparison.get_results_dataframe()
        
        if results_df.empty:
            print("No results available for plotting")
            return
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. R² Score Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(results_df['Model'], results_df['R² Score'], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('R² Score Comparison', fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. MAE Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(results_df['Model'], results_df['MAE'],
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_title('Mean Absolute Error (MAE)', fontweight='bold')
        ax2.set_ylabel('MAE ($)')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'${height:.1f}', ha='center', va='bottom')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. RMSE Comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(results_df['Model'], results_df['RMSE'],
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax3.set_title('Root Mean Square Error (RMSE)', fontweight='bold')
        ax3.set_ylabel('RMSE ($)')
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'${height:.1f}', ha='center', va='bottom')
        
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Combined Metrics Radar Chart
        ax4 = axes[1, 1]
        
        # Normalize metrics for radar chart (0-1 scale)
        normalized_df = results_df.copy()
        normalized_df['R² Score'] = results_df['R² Score']  # Already 0-1
        normalized_df['MAE_norm'] = 1 - (results_df['MAE'] - results_df['MAE'].min()) / (results_df['MAE'].max() - results_df['MAE'].min())
        normalized_df['RMSE_norm'] = 1 - (results_df['RMSE'] - results_df['RMSE'].min()) / (results_df['RMSE'].max() - results_df['RMSE'].min())
        
        # Simple line plot showing overall performance
        models = normalized_df['Model'].tolist()
        r2_scores = normalized_df['R² Score'].tolist()
        mae_norm = normalized_df['MAE_norm'].tolist()
        rmse_norm = normalized_df['RMSE_norm'].tolist()
        
        x_pos = range(len(models))
        ax4.plot(x_pos, r2_scores, 'o-', label='R² Score', linewidth=2, markersize=8)
        ax4.plot(x_pos, mae_norm, 's-', label='MAE (normalized)', linewidth=2, markersize=8)
        ax4.plot(x_pos, rmse_norm, '^-', label='RMSE (normalized)', linewidth=2, markersize=8)
        
        ax4.set_title('Overall Performance Comparison', fontweight='bold')
        ax4.set_ylabel('Normalized Score (Higher = Better)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions_vs_actual(self, save_path=None):
        """Plot predictions vs actual values for all models"""
        if not self.model_comparison.results:
            print("No results available for plotting")
            return
        
        n_models = len(self.model_comparison.results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Predictions vs Actual Values', fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (model_name, results) in enumerate(self.model_comparison.results.items()):
            ax = axes_flat[idx]
            
            y_pred = results['predictions']
            
            # Scatter plot
            ax.scatter(self.y_test, y_pred, alpha=0.6, color=colors[idx])
            
            # Perfect prediction line
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Labels and title
            ax.set_xlabel('Actual Price ($)')
            ax.set_ylabel('Predicted Price ($)')
            ax.set_title(f'{model_name}\nR² = {results["r2_score"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, save_path=None):
        """Plot feature importance for tree-based models"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # Random Forest feature importance
        rf_model = self.model_comparison.models.get('Random Forest')
        if rf_model and rf_model.is_fitted:
            rf_importance = rf_model.get_feature_importance(self.feature_names)
            top_features_rf = rf_importance.head(10)
            
            ax1 = axes[0]
            bars1 = ax1.barh(range(len(top_features_rf)), top_features_rf['importance'])
            ax1.set_yticks(range(len(top_features_rf)))
            ax1.set_yticklabels(top_features_rf['feature'])
            ax1.set_xlabel('Importance')
            ax1.set_title('Random Forest - Top 10 Features')
            ax1.grid(True, alpha=0.3)
        
        # XGBoost feature importance
        xgb_model = self.model_comparison.models.get('XGBoost')
        if xgb_model and xgb_model.is_fitted:
            xgb_importance = xgb_model.get_feature_importance(self.feature_names)
            top_features_xgb = xgb_importance.head(10)
            
            ax2 = axes[1]
            bars2 = ax2.barh(range(len(top_features_xgb)), top_features_xgb['importance'])
            ax2.set_yticks(range(len(top_features_xgb)))
            ax2.set_yticklabels(top_features_xgb['feature'])
            ax2.set_xlabel('Importance')
            ax2.set_title('XGBoost - Top 10 Features')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def generate_model_report(self):
        """Generate a comprehensive model performance report"""
        results_df = self.model_comparison.get_results_dataframe()
        best_model_name, best_model = self.model_comparison.get_best_model()
        
        print("\n" + "="*60)
        print("SNEAKER PRICE PREDICTION - MODEL COMPARISON REPORT")
        print("="*60)
        
        print(f"\nDataset Information:")
        print(f"  • Total samples: {len(self.X)}")
        print(f"  • Features: {len(self.feature_names)}")
        print(f"  • Price range: ${self.y.min():.2f} - ${self.y.max():.2f}")
        print(f"  • Average price: ${self.y.mean():.2f}")
        
        print(f"\nData Split:")
        print(f"  • Training: {len(self.X_train)} samples ({len(self.X_train)/len(self.X)*100:.1f}%)")
        print(f"  • Validation: {len(self.X_val)} samples ({len(self.X_val)/len(self.X)*100:.1f}%)")
        print(f"  • Testing: {len(self.X_test)} samples ({len(self.X_test)/len(self.X)*100:.1f}%)")
        
        print(f"\nModel Performance Results:")
        print("-" * 60)
        for _, row in results_df.iterrows():
            print(f"  {row['Model']:20} | R²: {row['R² Score']:.4f} | MAE: ${row['MAE']:6.1f} | RMSE: ${row['RMSE']:6.1f}")
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   R² Score: {results_df.iloc[0]['R² Score']:.4f}")
        print(f"   This model explains {results_df.iloc[0]['R² Score']*100:.1f}% of the price variance")
        
        print(f"\nModel Rankings:")
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            print(f"  {i}. {row['Model']} (R² = {row['R² Score']:.4f})")
        
        print("\n" + "="*60)
    
    def run_complete_analysis(self, save_plots=True):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Complete Sneaker Price Prediction Analysis")
        print("="*60)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Split data
        self.split_data()
        
        # Step 3: Initialize models
        self.initialize_models()
        
        # Step 4: Train models
        self.train_all_models()
        
        # Step 5: Evaluate models
        results_df = self.evaluate_models()
        
        # Step 6: Generate visualizations
        if save_plots:
            self.plot_model_comparison('model_comparison.png')
            self.plot_predictions_vs_actual('predictions_vs_actual.png')
            self.plot_feature_importance('feature_importance.png')
        else:
            self.plot_model_comparison()
            self.plot_predictions_vs_actual()
            self.plot_feature_importance()
        
        # Step 7: Generate report
        self.generate_model_report()
        
        print("\n✅ Complete analysis finished!")
        return True

def main():
    """Main function to run the analysis"""
    # Update this path to your actual data file
    data_path = "media/Clean_Shoe_Data.csv"  # Adjust based on your file structure
    
    # Initialize analysis
    analysis = SneakerPriceAnalysis(data_path)
    
    # Run complete analysis
    success = analysis.run_complete_analysis(save_plots=True)
    
    if success:
        print("\n📊 Analysis completed successfully!")
        print("Check the generated plots and report above.")
    else:
        print("\n❌ Analysis failed. Please check your data file path.")

if __name__ == "__main__":
    main()

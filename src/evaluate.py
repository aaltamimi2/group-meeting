import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import List
from src.utils import setup_logging, load_config, ensure_directory
from src.data_download import PubChemDownloader
from src.featurize import MolecularFeaturizer

logger = setup_logging(__name__)

# Set style
sns.set_style('whitegrid')

class ModelEvaluator:
    """Evaluate trained models on new molecules."""

    def __init__(self, config: dict):
        self.config = config
        self.model_dir = config['output']['model_dir']
        self.reports_dir = config['output']['reports_dir']

        # Load all available models
        self.models = self.load_all_models()

        # Initialize downloader and featurizer
        self.downloader = PubChemDownloader(config)
        self.featurizer = MolecularFeaturizer(config)

    def load_all_models(self):
        """Load all trained models from models directory."""
        models = {}
        model_files = {
            'Ridge': 'ridge_model.pkl',
            'Lasso': 'lasso_model.pkl',
            'ElasticNet': 'elasticnet_model.pkl',
            'Decision Tree': 'decision_tree_model.pkl',
            'KNN': 'knn_model.pkl',
            'Random Forest': 'random_forest_model.pkl'
        }

        for name, filename in model_files.items():
            model_path = os.path.join(self.model_dir, filename)
            if os.path.exists(model_path):
                models[name] = joblib.load(model_path)
                logger.info(f"Loaded {name} model")
            else:
                logger.warning(f"Model not found: {model_path}")

        return models

    def compute_ground_truth(self, df: pd.DataFrame) -> np.ndarray:
        """Compute ground truth target using same formula as training."""
        target = (
            0.3 * df['MW'].fillna(0) / 500 +
            0.2 * df['LogP'].fillna(0) +
            0.25 * df['TPSA'].fillna(0) / 100 +
            0.15 * df['NumHDonors'].fillna(0) +
            0.1 * df['NumHAcceptors'].fillna(0)
        )
        return target.values

    def predict_molecules(self, molecule_names: List[str]) -> pd.DataFrame:
        """Run full pipeline: download -> featurize -> predict."""
        logger.info(f"Processing {len(molecule_names)} molecules for evaluation...")

        # Download data
        df = self.downloader.download_molecules(molecule_names)

        # Featurize
        df_features = self.featurizer.featurize_dataset(df)

        # Prepare features
        feature_cols = [col for col in df_features.columns if
                       col.startswith('morgan_') or
                       col.startswith('maccs_') or
                       col in ['MW', 'LogP', 'TPSA', 'NumHDonors',
                              'NumHAcceptors', 'NumRotatableBonds',
                              'NumAromaticRings', 'NumAliphaticRings']]

        X = df_features[feature_cols].fillna(0)

        # Compute ground truth
        y_true = self.compute_ground_truth(df_features)

        # Make predictions with all models
        results = df_features[['name', 'cid', 'smiles', 'molecular_weight']].copy()
        results['true_value'] = y_true

        for model_name, model in self.models.items():
            try:
                predictions = model.predict(X)
                results[f'{model_name}_pred'] = predictions
                logger.info(f"  ✓ {model_name} predictions complete")
            except Exception as e:
                logger.error(f"  ✗ {model_name} failed: {e}")
                results[f'{model_name}_pred'] = np.nan

        logger.info("All predictions complete")
        return results

    def create_evaluation_visualizations(self, results: pd.DataFrame):
        """Create comprehensive evaluation visualizations."""
        logger.info("Creating evaluation visualizations...")

        # Get prediction columns
        pred_cols = [col for col in results.columns if col.endswith('_pred')]
        model_names = [col.replace('_pred', '') for col in pred_cols]

        # Remove failed models (all NaN)
        valid_models = [name for name, col in zip(model_names, pred_cols)
                       if not results[col].isna().all()]

        if not valid_models:
            logger.error("No valid model predictions found!")
            return

        logger.info(f"Creating visualizations for {len(valid_models)} models")

        # 1. Prediction vs Ground Truth (all models)
        self._plot_predictions_vs_truth(results, valid_models)

        # 2. Prediction Error Distribution
        self._plot_error_distribution(results, valid_models)

        # 3. Model Performance Comparison
        self._plot_model_performance_comparison(results, valid_models)

        # 4. Prediction Correlation Heatmap
        self._plot_prediction_correlations(results, valid_models)

        # 5. Individual Model Scatter Plots (best 4 models)
        self._plot_top_model_scatter(results, valid_models)

        logger.info("Evaluation visualizations complete")

    def _plot_predictions_vs_truth(self, results: pd.DataFrame, model_names: List[str]):
        """Plot predicted vs ground truth for all models."""
        n_models = len(model_names)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]

        y_true = results['true_value']

        for idx, model_name in enumerate(model_names):
            y_pred = results[f'{model_name}_pred']
            valid_mask = ~(y_pred.isna() | y_true.isna())

            if valid_mask.sum() == 0:
                continue

            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]

            # Scatter plot
            axes[idx].scatter(y_true_valid, y_pred_valid, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)

            # Perfect prediction line
            min_val = min(y_true_valid.min(), y_pred_valid.min())
            max_val = max(y_true_valid.max(), y_pred_valid.max())
            axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

            # Calculate R²
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_true_valid, y_pred_valid)
            mse = mean_squared_error(y_true_valid, y_pred_valid)

            axes[idx].set_xlabel('Ground Truth', fontsize=10)
            axes[idx].set_ylabel('Predicted', fontsize=10)
            axes[idx].set_title(f'{model_name}\nR² = {r2:.3f}, MSE = {mse:.4f}', fontsize=11)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(model_names), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, 'evaluation_pred_vs_truth.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved prediction vs truth plot to {plot_path}")

    def _plot_error_distribution(self, results: pd.DataFrame, model_names: List[str]):
        """Plot error distribution for each model."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        y_true = results['true_value']
        errors = {}

        for model_name in model_names:
            y_pred = results[f'{model_name}_pred']
            valid_mask = ~(y_pred.isna() | y_true.isna())
            if valid_mask.sum() > 0:
                errors[model_name] = (y_pred[valid_mask] - y_true[valid_mask]).values

        # Box plot
        if errors:
            axes[0].boxplot(errors.values(), labels=errors.keys(), vert=True)
            axes[0].set_ylabel('Prediction Error', fontsize=11)
            axes[0].set_title('Prediction Error Distribution by Model', fontsize=12, fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')
            axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            axes[0].legend()

            # Histogram of absolute errors
            for model_name, error in errors.items():
                axes[1].hist(np.abs(error), alpha=0.5, bins=20, label=model_name, edgecolor='black')

            axes[1].set_xlabel('Absolute Prediction Error', fontsize=11)
            axes[1].set_ylabel('Frequency', fontsize=11)
            axes[1].set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, 'evaluation_error_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved error distribution plot to {plot_path}")

    def _plot_model_performance_comparison(self, results: pd.DataFrame, model_names: List[str]):
        """Compare model performance on evaluation set."""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        y_true = results['true_value']
        metrics = []

        for model_name in model_names:
            y_pred = results[f'{model_name}_pred']
            valid_mask = ~(y_pred.isna() | y_true.isna())

            if valid_mask.sum() > 0:
                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask]

                metrics.append({
                    'Model': model_name,
                    'R²': r2_score(y_true_valid, y_pred_valid),
                    'MSE': mean_squared_error(y_true_valid, y_pred_valid),
                    'MAE': mean_absolute_error(y_true_valid, y_pred_valid)
                })

        metrics_df = pd.DataFrame(metrics).sort_values('R²', ascending=False)

        # Create bar plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # R² Score
        axes[0].barh(metrics_df['Model'], metrics_df['R²'], color='forestgreen', edgecolor='black')
        axes[0].set_xlabel('R² Score', fontsize=11)
        axes[0].set_title('Model R² on Evaluation Set', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')

        # MSE
        axes[1].barh(metrics_df['Model'], metrics_df['MSE'], color='steelblue', edgecolor='black')
        axes[1].set_xlabel('MSE', fontsize=11)
        axes[1].set_title('Model MSE on Evaluation Set', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        # MAE
        axes[2].barh(metrics_df['Model'], metrics_df['MAE'], color='coral', edgecolor='black')
        axes[2].set_xlabel('MAE', fontsize=11)
        axes[2].set_title('Model MAE on Evaluation Set', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, 'evaluation_model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved model comparison plot to {plot_path}")

        # Save metrics to CSV
        metrics_path = os.path.join(self.reports_dir, 'evaluation_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved evaluation metrics to {metrics_path}")

        return metrics_df

    def _plot_prediction_correlations(self, results: pd.DataFrame, model_names: List[str]):
        """Plot correlation heatmap between model predictions."""
        pred_cols = [f'{name}_pred' for name in model_names]
        pred_data = results[pred_cols + ['true_value']].dropna()

        if len(pred_data) == 0:
            logger.warning("No valid data for correlation plot")
            return

        # Rename columns for display
        display_names = model_names + ['Ground Truth']
        pred_data.columns = display_names

        corr_matrix = pred_data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0.5, square=True, linewidths=1,
                   cbar_kws={'shrink': 0.8})
        plt.title('Prediction Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        plot_path = os.path.join(self.reports_dir, 'evaluation_prediction_correlations.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved prediction correlation plot to {plot_path}")

    def _plot_top_model_scatter(self, results: pd.DataFrame, model_names: List[str]):
        """Create detailed scatter plots for top 4 models."""
        from sklearn.metrics import r2_score

        y_true = results['true_value']

        # Calculate R² for each model
        model_scores = []
        for model_name in model_names:
            y_pred = results[f'{model_name}_pred']
            valid_mask = ~(y_pred.isna() | y_true.isna())
            if valid_mask.sum() > 0:
                r2 = r2_score(y_true[valid_mask], y_pred[valid_mask])
                model_scores.append((model_name, r2))

        # Sort and take top 4
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = [name for name, _ in model_scores[:4]]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for idx, model_name in enumerate(top_models):
            y_pred = results[f'{model_name}_pred']
            valid_mask = ~(y_pred.isna() | y_true.isna())

            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]

            # Scatter with density coloring
            axes[idx].scatter(y_true_valid, y_pred_valid, alpha=0.6, s=50,
                            c=range(len(y_true_valid)), cmap='viridis',
                            edgecolors='black', linewidth=0.5)

            # Perfect prediction line
            min_val = min(y_true_valid.min(), y_pred_valid.min())
            max_val = max(y_true_valid.max(), y_pred_valid.max())
            axes[idx].plot([min_val, max_val], [min_val, max_val],
                          'r--', linewidth=3, label='Perfect Prediction')

            # Calculate metrics
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            r2 = r2_score(y_true_valid, y_pred_valid)
            mse = mean_squared_error(y_true_valid, y_pred_valid)
            mae = mean_absolute_error(y_true_valid, y_pred_valid)

            axes[idx].set_xlabel('Ground Truth', fontsize=12)
            axes[idx].set_ylabel('Predicted Value', fontsize=12)
            axes[idx].set_title(f'{model_name}\nR²={r2:.3f}, MSE={mse:.4f}, MAE={mae:.4f}',
                              fontsize=13, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, 'evaluation_top_models_detailed.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved top models detailed plot to {plot_path}")

    def generate_report(self, results: pd.DataFrame, metrics_df: pd.DataFrame):
        """Generate comprehensive evaluation report."""
        report_path = os.path.join(self.reports_dir, 'evaluation_report.md')

        with open(report_path, 'w') as f:
            f.write("# Model Evaluation Report - New Molecules\n\n")
            f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Evaluation Summary\n\n")
            f.write(f"- Total molecules evaluated: **{len(results)}**\n")
            f.write(f"- Successfully processed: **{results['true_value'].notna().sum()}**\n")
            f.write(f"- Models evaluated: **{len(metrics_df)}**\n\n")

            f.write("## Model Performance Rankings\n\n")
            f.write("Models ranked by R² score:\n\n")
            f.write("| Rank | Model | R² | MSE | MAE |\n")
            f.write("|------|-------|-----|-----|-----|\n")

            for rank, (_, row) in enumerate(metrics_df.iterrows(), 1):
                f.write(f"| {rank} | {row['Model']} | {row['R²']:.4f} | {row['MSE']:.4f} | {row['MAE']:.4f} |\n")

            f.write("\n## Best Model\n\n")
            best_model = metrics_df.iloc[0]
            f.write(f"🏆 **{best_model['Model']}**\n\n")
            f.write(f"- R² Score: **{best_model['R²']:.4f}** (explains {best_model['R²']*100:.1f}% of variance)\n")
            f.write(f"- MSE: **{best_model['MSE']:.4f}**\n")
            f.write(f"- MAE: **{best_model['MAE']:.4f}**\n\n")

            f.write("## Visualizations\n\n")
            f.write("### 1. Prediction vs Ground Truth\n")
            f.write("![Pred vs Truth](evaluation_pred_vs_truth.png)\n\n")

            f.write("### 2. Error Distribution\n")
            f.write("![Error Distribution](evaluation_error_distribution.png)\n\n")

            f.write("### 3. Model Comparison\n")
            f.write("![Model Comparison](evaluation_model_comparison.png)\n\n")

            f.write("### 4. Prediction Correlations\n")
            f.write("![Correlations](evaluation_prediction_correlations.png)\n\n")

            f.write("### 5. Top Models Detailed\n")
            f.write("![Top Models](evaluation_top_models_detailed.png)\n\n")

        logger.info(f"Saved evaluation report to {report_path}")

def main():
    """Main function to evaluate models."""
    config = load_config()

    # Get evaluation molecules from config
    eval_molecules = config['evaluation']['molecules']
    logger.info(f"Evaluating models on {len(eval_molecules)} new molecules")

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    # Make predictions
    results = evaluator.predict_molecules(eval_molecules)

    # Save results
    ensure_directory(config['output']['reports_dir'])
    results_path = os.path.join(config['output']['reports_dir'], 'evaluation_results.csv')
    results.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")

    # Get prediction columns
    pred_cols = [col for col in results.columns if col.endswith('_pred')]
    model_names = [col.replace('_pred', '') for col in pred_cols]
    valid_models = [name for name, col in zip(model_names, pred_cols)
                   if not results[col].isna().all()]

    # Generate visualizations
    evaluator.create_evaluation_visualizations(results)

    # Calculate and display metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y_true = results['true_value']

    metrics = []
    for model_name in valid_models:
        y_pred = results[f'{model_name}_pred']
        valid_mask = ~(y_pred.isna() | y_true.isna())

        if valid_mask.sum() > 0:
            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]

            metrics.append({
                'Model': model_name,
                'R²': r2_score(y_true_valid, y_pred_valid),
                'MSE': mean_squared_error(y_true_valid, y_pred_valid),
                'MAE': mean_absolute_error(y_true_valid, y_pred_valid)
            })

    metrics_df = pd.DataFrame(metrics).sort_values('R²', ascending=False)

    # Generate report
    evaluator.generate_report(results, metrics_df)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"\n{metrics_df.to_string(index=False)}\n")
    logger.info(f"{'='*60}")

    return results, metrics_df

if __name__ == "__main__":
    main()

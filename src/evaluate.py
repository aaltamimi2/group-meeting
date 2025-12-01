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
    """Evaluate trained models on new molecules for lipophilicity prediction."""

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

    def predict_molecules(self, molecule_names: List[str]) -> pd.DataFrame:
        """Run full pipeline: download -> featurize -> predict lipophilicity."""
        logger.info(f"Processing {len(molecule_names)} molecules for lipophilicity prediction...")

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

        # Make predictions with all models
        results = df_features[['name', 'cid', 'smiles', 'molecular_weight']].copy()

        for model_name, model in self.models.items():
            try:
                predictions = model.predict(X)
                results[f'{model_name}_pred'] = predictions
                logger.info(f"  ✓ {model_name} predictions complete")
            except Exception as e:
                logger.error(f"  ✗ {model_name} failed: {e}")
                results[f'{model_name}_pred'] = np.nan

        # Add consensus prediction (mean of all models)
        pred_cols = [col for col in results.columns if col.endswith('_pred')]
        results['consensus_pred'] = results[pred_cols].mean(axis=1)
        results['prediction_std'] = results[pred_cols].std(axis=1)

        logger.info("All predictions complete")
        return results

    def create_evaluation_visualizations(self, results: pd.DataFrame):
        """Create comprehensive visualizations for lipophilicity predictions."""
        logger.info("Creating lipophilicity prediction visualizations...")

        # Get prediction columns
        pred_cols = [col for col in results.columns if col.endswith('_pred') and col != 'consensus_pred']
        model_names = [col.replace('_pred', '') for col in pred_cols]

        # Remove failed models (all NaN)
        valid_models = [name for name, col in zip(model_names, pred_cols)
                       if not results[col].isna().all()]

        if not valid_models:
            logger.error("No valid model predictions found!")
            return

        logger.info(f"Creating visualizations for {len(valid_models)} models")

        # 1. Prediction distributions by model
        self._plot_prediction_distributions(results, valid_models)

        # 2. Prediction correlation heatmap (model agreement)
        self._plot_prediction_correlations(results, valid_models)

        # 3. Consensus predictions with uncertainty
        self._plot_consensus_predictions(results)

        # 4. Top/bottom molecules by predicted lipophilicity
        self._plot_top_molecules(results)

        logger.info("Evaluation visualizations complete")

    def _plot_prediction_distributions(self, results: pd.DataFrame, model_names: List[str]):
        """Plot distribution of predictions for each model."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, model_name in enumerate(model_names[:6]):
            predictions = results[f'{model_name}_pred'].dropna()

            # Histogram
            axes[idx].hist(predictions, bins=25, edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx].set_xlabel('Predicted LogD', fontsize=11)
            axes[idx].set_ylabel('Count', fontsize=11)
            axes[idx].set_title(f'{model_name}\nMean: {predictions.mean():.2f}, Std: {predictions.std():.2f}',
                              fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].axvline(predictions.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            axes[idx].legend()

        # Hide unused subplots
        for idx in range(len(model_names), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, 'evaluation_prediction_distributions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved prediction distributions to {plot_path}")

    def _plot_prediction_correlations(self, results: pd.DataFrame, model_names: List[str]):
        """Plot correlation heatmap between model predictions."""
        pred_cols = [f'{name}_pred' for name in model_names]
        pred_data = results[pred_cols].dropna()

        if len(pred_data) == 0:
            logger.warning("No valid data for correlation plot")
            return

        # Rename columns for display
        pred_data.columns = model_names

        corr_matrix = pred_data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0.5, square=True, linewidths=1,
                   cbar_kws={'shrink': 0.8}, vmin=0, vmax=1)
        plt.title('Model Prediction Correlation Matrix\n(Model Agreement)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        plot_path = os.path.join(self.reports_dir, 'evaluation_prediction_correlations.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved prediction correlation plot to {plot_path}")

    def _plot_consensus_predictions(self, results: pd.DataFrame):
        """Plot consensus predictions with uncertainty bands."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Sort by consensus prediction
        sorted_results = results.sort_values('consensus_pred')
        x = range(len(sorted_results))

        # Plot 1: Consensus with error bars
        axes[0].errorbar(x, sorted_results['consensus_pred'],
                        yerr=sorted_results['prediction_std'],
                        fmt='o', markersize=4, alpha=0.6, capsize=3, elinewidth=1)
        axes[0].set_xlabel('Molecule Index (sorted by LogD)', fontsize=11)
        axes[0].set_ylabel('Predicted LogD', fontsize=11)
        axes[0].set_title('Consensus Predictions with Uncertainty\n(error bars = std across models)',
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Prediction uncertainty distribution
        axes[1].hist(sorted_results['prediction_std'], bins=30, edgecolor='black', alpha=0.7, color='coral')
        axes[1].set_xlabel('Prediction Std Dev', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title(f'Prediction Uncertainty Distribution\nMean Uncertainty: {sorted_results["prediction_std"].mean():.3f}',
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(sorted_results['prediction_std'].mean(), color='red',
                       linestyle='--', linewidth=2, label='Mean')
        axes[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, 'evaluation_consensus_predictions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved consensus predictions plot to {plot_path}")

    def _plot_top_molecules(self, results: pd.DataFrame):
        """Plot top and bottom molecules by predicted lipophilicity."""
        # Sort by consensus prediction
        sorted_results = results.sort_values('consensus_pred')

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Top 20 most lipophilic
        top_20 = sorted_results.tail(20)
        axes[0].barh(range(len(top_20)), top_20['consensus_pred'],
                    xerr=top_20['prediction_std'], color='darkgreen',
                    alpha=0.7, edgecolor='black', capsize=3)
        axes[0].set_yticks(range(len(top_20)))
        axes[0].set_yticklabels(top_20['name'], fontsize=9)
        axes[0].set_xlabel('Predicted LogD', fontsize=11)
        axes[0].set_title('Top 20 Most Lipophilic Molecules (Highest LogD)',
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')

        # Bottom 20 least lipophilic
        bottom_20 = sorted_results.head(20)
        axes[1].barh(range(len(bottom_20)), bottom_20['consensus_pred'],
                    xerr=bottom_20['prediction_std'], color='darkblue',
                    alpha=0.7, edgecolor='black', capsize=3)
        axes[1].set_yticks(range(len(bottom_20)))
        axes[1].set_yticklabels(bottom_20['name'], fontsize=9)
        axes[1].set_xlabel('Predicted LogD', fontsize=11)
        axes[1].set_title('Top 20 Least Lipophilic Molecules (Lowest LogD)',
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, 'evaluation_top_molecules.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved top molecules plot to {plot_path}")

    def generate_report(self, results: pd.DataFrame):
        """Generate comprehensive evaluation report for lipophilicity predictions."""
        report_path = os.path.join(self.reports_dir, 'evaluation_report.md')

        # Get model names
        pred_cols = [col for col in results.columns if col.endswith('_pred') and col != 'consensus_pred']
        model_names = [col.replace('_pred', '') for col in pred_cols]

        with open(report_path, 'w') as f:
            f.write("# Lipophilicity Prediction Report - Evaluation Set\n\n")
            f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- Total molecules evaluated: **{len(results)}**\n")
            f.write(f"- Models used: **{len(model_names)}** ({', '.join(model_names)})\n")
            f.write(f"- Consensus mean LogD: **{results['consensus_pred'].mean():.2f} ± {results['consensus_pred'].std():.2f}**\n")
            f.write(f"- Average prediction uncertainty: **{results['prediction_std'].mean():.3f}**\n\n")

            f.write("## Prediction Statistics by Model\n\n")
            f.write("| Model | Mean LogD | Std Dev | Min | Max |\n")
            f.write("|-------|-----------|---------|-----|-----|\n")

            for model_name in model_names:
                preds = results[f'{model_name}_pred'].dropna()
                f.write(f"| {model_name} | {preds.mean():.2f} | {preds.std():.2f} | {preds.min():.2f} | {preds.max():.2f} |\n")

            f.write(f"\n## Top 10 Most Lipophilic Molecules\n\n")
            top_10 = results.nlargest(10, 'consensus_pred')
            f.write("| Rank | Molecule | Predicted LogD | Uncertainty |\n")
            f.write("|------|----------|----------------|-------------|\n")
            for rank, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"| {rank} | {row['name']} | {row['consensus_pred']:.2f} | ±{row['prediction_std']:.3f} |\n")

            f.write(f"\n## Top 10 Least Lipophilic Molecules\n\n")
            bottom_10 = results.nsmallest(10, 'consensus_pred')
            f.write("| Rank | Molecule | Predicted LogD | Uncertainty |\n")
            f.write("|------|----------|----------------|-------------|\n")
            for rank, (_, row) in enumerate(bottom_10.iterrows(), 1):
                f.write(f"| {rank} | {row['name']} | {row['consensus_pred']:.2f} | ±{row['prediction_std']:.3f} |\n")

            f.write("\n## Visualizations\n\n")
            f.write("### 1. Prediction Distributions\n")
            f.write("![Prediction Distributions](evaluation_prediction_distributions.png)\n\n")

            f.write("### 2. Model Agreement (Correlations)\n")
            f.write("![Model Correlations](evaluation_prediction_correlations.png)\n\n")

            f.write("### 3. Consensus Predictions\n")
            f.write("![Consensus](evaluation_consensus_predictions.png)\n\n")

            f.write("### 4. Top/Bottom Molecules\n")
            f.write("![Top Molecules](evaluation_top_molecules.png)\n\n")

            f.write("## Notes\n\n")
            f.write("- LogD is the octanol-water distribution coefficient at pH 7.4\n")
            f.write("- Higher LogD = more lipophilic (fat-soluble)\n")
            f.write("- Lower LogD = more hydrophilic (water-soluble)\n")
            f.write("- Typical drug LogD range: -0.4 to 5.6\n")
            f.write("- Consensus prediction is the mean across all models\n")
            f.write("- Uncertainty is the standard deviation across model predictions\n")

        logger.info(f"Saved evaluation report to {report_path}")

def main():
    """Main function to evaluate models."""
    config = load_config()

    # Get evaluation molecules from config
    eval_molecules = config['evaluation']['molecules']
    logger.info(f"Evaluating lipophilicity for {len(eval_molecules)} molecules")

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    # Make predictions
    results = evaluator.predict_molecules(eval_molecules)

    # Save results
    ensure_directory(config['output']['reports_dir'])
    results_path = os.path.join(config['output']['reports_dir'], 'evaluation_results.csv')
    results.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")

    # Generate visualizations
    evaluator.create_evaluation_visualizations(results)

    # Generate report
    evaluator.generate_report(results)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("LIPOPHILICITY PREDICTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Molecules processed: {len(results)}")
    logger.info(f"Mean predicted LogD: {results['consensus_pred'].mean():.2f}")
    logger.info(f"LogD range: [{results['consensus_pred'].min():.2f}, {results['consensus_pred'].max():.2f}]")
    logger.info(f"Average uncertainty: {results['prediction_std'].mean():.3f}")
    logger.info(f"{'='*60}")

    return results

if __name__ == "__main__":
    main()

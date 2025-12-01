import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from typing import List
from src.utils import setup_logging, load_config, ensure_directory
from src.data_download import PubChemDownloader
from src.featurize import MolecularFeaturizer

logger = setup_logging(__name__)

class ModelEvaluator:
    """Evaluate trained models on new molecules."""

    def __init__(self, config: dict):
        self.config = config
        self.model_dir = config['output']['model_dir']
        self.reports_dir = config['output']['reports_dir']

        # Load models
        self.ridge_model = self.load_model('ridge_model.pkl')
        self.rf_model = self.load_model('random_forest_model.pkl')

        # Initialize downloader and featurizer
        self.downloader = PubChemDownloader(config)
        self.featurizer = MolecularFeaturizer(config)

    def load_model(self, model_name: str):
        """Load a trained model."""
        model_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            return None
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model

    def predict_molecules(self, molecule_names: List[str]) -> pd.DataFrame:
        """Run full pipeline: download -> featurize -> predict."""
        logger.info(f"Processing {len(molecule_names)} molecules...")

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

        # Make predictions
        results = df_features[['name', 'cid', 'smiles', 'molecular_weight']].copy()

        if self.ridge_model is not None:
            results['ridge_prediction'] = self.ridge_model.predict(X)

        if self.rf_model is not None:
            results['rf_prediction'] = self.rf_model.predict(X)

        logger.info("Predictions complete")
        return results

    def plot_predictions(self, results: pd.DataFrame):
        """Plot prediction results."""
        if 'ridge_prediction' not in results.columns or 'rf_prediction' not in results.columns:
            logger.warning("Cannot plot: predictions not available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Ridge predictions
        axes[0].bar(range(len(results)), results['ridge_prediction'])
        axes[0].set_xlabel('Molecule Index')
        axes[0].set_ylabel('Predicted Value')
        axes[0].set_title('Ridge Model Predictions')
        axes[0].tick_params(axis='x', rotation=45)

        # Random Forest predictions
        axes[1].bar(range(len(results)), results['rf_prediction'])
        axes[1].set_xlabel('Molecule Index')
        axes[1].set_ylabel('Predicted Value')
        axes[1].set_title('Random Forest Model Predictions')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        plot_path = os.path.join(self.reports_dir, 'evaluation_predictions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved prediction plot to {plot_path}")

    def generate_report(self, results: pd.DataFrame):
        """Generate evaluation report."""
        report_path = os.path.join(self.reports_dir, 'model_performance.md')

        with open(report_path, 'w') as f:
            f.write("# Model Evaluation Report\n\n")
            f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Evaluation Molecules\n\n")
            f.write(f"Total molecules evaluated: {len(results)}\n\n")

            f.write("## Prediction Summary\n\n")

            if 'ridge_prediction' in results.columns:
                f.write("### Ridge Model\n")
                f.write(f"- Mean prediction: {results['ridge_prediction'].mean():.4f}\n")
                f.write(f"- Std prediction: {results['ridge_prediction'].std():.4f}\n")
                f.write(f"- Min prediction: {results['ridge_prediction'].min():.4f}\n")
                f.write(f"- Max prediction: {results['ridge_prediction'].max():.4f}\n\n")

            if 'rf_prediction' in results.columns:
                f.write("### Random Forest Model\n")
                f.write(f"- Mean prediction: {results['rf_prediction'].mean():.4f}\n")
                f.write(f"- Std prediction: {results['rf_prediction'].std():.4f}\n")
                f.write(f"- Min prediction: {results['rf_prediction'].min():.4f}\n")
                f.write(f"- Max prediction: {results['rf_prediction'].max():.4f}\n\n")

            f.write("## Detailed Results\n\n")
            f.write("| Molecule | CID | MW | Ridge Pred | RF Pred |\n")
            f.write("|----------|-----|-------|------------|----------|\n")

            for _, row in results.iterrows():
                ridge_pred = row.get('ridge_prediction', 'N/A')
                rf_pred = row.get('rf_prediction', 'N/A')

                if isinstance(ridge_pred, float):
                    ridge_pred = f"{ridge_pred:.4f}"
                if isinstance(rf_pred, float):
                    rf_pred = f"{rf_pred:.4f}"

                # Handle molecular_weight which might be string or float
                mw = row['molecular_weight']
                if pd.notna(mw):
                    try:
                        mw_str = f"{float(mw):.2f}"
                    except (ValueError, TypeError):
                        mw_str = str(mw)
                else:
                    mw_str = "N/A"

                f.write(f"| {row['name']} | {row['cid']} | {mw_str} | {ridge_pred} | {rf_pred} |\n")

            f.write("\n## Visualization\n\n")
            f.write("![Prediction Plot](evaluation_predictions.png)\n")

        logger.info(f"Saved evaluation report to {report_path}")

def main():
    """Main function to evaluate models."""
    config = load_config()

    # Get evaluation molecules from config
    eval_molecules = config['evaluation']['molecules']

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    # Make predictions
    results = evaluator.predict_molecules(eval_molecules)

    # Save results
    ensure_directory(config['output']['reports_dir'])
    results_path = os.path.join(config['output']['reports_dir'], 'evaluation_results.csv')
    results.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")

    # Generate plots
    evaluator.plot_predictions(results)

    # Generate report
    evaluator.generate_report(results)

    return results

if __name__ == "__main__":
    main()

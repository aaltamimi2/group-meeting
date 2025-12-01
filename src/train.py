import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from src.utils import setup_logging, load_config, ensure_directory

logger = setup_logging(__name__)

class ModelTrainer:
    """Train machine learning models for molecular property prediction."""

    def __init__(self, config: dict):
        self.config = config
        self.test_size = config['model']['test_size']
        self.random_state = config['model']['random_state']
        self.cv_folds = config['model']['cv_folds']
        self.model_dir = config['output']['model_dir']
        self.reports_dir = config['output']['reports_dir']

        ensure_directory(self.model_dir)
        ensure_directory(self.reports_dir)

    def create_synthetic_target(self, df: pd.DataFrame) -> np.ndarray:
        """Create synthetic regression target from molecular features."""
        logger.info("Creating synthetic target variable...")

        # Use a combination of physicochemical properties
        target = (
            0.3 * df['MW'].fillna(0) / 500 +
            0.2 * df['LogP'].fillna(0) +
            0.25 * df['TPSA'].fillna(0) / 100 +
            0.15 * df['NumHDonors'].fillna(0) +
            0.1 * df['NumHAcceptors'].fillna(0)
        )

        # Add some noise
        np.random.seed(self.random_state)
        noise = np.random.normal(0, 0.1, size=len(target))
        target = target + noise

        return target.values

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix."""
        # Select feature columns (morgan, maccs, descriptors)
        feature_cols = [col for col in df.columns if
                       col.startswith('morgan_') or
                       col.startswith('maccs_') or
                       col in ['MW', 'LogP', 'TPSA', 'NumHDonors',
                              'NumHAcceptors', 'NumRotatableBonds',
                              'NumAromaticRings', 'NumAliphaticRings']]

        X = df[feature_cols].fillna(0)
        logger.info(f"Prepared feature matrix with shape {X.shape}")
        return X

    def train_ridge_model(self, X_train, y_train, X_test, y_test):
        """Train Ridge regression model."""
        logger.info("Training Ridge regression model...")

        alpha = self.config['model']['ridge']['alpha']
        model = Ridge(alpha=alpha, random_state=self.random_state)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=self.cv_folds,
                                    scoring='neg_mean_squared_error')
        logger.info(f"Ridge CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Train on full training set
        model.fit(X_train, y_train)

        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }

        logger.info(f"Ridge Test MSE: {metrics['test_mse']:.4f}, R2: {metrics['test_r2']:.4f}")

        # Save model
        model_path = os.path.join(self.model_dir, 'ridge_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"Saved Ridge model to {model_path}")

        return model, metrics

    def train_random_forest_model(self, X_train, y_train, X_test, y_test):
        """Train Random Forest regression model."""
        logger.info("Training Random Forest model...")

        n_estimators = self.config['model']['random_forest']['n_estimators']
        max_depth = self.config['model']['random_forest']['max_depth']
        random_state = self.config['model']['random_forest']['random_state']

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=self.cv_folds,
                                    scoring='neg_mean_squared_error')
        logger.info(f"RandomForest CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Train on full training set
        model.fit(X_train, y_train)

        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }

        logger.info(f"RandomForest Test MSE: {metrics['test_mse']:.4f}, R2: {metrics['test_r2']:.4f}")

        # Save model
        model_path = os.path.join(self.model_dir, 'random_forest_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"Saved Random Forest model to {model_path}")

        # Plot feature importance
        self.plot_feature_importance(model, X_train.columns)

        return model, metrics

    def plot_feature_importance(self, model, feature_names, top_n=20):
        """Plot feature importance for tree-based models."""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()

        plot_path = os.path.join(self.reports_dir, 'feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved feature importance plot to {plot_path}")

def main():
    """Main function to train models."""
    config = load_config()

    # Load processed data
    data_path = config['output']['processed_data']
    logger.info(f"Loading processed data from {data_path}")
    df = pd.read_csv(data_path)

    # Initialize trainer
    trainer = ModelTrainer(config)

    # Create synthetic target
    y = trainer.create_synthetic_target(df)

    # Prepare features
    X = trainer.prepare_features(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=trainer.test_size, random_state=trainer.random_state
    )

    logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    # Train models
    ridge_model, ridge_metrics = trainer.train_ridge_model(X_train, y_train, X_test, y_test)
    rf_model, rf_metrics = trainer.train_random_forest_model(X_train, y_train, X_test, y_test)

    # Save metrics
    metrics_df = pd.DataFrame({
        'Model': ['Ridge', 'Random Forest'],
        'Train MSE': [ridge_metrics['train_mse'], rf_metrics['train_mse']],
        'Test MSE': [ridge_metrics['test_mse'], rf_metrics['test_mse']],
        'Train MAE': [ridge_metrics['train_mae'], rf_metrics['train_mae']],
        'Test MAE': [ridge_metrics['test_mae'], rf_metrics['test_mae']],
        'Train R2': [ridge_metrics['train_r2'], rf_metrics['train_r2']],
        'Test R2': [ridge_metrics['test_r2'], rf_metrics['test_r2']]
    })

    metrics_path = os.path.join(config['output']['reports_dir'], 'training_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved training metrics to {metrics_path}")

    return ridge_model, rf_model, metrics_df

if __name__ == "__main__":
    main()

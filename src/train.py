import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import time
from src.utils import setup_logging, load_config, ensure_directory
from src.lipophilicity_data import load_lipophilicity_dataset
from src.featurize import MolecularFeaturizer

# Set style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

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

    def train_model_generic(self, model, model_name, X_train, y_train, X_test, y_test):
        """Generic method to train and evaluate any model."""
        logger.info(f"Training {model_name} model...")

        start_time = time.time()

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=self.cv_folds,
                                    scoring='neg_mean_squared_error')
        cv_time = time.time() - start_time
        logger.info(f"{model_name} CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f}) [{cv_time:.2f}s]")

        # Train on full training set
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start

        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'cv_time': cv_time,
            'train_time': train_time
        }

        logger.info(f"{model_name} - Test MSE: {metrics['test_mse']:.4f}, R2: {metrics['test_r2']:.4f}, Time: {train_time:.2f}s")

        # Save model
        model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
        model_path = os.path.join(self.model_dir, model_filename)
        joblib.dump(model, model_path)
        logger.info(f"Saved {model_name} model to {model_path}")

        return model, metrics

    def visualize_dataset(self, df: pd.DataFrame, y: np.ndarray):
        """Create comprehensive visualizations of the training dataset."""
        logger.info("Creating dataset visualizations...")

        # Molecular descriptors to visualize
        descriptor_cols = ['MW', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                          'NumRotatableBonds', 'NumAromaticRings', 'NumAliphaticRings']

        # Filter to available columns
        available_descriptors = [col for col in descriptor_cols if col in df.columns]

        if not available_descriptors:
            logger.warning("No molecular descriptors found in dataset")
            return

        # 1. Molecular Property Distributions
        self._plot_property_distributions(df, available_descriptors)

        # 2. Correlation Heatmap
        self._plot_correlation_heatmap(df, available_descriptors)

        # 3. Target Variable Distribution
        self._plot_target_distribution(y)

        # 4. Property Relationships
        self._plot_property_relationships(df, available_descriptors, y)

        logger.info("Dataset visualizations complete")

    def _plot_property_distributions(self, df: pd.DataFrame, descriptors: list):
        """Plot distributions of molecular properties."""
        n_plots = len(descriptors)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for idx, col in enumerate(descriptors):
            if col in df.columns:
                data = df[col].dropna()
                axes[idx].hist(data, bins=20, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'{col} Distribution')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Count')
                axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(descriptors), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, 'dataset_property_distributions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved property distributions to {plot_path}")

    def _plot_correlation_heatmap(self, df: pd.DataFrame, descriptors: list):
        """Plot correlation heatmap of molecular descriptors."""
        available = [col for col in descriptors if col in df.columns]
        if len(available) < 2:
            return

        corr_matrix = df[available].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1,
                   cbar_kws={'shrink': 0.8})
        plt.title('Molecular Descriptor Correlation Matrix', fontsize=14, pad=20)
        plt.tight_layout()

        plot_path = os.path.join(self.reports_dir, 'dataset_correlation_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved correlation heatmap to {plot_path}")

    def _plot_target_distribution(self, y: np.ndarray):
        """Plot distribution of target variable."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        axes[0].hist(y, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_title('Lipophilicity (LogD) Distribution')
        axes[0].set_xlabel('LogD at pH 7.4')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(y, vert=True)
        axes[1].set_title('Lipophilicity (LogD) Box Plot')
        axes[1].set_ylabel('LogD at pH 7.4')
        axes[1].grid(True, alpha=0.3)

        # Add statistics
        stats_text = f'Mean: {y.mean():.3f}\nStd: {y.std():.3f}\nMin: {y.min():.3f}\nMax: {y.max():.3f}'
        axes[1].text(1.15, y.mean(), stats_text, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, 'dataset_target_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved target distribution to {plot_path}")

    def _plot_property_relationships(self, df: pd.DataFrame, descriptors: list, y: np.ndarray):
        """Plot relationships between key molecular properties and target."""
        # Select key properties for scatter plots
        key_props = ['MW', 'LogP', 'TPSA', 'NumHDonors']
        available = [col for col in key_props if col in df.columns]

        if not available:
            return

        n_plots = len(available)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for idx, col in enumerate(available):
            axes[idx].scatter(df[col], y, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            axes[idx].set_xlabel(col, fontsize=11)
            axes[idx].set_ylabel('LogD at pH 7.4', fontsize=11)
            axes[idx].set_title(f'LogD vs {col}', fontsize=12)
            axes[idx].grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(df[col].fillna(0), y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df[col].min(), df[col].max(), 100)
            axes[idx].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, 'dataset_property_relationships.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved property relationships to {plot_path}")

def create_model_comparison_plots(metrics_df: pd.DataFrame, reports_dir: str):
    """Create comprehensive model comparison visualizations."""

    # 1. Test MSE Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Test MSE
    axes[0, 0].barh(metrics_df['Model'], metrics_df['Test MSE'], color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Test MSE', fontsize=11)
    axes[0, 0].set_title('Model Comparison: Test MSE (lower is better)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # Test R² Score
    axes[0, 1].barh(metrics_df['Model'], metrics_df['Test R2'], color='forestgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Test R² Score', fontsize=11)
    axes[0, 1].set_title('Model Comparison: Test R² (higher is better)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # Training Time
    axes[1, 0].barh(metrics_df['Model'], metrics_df['Train Time (s)'], color='coral', edgecolor='black')
    axes[1, 0].set_xlabel('Training Time (seconds)', fontsize=11)
    axes[1, 0].set_title('Model Comparison: Training Time', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Test MAE
    axes[1, 1].barh(metrics_df['Model'], metrics_df['Test MAE'], color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Test MAE', fontsize=11)
    axes[1, 1].set_title('Model Comparison: Test MAE (lower is better)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plot_path = os.path.join(reports_dir, 'model_comparison_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved model comparison plot to {plot_path}")

    # 2. Train vs Test Performance
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(metrics_df))
    width = 0.35

    ax.bar(x - width/2, metrics_df['Train MSE'], width, label='Train MSE', color='lightblue', edgecolor='black')
    ax.bar(x + width/2, metrics_df['Test MSE'], width, label='Test MSE', color='steelblue', edgecolor='black')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Train vs Test MSE by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(reports_dir, 'model_train_vs_test.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved train vs test comparison to {plot_path}")

    # 3. Performance vs Speed Trade-off
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    scatter = ax.scatter(metrics_df['Train Time (s)'], metrics_df['Test R2'],
                        s=200, c=range(len(metrics_df)), cmap='viridis',
                        edgecolors='black', linewidth=2, alpha=0.7)

    # Annotate points
    for idx, row in metrics_df.iterrows():
        ax.annotate(row['Model'], (row['Train Time (s)'], row['Test R2']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('Test R² Score', fontsize=12)
    ax.set_title('Model Performance vs Training Speed', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(reports_dir, 'model_performance_vs_speed.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved performance vs speed plot to {plot_path}")

def main():
    """Main function to train models."""
    config = load_config()

    # Load lipophilicity dataset
    logger.info("="*60)
    logger.info("LOADING LIPOPHILICITY DATASET (4,200 molecules)")
    logger.info("="*60)
    mol_df, y = load_lipophilicity_dataset()

    # Featurize the molecules
    pkl_cache_path = 'data/processed/lipophilicity_features.pkl'

    if os.path.exists(pkl_cache_path):
        logger.info(f"\nLoading cached features from {pkl_cache_path}")
        df = joblib.load(pkl_cache_path)
        logger.info(f"Loaded {len(df)} featurized molecules from cache")
    else:
        logger.info("\nFeaturizing molecules (this may take a few minutes)...")
        featurizer = MolecularFeaturizer(config)
        df = featurizer.featurize_dataset(mol_df)

        # Save featurized data for faster loading
        ensure_directory('data/processed')
        joblib.dump(df, pkl_cache_path)
        logger.info(f"Saved featurized data to {pkl_cache_path}")

    # Initialize trainer
    trainer = ModelTrainer(config)

    # Visualize dataset before training
    trainer.visualize_dataset(df, y)

    # Prepare features
    X = trainer.prepare_features(df)

    # Check for cached train/test split
    cache_path = os.path.join('data/processed', 'train_test_cache.pkl')
    if os.path.exists(cache_path):
        logger.info("Loading cached train/test split (for consistent model comparison)...")
        cache_data = joblib.load(cache_path)
        X_train, X_test, y_train, y_test = cache_data['X_train'], cache_data['X_test'], cache_data['y_train'], cache_data['y_test']
        logger.info(f"Loaded cached split: {len(X_train)} train, {len(X_test)} test")
    else:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=trainer.test_size, random_state=trainer.random_state
        )
        # Cache the split for consistent comparisons
        cache_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        joblib.dump(cache_data, cache_path)
        logger.info(f"Created new train/test split and cached to {cache_path}")

    logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    logger.info(f"Feature dimensions: {X_train.shape[1]} features")

    # Define all models to train
    models = [
        (Ridge(alpha=1.0, random_state=trainer.random_state), 'Ridge'),
        (Lasso(alpha=0.1, random_state=trainer.random_state, max_iter=5000), 'Lasso'),
        (ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=trainer.random_state, max_iter=5000), 'ElasticNet'),
        (DecisionTreeRegressor(max_depth=10, random_state=trainer.random_state), 'Decision Tree'),
        (KNeighborsRegressor(n_neighbors=5, n_jobs=-1), 'KNN'),
        (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=trainer.random_state, n_jobs=-1), 'Random Forest'),
    ]

    # Train all models and collect metrics
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {len(models)} models...")
    logger.info(f"{'='*60}\n")

    all_models = {}
    all_metrics = {}

    for model, name in models:
        trained_model, metrics = trainer.train_model_generic(model, name, X_train, y_train, X_test, y_test)
        all_models[name] = trained_model
        all_metrics[name] = metrics

        # Plot feature importance for tree-based models
        if name in ['Decision Tree', 'Random Forest'] and hasattr(trained_model, 'feature_importances_'):
            plot_name = name.lower().replace(' ', '_')
            trainer.plot_feature_importance(trained_model, X_train.columns, top_n=20)
            # Rename to include model name
            old_path = os.path.join(trainer.reports_dir, 'feature_importance.png')
            new_path = os.path.join(trainer.reports_dir, f'feature_importance_{plot_name}.png')
            if os.path.exists(old_path):
                os.rename(old_path, new_path)

    # Create comparison metrics DataFrame
    metrics_data = []
    for name in all_metrics:
        metrics = all_metrics[name]
        metrics_data.append({
            'Model': name,
            'Train MSE': metrics['train_mse'],
            'Test MSE': metrics['test_mse'],
            'Train MAE': metrics['train_mae'],
            'Test MAE': metrics['test_mae'],
            'Train R2': metrics['train_r2'],
            'Test R2': metrics['test_r2'],
            'CV Time (s)': metrics['cv_time'],
            'Train Time (s)': metrics['train_time']
        })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.sort_values('Test MSE')  # Sort by test error

    # Save metrics
    metrics_path = os.path.join(config['output']['reports_dir'], 'training_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"\nSaved training metrics to {metrics_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("MODEL PERFORMANCE SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"\n{metrics_df.to_string(index=False)}\n")
    logger.info(f"{'='*60}")

    # Create model comparison visualizations
    logger.info("\nCreating model comparison visualizations...")
    create_model_comparison_plots(metrics_df, trainer.reports_dir)

    return all_models, metrics_df

if __name__ == "__main__":
    main()

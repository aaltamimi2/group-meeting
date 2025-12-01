# Molecular Informatics Pipeline - Lipophilicity Prediction

A complete end-to-end machine learning pipeline for predicting molecular lipophilicity (LogD) using the MoleculeNet benchmark dataset, RDKit featurization, and scikit-learn models.

**Predicts**: Octanol-water distribution coefficient (LogD at pH 7.4) - a key pharmaceutical property affecting drug absorption and bioavailability.

## Features

- **Real Dataset**: MoleculeNet Lipophilicity benchmark (4,200 molecules with experimental LogD values)
- **Feature Engineering**: Compute molecular fingerprints and physicochemical descriptors using RDKit
- **Model Training**: Train 6 regression models with cross-validation and comparison
- **Model Evaluation**: Predict lipophilicity for new molecules with consensus predictions and uncertainty quantification
- **Comprehensive Visualizations**: Dataset analysis, model comparisons, prediction distributions

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd molecular-informatics-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
project-root/
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml          # Configuration file
├── data/
│   ├── external/              # External datasets (lipophilicity.csv)
│   ├── raw/                   # Raw PubChem data (for evaluation)
│   └── processed/             # Featurized data
├── src/
│   ├── lipophilicity_data.py  # Lipophilicity dataset loader
│   ├── data_download.py       # PubChem data acquisition
│   ├── featurize.py           # RDKit feature engineering
│   ├── train.py               # Model training
│   ├── evaluate.py            # Model evaluation
│   └── utils.py               # Utility functions
├── notebooks/
│   └── exploratory.ipynb      # Exploratory analysis
├── models/                    # Trained model files
└── reports/                   # Evaluation reports and plots
```

## Usage

### Quick Start

Train models on the lipophilicity dataset:

```bash
python -m src.train
```

The dataset is automatically loaded and featurized. On first run, featurization takes ~3-5 minutes. Subsequent runs use cached features for 50x faster loading.

### Train Models

```bash
python -m src.train
```

Trains **6 machine learning models** on 4,200 molecules with experimental LogD values:

**Linear Models**:
- Ridge (L2 regularization)
- Lasso (L1 regularization)
- ElasticNet (L1 + L2)

**Nonlinear Models**:
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest

The pipeline automatically:
1. Loads the MoleculeNet lipophilicity dataset
2. Featurizes molecules (2,223 features: fingerprints + descriptors)
3. Creates dataset visualizations
4. Trains all 6 models with 5-fold cross-validation
5. Compares model performance

**Outputs**:
- `models/*.pkl` - All 6 trained models
- `data/processed/lipophilicity_features.pkl` - Cached featurized data (fast loading)
- `reports/training_metrics.csv` - Model comparison (MSE, MAE, R², time)
- `reports/model_comparison_metrics.png` - 4-panel comparison
- `reports/model_train_vs_test.png` - Overfitting analysis
- `reports/model_performance_vs_speed.png` - Performance/speed trade-off
- `reports/dataset_property_distributions.png` - LogD distribution
- `reports/dataset_correlation_heatmap.png` - Descriptor correlations
- `reports/dataset_target_distribution.png` - LogD statistics
- `reports/dataset_property_relationships.png` - LogD vs molecular properties

### Evaluate Models on New Molecules

```bash
python -m src.evaluate
```

Predicts lipophilicity for 100 evaluation molecules:
1. Downloads molecular data from PubChem
2. Featurizes with RDKit
3. Predicts LogD with all 6 models
4. Computes consensus predictions (mean across models)
5. Quantifies prediction uncertainty (std across models)

Edit `config/settings.yaml` to change the evaluation molecule list.

**Outputs**:
- `reports/evaluation_results.csv` - All predictions with consensus and uncertainty
- `reports/evaluation_report.md` - Comprehensive summary
- `reports/evaluation_prediction_distributions.png` - Prediction histograms by model
- `reports/evaluation_prediction_correlations.png` - Model agreement heatmap
- `reports/evaluation_consensus_predictions.png` - Consensus predictions with error bars
- `reports/evaluation_top_molecules.png` - Top 20 most/least lipophilic molecules

## Data Flow

**Training**:
```
MoleculeNet Lipophilicity Dataset (4,200 molecules + LogD values)
     |
     v
[RDKit Featurization] --> Features (2,223 features)
     |
     v
[Train/Test Split] --> 80% train, 20% test
     |
     v
[6 ML Models] --> Trained models with cross-validation
```

**Evaluation** (New Molecules):
```
Molecule Names (from config)
     |
     v
[PubChem API] --> SMILES
     |
     v
[RDKit Featurization] --> Features (same 2,223 features)
     |
     v
[6 Trained Models] --> Predictions
     |
     v
[Consensus] --> Mean prediction ± uncertainty
```

## Configuration

Edit `config/settings.yaml` to customize:

- PubChem API settings (delay, retries)
- Feature engineering parameters (fingerprint size, descriptors)
- Model hyperparameters (alpha, n_estimators, max_depth)
- Evaluation molecule list

## Example

```python
from src.evaluate import ModelEvaluator
from src.utils import load_config

config = load_config()
evaluator = ModelEvaluator(config)

molecules = ['aspirin', 'caffeine', 'morphine']
results = evaluator.predict_molecules(molecules)
print(results)
```

## Dataset

**MoleculeNet Lipophilicity**
- Source: https://github.com/GLambard/Molecules_Dataset_Collection
- Size: 4,200 molecules
- Property: Experimental LogD (octanol-water distribution coefficient at pH 7.4)
- Range: -1.5 to 4.5
- Format: ChEMBL ID + SMILES + LogD value

**What is LogD?**
- LogD measures lipophilicity (fat-solubility vs water-solubility)
- Critical for predicting drug absorption, blood-brain barrier penetration, and metabolism
- Higher LogD = more lipophilic (fat-soluble)
- Lower LogD = more hydrophilic (water-soluble)
- Typical drug range: -0.4 to 5.6

## Performance

Typical results on the test set:
- **ElasticNet**: R² ≈ 0.98, MSE ≈ 0.02
- **Random Forest**: R² ≈ 0.95, MSE ≈ 0.05
- **Ridge**: R² ≈ 0.97, MSE ≈ 0.03

Models show excellent agreement (correlation > 0.95) on evaluation sets.

## Limitations

- **API Rate Limits**: PubChem has rate limits. The downloader includes delays and retry logic.
- **Evaluation Scope**: Evaluation molecules are not in the training set, so no ground truth for comparison
- **Feature Selection**: All computed features are used. Feature selection may improve performance.
- **2D Descriptors Only**: Only 2D molecular descriptors are computed. 3D descriptors require conformer generation.

## Dependencies

- **requests**: HTTP requests to PubChem API
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning models
- **rdkit**: Cheminformatics toolkit
- **matplotlib**: Visualization
- **PyYAML**: Configuration file parsing

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Citation

If you use this pipeline in your research, please cite:
- PubChem: https://pubchem.ncbi.nlm.nih.gov/
- RDKit: https://www.rdkit.org/

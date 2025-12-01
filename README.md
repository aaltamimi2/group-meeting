# Molecular Informatics Pipeline

A complete end-to-end machine learning pipeline for molecular property prediction using PubChem data, RDKit featurization, and scikit-learn models.

## Features

- **Data Acquisition**: Download molecular data from PubChem REST API
- **Feature Engineering**: Compute molecular fingerprints and physicochemical descriptors using RDKit
- **Model Training**: Train Ridge and Random Forest regression models with cross-validation
- **Model Evaluation**: Evaluate models on new molecules with automated reporting

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
│   ├── raw/                   # Raw downloaded data
│   └── processed/             # Featurized data
├── src/
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

### 1. Download Molecular Data

```bash
python -m src.data_download
```

This downloads molecular data from PubChem for a predefined list of molecules and saves it to `data/raw/molecules.csv`.

**Output**:
- Canonical SMILES
- InChI
- Molecular weight
- XLogP3
- TPSA
- Hydrogen bond donor/acceptor counts

### 2. Featurize Molecules

```bash
python -m src.featurize
```

This generates molecular features:
- Morgan fingerprints (radius=2, 2048 bits)
- MACCS keys (167 bits)
- Physicochemical descriptors (MW, LogP, TPSA, etc.)

**Output**: `data/processed/processed.csv`

### 3. Train Models

```bash
python -m src.train
```

Trains two models:
- Ridge regression (linear model)
- Random Forest (nonlinear ensemble model)

Also creates comprehensive dataset visualizations before training.

**Outputs**:
- `models/ridge_model.pkl` - Trained Ridge model
- `models/random_forest_model.pkl` - Trained Random Forest model
- `reports/training_metrics.csv` - Model performance metrics
- `reports/feature_importance.png` - Top 20 important features
- `reports/dataset_property_distributions.png` - Molecular property histograms
- `reports/dataset_correlation_heatmap.png` - Descriptor correlation matrix
- `reports/dataset_target_distribution.png` - Target variable distribution
- `reports/dataset_property_relationships.png` - Property vs target scatter plots

### 4. Evaluate Models

```bash
python -m src.evaluate
```

Runs the full pipeline on new molecules:
1. Download from PubChem
2. Featurize
3. Predict with both models

**Outputs**:
- `reports/evaluation_results.csv`
- `reports/evaluation_predictions.png`
- `reports/model_performance.md`

## Data Flow

```
Molecule Names
     |
     v
[PubChem API] --> Raw Data (SMILES, InChI, Properties)
     |
     v
[RDKit Featurization] --> Features (Fingerprints + Descriptors)
     |
     v
[ML Training] --> Trained Models (Ridge + RandomForest)
     |
     v
[Prediction] --> Predicted Properties
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

## Limitations

- **Synthetic Target**: The current implementation uses a synthetic regression target derived from physicochemical properties. For real-world applications, replace with actual experimental data.
- **API Rate Limits**: PubChem has rate limits. The downloader includes delays and retry logic.
- **Feature Selection**: All computed features are used. Consider feature selection for large-scale applications.
- **Model Scope**: Models are trained on a small dataset. Performance may vary with larger datasets.
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

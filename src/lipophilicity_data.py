"""
Load and prepare lipophilicity dataset for training.

Dataset: MoleculeNet Lipophilicity
Source: https://github.com/GLambard/Molecules_Dataset_Collection
Size: 4,200 molecules
Property: Octanol-water distribution coefficient (logD at pH 7.4)
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from src.utils import setup_logging

logger = setup_logging(__name__)

class LipophilicityDataset:
    """Load and prepare the lipophilicity dataset."""

    def __init__(self, data_path='data/external/lipophilicity.csv'):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """Load the lipophilicity CSV file."""
        logger.info(f"Loading lipophilicity dataset from {self.data_path}")

        self.df = pd.read_csv(self.data_path)

        # Rename columns for consistency
        self.df = self.df.rename(columns={
            'CMPD_CHEMBLID': 'chembl_id',
            'exp': 'logD',
            'smiles': 'smiles'
        })

        logger.info(f"Loaded {len(self.df)} molecules from lipophilicity dataset")
        logger.info(f"LogD range: {self.df['logD'].min():.2f} to {self.df['logD'].max():.2f}")

        return self.df

    def validate_smiles(self):
        """Validate SMILES strings and remove invalid ones."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Validating SMILES strings...")

        valid_mask = []
        for idx, smiles in enumerate(self.df['smiles']):
            mol = Chem.MolFromSmiles(smiles)
            valid_mask.append(mol is not None)

        initial_count = len(self.df)
        self.df = self.df[valid_mask].reset_index(drop=True)
        invalid_count = initial_count - len(self.df)

        if invalid_count > 0:
            logger.warning(f"Removed {invalid_count} molecules with invalid SMILES")
        else:
            logger.info("All SMILES are valid")

        return self.df

    def get_smiles_and_targets(self):
        """Get SMILES and target values for training."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        smiles = self.df['smiles'].values
        targets = self.df['logD'].values

        logger.info(f"Prepared {len(smiles)} SMILES with logD targets")
        logger.info(f"Target statistics: mean={targets.mean():.2f}, std={targets.std():.2f}")

        return smiles, targets

    def create_molecule_dataframe(self):
        """Create a DataFrame suitable for featurization pipeline."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Create DataFrame with format expected by featurizer
        mol_df = pd.DataFrame({
            'name': self.df['chembl_id'],
            'cid': self.df['chembl_id'],  # Use ChEMBL ID as identifier
            'smiles': self.df['smiles'],
            'molecular_weight': np.nan  # Will be computed during featurization
        })

        logger.info(f"Created molecule DataFrame with {len(mol_df)} entries")

        return mol_df

    def get_statistics(self):
        """Get dataset statistics."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        stats = {
            'n_molecules': len(self.df),
            'logD_min': self.df['logD'].min(),
            'logD_max': self.df['logD'].max(),
            'logD_mean': self.df['logD'].mean(),
            'logD_std': self.df['logD'].std(),
            'logD_median': self.df['logD'].median()
        }

        return stats


def load_lipophilicity_dataset(data_path='data/external/lipophilicity.csv'):
    """
    Convenience function to load and validate lipophilicity dataset.

    Returns:
        tuple: (molecule_df, targets) ready for featurization and training
    """
    dataset = LipophilicityDataset(data_path)
    dataset.load_data()
    dataset.validate_smiles()

    mol_df = dataset.create_molecule_dataframe()
    _, targets = dataset.get_smiles_and_targets()

    # Add targets to molecule DataFrame
    mol_df['logD'] = targets

    stats = dataset.get_statistics()
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  N molecules: {stats['n_molecules']}")
    logger.info(f"  LogD range: [{stats['logD_min']:.2f}, {stats['logD_max']:.2f}]")
    logger.info(f"  LogD mean: {stats['logD_mean']:.2f} ± {stats['logD_std']:.2f}")
    logger.info(f"  LogD median: {stats['logD_median']:.2f}\n")

    return mol_df, targets


if __name__ == "__main__":
    # Test the loader
    mol_df, targets = load_lipophilicity_dataset()
    print(f"\nLoaded {len(mol_df)} molecules")
    print(f"\nFirst 5 entries:")
    print(mol_df.head())
    print(f"\nTarget statistics:")
    print(f"  Min: {targets.min():.2f}")
    print(f"  Max: {targets.max():.2f}")
    print(f"  Mean: {targets.mean():.2f}")
    print(f"  Std: {targets.std():.2f}")

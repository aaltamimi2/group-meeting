import pandas as pd
import numpy as np
from typing import List, Tuple
from src.utils import (
    setup_logging, load_config, ensure_directory,
    smiles_to_mol, compute_morgan_fingerprint,
    compute_maccs_keys, compute_physicochemical_descriptors,
    validate_smiles
)

logger = setup_logging(__name__)

class MolecularFeaturizer:
    """Featurize molecules using RDKit."""

    def __init__(self, config: dict):
        self.config = config
        self.morgan_radius = config['featurization']['morgan_fingerprint']['radius']
        self.morgan_nbits = config['featurization']['morgan_fingerprint']['n_bits']
        self.use_maccs = config['featurization']['use_maccs']

    def featurize_molecule(self, smiles: str) -> Tuple[np.ndarray, dict]:
        """Featurize a single molecule."""
        mol = smiles_to_mol(smiles)

        # Morgan fingerprint
        morgan_fp = compute_morgan_fingerprint(mol, self.morgan_radius, self.morgan_nbits)

        # MACCS keys
        if self.use_maccs:
            maccs = compute_maccs_keys(mol)
        else:
            maccs = np.array([])

        # Physicochemical descriptors
        descriptors = compute_physicochemical_descriptors(mol)

        return morgan_fp, maccs, descriptors

    def featurize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Featurize entire dataset."""
        logger.info("Starting featurization...")

        initial_count = len(df)
        logger.info(f"Initial dataset size: {initial_count} molecules")

        # Filter valid SMILES
        df = df[df['smiles'].notna()].copy()
        logger.info(f"Molecules with non-null SMILES: {len(df)}")

        df['valid_smiles'] = df['smiles'].apply(validate_smiles)
        valid_count = df['valid_smiles'].sum()
        logger.info(f"Molecules with valid SMILES: {valid_count}")

        if valid_count == 0:
            logger.error("No valid SMILES found! Check your input data.")
            logger.error("Common issues:")
            logger.error("  1. PubChem download failed to retrieve SMILES")
            logger.error("  2. Molecule names not found in PubChem")
            logger.error("  3. SMILES strings are malformed")
            logger.error("\nPlease run 'python diagnose.py' to debug the data.")
            raise ValueError("No valid SMILES found in dataset")

        df = df[df['valid_smiles']].copy()
        logger.info(f"Processing {len(df)} molecules with valid SMILES")

        # Initialize feature lists
        morgan_features = []
        maccs_features = []
        descriptor_features = []

        for idx, row in df.iterrows():
            smiles = row['smiles']
            morgan_fp, maccs, descriptors = self.featurize_molecule(smiles)

            morgan_features.append(morgan_fp)
            if self.use_maccs:
                maccs_features.append(maccs)
            descriptor_features.append(descriptors)

        # Create feature DataFrames
        morgan_df = pd.DataFrame(
            morgan_features,
            columns=[f'morgan_{i}' for i in range(self.morgan_nbits)]
        )

        if self.use_maccs:
            maccs_df = pd.DataFrame(
                maccs_features,
                columns=[f'maccs_{i}' for i in range(167)]
            )
        else:
            maccs_df = pd.DataFrame()

        descriptors_df = pd.DataFrame(descriptor_features)

        # Combine all features
        df_reset = df.reset_index(drop=True)
        morgan_df_reset = morgan_df.reset_index(drop=True)
        descriptors_df_reset = descriptors_df.reset_index(drop=True)

        if not maccs_df.empty:
            maccs_df_reset = maccs_df.reset_index(drop=True)
            result_df = pd.concat([df_reset, morgan_df_reset, maccs_df_reset, descriptors_df_reset], axis=1)
        else:
            result_df = pd.concat([df_reset, morgan_df_reset, descriptors_df_reset], axis=1)

        logger.info(f"Featurization complete. Shape: {result_df.shape}")
        logger.info(f"Feature columns created: {len([c for c in result_df.columns if c.startswith('morgan_') or c.startswith('maccs_') or c in ['MW', 'LogP', 'TPSA']])} features")
        return result_df

def main():
    """Main function to featurize molecules."""
    config = load_config()

    # Load raw data
    raw_data_path = config['output']['raw_data']
    logger.info(f"Loading data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    # Featurize
    featurizer = MolecularFeaturizer(config)
    processed_df = featurizer.featurize_dataset(df)

    # Save processed data
    output_path = config['output']['processed_data']
    ensure_directory('data/processed')
    processed_df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")

    return processed_df

if __name__ == "__main__":
    main()

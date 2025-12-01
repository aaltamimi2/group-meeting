import os
import logging
import yaml
from typing import Dict, Any
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
import numpy as np

def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)

def smiles_to_mol(smiles: str) -> Chem.Mol:
    """Convert SMILES string to RDKit Mol object."""
    if pd.isna(smiles) or smiles == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    return mol

def compute_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprint for a molecule."""
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def compute_maccs_keys(mol: Chem.Mol) -> np.ndarray:
    """Compute MACCS keys for a molecule."""
    if mol is None:
        return np.zeros(167)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    return np.array(maccs)

def compute_physicochemical_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Compute physicochemical descriptors for a molecule."""
    if mol is None:
        return {
            'MW': np.nan,
            'LogP': np.nan,
            'TPSA': np.nan,
            'NumHDonors': np.nan,
            'NumHAcceptors': np.nan,
            'NumRotatableBonds': np.nan,
            'NumAromaticRings': np.nan,
            'NumAliphaticRings': np.nan
        }

    descriptors = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol)
    }

    return descriptors

def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string."""
    if pd.isna(smiles) or smiles == "":
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

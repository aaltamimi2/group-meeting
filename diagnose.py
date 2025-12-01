"""Diagnostic script to check data issues"""
import pandas as pd
from src.utils import load_config, validate_smiles

config = load_config()
raw_data_path = config['output']['raw_data']

print("="*60)
print("DIAGNOSTIC REPORT")
print("="*60)

# Load raw data
df = pd.read_csv(raw_data_path)
print(f"\n1. Raw data shape: {df.shape}")
print(f"\n2. Columns: {list(df.columns)}")
print(f"\n3. First few rows:")
print(df.head())

print(f"\n4. SMILES column info:")
print(f"   - Total rows: {len(df)}")
print(f"   - Non-null SMILES: {df['smiles'].notna().sum()}")
print(f"   - Null SMILES: {df['smiles'].isna().sum()}")

if df['smiles'].notna().any():
    print(f"\n5. Sample SMILES values:")
    print(df[df['smiles'].notna()]['smiles'].head())

    print(f"\n6. SMILES validation:")
    df['valid'] = df['smiles'].apply(validate_smiles)
    print(f"   - Valid SMILES: {df['valid'].sum()}")
    print(f"   - Invalid SMILES: {(~df['valid']).sum()}")

    if df['valid'].sum() == 0:
        print("\n⚠️  WARNING: All SMILES are invalid!")
        print("   Sample invalid SMILES:")
        print(df[~df['valid']][['name', 'smiles']].head())
else:
    print("\n⚠️  WARNING: No SMILES data found!")

print("\n" + "="*60)

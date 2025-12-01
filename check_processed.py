"""Check processed data"""
import pandas as pd

df = pd.read_csv('data/processed/processed.csv')
print(f"Processed data shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)

"""Test download with debug output"""
import logging
from src.data_download import PubChemDownloader
from src.utils import load_config, setup_logging

# Enable debug logging
logger = setup_logging(__name__, level=logging.DEBUG)

# Load config
config = load_config()

# Create downloader
downloader = PubChemDownloader(config)

# Test with just aspirin
print("\n" + "="*60)
print("Testing with aspirin only...")
print("="*60 + "\n")

# Get CID
cid = downloader.get_compound_cid('aspirin')
print(f"\nCID for aspirin: {cid}")

if cid:
    # Get properties
    props = downloader.get_compound_properties(cid)
    print(f"\nProperties returned: {props is not None}")

    if props:
        print(f"\nAll keys in properties dict:")
        for key, value in props.items():
            print(f"  {key}: {value}")

        print(f"\n" + "="*60)
        print("Checking SMILES extraction:")
        print(f"  props.get('CanonicalSMILES'): {props.get('CanonicalSMILES')}")
        print("="*60)

# Now test full download
print("\n\nTesting full download with 3 molecules...")
df = downloader.download_molecules(['aspirin', 'caffeine', 'glucose'])
print(f"\nResulting DataFrame:")
print(df[['name', 'cid', 'smiles']])

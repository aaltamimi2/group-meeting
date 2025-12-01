"""Debug script to check PubChem API response structure"""
import requests
import json

# Test with aspirin
cid = 2244  # Aspirin CID

properties = [
    'IsomericSMILES',
    'InChI',
    'MolecularWeight',
    'XLogP',
    'TPSA',
    'HBondDonorCount',
    'HBondAcceptorCount'
]

props_str = ','.join(properties)
base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
url = f"{base_url}/compound/cid/{cid}/property/{props_str}/JSON"

print(f"Testing PubChem API with aspirin (CID: {cid})")
print(f"\nURL: {url}")
print("\n" + "="*60)

response = requests.get(url, timeout=30)
print(f"Status Code: {response.status_code}")
print("\n" + "="*60)
print("Raw Response:")
print(json.dumps(response.json(), indent=2))
print("\n" + "="*60)

if response.status_code == 200:
    data = response.json()
    if 'PropertyTable' in data:
        props = data['PropertyTable']['Properties'][0]
        print("\nExtracted Properties:")
        for key, value in props.items():
            print(f"  {key}: {value}")

        print("\n" + "="*60)
        print("Checking for SMILES:")
        print(f"  SMILES exists: {'SMILES' in props}")
        print(f"  SMILES value: {props.get('SMILES', 'NOT FOUND')}")
        print(f"  IsomericSMILES exists: {'IsomericSMILES' in props}")
        print(f"  IsomericSMILES value: {props.get('IsomericSMILES', 'NOT FOUND')}")
        print(f"  CanonicalSMILES exists: {'CanonicalSMILES' in props}")
        print(f"  CanonicalSMILES value: {props.get('CanonicalSMILES', 'NOT FOUND')}")
        print(f"  ConnectivitySMILES exists: {'ConnectivitySMILES' in props}")
        print(f"  ConnectivitySMILES value: {props.get('ConnectivitySMILES', 'NOT FOUND')}")
    else:
        print("\n⚠️  WARNING: 'PropertyTable' not found in response!")
else:
    print(f"\n⚠️  ERROR: Request failed with status {response.status_code}")

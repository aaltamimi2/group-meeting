import requests
import pandas as pd
import time
from typing import List, Dict, Any, Optional
from src.utils import setup_logging, load_config, ensure_directory

logger = setup_logging(__name__)

class PubChemDownloader:
    """Download molecular data from PubChem REST API."""

    def __init__(self, config: Dict[str, Any]):
        self.base_url = config['pubchem']['base_url']
        self.request_delay = config['pubchem']['request_delay']
        self.max_retries = config['pubchem']['max_retries']

    def _make_request(self, url: str, retries: int = 0) -> Optional[Dict]:
        """Make HTTP request with retry logic."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed: {e}")
            if retries < self.max_retries:
                wait_time = 2 ** retries
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                return self._make_request(url, retries + 1)
            logger.error(f"Max retries exceeded for URL: {url}")
            return None

    def get_compound_cid(self, name: str) -> Optional[int]:
        """Get PubChem CID from compound name."""
        url = f"{self.base_url}/compound/name/{name}/cids/JSON"
        data = self._make_request(url)

        if data and 'IdentifierList' in data:
            cid = data['IdentifierList']['CID'][0]
            logger.info(f"Found CID {cid} for {name}")
            return cid

        logger.warning(f"Could not find CID for {name}")
        return None

    def get_compound_properties(self, cid: int) -> Optional[Dict[str, Any]]:
        """Get compound properties from PubChem."""
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
        url = f"{self.base_url}/compound/cid/{cid}/property/{props_str}/JSON"

        data = self._make_request(url)

        if data and 'PropertyTable' in data:
            props = data['PropertyTable']['Properties'][0]
            logger.info(f"Retrieved properties for CID {cid}")
            return props

        logger.warning(f"Could not retrieve properties for CID {cid}")
        return None

    def download_molecules(self, molecule_names: List[str]) -> pd.DataFrame:
        """Download molecular data for a list of molecules."""
        results = []
        success_count = 0
        failed_cid_count = 0
        failed_properties_count = 0

        for name in molecule_names:
            logger.info(f"Processing {name}...")

            cid = self.get_compound_cid(name)
            if cid is None:
                failed_cid_count += 1
                results.append({
                    'name': name,
                    'cid': None,
                    'smiles': None,
                    'inchi': None,
                    'molecular_weight': None,
                    'xlogp': None,
                    'tpsa': None,
                    'hbd_count': None,
                    'hba_count': None
                })
                continue

            properties = self.get_compound_properties(cid)
            if properties is None:
                failed_properties_count += 1
                results.append({
                    'name': name,
                    'cid': cid,
                    'smiles': None,
                    'inchi': None,
                    'molecular_weight': None,
                    'xlogp': None,
                    'tpsa': None,
                    'hbd_count': None,
                    'hba_count': None
                })
                continue

            # PubChem may return IsomericSMILES, CanonicalSMILES, or ConnectivitySMILES
            smiles = (properties.get('IsomericSMILES') or
                     properties.get('CanonicalSMILES') or
                     properties.get('ConnectivitySMILES'))

            if smiles:
                success_count += 1
                logger.info(f"  ✓ Successfully retrieved SMILES for {name}")
            else:
                logger.warning(f"  ✗ No SMILES found for {name}")

            results.append({
                'name': name,
                'cid': cid,
                'smiles': smiles,
                'inchi': properties.get('InChI'),
                'molecular_weight': properties.get('MolecularWeight'),
                'xlogp': properties.get('XLogP'),
                'tpsa': properties.get('TPSA'),
                'hbd_count': properties.get('HBondDonorCount'),
                'hba_count': properties.get('HBondAcceptorCount')
            })

            time.sleep(self.request_delay)

        df = pd.DataFrame(results)
        logger.info(f"\n{'='*60}")
        logger.info(f"Download Summary:")
        logger.info(f"  Total molecules requested: {len(molecule_names)}")
        logger.info(f"  Successfully retrieved SMILES: {success_count}")
        logger.info(f"  Failed to find CID: {failed_cid_count}")
        logger.info(f"  Failed to get properties: {failed_properties_count}")
        logger.info(f"  Molecules with missing SMILES: {len(df) - success_count}")
        logger.info(f"{'='*60}")
        return df

def main():
    """Main function to download molecular data."""
    config = load_config()

    # Example molecule list
    molecules = [
        'aspirin', 'caffeine', 'ibuprofen', 'paracetamol',
        'morphine', 'codeine', 'nicotine', 'glucose',
        'dopamine', 'serotonin', 'acetaminophen', 'warfarin',
        'penicillin', 'insulin', 'metformin'
    ]

    downloader = PubChemDownloader(config)
    df = downloader.download_molecules(molecules)

    # Save to CSV
    output_path = config['output']['raw_data']
    ensure_directory('data/raw')
    df.to_csv(output_path, index=False)
    logger.info(f"Saved raw data to {output_path}")

    return df

if __name__ == "__main__":
    main()

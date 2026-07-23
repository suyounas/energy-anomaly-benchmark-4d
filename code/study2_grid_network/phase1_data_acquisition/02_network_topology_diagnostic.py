# Run in the same Colab session. Downloads and inspects europe_network.json
# to understand its actual schema BEFORE assuming how to identify France's
# interconnector lines to neighbouring countries. Small file (4.3MB), fast.

import json
import os

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.chdir(DATA_DIR)

# Reuse download_file if still in session, else define minimal version
if 'download_file' not in dir():
    import requests, time
    def download_file(filename, retries=5, initial_delay=5, backoff_factor=2, timeout=120):
        if os.path.exists(filename):
            print(f'{filename} already exists — skipping.')
            return
        url = f'https://zenodo.org/records/13378476/files/{filename}?download=1'
        for attempt in range(retries):
            try:
                print(f'Downloading {filename} (attempt {attempt+1}/{retries})...')
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                with open(filename, 'wb') as f:
                    f.write(r.content)
                print(f'  -> {len(r.content)/1024:.1f} KB saved')
                return
            except requests.exceptions.RequestException as e:
                delay = initial_delay * (backoff_factor ** attempt)
                print(f'  Error: {e}. Retrying in {delay:.1f}s...')
                time.sleep(delay)
        raise ConnectionError(f'Failed to download {filename}')

download_file('europe_network.json')

print('\nLoading and parsing network JSON...')
with open('europe_network.json') as f:
    network = json.load(f)

print('\n' + '=' * 70)
print('TOP-LEVEL STRUCTURE')
print('=' * 70)
print(f'Top-level keys: {list(network.keys())}')
for k, v in network.items():
    if isinstance(v, dict):
        print(f'  {k}: dict with {len(v)} entries')
    elif isinstance(v, list):
        print(f'  {k}: list with {len(v)} entries')
    else:
        print(f'  {k}: {v}')

# %% Look for the bus data structure ----------------------------------------
print('\n' + '=' * 70)
print('BUS DATA (sample)')
print('=' * 70)
bus_key = None
for candidate in ['bus', 'buses']:
    if candidate in network:
        bus_key = candidate
        break

if bus_key:
    buses = network[bus_key]
    if isinstance(buses, dict):
        sample_keys = list(buses.keys())[:3]
        for k in sample_keys:
            print(f'Bus ID "{k}": {buses[k]}')
    elif isinstance(buses, list):
        for b in buses[:3]:
            print(b)
else:
    print('No obvious "bus"/"buses" key found — printing full top-level keys again:')
    print(list(network.keys()))

# %% Look for the branch/line data structure --------------------------------
print('\n' + '=' * 70)
print('BRANCH/LINE DATA (sample)')
print('=' * 70)
branch_key = None
for candidate in ['branch', 'branches', 'line', 'lines']:
    if candidate in network:
        branch_key = candidate
        break

if branch_key:
    branches = network[branch_key]
    if isinstance(branches, dict):
        sample_keys = list(branches.keys())[:3]
        for k in sample_keys:
            print(f'Branch ID "{k}": {branches[k]}')
    elif isinstance(branches, list):
        for b in branches[:3]:
            print(b)
else:
    print('No obvious "branch"/"line" key found.')

# %% Cross-check: do load column IDs from loads_by_country.csv appear ------
# as bus identifiers anywhere in the network file? This tells us whether
# we can link loads_by_country.csv's FR list directly to bus IDs.
print('\n' + '=' * 70)
print('CROSS-CHECK: do FR load IDs match bus IDs in the network file?')
print('=' * 70)
import pandas as pd
loads_by_country = pd.read_csv('loads_by_country.csv', dtype=str)
fr_load_ids = loads_by_country['FR'].dropna().tolist()
print(f'Sample FR load IDs from loads_by_country.csv: {fr_load_ids[:5]}')

if bus_key:
    buses = network[bus_key]
    if isinstance(buses, dict):
        network_bus_ids = list(buses.keys())[:20]
    else:
        network_bus_ids = [str(b.get('bus_i', b.get('index', '?'))) for b in buses[:20]]
    print(f'Sample bus IDs from network file: {network_bus_ids}')
    overlap = set(fr_load_ids) & set(network_bus_ids if bus_key else [])
    print(f'\nDo any FR load IDs appear directly in the first 20 network bus IDs? '
          f'{len(overlap) > 0} (small sample check only, not conclusive either way)')

print('\nPhase 2c-diagnostic complete. Review the structure above — this')
print('determines how we identify France-neighbour interconnector lines')
print('before downloading the large lines_*.zip files.')

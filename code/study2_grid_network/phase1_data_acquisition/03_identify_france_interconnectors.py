# Run in the same Colab session. Uses the already-downloaded
# europe_network.json to identify exactly which line/branch IDs are
# France-neighbour interconnectors, BEFORE downloading any large lines_*.zip.

import json
import os

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.chdir(DATA_DIR)

with open('europe_network.json') as f:
    network = json.load(f)

# %% Build bus_id (str) -> country mapping -----------------------------------
bus_country = {bus_id: info['country'] for bus_id, info in network['bus'].items()}

countries_present = sorted(set(bus_country.values()))
print(f'Countries present in network: {countries_present}')
print(f'Total buses: {len(bus_country)}')

FRANCE_NEIGHBOURS = ['DE', 'CH', 'IT', 'ES', 'BE', 'LU']
print(f'\nChecking France + neighbours are all present: '
      f'{[c for c in ["FR"]+FRANCE_NEIGHBOURS if c in countries_present]}')
missing = [c for c in ["FR"]+FRANCE_NEIGHBOURS if c not in countries_present]
if missing:
    print(f'⚠️ NOT present in network: {missing}')

# %% Find all France-neighbour interconnector branches -----------------------
interconnectors = {}  # neighbour_country -> list of (branch_id, f_bus, t_bus)

for branch_id, info in network['branch'].items():
    f_bus = str(info['f_bus'])
    t_bus = str(info['t_bus'])
    f_country = bus_country.get(f_bus, 'UNKNOWN')
    t_country = bus_country.get(t_bus, 'UNKNOWN')

    countries_on_this_line = {f_country, t_country}
    if 'FR' in countries_on_this_line and len(countries_on_this_line) == 2:
        other = (countries_on_this_line - {'FR'}).pop()
        if other in FRANCE_NEIGHBOURS:
            interconnectors.setdefault(other, []).append(
                (branch_id, f_bus, f_country, t_bus, t_country)
            )

print('\n' + '=' * 70)
print('FRANCE-NEIGHBOUR INTERCONNECTOR LINES FOUND')
print('=' * 70)
all_line_ids = []
for neighbour, lines in interconnectors.items():
    print(f'\n{neighbour}: {len(lines)} interconnector line(s)')
    for branch_id, f_bus, f_c, t_bus, t_c in lines:
        print(f'  Line ID {branch_id}: bus {f_bus} ({f_c}) <-> bus {t_bus} ({t_c})')
        all_line_ids.append(branch_id)

print(f'\nTotal France-neighbour interconnector lines: {len(all_line_ids)}')
print(f'Line IDs (these are the column names to extract from lines_*.csv):')
print(all_line_ids)

# Save for use in the next step (downloading + extracting only these columns)
with open('france_interconnector_line_ids.json', 'w') as f:
    json.dump(all_line_ids, f)
print(f'\nSaved: france_interconnector_line_ids.json')

if len(all_line_ids) == 0:
    print('\n⚠️ NO interconnector lines found — something is wrong with the')
    print('country-matching logic above. Do not proceed to download lines_*.zip')
    print('until this is resolved.')
elif len(all_line_ids) > 30:
    print(f'\n⚠️ {len(all_line_ids)} lines found — more than expected for major')
    print('interconnectors alone. This may include minor local cross-border')
    print('lines, not just major interconnectors. Review the list above before')
    print('proceeding — may want to filter further by rate_a (line capacity).')
else:
    print(f'\n✅ {len(all_line_ids)} lines is a tractable number to extract and model.')

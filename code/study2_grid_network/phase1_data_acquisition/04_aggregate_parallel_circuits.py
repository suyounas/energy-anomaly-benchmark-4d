# Run in the same Colab session. Groups the 62 France-neighbour lines into
# their 28 distinct PHYSICAL interconnection points (aggregating parallel
# circuits at the same bus-pair), before downloading any lines_*.zip.

import json
import os
from collections import defaultdict

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.chdir(DATA_DIR)

with open('europe_network.json') as f:
    network = json.load(f)
with open('france_interconnector_line_ids.json') as f:
    all_line_ids = json.load(f)

bus_country = {bus_id: info['country'] for bus_id, info in network['bus'].items()}

# %% Group lines by physical (bus_pair) crossing point, not by individual line
crossing_points = defaultdict(list)  # (sorted bus pair) -> list of line_ids

for line_id in all_line_ids:
    info = network['branch'][line_id]
    f_bus, t_bus = str(info['f_bus']), str(info['t_bus'])
    key = tuple(sorted([f_bus, t_bus]))  # order-independent pairing
    crossing_points[key].append(line_id)

print('=' * 70)
print(f'DISTINCT PHYSICAL CROSSING POINTS: {len(crossing_points)}')
print('=' * 70)

crossing_point_map = {}  # crossing_point_name -> list of line_ids
for (bus_a, bus_b), line_ids in crossing_points.items():
    country_a = bus_country[bus_a]
    country_b = bus_country[bus_b]
    neighbour = country_b if country_a == 'FR' else country_a
    name = f'FR-{neighbour}_{bus_a}-{bus_b}'
    crossing_point_map[name] = line_ids
    print(f'{name}: {len(line_ids)} parallel circuit(s) -> line IDs {line_ids}')

print(f'\nTotal crossing points: {len(crossing_point_map)}')
print(f'Total individual lines being aggregated: {sum(len(v) for v in crossing_point_map.values())}')

with open('crossing_point_map.json', 'w') as f:
    json.dump(crossing_point_map, f, indent=2)
print('\nSaved: crossing_point_map.json')
print('This maps each physical crossing point name to the line IDs (columns)')
print('that need to be SUMMED together once lines_*.zip is downloaded.')

# %% Sanity check: do the "duplicate" bus-pair lines actually look redundant?
# We can't check this yet without the actual flow data, but we CAN check
# their electrical parameters (br_x = reactance) — near-identical parallel
# circuits should have similar reactance values.
print('\n' + '=' * 70)
print('ELECTRICAL PARAMETER CHECK (do parallel circuits look genuinely similar?)')
print('=' * 70)
for name, line_ids in list(crossing_point_map.items()):
    if len(line_ids) > 1:
        reactances = [network['branch'][lid]['br_x'] for lid in line_ids]
        rates = [network['branch'][lid]['rate_a'] for lid in line_ids]
        print(f'{name}: reactances={[round(r,4) for r in reactances]}, '
              f'ratings={rates}')

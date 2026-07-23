# Run in the same Colab session. Tests whether our sign convention is
# GLOBALLY inverted, using France->Italy as the test case — Italy had zero
# domestic nuclear generation throughout 2016-2018 (shut down since the
# 1987 referendum) and was a large, structural electricity importer, so
# France->Italy flow should be strongly and consistently POSITIVE under
# a correct convention.

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import os

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.chdir(DATA_DIR)
print(f'Working directory: {os.getcwd()}')
print(f'france_export_to_IT.csv present: {os.path.exists("france_export_to_IT.csv")}')

BASE_MVA = 100

it_df = pd.read_csv('france_export_to_IT.csv', parse_dates=['Date'])
it_df['GW'] = it_df['NetExport'] * BASE_MVA / 1000

print('=' * 70)
print('FRANCE -> ITALY: sign convention test')
print('=' * 70)
print(f'Mean (current convention): {it_df.GW.mean():+.2f} GW')
print(f'% of hours positive: {(it_df.GW > 0).mean()*100:.1f}%')
print(f'% of hours negative: {(it_df.GW < 0).mean()*100:.1f}%')
print(f'Max: {it_df.GW.max():+.2f} GW, Min: {it_df.GW.min():+.2f} GW')

print(f'\nIf this is DOMINANTLY negative (>70% of hours), that strongly')
print(f'supports a GLOBAL sign inversion — try flipping the sign of')
print(f'EVERY line (not just the 10 originally flipped) and re-check.')
print(f'\nIf it is mixed (40-60% either way) with a small negative mean,')
print(f'that looks more like normal bidirectional trading with a')
print(f'seasonal/annual quirk, not a wholesale convention error.')

# %% Quick test: what does a GLOBAL flip look like? --------------------------
print('\n' + '=' * 70)
print('WHAT IF WE GLOBALLY FLIP EVERYTHING?')
print('=' * 70)
it_flipped_mean = -it_df.GW.mean()
print(f'France -> Italy mean UNDER GLOBAL FLIP: {it_flipped_mean:+.2f} GW')
print(f'(Does this look more like the expected strongly-positive real-world value?)')

total_df = pd.read_csv('france_total_net_export.csv', parse_dates=['Date'])
total_df['GW'] = total_df['TotalNetExport'] * BASE_MVA / 1000
total_flipped_mean = -total_df.GW.mean()
print(f'\nFrance TOTAL net export, full available period, current convention: '
      f'{total_df.GW.mean():+.2f} GW')
print(f'France TOTAL net export, full available period, GLOBAL FLIP: '
      f'{total_flipped_mean:+.2f} GW')
print(f'(France is a well-documented net exporter overall — which of these')
print(f' two numbers is more plausible?)')

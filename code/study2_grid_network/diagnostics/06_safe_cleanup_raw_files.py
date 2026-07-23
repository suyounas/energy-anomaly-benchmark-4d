# Run in your Colab session. Step 1 just LISTS sizes — review before
# running Step 2, which actually deletes.

from google.colab import drive
drive.mount('/content/drive')

import os

DATA_DIR = '/content/drive/MyDrive/grid_data'
os.chdir(DATA_DIR)

# %% [1] List all files by size, largest first — REVIEW BEFORE DELETING ----
files = [(f, os.path.getsize(f)) for f in os.listdir('.') if os.path.isfile(f)]
files.sort(key=lambda x: -x[1])

print('=' * 70)
print('ALL FILES BY SIZE')
print('=' * 70)
total = 0
for fname, size in files:
    mb = size / (1024*1024)
    total += mb
    print(f'  {mb:8.1f} MB  {fname}')
print(f'\nTotal: {total:.1f} MB ({total/1024:.2f} GB)')

# %% [2] Identify the safe-to-delete raw files specifically -----------------
RAW_FILES_SAFE_TO_DELETE = (
    [f'loads_{y}_1.csv' for y in [2016, 2017, 2018, 2019, 2020]] +
    [f'lines_{y}_1.csv' for y in [2016, 2017, 2018]]
)

print('\n' + '=' * 70)
print('FILES THAT WOULD BE DELETED (raw wide-format, already aggregated)')
print('=' * 70)
to_delete = []
freed_mb = 0
for fname in RAW_FILES_SAFE_TO_DELETE:
    if os.path.exists(fname):
        size_mb = os.path.getsize(fname) / (1024*1024)
        freed_mb += size_mb
        to_delete.append(fname)
        print(f'  {size_mb:8.1f} MB  {fname}')
    else:
        print(f'  (not found, skipping)  {fname}')

print(f'\nTotal space that would be freed: {freed_mb:.1f} MB ({freed_mb/1024:.2f} GB)')

# %% [3] Verify the aggregated outputs actually exist BEFORE deleting -------
# Safety check: refuse to delete a raw file if its aggregated output isn't
# confirmed present on disk.
print('\n' + '=' * 70)
print('SAFETY CHECK: confirming aggregated outputs exist before deleting')
print('=' * 70)

safe_to_proceed = True

loads_years_present = [y for y in [2016,2017,2018,2019,2020]
                        if os.path.exists(f'loads_{y}_1.csv')]
if loads_years_present:
    sample_country_file = 'national_load_FR_full.csv'
    if os.path.exists(sample_country_file):
        print(f'✅ {sample_country_file} exists — loads aggregation confirmed.')
    else:
        print(f'❌ {sample_country_file} NOT FOUND — do not delete loads_*_1.csv yet.')
        safe_to_proceed = False

lines_years_present = [y for y in [2016,2017,2018] if os.path.exists(f'lines_{y}_1.csv')]
if lines_years_present:
    if os.path.exists('france_total_net_export.csv'):
        print('✅ france_total_net_export.csv exists — lines aggregation confirmed.')
    else:
        print('❌ france_total_net_export.csv NOT FOUND — do not delete lines_*_1.csv yet.')
        safe_to_proceed = False

# %% [4] ACTUAL DELETION — only runs if safety check passed -----------------
if safe_to_proceed and len(to_delete) > 0:
    print('\n' + '=' * 70)
    print('DELETING NOW')
    print('=' * 70)
    for fname in to_delete:
        os.remove(fname)
        print(f'  Deleted: {fname}')
    print(f'\n✅ Freed approximately {freed_mb:.1f} MB ({freed_mb/1024:.2f} GB)')
elif not safe_to_proceed:
    print('\n⚠️ SAFETY CHECK FAILED — nothing was deleted. Fix the missing')
    print('aggregated output(s) above before re-running this step.')
else:
    print('\nNothing to delete — no matching raw files found.')

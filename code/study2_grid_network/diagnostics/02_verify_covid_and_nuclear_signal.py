# Run in the same Colab session — checks whether the two real-world events
# are actually visible as anomalies in this SYNTHETIC dataset, before we
# trust them as ground truth for modelling.

import pandas as pd
import matplotlib.pyplot as plt

fr = pd.read_csv('national_load_FR_full.csv', parse_dates=['Date'])

print('=' * 70)
print('CHECK 1: French nuclear crisis (Oct 2016 - Jan 2017)')
print('=' * 70)
print('Real-world story: France became a NET IMPORTER for ~110 days, exports')
print('ran ~7.2 GW below average. This should show as reduced FRENCH')
print('domestic load pressure is NOT directly visible in load data (load ≈')
print('domestic demand, not imports/exports) — this event is fundamentally')
print('an EXPORT/GENERATION story, not a LOAD story. We may be looking in')
print('the wrong data type entirely for this specific event.')
print()

window1 = fr[(fr.Date >= '2016-09-01') & (fr.Date <= '2017-02-15')].copy()
daily1 = window1.groupby(window1.Date.dt.date)['TotalLoad'].mean().reset_index()
daily1.columns = ['Date', 'MeanLoad']

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(pd.to_datetime(daily1['Date']), daily1['MeanLoad'])
ax.axvspan(pd.Timestamp('2016-10-10'), pd.Timestamp('2017-01-27'),
           alpha=0.2, color='red', label='Documented nuclear crisis window')
ax.set_title('France TOTAL LOAD (demand) around the 2016-17 nuclear crisis')
ax.set_ylabel('Daily mean load')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('nuclear_crisis_check.png', dpi=120)
plt.show()
print('Saved: nuclear_crisis_check.png — VISUALLY INSPECT: is there any')
print('structural anomaly in LOAD during the shaded window, or does it look')
print('like normal winter seasonal demand (which would be expected either way,')
print('since winter demand is naturally high regardless of the nuclear crisis)?')

print('\n' + '=' * 70)
print('CHECK 2: COVID-19 demand collapse (Mar-May 2020)')
print('=' * 70)
print('Real-world story: lockdowns reduced INDUSTRIAL/COMMERCIAL demand')
print('directly. This IS fundamentally a LOAD story, so this is the more')
print('promising of the two events to expect visible in load data.')
print()

window2 = fr[(fr.Date >= '2020-01-01') & (fr.Date <= '2020-07-01')].copy()
daily2 = window2.groupby(window2.Date.dt.date)['TotalLoad'].mean().reset_index()
daily2.columns = ['Date', 'MeanLoad']

pre_covid = daily2[(pd.to_datetime(daily2['Date']) >= '2020-02-01') &
                    (pd.to_datetime(daily2['Date']) < '2020-03-16')]['MeanLoad'].mean()
during_covid = daily2[(pd.to_datetime(daily2['Date']) >= '2020-03-16') &
                       (pd.to_datetime(daily2['Date']) <= '2020-05-11')]['MeanLoad'].mean()
post_covid = daily2[(pd.to_datetime(daily2['Date']) > '2020-05-11') &
                     (pd.to_datetime(daily2['Date']) <= '2020-06-15')]['MeanLoad'].mean()

pct_drop = 100 * (during_covid - pre_covid) / pre_covid
print(f'Mean load Feb 1 - Mar 15 (pre-lockdown):  {pre_covid:.1f}')
print(f'Mean load Mar 16 - May 11 (lockdown):      {during_covid:.1f}')
print(f'Mean load May 12 - Jun 15 (post-lockdown):  {post_covid:.1f}')
print(f'Pre -> during change: {pct_drop:+.1f}%')
print(f'(Real-world reporting for France/Europe during COVID: demand fell')
print(f' roughly 15-20% at the trough. If this synthetic data shows a')
print(f' comparable drop, that is strong evidence the event IS captured.)')

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(pd.to_datetime(daily2['Date']), daily2['MeanLoad'])
ax.axvspan(pd.Timestamp('2020-03-16'), pd.Timestamp('2020-05-11'),
           alpha=0.2, color='red', label='Documented COVID lockdown window')
ax.set_title('France TOTAL LOAD (demand) around COVID-19 lockdown')
ax.set_ylabel('Daily mean load')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('covid_check.png', dpi=120)
plt.show()
print('\nSaved: covid_check.png')

print('\n' + '=' * 70)
print('INTERPRETATION GUIDE')
print('=' * 70)
print('If COVID shows a clear, sustained ~15-20% drop matching real-world')
print('reporting: this event is validly usable as ground truth for LOAD data.')
print()
print('If the nuclear crisis shows NO visible anomaly in load: this is')
print('EXPECTED and not a problem with the dataset — the real event was an')
print('export/generation phenomenon, not a demand phenomenon. It may be a')
print('better fit for LINE FLOW data (France-neighbour interconnectors)')
print('than for load data. Worth re-scoping that event to lines, not loads,')
print('if load shows nothing.')

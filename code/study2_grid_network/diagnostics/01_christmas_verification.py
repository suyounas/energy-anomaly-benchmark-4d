# Run in the same Colab session — verifies date alignment using the
# already-saved national_load_*.csv files. No new downloads needed.

import pandas as pd
import matplotlib.pyplot as plt

print('=' * 70)
print('VERIFICATION 1: Christmas Day 2016 demand dip (expect a sharp drop)')
print('=' * 70)

fr = pd.read_csv('national_load_FR.csv', parse_dates=['Date'])

# Daily average load across late December
window = fr[(fr.Date >= '2016-12-18') & (fr.Date <= '2017-01-02')].copy()
window['DayOfWeek'] = window.Date.dt.day_name()
daily = window.groupby(window.Date.dt.date).agg(
    MeanLoad=('TotalLoad', 'mean'),
).reset_index()
daily['DayOfWeek'] = pd.to_datetime(daily['Date']).dt.day_name()

print('\nFrance daily mean load, 18 Dec 2016 - 2 Jan 2017:')
print(daily.to_string(index=False))

christmas_load = daily[pd.to_datetime(daily['Date']).dt.date == pd.Timestamp('2016-12-25').date()]
baseline = daily[~daily['Date'].isin([pd.Timestamp('2016-12-25').date(), pd.Timestamp('2016-12-24').date(),
                                        pd.Timestamp('2016-12-26').date()])]['MeanLoad'].mean()

if len(christmas_load) > 0:
    xmas_val = christmas_load['MeanLoad'].values[0]
    xmas_day = christmas_load['DayOfWeek'].values[0]
    pct_diff = 100 * (xmas_val - baseline) / baseline
    print(f'\n25 Dec 2016 falls on: {xmas_day}  (should be "Sunday" if alignment is correct)')
    print(f'25 Dec 2016 mean load: {xmas_val:.1f}')
    print(f'Surrounding baseline mean load: {baseline:.1f}')
    print(f'Difference: {pct_diff:+.1f}%  (expect a clear negative dip, typically -15% to -30%'
          f' for a major holiday+weekend combination)')
else:
    print('\n⚠️ 25 Dec 2016 not found in this date range — check the data.')

print('\n' + '=' * 70)
print('VERIFICATION 2: Weekday vs weekend pattern (full year)')
print('=' * 70)

fr['DayOfWeek'] = fr.Date.dt.day_name()
by_dow = fr.groupby('DayOfWeek')['TotalLoad'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)
print('\nMean load by day of week (full 2016 dataset):')
print(by_dow.to_string())
print('\nExpectation if alignment is correct: Mon-Fri should be visibly higher')
print('than Sat-Sun (industrial/commercial demand on weekdays).')
weekday_avg = by_dow[['Monday','Tuesday','Wednesday','Thursday','Friday']].mean()
weekend_avg = by_dow[['Saturday','Sunday']].mean()
print(f'\nWeekday average: {weekday_avg:.1f}')
print(f'Weekend average: {weekend_avg:.1f}')
print(f'Weekday/weekend ratio: {weekday_avg/weekend_avg:.3f}  (expect > 1.05 if pattern is real)')

# %% Visual check
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(pd.to_datetime(daily['Date']), daily['MeanLoad'], marker='o')
xmas_idx = daily[pd.to_datetime(daily['Date']).dt.date == pd.Timestamp('2016-12-25').date()].index
if len(xmas_idx) > 0:
    ax.axvline(pd.Timestamp('2016-12-25'), color='red', linestyle='--', label='25 Dec (Christmas)')
ax.set_title('France daily mean load, late Dec 2016 - early Jan 2017')
ax.set_ylabel('Mean load (per-unit)')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('christmas_verification.png', dpi=120)
plt.show()
print('\nSaved plot: christmas_verification.png')

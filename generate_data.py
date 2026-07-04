"""
Generates a realistic synthetic hourly energy demand dataset.
Mimics real grid load patterns: daily cycle, weekly cycle, seasonal trend,
holiday dips, and random noise -- similar in structure to public datasets
like PJM Hourly Energy Consumption (Kaggle), but self-contained and
reproducible so no external download/login is needed.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# 2 years of hourly data
start = pd.Timestamp("2024-01-01 00:00:00")
end = pd.Timestamp("2025-12-31 23:00:00")
timestamps = pd.date_range(start, end, freq="h")

n = len(timestamps)
hours = timestamps.hour.values
dow = timestamps.dayofweek.values          # 0=Mon ... 6=Sun
day_of_year = timestamps.dayofyear.values

# ---- Base load ----
base_load = 30000  # MW, arbitrary baseline

# ---- Daily seasonality: two peaks (morning + evening), dip at night ----
daily_pattern = (
    4000 * np.sin((hours - 6) / 24 * 2 * np.pi) +
    3000 * np.exp(-((hours - 19) ** 2) / (2 * 3 ** 2)) +   # evening peak
    2000 * np.exp(-((hours - 9) ** 2) / (2 * 2 ** 2))      # morning peak
)

# ---- Weekly seasonality: lower demand on weekends ----
weekend_dip = np.where(dow >= 5, -3000, 0)

# ---- Yearly seasonality: higher demand in summer (cooling) and winter (heating) ----
yearly_pattern = 5000 * np.cos((day_of_year - 200) / 365 * 2 * np.pi) ** 2 - 2000

# ---- Holidays: known fixed-date holidays get a demand dip ----
holidays = pd.to_datetime([
    "2024-01-01", "2024-07-04", "2024-12-25", "2024-11-28",
    "2025-01-01", "2025-07-04", "2025-12-25", "2025-11-27",
])
is_holiday = np.isin(timestamps.normalize(), holidays)
holiday_dip = np.where(is_holiday, -4000, 0)

# ---- Random noise ----
noise = np.random.normal(0, 800, n)

# ---- Combine ----
load = base_load + daily_pattern + weekend_dip + yearly_pattern + holiday_dip + noise
load = np.clip(load, 15000, None)  # keep values realistic (no negative/zero load)

df = pd.DataFrame({
    "Datetime": timestamps,
    "Demand_MW": load.round(1),
    "Is_Holiday": is_holiday.astype(int),
})

out_path = "/home/claude/energy_project/data/energy_demand.csv"
df.to_csv(out_path, index=False)

print(f"Saved {len(df)} rows to {out_path}")
print(df.head())
print(df.tail())
print("\nStats:")
print(df["Demand_MW"].describe())

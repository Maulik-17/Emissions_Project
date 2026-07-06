"""
Script 2: Feature Engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load filtered data from Script 1
df = pd.read_csv("filtered_co2_data.csv")

# ------------------------------------------------------------------------------
# 2.1 Time-Based Features
# ------------------------------------------------------------------------------
df['decade'] = (df['year'] // 10) * 10
df['years_since_1990'] = df['year'] - 1990

# For each country, compute 5-year rolling mean of co2 (using current and past 4 years)
df['co2_5yr_rolling_mean'] = df.groupby('country')['co2'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)

# ------------------------------------------------------------------------------
# 2.2 Lag Features
# ------------------------------------------------------------------------------
for lag in [1, 2, 3]:
    df[f'co2_lag{lag}'] = df.groupby('country')['co2'].shift(lag)

print("\n--- 2.2 Lag Features ---")
print("Lag features (co2_lag1, co2_lag2, co2_lag3) represent the CO₂ emissions of the previous 1, 2, and 3 years, respectively.")
print("They capture temporal dependencies and are essential for time-series prediction because past emissions often influence future emissions due to inertia in energy systems and economic activity.")

# ------------------------------------------------------------------------------
# 2.3 Per-Capita and Intensity Features
# ------------------------------------------------------------------------------
# Verify co2_per_capita = co2 / population for 3 countries and 3 years
test_cases = [
    ('China', 2010),
    ('United States', 2015),
    ('India', 2020)
]
print("\n--- 2.3 Verification of co2_per_capita ---")
for country, year in test_cases:
    row = df[(df['country'] == country) & (df['year'] == year)]
    if not row.empty:
        co2 = row['co2'].values[0]
        pop = row['population'].values[0]
        computed = co2 / pop
        reported = row['co2_per_capita'].values[0]
        print(f"{country} {year}: computed={computed:.4f}, reported={reported:.4f}, match? {np.isclose(computed, reported, rtol=1e-3)}")
    else:
        print(f"{country} {year}: data not available.")

# Create ghg_intensity = total_ghg / gdp (where both exist)
df['ghg_intensity'] = df['total_ghg'] / df['gdp']
print("\nCountries and years where ghg_intensity cannot be computed (missing GDP or total_ghg):")
missing_intensity = df[df['ghg_intensity'].isnull()]
print(missing_intensity[['country', 'year']].head(10))
print(f"Total missing: {missing_intensity.shape[0]} rows")

# ------------------------------------------------------------------------------
# 2.4 Growth Rate Features
# ------------------------------------------------------------------------------
# Year-on-year absolute change
df['co2_yoy_change'] = df.groupby('country')['co2'].diff()
# Year-on-year percentage change
df['co2_yoy_pct_change'] = df.groupby('country')['co2'].pct_change() * 100

# Top 5 countries with highest average annual CO₂ growth rate since 1990
# Compute average of yoy pct change per country (excluding first year and NaNs)
growth_rates = df[df['year'] >= 1990].groupby('country')['co2_yoy_pct_change'].mean().dropna()
top5_growth = growth_rates.nlargest(5)
print("\nTop 5 countries with highest average annual CO₂ growth rate since 1990:")
print(top5_growth)

# Top 5 countries with largest CO₂ reductions since 1990
# Compute total change from 1990 to latest year (per country)
first_last = df[df['year'].isin([1990, df['year'].max()])].groupby('country')
def total_change(group):
    if len(group) == 2:
        return group[group['year'] == group['year'].max()]['co2'].values[0] - group[group['year'] == group['year'].min()]['co2'].values[0]
    else:
        return np.nan
reductions = df.groupby('country').apply(total_change).dropna()
top5_reductions = reductions.nsmallest(5)
print("\nTop 5 countries with largest CO₂ reductions since 1990 (most negative change):")
print(top5_reductions)

# ------------------------------------------------------------------------------
# 2.5 Final Feature Dataset for 10 Project Countries
# ------------------------------------------------------------------------------
# Select top 10 countries by total CO₂ emissions (sovereign)
top10_countries = df.groupby('country')['co2'].sum().nlargest(10).index.tolist()
print(f"\nTop 10 project countries: {top10_countries}")

required_cols = [
    'country', 'year', 'co2', 'co2_per_capita', 'co2_5yr_rolling_mean',
    'co2_lag1', 'co2_lag2', 'co2_lag3', 'co2_yoy_pct_change', 'ghg_intensity'
]
df_final = df[df['country'].isin(top10_countries)][required_cols].copy()

# Drop rows where any feature (excluding possibly ghg_intensity) is NaN
# We'll keep ghg_intensity as is (may have NaN) for modelling; will handle later.
df_final = df_final.dropna(subset=['co2', 'co2_per_capita', 'co2_5yr_rolling_mean',
                                   'co2_lag1', 'co2_lag2', 'co2_lag3', 'co2_yoy_pct_change'])

# Save
df_final.to_csv("ghg_features.csv", index=False)
print(f"Final feature dataset saved to 'ghg_features.csv' with shape {df_final.shape}")

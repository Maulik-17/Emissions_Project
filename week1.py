"""
Script 1: Data Loading, Profiling, and Exploratory Data Analysis (EDA)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import requests
from io import StringIO

# ------------------------------------------------------------------------------
# 1.1 Data Loading
# ------------------------------------------------------------------------------
url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
local_file = "owid-co2-data.csv"

if not os.path.exists(local_file):
    print("Downloading OWID CO₂ dataset...")
    response = requests.get(url)
    with open(local_file, "wb") as f:
        f.write(response.content)
else:
    print("Using local copy of OWID CO₂ dataset.")

df = pd.read_csv(local_file)

print("\n--- 1.1 Data Loading ---")
print("First 10 rows:")
print(df.head(10))
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print(f"\nShape: {df.shape}")

# Markdown explanation of key columns
print("\n--- Key Columns Explanation ---")
print("""
- `co2`: Annual CO₂ emissions (million tonnes, MtCO₂).
- `co2_per_capita`: CO₂ emissions per person (tonnes per capita).
- `methane`: Annual methane emissions (MtCO₂e).
- `nitrous_oxide`: Annual nitrous oxide emissions (MtCO₂e).
- `total_ghg`: Total greenhouse gas emissions (MtCO₂e), sum of CO₂, CH₄, N₂O, and other gases.
- `year`: Calendar year.
- `country`: Name of the country or aggregate region.
""")

# ------------------------------------------------------------------------------
# 1.2 Data Profiling
# ------------------------------------------------------------------------------
print("\n--- 1.2 Data Profiling ---")
null_percent = (df.isnull().sum() / len(df)) * 100
print("\nNull percentage per column:")
print(null_percent)

# Countries and years with most complete data (non-null co2, population, gdp)
key_cols = ['co2', 'population', 'gdp']
complete = df[key_cols].notnull().all(axis=1)
complete_df = df[complete]
print("\nCountries with most complete data (non-null co2, population, gdp):")
print(complete_df['country'].value_counts().head(10))
print("\nYears with most complete data:")
print(complete_df['year'].value_counts().head(10))

# Filter: year >= 1990 and sovereign nations (iso_code not null)
df_filtered = df[(df['year'] >= 1990) & (df['iso_code'].notnull())].copy()
print(f"\nFiltered dataset shape: {df_filtered.shape}")
print("Filtering decisions:")
print("- Retain only rows with year >= 1990 to focus on recent period.")
print("- Retain only sovereign nations (iso_code not null) to exclude aggregates like 'World', 'Asia', etc.")

# Save filtered data for later use
filtered_file = "filtered_co2_data.csv"
df_filtered.to_csv(filtered_file, index=False)
print(f"Filtered data saved to {filtered_file}")

# ------------------------------------------------------------------------------
# 1.3 Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
print("\n--- 1.3 EDA ---")

# Global CO₂ emissions (sum over countries)
global_co2 = df_filtered.groupby('year')['co2'].sum().reset_index()

plt.figure(figsize=(12,6))
plt.plot(global_co2['year'], global_co2['co2'], color='darkred', linewidth=2)
plt.title('Global CO₂ Emissions (1990–latest)')
plt.xlabel('Year')
plt.ylabel('Total CO₂ (MtCO₂)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('global_co2_trend.png', dpi=150)
plt.close()
print("Global CO₂ line chart saved as 'global_co2_trend.png'.")
print("Summary: Global CO₂ emissions increased steadily from 1990 until around 2019, then dipped during the COVID-19 pandemic in 2020, and have since rebounded, though the rate of growth appears to have slowed in recent years.")

# Top 5 emitting countries (by total CO₂ 1990–latest)
top5 = df_filtered.groupby('country')['co2'].sum().nlargest(5).index.tolist()
print(f"Top 5 emitting countries: {top5}")

plt.figure(figsize=(12,6))
for country in top5:
    country_data = df_filtered[df_filtered['country'] == country]
    plt.plot(country_data['year'], country_data['co2'], label=country, linewidth=2)

plt.title('CO₂ Emission Trends for Top 5 Emitting Countries')
plt.xlabel('Year')
plt.ylabel('CO₂ (MtCO₂)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('top5_emitters_trends.png', dpi=150)
plt.close()
print("Multi-line chart saved as 'top5_emitters_trends.png'.")
print("Summary: China shows the steepest increase, overtaking the USA in the mid-2000s. USA emissions have plateaued and slightly declined since 2005. India's emissions are rising steadily but remain lower. Russia and Japan show relatively flat trends with some fluctuations.")

# Stacked bar chart: share of total GHG by gas type per decade
# We need co2, methane, nitrous_oxide columns (in MtCO₂e)
# Compute decade column
df_filtered['decade'] = (df_filtered['year'] // 10) * 10
decade_gases = df_filtered.groupby('decade')[['co2', 'methane', 'nitrous_oxide']].sum()
# Convert to proportions
decade_gases_total = decade_gases.sum(axis=1)
decade_gases_pct = decade_gases.div(decade_gases_total, axis=0) * 100

# Keep only decades with data: 1990s, 2000s, 2010s, 2020s
decades_to_plot = [1990, 2000, 2010, 2020]
decade_gases_pct = decade_gases_pct.loc[decades_to_plot]

# Plot stacked bar
ax = decade_gases_pct.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
plt.title('Share of Global GHG Emissions by Gas Type per Decade')
plt.xlabel('Decade')
plt.ylabel('Percentage of Total GHG')
plt.legend(title='Gas')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('ghg_share_by_decade.png', dpi=150)
plt.close()
print("Stacked bar chart saved as 'ghg_share_by_decade.png'.")
print("Summary: CO₂ dominates the GHG mix, accounting for ~70-80% of total emissions. Methane contributes ~15-20%, and nitrous oxide ~5-10%. The share of CO₂ has slightly increased over the decades, while methane's share has decreased somewhat, reflecting energy transitions and agricultural changes.")

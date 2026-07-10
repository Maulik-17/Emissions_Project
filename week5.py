# %% [markdown]
# # OWID CO2 — Mitigation Scenario Planning (2025-2040)
#
# Builds on `co2_ets_forecasting.py`. Takes the ETS(A,Ad,N) Business-as-Usual
# forecast from Week 4 and derives two illustrative mitigation scenarios,
# then visualises and summarises the cumulative impact per country.

# %%
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)
np.random.seed(42)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
RAW_LOCAL_PATH = os.path.join(OUTPUT_DIR, "owid-co2-data.csv")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "ghg_features.csv")

PROJECT_COUNTRIES = [
    "China", "United States", "India", "Russia", "Japan",
    "Germany", "United Kingdom", "Brazil", "South Africa", "Australia",
]

TRAIN_YEAR_MIN, TRAIN_YEAR_MAX = 1990, 2018
PLOT_HIST_YEAR_MAX = 2024      # historical-actuals reference line, per spec
SCENARIO_YEAR_MIN, SCENARIO_YEAR_MAX = 2025, 2040
CHART_YEAR_MIN = 2020          # scenario overlay window per spec (2020-2040)
FORECAST_YEAR_MAX = 2043       # keep consistent with Week 4

MODERATE_ANNUAL_REDUCTION = 0.02   # 2% per year
AGGRESSIVE_ANNUAL_REDUCTION = 0.05  # 5% per year

COLORS = {"BAU": "#1f77b4", "Moderate Mitigation": "#ff7f0e", "Aggressive Mitigation": "#2ca02c"}

# %% [markdown]
# ## Load feature data & rebuild ETS BAU forecasts
#
# Rebuilds the Week 4 ETS(A,Ad,N) fits (trained on 1990-2018) here so this
# script can run standalone; if you already have the fitted models in memory
# from `co2_ets_forecasting.py`, feel free to reuse those instead.

# %%
def build_feature_dataset() -> pd.DataFrame:
    if not os.path.exists(RAW_LOCAL_PATH):
        print(f"Downloading dataset from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, RAW_LOCAL_PATH)
    raw = pd.read_csv(RAW_LOCAL_PATH)

    aggregate_names = {
        "World", "Africa", "Asia", "Europe", "European Union (27)", "European Union (28)",
        "North America", "South America", "Oceania", "Antarctica",
        "High-income countries", "Low-income countries", "Lower-middle-income countries",
        "Upper-middle-income countries", "International transport", "International aviation",
        "International shipping", "Asia (excl. China and India)", "Europe (excl. EU-27)",
        "Europe (excl. EU-28)", "North America (excl. USA)", "Non-OECD (GCP)", "OECD (GCP)",
        "G7", "G20", "European Union (Kyoto Protocol)",
    }
    f = raw[
        (raw["year"] >= 1990)
        & (raw["iso_code"].notna())
        & (raw["iso_code"].str.len() == 3)
        & (~raw["iso_code"].str.startswith("OWID"))
        & (~raw["country"].isin(aggregate_names))
    ].copy()
    return f[f["country"].isin(PROJECT_COUNTRIES)][["country", "year", "co2"]].sort_values(
        ["country", "year"]
    ).reset_index(drop=True)


if os.path.exists(FEATURES_PATH):
    features_df = pd.read_csv(FEATURES_PATH)[["country", "year", "co2"]]
else:
    print("ghg_features.csv not found — rebuilding...")
    features_df = build_feature_dataset()

latest_actual_year = int(features_df["year"].max())
print(f"Latest actual year available in data: {latest_actual_year}")


def get_country_series(country: str, year_min: int, year_max: int) -> pd.Series:
    sub = features_df[
        (features_df["country"] == country)
        & (features_df["year"] >= year_min)
        & (features_df["year"] <= year_max)
    ].dropna(subset=["co2"]).sort_values("year")
    idx = pd.date_range(start=f"{int(sub['year'].min())}-01-01", periods=len(sub), freq="YS")
    return pd.Series(sub["co2"].values, index=idx)


bau_forecast = {}  # country -> {year: co2}
for country in PROJECT_COUNTRIES:
    train_series = get_country_series(country, TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)
    if len(train_series) < 10:
        print(f"Skipping {country}: insufficient training data")
        continue
    fit = ExponentialSmoothing(train_series, trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
    steps = FORECAST_YEAR_MAX - TRAIN_YEAR_MAX  # 2019..2043
    forecast_mean = fit.get_forecast(steps=steps).predicted_mean
    years = np.arange(TRAIN_YEAR_MAX + 1, FORECAST_YEAR_MAX + 1)
    bau_forecast[country] = dict(zip(years, forecast_mean.values))

print(f"BAU forecasts ready for {len(bau_forecast)} countries.")

# %% [markdown]
# ## 5.1 Scenario Design
#
# - **Scenario A - Business as Usual (BAU):** No policy change — simply the
#   ETS(A,Ad,N) damped-trend forecast from Week 4, extrapolating each
#   country's historical 1990-2018 trend forward with no additional
#   adjustment.
# - **Scenario B - Moderate Mitigation:** Starting in 2025, the BAU forecast
#   for each year is scaled down by a **compounding 2%-per-year** reduction
#   factor — i.e. `co2_scenario(year) = co2_BAU(year) * (1 - 0.02)^(year - 2024)`
#   for `year >= 2025`. This represents a steady, moderate policy push (e.g.
#   incremental efficiency standards, gradual renewables build-out) layered
#   on top of whatever the BAU trend was already doing.
# - **Scenario C - Aggressive Mitigation:** Same mechanism, with a
#   **compounding 5%-per-year** reduction factor from 2025 onward —
#   representing a much stronger policy push (e.g. binding emissions caps,
#   rapid fossil-fuel phase-out).
#
# ### Basis and limitations (read before interpreting the numbers)
# - **Why compounding, not a straight-line subtraction:** a constant *rate*
#   (e.g. "2% per year") compounds multiplicatively over time — this is the
#   standard way emissions-reduction targets are expressed in policy
#   discourse (e.g. "cut emissions by X% per year"), so a compounding factor
#   applied to the BAU trajectory was used rather than an equal absolute (Mt)
#   cut every year.
# - **These are illustrative, not scientifically calibrated scenarios.** The
#   2% / 5% figures are simple, round, hypothetical policy-stringency
#   examples for comparison purposes — they are **not** derived from any
#   specific country's actual climate policy, NDC (Nationally Determined
#   Contribution) target, or sector-level decarbonisation modelling. Real
#   climate scenario work (e.g. IPCC pathways, IEA net-zero scenarios) is
#   built from detailed technology, energy-system, and economic models, not
#   a flat percentage applied to a statistical forecast.
# - **The same percentage is applied uniformly to every country**, which
#   ignores real-world differences in feasibility, starting emissions
#   intensity, economic structure, and existing policy commitments — a
#   country already near its practical mitigation ceiling and a
#   fast-industrializing country face very different realities for hitting
#   "5% per year," even though this model treats them identically.
# - **The scenarios modify only the BAU curve's scale, not its shape** — they
#   don't represent specific interventions (e.g. a coal phase-out year, an EV
#   mandate) and can't capture non-linear effects like a sudden shock,
#   tipping point, or policy cliff-edge.

# %% [markdown]
# ## 5.2 Scenario Calculation

# %%
scenario_rows = []
for country, forecast_by_year in bau_forecast.items():
    for year in range(SCENARIO_YEAR_MIN, SCENARIO_YEAR_MAX + 1):
        bau_val = forecast_by_year.get(year, np.nan)
        if pd.isna(bau_val):
            continue
        years_since_2024 = year - 2024
        moderate_val = bau_val * (1 - MODERATE_ANNUAL_REDUCTION) ** years_since_2024
        aggressive_val = bau_val * (1 - AGGRESSIVE_ANNUAL_REDUCTION) ** years_since_2024

        scenario_rows.append({"country": country, "year": year, "scenario": "BAU", "co2_projected": bau_val})
        scenario_rows.append({"country": country, "year": year, "scenario": "Moderate Mitigation", "co2_projected": moderate_val})
        scenario_rows.append({"country": country, "year": year, "scenario": "Aggressive Mitigation", "co2_projected": aggressive_val})

scenario_df = pd.DataFrame(scenario_rows).sort_values(["country", "scenario", "year"]).reset_index(drop=True)
print("Scenario projections shape:", scenario_df.shape)
print(scenario_df.head(12).to_string(index=False))

SCENARIO_CSV_PATH = os.path.join(OUTPUT_DIR, "scenario_projections.csv")
scenario_df.to_csv(SCENARIO_CSV_PATH, index=False)
print(f"\nSaved {SCENARIO_CSV_PATH}")

# %% [markdown]
# ### Committing `scenario_projections.csv` to GitHub
#
# This sandboxed environment has no network access and isn't connected to a
# git remote, so the commit/push step can't be executed from here. Run the
# following locally, from the folder containing `scenario_projections.csv`
# (adjust the remote/branch to your repo):
#
# ```bash
# git add scenario_projections.csv
# git commit -m "Add Week 5 scenario projections (BAU / Moderate / Aggressive, 2025-2040)"
# git push origin main
# ```

# %% [markdown]
# ## 5.3 Scenario Visualisations

# %%
def get_full_scenario_series(country: str) -> pd.DataFrame:
    """BAU/Moderate/Aggressive values for CHART_YEAR_MIN..SCENARIO_YEAR_MAX.
    Before 2025 all three scenarios equal the BAU forecast (no divergence yet)."""
    rows = []
    forecast_by_year = bau_forecast[country]
    for year in range(CHART_YEAR_MIN, SCENARIO_YEAR_MAX + 1):
        bau_val = forecast_by_year.get(year, np.nan)
        if year < SCENARIO_YEAR_MIN:
            mod_val, agg_val = bau_val, bau_val
        else:
            years_since_2024 = year - 2024
            mod_val = bau_val * (1 - MODERATE_ANNUAL_REDUCTION) ** years_since_2024
            agg_val = bau_val * (1 - AGGRESSIVE_ANNUAL_REDUCTION) ** years_since_2024
        rows.append({"year": year, "BAU": bau_val, "Moderate Mitigation": mod_val, "Aggressive Mitigation": agg_val})
    return pd.DataFrame(rows)


def plot_country_scenarios(country: str, save: bool = True):
    hist = features_df[
        (features_df["country"] == country)
        & (features_df["year"] >= 1990)
        & (features_df["year"] <= min(PLOT_HIST_YEAR_MAX, latest_actual_year))
    ].sort_values("year")
    benchmark_1990 = hist[hist["year"] == 1990]["co2"]
    benchmark_1990 = benchmark_1990.values[0] if not benchmark_1990.empty else np.nan

    scenarios = get_full_scenario_series(country)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(hist["year"], hist["co2"], color="grey", linewidth=1.5, label="Historical actuals (1990-2024)")

    for scenario_name, color in COLORS.items():
        ax.plot(scenarios["year"], scenarios[scenario_name], color=color, linewidth=2, label=scenario_name)

    if pd.notna(benchmark_1990):
        ax.axhline(benchmark_1990, color="black", linestyle=":", linewidth=1,
                    label=f"1990 level ({benchmark_1990:.0f} Mt)")

    ax.set_title(f"CO2 Mitigation Scenarios — {country} (2020-2040)")
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2 emissions (million tonnes)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        safe_name = country.lower().replace(" ", "_")
        fig.savefig(os.path.join(OUTPUT_DIR, f"scenarios_{safe_name}.png"), dpi=150)
    plt.show()


for country in PROJECT_COUNTRIES:
    if country in bau_forecast:
        plot_country_scenarios(country)

# %%
# --- Global aggregate chart: sum of all 10 countries' projections per scenario ---
global_rows = []
for year in range(CHART_YEAR_MIN, SCENARIO_YEAR_MAX + 1):
    totals = {"BAU": 0.0, "Moderate Mitigation": 0.0, "Aggressive Mitigation": 0.0}
    for country in bau_forecast:
        s = get_full_scenario_series(country)
        row = s[s["year"] == year]
        if row.empty:
            continue
        for scenario_name in totals:
            totals[scenario_name] += row[scenario_name].values[0]
    global_rows.append({"year": year, **totals})

global_df = pd.DataFrame(global_rows)

fig, ax = plt.subplots(figsize=(11, 5.5))
for scenario_name, color in COLORS.items():
    ax.plot(global_df["year"], global_df[scenario_name], color=color, linewidth=2.5, label=scenario_name)
ax.set_title("Aggregate CO2 Emissions Across 10 Project Countries — Scenario Comparison")
ax.set_xlabel("Year")
ax.set_ylabel("Total CO2 emissions (million tonnes)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "scenarios_global_aggregate.png"), dpi=150)
plt.show()

# %% [markdown]
# ## 5.4 Impact Summary

# %%
cumulative_rows = []
for country in bau_forecast:
    country_scenarios = scenario_df[scenario_df["country"] == country]
    for scenario_name in ["BAU", "Moderate Mitigation", "Aggressive Mitigation"]:
        total = country_scenarios[country_scenarios["scenario"] == scenario_name]["co2_projected"].sum()
        cumulative_rows.append({"country": country, "scenario": scenario_name, "cumulative_co2_2025_2040": total})

cumulative_df = pd.DataFrame(cumulative_rows)
cumulative_pivot = cumulative_df.pivot(index="country", columns="scenario", values="cumulative_co2_2025_2040")
cumulative_pivot = cumulative_pivot[["BAU", "Moderate Mitigation", "Aggressive Mitigation"]]
print("Cumulative CO2 emissions, 2025-2040 (million tonnes):")
print(cumulative_pivot.round(0))

cumulative_pivot.to_csv(os.path.join(OUTPUT_DIR, "cumulative_emissions_by_scenario.csv"))

# %%
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(cumulative_pivot.index))
width = 0.25

for i, (scenario_name, color) in enumerate(COLORS.items()):
    ax.bar(x + (i - 1) * width, cumulative_pivot[scenario_name], width, label=scenario_name, color=color)

ax.set_xticks(x)
ax.set_xticklabels(cumulative_pivot.index, rotation=45, ha="right")
ax.set_ylabel("Cumulative CO2 emissions, 2025-2040 (million tonnes)")
ax.set_title("Cumulative CO2 Emissions by Country and Scenario (2025-2040)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cumulative_emissions_grouped_bar.png"), dpi=150)
plt.show()

# %%
cumulative_pivot["BAU_minus_Aggressive"] = cumulative_pivot["BAU"] - cumulative_pivot["Aggressive Mitigation"]
biggest_absolute_savers = cumulative_pivot["BAU_minus_Aggressive"].sort_values(ascending=False)
print("Countries ranked by absolute cumulative CO2 avoided under Aggressive Mitigation (vs BAU):")
print(biggest_absolute_savers.round(0))

# %% [markdown]
# ### Interpretation
#
# The countries with the largest *absolute* emissions avoided under
# Aggressive Mitigation are, unsurprisingly, whichever of the 10 countries
# have the largest BAU forecasts to begin with (typically China and the
# United States) — a fixed percentage cut naturally removes more total
# tonnes from a larger baseline. Measured in *relative* terms, however, every
# country benefits identically by construction, since the 5% compounding
# reduction is applied uniformly — this scenario design mechanically cannot
# show one country mitigating "more effectively" than another in percentage
# terms, only in absolute tonnes. The countries whose BAU trend is already
# flattest or declining (e.g. UK, Germany, Japan) end up furthest below their
# own 1990 benchmark line even under Moderate Mitigation, while
# fast-growing-BAU countries (e.g. India) still likely remain above their
# 1990 level under all three scenarios within this horizon — meaning
# aggressive mitigation slows their growth but may not be enough, on its own
# and at this stylised rate, to bring them back to a 1990 baseline by 2040.
# This is a direct consequence of the scenario design rather than a
# real-world policy insight, and reinforces the 5.1 caveat that these
# percentages are illustrative, not calibrated to each country's actual
# mitigation potential.

# %% [markdown]
# ## Notes on running this script
# - Requires: `pandas`, `numpy`, `matplotlib`, `statsmodels`.
# - Rebuilds ETS BAU forecasts from `ghg_features.csv` (or from scratch if
#   that's missing) so it can run standalone.
# - Outputs written to `outputs/`: `scenario_projections.csv`, one
#   `scenarios_<country>.png` per country, `scenarios_global_aggregate.png`,
#   `cumulative_emissions_by_scenario.csv`,
#   `cumulative_emissions_grouped_bar.png`.
# - The GitHub commit step must be run locally — see the markdown cell above
#   for the exact commands.

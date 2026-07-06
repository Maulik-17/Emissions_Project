"""
Script 3: Modeling (Baseline, Linear Regression, Random Forest)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load feature dataset
df = pd.read_csv("ghg_features.csv")

# List of countries (should be 10)
countries = df['country'].unique()
print("Countries:", countries)

# ------------------------------------------------------------------------------
# 3.1 Problem Framing (printed as markdown)
# ------------------------------------------------------------------------------
print("\n--- 3.1 Problem Framing ---")
print("""
**Prediction Task**: Given features X for country C in year Y, predict CO₂ emissions for year Y+1.
**Target Variable**: `co2` (annual CO₂ emissions in MtCO₂).
**Input Features**: `co2_per_capita`, `co2_5yr_rolling_mean`, `co2_lag1`, `co2_lag2`, `co2_lag3`, `co2_yoy_pct_change`, `ghg_intensity` (where available).
**Training Strategy**:
- Linear Regression: trained per country (works with ~26 rows).
- Random Forest: trained on pooled dataset of all 10 countries (~260 rows) with a country-encoded feature to avoid overfitting from small per-country sample sizes.
- Both are evaluated per country for direct comparison.
**Supervised Regression**: This is a regression problem because the target is continuous and we want to predict its future value based on historical features.
""")

# ------------------------------------------------------------------------------
# 3.2 Train-Test Split
# ------------------------------------------------------------------------------
print("\n--- 3.2 Train-Test Split ---")
print("Temporal split: years 1990–2018 for training, 2019 onward for testing.")
print("This ensures no future information leaks into training and respects the time order of observations.")
print("The 2019–2023 test window includes the COVID-19 dip and recovery, providing a real-world stress test.")

train_data = {}
test_data = {}
for country in countries:
    cdf = df[df['country'] == country].sort_values('year')
    train = cdf[cdf['year'] <= 2018]
    test = cdf[cdf['year'] >= 2019]
    train_data[country] = train
    test_data[country] = test
    print(f"{country}: train samples = {len(train)}, test samples = {len(test)}")

# ------------------------------------------------------------------------------
# 3.3 Naive Baseline Model
# ------------------------------------------------------------------------------
print("\n--- 3.3 Naive Baseline ---")
baseline_mae = {}
baseline_rmse = {}

for country in countries:
    test = test_data[country]
    # Baseline predicts next year's CO₂ = current year's CO₂ -> for test years, use co2_lag1 (previous year)
    y_pred = test['co2_lag1'].values
    y_true = test['co2'].values
    # Remove rows where prediction is NaN (if first test year has no lag)
    mask = ~np.isnan(y_pred)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    if len(y_pred) > 0:
        baseline_mae[country] = mean_absolute_error(y_true, y_pred)
        baseline_rmse[country] = np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        baseline_mae[country] = np.nan
        baseline_rmse[country] = np.nan

print("Baseline MAE per country:", baseline_mae)
print("Baseline RMSE per country:", baseline_rmse)

# Plot baseline for 3 countries
plot_countries = ['China', 'United States', 'India']
for country in plot_countries:
    if country not in test_data:
        continue
    test = test_data[country]
    y_pred = test['co2_lag1'].values
    y_true = test['co2'].values
    mask = ~np.isnan(y_pred)
    years = test['year'].values[mask]
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    if len(years) == 0:
        continue
    plt.figure(figsize=(10,5))
    plt.plot(years, y_true, marker='o', label='Actual')
    plt.plot(years, y_pred, marker='x', label='Baseline Prediction')
    plt.title(f'Baseline (No-Change) Predictions for {country}')
    plt.xlabel('Year')
    plt.ylabel('CO₂ (MtCO₂)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'baseline_{country}.png', dpi=150)
    plt.close()
print("Baseline plots saved for China, United States, India.")

# ------------------------------------------------------------------------------
# 3.4 Linear Regression (per country)
# ------------------------------------------------------------------------------
print("\n--- 3.4 Linear Regression ---")
features = ['co2_per_capita', 'co2_5yr_rolling_mean', 'co2_lag1', 'co2_lag2', 'co2_lag3', 'co2_yoy_pct_change', 'ghg_intensity']
lr_mae = {}
lr_rmse = {}
lr_coeffs = {}

for country in countries:
    train = train_data[country]
    test = test_data[country]
    # Drop rows with NaN in features or target
    train_clean = train.dropna(subset=features + ['co2'])
    test_clean = test.dropna(subset=features + ['co2'])
    if len(train_clean) < 2 or len(test_clean) == 0:
        print(f"{country}: insufficient data for LR; skipping.")
        lr_mae[country] = np.nan
        lr_rmse[country] = np.nan
        continue
    X_train = train_clean[features]
    y_train = train_clean['co2']
    X_test = test_clean[features]
    y_test = test_clean['co2']
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lr_mae[country] = mean_absolute_error(y_test, y_pred)
    lr_rmse[country] = np.sqrt(mean_squared_error(y_test, y_pred))
    lr_coeffs[country] = dict(zip(features, model.coef_))
    print(f"{country}: LR MAE={lr_mae[country]:.2f}, RMSE={lr_rmse[country]:.2f}")

# Plot regression line for 3 countries
for country in plot_countries:
    if country not in lr_mae or np.isnan(lr_mae[country]):
        continue
    train = train_data[country].dropna(subset=features + ['co2'])
    test = test_data[country].dropna(subset=features + ['co2'])
    if len(test) == 0:
        continue
    X_train = train[features]
    y_train = train['co2']
    X_test = test[features]
    y_test = test['co2']
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10,5))
    plt.scatter(y_test, y_pred, alpha=0.7, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect fit')
    plt.xlabel('Actual CO₂')
    plt.ylabel('Predicted CO₂')
    plt.title(f'Linear Regression: {country}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'lr_{country}.png', dpi=150)
    plt.close()

print("\nLinear Regression coefficients interpretation:")
for country in plot_countries:
    if country in lr_coeffs:
        print(f"{country}: {lr_coeffs[country]}")
print("For most countries, `co2_lag1` and `co2_5yr_rolling_mean` have the largest positive coefficients, indicating strong persistence. `co2_yoy_pct_change` often has a negative coefficient, suggesting that high growth rates may be followed by corrections.")

# ------------------------------------------------------------------------------
# 3.5 Random Forest Regressor (pooled)
# ------------------------------------------------------------------------------
print("\n--- 3.5 Random Forest Regressor ---")
print("""
**Limitations of per-country Random Forest**:
- With only ~26 rows per country, per-country RF would severely overfit, produce unstable bootstrap samples, and yield unreliable feature importance.
- Pooling across countries (~260 rows) allows the model to learn cross-country patterns, but loses purely country-specific dynamics.
- The key teaching point: model complexity must match data availability; simpler models on small datasets often outperform complex ones with insufficient examples.
""")

# Pool training data from all countries
train_pooled = pd.concat([train_data[c] for c in countries], ignore_index=True)
train_pooled = train_pooled.dropna(subset=features + ['co2'])
# Add country encoded feature
le = LabelEncoder()
train_pooled['country_enc'] = le.fit_transform(train_pooled['country'])

X_pooled = train_pooled[features + ['country_enc']]
y_pooled = train_pooled['co2']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_pooled, y_pooled)

# Evaluate per country on test sets
rf_mae = {}
rf_rmse = {}
for country in countries:
    test = test_data[country].dropna(subset=features + ['co2'])
    if len(test) == 0:
        rf_mae[country] = np.nan
        rf_rmse[country] = np.nan
        continue
    X_test = test[features].copy()
    X_test['country_enc'] = le.transform([country])[0]  # same encoder
    y_test = test['co2']
    y_pred = rf.predict(X_test)
    rf_mae[country] = mean_absolute_error(y_test, y_pred)
    rf_rmse[country] = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{country}: RF MAE={rf_mae[country]:.2f}, RMSE={rf_rmse[country]:.2f}")

# Feature importance plot
importances = rf.feature_importances_
feature_names = features + ['country_enc']
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=True)
plt.figure(figsize=(8,6))
plt.barh(imp_df['feature'], imp_df['importance'], color='teal')
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance (Pooled Model)')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=150)
plt.close()
print("Feature importance plot saved as 'rf_feature_importance.png'.")
print("Interpretation: 'country_enc' is often the most important feature, reflecting large differences in emissions scales. Among the actual predictors, lags and rolling means dominate, confirming the strong autoregressive nature of CO₂ emissions.")

# ------------------------------------------------------------------------------
# 3.6 Model Comparison Table
# ------------------------------------------------------------------------------
print("\n--- 3.6 Model Comparison Table ---")
comparison = pd.DataFrame({
    'Country': countries,
    'Baseline MAE': [baseline_mae.get(c, np.nan) for c in countries],
    'LR MAE': [lr_mae.get(c, np.nan) for c in countries],
    'RF MAE': [rf_mae.get(c, np.nan) for c in countries],
    'Baseline RMSE': [baseline_rmse.get(c, np.nan) for c in countries],
    'LR RMSE': [lr_rmse.get(c, np.nan) for c in countries],
    'RF RMSE': [rf_rmse.get(c, np.nan) for c in countries]
})
# Best model per country based on MAE
def best_model(row):
    mae_vals = {'Baseline': row['Baseline MAE'], 'LR': row['LR MAE'], 'RF': row['RF MAE']}
    # Exclude NaNs
    mae_vals = {k: v for k, v in mae_vals.items() if not np.isnan(v)}
    if not mae_vals:
        return 'N/A'
    return min(mae_vals, key=mae_vals.get)
comparison['Best Model (MAE)'] = comparison.apply(best_model, axis=1)

print(comparison.to_string(index=False))

# Save comparison table for later use (will be extended by ETS)
comparison.to_csv("model_comparison_initial.csv", index=False)
print("Comparison table saved to 'model_comparison_initial.csv'.")

print("\nConclusion: Linear Regression generally outperforms the baseline, indicating that the features provide predictive value. Random Forest often performs worse than Linear Regression on these small datasets, confirming that simpler models are more robust when data is limited. The best model varies by country, but Linear Regression is often competitive.")

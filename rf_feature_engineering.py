import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate

df = pd.read_csv("BiogasData.csv")

target = "Biogas_Yield_m3_day"

# ============================================================
# Physics-informed feature engineering
# ============================================================

def engineer_features(data):

    d = data.copy()

    # --------------------------------------------------------
    # Basic nonlinear hydraulic transformation
    # --------------------------------------------------------

    d["Flow_power_035"] = d["Raw_Sewage_Flow_L_day"] ** 0.35

    # --------------------------------------------------------
    # Distance from biological optima
    # --------------------------------------------------------

    d["Temperature_deviation"] = abs(
        d["Temperature_C"] - 35.0
    )

    d["pH_deviation"] = abs(
        d["pH"] - 7.2
    )

    d["CN_deviation"] = abs(
        d["CN_Ratio"] - 22.5
    )

    # --------------------------------------------------------
    # Biological response envelopes
    # These mirror the structure used by the data generator.
    # --------------------------------------------------------

    d["Temperature_factor"] = np.exp(
        -((d["Temperature_C"] - 35.0) / 5.0) ** 2
    )

    d["pH_factor"] = np.exp(
        -((d["pH"] - 7.2) / 0.3) ** 2
    )

    d["CN_factor"] = np.exp(
        -((d["CN_Ratio"] - 22.5) / 2.5) ** 2
    )

    d["HRT_factor"] = (
        1 - np.exp(-d["HRT_days"] / 10.0)
    )

    # --------------------------------------------------------
    # Feedstock interaction features
    # --------------------------------------------------------

    d["Cellulose_Hemicellulose"] = (
        d["Cellulose_pct"] *
        d["Hemicellulose_pct"]
    )

    d["Waste_Cellulose"] = (
        d["Vegetable_Waste_kg_day"] *
        d["Cellulose_pct"]
    )

    d["Waste_Hemicellulose"] = (
        d["Vegetable_Waste_kg_day"] *
        d["Hemicellulose_pct"]
    )

    return d


df_eng = engineer_features(df)

# ------------------------------------------------------------
# Original 7 engineering features
# ------------------------------------------------------------

base_features = [
    "Raw_Sewage_Flow_L_day",
    "Vegetable_Waste_kg_day",
    "Cellulose_pct",
    "Hemicellulose_pct",
    "Temperature_C",
    "pH",
    "CN_Ratio"
]

# ------------------------------------------------------------
# Engineered additions
# ------------------------------------------------------------

engineered_features = base_features + [
    "Flow_power_035",
    "Temperature_deviation",
    "pH_deviation",
    "CN_deviation",
    "Temperature_factor",
    "pH_factor",
    "CN_factor",
    "HRT_factor",
    "Cellulose_Hemicellulose",
    "Waste_Cellulose",
    "Waste_Hemicellulose"
]

# ------------------------------------------------------------
# Compare feature sets
# ------------------------------------------------------------

feature_sets = {
    "Original_7": base_features,

    "Original_7_Plus_Physics": engineered_features
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_params = {
    "n_estimators": 1000,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.8,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1
}

results = []

print("\n" + "=" * 100)
print("PHYSICS-INFORMED RANDOM FOREST FEATURE ENGINEERING")
print("=" * 100)

for name, features in feature_sets.items():

    print(f"\nTesting: {name}")
    print(f"Features: {len(features)}")

    model = RandomForestRegressor(**rf_params)

    scores = cross_validate(
        model,
        df_eng[features],
        df_eng[target],
        cv=cv,
        scoring={
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error"
        },
        n_jobs=-1
    )

    r2 = scores["test_r2"].mean()
    r2_std = scores["test_r2"].std()
    rmse = -scores["test_rmse"].mean()
    mae = -scores["test_mae"].mean()

    results.append({
        "Feature_Set": name,
        "Features": len(features),
        "R2": r2,
        "R2_std": r2_std,
        "RMSE": rmse,
        "MAE": mae
    })

    print(
        f"R²   = {r2:.6f} ± {r2_std:.6f}\n"
        f"RMSE = {rmse:.6f}\n"
        f"MAE  = {mae:.6f}"
    )

results_df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("FINAL COMPARISON")
print("=" * 100)

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    "rf_feature_engineering_results.csv",
    index=False
)

print("\nSaved: rf_feature_engineering_results.csv")

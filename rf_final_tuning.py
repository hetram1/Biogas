import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate

df = pd.read_csv("BiogasData.csv")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["Temperature_deviation"] = abs(
    df["Temperature_C"] - 35.0
)

df["pH_deviation"] = abs(
    df["pH"] - 7.2
)

df["CN_deviation"] = abs(
    df["CN_Ratio"] - 22.5
)

features = [
    "Raw_Sewage_Flow_L_day",
    "Vegetable_Waste_kg_day",
    "Cellulose_pct",
    "Hemicellulose_pct",
    "Temperature_C",
    "pH",
    "CN_Ratio",
    "Temperature_deviation",
    "pH_deviation",
    "CN_deviation"
]

X = df[features]
y = df["Biogas_Yield_m3_day"]

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ============================================================
# FINAL RF SEARCH SPACE
# ============================================================

configs = []

for max_depth in [15, 20, 25, 30, None]:

    for min_samples_split in [2, 5, 10]:

        for min_samples_leaf in [1, 2, 4]:

            for max_features in [0.6, 0.8, 1.0]:

                configs.append({
                    "n_estimators": 1000,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                    "bootstrap": True,
                    "criterion": "squared_error"
                })

# Add selected alternative criteria
for criterion in ["friedman_mse", "absolute_error", "poisson"]:

    configs.append({
        "n_estimators": 1000,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 0.8,
        "bootstrap": True,
        "criterion": criterion
    })


results = []

print("\n" + "=" * 110)
print("FINAL RANDOM FOREST TUNING — 10 PHYSICS-INFORMED FEATURES")
print("=" * 110)

print(f"\nTotal configurations: {len(configs)}")

for i, params in enumerate(configs, 1):

    model = RandomForestRegressor(
        **params,
        random_state=42,
        n_jobs=-1
    )

    scores = cross_validate(
        model,
        X,
        y,
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
        **params,
        "R2": r2,
        "R2_std": r2_std,
        "RMSE": rmse,
        "MAE": mae
    })

    if i % 10 == 0:
        print(f"Completed {i}/{len(configs)}")


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["R2", "MAE"],
    ascending=[False, True]
)

print("\n" + "=" * 110)
print("TOP 15 CONFIGURATIONS")
print("=" * 110)

print(
    results_df[
        [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "criterion",
            "R2",
            "R2_std",
            "RMSE",
            "MAE"
        ]
    ].head(15).to_string(index=False)
)

results_df.to_csv(
    "rf_final_tuning_results.csv",
    index=False
)

print("\nSaved: rf_final_tuning_results.csv")

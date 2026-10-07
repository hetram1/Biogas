import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate

df = pd.read_csv("BiogasData.csv")

features = [
    "Raw_Sewage_Flow_L_day",
    "Vegetable_Waste_kg_day",
    "Cellulose_pct",
    "Hemicellulose_pct",
    "Temperature_C",
    "pH",
    "CN_Ratio"
]

X = df[features]
y = df["Biogas_Yield_m3_day"]

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

configs = [
    # Baseline
    {
        "n_estimators": 500,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "bootstrap": True
    },

    # More trees
    {
        "n_estimators": 1000,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "bootstrap": True
    },

    {
        "n_estimators": 1500,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "bootstrap": True
    },

    # Deeper trees
    {
        "n_estimators": 1000,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "bootstrap": True
    },

    {
        "n_estimators": 1000,
        "max_depth": 25,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "bootstrap": True
    },

    {
        "n_estimators": 1000,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "bootstrap": True
    },

    # Regularization
    {
        "n_estimators": 1000,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "bootstrap": True
    },

    {
        "n_estimators": 1000,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 4,
        "max_features": 1.0,
        "bootstrap": True
    },

    {
        "n_estimators": 1000,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "bootstrap": True
    },

    # Feature subsampling
    {
        "n_estimators": 1000,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True
    },

    {
        "n_estimators": 1000,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 0.7,
        "bootstrap": True
    },

    {
        "n_estimators": 1000,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 0.8,
        "bootstrap": True
    },

    # More aggressive regularization
    {
        "n_estimators": 1000,
        "max_depth": 25,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "bootstrap": True
    },

    {
        "n_estimators": 1000,
        "max_depth": 30,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "bootstrap": True
    },

    # Extremely randomized bootstrap behaviour
    {
        "n_estimators": 1000,
        "max_depth": 25,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "bootstrap": False
    },
]

results = []

print("\n" + "=" * 110)
print("RANDOM FOREST HYPERPARAMETER OPTIMIZATION")
print("=" * 110)

for i, params in enumerate(configs, 1):

    print(f"\nTesting configuration {i}/{len(configs)}...")

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

    print(
        f"R²={r2:.6f} | "
        f"RMSE={rmse:.6f} | "
        f"MAE={mae:.6f}"
    )

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n" + "=" * 110)
print("TOP RANDOM FOREST CONFIGURATIONS")
print("=" * 110)

print(
    results_df[
        [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "R2",
            "R2_std",
            "RMSE",
            "MAE"
        ]
    ].head(10).to_string(index=False)
)

results_df.to_csv(
    "rf_hyperparameter_results.csv",
    index=False
)

print("\nSaved: rf_hyperparameter_results.csv")

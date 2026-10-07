import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate

df = pd.read_csv("BiogasData.csv")

target = "Biogas_Yield_m3_day"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

d = df.copy()

d["Flow_power_035"] = (
    d["Raw_Sewage_Flow_L_day"] ** 0.35
)

d["Temperature_deviation"] = abs(
    d["Temperature_C"] - 35.0
)

d["pH_deviation"] = abs(
    d["pH"] - 7.2
)

d["CN_deviation"] = abs(
    d["CN_Ratio"] - 22.5
)

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


# ============================================================
# BASE FEATURES
# ============================================================

base = [
    "Raw_Sewage_Flow_L_day",
    "Vegetable_Waste_kg_day",
    "Cellulose_pct",
    "Hemicellulose_pct",
    "Temperature_C",
    "pH",
    "CN_Ratio"
]


# ============================================================
# GROUPS
# ============================================================

flow_group = [
    "Flow_power_035"
]

biology_group = [
    "Temperature_factor",
    "pH_factor",
    "CN_factor",
    "HRT_factor"
]

deviation_group = [
    "Temperature_deviation",
    "pH_deviation",
    "CN_deviation"
]

interaction_group = [
    "Cellulose_Hemicellulose",
    "Waste_Cellulose",
    "Waste_Hemicellulose"
]


# ============================================================
# FEATURE SETS
# ============================================================

feature_sets = {

    "Original_7":
        base,

    "Base + Flow":
        base + flow_group,

    "Base + Biology":
        base + biology_group,

    "Base + Deviations":
        base + deviation_group,

    "Base + Interactions":
        base + interaction_group,

    "Base + Flow + Biology":
        base + flow_group + biology_group,

    "Base + Flow + Biology + Deviations":
        base + flow_group + biology_group + deviation_group,

    "Base + Flow + Biology + Interactions":
        base + flow_group + biology_group + interaction_group,

    "ALL_18":
        base
        + flow_group
        + biology_group
        + deviation_group
        + interaction_group
}


# ============================================================
# MODEL
# ============================================================

rf = RandomForestRegressor(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.8,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# RUN
# ============================================================

results = []

print("\n" + "=" * 110)
print("RANDOM FOREST FEATURE ABLATION")
print("=" * 110)

for name, features in feature_sets.items():

    print(f"\nTesting: {name}")

    scores = cross_validate(
        rf,
        d[features],
        d[target],
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
        "Feature_Count": len(features),
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
    "R2",
    ascending=False
)

print("\n" + "=" * 110)
print("RANKED RESULTS")
print("=" * 110)

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    "rf_feature_ablation_results.csv",
    index=False
)

print("\nSaved: rf_feature_ablation_results.csv")

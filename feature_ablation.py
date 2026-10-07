import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

df = pd.read_csv("BiogasData.csv")

target = "Biogas_Yield_m3_day"

all_features = [
    "Digester_Volume_m3",
    "Raw_Sewage_Flow_L_day",
    "Vegetable_Waste_kg_day",
    "Cellulose_pct",
    "Hemicellulose_pct",
    "Temperature_C",
    "pH",
    "HRT_days",
    "CN_Ratio"
]

feature_sets = {
    "All Features": all_features,

    "Remove Volume": [
        f for f in all_features
        if f != "Digester_Volume_m3"
    ],

    "Remove Flow": [
        f for f in all_features
        if f != "Raw_Sewage_Flow_L_day"
    ],

    "Remove Volume + Flow": [
        f for f in all_features
        if f not in [
            "Digester_Volume_m3",
            "Raw_Sewage_Flow_L_day"
        ]
    ],

    "Remove Cellulose": [
        f for f in all_features
        if f != "Cellulose_pct"
    ],

    "Remove Hemicellulose": [
        f for f in all_features
        if f != "Hemicellulose_pct"
    ],
}

models = {
    "XGBoost": XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ),

    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(
            kernel="rbf",
            C=100,
            gamma="scale",
            epsilon=0.001
        ))
    ])
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rows = []

for feature_name, features in feature_sets.items():

    X = df[features]
    y = df[target]

    for model_name, model in models.items():

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

        rows.append({
            "Feature_Set": feature_name,
            "Model": model_name,
            "R2_Mean": scores["test_r2"].mean(),
            "RMSE_Mean": -scores["test_rmse"].mean(),
            "MAE_Mean": -scores["test_mae"].mean()
        })

results = pd.DataFrame(rows)

results = results.sort_values(
    ["Model", "R2_Mean"],
    ascending=[True, False]
)

print("\n" + "=" * 100)
print("FEATURE ABLATION — 5-FOLD CROSS VALIDATION")
print("=" * 100)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

results.to_csv("feature_ablation.csv", index=False)

print("\nSaved: feature_ablation.csv")

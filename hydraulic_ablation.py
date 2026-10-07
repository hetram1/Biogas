import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

df = pd.read_csv("BiogasData.csv")

target = "Biogas_Yield_m3_day"

base_features = [
    "Vegetable_Waste_kg_day",
    "Cellulose_pct",
    "Hemicellulose_pct",
    "Temperature_C",
    "pH",
    "CN_Ratio"
]

hydraulic_sets = {
    "Flow + HRT": base_features + [
        "Raw_Sewage_Flow_L_day",
        "HRT_days"
    ],

    "Volume + HRT": base_features + [
        "Digester_Volume_m3",
        "HRT_days"
    ],

    "Flow Only": base_features + [
        "Raw_Sewage_Flow_L_day"
    ],

    "Volume Only": base_features + [
        "Digester_Volume_m3"
    ],

    "Flow + Volume + HRT": base_features + [
        "Raw_Sewage_Flow_L_day",
        "Digester_Volume_m3",
        "HRT_days"
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
        n_jobs=1
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

results = []

for feature_set, features in hydraulic_sets.items():

    for model_name, model in models.items():

        print(f"Running: {model_name} — {feature_set}")

        scores = cross_validate(
            model,
            df[features],
            df[target],
            cv=cv,
            scoring={
                "r2": "r2",
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error"
            },
            n_jobs=1
        )

        results.append({
            "Feature_Set": feature_set,
            "Model": model_name,
            "R2": scores["test_r2"].mean(),
            "RMSE": -scores["test_rmse"].mean(),
            "MAE": -scores["test_mae"].mean()
        })

results = pd.DataFrame(results)

print("\n" + "=" * 90)
print("HYDRAULIC FEATURE ABLATION")
print("=" * 90)

print(
    results.sort_values(
        "R2",
        ascending=False
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

results.to_csv(
    "hydraulic_ablation.csv",
    index=False
)

print("\nSaved: hydraulic_ablation.csv")

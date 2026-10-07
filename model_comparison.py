import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

df = pd.read_csv("BiogasData.csv")

FEATURES = [
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

X = df[FEATURES]
y = df["Biogas_Yield_m3_day"]

models = {
    "XGBoost": XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=-1
    ),

    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.001))
    ]),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05,
        max_depth=5, random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=500, max_depth=15,
        random_state=42, n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500, max_depth=15,
        random_state=42, n_jobs=-1
    ),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(
            n_neighbors=10, weights="distance", p=2, n_jobs=-1
        ))
    ])
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "R2": "r2",
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error"
}

results = []

for name, model in models.items():
    print(f"Evaluating {name}...")

    scores = cross_validate(
        model, X, y, cv=kf,
        scoring=scoring, n_jobs=-1
    )

    r2 = scores["test_R2"]
    rmse = -scores["test_RMSE"]
    mae = -scores["test_MAE"]

    results.append({
        "Model": name,
        "R2 Mean": r2.mean(),
        "R2 Std": r2.std(),
        "RMSE Mean": rmse.mean(),
        "RMSE Std": rmse.std(),
        "MAE Mean": mae.mean(),
        "MAE Std": mae.std()
    })

results_df = pd.DataFrame(results).sort_values(
    "R2 Mean", ascending=False
)

print("\n" + "=" * 80)
print("MODEL COMPARISON — 5-FOLD CROSS VALIDATION")
print("=" * 80)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

results_df.to_csv("model_comparison.csv", index=False)

print("\nResults saved to: model_comparison.csv")
print(f"Best Model: {results_df.iloc[0]['Model']}")

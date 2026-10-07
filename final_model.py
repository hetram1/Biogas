import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("BiogasData.csv")


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def create_features(df):

    X = df[
        [
            "Raw_Sewage_Flow_L_day",
            "Vegetable_Waste_kg_day",
            "Cellulose_pct",
            "Hemicellulose_pct",
            "Temperature_C",
            "pH",
            "CN_Ratio",
        ]
    ].copy()

    X["Temperature_deviation"] = np.abs(
        X["Temperature_C"] - 35
    )

    X["pH_deviation"] = np.abs(
        X["pH"] - 7.2
    )

    X["CN_deviation"] = np.abs(
        X["CN_Ratio"] - 22.5
    )

    return X


X = create_features(df)
y = df["Biogas_Yield_m3_day"]


# ============================================================
# 3. UNTOUCHED TEST SET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("=" * 70)
print("FINAL RANDOM FOREST EVALUATION")
print("=" * 70)

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Features         : {X.shape[1]}")


# ============================================================
# 4. FINAL MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.8,
    criterion="poisson",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. TRAIN
# ============================================================

print("\nTraining final model...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 6. TEST PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. FINAL METRICS
# ============================================================

r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

mae = mean_absolute_error(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("FINAL TEST PERFORMANCE")
print("=" * 70)

print(f"\nR²   : {r2:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(index=False)
)


# ============================================================
# 9. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "biogas_model_final.pkl"
)


# ============================================================
# 10. SAVE TEST PREDICTIONS
# ============================================================

parity = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

parity.to_csv(
    "final_parity_data.csv",
    index=False
)


# ============================================================
# 11. SAVE FEATURE IMPORTANCE
# ============================================================

importance.to_csv(
    "final_feature_importance.csv",
    index=False
)


print("\nSaved:")
print("  biogas_model_final.pkl")
print("  final_parity_data.csv")
print("  final_feature_importance.csv")

print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETE")
print("=" * 70)

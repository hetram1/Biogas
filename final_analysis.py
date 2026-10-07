import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# LOAD FINAL TEST PREDICTIONS
# ============================================================

df = pd.read_csv("final_parity_data.csv")

actual = df["Actual"]
predicted = df["Predicted"]

df["Residual"] = actual - predicted
df["Absolute_Error"] = np.abs(df["Residual"])
df["Relative_Error_%"] = (
    np.abs(df["Residual"]) / np.maximum(actual, 1e-10)
) * 100


# ============================================================
# BASIC METRICS
# ============================================================

r2 = r2_score(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)

print("=" * 70)
print("FINAL RESIDUAL ANALYSIS")
print("=" * 70)

print(f"\nR²   : {r2:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")


# ============================================================
# RESIDUAL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("RESIDUAL STATISTICS")
print("=" * 70)

print(f"\nMean residual       : {df['Residual'].mean():.8f}")
print(f"Median residual     : {df['Residual'].median():.8f}")
print(f"Std residual        : {df['Residual'].std():.8f}")
print(f"Mean absolute error : {df['Absolute_Error'].mean():.8f}")


# ============================================================
# RELATIVE ERROR
# ============================================================

print("\n" + "=" * 70)
print("RELATIVE ERROR")
print("=" * 70)

print(f"\nMean relative error   : {df['Relative_Error_%'].mean():.2f}%")
print(f"Median relative error : {df['Relative_Error_%'].median():.2f}%")
print(f"90th percentile      : {df['Relative_Error_%'].quantile(0.90):.2f}%")
print(f"95th percentile      : {df['Relative_Error_%'].quantile(0.95):.2f}%")
print(f"Maximum               : {df['Relative_Error_%'].max():.2f}%")


# ============================================================
# WORST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("10 WORST PREDICTIONS")
print("=" * 70)

worst = df.nlargest(10, "Absolute_Error")

print(
    worst[
        [
            "Actual",
            "Predicted",
            "Residual",
            "Absolute_Error",
            "Relative_Error_%",
        ]
    ].to_string(index=False)
)


# ============================================================
# BEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("10 BEST PREDICTIONS")
print("=" * 70)

best = df.nsmallest(10, "Absolute_Error")

print(
    best[
        [
            "Actual",
            "Predicted",
            "Residual",
            "Absolute_Error",
            "Relative_Error_%",
        ]
    ].to_string(index=False)
)


# ============================================================
# BIAS CHECK
# ============================================================

positive = (df["Residual"] > 0).sum()
negative = (df["Residual"] < 0).sum()

print("\n" + "=" * 70)
print("BIAS CHECK")
print("=" * 70)

print(f"\nUnder-predictions : {positive}")
print(f"Over-predictions  : {negative}")

if df["Residual"].mean() > 0:
    print("Overall tendency  : slight UNDER-PREDICTION")
elif df["Residual"].mean() < 0:
    print("Overall tendency  : slight OVER-PREDICTION")
else:
    print("Overall tendency  : essentially unbiased")


# ============================================================
# YIELD RANGE ANALYSIS
# ============================================================

df["Yield_Band"] = pd.cut(
    actual,
    bins=5,
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

band_results = (
    df.groupby("Yield_Band", observed=True)
    .agg(
        Samples=("Actual", "count"),
        Actual_Mean=("Actual", "mean"),
        Predicted_Mean=("Predicted", "mean"),
        MAE=("Absolute_Error", "mean"),
        Mean_Relative_Error=("Relative_Error_%", "mean")
    )
)

print("\n" + "=" * 70)
print("ERROR BY YIELD RANGE")
print("=" * 70)

print("\n" + band_results.to_string())


# ============================================================
# SAVE ENRICHED ANALYSIS
# ============================================================

df.to_csv(
    "final_residual_analysis.csv",
    index=False
)

band_results.to_csv(
    "error_by_yield_band.csv"
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("\nfinal_residual_analysis.csv")
print("error_by_yield_band.csv")

print("\nAnalysis complete.")

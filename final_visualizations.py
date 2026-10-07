import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LOAD DATA
# ============================================================

parity = pd.read_csv("final_parity_data.csv")
importance = pd.read_csv("final_feature_importance.csv")

actual = parity["Actual"].values
predicted = parity["Predicted"].values
residual = actual - predicted


# ============================================================
# 1. ACTUAL vs PREDICTED
# ============================================================

plt.figure(figsize=(8, 7))

plt.scatter(
    actual,
    predicted,
    alpha=0.55,
    s=25
)

min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual Biogas Yield (m³/day)")
plt.ylabel("Predicted Biogas Yield (m³/day)")
plt.title("Random Forest — Actual vs Predicted")

plt.tight_layout()
plt.savefig(
    "final_parity_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. RESIDUAL DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    residual,
    bins=40,
    edgecolor="black",
    alpha=0.75
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2
)

plt.xlabel("Residual (Actual − Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")

plt.tight_layout()
plt.savefig(
    "final_residual_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. RESIDUAL vs PREDICTED
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    predicted,
    residual,
    alpha=0.55,
    s=25
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=2
)

plt.xlabel("Predicted Biogas Yield (m³/day)")
plt.ylabel("Residual (Actual − Predicted)")
plt.title("Residuals vs Predicted Yield")

plt.tight_layout()
plt.savefig(
    "final_residual_vs_prediction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. FEATURE IMPORTANCE
# ============================================================

importance = importance.sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(9, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.tight_layout()
plt.savefig(
    "final_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 65)
print("FINAL VISUALIZATIONS CREATED")
print("=" * 65)

print("\nCreated:")

print("  final_parity_plot.png")
print("  final_residual_distribution.png")
print("  final_residual_vs_prediction.png")
print("  final_feature_importance.png")

print("\nAll plots saved at 300 DPI.")

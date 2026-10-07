import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

df = pd.read_csv("BiogasData.csv")

target = "Biogas_Yield_m3_day"

# ---------------------------------------------------------
# Feature sets
# ---------------------------------------------------------

flow_features = [
    "Raw_Sewage_Flow_L_day"
]

engineering_features = [
    "Raw_Sewage_Flow_L_day",
    "Vegetable_Waste_kg_day",
    "Cellulose_pct",
    "Hemicellulose_pct",
    "Temperature_C",
    "pH",
    "HRT_days",
    "CN_Ratio"
]

# ---------------------------------------------------------
# Train both models
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df,
    df[target],
    test_size=0.20,
    random_state=42
)

flow_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVR(
        kernel="rbf",
        C=100,
        gamma="scale",
        epsilon=0.001
    ))
])

engineering_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVR(
        kernel="rbf",
        C=100,
        gamma="scale",
        epsilon=0.001
    ))
])

flow_model.fit(
    X_train[flow_features],
    y_train
)

engineering_model.fit(
    X_train[engineering_features],
    y_train
)

# ---------------------------------------------------------
# Baseline operating condition
# ---------------------------------------------------------

baseline = {
    "Raw_Sewage_Flow_L_day": 6000,
    "Vegetable_Waste_kg_day": 150,
    "Cellulose_pct": 42.5,
    "Hemicellulose_pct": 22,
    "Temperature_C": 35,
    "pH": 7.2,
    "HRT_days": 40,
    "CN_Ratio": 22.5
}

def predict(models_data):
    flow_pred = flow_model.predict(
        pd.DataFrame([models_data])[flow_features]
    )[0]

    engineering_pred = engineering_model.predict(
        pd.DataFrame([models_data])[engineering_features]
    )[0]

    return flow_pred, engineering_pred


# ---------------------------------------------------------
# 1. Temperature sensitivity
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("TEMPERATURE SENSITIVITY")
print("=" * 80)

for temperature in [25, 28, 30, 32, 35, 37, 40]:

    x = baseline.copy()
    x["Temperature_C"] = temperature

    flow_pred, engineering_pred = predict(x)

    print(
        f"T={temperature:2d}°C | "
        f"Flow-only={flow_pred:.6f} | "
        f"Engineering={engineering_pred:.6f}"
    )


# ---------------------------------------------------------
# 2. pH sensitivity
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("pH SENSITIVITY")
print("=" * 80)

for ph in [6.8, 6.9, 7.0, 7.2, 7.4, 7.5, 7.6]:

    x = baseline.copy()
    x["pH"] = ph

    flow_pred, engineering_pred = predict(x)

    print(
        f"pH={ph:.1f} | "
        f"Flow-only={flow_pred:.6f} | "
        f"Engineering={engineering_pred:.6f}"
    )


# ---------------------------------------------------------
# 3. C/N sensitivity
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("C/N SENSITIVITY")
print("=" * 80)

for cn in [20, 21, 22, 22.5, 23, 24, 25]:

    x = baseline.copy()
    x["CN_Ratio"] = cn

    flow_pred, engineering_pred = predict(x)

    print(
        f"C/N={cn:4.1f} | "
        f"Flow-only={flow_pred:.6f} | "
        f"Engineering={engineering_pred:.6f}"
    )


# ---------------------------------------------------------
# 4. Vegetable waste sensitivity
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("VEGETABLE WASTE SENSITIVITY")
print("=" * 80)

for waste in [10, 50, 100, 150, 200, 250, 300]:

    x = baseline.copy()
    x["Vegetable_Waste_kg_day"] = waste

    # Keep composition approximately consistent with generator
    x["Cellulose_pct"] = np.clip(
        35 + 15 * (waste / 300),
        35,
        55
    )

    x["Hemicellulose_pct"] = np.clip(
        18 + 8 * (waste / 300),
        18,
        28
    )

    flow_pred, engineering_pred = predict(x)

    print(
        f"Waste={waste:3d} kg/day | "
        f"Flow-only={flow_pred:.6f} | "
        f"Engineering={engineering_pred:.6f}"
    )


# ---------------------------------------------------------
# 5. Flow sensitivity
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("FLOW SENSITIVITY")
print("=" * 80)

for flow in [1000, 3000, 5000, 6000, 8000, 10000, 12000]:

    x = baseline.copy()
    x["Raw_Sewage_Flow_L_day"] = flow

    flow_pred, engineering_pred = predict(x)

    print(
        f"Flow={flow:5d} L/day | "
        f"Flow-only={flow_pred:.6f} | "
        f"Engineering={engineering_pred:.6f}"
    )

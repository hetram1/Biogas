# ==========================================================
# BIOGAS AI PROJECT
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ==========================================================
# LOAD DATASET
# ==========================================================

data = pd.read_csv("BiogasData.csv")

# ==========================================================
# INPUT FEATURES
# ==========================================================

X = data[
[
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
]

# ==========================================================
# TARGET
# ==========================================================

y = data["Biogas_Yield_m3_day"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train,X_test,y_train,y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42
)

# ==========================================================
# MODEL
# ==========================================================

model = RandomForestRegressor(
n_estimators=500,
max_depth=15,
random_state=42
)

# ==========================================================
# TRAIN MODEL
# ==========================================================

model.fit(X_train,y_train)




importance = pd.DataFrame({

"Feature": X.columns,

"Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "feature_importance.csv",
    index=False
)

print()
print("="*60)
print("FEATURE IMPORTANCE")
print("="*60)
print(importance)



import joblib

# Save model
joblib.dump(
{
    "model": model,
    "features": list(X.columns)
},
"biogas_model.pkl"
)

print("Model saved successfully")





# ==========================================================
# ACCURACY
# ==========================================================

predictions = model.predict(X_test)

parity_df = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": predictions

})

parity_df.to_csv(
    "parity_data.csv",
    index=False
)

r2 = r2_score(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

mae = mean_absolute_error(y_test, predictions)

print()
print("R² Score =", round(r2,4))
print("RMSE     =", round(rmse,4))
print("MAE      =", round(mae,4))

# ==========================================================
# USER INPUT
# ==========================================================

print("\nENTER DIGESTER CONDITIONS\n")

raw_sewage_flow = float(
input("Raw Sewage Flow (L/day): ")
)

vegetable_waste = float(
input("Vegetable Waste (kg/day): ")
)

cellulose = float(input("Cellulose (%): "))

hemicellulose = float(input("Hemicellulose (%): "))

cn_ratio = float(input("C/N Ratio: "))

temperature = float(
input("Temperature (°C): ")
)

ph = float(
input("pH: ")
)

hrt = float(
input("HRT (days): ")
)

volume = float(
input("Digester Volume (m³): ")
)

# ==========================================================
# CREATE INPUT DATAFRAME
# ==========================================================

new_data = pd.DataFrame({

"Digester_Volume_m3":[volume],

"Raw_Sewage_Flow_L_day":[raw_sewage_flow],

"Vegetable_Waste_kg_day":[vegetable_waste],

"Cellulose_pct":[cellulose],

"Hemicellulose_pct":[hemicellulose],

"Temperature_C":[temperature],

"pH":[ph],

"HRT_days":[hrt],

"CN_Ratio":[cn_ratio]

})
# ==========================================================
# MACHINE LEARNING PREDICTION
# ==========================================================

ml_prediction = model.predict(
new_data
)[0]

# ==========================================================
# PHYSICS MODEL
# ==========================================================

temp_factor = np.exp(-((temperature-35)/7)**2)

ph_factor = np.exp(-((ph-7)/0.5)**2)

hrt_factor = 1 - np.exp(-hrt/10)

cellulose_factor = cellulose / 100

hemi_factor = hemicellulose / 100


cn_factor = np.exp(-((cn_ratio-27.5)/5)**2)

flow_factor = (raw_sewage_flow / 5000)**0.6

vegetable_factor = 1 + 0.0025 * vegetable_waste

physics_prediction = (
    volume
    * flow_factor
    * vegetable_factor
    * cellulose_factor
    * hemi_factor
    * temp_factor
    * ph_factor
    * hrt_factor
    * cn_factor
    * 10
)

# ==========================================================
# HYBRID PREDICTION
# ==========================================================

final_prediction = (
0.95*ml_prediction +
0.05*physics_prediction
)

# ==========================================================
# PROCESS STATUS
# ==========================================================

print()
print("="*60)
print("ANAEROBIC DIGESTER REPORT")
print("="*60)

print()

print("INPUT CONDITIONS")

print("Raw Sewage Flow :", raw_sewage_flow, "L/day")

print("Vegetable Waste :", vegetable_waste, "kg/day")

print("Cellulose       :", cellulose, "%")

print("Hemicellulose   :", hemicellulose, "%")

print("Temperature     :", temperature, "°C")

print("pH              :", ph)

print("HRT             :", hrt, "days")

print("C/N Ratio       :", cn_ratio)

print("Volume          :", volume, "m³")


# ==========================================================
# BIOLOGY
# ==========================================================

print()
print("="*60)
print("BIOLOGICAL STAGES")
print("="*60)

print()

print("Hydrolysis")
print("Complex organics broken down")

print()

print("Acidogenesis")
print("VFAs produced")

print()

print("Acetogenesis")
print("Acetate + H2 + CO2 formed")

print()

print("Methanogenesis")
print("Methanogens convert acetate into methane")

# ==========================================================
# DIGESTER HEALTH
# ==========================================================

health = 100

if temperature < 30 or temperature > 40:
    health -= 20

if ph < 6.8 or ph > 7.3:
    health -= 20

if hrt < 10:
    health -= 15

if cn_ratio < 20 or cn_ratio > 35:
    health -= 10


if cellulose < 20:
    health -= 5

# ==========================================================
# FINAL OUTPUT
# ==========================================================

print()
print("="*60)
print("FINAL RESULTS")
print("="*60)

print()

print(
"Machine Learning Prediction =",
round(ml_prediction,2),
"m³/day"
)

print(
"Physics Prediction =",
round(physics_prediction,2),
"m³/day"
)

print(
"Final Biogas Yield =",
round(final_prediction,2),
"m³/day"
)

print()

print(
"Digester Health Score =",
health,
"/100"
)

print()

if health >= 90:
    print("Status : Excellent")

elif health >= 75:
    print("Status : Good")

elif health >= 60:
    print("Status : Moderate")

else:
    print("Status : Poor")

print()

print("="*60)

# ==========================================================
# PART 3 : PERFORMANCE REPORT
# ==========================================================

print()
print("="*70)
print("AUDST PERFORMANCE REPORT")
print("="*70)

# Temperature condition

if 32 <= temperature <= 38:
    temp_status = "Mesophilic Zone (Optimal)"
elif temperature < 32:
    temp_status = "Low Temperature Activity"
else:
    temp_status = "High Temperature"

# pH condition

if 6.8 <= ph <= 7.3:
    ph_status = "Healthy Methanogenesis"
else:
    ph_status = "Possible Inhibition"

# HRT condition

if hrt < 10:
    hrt_status = "Insufficient Retention Time"

elif hrt < 20:
    hrt_status = "Moderate Retention Time"

else:
    hrt_status = "High Retention Time"

print()
print("Temperature Status :", temp_status)

print("pH Status          :", ph_status)

print("HRT Status         :", hrt_status)

# ==========================================================
# GAS CONTRIBUTION
# ==========================================================

print()
print("="*70)
print("BIOGAS CONTRIBUTION")
print("="*70)

print()

print(
"Cellulose Contribution :",
round(cellulose_factor*100,2),
"%"
)

print(
"Hemicellulose Contribution :",
round(hemi_factor*100,2),
"%"
)


print(
"C/N Ratio Factor :",
round(cn_factor,2)
)

# ==========================================================
# METHANOGENESIS
# ==========================================================

methane_fraction = 0.62

co2_fraction = 0.35

other_fraction = 0.03

methane = methane_fraction * final_prediction

co2 = co2_fraction * final_prediction

others = other_fraction * final_prediction


print(
"Vegetable Waste Feed :",
round(vegetable_waste,2),
"kg/day"
)


print()
print("="*70)
print("BIOGAS COMPOSITION")
print("="*70)

print()

print(
"Methane (CH4) :",
round(methane,2),
"m³/day"
)

print(
"Carbon Dioxide (CO2) :",
round(co2,2),
"m³/day"
)

print(
"Other Gases :",
round(others,2),
"m³/day"
)

# ==========================================================
# DIGESTER EFFICIENCY
# ==========================================================

methane_productivity = methane / volume

print()

print("="*70)
print("METHANOGENIC PERFORMANCE")
print("="*70)

print()

print(
"Methane Productivity =",
round(methane_productivity,3),
"m³ CH4/m³ Digester"
)

# ==========================================================
# PROCESS SUMMARY
# ==========================================================

print()

print("="*70)
print("PROCESS FLOW")
print("="*70)

print()

print("Raw Sewage + Vegetable Waste Feed")

print("↓")

print("Hydrolysis")

print("↓")

print("Acidogenesis")

print("↓")

print("Acetogenesis")

print("↓")

print("Methanogenesis")

print("↓")

print(
"Biogas Production =",
round(final_prediction,2),
"m³/day"
)

print()

print("="*70)
print("END OF REPORT")
print("="*70)
# ==========================================================
# GENERATE SCALED BIOGAS DATASET WITH EXPLICIT UNITS
# ==========================================================

import numpy as np
import pandas as pd

# Set seed for reproducibility (Elsevier/Springer requirement)
np.random.seed(42)

# Number of samples
n = 5000

# ==========================================================
# 1. GENERATE OPERATIONAL VOLUMES AND RETENTION TIMES
# ==========================================================

# Digester Volume varied from 5 m³ to 500 m³
digester_volume_m3 = np.random.uniform(5, 500, n)

# HRT varied between 35 and 45 days
hrt_days = np.random.uniform(35, 45, n)

# Calculate required influent flow rate (Q = V / HRT) in m³/day
flow_m3_day = digester_volume_m3 / hrt_days

# Convert flow to Liters/day for Raw Sewage Flow tracking
raw_sewage_L_day = flow_m3_day * 1000

# Vegetable waste feed (kg/day)
vegetable_waste_kg_day = np.random.uniform(10, 300, n)

# ==========================================================
# 2. GENERATE CHEMICAL COMPOSITION ACCORDING TO SPECS
# ==========================================================



# Composition features are in percentage (%) of total volatile solids
cellulose_pct = (
    35
    + 15 * (vegetable_waste_kg_day / 300)
    + np.random.normal(0, 2, n)
)

cellulose_pct = np.clip(cellulose_pct, 35, 55)
hemicellulose_pct = (
    18
    + 8 * (vegetable_waste_kg_day / 300)
    + np.random.normal(0, 1.5, n)
)

hemicellulose_pct = np.clip(hemicellulose_pct, 18, 28)

temperature_C = np.random.uniform(25, 40, n)      # °C
ph = np.random.uniform(6.8, 7.6, n)                # pH (dimensionless)
cn_ratio = np.random.uniform(20, 25, n)            # C:N Ratio (dimensionless ratio)

# ==========================================================
# 3. DEFINE BIOLOGICAL ENVELOPE FUNCTIONS (f(x))
# ==========================================================

f_T = np.exp(-((temperature_C - 35) / 5) ** 2)
f_pH = np.exp(-((ph - 7.2) / 0.3) ** 2)
f_CN = np.exp(-((cn_ratio - 22.5) / 2.5) ** 2)
f_HRT = 1 - np.exp(-hrt_days / 10)

# Combined environmental efficiency multiplier (ranges from 0 to 1)
env_efficiency = f_T * f_pH * f_CN * f_HRT

# ==========================================================
# 4. CALCULATE SCALED BIOGAS YIELD via CORE EMPIRICAL FUNCTION
# ==========================================================

# Base scaling factor (k) calibrated for larger volumetric flow units
k = 0.001

# Biogas yield calculation (m³/day)
feed_factor = 1 + 0.005 * vegetable_waste_kg_day

biogas_yield_m3_day = (
    k
    * (raw_sewage_L_day ** 0.35)
    * feed_factor
    * (1 + 0.016 * (cellulose_pct - 35))
    * (1 + 0.014 * (hemicellulose_pct - 18))
    * env_efficiency
)

# Apply realistic operational noise to total yield (±4% Gaussian noise)
yield_noise = np.random.normal(0, 0.04 * biogas_yield_m3_day)
biogas_yield_m3_day = np.maximum(biogas_yield_m3_day + yield_noise, 0.01)

# ==========================================================
# 5. CALCULATE CH4 AND CO2 PERCENTAGES DYNAMICALLY
# ==========================================================

# Base methane % derived from chemical composition
base_ch4 = (
    60.0 +
    0.70 * (cellulose_pct - 35) +
    0.50 * (hemicellulose_pct - 18)
)

# Penalty term: If environmental conditions are bad, methane % drops
ch4_penalty = 8.0 * (1.0 - env_efficiency)

# Calculate theoretical methane percentage (%)
methane_pct = base_ch4 - ch4_penalty

# Add small local process fluctuations to gas quality (±1.5% random noise)
quality_noise = np.random.normal(0, 1.5, n)
methane_pct = methane_pct + quality_noise

# Constrain methane to physical boundary limits [58%, 72%]
methane_pct = np.clip(methane_pct, 58.0, 72.0)

# Carbon Dioxide (%) balance (summing to ~98.5% with trace gases)
co2_pct = 98.5 - methane_pct

# ==========================================================
# 6. CREATE AND SAVE THE DATAFRAME WITH EXPLICIT COLUMN UNITS
# ==========================================================

df = pd.DataFrame({
    "Digester_Volume_m3": digester_volume_m3,
    "Raw_Sewage_Flow_L_day": raw_sewage_L_day,
    "Vegetable_Waste_kg_day": vegetable_waste_kg_day,
    "Cellulose_pct": cellulose_pct,
    "Hemicellulose_pct": hemicellulose_pct,
    "Temperature_C": temperature_C,
    "pH": ph,
    "HRT_days": hrt_days,
    "CN_Ratio": cn_ratio,
    "Biogas_Yield_m3_day": biogas_yield_m3_day,
    "Methane_pct": methane_pct,
    "CO2_pct": co2_pct
})

# Save output to CSV
df.to_csv("BiogasData.csv", index=False)

# ==========================================================
# DISPLAY DATASET SUMMARY
# ==========================================================

print("==========================================================")
print("  DATASET GENERATED SUCCESSFULLY (WITH EXPLICIT UNITS)   ")
print("==========================================================")
print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

print("First 5 rows of the generated dataset:")
print(df.head().round(3))

print("\n--- Summary Statistics of Target Variables ---")
print(df[["Biogas_Yield_m3_day", "Methane_pct", "CO2_pct"]].describe().round(2))
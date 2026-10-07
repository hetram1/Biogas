import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from PIL import Image

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AUDST Biogas Dashboard",
    layout="wide"
)

st.title("🟢 AUDST BIOGAS PREDICTION SYSTEM")

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load("biogas_model_final.pkl")

feature_order = [
    "Raw_Sewage_Flow_L_day",
    "Vegetable_Waste_kg_day",
    "Cellulose_pct",
    "Hemicellulose_pct",
    "Temperature_C",
    "pH",
    "CN_Ratio",
    "Temperature_deviation",
    "pH_deviation",
    "CN_deviation"
]

importance = pd.read_csv("final_feature_importance.csv")

data = pd.read_csv("BiogasData.csv")

parity_data = pd.read_csv("final_parity_data.csv")

# =====================================
# SIDEBAR INPUTS
# =====================================

st.sidebar.header("Input Conditions")

# =====================================
# RAW SEWAGE INPUT
# =====================================

# Dry biomass (Volatile Suspended Solids)
# Typical municipal sewage:
# 150–250 mg/L
# Midpoint = 200 mg/L = 0.0002 kg/L

VSS_CONCENTRATION = 0.0002      # kg dry biomass per L sewage

raw_unit = st.sidebar.selectbox(
    "Raw Sewage Unit",
    [
        "L/day",
        "m³/day",
        "Dry Biomass (kg/day)"
    ]
)

# ----------------------------
# INPUT IN L/day
# ----------------------------

if raw_unit == "L/day":

    raw_sewage = st.sidebar.slider(
        "Raw Sewage Flow",
        500,
        5000,
        750
    )

    raw_sewage_m3 = raw_sewage / 1000

    raw_sewage_kg = raw_sewage * VSS_CONCENTRATION

    st.sidebar.success(
        f"{raw_sewage_m3:.3f} m³/day | {raw_sewage_kg:.3f} kg Dry Biomass/day"
    )

# ----------------------------
# INPUT IN m³/day
# ----------------------------

elif raw_unit == "m³/day":

    raw_sewage_m3 = st.sidebar.slider(
        "Raw Sewage Flow",
        0.50,
        5.00,
        0.75,
        0.01
    )

    raw_sewage = raw_sewage_m3 * 1000

    raw_sewage_kg = raw_sewage * VSS_CONCENTRATION

    st.sidebar.success(
        f"{raw_sewage:.0f} L/day | {raw_sewage_kg:.3f} kg Dry Biomass/day"
    )

# ----------------------------
# INPUT IN DRY BIOMASS
# ----------------------------

else:

    raw_sewage_kg = st.sidebar.slider(
        "Dry Biomass Feed (kg/day)",
        0.1,
        10.0,
        2.0,
        0.1
    )

    # Fixed hydraulic flow (user can choose a representative value)
    raw_sewage = 1000          # L/day 
    raw_sewage_m3 = 1.0        # m³/day

    st.sidebar.success(
    f"Hydraulic Flow = {raw_sewage:.0f} L/day\n"
    f"Dry Biomass = {raw_sewage_kg:.3f} kg/day"
    )

# =====================================
# INFORMATION
# =====================================

st.sidebar.info(
"""
Typical Municipal Raw Sewage

• Dry Biomass (VSS): 150–250 mg/L

Engineering assumption:

• 200 mg/L = 0.0002 kg/L

Equivalent:

• 1 L sewage ≈ 0.0002 kg dry biomass

• 1000 L sewage ≈ 0.20 kg dry biomass

• 1 kg dry biomass ≈ 5000 L sewage ≈ 5 m³ sewage
"""
)

vegetable_density = 650      # kg/m³

veg_unit = st.sidebar.selectbox(
    "Vegetable Waste Unit",
    ["kg/day", "m³/day"]
)

if veg_unit == "kg/day":

    vegetable_waste = st.sidebar.slider(
        "Vegetable Waste",
        10,
        300,
        100
    )

    vegetable_volume = vegetable_waste / vegetable_density

    st.sidebar.success(f"= {vegetable_volume:.3f} m³/day")

else:

    vegetable_volume = st.sidebar.slider(
        "Vegetable Waste",
        0.01,
        0.50,
        0.15,
        0.005
    )

    vegetable_waste = vegetable_volume * vegetable_density

    st.sidebar.success(f"= {vegetable_waste:.1f} kg/day")

cellulose = st.sidebar.slider(
    "Cellulose (%)",
    10,
    70,
    40
)

hemicellulose = st.sidebar.slider(
    "Hemicellulose (%)",
    5,
    40,
    20
)


cn_ratio = st.sidebar.slider(
    "C/N Ratio",
    10,
    40,
    25
)

hrt = st.sidebar.slider(
    "HRT (days)",
    5,
    60,
    20
)

temp = st.sidebar.slider(
    "Temperature (°C)",
    20,
    40,
    35
)

ph = st.sidebar.slider(
    "pH",
    6.5,
    7.8,
    7.0
)


volume_unit = st.sidebar.selectbox(
    "Digester Volume Unit",
    ["m³", "L"]
)

if volume_unit == "m³":

    volume = st.sidebar.slider(
        "Digester Volume",
        2,
        150,
        40
    )

    st.sidebar.success(f"= {volume*1000:.0f} L")

else:

    volume_liters = st.sidebar.slider(
        "Digester Volume",
        2000,
        150000,
        40000,
        500
    )

    volume = volume_liters / 1000

    st.sidebar.success(f"= {volume:.2f} m³")


# =====================================
# DIGESTER LOADING CALCULATION
# =====================================

# Sewage flow (m³/day)
daily_flow = raw_sewage / 1000

# Vegetable waste equivalent volume (assuming slurry density)
vegetable_density = 650      # kg/m³

vegetable_volume = vegetable_waste / vegetable_density

# Total influent volume
total_feed_volume = daily_flow + vegetable_volume

# Required digester volume
required_volume = total_feed_volume * hrt

# Working volume (80% of total digester volume)
working_volume = volume * 0.80

# Digester utilization
fill_percentage = (required_volume / working_volume) * 100


# ===============================
# PHYSICS MODEL
# ===============================





new_data = pd.DataFrame({

"Digester_Volume_m3":[volume],

"Raw_Sewage_Flow_L_day":[raw_sewage],

"Vegetable_Waste_kg_day":[vegetable_waste],

"Cellulose_pct":[cellulose],

"Hemicellulose_pct":[hemicellulose],

"Temperature_C":[temp],

"pH":[ph],

"HRT_days":[hrt],

"CN_Ratio":[cn_ratio]

})



# =====================================
# MACHINE LEARNING PREDICTION
# =====================================

def prepare_model_input(df):
    result = pd.DataFrame({
        "Raw_Sewage_Flow_L_day": df["Raw_Sewage_Flow_L_day"].values,
        "Vegetable_Waste_kg_day": df["Vegetable_Waste_kg_day"].values,
        "Cellulose_pct": df["Cellulose_pct"].values,
        "Hemicellulose_pct": df["Hemicellulose_pct"].values,
        "Temperature_C": df["Temperature_C"].values,
        "pH": df["pH"].values,
        "CN_Ratio": df["CN_Ratio"].values,
        "Temperature_deviation": np.abs(df["Temperature_C"].values - 35.0),
        "pH_deviation": np.abs(df["pH"].values - 7.2),
        "CN_deviation": np.abs(df["CN_Ratio"].values - 22.5)
    })

    return result[feature_order]


ml_prediction = model.predict(
    prepare_model_input(new_data)
)[0]


temp_factor = np.exp(-((temp-35)/7)**2)

ph_factor = np.exp(-((ph-7)/0.5)**2)

hrt_factor = 1 - np.exp(-hrt/10)

cellulose_factor = cellulose/100

hemi_factor = hemicellulose/100

cn_factor = np.exp(-((cn_ratio-27.5)/5)**2)

flow_factor = (raw_sewage/5000)**0.5

# Reference biomass = 2 kg/day
biomass_factor = 0.5 + (raw_sewage_kg / 2.0)

# Prevent unrealistic values
biomass_factor = np.clip(biomass_factor, 0.5, 5.0)

vegetable_factor = 1 + 0.006 * vegetable_waste

physics_prediction = (
    volume
    * flow_factor
    * biomass_factor
    * vegetable_factor
    * cellulose_factor
    * hemi_factor
    * temp_factor
    * ph_factor
    * hrt_factor
    * cn_factor
    * 10
)

final_prediction = ml_prediction


# =====================================
# GAS COMPOSITION
# =====================================

methane = 0.62*final_prediction

co2 = 0.35*final_prediction

others = 0.03*final_prediction




# =====================================
# GAS OUTPUT UNIT
# =====================================

gas_unit = st.radio(
    "Gas Output Unit",
    ["m³/day", "L/day", "kg/day"],
    horizontal=True
)


# =====================================
# METRICS
# =====================================


c1,c2,c3 = st.columns(3)

c4,c5,c6 = st.columns(3)


st.subheader("Digester Utilization")

st.progress(min(fill_percentage/100,1.0))

st.write(f"Working Volume Utilization : {fill_percentage:.1f}%")

if fill_percentage > 100:

    st.error(
        "⚠ Digester Over-Filling Risk! Increase digester volume or reduce wastewater flow."
    )

elif fill_percentage > 90:

    st.warning(
        "⚠ Digester is operating close to maximum working capacity."
    )

else:

    st.success(
        "✓ Digester operating within safe working volume."
    )


# =====================================
# UNIT CONVERSIONS
# =====================================

# Raw sewage
raw_sewage_m3 = raw_sewage / 1000          # m³/day
raw_sewage_liters = raw_sewage             # L/day

# Dry biomass (VSS)
raw_sewage_dry_biomass = raw_sewage_kg     # kg/day

# Vegetable waste
VEGETABLE_DENSITY = 650                    # kg/m³

vegetable_volume = vegetable_waste / VEGETABLE_DENSITY
vegetable_liters = vegetable_volume * 1000

# Digester
volume_liters = volume * 1000

# Gas production
biogas_liters = final_prediction * 1000
methane_liters = methane * 1000
co2_liters = co2 * 1000


# =====================================
# GAS PROPERTIES (STP: 0°C, 1 atm)
# =====================================

MOLAR_VOLUME = 22.414       # L/mol

CH4_MOLAR_MASS = 16.04      # g/mol
CO2_MOLAR_MASS = 44.01      # g/mol

CH4_DENSITY = CH4_MOLAR_MASS / MOLAR_VOLUME      # g/L = kg/m³
CO2_DENSITY = CO2_MOLAR_MASS / MOLAR_VOLUME      # g/L = kg/m³

# Biogas composition
CH4_FRACTION = 0.62
CO2_FRACTION = 0.35
OTHER_FRACTION = 0.03

BIOGAS_DENSITY = (
    CH4_FRACTION * CH4_DENSITY +
    CO2_FRACTION * CO2_DENSITY
)

biogas_kg = final_prediction * BIOGAS_DENSITY
methane_kg = methane * CH4_DENSITY


st.markdown("---")

with st.expander("📐 Unit Conversions"):

    st.write(
    f"**Raw Sewage:** "
    f"{raw_sewage_liters:.0f} L/day = "
    f"{raw_sewage_m3:.3f} m³/day = "
    f"{raw_sewage_dry_biomass:.3f} kg Dry Biomass/day"
)

    st.write(
    f"**Vegetable Waste:** "
    f"{vegetable_waste:.1f} kg/day = "
    f"{vegetable_volume:.3f} m³/day "
    f"({vegetable_liters:.0f} L/day)"
)

    st.write(f"**Digester Volume:** {volume:.2f} m³ = {volume_liters:.0f} L")

    st.write(f"**Biogas Production:** {final_prediction:.2f} m³/day = {biogas_liters:.0f} L/day")

    st.write(f"**Methane:** {methane:.2f} m³/day = {methane_liters:.0f} L/day")

    st.write(f"**CO₂:** {co2:.2f} m³/day = {co2_liters:.0f} L/day")

st.subheader("Design Check")

design_df = pd.DataFrame({

"Parameter":[
"Daily Sewage Flow",
"Vegetable Waste Feed",
"Equivalent Feed Volume",
"HRT",
"Required Volume",
"Working Volume",
"Digester Utilization"
],

"Value":[
f"{daily_flow:.2f} m³/day",
f"{vegetable_waste:.1f} kg/day",
f"{vegetable_volume:.2f} m³/day",
f"{hrt:.1f} days",
f"{required_volume:.2f} m³",
f"{working_volume:.2f} m³",
f"{fill_percentage:.1f}%"
]
})
st.table(design_df)




image = Image.open("AUDST.png")

st.image(
    image,
    caption="Anaerobic Upflow Domestic Septic Tank (AUDST)",
    width='stretch'
)

# =====================================
# HEALTH STATUS
# =====================================

health = 100

if temp < 30 or temp > 40:
    health -= 20

if ph < 6.8 or ph > 7.3:
    health -= 20

if hrt < 10:
    health -= 15

if cn_ratio < 20 or cn_ratio > 35:
    health -= 10

if cellulose < 20:
    health -= 5


# =====================================
# BIOLOGICAL STAGES & PROCESS STATUS
# =====================================

col1, col2 = st.columns(2)

with col1:

    st.header("Biological Stages")

    st.write("🟢 Hydrolysis : Breakdown of complex organics")

    st.write("🟢 Acidogenesis : Volatile fatty acid production")

    st.write("🟢 Acetogenesis : Acetate production")

    st.write("🟢 Methanogenesis : Methane generation")


with col2:

    st.header("Process Status")

    if 32 <= temp <= 38:
        st.success("Mesophilic Temperature Region")
    else:
        st.warning("Non-optimal Temperature")



if gas_unit == "m³/day":
    biogas_display = f"{final_prediction:.2f} m³/day"

elif gas_unit == "L/day":
    biogas_display = f"{biogas_liters:.0f} L/day"

else:
    biogas_display = f"{biogas_kg:.2f} kg/day"

c1.metric(
    "Biogas",
    biogas_display
)

if gas_unit == "m³/day":
    methane_display = f"{methane:.2f} m³/day"

elif gas_unit == "L/day":
    methane_display = f"{methane_liters:.0f} L/day"

else:
    methane_display = f"{methane_kg:.2f} kg/day"

c2.metric(
    "CH₄",
    methane_display
)

c3.metric(
    "Health",
    f"{health}/100"
)

c4.metric(
"Vegetable Waste",
f"{vegetable_waste:.1f} kg/day"
)

c5.metric(
    "pH",
    f"{ph:.2f}"
)

c6.metric(
    "HRT",
    f"{hrt:.1f} days"
)





# =====================================
# PIE CHART & BAR CHART
# =====================================

col1, col2 = st.columns(2)

# -------------------------------
# Pie Chart
# -------------------------------

gas_df = pd.DataFrame({

    "Gas": ["CH4", "CO2", "Others"],

    "Volume": [methane, co2, others]

})

fig1 = px.pie(

    gas_df,

    values="Volume",

    names="Gas",

    title="Biogas Composition"

)

with col1:

    st.plotly_chart(
        fig1,
        width='stretch'
    )

# -------------------------------
# Bar Chart
# -------------------------------

bar_df = pd.DataFrame({

"Component":[
"Cellulose",
"Hemicellulose"
],

"Percentage":[
cellulose,
hemicellulose
]

})

fig2 = px.bar(

    bar_df,

    x="Component",

    y="Percentage",

    title="Biomass Composition"

)

with col2:

    st.plotly_chart(
        fig2,
        width='stretch'
    )







st.header("Current Prediction")

result = pd.DataFrame({

"Parameter":[
"Biogas Yield",
"CH4",
"CO2",
"Health Score"
],

"Value":[
round(final_prediction,2),
round(methane,2),
round(co2,2),
health
]

})

st.dataframe(result)



st.header("Sensitivity Analysis")
st.info(
    "Each graph varies one input parameter while keeping all other operating conditions constant. "
    "This illustrates the sensitivity of the trained machine learning model to individual process variables."
)

col1, col2 = st.columns(2)

cellulose_range = np.linspace(10,70,40)

yield_list = []

for c in cellulose_range:

    temp_df = new_data.copy()

    temp_df["Cellulose_pct"] = c

    pred = model.predict(
        prepare_model_input(temp_df)
    )[0]

    yield_list.append(pred)

curve_df = pd.DataFrame({

"Cellulose (%)":cellulose_range,

"Biogas Yield (m³/day)":yield_list

})

fig = px.line(
    curve_df,
    x="Cellulose (%)",
    y="Biogas Yield (m³/day)",
    title="Effect of Cellulose on Biogas Yield"
)

with col1:
    st.plotly_chart(fig,width='stretch')


cn_range = np.linspace(10,40,40)

yield_list = []

for cn in cn_range:

    temp_df = new_data.copy()

    temp_df["CN_Ratio"] = cn

    pred = model.predict(
        prepare_model_input(temp_df)
    )[0]

    yield_list.append(pred)

curve_df = pd.DataFrame({

"C/N Ratio":cn_range,

"Biogas Yield (m³/day)":yield_list

})

fig = px.line(
    curve_df,
    x="C/N Ratio",
    y="Biogas Yield (m³/day)",
    title="Effect of C/N Ratio on Biogas Yield"
)

with col2:
    st.plotly_chart(fig,width='stretch')



col1, col2 = st.columns(2)


hrt_range = np.linspace(5,60,40)

yield_list=[]

for h in hrt_range:

    temp_df=new_data.copy()

    temp_df["HRT_days"]=h

    pred=model.predict(
        prepare_model_input(temp_df)
    )[0]

    yield_list.append(pred)

curve_df=pd.DataFrame({

"HRT (days)":hrt_range,

"Biogas Yield (m³/day)":yield_list

})

fig=px.line(

curve_df,

x="HRT (days)",

y="Biogas Yield (m³/day)",

title="Effect of HRT on Biogas Yield"

)

with col1:
    st.plotly_chart(fig,width='stretch')






temp_range = np.linspace(20,45,40)

yield_list = []

for t in temp_range:

    temp_df = new_data.copy()

    temp_df["Temperature_C"] = t

    pred = model.predict(
        prepare_model_input(temp_df)
    )[0]

    yield_list.append(pred)

curve_df = pd.DataFrame({

"Temperature (°C)":temp_range,

"Biogas Yield (m³/day)":yield_list

})

fig = px.line(

curve_df,

x="Temperature (°C)",

y="Biogas Yield (m³/day)",

title="Effect of Temperature on Biogas Yield"

)

with col2:
    st.plotly_chart(fig,width='stretch')

col1, col2 = st.columns(2)

ph_range = np.linspace(6.5,7.8,40)

yield_list = []

for p in ph_range:

    temp_df = new_data.copy()

    temp_df["pH"] = p

    pred = model.predict(
        prepare_model_input(temp_df)
    )[0]

    yield_list.append(pred)

curve_df = pd.DataFrame({

"pH":ph_range,

"Biogas Yield (m³/day)":yield_list

})

fig = px.line(

curve_df,

x="pH",

y="Biogas Yield (m³/day)",

title="Effect of pH on Biogas Yield"

)

with col1:
    st.plotly_chart(fig,width='stretch')

with col2:

    fig_importance = px.bar(

        importance.sort_values(
            "Importance",
            ascending=True
        ),

        x="Importance",

        y="Feature",

        orientation="h",

        title="Random Forest Feature Importance"

    )

    st.plotly_chart(
        fig_importance,
        width='stretch'
    )



st.header("Correlation Analysis")

corr = data.corr(numeric_only=True)

fig_corr = px.imshow(

    corr,

    text_auto=".2f",

    aspect="auto",

    color_continuous_scale="RdBu_r",

    title="Feature Correlation Heatmap"

)

st.plotly_chart(
    fig_corr,
    width='stretch'
)




st.header("Model Validation")

fig_parity = px.scatter(

    parity_data,

    x="Actual",

    y="Predicted",

    title="Actual vs Predicted Biogas Yield"

)

fig_parity.add_shape(

    type="line",

    x0=parity_data["Actual"].min(),

    y0=parity_data["Actual"].min(),

    x1=parity_data["Actual"].max(),

    y1=parity_data["Actual"].max(),

    line=dict(color="red", dash="dash")

)

st.plotly_chart(
    fig_parity,
    width='stretch'
)





st.header("Operating Condition Assessment")

radar_df = pd.DataFrame({

"Factor":[
"Temperature",
"pH",
"HRT",
"C/N Ratio",
"Cellulose"
],

"Score":[

temp_factor*100,

ph_factor*100,

hrt_factor*100,

cn_factor*100,

cellulose_factor*100

]

})

fig_radar = px.line_polar(

radar_df,

r="Score",

theta="Factor",

line_close=True,

title="Current Operating Factors"

)

fig_radar.update_traces(fill="toself")

st.plotly_chart(
    fig_radar,
    width='stretch'
)


st.header("Biogas Production Over Retention Time")

days = np.arange(1, hrt + 1)

# Simple cumulative production model
daily_production = final_prediction * (1 - np.exp(-days / (hrt / 3)))

production_df = pd.DataFrame({

    "Day": days,

    "Biogas Production (m³/day)": daily_production

})

fig_time = px.line(

    production_df,

    x="Day",

    y="Biogas Production (m³/day)",

    title="Predicted Biogas Production During Hydraulic Retention Time"

)

st.plotly_chart(
    fig_time,
    width='stretch'
)


st.header("Biological Interpretation")

st.write("""
Hydrolysis degrades cellulose and hemicellulose into soluble sugars.

Acidogenesis converts these sugars into volatile fatty acids (VFAs).

Acetogenesis converts VFAs into acetate, hydrogen and carbon dioxide.

Methanogenesis converts acetate and hydrogen into methane-rich biogas.

Temperature, pH, hydraulic retention time (HRT), biomass composition and C/N ratio strongly influence microbial activity and overall biogas yield.
""")

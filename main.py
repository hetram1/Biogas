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

saved = joblib.load("biogas_model.pkl")

model = saved["model"]

feature_order = saved["features"]

importance = pd.read_csv("feature_importance.csv")

data = pd.read_csv("BiogasData.csv")

parity_data = pd.read_csv("parity_data.csv")

# =====================================
# SIDEBAR INPUTS
# =====================================

st.sidebar.header("Input Conditions")

raw_sewage = st.sidebar.slider(
    "Raw Sewage Flow (L/day)",
    500,
    5000,
    750
)

vegetable_waste = st.sidebar.slider(
    "Vegetable Waste (kg/day)",
    10,
    300,
    100
)

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


volume = st.sidebar.slider(
    "Digester Volume (m³)",
    2,
    150,
    40
)


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

new_data = new_data[feature_order]

ml_prediction = model.predict(new_data)[0]


temp_factor = np.exp(-((temp-35)/7)**2)

ph_factor = np.exp(-((ph-7)/0.5)**2)

hrt_factor = 1 - np.exp(-hrt/10)

cellulose_factor = cellulose/100

hemi_factor = hemicellulose/100

cn_factor = np.exp(-((cn_ratio-27.5)/5)**2)

flow_factor = (raw_sewage/5000)**0.5

vegetable_factor = 1 + 0.006 * vegetable_waste

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

final_prediction = (
    0.95*ml_prediction +
    0.05*physics_prediction
)


# =====================================
# GAS COMPOSITION
# =====================================

methane = 0.62*final_prediction

co2 = 0.35*final_prediction

others = 0.03*final_prediction

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
    use_container_width=True
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



c1.metric(
"Biogas",
f"{final_prediction:.2f} m³/day"
)

c2.metric(
"CH₄",
f"{methane:.2f} m³/day"
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
        use_container_width=True
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
        use_container_width=True
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

    temp_df = temp_df[feature_order]

    pred = model.predict(temp_df)[0]

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
    st.plotly_chart(fig,use_container_width=True)


cn_range = np.linspace(10,40,40)

yield_list = []

for cn in cn_range:

    temp_df = new_data.copy()

    temp_df["CN_Ratio"] = cn

    temp_df = temp_df[feature_order]

    pred = model.predict(temp_df)[0]

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
    st.plotly_chart(fig,use_container_width=True)



col1, col2 = st.columns(2)


hrt_range = np.linspace(5,60,40)

yield_list=[]

for h in hrt_range:

    temp_df=new_data.copy()

    temp_df["HRT_days"]=h

    temp_df=temp_df[feature_order]

    pred=model.predict(temp_df)[0]

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
    st.plotly_chart(fig,use_container_width=True)






temp_range = np.linspace(20,45,40)

yield_list = []

for t in temp_range:

    temp_df = new_data.copy()

    temp_df["Temperature_C"] = t

    temp_df = temp_df[feature_order]

    pred = model.predict(temp_df)[0]

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
    st.plotly_chart(fig,use_container_width=True)

col1, col2 = st.columns(2)

ph_range = np.linspace(6.5,7.8,40)

yield_list = []

for p in ph_range:

    temp_df = new_data.copy()

    temp_df["pH"] = p

    temp_df = temp_df[feature_order]

    pred = model.predict(temp_df)[0]

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
    st.plotly_chart(fig,use_container_width=True)

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
        use_container_width=True
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
    use_container_width=True
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
    use_container_width=True
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
    use_container_width=True
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
    use_container_width=True
)


st.header("Biological Interpretation")

st.write("""
Hydrolysis degrades cellulose and hemicellulose into soluble sugars.

Acidogenesis converts these sugars into volatile fatty acids (VFAs).

Acetogenesis converts VFAs into acetate, hydrogen and carbon dioxide.

Methanogenesis converts acetate and hydrogen into methane-rich biogas.

Temperature, pH, hydraulic retention time (HRT), biomass composition and C/N ratio strongly influence microbial activity and overall biogas yield.
""")
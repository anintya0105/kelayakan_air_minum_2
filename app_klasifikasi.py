import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="DSS Kelayakan Air Minum - Enterprise",
    page_icon="💧",
    layout="wide"
)

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
.main {background-color: #f4f6f9;}
[data-testid="stSidebar"] {background-color: #0E1117;}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {color: white;}
.stButton>button {border-radius: 10px;height:45px;font-weight:bold;width:100%;}
.info-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD MODEL
# =============================
model = joblib.load("model_random_forest.pkl")

# =============================
# HEADER
# =============================
st.title("💧 Enterprise Decision Support System")
st.subheader("Analisis Kelayakan Air Minum Berbasis Machine Learning")

# =============================
# SIDEBAR INPUT
# =============================
st.sidebar.header("⚙ Control Panel")

ph = st.sidebar.number_input("pH", 0.0, 14.0, 7.0)
hardness = st.sidebar.number_input("Hardness", 0.0, 500.0, 150.0)
solids = st.sidebar.number_input("Solids (TDS)", 0.0, 50000.0, 20000.0)
chloramines = st.sidebar.number_input("Chloramines", 0.0, 20.0, 7.0)
sulfate = st.sidebar.number_input("Sulfate", 0.0, 500.0, 300.0)
conductivity = st.sidebar.number_input("Conductivity", 0.0, 1000.0, 400.0)
organic_carbon = st.sidebar.number_input("Organic Carbon", 0.0, 50.0, 10.0)
trihalomethanes = st.sidebar.number_input("Trihalomethanes", 0.0, 200.0, 70.0)
turbidity = st.sidebar.number_input("Turbidity", 0.0, 10.0, 4.0)

analyze = st.sidebar.button("🚀 Analisis Sekarang")

# =============================
# RISK FUNCTION
# =============================
def risk_label(prob):
    if prob >= 0.75:
        return "🟢 Risiko Rendah"
    elif prob >= 0.5:
        return "🟡 Risiko Sedang"
    else:
        return "🔴 Risiko Tinggi"

# =============================
# ANALYSIS
# =============================
if analyze:

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    input_df = pd.DataFrame([[ 
        ph, hardness, solids, chloramines,
        sulfate, conductivity, organic_carbon,
        trihalomethanes, turbidity
    ]], columns=[
        "ph","Hardness","Solids","Chloramines",
        "Sulfate","Conductivity","Organic_carbon",
        "Trihalomethanes","Turbidity"
    ])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("## 📊 Hasil Analisis")

    col1, col2 = st.columns(2)

    # GAUGE
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Probabilitas Kelayakan (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "green"},
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Tingkat Risiko", risk_label(prob))
        if pred == 1:
            st.success("### ✅ Air Layak Minum")
        else:
            st.error("### ❌ Air Tidak Layak")

    # =============================
    # FEATURE IMPORTANCE
    # =============================
    st.markdown("## 📈 Parameter Paling Berpengaruh")

    importance_df = pd.DataFrame({
        "Parameter": input_df.columns,
        "Tingkat Pengaruh": model.feature_importances_
    }).sort_values(by="Tingkat Pengaruh", ascending=True)

    fig2 = px.bar(
        importance_df,
        x="Tingkat Pengaruh",
        y="Parameter",
        orientation='h'
    )

    st.plotly_chart(fig2, use_container_width=True)

# =============================
# PENJELASAN PARAMETER
# =============================
st.markdown("## 📘 Penjelasan Parameter Kualitas Air")

st.markdown("""
### 1️⃣ pH
Menunjukkan tingkat keasaman atau kebasaan air.  
Rentang ideal air minum: **6.5 – 8.5**.  
- pH terlalu rendah → air bersifat asam, dapat menyebabkan korosi pipa dan gangguan pencernaan.  
- pH terlalu tinggi → rasa pahit dan dapat mengganggu metabolisme tubuh.

### 2️⃣ Hardness
Mengukur kadar kalsium dan magnesium.  
- Terlalu tinggi → menyebabkan kerak dan gangguan ginjal jangka panjang.  
- Terlalu rendah → air terasa hambar.

### 3️⃣ Solids (TDS)
Total zat terlarut dalam air.  
- Tinggi → rasa tidak enak dan kemungkinan kontaminasi.  
- Terlalu rendah → air miskin mineral.

### 4️⃣ Chloramines
Digunakan sebagai desinfektan.  
- Berlebihan → iritasi kulit dan mata.  
- Kurang → risiko mikroorganisme tidak mati.

### 5️⃣ Sulfate
Mineral alami dalam air.  
- Tinggi → dapat menyebabkan efek laksatif.  

### 6️⃣ Conductivity
Kemampuan air menghantarkan listrik (indikasi ion terlarut).  
- Tinggi → kandungan mineral tinggi.

### 7️⃣ Organic Carbon
Menunjukkan kandungan bahan organik.  
- Tinggi → indikasi potensi pertumbuhan bakteri.

### 8️⃣ Trihalomethanes
Produk sampingan klorinasi.  
- Tinggi → berpotensi berdampak jangka panjang terhadap kesehatan.

### 9️⃣ Turbidity
Tingkat kekeruhan air.  
- Tinggi → menunjukkan adanya partikel tersuspensi dan mikroorganisme.
""")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("DSS Kelayakan Air Minum © 2026 | Enterprise ML-Based Decision Support System")

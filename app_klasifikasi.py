import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="DSS Kelayakan Air Minum",
    page_icon="💧",
    layout="wide"
)

# =============================
# CUSTOM CSS (Modern Look)
# =============================
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD MODEL
# =============================
model = joblib.load("model.pkl")

# =============================
# HEADER
# =============================
st.title("💧 Decision Support System")
st.subheader("Analisis Kelayakan Air Minum Berbasis Machine Learning")

# =============================
# EXPLAIN VARIABLES
# =============================
with st.expander("📘 Penjelasan Parameter Kualitas Air"):
    st.markdown("""
**pH** → Tingkat keasaman/kebasaan air (6.5–8.5 ideal)  
**Hardness** → Kandungan mineral kalsium & magnesium  
**Solids (TDS)** → Total zat terlarut  
**Chloramines** → Desinfektan air  
**Sulfate** → Kandungan sulfat  
**Conductivity** → Kemampuan menghantarkan listrik  
**Organic Carbon** → Kandungan karbon organik  
**Trihalomethanes** → Produk sampingan klorin  
**Turbidity** → Tingkat kekeruhan air  
""")

# =============================
# INPUT MODE
# =============================
st.sidebar.header("⚙ Pengaturan")
mode = st.sidebar.radio("Metode Input:", ["Manual", "Upload CSV"])

# =============================
# RISK LABEL FUNCTION
# =============================
def risk_label(prob):
    if prob >= 0.75:
        return "🟢 Low Risk"
    elif prob >= 0.5:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

# =============================
# MANUAL INPUT
# =============================
if mode == "Manual":

    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input("pH", 0.0, 14.0, 7.0)
        hardness = st.number_input("Hardness", 0.0, 500.0, 150.0)
        solids = st.number_input("Solids", 0.0, 50000.0, 20000.0)

    with col2:
        chloramines = st.number_input("Chloramines", 0.0, 20.0, 7.0)
        sulfate = st.number_input("Sulfate", 0.0, 500.0, 300.0)
        conductivity = st.number_input("Conductivity", 0.0, 1000.0, 400.0)

    with col3:
        organic_carbon = st.number_input("Organic Carbon", 0.0, 50.0, 10.0)
        trihalomethanes = st.number_input("Trihalomethanes", 0.0, 200.0, 70.0)
        turbidity = st.number_input("Turbidity", 0.0, 10.0, 4.0)

    if st.button("🚀 Analisis"):

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

        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Probabilitas Layak", f"{prob*100:.2f}%")

        with colB:
            st.metric("Risk Level", risk_label(prob))

        with colC:
            if pred == 1:
                st.success("Air Layak Minum")
            else:
                st.error("Air Tidak Layak")

# =============================
# CSV MODE
# =============================
else:
    file = st.file_uploader("Upload file CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df)

        if st.button("🚀 Analisis Semua Data"):

            pred = model.predict(df)
            prob = model.predict_proba(df)[:,1]

            df["Prediction"] = pred
            df["Probability"] = prob
            df["Risk_Level"] = df["Probability"].apply(risk_label)

            df = df.sort_values("Probability", ascending=False)
            df["Ranking"] = range(1, len(df)+1)

            st.markdown("## 📊 Hasil Ranking")
            st.dataframe(df)

            # Summary Dashboard
            st.markdown("## 📈 Dashboard Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Sampel", len(df))

            with col2:
                st.metric("Rata-rata Probabilitas", f"{df['Probability'].mean()*100:.2f}%")

            with col3:
                st.metric("Jumlah Layak", int(df["Prediction"].sum()))

            # Chart
            st.markdown("## 📊 Distribusi Probabilitas")
            fig = plt.figure()
            plt.hist(df["Probability"], bins=10)
            plt.xlabel("Probabilitas")
            plt.ylabel("Jumlah")
            st.pyplot(fig)

            # Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Hasil",
                csv,
                "hasil_analisis_air.csv",
                "text/csv"
            )

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("DSS Kelayakan Air Minum © 2026 | Sistem Pendukung Keputusan Berbasis Machine Learning")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="DSS Kelayakan Air", layout="wide")

# =============================
# DARK MODE TOGGLE
# =============================
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
    body { background-color: #111; color: white; }
    </style>
    """, unsafe_allow_html=True)

# =============================
# LOAD MODEL
# =============================
model = joblib.load("model.pkl")

# =============================
# HEADER
# =============================
st.title("💧 Enterprise Decision Support System")
st.caption("Analisis Kelayakan Air Minum Berbasis Machine Learning")

# =============================
# INPUT
# =============================
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

# =============================
# ANALYSIS
# =============================
if st.button("🚀 Analisis Premium"):

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

    st.markdown("## 📊 Hasil Prediksi")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Probabilitas Layak", f"{prob*100:.2f}%")

    with colB:
        if pred == 1:
            st.success("Air Layak Minum")
        else:
            st.error("Air Tidak Layak")

    # =============================
    # GAUGE CHART
    # =============================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text': "Kelayakan (%)"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [0,50], 'color': "red"},
                {'range': [50,75], 'color': "yellow"},
                {'range': [75,100], 'color': "green"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # FEATURE IMPORTANCE
    # =============================
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        features = input_df.columns

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        st.markdown("## 📈 Feature Importance")
        fig2 = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)

    # =============================
    # SHAP EXPLAINABILITY
    # =============================
    try:
        st.markdown("## 🧠 SHAP Explainability")

        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        shap_fig = shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(bbox_inches="tight")

    except:
        st.info("SHAP tidak tersedia untuk model ini.")

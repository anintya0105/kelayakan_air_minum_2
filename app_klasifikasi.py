import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

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
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD MODEL & DATA
# =============================
model = joblib.load("model_random_forest.pkl")

# Dataset untuk evaluasi (WAJIB ADA DI REPO)
dataset = pd.read_csv("water_potability.csv")

X = dataset.drop("Potability", axis=1)
y = dataset["Potability"]

# =============================
# HEADER
# =============================
st.title("💧 Enterprise Decision Support System")
st.subheader("Analisis & Evaluasi Kelayakan Air Minum")

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
        return "🟢 Low Risk"
    elif prob >= 0.5:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

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
    ]], columns=X.columns)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("## 📊 Hasil Prediksi")

    col1, col2 = st.columns(2)

    # GAUGE
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Probabilitas Layak (%)"},
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

    # RESULT
    with col2:
        st.metric("Risk Level", risk_label(prob))
        if pred == 1:
            st.success("### ✅ Air Layak Minum")
        else:
            st.error("### ❌ Air Tidak Layak")

    # =============================
    # FEATURE IMPORTANCE
    # =============================
    st.markdown("## 📈 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    fig2 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation='h'
    )
    st.plotly_chart(fig2, use_container_width=True)

    # =============================
    # CONFUSION MATRIX
    # =============================
    st.markdown("## 📊 Confusion Matrix")

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    fig3 = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    st.pyplot(fig3)

    # =============================
    # ROC CURVE
    # =============================
    st.markdown("## 📈 ROC Curve")

    y_prob = model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig4 = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    st.pyplot(fig4)

    # =============================
    # DISTRIBUTION
    # =============================
    st.markdown("## 📊 Distribusi Probabilitas Dataset")

    fig5 = plt.figure()
    plt.hist(y_prob, bins=20)
    plt.axvline(prob, linestyle="dashed")
    plt.title("Distribusi Probabilitas")
    st.pyplot(fig5)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("DSS Kelayakan Air Minum © 2026 | Enterprise ML Dashboard")

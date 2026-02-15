import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Water Quality AI System",
    page_icon="💧",
    layout="wide"
)

# ======================================
# CUSTOM DARK UI
# ======================================
st.markdown("""
<style>
.big-title {
    font-size:42px !important;
    font-weight:800;
    color:#00C9A7;
}
.card {
    background-color:#1E1E2F;
    padding:20px;
    border-radius:15px;
    box-shadow:0 4px 15px rgba(0,0,0,0.4);
}
.stMetric {
    background-color:#1E1E2F;
    padding:15px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">💧 Enterprise Water Quality AI</p>', unsafe_allow_html=True)
st.caption("Random Forest vs SVM vs Voting Ensemble")

# ======================================
# LOAD DATA
# ======================================
df = pd.read_csv("water_potability.csv")
df = df.dropna()

X = df.drop("Potability", axis=1)
y = df["Potability"]

# ======================================
# LOAD MODELS (.pkl)
# ======================================
rf = joblib.load("random_forest.pkl")
svm = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
ensemble = joblib.load("ensemble.pkl")

# ======================================
# MODEL ACCURACY
# ======================================
rf_acc = accuracy_score(y, rf.predict(X))

X_scaled = scaler.transform(X)
svm_acc = accuracy_score(y, svm.predict(X_scaled))
ensemble_acc = accuracy_score(y, ensemble.predict(X_scaled))

models = ["Random Forest", "SVM", "Ensemble"]
scores = [rf_acc, svm_acc, ensemble_acc]

best_model_name = models[np.argmax(scores)]

# ======================================
# ACCURACY SECTION
# ======================================
st.markdown("## 📊 Model Performance Comparison")

col1, col2, col3 = st.columns(3)

col1.metric("🌳 Random Forest", f"{rf_acc:.3f}")
col2.metric("🔵 SVM", f"{svm_acc:.3f}")
col3.metric("🤖 Ensemble", f"{ensemble_acc:.3f}")

# Performance Chart
fig, ax = plt.subplots()
ax.bar(models, scores)
ax.set_ylim(0,1)
ax.set_ylabel("Accuracy")
ax.set_facecolor("#1E1E2F")
fig.patch.set_facecolor("#0E1117")
st.pyplot(fig)

st.success(f"🏆 Best Model Automatically Selected: {best_model_name}")

# ======================================
# INPUT SECTION
# ======================================
st.markdown("## 🔎 Water Parameter Input")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH", 0.0, 14.0, 7.5)
    hardness = st.number_input("Hardness", 0.0, 500.0, 100.0)
    solids = st.number_input("Solids", 0.0, 50000.0, 300.0)

with col2:
    chloramines = st.number_input("Chloramines", 0.0, 15.0, 2.5)
    sulfate = st.number_input("Sulfate", 0.0, 1000.0, 100.0)
    conductivity = st.number_input("Conductivity", 0.0, 1000.0, 300.0)

with col3:
    organic_carbon = st.number_input("Organic Carbon", 0.0, 30.0, 3.0)
    trihalomethanes = st.number_input("Trihalomethanes", 0.0, 200.0, 50.0)
    turbidity = st.number_input("Turbidity", 0.0, 10.0, 2.0)

input_data = np.array([[ph, hardness, solids, chloramines,
                        sulfate, conductivity, organic_carbon,
                        trihalomethanes, turbidity]])

# ======================================
# PREDICTION
# ======================================
if st.button("🚀 Analyze Water Quality"):

    if best_model_name == "Random Forest":
        prediction = rf.predict(input_data)
        probability = rf.predict_proba(input_data)

    elif best_model_name == "SVM":
        scaled = scaler.transform(input_data)
        prediction = svm.predict(scaled)
        probability = svm.predict_proba(scaled)

    else:
        scaled = scaler.transform(input_data)
        prediction = ensemble.predict(scaled)
        probability = ensemble.predict_proba(scaled)

    prob = probability[0][1]

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Potability Probability (%)"},
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

    if prediction[0] == 1:
        st.success("✅ Water is SAFE for Drinking")
    else:
        st.error("❌ Water is NOT Safe for Drinking")

    st.write(f"Confidence Score: {prob:.3f}")

# ======================================
# FOOTER
# ======================================
st.markdown("---")
st.caption("Enterprise AI Decision Support System © 2026")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="DSS Kelayakan Air Minum",
    page_icon="💧",
    layout="wide"
)

model = joblib.load("model_random_forest.pkl")

st.title("💧 Sistem Pendukung Keputusan Kelayakan Air Minum")

st.write("""
Sistem ini membantu memprioritaskan sampel air 
yang memerlukan penanganan berdasarkan hasil klasifikasi machine learning.
""")

st.divider()

# =========================
# MODE PILIHAN
# =========================
mode = st.radio("Pilih Mode Analisis:",
                ["Analisis 1 Sampel", "Analisis Banyak Sampel (Upload CSV)"])

# =====================================================
# MODE 1: ANALISIS SATU SAMPEL
# =====================================================
if mode == "Analisis 1 Sampel":

    st.subheader("Input Parameter")

    ph = st.number_input("pH", 0.0, 14.0, 7.0)
    hardness = st.number_input("Hardness", value=150.0)
    solid = st.number_input("Total Dissolved Solids", value=20000.0)
    chloramines = st.number_input("Chloramines", value=7.0)
    sulfate = st.number_input("Sulfate", value=300.0)
    conductivity = st.number_input("Conductivity", value=400.0)
    organic_carbon = st.number_input("Organic Carbon", value=10.0)
    trihalomethanes = st.number_input("Trihalomethanes", value=70.0)
    turbidity = st.number_input("Turbidity", value=4.0)

    if st.button("Analisis Sampel"):

        input_data = pd.DataFrame({
            "ph": [ph],
            "hardness": [hardness],
            "solid": [solid],
            "chloramines": [chloramines],
            "sulfate": [sulfate],
            "conductivity": [conductivity],
            "organic_carbon": [organic_carbon],
            "trihalomethanes": [trihalomethanes],
            "turbidity": [turbidity]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.metric("Probabilitas Layak", f"{probability*100:.2f}%")
        st.progress(float(probability))

        if probability >= 0.75:
            st.success("RISIKO RENDAH")
        elif probability >= 0.5:
            st.warning("RISIKO SEDANG")
        else:
            st.error("RISIKO TINGGI - Prioritas Pengujian")

# =====================================================
# MODE 2: ANALISIS BANYAK SAMPEL
# =====================================================
else:

    st.subheader("Upload File CSV")

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        required_columns = [
            "ph","hardness","solid","chloramines","sulfate",
            "conductivity","organic_carbon","trihalomethanes","turbidity"
        ]

        if all(col in df.columns for col in required_columns):

            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:,1]

            df["Prediksi"] = np.where(predictions==1,"Layak","Tidak Layak")
            df["Probabilitas_Layak"] = probabilities

            # Level Risiko
            def risiko(prob):
                if prob >= 0.75:
                    return "Rendah"
                elif prob >= 0.5:
                    return "Sedang"
                else:
                    return "Tinggi"

            df["Level_Risiko"] = df["Probabilitas_Layak"].apply(risiko)

            # Ranking Prioritas (probabilitas terendah = prioritas tinggi)
            df = df.sort_values("Probabilitas_Layak")

            st.subheader("📊 Hasil Analisis & Ranking Prioritas")
            st.dataframe(df)

            # Download hasil
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Hasil Analisis",
                data=csv,
                file_name='hasil_analisis_air.csv',
                mime='text/csv'
            )

            # Visualisasi distribusi risiko
            st.subheader("Distribusi Level Risiko")

            risk_counts = df["Level_Risiko"].value_counts()

            fig, ax = plt.subplots()
            ax.bar(risk_counts.index, risk_counts.values)
            ax.set_ylabel("Jumlah Sampel")
            ax.set_title("Distribusi Risiko")

            st.pyplot(fig)

        else:
            st.error("Format kolom tidak sesuai dengan model.")

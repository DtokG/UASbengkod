import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load semua file yang sudah disimpan
model = joblib.load('model_churn_tuned.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('feature_columns.pkl')

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📊 Telco Churn Prediction App")
st.markdown("Aplikasi ini menggunakan **Tuned Voting Classifier** untuk memprediksi potensi churn pelanggan.")

# Form Input Pelanggan
with st.container():
    st.subheader("Input Data Pelanggan")
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (Bulan)", 0, 100, 12)
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    with col2:
        total = st.number_input("Total Charges", 0.0, 10000.0, 800.0)
        # Tambahkan input lain jika diperlukan

# Tombol Prediksi
if st.button("Prediksi Sekarang"):
    # Siapkan data frame kosong dengan kolom yang sama saat training
    df_input = pd.DataFrame(np.zeros((1, len(features))), columns=features)
    
    # Masukkan nilai input ke kolom yang sesuai
    df_input['tenure'] = tenure
    df_input['MonthlyCharges'] = monthly
    df_input['TotalCharges'] = total
    # (Opsional: tambahkan mapping untuk kolom One-Hot Encoding lainnya di sini)

    # Scaling
    df_scaled = scaler.transform(df_input)
    
    # Prediksi
    prediction = model.predict(df_scaled)
    prob = model.predict_proba(df_scaled)[0][1]

    # Hasil
    st.divider()
    if prediction[0] == 1:
        st.error(f"⚠️ Hasil: Pelanggan Berpotensi **CHURN** (Peluang: {prob:.2%})")
    else:
        st.success(f"✅ Hasil: Pelanggan Cenderung **TETAP** (Peluang Churn: {prob:.2%})")
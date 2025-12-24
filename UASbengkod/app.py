import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

st.title("📊 Telco Customer Churn Prediction App")
st.markdown("""
Aplikasi ini memprediksi kemungkinan pelanggan berhenti berlangganan berdasarkan data demografi, layanan, dan kontrak.
""")

# Pengecekan file model
base_dir = Path(__file__).resolve().parent
model_path = base_dir / 'model_churn_tuned.pkl'
scaler_path = base_dir / 'scaler.pkl'
features_path = base_dir / 'feature_columns.pkl'

missing = [f.name for f in [model_path, scaler_path, features_path] if not f.exists()]

if missing:
    st.error(f"File berikut tidak ditemukan: {', '.join(missing)}")
    st.stop()

# Load Model & Peralatan
model = joblib.load(str(model_path))
scaler = joblib.load(str(scaler_path))
features = joblib.load(str(features_path))

# --- FORM INPUT ---
with st.form("churn_form"):
    st.subheader("📋 Masukkan Data Pelanggan")
    
    # Baris 1: Demografi
    st.markdown("### 👤 Informasi Demografi")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        senior = st.selectbox("Senior Citizen (Lansia)", [0, 1], help="0 = Tidak, 1 = Ya")
    with c3:
        partner = st.selectbox("Memiliki Pasangan?", ["Yes", "No"])
    with c4:
        dependents = st.selectbox("Memiliki Tanggungan?", ["Yes", "No"])

    st.divider()

    # Baris 2: Layanan Telepon & Internet
    st.markdown("### 🌐 Informasi Layanan")
    c5, c6, c7 = st.columns(3)
    with c5:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with c6:
        security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    with c7:
        support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

    st.divider()

    # Baris 3: Informasi Kontrak & Biaya
    st.markdown("### 💰 Informasi Kontrak & Pembayaran")
    c8, c9, c10 = st.columns(3)
    with c8:
        tenure = st.number_input("Tenure (Lama Berlangganan/Bulan)", 0, 100, 12)
        contract = st.selectbox("Tipe Kontrak", ["Month-to-month", "One year", "Two year"])
    with c9:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Metode Pembayaran", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with c10:
        monthly = st.number_input("Biaya Bulanan ($)", 0.0, 200.0, 70.0)
        total = st.number_input("Total Biaya ($)", 0.0, 10000.0, 800.0)

    submitted = st.form_submit_button("🚀 Prediksi Sekarang")

# --- PROSES PREDIKSI ---
if submitted:
    # 1. Simpan input ke Dictionary
    input_dict = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple_lines,
        'InternetService': internet, 'OnlineSecurity': security, 'OnlineBackup': backup,
        'DeviceProtection': protection, 'TechSupport': support, 'StreamingTV': tv,
        'StreamingMovies': movies, 'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total
    }

    # 2. Buat DataFrame awal dengan kolom nol (sesuai jumlah fitur training)
    df_input = pd.DataFrame(np.zeros((1, len(features))), columns=features)

    # 3. Masukkan Fitur Numerik
    numerics = ['tenure', 'SeniorCitizen', 'MonthlyCharges', 'TotalCharges']
    for col in numerics:
        if col in df_input.columns:
            df_input[col] = input_dict[col]

    # 4. Handling One-Hot Encoding Manual
    # Mencocokkan input kategori user ke kolom biner (misal: Contract_One year)
    for col, val in input_dict.items():
        if col not in numerics:
            column_name = f"{col}_{val}"
            if column_name in df_input.columns:
                df_input[column_name] = 1

    # 5. Scaling & Prediksi
    try:
        df_scaled = scaler.transform(df_input)
        prediction = model.predict(df_scaled)
        prob = model.predict_proba(df_scaled)[0][1]

        # 6. Tampilkan Hasil
        st.subheader("🎯 Hasil Analisis")
        if prediction[0] == 1:
            st.error(f"🚨 **PREDIKSI: CHURN** (Pelanggan kemungkinan besar akan BERHENTI)")
            st.warning(f"Probabilitas Churn: **{prob:.2%}.** Segera berikan penawaran khusus!")
        else:
            st.success(f"✅ **PREDIKSI: TETAP (NOT CHURN)** (Pelanggan kemungkinan besar akan BERTAHAN)")
            st.info(f"Probabilitas Churn hanya: **{prob:.2%}.** Pertahankan kualitas layanan.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat pemrosesan: {e}")
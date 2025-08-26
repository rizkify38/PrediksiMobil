import streamlit as st
import pandas as pd
import joblib
import altair as alt

# ==============================
# Load Model
# ==============================
model = joblib.load("best_model_car_price.pkl")

# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(page_title="Prediksi Kelayakan Mobil", layout="wide")

# ==============================
# Sidebar Navigasi
# ==============================
menu = st.sidebar.radio("Navigasi", ["Beranda", "Prediksi Mobil", "Analitik"])

# ==============================
# Halaman Beranda
# ==============================
if menu == "Beranda":
    st.title("🚗 Prediksi Kelayakan Mobil")
    st.write("""
    **Aplikasi ini membantu Anda memprediksi apakah harga mobil layak (Bagus) atau tidak, 
    berdasarkan fitur-fitur seperti merek, jenis bahan bakar, aspirasi, harga, dan lainnya.**

    ### 🔍 Fitur Utama:
    - Prediksi **Bagus / Tidak Bagus** berdasarkan data mobil
    - Tampilkan **probabilitas prediksi** dalam bentuk grafik
    - Dashboard **analitik distribusi prediksi**
    """)

# ==============================
# Halaman Prediksi
# ==============================
elif menu == "Prediksi Mobil":
    st.title("🔍 Prediksi Mobil")

    # Form input
    with st.form("form_prediksi"):
        st.subheader("Masukkan Detail Mobil")

        make = st.selectbox("Merek Mobil", ['toyota', 'honda', 'nissan', 'mazda', 'bmw', 'mercedes'])
        fuel_type = st.selectbox("Jenis Bahan Bakar", ['gas', 'diesel'])
        aspiration = st.selectbox("Aspirasi", ['std', 'turbo'])

        col1, col2, col3 = st.columns(3)
        with col1:
            horsepower = st.number_input("Horsepower", min_value=40, max_value=300, value=100)
        with col2:
            peak_rpm = st.number_input("Peak RPM", min_value=4000, max_value=7000, value=5000)
        with col3:
            year = st.number_input("Tahun", min_value=1980, max_value=2025, value=2015)

        price = st.number_input("Harga (USD)", min_value=500, max_value=100000, value=20000)
        resale_price = st.number_input("Harga Jual Kembali (USD)", min_value=500, max_value=100000, value=15000)

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        # Hitung fitur turunan
        price_ratio = resale_price / price
        depreciation = price - resale_price
        car_age = 2025 - year

        # Data untuk prediksi
        input_data = pd.DataFrame({
            'make': [make],
            'fuel-type': [fuel_type],
            'aspiration': [aspiration],
            'horsepower': [horsepower],
            'peak-rpm': [peak_rpm],
            'price': [price],
            'year': [year],
            'price_ratio': [price_ratio],
            'depreciation': [depreciation],
            'car_age': [car_age]
        })

        # Prediksi
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        label = "Bagus" if prediction == 1 else "Tidak Bagus"
        st.success(f"✅ Prediksi: **{label}**")
        st.write(f"**Probabilitas Bagus:** {probability[1]*100:.2f}%")
        st.write(f"**Probabilitas Tidak Bagus:** {probability[0]*100:.2f}%")

        # Visualisasi probabilitas
        prob_df = pd.DataFrame({
            'Kategori': ['Tidak Bagus', 'Bagus'],
            'Probabilitas': [probability[0]*100, probability[1]*100]
        })

        chart = alt.Chart(prob_df).mark_bar().encode(
            x=alt.X('Kategori', sort=None),
            y='Probabilitas',
            color='Kategori'
        ).properties(title="Visualisasi Probabilitas Prediksi")

        st.altair_chart(chart, use_container_width=True)

# ==============================
# Halaman Analitik
# ==============================
elif menu == "Analitik":
    st.title("📊 Dashboard Analitik")
    st.write("Distribusi prediksi model berdasarkan data historis")

    try:
        df = pd.read_csv("hasil_prediksi.csv")
        if 'target' not in df.columns:
            st.warning("Data historis belum memiliki kolom target.")
        else:
            distribusi = df['target'].value_counts().reset_index()
            distribusi.columns = ['Kategori', 'Jumlah']
            distribusi['Kategori'] = distribusi['Kategori'].map({0: 'Tidak Bagus', 1: 'Bagus'})

            chart = alt.Chart(distribusi).mark_arc().encode(
                theta='Jumlah',
                color='Kategori'
            ).properties(title="Distribusi Target (Bagus vs Tidak Bagus)")

            st.altair_chart(chart, use_container_width=True)
    except FileNotFoundError:
        st.error("File hasil_prediksi.csv tidak ditemukan. Upload file untuk melihat analitik.")

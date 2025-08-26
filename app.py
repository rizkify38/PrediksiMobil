import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Load Model
# ==============================
model = joblib.load("best_model_car_price.pkl")

# ==============================
# Load Data untuk Dashboard
# ==============================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("hasil_prediksi.csv")
        return df
    except FileNotFoundError:
        st.warning("File hasil_prediksi.csv tidak ditemukan. Dashboard akan menampilkan data dummy.")
        return None

dashboard_data = load_data()

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Prediksi Mobil Bagus", layout="wide")

# ==============================
# Sidebar Navigasi
# ==============================
menu = st.sidebar.radio("Navigasi", ["🏠 Beranda", "🔍 Prediksi", "📊 Dashboard Analitik"])

# ==============================
# Halaman Beranda
# ==============================
if menu == "🏠 Beranda":
    st.title("🚗 Prediksi Mobil Bagus")
    st.write("""
    Aplikasi ini menggunakan **Machine Learning (Random Forest)** untuk memprediksi apakah sebuah mobil termasuk **Bagus** atau **Tidak Bagus**.
    
    ### 🔍 Cara Kerja Model:
    - Menggunakan fitur seperti merek, bahan bakar, aspirasi, horsepower, harga, tahun, dll.
    - Menambahkan fitur baru: **price_ratio, depreciation, car_age**.
    - Target: Mobil dianggap **Bagus** jika `price_ratio > 0.8`.

    ### 📌 Fitur Aplikasi:
    ✅ Prediksi mobil berdasarkan input data  
    ✅ Visualisasi probabilitas hasil prediksi  
    ✅ Dashboard analitik untuk melihat distribusi data dan insight  

    Klik menu **Prediksi** di sidebar untuk mulai.
    """)

# ==============================
# Halaman Prediksi
# ==============================
elif menu == "🔍 Prediksi":
    st.title("🔍 Prediksi Mobil Bagus atau Tidak")
    st.write("Isi data berikut untuk melakukan prediksi:")

    # Input Form
    col1, col2 = st.columns(2)
    with col1:
        make = st.selectbox("Merek Mobil", ["toyota", "honda", "bmw", "audi", "nissan", "daihatsu", "suzuki"])
        fuel_type = st.selectbox("Jenis Bahan Bakar", ["gas", "diesel"])
        aspiration = st.selectbox("Aspirasi", ["std", "turbo"])
        year = st.number_input("Tahun", min_value=1980, max_value=2025, value=2015)
    with col2:
        horsepower = st.number_input("Horsepower", min_value=40, max_value=400, value=100)
        peak_rpm = st.number_input("Peak RPM", min_value=3000, max_value=7000, value=5000)
        price = st.number_input("Harga (price)", min_value=500, value=10000)
        resale_price = st.number_input("Harga Jual Kembali (resale_price)", min_value=500, value=8000)

    if st.button("Prediksi"):
        try:
            # Feature Engineering
            price_ratio = resale_price / price
            depreciation = price - resale_price
            car_age = 2025 - year

            input_data = pd.DataFrame([{
                "make": make,
                "fuel-type": fuel_type,
                "aspiration": aspiration,
                "horsepower": horsepower,
                "peak-rpm": peak_rpm,
                "price": price,
                "year": year,
                "price_ratio": price_ratio,
                "depreciation": depreciation,
                "car_age": car_age
            }])

            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            # Hasil Prediksi
            label = "Bagus ✅" if prediction == 1 else "Tidak Bagus ❌"
            st.subheader(f"Hasil Prediksi: **{label}**")
            st.write(f"Probabilitas Bagus: {prediction_proba[1]*100:.2f}%")
            st.write(f"Probabilitas Tidak Bagus: {prediction_proba[0]*100:.2f}%")

            # Visualisasi Probabilitas (Bar Chart)
            st.subheader("Visualisasi Probabilitas")
            fig, ax = plt.subplots()
            ax.bar(["Tidak Bagus", "Bagus"], prediction_proba, color=["red", "green"])
            ax.set_ylabel("Probabilitas")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")

# ==============================
# Halaman Dashboard Analitik
# ==============================
elif menu == "📊 Dashboard Analitik":
    st.title("📊 Dashboard Analitik")
    if dashboard_data is not None:
        st.write("Analisis distribusi prediksi dan fitur dari dataset **hasil_prediksi.csv**.")

        # Pastikan kolom prediction ada
        if "prediction" not in dashboard_data.columns:
            st.error("Kolom 'prediction' tidak ditemukan dalam file hasil_prediksi.csv.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribusi Prediksi")
                pred_counts = dashboard_data["prediction"].value_counts()
                labels = ["Tidak Bagus", "Bagus"] if len(pred_counts) == 2 else pred_counts.index.astype(str)
                fig1, ax1 = plt.subplots()
                ax1.pie(pred_counts, labels=labels, autopct='%1.1f%%', colors=["red", "green"])
                st.pyplot(fig1)

            with col2:
                st.subheader("Rata-rata Harga per Kategori Prediksi")
                avg_price = dashboard_data.groupby("prediction")["price"].mean()
                fig2, ax2 = plt.subplots()
                ax2.bar(labels, avg_price, color=["red", "green"])
                ax2.set_ylabel("Harga Rata-rata")
                st.pyplot(fig2)

            # Tambahan: Distribusi Horsepower
            st.subheader("Distribusi Horsepower per Prediksi")
            fig3, ax3 = plt.subplots()
            for pred in dashboard_data["prediction"].unique():
                subset = dashboard_data[dashboard_data["prediction"] == pred]
                ax3.hist(subset["horsepower"], bins=10, alpha=0.5, label=f"Prediksi {pred}")
            ax3.set_xlabel("Horsepower")
            ax3.set_ylabel("Frekuensi")
            ax3.legend()
            st.pyplot(fig3)

    else:
        st.warning("Data untuk dashboard tidak tersedia. Upload file hasil_prediksi.csv ke project Anda.")

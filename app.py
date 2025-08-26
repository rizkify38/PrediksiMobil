import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ======================
# Load data dan model
# ======================
hasil_prediksi = pd.read_csv("hasil_prediksi.csv")
prediksi_mobil = pd.read_csv("prediksi_mobil.csv")
model = joblib.load("mobil_harga_model.joblib")

# ======================
# Sidebar navigasi
# ======================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", 
                        ["Informasi Mobil", "Prediksi Mobil", "Tren Harga Mobil"])

# ======================
# Halaman 1: Informasi Mobil
# ======================
if page == "Informasi Mobil":
    st.title("📊 Informasi Mobil - Hasil Prediksi")
    st.write("Berikut adalah data hasil prediksi mobil:")
    st.dataframe(hasil_prediksi)

# ======================
# Halaman 2: Prediksi Mobil
# ======================
elif page == "Prediksi Mobil":
    st.title("🚗 Prediksi Harga Mobil")
    st.write("Isi form berikut untuk memprediksi harga mobil:")

    # Form input
    make = st.text_input("Make (Merek Mobil)")
    fuel_type = st.selectbox("Fuel Type", ["gas", "diesel"])
    aspiration = st.selectbox("Aspiration", ["std", "turbo"])
    horsepower = st.number_input("Horsepower", min_value=40, max_value=500, value=100)
    peak_rpm = st.number_input("Peak RPM", min_value=4000, max_value=7000, value=5000)
    price = st.number_input("Price", min_value=500, value=1000)
    year = st.number_input("Year", min_value=1980, max_value=2025, value=2015)
    price_ratio = st.number_input("Price Ratio", min_value=0.1, value=1.0)
    depreciation = st.number_input("Depreciation", min_value=0.0, value=0.1)
    car_age = st.number_input("Car Age", min_value=0, max_value=40, value=5)

    if st.button("Prediksi Harga"):
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

        try:
            prediction = model.predict(input_data)[0]
            st.success(f"💰 Prediksi harga mobil: {prediction:,.2f}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ======================
# Halaman 3: Tren Harga Mobil
# ======================
elif page == "Tren Harga Mobil":
    st.title("📈 Tren Harga Mobil")
    st.write("Visualisasi tren harga mobil berdasarkan prediksi:")

    if "year" in prediksi_mobil.columns and "price" in prediksi_mobil.columns:
        fig, ax = plt.subplots()
        prediksi_mobil.groupby("year")["price"].mean().plot(ax=ax, marker="o")
        ax.set_ylabel("Harga Rata-rata")
        ax.set_xlabel("Tahun")
        ax.set_title("Tren Harga Mobil per Tahun")
        st.pyplot(fig)
    else:
        st.error("Data prediksi_mobil tidak memiliki kolom 'year' dan 'price'.")

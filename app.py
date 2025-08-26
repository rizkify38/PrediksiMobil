import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# ======================
# Load data dan model dengan error handling
# ======================
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"File {file_path} tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py")
        return pd.DataFrame()

@st.cache_resource
def load_model(file_path):
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        st.error(f"Model {file_path} tidak ditemukan.")
        return None

hasil_prediksi = load_data("hasil_prediksi.csv")
prediksi_mobil = load_data("prediksi_mobil.csv")
model = load_model("mobil_harga_model.joblib")

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
    if not hasil_prediksi.empty:
        st.dataframe(hasil_prediksi)
    else:
        st.warning("Data hasil_prediksi kosong atau file tidak ditemukan.")

# ======================
# Halaman 2: Prediksi Mobil
# ======================
elif page == "Prediksi Mobil":
    st.title("🚗 Prediksi Harga Mobil")

    if model is not None:
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
                # Prediksi
                prediction = model.predict(input_data)[0]

                # Jika model mendukung probabilitas
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_data)[0]
                    st.subheader("📊 Probabilitas Prediksi")
                    st.write(f"🔻 Probabilitas Turun: **{proba[0]:.2f}**")
                    st.write(f"🔺 Probabilitas Naik : **{proba[1]:.2f}**")

                # Hasil akhir
                st.success(f"💰 Prediksi akhir: {'Naik' if prediction == 1 else 'Turun'}")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
    else:
        st.warning("Model belum dimuat. Pastikan file model ada.")

# ======================
# Halaman 3: Tren Harga Mobil
# ======================
elif page == "Tren Harga Mobil":
    st.title("📈 Tren Harga Mobil")
    if not prediksi_mobil.empty:
        if "year" in prediksi_mobil.columns and "price" in prediksi_mobil.columns:
            fig, ax = plt.subplots()
            prediksi_mobil.groupby("year")["price"].mean().plot(ax=ax, marker="o")
            ax.set_ylabel("Harga Rata-rata")
            ax.set_xlabel("Tahun")
            ax.set_title("Tren Harga Mobil per Tahun")
            st.pyplot(fig)
        else:
            st.error("Data prediksi_mobil tidak memiliki kolom 'year' dan 'price'.")
    else:
        st.warning("Data prediksi_mobil kosong atau file tidak ditemukan.")


import streamlit as st
import pandas as pd
import joblib
import pickle
import os

# ==============================
# Fungsi konversi model lama ke format joblib terbaru
# ==============================
def convert_old_pickle_to_joblib(old_file, new_file):
    try:
        with open(old_file, "rb") as f:
            model = pickle.load(f, encoding="latin1")
        joblib.dump(model, new_file)
        return model
    except Exception as e:
        st.error(f"Gagal mengonversi model lama: {e}")
        return None

# ==============================
# Fungsi memuat model
# ==============================
@st.cache_resource
def load_model():
    old_model_path = "best_model_car_price.pkl"
    new_model_path = "converted_model.joblib"

    if not os.path.exists(old_model_path):
        st.error("File model tidak ditemukan.")
        return None

    # Coba load dengan joblib
    try:
        return joblib.load(old_model_path)
    except Exception as e1:
        st.warning(f"Joblib gagal memuat model asli: {e1}")
        # Coba pickle biasa
        try:
            with open(old_model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            st.warning(f"Pickle standar gagal: {e2}")
            # Coba pickle encoding latin1
            try:
                return convert_old_pickle_to_joblib(old_model_path, new_model_path)
            except Exception as e3:
                st.error(f"Gagal memuat model: {e3}")
                return None

# ==============================
# Fungsi memuat data CSV
# ==============================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("hasil_prediksi.csv")
    except Exception as e:
        st.error(f"Gagal memuat CSV: {e}")
        return pd.DataFrame()

# ==============================
# Muat data dan model
# ==============================
data = load_data()
model = load_model()

# ==============================
# Sidebar Navigasi
# ==============================
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Informasi Mobil", "Prediksi Mobil"])

# ==============================
# Halaman 1: Informasi Mobil
# ==============================
if menu == "Informasi Mobil":
    st.title("📊 Informasi Data Mobil")
    if not data.empty:
        st.dataframe(data)
        st.write(f"Jumlah data: {len(data)}")
        if "harga" in data.columns:
            st.subheader("Ringkasan Harga")
            st.write(data["harga"].describe())
    else:
        st.warning("Dataset tidak tersedia.")

# ==============================
# Halaman 2: Prediksi Mobil
# ==============================
elif menu == "Prediksi Mobil":
    st.title("🔍 Prediksi Harga Mobil")

    if model is None:
        st.error("Model belum berhasil dimuat.")
    else:
        st.subheader("Masukkan Informasi Mobil")

        # Tentukan input fitur
        if not data.empty:
            fitur_text = [col for col in data.columns if col != "harga"]
        else:
            fitur_text = ["merk", "model", "tahun", "transmisi", "kilometer", "bahan_bakar", "warna"]

        input_data = {}
        for col in fitur_text:
            if col.lower() in ["tahun", "kilometer"]:
                input_data[col] = st.number_input(f"{col}", min_value=0)
            else:
                input_data[col] = st.text_input(f"{col}")

        if st.button("Prediksi Harga"):
            try:
                df_input = pd.DataFrame([input_data])
                prediksi = model.predict(df_input)[0]
                st.success(f"Prediksi Harga Mobil: Rp {prediksi:,.0f}")
            except Exception as e:
                st.error(f"Gagal prediksi: {e}")

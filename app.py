import streamlit as st
import pandas as pd
import joblib
import pickle
import os

# ==============================
# Fungsi untuk memuat data CSV
# ==============================
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("hasil_prediksi.csv")
        return data
    except Exception as e:
        st.error(f"Gagal memuat data CSV: {e}")
        return pd.DataFrame()

# ==============================
# Fungsi untuk memuat model PKL
# ==============================
@st.cache_resource
def load_model():
    model_path = "best_model_car_price.pkl"
    if not os.path.exists(model_path):
        st.error("File model tidak ditemukan.")
        return None

    try:
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Joblib gagal memuat model: {e}")
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            try:
                with open(model_path, "rb") as f:
                    return pickle.load(f, encoding="latin1")
            except Exception as e3:
                st.error(f"Gagal memuat model: {e3}")
                return None

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
    st.write("Data ini berisi informasi mobil yang tersedia dalam dataset.")
    
    if not data.empty:
        st.dataframe(data)
        st.write(f"Jumlah data: {len(data)}")
        
        # Tambahkan analisis sederhana
        if "harga" in data.columns:
            st.subheader("Ringkasan Harga")
            st.write(data["harga"].describe())
    else:
        st.warning("Dataset tidak tersedia atau gagal dimuat.")

# ==============================
# Halaman 2: Prediksi Mobil
# ==============================
elif menu == "Prediksi Mobil":
    st.title("🔍 Prediksi Harga Mobil")

    if model is None:
        st.error("Model belum berhasil dimuat. Pastikan file PKL valid.")
    else:
        # Form input prediksi
        st.subheader("Masukkan Informasi Mobil")
        
        # Tentukan input sesuai dengan kolom fitur model
        # Jika pipeline, kita ambil dari dataset contoh
        if not data.empty:
            example = data.iloc[0]
            fitur_text = [col for col in data.columns if col != "harga"]
        else:
            # Jika data kosong, tentukan fitur manual
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
                st.error(f"Gagal melakukan prediksi: {e}")



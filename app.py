import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load Data & Model
# ==============================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("hasil_prediksi.csv")
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        return joblib.load("best_model_car_price.pkl")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

data = load_data()
model = load_model()

st.set_page_config(page_title="Prediksi Harga Mobil", layout="wide")

# Debugging: tampilkan info model dan data
with st.expander("🔍 Debug Info"):
    st.write("**Data Columns:**", data.columns.tolist() if not data.empty else "Data kosong")
    st.write("**Model Type:**", type(model))
    if model is not None:
        st.write(model)

# Sidebar
menu = st.sidebar.radio("Pilih Halaman:", ["📄 Informasi Mobil", "🔍 Prediksi Harga Mobil"])

# ==============================
# Halaman 1: Informasi Mobil
# ==============================
if menu == "📄 Informasi Mobil":
    st.title("📄 Informasi Mobil")
    if data.empty:
        st.error("Dataset tidak ditemukan atau kosong.")
    else:
        st.dataframe(data)

        st.subheader("Ringkasan Statistik")
        st.write(data.describe())

        # Filter merk jika ada
        if 'merk' in data.columns:
            merk = st.selectbox("Pilih Merk Mobil:", options=["Semua"] + sorted(data['merk'].dropna().unique()))
            if merk != "Semua":
                st.write(data[data['merk'] == merk])
            else:
                st.write(data)
        else:
            st.warning("Kolom 'merk' tidak ditemukan di dataset.")

# ==============================
# Halaman 2: Prediksi Harga Mobil
# ==============================
elif menu == "🔍 Prediksi Harga Mobil":
    st.title("🔍 Prediksi Harga Mobil")

    if model is None:
        st.error("Model belum berhasil dimuat.")
    else:
        # Tentukan fitur berdasarkan dataset (fallback jika model tidak punya feature_names_in_)
        if not data.empty:
            feature_names = [col for col in data.columns if col != 'harga']
        else:
            feature_names = ["merk", "tahun", "transmisi", "jarak_tempuh", "bahan_bakar", "kapasitas_mesin"]

        st.write("Masukkan detail mobil untuk prediksi:")

        # Buat form input otomatis
        input_dict = {}
        for col in feature_names:
            if col in ['tahun', 'kapasitas_mesin', 'jarak_tempuh']:
                input_dict[col] = st.number_input(f"{col}", min_value=0, value=2020 if col == 'tahun' else 1000)
            else:
                if col in data.columns:
                    options = sorted(data[col].dropna().unique())
                    if len(options) > 0:
                        input_dict[col] = st.selectbox(f"{col}", options=options)
                    else:
                        input_dict[col] = st.text_input(f"{col}")
                else:
                    input_dict[col] = st.text_input(f"{col}")

        if st.button("Prediksi Harga"):
            try:
                input_df = pd.DataFrame([input_dict])
                prediksi = model.predict(input_df)[0]
                st.success(f"💰 Prediksi Harga Mobil: Rp {prediksi:,.0f}")
            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")

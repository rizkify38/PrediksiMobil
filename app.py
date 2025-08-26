import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load Data & Model
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("hasil_prediksi.csv")

@st.cache_resource
def load_model():
    return joblib.load("best_model_car_price.pkl")

data = load_data()
model = load_model()

# ==============================
# Streamlit Pages
# ==============================
st.set_page_config(page_title="Prediksi Harga Mobil", layout="wide")

# Sidebar untuk navigasi
menu = st.sidebar.radio("Pilih Halaman:", ["📄 Informasi Mobil", "🔍 Prediksi Harga Mobil"])

# ==============================
# Halaman 1: Informasi Mobil
# ==============================
if menu == "📄 Informasi Mobil":
    st.title("📄 Informasi Mobil")
    st.write("Data ini berisi hasil prediksi harga mobil yang sudah diproses sebelumnya.")

    # Tampilkan data
    st.dataframe(data)

    # Statistik ringkas
    st.subheader("Ringkasan Statistik")
    st.write(data.describe())

    # Filter berdasarkan merk
    merk = st.selectbox("Pilih Merk Mobil:", options=["Semua"] + list(data['merk'].unique()))
    if merk != "Semua":
        st.write(data[data['merk'] == merk])
    else:
        st.write(data)

# ==============================
# Halaman 2: Prediksi Harga Mobil
# ==============================
elif menu == "🔍 Prediksi Harga Mobil":
    st.title("🔍 Prediksi Harga Mobil")
    st.write("Masukkan detail mobil untuk memprediksi harga.")

    # Form input
    merk = st.selectbox("Merk", options=data['merk'].unique())
    tahun = st.number_input("Tahun Pembuatan", min_value=1990, max_value=2025, value=2020)
    transmisi = st.selectbox("Transmisi", options=data['transmisi'].unique())
    jarak_tempuh = st.number_input("Jarak Tempuh (km)", min_value=0, value=50000, step=1000)
    bahan_bakar = st.selectbox("Bahan Bakar", options=data['bahan_bakar'].unique())
    kapasitas_mesin = st.number_input("Kapasitas Mesin (cc)", min_value=600, max_value=5000, value=1500, step=100)

    # Tombol Prediksi
    if st.button("Prediksi Harga"):
        # Siapkan data input
        input_data = pd.DataFrame({
            'merk': [merk],
            'tahun': [tahun],
            'transmisi': [transmisi],
            'jarak_tempuh': [jarak_tempuh],
            'bahan_bakar': [bahan_bakar],
            'kapasitas_mesin': [kapasitas_mesin]
        })

        # Prediksi
        prediksi = model.predict(input_data)[0]

        st.success(f"💰 Prediksi Harga Mobil: Rp {prediksi:,.0f}")


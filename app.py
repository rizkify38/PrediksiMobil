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

# Ambil feature names dari pipeline (jika ada)
try:
    feature_names = model.feature_names_in_
except AttributeError:
    feature_names = data.columns.tolist()

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

    # Filter berdasarkan kolom jika ada 'merk'
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
    st.write("Masukkan detail mobil untuk memprediksi harga.")

    # Buat input sesuai feature_names
    input_dict = {}
    for col in feature_names:
        if col in ['tahun', 'kapasitas_mesin', 'jarak_tempuh']:
            input_dict[col] = st.number_input(f"{col}", min_value=0, value=2020 if col == 'tahun' else 1000)
        else:
            # Untuk kolom kategori, ambil opsi dari dataset
            if col in data.columns:
                options = sorted(data[col].dropna().unique())
                if len(options) > 0:
                    input_dict[col] = st.selectbox(f"{col}", options=options)
                else:
                    input_dict[col] = st.text_input(f"{col}")
            else:
                input_dict[col] = st.text_input(f"{col}")

    # Tombol Prediksi
    if st.button("Prediksi Harga"):
        input_df = pd.DataFrame([input_dict])
        try:
            prediksi = model.predict(input_df)[0]
            st.success(f"💰 Prediksi Harga Mobil: Rp {prediksi:,.0f}")
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")



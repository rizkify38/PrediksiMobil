import streamlit as st
import pandas as pd
import joblib
import pickle

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Prediksi Harga Mobil", layout="wide")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("hasil_prediksi.csv")
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# ==============================
# LOAD MODEL (dengan fallback encoding latin1)
# ==============================
@st.cache_resource
def load_model():
    model = None
    try:
        model = joblib.load("best_model_car_price.pkl")
    except Exception:
        try:
            with open("best_model_car_price.pkl", "rb") as f:
                model = pickle.load(f)
        except Exception:
            try:
                with open("best_model_car_price.pkl", "rb") as f:
                    model = pickle.load(f, encoding="latin1")
            except Exception as e:
                st.error(f"Gagal memuat model (semua metode gagal): {e}")
                return None
    return model

# ==============================
# LOAD
# ==============================
data = load_data()
model = load_model()

# ==============================
# DEBUG INFO
# ==============================
with st.expander("🔍 Debug Info"):
    st.write("**Kolom Dataset:**", data.columns.tolist() if not data.empty else "Dataset kosong")
    st.write("**Tipe Model:**", type(model))

# ==============================
# SIDEBAR MENU
# ==============================
menu = st.sidebar.radio("Pilih Halaman:", ["📄 Informasi Mobil", "🔍 Prediksi Harga Mobil"])

# ==============================
# HALAMAN 1: INFORMASI MOBIL
# ==============================
if menu == "📄 Informasi Mobil":
    st.title("📄 Informasi Mobil")
    if data.empty:
        st.error("Dataset tidak ditemukan atau kosong.")
    else:
        st.dataframe(data)

        st.subheader("Ringkasan Statistik")
        st.write(data.describe())

        # Filter jika ada kolom kategorikal
        cat_cols = data.select_dtypes(include=['object']).columns.tolist()
        if len(cat_cols) > 0:
            filter_col = st.selectbox("Pilih kolom untuk filter:", options=cat_cols)
            unique_vals = ["Semua"] + sorted(data[filter_col].dropna().unique())
            selected_val = st.selectbox(f"Pilih {filter_col}:", options=unique_vals)
            if selected_val != "Semua":
                st.write(data[data[filter_col] == selected_val])
            else:
                st.write(data)
        else:
            st.warning("Tidak ada kolom kategorikal untuk filter.")

# ==============================
# HALAMAN 2: PREDIKSI HARGA MOBIL
# ==============================
elif menu == "🔍 Prediksi Harga Mobil":
    st.title("🔍 Prediksi Harga Mobil")

    if model is None:
        st.error("Model belum berhasil dimuat.")
    else:
        # Tentukan fitur berdasarkan dataset (kecuali kolom harga)
        if not data.empty:
            feature_names = [col for col in data.columns if col.lower() not in ['harga', 'prediksi', 'price']]
        else:
            st.warning("Dataset kosong, gunakan input default.")
            feature_names = ["merk", "tahun", "transmisi", "jarak_tempuh", "bahan_bakar", "kapasitas_mesin"]

        st.write("Masukkan detail mobil untuk prediksi:")

        input_dict = {}
        for col in feature_names:
            if not data.empty and col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    default_val = int(data[col].median())
                    input_dict[col] = st.number_input(f"{col}", min_value=0, value=default_val)
                else:
                    options = sorted(data[col].dropna().unique())
                    input_dict[col] = st.selectbox(f"{col}", options=options)
            else:
                input_dict[col] = st.text_input(f"{col}")

        if st.button("Prediksi Harga"):
            try:
                input_df = pd.DataFrame([input_dict])
                prediksi = model.predict(input_df)[0]
                st.success(f"💰 Prediksi Harga Mobil: Rp {prediksi:,.0f}")
            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")


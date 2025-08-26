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

# Debug Info
with st.expander("🔍 Debug Info"):
    st.write("**Kolom Dataset:**", data.columns.tolist() if not data.empty else "Dataset kosong")
    st.write("**Tipe Model:**", type(model))

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

        # Cari kolom kategorikal untuk filter (jika ada)
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
# Halaman 2: Prediksi Harga Mobil
# ==============================
elif menu == "🔍 Prediksi Harga Mobil":
    st.title("🔍 Prediksi Harga Mobil")

    if model is None:
        st.error("Model belum berhasil dimuat.")
    else:
        # Gunakan kolom dataset selain target (misal 'harga')
        if not data.empty:
            feature_names = [col for col in data.columns if col.lower() not in ['harga', 'prediksi', 'price']]
        else:
            feature_names = []

        st.write("Masukkan detail mobil untuk prediksi:")

        input_dict = {}
        for col in feature_names:
            if pd.api.types.is_numeric_dtype(data[col]):
                input_dict[col] = st.number_input(f"{col}", min_value=0, value=int(data[col].median()))
            else:
                options = sorted(data[col].dropna().unique())
                if len(options) > 0:
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

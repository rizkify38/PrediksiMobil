import streamlit as st
import pandas as pd
import joblib
import pickle
import os

# ==============================
# Fungsi konversi model lama ke joblib
# ==============================
def convert_old_pickle_to_joblib(old_file, new_file):
    try:
        with open(old_file, "rb") as f:
            model = pickle.load(f, encoding="latin1")
        joblib.dump(model, new_file)
        return model
    except Exception as e:
        st.error(f"Gagal konversi model: {e}")
        return None

# ==============================
# Fungsi memuat model
# ==============================
def load_model_from_file(file):
    try:
        return joblib.load(file)
    except Exception as e1:
        st.warning(f"Joblib gagal: {e1}")
        try:
            with open(file, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            st.warning(f"Pickle standar gagal: {e2}")
            try:
                new_file = "converted_model.joblib"
                return convert_old_pickle_to_joblib(file, new_file)
            except Exception as e3:
                st.error(f"Gagal memuat model: {e3}")
                return None

# ==============================
# Sidebar Upload
# ==============================
st.sidebar.title("⚙️ Pengaturan")
uploaded_model = st.sidebar.file_uploader("Upload Model (.pkl)", type=["pkl", "joblib"])
uploaded_csv = st.sidebar.file_uploader("Upload Data CSV", type=["csv"])

# ==============================
# Load Data
# ==============================
if uploaded_csv is not None:
    data = pd.read_csv(uploaded_csv)
else:
    data = pd.DataFrame()

# ==============================
# Load Model
# ==============================
model = None
if uploaded_model is not None:
    temp_model_path = "uploaded_model.pkl"
    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.read())
    model = load_model_from_file(temp_model_path)

# ==============================
# Navigasi
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
        st.warning("Dataset belum diupload.")

# ==============================
# Halaman 2: Prediksi Mobil
# ==============================
elif menu == "Prediksi Mobil":
    st.title("🔍 Prediksi Harga Mobil")
    if model is None:
        st.error("Model belum berhasil dimuat. Upload file model yang benar.")
    else:
        if data.empty:
            st.warning("Upload dataset agar fitur input otomatis tersedia.")
            fitur_text = ["merk", "model", "tahun", "transmisi", "kilometer", "bahan_bakar", "warna"]
        else:
            fitur_text = [col for col in data.columns if col.lower() != "harga"]

        st.subheader("Masukkan Informasi Mobil")
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

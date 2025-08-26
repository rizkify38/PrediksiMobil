import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================
# Load data & model
# ==========================
@st.cache_resource
def load_resources():
    df = pd.read_csv("prediksi_mobil.csv")
    model = joblib.load("mobil_harga_model.joblib")
    return df, model

df, model = load_resources()

# ==========================
# Sidebar Navigasi
# ==========================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi Mobil", "Prediksi Mobil", "Tren Harga Mobil"])

# ==========================
# Halaman 1: Informasi Mobil
# ==========================
if page == "Informasi Mobil":
    st.title("📊 Informasi Mobil")
    st.write("Data dari **prediksi_mobil.csv**:")

    st.dataframe(df)

    st.subheader("Ringkasan Statistik")
    st.write(df.describe(include="all"))

# ==========================
# Halaman 2: Prediksi Mobil
# ==========================
elif page == "Prediksi Mobil":
    st.title("🔮 Prediksi Mobil")
    st.write("Isi form berikut untuk memprediksi kelas mobil:")

    # ==== INPUT FORM ====
    col1, col2 = st.columns(2)

    with col1:
        make = st.selectbox("Make", sorted(df["make"].unique()))
        fuel_type = st.selectbox("Fuel Type", df["fuel-type"].unique())
        aspiration = st.selectbox("Aspiration", df["aspiration"].unique())
        horsepower = st.number_input("Horsepower", value=100)
        peak_rpm = st.number_input("Peak RPM", value=5000)

    with col2:
        price = st.number_input("Price", value=10000.0)
        year = st.number_input("Year", value=2015)
        price_ratio = st.number_input("Price Ratio", value=1.0)
        depreciation = st.number_input("Depreciation", value=2000.0)
        car_age = st.number_input("Car Age", value=5)

    # Tombol Prediksi
    if st.button("Prediksi"):
        input_data = pd.DataFrame([[
            make, fuel_type, aspiration, horsepower, peak_rpm,
            price, year, price_ratio, depreciation, car_age
        ]], columns=[
            "make", "fuel-type", "aspiration", "horsepower", "peak-rpm",
            "price", "year", "price_ratio", "depreciation", "car_age"
        ])

        try:
            prediction = model.predict(input_data)
            st.success(f"✅ Hasil Prediksi: {prediction[0]}")
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")

# ==========================
# Halaman 3: Tren Harga Mobil
# ==========================
elif page == "Tren Harga Mobil":
    st.title("📈 Tren Harga Mobil")
    st.write("Visualisasi tren harga berdasarkan data historis dari **prediksi_mobil.csv**.")

    selected_make = st.selectbox("Pilih Merek Mobil", sorted(df["make"].unique()))
    df_make = df[df["make"] == selected_make]

    if df_make.empty:
        st.warning("Data tidak tersedia untuk merek ini.")
    else:
        fig, ax = plt.subplots()
        ax.plot(df_make["year"], df_make["price"], marker="o", label="Harga")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Harga")
        ax.set_title(f"Tren Harga Mobil - {selected_make}")
        ax.legend()
        st.pyplot(fig)

import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load data & model
# ==========================
@st.cache_resource
def load_data():
    return pd.read_csv("hasil_prediksi.csv")

@st.cache_resource
def load_model():
    return joblib.load("car_classification_model.joblib")

df = load_data()
model = load_model()

# ==========================
# Sidebar Navigasi
# ==========================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi Mobil", "Prediksi Mobil"])

# ==========================
# Halaman 1: Informasi Mobil
# ==========================
if page == "Informasi Mobil":
    st.title("📊 Informasi Mobil")
    st.write("Data dari **hasil_prediksi.csv**:")

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
        symboling = st.number_input("Symboling", value=0)
        normalized_losses = st.number_input("Normalized Losses", value=100.0)
        make = st.selectbox("Make", sorted(df["make"].unique()))
        fuel_type = st.selectbox("Fuel Type", df["fuel-type"].unique())
        aspiration = st.selectbox("Aspiration", df["aspiration"].unique())
        num_of_doors = st.selectbox("Num of Doors", df["num-of-doors"].unique())
        body_style = st.selectbox("Body Style", df["body-style"].unique())
        drive_wheels = st.selectbox("Drive Wheels", df["drive-wheels"].unique())
        engine_location = st.selectbox("Engine Location", df["engine-location"].unique())
        wheel_base = st.number_input("Wheel Base", value=95.0)
        length = st.number_input("Length", value=160.0)
        width = st.number_input("Width", value=65.0)
        height = st.number_input("Height", value=50.0)
        curb_weight = st.number_input("Curb Weight", value=2000)

    with col2:
        engine_type = st.selectbox("Engine Type", df["engine-type"].unique())
        num_of_cylinders = st.selectbox("Num of Cylinders", df["num-of-cylinders"].unique())
        engine_size = st.number_input("Engine Size", value=120)
        fuel_system = st.selectbox("Fuel System", df["fuel-system"].unique())
        bore = st.number_input("Bore", value=3.0)
        stroke = st.number_input("Stroke", value=3.0)
        compression_ratio = st.number_input("Compression Ratio", value=9.0)
        horsepower = st.number_input("Horsepower", value=100)
        peak_rpm = st.number_input("Peak RPM", value=5000)
        city_mpg = st.number_input("City MPG", value=25)
        highway_mpg = st.number_input("Highway MPG", value=30)
        price = st.number_input("Price", value=10000.0)
        year = st.number_input("Year", value=2015)

    # Tombol Prediksi
    if st.button("Prediksi"):
        input_data = pd.DataFrame([[
            symboling, normalized_losses, make, fuel_type, aspiration,
            num_of_doors, body_style, drive_wheels, engine_location,
            wheel_base, length, width, height, curb_weight, engine_type,
            num_of_cylinders, engine_size, fuel_system, bore, stroke,
            compression_ratio, horsepower, peak_rpm, city_mpg, highway_mpg,
            price, year
        ]], columns=[
            "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
            "num-of-doors", "body-style", "drive-wheels", "engine-location",
            "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
            "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
            "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",
            "price", "year"
        ])

        try:
            prediction = model.predict(input_data)
            st.success(f"✅ Hasil Prediksi: {prediction[0]}")
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")



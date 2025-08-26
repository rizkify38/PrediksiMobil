import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Car Classification App",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CSS Styling
# ==============================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .probability-bar {
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Data & Model Functions
# ==============================
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        data = pd.read_csv("/mnt/data/hasil_prediksi.csv")
        return data
    except FileNotFoundError:
        st.error("❌ File dataset tidak ditemukan!")
        return None

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    try:
        model = joblib.load("/mnt/data/car_classification_model.joblib")
        return model
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan!")
        return None

# ==============================
# Utility Functions
# ==============================
def detect_column_type(series):
    """Automatically detect the best input type for a column"""
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 10 and all(isinstance(x, (int, np.integer)) for x in series.dropna()):
            return "select_numeric"  # Integer with few unique values
        else:
            return "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    elif pd.api.types.is_bool_dtype(series):
        return "boolean"
    else:
        return "categorical"

def create_smart_input(col_name, series, key_prefix=""):
    """Create appropriate input widget based on column type"""
    col_type = detect_column_type(series)
    
    if col_type == "numeric":
        col1, col2 = st.columns([1, 1])
        with col1:
            min_val = float(series.min())
            max_val = float(series.max())
            mean_val = float(series.mean())
            
            # Determine step size based on data range
            if max_val - min_val > 1000:
                step = 10.0
            elif max_val - min_val > 100:
                step = 1.0
            else:
                step = 0.1
            
            value = st.number_input(
                f"📊 {col_name}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=step,
                key=f"{key_prefix}_{col_name}",
                help=f"Range: {min_val:.2f} - {max_val:.2f}"
            )
        
        with col2:
            # Show distribution plot
            fig = px.histogram(
                x=series.dropna(),
                title=f"Distribusi {col_name}",
                height=200
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
                font=dict(size=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        return value
    
    elif col_type == "select_numeric":
        unique_vals = sorted(series.dropna().unique())
        return st.selectbox(
            f"🔢 {col_name}",
            options=unique_vals,
            index=len(unique_vals)//2,
            key=f"{key_prefix}_{col_name}",
            help=f"Pilihan: {unique_vals}"
        )
    
    elif col_type == "categorical":
        unique_vals = series.dropna().unique().tolist()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            value = st.selectbox(
                f"📝 {col_name}",
                options=unique_vals,
                key=f"{key_prefix}_{col_name}",
                help=f"Total pilihan: {len(unique_vals)}"
            )
        
        with col2:
            # Show value counts
            value_counts = series.value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribusi {col_name}",
                height=200
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
                font=dict(size=8)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        return value
    
    elif col_type == "boolean":
        return st.checkbox(
            f"☑️ {col_name}",
            value=bool(series.mode().iloc[0] if not series.mode().empty else False),
            key=f"{key_prefix}_{col_name}"
        )
    
    elif col_type == "date":
        min_date = series.min()
        max_date = series.max()
        default_date = series.median()
        return st.date_input(
            f"📅 {col_name}",
            value=default_date,
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_{col_name}"
        )
    
    else:
        return st.text_input(f"✏️ {col_name}", key=f"{key_prefix}_{col_name}")

def create_prediction_visualization(probabilities, classes):
    """Create interactive visualization for prediction probabilities"""
    # Sort probabilities in descending order
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_classes = [classes[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=sorted_classes,
            x=sorted_probs,
            orientation='h',
            marker=dict(
                color=sorted_probs,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Probability")
            ),
            text=[f"{p:.1%}" for p in sorted_probs],
            textposition='inside'
        )
    ])
    
    fig.update_layout(
        title="Probabilitas Prediksi untuk Setiap Kelas",
        xaxis_title="Probabilitas",
        yaxis_title="Kelas",
        height=max(300, len(classes) * 50),
        showlegend=False
    )
    
    return fig

# ==============================
# Load Data and Model
# ==============================
data = load_data()
model = load_model()

if data is None or model is None:
    st.stop()

# ==============================
# Sidebar Navigation
# ==============================
st.sidebar.markdown("# 🚗 Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Dashboard", "📊 Eksplorasi Data", "🔮 Prediksi Cerdas", "📈 Analisis Model"],
    index=0
)

# ==============================
# Dashboard
# ==============================
if menu == "🏠 Dashboard":
    st.markdown('<h1 class="main-header">🚗 Sistem Klasifikasi Mobil Cerdas</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Data",
            value=f"{len(data):,}",
            delta="rows"
        )
    
    with col2:
        st.metric(
            label="🔢 Jumlah Fitur",
            value=len([col for col in data.columns if col != "prediksi"]),
            delta="features"
        )
    
    with col3:
        unique_predictions = data['prediksi'].nunique() if 'prediksi' in data.columns else 0
        st.metric(
            label="🏷️ Kelas Target",
            value=unique_predictions,
            delta="classes"
        )
    
    with col4:
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        st.metric(
            label="📈 Kolom Numerik",
            value=numeric_cols,
            delta="columns"
        )
    
    st.markdown("---")
    
    # Dataset preview
    st.subheader("👁️ Preview Dataset")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Quick insights
    if 'prediksi' in data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribusi Kelas Target")
            target_counts = data['prediksi'].value_counts()
            fig = px.pie(
                values=target_counts.values,
                names=target_counts.index,
                title="Distribusi Kelas Prediksi"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Ringkasan Statistik")
            st.dataframe(data.describe(), use_container_width=True)

# ==============================
# Data Exploration Page
# ==============================
elif menu == "📊 Eksplorasi Data":
    st.markdown('<h1 class="main-header">📊 Eksplorasi Data Mendalam</h1>', unsafe_allow_html=True)
    
    # Data info section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ℹ️ Informasi Dataset")
        info_data = {
            "Jumlah Baris": len(data),
            "Jumlah Kolom": len(data.columns),
            "Missing Values": data.isnull().sum().sum(),
            "Duplicates": data.duplicated().sum(),
            "Memory Usage": f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        for key, value in info_data.items():
            st.metric(key, value)
    
    with col2:
        st.subheader("📋 Tipe Data Kolom")
        dtype_df = pd.DataFrame({
            'Column': data.columns,
            'Data Type': [str(dt) for dt in data.dtypes],
            'Non-Null Count': [data[col].notna().sum() for col in data.columns],
            'Unique Values': [data[col].nunique() for col in data.columns]
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    # Interactive column analysis
    st.subheader("🔍 Analisis Kolom Interaktif")
    selected_column = st.selectbox(
        "Pilih kolom untuk dianalisis:",
        data.columns.tolist()
    )
    
    if selected_column:
        col_data = data[selected_column]
        col1, col2 = st.columns(2)
        
        with col1:
            if pd.api.types.is_numeric_dtype(col_data):
                fig = px.histogram(
                    data,
                    x=selected_column,
                    title=f"Distribusi {selected_column}",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                value_counts = col_data.value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Frekuensi {selected_column}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Statistik Deskriptif:**")
            if pd.api.types.is_numeric_dtype(col_data):
                stats = col_data.describe()
                st.dataframe(stats.to_frame(), use_container_width=True)
            else:
                stats = {
                    'Count': col_data.count(),
                    'Unique': col_data.nunique(),
                    'Top': col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A',
                    'Freq': col_data.value_counts().iloc[0] if not col_data.empty else 0,
                    'Missing': col_data.isnull().sum()
                }
                st.dataframe(pd.Series(stats).to_frame(name='Value'), use_container_width=True)

# ==============================
# Smart Prediction Page
# ==============================
elif menu == "🔮 Prediksi Cerdas":
    st.markdown('<h1 class="main-header">🔮 Prediksi Mobil Cerdas</h1>', unsafe_allow_html=True)
    
    # Get feature columns (exclude target if exists)
    feature_columns = [col for col in data.columns if col not in ['prediksi', 'prediction', 'target']]
    
    st.write("### 📝 Masukkan Fitur Mobil")
    st.write("*Sistem akan secara otomatis menyesuaikan tipe input berdasarkan karakteristik data*")
    
    # Create input form
    with st.form("prediction_form"):
        input_data = {}
        
        # Group columns by type for better organization
        numeric_cols = []
        categorical_cols = []
        
        for col in feature_columns:
            col_type = detect_column_type(data[col])
            if col_type in ["numeric", "select_numeric"]:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Numeric features section
        if numeric_cols:
            st.subheader("🔢 Fitur Numerik")
            for col in numeric_cols:
                input_data[col] = create_smart_input(col, data[col], "pred")
        
        # Categorical features section
        if categorical_cols:
            st.subheader("📝 Fitur Kategorikal")
            for col in categorical_cols:
                input_data[col] = create_smart_input(col, data[col], "pred")
        
        # Prediction settings
        st.subheader("⚙️ Pengaturan Prediksi")
        show_probabilities = st.checkbox("Tampilkan probabilitas semua kelas", value=True)
        confidence_threshold = st.slider(
            "Threshold confidence minimum (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        ) / 100
        
        # Submit button
        submitted = st.form_submit_button("🚀 Prediksi", use_container_width=True)
    
    # Make prediction when form is submitted
    if submitted:
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_
                max_prob = np.max(probabilities)
                
                # Display main prediction result
                confidence_color = "🟢" if max_prob >= confidence_threshold else "🟡" if max_prob >= 0.3 else "🔴"
                
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>{confidence_color} Hasil Prediksi</h2>
                    <h1>🚗 {prediction}</h1>
                    <p>Confidence: <strong>{max_prob:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show confidence interpretation
                if max_prob >= confidence_threshold:
                    st.success(f"✅ Prediksi dengan confidence tinggi ({max_prob:.1%})")
                elif max_prob >= 0.3:
                    st.warning(f"⚠️ Prediksi dengan confidence sedang ({max_prob:.1%})")
                else:
                    st.error(f"❌ Prediksi dengan confidence rendah ({max_prob:.1%})")
                
                # Show probability visualization
                if show_probabilities:
                    st.subheader("📊 Probabilitas Semua Kelas")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = create_prediction_visualization(probabilities, classes)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("📋 Detail Probabilitas")
                        prob_df = pd.DataFrame({
                            'Kelas': classes,
                            'Probabilitas': probabilities,
                            'Persentase': [f"{p:.1%}" for p in probabilities]
                        }).sort_values('Probabilitas', ascending=False)
                        
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            else:
                # Model doesn't support probability prediction
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>🚗 Hasil Prediksi</h2>
                    <h1>{prediction}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("ℹ️ Model ini tidak mendukung prediksi probabilitas")
            
            # Show input summary
            st.subheader("📋 Ringkasan Input")
            input_summary_df = pd.DataFrame([input_data]).T
            input_summary_df.columns = ['Nilai Input']
            st.dataframe(input_summary_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat melakukan prediksi: {str(e)}")
            st.write("🔍 Detail error untuk debugging:")
            st.code(str(e))

# ==============================
# Model Analysis Page
# ==============================
elif menu == "📈 Analisis Model":
    st.markdown('<h1 class="main-header">📈 Analisis Model</h1>', unsafe_allow_html=True)
    
    # Model information
    st.subheader("ℹ️ Informasi Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Tipe Model:** {type(model).__name__}")
    
    with col2:
        if hasattr(model, 'classes_'):
            st.info(f"**Jumlah Kelas:** {len(model.classes_)}")
    
    with col3:
        feature_count = len([col for col in data.columns if col not in ['prediksi', 'prediction', 'target']])
        st.info(f"**Jumlah Fitur:** {feature_count}")
    
    # Model parameters (if available)
    if hasattr(model, 'get_params'):
        st.subheader("⚙️ Parameter Model")
        params = model.get_params()
        
        # Filter out None values and organize parameters
        important_params = {}
        other_params = {}
        
        important_keys = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 
                         'random_state', 'C', 'kernel', 'gamma', 'n_neighbors']
        
        for key, value in params.items():
            if value is not None:
                if any(imp_key in key for imp_key in important_keys):
                    important_params[key] = value
                else:
                    other_params[key] = value
        
        col1, col2 = st.columns(2)
        
        with col1:
            if important_params:
                st.write("**Parameter Utama:**")
                st.json(important_params)
        
        with col2:
            if other_params:
                st.write("**Parameter Lainnya:**")
                with st.expander("Lihat semua parameter"):
                    st.json(other_params)
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("🎯 Tingkat Kepentingan Fitur")
        
        feature_columns = [col for col in data.columns if col not in ['prediksi', 'prediction', 'target']]
        importances = model.feature_importances_
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Fitur': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Fitur',
                orientation='h',
                title="Tingkat Kepentingan Fitur",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=max(400, len(feature_columns) * 30))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Top 5 Fitur Penting:**")
            st.dataframe(
                importance_df.head().round(4),
                use_container_width=True,
                hide_index=True
            )
    
    # Classes information (if available)
    if hasattr(model, 'classes_'):
        st.subheader("🏷️ Kelas Target")
        classes_df = pd.DataFrame({
            'Index': range(len(model.classes_)),
            'Kelas': model.classes_
        })
        st.dataframe(classes_df, use_container_width=True, hide_index=True)
    
    # Model performance on sample data
    st.subheader("🎯 Performa pada Sample Data")
    
    if st.button("Test Model pada 10 Sample Random"):
        try:
            sample_data = data.sample(n=min(10, len(data)), random_state=42)
            feature_columns = [col for col in data.columns if col not in ['prediksi', 'prediction', 'target']]
            
            X_sample = sample_data[feature_columns]
            predictions = model.predict(X_sample)
            
            result_df = sample_data.copy()
            result_df['Prediksi_Model'] = predictions
            
            if 'prediksi' in sample_data.columns:
                result_df['Match'] = result_df['prediksi'] == result_df['Prediksi_Model']
                accuracy = result_df['Match'].mean()
                st.success(f"Akurasi pada sample: {accuracy:.1%}")
            
            st.dataframe(result_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error saat testing model: {str(e)}")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚗 Sistem Klasifikasi Mobil Cerdas | Dibuat dengan ❤️ menggunakan Streamlit</p>
    <p>📊 Powered by Machine Learning & Interactive Visualization</p>
</div>
""", unsafe_allow_html=True)

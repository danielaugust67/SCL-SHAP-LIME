import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- FUNGSI UNTUK MEMUAT ASET (HANYA DIJALANKAN SEKALI) ---
@st.cache_resource
def load_all_assets():
    """
    Memuat semua model, data, dan objek yang telah disimpan dari folder aset.
    Decorator @st.cache_resource memastikan ini hanya berjalan sekali.
    """
    asset_path = "streamlit_dashboard_assets"
    assets = {}

    # Memuat model
    assets['model_A'] = joblib.load(f"{asset_path}/model_A_without_SCL.pkl")
    assets['model_B'] = joblib.load(f"{asset_path}/model_B_with_SCL.pkl")

    # Memuat pre-processor dan nama
    assets['le'] = joblib.load(f"{asset_path}/label_encoder.pkl")
    assets['scaler'] = joblib.load(f"{asset_path}/standard_scaler.pkl")
    assets['class_names'] = joblib.load(f"{asset_path}/class_names.pkl")
    assets['feature_names_A'] = joblib.load(f"{asset_path}/feature_names_A.pkl")
    assets['feature_names_B'] = joblib.load(f"{asset_path}/feature_names_B.pkl")

    # Memuat hasil klasifikasi
    assets['report_A'] = pd.read_csv(f"{asset_path}/classification_report_A.csv", index_col=0)
    assets['report_B'] = pd.read_csv(f"{asset_path}/classification_report_B.csv", index_col=0)

    # Memuat data & nilai SHAP
    assets['shap_values_A'] = joblib.load(f"{asset_path}/shap_values_A.pkl")
    assets['shap_data_A'] = pd.read_csv(f"{asset_path}/shap_data_A.csv")
    assets['shap_values_B'] = joblib.load(f"{asset_path}/shap_values_B.pkl")
    assets['shap_data_B'] = pd.read_csv(f"{asset_path}/shap_data_B.csv")

    # Memuat penjelasan LIME
    assets['lime_explanation_A'] = joblib.load(f"{asset_path}/lime_explanation_A.pkl")
    assets['lime_explanation_B'] = joblib.load(f"{asset_path}/lime_explanation_B.pkl")

    # Memuat analisis semantik
    assets['correlation_df'] = pd.read_csv(f"{asset_path}/embedding_semantic_correlation.csv", index_col=0)

    return assets

# --- MEMBANGUN UI STREAMLIT ---

st.set_page_config(layout="wide")

# Muat semua aset
try:
    assets = load_all_assets()
except FileNotFoundError:
    st.error("Folder 'streamlit_dashboard_assets' tidak ditemukan. Pastikan folder tersebut berada di direktori yang sama dengan file app.py Anda.")
    st.stop()


# --- SIDEBAR UNTUK KAMUS EMBEDDING ---
st.sidebar.title("📖 Kamus Makna Embedding")
st.sidebar.info(
    "Sidebar ini menunjukkan fitur-fitur asli yang paling kuat berkorelasi "
    "dengan embedding yang dianggap penting oleh model. Ini membantu kita memahami "
    "makna dari setiap dimensi embedding."
)
st.sidebar.header(f"Makna dari {assets['correlation_df'].columns[0]}")
st.sidebar.dataframe(assets['correlation_df'])


# --- KONTEN UTAMA ---
st.title("Dashboard Perbandingan Model Prediksi Status Mahasiswa")
st.success("Semua aset model berhasil dimuat! Gunakan sidebar untuk melihat makna embedding.")


# Tampilkan hasil dalam tab agar rapi
tab1, tab2, tab3 = st.tabs([
    "📊 Laporan Klasifikasi",
    "🌍 Analisis Global (SHAP)",
    "📍 Analisis Lokal (LIME)"
])

with tab1:
    st.header("Perbandingan Laporan Klasifikasi")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model A (Baseline Tanpa SCL)")
        st.dataframe(assets['report_A'])
    with col2:
        st.subheader("Model B (Proposed Dengan SCL)")
        st.dataframe(assets['report_B'])

with tab2:
    st.header("Analisis Kepentingan Fitur Global (SHAP)")
    st.set_option('deprecation.showPyplotGlobalUse', False) # Menonaktifkan warning

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model A (Fitur Asli)")
        fig_A, ax_A = plt.subplots(figsize=(10, 8))
        shap.summary_plot(assets['shap_values_A'], assets['shap_data_A'], plot_type='bar', feature_names=assets['feature_names_A'], show=False)
        st.pyplot(fig_A)

    with col2:
        st.subheader("Model B (Fitur Embedding)")
        fig_B, ax_B = plt.subplots(figsize=(10, 8))
        shap.summary_plot(assets['shap_values_B'], assets['shap_data_B'], plot_type='bar', feature_names=assets['feature_names_B'], show=False)
        st.pyplot(fig_B)


with tab3:
    st.header("Analisis Prediksi Lokal untuk Satu Sampel (LIME)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Penjelasan Model A")
        st.write("Fitur-fitur yang mendukung/menentang prediksi:")
        lime_df_A = pd.DataFrame(assets['lime_explanation_A'], columns=['Fitur', 'Kontribusi'])
        st.dataframe(lime_df_A)

    with col2:
        st.subheader("Penjelasan Model B")
        st.write("Embedding yang mendukung/menentang prediksi:")
        lime_df_B = pd.DataFrame(assets['lime_explanation_B'], columns=['Embedding', 'Kontribusi'])
        st.dataframe(lime_df_B)

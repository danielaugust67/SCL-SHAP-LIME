import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Dashboard Perbandingan Model Prediksi Status Mahasiswa")

# --- FUNGSI LAZY LOADING UNTUK SETIAP ASET ---

@st.cache_resource
def get_classification_reports():
    report_A = pd.read_csv("streamlit_dashboard_assets/classification_report_A.csv", index_col=0)
    report_B = pd.read_csv("streamlit_dashboard_assets/classification_report_B.csv", index_col=0)
    return report_A, report_B

@st.cache_resource
def get_shap_assets():
    shap_values_A = joblib.load("streamlit_dashboard_assets/shap_values_A.pkl")
    shap_data_A = pd.read_csv("streamlit_dashboard_assets/shap_data_A.csv")
    feature_names_A = joblib.load("streamlit_dashboard_assets/feature_names_A.pkl")

    shap_values_B = joblib.load("streamlit_dashboard_assets/shap_values_B.pkl")
    shap_data_B = pd.read_csv("streamlit_dashboard_assets/shap_data_B.csv")
    feature_names_B = joblib.load("streamlit_dashboard_assets/feature_names_B.pkl")
    return (shap_values_A, shap_data_A, feature_names_A), (shap_values_B, shap_data_B, feature_names_B)

@st.cache_resource
def get_lime_assets():
    lime_A = pd.DataFrame(joblib.load("streamlit_dashboard_assets/lime_explanation_A.pkl"), columns=['Fitur', 'Kontribusi'])
    lime_B = pd.DataFrame(joblib.load("streamlit_dashboard_assets/lime_explanation_B.pkl"), columns=['Embedding', 'Kontribusi'])
    return lime_A, lime_B

@st.cache_resource
def get_semantic_assets():
    return pd.read_csv("streamlit_dashboard_assets/embedding_semantic_correlation.csv", index_col=0)


# --- SIDEBAR (Selalu dimuat karena ringan) ---
try:
    correlation_df = get_semantic_assets()
    st.sidebar.title("📖 Kamus Makna Embedding")
    st.sidebar.info("Sidebar ini menunjukkan fitur asli yang paling kuat berkorelasi dengan embedding penting.")
    st.sidebar.header(f"Makna dari {correlation_df.columns[0]}")
    st.sidebar.dataframe(correlation_df)
except Exception as e:
    st.sidebar.error(f"Gagal memuat kamus embedding: {e}")


# --- KONTEN UTAMA DENGAN TAB ---
tab1, tab2, tab3 = st.tabs([
    "📊 Laporan Klasifikasi",
    "🌍 Analisis Global (SHAP)",
    "📍 Analisis Lokal (LIME)"
])

with tab1:
    st.header("Perbandingan Laporan Klasifikasi")
    try:
        report_A, report_B = get_classification_reports()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model A (Baseline Tanpa SCL)")
            st.dataframe(report_A)
        with col2:
            st.subheader("Model B (Proposed Dengan SCL)")
            st.dataframe(report_B)
    except Exception as e:
        st.error(f"Gagal memuat laporan klasifikasi: {e}")

with tab2:
    st.header("Analisis Kepentingan Fitur Global (SHAP)")
    st.info("Aset SHAP berukuran besar dan akan dimuat saat tab ini pertama kali dibuka. Mohon tunggu sebentar...")
    try:
        (shap_values_A, shap_data_A, features_A), (shap_values_B, shap_data_B, features_B) = get_shap_assets()
        st.success("Aset SHAP berhasil dimuat!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model A (Fitur Asli)")
            fig_A, ax_A = plt.subplots()
            shap.summary_plot(shap_values_A, shap_data_A, plot_type='bar', feature_names=features_A, show=False)
            st.pyplot(fig_A)

        with col2:
            st.subheader("Model B (Fitur Embedding)")
            fig_B, ax_B = plt.subplots()
            shap.summary_plot(shap_values_B, shap_data_B, plot_type='bar', feature_names=features_B, show=False)
            st.pyplot(fig_B)
    except Exception as e:
        st.error(f"Gagal memuat aset SHAP. Kemungkinan file terlalu besar untuk memori yang tersedia. Error: {e}")

with tab3:
    st.header("Analisis Prediksi Lokal untuk Satu Sampel (LIME)")
    try:
        lime_df_A, lime_df_B = get_lime_assets()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Penjelasan Model A")
            st.dataframe(lime_df_A)
        with col2:
            st.subheader("Penjelasan Model B")
            st.dataframe(lime_df_B)
    except Exception as e:
        st.error(f"Gagal memuat penjelasan LIME: {e}")

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Konfigurasi halaman dan judul utama
st.set_page_config(layout="wide")
st.title("📊 Dashboard Perbandingan Model Prediksi Status Mahasiswa")

# --- Fungsi Bantuan untuk Plotting LIME ---
def plot_lime_explanation(lime_df, title):
    lime_df['abs_contribution'] = lime_df['Kontribusi'].abs()
    lime_df = lime_df.sort_values('abs_contribution', ascending=True)
    colors = ['#1f77b4' if x > 0 else '#ff7f0e' for x in lime_df['Kontribusi']]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(lime_df['Fitur'], lime_df['Kontribusi'], color=colors)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Kontribusi terhadap Prediksi")
    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

# --- Fungsi Lazy Loading untuk Aset (di-cache) ---
@st.cache_resource
def get_classification_reports():
    return (pd.read_csv("dashboard_assets/classification_report_A.csv", index_col=0),
            pd.read_csv("dashboard_assets/classification_report_B.csv", index_col=0))

@st.cache_resource
def get_shap_assets():
    assets = {
        'shap_values_A': joblib.load("dashboard_assets/shap_values_A.pkl"),
        'shap_data_A': pd.read_csv("dashboard_assets/shap_data_A.csv"),
        'features_A': joblib.load("dashboard_assets/feature_names_A.pkl"),
        'shap_values_B': joblib.load("dashboard_assets/shap_values_B.pkl"),
        'shap_data_B': pd.read_csv("dashboard_assets/shap_data_B.csv"),
        'features_B': joblib.load("dashboard_assets/feature_names_B.pkl")
    }
    return assets

@st.cache_resource
def get_lime_assets():
    return (pd.DataFrame(joblib.load("dashboard_assets/lime_explanation_A.pkl"), columns=['Fitur', 'Kontribusi']),
            pd.DataFrame(joblib.load("dashboard_assets/lime_explanation_B.pkl"), columns=['Fitur', 'Kontribusi']))

@st.cache_resource
def get_semantic_assets():
    return pd.read_csv("dashboard_assets/embedding_semantic_correlation.csv", index_col=0)

# --- Sidebar ---
try:
    correlation_df = get_semantic_assets()
    st.sidebar.title("📖 Kamus Makna Embedding")
    st.sidebar.info("Menunjukkan fitur asli yang paling berkorelasi dengan embedding penting (Emb_86).")
    st.sidebar.dataframe(correlation_df.abs().sort_values(ascending=False).head(10))
except Exception as e:
    st.sidebar.error(f"Gagal memuat kamus embedding: {e}")

# --- Konten Utama dengan Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📊 Laporan Klasifikasi",
    "🌍 Analisis Global (SHAP)",
    "📍 Analisis Lokal (LIME)"
])

with tab1:
    st.header("Perbandingan Kinerja Model")
    try:
        report_A, report_B = get_classification_reports()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model A (Baseline Tanpa SCL)")
            st.dataframe(report_A)
        with col2:
            st.subheader("Model B (Dengan SCL)")
            st.dataframe(report_B)
    except Exception as e:
        st.error(f"Gagal memuat laporan klasifikasi: {e}")

with tab2:
    st.header("Analisis Kepentingan Fitur Global (SHAP)")
    st.info("Aset SHAP akan dimuat saat tab ini dibuka. Mohon tunggu sebentar...")
    try:
        shap_assets = get_shap_assets()
        st.success("Aset SHAP berhasil dimuat!")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model A (Fitur Asli)")
            fig_A, _ = plt.subplots()
            shap.summary_plot(shap_assets['shap_values_A'], shap_assets['shap_data_A'], plot_type='bar', feature_names=shap_assets['features_A'], show=False)
            st.pyplot(fig_A, use_container_width=True)
            plt.close(fig_A)
        with col2:
            st.subheader("Model B (Fitur Embedding)")
            fig_B, _ = plt.subplots()
            shap.summary_plot(shap_assets['shap_values_B'], shap_assets['shap_data_B'], plot_type='bar', feature_names=shap_assets['features_B'], show=False)
            st.pyplot(fig_B, use_container_width=True)
            plt.close(fig_B)
    except Exception as e:
        st.error(f"Gagal memuat aset SHAP. Error: {e}")

with tab3:
    st.header("Analisis Prediksi Lokal (LIME)")
    try:
        lime_df_A, lime_df_B = get_lime_assets()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Penjelasan Model A")
            fig_lime_A = plot_lime_explanation(lime_df_A, "Kontribusi Fitur - Model A")
            st.pyplot(fig_lime_A, use_container_width=True)
            with st.expander("Lihat data mentah"):
                st.dataframe(lime_df_A)
        with col2:
            st.subheader("Penjelasan Model B")
            fig_lime_B = plot_lime_explanation(lime_df_B, "Kontribusi Embedding - Model B")
            st.pyplot(fig_lime_B, use_container_width=True)
            with st.expander("Lihat data mentah"):
                st.dataframe(lime_df_B)
    except Exception as e:
        st.error(f"Gagal memuat penjelasan LIME: {e}")

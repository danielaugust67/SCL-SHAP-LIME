import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.title("Dashboard Perbandingan Model Prediksi Status Mahasiswa")

# ---------- FUNGSI PLOTTING LIME ----------
def plot_lime_explanation(lime_df, title):
    lime_df['abs_contribution'] = lime_df['Kontribusi'].abs()
    lime_df = lime_df.sort_values('abs_contribution', ascending=True)
    colors = ['#1f77b4' if x > 0 else '#ff7f0e' for x in lime_df['Kontribusi']]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(lime_df['Fitur'], lime_df['Kontribusi'], color=colors)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Kontribusi terhadap Prediksi", fontsize=12)
    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    return fig

# ---------- FUNGSI CACHE LOADER ----------
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
    # Memuat dict hasil export dari notebook
    lime_A_data = joblib.load("streamlit_dashboard_assets/lime_explanation_A.pkl")
    lime_B_data = joblib.load("streamlit_dashboard_assets/lime_explanation_B.pkl")
    lime_df_A = pd.DataFrame(lime_A_data["feature_contributions"], columns=['Fitur', 'Kontribusi'])
    lime_df_B = pd.DataFrame(lime_B_data["feature_contributions"], columns=['Fitur', 'Kontribusi'])
    return lime_A_data, lime_df_A, lime_B_data, lime_df_B

@st.cache_resource
def get_semantic_assets():
    return pd.read_csv("streamlit_dashboard_assets/embedding_semantic_correlation.csv", index_col=0)

# ---------- SIDEBAR ----------
try:
    correlation_df = get_semantic_assets()
    st.sidebar.title("📖 Kamus Makna Embedding")
    st.sidebar.info("Sidebar ini menunjukkan fitur asli yang paling kuat berkorelasi dengan embedding penting.")
    st.sidebar.header(f"Makna dari {correlation_df.columns[0]}")
    st.sidebar.dataframe(correlation_df)
except Exception as e:
    st.sidebar.error(f"Gagal memuat kamus embedding: {e}")

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs([
    "📊 Laporan Klasifikasi",
    "🌍 Analisis Global (SHAP)",
    "📍 Analisis Lokal (LIME)"
])

# ----- TAB 1: Laporan Klasifikasi -----
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

# ----- TAB 2: SHAP Global -----
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
            st.pyplot(fig_A, use_container_width=True)
            plt.close(fig_A)
        with col2:
            st.subheader("Model B (Fitur Embedding)")
            fig_B, ax_B = plt.subplots()
            shap.summary_plot(shap_values_B, shap_data_B, plot_type='bar', feature_names=features_B, show=False)
            st.pyplot(fig_B, use_container_width=True)
            plt.close(fig_B)
    except Exception as e:
        st.error(f"Gagal memuat aset SHAP. Kemungkinan file terlalu besar untuk memori yang tersedia. Error: {e}")

# ----- TAB 3: LIME Lokal -----
with tab3:
    st.header("Analisis Prediksi Lokal untuk Satu Sampel (LIME)")
    try:
        lime_A_data, lime_df_A, lime_B_data, lime_df_B = get_lime_assets()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Penjelasan Model A")
            st.markdown(f"""
                - **Index Sampel:** {lime_A_data['instance_idx']}  
                - **Label Asli:** {lime_A_data['true_label_name']} ({lime_A_data['true_label']})  
                - **Prediksi Model:** {lime_A_data['pred_label_name']} ({lime_A_data['pred_label']})
            """)
            fig_lime_A = plot_lime_explanation(lime_df_A, "Kontribusi Fitur - Model A")
            st.pyplot(fig_lime_A, use_container_width=True)
            with st.expander("Lihat data mentah penjelasan LIME Model A"):
                st.dataframe(lime_df_A)
        with col2:
            st.subheader("Penjelasan Model B")
            st.markdown(f"""
                - **Index Sampel:** {lime_B_data['instance_idx']}  
                - **Label Asli:** {lime_B_data['true_label_name']} ({lime_B_data['true_label']})  
                - **Prediksi Model:** {lime_B_data['pred_label_name']} ({lime_B_data['pred_label']})
            """)
            fig_lime_B = plot_lime_explanation(lime_df_B, "Kontribusi Embedding - Model B")
            st.pyplot(fig_lime_B, use_container_width=True)
            with st.expander("Lihat data mentah penjelasan LIME Model B"):
                st.dataframe(lime_df_B)
    except Exception as e:
        st.error(f"Gagal memuat atau memvisualisasikan penjelasan LIME: {e}")

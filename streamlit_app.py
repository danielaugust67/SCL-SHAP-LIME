# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
import shap
import lime
import lime.lime_tabular
import time

# Mengabaikan warning untuk tampilan yang lebih bersih di dashboard
warnings.filterwarnings('ignore')

# Konfigurasi halaman Streamlit
st.set_page_config(layout="wide", page_title="Analisis Model Prediksi Kelulusan Mahasiswa")

# --- Bagian Cache: Fungsi untuk memuat dan melatih model ---
# Menggunakan cache agar proses yang berat tidak diulang setiap kali ada interaksi

@st.cache_data
def load_and_process_data():
    """
    Memuat data, melakukan pra-pemrosesan, dan mengembalikan set data yang siap pakai.
    Fungsi ini di-cache agar tidak perlu memuat ulang data setiap saat.
    """
    df = pd.read_csv("Final_Cleaned_Student_Dataset.csv")
    
    target_column = "Target"
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    class_names = le.classes_
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X = pd.get_dummies(X)
    X_columns = X.columns.tolist()
    
    smote = SMOTETomek(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)
    
    return X_bal, y_bal, X_scaled, X_columns, class_names

@st.cache_data
def train_scl_and_get_embeddings(_X_scaled, _y_bal, best_temp=0.5, best_embed_dim=128):
    """
    Melatih model Supervised Contrastive Learning (SCL) dan menghasilkan embeddings.
    Fungsi ini di-cache untuk menghindari pelatihan ulang SCL.
    """
    class ProjectionHead(nn.Module):
        def __init__(self, input_dim, output_dim=128):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        def forward(self, x):
            return F.normalize(self.fc(x), dim=1)

    class ContrastiveModel(nn.Module):
        def __init__(self, input_dim, embed_dim=128):
            super().__init__()
            self.encoder = nn.Linear(input_dim, embed_dim)
            self.projector = ProjectionHead(embed_dim, embed_dim)
        def forward(self, x):
            return self.projector(F.relu(self.encoder(x)))

    def supervised_contrastive_loss(features, labels, temperature=0.5):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - torch.eye(features.shape[0]).to(device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        return -mean_log_prob_pos.mean()

    device = torch.device("cpu") # Streamlit cloud biasanya CPU
    input_dim = _X_scaled.shape[1]
    scl_model = ContrastiveModel(input_dim, embed_dim=best_embed_dim).to(device)
    optimizer = torch.optim.Adam(scl_model.parameters(), lr=0.001)

    X_tensor = torch.tensor(_X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(_y_bal.values, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Progress bar untuk pelatihan SCL
    progress_bar = st.progress(0)
    status_text = st.empty()
    num_epochs = 50

    scl_model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            features = scl_model(batch_x)
            loss = supervised_contrastive_loss(features, batch_y, temperature=best_temp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        progress = (epoch + 1) / num_epochs
        progress_bar.progress(progress)
        status_text.text(f"Melatih Model SCL... Epoch {epoch+1}/{num_epochs}")

    scl_model.eval()
    with torch.no_grad():
        embeddings = scl_model(X_tensor.to(device)).cpu().numpy()
    
    status_text.text("Pelatihan SCL selesai!")
    progress_bar.empty()

    return embeddings

@st.cache_resource
def train_classifier(X_train, y_train):
    """Melatih dan mengembalikan model VotingClassifier."""
    rf = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')
    model.fit(X_train, y_train)
    return model

# --- Judul Utama Dashboard ---
st.title("📊 Dashboard Analisis & Interpretasi Model Prediksi Kelulusan Mahasiswa")
st.write("""
Aplikasi ini membandingkan dua pendekatan model machine learning untuk memprediksi status kelulusan mahasiswa:
1.  **Model A (Baseline)**: Menggunakan fitur asli mahasiswa.
2.  **Model B (Proposed)**: Menggunakan *embeddings* yang dihasilkan oleh *Supervised Contrastive Learning* (SCL).
Jelajahi berbagai tab di sidebar untuk melihat perbandingan performa, interpretasi global (SHAP), interpretasi lokal (LIME), dan analisis makna dari *embeddings*.
""")

# --- Memuat Data dan Melatih Model (dengan Spinner) ---
with st.spinner("Langkah 1/3: Memuat dan memproses data... Ini mungkin memakan waktu beberapa saat."):
    X_bal, y_bal, X_scaled, X_columns, class_names = load_and_process_data()

with st.spinner("Langkah 2/3: Melatih model SCL untuk menghasilkan embeddings... Ini adalah proses yang paling lama pada pemuatan pertama."):
    embeddings = train_scl_and_get_embeddings(X_scaled, y_bal)
    embedding_feature_names = [f'Emb_{i}' for i in range(embeddings.shape[1])]

# Membagi data setelah semua diproses
X_train_orig, X_test_orig, y_train, y_test, emb_train, emb_test = train_test_split(
    X_scaled, y_bal, embeddings, test_size=0.25, stratify=y_bal, random_state=42
)
X_train_df = pd.DataFrame(X_train_orig, columns=X_columns)

with st.spinner("Langkah 3/3: Melatih model klasifikasi A dan B..."):
    model_A = train_classifier(X_train_orig, y_train)
    model_B = train_classifier(emb_train, y_train)

st.success("Semua data dan model berhasil dimuat! Silakan pilih analisis dari sidebar.")
st.markdown("---")


# --- Sidebar untuk Navigasi ---
st.sidebar.title("Navigasi Analisis")
analysis_choice = st.sidebar.radio(
    "Pilih Halaman:",
    ("Ringkasan & Performa Model", "Analisis Global (SHAP)", "Analisis Lokal (LIME)", "Analisis Semantik Embedding")
)


# --- Tampilan Berdasarkan Pilihan di Sidebar ---

if analysis_choice == "Ringkasan & Performa Model":
    st.header("Ringkasan & Performa Model")
    st.write("Perbandingan laporan klasifikasi antara Model A (tanpa SCL) dan Model B (dengan SCL) pada data tes.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model A (Baseline - Tanpa SCL)")
        y_pred_A = model_A.predict(X_test_orig)
        report_A = classification_report(y_test, y_pred_A, target_names=class_names, output_dict=True)
        st.dataframe(pd.DataFrame(report_A).transpose())

    with col2:
        st.subheader("Model B (Proposed - Dengan SCL)")
        y_pred_B = model_B.predict(emb_test)
        report_B = classification_report(y_test, y_pred_B, target_names=class_names, output_dict=True)
        st.dataframe(pd.DataFrame(report_B).transpose())

elif analysis_choice == "Analisis Global (SHAP)":
    st.header("Analisis Keterpenjelasan Global dengan SHAP")
    st.write("SHAP (SHapley Additive exPlanations) menunjukkan kontribusi rata-rata dari setiap fitur/embedding terhadap prediksi model secara keseluruhan. Plot di bawah ini adalah *bar plot* yang merangkum *mean absolute SHAP value*.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model A (Baseline - Fitur Asli)")
        with st.spinner("Menghitung SHAP values untuk Model A..."):
            explainer_A = shap.KernelExplainer(model_A.predict_proba, X_train_orig[:50])
            shap_values_A = explainer_A.shap_values(X_test_orig[:25])
            fig_A, ax_A = plt.subplots()
            shap.summary_plot(shap_values_A, X_test_orig[:25], plot_type='bar', feature_names=X_columns, show=False)
            st.pyplot(fig_A)

    with col2:
        st.subheader("Model B (Proposed - Fitur Embedding)")
        with st.spinner("Menghitung SHAP values untuk Model B..."):
            explainer_B = shap.KernelExplainer(model_B.predict_proba, emb_train[:50])
            shap_values_B = explainer_B.shap_values(emb_test[:25])
            fig_B, ax_B = plt.subplots()
            shap.summary_plot(shap_values_B, emb_test[:25], plot_type='bar', feature_names=embedding_feature_names, show=False)
            st.pyplot(fig_B)


elif analysis_choice == "Analisis Lokal (LIME)":
    st.header("Analisis Keterpenjelasan Lokal dengan LIME")
    st.write("LIME (Local Interpretable Model-agnostic Explanations) menjelaskan prediksi untuk satu sampel data (satu mahasiswa). Pilih indeks mahasiswa dari data tes untuk dianalisis.")

    instance_idx = st.slider("Pilih Indeks Mahasiswa dari Test Set:", 0, len(X_test_orig) - 1, 0)
    
    instance_A = X_test_orig[instance_idx]
    instance_B = emb_test[instance_idx]
    true_label = class_names[y_test.iloc[instance_idx]]
    pred_A = class_names[model_A.predict(instance_A.reshape(1, -1))[0]]
    pred_B = class_names[model_B.predict(instance_B.reshape(1, -1))[0]]

    st.markdown(f"**Menganalisis Mahasiswa ke-{instance_idx}**")
    st.write(f"- Label Asli: **{true_label}**")
    st.write(f"- Prediksi Model A: **{pred_A}**")
    st.write(f"- Prediksi Model B: **{pred_B}**")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Penjelasan LIME untuk Model A")
        with st.spinner("Membuat penjelasan LIME untuk Model A..."):
            explainer_lime_A = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_orig, mode='classification', feature_names=X_columns, class_names=class_names, verbose=False
            )
            explanation_A = explainer_lime_A.explain_instance(instance_A, model_A.predict_proba, num_features=10, top_labels=3)
            st.components.v1.html(explanation_A.as_html(), height=800)
    
    with col2:
        st.subheader("Penjelasan LIME untuk Model B")
        with st.spinner("Membuat penjelasan LIME untuk Model B..."):
             explainer_lime_B = lime.lime_tabular.LimeTabularExplainer(
                training_data=emb_train, mode='classification', feature_names=embedding_feature_names, class_names=class_names, verbose=False
            )
             explanation_B = explainer_lime_B.explain_instance(instance_B, model_B.predict_proba, num_features=10, top_labels=3)
             st.components.v1.html(explanation_B.as_html(), height=800)

elif analysis_choice == "Analisis Semantik Embedding":
    st.header("Analisis Semantik: Mencari Makna di Balik Embedding Penting")
    st.write("""
    Model B tidak menggunakan fitur asli secara langsung, melainkan *embedding* hasil SCL.
    Untuk memahami apa yang direpresentasikan oleh sebuah embedding, kita bisa mencari fitur-fitur asli mana yang paling kuat korelasinya dengan embedding tersebut.
    Gunakan menu di bawah untuk memilih embedding yang ingin dianalisis (misalnya, yang paling penting menurut plot SHAP).
    """)

    # Membuat DataFrame dari data training untuk korelasi
    emb_train_df = pd.DataFrame(emb_train, columns=embedding_feature_names)

    # Widget untuk memilih embedding
    important_embedding_name = st.selectbox(
        "Pilih Embedding untuk dianalisis:",
        options=embedding_feature_names,
        index=86 # Default ke Emb_86 seperti di skrip asli
    )

    if important_embedding_name:
        # Hitung korelasi antara embedding yang dipilih dengan semua fitur asli
        correlation_series = X_train_df.corrwith(emb_train_df[important_embedding_name])
        
        # Urutkan berdasarkan nilai korelasi absolut (paling kuat)
        top_correlated_features = correlation_series.abs().sort_values(ascending=False).head(10)
        
        st.subheader(f"Top 10 Fitur Asli yang Paling Berkolerasi dengan {important_embedding_name}")
        
        # Tampilkan dalam bentuk tabel dan plot
        col1, col2 = st.columns([1, 2])

        with col1:
             st.write("Tabel Korelasi:")
             st.dataframe(top_correlated_features.rename("Korelasi Absolut"))

        with col2:
            st.write("Visualisasi Korelasi:")
            # Ambil nilai korelasi asli (bukan absolut) untuk plot
            plot_data = correlation_series.loc[top_correlated_features.index].sort_values()
            st.bar_chart(plot_data)
            st.caption("Batang positif berarti korelasi positif (searah), batang negatif berarti korelasi negatif (berlawanan arah).")

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Mengabaikan warning
warnings.filterwarnings('ignore')

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide", page_title="Dashboard Analisis Model SCL")

# --- Styling CSS Kustom (Opsional) ---
st.markdown("""
<style>
    /* Mengubah font */
    html, body, [class*="css"]  {
       font-family: 'Source Sans Pro', sans-serif;
    }
    /* Style untuk subheader */
    h2 {
        color: #2E86C1;
        border-bottom: 2px solid #D6EAF8;
        padding-bottom: 10px;
    }
    /* Style untuk header di kolom utama */
    h3 {
        color: #17202A;
    }
</style>
""", unsafe_allow_html=True)


# --- Fungsi-fungsi Cache untuk Data dan Model (Tidak Berubah) ---
@st.cache_data
def load_and_process_data():
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
    class ProjectionHead(nn.Module):
        def __init__(self, input_dim, output_dim=128): super().__init__(); self.fc = nn.Linear(input_dim, output_dim)
        def forward(self, x): return F.normalize(self.fc(x), dim=1)
    class ContrastiveModel(nn.Module):
        def __init__(self, input_dim, embed_dim=128): super().__init__(); self.encoder = nn.Linear(input_dim, embed_dim); self.projector = ProjectionHead(embed_dim, embed_dim)
        def forward(self, x): return self.projector(F.relu(self.encoder(x)))
    def supervised_contrastive_loss(features, labels, temperature=0.5):
        device=features.device; labels=labels.contiguous().view(-1,1); mask=torch.eq(labels,labels.T).float().to(device); anchor_dot_contrast=torch.div(torch.matmul(features,features.T),temperature); logits_max,_=torch.max(anchor_dot_contrast,dim=1,keepdim=True); logits=anchor_dot_contrast-logits_max.detach(); exp_logits=torch.exp(logits)*(1-torch.eye(features.shape[0]).to(device)); log_prob=logits-torch.log(exp_logits.sum(1,keepdim=True)+1e-10); mean_log_prob_pos=(mask*log_prob).sum(1)/(mask.sum(1)+1e-10); return -mean_log_prob_pos.mean()
    device=torch.device("cpu"); input_dim=_X_scaled.shape[1]; scl_model=ContrastiveModel(input_dim,embed_dim=best_embed_dim).to(device); optimizer=torch.optim.Adam(scl_model.parameters(),lr=0.001); X_tensor=torch.tensor(_X_scaled,dtype=torch.float32); y_tensor=torch.tensor(_y_bal.values,dtype=torch.long); dataset=TensorDataset(X_tensor,y_tensor); loader=DataLoader(dataset,batch_size=128,shuffle=True); num_epochs=50; scl_model.train()
    for epoch in range(num_epochs):
        for batch_x,batch_y in loader: batch_x,batch_y=batch_x.to(device),batch_y.to(device); features=scl_model(batch_x); loss=supervised_contrastive_loss(features,batch_y,temperature=best_temp); optimizer.zero_grad(); loss.backward(); optimizer.step()
    scl_model.eval()
    with torch.no_grad(): embeddings=scl_model(X_tensor.to(device)).cpu().numpy()
    return embeddings

@st.cache_resource
def train_classifier(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')
    model.fit(X_train, y_train)
    return model

# --- Memuat Data dan Melatih Model ---
with st.spinner("Mempersiapkan data dan melatih model... Proses ini hanya berjalan sekali."):
    X_bal, y_bal, X_scaled, X_columns, class_names = load_and_process_data()
    embeddings = train_scl_and_get_embeddings(X_scaled, y_bal)
    embedding_feature_names = [f'Emb_{i}' for i in range(embeddings.shape[1])]
    
    X_train_orig, X_test_orig, y_train, y_test, emb_train, emb_test = train_test_split(
        X_scaled, y_bal, embeddings, test_size=0.25, stratify=y_bal, random_state=42
    )
    X_train_df_orig = pd.DataFrame(X_train_orig, columns=X_columns)
    emb_train_df = pd.DataFrame(emb_train, columns=embedding_feature_names)
    
    model_A = train_classifier(X_train_orig, y_train)
    model_B = train_classifier(emb_train, y_train)


# --- Sidebar: Kamus Embedding ---
st.sidebar.title("📖 Kamus Embedding")
st.sidebar.markdown("Gunakan panel ini untuk memahami makna dari setiap *embedding* yang dihasilkan oleh Model B.")
selected_embedding = st.sidebar.selectbox(
    "Pilih Embedding untuk dianalisis:",
    options=embedding_feature_names,
    index=86, # Default
    help="Pilih embedding yang ingin Anda ketahui maknanya."
)
if selected_embedding:
    correlation_series = X_train_df_orig.corrwith(emb_train_df[selected_embedding])
    top_correlated_features = correlation_series.abs().sort_values(ascending=False).head(10)
    
    st.sidebar.markdown(f"**Fitur Teratas yang Berkolerasi dengan `{selected_embedding}`**")
    correlation_df = correlation_series.loc[top_correlated_features.index].rename("Nilai Korelasi").to_frame()
    st.sidebar.dataframe(correlation_df, use_container_width=True)
    st.sidebar.caption("Nilai positif berarti korelasi searah, nilai negatif berarti berlawanan arah.")


# --- Judul Utama Dashboard ---
st.title("🔬 Dashboard Analisis & Interpretasi Model Prediksi Kelulusan")
st.markdown("Sebuah dasbor interaktif untuk membedah dan membandingkan performa model machine learning dengan dan tanpa *Supervised Contrastive Learning* (SCL).")
st.divider()

# --- Navigasi Utama Menggunakan Tab ---
tab1, tab2, tab3 = st.tabs(["📊 Ringkasan & Performa", "🌍 Analisis Global (SHAP)", "🔬 Analisis Lokal (LIME)"])


# --- Isi Tab 1: Performa Model ---
with tab1:
    st.header("Perbandingan Performa Model")
    st.write("Laporan klasifikasi lengkap untuk Model A (Baseline) dan Model B (SCL) pada data tes. Metrik seperti *precision*, *recall*, dan *f1-score* menunjukkan kemampuan model pada setiap kelas.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model A (Baseline - Tanpa SCL)")
        y_pred_A = model_A.predict(X_test_orig)
        report_A = classification_report(y_test, y_pred_A, target_names=class_names, output_dict=True)
        st.dataframe(pd.DataFrame(report_A).transpose(), use_container_width=True)

    with col2:
        st.subheader("Model B (Proposed - Dengan SCL)")
        y_pred_B = model_B.predict(emb_test)
        report_B = classification_report(y_test, y_pred_B, target_names=class_names, output_dict=True)
        st.dataframe(pd.DataFrame(report_B).transpose(), use_container_width=True)
    
    st.info("""
    **Kesimpulan Awal:** Perhatikan perbedaan skor F1 (rata-rata terbobot) antara kedua model. Model dengan SCL (Model B) diharapkan menunjukkan peningkatan karena kemampuannya mempelajari representasi data yang lebih baik.
    """)

# --- Isi Tab 2: Analisis SHAP ---
with tab2:
    st.header("Analisis Keterpenjelasan Global (SHAP)")
    st.markdown("SHAP menunjukkan fitur mana yang paling berdampak pada prediksi model secara keseluruhan. Plot ini merangkum dampak rata-rata dari setiap fitur untuk setiap kelas target.")
    st.info("💡 Gunakan **Kamus Embedding** di sidebar untuk mencari tahu arti dari embedding yang paling berpengaruh pada plot Model B!")

    # SHAP Model A
    st.subheader("Model A (Baseline - Fitur Asli)")
    with st.spinner("Menghitung SHAP values untuk Model A..."):
        explainer_A = shap.KernelExplainer(model_A.predict_proba, X_train_orig[:50])
        shap_values_A = explainer_A.shap_values(X_test_orig[:25])
        fig_A, _ = plt.subplots(figsize=(10, 8))
        # Menggunakan nama kelas asli
        shap.summary_plot(shap_values_A, X_test_orig[:25], plot_type='bar', feature_names=X_columns, class_names=class_names, show=False)
        plt.title("Kontribusi Fitur Global - Model A")
        plt.tight_layout()
        st.pyplot(fig_A, use_container_width=True)
    
    st.divider()

    # SHAP Model B
    st.subheader("Model B (Proposed - Embedding SCL)")
    with st.spinner("Menghitung SHAP values untuk Model B..."):
        explainer_B = shap.KernelExplainer(model_B.predict_proba, emb_train[:50])
        shap_values_B = explainer_B.shap_values(emb_test[:25])
        fig_B, _ = plt.subplots(figsize=(10, 8))
        # Menggunakan nama kelas asli
        shap.summary_plot(shap_values_B, emb_test[:25], plot_type='bar', feature_names=embedding_feature_names, class_names=class_names, show=False)
        plt.title("Kontribusi Embedding Global - Model B")
        plt.tight_layout()
        st.pyplot(fig_B, use_container_width=True)

# --- Isi Tab 3: Analisis LIME ---
with tab3:
    st.header("Analisis Keterpenjelasan Lokal (LIME)")
    st.markdown("LIME menjelaskan mengapa model membuat prediksi tertentu untuk **satu mahasiswa spesifik**.")
    st.info("💡 Gunakan **Kamus Embedding** di sidebar untuk mengurai embedding yang muncul di penjelasan LIME untuk Model B.")
    
    instance_idx = st.slider("Pilih Indeks Mahasiswa dari Test Set:", 0, len(X_test_orig) - 1, 0,
                             help="Geser untuk memilih mahasiswa yang berbeda dan lihat bagaimana penjelasan model berubah.")
    
    instance_A = X_test_orig[instance_idx]
    instance_B = emb_test[instance_idx]
    true_label = class_names[y_test.iloc[instance_idx]]
    pred_A = class_names[model_A.predict(instance_A.reshape(1, -1))[0]]
    pred_B = class_names[model_B.predict(instance_B.reshape(1, -1))[0]]

    st.markdown(f"#### Menganalisis Mahasiswa ke-{instance_idx}")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Label Asli", true_label)
    info_col2.metric("Prediksi Model A", pred_A)
    info_col3.metric("Prediksi Model B", pred_B)
    
    # LIME Model A
    st.subheader("Penjelasan LIME untuk Model A")
    with st.spinner("Membuat penjelasan LIME untuk Model A..."):
        explainer_lime_A = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_orig, mode='classification', feature_names=X_columns, class_names=class_names, verbose=False
        )
        explanation_A = explainer_lime_A.explain_instance(instance_A, model_A.predict_proba, num_features=10, top_labels=3)
        st.components.v1.html(explanation_A.as_html(), height=400)
        
    st.divider()
        
    # LIME Model B
    st.subheader("Penjelasan LIME untuk Model B")
    with st.spinner("Membuat penjelasan LIME untuk Model B..."):
         explainer_lime_B = lime.lime_tabular.LimeTabularExplainer(
            training_data=emb_train, mode='classification', feature_names=embedding_feature_names, class_names=class_names, verbose=False
        )
         explanation_B = explainer_lime_B.explain_instance(instance_B, model_B.predict_proba, num_features=10, top_labels=3)
         st.components.v1.html(explanation_B.as_html(), height=400)

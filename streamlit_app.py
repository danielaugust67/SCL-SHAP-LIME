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

# --- Styling CSS Kustom ---
st.markdown("""
<style>
    # ... (CSS styling Anda sebelumnya bisa diletakkan di sini) ...
    [data-testid="stSidebar"] {
        background-color: #111111;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #EAECEE;
    }
    [data-testid="stSidebar"] .st-caption {
        color: #AAB7B8;
    }
</style>
""", unsafe_allow_html=True)


# --- Fungsi-fungsi Cache untuk Data dan Model ---
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
def train_scl_and_get_embeddings(_X_scaled, _y_bal):
    # ... (Isi fungsi ini sama persis seperti sebelumnya) ...
    class ProjectionHead(nn.Module):
        def __init__(self, input_dim, output_dim=128): super().__init__(); self.fc = nn.Linear(input_dim, output_dim)
        def forward(self, x): return F.normalize(self.fc(x), dim=1)
    class ContrastiveModel(nn.Module):
        def __init__(self, input_dim, embed_dim=128): super().__init__(); self.encoder = nn.Linear(input_dim, embed_dim); self.projector = ProjectionHead(embed_dim, embed_dim)
        def forward(self, x): return self.projector(F.relu(self.encoder(x)))
    def supervised_contrastive_loss(features, labels, temperature=0.5):
        device=features.device; labels=labels.contiguous().view(-1,1); mask=torch.eq(labels,labels.T).float().to(device); anchor_dot_contrast=torch.div(torch.matmul(features,features.T),temperature); logits_max,_=torch.max(anchor_dot_contrast,dim=1,keepdim=True); logits=anchor_dot_contrast-logits_max.detach(); exp_logits=torch.exp(logits)*(1-torch.eye(features.shape[0]).to(device)); log_prob=logits-torch.log(exp_logits.sum(1,keepdim=True)+1e-10); mean_log_prob_pos=(mask*log_prob).sum(1)/(mask.sum(1)+1e-10); return -mean_log_prob_pos.mean()
    device=torch.device("cpu"); input_dim=_X_scaled.shape[1]; scl_model=ContrastiveModel(input_dim,embed_dim=128).to(device); optimizer=torch.optim.Adam(scl_model.parameters(),lr=0.001); X_tensor=torch.tensor(_X_scaled,dtype=torch.float32); y_tensor=torch.tensor(_y_bal.values,dtype=torch.long); dataset=TensorDataset(X_tensor,y_tensor); loader=DataLoader(dataset,batch_size=128,shuffle=True); num_epochs=50; scl_model.train()
    for epoch in range(num_epochs):
        for batch_x,batch_y in loader: batch_x,batch_y=batch_x.to(device),batch_y.to(device); features=scl_model(batch_x); loss=supervised_contrastive_loss(features,batch_y,temperature=0.5); optimizer.zero_grad(); loss.backward(); optimizer.step()
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

# --- FUNGSI BARU UNTUK CACHING KALKULASI BERAT ---

@st.cache_data(max_entries=5, show_spinner=False)
def get_shap_values(_model, _X_train_bg, _X_test_explain):
    """Menghitung dan menyimpan cache SHAP values."""
    explainer = shap.KernelExplainer(_model.predict_proba, _X_train_bg)
    shap_values = explainer.shap_values(_X_test_explain)
    return shap_values

@st.cache_data(max_entries=20, show_spinner=False)
def get_lime_explanation(_model, training_data, feature_names, class_names, instance_to_explain):
    """Membuat dan menyimpan cache LIME explanation."""
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data, 
        mode='classification', 
        feature_names=feature_names, 
        class_names=class_names, 
        verbose=False
    )
    explanation = explainer_lime.explain_instance(
        instance_to_explain, 
        _model.predict_proba, 
        num_features=10, 
        top_labels=3
    )
    return explanation

# --- Memuat Data dan Melatih Model ---
with st.spinner("Mempersiapkan data dan melatih model... Proses ini hanya berjalan sekali."):
    # ... (Isi bagian ini sama persis seperti sebelumnya) ...
    X_bal, y_bal, X_scaled, X_columns, class_names = load_and_process_data()
    embeddings = train_scl_and_get_embeddings(X_scaled, y_bal)
    embedding_feature_names = [f'Emb_{i}' for i in range(embeddings.shape[1])]
    X_train_orig, X_test_orig, y_train, y_test, emb_train, emb_test = train_test_split(
        X_scaled, y_bal, embeddings, test_size=0.25, stratify=y_bal, random_state=42)
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
    index=86,
    help="Pilih embedding yang ingin Anda ketahui maknanya."
)
if selected_embedding:
    correlation_series = X_train_df_orig.corrwith(emb_train_df[selected_embedding])
    top_correlated_features = correlation_series.abs().sort_values(ascending=False).head(10)
    correlation_df = correlation_series.loc[top_correlated_features.index].rename("Nilai Korelasi").to_frame()
    st.sidebar.dataframe(correlation_df, use_container_width=True)
    st.sidebar.caption("Nilai positif berarti korelasi searah, nilai negatif berarti berlawanan arah.")


# --- Judul Utama dan Navigasi ---
st.title("🔬 Dashboard Analisis & Interpretasi Model Prediksi Kelulusan")
st.divider()
tab1, tab2, tab3 = st.tabs(["📊 Ringkasan & Performa", "🌍 Analisis Global (SHAP)", "🔬 Analisis Lokal (LIME)"])


# --- Isi Tab 1: Performa Model ---
with tab1:
    # ... (Isi tab ini sama persis seperti sebelumnya) ...
    st.header("Perbandingan Performa Model")
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


# --- Isi Tab 2: Analisis SHAP (Menggunakan Fungsi Cache) ---
with tab2:
    st.header("Analisis Keterpenjelasan Global (SHAP)")
    st.info("💡 Gunakan **Kamus Embedding** di sidebar untuk mencari tahu arti dari embedding yang paling berpengaruh pada plot Model B!")

    with st.spinner("Menyiapkan plot SHAP... (kalkulasi berat hanya berjalan sekali)"):
        # SHAP Model A
        st.subheader("Model A (Baseline - Fitur Asli)")
        shap_values_A = get_shap_values(model_A, X_train_orig[:50], X_test_orig[:25]) # Panggil fungsi cache
        fig_A, _ = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values_A, X_test_orig[:25], plot_type='bar', feature_names=X_columns, class_names=class_names, show=False)
        plt.tight_layout()
        st.pyplot(fig_A, use_container_width=True)
        
        st.divider()

        # SHAP Model B
        st.subheader("Model B (Proposed - Embedding SCL)")
        shap_values_B = get_shap_values(model_B, emb_train[:50], emb_test[:25]) # Panggil fungsi cache
        fig_B, _ = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values_B, emb_test[:25], plot_type='bar', feature_names=embedding_feature_names, class_names=class_names, show=False)
        plt.tight_layout()
        st.pyplot(fig_B, use_container_width=True)

# --- Isi Tab 3: Analisis LIME (Menggunakan Fungsi Cache) ---
with tab3:
    st.header("Analisis Keterpenjelasan Lokal (LIME)")
    st.info("💡 Gunakan **Kamus Embedding** di sidebar untuk mengurai embedding yang muncul di penjelasan LIME untuk Model B.")
    
    instance_idx = st.slider("Pilih Indeks Mahasiswa dari Test Set:", 0, len(X_test_orig) - 1, 0)
    
    # Ambil instance data berdasarkan slider
    instance_A = X_test_orig[instance_idx]
    instance_B = emb_test[instance_idx]
    
    # ... (Info metric sama seperti sebelumnya) ...
    true_label = class_names[y_test.iloc[instance_idx]]
    pred_A = class_names[model_A.predict(instance_A.reshape(1, -1))[0]]
    pred_B = class_names[model_B.predict(instance_B.reshape(1, -1))[0]]
    st.markdown(f"#### Menganalisis Mahasiswa ke-{instance_idx}")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Label Asli", true_label)
    info_col2.metric("Prediksi Model A", pred_A)
    info_col3.metric("Prediksi Model B", pred_B)
    
    with st.spinner("Menyiapkan penjelasan LIME... (kalkulasi hanya berjalan jika slider digerakkan)"):
        # LIME Model A
        st.subheader("Penjelasan LIME untuk Model A")
        explanation_A = get_lime_explanation(model_A, X_train_orig, X_columns, class_names, instance_A) # Panggil fungsi cache
        st.components.v1.html(explanation_A.as_html(), height=400)
            
        st.divider()
            
        # LIME Model B
        st.subheader("Penjelasan LIME untuk Model B")
        explanation_B = get_lime_explanation(model_B, emb_train, embedding_feature_names, class_names, instance_B) # Panggil fungsi cache
        st.components.v1.html(explanation_B.as_html(), height=400)

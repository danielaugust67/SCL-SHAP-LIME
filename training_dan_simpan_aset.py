# Nama file: training_dan_simpan_aset.py

# --- Bagian 0: Setup dan Import Library ---
import pandas as pd
import numpy as np
import warnings
import joblib
import os

# Sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# XAI Libraries
import shap
import lime
import lime.lime_tabular

# Mengabaikan warning dan setup
warnings.filterwarnings('ignore')
print("Memulai proses training dan penyimpanan aset...")

# --- Bagian 1: Persiapan Data ---
print("Langkah 1: Memuat dan Memproses Data...")
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
print("Data berhasil diproses.")

# --- Bagian 2: Definisi & Pelatihan SCL ---
print("Langkah 2: Melatih SCL...")
BEST_TEMP = 0.5
BEST_EMBED_DIM = 128

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_scaled.shape[1]
scl_model = ContrastiveModel(input_dim, embed_dim=BEST_EMBED_DIM).to(device)
optimizer = torch.optim.Adam(scl_model.parameters(), lr=0.001)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_bal.values, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

scl_model.train()
for epoch in range(50):
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        features = scl_model(batch_x)
        loss = supervised_contrastive_loss(features, batch_y, temperature=BEST_TEMP)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

scl_model.eval()
with torch.no_grad():
    embeddings = scl_model(X_tensor.to(device)).cpu().numpy()
print("SCL selesai dilatih.")

# --- Bagian 3: Pelatihan Model Klasifikasi ---
print("Langkah 3: Melatih model klasifikasi...")
X_train, X_test, y_train, y_test, emb_train, emb_test = train_test_split(
    X_scaled, y_bal, embeddings, test_size=0.25, stratify=y_bal, random_state=42
)

# Model A (Baseline)
rf_A = RandomForestClassifier(random_state=42)
xgb_A = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model_A = VotingClassifier(estimators=[('rf', rf_A), ('xgb', xgb_A)], voting='soft')
model_A.fit(X_train, y_train)
y_pred_A = model_A.predict(X_test)

# Model B (Dengan SCL)
rf_B = RandomForestClassifier(random_state=42)
xgb_B = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model_B = VotingClassifier(estimators=[('rf', rf_B), ('xgb', xgb_B)], voting='soft')
model_B.fit(emb_train, y_train)
y_pred_B = model_B.predict(emb_test)
print("Model selesai dilatih.")

# --- Bagian 4 & 5: Kalkulasi SHAP dan LIME ---
print("Langkah 4 & 5: Menghitung penjelasan SHAP dan LIME (ini mungkin lama)...")
# SHAP
explainer_A = shap.KernelExplainer(model_A.predict_proba, X_train[:100])
shap_values_A = explainer_A.shap_values(X_test[:50])
embedding_feature_names = [f'Emb_{i}' for i in range(emb_train.shape[1])]
explainer_B = shap.KernelExplainer(model_B.predict_proba, emb_train[:100])
shap_values_B = explainer_B.shap_values(emb_test[:50])

# LIME
instance_idx = 0
instance_A = X_test[instance_idx]
instance_B = emb_test[instance_idx]
explainer_lime_A = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification', feature_names=X_columns, class_names=class_names)
explanation_A = explainer_lime_A.explain_instance(instance_A, model_A.predict_proba, num_features=10)
explainer_lime_B = lime.lime_tabular.LimeTabularExplainer(emb_train, mode='classification', feature_names=embedding_feature_names, class_names=class_names)
explanation_B = explainer_lime_B.explain_instance(instance_B, model_B.predict_proba, num_features=10)
print("Penjelasan XAI selesai dihitung.")

# --- Bagian 6: Analisis Semantik ---
print("Langkah 6: Melakukan analisis semantik...")
important_embedding_idx = 86
X_train_df = pd.DataFrame(X_train, columns=X_columns)
emb_train_df = pd.DataFrame(emb_train, columns=embedding_feature_names)
correlation_series = X_train_df.corrwith(emb_train_df[f'Emb_{important_embedding_idx}'])
print("Analisis semantik selesai.")

# --- BAGIAN TERAKHIR: SIMPAN SEMUA ASET ---
print("Langkah Terakhir: Menyimpan semua aset ke folder 'dashboard_assets'...")
output_dir = "dashboard_assets"
os.makedirs(output_dir, exist_ok=True)

# Simpan Laporan Klasifikasi
report_A_df = pd.DataFrame(classification_report(y_test, y_pred_A, target_names=class_names, output_dict=True)).transpose()
report_B_df = pd.DataFrame(classification_report(y_test, y_pred_B, target_names=class_names, output_dict=True)).transpose()
report_A_df.to_csv(os.path.join(output_dir, "classification_report_A.csv"))
report_B_df.to_csv(os.path.join(output_dir, "classification_report_B.csv"))

# Simpan Aset SHAP
joblib.dump(shap_values_A, os.path.join(output_dir, 'shap_values_A.pkl'))
pd.DataFrame(X_test[:50], columns=X_columns).to_csv(os.path.join(output_dir, 'shap_data_A.csv'), index=False)
joblib.dump(X_columns, os.path.join(output_dir, 'feature_names_A.pkl'))
joblib.dump(shap_values_B, os.path.join(output_dir, 'shap_values_B.pkl'))
pd.DataFrame(emb_test[:50], columns=embedding_feature_names).to_csv(os.path.join(output_dir, 'shap_data_B.csv'), index=False)
joblib.dump(embedding_feature_names, os.path.join(output_dir, 'feature_names_B.pkl'))

# Simpan Aset LIME
joblib.dump(explanation_A.as_list(), os.path.join(output_dir, 'lime_explanation_A.pkl'))
joblib.dump(explanation_B.as_list(), os.path.join(output_dir, 'lime_explanation_B.pkl'))

# Simpan Aset Semantik
correlation_series.to_csv(os.path.join(output_dir, 'embedding_semantic_correlation.csv'))

print("-" * 50)
print("✅ Semua aset berhasil dibuat dan disimpan di folder 'dashboard_assets'.")
print("Anda sekarang siap untuk menjalankan aplikasi Streamlit.")
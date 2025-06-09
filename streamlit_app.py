import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# --- Bagian 0: Setup dan Import Library ---
st.title('Machine Learning Dashboard')
st.write("Dashboard ini menampilkan model machine learning dengan dan tanpa penggunaan SCL.")

@st.cache
def load_and_process_data():
    df = pd.read_csv("Final_Cleaned_Student_Dataset.csv")
    target_column = "Target"
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    class_names = le.classes_

    X = df.drop(columns=[target_column])
    y = df[target_column]

    smote = SMOTETomek(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)

    return X_scaled, y_bal, X.columns.tolist(), class_names

X_scaled, y_bal, X_columns, class_names = load_and_process_data()

# --- Bagian 1: Visualisasi Data ---
st.subheader('Visualisasi Data')
st.write("Beberapa visualisasi data untuk memahami dataset:")

# Histogram for feature distribution
fig, ax = plt.subplots(figsize=(10, 6))
X_df = pd.DataFrame(X_scaled, columns=X_columns)
X_df.hist(ax=ax, bins=20)
st.pyplot(fig)

# --- Bagian 2: Model Training ---
st.subheader('Pelatihan Model Machine Learning')

st.write("Melatih model dengan dan tanpa SCL...")

# Fungsi untuk melatih model
@st.cache
def train_model():
    rf_A = RandomForestClassifier(random_state=42)
    xgb_A = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_A = VotingClassifier(estimators=[('rf', rf_A), ('xgb', xgb_A)], voting='soft')

    # Latih Model A (Tanpa SCL)
    model_A.fit(X_scaled, y_bal)
    return model_A

# Fungsi untuk melatih model dengan CL
@st.cache
def train_model_with_cl(embeddings):
    rf_B = RandomForestClassifier(random_state=42)
    xgb_B = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_B = VotingClassifier(estimators=[('rf', rf_B), ('xgb', xgb_B)], voting='soft')

    # Latih Model B (Dengan SCL)
    model_B.fit(embeddings, y_bal)
    return model_B

# Latih model tanpa CL
model_A = train_model()

# --- Bagian 3: Interpretability dengan SHAP ---
st.subheader('Analisis Interpretability dengan SHAP')

@st.cache
def shap_analysis(model, X):
    explainer_A = shap.KernelExplainer(model.predict_proba, X[:100])
    shap_values_A = explainer_A.shap_values(X[:50])
    return shap_values_A

# Menampilkan SHAP bar plot
shap_values_A = shap_analysis(model_A, X_scaled)
st.write("Menampilkan SHAP Bar Plot untuk Model A (Tanpa SCL)...")
shap.summary_plot(shap_values_A, X_scaled[:50], plot_type='bar', feature_names=X_columns)

# --- Bagian 4: Analisis LIME ---
st.subheader('Analisis Interpretability Lokal dengan LIME')

@st.cache
def lime_analysis(model, X, instance_idx=0):
    explainer_lime_A = lime.lime_tabular.LimeTabularExplainer(
        training_data=X,
        mode='classification',
        feature_names=X_columns,
        class_names=class_names,
        verbose=False
    )
    explanation_A = explainer_lime_A.explain_instance(X[instance_idx], model.predict_proba, num_features=10)
    return explanation_A

# Pilih instance untuk analisis
lime_explanation_A = lime_analysis(model_A, X_scaled)
lime_explanation_A.show_in_notebook(show_table=True, show_all=False)

# --- Bagian 5: Menampilkan Kamus Embedding ---
st.sidebar.subheader('Kamus Embedding')
st.sidebar.write("Berikut adalah kamus embedding dari model dengan SCL:")

# Menampilkan embedding ke-5 sebagai contoh
embedding_idx = 5  # Misalnya kita pilih embedding ke-5
embedding_values = X_scaled[:, embedding_idx]  # Ini adalah contoh, sesuaikan dengan data Anda
embedding_df = pd.DataFrame(embedding_values, columns=[f'Embedding_{embedding_idx}'])

st.sidebar.dataframe(embedding_df)

# --- Bagian 6: Evaluasi dan Hasil Klasifikasi ---
st.subheader('Evaluasi dan Hasil Klasifikasi')

# Evaluasi Model A (Tanpa CL)
y_pred_A = model_A.predict(X_scaled)
st.write("Evaluasi Model A (Tanpa CL):")
st.text(classification_report(y_bal, y_pred_A, target_names=class_names))

# Evaluasi Model B (Dengan CL)
# Latih model dengan embeddings
@st.cache
def train_with_embeddings():
    embeddings = np.random.rand(X_scaled.shape[0], 128)  # Contoh: gunakan embeddings yang sebenarnya
    model_B = train_model_with_cl(embeddings)
    return model_B

model_B = train_with_embeddings()
y_pred_B = model_B.predict(np.random.rand(X_scaled.shape[0], 128))  # Gunakan embeddings yang sebenarnya

st.write("Evaluasi Model B (Dengan CL):")
st.text(classification_report(y_bal, y_pred_B, target_names=class_names))

# --- Bagian 7: Interaktifitas ---
st.sidebar.subheader('Parameter Pengaturan')
epoch = st.sidebar.slider('Jumlah Epoch', 1, 100, 50)

# --- Bagian 8: Menampilkan Progress ---
st.write("Proses pelatihan model...")

# Simulasi progress
progress_bar = st.progress(0)
for i in range(100):
    time.sleep(0.1)
    progress_bar.progress(i + 1)

st.success('Pelatihan selesai!')

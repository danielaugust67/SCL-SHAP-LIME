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
from sklearn.metrics import classification_report

# --- Bagian 0: Setup dan Import Library ---
st.title('Machine Learning Dashboard')
st.write("Dashboard ini menampilkan model machine learning dengan dan tanpa penggunaan SCL.")

# --- Bagian 1: Memuat dan Memproses Data ---
@st.cache
def load_and_process_data():
    # Simulasi pembacaan data
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

# --- Bagian 2: Menampilkan Sidebar dan Kamus Embedding ---
# Menyimpan embedding dalam session_state agar tidak dihitung ulang
if 'embedding_values' not in st.session_state:
    st.session_state.embedding_values = np.random.rand(100, 10)  # Contoh embedding

# Sidebar untuk update embedding
st.sidebar.subheader('Kamus Embedding')
embedding_idx = st.sidebar.slider('Pilih embedding index', 0, 9, 5)  # Pilih embedding index antara 0 dan 9
embedding_values = st.session_state.embedding_values[:, embedding_idx]

# Tampilkan nilai embedding yang dipilih di sidebar
st.sidebar.write(f"Embedding pada index {embedding_idx}:")
st.sidebar.dataframe(pd.DataFrame(embedding_values))

# --- Bagian 3: Model Tanpa SCL (Tanpa Contrastive Learning) ---
@st.cache
def train_model_without_cl(X_scaled, y_bal):
    rf_A = RandomForestClassifier(random_state=42)
    xgb_A = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_A = VotingClassifier(estimators=[('rf', rf_A), ('xgb', xgb_A)], voting='soft')
    model_A.fit(X_scaled, y_bal)
    return model_A

model_A = train_model_without_cl(load_and_process_data()[0], load_and_process_data()[1])

# Hasil Klasifikasi Model A (Tanpa CL)
@st.cache
def evaluate_model_without_cl(model, X_scaled, y_bal):
    y_pred_A = model.predict(X_scaled)
    return classification_report(y_bal, y_pred_A)

st.subheader("Evaluasi Model Tanpa CL")
st.text(evaluate_model_without_cl(model_A, load_and_process_data()[0], load_and_process_data()[1]))

# --- Bagian 4: Model Dengan SCL (Dengan Contrastive Learning) ---
@st.cache
def train_model_with_cl(embeddings, y_bal):
    rf_B = RandomForestClassifier(random_state=42)
    xgb_B = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_B = VotingClassifier(estimators=[('rf', rf_B), ('xgb', xgb_B)], voting='soft')
    model_B.fit(embeddings, y_bal)
    return model_B

# Menggunakan embeddings yang sudah dipilih untuk training
embeddings = st.session_state.embedding_values
model_B = train_model_with_cl(embeddings, load_and_process_data()[1])

# Hasil Klasifikasi Model B (Dengan CL)
@st.cache
def evaluate_model_with_cl(model, embeddings, y_bal):
    y_pred_B = model.predict(embeddings)
    return classification_report(y_bal, y_pred_B)

st.subheader("Evaluasi Model Dengan CL")
st.text(evaluate_model_with_cl(model_B, embeddings, load_and_process_data()[1]))

# --- Bagian 5: SHAP (Analisis Interpretability Global) ---
@st.cache
def shap_analysis(model, X):
    explainer_A = shap.KernelExplainer(model.predict_proba, X[:100])
    shap_values_A = explainer_A.shap_values(X[:50])
    return shap_values_A

# Menampilkan SHAP Bar Plot
shap_values_A = shap_analysis(model_A, load_and_process_data()[0])
st.subheader("SHAP Bar Plot untuk Model Tanpa CL")
shap.summary_plot(shap_values_A, load_and_process_data()[0][:50], plot_type='bar', feature_names=load_and_process_data()[2])

# --- Bagian 6: LIME (Analisis Interpretability Lokal) ---
@st.cache
def lime_analysis(model, X, instance_idx=0):
    explainer_lime_A = lime.lime_tabular.LimeTabularExplainer(
        training_data=X,
        mode='classification',
        feature_names=load_and_process_data()[2],
        class_names=load_and_process_data()[3],
        verbose=False
    )
    explanation_A = explainer_lime_A.explain_instance(X[instance_idx], model.predict_proba, num_features=10)
    return explanation_A

lime_explanation_A = lime_analysis(model_A, load_and_process_data()[0])
lime_explanation_A.show_in_notebook(show_table=True, show_all=False)

# --- Bagian 7: Menampilkan Progress Bar ---
st.write("Proses pelatihan model...")

# Simulasi progress bar
progress_bar = st.progress(0)
for i in range(100):
    time.sleep(0.1)
    progress_bar.progress(i + 1)

st.success('Pelatihan selesai!')


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Predicción de Ventilación Mecánica (12-72 horas)", layout="centered")

st.title("Predicción de Ventilación Mecánica (12-72 horas)")
st.write("Esta aplicación entrena el modelo directamente en la nube usando el dataset incluido en el repositorio.")

# Cargar el dataset desde el repositorio
@st.cache_data
def cargar_datos():
    return pd.read_csv("Mech_Vent_Dataset.csv")

df = cargar_datos()

# Variables seleccionadas
variables = [
    "DiasABP_first", "GCS_first", "Glucose_first", "HR_first", "MAP_first", "RespRate_first",
    "SaO2_first", "SysABP_first", "Temp_first", "Creatinine_first", "FiO2_first", "HCO3_first",
    "HCT_first", "K_first", "Lactate_first", "Na_first", "PaCO2_first", "PaO2_first",
    "Platelets_first", "WBC_first", "Weight_first", "pH_first"
]

objetivo = "MechVentBinary"

# Separar datos
X = df[variables]
y = df[objetivo]

# Imputación y escalado
imputador = SimpleImputer(strategy="mean")
escalador = StandardScaler()
X_imputado = imputador.fit_transform(X)
X_escalado = escalador.fit_transform(X_imputado)

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X_escalado, y, test_size=0.2, random_state=42)
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Evaluación
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:,1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

st.subheader("Rendimiento del Modelo")
st.write(f"**Precisión:** {accuracy:.2f}")
st.write(f"**ROC AUC:** {roc_auc:.2f}")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend()
st.pyplot(plt)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No VM", "Sí VM"], yticklabels=["No VM", "Sí VM"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
st.pyplot(plt)

st.subheader("Predicción para un Paciente Nuevo")

# Formulario en español
valores_usuario = {}
for var in variables:
    valores_usuario[var] = st.number_input(f"{var}", value=0.0, step=0.1)

if st.button("Predecir"):
    entrada = pd.DataFrame([valores_usuario])
    entrada_imp = imputador.transform(entrada)
    entrada_esc = escalador.transform(entrada_imp)
    pred_prob = modelo.predict_proba(entrada_esc)[:,1][0]
    pred_clase = modelo.predict(entrada_esc)[0]
    st.write(f"**Probabilidad de necesitar ventilación mecánica (12-72h):** {pred_prob*100:.2f}%")

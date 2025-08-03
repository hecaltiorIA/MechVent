
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Predicción de Ventilación Mecánica (12-72 horas)")
st.write("Esta aplicación entrena el modelo directamente con el dataset subido.")

# Subir CSV
archivo = st.file_uploader("Sube el archivo CSV con los datos clínicos", type=["csv"])

# Variables seleccionadas
variables_seleccionadas = [
    "DiasABP_first", "GCS_first", "Glucose_first", "HR_first", "MAP_first",
    "RespRate_first", "SaO2_first", "SysABP_first", "Temp_first", "Creatinine_first",
    "FiO2_first", "HCO3_first", "HCT_first", "K_first", "Lactate_first",
    "Na_first", "PaCO2_first", "PaO2_first", "Platelets_first", "WBC_first",
    "Weight_first", "pH_first"
]

if archivo is not None:
    df = pd.read_csv(archivo)

    # Eliminar filas sin la variable objetivo
    df = df.dropna(subset=["MechVentBinary"])

    X = df[variables_seleccionadas]
    y = df["MechVentBinary"]

    # Imputar valores faltantes
    imputador = SimpleImputer(strategy="mean")
    X_imp = imputador.fit_transform(X)

    # Escalar
    escalador = StandardScaler()
    X_scaled = escalador.fit_transform(X_imp)

    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)

    # Métricas
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Precisión del modelo:** {acc*100:.2f}%")

    st.subheader("Ingresar datos para predicción")
    valores = []
    for var in variables_seleccionadas:
        valores.append(st.number_input(var, value=0.0, format="%.2f"))

    if st.button("Predecir"):
        datos_nuevo = np.array(valores).reshape(1, -1)
        datos_nuevo_imp = imputador.transform(datos_nuevo)
        datos_nuevo_scaled = escalador.transform(datos_nuevo_imp)
        prob = modelo.predict_proba(datos_nuevo_scaled)[0][1]
        pred = modelo.predict(datos_nuevo_scaled)[0]

        st.write(f"**Probabilidad:** {prob*100:.2f}%")
        if pred == 1:
            st.error("Alto riesgo: probablemente requerirá ventilación mecánica.")
        else:
            st.success("Bajo riesgo: probablemente NO requerirá ventilación mecánica.")

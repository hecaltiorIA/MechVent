
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo, imputador y escalador
modelo = joblib.load("modelo_ventilacion_vars_seleccionadas.pkl")
imputador = joblib.load("imputador_ventilacion_vars_seleccionadas.pkl")
escalador = joblib.load("escalador_ventilacion_vars_seleccionadas.pkl")

# Lista de variables en el mismo orden que se entrenó
variables = [
    "DiasABP_first", "GCS_first", "Glucose_first", "HR_first", "MAP_first",
    "RespRate_first", "SaO2_first", "SysABP_first", "Temp_first", "Creatinine_first",
    "FiO2_first", "HCO3_first", "HCT_first", "K_first", "Lactate_first", "Na_first",
    "PaCO2_first", "PaO2_first", "Platelets_first", "WBC_first", "Weight_first", "pH_first"
]

st.set_page_config(page_title="Predicción de Ventilación Mecánica", layout="centered")

st.title("🫁 Predicción de Necesidad de Ventilación Mecánica (12-72h)")
st.write("Introduce los valores iniciales del paciente para estimar el riesgo.")

# Crear formulario
valores_usuario = []
with st.form("formulario_vm"):
    for var in variables:
        valor = st.number_input(f"{var}", value=0.0, format="%.2f")
        valores_usuario.append(valor)
    submit = st.form_submit_button("Predecir")

# Cuando el usuario envía datos
if submit:
    datos_df = pd.DataFrame([valores_usuario], columns=variables)

    # Preprocesar
    datos_imputados = imputador.transform(datos_df)
    datos_escalados = escalador.transform(datos_imputados)

    # Predicción
    probabilidad = modelo.predict_proba(datos_escalados)[0][1]
    prediccion = modelo.predict(datos_escalados)[0]

    # Mostrar resultados
    st.subheader("Resultados")
    st.write(f"**Probabilidad de requerir ventilación mecánica:** {probabilidad*100:.2f}%")
    
    if prediccion == 1:
        st.error("⚠️ Alto riesgo de ventilación mecánica en las próximas 12-72 horas.")
    else:
        st.success("✅ Bajo riesgo de ventilación mecánica en las próximas 12-72 horas.")

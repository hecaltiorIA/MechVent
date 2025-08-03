
import streamlit as st
import joblib
import numpy as np

# Cargar modelo, imputador y escalador
modelo = joblib.load("modelo_ventilacion_vars_seleccionadas.pkl")
imputador = joblib.load("imputador_ventilacion_vars_seleccionadas.pkl")
escalador = joblib.load("escalador_ventilacion_vars_seleccionadas.pkl")

st.title("Predicción de Necesidad de Ventilación Mecánica (12-72 horas)")
st.write("Ingrese los valores iniciales del paciente para estimar el riesgo.")

# Variables en español
variables = [
    "Presión Arterial Diastólica Inicial (DiasABP_first)",
    "Glasgow Inicial (GCS_first)",
    "Glucosa Inicial (Glucose_first)",
    "Frecuencia Cardíaca Inicial (HR_first)",
    "Presión Arterial Media Inicial (MAP_first)",
    "Frecuencia Respiratoria Inicial (RespRate_first)",
    "Saturación de Oxígeno Inicial (SaO2_first)",
    "Presión Arterial Sistólica Inicial (SysABP_first)",
    "Temperatura Inicial (Temp_first)",
    "Creatinina Inicial (Creatinine_first)",
    "Fracción Inspirada de Oxígeno Inicial (FiO2_first)",
    "Bicarbonato Inicial (HCO3_first)",
    "Hematocrito Inicial (HCT_first)",
    "Potasio Inicial (K_first)",
    "Lactato Inicial (Lactate_first)",
    "Sodio Inicial (Na_first)",
    "Presión Parcial de CO2 Inicial (PaCO2_first)",
    "Presión Parcial de O2 Inicial (PaO2_first)",
    "Plaquetas Iniciales (Platelets_first)",
    "Leucocitos Iniciales (WBC_first)",
    "Peso Inicial (Weight_first)",
    "pH Inicial (pH_first)"
]

# Crear formulario
inputs = []
for var in variables:
    val = st.number_input(var, value=0.0, format="%.2f")
    inputs.append(val)

if st.button("Predecir"):
    X = np.array([inputs])
    X_imp = imputador.transform(X)
    X_scaled = escalador.transform(X_imp)
    prob = modelo.predict_proba(X_scaled)[0][1]
    pred = modelo.predict(X_scaled)[0]

    st.write(f"**Probabilidad de necesitar ventilación mecánica:** {prob*100:.2f}%")
    if pred == 1:
        st.error("Alto riesgo: El paciente probablemente requerirá ventilación mecánica en 12-72h.")
    else:
        st.success("Bajo riesgo: El paciente probablemente NO requerirá ventilación mecánica en 12-72h.")

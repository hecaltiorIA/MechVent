
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar modelo, imputador y escalador con pickle
with open("modelo_ventilacion_vars_seleccionadas.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("imputador_ventilacion_vars_seleccionadas.pkl", "rb") as f:
    imputador = pickle.load(f)

with open("escalador_ventilacion_vars_seleccionadas.pkl", "rb") as f:
    escalador = pickle.load(f)

# Título
st.title("Predicción de Ventilación Mecánica 12-72h")

# Variables en español
variables_dict = {
    'DiasABP_first': 'Presión arterial diastólica (mmHg)',
    'GCS_first': 'Escala de Glasgow',
    'Glucose_first': 'Glucosa (mg/dL)',
    'HR_first': 'Frecuencia cardíaca (lpm)',
    'MAP_first': 'Presión arterial media (mmHg)',
    'RespRate_first': 'Frecuencia respiratoria (rpm)',
    'SaO2_first': 'Saturación de oxígeno (%)',
    'SysABP_first': 'Presión arterial sistólica (mmHg)',
    'Temp_first': 'Temperatura (°C)',
    'Creatinine_first': 'Creatinina (mg/dL)',
    'FiO2_first': 'Fracción inspirada de oxígeno (%)',
    'HCO3_first': 'Bicarbonato (mmol/L)',
    'HCT_first': 'Hematocrito (%)',
    'K_first': 'Potasio (mmol/L)',
    'Lactate_first': 'Lactato (mmol/L)',
    'Na_first': 'Sodio (mmol/L)',
    'PaCO2_first': 'Presión parcial de CO₂ (mmHg)',
    'PaO2_first': 'Presión parcial de O₂ (mmHg)',
    'Platelets_first': 'Plaquetas (10³/µL)',
    'WBC_first': 'Leucocitos (10³/µL)',
    'Weight_first': 'Peso (kg)',
    'pH_first': 'pH arterial'
}

# Entradas de usuario
input_data = {}
for var, label in variables_dict.items():
    input_data[var] = st.number_input(label, value=0.0)

if st.button("Predecir"):
    df_input = pd.DataFrame([input_data])
    df_input_imputado = imputador.transform(df_input)
    df_input_escalado = escalador.transform(df_input_imputado)
    prediccion = modelo.predict(df_input_escalado)[0]
    prob = modelo.predict_proba(df_input_escalado)[0][1]
    
    st.write(f"**Predicción:** {'Requiere ventilación mecánica' if prediccion == 1 else 'No requiere ventilación mecánica'}")
    st.write(f"**Probabilidad:** {prob:.2%}")

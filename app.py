import pandas as pd
import streamlit as st

# Subir los archivos CSV
st.title('Modelo Predictivo de Demanda de Envíos')
st.write("Cargar los datasets para analizar la demanda de envíos.")

dataset_1_file = st.file_uploader("Subir el primer dataset", type=["csv"])
dataset_2_file = st.file_uploader("Subir el segundo dataset", type=["csv"])

# Mostrar las primeras 5 filas del dataset
print(dataset_1_file.head())
import pandas as pd
import streamlit as st

# Configuración de la aplicación
st.title('Modelo Predictivo de Demanda de Envíos')
st.write("Cargar los datasets para analizar la demanda de envíos.")

# Subir los archivos CSV
dataset_1_file = st.file_uploader("Subir el primer dataset", type=["csv"])
dataset_2_file = st.file_uploader("Subir el segundo dataset", type=["csv"])

# Procesar el primer archivo si se carga
if dataset_1_file is not None:
    try:
        dataset_1 = pd.read_csv(dataset_1_file, sep=",", on_bad_lines='skip')  # Manejar errores de líneas
        st.write("Primer dataset cargado con éxito:")
        st.write("Dimensiones del dataset:", dataset_1.shape)
        st.write("Primeras filas del dataset:")
        st.write(dataset_1.head())  # Mostrar las primeras 5 filas
    except Exception as e:
        st.error(f"Error al leer el primer dataset: {e}")

# Procesar el segundo archivo si se carga
if dataset_2_file is not None:
    try:
        dataset_2 = pd.read_csv(dataset_2_file, sep=",", on_bad_lines='skip')  # Manejar errores de líneas
        st.write("Segundo dataset cargado con éxito:")
        st.write("Dimensiones del dataset:", dataset_2.shape)
        st.write("Primeras filas del dataset:")
        st.write(dataset_2.head())  # Mostrar las primeras 5 filas
    except Exception as e:
        st.error(f"Error al leer el segundo dataset: {e}")

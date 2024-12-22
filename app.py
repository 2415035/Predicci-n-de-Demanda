import pandas as pd
import streamlit as st
import csv

# Configuración de la aplicación
st.title('Modelo Predictivo de Demanda de Envíos')
st.write("Cargar los datasets para analizar la demanda de envíos.")

# Función para detectar delimitador
def detectar_delimitador(file):
    try:
        sample = file.read(1024).decode()  # Leer una muestra del archivo
        file.seek(0)  # Volver al inicio del archivo
        sniffer = csv.Sniffer()
        delimitador = sniffer.sniff(sample).delimiter
        return delimitador
    except Exception as e:
        st.error(f"Error al detectar delimitador: {e}")
        return None

# Subir los archivos CSV
dataset_1_file = st.file_uploader("Subir el primer dataset", type=["csv"])
dataset_2_file = st.file_uploader("Subir el segundo dataset", type=["csv"])

# Procesar el primer archivo si se carga
if dataset_1_file is not None:
    delimitador_1 = detectar_delimitador(dataset_1_file)
    if delimitador_1:
        dataset_1 = pd.read_csv(dataset_1_file, sep=delimitador_1, quotechar='"', on_bad_lines='skip')
        st.write("Primer dataset cargado con éxito:")
        st.write(dataset_1.head())
    else:
        st.error("No se pudo detectar el delimitador del primer dataset.")

# Procesar el segundo archivo si se carga
if dataset_2_file is not None:
    delimitador_2 = detectar_delimitador(dataset_2_file)
    if delimitador_2:
        dataset_2 = pd.read_csv(dataset_2_file, sep=delimitador_2, quotechar='"', on_bad_lines='skip')
        st.write("Segundo dataset cargado con éxito:")
        st.write(dataset_2.head())
    else:
        st.error("No se pudo detectar el delimitador del segundo dataset.")

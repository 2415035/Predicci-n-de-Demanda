import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Subir los archivos CSV
st.title('Modelo Predictivo de Demanda de Envíos')
st.write("Cargar los datasets para analizar la demanda de envíos.")

dataset_1_file = st.file_uploader("Subir el primer dataset", type=["csv"])
dataset_2_file = st.file_uploader("Subir el segundo dataset", type=["csv"])

if dataset_1_file and dataset_2_file:
    # Cargar los datasets
    dataset_1 = pd.read_csv(dataset_1_file, delimiter=';')  # Cambiar por el delimitador correcto
    dataset_2 = pd.read_csv(dataset_2_file, delimiter=';')  # Cambiar por el delimitador correcto

    # Verificar las columnas de los datasets
    st.write("Columnas del primer dataset:", dataset_1.columns)
    st.write("Columnas del segundo dataset:", dataset_2.columns)

    print(dataset_1.head())  # Para imprimir las primeras 5 filas del primer dataset
    print(dataset_2.head())  # Para imprimir las primeras 5 filas del segundo dataset

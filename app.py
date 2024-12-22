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
    dataset_1 = pd.read_csv(dataset_1_file)
    dataset_2 = pd.read_csv(dataset_2_file)

    # Preprocesamiento para el primer dataset (convertir columnas categóricas)
    dataset_1 = pd.get_dummies(dataset_1, drop_first=True)

    # Preprocesamiento para el segundo dataset (convertir columnas categóricas)
    dataset_2 = pd.get_dummies(dataset_2, drop_first=True)

    # Preparación del primer dataset (definir X e y)
    columnas_a_eliminar_1 = ['Order ID', 'Date', 'Status', 'Fulfilment', 'Sales Channel', 
                             'ship-service-level', 'Category', 'Size', 'Courier Status', 
                             'currency', 'ship-city', 'ship-state', 'ship-postal-code', 'ship-country', 
                             'B2B', 'fulfilled-by', 'New', 'PendingS']
    columnas_a_eliminar_1 = [col for col in columnas_a_eliminar_1 if col in dataset_1.columns]
    X1 = dataset_1.drop(columns=columnas_a_eliminar_1)
    y1 = dataset_1['Qty']  # Variable objetivo

    # Preparación del segundo dataset (definir X e y)
    columnas_a_eliminar_2 = ['OBJECTID', 'FEATURE_ID', 'NAME', 'ADDRESS', 'ADDRESS2']
    columnas_a_eliminar_2 = [col for col in columnas_a_eliminar_2 if col in dataset_2.columns]
    X2 = dataset_2.drop(columns=columnas_a_eliminar_2)
    y2 = dataset_2['Qty']  # Variable objetivo (asegurarse que esta columna exista)

    # Dividir los datasets en entrenamiento y prueba
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

    # Entrenar el modelo en el primer dataset
    model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
    model_1.fit(X1_train, y1_train)

    # Realizar predicciones para el primer dataset
    y1_pred = model_1.predict(X1_test)

    # Evaluar el rendimiento del primer modelo
    r2_1 = r2_score(y1_test, y1_pred)
    mae_1 = mean_absolute_error(y1_test, y1_pred)
    mse_1 = mean_squared_error(y1_test, y1_pred)

    # Entrenar el modelo en el segundo dataset
    model_2 = RandomForestRegressor(n_estimators=100, random_state=42)
    model_2.fit(X2_train, y2_train)

    # Realizar predicciones para el segundo dataset
    y2_pred = model_2.predict(X2_test)

    # Evaluar el rendimiento del segundo modelo
    r2_2 = r2_score(y2_test, y2_pred)
    mae_2 = mean_absolute_error(y2_test, y2_pred)
    mse_2 = mean_squared_error(y2_test, y2_pred)

    # Mostrar los resultados en Streamlit
    st.subheader("Resultados para el Dataset 1:")
    st.write(f"R²: {r2_1}")
    st.write(f"MAE: {mae_1}")
    st.write(f"MSE: {mse_1}")

    st.subheader("Resultados para el Dataset 2:")
    st.write(f"R²: {r2_2}")
    st.write(f"MAE: {mae_2}")
    st.write(f"MSE: {mse_2}")

    if r2_1 >= 0.8 and r2_2 >= 0.8:
        st.success("Ambos modelos alcanzaron un R² mayor al 80%. El modelo es confiable.")
    else:
        st.warning("Uno o ambos modelos no alcanzaron el rendimiento objetivo del 80%. Puede que sea necesario ajustar el modelo.")
else:
    st.write("Por favor, sube ambos archivos CSV para realizar la predicción.")

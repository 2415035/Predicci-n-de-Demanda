import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuración de la aplicación
st.title("Modelo Predictivo para Dos Datasets")
st.write("Sube los datasets y el modelo predecirá la cantidad de envíos en cada caso.")

# Cargar archivos CSV
dataset_1_file = st.file_uploader("Subir el primer dataset (con columna 'PCS')", type=["csv"])
dataset_2_file = st.file_uploader("Subir el segundo dataset (con columna 'Qty')", type=["csv"])

if dataset_1_file and dataset_2_file:
    # Cargar datasets
    dataset_1 = pd.read_csv(dataset_1_file)
    dataset_2 = pd.read_csv(dataset_2_file)

    # Mostrar vistas previas de los datos
    st.subheader("Vista previa del primer dataset:")
    st.write(dataset_1.head())

    st.subheader("Vista previa del segundo dataset:")
    st.write(dataset_2.head())

    # Preprocesamiento para el primer dataset
    st.subheader("Preprocesamiento del Primer Dataset")
    if 'PCS' in dataset_1.columns:
        # Manejar valores faltantes solo en columnas numéricas
        num_cols_1 = dataset_1.select_dtypes(include=['number']).columns
        dataset_1[num_cols_1] = dataset_1[num_cols_1].fillna(dataset_1[num_cols_1].mean())

        # Preparar X e y
        X1 = dataset_1.drop(columns=['PCS', 'DATE', 'GROSS AMT'], errors='ignore')
        y1 = dataset_1['PCS']
        X1 = pd.get_dummies(X1, drop_first=True)

        # Validar si quedan valores faltantes
        if X1.isnull().sum().sum() > 0 or y1.isnull().sum() > 0:
            st.error("El primer dataset aún contiene valores faltantes después del preprocesamiento.")
        else:
            # Dividir en conjuntos de entrenamiento y prueba
            X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

            # Entrenar modelo
            model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
            model_1.fit(X1_train, y1_train)

            # Realizar predicciones y evaluar
            y1_pred = model_1.predict(X1_test)
            r2_1 = r2_score(y1_test, y1_pred)
            mae_1 = mean_absolute_error(y1_test, y1_pred)
            mse_1 = mean_squared_error(y1_test, y1_pred)

            st.subheader("Resultados del Primer Dataset:")
            st.write(f"R²: {r2_1}")
            st.write(f"MAE: {mae_1}")
            st.write(f"MSE: {mse_1}")
    else:
        st.error("La columna 'PCS' no está presente en el primer dataset.")

    # Preprocesamiento para el segundo dataset
    st.subheader("Preprocesamiento del Segundo Dataset")
    if 'Qty' in dataset_2.columns:
        # Manejar valores faltantes solo en columnas numéricas
        num_cols_2 = dataset_2.select_dtypes(include=['number']).columns
        dataset_2[num_cols_2] = dataset_2[num_cols_2].fillna(dataset_2[num_cols_2].mean())

        # Preparar X e y
        X2 = dataset_2.drop(columns=['Qty', 'NAME', 'ADDRESS', 'ADDRESS2'], errors='ignore')
        y2 = dataset_2['Qty']
        X2 = pd.get_dummies(X2, drop_first=True)

        # Validar si quedan valores faltantes
        if X2.isnull().sum().sum() > 0 or y2.isnull().sum() > 0:
            st.error("El segundo dataset aún contiene valores faltantes después del preprocesamiento.")
        else:
            # Dividir en conjuntos de entrenamiento y prueba
            X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

            # Entrenar modelo
            model_2 = RandomForestRegressor(n_estimators=100, random_state=42)
            model_2.fit(X2_train, y2_train)

            # Realizar predicciones y evaluar
            y2_pred = model_2.predict(X2_test)
            r2_2 = r2_score(y2_test, y2_pred)
            mae_2 = mean_absolute_error(y2_test, y2_pred)
            mse_2 = mean_squared_error(y2_test, y2_pred)

            st.subheader("Resultados del Segundo Dataset:")
            st.write(f"R²: {r2_2}")
            st.write(f"MAE: {mae_2}")
            st.write(f"MSE: {mse_2}")
    else:
        st.error("La columna 'Qty' no está presente en el segundo dataset.")
else:
    st.write("Por favor, sube ambos archivos CSV para continuar.")

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuración de la aplicación
st.title('Modelo Predictivo de Cantidad de Envíos')
st.write("Sube los datasets para analizar la cantidad de envíos.")

# Subir los archivos CSV
dataset_1_file = st.file_uploader("Sube el primer dataset", type=["csv"])
dataset_2_file = st.file_uploader("Sube el segundo dataset", type=["csv"])

if dataset_1_file and dataset_2_file:
    # Cargar los datasets
    dataset_1 = pd.read_csv(dataset_1_file)
    dataset_2 = pd.read_csv(dataset_2_file)

    # Mostrar las primeras filas de ambos datasets para verificar su estructura
    st.subheader("Vista previa del primer dataset:")
    st.write(dataset_1.head())

    st.subheader("Vista previa del segundo dataset:")
    st.write(dataset_2.head())

    # Preprocesamiento para el primer dataset
    if 'PCS' in dataset_1.columns:
        X1 = dataset_1.drop(columns=['PCS', 'DATE', 'GROSS AMT'], errors='ignore')  # Eliminar columnas no predictoras
        y1 = dataset_1['PCS']  # Variable objetivo
        X1 = pd.get_dummies(X1, drop_first=True)  # Convertir categóricas en variables dummy

        # Dividir los datos en entrenamiento y prueba
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

        # Entrenar el modelo para el primer dataset
        model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
        model_1.fit(X1_train, y1_train)

        # Realizar predicciones y evaluar el modelo
        y1_pred = model_1.predict(X1_test)
        r2_1 = r2_score(y1_test, y1_pred)
        mae_1 = mean_absolute_error(y1_test, y1_pred)
        mse_1 = mean_squared_error(y1_test, y1_pred)

        # Mostrar resultados para el primer dataset
        st.subheader("Resultados para el Primer Dataset (Predicción de PCS):")
        st.write(f"R²: {r2_1}")
        st.write(f"MAE: {mae_1}")
        st.write(f"MSE: {mse_1}")
    else:
        st.error("La columna 'PCS' no está presente en el primer dataset.")

    # Preprocesamiento para el segundo dataset
    if 'Qty' in dataset_2.columns:
        X2 = dataset_2.drop(columns=['Qty', 'NAME', 'ADDRESS', 'ADDRESS2'], errors='ignore')  # Eliminar columnas no predictoras
        y2 = dataset_2['Qty']  # Variable objetivo
        X2 = pd.get_dummies(X2, drop_first=True)  # Convertir categóricas en variables dummy

        # Dividir los datos en entrenamiento y prueba
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

        # Entrenar el modelo para el segundo dataset
        model_2 = RandomForestRegressor(n_estimators=100, random_state=42)
        model_2.fit(X2_train, y2_train)

        # Realizar predicciones y evaluar el modelo
        y2_pred = model_2.predict(X2_test)
        r2_2 = r2_score(y2_test, y2_pred)
        mae_2 = mean_absolute_error(y2_test, y2_pred)
        mse_2 = mean_squared_error(y2_test, y2_pred)

        # Mostrar resultados para el segundo dataset
        st.subheader("Resultados para el Segundo Dataset (Predicción de Qty):")
        st.write(f"R²: {r2_2}")
        st.write(f"MAE: {mae_2}")
        st.write(f"MSE: {mse_2}")
    else:
        st.error("La columna 'Qty' no está presente en el segundo dataset.")

    # Evaluar la consistencia de los modelos
    if 'PCS' in dataset_1.columns and 'Qty' in dataset_2.columns:
        st.subheader("Comparativa entre modelos:")
        if r2_1 >= 0.8 and r2_2 >= 0.8:
            st.success("Ambos modelos alcanzaron un R² mayor al 80%. Los modelos son consistentes.")
        else:
            st.warning("Uno o ambos modelos no alcanzaron el rendimiento objetivo del 80%. Puede que sea necesario ajustar los modelos.")
else:
    st.write("Por favor, sube ambos archivos CSV para realizar la predicción.")

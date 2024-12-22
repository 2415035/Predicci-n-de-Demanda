import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

file_path = "Amazon Sale Report.csv"

if os.path.exists(file_path):
    print(f"El archivo '{file_path}' existe.")
    print(f"Tamaño del archivo: {os.path.getsize(file_path)} bytes")
else:
    print(f"El archivo '{file_path}' no existe.")

# Cargar datasets
dataset_1 = pd.read_csv("Amazon Sale Report.csv")  # Primer dataset
dataset_2 = pd.read_csv("DHL_Facilities.csv")  # Segundo dataset

# --- Preprocesamiento del Dataset 1 ---
dataset_1['Date'] = pd.to_datetime(dataset_1['Date'])  # Convertir fechas
dataset_1['Month'] = dataset_1['Date'].dt.month  # Extraer mes
dataset_1['Day'] = dataset_1['Date'].dt.day  # Extraer día
X1 = dataset_1[['Month', 'Day', 'Qty']]  # Variables independientes
y1 = dataset_1['Amount']  # Variable dependiente

# --- Preprocesamiento del Dataset 2 ---
dataset_2.rename(columns={'CITY': 'City'}, inplace=True)  # Renombrar columnas
X2 = dataset_2[['LATITUDE', 'LONGITUDE', 'CENSUS_CODE']]  # Variables independientes
y2 = dataset_2['LOCATION_TY']  # Variable dependiente (suponiendo que es una cantidad)

# --- Entrenamiento del Modelo para Dataset 1 ---
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model1.fit(X1_train, y1_train)
y1_pred = model1.predict(X1_test)

# Métricas para Dataset 1
mae1 = mean_absolute_error(y1_test, y1_pred)
rmse1 = np.sqrt(mean_squared_error(y1_test, y1_pred))
r2_1 = r2_score(y1_test, y1_pred)

# --- Entrenamiento del Modelo para Dataset 2 ---
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2 = RandomForestRegressor(n_estimators=100, random_state=42)
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

# Métricas para Dataset 2
mae2 = mean_absolute_error(y2_test, y2_pred)
rmse2 = np.sqrt(mean_squared_error(y2_test, y2_pred))
r2_2 = r2_score(y2_test, y2_pred)

# --- Comparación de Resultados ---
print("Resultados para Dataset 1:")
print(f"MAE: {mae1}, RMSE: {rmse1}, R²: {r2_1}")

print("Resultados para Dataset 2:")
print(f"MAE: {mae2}, RMSE: {rmse2}, R²: {r2_2}")

# Visualización Comparativa
labels = ['Dataset 1', 'Dataset 2']
mae_values = [mae1, mae2]
rmse_values = [rmse1, rmse2]
r2_values = [r2_1, r2_2]

x = np.arange(len(labels))  # Etiquetas
width = 0.25  # Ancho de las barras

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, mae_values, width, label='MAE')
ax.bar(x, rmse_values, width, label='RMSE')
ax.bar(x + width, r2_values, width, label='R²')

ax.set_xlabel('Dataset')
ax.set_title('Comparación de Métricas entre Datasets')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

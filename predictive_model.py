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

# Leer el archivo CSV, eliminando comillas y manejando columnas con espacios
dataset = pd.read_csv("Amazon Sale Report.csv", quotechar='"')

# Eliminar espacios extra en los nombres de las columnas
dataset.columns = dataset.columns.str.strip()

# Eliminar la columna innecesaria "Unnamed: 22"
dataset = dataset.drop(columns=['Unnamed: 22'])

# Convertir la columna 'Date' al formato de fecha estándar (YYYY-MM-DD)
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m-%d-%y')

# Revisar los primeros 5 registros del dataframe
print(dataset.head())

# Guardar el archivo corregido, si es necesario
dataset.to_csv("Amazon_Sale_Report_Corregido.csv", index=False)
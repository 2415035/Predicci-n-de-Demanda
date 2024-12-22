import pandas as pd

# Cargar el archivo CSV
dataset_1 = pd.read_csv('Amazon Sale Report.csv')

# Mostrar las primeras 5 filas del dataset
print(dataset_1.head())
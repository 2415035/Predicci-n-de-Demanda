import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Estimación del Número de Pedidos en Días Festivos y otros valores Independientes")
st.write("Subir archivo CSV y obtén una estimación del número de pedidos en días festivos y demás datos.")

uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("Vista previa del archivo:")
    st.dataframe(data.head())

    st.write("Procesando datos...")
    data['Order_Date'] = pd.to_datetime(data['Order_Date'])
    data['Year'] = data['Order_Date'].dt.year
    data['Month'] = data['Order_Date'].dt.month
    data['Day'] = data['Order_Date'].dt.day
    data['Day_of_Week'] = data['Order_Date'].dt.dayofweek
    data['Is_Weekend'] = data['Day_of_Week'].isin([5, 6]).astype(int)

    data['Weatherconditions'].fillna("Desconocido", inplace=True)
    data['Road_traffic_density'].fillna("Desconocido", inplace=True)
    data['Festival'].fillna("No", inplace=True)
    data['City'].fillna("Desconocido", inplace=True)

    label_encoders = {}
    categorical_columns = ['Weatherconditions', 'Road_traffic_density', 'Festival', 'City']

    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    data['Order_Count'] = data.groupby(['Order_Date'])['Delivery_person_ID'].transform('count')

    features = ['Year', 'Month', 'Day', 'Day_of_Week', 'Is_Weekend',
                'Weatherconditions', 'Road_traffic_density', 'Festival', 'City']
    X = data[features]
    y = data['Order_Count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo de regresión
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Métricas de regresión
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    r2_percentage = r2 * 100

    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"R2 Score: {r2:.2f}")
    st.write(f"Porcentaje de acertividad del modelo: {r2_percentage:.2f}%")

    # Validación cruzada
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = (-cv_scores) ** 0.5

    st.write("Resultados de la validación cruzada (RMSE):")
    st.write(cv_rmse)
    st.write(f"Promedio RMSE (CV): {cv_rmse.mean():.2f}")

    st.write("Introduce datos para estimar los pedidos:")
    year = st.number_input("Año", min_value=2000, max_value=2100, value=2024)
    month = st.number_input("Mes", min_value=1, max_value=12, value=12)
    day = st.number_input("Día", min_value=1, max_value=31, value=25)
    day_of_week = st.number_input("Día de la semana (0=Lunes, 6=Domingo)", min_value=0, max_value=6, value=2)
    is_weekend = st.selectbox("¿Es fin de semana?", [0, 1])
    weather = st.selectbox("Condición climática", label_encoders['Weatherconditions'].classes_)
    traffic = st.selectbox("Densidad de tráfico", label_encoders['Road_traffic_density'].classes_)
    festival = st.selectbox("¿Es día festivo?", label_encoders['Festival'].classes_)
    city = st.selectbox("Ciudad", label_encoders['City'].classes_)

    example_data = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Day_of_Week': [day_of_week],
        'Is_Weekend': [is_weekend],
        'Weatherconditions': [label_encoders['Weatherconditions'].transform([weather])[0]],
        'Road_traffic_density': [label_encoders['Road_traffic_density'].transform([traffic])[0]],
        'Festival': [label_encoders['Festival'].transform([festival])[0]],
        'City': [label_encoders['City'].transform([city])[0]]
    })

    prediction = model.predict(example_data)
    st.write(f"Pedidos estimados: {prediction[0]:.2f}")

    st.write("Distribución de pedidos por mes:")
    fig, ax = plt.subplots()
    sns.barplot(x=data['Month'], y=data['Order_Count'], ax=ax)
    st.pyplot(fig)

    st.write("Gráfico de Predicciones Reales vs Predichas:")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # Línea de referencia
    ax.set_xlabel("Pedidos Reales")
    ax.set_ylabel("Pedidos Predichos")
    ax.set_title("Predicciones Reales vs Predichas")
    st.pyplot(fig)


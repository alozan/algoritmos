import pandas as pd
import numpy as np
from datetime import timedelta

# Modelos
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def preparar_features_futuras(df, col_fecha, dias_futuro):
    """Genera las fechas futuras y extrae sus características numéricas."""
    ultima_fecha = pd.to_datetime(df[col_fecha]).max()
    fechas_futuras = [ultima_fecha + timedelta(days=x) for x in range(1, dias_futuro + 1)]
    
    df_futuro = pd.DataFrame({col_fecha: fechas_futuras})
    for data in [df, df_futuro]:
        data['dia'] = pd.to_datetime(data[col_fecha]).dt.day
        data['mes'] = pd.to_datetime(data[col_fecha]).dt.month
        data['dia_semana'] = pd.to_datetime(data[col_fecha]).dt.dayofweek
        
    return df, df_futuro, fechas_futuras

def generar_pronosticos(ruta_csv, col_fecha, col_ventas, dias_futuro=30):
    df = pd.read_csv(ruta_csv)
    df[col_fecha] = pd.to_datetime(df[col_fecha])
    df = df.sort_values(col_fecha)
    
    # 1. Preparar datos para modelos Tabulares (LR, SVM, XGB)
    df, df_futuro, fechas_futuras = preparar_features_futuras(df, col_fecha, dias_futuro)
    
    features = ['dia', 'mes', 'dia_semana']
    X_train = df[features]
    y_train = df[col_ventas]
    X_futuro = df_futuro[features]

    # DataFrame para guardar todos los resultados
    resultados = pd.DataFrame({'Fecha': fechas_futuras})

    print("Entrenando Regresión Lineal...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    resultados['Pred_Regresion_Lineal'] = lr_model.predict(X_futuro)

    print("Entrenando Support Vector Machine (SVR)...")
    # SVM requiere datos escalados estrictamente
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_futuro_scaled = scaler_X.transform(X_futuro)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svm_model.fit(X_train_scaled, y_train_scaled)
    preds_svm_scaled = svm_model.predict(X_futuro_scaled)
    resultados['Pred_SVM'] = scaler_y.inverse_transform(preds_svm_scaled.reshape(-1, 1)).ravel()

    print("Entrenando XGBoost (Acelerado por GPU)...")
    # Utilizamos el backend CUDA para aprovechar la infraestructura NVIDIA
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, tree_method='hist', device='cuda')
    xgb_model.fit(X_train, y_train)
    resultados['Pred_XGBoost'] = xgb_model.predict(X_futuro)

    print("Entrenando ARIMA...")
    # ARIMA puro usa solo la serie histórica (p=5, d=1, q=0 es un ejemplo base)
    arima_model = ARIMA(df[col_ventas].values, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    resultados['Pred_ARIMA'] = arima_fit.forecast(steps=dias_futuro)

    print("Entrenando LSTM (Deep Learning)...")
    # LSTM requiere secuencias 3D: [muestras, pasos_tiempo, variables]
    X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_futuro_lstm = np.reshape(X_futuro_scaled, (X_futuro_scaled.shape[0], 1, X_futuro_scaled.shape[1]))
    
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, len(features))),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    # verbose=0 para no saturar la consola
    lstm_model.fit(X_train_lstm, y_train_scaled, epochs=50, batch_size=16, verbose=0) 
    
    preds_lstm_scaled = lstm_model.predict(X_futuro_lstm)
    resultados['Pred_LSTM'] = scaler_y.inverse_transform(preds_lstm_scaled).ravel()

    return resultados

# --- Ejecución ---
# df_estacionalidad = analizar_estacionalidad(pd.read_csv('datos.csv'), 'fecha', 'ventas', periodo=7)
# proyecciones = generar_pronosticos('datos.csv', 'fecha', 'ventas', dias_futuro=30)
# proyecciones.to_csv('pronosticos_multi_modelo.csv', index=False)

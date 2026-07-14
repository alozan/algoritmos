import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def preparar_features_temporales(df, col_fecha):
    """
    Extrae características útiles de la fecha para que el modelo entienda 
    patrones como el día de la semana, mes, o si es fin de semana.
    """
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha])
    df = df.sort_values(col_fecha)
    
    # Extraer componentes de la fecha
    df['dia_semana'] = df[col_fecha].dt.dayofweek
    df['dia_mes'] = df[col_fecha].dt.day
    df['mes'] = df[col_fecha].dt.month
    df['trimestre'] = df[col_fecha].dt.quarter
    df['es_fin_semana'] = np.where(df['dia_semana'] >= 5, 1, 0)
    
    return df

def entrenar_predictor_ventas(ruta_csv, col_fecha, col_ventas, dias_prediccion=30):
    """
    Entrena un modelo XGBoost para predecir ventas usando validación cronológica.
    """
    print(f"Cargando y procesando datos de: {ruta_csv}...")
    
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        return "Error: No se encontró el archivo CSV."

    # 1. Ingeniería de características temporales
    df = preparar_features_temporales(df, col_fecha)
    
    # 2. Crear variables de rezago (Lags) - ¿Cuánto se vendió hace 1 día, 7 días?
    # Esto es crucial para que el modelo entienda la inercia de las ventas
    df['ventas_lag_1'] = df[col_ventas].shift(1)
    df['ventas_lag_7'] = df[col_ventas].shift(7)
    df = df.dropna() # Eliminar las primeras filas que quedan sin lags

    # 3. Separar características (X) y objetivo (y)
    # Excluimos la fecha original porque los modelos de ML solo entienden números
    X = df.drop(columns=[col_fecha, col_ventas])
    y = df[col_ventas]

    # 4. División Cronológica (Train / Test)
    # Reservamos los últimos 'dias_prediccion' para evaluar el modelo
    limite = len(df) - dias_prediccion
    
    X_train, X_test = X.iloc[:limite], X.iloc[limite:]
    y_train, y_test = y.iloc[:limite], y.iloc[limite:]

    # 5. Configurar y entrenar XGBoost
    # Nota: Si usas hardware NVIDIA, cambia tree_method='hist' y device='cuda'
    modelo = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1, 
        random_state=42
    )
    
    print("Entrenando XGBoost Regressor...")
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50 # Imprime el error cada 50 iteraciones
    )

    # 6. Evaluación
    predicciones = modelo.predict(X_test)

    print("\n" + "="*40)
    print("MÉTRICAS DE RENDIMIENTO (TEST SET)")
    print("="*40)
    
    mae = mean_absolute_error(y_test, predicciones)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    mape = mean_absolute_percentage_error(y_test, predicciones) * 100

    print(f"Error Absoluto Medio (MAE): {mae:.2f} unidades")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} unidades")
    print(f"Error Porcentual (MAPE): {mape:.2f}%")

    return modelo, X.columns

# --- Ejemplo de Uso ---
# modelo, columnas = entrenar_predictor_ventas('historial_ventas.csv', 'fecha_venta', 'total_vendido', dias_prediccion=30)

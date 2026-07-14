import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
# import joblib # Descomentar para guardar el modelo

def entrenar_clasificador_csv(ruta_csv, columna_objetivo):
    """
    Carga un CSV, entrena un modelo de clasificación y muestra las métricas.
    
    Args:
        ruta_csv (str): Ruta al archivo .csv.
        columna_objetivo (str): El nombre de la columna que el modelo debe aprender a predecir.
    """
    print(f"Cargando datos de: {ruta_csv}...")
    
    # 1. Cargar datos
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        return "Error: No se encontró el archivo CSV."
        
    # Eliminar filas donde la variable objetivo sea nula (opcional pero recomendado)
    df = df.dropna(subset=[columna_objetivo])

    # Separar características (X) de la etiqueta a predecir (y)
    X = df.drop(columns=[columna_objetivo])
    y = df[columna_objetivo]

    # 2. Identificar tipos de datos para el preprocesamiento automático
    cols_numericas = X.select_dtypes(include=['int64', 'float64']).columns
    cols_categoricas = X.select_dtypes(include=['object', 'category']).columns

    # 3. Configurar transformadores
    # StandardScaler: Normaliza los números (media 0, varianza 1)
    # OneHotEncoder: Convierte categorías de texto en columnas binarias
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), cols_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cols_categoricas)
        ])

    # 4. Construir el Pipeline
    pipeline = Pipeline(steps=[
        ('preprocesador', preprocesador),
        ('clasificador', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # 5. Dividir los datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Entrenando el modelo...")
    # 6. Entrenar el modelo
    pipeline.fit(X_train, y_train)

    # 7. Evaluar el modelo
    y_pred = pipeline.predict(X_test)

    # --- Resultados ---
    print("\n" + "="*40)
    print("MÉTRICAS DE EVALUACIÓN")
    print("="*40)
    
    print("\n1. Reporte de Clasificación:")
    # Muestra Precisión (Precision), Sensibilidad (Recall) y F1-Score por clase
    print(classification_report(y_test, y_pred))

    print("\n2. Matriz de Confusión:")
    # Muestra los falsos positivos y falsos negativos
    print(confusion_matrix(y_test, y_pred))
    
    # 8. Guardar el pipeline completo para producción (Opcional)
    # joblib.dump(pipeline, 'modelo_entrenado.pkl')
    # print("\nModelo guardado como 'modelo_entrenado.pkl'")

    return pipeline

# --- Ejemplo de Uso ---
# pipeline_final = entrenar_clasificador_csv('datos.csv', 'etiqueta_a_predecir')

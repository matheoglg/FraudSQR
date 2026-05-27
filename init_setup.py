# init_setup.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def preparar_entorno():
    print("=== Iniciando Configuración Previa del Modelo de ML ===")
    
    # 1. Crear las carpetas de destino si no existen
    os.makedirs("src/models/saved_models", exist_ok=True)
    
    # 2. Verificar que existan los datos crudos
    ruta_datos = "data/raw/siniestros.csv"
    if not os.path.exists(ruta_datos):
        print(f"❌ Error: No se encontró el archivo de datos en '{ruta_datos}'.")
        print("Por favor, asegúrate de colocar tus archivos CSV en la carpeta 'data/raw/' antes de continuar.")
        return

    # 3. Leer los datos para el entrenamiento inicial
    print("📥 Leyendo dataset histórico de siniestros...")
    df_siniestros = pd.read_csv(ruta_datos)
    
    # 4. Alinear las características exactamente como las espera build_features.py
    X = df_siniestros[['monto_reclamado', 'dias_desde_inicio_poliza', 
                       'dias_desde_fin_poliza', 'dias_entre_ocurrencia_reporte', 
                       'historial_siniestros_asegurado']].fillna(0)
    
    y = df_siniestros['etiqueta_fraude_simulada'].fillna(0)
    
    # 5. Entrenar el clasificador base
    print("🧠 Entrenando el modelo predictivo RandomForest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y)
    
    # 6. Serializar y exportar el artefacto .pkl
    ruta_modelo = "src/models/saved_models/fraud_model.pkl"
    joblib.dump(model, ruta_modelo)
    
    print(f"✅ ¡Éxito! El artefacto del modelo ha sido guardado en: '{ruta_modelo}'")
    print("=== Configuración completada. Ya puedes ejecutar 'streamlit run src/app/main.py' ===")

if __name__ == "__main__":
    preparar_entorno()
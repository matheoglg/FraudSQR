# src/models/fraud_model.py
import os
import joblib
import pandas as pd

class FraudMLModel:
    def __init__(self, model_path: str = "src/models/saved_models/fraud_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.load_model_artifact()

    def load_model_artifact(self):
        """Carga el modelo pre-entrenado de forma rápida y limpia sin re-entrenamientos."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
        else:
            print(f"CRITICAL WARNING: No se encontró el artefacto serializado en {self.model_path}.")

    def predict_probability(self, feature_vector: pd.DataFrame) -> float:
        """Realiza la inferencia estadística pura en tiempo real (0.0 a 1.0)."""
        if not self.is_loaded or self.model is None:
            return 0.0
        try:
            return float(self.model.predict_proba(feature_vector)[0][1])
        except Exception as e:
            print(f"Error en el motor predictivo de ML: {str(e)}")
            return 0.0
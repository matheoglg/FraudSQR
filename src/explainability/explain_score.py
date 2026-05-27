# src/explainability/explain_score.py
import pandas as pd
from src.rules.fraud_rules import FraudRulesEvaluator
from src.models.fraud_model import FraudMLModel
from src.features.build_features import FeatureBuilder

class ScoreExplainerEngine:
    def __init__(self, datasets: dict):
        self.siniestros = datasets["siniestros"]
        self.documentos = datasets["documentos"]
        self.proveedores = datasets["proveedores"]
        
        self.rules_evaluator = FraudRulesEvaluator(self.siniestros, self.documentos, self.proveedores)
        self.ml_model = FraudMLModel()

    def compute_hybrid_score(self, siniestro_row: pd.Series) -> dict:
        """Fusiona el conocimiento determinista con la probabilidad de Machine Learning."""
        # 1. Evaluar Reglas de Negocio
        rules_res = self.rules_evaluator.evaluate_rules(siniestro_row)
        
        # 2. Extraer Variables e Inferir con el Modelo .pkl cargado
        features = FeatureBuilder.prepare_inference_vector(siniestro_row)
        ml_prob = self.ml_model.predict_probability(features)
        ml_score_component = int(ml_prob * 30) # El bosque aleatorio aporta un peso máximo de 30 puntos

        # Score Híbrido Final Acotado
        total_score = min(rules_res['score_rules'] + ml_score_component, 100)
        
        alertas = rules_res['alertas'].copy()
        if ml_prob > 0.65:
            alertas.append(f"ML-Engine: Patrón anómalo multivariable detectado (Confianza estadística: {round(ml_prob*100, 1)}%).")

        # Semáforo de Riesgo Exigido
        if total_score <= 40:
            semaforo, color = "Verde", "#10B981"
        elif total_score <= 75:
            semaforo, color = "Amarillo", "#F59E0B"
        else:
            semaforo, color = "Rojo", "#EF4444"

        # Buscar proveedor real
        prov_match = self.proveedores[self.proveedores['id_proveedor'] == siniestro_row['id_proveedor']]
        nombre_prov = prov_match.iloc[0]['nombre'] if not prov_match.empty else "No Identificado"

        return {
            "id_siniestro": siniestro_row.get('id_siniestro', 9999),
            "id_poliza": siniestro_row['id_poliza'],
            "id_asegurado": siniestro_row['id_asegurado'],
            "score": total_score,
            "ml_probability": ml_prob,
            "semaforo": semaforo,
            "color": color,
            "alertas": alertas,
            "descripcion": siniestro_row['descripcion'],
            "sucursal": siniestro_row['sucursal'],
            "monto": float(siniestro_row['monto_reclamado']),
            "id_proveedor": siniestro_row['id_proveedor'],
            "nombre_proveedor": nombre_prov,
            "historial_siniestros_asegurado": siniestro_row['historial_siniestros_asegurado']
        }

    def process_entire_dataset(self) -> pd.DataFrame:
        """Evalúa masivamente la tabla completa para el Dashboard Global."""
        records = []
        for _, row in self.siniestros.iterrows():
            records.append(self.compute_hybrid_score(row))
        return pd.DataFrame(records).sort_values(by="score", ascending=False)
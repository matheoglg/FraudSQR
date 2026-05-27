# src/rules/fraud_rules.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FraudRulesEvaluator:
    def __init__(self, df_siniestros, df_documentos, df_proveedores):
        self.siniestros = df_siniestros
        self.documentos = df_documentos
        self.proveedores = df_proveedores
        self._initialize_nlp_matrix()

    def _initialize_nlp_matrix(self):
        """RF-07: Vectoriza el corpus lingüístico histórico de narrativas."""
        self.siniestros['descripcion'] = self.siniestros['descripcion'].fillna("")
        self.vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.siniestros['descripcion'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def evaluate_rules(self, siniestro_row: pd.Series) -> dict:
        """Aplica las penalizaciones paramétricas del negocio."""
        score = 0
        alertas = []
        id_siniestro = siniestro_row['id_siniestro']

        # RF-05: Regla de Borde de Vigencia
        if siniestro_row['dias_desde_inicio_poliza'] <= 5:
            score += 30
            alertas.append(f"RF-05: Emisión Reciente (Ocurrido a los {siniestro_row['dias_desde_inicio_poliza']} días de vigencia).")
        elif siniestro_row['dias_desde_fin_poliza'] <= 5:
            score += 20
            alertas.append(f"RF-05: Fin de Vigencia Cercano ({siniestro_row['dias_desde_fin_poliza']} días restantes).")

        # RF-02: Gestión Documental Anomalías
        docs = self.documentos[self.documentos['id_siniestro'] == id_siniestro]
        if not docs.empty:
            inc_docs = docs[docs['inconsistencia_detectada'] == 'Sí']
            if not inc_docs.empty:
                score += 25
                alertas.append(f"RF-02: Inconsistencia en {inc_docs.iloc[0]['tipo_documento']}: {inc_docs.iloc[0]['observacion']}")

        # RF-03: Listas Restrictivas de Proveedores
        prov = self.proveedores[self.proveedores['id_proveedor'] == siniestro_row['id_proveedor']]
        if not prov.empty and prov.iloc[0]['porcentaje_de_casos_observados'] > 10.0:
            score += 20
            alertas.append(f"RF-03: Alerta Proveedor - {prov.iloc[0]['nombre']} registra {prov.iloc[0]['porcentaje_de_casos_observados']}% de casos observados.")

        # RF-07: Similitud Coseno (Solo aplica si el registro existe en el dataset histórico)
        matching_indices = self.siniestros[self.siniestros['id_siniestro'] == id_siniestro].index
        if not matching_indices.empty:
            idx = matching_indices[0]
            similares = self.similarity_matrix[idx]
            mask = np.ones(similares.shape, dtype=bool)
            mask[idx] = False
            max_sim = similares[mask].max() if len(similares) > 1 else 0
            if max_sim >= 0.85:
                score += 25
                alertas.append(f"RF-07: Relato Duplicado - Similitud del {round(max_sim*100, 1)}% con otra narrativa del sistema.")

        return {"score_rules": score, "alertas": alertas}
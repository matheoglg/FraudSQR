# src/features/build_features.py
import pandas as pd

class FeatureBuilder:
    @staticmethod
    def prepare_inference_vector(siniestro_row: pd.Series) -> pd.DataFrame:
        """Garantiza la alineación exacta de columnas requerida por el modelo .pkl."""
        data = {
            'monto_reclamado': float(siniestro_row.get('monto_reclamado', 0)),
            'dias_desde_inicio_poliza': int(siniestro_row.get('dias_desde_inicio_poliza', 0)),
            'dias_desde_fin_poliza': int(siniestro_row.get('dias_desde_fin_poliza', 0)),
            'dias_entre_ocurrencia_reporte': int(siniestro_row.get('dias_entre_ocurrencia_reporte', 0)),
            'historial_siniestros_asegurado': int(siniestro_row.get('historial_siniestros_asegurado', 0))
        }
        return pd.DataFrame([data])
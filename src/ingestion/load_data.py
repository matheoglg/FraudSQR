# src/ingestion/load_data.py
import os
import pandas as pd

class DataLoader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir

    def load_all_datasets(self) -> dict:
        """Carga de manera estricta y segura las tablas relacionales del ecosistema."""
        datasets = {}
        required_files = {
            "siniestros": "siniestros.csv",
            "polizas": "polizas.csv",
            "asegurados": "asegurados.csv",
            "proveedores": "proveedores.csv",
            "documentos": "documentos.csv"
        }
        
        for key, filename in required_files.items():
            path = os.path.join(self.data_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Error Crítico: No se encontró el archivo mandatorio: {path}")
            datasets[key] = pd.read_csv(path)
            
        return datasets
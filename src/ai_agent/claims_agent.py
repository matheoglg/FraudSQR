# src/ai_agent/claims_agent.py
import google.generativeai as genai
import pandas as pd

class ClaimsIAAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Uso nativo del modelo rápido y avanzado solicitado para la HacklAthon
        self.model = genai.GenerativeModel('gemini-3.1-flash-lite')

    def generate_explicability_report(self, case_summary: dict) -> str:
        """Capa Ética: Genera explicabilidad detallada para mitigar acusaciones automáticas."""
        prompt = f"""
        Actúa como un Auditor de Riesgos Senior para Aseguradora del Sur.
        Analiza las señales encontradas por nuestro ecosistema relacional para el Siniestro #{case_summary['id_siniestro']}:
        - Score Híbrido Asignado: {case_summary['score']}/100
        - Clasificación Semáforo: {case_summary['semaforo']}
        - Alertas Operativas: {case_summary['alertas']}
        - Declaración del Asegurado: "{case_summary['descripcion']}"
        - Monto en disputa: ${case_summary['monto']}

        Genera una argumentación técnica y justificada en español para el analista humano. 
        Explica las razones del nivel de riesgo y proporciona recomendaciones claras para la auditoría física o documental (ej. auditar facturas, inspeccionar el taller, corroborar fechas). 
        Mantén un tono neutral y corporativo, recordando que es una herramienta de apoyo al diagnóstico, no un veredicto legal automático.
        """
        try:
            return self.model.generate_content(prompt).text
        except Exception as e:
            return f"Error al invocar la API de Gemini: {str(e)}"

    def execute_agentic_analytics(self, query: str, df_global: pd.DataFrame) -> str:
        """Responde con solidez las Pruebas de Fuego estadísticas planteadas por el jurado."""
        total_casos = len(df_global)
        df_rojos = df_global[df_global['semaforo'] == 'Rojo']
        
        talleres_criticos = ""
        if not df_rojos.empty:
            counts = df_rojos['nombre_proveedor'].value_counts()
            talleres_criticos = "\n".join([f"- {name}: {count} casos críticos" for name, count in counts.items()])
            
        contexto = f"""
        Eres el Agente Virtual Analítico de FraudIA. Tienes acceso completo en memoria a las métricas del cuadro de mando de Aseguradora del Sur:
        - Volumen Total de Reclamos en este Lote: {total_casos}
        - Total de Casos de Máximo Riesgo (Semáforo Rojo): {len(df_rojos)}
        - Concentración de Proveedores y Talleres en Casos Rojos:\n{talleres_criticos}

        Usa estos datos exactos consolidados para responder la pregunta del analista de manera corporativa, estructurada y persuasiva.
        """
        try:
            return self.model.generate_content(f"{contexto}\n\nPregunta del Jurado: {query}\nRespuesta Analítica:").text
        except Exception as e:
            return f"Error en la consulta agéntica de IA: {str(e)}"
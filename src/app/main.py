# src/app/main.py
import streamlit as st
import pandas as pd
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.ingestion.load_data import DataLoader
from src.explainability.explain_score import ScoreExplainerEngine
from src.ai_agent.claims_agent import ClaimsIAAgent

load_dotenv()

st.set_page_config(page_title="FraudIA Suite - Aseguradora del Sur", layout="wide")

# Carga de estilos Premium (Norman / Nielsen)
if os.path.exists("src/app/styles.css"):
    with open("src/app/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def init_core_pipeline():
    loader = DataLoader()
    datasets = loader.load_all_datasets()
    engine_object = ScoreExplainerEngine(datasets)
    return engine_object

engine = init_core_pipeline()
df_dashboard = engine.process_entire_dataset()

# Inicialización segura de Gemini con llave del .env
api_key = os.getenv("GEMINI_API_KEY", "")
agent = ClaimsIAAgent(api_key) if api_key else None

# --- SIDEBAR: CONTROL DE NAVEGACIÓN ---
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=55)
st.sidebar.markdown("## FraudIA Workstation")
menu_option = st.sidebar.radio("Módulos Operativos:", [
    "📊 Control Global (Dashboard)",
    "🔍 Auditoría de Casos & Explicabilidad",
    "📥 Cargar Nuevo Siniestro (Inferencia Real-Time)",
    "💬 Asistente Virtual Agéntico"
])

st.sidebar.markdown("---")
if api_key:
    st.sidebar.success("⚡ Conectado a Gemini 3.1 Flash Exp")
else:
    st.sidebar.error("⚠️ Llave API ausente en el .env")

# ==========================================
# MÓDULO 1: DASHBOARD GLOBAL
# ==========================================
if menu_option == "📊 Control Global (Dashboard)":
    st.markdown("# Panel de Control Macroscópico de Riesgos")
    st.markdown("---")
    
    # Grid de Indicadores Clave
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi-card'><h4 style='color:#EF4444;margin:0;'>🚨 Riesgo Crítico (Rojo)</h4><h2>{len(df_dashboard[df_dashboard['semaforo']=='Rojo'])}</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><h4 style='color:#F59E0B;margin:0;'>⚠️ Observados (Amarillo)</h4><h2>{len(df_dashboard[df_dashboard['semaforo']=='Amarillo'])}</h2></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><h4 style='color:#10B981;margin:0;'>✅ Estables (Verde)</h4><h2>{len(df_dashboard[df_dashboard['semaforo']=='Verde'])}</h2></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi-card'><h4 style='margin:0;color:#475569;'>💰 Monto Global Auditado</h4><h2>${df_dashboard['monto'].sum():,.2f}</h2></div>", unsafe_allow_html=True)

    st.markdown("### 📋 Casos Priorizados y Ordenados por Gravedad")
    df_show = df_dashboard[['id_siniestro', 'id_poliza', 'sucursal', 'monto', 'score', 'semaforo']].copy()
    df_show.columns = ['ID Siniestro', 'ID Póliza', 'Sucursal', 'Monto ($)', 'Score Combinado (Rules+ML)', 'Semáforo']
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    # Descarga de reportes (Requerimiento Funcional)
    csv_data = df_dashboard.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Reporte de Casos de Mayor Riesgo (CSV)", data=csv_data, file_name="reporte_priorizacion_fraude.csv", mime="text/csv")

# ==========================================
# MÓDULO 2: AUDITORÍA DE CASOS & EXPLICABILIDAD
# ==========================================
elif menu_option == "🔍 Auditoría de Casos & Explicabilidad":
    st.markdown("# Centro de Peritaje Analítico Individual")
    
    target_id = st.selectbox("Seleccione un Siniestro de la Base Histórica:", options=df_dashboard['id_siniestro'].tolist())
    
    if target_id:
        # Recuperar registro pre-calculado
        case = df_dashboard[df_dashboard['id_siniestro'] == target_id].iloc[0].to_dict()
        
        st.markdown(f"### Nivel de Alerta: <span style='color:{case['color']};font-weight:700;'>{case['semaforo'].upper()} (Score Híbrido: {case['score']}/100)</span>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📁 Ficha Relacional Cruzada")
            st.write(f"**Póliza Asociada:** `{case['id_poliza']}`")
            st.write(f"**Identificador de Asegurado:** `{case['id_asegurado']}`")
            st.write(f"**Establecimiento / Proveedor:** {case['nombre_proveedor']} (`{case['id_proveedor']}`)")
            st.write(f"**Sucursal Emisora:** {case['sucursal']}")
            st.write(f"**Monto Exigido:** ${case['monto']:,.2f}")
            st.info(f"**Descripción del Hecho:** {case['descripcion']}")
            
        with col2:
            st.markdown("#### ⚠️ Disparadores de Alertas Combinadas")
            if case['alertas']:
                for alerta in case['alertas']:
                    st.markdown(f"<div class='alert-pill-box'>{alerta}</div>", unsafe_allow_html=True)
            else:
                st.success("No se detectan anomalías paramétricas en las reglas del negocio.")
                
            st.markdown("---")
            st.markdown("#### 🧠 Justificación Cognitiva (Explicabilidad Avanzada de Gemini)")
            if not agent:
                st.info("Configura la API Key para desplegar el análisis del modelo de lenguaje.")
            else:
                with st.spinner("Generando reporte argumentado explicable..."):
                    st.markdown(agent.generate_explicability_report(case))

# ==========================================
# MÓDULO 3: CARGAR NUEVO SINIESTRO
# ==========================================
elif menu_option == "📥 Cargar Nuevo Siniestro (Inferencia Real-Time)":
    st.markdown("# Evaluación Inmediata de Nuevos Reclamos")
    st.markdown("<p style='color:#64748B;'>Simula la carga interactiva de un nuevo siniestro ingresado al sistema. El modelo pre-entrenado `.pkl` calculará el score al instante sin requerir re-entrenamiento.</p>", unsafe_allow_html=True)
    
    with st.form("new_claim_form"):
        cc1, cc2 = st.columns(2)
        with cc1:
            in_id_poliza = st.text_input("Código de Póliza:", value="POL-999999")
            in_id_asegurado = st.text_input("ID Asegurado:", value="ASEG-111111")
            in_id_proveedor = st.selectbox("Proveedor Asociado:", options=engine.proveedores['id_proveedor'].tolist())
            in_sucursal = st.selectbox("Sucursal:", options=["Guayaquil", "Quito", "Cuenca", "Azogues", "Machala", "Loja", "Zamora"])
            in_monto = st.number_input("Monto Reclamado ($):", min_value=0.0, value=1500.0)
        with cc2:
            in_dias_ini = st.number_input("Días transcurridos desde inicio de la póliza:", min_value=0, value=3)
            in_dias_fin = st.number_input("Días faltantes para fin de la póliza:", min_value=0, value=360)
            in_dias_rep = st.number_input("Días transcurridos entre ocurrencia y reporte:", min_value=0, value=1)
            in_historial = st.number_input("Siniestros previos del asegurado (últimos 12 meses):", min_value=0, value=4)
            
        in_descripcion = st.text_area("Narrativa del Hecho:", value="El cliente reporta atención médica de emergencia médica ocurrida pocas horas después de la activación del contrato.")
        
        submitted = st.form_submit_button("🛡️ Evaluar Siniestro Inmediatamente")
        
        if submitted:
            # Crear fila virtual en formato Series para evaluar en tiempo real
            new_row = pd.Series({
                'id_siniestro': 9999, # ID temporal fuera de la base estática
                'id_poliza': in_id_poliza,
                'id_asegurado': in_id_asegurado,
                'id_proveedor': in_id_proveedor,
                'sucursal': in_sucursal,
                'monto_reclamado': in_monto,
                'dias_desde_inicio_poliza': in_dias_ini,
                'dias_desde_fin_poliza': in_dias_fin,
                'dias_entre_ocurrencia_reporte': in_dias_rep,
                'historial_siniestros_asegurado': in_historial,
                'descripcion': in_descripcion
            })
            
            # Ejecutar inferencia limpia sobre la estructura pre-cargada
            res = engine.compute_hybrid_score(new_row)
            
            st.markdown("---")
            st.markdown(f"### Resultado del Análisis Instantáneo: <span style='color:{res['color']};font-weight:700;'>{res['semaforo'].upper()} (Score: {res['score']}/100)</span>", unsafe_allow_html=True)
            
            st.markdown("#### Banderas Activas:")
            for al in res['alertas']:
                st.error(al)
                
            if agent:
                with st.spinner("Gemini derivando evaluación fáctica sobre los nuevos datos..."):
                    st.markdown("#### Resumen Analítico de la IA:")
                    st.markdown(agent.generate_explicability_report(res))

# ==========================================
# MÓDULO 4: ASISTENTE VIRTUAL AGÉNTICO
# ==========================================
elif menu_option == "💬 Asistente Virtual Agéntico":
    st.markdown("# Copiloto Conversacional Antifraude")
    st.markdown("<p style='color:#64748B;'>Permite interrogar la base consolidada mediante lenguaje natural para el soporte en el Pitch en vivo.</p>", unsafe_allow_html=True)
    
    st.info("**Prueba de Fuego Mandatoria (Pág 13 del PDF):** Ingresa exactamente la consulta: *¿Qué proveedores concentran las alertas rojas?*")
    
    user_query = st.text_input("Escribe tu consulta analítica corporativa:", placeholder="Ej. ¿Cuántos casos están en semáforo rojo actualmente?")
    
    if user_query:
        if not agent:
            st.error("Por favor, asegúrate de configurar tu archivo .env con la API Key para usar este módulo.")
        else:
            with st.spinner("Gemini analizando la distribución agregada de las tablas relacionales..."):
                st.markdown("---")
                st.markdown(agent.execute_agentic_analytics(user_query, df_dashboard))
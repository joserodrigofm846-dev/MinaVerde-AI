
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="MinaVerde AI",
    page_icon="🌱",
    layout="wide"
)

# ======================================
# FASE 5 - DISEÑO PROFESIONAL
# ======================================

st.markdown("""
<style>

/* Fondo de la aplicación */
.stApp {
    background-color: #0B1220;
    color: #F1F5F9;
}

/* Encabezados */
h1, h2, h3 {
    color: #E2E8F0;
    font-family: Arial, sans-serif;
}

/* Indicadores */
[data-testid="stMetric"] {
    background-color: #172338;
    border: 1px solid #263A50;
    padding: 20px;
    border-radius: 12px;
}

/* Valores de los indicadores */
[data-testid="stMetricValue"] {
    color: #36D399;
}

/* Botones */
.stDownloadButton button {
    background-color: #14866D;
    color: white;
    border-radius: 9px;
    border: none;
}

.stDownloadButton button:hover {
    background-color: #1AA784;
    color: white;
}

</style>
""", unsafe_allow_html=True)


st.title("🌱 MinaVerde AI")
st.subheader("Monitoreo ambiental minero")
st.info("Prototipo educativo con datos simulados.")

# Crear datos de prueba reproducibles
@st.cache_data
def generar_datos():
    rng = np.random.default_rng(42)
    fechas = pd.date_range("2026-01-01", periods=60)

    registros = []
    for punto in ["Punto A", "Punto B", "Punto C"]:
        for fecha in fechas:
            registros.append({
                "fecha": fecha,
                "punto": punto,
                "ph": rng.normal(7.2, 0.65),
                "turbidez": max(0, rng.normal(12, 4)),
                "cobre_mg_l": max(0, rng.normal(0.4, 0.15))
            })

    return pd.DataFrame(registros)

archivo = st.file_uploader(
    "Sube tus datos CSV (opcional)",
    type=["csv"]
)

if archivo is not None:
    df = pd.read_csv(archivo)
else:
    df = generar_datos()

# Validar los datos antes de utilizarlos
columnas = [
    "fecha", "punto", "ph",
    "turbidez", "cobre_mg_l"
]

if not all(c in df.columns for c in columnas):
    st.error("El CSV no contiene las columnas necesarias.")
    st.stop()

df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

for columna in ["ph", "turbidez", "cobre_mg_l"]:
    df[columna] = pd.to_numeric(df[columna], errors="coerce")

df = df.dropna(subset=columnas)

if df.empty:
    st.error("No existen registros válidos para analizar.")
    st.stop()

# Selección de punto de monitoreo
punto = st.selectbox(
    "Selecciona un punto de monitoreo",
    sorted(df["punto"].unique())
)

datos = df[df["punto"] == punto].sort_values("fecha").copy()

# Umbrales de demostración, NO son ECA legales
ph_min = 6.5
ph_max = 8.5
turbidez_max = 15.0

datos["alerta"] = (
    (datos["ph"] < ph_min) |
    (datos["ph"] > ph_max) |
    (datos["turbidez"] > turbidez_max)
)

# Indicadores generales
a, b, c = st.columns(3)

a.metric("pH promedio", f"{datos['ph'].mean():.2f}")
b.metric(
    "Turbidez promedio",
    f"{datos['turbidez'].mean():.2f}"
)
c.metric(
    "Registros con alerta",
    int(datos["alerta"].sum())
)

# Gráficos
st.subheader("Evolución de los parámetros")

parametro = st.selectbox(
    "Parámetro",
    ["ph", "turbidez", "cobre_mg_l"]
)

st.line_chart(
    datos.set_index("fecha")[[parametro]]
)

# Mostrar alertas

# ======================================
# INTELIGENCIA ARTIFICIAL - MINAVERDE AI
# ======================================

from sklearn.ensemble import IsolationForest

st.subheader("🤖 Análisis con Inteligencia Artificial")

# Seleccionamos los datos que analizará la IA
variables = datos[["ph", "turbidez", "cobre_mg_l"]]

# Creamos nuestro modelo de Machine Learning
modelo = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

# Entrenamos el modelo con los datos ambientales
modelo.fit(variables)

# Analizamos las mediciones
datos["resultado_ia"] = modelo.predict(variables)

# Identificamos registros inusuales
anomalias = datos[datos["resultado_ia"] == -1]

st.metric("Anomalías detectadas por IA", len(anomalias))

if len(anomalias) > 0:
    st.warning("La IA encontró mediciones inusuales.")
    st.dataframe(anomalias)
else:
    st.success("No se detectaron anomalías.")

# Gráfico para visualizar las anomalías
st.subheader("Visualización de anomalías")

st.scatter_chart(
    datos,
    x="ph",
    y="turbidez",
    color="resultado_ia"
)

st.subheader("Resultados del monitoreo")

if datos["alerta"].any():
    st.warning("Se detectaron registros fuera de los rangos de prueba.")
else:
    st.success("Sin alertas bajo los umbrales de prueba.")

st.dataframe(datos, use_container_width=True)

# Exportar resultados
csv = datos.to_csv(index=False).encode("utf-8")

st.download_button(
    "Descargar resultados CSV",
    data=csv,
    file_name="monitoreo_ambiental.csv",
    mime="text/csv"
)

# =======================================
# FASE 3 - MAPA DE MONITOREO AMBIENTAL
# =======================================

st.divider()

st.header("🗺️ Mapa de Monitoreo Ambiental")

st.write(
    "Ubicación simulada de los puntos de "
    "monitoreo de calidad del agua."
)

# Coordenadas ficticias para fines educativos
puntos_mapa = pd.DataFrame({
    "punto": ["Punto A", "Punto B", "Punto C"],
    "lat": [-9.50, -9.52, -9.54],
    "lon": [-77.30, -77.32, -77.28]
})

# Mostrar mapa interactivo
st.map(
    puntos_mapa,
    latitude="lat",
    longitude="lon",
    size=150
)

st.subheader("Coordenadas de monitoreo")
st.dataframe(puntos_mapa, hide_index=True)

st.caption(
    "Las ubicaciones son ilustrativas y no "
    "representan puntos reales de una mina."
)

# ==========================================
# FASE 4 - GENERADOR DE REPORTES PDF
# ==========================================

from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

st.divider()
st.header("📄 Reportes Ambientales")

st.write(
    "Genera un reporte del punto de monitoreo seleccionado."
)


def generar_pdf(datos, punto):
    memoria = BytesIO()

    pdf = canvas.Canvas(memoria, pagesize=A4)
    ancho, alto = A4

    # Título
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(50, alto - 60, "MINAVERDE AI")

    pdf.setFont("Helvetica", 13)
    pdf.drawString(
        50, alto - 85,
        "Reporte de Monitoreo Ambiental"
    )

    # Información general
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(
        50, alto - 130,
        f"Punto de monitoreo: {punto}"
    )

    pdf.setFont("Helvetica", 11)
    pdf.drawString(
        50, alto - 155,
        f"Registros analizados: {len(datos)}"
    )

    # Calcular indicadores
    ph = datos["ph"].mean()
    turbidez = datos["turbidez"].mean()
    cobre = datos["cobre_mg_l"].mean()

    alertas = int(datos["alerta"].sum())
    anomalias = int((datos["resultado_ia"] == -1).sum())

    # Escribir resultados
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(
        50, alto - 205,
        "Indicadores ambientales"
    )

    pdf.setFont("Helvetica", 11)

    resultados = [
        f"pH promedio: {ph:.2f}",
        f"Turbidez promedio: {turbidez:.2f}",
        f"Cobre promedio: {cobre:.4f} mg/L",
        f"Registros con alerta: {alertas}",
        f"Anomalias detectadas por IA: {anomalias}"
    ]

    y = alto - 235

    for resultado in resultados:
        pdf.drawString(50, y, resultado)
        y -= 25

    # Advertencia
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(
        50, y - 45,
        "Reporte educativo con datos simulados."
    )

    pdf.drawString(
        50, y - 60,
        "No representa una evaluacion de cumplimiento legal."
    )

    # Finalizar PDF
    pdf.save()
    memoria.seek(0)

    return memoria.getvalue()


# Crear el PDF
reporte = generar_pdf(datos, punto)

# Botón de descarga
st.download_button(
    label="📥 Descargar reporte PDF",
    data=reporte,
    file_name="reporte_ambiental.pdf",
    mime="application/pdf"
)



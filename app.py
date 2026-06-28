# -*- coding: utf-8 -*-
"""
app.py — Sistema de Alerta Temprana y Predicción de Riesgo Hídrico
Subcuenca Represa El Coyolar · Modelo V13 · GEE + Sentinel-2 + Landsat 8 + CHIRPS

Modelo final : RandomForestClassifier
               n_estimators=200, max_depth=6, random_state=42
Features (18): NDVI, NDWI_agua, NDWI_veg, LST, SPI_3  (base ×5)
               + lag1 de las 5 bases (×5)
               + lag2 y delta solo de NDVI, NDWI_agua, SPI_3 (×3 c/u)
               + mes_sin, mes_cos
Target       : NRHF — Nivel de Riesgo Hídrico Futuro (próximo mes, clases 0-3)
Archivo pkl  : modelo_sequia_futuro.pkl
ROI          : HydroATLAS nivel 12 — punto (-87.50904715160283, 14.333509533380646)
"""

import tempfile
import json
import streamlit as st
import pandas as pd
import numpy as np
import ee
import joblib
import folium
from streamlit_folium import st_folium # Added for displaying folium maps in Streamlit
import plotly.graph_objects as go
from groq import Groq
from scipy.stats import gamma, norm

# ==============================================================================
# 0. CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Sistema Predictivo de Sequías · El Coyolar",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric { background: #f7f6f3; border-radius: 10px; padding: .5rem .75rem; }
    .badge-riesgo {
        padding: 18px 24px; border-radius: 12px; text-align: center;
        font-size: 22px; font-weight: 600; color: white; margin-bottom: 1rem;
    }
    .interp-tag {
        display: inline-block; padding: 3px 10px; border-radius: 5px;
        font-size: 13px; font-weight: 600; margin-right: 6px;
    }
    .section-title {
        font-size: 16px; font-weight: 600; color: #1a1a19;
        margin: 1.2rem 0 .6rem; border-left: 4px solid #2a78d6;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("💧 Sistema de Alerta Temprana y Predicción de Riesgo Hídrico")
st.caption("Subcuenca Represa El Coyolar · GEE + Sentinel-2 + Landsat 8 + CHIRPS · Modelo V13")

# ==============================================================================
# 1. CLIENTES Y RECURSOS
# ==============================================================================
GROQ_API_KEY = st.secrets["DB_TOKEN"]
groq_client  = Groq(api_key=GROQ_API_KEY)

# ── Helper GEE (debe definirse ANTES de cargar_recursos) ──────────────────────
def _obtener_subcuenca_gee(lon: float, lat: float, nivel: int = 12) -> "ee.Geometry":
    """
    Devuelve la geometría HydroATLAS que contiene el punto dado.
    Idéntico a obtener_subcuenca_automatica() del script de entrenamiento V13.
    """
    nivel_str = str(nivel).zfill(2)
    cuencas   = ee.FeatureCollection(f'WWF/HydroATLAS/v1/Basins/level{nivel_str}')
    punto     = ee.Geometry.Point([lon, lat])
    return cuencas.filterBounds(punto).first().geometry()

@st.cache_resource(show_spinner="Inicializando Google Earth Engine y modelo…")
def cargar_recursos():
    """
    Inicializa GEE con cuenta de servicio y carga el modelo entrenado.
    - Modelo : modelo_sequia_futuro.pkl  (RandomForestClassifier V13)
    - ROI    : HydroATLAS nivel 12, mismo punto que el entrenamiento
    """
    credentials_json = json.loads(st.secrets["GEE_CREDENTIALS"])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(credentials_json, f)
        temp_file = f.name
    credentials = ee.ServiceAccountCredentials(credentials_json["client_email"], temp_file)
    ee.Initialize(credentials)

    # Mismo nombre de archivo que joblib.dump() en el entrenamiento V13
    modelo = joblib.load("modelo_sequia_futuro.pkl")

    # Mismo punto de referencia y nivel que en el script de entrenamiento V13
    roi = _obtener_subcuenca_gee(-87.50904715160283, 14.333509533380646, nivel=12)

    return modelo, roi

modelo, roi_subcuenca = cargar_recursos()
st.sidebar.success("✅ Modelo (RandomForest V13) + GEE listos")

# ==============================================================================
# 2. CONFIGURACIÓN DE RIESGO
# ==============================================================================
RIESGO_INFO = {
    0: {"nombre": "Sin sequía",       "emoji": "🟢", "color": "#2ecc71", "bg": "#EAF3DE", "txt": "#27500A"},
    1: {"nombre": "Sequía leve",      "emoji": "🟡", "color": "#f1c40f", "bg": "#FAEEDA", "txt": "#633806"},
    2: {"nombre": "Sequía moderada",  "emoji": "🟠", "color": "#e67e22", "bg": "#FAECE7", "txt": "#712B13"},
    3: {"nombre": "Sequía severa",    "emoji": "🔴", "color": "#e74c3c", "bg": "#FCEBEB", "txt": "#791F1F"},
}

INTERPRETACIONES = {
    "NDVI": [
        (lambda v: v < 0.35,  "#FCEBEB", "#791F1F", "< 0.35",        "Vegetación muy degradada. Cobertura mínima."),
        (lambda v: v < 0.50,  "#FAECE7", "#712B13", "0.35 – 0.50",   "Vegetación moderada a baja. Posible estrés."),
        (lambda v: v < 0.65,  "#EAF3DE", "#27500A", "0.50 – 0.65",   "Vegetación en condición normal."),
        (lambda v: True,       "#EAF3DE", "#27500A", "> 0.65",        "Vegetación densa y sana."),
    ],
    "NDWI_agua": [
        (lambda v: v < 0.00,  "#FCEBEB", "#791F1F", "< 0.00",        "Embalse muy bajo o seco. Situación crítica."),
        (lambda v: v < 0.10,  "#FAECE7", "#712B13", "0.00 – 0.10",   "Nivel bajo del embalse."),
        (lambda v: v < 0.22,  "#EAF3DE", "#27500A", "0.10 – 0.22",   "Nivel normal del embalse."),
        (lambda v: True,       "#EAF3DE", "#27500A", "> 0.22",        "Nivel alto del embalse."),
    ],
    "NDWI_veg": [
        (lambda v: v < -0.15, "#FCEBEB", "#791F1F", "< -0.15",       "Vegetación muy estresada, sin agua foliar."),
        (lambda v: v < -0.05, "#FAECE7", "#712B13", "-0.15 – -0.05", "Estrés hídrico moderado en plantas."),
        (lambda v: v < 0.05,  "#EAF3DE", "#27500A", "-0.05 – 0.05",  "Contenido de agua foliar normal."),
        (lambda v: True,       "#EAF3DE", "#27500A", "> 0.05",        "Vegetación con alto contenido de agua."),
    ],
    "LST": [
        (lambda v: v > 35,    "#FCEBEB", "#791F1F", "> 35 °C",       "Temperatura muy alta. Alto estrés térmico."),
        (lambda v: v > 32,    "#FAECE7", "#712B13", "32 – 35 °C",    "Temperatura elevada."),
        (lambda v: v > 27,    "#EAF3DE", "#27500A", "27 – 32 °C",    "Temperatura normal."),
        (lambda v: True,       "#EAF3DE", "#27500A", "< 27 °C",       "Temperatura fresca. Buena cobertura nubosa."),
    ],
    "SPI_3": [
        (lambda v: v < -1.5,  "#FCEBEB", "#791F1F", "< -1.5",        "Sequía severa. Precipitación muy por debajo del normal."),
        (lambda v: v < -1.0,  "#FAECE7", "#712B13", "-1.5 – -1.0",   "Sequía moderada. Déficit hídrico significativo."),
        (lambda v: v < 0.5,   "#EAF3DE", "#27500A", "-1.0 – 0.5",    "Condición normal a húmeda."),
        (lambda v: True,       "#EAF3DE", "#27500A", "> 0.5",         "Exceso de lluvia sobre el promedio histórico."),
    ],
}

def obtener_interpretacion(key, valor):
    for cond, bg, txt, rango, desc in INTERPRETACIONES[key]:
        if cond(valor):
            return bg, txt, rango, desc
    return "#EAF3DE", "#27500A", "—", "—"

# ==============================================================================
# 3. FUNCIONES GEE — extracción de variables (igual que en V13)
# ==============================================================================
def remover_nubes_sentinel(img):
    """Enmascara nubes y sombras Sentinel-2 usando la banda SCL (V13 l.310-323)."""
    scl  = img.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask).clip(roi_subcuenca)

def remover_nubes_landsat(img):
    """Enmascara nubes y sombras Landsat 8 usando QA_PIXEL (V13 l.380-386)."""
    qa   = img.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return img.updateMask(mask)

def obtener_datos_satelitales(anio: int, mes: int) -> pd.DataFrame:
    """
    Extrae NDVI, NDWI_agua, NDWI_veg, LST y Lluvia_mm para el mes/año dado.
    Lógica idéntica a extraer_variables() del entrenamiento V13 (l.326-435):
      · NDWI_agua → solo píxeles de agua permanente/semipermanente (JRC occurrence > 10 %)
      · NDWI_veg  → solo píxeles de vegetación/suelo (complemento de la máscara anterior)
      · LST       → Landsat 8 Collection 2 L2, fórmula USGS: DN×0.00341802 + 149.0 - 273.15
    """
    inicio = ee.Date.fromYMD(anio, mes, 1)
    fin    = inicio.advance(1, 'month')

    # Sentinel-2 SR Harmonized — nube filtrada, mediana mensual
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(roi_subcuenca)
          .filterDate(inicio, fin)
          .map(remover_nubes_sentinel)
          .median())

    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # Máscaras espaciales separadas para los dos NDWI (igual que V13)
    gsw          = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    mascara_agua = gsw.select('occurrence').gt(10)   # agua estacional + permanente
    mascara_veg  = mascara_agua.Not()                # tierra / vegetación

    ndwi_base = s2.normalizedDifference(['B3', 'B8'])
    ndwi_agua = ndwi_base.updateMask(mascara_agua).rename('NDWI_agua')
    ndwi_veg  = ndwi_base.updateMask(mascara_veg).rename('NDWI_veg')

    # Landsat 8 — LST (fórmula USGS Collection 2 Level 2, igual que V13 l.398)
    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterBounds(roi_subcuenca)
          .filterDate(inicio, fin)
          .map(remover_nubes_landsat)
          .median())
    lst = l8.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')

    # CHIRPS — precipitación mensual total
    lluvia = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
              .filterBounds(roi_subcuenca)
              .filterDate(inicio, fin)
              .sum()
              .reduceRegion(reducer=ee.Reducer.mean(),
                            geometry=roi_subcuenca, scale=5000)
              .get('precipitation'))

    # Un único reduceRegion para todo el stack (más eficiente)
    stack = ee.Image.cat([ndvi, ndwi_agua, ndwi_veg, lst])
    stats = stack.reduceRegion(reducer=ee.Reducer.mean(),
                               geometry=roi_subcuenca, scale=30)

    return pd.DataFrame([{
        "Anio":      anio,
        "Mes":       mes,
        "NDVI":      stats.get('NDVI').getInfo(),
        "NDWI_agua": stats.get('NDWI_agua').getInfo(),
        "NDWI_veg":  stats.get('NDWI_veg').getInfo(),
        "LST":       stats.get('LST').getInfo(),
        "Lluvia_mm": lluvia.getInfo() if lluvia else None,
    }])

def obtener_geometria_rio(roi) -> dict:
    """Retorna GeoJSON del cuerpo de agua permanente más grande dentro del ROI."""
    gsw  = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    agua = gsw.select('occurrence').gt(50).selfMask()
    vecs = agua.reduceToVectors(
        geometry=roi, scale=30, geometryType='polygon',
        eightConnected=True, maxPixels=1e9
    ).map(lambda f: f.set('area_m2', f.geometry().area(1)))
    return ee.Feature(vecs.sort('area_m2', False).first()).geometry().getInfo()

# ==============================================================================
# 4. SPI-3 EN PRODUCCIÓN
# ==============================================================================
# Estadísticos CHIRPS 1981-2025 para la subcuenca El Coyolar (un valor por mes).
# Estos se calculan en el entrenamiento con ajustar_distribucion_gamma_mensual().
# Reemplaza con los valores reales de tu notebook si los tienes exportados.
SPI_MEDIA_MES = {1:55, 2:38, 3:25, 4:30, 5:85,  6:125,
                 7:110,8:120,9:145,10:135,11:90, 12:70}
SPI_STD_MES   = {1:32, 2:28, 3:20, 4:24, 5:40,  6:45,
                 7:42, 8:43, 9:48, 10:46, 11:38, 12:36}

def calcular_spi3_simple(lluvia_mm: float, mes: int) -> float:
    """
    Aproximación lineal del SPI-3 usando media y desviación estándar mensuales
    históricos de CHIRPS 1981-2025.  En el entrenamiento se usa la transformación
    gamma completa (ajustar_distribucion_gamma_mensual + transformar_a_spi);
    esta función es el equivalente simplificado para la app.
    """
    mu  = SPI_MEDIA_MES.get(mes, 85.4)
    std = SPI_STD_MES.get(mes, 42.1)
    return float((lluvia_mm - mu) / std) if std > 0 else 0.0

# ==============================================================================
# 5. FEATURE ENGINEERING — orden exacto del entrenamiento V13
# ==============================================================================
# Extraído de entrenamiento_modelo_v13.py líneas 702-706:
#   X = df[['NDVI','NDWI_agua','NDWI_veg','LST','SPI_3',
#           'NDVI_lag1','NDWI_agua_lag1','NDWI_veg_lag1','LST_lag1','SPI_3_lag1',
#           'NDVI_lag2','NDWI_agua_lag2','SPI_3_lag2',
#           'NDVI_delta','NDWI_agua_delta','SPI_3_delta',
#           'mes_sin','mes_cos']]
# IMPORTANTE: el orden de la lista debe ser idéntico al del entrenamiento;
# cualquier cambio provocará predicciones incorrectas.
FEATURES_V13 = [
    # Base (5)
    'NDVI', 'NDWI_agua', 'NDWI_veg', 'LST', 'SPI_3',
    # Lag 1 — las 5 variables base (5)
    'NDVI_lag1', 'NDWI_agua_lag1', 'NDWI_veg_lag1', 'LST_lag1', 'SPI_3_lag1',
    # Lag 2 — solo NDVI, NDWI_agua, SPI_3 (3)
    'NDVI_lag2', 'NDWI_agua_lag2', 'SPI_3_lag2',
    # Delta — solo NDVI, NDWI_agua, SPI_3 (3)
    'NDVI_delta', 'NDWI_agua_delta', 'SPI_3_delta',
    # Estacionalidad cíclica (2)
    'mes_sin', 'mes_cos',
]  # Total: 18 features

def construir_features(df_actual: pd.DataFrame, df_historial=None) -> pd.DataFrame:
    """
    Construye el vector de 18 features en el orden exacto del entrenamiento V13.

    Parámetros
    ----------
    df_actual   : DataFrame con los datos del mes a predecir (1 fila).
    df_historial: DataFrame con los meses previos ya procesados (opcional).
                  Si se pasa ≥2 filas, se usan las últimas dos como lag1 y lag2.
                  Si se pasa 1 fila, esa fila actúa como lag1 y lag2.
                  Si es None, los valores actuales se usan como proxy para los lags
                  (introduce pequeño sesgo; aceptable en ausencia de historial).

    Notas
    -----
    · lag2 y delta solo existen para NDVI, NDWI_agua y SPI_3 (igual que en V13).
    · NDWI_veg y LST solo tienen lag1 (no lag2 ni delta).
    """
    row = df_actual.iloc[0]

    if df_historial is not None and len(df_historial) >= 2:
        lag1 = df_historial.iloc[-1]
        lag2 = df_historial.iloc[-2]
    elif df_historial is not None and len(df_historial) == 1:
        lag1 = df_historial.iloc[-1]
        lag2 = df_historial.iloc[-1]
    else:
        # Sin historial: proxy — lags iguales al mes actual
        lag1 = row
        lag2 = row

    mes = int(row['Mes'])
    f   = {}

    # Base
    for c in ['NDVI', 'NDWI_agua', 'NDWI_veg', 'LST', 'SPI_3']:
        f[c] = float(row[c])

    # Lag 1 — todas las variables base
    for c in ['NDVI', 'NDWI_agua', 'NDWI_veg', 'LST', 'SPI_3']:
        f[f'{c}_lag1'] = float(lag1[c])

    # Lag 2 — solo NDVI, NDWI_agua, SPI_3
    for c in ['NDVI', 'NDWI_agua', 'SPI_3']:
        f[f'{c}_lag2'] = float(lag2[c])

    # Delta (t − t-1) — solo NDVI, NDWI_agua, SPI_3
    for c in ['NDVI', 'NDWI_agua', 'SPI_3']:
        f[f'{c}_delta'] = float(row[c]) - float(lag1[c])

    # Estacionalidad cíclica
    f['mes_sin'] = float(np.sin(2 * np.pi * mes / 12))
    f['mes_cos'] = float(np.cos(2 * np.pi * mes / 12))

    # Garantiza el orden exacto del entrenamiento
    return pd.DataFrame([f])[FEATURES_V13]

# ==============================================================================
# 6. EXPLICACIÓN IA (Groq · LLaMA 3)
# ==============================================================================
def explicar_riesgo_ia(datos: dict, nivel: int, probabilidades: list) -> str:
    info     = RIESGO_INFO[nivel]
    prob_str = ", ".join([f"clase {i}: {p:.0%}" for i, p in enumerate(probabilidades)])
    prompt   = f"""
Eres un experto en hidrología y gestión de recursos hídricos en Honduras.

Datos satelitales del mes analizado (Subcuenca Represa El Coyolar):
- NDVI (vegetación):              {datos['NDVI']:.3f}
- NDWI agua (nivel del embalse):  {datos['NDWI_agua']:.3f}
- NDWI veg (estrés hídrico):      {datos['NDWI_veg']:.3f}
- LST temperatura superficial:    {datos['LST']:.1f} °C
- SPI-3 (índice de precipitación):{datos['SPI_3']:.2f}
- Lluvia acumulada del mes:        {datos['Lluvia_mm']:.1f} mm

Predicción del modelo V13 (RandomForestClassifier):
  Nivel de riesgo: {info['emoji']} {info['nombre']}
  Probabilidades por clase: {prob_str}

Redacta en español claro y accesible (máximo 200 palabras):
1. Situación hídrica actual de la subcuenca
2. Causas principales que explican el nivel de riesgo predicho
3. Recomendaciones concretas para operadores del embalse y comunidades
"""
    resp = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    return resp.choices[0].message.content

# ==============================================================================
# 7. MAPA FOLIUM — estilo imagen de referencia
# ==============================================================================
def construir_mapa(nivel: int, datos_row: dict) -> folium.Map:
    """
    Mapa con estilo de la imagen de referencia:
    · Base OpenStreetMap
    · Subcuenca: borde azul sólido #2196F3 + relleno semitransparente según riesgo
    · Ríos/embalse: azul claro #64B5F6, grosor dinámico por NDWI_agua
    · Tooltip interactivo con los 5 índices
    · Leyenda con barras de progreso en esquina inferior derecha
    """
    lat, lon  = 14.3335, -87.5090
    info      = RIESGO_INFO[nivel]

    FILL_COLOR = {0: "#64B5F6", 1: "#FFD54F", 2: "#FF8A65", 3: "#EF9A9A"}
    FILL_OPAC  = {0: 0.35,      1: 0.40,      2: 0.42,      3: 0.45}

    fill_color = FILL_COLOR[nivel]
    fill_opac  = FILL_OPAC[nivel]

    mapa = folium.Map(location=[lat, lon], zoom_start=12,
                      tiles="OpenStreetMap", prefer_canvas=True)

    # ── Subcuenca ─────────────────────────────────────────────────────────
    geom_cuenca = roi_subcuenca.getInfo()

    ndwi_a = float(datos_row.get("NDWI_agua", 0))
    ndvi   = float(datos_row.get("NDVI",      0))
    lst    = float(datos_row.get("LST",        0))
    spi    = float(datos_row.get("SPI_3",      0))
    lluvia = float(datos_row.get("Lluvia_mm",  0))
    ndwi_v = float(datos_row.get("NDWI_veg",   0))

    tooltip_html = f"""
    <div style="font-family:sans-serif;font-size:13px;min-width:210px">
        <b>Subcuenca El Coyolar</b>
        <hr style="margin:4px 0">
        <span style="background:{info['color']};color:white;padding:2px 8px;
                     border-radius:4px;font-weight:600;font-size:12px">
            {info['emoji']} {info['nombre']}
        </span><br><br>
        🌿 NDVI: <b>{ndvi:.3f}</b><br>
        💧 NDWI agua: <b>{ndwi_a:.3f}</b><br>
        🌱 NDWI veg: <b>{ndwi_v:.3f}</b><br>
        🌡️ LST: <b>{lst:.1f} °C</b><br>
        🌧️ SPI-3: <b>{spi:.2f}</b><br>
        ☁️ Lluvia: <b>{lluvia:.1f} mm</b>
    </div>"""

    folium.GeoJson(
        geom_cuenca,
        name="Subcuenca El Coyolar",
        style_function=lambda _: {
            "fillColor":   fill_color,
            "color":       "#2196F3",   # borde azul sólido
            "weight":      2.5,
            "fillOpacity": fill_opac,
        },
        highlight_function=lambda _: {
            "fillOpacity": min(fill_opac + 0.15, 0.75),
            "weight":      3.5,
            "color":       "#1565C0",
        },
        tooltip=folium.Tooltip(tooltip_html, sticky=True),
    ).add_to(mapa)

    # ── Ríos / embalse ─────────────────────────────────────────────────────
    try:
        geom_rio   = obtener_geometria_rio(roi_subcuenca)
        grosor_rio = max(1.5, min(5.5, 1.5 + ndwi_a * 14))
        folium.GeoJson(
            geom_rio,
            name="Embalse / Ríos",
            style_function=lambda _: {
                "color":       "#64B5F6",
                "weight":      grosor_rio,
                "fillColor":   "#90CAF9",
                "fillOpacity": 0.55,
            },
            tooltip=folium.Tooltip(
                f"<b>Embalse / Río principal</b><br>NDWI agua: {ndwi_a:.3f}",
                sticky=False
            ),
        ).add_to(mapa)
    except Exception:
        pass

    # ── Leyenda con barras de progreso ─────────────────────────────────────
    def _barra(label, valor, min_v, max_v, color, unidad=""):
        pct = max(0, min(100, int((valor - min_v) / (max_v - min_v) * 100)))
        return (f'<div style="margin-bottom:6px">'
                f'<div style="display:flex;justify-content:space-between;font-size:11px">'
                f'<span>{label}</span>'
                f'<span style="font-weight:600">{valor:.2f}{unidad}</span></div>'
                f'<div style="background:#e0e0e0;border-radius:3px;height:6px;margin-top:2px">'
                f'<div style="background:{color};width:{pct}%;height:6px;border-radius:3px">'
                f'</div></div></div>')

    leyenda_html = (
        f'<div style="position:fixed;bottom:28px;right:12px;z-index:9999;'
        f'background:rgba(255,255,255,0.97);padding:14px 16px;border-radius:10px;'
        f'border:1.5px solid #BBDEFB;font-family:sans-serif;min-width:215px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.15)">'
        f'<div style="font-size:13px;font-weight:700;margin-bottom:8px;color:#0D47A1">'
        f'🗺️ Subcuenca El Coyolar</div>'
        f'<div style="background:{info["color"]};color:white;padding:5px 10px;'
        f'border-radius:6px;font-size:13px;font-weight:600;margin-bottom:10px;text-align:center">'
        f'{info["emoji"]} {info["nombre"]}</div>'
        f'<div style="font-size:11px;color:#555;margin-bottom:8px;font-weight:600;'
        f'border-bottom:1px solid #e3f2fd;padding-bottom:4px">ÍNDICES SATELITALES</div>'
        + _barra("🌿 NDVI",       ndvi,   0.20, 0.80, '#43A047')
        + _barra("💧 NDWI agua",  ndwi_a, -0.10, 0.40, '#1E88E5')
        + _barra("🌱 NDWI veg",   ndwi_v, -0.30, 0.15, '#26A69A')
        + _barra("🌡️ LST",        lst,    24.0,  38.0, '#E53935', ' °C')
        + _barra("🌧️ SPI-3",
                 max(-2.5, min(2.5, spi)), -2.5, 2.5,
                 '#1565C0' if spi >= 0 else '#E53935')
        + '<div style="font-size:10px;color:#90A4AE;margin-top:8px;'
          'border-top:1px solid #e3f2fd;padding-top:6px">'
          '■ <span style="color:#2196F3">Límite subcuenca</span> &nbsp;'
          '■ <span style="color:#64B5F6">Ríos / embalse</span></div></div>'
    )

    mapa.get_root().html.add_child(folium.Element(leyenda_html))
    folium.LayerControl(collapsed=False).add_to(mapa)
    return mapa

# ==============================================================================
# 8. GRÁFICOS PLOTLY
# ==============================================================================
MESES_LABELS = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']

# Medianas históricas 2015-2025 (referencia visual en los gráficos)
HISTORICO = {
    'NDVI':      [0.52,0.48,0.44,0.46,0.62,0.70,0.65,0.68,0.72,0.71,0.65,0.58],
    'NDWI_agua': [0.12,0.08,0.04,0.02,0.15,0.28,0.24,0.26,0.32,0.30,0.22,0.16],
    'NDWI_veg':  [-0.05,-0.10,-0.16,-0.18,-0.04,0.08,0.04,0.06,0.10,0.09,0.02,-0.02],
    'LST':       [32.4,33.8,35.6,36.2,30.1,27.4,28.8,28.2,26.8,27.0,29.5,31.2],
    'SPI_3':     [-0.6,-0.9,-1.2,-1.5,-0.3,0.8,0.4,0.6,1.1,0.9,0.3,-0.3],
}

COLORES_INDICES = {
    'NDVI':      '#2a78d6',
    'NDWI_agua': '#1baf7a',
    'NDWI_veg':  '#4a3aa7',
    'LST':       '#e34948',
    'SPI_3':     '#eda100',
}
TITULOS_INDICES = {
    'NDVI':      'NDVI — Vegetación',
    'NDWI_agua': 'NDWI agua — Nivel del embalse',
    'NDWI_veg':  'NDWI veg — Estrés hídrico foliar',
    'LST':       'LST — Temperatura superficial (°C)',
    'SPI_3':     'SPI-3 — Índice estandarizado de precipitación',
}

def grafico_indice(key: str, valor_actual: float, mes_idx: int) -> go.Figure:
    hist   = HISTORICO[key]
    color  = COLORES_INDICES[key]
    barras = [color if i == mes_idx else 'rgba(180,178,169,0.55)' for i in range(12)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=MESES_LABELS, y=hist,
        marker_color=barras,
        name='Histórico 2015-2025',
        hovertemplate='%{x}: %{y:.3f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[MESES_LABELS[mes_idx]], y=[valor_actual],
        mode='markers',
        marker=dict(color=color, size=13, symbol='diamond',
                    line=dict(color='white', width=2)),
        name='Valor GEE (mes analizado)',
        hovertemplate=f'GEE: {valor_actual:.3f}<extra></extra>',
    ))

    if key == 'SPI_3':
        for y_val, label, lcolor in [
            (-1.0, 'Umbral sequía moderada', '#e67e22'),
            (-1.5, 'Umbral sequía severa',   '#e74c3c'),
        ]:
            fig.add_hline(y=y_val, line_dash='dot', line_color=lcolor,
                          annotation_text=label,
                          annotation_position='bottom right',
                          annotation_font_size=10)

    fig.update_layout(
        title=dict(text=TITULOS_INDICES[key], font_size=13, x=0),
        height=230,
        margin=dict(l=40, r=10, t=36, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font_size=10, orientation='h', yanchor='bottom',
                    y=1.02, xanchor='right', x=1),
        xaxis=dict(showgrid=False, tickfont_size=10),
        yaxis=dict(gridcolor='#e1e0d9', tickfont_size=10),
        bargap=0.25,
    )
    return fig

def grafico_probabilidades(probabilidades: list) -> go.Figure:
    labels  = [f"{RIESGO_INFO[i]['emoji']} {RIESGO_INFO[i]['nombre']}" for i in range(4)]
    colors  = [RIESGO_INFO[i]['color'] for i in range(4)]
    valores = [round(p * 100, 1) for p in probabilidades]

    fig = go.Figure(go.Bar(
        x=labels, y=valores,
        marker_color=colors,
        text=[f"{v}%" for v in valores],
        textposition='outside',
        hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Probabilidad por nivel de riesgo (%)', font_size=13, x=0),
        height=250,
        yaxis=dict(range=[0, 115], showgrid=False, tickfont_size=10),
        xaxis=dict(tickfont_size=11),
        margin=dict(l=20, r=10, t=36, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    return fig

# ==============================================================================
# 9. BARRA LATERAL — CONTROLES
# ==============================================================================
st.sidebar.header("⚙️ Parámetros de análisis")

MESES_NOMBRES = {
    1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
    7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
}

# Rango permitido: enero 2019 → mes siguiente al actual
_hoy      = pd.Timestamp.now()
_mes_max  = (_hoy + pd.DateOffset(months=1)).month
_anio_max = (_hoy + pd.DateOffset(months=1)).year

_opciones_fecha = []
for _a in range(2019, _anio_max + 1):
    for _m in range(1, 13):
        if _a == _anio_max and _m > _mes_max:
            break
        _opciones_fecha.append((_a, _m))

def _label_fecha(t):
    return f"{MESES_NOMBRES[t[1]]} {t[0]}"

_default_idx = next(
    (i for i, t in enumerate(_opciones_fecha) if t == (_hoy.year, _hoy.month)),
    len(_opciones_fecha) - 1
)

fecha_sel = st.sidebar.selectbox(
    "Mes y año a analizar",
    options=_opciones_fecha,
    format_func=_label_fecha,
    index=_default_idx,
)
anio_sel, mes_sel = fecha_sel

usar_datos_reales = st.sidebar.checkbox(
    "Obtener datos reales desde GEE", value=True,
    help="Descarga índices satelitales desde Google Earth Engine. "
         "Desactiva para usar datos de demostración (más rápido)."
)

ejecutar = st.sidebar.button("🔄 Ejecutar análisis", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Modelo V13** · RandomForestClassifier  \n"
    "18 features: base ×5, lag1 ×5, lag2/delta ×3 c/u, estacionalidad  \n"
    f"Rango disponible: Enero 2019 – {_label_fecha(_opciones_fecha[-1])}"
)

# ==============================================================================
# 10. DATOS DE DEMOSTRACIÓN
# ==============================================================================
DEMO_DATOS = {
    m: {
        "NDVI":      HISTORICO["NDVI"][m-1],
        "NDWI_agua": HISTORICO["NDWI_agua"][m-1],
        "NDWI_veg":  HISTORICO["NDWI_veg"][m-1],
        "LST":       HISTORICO["LST"][m-1],
        "Lluvia_mm": [55,38,25,30,85,125,110,120,145,135,90,70][m-1],
        "SPI_3":     HISTORICO["SPI_3"][m-1],
    }
    for m in range(1, 13)
}

# ==============================================================================
# 11. LÓGICA PRINCIPAL
# ==============================================================================
if ejecutar:
    with st.spinner(f"Analizando {MESES_NOMBRES[mes_sel]} {anio_sel}…"):

        # ── Datos satelitales ─────────────────────────────────────────────
        try:
            if usar_datos_reales:
                df_raw = obtener_datos_satelitales(anio_sel, mes_sel)
                df_raw["SPI_3"] = calcular_spi3_simple(
                    float(df_raw["Lluvia_mm"].iloc[0]), mes_sel
                )
                st.sidebar.success("✅ Datos GEE obtenidos")
            else:
                demo   = DEMO_DATOS[mes_sel]
                df_raw = pd.DataFrame([{**demo, "Anio": anio_sel, "Mes": mes_sel}])
                st.sidebar.info("ℹ️ Usando datos de demostración")
        except Exception as e:
            st.error(f"Error al obtener datos satelitales: {e}")
            st.stop()

        # ── Feature engineering (18 features V13) ─────────────────────────
        # Para usar historial real, carga los meses anteriores desde tu BD
        # y pásalos como df_historial.
        X = construir_features(df_raw, df_historial=None)

        # ── Predicción: NRHF (Nivel de Riesgo Hídrico Futuro) ─────────────
        pred  = int(modelo.predict(X)[0])
        probs = modelo.predict_proba(X)[0].tolist()
        info  = RIESGO_INFO[pred]

        datos_row = df_raw.iloc[0].to_dict()

    # ── Semáforo de riesgo ─────────────────────────────────────────────────
    st.markdown(
        f"<p class='section-title'>🚦 Riesgo hídrico predicho — "
        f"{MESES_NOMBRES[mes_sel]} {anio_sel}</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="badge-riesgo" style="background:{info["color"]}">'
        f'{info["emoji"]} {info["nombre"]} &nbsp;·&nbsp; '
        f'Probabilidad: {round(probs[pred]*100)}%</div>',
        unsafe_allow_html=True
    )

    # ── Métricas ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("NDVI",       f"{datos_row['NDVI']:.3f}")
    c2.metric("NDWI agua",  f"{datos_row['NDWI_agua']:.3f}")
    c3.metric("NDWI veg",   f"{datos_row['NDWI_veg']:.3f}")
    c4.metric("LST",        f"{datos_row['LST']:.1f} °C")
    c5.metric("SPI-3",      f"{datos_row['SPI_3']:.2f}")

    # ── Interpretaciones ──────────────────────────────────────────────────
    cols_i = st.columns(5)
    for col, key in zip(cols_i, ["NDVI","NDWI_agua","NDWI_veg","LST","SPI_3"]):
        bg, txt, rango, desc = obtener_interpretacion(key, datos_row[key])
        col.markdown(
            f"<span class='interp-tag' style='background:{bg};color:{txt}'>{rango}</span>"
            f"<span style='font-size:11px;color:#52514e'>{desc}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── IA + Probabilidades ────────────────────────────────────────────────
    col_ia, col_prob = st.columns([3, 2])

    with col_ia:
        st.markdown("<p class='section-title'>🧠 Explicación IA (Groq · LLaMA 3)</p>",
                    unsafe_allow_html=True)
        with st.spinner("Generando análisis…"):
            try:
                explicacion = explicar_riesgo_ia(datos_row, pred, probs)
                st.info(explicacion)
            except Exception as e:
                st.warning(f"No se pudo generar la explicación IA: {e}")

    with col_prob:
        st.markdown("<p class='section-title'>📊 Distribución de probabilidades</p>",
                    unsafe_allow_html=True)
        st.plotly_chart(grafico_probabilidades(probs), use_container_width=True)

    st.markdown("---")

    # ── Mapa Folium ────────────────────────────────────────────────────────
    st.markdown("<p class='section-title'>🗺️ Mapa de riesgo — Subcuenca El Coyolar</p>",
                unsafe_allow_html=True)
    with st.spinner("Cargando mapa…"):
        try:
            mapa = construir_mapa(pred, datos_row)
            st_folium(mapa, width=None, height=480, returned_objects=[])
        except Exception as e:
            st.warning(f"No se pudo cargar el mapa: {e}")

    st.markdown("---")

    # ── Gráficos de índices ────────────────────────────────────────────────
    st.markdown(
        "<p class='section-title'>📈 Índices satelitales — histórico mensual vs. valor actual</p>",
        unsafe_allow_html=True
    )
    mi = mes_sel - 1  # índice 0-based para HISTORICO

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(grafico_indice("NDVI",      datos_row["NDVI"],      mi), use_container_width=True)
    with r1c2:
        st.plotly_chart(grafico_indice("NDWI_agua", datos_row["NDWI_agua"], mi), use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(grafico_indice("NDWI_veg",  datos_row["NDWI_veg"],  mi), use_container_width=True)
    with r2c2:
        st.plotly_chart(grafico_indice("LST",        datos_row["LST"],       mi), use_container_width=True)

    st.plotly_chart(grafico_indice("SPI_3", datos_row["SPI_3"], mi), use_container_width=True)

    # ── Tablas de detalle ──────────────────────────────────────────────────
    with st.expander("🔎 Ver datos satelitales crudos del mes"):
        st.dataframe(df_raw.style.format({
            "NDVI": "{:.4f}", "NDWI_agua": "{:.4f}", "NDWI_veg": "{:.4f}",
            "LST":  "{:.2f}", "Lluvia_mm": "{:.1f}",  "SPI_3":    "{:.3f}",
        }), use_container_width=True)

    with st.expander("🔎 Ver vector de features enviado al modelo (18 features V13)"):
        st.dataframe(
            X.T.rename(columns={0: "valor"}).style.format("{:.5f}"),
            use_container_width=True
        )

else:
    # ── Pantalla de bienvenida ─────────────────────────────────────────────
    st.info("👈 Selecciona el mes y el año en el panel lateral, luego presiona **Ejecutar análisis**.")
    st.markdown("### ¿Qué analiza este sistema?")
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown(
        "🛰️ **Datos satelitales**  \n"
        "Sentinel-2 (NDVI, NDWI), Landsat 8 (LST) y CHIRPS (lluvia / SPI-3) "
        "vía Google Earth Engine."
    )
    col_b.markdown(
        "🤖 **Modelo V13**  \n"
        "RandomForestClassifier · 18 features · target: NRHF  \n"
        "Predice el nivel de riesgo hídrico del **próximo mes** (clases 0-3)."
    )
    col_c.markdown(
        "🧠 **IA explicativa**  \n"
        "Groq + LLaMA 3 genera un análisis en lenguaje natural con causas y "
        "recomendaciones para operadores del embalse y comunidades."
    )

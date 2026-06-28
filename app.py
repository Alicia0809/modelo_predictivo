# -*- coding: utf-8 -*-
"""
app.py — Sistema de Predicción de Riesgo Hídrico
Subcuenca Represa El Coyolar · Modelo V13 · GEE + Sentinel-2 + Landsat 8 + CHIRPS

Modelo final : RandomForestClassifier (n_estimators=200, max_depth=6, random_state=42)
Features (18): NDVI, NDWI_agua, NDWI_veg, LST, SPI_3 (base ×5)
               + lag1 de las 5 bases (×5)
               + lag2 y delta solo de NDVI, NDWI_agua, SPI_3 (×3 c/u)
               + mes_sin, mes_cos
Target       : NRHF — Nivel de Riesgo Hídrico Futuro (próximo mes, clases 0-3)
Archivo pkl  : modelo_sequia_futuro.pkl
CSV historial: variables.csv  (columnas: Anio, Mes, NDVI, NDWI_agua, NDWI_veg,
                                          LST, Lluvia_mm, SPI_3, NRHF, ...)
ROI          : HydroATLAS nivel 12 — punto (-87.50904715160283, 14.333509533380646)

MODOS DE PREDICCIÓN
───────────────────
  "historico"    → Ene 2019 – Dic 2025
                   Usa variables.csv (datos ya extraídos durante el entrenamiento).
                   No llama a GEE.

  "gee_futuro"   → Ene 2026 en adelante (excepto el mes siguiente al actual)
                   Extrae índices del mes solicitado desde GEE y predice.

  "gee_siguiente"→ El mes inmediatamente siguiente al mes actual.
                   Extrae índices del mes ACTUAL desde GEE y predice el riesgo
                   del mes SIGUIENTE (comportamiento natural del modelo).
"""

from datetime import datetime
import tempfile
import json

import streamlit as st
import pandas as pd
import numpy as np
import ee
import joblib
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from groq import Groq

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
    .modo-badge {
        display: inline-block; padding: 4px 12px; border-radius: 6px;
        font-size: 12px; font-weight: 600; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("💧 Sistema de Predicción de Riesgo Hídrico")
st.caption("Subcuenca Represa El Coyolar · GEE + Sentinel-2 + Landsat 8 + CHIRPS · Modelo V13")

# ==============================================================================
# 1. DETERMINAR MODO SEGÚN EL MES/AÑO SOLICITADO
# ==============================================================================
LIMITE_HISTORICO_ANIO = 2025
LIMITE_HISTORICO_MES  = 12   # Dic 2025 es el último mes con datos en variables.csv

def determinar_modo(anio: int, mes: int) -> str:
    """
    Devuelve el modo de predicción según el mes/año solicitado:

    "historico"     → Ene 2019 – Dic 2025
                      Se leen los datos ya extraídos de variables.csv.

    "gee_siguiente" → El mes inmediatamente siguiente al mes actual.
                      Se extraen índices del mes ACTUAL desde GEE y el modelo
                      predice el riesgo del mes SIGUIENTE.

    "gee_futuro"    → Cualquier mes posterior a Dic 2025 que NO sea el mes
                      siguiente al actual.
                      Se extraen índices del mes SOLICITADO desde GEE.
    """
    hoy              = datetime.today()
    fecha_solicitada = anio * 12 + mes
    fecha_actual     = hoy.year * 12 + hoy.month
    fecha_limite     = LIMITE_HISTORICO_ANIO * 12 + LIMITE_HISTORICO_MES

    if fecha_solicitada <= fecha_limite:
        return "historico"
    elif fecha_solicitada == fecha_actual + 1:
        return "gee_siguiente"
    else:
        return "gee_futuro"

# Etiquetas y colores para mostrar el modo en la UI
MODO_UI = {
    "historico": {
        "label":   "📁 Datos históricos (variables.csv)",
        "color":   "#e3f2fd",
        "txt":     "#0D47A1",
        "detalle": "Los índices provienen del CSV generado durante el entrenamiento.",
    },
    "gee_siguiente": {
        "label":   "🛰️ Predicción mes siguiente (GEE)",
        "color":   "#e8f5e9",
        "txt":     "#1B5E20",
        "detalle": "Se extraen los índices del mes ACTUAL desde GEE. "
                   "El modelo predice el riesgo del mes SIGUIENTE.",
    },
    "gee_futuro": {
        "label":   "🔭 Predicción futura (GEE)",
        "color":   "#fff3e0",
        "txt":     "#E65100",
        "detalle": "Se extraen los índices del mes solicitado desde GEE. "
                   "El modelo predice el riesgo del mes siguiente a ese.",
    },
}

# ==============================================================================
# 2. CLIENTES Y RECURSOS
# ==============================================================================
GROQ_API_KEY = st.secrets["DB_TOKEN"]
groq_client  = Groq(api_key=GROQ_API_KEY)

def _obtener_subcuenca_gee(lon: float, lat: float, nivel: int = 12):
    """Devuelve la geometría HydroATLAS que contiene el punto dado."""
    nivel_str = str(nivel).zfill(2)
    cuencas   = ee.FeatureCollection(f'WWF/HydroATLAS/v1/Basins/level{nivel_str}')
    punto     = ee.Geometry.Point([lon, lat])
    return cuencas.filterBounds(punto).first().geometry()

@st.cache_resource(show_spinner="Inicializando GEE, modelo y datos históricos…")
def cargar_recursos():
    """
    Inicializa GEE, carga el modelo y el CSV histórico.
    Se ejecuta una sola vez gracias a @st.cache_resource.
    """
    # GEE
    credentials_json = json.loads(st.secrets["GEE_CREDENTIALS"])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(credentials_json, f)
        temp_file = f.name
    creds = ee.ServiceAccountCredentials(credentials_json["client_email"], temp_file)
    ee.Initialize(creds)

    # Modelo entrenado
    modelo = joblib.load("modelo_sequia_futuro.pkl")

    # ROI — mismo punto que en el entrenamiento V13
    roi = _obtener_subcuenca_gee(-87.50904715160283, 14.333509533380646, nivel=12)

    # CSV histórico — columnas mínimas requeridas:
    # Anio, Mes, NDVI, NDWI_agua, NDWI_veg, LST, Lluvia_mm, SPI_3, NRHF
    #
    # El archivo usa coma como separador decimal (formato regional español,
    # ej: "0,7208314795"). Se lee con decimal=',' para que pandas convierta
    # directamente a float64 sin pasos intermedios.
    historical_df = pd.read_csv("variables.csv", decimal=",")

    # Garantizar tipos correctos
    historical_df["Anio"] = historical_df["Anio"].astype(int)
    historical_df["Mes"]  = historical_df["Mes"].astype(int)

    _COLS_NUMERICAS = ["NDVI", "NDWI_agua", "NDWI_veg", "LST", "Lluvia_mm", "SPI_3"]
    for _c in _COLS_NUMERICAS:
        if _c in historical_df.columns:
            historical_df[_c] = pd.to_numeric(historical_df[_c], errors="coerce")

    return modelo, roi, historical_df

modelo, roi_subcuenca, historical_df = cargar_recursos()
st.sidebar.success("✅ Modelo (RandomForest V13) + GEE listos")

# ==============================================================================
# 3. CONFIGURACIÓN DE RIESGO E INTERPRETACIONES
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
# 4. EXTRACCIÓN GEE
# ==============================================================================
def remover_nubes_sentinel(img):
    scl  = img.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask).clip(roi_subcuenca)

def remover_nubes_landsat(img):
    qa   = img.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return img.updateMask(mask)

def obtener_datos_satelitales_gee(anio: int, mes: int) -> pd.DataFrame:
    """
    Extrae NDVI, NDWI_agua, NDWI_veg, LST y Lluvia_mm desde GEE
    para el mes/año dado. Lógica idéntica al entrenamiento V13.
    """
    inicio = ee.Date.fromYMD(anio, mes, 1)
    fin    = inicio.advance(1, 'month')

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(roi_subcuenca)
          .filterDate(inicio, fin)
          .map(remover_nubes_sentinel)
          .median())

    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')

    gsw          = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    mascara_agua = gsw.select('occurrence').gt(10)
    mascara_veg  = mascara_agua.Not()

    ndwi_base = s2.normalizedDifference(['B3', 'B8'])
    ndwi_agua = ndwi_base.updateMask(mascara_agua).rename('NDWI_agua')
    ndwi_veg  = ndwi_base.updateMask(mascara_veg).rename('NDWI_veg')

    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterBounds(roi_subcuenca)
          .filterDate(inicio, fin)
          .map(remover_nubes_landsat)
          .median())
    lst = l8.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')

    lluvia_stats = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                    .filterBounds(roi_subcuenca)
                    .filterDate(inicio, fin)
                    .sum()
                    .reduceRegion(reducer=ee.Reducer.mean(),
                                  geometry=roi_subcuenca, scale=5000, maxPixels=1e9))
    lluvia_val = lluvia_stats.get('precipitation').getInfo() or 0.0

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
        "Lluvia_mm": lluvia_val,
    }])

def obtener_geometria_rio(roi) -> dict:
    gsw  = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    agua = gsw.select('occurrence').gt(50).selfMask()
    vecs = agua.reduceToVectors(
        geometry=roi, scale=30, geometryType='polygon',
        eightConnected=True, maxPixels=1e9
    ).map(lambda f: f.set('area_m2', f.geometry().area(1)))
    return ee.Feature(vecs.sort('area_m2', False).first()).geometry().getInfo()

# ==============================================================================
# 5. SPI-3
# ==============================================================================
SPI_MEDIA_MES = {1:55,  2:38,  3:25,  4:30,  5:85,  6:125,
                 7:110, 8:120, 9:145, 10:135, 11:90, 12:70}
SPI_STD_MES   = {1:32,  2:28,  3:20,  4:24,  5:40,  6:45,
                 7:42,  8:43,  9:48,  10:46,  11:38, 12:36}

def calcular_spi3(lluvia_mm: float, mes: int) -> float:
    mu  = SPI_MEDIA_MES.get(mes, 85.4)
    std = SPI_STD_MES.get(mes, 42.1)
    return float((lluvia_mm - mu) / std) if std > 0 else 0.0

# ==============================================================================
# 6. FEATURE ENGINEERING — 18 features, orden exacto del entrenamiento V13
# ==============================================================================
FEATURES_V13 = [
    'NDVI', 'NDWI_agua', 'NDWI_veg', 'LST', 'SPI_3',
    'NDVI_lag1', 'NDWI_agua_lag1', 'NDWI_veg_lag1', 'LST_lag1', 'SPI_3_lag1',
    'NDVI_lag2', 'NDWI_agua_lag2', 'SPI_3_lag2',
    'NDVI_delta', 'NDWI_agua_delta', 'SPI_3_delta',
    'mes_sin', 'mes_cos',
]

def _fila_a_features(row, lag1, lag2) -> pd.DataFrame:
    """
    Construye el vector de 18 features dado el mes actual (row)
    y los dos meses anteriores (lag1, lag2) como Series o dict-like.
    """
    mes = int(row['Mes'])
    f   = {}
    for c in ['NDVI', 'NDWI_agua', 'NDWI_veg', 'LST', 'SPI_3']:
        f[c]            = float(row[c])
        f[f'{c}_lag1']  = float(lag1[c])
    for c in ['NDVI', 'NDWI_agua', 'SPI_3']:
        f[f'{c}_lag2']  = float(lag2[c])
        f[f'{c}_delta'] = float(row[c]) - float(lag1[c])
    f['mes_sin'] = float(np.sin(2 * np.pi * mes / 12))
    f['mes_cos'] = float(np.cos(2 * np.pi * mes / 12))
    return pd.DataFrame([f])[FEATURES_V13]

def _mes_anterior(anio: int, mes: int):
    """Devuelve (anio, mes) del mes anterior."""
    if mes == 1:
        return anio - 1, 12
    return anio, mes - 1

def _buscar_en_csv(anio: int, mes: int) -> pd.Series:
    """
    Busca una fila en el CSV histórico para el mes/año exacto.
    Si no existe (mes faltante por cobertura nubosa u otro motivo),
    busca el mes disponible más cercano cronológicamente.
    Lanza ValueError solo si el CSV está completamente vacío.
    """
    fila = historical_df[(historical_df["Anio"] == anio) &
                         (historical_df["Mes"]  == mes)]
    if not fila.empty:
        return fila.iloc[0]

    # Mes faltante: buscar el registro más cercano en tiempo
    fecha_objetivo = anio * 12 + mes
    historical_df["_dist"] = abs(
        historical_df["Anio"] * 12 + historical_df["Mes"] - fecha_objetivo
    )
    mas_cercano = historical_df.sort_values("_dist").iloc[0]
    historical_df.drop(columns=["_dist"], inplace=True)

    a_cercano = int(mas_cercano["Anio"])
    m_cercano = int(mas_cercano["Mes"])
    st.warning(
        f"⚠️ No hay datos en variables.csv para {mes}/{anio}. "
        f"Se usará el mes más cercano disponible: {m_cercano}/{a_cercano}.",
        icon="ℹ️"
    )
    return mas_cercano

# ==============================================================================
# 7. TRES FUNCIONES DE PREDICCIÓN — una por modo
# ==============================================================================

def predecir_historico(anio: int, mes: int) -> tuple:
    """
    MODO HISTÓRICO (Ene 2019 – Dic 2025)
    Obtiene la fila del mes solicitado y sus dos anteriores desde variables.csv.
    No llama a GEE.

    Retorna: (pred, probs, datos_row_dict)
        datos_row_dict → índices del mes solicitado (para mostrar en UI)
    """
    fila  = _buscar_en_csv(anio, mes)
    a1, m1 = _mes_anterior(anio, mes)
    a2, m2 = _mes_anterior(a1, m1)

    lag1 = _buscar_en_csv(a1, m1)
    lag2 = _buscar_en_csv(a2, m2)

    X     = _fila_a_features(fila, lag1, lag2)
    pred  = int(modelo.predict(X)[0])
    probs = modelo.predict_proba(X)[0].tolist()

    datos_row = fila.to_dict()
    # Asegura que SPI_3 esté presente (puede ya estar en el CSV)
    if "SPI_3" not in datos_row or pd.isna(datos_row.get("SPI_3")):
        datos_row["SPI_3"] = calcular_spi3(float(datos_row.get("Lluvia_mm", 0)), mes)

    return pred, probs, datos_row


def predecir_mes_siguiente(anio_siguiente: int, mes_siguiente: int) -> tuple:
    """
    MODO MES SIGUIENTE
    El usuario pidió el mes inmediatamente siguiente al actual.

    Extrae desde GEE:
      · El mes ACTUAL            → fila actual (t)   ← entrada real del modelo
      · El mes anterior          → lag1        (t-1)
      · El mes anterior al anterior → lag2     (t-2)

    Para lag1 y lag2 intenta primero el CSV (están dentro del rango histórico);
    si por alguna razón no existieran, los descarga desde GEE.

    El modelo predice el riesgo del mes SIGUIENTE (comportamiento natural del target NRHF).

    Retorna: (pred, probs, datos_row_dict)
        datos_row_dict → índices del mes ACTUAL (la entrada real del modelo)
    """
    hoy = datetime.today()
    anio_actual, mes_actual = hoy.year, hoy.month

    # Descarga el mes actual desde GEE (t)
    df_actual = obtener_datos_satelitales_gee(anio_actual, mes_actual)
    df_actual["SPI_3"] = calcular_spi3(float(df_actual["Lluvia_mm"].iloc[0]), mes_actual)
    fila_actual = df_actual.iloc[0]

    # lag1 y lag2: CSV primero, GEE como fallback
    a1, m1 = _mes_anterior(anio_actual, mes_actual)
    a2, m2 = _mes_anterior(a1, m1)

    def _obtener_lag(a: int, m: int) -> pd.Series:
        try:
            return _buscar_en_csv(a, m)
        except ValueError:
            df_tmp = obtener_datos_satelitales_gee(a, m)
            df_tmp["SPI_3"] = calcular_spi3(float(df_tmp["Lluvia_mm"].iloc[0]), m)
            return df_tmp.iloc[0]

    lag1 = _obtener_lag(a1, m1)
    lag2 = _obtener_lag(a2, m2)

    X     = _fila_a_features(fila_actual, lag1, lag2)
    pred  = int(modelo.predict(X)[0])
    probs = modelo.predict_proba(X)[0].tolist()

    datos_row        = fila_actual.to_dict()
    datos_row["Mes"] = mes_actual   # los índices mostrados son del mes actual

    return pred, probs, datos_row


def predecir_gee_futuro(anio: int, mes: int) -> tuple:
    """
    MODO GEE FUTURO (Ene 2026 en adelante, excepto el mes siguiente al actual)

    Extrae desde GEE:
      · El mes SOLICITADO        → fila actual (t)
      · El mes anterior          → lag1        (t-1)
      · El mes anterior al anterior → lag2     (t-2)

    Para cada mes intenta primero el CSV (más rápido); si no existe en el CSV
    (porque es posterior a Dic 2025) lo descarga desde GEE.

    El modelo predice el riesgo del mes SIGUIENTE al solicitado.

    Retorna: (pred, probs, datos_row_dict)
        datos_row_dict → índices del mes solicitado (entrada real del modelo)
    """
    def _obtener_fila(a: int, m: int) -> pd.Series:
        """Devuelve una fila con los 5 índices base + SPI_3, priorizando el CSV."""
        try:
            return _buscar_en_csv(a, m)
        except ValueError:
            df_tmp = obtener_datos_satelitales_gee(a, m)
            df_tmp["SPI_3"] = calcular_spi3(float(df_tmp["Lluvia_mm"].iloc[0]), m)
            return df_tmp.iloc[0]

    a1, m1 = _mes_anterior(anio, mes)
    a2, m2 = _mes_anterior(a1, m1)

    fila_actual = _obtener_fila(anio, mes)
    lag1        = _obtener_fila(a1, m1)
    lag2        = _obtener_fila(a2, m2)

    X     = _fila_a_features(fila_actual, lag1, lag2)
    pred  = int(modelo.predict(X)[0])
    probs = modelo.predict_proba(X)[0].tolist()

    return pred, probs, fila_actual.to_dict()

# ==============================================================================
# 8. EXPLICACIÓN IA
# ==============================================================================
def explicar_riesgo_ia(datos: dict, nivel: int, probs: list, modo: str) -> str:
    info     = RIESGO_INFO[nivel]
    prob_str = ", ".join([f"clase {i}: {p:.0%}" for i, p in enumerate(probs)])
    contexto_modo = {
        "historico":     "Los índices corresponden a datos históricos ya registrados (variables.csv).",
        "gee_siguiente": "Los índices son del mes ACTUAL extraídos de GEE; el modelo predice el riesgo del MES SIGUIENTE.",
        "gee_futuro":    "Los índices son del mes solicitado extraídos de GEE; el modelo predice el riesgo del MES SIGUIENTE a ese.",
    }.get(modo, "")

    prompt = f"""
Eres un experto en hidrología y gestión de recursos hídricos en Honduras.

Contexto: {contexto_modo}

Índices satelitales (Subcuenca Represa El Coyolar):
- NDVI (vegetación):             {datos.get('NDVI', 0):.3f}
- NDWI agua (nivel del embalse): {datos.get('NDWI_agua', 0):.3f}
- NDWI veg (estrés hídrico):     {datos.get('NDWI_veg', 0):.3f}
- LST temperatura superficial:   {datos.get('LST', 0):.1f} °C
- SPI-3 (precipitación):         {datos.get('SPI_3', 0):.2f}
- Lluvia acumulada:               {datos.get('Lluvia_mm', 0):.1f} mm

Predicción del modelo V13 (RandomForestClassifier):
  Nivel de riesgo: {info['emoji']} {info['nombre']}
  Probabilidades: {prob_str}

Redacta en español claro (máximo 200 palabras):
1. Situación hídrica de la subcuenca
2. Causas del nivel de riesgo predicho
3. Recomendaciones para operadores del embalse y comunidades
"""
    resp = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    return resp.choices[0].message.content

# ==============================================================================
# 9. MAPA FOLIUM
# ==============================================================================
def construir_mapa(nivel: int, datos_row: dict) -> folium.Map:
    lat, lon  = 14.3335, -87.5090
    info      = RIESGO_INFO[nivel]
    FILL_COLOR = {0: "#64B5F6", 1: "#FFD54F", 2: "#FF8A65", 3: "#EF9A9A"}
    FILL_OPAC  = {0: 0.35,      1: 0.40,      2: 0.42,      3: 0.45}

    mapa = folium.Map(location=[lat, lon], zoom_start=12,
                      tiles="OpenStreetMap", prefer_canvas=True)

    ndwi_a = float(datos_row.get("NDWI_agua", 0))
    ndvi   = float(datos_row.get("NDVI",      0))
    lst    = float(datos_row.get("LST",        0))
    spi    = float(datos_row.get("SPI_3",      0))
    lluvia = float(datos_row.get("Lluvia_mm",  0))
    ndwi_v = float(datos_row.get("NDWI_veg",   0))

    fill_color = FILL_COLOR[nivel]
    fill_opac  = FILL_OPAC[nivel]

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
        roi_subcuenca.getInfo(),
        name="Subcuenca El Coyolar",
        style_function=lambda _: {
            "fillColor": fill_color, "color": "#2196F3",
            "weight": 2.5, "fillOpacity": fill_opac,
        },
        highlight_function=lambda _: {
            "fillOpacity": min(fill_opac + 0.15, 0.75),
            "weight": 3.5, "color": "#1565C0",
        },
        tooltip=folium.Tooltip(tooltip_html, sticky=True),
    ).add_to(mapa)

    try:
        geom_rio   = obtener_geometria_rio(roi_subcuenca)
        grosor_rio = max(1.5, min(5.5, 1.5 + ndwi_a * 14))
        folium.GeoJson(
            geom_rio, name="Embalse / Ríos",
            style_function=lambda _: {
                "color": "#64B5F6", "weight": grosor_rio,
                "fillColor": "#90CAF9", "fillOpacity": 0.55,
            },
            tooltip=folium.Tooltip(
                f"<b>Embalse / Río principal</b><br>NDWI agua: {ndwi_a:.3f}",
                sticky=False),
        ).add_to(mapa)
    except Exception:
        pass

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
        + _barra("🌿 NDVI",      ndvi,   0.20, 0.80, '#43A047')
        + _barra("💧 NDWI agua", ndwi_a, -0.10, 0.40, '#1E88E5')
        + _barra("🌱 NDWI veg",  ndwi_v, -0.30, 0.15, '#26A69A')
        + _barra("🌡️ LST",       lst,    24.0,  38.0, '#E53935', ' °C')
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
# 10. GRÁFICOS PLOTLY
# ==============================================================================
MESES_LABELS = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']

HISTORICO = {
    'NDVI':      [0.52,0.48,0.44,0.46,0.62,0.70,0.65,0.68,0.72,0.71,0.65,0.58],
    'NDWI_agua': [0.12,0.08,0.04,0.02,0.15,0.28,0.24,0.26,0.32,0.30,0.22,0.16],
    'NDWI_veg':  [-0.05,-0.10,-0.16,-0.18,-0.04,0.08,0.04,0.06,0.10,0.09,0.02,-0.02],
    'LST':       [32.4,33.8,35.6,36.2,30.1,27.4,28.8,28.2,26.8,27.0,29.5,31.2],
    'SPI_3':     [-0.6,-0.9,-1.2,-1.5,-0.3,0.8,0.4,0.6,1.1,0.9,0.3,-0.3],
}
COLORES_INDICES = {
    'NDVI': '#2a78d6', 'NDWI_agua': '#1baf7a',
    'NDWI_veg': '#4a3aa7', 'LST': '#e34948', 'SPI_3': '#eda100',
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
        x=MESES_LABELS, y=hist, marker_color=barras,
        name='Histórico 2015-2025',
        hovertemplate='%{x}: %{y:.3f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[MESES_LABELS[mes_idx]], y=[valor_actual],
        mode='markers',
        marker=dict(color=color, size=13, symbol='diamond',
                    line=dict(color='white', width=2)),
        name='Valor del mes analizado',
        hovertemplate=f'Valor: {valor_actual:.3f}<extra></extra>',
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
        height=230, margin=dict(l=40, r=10, t=36, b=30),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
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
        x=labels, y=valores, marker_color=colors,
        text=[f"{v}%" for v in valores], textposition='outside',
        hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Probabilidad por nivel de riesgo (%)', font_size=13, x=0),
        height=250, yaxis=dict(range=[0, 115], showgrid=False, tickfont_size=10),
        xaxis=dict(tickfont_size=11), margin=dict(l=20, r=10, t=36, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    return fig

# ==============================================================================
# 11. BARRA LATERAL — CONTROLES
# ==============================================================================
st.sidebar.header("⚙️ Parámetros de análisis")

MESES_NOMBRES = {
    1:"Enero", 2:"Febrero", 3:"Marzo",    4:"Abril",
    5:"Mayo",  6:"Junio",   7:"Julio",    8:"Agosto",
    9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"
}

_hoy      = pd.Timestamp.now()
_fecha_max = _hoy + pd.DateOffset(months=1)   # límite: mes siguiente al actual
_anio_max  = int(_fecha_max.year)
_mes_max   = int(_fecha_max.month)

# ── Selector de AÑO ───────────────────────────────────────────────────────────
anio_sel = st.sidebar.selectbox(
    "Año",
    options=list(range(2019, _anio_max + 1)),
    index=list(range(2019, _anio_max + 1)).index(
        min(_hoy.year, _anio_max)   # por defecto: año actual (o el máximo permitido)
    ),
)

# ── Meses disponibles según el año elegido ────────────────────────────────────
# - Si el año es 2019: todos los meses (Ene–Dic)
# - Si el año es el máximo: sólo hasta el mes siguiente al actual
if anio_sel == _anio_max:
    _meses_disponibles = list(range(1, _mes_max + 1))
else:
    _meses_disponibles = list(range(1, 13))

# Índice por defecto del mes:
# - Si el año es el actual, intenta apuntar al mes actual; si no está disponible, el último
# - En cualquier otro año, apunta a enero
if anio_sel == _hoy.year:
    _mes_default = _hoy.month if _hoy.month in _meses_disponibles else _meses_disponibles[-1]
else:
    _mes_default = _meses_disponibles[0]

mes_sel = st.sidebar.selectbox(
    "Mes",
    options=_meses_disponibles,
    format_func=lambda m: MESES_NOMBRES[m],
    index=_meses_disponibles.index(_mes_default),
)

# ── Badge de modo (feedback inmediato al usuario) ─────────────────────────────
_modo_preview = determinar_modo(anio_sel, mes_sel)
_mui          = MODO_UI[_modo_preview]
st.sidebar.markdown(
    f'<div class="modo-badge" style="background:{_mui["color"]};color:{_mui["txt"]}">'
    f'{_mui["label"]}</div>',
    unsafe_allow_html=True
)
st.sidebar.caption(_mui["detalle"])

ejecutar = st.sidebar.button("🔄 Ejecutar análisis", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Modelo V13** · RandomForestClassifier  \n"
    "18 features: base ×5, lag1 ×5, lag2/delta ×3 c/u, estacionalidad  \n"
    f"Rango disponible: Enero 2019 – {MESES_NOMBRES[_mes_max]} {_anio_max}"
)

# ==============================================================================
# 12. LÓGICA PRINCIPAL
# ==============================================================================
if ejecutar:

    modo = determinar_modo(anio_sel, mes_sel)
    mui  = MODO_UI[modo]

    # ── Etiqueta de modo en la UI principal ───────────────────────────────
    st.markdown(
        f'<div class="modo-badge" style="background:{mui["color"]};color:{mui["txt"]};'
        f'font-size:13px;padding:6px 14px">{mui["label"]} — {mui["detalle"]}</div>',
        unsafe_allow_html=True
    )

    # ── Ejecutar el modo correspondiente ──────────────────────────────────
    with st.spinner(f"Analizando {MESES_NOMBRES[mes_sel]} {anio_sel} [{modo}]…"):
        try:
            if modo == "historico":
                pred, probs, datos_row = predecir_historico(anio_sel, mes_sel)

            elif modo == "gee_siguiente":
                pred, probs, datos_row = predecir_mes_siguiente(anio_sel, mes_sel)

            else:  # gee_futuro
                pred, probs, datos_row = predecir_gee_futuro(anio_sel, mes_sel)

        except Exception as e:
            st.error("❌ Error durante el análisis.")
            st.exception(e)
            st.stop()

    info = RIESGO_INFO[pred]

    # ── Título del resultado con mes analizado / mes predicho ─────────────
    if modo == "gee_siguiente":
        hoy = datetime.today()
        mes_mostrado   = MESES_NOMBRES[hoy.month]
        anio_mostrado  = hoy.year
        titulo_riesgo  = (f"🚦 Riesgo predicho para {MESES_NOMBRES[mes_sel]} {anio_sel} "
                          f"— índices de {mes_mostrado} {anio_mostrado}")
    else:
        titulo_riesgo  = f"🚦 Riesgo hídrico predicho — {MESES_NOMBRES[mes_sel]} {anio_sel}"

    st.markdown(f"<p class='section-title'>{titulo_riesgo}</p>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="badge-riesgo" style="background:{info["color"]}">'
        f'{info["emoji"]} {info["nombre"]} &nbsp;·&nbsp; '
        f'Probabilidad: {round(probs[pred]*100)}%</div>',
        unsafe_allow_html=True
    )

    # ── Métricas ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("NDVI",      f"{float(datos_row.get('NDVI', 0)):.3f}")
    c2.metric("NDWI agua", f"{float(datos_row.get('NDWI_agua', 0)):.3f}")
    c3.metric("NDWI veg",  f"{float(datos_row.get('NDWI_veg', 0)):.3f}")
    c4.metric("LST",       f"{float(datos_row.get('LST', 0)):.1f} °C")
    c5.metric("SPI-3",     f"{float(datos_row.get('SPI_3', 0)):.2f}")

    # ── Interpretaciones ──────────────────────────────────────────────────
    cols_i = st.columns(5)
    for col, key in zip(cols_i, ["NDVI","NDWI_agua","NDWI_veg","LST","SPI_3"]):
        bg, txt, rango, desc = obtener_interpretacion(key, float(datos_row.get(key, 0)))
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
                explicacion = explicar_riesgo_ia(datos_row, pred, probs, modo)
                st.info(explicacion)
            except Exception as e:
                st.warning(f"No se pudo generar la explicación IA: {e}")

    with col_prob:
        st.markdown("<p class='section-title'>📊 Distribución de probabilidades</p>",
                    unsafe_allow_html=True)
        st.plotly_chart(grafico_probabilidades(probs), use_container_width=True)

    st.markdown("---")

    # ── Mapa ──────────────────────────────────────────────────────────────
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
        "<p class='section-title'>📈 Índices satelitales — histórico mensual vs. valor analizado</p>",
        unsafe_allow_html=True
    )
    mi = int(datos_row.get("Mes", mes_sel)) - 1  # usa el mes de los datos reales

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(grafico_indice("NDVI",      float(datos_row.get("NDVI", 0)),      mi), use_container_width=True)
    with r1c2:
        st.plotly_chart(grafico_indice("NDWI_agua", float(datos_row.get("NDWI_agua", 0)), mi), use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(grafico_indice("NDWI_veg",  float(datos_row.get("NDWI_veg", 0)),  mi), use_container_width=True)
    with r2c2:
        st.plotly_chart(grafico_indice("LST",        float(datos_row.get("LST", 0)),       mi), use_container_width=True)

    st.plotly_chart(grafico_indice("SPI_3", float(datos_row.get("SPI_3", 0)), mi), use_container_width=True)

    # ── Tablas de detalle ──────────────────────────────────────────────────
    with st.expander("🔎 Ver datos del mes analizado"):
        st.dataframe(
            pd.DataFrame([datos_row]).style.format({
                "NDVI": "{:.4f}", "NDWI_agua": "{:.4f}", "NDWI_veg": "{:.4f}",
                "LST":  "{:.2f}", "Lluvia_mm": "{:.1f}",  "SPI_3":   "{:.3f}",
            }),
            use_container_width=True
        )

    with st.expander("🔎 Ver vector de features enviado al modelo (18 features V13)"):
        # Reconstruir X para mostrarla (la predicción ya está hecha)
        try:
            fila_display = pd.Series(datos_row)
            if modo == "historico":
                a1, m1 = _mes_anterior(anio_sel, mes_sel)
                a2, m2 = _mes_anterior(a1, m1)
                l1 = _buscar_en_csv(a1, m1)
                l2 = _buscar_en_csv(a2, m2)
            else:
                l1 = fila_display
                l2 = fila_display
            X_display = _fila_a_features(fila_display, l1, l2)
            st.dataframe(X_display.T.rename(columns={0: "valor"}).style.format("{:.5f}"),
                         use_container_width=True)
        except Exception:
            st.info("No se pudo reconstruir el vector de features para esta vista.")

else:
    # ── Pantalla de bienvenida ─────────────────────────────────────────────
    st.info("👈 Selecciona el mes y el año en el panel lateral, luego presiona **Ejecutar análisis**.")

    st.markdown("### Modos de predicción")
    for clave, mui in MODO_UI.items():
        st.markdown(
            f'<div style="background:{mui["color"]};color:{mui["txt"]};'
            f'padding:10px 14px;border-radius:8px;margin-bottom:8px">'
            f'<b>{mui["label"]}</b><br>'
            f'<span style="font-size:13px">{mui["detalle"]}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### ¿Qué analiza este sistema?")
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown(
        "🛰️ **Datos satelitales**  \n"
        "Sentinel-2 (NDVI, NDWI), Landsat 8 (LST) y CHIRPS (lluvia/SPI-3) "
        "vía Google Earth Engine o variables.csv."
    )
    col_b.markdown(
        "🤖 **Modelo V13**  \n"
        "RandomForestClassifier · 18 features · target: NRHF  \n"
        "Predice el nivel de riesgo hídrico del **mes siguiente** (clases 0-3)."
    )
    col_c.markdown(
        "🧠 **IA explicativa**  \n"
        "Groq + LLaMA 3 genera un análisis en lenguaje natural con causas y "
        "recomendaciones para operadores del embalse y comunidades."
    )

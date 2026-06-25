# -*- coding: utf-8 -*-

import json
import streamlit as st
import pandas as pd
import numpy as np
import ee
import joblib
import folium
from groq import Groq

# ==============================================================================
# CONFIG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Sistema Predictivo de Sequías",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Sistema de Alerta Temprana y Predicción de Riesgo Hídrico")
st.subheader("Subcuenca Represa El Coyolar - IA + Satélites + GEE")

# ==============================================================================
# GROQ API
# ==============================================================================
GROQ_API_KEY = st.secrets["DB_TOKEN"]
client = Groq(api_key=GROQ_API_KEY)

# ==============================================================================
# EARTH ENGINE
# ==============================================================================
def initialize_earth_engine():
    if "GEE_SERVICE_ACCOUNT" in st.secrets and "GEE_PRIVATE_KEY" in st.secrets:
        service_account = st.secrets["GEE_SERVICE_ACCOUNT"]
        private_key = st.secrets["GEE_PRIVATE_KEY"]

        # Corrige el formateo de saltos de línea generado por Streamlit Secrets
        if isinstance(private_key, str):
            private_key = private_key.replace("\\n", "\n")
        elif isinstance(private_key, dict):
            private_key = json.dumps(private_key)

        credentials = ee.ServiceAccountCredentials(service_account, private_key)
        ee.Initialize(credentials)
    else:
        ee.Initialize()



@st.cache_resource
def cargar_modelo_y_gee():
    initialize_earth_engine()

    modelo = joblib.load("modelo_sequia_futuro.pkl")

    cuencas = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_12')
    punto = ee.Geometry.Point([-87.5077, 14.3404])
    roi = cuencas.filterBounds(punto).geometry()

    return modelo, roi


modelo_rf, roi_subcuenca = cargar_modelo_y_gee()

st.sidebar.success("Modelo + GEE cargados correctamente")


# ==============================================================================
# LIMPIEZA DE NUBES
# ==============================================================================
def remover_nubes_scl(img):
    scl = img.select('SCL')
    mask = (
        scl.eq(4)
        .Or(scl.eq(5))
        .Or(scl.eq(6))
        .Or(scl.eq(7))
        .Or(scl.eq(11))
    )
    return img.updateMask(mask).clip(roi_subcuenca)


# ==============================================================================
# DATOS SATELITALES
# ==============================================================================
def obtener_datos_mes_actual():
    hoy = ee.Date(pd.Timestamp.now().strftime('%Y-%m-%d'))
    inicio = hoy.advance(-1, 'month')

    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi_subcuenca) \
        .filterDate(inicio, hoy) \
        .map(remover_nubes_scl) \
        .median()

    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI')

    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(roi_subcuenca) \
        .filterDate(inicio, hoy) \
        .median()

    lst = l8.select('ST_B10') \
        .multiply(0.00341802) \
        .subtract(273.15) \
        .rename('LST')

    lluvia = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterBounds(roi_subcuenca) \
        .filterDate(inicio, hoy) \
        .sum() \
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi_subcuenca,
            scale=5000
        ).get('precipitation')

    stack = ee.Image.cat([ndvi, ndwi, lst])

    stats = stack.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi_subcuenca,
        scale=30
    )

    return pd.DataFrame([{
        "NDVI": stats.get("NDVI").getInfo(),
        "NDWI_agua": stats.get("NDWI").getInfo(),
        "NDWI_veg": stats.get("NDWI").getInfo(),
        "LST": stats.get("LST").getInfo(),
        "Lluvia_mm": lluvia.getInfo()
    }])


# ==============================================================================
# GEE GEOMETRÍAS
# ==============================================================================
def obtener_subcuenca(lon, lat):
    cuencas = ee.FeatureCollection('WWF/HydroATLAS/v1/Basins/level12')
    punto = ee.Geometry.Point([lon, lat])
    return cuencas.filterBounds(punto).first().geometry()


def obtener_rio(roi):
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    agua = gsw.select('occurrence').gt(50).selfMask()

    vectors = agua.reduceToVectors(
        geometry=roi,
        scale=30,
        geometryType='polygon',
        eightConnected=True,
        maxPixels=1e9
    )

    return ee.Feature(vectors.first()).geometry()


# ==============================================================================
# SEMÁFORO
# ==============================================================================
diccionario_riesgos = {
    0: {"nombre": "🟢 Riesgo Bajo", "color": "#2ecc71", "nivel": "bajo"},
    1: {"nombre": "🟡 Riesgo Leve", "color": "#f1c40f", "nivel": "leve"},
    2: {"nombre": "🟠 Riesgo Moderado", "color": "#e67e22", "nivel": "moderado"},
    3: {"nombre": "🔴 Riesgo Severo", "color": "#e74c3c", "nivel": "severo"}
}


# ==============================================================================
# IA GROQ
# ==============================================================================
def explicar_riesgo_ia(datos, resultado):
    prompt = f"""
Eres un experto en hidrología.

Datos:
NDVI: {datos['NDVI']}
NDWI: {datos['NDWI_agua']}
LST: {datos['LST']}
Lluvia: {datos['Lluvia_mm']}

Riesgo: {resultado['nombre']}

Explica:
- situación actual
- causas
- recomendación

en español claro.
"""

    res = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return res.choices[0].message.content


# ==============================================================================
# BOTÓN PRINCIPAL
# ==============================================================================
if st.sidebar.button("🔄 Ejecutar Sistema Inteligente"):

    try:
        df = obtener_datos_mes_actual()

        df["SPI_3"] = (df["Lluvia_mm"] - 85.4) / 42.1

        X = df[["NDVI", "NDWI_agua", "NDWI_veg", "LST", "SPI_3"]]

        pred = modelo_rf.predict(X)[0]
        prob = modelo_rf.predict_proba(X)[0]

        resultado = diccionario_riesgos[pred]

        # =========================
        # MÉTRICAS
        # =========================
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("NDVI", f"{df['NDVI'][0]:.2f}")
        col2.metric("NDWI", f"{df['NDWI_agua'][0]:.2f}")
        col3.metric("NDWI veg", f"{df['NDWI_veg'][0]:.2f}")
        col4.metric("LST", f"{df['LST'][0]:.1f} °C")
        col5.metric("Lluvia", f"{df['Lluvia_mm'][0]:.1f} mm")

        # =========================
        # SEMÁFORO
        # =========================
        st.markdown("## 🚦 Estado del Riesgo")

        st.markdown(
            f"""
            <div style="
                background-color:{resultado['color']};
                padding:20px;
                border-radius:10px;
                color:white;
                font-size:20px;
                text-align:center;
                font-weight:bold;">
                {resultado['nombre']}
            </div>
            """,
            unsafe_allow_html=True
        )

        # =========================
        # IA EXPLICACIÓN
        # =========================
        explicacion = explicar_riesgo_ia(df.iloc[0], resultado)

        st.markdown("## 🧠 Explicación IA (Groq)")
        st.info(explicacion)

        # =========================
        # MAPA
        # =========================
        lon, lat = -87.5090, 14.3335

        geom_cuenca = obtener_subcuenca(lon, lat)
        geom_rio = obtener_rio(geom_cuenca)

        map_ = folium.Map(location=[lat, lon], zoom_start=12)

        folium.GeoJson(
            geom_cuenca.getInfo(),
            style_function=lambda x: {
                "fillColor": resultado["color"],
                "color": resultado["color"],
                "weight": 2,
                "fillOpacity": 0.3
            }
        ).add_to(map_)

        folium.GeoJson(
            geom_rio.getInfo(),
            style_function=lambda x: {
                "color": "blue",
                "weight": 2
            }
        ).add_to(map_)

        st_folium(map_, width=1000, height=500)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Presiona el botón para ejecutar el sistema inteligente.")
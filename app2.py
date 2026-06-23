import panel as pn
import pandas as pd
import numpy as np
import ee
import joblib
import folium
import os

# Inicializar la extensión de Panel con soporte para Folium/Leaflet
pn.extension('folium', sizing_mode="stretch_width")

# ==============================================================================
# 1. AUTENTICACIÓN DINÁMICA MEDIANTE VARIABLE DE ENTORNO (TOKEN PERSISTENTE)
# ==============================================================================
def inicializar_servicios():
    # Hugging Face lee los Secrets directamente como variables de entorno del sistema (os.environ)
    token_secure = os.environ.get("EE_TOKEN")
    modelo = joblib.load("modelo_sequia_futuro.pkl")
    
    if token_secure:
        from ee import oauth
        ee.Initialize(credentials=oauth.OAuth2Credentials(
            client_id=oauth.CLIENT_ID, client_secret=oauth.CLIENT_SECRET, refresh_token=token_secure
        ))
    else:
        ee.Initialize() # Fallback para desarrollo local
        
    return modelo

modelo_rf = inicializar_servicios()

# ==============================================================================
# 2. FUNCIONES DE EXTRACCIÓN SATELITAL (HYDROATLAS Y JRC)
# ==============================================================================
lon_coyolar, lat_coyolar = -87.50904715160283, 14.333509533380646
cuencas = ee.FeatureCollection('WWF/HydroATLAS/v1/Basins/level12')
roi_subcuenca = cuencas.filterBounds(ee.Geometry.Point([lon_coyolar, lat_coyolar])).geometry()

def remover_nubes_scl(img):
    scl = img.select('SCL')
    mask = scl.eq(4).orAnomalies(scl.eq(5)).orAnomalies(scl.eq(6)).orAnomalies(scl.eq(7)).orAnomalies(scl.eq(11))
    return img.updateMask(mask).clip(roi_subcuenca)

def consultar_satelites_actuales():
    fecha_hoy = ee.Date(pd.Timestamp.now().strftime('%Y-%m-%d'))
    fecha_inicio = fecha_hoy.advance(-1, 'month')
    
    imagen_s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi_subcuenca).filterDate(fecha_inicio, fecha_hoy).map(remover_nubes_scl).median()
    ndvi = imagen_s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi_agua = imagen_s2.normalizedDifference(['B3', 'B8']).rename('NDWI_agua')
    ndwi_veg = imagen_s2.normalizedDifference(['B8', 'B11']).rename('NDWI_veg')
    
    imagen_l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(roi_subcuenca).filterDate(fecha_inicio, fecha_hoy).median()
    lst = imagen_l8.select('ST_B10').multiply(0.00341802).subtract(273.15).rename('LST')
    
    lluvia_mes = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterBounds(roi_subcuenca).filterDate(fecha_inicio, fecha_hoy).sum().reduceRegion(reducer=ee.Reducer.mean(), geometry=roi_subcuenca, scale=5000).get('precipitation')
    
    stats = ee.Image.cat([ndvi, ndwi_agua, ndwi_veg, lst]).reduceRegion(reducer=ee.Reducer.mean(), geometry=roi_subcuenca, scale=30)
    
    return pd.DataFrame([{
        'NDVI': stats.get('NDVI').getInfo(), 'NDWI_agua': stats.get('NDWI_agua').getInfo(),
        'NDWI_veg': stats.get('NDWI_veg').getInfo(), 'LST': stats.get('LST').getInfo(), 'Lluvia_mm': lluvia_mes.getInfo()
    }])

# ==============================================================================
# 3. LÓGICA REACTIVA DEL PANEL INTERACTIVO
# ==============================================================================
# Contenedores visuales vacíos (Placeholders) que Panel actualizará dinámicamente
html_alertas = pn.pane.HTML(width=800)
mapa_pane = pn.pane.Folium(width=1000, height=500)

def ejecutar_pipeline_evento(event):
    html_alertas.object = "<div style='color: orange;'>🔄 Analizando firmas espectrales en la nube de Google...</div>"
    try:
        df_actual = consultar_satelites_actuales()
        df_actual['SPI_3'] = (df_actual['Lluvia_mm'] - 85.4) / 42.1
        X_actual = df_actual[['NDVI', 'NDWI_agua', 'NDWI_veg', 'LST', 'SPI_3']]
        
        prediccion = int(modelo_rf.predict(X_actual)[0])
        dicc_colores = {0: ("Bajo (Sin Sequía)", "green"), 1: ("Leve", "yellow"), 2: ("Moderado", "orange"), 3: ("Severo (Alerta)", "red")}
        txt_riesgo, color_riesgo = dicc_colores[prediccion]
        
        # Actualizar la tarjeta de alerta en formato HTML limpio sin recargar la página
        html_alertas.object = f"""
        <div style='padding:15px; background-color:{color_riesgo}; color:white; border-radius:5px; font-weight:bold;'>
            🔮 Predicción de Riesgo Hídrico para el próximo mes en la Subcuenca: {txt_riesgo}
        </div>
        """
        
        # Renderizar el mapa dinámico nativo de GEE
        mapa_web = folium.Map(location=[lat_coyolar, lon_coyolar], zoom_start=13, tiles="OpenStreetMap")
        
        cuencas_altas = ee.FeatureCollection('WWF/HydroATLAS/v1/Basins/level12')
        geom_subcuenca = cuencas_altas.filterBounds(ee.Geometry.Point([lon_coyolar, lat_coyolar])).first().geometry().getInfo()
        
        folium.GeoJson(data=geom_subcuenca, style_function=lambda f: {'fillColor': color_riesgo, 'color': color_riesgo, 'weight': 2, 'fillOpacity': 0.35}).add_to(mapa_web)
        
        # Forzar el refresco asíncrono del componente del mapa
        mapa_pane.object = mapa_web
        
    except Exception as err:
        html_alertas.object = f"<div style='color: red;'>❌ Error: {str(err)}</div>"

# Botón interactivo nativo de HoloViz Panel
boton_calcular = pn.widgets.Button(name="🔄 Interrogar Satélites y Predecir", button_type="primary", width=300)
boton_calcular.on_click(ejecutar_pipeline_evento)

# ==============================================================================
# 4. MAQUETACIÓN E INTERFAZ GRÁFICA FINAL (LAYOUT DE TESIS)
# ==============================================================================
dashboard = pn.Column(
    pn.pane.Markdown("# 💧 SAT - Alerta Temprana de Sequías Fluviales"),
    pn.pane.Markdown("### Dashboard Científico de Control Hidroambiental - Represa El Coyolar, Honduras"),
    pn.Row(boton_calcular, html_alertas),
    pn.layout.Divider(),
    mapa_pane
)

# Servir la aplicación de forma pública para el servidor web
dashboard.servable()

import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import ee
import joblib
import plotly.graph_objects as go
import os

# Inicializar la aplicación Dash
app = dash.Dash(__name__, title="SAT - Predicción El Coyolar")
server = app.server  # Expone el servidor Flask para que Koyeb pueda ejecutarlo

# ==============================================================================
# 1. AUTENTICACIÓN DINÁMICA MEDIANTE VARIABLE DE ENTORNO (TOKEN PERSISTENTE)
# ==============================================================================
def inicializar_servicios():
    # Koyeb lee los Secrets directamente como variables de entorno (os.environ)
    token_secure = os.environ.get("EE_TOKEN")
    modelo = joblib.load("modelo_sequia_futuro.pkl")
    
    if token_secure:
        from ee import oauth
        ee.Initialize(credentials=oauth.OAuth2Credentials(
            client_id=oauth.CLIENT_ID, client_secret=oauth.CLIENT_SECRET, refresh_token=token_secure
        ))
    else:
        ee.Initialize() # Fallback local
        
    return modelo

modelo_rf = inicializar_servicios()

# Coordenadas e inicialización geográfica de la subcuenca (HydroATLAS)
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
# 2. DISEÑO DE LA INTERFAZ VISUAL (VIEW - HTML/CSS COMPONENTES)
# ==============================================================================
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1("💧 SAT - Alerta Temprana de Sequías Fluviales", style={'color': '#1a365d'}),
    html.H3("Dashboard Científico de Control Hidroambiental - Represa El Coyolar, Honduras", style={'color': '#4a5568'}),
    
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '25px'}, children=[
        html.Button("🔄 Interrogar Satélites y Predecir", id="btn-predict", n_clicks=0, 
                    style={'padding': '12px 24px', 'backgroundColor': '#3182ce', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'fontSize': '16px'}),
        html.Div(id="output-alerta", style={'flex': '1'})
    ]),
    
    html.Hr(),
    
    html.Div(children=[
        dcc.Graph(id="mapa-plotly", style={'height': '600px'})
    ])
])

# ==============================================================================
# 3. LÓGICA DE CONTROL (CONTROLLER - CALLBACK REACTIVO)
# ==============================================================================
@app.callback(
    [Output("output-alerta", "children"),
     Output("mapa-plotly", "figure")],
    [Input("btn-predict", "n_clicks")],
    prevent_initial_call=True
)
def ejecutar_prediccion_y_mapeo(n_clicks):
    # Mensaje de carga inicial si el botón no ha hecho procesamiento profundo
    if n_clicks == 0:
        return dash.no_update, dash.no_update
        
    try:
        # Extraer variables climáticas y aplicar el modelo
        df_actual = consultar_satelites_actuales()
        df_actual['SPI_3'] = (df_actual['Lluvia_mm'] - 85.4) / 42.1
        X_actual = df_actual[['NDVI', 'NDWI_agua', 'NDWI_veg', 'LST', 'SPI_3']]
        
        prediccion = int(modelo_rf.predict(X_actual))
        dicc_colores = {0: ("Bajo (Sin Sequía)", "#28a745"), 1: ("Leve", "#ffc107"), 2: ("Moderado", "#fd7e14"), 3: ("Severo (Alerta)", "#dc3545")}
        txt_riesgo, color_riesgo = dicc_colores[prediccion]
        
        # 1. Construir la tarjeta de Alerta en HTML para la vista
        alerta_componente = html.Div(
            f"🔮 Predicción de Riesgo Hídrico para el próximo mes en la Subcuenca: {txt_riesgo}",
            style={'padding': '15px', 'backgroundColor': color_riesgo, 'color': 'white', 'borderRadius': '5px', 'fontWeight': 'bold', 'fontSize': '16px'}
        )
        
        # 2. Consultar la geometría oficial de la cuenca en HydroATLAS
        cuencas_altas = ee.FeatureCollection('WWF/HydroATLAS/v1/Basins/level12')
        geojson_subcuenca = cuencas_altas.filterBounds(ee.Geometry.Point([lon_coyolar, lat_coyolar])).first().geometry().getInfo()
        
        # Extraer los puntos del polígono para graficarlos en las líneas de Plotly
        lons = [pt[0] for pt in geojson_subcuenca['coordinates'][0]]
        lats = [pt[1] for pt in geojson_subcuenca['coordinates'][1]] if len(geojson_subcuenca['coordinates']) > 1 else [pt[1] for pt in geojson_subcuenca['coordinates'][0]]
        if len(geojson_subcuenca['coordinates'][0][0]) == 2: # Manejo de estructuras GeoJSON estándar
            lons = [pt[0] for pt in geojson_subcuenca['coordinates'][0]]
            lats = [pt[1] for pt in geojson_subcuenca['coordinates'][0]]

        # 3. Renderizar el mapa de Mapbox usando componentes nativos de Plotly
        fig = go.Figure()
        
        # Dibujar el contorno del polígono de HydroATLAS con el color predictivo
        fig.add_trace(go.Scattermapbox(
            lon=lons, lat=lats,
            mode='lines',
            fill='toself',
            fillcolor=color_riesgo,
            opacity=0.4,
            line=dict(width=2.5, color=color_riesgo),
            text=f"Subcuenca El Coyolar - Alerta: {txt_riesgo}",
            hoverinfo='text'
        ))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=lat_coyolar, lon=lon_coyolar),
                zoom=12
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return alerta_componente, fig
        
    except Exception as err:
        error_comp = html.Div(f"❌ Error en el Pipeline: {str(err)}", style={'color': 'red', 'fontWeight': 'bold'})
        return error_comp, go.Figure()

# Ejecutar el servidor web cuando la nube lo invoque
if __name__ == '__main__':
    # El puerto lo asignará Koyeb automáticamente, se usa 8000 como fallback
    port = int(os.environ.get("PORT", 8000))
    app.run_server(host='0.0.0.0', port=port)

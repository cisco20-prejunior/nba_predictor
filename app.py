import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="NBA AI Predictor", page_icon="🏀", layout="wide")

# --- CARGAR MODELOS Y DATOS (CON CACHÉ PARA RAPIDEZ) ---
@st.cache_resource
def load_artifacts():
    model = load_model('nba_model_dvp.h5')
    scaler = joblib.load('nba_scaler.pkl')
    pos_map = joblib.load('pos_map.pkl')
    le_opp = joblib.load('opp_encoder.pkl')
    dvp_stats = pd.read_csv('defense_vs_position.csv')
    df = pd.read_csv('nba_data_final.csv') # Tu df_clean guardado
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return model, scaler, pos_map, le_opp, dvp_stats, df

try:
    model, scaler, pos_map, le_opp, dvp_stats, df = load_artifacts()
except:
    st.error("Error cargando archivos. Asegúrate de subir los .h5, .pkl y .csv")
    st.stop()

# --- FUNCIONES AUXILIARES ---
def get_defense_stats(rival, pos):
    stats = dvp_stats[(dvp_stats['OPPONENT_ABBREV'] == rival) & (dvp_stats['POSITION'] == pos)]
    if stats.empty: return 20.0, 5.0, 5.0
    return stats.iloc[0]['OPP_ALLOW_PTS'], stats.iloc[0]['OPP_ALLOW_REB'], stats.iloc[0]['OPP_ALLOW_AST']

def predict_player(name, rival, is_home):
    # Buscar datos recientes del jugador
    player_data = df[df['PLAYER_NAME'] == name].sort_values(by='GAME_DATE').tail(1)
    if player_data.empty: return None
    
    # Preparar inputs
    p_id = player_data['PLAYER_ID'].values[0]
    pos = pos_map.get(p_id, 'F')
    
    opp_pts, opp_reb, opp_ast = get_defense_stats(rival, pos)
    
    # Construir fila
    N_GAMES = 5
    stats_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FG_PCT', 'FG3_PCT']
    feats = [f'AVG_{s}_LAST_{N_GAMES}' for s in stats_cols]
    
    row = player_data[feats].copy()
    row['IS_HOME'] = 1 if is_home else 0
    row['OPP_ALLOW_PTS'] = opp_pts
    row['OPP_ALLOW_REB'] = opp_reb
    row['OPP_ALLOW_AST'] = opp_ast
    
    # Ordenar columnas como en el entrenamiento
    features_defense = ['OPP_ALLOW_PTS', 'OPP_ALLOW_REB', 'OPP_ALLOW_AST']
    final_cols = feats + ['IS_HOME'] + features_defense
    
    # Predecir
    X_scaled = scaler.transform(row[final_cols])
    pred = model.predict(X_scaled)
    return pred[0], pos, opp_pts

# --- INTERFAZ GRÁFICA ---
st.title("🏀 NBA AI Performance Predictor")
st.markdown("Predicciones basadas en **Redes Neuronales** y **Defensa por Posición (DvP)**.")

tab1, tab2 = st.tabs(["🔮 Predicción Individual", "📅 Cartelera de Hoy"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        jugador = st.selectbox("Selecciona un Jugador", df['PLAYER_NAME'].unique())
    with col2:
        rival = st.selectbox("Rival (Equipo)", dvp_stats['OPPONENT_ABBREV'].unique())
    with col3:
        localia = st.radio("¿Dónde juega?", ["Casa 🏠", "Visita ✈️"])

    if st.button("Predecir Rendimiento"):
        is_home = True if localia == "Casa 🏠" else False
        res = predict_player(jugador, rival, is_home)
        
        if res:
            (pts, reb, ast), pos, opp_pts_allow = res
            
            st.markdown(f"### Resultados para **{jugador}** ({pos})")
            
            # Métricas grandes
            c1, c2, c3 = st.columns(3)
            c1.metric("Puntos", f"{pts:.1f}")
            c2.metric("Rebotes", f"{reb:.1f}")
            c3.metric("Asistencias", f"{ast:.1f}")
            
            # Análisis Defensivo
            if opp_pts_allow < 18:
                st.warning(f"🛡️ **Alerta Defensiva:** {rival} es una muralla contra los {pos} (Permiten solo {opp_pts_allow:.1f} pts).")
            elif opp_pts_allow > 25:
                st.success(f"🎯 **Oportunidad:** {rival} tiene una defensa débil contra los {pos}.")
            
            # Mostrar promedios recientes para comparar
            st.caption("Comparación con sus últimos 5 partidos:")
            last_games = df[df['PLAYER_NAME'] == jugador].tail(5)[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST']]
            st.dataframe(last_games)
            
        else:
            st.error("No hay datos suficientes para este jugador.")

with tab2:
    if st.button("Cargar Partidos de Hoy"):
        today = datetime.now().strftime('%Y-%m-%d')
        # today = "2024-12-04" # Descomentar para pruebas
        st.info(f"Buscando partidos para: {today}")
        
        try:
            board = scoreboardv2.ScoreboardV2(game_date=today)
            games = board.get_data_frames()[0]
            
            if games.empty:
                st.warning("No hay partidos programados.")
            else:
                for i, game in games.iterrows():
                    home = game['HOME_TEAM_ID']
                    visita = game['VISITOR_TEAM_ID']
                    
                    # Intentar sacar nombres (fallback simple)
                    try: h_name = df[df['TEAM_ID'] == home]['TEAM_ABBREVIATION'].iloc[0]
                    except: h_name = "HOME"
                    try: v_name = df[df['TEAM_ID'] == visita]['TEAM_ABBREVIATION'].iloc[0]
                    except: v_name = "VISIT"
                    
                    with st.expander(f"{v_name} @ {h_name}"):
                        st.write("Análisis rápido de jugadores clave:")
                        # Aquí podrías iterar jugadores como en la Celda 5
                        # Por simplicidad en la web, mostramos solo el matchup
                        st.write("Usa la pestaña 'Predicción Individual' para analizar jugadores específicos de este partido.")
                        
        except Exception as e:
            st.error(f"Error conectando con NBA API: {e}")
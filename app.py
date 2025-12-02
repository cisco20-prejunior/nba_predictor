# --- AÑADIR ESTO AL PRINCIPIO DESPUÉS DE LOS IMPORTS ---
from nba_api.stats.library.http import NBAStatsHTTP

# Configuración Anti-Bloqueo: Aumentar timeout y usar User-Agent real
NBAStatsHTTP.headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Origin': 'https://www.nba.com',
    'Referer': 'https://www.nba.com/'
}
# Aumentar el tiempo de espera antes de rendirse
NBAStatsHTTP.timeout = 30

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="NBA AI Predictor", page_icon="🏀", layout="wide")

# --- CARGAR MODELOS Y DATOS ---

@st.cache_resource
def load_artifacts():
    # compile=False evita errores de métricas entre versiones
    model = load_model('nba_model_dvp.h5', compile=False)
    scaler = joblib.load('nba_scaler.pkl')
    pos_map = joblib.load('pos_map.pkl')
    le_opp = joblib.load('opp_encoder.pkl')
    dvp_stats = pd.read_csv('defense_vs_position.csv')
    
    # Cargar datos históricos
    df = pd.read_csv('nba_data_final.csv')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # --- CORRECCIÓN DEL ERROR ---
    # 1. Eliminar filas donde el nombre del jugador sea nulo (NaN)
    df = df.dropna(subset=['PLAYER_NAME'])
    # 2. Asegurar que todos los nombres sean texto (string)
    df['PLAYER_NAME'] = df['PLAYER_NAME'].astype(str)
    
    return model, scaler, pos_map, le_opp, dvp_stats, df

# Carga inicial
try:
    model, scaler, pos_map, le_opp, dvp_stats, df = load_artifacts()
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

# --- FUNCIONES AUXILIARES ---

def get_defense_stats(rival, pos):
    """Busca estadísticas defensivas del rival"""
    stats = dvp_stats[(dvp_stats['OPPONENT_ABBREV'] == rival) & (dvp_stats['POSITION'] == pos)]
    if stats.empty: return 20.0, 5.0, 5.0
    return stats.iloc[0]['OPP_ALLOW_PTS'], stats.iloc[0]['OPP_ALLOW_REB'], stats.iloc[0]['OPP_ALLOW_AST']

def get_active_roster(team_id, min_minutes=20):
    """Busca jugadores del equipo con minutos relevantes"""
    # Filtramos por ID de equipo
    team_df = df[df['TEAM_ID'] == team_id]
    if team_df.empty: return []
    
    # Buscamos el último partido de cada jugador único
    # y verificamos si sigue en el equipo y si juega suficientes minutos
    roster = []
    unique_players = team_df['PLAYER_NAME'].unique()
    
    for p in unique_players:
        p_data = team_df[team_df['PLAYER_NAME'] == p].sort_values('GAME_DATE').iloc[-1]
        
        # Validamos que su último partido haya sido con este equipo
        if p_data['TEAM_ID'] == team_id:
            # Validamos minutos (Columna AVG_MIN_LAST_5 debe existir en tu csv)
            if 'AVG_MIN_LAST_5' in p_data and p_data['AVG_MIN_LAST_5'] >= min_minutes:
                roster.append(p)
            elif 'AVG_MIN_LAST_5' not in p_data:
                # Si no existe la columna, lo agregamos por si acaso
                roster.append(p)
                
    return roster

def predict_player(name, rival, is_home):
    """Ejecuta la predicción para un jugador"""
    # 1. Buscar datos recientes
    player_data = df[df['PLAYER_NAME'] == name].sort_values(by='GAME_DATE').tail(1)
    if player_data.empty: return None
    
    # 2. Preparar inputs
    p_id = player_data['PLAYER_ID'].values[0]
    pos = pos_map.get(p_id, 'F') # Si falla el mapa, usamos 'F'
    
    opp_pts, opp_reb, opp_ast = get_defense_stats(rival, pos)
    
    # 3. Construir fila de características
    # Deben coincidir con las usadas en el entrenamiento
    stats_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FG_PCT', 'FG3_PCT']
    feats = [f'AVG_{s}_LAST_5' for s in stats_cols]
    
    try:
        row = player_data[feats].copy()
    except KeyError:
        return None # Faltan columnas
        
    row['IS_HOME'] = 1 if is_home else 0
    row['OPP_ALLOW_PTS'] = opp_pts
    row['OPP_ALLOW_REB'] = opp_reb
    row['OPP_ALLOW_AST'] = opp_ast
    
    # Orden correcto
    features_defense = ['OPP_ALLOW_PTS', 'OPP_ALLOW_REB', 'OPP_ALLOW_AST']
    final_cols = feats + ['IS_HOME'] + features_defense
    
    # 4. Predecir
    try:
        X_scaled = scaler.transform(row[final_cols])
        pred = model.predict(X_scaled, verbose=0)
        return pred[0], pos, opp_pts
    except:
        return None

# --- INTERFAZ GRÁFICA ---
st.title("🏀 NBA AI Predictor")

tab1, tab2 = st.tabs(["🔮 Predicción Individual", "📅 Cartelera de Hoy"])

# --- PESTAÑA 1: BUSCADOR ---
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        jugador = st.selectbox("Jugador", sorted(df['PLAYER_NAME'].unique()))
    with col2:
        # Lista de equipos rivales
        rivales = sorted(dvp_stats['OPPONENT_ABBREV'].unique())
        rival = st.selectbox("Rival", rivales)
    with col3:
        localia = st.radio("Condición", ["Casa 🏠", "Visita ✈️"])

    if st.button("Analizar Jugador"):
        is_home = True if localia == "Casa 🏠" else False
        res = predict_player(jugador, rival, is_home)
        
        if res:
            (pts, reb, ast), pos, opp_allowed = res
            
            st.markdown(f"### 📊 Proyección: {jugador} ({pos})")
            
            # Tarjetas de números
            c1, c2, c3 = st.columns(3)
            c1.metric("Puntos", f"{pts:.1f}")
            c2.metric("Rebotes", f"{reb:.1f}")
            c3.metric("Asistencias", f"{ast:.1f}")
            
            # Contexto defensivo
            if opp_allowed < 15: # Umbral de escudo estricto
                st.warning(f"🛡️ **Defensa Élite:** {rival} solo permite {opp_allowed:.1f} pts a su posición.")
            elif opp_allowed > 25:
                st.success(f"🔥 **Defensa Débil:** {rival} permite {opp_allowed:.1f} pts (Matchup favorable).")
            else:
                st.info(f"⚖️ **Defensa Promedio:** {rival} permite {opp_allowed:.1f} pts.")

        else:
            st.error("Datos insuficientes para predecir.")

# --- SUSTITUYE EL BLOQUE DE LA PESTAÑA 2 CON ESTO ---
with tab2:
    st.write("Generando predicciones automáticas para los partidos del día...")
    
    if st.button("🔄 Cargar Partidos de Hoy"):
        today = datetime.now().strftime('%Y-%m-%d')
        # today = "2024-12-04" # Descomentar para pruebas
        
        try:
            # Intentamos conectar con la API
            board = scoreboardv2.ScoreboardV2(game_date=today)
            games = board.get_data_frames()[0]
            
            if games.empty:
                st.warning(f"No hay partidos programados para hoy ({today}).")
            else:
                st.success(f"Se encontraron {len(games)} partidos.")
                
                # --- BUCLE DE PARTIDOS (Tu código original) ---
                for i, game in games.iterrows():
                    home_id = game['HOME_TEAM_ID']
                    away_id = game['VISITOR_TEAM_ID']
                    
                    try: h_abb = df[df['TEAM_ID'] == home_id]['TEAM_ABBREVIATION'].iloc[0]
                    except: h_abb = "HOME"
                    try: v_abb = df[df['TEAM_ID'] == away_id]['TEAM_ABBREVIATION'].iloc[0]
                    except: v_abb = "AWAY"
                    
                    with st.expander(f"🏀 {v_abb} @ {h_abb}", expanded=True):
                        col_away, col_home = st.columns(2)
                        
                        # --- EQUIPO VISITANTE ---
                        with col_away:
                            st.markdown(f"**✈️ {v_abb} (Visita)**")
                            roster = get_active_roster(away_id, min_minutes=20)
                            if not roster: st.caption("Sin datos de jugadores.")
                            for p in roster:
                                res = predict_player(p, h_abb, False)
                                if res:
                                    (pts, reb, ast), pos, opp_pts = res
                                    icon = "🛡️" if opp_pts < 15 else ""
                                    st.write(f"{icon} **{p}** ({pos}): {pts:.1f} PTS | {reb:.1f} REB | {ast:.1f} AST")

                        # --- EQUIPO LOCAL ---
                        with col_home:
                            st.markdown(f"**🏠 {h_abb} (Casa)**")
                            roster = get_active_roster(home_id, min_minutes=20)
                            if not roster: st.caption("Sin datos de jugadores.")
                            for p in roster:
                                res = predict_player(p, v_abb, True)
                                if res:
                                    (pts, reb, ast), pos, opp_pts = res
                                    icon = "🛡️" if opp_pts < 15 else ""
                                    st.write(f"{icon} **{p}** ({pos}): {pts:.1f} PTS | {reb:.1f} REB | {ast:.1f} AST")

        except Exception as e:
            # MENSAJE DE ERROR AMIGABLE
            st.error("⚠️ La API de la NBA ha rechazado la conexión temporalmente.")
            st.code(f"Error técnico: {e}")
            st.info("💡 Consejo: Intenta de nuevo en 1 minuto. Si estás en Streamlit Cloud, los servidores de la NBA a veces bloquean estas IPs compartidas.")
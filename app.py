import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import warnings
from tensorflow.keras.models import load_model
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.library.http import NBAStatsHTTP

# --- 0. CONFIGURACIÓN INICIAL ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="NBA AI Predictor 2026", page_icon="🏀", layout="wide")

# --- 1. CONFIGURACIÓN ANTI-BLOQUEO ---
NBAStatsHTTP.timeout = 60 
NBAStatsHTTP.headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Origin': 'https://www.nba.com',
    'Referer': 'https://www.nba.com/',
}

# --- 2. CARGAR MODELOS Y DATOS ---
@st.cache_resource
def load_artifacts():
    # Modelo
    model = load_model('nba_model_dvp.h5', compile=False)
    
    # Artefactos
    scaler = joblib.load('nba_scaler.pkl')
    pos_map = joblib.load('pos_map.pkl')
    le_opp = joblib.load('opp_encoder.pkl')
    dvp_stats = pd.read_csv('defense_vs_position.csv')
    
    # Datos Históricos
    df = pd.read_csv('nba_data_final.csv')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Limpieza
    df = df.dropna(subset=['PLAYER_NAME'])
    df['PLAYER_NAME'] = df['PLAYER_NAME'].astype(str)
    
    # --- CÁLCULO DE PROMEDIOS TEMPORADA ACTUAL (2025-26) ---
    inicio_temp_2026 = pd.to_datetime('2025-09-01')
    df_2026 = df[df['GAME_DATE'] >= inicio_temp_2026].copy()
    
    # Diccionario: Jugador -> Minutos Promedio en 2026
    promedio_minutos_2026 = df_2026.groupby('PLAYER_NAME')['MIN'].mean().to_dict()
    
    return model, scaler, pos_map, le_opp, dvp_stats, df, promedio_minutos_2026

try:
    model, scaler, pos_map, le_opp, dvp_stats, df, avg_min_2026 = load_artifacts()
except Exception as e:
    st.error(f"Error crítico cargando archivos: {e}")
    st.stop()

# --- 3. FUNCIONES AUXILIARES ---

def obtener_scoreboard_seguro(fecha):
    intentos = 0
    while intentos < 3:
        try:
            board = scoreboardv2.ScoreboardV2(game_date=fecha, timeout=60)
            games = board.get_data_frames()[0]
            return games
        except:
            intentos += 1
            time.sleep(1.5)
    raise Exception("API NBA sin respuesta.")

def get_defense_stats(rival, pos):
    stats = dvp_stats[(dvp_stats['OPPONENT_ABBREV'] == rival) & (dvp_stats['POSITION'] == pos)]
    if stats.empty: return 20.0, 5.0, 5.0
    return stats.iloc[0]['OPP_ALLOW_PTS'], stats.iloc[0]['OPP_ALLOW_REB'], stats.iloc[0]['OPP_ALLOW_AST']

def get_active_roster(team_id, min_minutes):
    """
    Busca jugadores que ACTUALMENTE pertenezcan al equipo.
    Lógica: El último partido registrado del jugador DEBE ser con este equipo.
    """
    # 1. Obtenemos lista de candidatos (cualquiera que haya jugado para este equipo alguna vez)
    # Esto es rápido para filtrar la lista gigante
    candidatos_df = df[df['TEAM_ID'] == team_id]
    if candidatos_df.empty: return []
    
    nombres_candidatos = candidatos_df['PLAYER_NAME'].unique()
    roster_final = []
    
    for p in nombres_candidatos:
        # 2. VALIDACIÓN CRÍTICA:
        # Buscamos el ÚLTIMO partido de este jugador en TODO el dataframe global (df),
        # no solo en los partidos de este equipo.
        ultimo_partido_global = df[df['PLAYER_NAME'] == p].sort_values('GAME_DATE').iloc[-1]
        
        # 3. Si su último partido global FUE con este equipo, entonces sigue aquí.
        if ultimo_partido_global['TEAM_ID'] == team_id:
            
            # 4. Aplicamos el filtro de minutos de la temporada actual
            promedio_real = avg_min_2026.get(p, 0)
            if promedio_real >= min_minutes:
                roster_final.append(p)
                
    return roster_final

def predict_player(name, rival, is_home):
    # Datos recientes
    player_data = df[df['PLAYER_NAME'] == name].sort_values(by='GAME_DATE').tail(1)
    if player_data.empty: return None
    
    # Inputs
    p_id = player_data['PLAYER_ID'].values[0]
    pos = pos_map.get(p_id, 'F')
    opp_pts, opp_reb, opp_ast = get_defense_stats(rival, pos)
    
    # Features
    stats_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FG_PCT', 'FG3_PCT']
    feats = [f'AVG_{s}_LAST_5' for s in stats_cols]
    
    try: row = player_data[feats].copy()
    except: return None
        
    row['IS_HOME'] = 1 if is_home else 0
    row['OPP_ALLOW_PTS'] = opp_pts
    row['OPP_ALLOW_REB'] = opp_reb
    row['OPP_ALLOW_AST'] = opp_ast
    
    final_cols = feats + ['IS_HOME', 'OPP_ALLOW_PTS', 'OPP_ALLOW_REB', 'OPP_ALLOW_AST']
    
    try:
        X_scaled = scaler.transform(row[final_cols])
        pred = model.predict(X_scaled, verbose=0)
        return pred[0], pos, opp_pts
    except: return None

# --- 4. INTERFAZ ---

st.title("🏀 NBA AI Predictor 2026")
st.markdown("Predicciones basadas en **Roster Actual** y **Defensa por Posición**.")

tab1, tab2 = st.tabs(["🔮 Predicción Individual", "📅 Cartelera de Hoy"])

# --- PESTAÑA 1 ---
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        jugador = st.selectbox("Jugador", sorted(df['PLAYER_NAME'].unique()))
    with col2:
        rival = st.selectbox("Rival", sorted(dvp_stats['OPPONENT_ABBREV'].unique()))
    with col3:
        localia = st.radio("Condición", ["Casa 🏠", "Visita ✈️"])

    if st.button("Analizar Jugador", type="primary"):
        res = predict_player(jugador, rival, localia == "Casa 🏠")
        if res:
            (pts, reb, ast), pos, opp = res
            st.markdown(f"### 📊 {jugador} ({pos})")
            c1, c2, c3 = st.columns(3)
            c1.metric("Puntos", f"{pts:.1f}")
            c2.metric("Rebotes", f"{reb:.1f}")
            c3.metric("Asistencias", f"{ast:.1f}")
            if opp < 15: st.warning("🛡️ Rival difícil.")
            elif opp > 25: st.success("🔥 Rival fácil.")
        else:
            st.error("Datos insuficientes.")

# --- PESTAÑA 2 ---
with tab2:
    col_slide, col_b = st.columns([2,1])
    with col_slide:
        min_filter = st.slider("Minutos Promedio (Temp 2025-26):", 10, 40, 25)
    
    if st.button("🔄 Cargar Partidos"):
        today = datetime.now().strftime('%Y-%m-%d')
        # today = "2024-12-04" # Descomentar para pruebas
        
        with st.spinner(f'Buscando partidos para {today}...'):
            try:
                games = obtener_scoreboard_seguro(today)
                if games.empty:
                    st.warning("No hay partidos hoy.")
                else:
                    st.success(f"✅ {len(games)} partidos encontrados.")
                    for i, game in games.iterrows():
                        h_id = game['HOME_TEAM_ID']
                        a_id = game['VISITOR_TEAM_ID']
                        
                        try: h_abb = df[df['TEAM_ID'] == h_id]['TEAM_ABBREVIATION'].iloc[0]
                        except: h_abb = "HOME"
                        try: a_abb = df[df['TEAM_ID'] == a_id]['TEAM_ABBREVIATION'].iloc[0]
                        except: a_abb = "AWAY"
                        
                        with st.expander(f"🏀 {a_abb} @ {h_abb}", expanded=True):
                            c_away, c_home = st.columns(2)
                            
                            with c_away:
                                st.markdown(f"**✈️ {a_abb}**")
                                roster = get_active_roster(a_id, min_filter)
                                if not roster: st.caption("Nadie cumple el filtro.")
                                for p in roster:
                                    res = predict_player(p, h_abb, False)
                                    if res:
                                        (pts, reb, ast), pos, opp = res
                                        icon = "🛡️" if opp < 15 else ""
                                        st.write(f"{icon} **{p}** ({pos}): {pts:.1f} PTS | {reb:.1f} REB")
                            
                            with c_home:
                                st.markdown(f"**🏠 {h_abb}**")
                                roster = get_active_roster(h_id, min_filter)
                                if not roster: st.caption("Nadie cumple el filtro.")
                                for p in roster:
                                    res = predict_player(p, a_abb, True)
                                    if res:
                                        (pts, reb, ast), pos, opp = res
                                        icon = "🛡️" if opp < 15 else ""
                                        st.write(f"{icon} **{p}** ({pos}): {pts:.1f} PTS | {reb:.1f} REB")
            except Exception as e:
                st.error("Error de conexión.")
                st.code(e)
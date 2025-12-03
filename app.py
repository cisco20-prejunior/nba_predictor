import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from tensorflow.keras.models import load_model
from datetime import datetime

# --- 0. CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="NBA AI Simulator", page_icon="🏀", layout="wide")

# LISTA OFICIAL DE LOS 30 EQUIPOS ACTUALES
NBA_TEAMS_CURRENT = [
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]

# --- 1. CARGA DE ARTEFACTOS (JUGADORES + EQUIPOS) ---
@st.cache_resource
def load_all_artifacts():
    try:
        # --- A. MODELO JUGADORES ---
        model_p = load_model('nba_model_dvp.h5', compile=False)
        scaler_p = joblib.load('nba_scaler.pkl')
        pos_map = joblib.load('pos_map.pkl')
        dvp_stats = pd.read_csv('defense_vs_position.csv')
        
        # Datos Jugadores
        df_p = pd.read_csv('nba_data_final.csv')
        df_p['GAME_DATE'] = pd.to_datetime(df_p['GAME_DATE'])
        df_p = df_p.dropna(subset=['PLAYER_NAME'])
        df_p['PLAYER_NAME'] = df_p['PLAYER_NAME'].astype(str)
        
        # Filtro Minutos 2026
        inicio_26 = pd.to_datetime('2025-09-01')
        df_p26 = df_p[df_p['GAME_DATE'] >= inicio_26].copy()
        avg_min_26 = df_p26.groupby('PLAYER_NAME')['MIN'].mean().to_dict()

        # --- B. MODELO EQUIPOS ---
        model_t = load_model('nba_model_team.h5', compile=False)
        scaler_t = joblib.load('team_scaler.pkl')
        df_t = pd.read_csv('nba_teams_data.csv')
        df_t['GAME_DATE'] = pd.to_datetime(df_t['GAME_DATE'])

        return (model_p, scaler_p, pos_map, dvp_stats, df_p, avg_min_26, 
                model_t, scaler_t, df_t)
    
    except Exception as e:
        st.error(f"Error detallado cargando archivos: {e}")
        return None, None, None, None, None, None, None, None, None

# Carga de variables globales
loaded_data = load_all_artifacts()

if loaded_data is None or loaded_data[0] is None:
    st.error("⚠️ Faltan archivos. Asegúrate de tener en la carpeta: 'nba_model_team.h5', 'nba_teams_data.csv', etc.")
    st.stop()

# Desempaquetado seguro
(model_p, scaler_p, pos_map, dvp_stats, df_p, avg_min_26, 
 model_t, scaler_t, df_t) = loaded_data

# --- 2. FUNCIONES DE LÓGICA ---

# A. PREDICCIÓN DE EQUIPOS (MARCADOR)
def get_team_recent_stats(team_abbr):
    """Obtiene las stats recientes del equipo para predecir score"""
    team_data = df_t[df_t['TEAM_ABBREVIATION'] == team_abbr].sort_values('GAME_DATE')
    if team_data.empty: return None
    return team_data.tail(1)

def predecir_marcador(visita, local):
    # 1. Datos recientes
    row_v = get_team_recent_stats(visita)
    row_l = get_team_recent_stats(local)
    
    if row_v is None or row_l is None: return None, None
    
    # 2. Features (Mismo orden que entrenamiento equipos)
    stats = ['PTS', 'PLUS_MINUS', 'FG_PCT', 'FG3_PCT', 'AST', 'REB', 'TOV']
    feats = [f'TEAM_{s}_LAST_5' for s in stats]
    
    try:
        input_v = row_v[feats].copy()
        input_l = row_l[feats].copy()
    except: return None, None
    
    # 3. Localía
    input_v['IS_HOME'] = 0
    input_l['IS_HOME'] = 1
    
    # 4. Predecir
    final_cols = feats + ['IS_HOME']
    
    try:
        score_v = model_t.predict(scaler_t.transform(input_v[final_cols]), verbose=0)[0][0]
        score_l = model_t.predict(scaler_t.transform(input_l[final_cols]), verbose=0)[0][0]
        return score_v, score_l
    except: return None, None

# B. PREDICCIÓN DE JUGADORES
def get_defense_stats(rival, pos):
    stats = dvp_stats[(dvp_stats['OPPONENT_ABBREV'] == rival) & (dvp_stats['POSITION'] == pos)]
    if stats.empty: return 20.0, 5.0, 5.0
    return stats.iloc[0]['OPP_ALLOW_PTS'], stats.iloc[0]['OPP_ALLOW_REB'], stats.iloc[0]['OPP_ALLOW_AST']

def get_active_roster(team_abbr, min_minutes):
    try: team_id = df_p[df_p['TEAM_ABBREVIATION'] == team_abbr]['TEAM_ID'].iloc[0]
    except: return []
    
    team_df = df_p[df_p['TEAM_ID'] == team_id]
    if team_df.empty: return []
    
    roster = []
    for p in team_df['PLAYER_NAME'].unique():
        last_game = df_p[df_p['PLAYER_NAME'] == p].sort_values('GAME_DATE').iloc[-1]
        if last_game['TEAM_ID'] == team_id:
            if avg_min_26.get(p, 0) >= min_minutes:
                roster.append(p)
    return roster

def predict_player(name, rival, is_home):
    player_history = df_p[df_p['PLAYER_NAME'] == name].sort_values(by='GAME_DATE')
    if player_history.empty: return None
    
    last_5_games = player_history.tail(5).sort_values(by='GAME_DATE', ascending=False).copy()
    last_game_row = player_history.tail(1)
    
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    last_date = last_game_row['GAME_DATE'].values[0]
    days_rest = max(1, min((today - last_date).days, 7))
    is_b2b = 1 if days_rest == 1 else 0
    
    p_id = last_game_row['PLAYER_ID'].values[0]
    pos = pos_map.get(p_id, 'F')
    opp_stats = get_defense_stats(rival, pos)
    
    stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FG_PCT', 'FG3_PCT']
    feats = [f'AVG_{s}_LAST_5' for s in stats]
    trends = ['TREND_PTS', 'TREND_FG_PCT']
    
    try: row = last_game_row[feats].copy()
    except: return None
    for t in trends: row[t] = last_game_row[t].values[0] if t in last_game_row else 0
    
    row['IS_HOME'] = 1 if is_home else 0
    row['DAYS_REST'] = days_rest
    row['IS_B2B'] = is_b2b
    row['OPP_ALLOW_PTS'], row['OPP_ALLOW_REB'], row['OPP_ALLOW_AST'] = opp_stats
    
    final_cols = feats + trends + ['IS_HOME', 'DAYS_REST', 'IS_B2B', 'OPP_ALLOW_PTS', 'OPP_ALLOW_REB', 'OPP_ALLOW_AST']
    
    try:
        pred = model_p.predict(scaler_p.transform(row[final_cols]), verbose=0)
        
        # Recuperamos la tendencia de puntos
        trend_val = last_game_row.get('TREND_PTS', pd.Series([0])).values[0]
        
        # CÁLCULO DE INACTIVIDAD (>30 días)
        days_since_last = (today - last_date).days
        is_inactive = True if days_since_last > 30 else False
        
        return pred[0], pos, opp_stats[0], is_b2b, trend_val, last_5_games, p_id, is_inactive
    except: return None

# --- 3. BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/logoman.png", width=100)
    st.title("Leyenda")
    st.markdown("""
    🔥 **En Racha:** Promedio +3 pts reciente.
    
    🛡️ **Defensa Dura:** Rival élite (<16 pts).
    
    🚨 **Inactivo:** >30 días sin jugar.
    """)

# --- 4. INTERFAZ PRINCIPAL ---

st.title("🧠 NBA AI Simulator")

tab1, tab2 = st.tabs(["👤 Oráculo Jugador", "🏟️ Simulador de Partido"])

# --- PESTAÑA 1: JUGADOR ---
with tab1:
    c1, c2, c3 = st.columns(3)
    
    all_players = sorted(df_p['PLAYER_NAME'].unique())
    jug = c1.selectbox("Jugador", all_players)
    
    # Filtro: Solo equipos NBA actuales
    all_rivals = sorted(dvp_stats['OPPONENT_ABBREV'].unique())
    nba_rivals = [r for r in all_rivals if r in NBA_TEAMS_CURRENT]
    
    riv = c2.selectbox("Rival", nba_rivals)
    loc = c3.radio("Sede", ["Casa 🏠", "Visita ✈️"], key="p1")
    
    if st.button("Analizar Jugador"):
        res = predict_player(jug, riv, loc == "Casa 🏠")
        if res:
            (pts, reb, ast), pos, opp, b2b, trend, last_5, pid, is_inactive = res
            
            col_img, col_info = st.columns([1, 4])
            with col_img:
                img_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(pid)}.png"
                st.image(img_url, use_container_width=True)
            with col_info:
                st.subheader(f"{jug} ({pos}) vs {riv}")
                if is_inactive: st.error(f"🚨 ALERTA: Jugador inactivo ({int((datetime.now() - last_5.iloc[0]['GAME_DATE']).days)} días).")
                elif b2b: st.warning("⚠️ Back-to-Back")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Puntos", f"{pts:.1f}", delta=f"{trend:.1f}")
                m2.metric("Rebotes", f"{reb:.1f}")
                m3.metric("Asistencias", f"{ast:.1f}")
            
            st.divider()
            st.caption("Historial reciente:")
            st.dataframe(last_5[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST']], hide_index=True)
        else: st.error("Sin datos.")

# --- PESTAÑA 2: SIMULADOR COMPLETO ---
with tab2:
    st.write("Predicción de Resultado Final y Rendimiento de Jugadores.")
    min_filter = st.slider("Minutos Mínimos >", 10, 40, 25)
    st.markdown("---")
    
    # Selectores
    col_v, col_vs, col_l = st.columns([2, 1, 2])
    
    # Filtro Equipos
    all_teams_dataset = sorted(df_t['TEAM_ABBREVIATION'].unique())
    equipos_filtrados = [t for t in all_teams_dataset if t in NBA_TEAMS_CURRENT]
    
    with col_v:
        visita = st.selectbox("✈️ Visitante", equipos_filtrados, index=0)
    with col_vs:
        st.markdown("<h2 style='text-align: center; margin-top: 20px;'>VS</h2>", unsafe_allow_html=True)
    with col_l:
        idx_local = 1 if len(equipos_filtrados) > 1 else 0
        local = st.selectbox("🏠 Local", equipos_filtrados, index=idx_local)
        
    if st.button("🚀 SIMULAR ENFRENTAMIENTO", type="primary", use_container_width=True):
        if visita == local:
            st.error("Elige equipos diferentes.")
        else:
            # 1. PREDICCIÓN DE MARCADOR
            score_v, score_l = predecir_marcador(visita, local)
            
            if score_v is None:
                st.error("No hay datos de equipo suficientes.")
            else:
                ganador = local if score_l > score_v else visita
                diff = abs(score_l - score_v)
                total = score_l + score_v
                
                # Visualización Score
                c_res_v, c_res_mid, c_res_l = st.columns([2, 2, 2])
                
                with c_res_v:
                    st.markdown(f"<h1 style='text-align: center; color: gray;'>{score_v:.0f}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center;'>{visita}</div>", unsafe_allow_html=True)
                    
                with c_res_mid:
                    st.markdown(f"""
                    <div style="text-align: center; background-color: rgba(0, 200, 0, 0.1); padding: 10px; border-radius: 10px; border: 1px solid green;">
                        <span style="color: green; font-weight: bold;">GANA {ganador}</span><br>
                        <small>por {diff:.1f} pts</small>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"<div style='text-align: center; margin-top: 5px;'>Total: {total:.0f} pts</div>", unsafe_allow_html=True)
                    
                with c_res_l:
                    st.markdown(f"<h1 style='text-align: center; color: gray;'>{score_l:.0f}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center;'>{local}</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # 2. PREDICCIÓN DE JUGADORES
                col_res_v, col_res_l = st.columns(2)
                
                def render_side(col, team_name, rival_name, is_home, color_borde):
                    with col:
                        st.subheader(f"{team_name}")
                        roster = get_active_roster(team_name, min_filter)
                        if not roster: st.warning("Sin jugadores.")
                        
                        for p in roster:
                            res = predict_player(p, rival_name, is_home)
                            if res:
                                # Desempaquetado correcto (9 valores)
                                (pts, reb, ast), pos, opp, b2b, trend, _, _, is_inactive = res
                                
                                icons = ""
                                if is_inactive: icons += "🚨"
                                
                                if trend > 3: icons += "🔥"
                                if opp < 16: icons += "🛡️"
                                
                                bg_color = "rgba(255, 0, 0, 0.1)" if is_inactive else "rgba(128, 128, 128, 0.1)"
                                
                                st.markdown(f"""
                                <div style="
                                    background-color: {bg_color}; 
                                    padding: 8px; 
                                    border-radius: 5px; 
                                    margin-bottom: 8px; 
                                    border-left: 4px solid {color_borde};">
                                    <strong>{p}</strong> <small>({pos})</small> {icons}<br>
                                    <span style="font-size: 16px; font-weight: bold;">🏀 {pts:.1f}</span> | 🙌 {reb:.1f} | 🤝 {ast:.1f}
                                </div>
                                """, unsafe_allow_html=True)

                render_side(col_res_v, visita, local, False, "#cc0000")
                render_side(col_res_l, local, visita, True, "#0000cc")
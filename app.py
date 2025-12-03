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

# --- 1. CARGA DE ARTEFACTOS (SOLO JUGADORES) ---
@st.cache_resource
def load_artifacts():
    try:
        # Cargamos solo lo necesario para predecir jugadores
        # compile=False es vital para evitar errores de versión
        model = load_model('nba_model_dvp.h5', compile=False)
        scaler = joblib.load('nba_scaler.pkl')
        pos_map = joblib.load('pos_map.pkl')
        dvp_stats = pd.read_csv('defense_vs_position.csv')
        
        # Datos históricos
        df = pd.read_csv('nba_data_final.csv')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        # Limpieza de nombres nulos
        df = df.dropna(subset=['PLAYER_NAME'])
        df['PLAYER_NAME'] = df['PLAYER_NAME'].astype(str)
        
        # Filtro de minutos (Temporada 2025-26)
        inicio_26 = pd.to_datetime('2025-09-01')
        df_26 = df[df['GAME_DATE'] >= inicio_26].copy()
        avg_min_26 = df_26.groupby('PLAYER_NAME')['MIN'].mean().to_dict()
        
        return model, scaler, pos_map, dvp_stats, df, avg_min_26
    
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        return None, None, None, None, None, None

# Ejecutar carga
model, scaler, pos_map, dvp_stats, df, avg_min_26 = load_artifacts()

if df is None:
    st.warning("⚠️ Faltan archivos en la carpeta. Asegúrate de tener: nba_model_dvp.h5, nba_scaler.pkl, nba_data_final.csv, etc.")
    st.stop()

# --- 2. FUNCIONES DE LÓGICA ---

def get_defense_stats(rival, pos):
    """Busca qué tan bien defiende el rival a esa posición"""
    stats = dvp_stats[(dvp_stats['OPPONENT_ABBREV'] == rival) & (dvp_stats['POSITION'] == pos)]
    if stats.empty: return 20.0, 5.0, 5.0
    return stats.iloc[0]['OPP_ALLOW_PTS'], stats.iloc[0]['OPP_ALLOW_REB'], stats.iloc[0]['OPP_ALLOW_AST']

def get_active_roster(team_abbr, min_minutes):
    """Obtiene los jugadores activos de un equipo por su abreviatura (ej: LAL)"""
    # 1. Buscar el ID del equipo basado en la abreviatura
    try:
        team_id = df[df['TEAM_ABBREVIATION'] == team_abbr]['TEAM_ID'].iloc[0]
    except:
        return [] # Equipo no encontrado en el CSV
    
    # 2. Filtrar dataframe
    team_df = df[df['TEAM_ID'] == team_id]
    if team_df.empty: return []
    
    roster = []
    for p in team_df['PLAYER_NAME'].unique():
        # Validar que su último partido registrado sea con este equipo
        last_game = df[df['PLAYER_NAME'] == p].sort_values('GAME_DATE').iloc[-1]
        
        if last_game['TEAM_ID'] == team_id:
            # Validar minutos en la temporada actual (si tiene datos)
            if avg_min_26.get(p, 0) >= min_minutes:
                roster.append(p)
    return roster

def predict_player(name, rival, is_home):
    # 1. Historial
    player_history = df[df['PLAYER_NAME'] == name].sort_values(by='GAME_DATE')
    if player_history.empty: return None
    
    # Últimos 5 partidos (para mostrar en tabla si es necesario)
    last_5_games = player_history.tail(5).sort_values(by='GAME_DATE', ascending=False).copy()
    
    last_game_row = player_history.tail(1)
    
    # 2. Contexto (Fatiga simulada)
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    last_date = last_game_row['GAME_DATE'].values[0]
    
    # Cálculo de días de descanso (min 1, max 7 para normalizar)
    days_rest = max(1, min((today - last_date).days, 7))
    is_b2b = 1 if days_rest == 1 else 0
    
    # 3. Inputs
    p_id = last_game_row['PLAYER_ID'].values[0]
    pos = pos_map.get(p_id, 'F')
    opp_pts, opp_reb, opp_ast = get_defense_stats(rival, pos)
    
    # 4. Features (Mismo orden que entrenamiento)
    stats_base = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FG_PCT', 'FG3_PCT']
    feats_short = [f'AVG_{s}_LAST_5' for s in stats_base]
    trends = ['TREND_PTS', 'TREND_FG_PCT']
    
    try: row = last_game_row[feats_short].copy()
    except: return None
    
    # Añadir tendencias si existen en el CSV
    for t in trends: 
        row[t] = last_game_row[t].values[0] if t in last_game_row else 0

    row['IS_HOME'] = 1 if is_home else 0
    row['DAYS_REST'] = days_rest
    row['IS_B2B'] = is_b2b
    row['OPP_ALLOW_PTS'] = opp_pts
    row['OPP_ALLOW_REB'] = opp_reb
    row['OPP_ALLOW_AST'] = opp_ast
    
    # Orden final de columnas para la red neuronal
    final_cols = feats_short + trends + ['IS_HOME', 'DAYS_REST', 'IS_B2B', 'OPP_ALLOW_PTS', 'OPP_ALLOW_REB', 'OPP_ALLOW_AST']
    
    try:
        X_scaled = scaler.transform(row[final_cols])
        pred = model.predict(X_scaled, verbose=0)
        
        # Recuperamos la tendencia de puntos para mostrarla en la UI
        trend_val = last_game_row.get('TREND_PTS', pd.Series([0])).values[0]
        
        return pred[0], pos, opp_pts, is_b2b, trend_val, last_5_games, p_id
    except: return None

# --- 3. INTERFAZ GRÁFICA ---

st.title("🧠 NBA AI Simulator")

tab1, tab2 = st.tabs(["👤 Oráculo Individual", "🆚 Enfrentamiento de Equipos"])

# --- PESTAÑA 1: INDIVIDUAL ---
with tab1:
    c1, c2, c3 = st.columns(3)
    # Listas ordenadas
    all_players = sorted(df['PLAYER_NAME'].unique())
    all_teams = sorted(dvp_stats['OPPONENT_ABBREV'].unique())
    
    jug = c1.selectbox("Jugador", all_players)
    riv = c2.selectbox("Rival", all_teams)
    loc = c3.radio("Sede", ["Casa", "Visita"], key="p1")
    
    if st.button("Analizar Jugador"):
        res = predict_player(jug, riv, loc == "Casa")
        if res:
            (pts, reb, ast), pos, opp, b2b, trend, last_5, pid = res
            
            # Encabezado con Foto
            col_img, col_info = st.columns([1, 4])
            with col_img:
                img_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(pid)}.png"
                st.image(img_url, use_container_width=True)
            with col_info:
                st.subheader(f"{jug} ({pos}) vs {riv}")
                if b2b: st.warning("⚠️ Juega cansado (Back-to-Back)")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Puntos", f"{pts:.1f}", delta=f"{trend:.1f}")
                m2.metric("Rebotes", f"{reb:.1f}")
                m3.metric("Asistencias", f"{ast:.1f}")
            
            st.divider()
            st.caption("Historial reciente (Últimos 5):")
            # Mostramos tabla limpia
            st.dataframe(last_5[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST']], hide_index=True)
        else:
            st.error("Datos insuficientes para este jugador.")

# --- PESTAÑA 2: ENFRENTAMIENTO (MANUAL) ---
with tab2:
    st.write("Simula el rendimiento de los jugadores en un partido específico.")
    
    # 1. Filtro de Minutos
    min_filter = st.slider("Filtrar jugadores con media de minutos >", 10, 40, 25)
    st.markdown("---")
    
    # 2. Selectores de Equipos
    col_v, col_vs, col_l = st.columns([2, 1, 2])
    equipos = sorted(df['TEAM_ABBREVIATION'].unique())
    
    with col_v:
        visita = st.selectbox("✈️ Visitante", equipos, index=0)
    with col_vs:
        st.markdown("<h2 style='text-align: center; margin-top: 20px;'>VS</h2>", unsafe_allow_html=True)
    with col_l:
        local = st.selectbox("🏠 Local", equipos, index=1)
        
    # 3. Botón de Acción
    st.markdown("---")
    if st.button("🚀 CALCULAR ENFRENTAMIENTO", type="primary", use_container_width=True):
        if visita == local:
            st.error("Elige equipos diferentes.")
        else:
            col_res_v, col_res_l = st.columns(2)
            
            # Función auxiliar para renderizar cada lado
            def render_side(col, team_name, rival_name, is_home, color_borde):
                with col:
                    st.header(f"{team_name}")
                    roster = get_active_roster(team_name, min_filter)
                    
                    if not roster:
                        st.warning("No hay jugadores que cumplan el filtro.")
                    
                    for p in roster:
                        # El jugador 'p' juega contra 'rival_name'
                        res = predict_player(p, rival_name, is_home)
                        if res:
                            (pts, reb, ast), pos, opp, b2b, trend, _, _ = res
                            
                            # Iconos
                            icon_b2b = "💤" if b2b else ""
                            icon_fire = "🔥" if trend > 3 else ""
                            icon_shield = "🛡️" if opp < 16 else ""
                            
                            # Diseño de Tarjeta (Compatible con Modo Oscuro)
                            st.markdown(f"""
                            <div style="
                                background-color: rgba(128, 128, 128, 0.1); 
                                padding: 10px; 
                                border-radius: 5px; 
                                margin-bottom: 10px; 
                                border-left: 4px solid {color_borde};">
                                <strong>{p}</strong> <small>({pos})</small> {icon_b2b}{icon_fire}{icon_shield}<br>
                                <span style="font-size: 18px; font-weight: bold;">🏀 {pts:.1f}</span> | 🙌 {reb:.1f} | 🤝 {ast:.1f}
                            </div>
                            """, unsafe_allow_html=True)

            # Renderizar Visitante (Borde Rojo)
            render_side(col_res_v, visita, local, False, "#cc0000")
            
            # Renderizar Local (Borde Azul)
            render_side(col_res_l, local, visita, True, "#0000cc")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import requests
from groq import Groq
from tensorflow.keras.models import load_model
from datetime import datetime
from tavily import TavilyClient
import time

# --- 0. CONFIGURACIÓN VISUAL Y CSS ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="NBA AI Simulator Ultimate", page_icon="🏀", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    h1, h2, h3 { font-family: 'Arial Black', sans-serif; color: #ff4b4b; }
    
    .player-card {
        background: linear-gradient(145deg, #1f2937, #111827);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    
    /* ESTILOS DE APUESTAS CORREGIDOS */
    .odds-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .odds-header {
        font-size: 1.1em;
        font-weight: bold;
        color: white;
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .comparison-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0f172a;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        font-family: sans-serif;
    }
    .edge-badge {
        background-color: #10b981;
        color: #000;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 0.85em;
        box-shadow: 0 0 5px rgba(16, 185, 129, 0.5);
    }
    .no-edge {
        color: #64748b;
        font-size: 0.8em;
        font-style: italic;
    }
    
    .stButton>button { border-radius: 8px; font-weight: bold; border: none; height: 50px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { background-color: #1f2937; color: white; border-radius: 4px 4px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b; color: white; }
    .metric-value { font-size: 1.3rem; font-weight: 800; color: #fff; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTES Y MAPEOS ---
NBA_TEAMS_CURRENT = [
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]

# DICCIONARIO MAESTRO: NOMBRE API -> ABREVIATURA MODELO
NAME_TO_ABBR = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

TEAM_LOGOS = {
    'ATL': 'https://a.espncdn.com/i/teamlogos/nba/500/atl.png',
    'BOS': 'https://a.espncdn.com/i/teamlogos/nba/500/bos.png',
    'BKN': 'https://a.espncdn.com/i/teamlogos/nba/500/bkn.png',
    'CHA': 'https://a.espncdn.com/i/teamlogos/nba/500/cha.png',
    'CHI': 'https://a.espncdn.com/i/teamlogos/nba/500/chi.png',
    'CLE': 'https://a.espncdn.com/i/teamlogos/nba/500/cle.png',
    'DAL': 'https://a.espncdn.com/i/teamlogos/nba/500/dal.png',
    'DEN': 'https://a.espncdn.com/i/teamlogos/nba/500/den.png',
    'DET': 'https://a.espncdn.com/i/teamlogos/nba/500/det.png',
    'GSW': 'https://a.espncdn.com/i/teamlogos/nba/500/gs.png',
    'HOU': 'https://a.espncdn.com/i/teamlogos/nba/500/hou.png',
    'IND': 'https://a.espncdn.com/i/teamlogos/nba/500/ind.png',
    'LAC': 'https://a.espncdn.com/i/teamlogos/nba/500/lac.png',
    'LAL': 'https://a.espncdn.com/i/teamlogos/nba/500/lal.png',
    'MEM': 'https://a.espncdn.com/i/teamlogos/nba/500/mem.png',
    'MIA': 'https://a.espncdn.com/i/teamlogos/nba/500/mia.png',
    'MIL': 'https://a.espncdn.com/i/teamlogos/nba/500/mil.png',
    'MIN': 'https://a.espncdn.com/i/teamlogos/nba/500/min.png',
    'NOP': 'https://a.espncdn.com/i/teamlogos/nba/500/no.png',
    'NYK': 'https://a.espncdn.com/i/teamlogos/nba/500/ny.png',
    'OKC': 'https://a.espncdn.com/i/teamlogos/nba/500/okc.png',
    'ORL': 'https://a.espncdn.com/i/teamlogos/nba/500/orl.png',
    'PHI': 'https://a.espncdn.com/i/teamlogos/nba/500/phi.png',
    'PHX': 'https://a.espncdn.com/i/teamlogos/nba/500/phx.png',
    'POR': 'https://a.espncdn.com/i/teamlogos/nba/500/por.png',
    'SAC': 'https://a.espncdn.com/i/teamlogos/nba/500/sac.png',
    'SAS': 'https://a.espncdn.com/i/teamlogos/nba/500/sas.png',
    'TOR': 'https://a.espncdn.com/i/teamlogos/nba/500/tor.png',
    'UTA': 'https://a.espncdn.com/i/teamlogos/nba/500/utah.png',
    'WAS': 'https://a.espncdn.com/i/teamlogos/nba/500/wsh.png'
}

# --- 1. CARGA DE ARTEFACTOS ---
@st.cache_resource
def load_all_artifacts():
    try:
        model_p = load_model('nba_model_dvp.h5', compile=False)
        scaler_p = joblib.load('nba_scaler.pkl')
        pos_map = joblib.load('pos_map.pkl')
        dvp_stats = pd.read_csv('defense_vs_position.csv')
        df_p = pd.read_csv('nba_data_final.csv')
        df_p['GAME_DATE'] = pd.to_datetime(df_p['GAME_DATE'])
        df_p = df_p.dropna(subset=['PLAYER_NAME'])
        df_p['PLAYER_NAME'] = df_p['PLAYER_NAME'].astype(str)
        model_t = load_model('nba_model_team.h5', compile=False)
        scaler_t = joblib.load('team_scaler.pkl')
        try: df_t = pd.read_csv('nba_teams_data_ready.csv')
        except: df_t = pd.read_csv('nba_teams_data.csv')
        df_t['GAME_DATE'] = pd.to_datetime(df_t['GAME_DATE'])
        return (model_p, scaler_p, pos_map, dvp_stats, df_p, model_t, scaler_t, df_t)
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        st.stop()

(model_p, scaler_p, pos_map, dvp_stats, df_p, model_t, scaler_t, df_t) = load_all_artifacts()

# --- 2. LÓGICA DE IA & APIS ---

def ask_groq(prompt, api_key):
    if not api_key: return "⚠️ Falta Groq API Key"
    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
        )
        return chat_completion.choices[0].message.content
    except Exception as e: return f"Error Groq: {e}"

def buscar_lesiones_tavily(equipo, key):
    if not key: return "⚠️ Falta Tavily Key"
    try:
        client = TavilyClient(api_key=key)
        hoy = datetime.now().strftime('%Y-%m-%d')
        query = f"NBA injury report {equipo} news today {hoy} ESPN CBS"
        response = client.search(query, search_depth="basic", max_results=2)
        texto = f"\n--- REPORTE {equipo} ---\n"
        for r in response.get('results', []):
            texto += f"📰 {r['title']}: {r['content']}\n"
        return texto
    except Exception as e: return f"Error buscando: {e}"

def consultar_ia_partido(local, visita, j_local, j_visita, groq_key, tavily_key):
    info_l = buscar_lesiones_tavily(local, tavily_key)
    info_v = buscar_lesiones_tavily(visita, tavily_key)
    prompt = f"""
    Analista NBA Experto. Partido HOY: {local} vs {visita}.
    NOTICIAS: {info_l} {info_v}
    PLANTILLAS: {j_local} vs {j_visita}
    TAREA: Lista BAJAS confirmadas para HOY y quién gana ventaja. Breve.
    """
    return ask_groq(prompt, groq_key)

def obtener_cuotas_nba(api_key):
    """Obtiene cuotas reales y las prepara para comparación"""
    if not api_key: return None, "⚠️ Falta The-Odds-API Key"
    
    url = f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=decimal&apiKey={api_key}'
    
    try:
        response = requests.get(url)
        if response.status_code != 200: return None, f"Error API: {response.text}"
        data = response.json()
        if not data: return None, "No hay partidos disponibles."
            
        partidos = []
        for game in data:
            home = game['home_team']
            away = game['away_team']
            
            bookmakers = game.get('bookmakers', [])
            if not bookmakers: continue
            
            bookie = next((b for b in bookmakers if b['key'] == 'draftkings'), bookmakers[0])
            
            odds = {
                'home_name': home, 'away_name': away,
                'home_abbr': NAME_TO_ABBR.get(home),
                'away_abbr': NAME_TO_ABBR.get(away),
                'bookie': bookie['title'],
                'total_line': None, 'spread_line': None, 'spread_team': None
            }
            
            for market in bookie.get('markets', []):
                if market['key'] == 'totals':
                    odds['total_line'] = market['outcomes'][0]['point']
                elif market['key'] == 'spreads':
                    odds['spread_line'] = market['outcomes'][0]['point']
                    odds['spread_team'] = market['outcomes'][0]['name']
            
            partidos.append(odds)
        return partidos, None
    except Exception as e: return None, str(e)

# --- 3. FUNCIONES MATEMÁTICAS ---
def get_team_recent_stats(team_abbr):
    team_data = df_t[df_t['TEAM_ABBREVIATION'] == team_abbr].sort_values('GAME_DATE')
    if team_data.empty: return None
    return team_data.tail(1)

def predecir_marcador(visita, local):
    row_v = get_team_recent_stats(visita)
    row_l = get_team_recent_stats(local)
    if row_v is None or row_l is None: return None, None, None, None
    stats = ['PTS', 'PLUS_MINUS', 'FG_PCT', 'FG3_PCT', 'AST', 'REB', 'TOV']
    feats = [f'TEAM_{s}_LAST_5' for s in stats]
    try:
        iv = row_v[feats].copy(); iv['IS_HOME'] = 0
        il = row_l[feats].copy(); il['IS_HOME'] = 1
        cols = feats + ['IS_HOME']
        sv = float(model_t.predict(scaler_t.transform(iv[cols]), verbose=0)[0][0])
        sl = float(model_t.predict(scaler_t.transform(il[cols]), verbose=0)[0][0])
        return sv, sl, row_v, row_l
    except: return None, None, None, None

def get_defense_stats(rival, pos):
    s = dvp_stats[(dvp_stats['OPPONENT_ABBREV'] == rival) & (dvp_stats['POSITION'] == pos)]
    if s.empty: return 20.0, 5.0, 5.0
    return s.iloc[0]['OPP_ALLOW_PTS'], s.iloc[0]['OPP_ALLOW_REB'], s.iloc[0]['OPP_ALLOW_AST']

def get_active_roster(team_abbr, min_minutes):
    try: team_id = df_p[df_p['TEAM_ABBREVIATION'] == team_abbr]['TEAM_ID'].iloc[0]
    except: return []
    team_df = df_p[df_p['TEAM_ID'] == team_id]
    if team_df.empty: return []
    last_date = team_df['GAME_DATE'].max()
    start_window = last_date - pd.Timedelta(days=90)
    active_period = team_df[team_df['GAME_DATE'] >= start_window]
    if active_period.empty: active_period = team_df
    avg_minutes = active_period.groupby('PLAYER_NAME')['MIN'].mean()
    roster = avg_minutes[avg_minutes >= min_minutes].index.tolist()
    return roster

def get_player_vs_rival(player_name, rival_abbr):
    ph = df_p[(df_p['PLAYER_NAME'] == player_name) & (df_p['MATCHUP'].str.contains(rival_abbr, case=False))]
    if ph.empty: return pd.DataFrame()
    return ph.sort_values('GAME_DATE', ascending=False).head(5)

def predict_player(name, rival, is_home):
    ph = df_p[df_p['PLAYER_NAME'] == name].sort_values('GAME_DATE')
    if ph.empty: return None
    last = ph.tail(1)
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    last_date = last['GAME_DATE'].values[0]
    ds = (today - last_date).days
    inactive = True if ds > 30 else False
    pos = pos_map.get(last['PLAYER_ID'].values[0], 'F')
    opp = get_defense_stats(rival, pos)
    stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FG_PCT', 'FG3_PCT']
    feats = [f'AVG_{s}_LAST_5' for s in stats]
    trends = ['TREND_PTS', 'TREND_FG_PCT']
    extra_cols = ['IS_HOME', 'DAYS_REST', 'IS_B2B', 'OPP_ALLOW_PTS', 'OPP_ALLOW_REB', 'OPP_ALLOW_AST']
    all_cols = feats + trends + extra_cols
    try:
        row = last.copy()
        row['IS_HOME'] = 1 if is_home else 0
        row['DAYS_REST'] = 2; row['IS_B2B'] = 0
        row['OPP_ALLOW_PTS'] = opp[0]; row['OPP_ALLOW_REB'] = opp[1]; row['OPP_ALLOW_AST'] = opp[2]
        missing = [c for c in all_cols if c not in row.columns]
        for c in missing: row[c] = 0
        X = row[all_cols]
        pred = model_p.predict(scaler_p.transform(X), verbose=0)
        pred_pts = float(pred[0][0])
        pred_reb = float(pred[0][1])
        pred_ast = float(pred[0][2])
        tr = last.get('TREND_PTS', pd.Series([0])).values[0]
        l5 = ph.tail(5).sort_values(by='GAME_DATE', ascending=False)
        pid = last['PLAYER_ID'].values[0]
        return (pred_pts, pred_reb, pred_ast), pos, opp, float(tr), l5, pid, inactive, ds
    except Exception as e: return None

# --- 4. INTERFAZ GRÁFICA (UI) ---

with st.sidebar:
    st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/logoman.png", width=80)
    st.markdown("### 🔑 Credenciales")
    k_groq = st.text_input("Groq API Key", type="password")
    k_tavily = st.text_input("Tavily Key", type="password")
    k_odds = st.text_input("The-Odds-API Key", type="password")
    st.markdown("### ⚙️ Filtros")
    min_min = st.slider("Minutos Mínimos", 10, 40, 20)

st.markdown("<h1 style='text-align: center;'>🏀 NBA AI PREDICTOR</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🏆 Simulador de Partido", "👤 Analizador de Jugador", "💰 Comparador de Apuestas (Edge)"])

# === PESTAÑA 1 ===
with tab1:
    col_v, col_vs, col_l = st.columns([2, 1, 2])
    with col_v:
        visita = st.selectbox("Visitante", NBA_TEAMS_CURRENT, index=0)
        st.image(TEAM_LOGOS.get(visita, ''), width=100)
    with col_vs: st.markdown("<br><h1 style='text-align: center;'>VS</h1>", unsafe_allow_html=True)
    with col_l:
        local = st.selectbox("Local", NBA_TEAMS_CURRENT, index=1)
        st.image(TEAM_LOGOS.get(local, ''), width=100)

    c_run, c_ai = st.columns([3, 1])
    simular = c_run.button("🚀 SIMULAR PARTIDO", type="primary", use_container_width=True)
    consultar = c_ai.button("🧠 Consultar IA (Groq)", use_container_width=True)

    if simular:
        sv, sl, row_v, row_l = predecir_marcador(visita, local)
        if sv:
            st.markdown("---")
            ganador = local if sl > sv else visita
            color_v = "#ff4b4b" if visita == ganador else "#6b7280"
            color_l = "#ff4b4b" if local == ganador else "#6b7280"
            
            c1, c2, c3 = st.columns([2,3,2])
            with c1: st.markdown(f"<h1 style='text-align:center; color:{color_v}; font-size:4em'>{sv:.0f}</h1>", unsafe_allow_html=True)
            with c2: 
                st.markdown(f"<div style='text-align:center; margin-top:20px'>GANA <b>{ganador}</b></div>", unsafe_allow_html=True)
                st.progress(float(sl / (sv+sl)))
            with c3: st.markdown(f"<h1 style='text-align:center; color:{color_l}; font-size:4em'>{sl:.0f}</h1>", unsafe_allow_html=True)
            
            st.markdown("#### 📊 Comparativa (Últimos 5)")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Puntos", f"{row_v['TEAM_PTS_LAST_5'].values[0]:.1f}")
                st.metric("Rebotes", f"{row_v['TEAM_REB_LAST_5'].values[0]:.1f}")
            with cols[1]: st.markdown("<h3 style='text-align:center'>VS</h3>", unsafe_allow_html=True)
            with cols[2]:
                st.metric("Puntos", f"{row_l['TEAM_PTS_LAST_5'].values[0]:.1f}", delta=f"{row_l['TEAM_PTS_LAST_5'].values[0] - row_v['TEAM_PTS_LAST_5'].values[0]:.1f}")
                st.metric("Rebotes", f"{row_l['TEAM_REB_LAST_5'].values[0]:.1f}", delta=f"{row_l['TEAM_REB_LAST_5'].values[0] - row_v['TEAM_REB_LAST_5'].values[0]:.1f}")

            st.markdown("---")
            st.markdown("### 🔥 Proyección de Jugadores")
            
            def render_expandable_cards(team, rival, is_home):
                roster = get_active_roster(team, min_min)
                if not roster: st.error(f"Sin datos para {team}."); return
                
                for p in roster:
                    res = predict_player(p, rival, is_home)
                    if res:
                        (pts, reb, ast), pos, (opp_pts, opp_reb, opp_ast), trend, l5, pid, inact, _ = res
                        fire = "🔥" if trend > 3 else ""
                        alert = "🚨 Inactivo" if inact else ""
                        dvp_color = "green" if opp_pts > 22 else ("red" if opp_pts < 18 else "gray")
                        
                        label = f"{p} ({pos}) | 🏀 {pts:.1f} PTS | {fire} {alert}"
                        
                        with st.expander(label):
                            c_img, c_stats = st.columns([1, 3])
                            with c_img:
                                st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(pid)}.png", width=100)
                                st.caption(f"Trend: {trend:.1f}")
                            with c_stats:
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Puntos", f"{pts:.1f}")
                                m2.metric("Rebotes", f"{reb:.1f}")
                                m3.metric("Asistencias", f"{ast:.1f}")
                                st.markdown(f"**Defensa {rival} vs {pos}:** <span style='color:{dvp_color}'>{opp_pts:.1f} pts</span>", unsafe_allow_html=True)
                            
                            t_stats, t_vs = st.tabs(["📅 Últimos 5", "⚔️ Vs Rival"])
                            with t_stats: st.dataframe(l5[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'MIN']], hide_index=True)
                            with t_vs:
                                vs_rival_df = get_player_vs_rival(p, rival)
                                if not vs_rival_df.empty: st.dataframe(vs_rival_df[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST']], hide_index=True)
                                else: st.info(f"Sin historial vs {rival}.")

            c1, c2 = st.columns(2)
            with c1: 
                st.markdown(f"<h3 style='text-align:center'>{visita}</h3>", unsafe_allow_html=True)
                render_expandable_cards(visita, local, False)
            with c2: 
                st.markdown(f"<h3 style='text-align:center'>{local}</h3>", unsafe_allow_html=True)
                render_expandable_cards(local, visita, True)

    if consultar:
        if not (k_groq and k_tavily): st.error("⚠️ Faltan API Keys")
        else:
            r_l = ', '.join(get_active_roster(local, 25)[:3])
            r_v = ', '.join(get_active_roster(visita, 25)[:3])
            with st.spinner("🤖 Analizando con Groq..."):
                st.info(consultar_ia_partido(local, visita, r_l, r_v, k_groq, k_tavily))

# === PESTAÑA 2 ===
with tab2:
    col1, col2, col3 = st.columns(3)
    p_name = col1.selectbox("Jugador", sorted(df_p['PLAYER_NAME'].unique()))
    p_riv = col2.selectbox("Rival", sorted(dvp_stats['OPPONENT_ABBREV'].unique()))
    p_loc = col3.radio("Condición", ["Casa 🏠", "Visita ✈️"])
    if st.button("🔮 Predecir"):
        res = predict_player(p_name, p_riv, p_loc == "Casa 🏠")
        if res:
            (pts, reb, ast), pos, _, trend, l5, pid, inact, d_out = res
            cpic, cdata = st.columns([1, 3])
            with cpic: st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(pid)}.png")
            with cdata:
                st.markdown(f"## {p_name}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Puntos", f"{pts:.1f}", f"{trend:.1f}")
                m2.metric("Rebotes", f"{reb:.1f}")
                m3.metric("Asistencias", f"{ast:.1f}")
            st.markdown("### Historial vs Rival")
            vs_rival = get_player_vs_rival(p_name, p_riv)
            if not vs_rival.empty: st.dataframe(vs_rival[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST']], hide_index=True)
            else: st.info("Sin historial reciente.")

# === PESTAÑA 3: COMPARADOR IA VS VEGAS (CORREGIDO) ===
with tab3:
    st.markdown("## 💰 Detector de Valor (Edge)")
    st.caption("Compara la predicción de tu IA contra las líneas de Las Vegas en tiempo real.")
    
    if st.button("🔄 Buscar Cuotas y Calcular Edge", type="primary"):
        if not k_odds:
            st.error("⚠️ Falta The-Odds-API Key en la barra lateral.")
        else:
            with st.spinner("📡 Obteniendo cuotas y ejecutando simulaciones IA..."):
                partidos, error = obtener_cuotas_nba(k_odds)
                
                if error:
                    st.error(error)
                else:
                    for p in partidos:
                        ai_total_txt = "N/A"
                        edge_badge = ""
                        
                        if p['home_abbr'] and p['away_abbr']:
                            sv, sl, _, _ = predecir_marcador(p['away_abbr'], p['home_abbr'])
                            if sv and sl:
                                ai_total = sv + sl
                                ai_total_txt = f"{ai_total:.1f}"
                                
                                if p['total_line']:
                                    diff = ai_total - p['total_line']
                                    if abs(diff) > 4:
                                        rec = "OVER ⬆️" if diff > 0 else "UNDER ⬇️"
                                        edge_badge = f"<span class='edge-badge'>✅ {rec} ({diff:+.1f})</span>"
                                    else:
                                        edge_badge = "<span class='no-edge'>Sin ventaja</span>"

                        html_odds = f"""<div class="odds-card"><div class="odds-header"><span>{p['away_name']} vs {p['home_name']}</span><span style="font-size:0.8em; color:#10b981">{p['bookie']}</span></div><div class="comparison-row"><span>🔢 <b>Total (O/U)</b></span>{edge_badge}</div><div class="comparison-row" style="background:#1e293b"><span>🏛️ Vegas: <b>{p['total_line']}</b></span><span>🤖 IA Predice: <b>{ai_total_txt}</b></span></div></div>"""
                        
                        st.markdown(html_odds, unsafe_allow_html=True)
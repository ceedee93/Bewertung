import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO, BytesIO
import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Energie-Beschaffungs-Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2rem; font-weight: 700; color: #1a5276; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.1rem; color: #5d6d7e; margin-bottom: 1.5rem;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;
    }
    .metric-card-red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px; border-radius: 8px 8px 0 0;
        font-weight: 600; font-size: 0.95rem;
    }
    div[data-testid="stMetricValue"] {font-size: 1.8rem;}
    .recommendation-box {
        background: #eaf2f8; border-left: 5px solid #2e86c1;
        padding: 1rem 1.5rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INITIALISIERUNG
# ─────────────────────────────────────────────
defaults = {
    "lastgang_df": None, "spot_df": None, "forward_df": None,
    "lastgang_cols": {}, "spot_cols": {}, "forward_cols": {},
    "backtesting_results": None, "strategies_config": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def parse_input(uploaded_file=None, pasted_text=None):
    """Parst CSV/XLSX Upload oder eingefügten Text."""
    df = None
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, sep=None, engine="python", parse_dates=True)
            except Exception:
                df = pd.read_csv(uploaded_file, sep=";", decimal=",", parse_dates=True)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, parse_dates=True)
    elif pasted_text and pasted_text.strip():
        text = pasted_text.strip()
        # Auto-detect delimiter
        for sep in ["\t", ";", ",", "|"]:
            try:
                df = pd.read_csv(StringIO(text), sep=sep, engine="python", parse_dates=True)
                if len(df.columns) > 1:
                    break
            except Exception:
                continue
    return df


def map_columns(df, key_prefix, required_cols):
    """Interaktive Spaltenzuordnung."""
    st.markdown("**🔧 Spaltenzuordnung**")
    mapping = {}
    cols = ["-- Nicht zugeordnet --"] + list(df.columns)

    # Auto-detect
    auto_map = {}
    for req_col, keywords in required_cols.items():
        for c in df.columns:
            c_lower = str(c).lower()
            for kw in keywords:
                if kw in c_lower:
                    auto_map[req_col] = c
                    break
            if req_col in auto_map:
                break

    col_layout = st.columns(min(len(required_cols), 4))
    for i, (req_col, keywords) in enumerate(required_cols.items()):
        with col_layout[i % len(col_layout)]:
            default_idx = 0
            if req_col in auto_map:
                try:
                    default_idx = cols.index(auto_map[req_col])
                except ValueError:
                    default_idx = 0
            selected = st.selectbox(
                f"📌 {req_col}", cols,
                index=default_idx,
                key=f"{key_prefix}_{req_col}"
            )
            if selected != "-- Nicht zugeordnet --":
                mapping[req_col] = selected
    return mapping


def apply_mapping(df, mapping):
    """Wendet Spaltenzuordnung an und gibt bereinigten DataFrame zurück."""
    rename_map = {v: k for k, v in mapping.items()}
    df_mapped = df[list(mapping.values())].rename(columns=rename_map).copy()
    # Parse datetime
    if "Zeitstempel" in df_mapped.columns:
        df_mapped["Zeitstempel"] = pd.to_datetime(df_mapped["Zeitstempel"], dayfirst=True, errors="coerce")
        df_mapped = df_mapped.dropna(subset=["Zeitstempel"]).sort_values("Zeitstempel").reset_index(drop=True)
    if "Datum" in df_mapped.columns:
        df_mapped["Datum"] = pd.to_datetime(df_mapped["Datum"], dayfirst=True, errors="coerce")
        df_mapped = df_mapped.dropna(subset=["Datum"]).sort_values("Datum").reset_index(drop=True)
    # Numerische Spalten
    for col in df_mapped.columns:
        if col not in ["Zeitstempel", "Datum", "Produkt"]:
            df_mapped[col] = pd.to_numeric(
                df_mapped[col].astype(str).str.replace(",", ".").str.strip(), errors="coerce"
            )
    return df_mapped


def calculate_strategy_costs(spot_prices, forward_price, volume_mwh, quota, strategy_type, params=None):
    """Berechnet Beschaffungskosten für eine gegebene Strategie."""
    n = len(spot_prices)
    if n == 0:
        return {}

    fixed_volume = volume_mwh * quota
    spot_volume = volume_mwh * (1 - quota)

    # Spot-Kosten
    spot_cost = spot_prices.values * (spot_volume / n) if n > 0 else np.zeros(n)

    # Fixierungs-Zeitpunkte je nach Strategie
    if strategy_type == "Gleichmäßig":
        # Gleichmäßige Tranchen über den gesamten Zeitraum
        fixed_cost_per_period = (fixed_volume * forward_price) / n
        fixed_costs = np.full(n, fixed_cost_per_period)

    elif strategy_type == "Frontloaded":
        # 70% in erster Hälfte, 30% in zweiter
        weights = np.zeros(n)
        mid = n // 2
        if mid > 0:
            weights[:mid] = 0.7 / mid
        if (n - mid) > 0:
            weights[mid:] = 0.3 / (n - mid)
        fixed_costs = weights * fixed_volume * forward_price

    elif strategy_type == "Backloaded":
        # 30% in erster Hälfte, 70% in zweiter
        weights = np.zeros(n)
        mid = n // 2
        if mid > 0:
            weights[:mid] = 0.3 / mid
        if (n - mid) > 0:
            weights[mid:] = 0.7 / (n - mid)
        fixed_costs = weights * fixed_volume * forward_price

    elif strategy_type == "Regelbasiert":
        # Fixiere wenn Spotpreis unter Schwelle
        threshold = params.get("threshold", forward_price * 0.95) if params else forward_price * 0.95
        below = spot_prices.values < threshold
        n_below = below.sum()
        if n_below > 0:
            weights = np.where(below, 1.0 / n_below, 0.0)
        else:
            weights = np.full(n, 1.0 / n)
        fixed_costs = weights * fixed_volume * forward_price

    elif strategy_type == "100% Spot":
        fixed_costs = np.zeros(n)
        spot_cost = spot_prices.values * (volume_mwh / n)

    elif strategy_type == "100% Termin":
        fixed_costs = np.full(n, (volume_mwh * forward_price) / n)
        spot_cost = np.zeros(n)

    else:
        fixed_costs = np.full(n, (fixed_volume * forward_price) / n)

    total_costs = spot_cost + fixed_costs
    cumulative = np.cumsum(total_costs)

    return {
        "spot_cost": spot_cost,
        "fixed_costs": fixed_costs,
        "total_costs": total_costs,
        "cumulative": cumulative,
        "total": cumulative[-1] if len(cumulative) > 0 else 0,
        "avg_price": cumulative[-1] / volume_mwh if volume_mwh > 0 else 0,
    }


def calculate_risk_metrics(cost_series):
    """Berechnet Risikokennzahlen."""
    if len(cost_series) < 2:
        return {"Volatilität": 0, "VaR_95": 0, "CVaR_95": 0, "Max_Drawdown": 0}

    returns = np.diff(cost_series) / (cost_series[:-1] + 1e-10)
    vol = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
    var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    # Max Drawdown
    cummax = np.maximum.accumulate(cost_series)
    drawdown = (cost_series - cummax) / (cummax + 1e-10)
    max_dd = drawdown.min() if len(drawdown) > 0 else 0

    return {
        "Volatilität": vol,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "Max_Drawdown": max_dd
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Energie-Engine")
    st.markdown("---")
    st.markdown("### 📊 Status")

    status_items = {
        "Lastgang": st.session_state.lastgang_df is not None,
        "Spotpreise": st.session_state.spot_df is not None,
        "Forwardpreise": st.session_state.forward_df is not None,
    }
    for name, loaded in status_items.items():
        icon = "✅" if loaded else "⬜"
        st.markdown(f"{icon} {name}")

    st.markdown("---")
    st.markdown("### ⚙️ Globale Parameter")

    currency = st.selectbox("Währung", ["EUR", "USD", "CHF"], index=0)
    unit = st.selectbox("Preiseinheit", ["€/MWh", "ct/kWh", "€/kWh"], index=0)

    st.markdown("---")
    st.markdown("### 📋 Anleitung")
    st.markdown("""
    1. **Daten importieren** — Lastgang & Preise
    2. **Spalten zuordnen** — Automatisch oder manuell
    3. **Strategien konfigurieren**
    4. **Backtesting starten**
    5. **Ergebnisse analysieren & exportieren**
    """)

    st.markdown("---")
    st.caption("v1.0 | Energie-Beschaffungs-Engine")

# ─────────────────────────────────────────────
# HAUPTBEREICH — TABS
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">⚡ Energie-Beschaffungs-Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Simulation · Backtesting · Bewertung von Beschaffungsstrategien</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📥 Datenimport", "📈 Marktdaten", "🔧 Strategien",
    "🔄 Backtesting", "📊 Ergebnisse", "💾 Export"
])

# ═════════════════════════════════════════════
# TAB 1: DATENIMPORT
# ═════════════════════════════════════════════
with tab1:
    st.markdown("### 📥 Datenimport & Spaltenzuordnung")

    import_tabs = st.tabs(["⚡ Lastgang", "📉 Spotpreise", "📊 Forwardpreise"])

    # --- Lastgang ---
    with import_tabs[0]:
        st.markdown("#### ⚡ Lastgang / Verbrauchsprofil")
        input_method_lg = st.radio("Eingabemethode", ["Datei-Upload", "Copy & Paste"], key="lg_method", horizontal=True)

        lg_df_raw = None
        if input_method_lg == "Datei-Upload":
            lg_file = st.file_uploader("Lastgang hochladen (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="lg_upload")
            if lg_file:
                lg_df_raw = parse_input(uploaded_file=lg_file)
        else:
            lg_text = st.text_area(
                "Daten einfügen (Tab/Semikolon/Komma-getrennt)",
                height=200, key="lg_paste",
                placeholder="Zeitstempel;Verbrauch_kW\n01.01.2024 00:00;1250\n01.01.2024 00:15;1180\n..."
            )
            if lg_text:
                lg_df_raw = parse_input(pasted_text=lg_text)

        if lg_df_raw is not None:
            st.success(f"✅ {len(lg_df_raw)} Zeilen, {len(lg_df_raw.columns)} Spalten erkannt")
            with st.expander("🔍 Rohdaten-Vorschau", expanded=False):
                st.dataframe(lg_df_raw.head(20), use_container_width=True)

            required_lg = {
                "Zeitstempel": ["zeit", "timestamp", "date", "datum", "time"],
                "Verbrauch_kW": ["verbrauch", "last", "load", "leistung", "kw", "mw", "wert", "value", "power"]
            }
            lg_mapping = map_columns(lg_df_raw, "lg", required_lg)

            if len(lg_mapping) >= 2:
                if st.button("✅ Lastgang übernehmen", key="lg_apply", type="primary"):
                    st.session_state.lastgang_df = apply_mapping(lg_df_raw, lg_mapping)
                    st.session_state.lastgang_cols = lg_mapping
                    st.success("Lastgang erfolgreich importiert!")
                    st.dataframe(st.session_state.lastgang_df.head(10), use_container_width=True)
            else:
                st.warning("Bitte alle erforderlichen Spalten zuordnen.")

    # --- Spotpreise ---
    with import_tabs[1]:
        st.markdown("#### 📉 Historische Spotpreise (z.B. EPEX Spot Day-Ahead)")
        input_method_sp = st.radio("Eingabemethode", ["Datei-Upload", "Copy & Paste"], key="sp_method", horizontal=True)

        sp_df_raw = None
        if input_method_sp == "Datei-Upload":
            sp_file = st.file_uploader("Spotpreise hochladen (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="sp_upload")
            if sp_file:
                sp_df_raw = parse_input(uploaded_file=sp_file)
        else:
            sp_text = st.text_area(
                "Daten einfügen", height=200, key="sp_paste",
                placeholder="Datum;Preis_EUR_MWh\n01.01.2024;85.50\n02.01.2024;92.30\n..."
            )
            if sp_text:
                sp_df_raw = parse_input(pasted_text=sp_text)

        if sp_df_raw is not None:
            st.success(f"✅ {len(sp_df_raw)} Zeilen, {len(sp_df_raw.columns)} Spalten erkannt")
            with st.expander("🔍 Rohdaten-Vorschau", expanded=False):
                st.dataframe(sp_df_raw.head(20), use_container_width=True)

            required_sp = {
                "Datum": ["datum", "date", "zeit", "timestamp", "time", "day"],
                "Spotpreis": ["preis", "price", "spot", "eur", "mwh", "wert", "value", "close"]
            }
            sp_mapping = map_columns(sp_df_raw, "sp", required_sp)

            if len(sp_mapping) >= 2:
                if st.button("✅ Spotpreise übernehmen", key="sp_apply", type="primary"):
                    st.session_state.spot_df = apply_mapping(sp_df_raw, sp_mapping)
                    st.session_state.spot_cols = sp_mapping
                    st.success("Spotpreise erfolgreich importiert!")
                    st.dataframe(st.session_state.spot_df.head(10), use_container_width=True)
            else:
                st.warning("Bitte alle erforderlichen Spalten zuordnen.")

    # --- Forwardpreise ---
    with import_tabs[2]:
        st.markdown("#### 📊 Historische Forward-/Futures-Preise")
        input_method_fw = st.radio("Eingabemethode", ["Datei-Upload", "Copy & Paste", "Manuell eingeben"], key="fw_method", horizontal=True)

        fw_df_raw = None
        if input_method_fw == "Datei-Upload":
            fw_file = st.file_uploader("Forwardpreise hochladen (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="fw_upload")
            if fw_file:
                fw_df_raw = parse_input(uploaded_file=fw_file)
        elif input_method_fw == "Copy & Paste":
            fw_text = st.text_area(
                "Daten einfügen", height=200, key="fw_paste",
                placeholder="Datum;Produkt;Preis_EUR_MWh\n01.01.2024;Cal-25 Base;82.50\n..."
            )
            if fw_text:
                fw_df_raw = parse_input(pasted_text=fw_text)
        else:
            st.markdown("**Manuelle Eingabe der Forward-Preise:**")
            n_products = st.number_input("Anzahl Produkte", 1, 20, 3, key="fw_n")
            manual_data = []
            for i in range(int(n_products)):
                c1, c2, c3 = st.columns(3)
                with c1:
                    prod = st.text_input(f"Produkt {i+1}", value=f"Cal-{2025+i} Base", key=f"fw_prod_{i}")
                with c2:
                    price = st.number_input(f"Preis (€/MWh) {i+1}", 0.0, 1000.0, 80.0, key=f"fw_price_{i}")
                with c3:
                    date = st.date_input(f"Datum {i+1}", value=datetime.date.today(), key=f"fw_date_{i}")
                manual_data.append({"Datum": date, "Produkt": prod, "Forwardpreis": price})
            if st.button("✅ Manuelle Eingabe übernehmen", key="fw_manual_apply"):
                st.session_state.forward_df = pd.DataFrame(manual_data)
                st.session_state.forward_df["Datum"] = pd.to_datetime(st.session_state.forward_df["Datum"])
                st.success("Forwardpreise erfolgreich importiert!")

        if fw_df_raw is not None:
            st.success(f"✅ {len(fw_df_raw)} Zeilen, {len(fw_df_raw.columns)} Spalten erkannt")
            with st.expander("🔍 Rohdaten-Vorschau", expanded=False):
                st.dataframe(fw_df_raw.head(20), use_container_width=True)

            required_fw = {
                "Datum": ["datum", "date", "zeit", "timestamp", "time"],
                "Produkt": ["produkt", "product", "name", "contract", "kontrakt"],
                "Forwardpreis": ["preis", "price", "forward", "futures", "eur", "wert", "value", "close", "settle"]
            }
            fw_mapping = map_columns(fw_df_raw, "fw", required_fw)

            if len(fw_mapping) >= 2:
                if st.button("✅ Forwardpreise übernehmen", key="fw_apply", type="primary"):
                    st.session_state.forward_df = apply_mapping(fw_df_raw, fw_mapping)
                    st.session_state.forward_cols = fw_mapping
                    st.success("Forwardpreise erfolgreich importiert!")
                    st.dataframe(st.session_state.forward_df.head(10), use_container_width=True)


# ═════════════════════════════════════════════
# TAB 2: MARKTDATEN VISUALISIERUNG
# ═════════════════════════════════════════════
with tab2:
    st.markdown("### 📈 Marktdaten & Lastprofil")

    viz_tabs = st.tabs(["⚡ Lastprofil", "📉 Spotpreise", "📊 Forwardpreise"])

    with viz_tabs[0]:
        if st.session_state.lastgang_df is not None:
            df_lg = st.session_state.lastgang_df

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Datenpunkte", f"{len(df_lg):,}")
            with col2:
                st.metric("Ø Verbrauch", f"{df_lg['Verbrauch_kW'].mean():,.1f} kW")
            with col3:
                st.metric("Max Verbrauch", f"{df_lg['Verbrauch_kW'].max():,.1f} kW")
            with col4:
                st.metric("Min Verbrauch", f"{df_lg['Verbrauch_kW'].min():,.1f} kW")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_lg["Zeitstempel"], y=df_lg["Verbrauch_kW"],
                mode="lines", name="Verbrauch",
                line=dict(color="#2e86c1", width=1),
                fill="tozeroy", fillcolor="rgba(46,134,193,0.1)"
            ))
            fig.update_layout(
                title="Lastprofil / Verbrauchsverlauf",
                xaxis_title="Zeit", yaxis_title="Verbrauch (kW)",
                template="plotly_white", height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tagesprofil
            if "Zeitstempel" in df_lg.columns:
                df_lg_copy = df_lg.copy()
                df_lg_copy["Stunde"] = df_lg_copy["Zeitstempel"].dt.hour
                hourly = df_lg_copy.groupby("Stunde")["Verbrauch_kW"].agg(["mean", "min", "max"]).reset_index()

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=hourly["Stunde"], y=hourly["max"], mode="lines",
                    name="Max", line=dict(color="#e74c3c", dash="dash")))
                fig2.add_trace(go.Scatter(x=hourly["Stunde"], y=hourly["mean"], mode="lines+markers",
                    name="Durchschnitt", line=dict(color="#2e86c1", width=3)))
                fig2.add_trace(go.Scatter(x=hourly["Stunde"], y=hourly["min"], mode="lines",
                    name="Min", line=dict(color="#27ae60", dash="dash")))
                fig2.update_layout(
                    title="Typisches Tagesprofil",
                    xaxis_title="Stunde", yaxis_title="Verbrauch (kW)",
                    template="plotly_white", height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("📥 Bitte zuerst Lastgang-Daten im Tab 'Datenimport' importieren.")

    with viz_tabs[1]:
        if st.session_state.spot_df is not None:
            df_sp = st.session_state.spot_df

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Datenpunkte", f"{len(df_sp):,}")
            with col2:
                st.metric("Ø Preis", f"{df_sp['Spotpreis'].mean():,.2f} €/MWh")
            with col3:
                st.metric("Max Preis", f"{df_sp['Spotpreis'].max():,.2f} €/MWh")
            with col4:
                st.metric("Min Preis", f"{df_sp['Spotpreis'].min():,.2f} €/MWh")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_sp["Datum"], y=df_sp["Spotpreis"],
                mode="lines", name="Spotpreis",
                line=dict(color="#e67e22", width=1.5)
            ))
            # Gleitender Durchschnitt
            if len(df_sp) > 30:
                df_sp_copy = df_sp.copy()
                df_sp_copy["MA30"] = df_sp_copy["Spotpreis"].rolling(30).mean()
                fig.add_trace(go.Scatter(
                    x=df_sp_copy["Datum"], y=df_sp_copy["MA30"],
                    mode="lines", name="30-Tage-Ø",
                    line=dict(color="#8e44ad", width=2.5)
                ))
            fig.update_layout(
                title="Historische Spotpreise",
                xaxis_title="Datum", yaxis_title="Preis (€/MWh)",
                template="plotly_white", height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Verteilung
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=df_sp["Spotpreis"], nbinsx=50,
                marker_color="#3498db", opacity=0.8
            ))
            fig3.update_layout(
                title="Preisverteilung Spotmarkt",
                xaxis_title="Preis (€/MWh)", yaxis_title="Häufigkeit",
                template="plotly_white", height=350
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("📥 Bitte zuerst Spotpreis-Daten im Tab 'Datenimport' importieren.")

    with viz_tabs[2]:
        if st.session_state.forward_df is not None:
            df_fw = st.session_state.forward_df
            st.dataframe(df_fw, use_container_width=True)

            if "Produkt" in df_fw.columns and "Forwardpreis" in df_fw.columns:
                if "Datum" in df_fw.columns and len(df_fw) > 1:
                    fig = px.line(df_fw, x="Datum", y="Forwardpreis", color="Produkt",
                        title="Forward-Preise über Zeit", template="plotly_white")
                    fig.update_layout(height=500, hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.bar(df_fw, x="Produkt", y="Forwardpreis",
                        title="Forward-Preise nach Produkt", template="plotly_white",
                        color="Forwardpreis", color_continuous_scale="blues")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📥 Bitte zuerst Forward-Daten im Tab 'Datenimport' importieren.")


# ═════════════════════════════════════════════
# TAB 3: STRATEGIEN KONFIGURATION
# ═════════════════════════════════════════════
with tab3:
    st.markdown("### 🔧 Strategien konfigurieren")

    st.markdown("#### Beschaffungsparameter")
    col1, col2 = st.columns(2)
    with col1:
        total_volume = st.number_input(
            "Jahresverbrauch (MWh)", min_value=1.0, value=10000.0, step=100.0,
            help="Gesamter Jahresverbrauch in MWh"
        )
        forward_price_input = st.number_input(
            "Referenz-Forwardpreis (€/MWh)", min_value=0.0, value=80.0, step=1.0,
            help="Aktueller/durchschnittlicher Forwardpreis für die Simulation"
        )
    with col2:
        spread_cost = st.number_input(
            "Transaktionskosten/Spread (€/MWh)", min_value=0.0, value=0.5, step=0.1,
            help="Optionale Transaktionskosten pro MWh"
        )
        risk_free_rate = st.number_input(
            "Risikoloser Zinssatz (%)", min_value=0.0, value=3.0, step=0.1
        )

    st.markdown("---")
    st.markdown("#### Fixierungsquoten")

    quota_mode = st.radio(
        "Quotenmodus", 
        ["Vordefinierte Quoten", "Benutzerdefiniert"],
        horizontal=True
    )

    if quota_mode == "Vordefinierte Quoten":
        selected_quotas = st.multiselect(
            "Quoten auswählen (%)",
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            default=[0, 20, 40, 60, 80, 100]
        )
        quotas = [q / 100 for q in selected_quotas]
    else:
        quota_text = st.text_input(
            "Quoten eingeben (kommagetrennt, in %)",
            value="0, 25, 50, 75, 100"
        )
        quotas = [float(q.strip()) / 100 for q in quota_text.split(",") if q.strip()]

    st.markdown("---")
    st.markdown("#### Strategietypen")

    available_strategies = ["Gleichmäßig", "Frontloaded", "Backloaded", "Regelbasiert", "100% Spot", "100% Termin"]
    selected_strategies = st.multiselect(
        "Strategien auswählen",
        available_strategies,
        default=["Gleichmäßig", "Frontloaded", "Backloaded", "100% Spot", "100% Termin"]
    )

    # Regelbasierte Parameter
    rule_params = {}
    if "Regelbasiert" in selected_strategies:
        st.markdown("**Parameter für regelbasierte Strategie:**")
        rule_threshold = st.slider(
            "Kaufschwelle (% des Forwardpreises)",
            50, 120, 95, 1,
            help="Fixiere wenn Spotpreis unter X% des Forwardpreises liegt"
        )
        rule_params["threshold"] = forward_price_input * rule_threshold / 100
        st.info(f"Kaufschwelle: {rule_params['threshold']:.2f} €/MWh")

    # Speichern
    st.session_state.strategies_config = {
        "volume": total_volume,
        "forward_price": forward_price_input,
        "spread": spread_cost,
        "quotas": quotas,
        "strategies": selected_strategies,
        "rule_params": rule_params
    }

    st.success(f"✅ {len(selected_strategies)} Strategien × {len(quotas)} Quoten = {len(selected_strategies) * len(quotas)} Szenarien konfiguriert")


# ═════════════════════════════════════════════
# TAB 4: BACKTESTING
# ═════════════════════════════════════════════
with tab4:
    st.markdown("### 🔄 Backtesting durchführen")

    if st.session_state.spot_df is None:
        st.warning("⚠️ Bitte zuerst Spotpreise importieren!")
    elif not st.session_state.strategies_config:
        st.warning("⚠️ Bitte zuerst Strategien konfigurieren!")
    else:
        config = st.session_state.strategies_config
        df_spot = st.session_state.spot_df

        st.markdown("#### Backtesting-Zeitraum")
        col1, col2 = st.columns(2)
        min_date = df_spot["Datum"].min().date() if hasattr(df_spot["Datum"].min(), "date") else df_spot["Datum"].min()
        max_date = df_spot["Datum"].max().date() if hasattr(df_spot["Datum"].max(), "date") else df_spot["Datum"].max()
        with col1:
            start_date = st.date_input("Von", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("Bis", value=max_date, min_value=min_date, max_value=max_date)

        st.markdown("---")

        if st.button("🚀 Backtesting starten", type="primary", use_container_width=True):
            with st.spinner("Backtesting läuft..."):
                # Filter Spotpreise
                mask = (df_spot["Datum"].dt.date >= start_date) & (df_spot["Datum"].dt.date <= end_date)
                spot_filtered = df_spot[mask].copy().reset_index(drop=True)

                if len(spot_filtered) == 0:
                    st.error("Keine Daten im gewählten Zeitraum!")
                else:
                    results = []
                    all_series = {}

                    for strategy in config["strategies"]:
                        for quota in config["quotas"]:
                            # Bei 100% Spot/Termin nur einmal berechnen
                            if strategy == "100% Spot" and quota != 0:
                                continue
                            if strategy == "100% Termin" and quota != 1.0:
                                continue
                            if strategy not in ["100% Spot", "100% Termin"] and quota in [0, 1.0]:
                                continue  # Redundant mit 100% Spot/Termin

                            calc = calculate_strategy_costs(
                                spot_filtered["Spotpreis"],
                                config["forward_price"] + config["spread"],
                                config["volume"],
                                quota, strategy,
                                config.get("rule_params")
                            )

                            if calc:
                                risk = calculate_risk_metrics(calc["cumulative"])
                                label = f"{strategy} ({int(quota*100)}%)" if strategy not in ["100% Spot", "100% Termin"] else strategy

                                results.append({
                                    "Strategie": label,
                                    "Typ": strategy,
                                    "Quote": quota,
                                    "Gesamtkosten_EUR": calc["total"],
                                    "Durchschnittspreis_EUR_MWh": calc["avg_price"],
                                    "Volatilität": risk["Volatilität"],
                                    "VaR_95": risk["VaR_95"],
                                    "CVaR_95": risk["CVaR_95"],
                                    "Max_Drawdown": risk["Max_Drawdown"],
                                })
                                all_series[label] = {
                                    "dates": spot_filtered["Datum"].values,
                                    "cumulative": calc["cumulative"],
                                    "total_costs": calc["total_costs"],
                                }

                    if results:
                        results_df = pd.DataFrame(results).sort_values("Gesamtkosten_EUR")
                        results_df["Rang"] = range(1, len(results_df) + 1)

                        # Spot-Benchmark
                        spot_total = (spot_filtered["Spotpreis"] * config["volume"] / len(spot_filtered)).sum()
                        results_df["vs_Spot_EUR"] = results_df["Gesamtkosten_EUR"] - spot_total
                        results_df["vs_Spot_pct"] = (results_df["vs_Spot_EUR"] / spot_total * 100)

                        st.session_state.backtesting_results = {
                            "results_df": results_df,
                            "series": all_series,
                            "spot_filtered": spot_filtered,
                            "config": config,
                            "spot_total": spot_total,
                        }

                        st.success(f"✅ Backtesting abgeschlossen! {len(results)} Szenarien berechnet.")

                        # Quick Preview
                        st.markdown("#### 🏆 Top 5 Strategien")
                        top5 = results_df.head(5)[["Rang", "Strategie", "Gesamtkosten_EUR", "Durchschnittspreis_EUR_MWh", "vs_Spot_pct"]]
                        top5.columns = ["Rang", "Strategie", "Gesamtkosten (€)", "Ø Preis (€/MWh)", "vs. Spot (%)"]
                        st.dataframe(
                            top5.style.format({
                                "Gesamtkosten (€)": "{:,.0f}",
                                "Ø Preis (€/MWh)": "{:.2f}",
                                "vs. Spot (%)": "{:+.2f}%"
                            }),
                            use_container_width=True, hide_index=True
                        )
                    else:
                        st.warning("Keine gültigen Szenarien berechnet.")


# ═════════════════════════════════════════════
# TAB 5: ERGEBNISSE & DASHBOARD
# ═════════════════════════════════════════════
with tab5:
    st.markdown("### 📊 Ergebnisse & Dashboard")

    if st.session_state.backtesting_results is None:
        st.info("🔄 Bitte zuerst Backtesting im Tab 'Backtesting' durchführen.")
    else:
        bt = st.session_state.backtesting_results
        results_df = bt["results_df"]
        series = bt["series"]
        spot_total = bt["spot_total"]
        config = bt["config"]

        # KPIs
        best = results_df.iloc[0]
        worst = results_df.iloc[-1]

        st.markdown("#### 🎯 Key Performance Indicators")
        kpi_cols = st.columns(5)
        with kpi_cols[0]:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:0.85rem;">Beste Strategie</div>
                <div style="font-size:1.3rem;font-weight:700;">{best['Strategie']}</div>
            </div>""", unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(f"""<div class="metric-card-green">
                <div style="font-size:0.85rem;">Niedrigste Kosten</div>
                <div style="font-size:1.3rem;font-weight:700;">{best['Gesamtkosten_EUR']:,.0f} €</div>
            </div>""", unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(f"""<div class="metric-card-red">
                <div style="font-size:0.85rem;">Höchste Kosten</div>
                <div style="font-size:1.3rem;font-weight:700;">{worst['Gesamtkosten_EUR']:,.0f} €</div>
            </div>""", unsafe_allow_html=True)
        with kpi_cols[3]:
            spread = worst['Gesamtkosten_EUR'] - best['Gesamtkosten_EUR']
            st.markdown(f"""<div class="metric-card-blue">
                <div style="font-size:0.85rem;">Kosten-Spread</div>
                <div style="font-size:1.3rem;font-weight:700;">{spread:,.0f} €</div>
            </div>""", unsafe_allow_html=True)
        with kpi_cols[4]:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:0.85rem;">100% Spot Kosten</div>
                <div style="font-size:1.3rem;font-weight:700;">{spot_total:,.0f} €</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Charts
        chart_tabs = st.tabs(["📊 Kostenvergleich", "📈 Kumulative Kosten", "🗺️ Heatmap", "⚠️ Risikoanalyse", "💡 Empfehlung"])

        with chart_tabs[0]:
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, (_, row) in enumerate(results_df.iterrows()):
                fig.add_trace(go.Bar(
                    x=[row["Strategie"]], y=[row["Gesamtkosten_EUR"]],
                    name=row["Strategie"],
                    marker_color=colors[i % len(colors)],
                    text=[f"{row['Gesamtkosten_EUR']:,.0f}€"],
                    textposition="outside"
                ))
            fig.add_hline(y=spot_total, line_dash="dash", line_color="red",
                annotation_text=f"100% Spot: {spot_total:,.0f}€")
            fig.update_layout(
                title="Gesamtkosten nach Strategie",
                yaxis_title="Gesamtkosten (€)", template="plotly_white",
                height=500, showlegend=False,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

            # Durchschnittspreis
            fig2 = px.bar(
                results_df, x="Strategie", y="Durchschnittspreis_EUR_MWh",
                color="Durchschnittspreis_EUR_MWh",
                color_continuous_scale="RdYlGn_r",
                title="Durchschnittlicher Beschaffungspreis",
                text="Durchschnittspreis_EUR_MWh"
            )
            fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig2.update_layout(template="plotly_white", height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

        with chart_tabs[1]:
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i, (label, s) in enumerate(series.items()):
                fig.add_trace(go.Scatter(
                    x=s["dates"], y=s["cumulative"],
                    mode="lines", name=label,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            fig.update_layout(
                title="Kumulative Beschaffungskosten über Zeit",
                xaxis_title="Datum", yaxis_title="Kumulative Kosten (€)",
                template="plotly_white", height=550,
                hovermode="x unified", legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart_tabs[2]:
            # Heatmap: Quote × Strategie → Kosten
            heatmap_data = results_df.pivot_table(
                index="Typ", columns="Quote", values="Gesamtkosten_EUR", aggfunc="first"
            )
            if not heatmap_data.empty:
                heatmap_data.columns = [f"{int(c*100)}%" for c in heatmap_data.columns]
                fig = px.imshow(
                    heatmap_data.values,
                    x=heatmap_data.columns.tolist(),
                    y=heatmap_data.index.tolist(),
                    color_continuous_scale="RdYlGn_r",
                    title="Heatmap: Strategietyp × Quote → Gesamtkosten",
                    text_auto=".0f",
                    aspect="auto"
                )
                fig.update_layout(template="plotly_white", height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nicht genug Daten für Heatmap.")

        with chart_tabs[3]:
            st.markdown("#### ⚠️ Risikokennzahlen")

            risk_cols = ["Strategie", "Volatilität", "VaR_95", "CVaR_95", "Max_Drawdown"]
            risk_df = results_df[risk_cols].copy()
            st.dataframe(
                risk_df.style.format({
                    "Volatilität": "{:.4f}",
                    "VaR_95": "{:.4f}",
                    "CVaR_95": "{:.4f}",
                    "Max_Drawdown": "{:.4f}"
                }).background_gradient(subset=["Volatilität"], cmap="Reds"),
                use_container_width=True, hide_index=True
            )

            # Risk-Return Scatter
            fig = px.scatter(
                results_df, x="Volatilität", y="Gesamtkosten_EUR",
                size=np.abs(results_df["Max_Drawdown"]) * 1000 + 10,
                color="Strategie",
                title="Risiko-Kosten-Profil (Größe = |Max Drawdown|)",
                template="plotly_white",
                hover_data=["Durchschnittspreis_EUR_MWh"]
            )
            fig.update_layout(height=500, yaxis_title="Gesamtkosten (€)")
            st.plotly_chart(fig, use_container_width=True)

        with chart_tabs[4]:
            st.markdown("#### 💡 Handlungsempfehlung")

            best_row = results_df.iloc[0]
            saving_vs_spot = spot_total - best_row["Gesamtkosten_EUR"]
            saving_pct = saving_vs_spot / spot_total * 100

            if saving_vs_spot > 0:
                st.markdown(f"""
                <div class="recommendation-box">
                <h4>🏆 Empfohlene Strategie: {best_row['Strategie']}</h4>
                <p>Basierend auf dem historischen Backtesting hätte die Strategie 
                <strong>{best_row['Strategie']}</strong> die niedrigsten Gesamtkosten erzielt:</p>
                <ul>
                    <li><strong>Gesamtkosten:</strong> {best_row['Gesamtkosten_EUR']:,.0f} €</li>
                    <li><strong>Ø Beschaffungspreis:</strong> {best_row['Durchschnittspreis_EUR_MWh']:.2f} €/MWh</li>
                    <li><strong>Ersparnis vs. 100% Spot:</strong> {saving_vs_spot:,.0f} € ({saving_pct:.1f}%)</li>
                    <li><strong>Volatilität:</strong> {best_row['Volatilität']:.4f}</li>
                    <li><strong>Max Drawdown:</strong> {best_row['Max_Drawdown']:.4f}</li>
                </ul>
                <p><em>Hinweis: Vergangene Performance ist kein Indikator für zukünftige Ergebnisse. 
                Die Empfehlung basiert ausschließlich auf historischem Backtesting.</em></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="recommendation-box">
                <h4>📊 Analyseergebnis</h4>
                <p>Im betrachteten Zeitraum hätte eine <strong>100% Spot-Beschaffung</strong> 
                die niedrigsten Kosten erzielt. Die nächstbeste Strategie war 
                <strong>{best_row['Strategie']}</strong> mit Gesamtkosten von 
                {best_row['Gesamtkosten_EUR']:,.0f} €.</p>
                </div>
                """, unsafe_allow_html=True)

            # Ranking-Tabelle
            st.markdown("#### 🏅 Vollständiges Ranking")
            display_df = results_df[["Rang", "Strategie", "Gesamtkosten_EUR", "Durchschnittspreis_EUR_MWh", "vs_Spot_pct", "Volatilität"]].copy()
            display_df.columns = ["Rang", "Strategie", "Gesamtkosten (€)", "Ø Preis (€/MWh)", "vs. Spot (%)", "Volatilität"]
            st.dataframe(
                display_df.style.format({
                    "Gesamtkosten (€)": "{:,.0f}",
                    "Ø Preis (€/MWh)": "{:.2f}",
                    "vs. Spot (%)": "{:+.2f}",
                    "Volatilität": "{:.4f}"
                }),
                use_container_width=True, hide_index=True
            )


# ═════════════════════════════════════════════
# TAB 6: EXPORT
# ═════════════════════════════════════════════
with tab6:
    st.markdown("### 💾 Ergebnisse exportieren")

    if st.session_state.backtesting_results is None:
        st.info("🔄 Bitte zuerst Backtesting durchführen.")
    else:
        bt = st.session_state.backtesting_results
        results_df = bt["results_df"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📄 CSV Export")
            csv_buffer = StringIO()
            results_df.to_csv(csv_buffer, index=False, sep=";", decimal=",")
            st.download_button(
                "📥 Ergebnisse als CSV herunterladen",
                csv_buffer.getvalue(),
                "backtesting_ergebnisse.csv",
                "text/csv",
                use_container_width=True
            )

        with col2:
            st.markdown("#### 📊 Excel Export")
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                results_df.to_excel(writer, sheet_name="Ergebnisse", index=False)
                if st.session_state.spot_df is not None:
                    bt["spot_filtered"].to_excel(writer, sheet_name="Spotpreise", index=False)
            st.download_button(
                "📥 Ergebnisse als Excel herunterladen",
                excel_buffer.getvalue(),
                "backtesting_ergebnisse.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("#### 📋 Zeitreihen-Export")

        series = bt["series"]
        if series:
            series_data = {}
            first_key = list(series.keys())[0]
            series_data["Datum"] = series[first_key]["dates"]
            for label, s in series.items():
                series_data[f"Kum. Kosten - {label}"] = s["cumulative"]

            series_df = pd.DataFrame(series_data)
            csv_ts = StringIO()
            series_df.to_csv(csv_ts, index=False, sep=";", decimal=",")
            st.download_button(
                "📥 Zeitreihen als CSV herunterladen",
                csv_ts.getvalue(),
                "backtesting_zeitreihen.csv",
                "text/csv",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#95a5a6;font-size:0.85rem;'>"
    "⚡ Energie-Beschaffungs-Engine v1.0 | Simulation · Backtesting · Bewertung"
    "</div>",
    unsafe_allow_html=True
)

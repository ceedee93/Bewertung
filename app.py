import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    .recommendation-box {
        background: #eaf2f8; border-left: 5px solid #2e86c1;
        padding: 1rem 1.5rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
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
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=";", decimal=",", parse_dates=True)
                except Exception:
                    st.error("CSV konnte nicht gelesen werden.")
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, parse_dates=True)
    elif pasted_text and pasted_text.strip():
        text = pasted_text.strip()
        for sep in ["\t", ";", ",", "|"]:
            try:
                df = pd.read_csv(StringIO(text), sep=sep, engine="python", parse_dates=True)
                if len(df.columns) > 1:
                    break
            except Exception:
                continue
    return df


def map_columns(df, key_prefix, required_cols):
    """Interaktive Spaltenzuordnung mit Auto-Erkennung."""
    st.markdown("**🔧 Spaltenzuordnung**")
    mapping = {}
    cols = ["-- Nicht zugeordnet --"] + list(df.columns)

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
    """Wendet Spaltenzuordnung an."""
    rename_map = {v: k for k, v in mapping.items()}
    df_mapped = df[list(mapping.values())].rename(columns=rename_map).copy()
    if "Zeitstempel" in df_mapped.columns:
        df_mapped["Zeitstempel"] = pd.to_datetime(df_mapped["Zeitstempel"], dayfirst=True, errors="coerce")
        df_mapped = df_mapped.dropna(subset=["Zeitstempel"]).sort_values("Zeitstempel").reset_index(drop=True)
    if "Datum" in df_mapped.columns:
        df_mapped["Datum"] = pd.to_datetime(df_mapped["Datum"], dayfirst=True, errors="coerce")
        df_mapped = df_mapped.dropna(subset=["Datum"]).sort_values("Datum").reset_index(drop=True)
    for col in df_mapped.columns:
        if col not in ["Zeitstempel", "Datum", "Produkt"]:
            df_mapped[col] = pd.to_numeric(
                df_mapped[col].astype(str).str.replace(",", ".").str.strip(), errors="coerce"
            )
    return df_mapped


def simulate_procurement(spot_prices, forward_price, volume_mwh, quota, strategy_type,
                         spread=0.0, rule_params=None):
    """
    Kernfunktion: Simuliert historische Beschaffungskosten.

    Logik:
    - Pro Periode (Tag/Stunde) wird ein bestimmtes Volumen benötigt: vol_per_period = volume_mwh / n
    - Die Fixierungsquote (quota) bestimmt, welcher Anteil zum Forwardpreis beschafft wird
      und welcher Anteil am Spotmarkt.
    - Die Strategie bestimmt, WIE die Fixierungsquote über die Zeit verteilt wird:
      * Gleichmäßig: Jede Periode hat die gleiche Quote
      * Frontloaded: Höhere Quote am Anfang, niedrigere am Ende (Durchschnitt = quota)
      * Backloaded: Niedrigere Quote am Anfang, höhere am Ende
      * Regelbasiert: Fixiere nur in Perioden, wo Spot < Schwelle

    Ergebnis: Für jede Periode die tatsächlichen Beschaffungskosten
    = (anteil_termin * vol_per_period * forward_price)
    + (anteil_spot * vol_per_period * spot_price_i)
    """
    n = len(spot_prices)
    if n == 0:
        return {}

    spot = spot_prices.values.astype(float)
    fwd = forward_price + spread
    vol_per_period = volume_mwh / n

    # Bestimme die effektive Fixierungsquote pro Periode
    # (Durchschnitt über alle Perioden = quota)
    if strategy_type == "Gleichmäßig":
        # Jede Periode gleich
        fix_quota = np.full(n, quota)

    elif strategy_type == "Frontloaded":
        # Linear fallend: Anfang hoch, Ende niedrig, Durchschnitt = quota
        if n > 1:
            # Lineare Verteilung: von (quota * 1.6) bis (quota * 0.4)
            fix_quota = np.linspace(quota * 1.6, quota * 0.4, n)
            # Normalisieren auf gewünschten Durchschnitt
            fix_quota = fix_quota * (quota / fix_quota.mean()) if fix_quota.mean() > 0 else np.full(n, quota)
        else:
            fix_quota = np.full(n, quota)
        fix_quota = np.clip(fix_quota, 0, 1)

    elif strategy_type == "Backloaded":
        # Linear steigend: Anfang niedrig, Ende hoch
        if n > 1:
            fix_quota = np.linspace(quota * 0.4, quota * 1.6, n)
            fix_quota = fix_quota * (quota / fix_quota.mean()) if fix_quota.mean() > 0 else np.full(n, quota)
        else:
            fix_quota = np.full(n, quota)
        fix_quota = np.clip(fix_quota, 0, 1)

    elif strategy_type == "Regelbasiert":
        # Fixiere verstärkt wenn Spot günstig (unter Schwelle)
        threshold = rule_params.get("threshold", fwd * 0.95) if rule_params else fwd * 0.95
        is_cheap = spot < threshold
        n_cheap = is_cheap.sum()

        if n_cheap > 0 and n_cheap < n:
            # In günstigen Perioden: höhere Fixierung, in teuren: niedrigere
            fix_quota = np.where(is_cheap, 
                                 quota * n / n_cheap * 0.8,  # 80% der Fixierung in günstigen Perioden
                                 quota * n / (n - n_cheap) * 0.2)  # 20% in teuren
            # Normalisieren
            fix_quota = fix_quota * (quota * n / fix_quota.sum()) if fix_quota.sum() > 0 else np.full(n, quota)
        else:
            fix_quota = np.full(n, quota)
        fix_quota = np.clip(fix_quota, 0, 1)

    else:
        fix_quota = np.full(n, quota)

    # Kosten berechnen
    termin_kosten = fix_quota * vol_per_period * fwd           # Terminmarkt-Anteil
    spot_kosten = (1 - fix_quota) * vol_per_period * spot      # Spotmarkt-Anteil
    total_kosten = termin_kosten + spot_kosten                 # Gesamtkosten pro Periode

    # Vergleich: Was hätte 100% Spot gekostet?
    spot_only_kosten = vol_per_period * spot
    # Was hätte 100% Termin gekostet?
    termin_only_kosten = np.full(n, vol_per_period * fwd)

    cumulative = np.cumsum(total_kosten)
    cum_spot_only = np.cumsum(spot_only_kosten)
    cum_termin_only = np.cumsum(termin_only_kosten)

    total = cumulative[-1]
    total_spot_only = cum_spot_only[-1]
    total_termin_only = cum_termin_only[-1]

    # Mark-to-Market: Wie viel hat die Strategie im Vergleich gespart/gekostet?
    pnl_vs_spot = total_spot_only - total  # positiv = Ersparnis
    pnl_vs_termin = total_termin_only - total

    # Durchschnittlicher Beschaffungspreis (gewichteter Mischpreis)
    avg_price = total / volume_mwh if volume_mwh > 0 else 0
    avg_spot_price = total_spot_only / volume_mwh if volume_mwh > 0 else 0

    return {
        "fix_quota": fix_quota,
        "termin_kosten": termin_kosten,
        "spot_kosten": spot_kosten,
        "total_kosten": total_kosten,
        "cumulative": cumulative,
        "cum_spot_only": cum_spot_only,
        "cum_termin_only": cum_termin_only,
        "total": total,
        "total_spot_only": total_spot_only,
        "total_termin_only": total_termin_only,
        "avg_price": avg_price,
        "avg_spot_price": avg_spot_price,
        "pnl_vs_spot": pnl_vs_spot,
        "pnl_vs_termin": pnl_vs_termin,
    }


def calculate_risk_metrics(total_kosten_series):
    """Berechnet Risikokennzahlen auf Basis der Periodenkosten."""
    if len(total_kosten_series) < 2:
        return {"Volatilität": 0, "VaR_95": 0, "CVaR_95": 0, "Max_Drawdown": 0, "Std_Kosten": 0}

    costs = np.array(total_kosten_series)
    std = np.std(costs)

    # Kumulierte Kosten für Drawdown
    cumulative = np.cumsum(costs)
    # Kosten-Änderungen
    changes = np.diff(costs)

    var_95 = np.percentile(costs, 95)  # 95%-Quantil der Periodenkosten
    tail = costs[costs >= var_95]
    cvar_95 = tail.mean() if len(tail) > 0 else var_95

    # Max Drawdown auf kumulierten Kosten (wie stark steigen Kosten über Erwartung)
    expected_cum = np.linspace(0, cumulative[-1], len(cumulative))
    deviation = cumulative - expected_cum
    max_dd = deviation.max() - deviation.min() if len(deviation) > 1 else 0

    # Volatilität der Periodenkosten (annualisiert)
    vol = std * np.sqrt(252) if std > 0 else 0

    return {
        "Volatilität": vol,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "Max_Drawdown": max_dd,
        "Std_Kosten": std,
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
    2. **Spalten zuordnen** — Auto oder manuell
    3. **Strategien konfigurieren**
    4. **Backtesting starten**
    5. **Ergebnisse analysieren & exportieren**
    """)
    st.markdown("---")
    st.caption("v2.0 | Energie-Beschaffungs-Engine")

# ─────────────────────────────────────────────
# HAUPTBEREICH
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">⚡ Energie-Beschaffungs-Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Simulation · Backtesting · Mark-to-Market Bewertung</p>', unsafe_allow_html=True)

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

    with import_tabs[0]:
        st.markdown("#### ⚡ Lastgang / Verbrauchsprofil")
        input_method_lg = st.radio("Eingabemethode", ["Datei-Upload", "Copy & Paste"], key="lg_method", horizontal=True)

        lg_df_raw = None
        if input_method_lg == "Datei-Upload":
            lg_file = st.file_uploader("Lastgang hochladen (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="lg_upload")
            if lg_file:
                lg_df_raw = parse_input(uploaded_file=lg_file)
        else:
            lg_text = st.text_area("Daten einfügen (Tab/Semikolon/Komma-getrennt)",
                height=200, key="lg_paste",
                placeholder="Zeitstempel;Verbrauch_kW\n01.01.2024 00:00;1250\n01.01.2024 00:15;1180")
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

    with import_tabs[1]:
        st.markdown("#### 📉 Historische Spotpreise")
        input_method_sp = st.radio("Eingabemethode", ["Datei-Upload", "Copy & Paste"], key="sp_method", horizontal=True)

        sp_df_raw = None
        if input_method_sp == "Datei-Upload":
            sp_file = st.file_uploader("Spotpreise hochladen (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="sp_upload")
            if sp_file:
                sp_df_raw = parse_input(uploaded_file=sp_file)
        else:
            sp_text = st.text_area("Daten einfügen", height=200, key="sp_paste",
                placeholder="Datum;Preis_EUR_MWh\n01.01.2024;85.50\n02.01.2024;92.30")
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

    with import_tabs[2]:
        st.markdown("#### 📊 Historische Forward-/Futures-Preise")
        input_method_fw = st.radio("Eingabemethode", ["Datei-Upload", "Copy & Paste", "Manuell eingeben"],
                                   key="fw_method", horizontal=True)

        fw_df_raw = None
        if input_method_fw == "Datei-Upload":
            fw_file = st.file_uploader("Forwardpreise hochladen (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="fw_upload")
            if fw_file:
                fw_df_raw = parse_input(uploaded_file=fw_file)
        elif input_method_fw == "Copy & Paste":
            fw_text = st.text_area("Daten einfügen", height=200, key="fw_paste",
                placeholder="Datum;Produkt;Preis_EUR_MWh\n01.01.2024;Cal-25 Base;82.50")
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
            fig.update_layout(title="Lastprofil / Verbrauchsverlauf",
                xaxis_title="Zeit", yaxis_title="Verbrauch (kW)",
                template="plotly_white", height=500, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

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
                fig2.update_layout(title="Typisches Tagesprofil",
                    xaxis_title="Stunde", yaxis_title="Verbrauch (kW)",
                    template="plotly_white", height=400)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("📥 Bitte zuerst Lastgang-Daten importieren.")

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
            fig.add_trace(go.Scatter(x=df_sp["Datum"], y=df_sp["Spotpreis"],
                mode="lines", name="Spotpreis", line=dict(color="#e67e22", width=1.5)))
            if len(df_sp) > 30:
                df_sp_copy = df_sp.copy()
                df_sp_copy["MA30"] = df_sp_copy["Spotpreis"].rolling(30).mean()
                fig.add_trace(go.Scatter(x=df_sp_copy["Datum"], y=df_sp_copy["MA30"],
                    mode="lines", name="30-Tage-Ø", line=dict(color="#8e44ad", width=2.5)))
            fig.update_layout(title="Historische Spotpreise",
                xaxis_title="Datum", yaxis_title="Preis (€/MWh)",
                template="plotly_white", height=500, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=df_sp["Spotpreis"], nbinsx=50,
                marker_color="#3498db", opacity=0.8))
            fig3.update_layout(title="Preisverteilung Spotmarkt",
                xaxis_title="Preis (€/MWh)", yaxis_title="Häufigkeit",
                template="plotly_white", height=350)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("📥 Bitte zuerst Spotpreis-Daten importieren.")

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
            st.info("📥 Bitte zuerst Forward-Daten importieren.")


# ═════════════════════════════════════════════
# TAB 3: STRATEGIEN KONFIGURATION
# ═════════════════════════════════════════════
with tab3:
    st.markdown("### 🔧 Strategien konfigurieren")

    st.markdown("""
    > **Idee:** Sie legen fest, welchen **Anteil (Quote)** Ihres Volumens Sie über den 
    > **Terminmarkt** (zum Forwardpreis) beschaffen und welchen Anteil am **Spotmarkt** 
    > (zum jeweiligen Tagesprice). Die **Strategie** bestimmt, wie die Terminbeschaffung 
    > über die Zeit verteilt wird.
    """)

    st.markdown("#### Beschaffungsparameter")
    col1, col2 = st.columns(2)
    with col1:
        total_volume = st.number_input("Jahresverbrauch (MWh)", min_value=1.0, value=10000.0, step=100.0,
            help="Gesamter Jahresverbrauch in MWh")
        forward_price_input = st.number_input("Referenz-Forwardpreis (€/MWh)", min_value=0.0, value=80.0, step=1.0,
            help="Der Terminpreis, zu dem die fixierte Menge beschafft worden wäre")
    with col2:
        spread_cost = st.number_input("Transaktionskosten/Spread (€/MWh)", min_value=0.0, value=0.5, step=0.1,
            help="Optionale Kosten pro MWh auf Termingeschäfte")
        st.info(f"Effektiver Terminpreis: **{forward_price_input + spread_cost:.2f} €/MWh**")

    st.markdown("---")
    st.markdown("#### Fixierungsquoten (Terminanteil)")
    st.caption("Quote = Anteil der Menge, die über den Terminmarkt beschafft wird. Rest wird am Spotmarkt gekauft.")

    quota_mode = st.radio("Quotenmodus", ["Vordefinierte Quoten", "Benutzerdefiniert"], horizontal=True)

    if quota_mode == "Vordefinierte Quoten":
        selected_quotas = st.multiselect("Quoten auswählen (%)",
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            default=[0, 20, 40, 60, 80, 100])
        quotas = sorted([q / 100 for q in selected_quotas])
    else:
        quota_text = st.text_input("Quoten eingeben (kommagetrennt, in %)", value="0, 25, 50, 75, 100")
        quotas = sorted([float(q.strip()) / 100 for q in quota_text.split(",") if q.strip()])

    st.markdown("---")
    st.markdown("#### Strategietypen")
    st.caption("Die Strategie bestimmt, WIE die Terminquote über die Zeit verteilt wird.")

    available_strategies = ["Gleichmäßig", "Frontloaded", "Backloaded", "Regelbasiert"]
    selected_strategies = st.multiselect("Strategien auswählen", available_strategies,
        default=["Gleichmäßig", "Frontloaded", "Backloaded"],
        help="Gleichmäßig = konstante Quote. Frontloaded = mehr am Anfang fixiert. Backloaded = mehr am Ende.")

    rule_params = {}
    if "Regelbasiert" in selected_strategies:
        st.markdown("**Parameter für regelbasierte Strategie:**")
        rule_threshold = st.slider("Kaufschwelle (% des Forwardpreises)", 50, 120, 95, 1,
            help="Fixiere verstärkt wenn Spotpreis unter X% des Forwardpreises liegt")
        rule_params["threshold"] = forward_price_input * rule_threshold / 100
        st.info(f"Kaufschwelle: {rule_params['threshold']:.2f} €/MWh")

    st.session_state.strategies_config = {
        "volume": total_volume,
        "forward_price": forward_price_input,
        "spread": spread_cost,
        "quotas": quotas,
        "strategies": selected_strategies,
        "rule_params": rule_params
    }

    n_scenarios = len(selected_strategies) * len(quotas)
    st.success(f"✅ {len(selected_strategies)} Strategien × {len(quotas)} Quoten = **{n_scenarios} Szenarien** konfiguriert")

    # Vorschau der Szenarien
    with st.expander("📋 Szenarien-Vorschau"):
        preview_data = []
        for s in selected_strategies:
            for q in quotas:
                preview_data.append({
                    "Strategie": s,
                    "Terminquote": f"{q*100:.0f}%",
                    "Spotquote": f"{(1-q)*100:.0f}%",
                    "Beschreibung": f"{q*100:.0f}% Termin @ {forward_price_input+spread_cost:.2f} €/MWh + {(1-q)*100:.0f}% Spot"
                })
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════
# TAB 4: BACKTESTING
# ═════════════════════════════════════════════
with tab4:
    st.markdown("### 🔄 Backtesting durchführen")

    st.markdown("""
    > **Was wird simuliert?** Für jeden Tag im gewählten Zeitraum wird berechnet, was die 
    > Beschaffung **tatsächlich gekostet hätte**, wenn Sie X% über den Terminmarkt (zum 
    > Forwardpreis) und den Rest am Spotmarkt (zum historischen Spotpreis) beschafft hätten.
    """)

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
                mask = (df_spot["Datum"].dt.date >= start_date) & (df_spot["Datum"].dt.date <= end_date)
                spot_filtered = df_spot[mask].copy().reset_index(drop=True)

                if len(spot_filtered) == 0:
                    st.error("Keine Daten im gewählten Zeitraum!")
                else:
                    results = []
                    all_series = {}

                    progress = st.progress(0)
                    total_scenarios = len(config["strategies"]) * len(config["quotas"])
                    scenario_idx = 0

                    for strategy in config["strategies"]:
                        for quota in config["quotas"]:
                            calc = simulate_procurement(
                                spot_filtered["Spotpreis"],
                                config["forward_price"],
                                config["volume"],
                                quota, strategy,
                                spread=config["spread"],
                                rule_params=config.get("rule_params")
                            )

                            if calc:
                                risk = calculate_risk_metrics(calc["total_kosten"])
                                label = f"{strategy} ({int(quota*100)}% Termin)"

                                results.append({
                                    "Strategie": label,
                                    "Typ": strategy,
                                    "Quote": quota,
                                    "Terminanteil": f"{quota*100:.0f}%",
                                    "Spotanteil": f"{(1-quota)*100:.0f}%",
                                    "Gesamtkosten_EUR": calc["total"],
                                    "Kosten_100pct_Spot_EUR": calc["total_spot_only"],
                                    "Kosten_100pct_Termin_EUR": calc["total_termin_only"],
                                    "Durchschnittspreis_EUR_MWh": calc["avg_price"],
                                    "Ø_Spotpreis_EUR_MWh": calc["avg_spot_price"],
                                    "PnL_vs_Spot_EUR": calc["pnl_vs_spot"],
                                    "PnL_vs_Termin_EUR": calc["pnl_vs_termin"],
                                    "Volatilität": risk["Volatilität"],
                                    "VaR_95": risk["VaR_95"],
                                    "CVaR_95": risk["CVaR_95"],
                                    "Max_Drawdown": risk["Max_Drawdown"],
                                    "Std_Kosten": risk["Std_Kosten"],
                                })
                                all_series[label] = {
                                    "dates": spot_filtered["Datum"].values,
                                    "cumulative": calc["cumulative"],
                                    "cum_spot_only": calc["cum_spot_only"],
                                    "cum_termin_only": calc["cum_termin_only"],
                                    "total_kosten": calc["total_kosten"],
                                    "fix_quota": calc["fix_quota"],
                                }

                            scenario_idx += 1
                            progress.progress(scenario_idx / total_scenarios)

                    progress.empty()

                    if results:
                        results_df = pd.DataFrame(results).sort_values("Gesamtkosten_EUR")
                        results_df["Rang"] = range(1, len(results_df) + 1)

                        spot_total = results_df["Kosten_100pct_Spot_EUR"].iloc[0]
                        termin_total = results_df["Kosten_100pct_Termin_EUR"].iloc[0]

                        st.session_state.backtesting_results = {
                            "results_df": results_df,
                            "series": all_series,
                            "spot_filtered": spot_filtered,
                            "config": config,
                            "spot_total": spot_total,
                            "termin_total": termin_total,
                        }

                        st.success(f"✅ Backtesting abgeschlossen! **{len(results)} Szenarien** berechnet.")

                        # Quick Preview
                        st.markdown("#### 🏆 Top 5 günstigste Strategien")
                        top5 = results_df.head(5)[["Rang", "Strategie", "Gesamtkosten_EUR",
                            "Durchschnittspreis_EUR_MWh", "PnL_vs_Spot_EUR"]].copy()
                        top5.columns = ["#", "Strategie", "Gesamtkosten (€)", "Ø Preis (€/MWh)", "PnL vs. Spot (€)"]
                        st.dataframe(
                            top5.style.format({
                                "Gesamtkosten (€)": "{:,.0f}",
                                "Ø Preis (€/MWh)": "{:.2f}",
                                "PnL vs. Spot (€)": "{:+,.0f}"
                            }),
                            use_container_width=True, hide_index=True
                        )

                        st.markdown("---")
                        st.markdown(f"""
                        **Referenzwerte:**
                        - 100% Spot hätte gekostet: **{spot_total:,.0f} €** (Ø {spot_total/config['volume']:.2f} €/MWh)
                        - 100% Termin hätte gekostet: **{termin_total:,.0f} €** (@ {config['forward_price']+config['spread']:.2f} €/MWh)
                        """)
                    else:
                        st.warning("Keine gültigen Szenarien berechnet.")


# ═════════════════════════════════════════════
# TAB 5: ERGEBNISSE & DASHBOARD
# ═════════════════════════════════════════════
with tab5:
    st.markdown("### 📊 Ergebnisse & Dashboard")

    if st.session_state.backtesting_results is None:
        st.info("🔄 Bitte zuerst Backtesting durchführen.")
    else:
        bt = st.session_state.backtesting_results
        results_df = bt["results_df"]
        series = bt["series"]
        spot_total = bt["spot_total"]
        termin_total = bt["termin_total"]
        config = bt["config"]

        best = results_df.iloc[0]
        worst = results_df.iloc[-1]

        # KPIs
        st.markdown("#### 🎯 Key Performance Indicators")
        kpi_cols = st.columns(5)
        with kpi_cols[0]:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:0.85rem;">Beste Strategie</div>
                <div style="font-size:1.1rem;font-weight:700;">{best['Strategie']}</div>
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
            st.markdown(f"""<div class="metric-card-blue">
                <div style="font-size:0.85rem;">100% Spot</div>
                <div style="font-size:1.3rem;font-weight:700;">{spot_total:,.0f} €</div>
            </div>""", unsafe_allow_html=True)
        with kpi_cols[4]:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:0.85rem;">100% Termin</div>
                <div style="font-size:1.3rem;font-weight:700;">{termin_total:,.0f} €</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        chart_tabs = st.tabs(["📊 Kostenvergleich", "📈 Kumulative Kosten", "🗺️ Heatmap",
                              "⚠️ Risiko & MtM", "💡 Empfehlung"])

        # --- Kostenvergleich ---
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
            fig.add_hline(y=termin_total, line_dash="dash", line_color="blue",
                annotation_text=f"100% Termin: {termin_total:,.0f}€")
            fig.update_layout(title="Gesamtkosten nach Strategie",
                yaxis_title="Gesamtkosten (€)", template="plotly_white",
                height=500, showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            # Durchschnittspreis
            fig2 = px.bar(results_df, x="Strategie", y="Durchschnittspreis_EUR_MWh",
                color="Durchschnittspreis_EUR_MWh", color_continuous_scale="RdYlGn_r",
                title="Durchschnittlicher Beschaffungspreis (€/MWh)",
                text="Durchschnittspreis_EUR_MWh")
            fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig2.update_layout(template="plotly_white", height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

        # --- Kumulative Kosten ---
        with chart_tabs[1]:
            fig = go.Figure()
            colors_plotly = px.colors.qualitative.Plotly

            # Referenzlinien: 100% Spot und 100% Termin
            first_key = list(series.keys())[0]
            fig.add_trace(go.Scatter(
                x=series[first_key]["dates"], y=series[first_key]["cum_spot_only"],
                mode="lines", name="100% Spot (Benchmark)",
                line=dict(color="red", width=2, dash="dash")))
            fig.add_trace(go.Scatter(
                x=series[first_key]["dates"], y=series[first_key]["cum_termin_only"],
                mode="lines", name="100% Termin (Benchmark)",
                line=dict(color="blue", width=2, dash="dash")))

            for i, (label, s) in enumerate(series.items()):
                fig.add_trace(go.Scatter(
                    x=s["dates"], y=s["cumulative"],
                    mode="lines", name=label,
                    line=dict(color=colors_plotly[i % len(colors_plotly)], width=2)))

            fig.update_layout(
                title="Kumulative Beschaffungskosten: Strategie vs. Benchmarks",
                xaxis_title="Datum", yaxis_title="Kumulative Kosten (€)",
                template="plotly_white", height=600,
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)

        # --- Heatmap ---
        with chart_tabs[2]:
            heatmap_data = results_df.pivot_table(
                index="Typ", columns="Quote", values="Gesamtkosten_EUR", aggfunc="first")
            if not heatmap_data.empty and heatmap_data.shape[1] > 1:
                heatmap_data.columns = [f"{int(c*100)}% Termin" for c in heatmap_data.columns]
                fig = px.imshow(
                    heatmap_data.values,
                    x=heatmap_data.columns.tolist(),
                    y=heatmap_data.index.tolist(),
                    color_continuous_scale="RdYlGn_r",
                    title="Heatmap: Strategietyp × Terminquote → Gesamtkosten (€)",
                    text_auto=",.0f", aspect="auto")
                fig.update_layout(template="plotly_white", height=450)
                st.plotly_chart(fig, use_container_width=True)

            # Heatmap: Preis
            heatmap_price = results_df.pivot_table(
                index="Typ", columns="Quote", values="Durchschnittspreis_EUR_MWh", aggfunc="first")
            if not heatmap_price.empty and heatmap_price.shape[1] > 1:
                heatmap_price.columns = [f"{int(c*100)}% Termin" for c in heatmap_price.columns]
                fig2 = px.imshow(
                    heatmap_price.values,
                    x=heatmap_price.columns.tolist(),
                    y=heatmap_price.index.tolist(),
                    color_continuous_scale="RdYlGn_r",
                    title="Heatmap: Strategietyp × Terminquote → Ø Beschaffungspreis (€/MWh)",
                    text_auto=".2f", aspect="auto")
                fig2.update_layout(template="plotly_white", height=450)
                st.plotly_chart(fig2, use_container_width=True)

        # --- Risiko & MtM ---
        with chart_tabs[3]:
            st.markdown("#### ⚠️ Risikokennzahlen & Mark-to-Market")

            # MtM / PnL Tabelle
            st.markdown("##### 💰 Mark-to-Market: PnL vs. Benchmarks")
            mtm_df = results_df[["Rang", "Strategie", "Gesamtkosten_EUR", "PnL_vs_Spot_EUR", "PnL_vs_Termin_EUR",
                                 "Durchschnittspreis_EUR_MWh", "Ø_Spotpreis_EUR_MWh"]].copy()
            mtm_df.columns = ["#", "Strategie", "Gesamtkosten (€)", "PnL vs. Spot (€)", "PnL vs. Termin (€)",
                              "Ø Preis (€/MWh)", "Ø Spotpreis (€/MWh)"]
            st.dataframe(
                mtm_df.style.format({
                    "Gesamtkosten (€)": "{:,.0f}",
                    "PnL vs. Spot (€)": "{:+,.0f}",
                    "PnL vs. Termin (€)": "{:+,.0f}",
                    "Ø Preis (€/MWh)": "{:.2f}",
                    "Ø Spotpreis (€/MWh)": "{:.2f}",
                }),
                use_container_width=True, hide_index=True
            )

            st.caption("PnL vs. Spot > 0 → Strategie war günstiger als 100% Spot. PnL vs. Termin > 0 → günstiger als 100% Termin.")

            st.markdown("---")

            # Risiko-Tabelle
            st.markdown("##### 📉 Risikokennzahlen")
            risk_df = results_df[["Strategie", "Std_Kosten", "Volatilität", "VaR_95", "CVaR_95"]].copy()
            risk_df.columns = ["Strategie", "Std Periodenkosten (€)", "Volatilität (ann.)", "VaR 95%", "CVaR 95%"]
            st.dataframe(
                risk_df.style.format({
                    "Std Periodenkosten (€)": "{:,.2f}",
                    "Volatilität (ann.)": "{:.4f}",
                    "VaR 95%": "{:,.2f}",
                    "CVaR 95%": "{:,.2f}"
                }),
                use_container_width=True, hide_index=True
            )

            # PnL Balkendiagramm
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Bar(
                x=results_df["Strategie"], y=results_df["PnL_vs_Spot_EUR"],
                name="PnL vs. Spot",
                marker_color=np.where(results_df["PnL_vs_Spot_EUR"] >= 0, "#27ae60", "#e74c3c"),
                text=[f"{v:+,.0f}€" for v in results_df["PnL_vs_Spot_EUR"]],
                textposition="outside"
            ))
            fig_pnl.update_layout(title="Mark-to-Market: Ersparnis (+) / Mehrkosten (-) vs. 100% Spot",
                yaxis_title="PnL (€)", template="plotly_white", height=450, xaxis_tickangle=-45)
            fig_pnl.add_hline(y=0, line_color="black", line_width=1)
            st.plotly_chart(fig_pnl, use_container_width=True)

            # Risk-Return Scatter
            fig_rr = px.scatter(results_df, x="Std_Kosten", y="Gesamtkosten_EUR",
                color="Typ", symbol="Typ",
                size=[max(10, abs(v)/1000) for v in results_df["PnL_vs_Spot_EUR"]],
                title="Risiko-Kosten-Profil (Größe = |PnL vs. Spot|)",
                template="plotly_white",
                hover_data=["Strategie", "Durchschnittspreis_EUR_MWh"],
                labels={"Std_Kosten": "Risiko (Std Periodenkosten €)", "Gesamtkosten_EUR": "Gesamtkosten (€)"})
            fig_rr.update_layout(height=500)
            st.plotly_chart(fig_rr, use_container_width=True)

        # --- Empfehlung ---
        with chart_tabs[4]:
            st.markdown("#### 💡 Handlungsempfehlung")

            best_row = results_df.iloc[0]
            pnl_spot = best_row["PnL_vs_Spot_EUR"]
            pnl_termin = best_row["PnL_vs_Termin_EUR"]

            if pnl_spot > 0:
                savings_text = f"**{pnl_spot:,.0f} € günstiger** als 100% Spot"
            else:
                savings_text = f"**{abs(pnl_spot):,.0f} € teurer** als 100% Spot"

            st.markdown(f"""
            <div class="recommendation-box">
            <h4>🏆 Empfohlene Strategie: {best_row['Strategie']}</h4>
            <p>Basierend auf dem historischen Backtesting hätte diese Strategie die 
            <strong>niedrigsten Gesamtbeschaffungskosten</strong> erzielt:</p>
            <table style="width:100%; border-collapse:collapse; margin:1rem 0;">
                <tr><td style="padding:4px 8px;"><strong>Gesamtkosten:</strong></td>
                    <td style="padding:4px 8px;">{best_row['Gesamtkosten_EUR']:,.0f} €</td></tr>
                <tr><td style="padding:4px 8px;"><strong>Ø Beschaffungspreis:</strong></td>
                    <td style="padding:4px 8px;">{best_row['Durchschnittspreis_EUR_MWh']:.2f} €/MWh</td></tr>
                <tr><td style="padding:4px 8px;"><strong>vs. 100% Spot:</strong></td>
                    <td style="padding:4px 8px;">{savings_text}</td></tr>
                <tr><td style="padding:4px 8px;"><strong>vs. 100% Termin:</strong></td>
                    <td style="padding:4px 8px;">{pnl_termin:+,.0f} €</td></tr>
                <tr><td style="padding:4px 8px;"><strong>Terminquote:</strong></td>
                    <td style="padding:4px 8px;">{best_row['Terminanteil']}</td></tr>
                <tr><td style="padding:4px 8px;"><strong>Spotquote:</strong></td>
                    <td style="padding:4px 8px;">{best_row['Spotanteil']}</td></tr>
            </table>
            <p><em>⚠️ Hinweis: Vergangene Performance ist kein verlässlicher Indikator für 
            zukünftige Ergebnisse. Die Empfehlung basiert ausschließlich auf historischem 
            Backtesting im gewählten Zeitraum.</em></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🏅 Vollständiges Ranking")
            display_df = results_df[["Rang", "Strategie", "Terminanteil", "Spotanteil",
                "Gesamtkosten_EUR", "Durchschnittspreis_EUR_MWh", "PnL_vs_Spot_EUR",
                "PnL_vs_Termin_EUR"]].copy()
            display_df.columns = ["#", "Strategie", "Termin", "Spot", "Gesamtkosten (€)",
                "Ø Preis (€/MWh)", "PnL vs Spot (€)", "PnL vs Termin (€)"]
            st.dataframe(
                display_df.style.format({
                    "Gesamtkosten (€)": "{:,.0f}",
                    "Ø Preis (€/MWh)": "{:.2f}",
                    "PnL vs Spot (€)": "{:+,.0f}",
                    "PnL vs Termin (€)": "{:+,.0f}",
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
            st.download_button("📥 Ergebnisse als CSV herunterladen",
                csv_buffer.getvalue(), "backtesting_ergebnisse.csv",
                "text/csv", use_container_width=True)

        with col2:
            st.markdown("#### 📊 Excel Export")
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                results_df.to_excel(writer, sheet_name="Ergebnisse", index=False)
                if st.session_state.spot_df is not None:
                    bt["spot_filtered"].to_excel(writer, sheet_name="Spotpreise", index=False)
            st.download_button("📥 Ergebnisse als Excel herunterladen",
                excel_buffer.getvalue(), "backtesting_ergebnisse.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📋 Zeitreihen-Export")
        series = bt["series"]
        if series:
            first_key = list(series.keys())[0]
            series_data = {"Datum": series[first_key]["dates"]}
            series_data["Kum_100pct_Spot"] = series[first_key]["cum_spot_only"]
            series_data["Kum_100pct_Termin"] = series[first_key]["cum_termin_only"]
            for label, s in series.items():
                series_data[f"Kum_{label}"] = s["cumulative"]

            series_df = pd.DataFrame(series_data)
            csv_ts = StringIO()
            series_df.to_csv(csv_ts, index=False, sep=";", decimal=",")
            st.download_button("📥 Zeitreihen als CSV herunterladen",
                csv_ts.getvalue(), "backtesting_zeitreihen.csv",
                "text/csv", use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#95a5a6;font-size:0.85rem;'>"
    "⚡ Energie-Beschaffungs-Engine v2.0 | Simulation · Backtesting · Mark-to-Market"
    "</div>",
    unsafe_allow_html=True
)

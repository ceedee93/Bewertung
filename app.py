import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(
    page_title="Energiebeschaffung Simulation & Backtesting",
    layout="wide",
    page_icon="⚡"
)

# ── Styling ──
st.markdown("""
<style>
    .stMetric { background: #1e1e2e; padding: 15px; border-radius: 10px; border-left: 4px solid #4CAF50; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    .big-number { font-size: 2rem; font-weight: bold; color: #4CAF50; }
    .info-box { background: #262640; padding: 15px; border-radius: 8px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──
defaults = {
    'load_df': None, 'spot_df': None, 'forward_df': None,
    'bt_results': None, 'bt_merged': None, 'bt_spot_total': None,
    'bt_demand_total': None, 'bt_avg_spot': None,
    'strategy_config': {
        'forward_shares': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'patterns': ['Gleichmäßig'],
        'n_tranches': 6,
        'tx_cost': 0.0
    }
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helper Functions ──
def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith('.csv'):
            raw = uploaded_file.read()
            uploaded_file.seek(0)
            text = raw.decode('utf-8', errors='replace')
            for sep in [';', ',', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep)
                    if len(df.columns) >= 2:
                        return df
                except:
                    continue
            return pd.read_csv(io.StringIO(text))
        elif name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")
    return None


def parse_pasted_text(text):
    """Parst eingefügten Text (Copy-Paste) mit Auto-Erkennung des Trennzeichens."""
    if not text or not text.strip():
        return None
    text = text.strip()
    for sep in ['\t', ';', ',', '|']:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine='python')
            if len(df.columns) >= 2 and len(df) > 0:
                return df
        except:
            continue
    # Letzter Versuch: Whitespace
    try:
        df = pd.read_csv(io.StringIO(text), sep=r'\s+', engine='python')
        if len(df.columns) >= 2:
            return df
    except:
        pass
    return None


def data_input_block(label, file_key, paste_key):
    """Wiederverwendbarer Datenimport-Block mit Upload + Copy-Paste."""
    input_method = st.radio(
        f"Eingabemethode für {label}",
        ["📁 Datei hochladen", "📋 Copy & Paste"],
        horizontal=True,
        key=f"method_{file_key}"
    )

    raw = None

    if input_method == "📁 Datei hochladen":
        uploaded = st.file_uploader(
            f"{label} hochladen (CSV, XLSX)",
            type=['csv', 'xlsx', 'xls'],
            key=file_key
        )
        if uploaded:
            raw = load_file(uploaded)

    else:  # Copy & Paste
        st.caption("Daten direkt einfügen – z.B. aus Excel kopiert (Tab-getrennt), CSV (Semikolon/Komma) etc.")
        pasted = st.text_area(
            f"Daten hier einfügen ({label})",
            height=200,
            key=paste_key,
            placeholder="Datum;Wert\n01.01.2025;42.5\n02.01.2025;38.2\n..."
        )
        if pasted and pasted.strip():
            raw = parse_pasted_text(pasted)
            if raw is None and pasted.strip():
                st.error("❌ Text konnte nicht als Tabelle erkannt werden. "
                         "Unterstützte Formate: Tab-getrennt, Semikolon, Komma, Pipe.")

    return raw


def detect_datetime_col(df):
    for col in df.columns:
        if df[col].dtype == 'object' or 'date' in col.lower() or 'zeit' in col.lower() or 'time' in col.lower():
            try:
                parsed = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                if parsed.notna().sum() > len(df) * 0.5:
                    return col
            except:
                continue
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() > len(df) * 0.5:
                return col
        except:
            continue
    return None


def detect_numeric_col(df, exclude=None):
    exclude = set(exclude or [])
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
        try:
            pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            if pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').notna().sum() > len(df) * 0.5:
                return col
        except:
            continue
    return None


def parse_and_clean(df, time_col, val_col, val_name):
    out = df[[time_col, val_col]].copy()
    out[time_col] = pd.to_datetime(out[time_col], dayfirst=True, errors='coerce')
    if out[val_col].dtype == object:
        out[val_col] = out[val_col].astype(str).str.replace(',', '.').str.strip()
    out[val_col] = pd.to_numeric(out[val_col], errors='coerce')
    out = out.dropna().rename(columns={time_col: 'datetime', val_col: val_name})
    out = out.sort_values('datetime').reset_index(drop=True)
    return out


# ── Sidebar Navigation ──
st.sidebar.title("⚡ Energiebeschaffung")
st.sidebar.caption("Simulation & Backtesting")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "📥 Datenimport",
    "⚙️ Strategien",
    "🔬 Backtesting",
    "📊 Dashboard"
])

# Status indicators in sidebar
st.sidebar.divider()
st.sidebar.markdown("### Datenstatus")
for label, key in [("Lastprofil", "load_df"), ("Spotpreise", "spot_df"), ("Terminpreise", "forward_df")]:
    if st.session_state[key] is not None:
        df = st.session_state[key]
        col_name = [c for c in df.columns if c != 'datetime'][0]
        st.sidebar.success(f"✅ {label}: {len(df)} Zeilen")
    else:
        st.sidebar.warning(f"⏳ {label}: fehlt")


# ══════════════════════════════════════════════════════════════
# 📥 DATENIMPORT
# ══════════════════════════════════════════════════════════════
if page == "📥 Datenimport":
    st.header("📥 Datenimport")

    st.markdown("""
> **Drei Datensätze werden benötigt:**
> 1. **Lastprofil** – Ihr Verbrauch während der **Lieferperiode** (z.B. stündlich/täglich für 2025)
> 2. **Spotpreise** – Historische Spotmarktpreise **während der Lieferperiode** (gleicher Zeitraum wie Last)
> 3. **Terminmarktpreise** – Forward-Preise **VOR der Lieferperiode** (z.B. Preise des Cal-25 Forwards in 2023/2024)
    """)

    # ── 1. Lastprofil ──
    st.subheader("1️⃣ Lastprofil (Lieferperiode)")
    st.caption("Zeitreihe Ihres Stromverbrauchs in MWh – z.B. stündlich oder täglich")
    raw = data_input_block("Lastprofil", "load_up", "load_paste")

    if raw is not None:
        with st.expander("Vorschau Rohdaten", expanded=True):
            st.dataframe(raw.head(10), use_container_width=True)

        cols = list(raw.columns)
        auto_t = detect_datetime_col(raw)
        auto_v = detect_numeric_col(raw, [auto_t] if auto_t else [])

        c1, c2 = st.columns(2)
        tc = c1.selectbox("Zeitspalte", cols, index=cols.index(auto_t) if auto_t and auto_t in cols else 0, key='ltc')
        vc = c2.selectbox("Verbrauch [MWh]", cols, index=cols.index(auto_v) if auto_v and auto_v in cols else min(1, len(cols)-1), key='lvc')

        if st.button("✅ Lastprofil übernehmen", key='lb'):
            df = parse_and_clean(raw, tc, vc, 'load_mwh')
            if len(df) > 0:
                st.session_state.load_df = df
                st.success(f"Lastprofil geladen: {len(df)} Einträge, {df['datetime'].min().date()} bis {df['datetime'].max().date()}, Gesamt: {df['load_mwh'].sum():,.1f} MWh")
            else:
                st.error("Keine gültigen Daten nach Parsing.")

    if st.session_state.load_df is not None:
        d = st.session_state.load_df
        st.info(f"📊 Lastprofil aktiv: **{d['datetime'].min().date()}** bis **{d['datetime'].max().date()}** | {len(d)} Einträge | {d['load_mwh'].sum():,.0f} MWh gesamt")

    st.divider()

    # ── 2. Spotpreise ──
    st.subheader("2️⃣ Spotpreise (während Lieferperiode)")
    st.caption("Historische Day-Ahead / Spotpreise in €/MWh – gleicher Zeitraum wie das Lastprofil")
    raw = data_input_block("Spotpreise", "spot_up", "spot_paste")

    if raw is not None:
        with st.expander("Vorschau Rohdaten", expanded=True):
            st.dataframe(raw.head(10), use_container_width=True)

        cols = list(raw.columns)
        auto_t = detect_datetime_col(raw)
        auto_v = detect_numeric_col(raw, [auto_t] if auto_t else [])

        c1, c2 = st.columns(2)
        tc = c1.selectbox("Zeitspalte", cols, index=cols.index(auto_t) if auto_t and auto_t in cols else 0, key='stc')
        vc = c2.selectbox("Preis [€/MWh]", cols, index=cols.index(auto_v) if auto_v and auto_v in cols else min(1, len(cols)-1), key='svc')

        if st.button("✅ Spotpreise übernehmen", key='sb'):
            df = parse_and_clean(raw, tc, vc, 'spot_price')
            if len(df) > 0:
                st.session_state.spot_df = df
                st.success(f"Spotpreise geladen: {len(df)} Einträge, Ø {df['spot_price'].mean():.2f} €/MWh")
            else:
                st.error("Keine gültigen Daten nach Parsing.")

    if st.session_state.spot_df is not None:
        d = st.session_state.spot_df
        st.info(f"💰 Spotpreise aktiv: **{d['datetime'].min().date()}** bis **{d['datetime'].max().date()}** | Ø {d['spot_price'].mean():.2f} €/MWh")

    st.divider()

    # ── 3. Terminmarktpreise ──
    st.subheader("3️⃣ Terminmarktpreise (VOR Lieferperiode)")
    st.caption("Forward-/Terminpreise für das Lieferprodukt, beobachtet VOR Lieferbeginn – z.B. tägliche Cal-25 Baseload-Preise im Jahr 2024")
    raw = data_input_block("Terminmarktpreise", "fwd_up", "fwd_paste")

    if raw is not None:
        with st.expander("Vorschau Rohdaten", expanded=True):
            st.dataframe(raw.head(10), use_container_width=True)

        cols = list(raw.columns)
        auto_t = detect_datetime_col(raw)
        auto_v = detect_numeric_col(raw, [auto_t] if auto_t else [])

        c1, c2 = st.columns(2)
        tc = c1.selectbox("Zeitspalte", cols, index=cols.index(auto_t) if auto_t and auto_t in cols else 0, key='ftc')
        vc = c2.selectbox("Preis [€/MWh]", cols, index=cols.index(auto_v) if auto_v and auto_v in cols else min(1, len(cols)-1), key='fvc')

        if st.button("✅ Terminmarktpreise übernehmen", key='fb'):
            df = parse_and_clean(raw, tc, vc, 'forward_price')
            if len(df) > 0:
                st.session_state.forward_df = df
                st.success(f"Terminmarktpreise geladen: {len(df)} Einträge, Ø {df['forward_price'].mean():.2f} €/MWh")
            else:
                st.error("Keine gültigen Daten nach Parsing.")

    if st.session_state.forward_df is not None:
        d = st.session_state.forward_df
        st.info(f"📈 Terminmarktpreise aktiv: **{d['datetime'].min().date()}** bis **{d['datetime'].max().date()}** | Ø {d['forward_price'].mean():.2f} €/MWh")


# ══════════════════════════════════════════════════════════════
# ⚙️ STRATEGIEN
# ══════════════════════════════════════════════════════════════
elif page == "⚙️ Strategien":
    st.header("⚙️ Beschaffungsstrategien konfigurieren")

    st.markdown("""
> **Wie funktioniert die Simulation?**
>
> Sie bestimmen, welchen **Anteil** Ihres Gesamtbedarfs Sie **vorab am Terminmarkt** sichern.
> Der Rest wird **während der Lieferung am Spotmarkt** beschafft.
>
> - **Terminanteil (z.B. 40%):** Wird VOR Lieferbeginn in Tranchen zu Forward-Preisen gekauft
> - **Spotanteil (z.B. 60%):** Wird WÄHREND der Lieferung zum jeweiligen Spotpreis gekauft
> - **Beschaffungsmuster:** Bestimmt, WANN und WIE die Terminmarkt-Tranchen verteilt werden
    """)

    st.subheader("Terminmarkt-Anteile")
    mode = st.radio("Modus", ["Standard (0% bis 100% in 10er-Schritten)", "Benutzerdefiniert"], horizontal=True)

    if mode.startswith("Standard"):
        forward_shares = [i/100 for i in range(0, 101, 10)]
    else:
        txt = st.text_input("Terminanteile in % (kommagetrennt)", "0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100")
        try:
            forward_shares = sorted(set([max(0, min(1, float(x.strip())/100)) for x in txt.split(',')]))
        except:
            forward_shares = [i/100 for i in range(0, 101, 10)]
            st.warning("Ungültige Eingabe, verwende Standard.")

    st.write("**Simulierte Anteile:**", ", ".join([f"{s*100:.0f}%" for s in forward_shares]))

    st.subheader("Beschaffungsmuster am Terminmarkt")
    st.caption("Wie werden die Forward-Käufe über den Beschaffungszeitraum verteilt?")

    patterns = st.multiselect(
        "Muster",
        ["Gleichmäßig", "Frontloaded (früh mehr kaufen)", "Backloaded (spät mehr kaufen)"],
        default=["Gleichmäßig"]
    )
    if not patterns:
        patterns = ["Gleichmäßig"]

    n_tranches = st.slider("Anzahl Beschaffungstranchen (Kaufzeitpunkte vor Lieferung)", 2, 36, 6)

    st.subheader("Transaktionskosten (optional)")
    tx_cost = st.number_input("Aufschlag pro MWh Forward-Kauf [€/MWh]", min_value=0.0, value=0.0, step=0.05, format="%.2f")

    st.session_state.strategy_config = {
        'forward_shares': forward_shares,
        'patterns': patterns,
        'n_tranches': n_tranches,
        'tx_cost': tx_cost
    }

    st.success(f"Konfiguration: {len(forward_shares)} Anteile × {len(patterns)} Muster = **{len(forward_shares)*len(patterns)} Szenarien**")

    # ── Preview procurement dates ──
    if st.session_state.forward_df is not None:
        st.divider()
        st.subheader("Vorschau: Beschaffungszeitpunkte & Gewichte")

        fwd = st.session_state.forward_df
        n_fwd = len(fwd)
        actual_tranches = min(n_tranches, n_fwd)

        for pat in patterns:
            indices = np.round(np.linspace(0, n_fwd - 1, actual_tranches)).astype(int)

            if pat == "Gleichmäßig":
                weights = np.ones(actual_tranches) / actual_tranches
            elif pat.startswith("Frontloaded"):
                w = np.linspace(2, 0.5, actual_tranches)
                weights = w / w.sum()
            elif pat.startswith("Backloaded"):
                w = np.linspace(0.5, 2, actual_tranches)
                weights = w / w.sum()
            else:
                weights = np.ones(actual_tranches) / actual_tranches

            preview = pd.DataFrame({
                'Nr.': range(1, actual_tranches + 1),
                'Kaufdatum': fwd.iloc[indices]['datetime'].dt.date.values,
                'Forward-Preis [€/MWh]': fwd.iloc[indices]['forward_price'].values,
                'Gewicht': [f"{w*100:.1f}%" for w in weights],
                'Gew. Preis [€/MWh]': fwd.iloc[indices]['forward_price'].values * weights
            })

            st.markdown(f"**{pat}** – Ø gewichteter Forward-Preis: **{preview['Gew. Preis [€/MWh]'].sum():.2f} €/MWh**")
            st.dataframe(preview, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# 🔬 BACKTESTING
# ══════════════════════════════════════════════════════════════
elif page == "🔬 Backtesting":
    st.header("🔬 Backtesting durchführen")

    # ── Validate data ──
    missing = []
    if st.session_state.load_df is None:
        missing.append("Lastprofil")
    if st.session_state.spot_df is None:
        missing.append("Spotpreise")
    if st.session_state.forward_df is None:
        missing.append("Terminmarktpreise")

    if missing:
        st.warning(f"⚠️ Fehlende Daten: **{', '.join(missing)}** – bitte im Tab 'Datenimport' hochladen.")
        st.stop()

    load_df = st.session_state.load_df
    spot_df = st.session_state.spot_df
    fwd_df = st.session_state.forward_df
    config = st.session_state.strategy_config

    # ── Data summary ──
    delivery_start = load_df['datetime'].min()
    delivery_end = load_df['datetime'].max()
    fwd_start = fwd_df['datetime'].min()
    fwd_end = fwd_df['datetime'].max()

    st.markdown(f"""
### Datenbasis

| Datensatz | Zeitraum | Details |
|-----------|----------|---------|
| 📊 **Lastprofil** | {delivery_start.date()} – {delivery_end.date()} | {load_df['load_mwh'].sum():,.0f} MWh, {len(load_df)} Einträge |
| 💰 **Spotpreise** | {spot_df['datetime'].min().date()} – {spot_df['datetime'].max().date()} | Ø {spot_df['spot_price'].mean():.2f} €/MWh |
| 📈 **Terminmarkt** | {fwd_start.date()} – {fwd_end.date()} | Ø {fwd_df['forward_price'].mean():.2f} €/MWh, {len(fwd_df)} Handelstage |
    """)

    # Check if forward data is before delivery
    if fwd_end >= delivery_start:
        st.warning(f"⚠️ Hinweis: Terminmarktdaten reichen bis {fwd_end.date()}, die Lieferperiode beginnt am {delivery_start.date()}. "
                   f"Es werden nur Forward-Preise **vor** Lieferbeginn verwendet.")

    st.info("""
**Was wird simuliert?**

Für jede Kombination aus Terminanteil und Beschaffungsmuster wird berechnet:
- **Terminkosten:** Der Terminanteil Ihres Bedarfs wird VOR Lieferbeginn in Tranchen am Forward-Markt gekauft
- **Spotkosten:** Der Restbedarf wird WÄHREND der Lieferung zum jeweiligen Spotpreis beschafft
- **Benchmark:** 100% Spot = Was es gekostet hätte, alles am Spotmarkt zu kaufen
    """)

    if st.button("🚀 Backtesting starten", type="primary", use_container_width=True):

        with st.spinner("Berechne Szenarien..."):

            # ── Filter forward data: only before delivery start ──
            fwd_before = fwd_df[fwd_df['datetime'] < delivery_start].copy()
            if len(fwd_before) == 0:
                st.error("❌ Keine Terminmarktdaten VOR der Lieferperiode vorhanden! "
                         "Terminmarktpreise müssen zeitlich VOR dem Lastprofil liegen.")
                st.stop()

            # ── Merge load + spot ──
            # Handle different granularities by merging on date
            load_c = load_df.copy()
            spot_c = spot_df.copy()

            # Try exact datetime merge first
            merged = pd.merge(load_c, spot_c, on='datetime', how='inner')

            # If no match, try matching on date only
            if len(merged) == 0:
                load_c['date'] = load_c['datetime'].dt.date
                spot_c['date'] = spot_c['datetime'].dt.date
                # Aggregate load to daily and spot to daily avg
                load_daily = load_c.groupby('date').agg({'load_mwh': 'sum', 'datetime': 'first'}).reset_index()
                spot_daily = spot_c.groupby('date').agg({'spot_price': 'mean'}).reset_index()
                merged = pd.merge(load_daily, spot_daily, on='date', how='inner')

            if len(merged) == 0:
                st.error("❌ Keine überlappenden Zeitpunkte zwischen Lastprofil und Spotpreisen! "
                         "Stellen Sie sicher, dass beide den gleichen Zeitraum abdecken.")
                st.stop()

            # ── Calculate benchmark: 100% Spot ──
            merged['spot_cost_per_period'] = merged['load_mwh'] * merged['spot_price']
            total_demand = merged['load_mwh'].sum()
            total_spot_cost = merged['spot_cost_per_period'].sum()
            avg_spot = total_spot_cost / total_demand if total_demand > 0 else 0

            # ── Cumulative spot cost for time-series chart ──
            merged = merged.sort_values('datetime').reset_index(drop=True)
            merged['cum_spot_cost'] = merged['spot_cost_per_period'].cumsum()
            merged['cum_load'] = merged['load_mwh'].cumsum()

            # ── Run scenarios ──
            fwd_vals = fwd_before['forward_price'].values
            fwd_dts = fwd_before['datetime'].values
            n_fwd = len(fwd_before)

            results = []
            cumulative_data = {}

            for pat in config['patterns']:
                actual_tranches = min(config['n_tranches'], n_fwd)
                indices = np.round(np.linspace(0, n_fwd - 1, actual_tranches)).astype(int)

                if pat == "Gleichmäßig":
                    weights = np.ones(actual_tranches) / actual_tranches
                elif pat.startswith("Frontloaded"):
                    w = np.linspace(2, 0.5, actual_tranches)
                    weights = w / w.sum()
                elif pat.startswith("Backloaded"):
                    w = np.linspace(0.5, 2, actual_tranches)
                    weights = w / w.sum()
                else:
                    weights = np.ones(actual_tranches) / actual_tranches

                # Weighted average forward price
                tranche_prices = fwd_vals[indices]
                wavg_fwd = np.sum(weights * tranche_prices)

                for fwd_share in config['forward_shares']:
                    spot_share = 1.0 - fwd_share

                    # Forward cost: fwd_share of total demand at weighted avg forward price
                    fwd_volume = total_demand * fwd_share
                    fwd_cost = fwd_volume * wavg_fwd

                    # Transaction costs
                    tx_total = fwd_volume * config['tx_cost']

                    # Spot cost: spot_share of each period's demand at that period's spot price
                    spot_cost = (merged['load_mwh'] * spot_share * merged['spot_price']).sum()

                    total_cost = fwd_cost + spot_cost + tx_total
                    avg_price = total_cost / total_demand if total_demand > 0 else 0
                    pnl_vs_spot = total_spot_cost - total_cost
                    pct_saving = (pnl_vs_spot / total_spot_cost * 100) if total_spot_cost != 0 else 0

                    name = f"{pat.split('(')[0].strip()} ({fwd_share*100:.0f}% Termin / {spot_share*100:.0f}% Spot)"

                    results.append({
                        'Strategie': name,
                        'Muster': pat.split('(')[0].strip(),
                        'Terminanteil [%]': fwd_share * 100,
                        'Spotanteil [%]': spot_share * 100,
                        'Ø Forward-Preis [€/MWh]': wavg_fwd,
                        'Terminvolumen [MWh]': fwd_volume,
                        'Terminkosten [€]': fwd_cost,
                        'Spotvolumen [MWh]': total_demand * spot_share,
                        'Ø Spot-Preis [€/MWh]': avg_spot,
                        'Spotkosten [€]': spot_cost,
                        'TX-Kosten [€]': tx_total,
                        'Gesamtkosten [€]': total_cost,
                        'Ø Beschaffungspreis [€/MWh]': avg_price,
                        'PnL vs. Spot [€]': pnl_vs_spot,
                        'Ersparnis [%]': pct_saving
                    })

                    # Cumulative cost per period for this strategy
                    # Each period: fwd_share * load * wavg_fwd + spot_share * load * spot_price + fwd_share * load * tx
                    period_cost = (merged['load_mwh'] * fwd_share * (wavg_fwd + config['tx_cost'])
                                   + merged['load_mwh'] * spot_share * merged['spot_price'])
                    cumulative_data[name] = period_cost.cumsum().values

            results_df = pd.DataFrame(results).sort_values('Gesamtkosten [€]').reset_index(drop=True)
            results_df.index += 1

            # Store
            st.session_state.bt_results = results_df
            st.session_state.bt_merged = merged
            st.session_state.bt_spot_total = total_spot_cost
            st.session_state.bt_demand_total = total_demand
            st.session_state.bt_avg_spot = avg_spot
            st.session_state.bt_cumulative = cumulative_data
            st.session_state.bt_fwd_before = fwd_before

        st.success(f"✅ Backtesting abgeschlossen! **{len(results)}** Szenarien berechnet.")
        st.rerun()

    # ── Display Results ──
    if st.session_state.bt_results is not None:
        results_df = st.session_state.bt_results
        merged = st.session_state.bt_merged
        total_spot_cost = st.session_state.bt_spot_total
        total_demand = st.session_state.bt_demand_total
        avg_spot = st.session_state.bt_avg_spot

        # ── Top KPIs ──
        best = results_df.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏆 Beste Strategie", f"{best['Terminanteil [%]']:.0f}%/{best['Spotanteil [%]']:.0f}%",
                     f"{best['PnL vs. Spot [€]']:+,.0f} € vs Spot")
        col2.metric("Gesamtkosten (beste)", f"{best['Gesamtkosten [€]']:,.0f} €",
                     f"{best['Ersparnis [%]']:+.1f}%")
        col3.metric("100% Spot (Benchmark)", f"{total_spot_cost:,.0f} €",
                     f"Ø {avg_spot:.2f} €/MWh")
        col4.metric("Gesamtbedarf", f"{total_demand:,.0f} MWh",
                     f"{len(merged)} Perioden")

        st.divider()

        # ── Top 5 cheapest ──
        st.subheader("🏆 Top 5 günstigste Strategien")
        top5 = results_df.head(5).copy()
        display_cols = ['Strategie', 'Gesamtkosten [€]', 'Ø Beschaffungspreis [€/MWh]',
                        'PnL vs. Spot [€]', 'Ersparnis [%]', 'Ø Forward-Preis [€/MWh]']
        fmt = top5[display_cols].copy()
        fmt['Gesamtkosten [€]'] = fmt['Gesamtkosten [€]'].apply(lambda x: f"{x:,.2f}")
        fmt['Ø Beschaffungspreis [€/MWh]'] = fmt['Ø Beschaffungspreis [€/MWh]'].apply(lambda x: f"{x:.2f}")
        fmt['PnL vs. Spot [€]'] = fmt['PnL vs. Spot [€]'].apply(lambda x: f"{x:+,.2f}")
        fmt['Ersparnis [%]'] = fmt['Ersparnis [%]'].apply(lambda x: f"{x:+.2f}%")
        fmt['Ø Forward-Preis [€/MWh]'] = fmt['Ø Forward-Preis [€/MWh]'].apply(lambda x: f"{x:.2f}")
        st.dataframe(fmt, use_container_width=True)

        st.divider()

        # ── Full results (expandable) ──
        with st.expander("📋 Alle Ergebnisse anzeigen"):
            full_fmt = results_df.copy()
            for col in ['Terminkosten [€]', 'Spotkosten [€]', 'TX-Kosten [€]', 'Gesamtkosten [€]', 'PnL vs. Spot [€]']:
                full_fmt[col] = full_fmt[col].apply(lambda x: f"{x:,.2f}")
            for col in ['Terminvolumen [MWh]', 'Spotvolumen [MWh]']:
                full_fmt[col] = full_fmt[col].apply(lambda x: f"{x:,.1f}")
            for col in ['Ø Forward-Preis [€/MWh]', 'Ø Spot-Preis [€/MWh]', 'Ø Beschaffungspreis [€/MWh]']:
                full_fmt[col] = full_fmt[col].apply(lambda x: f"{x:.2f}")
            full_fmt['Ersparnis [%]'] = full_fmt['Ersparnis [%]'].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(full_fmt, use_container_width=True)

        st.divider()

        # ══════════ CHARTS ══════════
        st.subheader("📊 Visualisierungen")

        # ── 1. Cost comparison bar chart ──
        fig1 = go.Figure()
        colors = ['#2ecc71' if x > 0 else '#e74c3c' if x < 0 else '#95a5a6' for x in results_df['PnL vs. Spot [€]']]
        fig1.add_trace(go.Bar(
            x=results_df['Strategie'],
            y=results_df['Gesamtkosten [€]'],
            marker_color=colors,
            text=results_df['Gesamtkosten [€]'].apply(lambda x: f"{x:,.0f}€"),
            textposition='outside',
            textfont_size=9
        ))
        fig1.add_hline(y=total_spot_cost, line_dash="dash", line_color="red", line_width=2,
                       annotation_text=f"100% Spot: {total_spot_cost:,.0f}€",
                       annotation_position="top right")
        fig1.update_layout(
            title="Gesamtkosten je Strategie vs. 100% Spot-Benchmark",
            xaxis_title="Strategie", yaxis_title="Gesamtkosten [€]",
            xaxis_tickangle=-45, height=550, template="plotly_dark"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ── 2. Average price per strategy ──
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=results_df['Strategie'],
            y=results_df['Ø Beschaffungspreis [€/MWh]'],
            marker_color='steelblue',
            text=results_df['Ø Beschaffungspreis [€/MWh]'].apply(lambda x: f"{x:.2f}"),
            textposition='outside', textfont_size=9
        ))
        fig2.add_hline(y=avg_spot, line_dash="dash", line_color="red",
                       annotation_text=f"Ø Spot: {avg_spot:.2f} €/MWh")
        if st.session_state.bt_fwd_before is not None:
            avg_fwd = st.session_state.bt_fwd_before['forward_price'].mean()
            fig2.add_hline(y=avg_fwd, line_dash="dot", line_color="orange",
                           annotation_text=f"Ø Forward: {avg_fwd:.2f} €/MWh")
        fig2.update_layout(
            title="Durchschnittlicher Beschaffungspreis je Strategie",
            xaxis_title="Strategie", yaxis_title="Ø Preis [€/MWh]",
            xaxis_tickangle=-45, height=500, template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── 3. Savings bar chart ──
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=results_df['Strategie'],
            y=results_df['PnL vs. Spot [€]'],
            marker_color=['#2ecc71' if x > 0 else '#e74c3c' for x in results_df['PnL vs. Spot [€]']],
            text=results_df['PnL vs. Spot [€]'].apply(lambda x: f"{x:+,.0f}€"),
            textposition='outside', textfont_size=9
        ))
        fig3.add_hline(y=0, line_color="white", line_width=1)
        fig3.update_layout(
            title="Ersparnis vs. 100% Spot (positiv = günstiger)",
            xaxis_title="Strategie", yaxis_title="Ersparnis [€]",
            xaxis_tickangle=-45, height=500, template="plotly_dark"
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── 4. Cumulative cost comparison ──
        if st.session_state.get('bt_cumulative'):
            st.subheader("📈 Kumulative Kosten über Lieferperiode")

            cum_data = st.session_state.bt_cumulative
            # Show a subset of strategies for readability
            show_strategies = st.multiselect(
                "Strategien auswählen",
                list(cum_data.keys()),
                default=list(cum_data.keys())[:5] if len(cum_data) > 5 else list(cum_data.keys())
            )

            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=merged['datetime'], y=merged['cum_spot_cost'],
                mode='lines', name='100% Spot (Benchmark)',
                line=dict(color='red', width=3, dash='dash')
            ))
            color_palette = ['#2ecc71', '#3498db', '#e67e22', '#9b59b6', '#1abc9c',
                             '#f39c12', '#e74c3c', '#2980b9', '#27ae60', '#8e44ad', '#d35400']
            for i, strat in enumerate(show_strategies):
                if strat in cum_data:
                    fig4.add_trace(go.Scatter(
                        x=merged['datetime'], y=cum_data[strat],
                        mode='lines', name=strat,
                        line=dict(color=color_palette[i % len(color_palette)], width=2)
                    ))
            fig4.update_layout(
                title="Kumulative Beschaffungskosten über die Lieferperiode",
                xaxis_title="Datum", yaxis_title="Kumulative Kosten [€]",
                height=550, template="plotly_dark",
                legend=dict(font=dict(size=9))
            )
            st.plotly_chart(fig4, use_container_width=True)

        # ── 5. Forward price development ──
        if st.session_state.get('bt_fwd_before') is not None:
            st.subheader("📈 Terminmarktpreis-Entwicklung vor Lieferbeginn")
            fwd_b = st.session_state.bt_fwd_before

            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=fwd_b['datetime'], y=fwd_b['forward_price'],
                mode='lines', name='Forward-Preis',
                line=dict(color='#3498db', width=2),
                fill='tozeroy', fillcolor='rgba(52,152,219,0.1)'
            ))

            # Mark procurement dates for best strategy
            n_fwd = len(fwd_b)
            actual_t = min(config['n_tranches'], n_fwd)
            idxs = np.round(np.linspace(0, n_fwd - 1, actual_t)).astype(int)
            fig5.add_trace(go.Scatter(
                x=fwd_b.iloc[idxs]['datetime'],
                y=fwd_b.iloc[idxs]['forward_price'],
                mode='markers', name='Kaufzeitpunkte (Gleichmäßig)',
                marker=dict(color='#e74c3c', size=10, symbol='triangle-up')
            ))

            fig5.add_hline(y=fwd_b['forward_price'].mean(), line_dash="dot", line_color="orange",
                           annotation_text=f"Ø Forward: {fwd_b['forward_price'].mean():.2f}")
            fig5.update_layout(
                title="Forward-Preise vor Lieferbeginn + Beschaffungszeitpunkte",
                xaxis_title="Datum", yaxis_title="Forward-Preis [€/MWh]",
                height=450, template="plotly_dark"
            )
            st.plotly_chart(fig5, use_container_width=True)

        # ── 6. Spot price + Load during delivery ──
        st.subheader("⚡ Spotpreis & Last während Lieferperiode")
        fig6 = make_subplots(specs=[[{"secondary_y": True}]])
        fig6.add_trace(
            go.Scatter(x=merged['datetime'], y=merged['spot_price'],
                       name='Spotpreis [€/MWh]', line=dict(color='#e67e22', width=1.5)),
            secondary_y=False
        )
        fig6.add_trace(
            go.Bar(x=merged['datetime'], y=merged['load_mwh'],
                   name='Last [MWh]', marker_color='rgba(52,152,219,0.3)'),
            secondary_y=True
        )
        fig6.update_layout(
            title="Spotpreise und Lastprofil während der Lieferperiode",
            height=450, template="plotly_dark"
        )
        fig6.update_yaxes(title_text="Spotpreis [€/MWh]", secondary_y=False)
        fig6.update_yaxes(title_text="Last [MWh]", secondary_y=True)
        st.plotly_chart(fig6, use_container_width=True)

        # ── 7. Sensitivity: cost vs forward share ──
        st.subheader("🎯 Sensitivität: Kosten vs. Terminanteil")

        for pat in config['patterns']:
            pat_name = pat.split('(')[0].strip()
            pat_data = results_df[results_df['Muster'] == pat_name].sort_values('Terminanteil [%]')
            if len(pat_data) > 1:
                fig7 = go.Figure()
                fig7.add_trace(go.Scatter(
                    x=pat_data['Terminanteil [%]'], y=pat_data['Gesamtkosten [€]'],
                    mode='lines+markers', name='Gesamtkosten',
                    line=dict(color='#2ecc71', width=2),
                    marker=dict(size=8)
                ))
                fig7.add_hline(y=total_spot_cost, line_dash="dash", line_color="red",
                               annotation_text="100% Spot")
                fig7.update_layout(
                    title=f"Gesamtkosten vs. Terminanteil – {pat_name}",
                    xaxis_title="Terminanteil [%]", yaxis_title="Gesamtkosten [€]",
                    height=400, template="plotly_dark"
                )
                st.plotly_chart(fig7, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 📊 DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.header("📊 Dashboard & Empfehlung")

    if st.session_state.bt_results is None:
        st.warning("⚠️ Bitte zunächst Backtesting durchführen.")
        st.stop()

    results_df = st.session_state.bt_results
    total_spot = st.session_state.bt_spot_total
    total_demand = st.session_state.bt_demand_total
    avg_spot = st.session_state.bt_avg_spot

    best = results_df.iloc[0]
    worst = results_df.iloc[-1]

    # ── KPIs ──
    st.subheader("🎯 Kennzahlen")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Optimale Strategie", f"{best['Terminanteil [%]']:.0f}%T / {best['Spotanteil [%]']:.0f}%S")
    c2.metric("Beste Gesamtkosten", f"{best['Gesamtkosten [€]']:,.0f} €",
              f"{best['PnL vs. Spot [€]']:+,.0f} € vs Spot")
    c3.metric("Schlechteste Strategie", f"{worst['Terminanteil [%]']:.0f}%T / {worst['Spotanteil [%]']:.0f}%S",
              f"{worst['PnL vs. Spot [€]']:+,.0f} €")
    c4.metric("Spanne (Best–Worst)", f"{worst['Gesamtkosten [€]'] - best['Gesamtkosten [€]']:,.0f} €")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gesamtbedarf", f"{total_demand:,.0f} MWh")
    c2.metric("100% Spot Kosten", f"{total_spot:,.0f} €")
    c3.metric("Ø Spot [€/MWh]", f"{avg_spot:.2f}")
    if st.session_state.bt_fwd_before is not None:
        avg_fwd = st.session_state.bt_fwd_before['forward_price'].mean()
        c4.metric("Ø Forward [€/MWh]", f"{avg_fwd:.2f}",
                  f"{avg_fwd - avg_spot:+.2f} vs Spot")

    st.divider()

    # ── Risk metrics ──
    st.subheader("📉 Risikokennzahlen")

    costs = results_df['Gesamtkosten [€]'].values
    prices = results_df['Ø Beschaffungspreis [€/MWh]'].values

    risk_data = {
        'Kennzahl': [
            'Min. Gesamtkosten',
            'Max. Gesamtkosten',
            'Ø Gesamtkosten',
            'Standardabweichung Kosten',
            'Spannweite (Max–Min)',
            'Min. Ø-Preis [€/MWh]',
            'Max. Ø-Preis [€/MWh]',
            'Volatilität Ø-Preis',
        ],
        'Wert': [
            f"{costs.min():,.2f} €",
            f"{costs.max():,.2f} €",
            f"{costs.mean():,.2f} €",
            f"{costs.std():,.2f} €",
            f"{costs.max() - costs.min():,.2f} €",
            f"{prices.min():.2f} €/MWh",
            f"{prices.max():.2f} €/MWh",
            f"{prices.std():.2f} €/MWh",
        ]
    }
    st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

    # ── VaR / CVaR on PnL ──
    pnl_values = results_df['PnL vs. Spot [€]'].values
    if len(pnl_values) > 2:
        var_95 = np.percentile(pnl_values, 5)  # 5th percentile of PnL = worst 5%
        cvar_95 = pnl_values[pnl_values <= var_95].mean() if (pnl_values <= var_95).any() else var_95
        max_loss = pnl_values.min()

        st.markdown(f"""
**Risikomaße (über alle Szenarien):**
- **Value at Risk (95%):** {var_95:+,.2f} € – Im schlechtesten 5% der Szenarien verlieren Sie mindestens diesen Betrag vs. Spot
- **Conditional VaR (95%):** {cvar_95:+,.2f} € – Erwarteter Verlust in den schlechtesten 5% der Szenarien
- **Maximaler Verlust vs. Spot:** {max_loss:+,.2f} €
        """)

    st.divider()

    # ── Recommendation ──
    st.subheader("💡 Empfehlung")

    if best['PnL vs. Spot [€]'] > 0:
        st.success(f"""
**Die optimale Strategie wäre gewesen: {best['Strategie']}**

- Gesamtkosten: **{best['Gesamtkosten [€]']:,.0f} €** (statt {total_spot:,.0f} € bei 100% Spot)
- Ersparnis: **{best['PnL vs. Spot [€]']:+,.0f} €** ({best['Ersparnis [%]']:+.1f}%)
- Ø Beschaffungspreis: **{best['Ø Beschaffungspreis [€/MWh]']:.2f} €/MWh** (statt {avg_spot:.2f} €/MWh)

→ Eine Vorab-Beschaffung von **{best['Terminanteil [%]']:.0f}%** am Terminmarkt hätte sich gelohnt.
        """)
    else:
        st.info(f"""
**In diesem Zeitraum wäre 100% Spotmarkt die günstigste Option gewesen.**

- 100% Spot: **{total_spot:,.0f} €** (Ø {avg_spot:.2f} €/MWh)
- Beste Alternative: {best['Strategie']} mit **{best['Gesamtkosten [€]']:,.0f} €**
- Differenz: **{abs(best['PnL vs. Spot [€]']):,.0f} €** Mehrkosten

→ Die Forward-Preise lagen über den späteren Spotpreisen – Terminbeschaffung hätte sich nicht gelohnt.
        """)

    # ── Sensitivity heatmap-style chart ──
    if len(config['patterns']) > 1:
        st.subheader("🗺️ Strategievergleich nach Muster")
        for pat in config['patterns']:
            pat_name = pat.split('(')[0].strip()
            sub = results_df[results_df['Muster'] == pat_name].sort_values('Terminanteil [%]')
            if len(sub) > 0:
                st.markdown(f"**{pat_name}:**")
                mini = sub[['Terminanteil [%]', 'Gesamtkosten [€]', 'Ø Beschaffungspreis [€/MWh]', 'PnL vs. Spot [€]', 'Ersparnis [%]']].copy()
                mini['Gesamtkosten [€]'] = mini['Gesamtkosten [€]'].apply(lambda x: f"{x:,.0f}")
                mini['Ø Beschaffungspreis [€/MWh]'] = mini['Ø Beschaffungspreis [€/MWh]'].apply(lambda x: f"{x:.2f}")
                mini['PnL vs. Spot [€]'] = mini['PnL vs. Spot [€]'].apply(lambda x: f"{x:+,.0f}")
                mini['Ersparnis [%]'] = mini['Ersparnis [%]'].apply(lambda x: f"{x:+.2f}%")
                st.dataframe(mini, use_container_width=True, hide_index=True)

    # ── Download results ──
    st.divider()
    st.subheader("💾 Ergebnisse exportieren")
    csv = results_df.to_csv(index=False, sep=';', decimal=',')
    st.download_button(
        "📥 Ergebnisse als CSV herunterladen",
        csv.encode('utf-8'),
        "backtesting_ergebnisse.csv",
        "text/csv"
    )
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

req = """streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
openpyxl>=3.1.0
"""
with open('requirements.txt', 'w') as f:
    f.write(req)

print(f"app.py: {len(app_code)} Zeichen")
print(f"requirements.txt geschrieben")app_code = r'''import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(
    page_title="Energiebeschaffung Simulation & Backtesting",
    layout="wide",
    page_icon="⚡"
)

# ── Styling ──
st.markdown("""
<style>
    .stMetric { background: #1e1e2e; padding: 15px; border-radius: 10px; border-left: 4px solid #4CAF50; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    .big-number { font-size: 2rem; font-weight: bold; color: #4CAF50; }
    .info-box { background: #262640; padding: 15px; border-radius: 8px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──
defaults = {
    'load_df': None, 'spot_df': None, 'forward_df': None,
    'bt_results': None, 'bt_merged': None, 'bt_spot_total': None,
    'bt_demand_total': None, 'bt_avg_spot': None,
    'strategy_config': {
        'forward_shares': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'patterns': ['Gleichmäßig'],
        'n_tranches': 6,
        'tx_cost': 0.0
    }
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helper Functions ──
def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith('.csv'):
            raw = uploaded_file.read()
            uploaded_file.seek(0)
            text = raw.decode('utf-8', errors='replace')
            for sep in [';', ',', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep)
                    if len(df.columns) >= 2:
                        return df
                except:
                    continue
            return pd.read_csv(io.StringIO(text))
        elif name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")
    return None


def parse_pasted_text(text):
    """Parst eingefügten Text (Copy-Paste) mit Auto-Erkennung des Trennzeichens."""
    if not text or not text.strip():
        return None
    text = text.strip()
    for sep in ['\t', ';', ',', '|']:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine='python')
            if len(df.colum

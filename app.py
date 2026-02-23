import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from typing import Optional, Dict, List, Tuple

# ══════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════

st.set_page_config(page_title="⚡ Energiebeschaffung", layout="wide", page_icon="⚡")

C_POS, C_NEG, C_NEUTRAL = "#2ecc71", "#e74c3c", "#95a5a6"
C_BLUE, C_ORANGE = "#3498db", "#e67e22"
PALETTE = [
    "#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#1abc9c",
    "#f39c12", "#e74c3c", "#2980b9", "#27ae60", "#8e44ad",
]
TPL = "plotly_dark"

st.markdown("""<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px; border-radius: 8px 8px 0 0; font-weight: 600;
    }
    textarea {
        font-family: 'Courier New', monospace !important;
        font-size: 13px !important;
    }
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════

_DEFAULTS = dict(
    load_df=None, spot_df=None, forward_df=None,
    bt_results=None, bt_merged=None, bt_spot_total=None,
    bt_demand_total=None, bt_avg_spot=None,
    bt_cumulative=None, bt_fwd_before=None,
    config=dict(
        forward_shares=[i / 100 for i in range(0, 101, 10)],
        patterns=["Gleichmäßig"],
        n_tranches=6,
        tx_cost=0.0,
    ),
)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v.copy() if isinstance(_v, (dict, list)) else _v


# ══════════════════════════════════════════════════
# DATEN-FUNKTIONEN
# ══════════════════════════════════════════════════


def parse_text(text: str) -> Optional[pd.DataFrame]:
    """Parst eingefügten Text (Tab / Semikolon / Komma) zu DataFrame."""
    if not text or not text.strip():
        return None
    for sep in ["\t", ";", ",", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text.strip()), sep=sep, engine="python")
            if len(df.columns) >= 2 and len(df) > 0:
                return df
        except Exception:
            continue
    try:
        df = pd.read_csv(io.StringIO(text.strip()), sep=r"\s+", engine="python")
        if len(df.columns) >= 2:
            return df
    except Exception:
        pass
    return None


def load_file(f) -> Optional[pd.DataFrame]:
    """Lädt CSV oder Excel – nutzt parse_text für CSV."""
    if f is None:
        return None
    try:
        if f.name.lower().endswith(".csv"):
            raw = f.read().decode("utf-8", errors="replace")
            f.seek(0)
            return parse_text(raw)
        return pd.read_excel(f)
    except Exception as e:
        st.error(f"Fehler: {e}")
        return None


def find_datetime_col(df: pd.DataFrame) -> Optional[str]:
    """Findet die wahrscheinlichste Datumsspalte."""
    prio = sorted(
        df.columns,
        key=lambda c: any(k in c.lower() for k in ("date", "datum", "zeit", "time")),
        reverse=True,
    )
    for col in prio:
        try:
            parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            if parsed.notna().mean() > 0.5:
                return col
        except Exception:
            continue
    return None


def find_value_col(df: pd.DataFrame, exclude: Optional[str] = None) -> Optional[str]:
    """Findet die erste numerische Spalte."""
    for col in df.columns:
        if col == exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
        try:
            conv = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."), errors="coerce"
            )
            if conv.notna().mean() > 0.5:
                return col
        except Exception:
            continue
    return None


def clean_data(
    df: pd.DataFrame, tcol: str, vcol: str, vname: str
) -> pd.DataFrame:
    """Bereinigt Rohdaten: parst Datum, Werte, entfernt NaN/Duplikate."""
    out = df[[tcol, vcol]].copy()
    out[tcol] = pd.to_datetime(out[tcol], dayfirst=True, errors="coerce")
    if out[vcol].dtype == object:
        out[vcol] = out[vcol].astype(str).str.replace(",", ".").str.strip()
    out[vcol] = pd.to_numeric(out[vcol], errors="coerce")
    out = out.dropna().rename(columns={tcol: "datetime", vcol: vname})
    n = len(out)
    out = out.groupby("datetime", as_index=False).agg({vname: "mean"})
    if len(out) < n:
        st.caption(f"ℹ️ {n - len(out)} Duplikate zusammengefasst (Mittelwert)")
    return out.sort_values("datetime").reset_index(drop=True)


# ══════════════════════════════════════════════════
# DEMO-DATEN
# ══════════════════════════════════════════════════


def generate_demo():
    """Erzeugt realistische Beispieldaten für Last, Spot und Forward."""
    rng = np.random.default_rng(42)

    dates_25 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    n = len(dates_25)
    season = 20 * np.sin(np.linspace(0, 2 * np.pi, n))

    load = 100 + season + rng.normal(0, 8, n)
    load_df = pd.DataFrame(
        {"datetime": dates_25, "load_mwh": np.maximum(load, 20)}
    )

    spot = 75 + season * 0.75 + rng.normal(0, 12, n)
    spot_df = pd.DataFrame(
        {"datetime": dates_25, "spot_price": np.maximum(spot, 5)}
    )

    dates_24 = pd.bdate_range("2024-01-02", "2024-12-30")[:250]
    fwd = 82 + np.cumsum(rng.normal(-0.02, 0.8, len(dates_24)))
    fwd_df = pd.DataFrame(
        {"datetime": dates_24, "forward_price": np.maximum(fwd, 40)}
    )

    return load_df, spot_df, fwd_df


# ══════════════════════════════════════════════════
# BACKTESTING ENGINE
# ══════════════════════════════════════════════════


def compute_weights(pattern: str, n: int) -> np.ndarray:
    """Gewichte je nach Beschaffungsmuster."""
    if "Frontloaded" in pattern:
        w = np.linspace(2.0, 0.5, n)
    elif "Backloaded" in pattern:
        w = np.linspace(0.5, 2.0, n)
    else:
        w = np.ones(n)
    return w / w.sum()


def merge_load_spot(
    load_df: pd.DataFrame, spot_df: pd.DataFrame
) -> Tuple[pd.DataFrame, str]:
    """Merged Last + Spot (exakt, sonst täglicher Fallback)."""
    m = pd.merge(load_df, spot_df, on="datetime", how="inner")
    if len(m) > 0:
        return m, "exakt"
    ld = (
        load_df.assign(d=load_df.datetime.dt.date)
        .groupby("d")
        .agg(load_mwh=("load_mwh", "sum"))
        .reset_index()
    )
    sd = (
        spot_df.assign(d=spot_df.datetime.dt.date)
        .groupby("d")
        .agg(spot_price=("spot_price", "mean"))
        .reset_index()
    )
    m = pd.merge(ld, sd, on="d", how="inner")
    if len(m) > 0:
        m["datetime"] = pd.to_datetime(m["d"])
        return m.drop(columns="d"), "täglich aggregiert"
    return pd.DataFrame(), "kein Überlapp"


def run_backtest(
    load_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    fwd_df: pd.DataFrame,
    cfg: Dict,
    progress=None,
) -> Dict:
    """Komplette Backtest-Berechnung – UI-frei."""
    delivery_start = load_df.datetime.min()

    fwd_b = fwd_df[fwd_df.datetime < delivery_start].copy()
    if fwd_b.empty:
        raise ValueError(
            "Keine Forward-Daten VOR Lieferbeginn vorhanden! "
            "Terminpreise müssen zeitlich VOR dem Lastprofil liegen."
        )

    merged, merge_mode = merge_load_spot(load_df, spot_df)
    if merged.empty:
        raise ValueError(
            "Kein zeitlicher Überlapp zwischen Last und Spot! "
            "Beide müssen denselben Lieferzeitraum abdecken."
        )

    merged["cost_spot"] = merged.load_mwh * merged.spot_price
    total_demand = float(merged.load_mwh.sum())
    total_spot = float(merged.cost_spot.sum())
    avg_spot = total_spot / total_demand if total_demand else 0.0

    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged["cum_spot"] = merged.cost_spot.cumsum()

    fwd_vals = fwd_b.forward_price.values
    n_fwd = len(fwd_b)
    n_total = len(cfg["patterns"]) * len(cfg["forward_shares"])

    results: List[Dict] = []
    cum_data: Dict[str, np.ndarray] = {}
    done = 0

    for pat in cfg["patterns"]:
        n_t = min(cfg["n_tranches"], n_fwd)
        idx = np.round(np.linspace(0, n_fwd - 1, n_t)).astype(int)
        w = compute_weights(pat, n_t)
        wavg = float(np.dot(w, fwd_vals[idx]))
        pname = pat.split("(")[0].strip()

        for fs in cfg["forward_shares"]:
            ss = 1.0 - fs
            fwd_vol = total_demand * fs
            fwd_cost = fwd_vol * wavg
            tx = fwd_vol * cfg["tx_cost"]
            spot_cost = float((merged.load_mwh * ss * merged.spot_price).sum())
            total = fwd_cost + spot_cost + tx
            avg_p = total / total_demand if total_demand else 0.0
            pnl = total_spot - total
            pct = (pnl / total_spot * 100) if total_spot else 0.0

            name = f"{pname} ({fs * 100:.0f}%T / {ss * 100:.0f}%S)"

            results.append(
                {
                    "Strategie": name,
                    "Muster": pname,
                    "Terminanteil [%]": fs * 100,
                    "Spotanteil [%]": ss * 100,
                    "Ø Forward [€/MWh]": wavg,
                    "Terminvol. [MWh]": fwd_vol,
                    "Terminkosten [€]": fwd_cost,
                    "Spotvol. [MWh]": total_demand * ss,
                    "Ø Spot [€/MWh]": avg_spot,
                    "Spotkosten [€]": spot_cost,
                    "TX [€]": tx,
                    "Gesamt [€]": total,
                    "Ø Preis [€/MWh]": avg_p,
                    "PnL vs Spot [€]": pnl,
                    "Ersparnis [%]": pct,
                }
            )

            period_cost = merged.load_mwh * fs * (wavg + cfg["tx_cost"]) + (
                merged.load_mwh * ss * merged.spot_price
            )
            cum_data[name] = period_cost.cumsum().values

            done += 1
            if progress:
                progress.progress(done / n_total, f"Szenario {done}/{n_total}")

    rdf = (
        pd.DataFrame(results)
        .sort_values("Gesamt [€]")
        .reset_index(drop=True)
    )
    rdf.index += 1

    return dict(
        results=rdf,
        merged=merged,
        total_spot=total_spot,
        demand=total_demand,
        avg_spot=avg_spot,
        cum=cum_data,
        fwd_before=fwd_b,
        merge_mode=merge_mode,
    )


# ══════════════════════════════════════════════════
# CHART-FUNKTIONEN
# ══════════════════════════════════════════════════


def _pnl_colors(vals):
    return [C_POS if v > 0 else C_NEG if v < 0 else C_NEUTRAL for v in vals]


def chart_costs(df, ref):
    fig = go.Figure(
        go.Bar(
            x=df["Strategie"],
            y=df["Gesamt [€]"],
            marker_color=_pnl_colors(df["PnL vs Spot [€]"]),
            text=df["Gesamt [€]"].apply(lambda v: f"{v:,.0f}€"),
            textposition="outside",
            textfont_size=9,
        )
    )
    fig.add_hline(
        y=ref,
        line_dash="dash",
        line_color="red",
        annotation_text=f"100% Spot: {ref:,.0f}€",
    )
    fig.update_layout(
        title="Gesamtkosten je Strategie",
        xaxis_tickangle=-45,
        height=500,
        template=TPL,
        yaxis_title="€",
    )
    return fig


def chart_savings(df):
    fig = go.Figure(
        go.Bar(
            x=df["Strategie"],
            y=df["PnL vs Spot [€]"],
            marker_color=_pnl_colors(df["PnL vs Spot [€]"]),
            text=df["PnL vs Spot [€]"].apply(lambda v: f"{v:+,.0f}€"),
            textposition="outside",
            textfont_size=9,
        )
    )
    fig.add_hline(y=0, line_color="white", line_width=1)
    fig.update_layout(
        title="Ersparnis vs. 100% Spot",
        xaxis_tickangle=-45,
        height=450,
        template=TPL,
        yaxis_title="€",
    )
    return fig


def chart_cumulative(merged, cum, keys):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged.datetime,
            y=merged.cum_spot,
            mode="lines",
            name="100% Spot",
            line=dict(color="red", width=3, dash="dash"),
        )
    )
    for i, k in enumerate(keys):
        if k in cum:
            fig.add_trace(
                go.Scatter(
                    x=merged.datetime,
                    y=cum[k],
                    mode="lines",
                    name=k,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                )
            )
    fig.update_layout(
        title="Kumulative Kosten",
        height=500,
        template=TPL,
        xaxis_title="Datum",
        yaxis_title="€",
        legend=dict(font_size=9),
    )
    return fig


def chart_forward(fwd, n_t):
    n = len(fwd)
    idx = np.round(np.linspace(0, n - 1, min(n_t, n))).astype(int)
    avg = fwd.forward_price.mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fwd.datetime,
            y=fwd.forward_price,
            mode="lines",
            name="Forward",
            line=dict(color=C_BLUE, width=2),
            fill="tozeroy",
            fillcolor="rgba(52,152,219,0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fwd.iloc[idx].datetime,
            y=fwd.iloc[idx].forward_price,
            mode="markers",
            name="Kaufzeitpunkte",
            marker=dict(color=C_NEG, size=10, symbol="triangle-up"),
        )
    )
    fig.add_hline(
        y=avg,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Ø {avg:.2f}",
    )
    fig.update_layout(
        title="Forward-Preisverlauf + Kaufzeitpunkte",
        height=400,
        template=TPL,
        yaxis_title="€/MWh",
    )
    return fig


def chart_spot_load(merged):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=merged.datetime,
            y=merged.spot_price,
            name="Spot [€/MWh]",
            line=dict(color=C_ORANGE, width=1.5),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=merged.datetime,
            y=merged.load_mwh,
            name="Last [MWh]",
            marker_color="rgba(52,152,219,0.25)",
        ),
        secondary_y=True,
    )
    fig.update_layout(title="Spot & Last", height=400, template=TPL)
    fig.update_yaxes(title_text="€/MWh", secondary_y=False)
    fig.update_yaxes(title_text="MWh", secondary_y=True)
    return fig


def chart_sensitivity(df, pattern, ref):
    sub = df[df["Muster"] == pattern].sort_values("Terminanteil [%]")
    if len(sub) < 2:
        return None
    best = sub.loc[sub["Gesamt [€]"].idxmin()]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["Terminanteil [%]"],
            y=sub["Gesamt [€]"],
            mode="lines+markers",
            name="Kosten",
            line=dict(color=C_POS, width=2),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[best["Terminanteil [%]"]],
            y=[best["Gesamt [€]"]],
            mode="markers",
            name=f"Optimum ({best['Terminanteil [%]']:.0f}%)",
            marker=dict(color="gold", size=14, symbol="star"),
        )
    )
    fig.add_hline(
        y=ref, line_dash="dash", line_color="red", annotation_text="100% Spot"
    )
    fig.update_layout(
        title=f"Sensitivität – {pattern}",
        height=380,
        template=TPL,
        xaxis_title="Terminanteil [%]",
        yaxis_title="€",
    )
    return fig


# ══════════════════════════════════════════════════
# UI-KOMPONENTE: DATEN-EINGABE  (Copy-Paste + Upload)
# ══════════════════════════════════════════════════


def data_input(
    title: str,
    key: str,
    state_key: str,
    val_label: str,
    val_name: str,
    placeholder: str,
    unit: str = "",
):
    """Wiederverwendbare Eingabe mit **Text-Einfügen** und Datei-Upload."""

    with st.container(border=True):
        st.markdown(f"**{title}**")

        # ── Daten bereits geladen → Zusammenfassung ──
        if st.session_state[state_key] is not None:
            df = st.session_state[state_key]
            c1, c2 = st.columns([5, 1])
            c1.success(
                f"✅ **{len(df)} Einträge** · "
                f"{df.datetime.min().date()} → {df.datetime.max().date()} · "
                f"Ø {df[val_name].mean():.1f} {unit}"
            )
            if c2.button("🗑️", key=f"{key}_del", help="Entfernen"):
                st.session_state[state_key] = None
                st.session_state.pop(f"{key}_raw", None)
                st.rerun()
            return

        # ── Eingabe: Paste (primär) | Upload (sekundär) ──
        tab_paste, tab_upload = st.tabs(
            ["📋 Einfügen (Copy & Paste)", "📁 Datei hochladen"]
        )

        with tab_paste:
            pasted = st.text_area(
                "Daten hier einfügen",
                height=180,
                key=f"{key}_txt",
                placeholder=placeholder,
                help=(
                    "Markieren Sie Ihre Daten in Excel / Google Sheets / CSV "
                    "und fügen Sie sie hier mit Strg+V ein."
                ),
            )
            if st.button(
                "🔄 Eingefügte Daten verarbeiten",
                key=f"{key}_go",
                type="primary",
                use_container_width=True,
                disabled=not pasted,
            ):
                raw = parse_text(pasted)
                if raw is not None and len(raw.columns) >= 2:
                    st.session_state[f"{key}_raw"] = raw
                    st.rerun()
                else:
                    st.error(
                        "❌ Format nicht erkannt – mind. 2 Spalten nötig "
                        "(Datum + Wert, getrennt durch Tab / Semikolon / Komma)."
                    )

        with tab_upload:
            f = st.file_uploader(
                "Datei wählen",
                ["csv", "xlsx", "xls"],
                key=f"{key}_f",
                label_visibility="collapsed",
            )
            if f and st.button(
                "📁 Datei laden",
                key=f"{key}_fgo",
                use_container_width=True,
            ):
                raw = load_file(f)
                if raw is not None:
                    st.session_state[f"{key}_raw"] = raw
                    st.rerun()

        # ── Spaltenauswahl (wenn Rohdaten vorhanden) ──
        raw = st.session_state.get(f"{key}_raw")
        if raw is not None and len(raw) > 0:
            st.divider()
            st.caption(
                f"Vorschau ({len(raw)} Zeilen × {len(raw.columns)} Spalten)"
            )
            st.dataframe(raw.head(8), use_container_width=True, height=200)

            cols = list(raw.columns)
            auto_t = find_datetime_col(raw)
            auto_v = find_value_col(raw, auto_t)

            c1, c2, c3 = st.columns([2, 2, 1])
            tc = c1.selectbox(
                "📅 Zeitspalte",
                cols,
                index=cols.index(auto_t) if auto_t in cols else 0,
                key=f"{key}_tc",
            )
            vc = c2.selectbox(
                f"📊 {val_label}",
                cols,
                index=(
                    cols.index(auto_v)
                    if auto_v in cols
                    else min(1, len(cols) - 1)
                ),
                key=f"{key}_vc",
            )
            with c3:
                st.markdown("")  # Spacer
                if st.button(
                    "✅ Übernehmen",
                    key=f"{key}_ok",
                    type="primary",
                    use_container_width=True,
                ):
                    df = clean_data(raw, tc, vc, val_name)
                    if df.empty:
                        st.error("Keine gültigen Daten nach Bereinigung.")
                    else:
                        st.session_state[state_key] = df
                        st.session_state.pop(f"{key}_raw", None)
                        st.rerun()


# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════

st.sidebar.title("⚡ Energiebeschaffung")
st.sidebar.caption("Simulation & Backtesting")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["📥 Datenimport", "⚙️ Strategien", "🔬 Backtesting", "📊 Dashboard"],
)

st.sidebar.divider()
st.sidebar.markdown("##### Status")
for sk, label in [
    ("load_df", "Lastprofil"),
    ("spot_df", "Spotpreise"),
    ("forward_df", "Terminpreise"),
]:
    d = st.session_state[sk]
    if d is not None:
        st.sidebar.success(f"✅ {label} ({len(d)})")
    else:
        st.sidebar.warning(f"⏳ {label}")

st.sidebar.divider()
sb1, sb2 = st.sidebar.columns(2)
if sb1.button("🗑️ Reset", use_container_width=True):
    for k, v in _DEFAULTS.items():
        st.session_state[k] = (
            v.copy() if isinstance(v, (dict, list)) else v
        )
    st.rerun()
if sb2.button("🎲 Demo", use_container_width=True, help="Beispieldaten laden"):
    ld, sd, fd = generate_demo()
    st.session_state.load_df = ld
    st.session_state.spot_df = sd
    st.session_state.forward_df = fd
    st.rerun()


# ══════════════════════════════════════════════════
# SEITE: DATENIMPORT
# ══════════════════════════════════════════════════

if page == "📥 Datenimport":
    st.header("📥 Datenimport")
    st.caption(
        "Kopieren Sie Daten direkt aus Excel (Strg+C → Strg+V) "
        "oder laden Sie eine Datei hoch."
    )

    data_input(
        title="1️⃣  Lastprofil  (Lieferperiode, z. B. 2025)",
        key="load",
        state_key="load_df",
        val_label="Verbrauch [MWh]",
        val_name="load_mwh",
        placeholder=(
            "Datum\tVerbrauch_MWh\n"
            "01.01.2025\t120.5\n"
            "02.01.2025\t115.3\n"
            "03.01.2025\t118.7\n"
            "…"
        ),
        unit="MWh",
    )

    data_input(
        title="2️⃣  Spotpreise  (gleicher Zeitraum wie Lastprofil)",
        key="spot",
        state_key="spot_df",
        val_label="Preis [€/MWh]",
        val_name="spot_price",
        placeholder=(
            "Datum\tPreis_EUR\n"
            "01.01.2025\t85.20\n"
            "02.01.2025\t92.10\n"
            "03.01.2025\t78.40\n"
            "…"
        ),
        unit="€/MWh",
    )

    data_input(
        title="3️⃣  Terminmarktpreise  (VOR Lieferperiode, z. B. 2024)",
        key="fwd",
        state_key="forward_df",
        val_label="Forward [€/MWh]",
        val_name="forward_price",
        placeholder=(
            "Datum\tForward_EUR\n"
            "02.01.2024\t78.50\n"
            "03.01.2024\t79.20\n"
            "04.01.2024\t77.80\n"
            "…"
        ),
        unit="€/MWh",
    )

    all_loaded = all(
        st.session_state[k] is not None
        for k in ["load_df", "spot_df", "forward_df"]
    )
    if all_loaded:
        st.success("✅ Alle drei Datensätze geladen — weiter zu **⚙️ Strategien**")


# ══════════════════════════════════════════════════
# SEITE: STRATEGIEN
# ══════════════════════════════════════════════════

elif page == "⚙️ Strategien":
    st.header("⚙️ Strategiekonfiguration")

    with st.container(border=True):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("##### Terminmarkt-Anteile")
            mode = st.radio(
                "Modus",
                ["Standard (0–100 %)", "Benutzerdefiniert"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if mode.startswith("Standard"):
                shares = [i / 100 for i in range(0, 101, 10)]
            else:
                txt = st.text_input(
                    "Anteile in %",
                    "0,10,20,30,40,50,60,70,80,90,100",
                )
                try:
                    shares = sorted(
                        {max(0.0, min(1.0, float(x) / 100)) for x in txt.split(",")}
                    )
                except ValueError:
                    shares = [i / 100 for i in range(0, 101, 10)]
            st.caption(
                f"{len(shares)} Stufen: "
                + ", ".join(f"{s:.0%}" for s in shares)
            )

        with c2:
            st.markdown("##### Beschaffungsmuster")
            patterns = st.multiselect(
                "Muster",
                [
                    "Gleichmäßig",
                    "Frontloaded (früh mehr)",
                    "Backloaded (spät mehr)",
                ],
                default=["Gleichmäßig"],
                label_visibility="collapsed",
            )
            if not patterns:
                patterns = ["Gleichmäßig"]
            n_tranches = st.slider("Tranchen", 2, 36, 6)
            tx = st.number_input(
                "TX-Kosten [€/MWh]", 0.0, step=0.05, format="%.2f"
            )

    st.session_state.config = dict(
        forward_shares=shares,
        patterns=patterns,
        n_tranches=n_tranches,
        tx_cost=tx,
    )
    n_sc = len(shares) * len(patterns)
    st.info(
        f"📐 **{n_sc} Szenarien** "
        f"({len(shares)} Stufen × {len(patterns)} Muster)"
    )

    # Vorschau Kaufzeitpunkte
    if st.session_state.forward_df is not None:
        fwd = st.session_state.forward_df
        n_fwd = len(fwd)
        at = min(n_tranches, n_fwd)
        idx = np.round(np.linspace(0, n_fwd - 1, at)).astype(int)

        for pat in patterns:
            w = compute_weights(pat, at)
            p = fwd.iloc[idx].forward_price.values
            wavg = float(np.dot(w, p))
            with st.expander(f"🔍 {pat} — Ø {wavg:.2f} €/MWh"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "#": range(1, at + 1),
                            "Datum": fwd.iloc[idx].datetime.dt.date.values,
                            "Preis": p,
                            "Gewicht": [f"{x:.1%}" for x in w],
                            "Gew. Preis": p * w,
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


# ══════════════════════════════════════════════════
# SEITE: BACKTESTING
# ══════════════════════════════════════════════════

elif page == "🔬 Backtesting":
    st.header("🔬 Backtesting")

    missing = [
        l
        for k, l in [
            ("load_df", "Lastprofil"),
            ("spot_df", "Spotpreise"),
            ("forward_df", "Terminpreise"),
        ]
        if st.session_state[k] is None
    ]
    if missing:
        st.warning(f"⚠️ Fehlend: **{', '.join(missing)}** → 📥 Datenimport")
        st.stop()

    cfg = st.session_state.config
    ld = st.session_state.load_df
    sd = st.session_state.spot_df
    fd = st.session_state.forward_df

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "📊 Last",
            f"{ld.load_mwh.sum():,.0f} MWh",
            f"{ld.datetime.min().date()} → {ld.datetime.max().date()}",
        )
        c2.metric(
            "💰 Spot",
            f"Ø {sd.spot_price.mean():.2f} €/MWh",
            f"{len(sd)} Einträge",
        )
        c3.metric(
            "📈 Forward",
            f"Ø {fd.forward_price.mean():.2f} €/MWh",
            f"{len(fd)} Handelstage",
        )

    if st.button(
        "🚀 Backtesting starten",
        type="primary",
        use_container_width=True,
    ):
        prog = st.progress(0, "Starte …")
        try:
            bt = run_backtest(ld, sd, fd, cfg, prog)
        except ValueError as e:
            prog.empty()
            st.error(f"❌ {e}")
            st.stop()
        prog.empty()

        st.session_state.update(
            bt_results=bt["results"],
            bt_merged=bt["merged"],
            bt_spot_total=bt["total_spot"],
            bt_demand_total=bt["demand"],
            bt_avg_spot=bt["avg_spot"],
            bt_cumulative=bt["cum"],
            bt_fwd_before=bt["fwd_before"],
        )
        st.success(
            f"✅ **{len(bt['results'])} Szenarien** berechnet "
            f"(Merge: {bt['merge_mode']})"
        )
        st.rerun()

    # ── ERGEBNISSE ──
    if st.session_state.bt_results is not None:
        R = st.session_state.bt_results
        M = st.session_state.bt_merged
        TS = st.session_state.bt_spot_total
        TD = st.session_state.bt_demand_total
        AS = st.session_state.bt_avg_spot
        CUM = st.session_state.bt_cumulative or {}
        FB = st.session_state.bt_fwd_before

        best = R.iloc[0]

        st.divider()

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "🏆 Beste",
            best["Strategie"].split("(")[1].rstrip(")"),
            f"{best['PnL vs Spot [€]']:+,.0f} €",
        )
        c2.metric(
            "Kosten (beste)",
            f"{best['Gesamt [€]']:,.0f} €",
            f"{best['Ersparnis [%]']:+.1f} %",
        )
        c3.metric(
            "100 % Spot",
            f"{TS:,.0f} €",
            f"Ø {AS:.2f} €/MWh",
        )
        c4.metric("Bedarf", f"{TD:,.0f} MWh", f"{len(M)} Perioden")

        # Top 5
        st.divider()
        st.subheader("🏆 Top 5")
        st.dataframe(
            R.head(5)[
                [
                    "Strategie",
                    "Gesamt [€]",
                    "Ø Preis [€/MWh]",
                    "PnL vs Spot [€]",
                    "Ersparnis [%]",
                ]
            ].style.format(
                {
                    "Gesamt [€]": "{:,.0f}",
                    "Ø Preis [€/MWh]": "{:.2f}",
                    "PnL vs Spot [€]": "{:+,.0f}",
                    "Ersparnis [%]": "{:+.1f} %",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("📋 Alle Ergebnisse"):
            st.dataframe(R, use_container_width=True)

        # Charts in Tabs
        st.divider()
        max_show = st.slider(
            "Max. Strategien in Diagrammen", 5, len(R), min(15, len(R))
        )
        show_df = R.head(max_show)

        t_c, t_s, t_k, t_f, t_sl, t_se = st.tabs(
            [
                "💰 Kosten",
                "📊 Ersparnis",
                "📈 Kumulativ",
                "🔵 Forward",
                "⚡ Spot+Last",
                "🎯 Sensitivität",
            ]
        )
        with t_c:
            st.plotly_chart(
                chart_costs(show_df, TS), use_container_width=True
            )
        with t_s:
            st.plotly_chart(chart_savings(show_df), use_container_width=True)
        with t_k:
            sel = st.multiselect(
                "Strategien wählen",
                list(CUM.keys()),
                default=list(CUM.keys())[:5],
            )
            if sel:
                st.plotly_chart(
                    chart_cumulative(M, CUM, sel), use_container_width=True
                )
        with t_f:
            if FB is not None:
                st.plotly_chart(
                    chart_forward(FB, cfg["n_tranches"]),
                    use_container_width=True,
                )
        with t_sl:
            st.plotly_chart(chart_spot_load(M), use_container_width=True)
        with t_se:
            for pat in cfg["patterns"]:
                pn = pat.split("(")[0].strip()
                fig = chart_sensitivity(R, pn, TS)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════
# SEITE: DASHBOARD
# ══════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.header("📊 Dashboard & Empfehlung")

    if st.session_state.bt_results is None:
        st.warning("⚠️ Erst Backtesting durchführen.")
        st.stop()

    R = st.session_state.bt_results
    TS = st.session_state.bt_spot_total
    TD = st.session_state.bt_demand_total
    AS = st.session_state.bt_avg_spot
    FB = st.session_state.bt_fwd_before
    cfg = st.session_state.config

    best, worst = R.iloc[0], R.iloc[-1]

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "🏆 Optimum",
            f"{best['Terminanteil [%]']:.0f} % T / "
            f"{best['Spotanteil [%]']:.0f} % S",
        )
        c2.metric(
            "Beste Kosten",
            f"{best['Gesamt [€]']:,.0f} €",
            f"{best['PnL vs Spot [€]']:+,.0f} € vs Spot",
        )
        c3.metric(
            "Schlechteste",
            f"{worst['Terminanteil [%]']:.0f} % T / "
            f"{worst['Spotanteil [%]']:.0f} % S",
            f"{worst['PnL vs Spot [€]']:+,.0f} €",
        )
        c4.metric(
            "Spanne",
            f"{worst['Gesamt [€]'] - best['Gesamt [€]']:,.0f} €",
        )

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bedarf", f"{TD:,.0f} MWh")
        c2.metric("100 % Spot", f"{TS:,.0f} €")
        c3.metric("Ø Spot", f"{AS:.2f} €/MWh")
        if FB is not None:
            af = FB.forward_price.mean()
            c4.metric(
                "Ø Forward",
                f"{af:.2f} €/MWh",
                f"{af - AS:+.2f} vs Spot",
            )

    st.divider()

    # Risiko
    st.subheader("📉 Risikokennzahlen")
    costs = R["Gesamt [€]"].values
    pnl = R["PnL vs Spot [€]"].values

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min. Kosten", f"{costs.min():,.0f} €")
        c2.metric("Max. Kosten", f"{costs.max():,.0f} €")
        c3.metric("Std.-Abw.", f"{costs.std():,.0f} €")
        if len(pnl) > 2:
            var95 = float(np.percentile(pnl, 5))
            c4.metric("VaR 95 %", f"{var95:+,.0f} €")

    st.divider()

    # Empfehlung
    st.subheader("💡 Empfehlung")
    if best["PnL vs Spot [€]"] > 0:
        st.success(
            f"**Optimale Strategie: {best['Strategie']}**\n\n"
            f"- Kosten: **{best['Gesamt [€]']:,.0f} €** "
            f"(statt {TS:,.0f} €)\n"
            f"- Ersparnis: **{best['PnL vs Spot [€]']:+,.0f} €** "
            f"({best['Ersparnis [%]']:+.1f} %)\n"
            f"- Ø Preis: **{best['Ø Preis [€/MWh]']:.2f} €/MWh** "
            f"(statt {AS:.2f})\n\n"
            f"→ **{best['Terminanteil [%]']:.0f} %** "
            f"Terminmarkt hätte sich gelohnt."
        )
    else:
        st.info(
            f"**100 % Spot wäre am günstigsten gewesen.**\n\n"
            f"- Spot: **{TS:,.0f} €** (Ø {AS:.2f} €/MWh)\n"
            f"- Nächstbeste: {best['Strategie']} → "
            f"{best['Gesamt [€]']:,.0f} €\n"
            f"- Mehrkosten: {abs(best['PnL vs Spot [€]']):,.0f} €"
        )

    # Mustervergleich
    if len(cfg["patterns"]) > 1:
        st.divider()
        st.subheader("🗺️ Mustervergleich")
        for pat in cfg["patterns"]:
            pn = pat.split("(")[0].strip()
            sub = R[R["Muster"] == pn].sort_values("Terminanteil [%]")
            if not sub.empty:
                with st.expander(f"**{pn}**"):
                    st.dataframe(
                        sub[
                            [
                                "Terminanteil [%]",
                                "Gesamt [€]",
                                "Ø Preis [€/MWh]",
                                "PnL vs Spot [€]",
                                "Ersparnis [%]",
                            ]
                        ].style.format(
                            {
                                "Gesamt [€]": "{:,.0f}",
                                "Ø Preis [€/MWh]": "{:.2f}",
                                "PnL vs Spot [€]": "{:+,.0f}",
                                "Ersparnis [%]": "{:+.1f} %",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    # Export
    st.divider()
    st.download_button(
        "📥 Ergebnisse als CSV",
        R.to_csv(index=False, sep=";", decimal=",").encode("utf-8"),
        "ergebnisse.csv",
        "text/csv",
        use_container_width=True,
    )

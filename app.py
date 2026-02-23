import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from typing import Optional, List, Dict, Tuple


# ═══════════════════════════════════════════════════════════════
# Konfiguration & Konstanten
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Energiebeschaffung Simulation & Backtesting",
    layout="wide",
    page_icon="⚡",
)

# FIX #12: Konstanten statt Magic Numbers / wiederholter Strings
COLOR_POSITIVE = "#2ecc71"
COLOR_NEGATIVE = "#e74c3c"
COLOR_NEUTRAL = "#95a5a6"
COLOR_FORWARD = "#3498db"
COLOR_SPOT = "#e67e22"
COLOR_PALETTE = [
    "#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#1abc9c",
    "#f39c12", "#e74c3c", "#2980b9", "#27ae60", "#8e44ad", "#d35400",
]
PLOTLY_TEMPLATE = "plotly_dark"
MIN_DATA_COVERAGE = 0.5  # mind. 50 % gültige Werte für Auto-Detect

st.markdown("""
<style>
    .stMetric {
        background: #1e1e2e; padding: 15px;
        border-radius: 10px; border-left: 4px solid #4CAF50;
    }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    .big-number { font-size: 2rem; font-weight: bold; color: #4CAF50; }
    .info-box {
        background: #262640; padding: 15px;
        border-radius: 8px; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Session State  –  FIX #1: ALLE Keys mit Defaults
# ═══════════════════════════════════════════════════════════════

DEFAULTS: Dict = {
    "load_df": None,
    "spot_df": None,
    "forward_df": None,
    "bt_results": None,
    "bt_merged": None,
    "bt_spot_total": None,
    "bt_demand_total": None,
    "bt_avg_spot": None,
    "bt_cumulative": None,    # ← fehlte vorher
    "bt_fwd_before": None,    # ← fehlte vorher
    "strategy_config": {
        "forward_shares": [i / 100 for i in range(0, 101, 10)],
        "patterns": ["Gleichmäßig"],
        "n_tranches": 6,
        "tx_cost": 0.0,
    },
}

for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        # deep-copy dicts, damit Defaults nicht mutiert werden
        st.session_state[_k] = _v.copy() if isinstance(_v, dict) else _v


# ═══════════════════════════════════════════════════════════════
# Hilfsfunktionen  –  FIX #2, #3, #4, #5
# ═══════════════════════════════════════════════════════════════

def load_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Lädt CSV / Excel mit Auto-Separator-Erkennung.

    FIX #3: Bare 'except' durch spezifische Exceptions ersetzt.
    """
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            raw_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            text = raw_bytes.decode("utf-8", errors="replace")

            for sep in [";", ",", "\t", "|"]:
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
                    if len(df.columns) >= 2:
                        return df
                except (pd.errors.ParserError, ValueError):
                    continue
            return pd.read_csv(io.StringIO(text))

        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)

        st.error(f"Nicht unterstütztes Format: {name}")
        return None

    except Exception as exc:
        st.error(f"Fehler beim Laden: {exc}")
        return None


def detect_datetime_col(df: pd.DataFrame) -> Optional[str]:
    """Auto-Detect der Datumsspalte (Name-Heuristik + Parse-Test)."""
    priority: List[str] = []
    rest: List[str] = []

    for col in df.columns:
        low = col.lower()
        if any(kw in low for kw in ("date", "datum", "zeit", "time", "timestamp")):
            priority.append(col)
        else:
            rest.append(col)

    for col in priority + rest:
        try:
            parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            if parsed.notna().mean() > MIN_DATA_COVERAGE:
                return col
        except (ValueError, TypeError, OverflowError):
            continue
    return None


def detect_numeric_col(
    df: pd.DataFrame, exclude: Optional[List[str]] = None
) -> Optional[str]:
    """Auto-Detect einer numerischen Spalte.

    FIX #2: Redundante doppelte Konvertierung entfernt.
    """
    exclude_set = set(exclude or [])

    for col in df.columns:
        if col in exclude_set:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
        # Versuch: Komma-Dezimaltrennzeichen → Punkt
        try:
            converted = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".").str.strip(),
                errors="coerce",
            )
            if converted.notna().mean() > MIN_DATA_COVERAGE:
                return col
        except (ValueError, TypeError):
            continue
    return None


def parse_and_clean(
    df: pd.DataFrame, time_col: str, val_col: str, val_name: str
) -> pd.DataFrame:
    """Parst Zeitspalte + Wertspalte, entfernt NaN und Duplikate.

    FIX #4: Duplikate werden per Mittelwert zusammengefasst.
    """
    out = df[[time_col, val_col]].copy()
    out[time_col] = pd.to_datetime(out[time_col], dayfirst=True, errors="coerce")

    if out[val_col].dtype == object:
        out[val_col] = out[val_col].astype(str).str.replace(",", ".").str.strip()
    out[val_col] = pd.to_numeric(out[val_col], errors="coerce")

    out = out.dropna().rename(columns={time_col: "datetime", val_col: val_name})

    n_before = len(out)
    out = out.groupby("datetime", as_index=False).agg({val_name: "mean"})
    n_dupes = n_before - len(out)
    if n_dupes > 0:
        st.info(f"ℹ️ {n_dupes} doppelte Zeitstempel zusammengefasst (Mittelwert).")

    return out.sort_values("datetime").reset_index(drop=True)


def validate_data(df: pd.DataFrame, val_col: str, label: str) -> None:
    """Prüft Datenqualität und zeigt Warnungen.

    FIX #5: Validierung für negative Werte, Lücken, Ausreißer.
    """
    n_neg = int((df[val_col] < 0).sum())
    if n_neg:
        st.warning(f"⚠️ {label}: {n_neg} negative Werte gefunden.")

    if len(df) > 2:
        diffs = df["datetime"].diff().dropna()
        median_diff = diffs.median()
        if median_diff > pd.Timedelta(0):
            large_gaps = diffs[diffs > median_diff * 3]
            if len(large_gaps) > 0:
                st.warning(
                    f"⚠️ {label}: {len(large_gaps)} Zeitlücken > 3× Median-Intervall "
                    f"({median_diff}) erkannt."
                )

    q1, q3 = df[val_col].quantile(0.25), df[val_col].quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        n_outliers = int(
            ((df[val_col] < q1 - 3 * iqr) | (df[val_col] > q3 + 3 * iqr)).sum()
        )
        if n_outliers:
            st.warning(f"⚠️ {label}: {n_outliers} statistische Ausreißer (>3× IQR).")


# ═══════════════════════════════════════════════════════════════
# Backtesting-Engine  –  FIX #6: von UI getrennt
# ═══════════════════════════════════════════════════════════════

def compute_weights(pattern: str, n_tranches: int) -> np.ndarray:
    """Tranchengewichte für ein Beschaffungsmuster (summieren sich zu 1)."""
    if "Frontloaded" in pattern:
        raw = np.linspace(2.0, 0.5, n_tranches)
    elif "Backloaded" in pattern:
        raw = np.linspace(0.5, 2.0, n_tranches)
    else:  # Gleichmäßig oder Fallback
        raw = np.ones(n_tranches)
    return raw / raw.sum()


def merge_load_and_spot(
    load_df: pd.DataFrame, spot_df: pd.DataFrame
) -> Tuple[pd.DataFrame, str]:
    """Merged Last + Spot.  Erst exakter Join, dann täglicher Fallback.

    FIX #11: Info über gewählte Granularität wird zurückgegeben.
    """
    merged = pd.merge(load_df, spot_df, on="datetime", how="inner")
    if len(merged) > 0:
        return merged, "exakt (Datetime-Match)"

    # Fallback: tägliche Aggregation
    load_daily = (
        load_df.assign(date=load_df["datetime"].dt.date)
        .groupby("date")
        .agg(load_mwh=("load_mwh", "sum"))
        .reset_index()
    )
    spot_daily = (
        spot_df.assign(date=spot_df["datetime"].dt.date)
        .groupby("date")
        .agg(spot_price=("spot_price", "mean"))
        .reset_index()
    )
    merged = pd.merge(load_daily, spot_daily, on="date", how="inner")
    if len(merged) > 0:
        merged["datetime"] = pd.to_datetime(merged["date"])
        merged = merged.drop(columns=["date"])
        return merged, "täglich (aggregiert)"

    return pd.DataFrame(), "keine Überlappung"


def run_backtesting(
    load_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    fwd_df: pd.DataFrame,
    config: Dict,
    progress_bar=None,
) -> Dict:
    """Kern-Berechnung des Backtestings – rein funktional, keine UI-Calls.

    FIX #6: Gesamte Logik extrahiert, testbar, wiederverwendbar.
    FIX #8: Optionaler Fortschrittsbalken.

    Returns
    -------
    dict  mit  results_df, merged, total_spot_cost, total_demand,
               avg_spot, cumulative_data, fwd_before, merge_info
    """
    delivery_start = load_df["datetime"].min()

    # Forward-Daten vor Lieferbeginn
    fwd_before = fwd_df[fwd_df["datetime"] < delivery_start].copy()
    if fwd_before.empty:
        raise ValueError(
            "Keine Terminmarktdaten VOR der Lieferperiode. "
            "Forward-Preise müssen zeitlich VOR dem Lastprofil liegen."
        )

    # Last + Spot zusammenführen
    merged, merge_info = merge_load_and_spot(load_df, spot_df)
    if merged.empty:
        raise ValueError(
            "Kein Zeitüberlapp zwischen Last und Spot. "
            "Beide müssen den gleichen Lieferzeitraum abdecken."
        )

    # Benchmark 100 % Spot
    merged["spot_cost_per_period"] = merged["load_mwh"] * merged["spot_price"]
    total_demand = merged["load_mwh"].sum()
    total_spot_cost = merged["spot_cost_per_period"].sum()
    avg_spot = total_spot_cost / total_demand if total_demand > 0 else 0.0

    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged["cum_spot_cost"] = merged["spot_cost_per_period"].cumsum()
    merged["cum_load"] = merged["load_mwh"].cumsum()

    # Szenarien
    fwd_vals = fwd_before["forward_price"].values
    n_fwd = len(fwd_before)

    total_scenarios = len(config["patterns"]) * len(config["forward_shares"])
    results: List[Dict] = []
    cumulative_data: Dict[str, np.ndarray] = {}
    done = 0

    for pat in config["patterns"]:
        actual_t = min(config["n_tranches"], n_fwd)
        indices = np.round(np.linspace(0, n_fwd - 1, actual_t)).astype(int)
        weights = compute_weights(pat, actual_t)
        wavg_fwd = float(np.dot(weights, fwd_vals[indices]))

        pat_short = pat.split("(")[0].strip()

        for fwd_share in config["forward_shares"]:
            spot_share = 1.0 - fwd_share
            fwd_volume = total_demand * fwd_share

            fwd_cost = fwd_volume * wavg_fwd
            tx_total = fwd_volume * config["tx_cost"]
            spot_cost = float(
                (merged["load_mwh"] * spot_share * merged["spot_price"]).sum()
            )

            total_cost = fwd_cost + spot_cost + tx_total
            avg_price = total_cost / total_demand if total_demand > 0 else 0.0
            pnl = total_spot_cost - total_cost
            pct = (pnl / total_spot_cost * 100) if total_spot_cost != 0 else 0.0

            name = f"{pat_short} ({fwd_share*100:.0f}%T / {spot_share*100:.0f}%S)"

            results.append(
                {
                    "Strategie": name,
                    "Muster": pat_short,
                    "Terminanteil [%]": fwd_share * 100,
                    "Spotanteil [%]": spot_share * 100,
                    "Ø Forward-Preis [€/MWh]": wavg_fwd,
                    "Terminvolumen [MWh]": fwd_volume,
                    "Terminkosten [€]": fwd_cost,
                    "Spotvolumen [MWh]": total_demand * spot_share,
                    "Ø Spot-Preis [€/MWh]": avg_spot,
                    "Spotkosten [€]": spot_cost,
                    "TX-Kosten [€]": tx_total,
                    "Gesamtkosten [€]": total_cost,
                    "Ø Beschaffungspreis [€/MWh]": avg_price,
                    "PnL vs. Spot [€]": pnl,
                    "Ersparnis [%]": pct,
                }
            )

            period_cost = (
                merged["load_mwh"] * fwd_share * (wavg_fwd + config["tx_cost"])
                + merged["load_mwh"] * spot_share * merged["spot_price"]
            )
            cumulative_data[name] = period_cost.cumsum().values

            done += 1
            if progress_bar is not None:
                progress_bar.progress(
                    done / total_scenarios,
                    text=f"Szenario {done}/{total_scenarios}: {name}",
                )

    results_df = (
        pd.DataFrame(results)
        .sort_values("Gesamtkosten [€]")
        .reset_index(drop=True)
    )
    results_df.index += 1

    return {
        "results_df": results_df,
        "merged": merged,
        "total_spot_cost": total_spot_cost,
        "total_demand": total_demand,
        "avg_spot": avg_spot,
        "cumulative_data": cumulative_data,
        "fwd_before": fwd_before,
        "merge_info": merge_info,
    }


# ═══════════════════════════════════════════════════════════════
# Chart-Funktionen  –  FIX #7: Wiederverwendbar extrahiert
# ═══════════════════════════════════════════════════════════════

def _pnl_colors(pnl_series: pd.Series) -> List[str]:
    return [
        COLOR_POSITIVE if v > 0 else COLOR_NEGATIVE if v < 0 else COLOR_NEUTRAL
        for v in pnl_series
    ]


def chart_cost_comparison(results_df: pd.DataFrame, total_spot: float) -> go.Figure:
    """Balken: Gesamtkosten je Strategie + Benchmark-Linie."""
    fig = go.Figure(
        go.Bar(
            x=results_df["Strategie"],
            y=results_df["Gesamtkosten [€]"],
            marker_color=_pnl_colors(results_df["PnL vs. Spot [€]"]),
            text=results_df["Gesamtkosten [€]"].apply(lambda v: f"{v:,.0f} €"),
            textposition="outside",
            textfont_size=9,
            hovertemplate="<b>%{x}</b><br>Kosten: %{y:,.0f} €<extra></extra>",
        )
    )
    fig.add_hline(
        y=total_spot, line_dash="dash", line_color="red", line_width=2,
        annotation_text=f"100 % Spot: {total_spot:,.0f} €",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Gesamtkosten je Strategie vs. 100 % Spot",
        xaxis_title="Strategie", yaxis_title="Gesamtkosten [€]",
        xaxis_tickangle=-45, height=550, template=PLOTLY_TEMPLATE,
    )
    return fig


def chart_avg_price(
    results_df: pd.DataFrame, avg_spot: float, avg_fwd: Optional[float] = None
) -> go.Figure:
    """Balken: Ø Beschaffungspreis."""
    fig = go.Figure(
        go.Bar(
            x=results_df["Strategie"],
            y=results_df["Ø Beschaffungspreis [€/MWh]"],
            marker_color="steelblue",
            text=results_df["Ø Beschaffungspreis [€/MWh]"].apply(
                lambda v: f"{v:.2f}"
            ),
            textposition="outside",
            textfont_size=9,
        )
    )
    fig.add_hline(
        y=avg_spot, line_dash="dash", line_color="red",
        annotation_text=f"Ø Spot: {avg_spot:.2f} €/MWh",
    )
    if avg_fwd is not None:
        fig.add_hline(
            y=avg_fwd, line_dash="dot", line_color="orange",
            annotation_text=f"Ø Forward: {avg_fwd:.2f} €/MWh",
        )
    fig.update_layout(
        title="Ø Beschaffungspreis je Strategie",
        xaxis_title="Strategie", yaxis_title="Ø Preis [€/MWh]",
        xaxis_tickangle=-45, height=500, template=PLOTLY_TEMPLATE,
    )
    return fig


def chart_savings(results_df: pd.DataFrame) -> go.Figure:
    """Balken: Ersparnis vs. 100 % Spot."""
    fig = go.Figure(
        go.Bar(
            x=results_df["Strategie"],
            y=results_df["PnL vs. Spot [€]"],
            marker_color=_pnl_colors(results_df["PnL vs. Spot [€]"]),
            text=results_df["PnL vs. Spot [€]"].apply(lambda v: f"{v:+,.0f} €"),
            textposition="outside",
            textfont_size=9,
        )
    )
    fig.add_hline(y=0, line_color="white", line_width=1)
    fig.update_layout(
        title="Ersparnis vs. 100 % Spot (positiv = günstiger)",
        xaxis_title="Strategie", yaxis_title="Ersparnis [€]",
        xaxis_tickangle=-45, height=500, template=PLOTLY_TEMPLATE,
    )
    return fig


def chart_cumulative(
    merged: pd.DataFrame,
    cum_data: Dict[str, np.ndarray],
    show_keys: List[str],
) -> go.Figure:
    """Linienchart: kumulative Kosten über die Lieferperiode."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["datetime"], y=merged["cum_spot_cost"],
            mode="lines", name="100 % Spot",
            line=dict(color="red", width=3, dash="dash"),
        )
    )
    for i, key in enumerate(show_keys):
        if key in cum_data:
            fig.add_trace(
                go.Scatter(
                    x=merged["datetime"], y=cum_data[key],
                    mode="lines", name=key,
                    line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=2),
                )
            )
    fig.update_layout(
        title="Kumulative Beschaffungskosten",
        xaxis_title="Datum", yaxis_title="Kosten [€]",
        height=550, template=PLOTLY_TEMPLATE,
        legend=dict(font=dict(size=9)),
    )
    return fig


def chart_forward_timeline(
    fwd_before: pd.DataFrame, n_tranches: int
) -> go.Figure:
    """Forward-Preisverlauf + eingezeichnete Kaufzeitpunkte."""
    n_fwd = len(fwd_before)
    actual_t = min(n_tranches, n_fwd)
    idxs = np.round(np.linspace(0, n_fwd - 1, actual_t)).astype(int)

    avg_fwd = fwd_before["forward_price"].mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fwd_before["datetime"], y=fwd_before["forward_price"],
            mode="lines", name="Forward-Preis",
            line=dict(color=COLOR_FORWARD, width=2),
            fill="tozeroy", fillcolor="rgba(52,152,219,0.1)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fwd_before.iloc[idxs]["datetime"],
            y=fwd_before.iloc[idxs]["forward_price"],
            mode="markers", name="Kaufzeitpunkte",
            marker=dict(color=COLOR_NEGATIVE, size=10, symbol="triangle-up"),
        )
    )
    fig.add_hline(
        y=avg_fwd, line_dash="dot", line_color="orange",
        annotation_text=f"Ø {avg_fwd:.2f} €/MWh",
    )
    fig.update_layout(
        title="Forward-Preise vor Lieferbeginn",
        xaxis_title="Datum", yaxis_title="Preis [€/MWh]",
        height=450, template=PLOTLY_TEMPLATE,
    )
    return fig


def chart_spot_load(merged: pd.DataFrame) -> go.Figure:
    """Dual-Achsen: Spotpreis + Last."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=merged["datetime"], y=merged["spot_price"],
            name="Spotpreis [€/MWh]",
            line=dict(color=COLOR_SPOT, width=1.5),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=merged["datetime"], y=merged["load_mwh"],
            name="Last [MWh]",
            marker_color="rgba(52,152,219,0.3)",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Spotpreise & Last während der Lieferperiode",
        height=450, template=PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(title_text="Spotpreis [€/MWh]", secondary_y=False)
    fig.update_yaxes(title_text="Last [MWh]", secondary_y=True)
    return fig


def chart_sensitivity(
    results_df: pd.DataFrame, pattern_name: str, total_spot: float
) -> Optional[go.Figure]:
    """Linien+Marker: Kosten vs. Terminanteil, mit Optimum-Stern."""
    sub = results_df[results_df["Muster"] == pattern_name].sort_values(
        "Terminanteil [%]"
    )
    if len(sub) <= 1:
        return None

    best_idx = sub["Gesamtkosten [€]"].idxmin()
    best = sub.loc[best_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["Terminanteil [%]"], y=sub["Gesamtkosten [€]"],
            mode="lines+markers", name="Gesamtkosten",
            line=dict(color=COLOR_POSITIVE, width=2), marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[best["Terminanteil [%]"]], y=[best["Gesamtkosten [€]"]],
            mode="markers",
            name=f"Optimum ({best['Terminanteil [%]']:.0f} %)",
            marker=dict(color="gold", size=14, symbol="star"),
        )
    )
    fig.add_hline(
        y=total_spot, line_dash="dash", line_color="red",
        annotation_text="100 % Spot",
    )
    fig.update_layout(
        title=f"Sensitivität – {pattern_name}",
        xaxis_title="Terminanteil [%]", yaxis_title="Gesamtkosten [€]",
        height=400, template=PLOTLY_TEMPLATE,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# Wiederverwend­bares Widget: Datei-Upload + Spalten-Auswahl
# ═══════════════════════════════════════════════════════════════

def data_upload_block(
    label: str,
    file_key: str,
    state_key: str,
    val_label: str,
    val_name: str,
) -> None:
    """Kapselt Upload → Vorschau → Spaltenauswahl → Übernahme.

    Reduziert den dreifach duplizierten Code im Datenimport-Tab.
    """
    uploaded = st.file_uploader(
        f"{label} hochladen", type=["csv", "xlsx", "xls"], key=file_key
    )
    if uploaded:
        raw = load_file(uploaded)
        if raw is not None:
            with st.expander("Vorschau Rohdaten", expanded=True):
                st.dataframe(raw.head(10), use_container_width=True)

            cols = list(raw.columns)
            auto_t = detect_datetime_col(raw)
            auto_v = detect_numeric_col(raw, [auto_t] if auto_t else [])

            c1, c2 = st.columns(2)
            tc = c1.selectbox(
                "Zeitspalte", cols,
                index=cols.index(auto_t) if auto_t in (cols or []) else 0,
                key=f"{file_key}_tc",
            )
            vc = c2.selectbox(
                val_label, cols,
                index=cols.index(auto_v) if auto_v in (cols or []) else min(1, len(cols) - 1),
                key=f"{file_key}_vc",
            )

            if st.button(f"✅ {label} übernehmen", key=f"{file_key}_btn"):
                df = parse_and_clean(raw, tc, vc, val_name)
                if df.empty:
                    st.error("Keine gültigen Daten nach Parsing.")
                else:
                    validate_data(df, val_name, label)
                    st.session_state[state_key] = df
                    st.success(
                        f"{label} geladen: **{len(df)}** Einträge, "
                        f"{df['datetime'].min().date()} – {df['datetime'].max().date()}"
                    )

    # Aktiv-Info
    df = st.session_state[state_key]
    if df is not None:
        st.info(
            f"📊 {label} aktiv: **{df['datetime'].min().date()}** – "
            f"**{df['datetime'].max().date()}** | {len(df)} Einträge"
        )


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════

st.sidebar.title("⚡ Energiebeschaffung")
st.sidebar.caption("Simulation & Backtesting")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["📥 Datenimport", "⚙️ Strategien", "🔬 Backtesting", "📊 Dashboard"],
)

st.sidebar.divider()
st.sidebar.markdown("### Datenstatus")
for key, (label, _) in {
    "load_df": ("Lastprofil", "load_mwh"),
    "spot_df": ("Spotpreise", "spot_price"),
    "forward_df": ("Terminpreise", "forward_price"),
}.items():
    if st.session_state[key] is not None:
        st.sidebar.success(f"✅ {label}: {len(st.session_state[key])} Zeilen")
    else:
        st.sidebar.warning(f"⏳ {label}: fehlt")

# FIX #9: Reset-Button
st.sidebar.divider()
if st.sidebar.button("🗑️ Alle Daten zurücksetzen", use_container_width=True):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v.copy() if isinstance(v, dict) else v
    st.rerun()


# ══════════════════════════════════════════════════════════════
# 📥 DATENIMPORT
# ══════════════════════════════════════════════════════════════

if page == "📥 Datenimport":
    st.header("📥 Datenimport")

    st.markdown("""
> **Drei Datensätze werden benötigt:**
> 1. **Lastprofil** – Verbrauch während der **Lieferperiode** (z. B. stündlich für 2025)
> 2. **Spotpreise** – Day-Ahead-Preise **während der Lieferperiode** (gleicher Zeitraum)
> 3. **Terminmarktpreise** – Forward-Preise **VOR der Lieferperiode**
    """)

    st.subheader("1️⃣ Lastprofil (Lieferperiode)")
    st.caption("Zeitreihe Ihres Stromverbrauchs in MWh")
    data_upload_block("Lastprofil", "load_up", "load_df", "Verbrauch [MWh]", "load_mwh")

    st.divider()

    st.subheader("2️⃣ Spotpreise (während Lieferperiode)")
    st.caption("Historische Day-Ahead- / Spotpreise in €/MWh")
    data_upload_block("Spotpreise", "spot_up", "spot_df", "Preis [€/MWh]", "spot_price")

    st.divider()

    st.subheader("3️⃣ Terminmarktpreise (VOR Lieferperiode)")
    st.caption("Forward-Preise für das Lieferprodukt, beobachtet VOR Lieferbeginn")
    data_upload_block(
        "Terminmarktpreise", "fwd_up", "forward_df",
        "Preis [€/MWh]", "forward_price",
    )


# ══════════════════════════════════════════════════════════════
# ⚙️ STRATEGIEN
# ══════════════════════════════════════════════════════════════

elif page == "⚙️ Strategien":
    st.header("⚙️ Beschaffungsstrategien konfigurieren")

    st.markdown("""
> **Terminanteil:** Wird VOR Lieferbeginn in Tranchen zu Forward-Preisen gekauft
> **Spotanteil:** Wird WÄHREND der Lieferung zum jeweiligen Spotpreis beschafft
> **Muster:** Bestimmt, WANN die Tranchen verteilt werden
    """)

    st.subheader("Terminmarkt-Anteile")
    mode = st.radio(
        "Modus",
        ["Standard (0 %–100 % in 10er-Schritten)", "Benutzerdefiniert"],
        horizontal=True,
    )

    if mode.startswith("Standard"):
        forward_shares = [i / 100 for i in range(0, 101, 10)]
    else:
        txt = st.text_input(
            "Terminanteile in % (kommagetrennt)",
            "0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100",
        )
        try:
            forward_shares = sorted(
                {max(0.0, min(1.0, float(x.strip()) / 100)) for x in txt.split(",")}
            )
        except ValueError:
            forward_shares = [i / 100 for i in range(0, 101, 10)]
            st.warning("Ungültige Eingabe – verwende Standard.")

    st.write(
        "**Simulierte Anteile:**",
        ", ".join(f"{s*100:.0f} %" for s in forward_shares),
    )

    st.subheader("Beschaffungsmuster am Terminmarkt")
    patterns = st.multiselect(
        "Muster",
        ["Gleichmäßig", "Frontloaded (früh mehr kaufen)", "Backloaded (spät mehr kaufen)"],
        default=["Gleichmäßig"],
    )
    if not patterns:
        patterns = ["Gleichmäßig"]

    n_tranches = st.slider(
        "Anzahl Beschaffungstranchen", 2, 36, 6
    )

    st.subheader("Transaktionskosten (optional)")
    tx_cost = st.number_input(
        "Aufschlag pro MWh Forward-Kauf [€/MWh]",
        min_value=0.0, value=0.0, step=0.05, format="%.2f",
    )

    st.session_state.strategy_config = {
        "forward_shares": forward_shares,
        "patterns": patterns,
        "n_tranches": n_tranches,
        "tx_cost": tx_cost,
    }

    n_scenarios = len(forward_shares) * len(patterns)
    st.success(
        f"Konfiguration: {len(forward_shares)} Anteile × {len(patterns)} Muster "
        f"= **{n_scenarios} Szenarien**"
    )

    # ── Vorschau Kaufzeitpunkte ──
    if st.session_state.forward_df is not None:
        st.divider()
        st.subheader("Vorschau: Beschaffungszeitpunkte & Gewichte")

        fwd = st.session_state.forward_df
        n_fwd = len(fwd)
        actual_tranches = min(n_tranches, n_fwd)
        indices = np.round(np.linspace(0, n_fwd - 1, actual_tranches)).astype(int)

        for pat in patterns:
            weights = compute_weights(pat, actual_tranches)
            prices = fwd.iloc[indices]["forward_price"].values

            preview = pd.DataFrame(
                {
                    "Nr.": range(1, actual_tranches + 1),
                    "Kaufdatum": fwd.iloc[indices]["datetime"].dt.date.values,
                    "Forward-Preis [€/MWh]": prices,
                    "Gewicht": [f"{w*100:.1f} %" for w in weights],
                    "Gew. Preis [€/MWh]": prices * weights,
                }
            )
            wavg = preview["Gew. Preis [€/MWh]"].sum()
            st.markdown(
                f"**{pat}** – Ø gewichteter Forward-Preis: **{wavg:.2f} €/MWh**"
            )
            st.dataframe(preview, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# 🔬 BACKTESTING
# ══════════════════════════════════════════════════════════════

elif page == "🔬 Backtesting":
    st.header("🔬 Backtesting durchführen")

    # ── Validierung ──
    missing = [
        label
        for key, label in [
            ("load_df", "Lastprofil"),
            ("spot_df", "Spotpreise"),
            ("forward_df", "Terminmarktpreise"),
        ]
        if st.session_state[key] is None
    ]
    if missing:
        st.warning(
            f"⚠️ Fehlende Daten: **{', '.join(missing)}** "
            f"– bitte im Tab 'Datenimport' hochladen."
        )
        st.stop()

    load_df = st.session_state.load_df
    spot_df = st.session_state.spot_df
    fwd_df = st.session_state.forward_df
    config = st.session_state.strategy_config

    delivery_start = load_df["datetime"].min()
    delivery_end = load_df["datetime"].max()
    fwd_start = fwd_df["datetime"].min()
    fwd_end = fwd_df["datetime"].max()

    st.markdown(f"""
### Datenbasis
| Datensatz | Zeitraum | Details |
|-----------|----------|---------|
| 📊 Lastprofil | {delivery_start.date()} – {delivery_end.date()} | {load_df['load_mwh'].sum():,.0f} MWh, {len(load_df)} Eintr. |
| 💰 Spotpreise | {spot_df['datetime'].min().date()} – {spot_df['datetime'].max().date()} | Ø {spot_df['spot_price'].mean():.2f} €/MWh |
| 📈 Terminmarkt | {fwd_start.date()} – {fwd_end.date()} | Ø {fwd_df['forward_price'].mean():.2f} €/MWh, {len(fwd_df)} Tage |
    """)

    if fwd_end >= delivery_start:
        st.warning(
            f"⚠️ Forward-Daten reichen bis {fwd_end.date()}, Lieferung beginnt "
            f"{delivery_start.date()}. Es werden nur Preise **vor** Lieferbeginn verwendet."
        )

    st.info("""
**Was wird simuliert?**
- **Terminkosten:** Terminanteil × Bedarf, gekauft in Tranchen am Forward-Markt
- **Spotkosten:** Restbedarf, gekauft während der Lieferung zum Spotpreis
- **Benchmark:** 100 % Spot = alles am Spotmarkt
    """)

    if st.button("🚀 Backtesting starten", type="primary", use_container_width=True):

        progress_bar = st.progress(0, text="Starte Berechnung …")

        try:
            bt = run_backtesting(load_df, spot_df, fwd_df, config, progress_bar)
        except ValueError as exc:
            st.error(f"❌ {exc}")
            st.stop()

        progress_bar.empty()

        # In Session State speichern
        st.session_state.bt_results = bt["results_df"]
        st.session_state.bt_merged = bt["merged"]
        st.session_state.bt_spot_total = bt["total_spot_cost"]
        st.session_state.bt_demand_total = bt["total_demand"]
        st.session_state.bt_avg_spot = bt["avg_spot"]
        st.session_state.bt_cumulative = bt["cumulative_data"]
        st.session_state.bt_fwd_before = bt["fwd_before"]

        st.success(
            f"✅ Backtesting abgeschlossen! **{len(bt['results_df'])}** Szenarien "
            f"berechnet. Merge-Modus: *{bt['merge_info']}*."
        )
        st.rerun()

    # ── Ergebnisse anzeigen ──
    if st.session_state.bt_results is not None:
        results_df = st.session_state.bt_results
        merged = st.session_state.bt_merged
        total_spot_cost = st.session_state.bt_spot_total
        total_demand = st.session_state.bt_demand_total
        avg_spot = st.session_state.bt_avg_spot
        cum_data = st.session_state.bt_cumulative or {}
        fwd_before = st.session_state.bt_fwd_before

        best = results_df.iloc[0]

        # ── KPIs ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "🏆 Beste Strategie",
            f"{best['Terminanteil [%]']:.0f} % / {best['Spotanteil [%]']:.0f} %",
            f"{best['PnL vs. Spot [€]']:+,.0f} € vs. Spot",
        )
        c2.metric(
            "Gesamtkosten (beste)",
            f"{best['Gesamtkosten [€]']:,.0f} €",
            f"{best['Ersparnis [%]']:+.1f} %",
        )
        c3.metric(
            "100 % Spot",
            f"{total_spot_cost:,.0f} €",
            f"Ø {avg_spot:.2f} €/MWh",
        )
        c4.metric(
            "Gesamtbedarf",
            f"{total_demand:,.0f} MWh",
            f"{len(merged)} Perioden",
        )

        st.divider()

        # ── Top 5 ──
        st.subheader("🏆 Top 5 günstigste Strategien")
        top5_cols = [
            "Strategie", "Gesamtkosten [€]", "Ø Beschaffungspreis [€/MWh]",
            "PnL vs. Spot [€]", "Ersparnis [%]", "Ø Forward-Preis [€/MWh]",
        ]
        top5_fmt = results_df.head(5)[top5_cols].copy()
        top5_fmt["Gesamtkosten [€]"] = top5_fmt["Gesamtkosten [€]"].map("{:,.2f}".format)
        top5_fmt["Ø Beschaffungspreis [€/MWh]"] = top5_fmt["Ø Beschaffungspreis [€/MWh]"].map("{:.2f}".format)
        top5_fmt["PnL vs. Spot [€]"] = top5_fmt["PnL vs. Spot [€]"].map("{:+,.2f}".format)
        top5_fmt["Ersparnis [%]"] = top5_fmt["Ersparnis [%]"].map("{:+.2f} %".format)
        top5_fmt["Ø Forward-Preis [€/MWh]"] = top5_fmt["Ø Forward-Preis [€/MWh]"].map("{:.2f}".format)
        st.dataframe(top5_fmt, use_container_width=True)

        st.divider()

        # ── Alle Ergebnisse ──
        with st.expander("📋 Alle Ergebnisse anzeigen"):
            st.dataframe(results_df, use_container_width=True)

        st.divider()

        # ══════ CHARTS ══════
        st.subheader("📊 Visualisierungen")

        # FIX #10: Bei vielen Strategien nur Teilmenge in Balken zeigen
        max_bars = st.slider(
            "Max. angezeigte Strategien in Balkendiagrammen",
            5, len(results_df), min(len(results_df), 15),
            key="max_bars",
        )
        plot_df = results_df.head(max_bars)

        st.plotly_chart(
            chart_cost_comparison(plot_df, total_spot_cost),
            use_container_width=True,
        )
        avg_fwd = fwd_before["forward_price"].mean() if fwd_before is not None else None
        st.plotly_chart(
            chart_avg_price(plot_df, avg_spot, avg_fwd),
            use_container_width=True,
        )
        st.plotly_chart(chart_savings(plot_df), use_container_width=True)

        # Kumulativ
        if cum_data:
            st.subheader("📈 Kumulative Kosten über Lieferperiode")
            show_strats = st.multiselect(
                "Strategien auswählen",
                list(cum_data.keys()),
                default=list(cum_data.keys())[:5],
            )
            st.plotly_chart(
                chart_cumulative(merged, cum_data, show_strats),
                use_container_width=True,
            )

        # Forward-Timeline
        if fwd_before is not None:
            st.subheader("📈 Terminmarktpreis-Entwicklung")
            st.plotly_chart(
                chart_forward_timeline(fwd_before, config["n_tranches"]),
                use_container_width=True,
            )

        # Spot + Last
        st.subheader("⚡ Spotpreis & Last")
        st.plotly_chart(chart_spot_load(merged), use_container_width=True)

        # Sensitivität
        st.subheader("🎯 Sensitivität: Kosten vs. Terminanteil")
        for pat in config["patterns"]:
            pat_name = pat.split("(")[0].strip()
            fig_s = chart_sensitivity(results_df, pat_name, total_spot_cost)
            if fig_s:
                st.plotly_chart(fig_s, use_container_width=True)


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
    fwd_before = st.session_state.bt_fwd_before
    config = st.session_state.strategy_config

    best = results_df.iloc[0]
    worst = results_df.iloc[-1]

    # ── KPIs ──
    st.subheader("🎯 Kennzahlen")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "🏆 Optimale Strategie",
        f"{best['Terminanteil [%]']:.0f} %T / {best['Spotanteil [%]']:.0f} %S",
    )
    c2.metric(
        "Beste Gesamtkosten",
        f"{best['Gesamtkosten [€]']:,.0f} €",
        f"{best['PnL vs. Spot [€]']:+,.0f} € vs. Spot",
    )
    c3.metric(
        "Schlechteste Strategie",
        f"{worst['Terminanteil [%]']:.0f} %T / {worst['Spotanteil [%]']:.0f} %S",
        f"{worst['PnL vs. Spot [€]']:+,.0f} €",
    )
    c4.metric(
        "Spanne",
        f"{worst['Gesamtkosten [€]'] - best['Gesamtkosten [€]']:,.0f} €",
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gesamtbedarf", f"{total_demand:,.0f} MWh")
    c2.metric("100 % Spot", f"{total_spot:,.0f} €")
    c3.metric("Ø Spot", f"{avg_spot:.2f} €/MWh")
    if fwd_before is not None:
        avg_fwd = fwd_before["forward_price"].mean()
        c4.metric(
            "Ø Forward",
            f"{avg_fwd:.2f} €/MWh",
            f"{avg_fwd - avg_spot:+.2f} vs. Spot",
        )

    st.divider()

    # ── Risikokennzahlen ──
    st.subheader("📉 Risikokennzahlen")
    costs = results_df["Gesamtkosten [€]"].values
    prices = results_df["Ø Beschaffungspreis [€/MWh]"].values

    risk_rows = [
        ("Min. Gesamtkosten", f"{costs.min():,.2f} €"),
        ("Max. Gesamtkosten", f"{costs.max():,.2f} €"),
        ("Ø Gesamtkosten", f"{costs.mean():,.2f} €"),
        ("Std.-Abw. Kosten", f"{costs.std():,.2f} €"),
        ("Spannweite", f"{costs.ptp():,.2f} €"),
        ("Min. Ø-Preis", f"{prices.min():.2f} €/MWh"),
        ("Max. Ø-Preis", f"{prices.max():.2f} €/MWh"),
        ("Volatilität Ø-Preis", f"{prices.std():.2f} €/MWh"),
    ]
    st.dataframe(
        pd.DataFrame(risk_rows, columns=["Kennzahl", "Wert"]),
        use_container_width=True,
        hide_index=True,
    )

    # VaR / CVaR
    pnl = results_df["PnL vs. Spot [€]"].values
    if len(pnl) > 2:
        var_95 = float(np.percentile(pnl, 5))
        mask = pnl <= var_95
        cvar_95 = float(pnl[mask].mean()) if mask.any() else var_95

        st.markdown(f"""
**Risikomaße:**
- **VaR (95 %):** {var_95:+,.2f} € – worst-case 5 % der Szenarien
- **CVaR (95 %):** {cvar_95:+,.2f} € – Erwartung in den schlechtesten 5 %
- **Max. Verlust vs. Spot:** {pnl.min():+,.2f} €
        """)

    st.divider()

    # ── Empfehlung ──
    st.subheader("💡 Empfehlung")

    if best["PnL vs. Spot [€]"] > 0:
        st.success(f"""
**Optimale Strategie: {best['Strategie']}**

- Gesamtkosten: **{best['Gesamtkosten [€]']:,.0f} €** (statt {total_spot:,.0f} €)
- Ersparnis: **{best['PnL vs. Spot [€]']:+,.0f} €** ({best['Ersparnis [%]']:+.1f} %)
- Ø Preis: **{best['Ø Beschaffungspreis [€/MWh]']:.2f} €/MWh** (statt {avg_spot:.2f})

→ **{best['Terminanteil [%]']:.0f} %** Terminmarkt hätte sich gelohnt.
        """)
    else:
        st.info(f"""
**100 % Spot wäre die günstigste Option gewesen.**

- Spot: **{total_spot:,.0f} €** (Ø {avg_spot:.2f} €/MWh)
- Beste Alternative: {best['Strategie']} → **{best['Gesamtkosten [€]']:,.0f} €**
- Mehrkosten: **{abs(best['PnL vs. Spot [€]']):,.0f} €**

→ Forward-Preise lagen über Spot – Terminbeschaffung hätte sich nicht gelohnt.
        """)

    # ── Mustervergleich ──
    if len(config["patterns"]) > 1:
        st.subheader("🗺️ Strategievergleich nach Muster")
        for pat in config["patterns"]:
            pat_name = pat.split("(")[0].strip()
            sub = results_df[results_df["Muster"] == pat_name].sort_values(
                "Terminanteil [%]"
            )
            if not sub.empty:
                st.markdown(f"**{pat_name}:**")
                mini = sub[
                    [
                        "Terminanteil [%]", "Gesamtkosten [€]",
                        "Ø Beschaffungspreis [€/MWh]", "PnL vs. Spot [€]",
                        "Ersparnis [%]",
                    ]
                ].copy()
                mini["Gesamtkosten [€]"] = mini["Gesamtkosten [€]"].map("{:,.0f}".format)
                mini["Ø Beschaffungspreis [€/MWh]"] = mini["Ø Beschaffungspreis [€/MWh]"].map("{:.2f}".format)
                mini["PnL vs. Spot [€]"] = mini["PnL vs. Spot [€]"].map("{:+,.0f}".format)
                mini["Ersparnis [%]"] = mini["Ersparnis [%]"].map("{:+.2f} %".format)
                st.dataframe(mini, use_container_width=True, hide_index=True)

    # ── Export ──
    st.divider()
    st.subheader("💾 Ergebnisse exportieren")

    csv_data = results_df.to_csv(index=False, sep=";", decimal=",")
    st.download_button(
        "📥 CSV herunterladen",
        csv_data.encode("utf-8"),
        "backtesting_ergebnisse.csv",
        "text/csv",
        use_container_width=True,
    )

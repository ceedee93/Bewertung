import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import re
import calendar
from datetime import date, timedelta
from typing import Optional, Dict, List, Tuple

# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════

st.set_page_config(page_title="⚡ Energiebeschaffung", layout="wide", page_icon="⚡")

C = dict(pos="#2ecc71", neg="#e74c3c", neut="#95a5a6", blue="#3498db", orange="#e67e22",
         purple="#9b59b6", teal="#1abc9c", gold="#f1c40f")
PAL = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#1abc9c",
       "#f39c12", "#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]
TPL = "plotly_dark"

st.markdown("""<style>
    .block-container { padding-top: 1.2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; font-weight: 700; }
    textarea { font-family: 'Courier New', monospace !important; font-size: 12px !important; }
    .stTabs [data-baseweb="tab"] { padding: 8px 14px; font-weight: 600; }
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════

_DEF = dict(
    load_df=None, spot_df=None, forward_df=None,
    deals_df=None,        # Echte Deals
    schedule_df=None,     # Beschaffungsplan
    bt_results=None, bt_merged=None, bt_spot_total=None,
    bt_demand_total=None, bt_avg_spot=None,
    bt_cumulative=None, bt_fwd_before=None,
    config=dict(
        forward_shares=[i / 100 for i in range(0, 101, 10)],
        patterns=["Gleichmäßig"], n_tranches=6, tx_cost=0.0,
    ),
)
for _k, _v in _DEF.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v.copy() if isinstance(_v, (dict, list)) else _v


# ══════════════════════════════════════════════════
# PARSE & DATA FUNCTIONS
# ══════════════════════════════════════════════════

def parse_text(text: str) -> Optional[pd.DataFrame]:
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
    if f is None:
        return None
    try:
        if f.name.lower().endswith(".csv"):
            return parse_text(f.read().decode("utf-8", errors="replace"))
        return pd.read_excel(f)
    except Exception as e:
        st.error(f"Fehler: {e}")
        return None


def find_col(df, keywords, numeric=False):
    """Findet Spalte per Keyword-Match oder Typ."""
    for col in df.columns:
        low = col.lower().replace("_", " ").replace("-", " ")
        if any(k in low for k in keywords):
            if numeric:
                try:
                    vals = pd.to_numeric(
                        df[col].astype(str).str.replace(",", "."), errors="coerce"
                    )
                    if vals.notna().mean() > 0.3:
                        return col
                except Exception:
                    continue
            else:
                return col
    if numeric:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
            try:
                vals = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "."), errors="coerce"
                )
                if vals.notna().mean() > 0.3:
                    return col
            except Exception:
                continue
    return None


def find_datetime_col(df):
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in ("date", "datum", "zeit", "time", "tag")):
            try:
                p = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
                if p.notna().mean() > 0.5:
                    return col
            except Exception:
                continue
    for col in df.columns:
        try:
            p = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            if p.notna().mean() > 0.5:
                return col
        except Exception:
            continue
    return None


def find_value_col(df, exclude=None):
    exc = {exclude} if exclude else set()
    for col in df.columns:
        if col in exc:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
        try:
            v = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."), errors="coerce"
            )
            if v.notna().mean() > 0.3:
                return col
        except Exception:
            continue
    return None


def clean_data(df, tcol, vcol, vname):
    out = df[[tcol, vcol]].copy()
    out[tcol] = pd.to_datetime(out[tcol], dayfirst=True, errors="coerce")
    if out[vcol].dtype == object:
        out[vcol] = out[vcol].astype(str).str.replace(",", ".").str.strip()
    out[vcol] = pd.to_numeric(out[vcol], errors="coerce")
    out = out.dropna().rename(columns={tcol: "datetime", vcol: vname})
    n = len(out)
    out = out.groupby("datetime", as_index=False).agg({vname: "mean"})
    if len(out) < n:
        st.caption(f"ℹ️ {n - len(out)} Duplikate zusammengefasst")
    return out.sort_values("datetime").reset_index(drop=True)


# ══════════════════════════════════════════════════
# PRODUKT-PARSER
# ══════════════════════════════════════════════════

MONTHS_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "mär": 3, "mrz": 3,
    "apr": 4, "mai": 5, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "okt": 10, "oct": 10,
    "nov": 11, "dez": 12, "dec": 12,
}


def parse_product_period(name: str) -> Tuple[Optional[date], Optional[date]]:
    """Erkennt Lieferzeitraum aus Produktnamen.

    Unterstützt: Cal/Year-25, Q1-25, H1-25, Jan-25, M01-25
    """
    if not name:
        return None, None
    s = name.strip().upper().replace("/", "-").replace(" ", "-")

    def _year(y):
        y = int(y)
        return y + 2000 if y < 100 else y

    # Cal / Year: Cal-25, Cal2025, Year-25, Y25
    m = re.search(r"(?:CAL|YEAR|Y|JA)-?(\d{2,4})", s)
    if m:
        yr = _year(m.group(1))
        return date(yr, 1, 1), date(yr, 12, 31)

    # Quarter: Q1-25, Q2/25
    m = re.search(r"Q([1-4])-?(\d{2,4})", s)
    if m:
        q, yr = int(m.group(1)), _year(m.group(2))
        sm = (q - 1) * 3 + 1
        em = q * 3
        ld = calendar.monthrange(yr, em)[1]
        return date(yr, sm, 1), date(yr, em, ld)

    # Half: H1-25, H2-25
    m = re.search(r"H([12])-?(\d{2,4})", s)
    if m:
        h, yr = int(m.group(1)), _year(m.group(2))
        if h == 1:
            return date(yr, 1, 1), date(yr, 6, 30)
        return date(yr, 7, 1), date(yr, 12, 31)

    # Month by name: Jan-25, Feb-25, Dez-25
    low = name.strip().lower()
    for mname, mnum in MONTHS_MAP.items():
        m = re.search(rf"{mname}\w*[- /]?(\d{{2,4}})", low)
        if m:
            yr = _year(m.group(1))
            ld = calendar.monthrange(yr, mnum)[1]
            return date(yr, mnum, 1), date(yr, mnum, ld)

    # Month by number: M01-25, M12-25
    m = re.search(r"M(\d{1,2})-?(\d{2,4})", s)
    if m:
        mn, yr = int(m.group(1)), _year(m.group(2))
        if 1 <= mn <= 12:
            ld = calendar.monthrange(yr, mn)[1]
            return date(yr, mn, 1), date(yr, mn, ld)

    return None, None


def parse_deals(raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Parst Deal-Tabelle mit intelligenter Spaltenerkennung.

    Erkennt: Produkt, Kaufdatum, Lieferstart/-ende, Leistung (MW), Menge (MWh), Preis (€/MWh)
    """
    if raw_df is None or raw_df.empty:
        return None

    cols = {c: c.lower().replace("_", " ").replace("-", " ") for c in raw_df.columns}
    result = pd.DataFrame()

    # Produkt
    for orig, low in cols.items():
        if any(k in low for k in ("produkt", "product", "name", "kontrakt", "contract")):
            result["produkt"] = raw_df[orig].astype(str).str.strip()
            break

    if "produkt" not in result.columns:
        result["produkt"] = raw_df.iloc[:, 0].astype(str).str.strip()

    # Parse product → delivery period
    starts, ends = [], []
    for p in result["produkt"]:
        s, e = parse_product_period(p)
        starts.append(s)
        ends.append(e)

    # Kaufdatum
    kd_col = None
    for orig, low in cols.items():
        if any(k in low for k in ("kaufdatum", "kauf", "buy", "trade", "handelstag")):
            kd_col = orig
            break
    if kd_col:
        result["kaufdatum"] = pd.to_datetime(raw_df[kd_col], dayfirst=True, errors="coerce")

    # Lieferstart / Lieferende – explizit oder aus Produkt
    start_col = end_col = None
    for orig, low in cols.items():
        if any(k in low for k in ("lieferstart", "start", "begin", "von", "from", "delivery start")):
            start_col = orig
        if any(k in low for k in ("lieferende", "ende", "end", "bis", "to", "delivery end")):
            end_col = orig

    if start_col:
        result["lieferstart"] = pd.to_datetime(raw_df[start_col], dayfirst=True, errors="coerce")
    else:
        result["lieferstart"] = pd.Series(starts).apply(
            lambda x: pd.Timestamp(x) if x else pd.NaT
        )

    if end_col:
        result["lieferende"] = pd.to_datetime(raw_df[end_col], dayfirst=True, errors="coerce")
    else:
        result["lieferende"] = pd.Series(ends).apply(
            lambda x: pd.Timestamp(x) if x else pd.NaT
        )

    # Leistung (MW)
    for orig, low in cols.items():
        if any(k in low for k in ("leistung", "mw", "power", "kapaz")):
            if "mwh" not in low:
                vals = raw_df[orig]
                if vals.dtype == object:
                    vals = vals.astype(str).str.replace(",", ".")
                result["leistung_mw"] = pd.to_numeric(vals, errors="coerce")
                break

    # Menge (MWh) – falls vorhanden
    for orig, low in cols.items():
        if any(k in low for k in ("menge", "mwh", "volume", "energy")):
            vals = raw_df[orig]
            if vals.dtype == object:
                vals = vals.astype(str).str.replace(",", ".")
            result["menge_mwh"] = pd.to_numeric(vals, errors="coerce")
            break

    # Preis
    for orig, low in cols.items():
        if any(k in low for k in ("preis", "price", "eur", "€")):
            vals = raw_df[orig]
            if vals.dtype == object:
                vals = vals.astype(str).str.replace(",", ".").str.replace("€", "").str.strip()
            result["preis"] = pd.to_numeric(vals, errors="coerce")
            break

    # Profil (Base/Peak) – optional
    for orig, low in cols.items():
        if any(k in low for k in ("profil", "profile", "typ", "type", "base", "peak")):
            result["profil"] = raw_df[orig].astype(str).str.strip()
            break
    if "profil" not in result.columns:
        # Versuch aus Produktname zu extrahieren
        result["profil"] = result["produkt"].apply(
            lambda x: "Peak" if "peak" in x.lower() else "Base"
        )

    # Berechne Menge aus Leistung wenn nötig
    if "leistung_mw" in result.columns and "menge_mwh" not in result.columns:
        hours = []
        for _, row in result.iterrows():
            if pd.notna(row.get("lieferstart")) and pd.notna(row.get("lieferende")):
                h = (row["lieferende"] - row["lieferstart"]).total_seconds() / 3600 + 24
                hours.append(h)
            else:
                hours.append(np.nan)
        result["stunden"] = hours
        result["menge_mwh"] = result.get("leistung_mw", 0) * result["stunden"]

    # Wenn kein Preis gefunden, Fehler
    if "preis" not in result.columns:
        return None

    return result.dropna(subset=["preis"]).reset_index(drop=True)


def parse_schedule(raw_df: pd.DataFrame, fwd_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Parst Beschaffungsplan: Datum, Anteil/Menge, optionaler Preis."""
    if raw_df is None or raw_df.empty:
        return None

    result = pd.DataFrame()

    # Datum
    dt_col = find_datetime_col(raw_df)
    if dt_col is None:
        return None
    result["datum"] = pd.to_datetime(raw_df[dt_col], dayfirst=True, errors="coerce")

    cols_lower = {c: c.lower().replace("_", " ").replace("-", " ") for c in raw_df.columns}

    # Anteil (%)
    anteil_col = None
    for orig, low in cols_lower.items():
        if any(k in low for k in ("anteil", "share", "prozent", "%", "pct")):
            anteil_col = orig
            break

    # Menge (MWh)
    menge_col = None
    for orig, low in cols_lower.items():
        if orig == anteil_col:
            continue
        if any(k in low for k in ("menge", "volume", "mwh", "energy")):
            menge_col = orig
            break

    if anteil_col:
        vals = raw_df[anteil_col]
        if vals.dtype == object:
            vals = vals.astype(str).str.replace(",", ".").str.replace("%", "").str.strip()
        result["anteil_pct"] = pd.to_numeric(vals, errors="coerce")
    elif menge_col:
        vals = raw_df[menge_col]
        if vals.dtype == object:
            vals = vals.astype(str).str.replace(",", ".")
        result["menge_mwh"] = pd.to_numeric(vals, errors="coerce")
    else:
        # Fallback: zweite Spalte als Anteil
        second = [c for c in raw_df.columns if c != dt_col]
        if second:
            vals = raw_df[second[0]]
            if vals.dtype == object:
                vals = vals.astype(str).str.replace(",", ".").str.replace("%", "")
            result["anteil_pct"] = pd.to_numeric(vals, errors="coerce")

    # Preis (optional)
    preis_col = None
    for orig, low in cols_lower.items():
        if orig in (dt_col, anteil_col, menge_col):
            continue
        if any(k in low for k in ("preis", "price", "eur", "€", "forward")):
            preis_col = orig
            break
    if preis_col:
        vals = raw_df[preis_col]
        if vals.dtype == object:
            vals = vals.astype(str).str.replace(",", ".").str.replace("€", "").str.strip()
        result["preis"] = pd.to_numeric(vals, errors="coerce")

    # Forward-Lookup für fehlende Preise
    if fwd_df is not None and "preis" in result.columns:
        for i, row in result.iterrows():
            if pd.isna(row["preis"]) and pd.notna(row["datum"]):
                # Nächsten Forward-Preis finden
                diffs = abs(fwd_df["datetime"] - row["datum"])
                nearest = diffs.idxmin()
                if diffs.loc[nearest] <= pd.Timedelta(days=5):
                    result.at[i, "preis"] = fwd_df.loc[nearest, "forward_price"]
    elif fwd_df is not None and "preis" not in result.columns:
        prices = []
        for _, row in result.iterrows():
            if pd.notna(row["datum"]):
                diffs = abs(fwd_df["datetime"] - row["datum"])
                nearest = diffs.idxmin()
                if diffs.loc[nearest] <= pd.Timedelta(days=5):
                    prices.append(fwd_df.loc[nearest, "forward_price"])
                else:
                    prices.append(np.nan)
            else:
                prices.append(np.nan)
        result["preis"] = prices

    return result.dropna(subset=["datum"]).reset_index(drop=True)


# ══════════════════════════════════════════════════
# DEMO DATA
# ══════════════════════════════════════════════════

def generate_demo():
    rng = np.random.default_rng(42)

    dates_25 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    n = len(dates_25)
    season = 20 * np.sin(np.linspace(0, 2 * np.pi, n))

    load_df = pd.DataFrame({
        "datetime": dates_25,
        "load_mwh": np.maximum(100 + season + rng.normal(0, 8, n), 20),
    })
    spot_df = pd.DataFrame({
        "datetime": dates_25,
        "spot_price": np.maximum(75 + season * 0.75 + rng.normal(0, 12, n), 5),
    })

    dates_24 = pd.bdate_range("2024-01-02", "2024-12-30")[:250]
    fwd_df = pd.DataFrame({
        "datetime": dates_24,
        "forward_price": np.maximum(82 + np.cumsum(rng.normal(-0.02, 0.8, len(dates_24))), 40),
    })

    deals_df = pd.DataFrame({
        "produkt": ["Cal-25 Base", "Q1-25 Base", "Q3-25 Base"],
        "kaufdatum": pd.to_datetime(["2024-03-15", "2024-06-01", "2024-09-10"]),
        "lieferstart": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-07-01"]),
        "lieferende": pd.to_datetime(["2025-12-31", "2025-03-31", "2025-09-30"]),
        "leistung_mw": [10.0, 5.0, 3.0],
        "preis": [82.50, 85.00, 78.20],
        "profil": ["Base", "Base", "Base"],
        "stunden": [8760.0, 2160.0, 2208.0],
        "menge_mwh": [87600.0, 10800.0, 6624.0],
    })

    schedule_df = pd.DataFrame({
        "datum": pd.to_datetime([
            "2024-02-15", "2024-04-15", "2024-06-15",
            "2024-08-15", "2024-10-15", "2024-12-02",
        ]),
        "anteil_pct": [10.0, 15.0, 20.0, 25.0, 20.0, 10.0],
        "preis": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    })

    return load_df, spot_df, fwd_df, deals_df, schedule_df


# ══════════════════════════════════════════════════
# BACKTESTING ENGINE
# ══════════════════════════════════════════════════

def compute_weights(pat: str, n: int) -> np.ndarray:
    if "Frontloaded" in pat:
        w = np.linspace(2.0, 0.5, n)
    elif "Backloaded" in pat:
        w = np.linspace(0.5, 2.0, n)
    else:
        w = np.ones(n)
    return w / w.sum()


def merge_load_spot(load_df, spot_df) -> Tuple[pd.DataFrame, str]:
    m = pd.merge(load_df, spot_df, on="datetime", how="inner")
    if len(m) > 0:
        return m, "exakt"
    ld = load_df.assign(d=load_df.datetime.dt.date).groupby("d").agg(
        load_mwh=("load_mwh", "sum")
    ).reset_index()
    sd = spot_df.assign(d=spot_df.datetime.dt.date).groupby("d").agg(
        spot_price=("spot_price", "mean")
    ).reset_index()
    m = pd.merge(ld, sd, on="d", how="inner")
    if len(m) > 0:
        m["datetime"] = pd.to_datetime(m["d"])
        return m.drop(columns="d"), "täglich"
    return pd.DataFrame(), "kein Überlapp"


def calc_deal_strategy(deals_df, merged, total_demand, total_spot):
    """Berechnet Kosten der echten Deals als Strategie."""
    if deals_df is None or deals_df.empty:
        return None

    # Stunden pro Periode ermitteln
    if len(merged) > 1:
        med = merged["datetime"].diff().dropna().median()
        h_per_period = max(med.total_seconds() / 3600, 0.25)
    else:
        h_per_period = 24.0

    merged_c = merged.copy()
    merged_c["deal_mwh"] = 0.0
    merged_c["deal_cost"] = 0.0

    for _, deal in deals_df.iterrows():
        if pd.isna(deal.get("lieferstart")) or pd.isna(deal.get("lieferende")):
            continue
        if pd.isna(deal.get("preis")):
            continue

        mask = (merged_c["datetime"] >= deal["lieferstart"]) & (
            merged_c["datetime"] <= deal["lieferende"]
        )

        if "leistung_mw" in deal and pd.notna(deal["leistung_mw"]):
            mwh_per_period = deal["leistung_mw"] * h_per_period
        elif "menge_mwh" in deal and pd.notna(deal["menge_mwh"]):
            n_periods = mask.sum()
            mwh_per_period = deal["menge_mwh"] / n_periods if n_periods > 0 else 0
        else:
            continue

        merged_c.loc[mask, "deal_mwh"] += mwh_per_period
        merged_c.loc[mask, "deal_cost"] += mwh_per_period * deal["preis"]

    # Deal-Volumen kann Last nicht übersteigen
    merged_c["deal_mwh_eff"] = merged_c[["deal_mwh", "load_mwh"]].min(axis=1)
    merged_c["spot_mwh"] = merged_c["load_mwh"] - merged_c["deal_mwh_eff"]
    merged_c["spot_cost"] = merged_c["spot_mwh"] * merged_c["spot_price"]

    # Deal-Kosten proportional kürzen wenn überdeckt
    ratio = np.where(
        merged_c["deal_mwh"] > 0,
        merged_c["deal_mwh_eff"] / merged_c["deal_mwh"],
        0,
    )
    merged_c["deal_cost_eff"] = merged_c["deal_cost"] * ratio

    deal_cost_total = float(merged_c["deal_cost_eff"].sum())
    spot_cost_total = float(merged_c["spot_cost"].sum())
    total_cost = deal_cost_total + spot_cost_total
    deal_vol = float(merged_c["deal_mwh_eff"].sum())
    deal_pct = deal_vol / total_demand * 100 if total_demand > 0 else 0
    avg_deal_price = deal_cost_total / deal_vol if deal_vol > 0 else 0

    pnl = total_spot - total_cost
    avg_price = total_cost / total_demand if total_demand > 0 else 0
    pct = pnl / total_spot * 100 if total_spot else 0

    cum_cost = (merged_c["deal_cost_eff"] + merged_c["spot_cost"]).cumsum().values

    return dict(
        name=f"📝 Echte Deals ({deal_pct:.0f}%T / {100-deal_pct:.0f}%S)",
        row={
            "Strategie": f"📝 Echte Deals ({deal_pct:.0f}%T / {100-deal_pct:.0f}%S)",
            "Muster": "Deals",
            "Terminanteil [%]": deal_pct,
            "Spotanteil [%]": 100 - deal_pct,
            "Ø Forward [€/MWh]": avg_deal_price,
            "Terminvol. [MWh]": deal_vol,
            "Terminkosten [€]": deal_cost_total,
            "Spotvol. [MWh]": total_demand - deal_vol,
            "Ø Spot [€/MWh]": spot_cost_total / (total_demand - deal_vol) if (total_demand - deal_vol) > 0 else 0,
            "Spotkosten [€]": spot_cost_total,
            "TX [€]": 0,
            "Gesamt [€]": total_cost,
            "Ø Preis [€/MWh]": avg_price,
            "PnL vs Spot [€]": pnl,
            "Ersparnis [%]": pct,
        },
        cum=cum_cost,
    )


def calc_schedule_strategy(schedule_df, fwd_df, merged, total_demand, total_spot, avg_spot, tx_cost=0):
    """Berechnet Kosten des Beschaffungsplans als Strategie."""
    if schedule_df is None or schedule_df.empty:
        return None

    delivery_start = merged["datetime"].min()
    fwd_before = fwd_df[fwd_df["datetime"] < delivery_start] if fwd_df is not None else pd.DataFrame()

    valid = schedule_df.dropna(subset=["datum"])
    if valid.empty:
        return None

    # Preise ermitteln
    prices = []
    dates_used = []
    weights = []
    for _, row in valid.iterrows():
        p = row.get("preis")
        if pd.isna(p) and not fwd_before.empty:
            diffs = abs(fwd_before["datetime"] - row["datum"])
            nearest_idx = diffs.idxmin()
            if diffs.loc[nearest_idx] <= pd.Timedelta(days=7):
                p = fwd_before.loc[nearest_idx, "forward_price"]
        if pd.isna(p):
            continue

        if "anteil_pct" in row and pd.notna(row["anteil_pct"]):
            w = row["anteil_pct"] / 100
        elif "menge_mwh" in row and pd.notna(row["menge_mwh"]):
            w = row["menge_mwh"] / total_demand if total_demand > 0 else 0
        else:
            w = 1.0 / len(valid)

        prices.append(p)
        dates_used.append(row["datum"])
        weights.append(w)

    if not prices:
        return None

    weights = np.array(weights)
    total_share = weights.sum()
    if total_share > 1:
        weights = weights / total_share
        total_share = 1.0

    prices = np.array(prices)
    wavg = float(np.dot(weights / weights.sum(), prices))

    fwd_share = min(total_share, 1.0)
    spot_share = 1.0 - fwd_share

    fwd_vol = total_demand * fwd_share
    fwd_cost = fwd_vol * wavg
    tx = fwd_vol * tx_cost
    spot_cost = float((merged["load_mwh"] * spot_share * merged["spot_price"]).sum())
    total_cost = fwd_cost + spot_cost + tx
    avg_price = total_cost / total_demand if total_demand > 0 else 0
    pnl = total_spot - total_cost
    pct = pnl / total_spot * 100 if total_spot else 0

    period_cost = (
        merged["load_mwh"] * fwd_share * (wavg + tx_cost)
        + merged["load_mwh"] * spot_share * merged["spot_price"]
    )
    cum = period_cost.cumsum().values

    return dict(
        name=f"📅 Plan ({fwd_share*100:.0f}%T / {spot_share*100:.0f}%S)",
        row={
            "Strategie": f"📅 Plan ({fwd_share*100:.0f}%T / {spot_share*100:.0f}%S)",
            "Muster": "Plan",
            "Terminanteil [%]": fwd_share * 100,
            "Spotanteil [%]": spot_share * 100,
            "Ø Forward [€/MWh]": wavg,
            "Terminvol. [MWh]": fwd_vol,
            "Terminkosten [€]": fwd_cost,
            "Spotvol. [MWh]": total_demand * spot_share,
            "Ø Spot [€/MWh]": avg_spot,
            "Spotkosten [€]": spot_cost,
            "TX [€]": tx,
            "Gesamt [€]": total_cost,
            "Ø Preis [€/MWh]": avg_price,
            "PnL vs Spot [€]": pnl,
            "Ersparnis [%]": pct,
        },
        cum=cum,
        schedule_detail=pd.DataFrame({
            "Datum": dates_used,
            "Preis [€/MWh]": prices,
            "Gewicht": weights,
        }),
    )


def run_backtest(load_df, spot_df, fwd_df, cfg, deals_df=None, schedule_df=None, progress=None):
    """Haupt-Backtest inkl. Deals und Beschaffungsplan."""
    delivery_start = load_df.datetime.min()

    fwd_before = fwd_df[fwd_df.datetime < delivery_start].copy()
    if fwd_before.empty:
        raise ValueError("Keine Forward-Daten VOR Lieferbeginn!")

    merged, mode = merge_load_spot(load_df, spot_df)
    if merged.empty:
        raise ValueError("Kein zeitlicher Überlapp Last ↔ Spot!")

    merged["cost_spot"] = merged.load_mwh * merged.spot_price
    total_demand = float(merged.load_mwh.sum())
    total_spot = float(merged.cost_spot.sum())
    avg_spot = total_spot / total_demand if total_demand else 0

    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged["cum_spot"] = merged.cost_spot.cumsum()

    fwd_vals = fwd_before.forward_price.values
    n_fwd = len(fwd_before)

    all_patterns = cfg["patterns"]
    all_shares = cfg["forward_shares"]
    n_total = len(all_patterns) * len(all_shares) + (1 if deals_df is not None else 0) + (1 if schedule_df is not None else 0)

    rows, cum_data = [], {}
    done = 0

    # ── Simulierte Strategien ──
    for pat in all_patterns:
        n_t = min(cfg["n_tranches"], n_fwd)
        idx = np.round(np.linspace(0, n_fwd - 1, n_t)).astype(int)
        w = compute_weights(pat, n_t)
        wavg = float(np.dot(w, fwd_vals[idx]))
        pn = pat.split("(")[0].strip()

        for fs in all_shares:
            ss = 1.0 - fs
            fv = total_demand * fs
            fc = fv * wavg
            tx = fv * cfg["tx_cost"]
            sc = float((merged.load_mwh * ss * merged.spot_price).sum())
            tc = fc + sc + tx
            ap = tc / total_demand if total_demand else 0
            pnl = total_spot - tc
            pct = pnl / total_spot * 100 if total_spot else 0
            name = f"{pn} ({fs*100:.0f}%T / {ss*100:.0f}%S)"

            rows.append({
                "Strategie": name, "Muster": pn,
                "Terminanteil [%]": fs * 100, "Spotanteil [%]": ss * 100,
                "Ø Forward [€/MWh]": wavg,
                "Terminvol. [MWh]": fv, "Terminkosten [€]": fc,
                "Spotvol. [MWh]": total_demand * ss, "Ø Spot [€/MWh]": avg_spot,
                "Spotkosten [€]": sc, "TX [€]": tx,
                "Gesamt [€]": tc, "Ø Preis [€/MWh]": ap,
                "PnL vs Spot [€]": pnl, "Ersparnis [%]": pct,
            })

            pc = merged.load_mwh * fs * (wavg + cfg["tx_cost"]) + merged.load_mwh * ss * merged.spot_price
            cum_data[name] = pc.cumsum().values

            done += 1
            if progress:
                progress.progress(done / n_total, f"{done}/{n_total}")

    # ── Deal-Strategie ──
    deal_result = None
    if deals_df is not None and not deals_df.empty:
        deal_result = calc_deal_strategy(deals_df, merged, total_demand, total_spot)
        if deal_result:
            rows.append(deal_result["row"])
            cum_data[deal_result["name"]] = deal_result["cum"]
        done += 1
        if progress:
            progress.progress(done / n_total, "Deals …")

    # ── Beschaffungsplan-Strategie ──
    sched_result = None
    if schedule_df is not None and not schedule_df.empty:
        sched_result = calc_schedule_strategy(
            schedule_df, fwd_df, merged, total_demand, total_spot, avg_spot, cfg["tx_cost"]
        )
        if sched_result:
            rows.append(sched_result["row"])
            cum_data[sched_result["name"]] = sched_result["cum"]
        done += 1
        if progress:
            progress.progress(done / n_total, "Plan …")

    rdf = pd.DataFrame(rows).sort_values("Gesamt [€]").reset_index(drop=True)
    rdf.index += 1

    return dict(
        results=rdf, merged=merged, total_spot=total_spot,
        demand=total_demand, avg_spot=avg_spot, cum=cum_data,
        fwd_before=fwd_before, merge_mode=mode,
        deal_result=deal_result, sched_result=sched_result,
    )


# ══════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════

def _pcol(vals):
    return [C["pos"] if v > 0 else C["neg"] if v < 0 else C["neut"] for v in vals]


def fig_costs(df, ref):
    fig = go.Figure(go.Bar(
        x=df["Strategie"], y=df["Gesamt [€]"],
        marker_color=_pcol(df["PnL vs Spot [€]"]),
        text=df["Gesamt [€]"].apply(lambda v: f"{v:,.0f}€"),
        textposition="outside", textfont_size=9,
    ))
    fig.add_hline(y=ref, line_dash="dash", line_color="red",
                  annotation_text=f"100% Spot: {ref:,.0f}€")
    fig.update_layout(title="Gesamtkosten", xaxis_tickangle=-45, height=500, template=TPL)
    return fig


def fig_savings(df):
    fig = go.Figure(go.Bar(
        x=df["Strategie"], y=df["PnL vs Spot [€]"],
        marker_color=_pcol(df["PnL vs Spot [€]"]),
        text=df["PnL vs Spot [€]"].apply(lambda v: f"{v:+,.0f}€"),
        textposition="outside", textfont_size=9,
    ))
    fig.add_hline(y=0, line_color="white")
    fig.update_layout(title="Ersparnis vs. Spot", xaxis_tickangle=-45, height=450, template=TPL)
    return fig


def fig_cum(merged, cum, keys):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.datetime, y=merged.cum_spot,
                             mode="lines", name="100% Spot",
                             line=dict(color="red", width=3, dash="dash")))
    for i, k in enumerate(keys):
        if k in cum:
            fig.add_trace(go.Scatter(x=merged.datetime, y=cum[k],
                                     mode="lines", name=k,
                                     line=dict(color=PAL[i % len(PAL)], width=2)))
    fig.update_layout(title="Kumulative Kosten", height=500, template=TPL,
                      legend=dict(font_size=9))
    return fig


def fig_fwd(fwd, n_t):
    n = len(fwd)
    idx = np.round(np.linspace(0, n - 1, min(n_t, n))).astype(int)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fwd.datetime, y=fwd.forward_price,
                             mode="lines", name="Forward",
                             line=dict(color=C["blue"], width=2),
                             fill="tozeroy", fillcolor="rgba(52,152,219,0.08)"))
    fig.add_trace(go.Scatter(x=fwd.iloc[idx].datetime, y=fwd.iloc[idx].forward_price,
                             mode="markers", name="Sim.-Kaufpunkte",
                             marker=dict(color=C["neg"], size=10, symbol="triangle-up")))
    avg = fwd.forward_price.mean()
    fig.add_hline(y=avg, line_dash="dot", line_color="orange",
                  annotation_text=f"Ø {avg:.2f}")
    fig.update_layout(title="Forward-Preise", height=400, template=TPL)
    return fig


def fig_sensitivity(df, pat, ref):
    sub = df[df["Muster"] == pat].sort_values("Terminanteil [%]")
    if len(sub) < 2:
        return None
    best = sub.loc[sub["Gesamt [€]"].idxmin()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["Terminanteil [%]"], y=sub["Gesamt [€]"],
                             mode="lines+markers", name="Kosten",
                             line=dict(color=C["pos"], width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=[best["Terminanteil [%]"]], y=[best["Gesamt [€]"]],
                             mode="markers", name=f"Optimum ({best['Terminanteil [%]']:.0f}%)",
                             marker=dict(color="gold", size=14, symbol="star")))
    fig.add_hline(y=ref, line_dash="dash", line_color="red", annotation_text="100% Spot")
    fig.update_layout(title=f"Sensitivität – {pat}", height=380, template=TPL,
                      xaxis_title="Terminanteil [%]", yaxis_title="€")
    return fig


# ══════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════

def data_input(title, key, state_key, val_label, val_name, placeholder, unit=""):
    with st.container(border=True):
        st.markdown(f"**{title}**")

        if st.session_state[state_key] is not None:
            df = st.session_state[state_key]
            c1, c2 = st.columns([5, 1])
            c1.success(
                f"✅ **{len(df)}** Eintr. · "
                f"{df.datetime.min().date()} → {df.datetime.max().date()} · "
                f"Ø {df[val_name].mean():.1f} {unit}"
            )
            if c2.button("🗑️", key=f"{key}_del", help="Löschen"):
                st.session_state[state_key] = None
                st.session_state.pop(f"{key}_raw", None)
                st.rerun()
            return

        tab_p, tab_f = st.tabs(["📋 Einfügen (Strg+V)", "📁 Datei"])

        with tab_p:
            txt = st.text_area("Daten einfügen", height=160, key=f"{key}_txt",
                               placeholder=placeholder)
            if st.button("🔄 Verarbeiten", key=f"{key}_go", type="primary",
                         use_container_width=True, disabled=not txt):
                raw = parse_text(txt)
                if raw is not None and len(raw.columns) >= 2:
                    st.session_state[f"{key}_raw"] = raw
                    st.rerun()
                else:
                    st.error("❌ Mind. 2 Spalten nötig (Datum + Wert)")

        with tab_f:
            f = st.file_uploader("Datei", ["csv", "xlsx", "xls"], key=f"{key}_f",
                                 label_visibility="collapsed")
            if f and st.button("📁 Laden", key=f"{key}_fl", use_container_width=True):
                raw = load_file(f)
                if raw is not None:
                    st.session_state[f"{key}_raw"] = raw
                    st.rerun()

        raw = st.session_state.get(f"{key}_raw")
        if raw is not None and len(raw) > 0:
            st.divider()
            st.caption(f"Vorschau ({len(raw)} × {len(raw.columns)})")
            st.dataframe(raw.head(6), use_container_width=True, height=180)

            cols = list(raw.columns)
            auto_t = find_datetime_col(raw)
            auto_v = find_value_col(raw, auto_t)

            c1, c2, c3 = st.columns([2, 2, 1])
            tc = c1.selectbox("📅 Datum", cols,
                              index=cols.index(auto_t) if auto_t in cols else 0,
                              key=f"{key}_tc")
            vc = c2.selectbox(f"📊 {val_label}", cols,
                              index=cols.index(auto_v) if auto_v in cols else min(1, len(cols) - 1),
                              key=f"{key}_vc")
            if c3.button("✅ OK", key=f"{key}_ok", type="primary", use_container_width=True):
                df = clean_data(raw, tc, vc, val_name)
                if df.empty:
                    st.error("Keine gültigen Daten.")
                else:
                    st.session_state[state_key] = df
                    st.session_state.pop(f"{key}_raw", None)
                    st.rerun()


# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════

st.sidebar.title("⚡ Energiebeschaffung")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "📥 Datenimport", "⚙️ Strategien", "🔬 Backtesting", "📊 Dashboard"
])

st.sidebar.divider()
st.sidebar.markdown("##### Status")
for sk, lbl in [("load_df", "Last"), ("spot_df", "Spot"), ("forward_df", "Forward"),
                ("deals_df", "Deals"), ("schedule_df", "Plan")]:
    d = st.session_state[sk]
    if d is not None:
        st.sidebar.success(f"✅ {lbl} ({len(d)})")
    else:
        st.sidebar.caption(f"⏳ {lbl}")

st.sidebar.divider()
sb1, sb2 = st.sidebar.columns(2)
if sb1.button("🗑️ Reset", use_container_width=True):
    for k, v in _DEF.items():
        st.session_state[k] = v.copy() if isinstance(v, (dict, list)) else v
    st.rerun()
if sb2.button("🎲 Demo", use_container_width=True):
    ld, sd, fd, dd, sch = generate_demo()
    st.session_state.update(load_df=ld, spot_df=sd, forward_df=fd, deals_df=dd, schedule_df=sch)
    st.rerun()


# ══════════════════════════════════════════════════
# PAGE: DATENIMPORT
# ══════════════════════════════════════════════════

if page == "📥 Datenimport":
    st.header("📥 Datenimport")
    st.caption("Daten per **Strg+C → Strg+V** aus Excel einfügen oder Datei hochladen.")

    data_input("1️⃣  Lastprofil (Lieferperiode)", "load", "load_df",
               "Verbrauch [MWh]", "load_mwh",
               "Datum\tMWh\n01.01.2025\t120.5\n02.01.2025\t115.3\n…", "MWh")

    data_input("2️⃣  Spotpreise (Lieferperiode)", "spot", "spot_df",
               "Preis [€/MWh]", "spot_price",
               "Datum\tEUR_MWh\n01.01.2025\t85.20\n02.01.2025\t92.10\n…", "€/MWh")

    data_input("3️⃣  Terminmarktpreise (VOR Lieferung)", "fwd", "forward_df",
               "Forward [€/MWh]", "forward_price",
               "Datum\tForward\n02.01.2024\t78.50\n03.01.2024\t79.20\n…", "€/MWh")

    if all(st.session_state[k] is not None for k in ("load_df", "spot_df", "forward_df")):
        st.success("✅ Pflichtdaten komplett — weiter zu **⚙️ Strategien**")


# ══════════════════════════════════════════════════
# PAGE: STRATEGIEN
# ══════════════════════════════════════════════════

elif page == "⚙️ Strategien":
    st.header("⚙️ Strategien & Beschaffung")

    tab_sim, tab_plan, tab_deals = st.tabs([
        "🔄 Simulation", "📅 Beschaffungsplan", "📝 Echte Deals"
    ])

    # ────── SIMULATION ──────
    with tab_sim:
        st.markdown("##### Terminanteil & Muster (automatische Szenarien)")

        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                mode = st.radio("Anteile", ["Standard 0–100%", "Eigene"], horizontal=True,
                                label_visibility="collapsed")
                if mode.startswith("Standard"):
                    shares = [i / 100 for i in range(0, 101, 10)]
                else:
                    txt = st.text_input("% kommagetrennt", "0,10,20,30,40,50,60,70,80,90,100")
                    try:
                        shares = sorted({max(0.0, min(1.0, float(x) / 100)) for x in txt.split(",")})
                    except ValueError:
                        shares = [i / 100 for i in range(0, 101, 10)]
                st.caption(f"{len(shares)} Stufen: " + ", ".join(f"{s:.0%}" for s in shares))

            with c2:
                patterns = st.multiselect("Muster", [
                    "Gleichmäßig", "Frontloaded (früh mehr)", "Backloaded (spät mehr)"
                ], default=["Gleichmäßig"], label_visibility="collapsed")
                if not patterns:
                    patterns = ["Gleichmäßig"]
                n_tr = st.slider("Tranchen", 2, 36, 6)
                tx = st.number_input("TX [€/MWh]", 0.0, step=0.05, format="%.2f")

        st.session_state.config = dict(
            forward_shares=shares, patterns=patterns, n_tranches=n_tr, tx_cost=tx)
        st.info(f"📐 **{len(shares) * len(patterns)} Szenarien** werden simuliert")

        # Vorschau
        if st.session_state.forward_df is not None:
            fwd = st.session_state.forward_df
            n_fwd = len(fwd)
            at = min(n_tr, n_fwd)
            idx = np.round(np.linspace(0, n_fwd - 1, at)).astype(int)
            for pat in patterns:
                w = compute_weights(pat, at)
                p = fwd.iloc[idx].forward_price.values
                with st.expander(f"🔍 {pat} — Ø {np.dot(w, p):.2f} €/MWh"):
                    st.dataframe(pd.DataFrame({
                        "#": range(1, at + 1),
                        "Datum": fwd.iloc[idx].datetime.dt.date.values,
                        "Preis": p, "Gewicht": [f"{x:.1%}" for x in w],
                    }), hide_index=True, use_container_width=True)

    # ────── BESCHAFFUNGSPLAN ──────
    with tab_plan:
        st.markdown("##### Eigene Kaufzeitpunkte am Terminmarkt")
        st.caption(
            "Geben Sie Datum + Anteil (%) ein. Preis wird aus der "
            "Forward-Kurve entnommen – oder Sie geben einen eigenen Preis ein."
        )

        with st.container(border=True):
            if st.session_state.schedule_df is not None:
                sched = st.session_state.schedule_df
                c1, c2 = st.columns([5, 1])
                c1.success(f"✅ **{len(sched)} Kaufzeitpunkte** hinterlegt")
                if c2.button("🗑️", key="sched_del"):
                    st.session_state.schedule_df = None
                    st.session_state.pop("sched_raw", None)
                    st.rerun()
                st.dataframe(sched, use_container_width=True, hide_index=True)
            else:
                sched_txt = st.text_area(
                    "Beschaffungsplan einfügen",
                    height=200, key="sched_txt",
                    placeholder=(
                        "Datum\tAnteil_%\tPreis_EUR_MWh\n"
                        "15.01.2024\t10\t\n"
                        "15.03.2024\t15\t79.50\n"
                        "15.05.2024\t20\t\n"
                        "15.07.2024\t25\t\n"
                        "15.09.2024\t20\t\n"
                        "15.11.2024\t10\t\n"
                        "\n(Leere Preis-Spalte = Preis aus Forward-Kurve)"
                    ),
                    help="Datum + Anteil (%) + optionaler Preis. Tab-getrennt.",
                )
                if st.button("🔄 Plan verarbeiten", key="sched_go", type="primary",
                             use_container_width=True, disabled=not sched_txt):
                    raw = parse_text(sched_txt)
                    if raw is not None:
                        fwd_for_lookup = st.session_state.forward_df
                        parsed = parse_schedule(raw, fwd_for_lookup)
                        if parsed is not None and not parsed.empty:
                            st.session_state.schedule_df = parsed
                            st.rerun()
                        else:
                            st.error("❌ Konnte Plan nicht parsen. Mind. Datum + Anteil/Menge nötig.")
                    else:
                        st.error("❌ Format nicht erkannt.")

                # Alternativ: Datei
                f = st.file_uploader("Oder Datei", ["csv", "xlsx"], key="sched_f",
                                     label_visibility="collapsed")
                if f and st.button("📁 Plan laden", key="sched_fl"):
                    raw = load_file(f)
                    if raw is not None:
                        parsed = parse_schedule(raw, st.session_state.forward_df)
                        if parsed is not None:
                            st.session_state.schedule_df = parsed
                            st.rerun()

    # ────── ECHTE DEALS ──────
    with tab_deals:
        st.markdown("##### Reale Beschaffungsgeschäfte hinterlegen")
        st.caption(
            "Produkte wie **Cal-25**, **Q1-25**, **Jan-25** werden automatisch "
            "erkannt. Leistung in MW oder Menge in MWh + Preis in €/MWh."
        )

        with st.container(border=True):
            if st.session_state.deals_df is not None:
                deals = st.session_state.deals_df
                c1, c2 = st.columns([5, 1])
                c1.success(f"✅ **{len(deals)} Deals** hinterlegt · "
                           f"Σ {deals.get('menge_mwh', pd.Series([0])).sum():,.0f} MWh")
                if c2.button("🗑️", key="deals_del"):
                    st.session_state.deals_df = None
                    st.session_state.pop("deals_raw", None)
                    st.rerun()

                display_cols = [c for c in [
                    "produkt", "kaufdatum", "lieferstart", "lieferende",
                    "leistung_mw", "menge_mwh", "preis", "profil"
                ] if c in deals.columns]
                st.dataframe(deals[display_cols], use_container_width=True, hide_index=True)
            else:
                deals_txt = st.text_area(
                    "Deals einfügen",
                    height=220, key="deals_txt",
                    placeholder=(
                        "Produkt\tKaufdatum\tLeistung_MW\tPreis_EUR_MWh\n"
                        "Cal-25 Base\t15.03.2024\t10\t82.50\n"
                        "Q1-25 Base\t01.06.2024\t5\t85.00\n"
                        "Q3-25 Peak\t15.07.2024\t3\t95.20\n"
                        "Jul-25 Base\t01.09.2024\t8\t72.30\n"
                        "\n"
                        "ODER mit explizitem Zeitraum:\n"
                        "Produkt\tLieferstart\tLieferende\tLeistung_MW\tPreis\n"
                        "Baseload\t01.01.2025\t31.12.2025\t10\t82.50"
                    ),
                    help=(
                        "Erkannte Produkte: Cal-25, Q1-25, H1-25, Jan-25, etc.\n"
                        "Leistung in MW → wird automatisch in MWh umgerechnet.\n"
                        "Oder: Lieferstart + Lieferende explizit angeben."
                    ),
                )

                if st.button("🔄 Deals verarbeiten", key="deals_go", type="primary",
                             use_container_width=True, disabled=not deals_txt):
                    raw = parse_text(deals_txt)
                    if raw is not None:
                        parsed = parse_deals(raw)
                        if parsed is not None and not parsed.empty:
                            # Prüfe ob Lieferzeiträume erkannt wurden
                            n_ok = parsed["lieferstart"].notna().sum()
                            if n_ok == 0:
                                st.error(
                                    "❌ Kein Lieferzeitraum erkannt. "
                                    "Verwenden Sie Produktnamen wie Cal-25, Q1-25, Jan-25 "
                                    "oder geben Sie Lieferstart + Lieferende als Spalten an."
                                )
                            else:
                                st.session_state.deals_df = parsed
                                st.rerun()
                        else:
                            st.error("❌ Konnte keine Deals erkennen. Mind. Produkt + Preis nötig.")
                    else:
                        st.error("❌ Format nicht erkannt.")

                f = st.file_uploader("Oder Datei", ["csv", "xlsx"], key="deals_f",
                                     label_visibility="collapsed")
                if f and st.button("📁 Deals laden", key="deals_fl"):
                    raw = load_file(f)
                    if raw is not None:
                        parsed = parse_deals(raw)
                        if parsed is not None:
                            st.session_state.deals_df = parsed
                            st.rerun()

        # Produkt-Parser Hilfe
        with st.expander("🔍 Unterstützte Produktformate"):
            st.markdown("""
| Eingabe | Erkannter Zeitraum |
|---------|-------------------|
| `Cal-25`, `Year-25`, `Y25` | 01.01.2025 – 31.12.2025 |
| `Q1-25`, `Q2-25` … `Q4-25` | Quartal 2025 |
| `H1-25`, `H2-25` | Halbjahr 2025 |
| `Jan-25`, `Feb-25` … `Dez-25` | Monat 2025 |
| `M01-25` … `M12-25` | Monat 2025 |

**Zusätzlich erkannte Spalten:** Kaufdatum, Lieferstart, Lieferende, Leistung (MW),
Menge (MWh), Preis (€/MWh), Profil (Base/Peak)
            """)


# ══════════════════════════════════════════════════
# PAGE: BACKTESTING
# ══════════════════════════════════════════════════

elif page == "🔬 Backtesting":
    st.header("🔬 Backtesting")

    missing = [l for k, l in [("load_df", "Last"), ("spot_df", "Spot"), ("forward_df", "Forward")]
               if st.session_state[k] is None]
    if missing:
        st.warning(f"⚠️ Fehlend: **{', '.join(missing)}** → 📥 Datenimport")
        st.stop()

    cfg = st.session_state.config
    ld = st.session_state.load_df
    sd = st.session_state.spot_df
    fd = st.session_state.forward_df
    dd = st.session_state.deals_df
    sch = st.session_state.schedule_df

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Last", f"{ld.load_mwh.sum():,.0f} MWh",
                  f"{ld.datetime.min().date()} → {ld.datetime.max().date()}")
        c2.metric("💰 Spot", f"Ø {sd.spot_price.mean():.2f} €/MWh", f"{len(sd)} Eintr.")
        c3.metric("📈 Forward", f"Ø {fd.forward_price.mean():.2f} €/MWh", f"{len(fd)} Tage")

    extras = []
    if dd is not None:
        extras.append(f"📝 {len(dd)} Deals")
    if sch is not None:
        extras.append(f"📅 {len(sch)} Planpunkte")
    n_sim = len(cfg["forward_shares"]) * len(cfg["patterns"])
    n_total = n_sim + (1 if dd is not None else 0) + (1 if sch is not None else 0)

    st.info(f"**{n_total} Strategien:** {n_sim} simuliert" +
            (f" + {', '.join(extras)}" if extras else ""))

    if st.button("🚀 Backtesting starten", type="primary", use_container_width=True):
        prog = st.progress(0, "Starte …")
        try:
            bt = run_backtest(ld, sd, fd, cfg, dd, sch, prog)
        except ValueError as e:
            prog.empty()
            st.error(f"❌ {e}")
            st.stop()
        prog.empty()

        st.session_state.update(
            bt_results=bt["results"], bt_merged=bt["merged"],
            bt_spot_total=bt["total_spot"], bt_demand_total=bt["demand"],
            bt_avg_spot=bt["avg_spot"], bt_cumulative=bt["cum"],
            bt_fwd_before=bt["fwd_before"],
        )
        st.success(f"✅ **{len(bt['results'])} Szenarien** berechnet (Merge: {bt['merge_mode']})")
        st.rerun()

    # ── RESULTS ──
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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Beste", best["Strategie"].split("(")[-1].rstrip(")"),
                  f"{best['PnL vs Spot [€]']:+,.0f} €")
        c2.metric("Kosten", f"{best['Gesamt [€]']:,.0f} €", f"{best['Ersparnis [%]']:+.1f}%")
        c3.metric("100% Spot", f"{TS:,.0f} €", f"Ø {AS:.2f} €/MWh")
        c4.metric("Bedarf", f"{TD:,.0f} MWh", f"{len(M)} Per.")

        # Highlight Deals & Plan
        special = R[R["Muster"].isin(["Deals", "Plan"])]
        if not special.empty:
            st.divider()
            st.subheader("⭐ Eigene Strategien")
            st.dataframe(
                special[["Strategie", "Gesamt [€]", "Ø Preis [€/MWh]",
                         "PnL vs Spot [€]", "Ersparnis [%]"]].style.format({
                    "Gesamt [€]": "{:,.0f}", "Ø Preis [€/MWh]": "{:.2f}",
                    "PnL vs Spot [€]": "{:+,.0f}", "Ersparnis [%]": "{:+.1f}%",
                }),
                use_container_width=True, hide_index=True,
            )

        # Top 5
        st.divider()
        st.subheader("🏆 Top 5 gesamt")
        st.dataframe(
            R.head(5)[["Strategie", "Gesamt [€]", "Ø Preis [€/MWh]",
                        "PnL vs Spot [€]", "Ersparnis [%]"]].style.format({
                "Gesamt [€]": "{:,.0f}", "Ø Preis [€/MWh]": "{:.2f}",
                "PnL vs Spot [€]": "{:+,.0f}", "Ersparnis [%]": "{:+.1f}%",
            }),
            use_container_width=True, hide_index=True,
        )

        with st.expander("📋 Alle Ergebnisse"):
            st.dataframe(R, use_container_width=True)

        # Charts
        st.divider()
        max_show = st.slider("Max. Strategien", 5, len(R), min(15, len(R)))
        show = R.head(max_show)

        tabs = st.tabs(["💰 Kosten", "📊 Ersparnis", "📈 Kumulativ",
                        "🔵 Forward", "🎯 Sensitivität"])

        with tabs[0]:
            st.plotly_chart(fig_costs(show, TS), use_container_width=True)
        with tabs[1]:
            st.plotly_chart(fig_savings(show), use_container_width=True)
        with tabs[2]:
            default_keys = [k for k in list(CUM.keys())[:3] if k.startswith("📝") or k.startswith("📅")]
            if not default_keys:
                default_keys = list(CUM.keys())[:5]
            sel = st.multiselect("Strategien", list(CUM.keys()), default=default_keys)
            if sel:
                st.plotly_chart(fig_cum(M, CUM, sel), use_container_width=True)
        with tabs[3]:
            if FB is not None:
                f5 = fig_fwd(FB, cfg["n_tranches"])
                # Beschaffungsplan-Punkte einzeichnen
                if sch is not None and st.session_state.schedule_df is not None:
                    sc = st.session_state.schedule_df
                    valid = sc.dropna(subset=["datum"])
                    if not valid.empty and "preis" in valid.columns:
                        f5.add_trace(go.Scatter(
                            x=valid["datum"], y=valid["preis"],
                            mode="markers", name="📅 Plan-Kaufpunkte",
                            marker=dict(color=C["purple"], size=12, symbol="diamond"),
                        ))
                # Deal-Kaufdaten einzeichnen
                if dd is not None and "kaufdatum" in dd.columns:
                    kd = dd.dropna(subset=["kaufdatum"])
                    if not kd.empty:
                        # Nächsten Forward-Preis für y-Achse suchen
                        y_vals = []
                        for _, row in kd.iterrows():
                            diffs = abs(FB.datetime - row["kaufdatum"])
                            nearest = diffs.idxmin()
                            y_vals.append(FB.loc[nearest, "forward_price"])
                        f5.add_trace(go.Scatter(
                            x=kd["kaufdatum"], y=y_vals,
                            mode="markers+text", name="📝 Deal-Kaufdaten",
                            marker=dict(color=C["gold"], size=12, symbol="star"),
                            text=kd["produkt"], textposition="top center",
                            textfont=dict(size=9),
                        ))
                st.plotly_chart(f5, use_container_width=True)
        with tabs[4]:
            for pat in cfg["patterns"]:
                pn = pat.split("(")[0].strip()
                f = fig_sensitivity(R, pn, TS)
                if f:
                    st.plotly_chart(f, use_container_width=True)


# ══════════════════════════════════════════════════
# PAGE: DASHBOARD
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
        c1.metric("🏆 Optimum", f"{best['Terminanteil [%]']:.0f}%T / {best['Spotanteil [%]']:.0f}%S")
        c2.metric("Kosten", f"{best['Gesamt [€]']:,.0f} €",
                  f"{best['PnL vs Spot [€]']:+,.0f} € vs Spot")
        c3.metric("Schlechteste", f"{worst['Terminanteil [%]']:.0f}%T / {worst['Spotanteil [%]']:.0f}%S",
                  f"{worst['PnL vs Spot [€]']:+,.0f} €")
        c4.metric("Spanne", f"{worst['Gesamt [€]'] - best['Gesamt [€]']:,.0f} €")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bedarf", f"{TD:,.0f} MWh")
        c2.metric("100% Spot", f"{TS:,.0f} €")
        c3.metric("Ø Spot", f"{AS:.2f} €/MWh")
        if FB is not None:
            af = FB.forward_price.mean()
            c4.metric("Ø Forward", f"{af:.2f} €/MWh", f"{af - AS:+.2f} vs Spot")

    # ── Deals & Plan Vergleich ──
    special = R[R["Muster"].isin(["Deals", "Plan"])]
    if not special.empty:
        st.divider()
        st.subheader("⭐ Ihre Strategien im Vergleich")
        for _, row in special.iterrows():
            rank = R.index[R["Strategie"] == row["Strategie"]].tolist()
            rank_pos = rank[0] if rank else "?"
            icon = "🟢" if row["PnL vs Spot [€]"] > 0 else "🔴"
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"{icon} {row['Strategie'].split('(')[0].strip()}",
                          f"{row['Gesamt [€]']:,.0f} €")
                c2.metric("Ersparnis", f"{row['PnL vs Spot [€]']:+,.0f} €",
                          f"{row['Ersparnis [%]']:+.1f}%")
                c3.metric("Ø Preis", f"{row['Ø Preis [€/MWh]']:.2f} €/MWh")
                c4.metric("Ranking", f"Platz {rank_pos}", f"von {len(R)}")

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
            c4.metric("VaR 95%", f"{np.percentile(pnl, 5):+,.0f} €")

    st.divider()

    # Empfehlung
    st.subheader("💡 Empfehlung")
    if best["PnL vs Spot [€]"] > 0:
        st.success(
            f"**Optimale Strategie: {best['Strategie']}**\n\n"
            f"- Kosten: **{best['Gesamt [€]']:,.0f} €** (statt {TS:,.0f} €)\n"
            f"- Ersparnis: **{best['PnL vs Spot [€]']:+,.0f} €** ({best['Ersparnis [%]']:+.1f}%)\n"
            f"- Ø Preis: **{best['Ø Preis [€/MWh]']:.2f} €/MWh** (statt {AS:.2f})"
        )
    else:
        st.info(
            f"**100% Spot wäre am günstigsten gewesen.**\n\n"
            f"- Spot: **{TS:,.0f} €** (Ø {AS:.2f} €/MWh)\n"
            f"- Nächstbeste: {best['Strategie']} → {best['Gesamt [€]']:,.0f} €"
        )

    # Mustervergleich
    all_muster = R["Muster"].unique()
    if len(all_muster) > 1:
        st.divider()
        st.subheader("🗺️ Vergleich nach Typ")
        for m in all_muster:
            sub = R[R["Muster"] == m].sort_values("Terminanteil [%]")
            if not sub.empty:
                with st.expander(f"**{m}** ({len(sub)} Szenarien)"):
                    st.dataframe(
                        sub[["Terminanteil [%]", "Gesamt [€]", "Ø Preis [€/MWh]",
                             "PnL vs Spot [€]", "Ersparnis [%]"]].style.format({
                            "Gesamt [€]": "{:,.0f}", "Ø Preis [€/MWh]": "{:.2f}",
                            "PnL vs Spot [€]": "{:+,.0f}", "Ersparnis [%]": "{:+.1f}%",
                        }),
                        use_container_width=True, hide_index=True,
                    )

    # Export
    st.divider()
    st.download_button(
        "📥 Ergebnisse als CSV",
        R.to_csv(index=False, sep=";", decimal=",").encode("utf-8"),
        "ergebnisse.csv", "text/csv", use_container_width=True,
    )

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import re
import calendar
from datetime import date
from typing import Optional, Dict, List, Tuple

# ═══════════════════════════════════════════════
st.set_page_config(page_title="⚡ Energiebeschaffung", layout="wide", page_icon="⚡")

C = dict(pos="#2ecc71", neg="#e74c3c", neut="#95a5a6", blue="#3498db",
         orange="#e67e22", purple="#9b59b6", gold="#f1c40f", teal="#1abc9c")
PAL = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#1abc9c",
       "#f39c12", "#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]
TPL = "plotly_dark"

st.markdown("""<style>
.block-container{padding-top:1rem}
div[data-testid="stMetricValue"]{font-size:1.15rem;font-weight:700}
textarea{font-family:'Courier New',monospace!important;font-size:12px!important}
.stTabs [data-baseweb="tab"]{padding:8px 14px;font-weight:600}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════

_DEF = dict(
    load_df=None, spot_df=None,
    forward_curves=None,   # Dict[produkt_name -> DataFrame(datetime, price)]
    deals_df=None,
    bt=None,
    config=dict(
        sim_products=[],     # Liste der Produkte für Simulation
        sim_shares=[i/10 for i in range(0, 11)],  # Skalierungsfaktoren 0-1
        dca_freq="Täglich", dca_window_months=0,
        tx_cost=0.0,
    ),
)
for _k, _v in _DEF.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v.copy() if isinstance(_v, (dict, list)) else _v


# ═══════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════

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


def find_datetime_col(df):
    for col in df.columns:
        if any(k in col.lower() for k in ("date", "datum", "zeit", "time", "tag")):
            try:
                if pd.to_datetime(df[col], dayfirst=True, errors="coerce").notna().mean() > 0.5:
                    return col
            except Exception:
                pass
    for col in df.columns:
        try:
            if pd.to_datetime(df[col], dayfirst=True, errors="coerce").notna().mean() > 0.5:
                return col
        except Exception:
            pass
    return None


def find_value_col(df, exclude=None):
    exc = {exclude} if exclude else set()
    for col in df.columns:
        if col in exc:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
        try:
            v = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
            if v.notna().mean() > 0.3:
                return col
        except Exception:
            continue
    return None


def clean_ts(df, tcol, vcol, vname):
    out = df[[tcol, vcol]].copy()
    out[tcol] = pd.to_datetime(out[tcol], dayfirst=True, errors="coerce")
    if out[vcol].dtype == object:
        out[vcol] = out[vcol].astype(str).str.replace(",", ".").str.strip()
    out[vcol] = pd.to_numeric(out[vcol], errors="coerce")
    out = out.dropna().rename(columns={tcol: "datetime", vcol: vname})
    out = out.groupby("datetime", as_index=False).agg({vname: "mean"})
    return out.sort_values("datetime").reset_index(drop=True)


# ═══════════════════════════════════════════════
# PRODUCT PARSER
# ═══════════════════════════════════════════════

MONTHS_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "mär": 3, "mrz": 3,
    "apr": 4, "mai": 5, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "okt": 10, "oct": 10,
    "nov": 11, "dez": 12, "dec": 12,
}


def parse_product_period(name: str) -> Tuple[Optional[date], Optional[date], str]:
    """Returns (start, end, product_type).
    product_type: 'cal', 'half', 'quarter', 'month'
    """
    if not name:
        return None, None, ""
    s = name.strip().upper().replace("/", "-").replace(" ", "-")
    
    def _yr(y):
        y = int(y)
        return y + 2000 if y < 100 else y

    m = re.search(r"(?:CAL|YEAR|Y|JA)-?(\d{2,4})", s)
    if m:
        yr = _yr(m.group(1))
        return date(yr, 1, 1), date(yr, 12, 31), "cal"
    m = re.search(r"Q([1-4])-?(\d{2,4})", s)
    if m:
        q, yr = int(m.group(1)), _yr(m.group(2))
        sm = (q - 1) * 3 + 1
        em = q * 3
        return date(yr, sm, 1), date(yr, em, calendar.monthrange(yr, em)[1]), "quarter"
    m = re.search(r"H([12])-?(\d{2,4})", s)
    if m:
        h, yr = int(m.group(1)), _yr(m.group(2))
        if h == 1:
            return date(yr, 1, 1), date(yr, 6, 30), "half"
        return date(yr, 7, 1), date(yr, 12, 31), "half"
    low = name.strip().lower()
    for mname, mnum in MONTHS_MAP.items():
        m2 = re.search(rf"{mname}\w*[- /]?(\d{{2,4}})", low)
        if m2:
            yr = _yr(m2.group(1))
            return date(yr, mnum, 1), date(yr, mnum, calendar.monthrange(yr, mnum)[1]), "month"
    m = re.search(r"M(\d{1,2})-?(\d{2,4})", s)
    if m:
        mn, yr = int(m.group(1)), _yr(m.group(2))
        if 1 <= mn <= 12:
            return date(yr, mn, 1), date(yr, mn, calendar.monthrange(yr, mn)[1]), "month"
    return None, None, ""


def detect_profile(name: str) -> str:
    """Base oder Peak aus Produktname."""
    low = name.lower()
    if "peak" in low:
        return "Peak"
    return "Base"


def is_peak_hour(dt) -> bool:
    """Mo-Fr 8-20 Uhr = Peak."""
    if dt.weekday() >= 5:
        return False
    return 8 <= dt.hour < 20


# ═══════════════════════════════════════════════
# DEAL PARSER – KORREKT MIT LIEFERZEITRAUM + LEISTUNG
# ═══════════════════════════════════════════════

def parse_deals(raw_df) -> Optional[pd.DataFrame]:
    """Parst Deal-Tabelle.
    Erkennt: Produkt → Lieferzeitraum, Leistung (MW), Preis, Profil.
    """
    if raw_df is None or raw_df.empty:
        return None
    
    cols = {c: c.lower().replace("_", " ").replace("-", " ") for c in raw_df.columns}
    r = pd.DataFrame()
    
    # Produkt
    pc = next((o for o, l in cols.items()
               if any(k in l for k in ("produkt", "product", "name", "kontrakt"))), None)
    r["produkt"] = raw_df[pc].astype(str).str.strip() if pc else raw_df.iloc[:, 0].astype(str).str.strip()
    
    # Lieferzeitraum aus Produkt oder explizit
    starts, ends, ptypes = [], [], []
    for p in r["produkt"]:
        s, e, pt = parse_product_period(p)
        starts.append(s)
        ends.append(e)
        ptypes.append(pt)
    
    sc = next((o for o, l in cols.items()
               if any(k in l for k in ("lieferstart", "start", "begin", "von", "from"))), None)
    ec = next((o for o, l in cols.items()
               if any(k in l for k in ("lieferende", "ende", "end", "bis", "to"))), None)
    r["lieferstart"] = pd.to_datetime(raw_df[sc], dayfirst=True, errors="coerce") if sc \
        else pd.Series(starts).apply(lambda x: pd.Timestamp(x) if x else pd.NaT)
    r["lieferende"] = pd.to_datetime(raw_df[ec], dayfirst=True, errors="coerce") if ec \
        else pd.Series(ends).apply(lambda x: pd.Timestamp(x) if x else pd.NaT)
    r["produkttyp"] = ptypes
    
    # Kaufdatum
    kd = next((o for o, l in cols.items()
               if any(k in l for k in ("kaufdatum", "kauf", "buy", "trade", "handel"))), None)
    if kd:
        r["kaufdatum"] = pd.to_datetime(raw_df[kd], dayfirst=True, errors="coerce")
    
    # Leistung (MW)
    mw_c = next((o for o, l in cols.items()
                 if any(k in l for k in ("leistung", "power", "kapaz")) or l.strip() == "mw"), None)
    if mw_c:
        v = raw_df[mw_c]
        if v.dtype == object:
            v = v.astype(str).str.replace(",", ".")
        r["leistung_mw"] = pd.to_numeric(v, errors="coerce")
    
    # Preis
    p_c = next((o for o, l in cols.items()
                if any(k in l for k in ("preis", "price", "eur", "€"))), None)
    if p_c:
        v = raw_df[p_c]
        if v.dtype == object:
            v = v.astype(str).str.replace(",", ".").str.replace("€", "").str.strip()
        r["preis"] = pd.to_numeric(v, errors="coerce")
    else:
        return None
    
    # Profil
    pf = next((o for o, l in cols.items()
               if any(k in l for k in ("profil", "profile", "typ", "type"))), None)
    if pf:
        r["profil"] = raw_df[pf].astype(str).str.strip()
    else:
        r["profil"] = r["produkt"].apply(detect_profile)
    
    return r.dropna(subset=["preis"]).reset_index(drop=True)


# ═══════════════════════════════════════════════
# FORWARD-KURVEN PARSER
# ═══════════════════════════════════════════════

def parse_forward_curves(raw_df) -> Optional[Dict[str, pd.DataFrame]]:
    """Parst Forward-Preise mit Produktzuordnung.
    
    Erwartet entweder:
    A) Mehrere Spalten: Datum | Cal-25 | Q1-25 | Q2-25 | ...
    B) Langes Format: Datum | Produkt | Preis
    """
    if raw_df is None or raw_df.empty:
        return None
    
    dt_col = find_datetime_col(raw_df)
    if dt_col is None:
        return None
    
    dates = pd.to_datetime(raw_df[dt_col], dayfirst=True, errors="coerce")
    other_cols = [c for c in raw_df.columns if c != dt_col]
    
    # Check: Is there a "Produkt" column? → Long format
    prod_col = None
    price_col = None
    for c in other_cols:
        low = c.lower().replace("_", " ")
        if any(k in low for k in ("produkt", "product", "kontrakt", "contract")):
            prod_col = c
        if any(k in low for k in ("preis", "price", "eur", "€", "close", "settle")):
            price_col = c
    
    curves = {}
    
    if prod_col and price_col:
        # Long format
        for prod_name, grp in raw_df.groupby(prod_col):
            name = str(prod_name).strip()
            sub = pd.DataFrame({
                "datetime": pd.to_datetime(grp[dt_col], dayfirst=True, errors="coerce"),
                "price": pd.to_numeric(
                    grp[price_col].astype(str).str.replace(",", ".").str.replace("€", "").str.strip(),
                    errors="coerce"
                ),
            }).dropna().sort_values("datetime").reset_index(drop=True)
            if len(sub) > 0:
                s, e, pt = parse_product_period(name)
                sub.attrs["lieferstart"] = s
                sub.attrs["lieferende"] = e
                sub.attrs["produkttyp"] = pt
                sub.attrs["profil"] = detect_profile(name)
                curves[name] = sub
    else:
        # Wide format: each column is a product
        for col in other_cols:
            vals = raw_df[col]
            if vals.dtype == object:
                vals = pd.to_numeric(
                    vals.astype(str).str.replace(",", ".").str.replace("€", "").str.strip(),
                    errors="coerce"
                )
            else:
                vals = pd.to_numeric(vals, errors="coerce")
            
            valid_pct = vals.notna().mean()
            if valid_pct < 0.3:
                continue
            
            sub = pd.DataFrame({
                "datetime": dates, "price": vals,
            }).dropna().sort_values("datetime").reset_index(drop=True)
            
            if len(sub) > 0:
                name = str(col).strip()
                s, e, pt = parse_product_period(name)
                sub.attrs["lieferstart"] = s
                sub.attrs["lieferende"] = e
                sub.attrs["produkttyp"] = pt
                sub.attrs["profil"] = detect_profile(name)
                curves[name] = sub
    
    return curves if curves else None


# ═══════════════════════════════════════════════
# KORREKTE BEWERTUNGS-ENGINE
# ═══════════════════════════════════════════════

def compute_delivery_profile(deals: pd.DataFrame, load_df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet periodengenau: Terminlieferung, Restlast, Spotmenge.
    
    Für jede Periode (Stunde/Tag) der Lastdaten:
    1. Gesamtlast auslesen
    2. Für jeden Deal: Prüfen ob Periode im Lieferzeitraum
    3. Liefermenge = Leistung [MW] × Periodenlänge [h]
    4. Bei Peak-Produkten: nur in Peak-Stunden
    5. Restlast = max(0, Last - Σ Terminlieferungen)
    6. Überdeckung = max(0, Σ Terminlieferungen - Last)
    """
    result = load_df[["datetime", "load_mwh"]].copy()
    result = result.sort_values("datetime").reset_index(drop=True)
    
    # Periodenlänge ermitteln
    if len(result) > 1:
        med_diff = result.datetime.diff().dropna().median()
        h_per_period = max(med_diff.total_seconds() / 3600, 0.25)
    else:
        h_per_period = 24.0
    
    # Ist es stündlich? Dann Peak/Base relevant
    is_hourly = h_per_period <= 1.5
    
    result["termin_mwh"] = 0.0
    result["termin_cost"] = 0.0
    
    deal_details = []
    
    for idx, deal in deals.iterrows():
        if pd.isna(deal.get("lieferstart")) or pd.isna(deal.get("lieferende")):
            continue
        if pd.isna(deal.get("preis")) or pd.isna(deal.get("leistung_mw")):
            continue
        
        ls = deal["lieferstart"]
        le = deal["lieferende"]
        mw = deal["leistung_mw"]
        price = deal["preis"]
        profil = deal.get("profil", "Base")
        
        # Maske: welche Perioden fallen in den Lieferzeitraum?
        mask = (result.datetime >= ls) & (result.datetime <= le)
        
        # Peak-Filter bei stündlichen Daten
        if is_hourly and profil == "Peak":
            peak_mask = result.datetime.apply(is_peak_hour)
            mask = mask & peak_mask
        
        # Liefermenge pro Periode
        mwh_per_period = mw * h_per_period
        
        n_periods = mask.sum()
        total_mwh = mwh_per_period * n_periods
        total_cost = total_mwh * price
        
        result.loc[mask, "termin_mwh"] += mwh_per_period
        result.loc[mask, "termin_cost"] += mwh_per_period * price
        
        deal_details.append({
            "deal_idx": idx,
            "produkt": deal.get("produkt", f"Deal {idx}"),
            "lieferstart": ls,
            "lieferende": le,
            "leistung_mw": mw,
            "preis": price,
            "profil": profil,
            "perioden": n_periods,
            "mwh_geliefert": total_mwh,
            "kosten": total_cost,
        })
    
    # Effektive Terminlieferung (≤ Last)
    result["termin_eff_mwh"] = result[["termin_mwh", "load_mwh"]].min(axis=1)
    result["überdeckung_mwh"] = (result["termin_mwh"] - result["load_mwh"]).clip(lower=0)
    result["restlast_mwh"] = (result["load_mwh"] - result["termin_eff_mwh"]).clip(lower=0)
    
    # Terminkosten proportional kürzen bei Überdeckung
    ratio = np.where(result.termin_mwh > 0,
                     result.termin_eff_mwh / result.termin_mwh, 0)
    result["termin_eff_cost"] = result.termin_cost * ratio
    
    return result, pd.DataFrame(deal_details) if deal_details else pd.DataFrame()


def evaluate_strategy(
    load_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    deals: pd.DataFrame,
    strategy_name: str = "Strategie",
) -> Dict:
    """Bewertet eine konkrete Beschaffungsstrategie periodengenau.
    
    1. Berechnet Terminlieferung + Restlast
    2. Merged Restlast mit Spotpreisen
    3. Berechnet Spotkosten periodengenau
    """
    # Delivery profile
    profile, deal_details = compute_delivery_profile(deals, load_df)
    
    # Merge mit Spot
    spot_c = spot_df.copy()
    
    # Exakter Join
    merged = pd.merge(profile, spot_c[["datetime", "spot_price"]], on="datetime", how="left")
    
    # Falls kein Match: täglicher Fallback
    if merged.spot_price.isna().mean() > 0.5:
        profile["date"] = profile.datetime.dt.date
        spot_daily = spot_c.assign(date=spot_c.datetime.dt.date).groupby("date").agg(
            spot_price=("spot_price", "mean")).reset_index()
        merged = pd.merge(profile, spot_daily, on="date", how="left")
        if "date" in merged.columns:
            merged = merged.drop(columns=["date"])
    
    merged["spot_price"] = merged["spot_price"].ffill().bfill()
    
    # Spot-Kosten für Restlast
    merged["spot_cost"] = merged.restlast_mwh * merged.spot_price
    
    # Überdeckungs-Erlös (Verkauf am Spot)
    merged["über_erlös"] = merged.überdeckung_mwh * merged.spot_price
    
    # Totals
    total_demand = float(merged.load_mwh.sum())
    termin_vol = float(merged.termin_eff_mwh.sum())
    termin_cost = float(merged.termin_eff_cost.sum())
    spot_vol = float(merged.restlast_mwh.sum())
    spot_cost = float(merged.spot_cost.sum())
    über_vol = float(merged.überdeckung_mwh.sum())
    über_erlös = float(merged.über_erlös.sum())
    
    total_cost = termin_cost + spot_cost - über_erlös
    
    # Benchmark: 100% Spot
    total_spot_cost = float((merged.load_mwh * merged.spot_price).sum())
    avg_spot = total_spot_cost / total_demand if total_demand else 0
    
    avg_price = total_cost / total_demand if total_demand else 0
    pnl = total_spot_cost - total_cost
    pct = pnl / total_spot_cost * 100 if total_spot_cost else 0
    
    termin_pct = termin_vol / total_demand * 100 if total_demand else 0
    avg_termin = termin_cost / termin_vol if termin_vol > 0 else 0
    
    merged["period_cost"] = merged.termin_eff_cost + merged.spot_cost - merged.über_erlös
    merged["cum_cost"] = merged.period_cost.cumsum()
    merged["cum_spot_only"] = (merged.load_mwh * merged.spot_price).cumsum()
    
    return dict(
        name=strategy_name,
        total_demand=total_demand,
        termin_vol=termin_vol, termin_cost=termin_cost, termin_pct=termin_pct,
        avg_termin=avg_termin,
        spot_vol=spot_vol, spot_cost=spot_cost, avg_spot=avg_spot,
        über_vol=über_vol, über_erlös=über_erlös,
        total_cost=total_cost, avg_price=avg_price,
        total_spot_cost=total_spot_cost,
        pnl=pnl, pct=pct,
        merged=merged, deal_details=deal_details,
    )


def scale_deals(deals: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Skaliert die Leistung aller Deals mit einem Faktor."""
    scaled = deals.copy()
    if "leistung_mw" in scaled.columns:
        scaled["leistung_mw"] = scaled["leistung_mw"] * factor
    if "menge_mwh" in scaled.columns:
        scaled["menge_mwh"] = scaled["menge_mwh"] * factor
    return scaled


def build_sim_deals(
    forward_curves: Dict[str, pd.DataFrame],
    products: List[str],
    base_mw: float,
    buy_date_idx: int,
) -> pd.DataFrame:
    """Baut simulierte Deals aus Forward-Kurven.
    
    Für jedes ausgewählte Produkt wird ein Deal erstellt
    mit dem Forward-Preis am Kaufdatum.
    """
    rows = []
    for prod_name in products:
        if prod_name not in forward_curves:
            continue
        curve = forward_curves[prod_name]
        s = curve.attrs.get("lieferstart")
        e = curve.attrs.get("lieferende")
        profil = curve.attrs.get("profil", "Base")
        
        if s is None or e is None:
            continue
        
        # Preis am Kaufdatum (Index)
        actual_idx = min(buy_date_idx, len(curve) - 1)
        price = curve.iloc[actual_idx]["price"]
        buy_dt = curve.iloc[actual_idx]["datetime"]
        
        rows.append({
            "produkt": prod_name,
            "kaufdatum": buy_dt,
            "lieferstart": pd.Timestamp(s),
            "lieferende": pd.Timestamp(e),
            "leistung_mw": base_mw,
            "preis": price,
            "profil": profil,
        })
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_dca_simulation(
    load_df, spot_df, forward_curves, products, base_mw, freq, window_months,
) -> List[Dict]:
    """DCA: Kauft an jedem Handelstag zum jeweiligen Forward-Preis."""
    delivery_start = load_df.datetime.min()
    
    results = []
    for prod_name in products:
        if prod_name not in forward_curves:
            continue
        curve = forward_curves[prod_name]
        fb = curve[curve.datetime < delivery_start].copy()
        
        if window_months > 0:
            start = delivery_start - pd.DateOffset(months=window_months)
            fb = fb[fb.datetime >= start]
        
        if freq == "Wöchentlich":
            fb = fb.set_index("datetime").resample("W").last().dropna().reset_index()
        elif freq == "Monatlich":
            fb = fb.set_index("datetime").resample("ME").last().dropna().reset_index()
        
        if not fb.empty:
            results.append({
                "produkt": prod_name,
                "n_käufe": len(fb),
                "avg_preis": float(fb.price.mean()),
                "min_preis": float(fb.price.min()),
                "max_preis": float(fb.price.max()),
                "kaufdaten": fb,
            })
    
    return results


def run_full_backtest(
    load_df, spot_df, forward_curves, deals_df, config, progress=None
) -> Dict:
    """Vollständiges Backtesting mit korrekter periodengenauer Bewertung."""
    results = []
    all_merged = None
    total_spot_cost = None
    
    # 1. Echte Deals bewerten
    if deals_df is not None and not deals_df.empty:
        r = evaluate_strategy(load_df, spot_df, deals_df, "📝 Echte Deals")
        results.append(r)
        all_merged = r["merged"]
        total_spot_cost = r["total_spot_cost"]
    
    # 2. Falls keine Deals: trotzdem Benchmark berechnen
    if total_spot_cost is None:
        empty_deals = pd.DataFrame(columns=["produkt", "lieferstart", "lieferende",
                                            "leistung_mw", "preis", "profil"])
        bench = evaluate_strategy(load_df, spot_df, empty_deals, "100% Spot")
        all_merged = bench["merged"]
        total_spot_cost = bench["total_spot_cost"]
    
    # 3. Skalierte Deals (verschiedene Terminquoten)
    if deals_df is not None and not deals_df.empty:
        for factor in config.get("sim_shares", []):
            if factor == 0 or factor == 1.0:
                continue  # 0 = 100% Spot (benchmark), 1.0 = echte Deals
            scaled = scale_deals(deals_df, factor)
            name = f"Deals ×{factor:.0%}"
            r = evaluate_strategy(load_df, spot_df, scaled, name)
            results.append(r)
    
    # 4. Simulierte Deals aus Forward-Kurven
    if forward_curves:
        products = config.get("sim_products", list(forward_curves.keys()))
        products = [p for p in products if p in forward_curves]
        
        if products:
            # Verschiedene Kaufzeitpunkte
            first_curve = forward_curves[products[0]]
            delivery_start = load_df.datetime.min()
            fb = first_curve[first_curve.datetime < delivery_start]
            n_dates = len(fb)
            
            if n_dates > 0:
                # 5 Zeitpunkte: früh, 25%, mitte, 75%, spät
                buy_indices = [0, n_dates // 4, n_dates // 2, 3 * n_dates // 4, n_dates - 1]
                buy_labels = ["Frühestmöglich", "25% Vorlauf", "Mitte", "75% Vorlauf", "Letztmöglich"]
                
                for mw_factor in [0.5, 1.0, 2.0]:
                    for bi, bl in zip(buy_indices, buy_labels):
                        sim_deals = build_sim_deals(forward_curves, products, mw_factor, bi)
                        if not sim_deals.empty:
                            name = f"Sim {mw_factor:.0f}MW × {bl}"
                            r = evaluate_strategy(load_df, spot_df, sim_deals, name)
                            results.append(r)
                
                # DCA
                dca_results = run_dca_simulation(
                    load_df, spot_df, forward_curves, products,
                    1.0, config.get("dca_freq", "Täglich"),
                    config.get("dca_window_months", 0),
                )
                if dca_results:
                    # Baue DCA-Deals: durchschnittlicher Preis pro Produkt
                    dca_deals = []
                    for dr in dca_results:
                        curve = forward_curves[dr["produkt"]]
                        dca_deals.append({
                            "produkt": f"DCA-{dr['produkt']}",
                            "lieferstart": pd.Timestamp(curve.attrs.get("lieferstart")),
                            "lieferende": pd.Timestamp(curve.attrs.get("lieferende")),
                            "leistung_mw": 1.0,
                            "preis": dr["avg_preis"],
                            "profil": curve.attrs.get("profil", "Base"),
                        })
                    if dca_deals:
                        dca_df = pd.DataFrame(dca_deals)
                        for factor in [0.5, 1.0, 2.0]:
                            scaled = scale_deals(dca_df, factor)
                            name = f"DCA {factor:.0f}MW"
                            r = evaluate_strategy(load_df, spot_df, scaled, name)
                            results.append(r)
    
    # Ergebnis-Tabelle
    rows = []
    cum_data = {}
    for r in results:
        rows.append({
            "Strategie": r["name"],
            "Terminanteil [%]": r["termin_pct"],
            "Spotanteil [%]": 100 - r["termin_pct"],
            "Terminvol. [MWh]": r["termin_vol"],
            "Ø Termin [€/MWh]": r["avg_termin"],
            "Terminkosten [€]": r["termin_cost"],
            "Spotvol. [MWh]": r["spot_vol"],
            "Ø Spot [€/MWh]": r["avg_spot"],
            "Spotkosten [€]": r["spot_cost"],
            "Überdeckung [MWh]": r["über_vol"],
            "Über-Erlös [€]": r["über_erlös"],
            "Gesamt [€]": r["total_cost"],
            "Ø Preis [€/MWh]": r["avg_price"],
            "PnL vs Spot [€]": r["pnl"],
            "Ersparnis [%]": r["pct"],
        })
        cum_data[r["name"]] = r["merged"]["cum_cost"].values
    
    rdf = pd.DataFrame(rows).sort_values("Gesamt [€]").reset_index(drop=True)
    rdf.index += 1
    
    return dict(
        results=rdf, merged=all_merged,
        total_spot=total_spot_cost, demand=float(load_df.load_mwh.sum()),
        avg_spot=total_spot_cost / float(load_df.load_mwh.sum()) if load_df.load_mwh.sum() > 0 else 0,
        cum=cum_data, details={r["name"]: r for r in results},
    )


# ═══════════════════════════════════════════════
# DEMO DATA
# ═══════════════════════════════════════════════

def generate_demo():
    rng = np.random.default_rng(42)
    d25 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    n = len(d25)
    s = 20 * np.sin(np.linspace(0, 2 * np.pi, n))
    ld = pd.DataFrame({"datetime": d25, "load_mwh": np.maximum(100 + s + rng.normal(0, 8, n), 20)})
    sd = pd.DataFrame({"datetime": d25, "spot_price": np.maximum(75 + s * .75 + rng.normal(0, 12, n), 5)})
    
    d24 = pd.bdate_range("2024-01-02", "2024-12-30")[:250]
    base = 82 + np.cumsum(rng.normal(-.02, .8, len(d24)))
    
    curves = {}
    for name, s_date, e_date, offset in [
        ("Cal-25 Base", "2025-01-01", "2025-12-31", 0),
        ("Q1-25 Base", "2025-01-01", "2025-03-31", 3),
        ("Q2-25 Base", "2025-04-01", "2025-06-30", -2),
        ("Q3-25 Base", "2025-07-01", "2025-09-30", -5),
        ("Q4-25 Base", "2025-10-01", "2025-12-31", 2),
    ]:
        prices = np.maximum(base + offset + rng.normal(0, 1, len(d24)), 40)
        df = pd.DataFrame({"datetime": d24, "price": prices})
        sp, ep, pt = parse_product_period(name)
        df.attrs["lieferstart"] = sp
        df.attrs["lieferende"] = ep
        df.attrs["produkttyp"] = pt
        df.attrs["profil"] = detect_profile(name)
        curves[name] = df
    
    deals = pd.DataFrame({
        "produkt": ["Cal-25 Base", "Q1-25 Base", "Q3-25 Base"],
        "kaufdatum": pd.to_datetime(["2024-03-15", "2024-06-01", "2024-09-10"]),
        "lieferstart": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-07-01"]),
        "lieferende": pd.to_datetime(["2025-12-31", "2025-03-31", "2025-09-30"]),
        "leistung_mw": [10.0, 5.0, 3.0],
        "preis": [82.50, 85.00, 78.20],
        "profil": ["Base", "Base", "Base"],
    })
    
    return ld, sd, curves, deals


# ═══════════════════════════════════════════════
# UI COMPONENT
# ═══════════════════════════════════════════════

def data_input(title, key, state_key, val_label, val_name, placeholder, unit=""):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if st.session_state[state_key] is not None:
            df = st.session_state[state_key]
            c1, c2 = st.columns([5, 1])
            c1.success(
                f"✅ **{len(df)}** Eintr. · "
                f"{df.datetime.min().date()} → {df.datetime.max().date()} · "
                f"Ø {df[val_name].mean():.1f} {unit}")
            if c2.button("🗑️", key=f"{key}_del"):
                st.session_state[state_key] = None
                st.session_state.pop(f"{key}_raw", None)
                st.rerun()
            return
        tp, tf = st.tabs(["📋 Einfügen", "📁 Datei"])
        with tp:
            txt = st.text_area("", height=140, key=f"{key}_txt", placeholder=placeholder)
            if st.button("🔄 Verarbeiten", key=f"{key}_go", type="primary",
                         use_container_width=True, disabled=not txt):
                raw = parse_text(txt)
                if raw is not None and len(raw.columns) >= 2:
                    st.session_state[f"{key}_raw"] = raw
                    st.rerun()
                else:
                    st.error("❌ Mind. 2 Spalten nötig")
        with tf:
            f = st.file_uploader("", ["csv", "xlsx", "xls"], key=f"{key}_f", label_visibility="collapsed")
            if f and st.button("📁 Laden", key=f"{key}_fl", use_container_width=True):
                raw = load_file(f)
                if raw is not None:
                    st.session_state[f"{key}_raw"] = raw
                    st.rerun()
        raw = st.session_state.get(f"{key}_raw")
        if raw is not None and len(raw) > 0:
            st.divider()
            st.dataframe(raw.head(5), use_container_width=True, height=160)
            cols = list(raw.columns)
            at = find_datetime_col(raw)
            av = find_value_col(raw, at)
            c1, c2, c3 = st.columns([2, 2, 1])
            tc = c1.selectbox("📅", cols, index=cols.index(at) if at in cols else 0, key=f"{key}_tc")
            vc = c2.selectbox(f"📊 {val_label}", cols, index=cols.index(av) if av in cols else min(1, len(cols) - 1), key=f"{key}_vc")
            if c3.button("✅", key=f"{key}_ok", type="primary", use_container_width=True):
                df = clean_ts(raw, tc, vc, val_name)
                if df.empty:
                    st.error("Keine gültigen Daten.")
                else:
                    st.session_state[state_key] = df
                    st.session_state.pop(f"{key}_raw", None)
                    st.rerun()


# ═══════════════════════════════════════════════
# CHART FUNCTIONS
# ═══════════════════════════════════════════════

def _pc(v):
    return [C["pos"] if x > 0 else C["neg"] if x < 0 else C["neut"] for x in v]


def fig_costs(df, ref):
    fig = go.Figure(go.Bar(x=df.Strategie, y=df["Gesamt [€]"],
                           marker_color=_pc(df["PnL vs Spot [€]"]),
                           text=df["Gesamt [€]"].apply(lambda v: f"{v:,.0f}€"),
                           textposition="outside", textfont_size=9))
    fig.add_hline(y=ref, line_dash="dash", line_color="red",
                  annotation_text=f"100% Spot: {ref:,.0f}€")
    fig.update_layout(title="Gesamtkosten", xaxis_tickangle=-45, height=500, template=TPL)
    return fig


def fig_savings(df):
    fig = go.Figure(go.Bar(x=df.Strategie, y=df["PnL vs Spot [€]"],
                           marker_color=_pc(df["PnL vs Spot [€]"]),
                           text=df["PnL vs Spot [€]"].apply(lambda v: f"{v:+,.0f}€"),
                           textposition="outside", textfont_size=9))
    fig.add_hline(y=0, line_color="white")
    fig.update_layout(title="Ersparnis vs. Spot", xaxis_tickangle=-45, height=450, template=TPL)
    return fig


def fig_cum(merged, cum, keys, ref_col="cum_spot_only"):
    fig = go.Figure()
    if ref_col in merged.columns:
        fig.add_trace(go.Scatter(x=merged.datetime, y=merged[ref_col], mode="lines",
                                 name="100% Spot", line=dict(color="red", width=3, dash="dash")))
    for i, k in enumerate(keys):
        if k in cum:
            fig.add_trace(go.Scatter(x=merged.datetime, y=cum[k], mode="lines",
                                     name=k, line=dict(color=PAL[i % len(PAL)], width=2)))
    fig.update_layout(title="Kumulative Kosten", height=500, template=TPL, legend=dict(font_size=9))
    return fig


def fig_delivery_profile(merged):
    """Zeigt Last, Terminlieferung, Restlast übereinander."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.datetime, y=merged.load_mwh,
                             mode="lines", name="Last [MWh]",
                             line=dict(color=C["blue"], width=1.5)))
    fig.add_trace(go.Scatter(x=merged.datetime, y=merged.termin_eff_mwh,
                             mode="lines", name="Terminlieferung [MWh]",
                             fill="tozeroy", fillcolor="rgba(46,204,113,0.3)",
                             line=dict(color=C["pos"], width=1)))
    fig.add_trace(go.Scatter(x=merged.datetime, y=merged.restlast_mwh,
                             mode="lines", name="Restlast (→ Spot) [MWh]",
                             line=dict(color=C["orange"], width=1, dash="dot")))
    if (merged.überdeckung_mwh > 0).any():
        fig.add_trace(go.Scatter(x=merged.datetime, y=merged.überdeckung_mwh,
                                 mode="lines", name="Überdeckung [MWh]",
                                 line=dict(color=C["neg"], width=1, dash="dash")))
    fig.update_layout(title="Lieferprofil: Last vs. Terminlieferung", height=450, template=TPL)
    return fig


def fig_forward_curves(curves, deals_df=None):
    fig = go.Figure()
    for i, (name, curve) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=curve.datetime, y=curve.price,
                                 mode="lines", name=name,
                                 line=dict(color=PAL[i % len(PAL)], width=2)))
    if deals_df is not None and "kaufdatum" in deals_df.columns:
        for _, d in deals_df.iterrows():
            if pd.notna(d.get("kaufdatum")) and pd.notna(d.get("preis")):
                fig.add_trace(go.Scatter(
                    x=[d.kaufdatum], y=[d.preis],
                    mode="markers+text", name=d.get("produkt", "Deal"),
                    marker=dict(color=C["gold"], size=12, symbol="star"),
                    text=[d.get("produkt", "")], textposition="top center",
                    textfont=dict(size=9), showlegend=False))
    fig.update_layout(title="Forward-Kurven", height=450, template=TPL, yaxis_title="€/MWh")
    return fig


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════

st.sidebar.title("⚡ Energiebeschaffung")
st.sidebar.divider()
page = st.sidebar.radio("Navigation", [
    "📥 Datenimport", "⚙️ Strategien", "🔬 Analyse", "📊 Dashboard"
])
st.sidebar.divider()
st.sidebar.markdown("##### Status")
for sk, lbl in [("load_df", "Last"), ("spot_df", "Spot"),
                ("forward_curves", "Forward-Kurven"), ("deals_df", "Deals")]:
    d = st.session_state[sk]
    if d is not None:
        n = len(d) if not isinstance(d, dict) else len(d)
        st.sidebar.success(f"✅ {lbl} ({n})")
    else:
        st.sidebar.caption(f"⏳ {lbl}")

st.sidebar.divider()
sb1, sb2 = st.sidebar.columns(2)
if sb1.button("🗑️ Reset", use_container_width=True):
    for k, v in _DEF.items():
        st.session_state[k] = v.copy() if isinstance(v, (dict, list)) else v
    st.rerun()
if sb2.button("🎲 Demo", use_container_width=True):
    ld, sd, fc, dd = generate_demo()
    st.session_state.update(load_df=ld, spot_df=sd, forward_curves=fc, deals_df=dd)
    st.rerun()


# ═══════════════════════════════════════════════
# PAGE: DATENIMPORT
# ═══════════════════════════════════════════════

if page == "📥 Datenimport":
    st.header("📥 Datenimport")

    data_input("1️⃣ Lastprofil (Lieferperiode)", "load", "load_df",
               "Verbrauch [MWh]", "load_mwh",
               "Datum\tMWh\n01.01.2025\t120.5\n02.01.2025\t115.3", "MWh")

    data_input("2️⃣ Spotpreise (Lieferperiode)", "spot", "spot_df",
               "Preis [€/MWh]", "spot_price",
               "Datum\tEUR\n01.01.2025\t85.20\n02.01.2025\t92.10", "€/MWh")

    st.divider()

    # ── Forward-Kurven (MEHRERE PRODUKTE) ──
    with st.container(border=True):
        st.markdown("**3️⃣ Forward-Kurven (VOR Lieferung, mehrere Produkte)**")
        st.caption(
            "Forward-Preise für verschiedene Lieferprodukte. "
            "Entweder **breites Format** (Datum | Cal-25 | Q1-25 | …) "
            "oder **langes Format** (Datum | Produkt | Preis)."
        )

        if st.session_state.forward_curves is not None:
            fc = st.session_state.forward_curves
            c1, c2 = st.columns([5, 1])
            c1.success(f"✅ **{len(fc)} Produkte**: {', '.join(fc.keys())}")
            if c2.button("🗑️", key="fc_del"):
                st.session_state.forward_curves = None
                st.session_state.pop("fc_raw", None)
                st.rerun()
            for name, curve in fc.items():
                s = curve.attrs.get("lieferstart", "?")
                e = curve.attrs.get("lieferende", "?")
                st.caption(f"  • {name}: {len(curve)} Tage, Ø {curve.price.mean():.2f} €/MWh, Lieferung {s} → {e}")
        else:
            tp, tf = st.tabs(["📋 Einfügen", "📁 Datei"])
            with tp:
                txt = st.text_area("", height=180, key="fc_txt",
                                   placeholder=(
                                       "Breites Format:\n"
                                       "Datum\tCal-25 Base\tQ1-25 Base\tQ2-25 Base\n"
                                       "02.01.2024\t82.50\t84.00\t80.30\n"
                                       "03.01.2024\t82.80\t84.20\t80.50\n\n"
                                       "ODER Langes Format:\n"
                                       "Datum\tProdukt\tPreis\n"
                                       "02.01.2024\tCal-25 Base\t82.50\n"
                                       "02.01.2024\tQ1-25 Base\t84.00"))
                if st.button("🔄 Forward-Kurven verarbeiten", key="fc_go",
                             type="primary", use_container_width=True, disabled=not txt):
                    raw = parse_text(txt)
                    if raw is not None:
                        curves = parse_forward_curves(raw)
                        if curves:
                            st.session_state.forward_curves = curves
                            st.rerun()
                        else:
                            st.error("❌ Konnte keine Produkte erkennen. Spaltenüberschriften müssen Produktnamen enthalten (Cal-25, Q1-25, …).")
            with tf:
                f = st.file_uploader("", ["csv", "xlsx", "xls"], key="fc_f", label_visibility="collapsed")
                if f and st.button("📁 Laden", key="fc_fl", use_container_width=True):
                    raw = load_file(f)
                    if raw is not None:
                        curves = parse_forward_curves(raw)
                        if curves:
                            st.session_state.forward_curves = curves
                            st.rerun()
                        else:
                            st.error("❌ Produkte nicht erkannt.")

    st.divider()

    # ── Deals ──
    with st.container(border=True):
        st.markdown("**4️⃣ Echte Deals (optional)**")
        st.caption("Abgeschlossene Termingeschäfte: Produkt, Leistung (MW), Preis (€/MWh)")

        if st.session_state.deals_df is not None:
            d = st.session_state.deals_df
            c1, c2 = st.columns([5, 1])
            c1.success(f"✅ **{len(d)} Deals**")
            if c2.button("🗑️", key="d_del"):
                st.session_state.deals_df = None
                st.rerun()
            dc = [c for c in ["produkt", "kaufdatum", "lieferstart", "lieferende",
                              "leistung_mw", "preis", "profil"] if c in d.columns]
            st.dataframe(d[dc], use_container_width=True, hide_index=True)
        else:
            txt = st.text_area("", height=160, key="d_txt",
                               placeholder="Produkt\tKaufdatum\tLeistung_MW\tPreis\nCal-25 Base\t15.03.2024\t10\t82.50\nQ1-25 Base\t01.06.2024\t5\t85.00")
            if st.button("🔄 Deals verarbeiten", key="d_go", type="primary",
                         use_container_width=True, disabled=not txt):
                raw = parse_text(txt)
                if raw is not None:
                    p = parse_deals(raw)
                    if p is not None and not p.empty and p.lieferstart.notna().sum() > 0:
                        st.session_state.deals_df = p
                        st.rerun()
                    else:
                        st.error("❌ Produkt + Preis + erkennbarer Zeitraum nötig.")

    if all(st.session_state[k] is not None for k in ("load_df", "spot_df")):
        has_fwd = st.session_state.forward_curves is not None
        has_deals = st.session_state.deals_df is not None
        if has_fwd or has_deals:
            st.success("✅ Bereit für Analyse → weiter zu **⚙️ Strategien**")
        else:
            st.warning("⚠️ Mindestens Forward-Kurven ODER Deals nötig für die Analyse.")


# ═══════════════════════════════════════════════
# PAGE: STRATEGIEN
# ═══════════════════════════════════════════════

elif page == "⚙️ Strategien":
    st.header("⚙️ Strategiekonfiguration")

    tab_sim, tab_dca, tab_deals = st.tabs(["🔄 Simulation", "📊 DCA", "📝 Deals"])

    with tab_sim:
        st.markdown("##### Welche Produkte simulieren?")
        fc = st.session_state.forward_curves
        if fc:
            products = st.multiselect("Produkte", list(fc.keys()), default=list(fc.keys()))
            st.session_state.config["sim_products"] = products

            st.markdown("##### Skalierungsfaktoren (Leistung)")
            st.caption("1.0 = Basis-Leistung, 0.5 = halbe Leistung, 2.0 = doppelt")
            scales_txt = st.text_input("Faktoren", "0.5, 1.0, 1.5, 2.0, 3.0")
            try:
                scales = sorted({float(x.strip()) for x in scales_txt.split(",") if float(x.strip()) > 0})
            except ValueError:
                scales = [0.5, 1.0, 2.0]
            st.session_state.config["sim_shares"] = scales
            st.caption(f"Faktoren: {', '.join(f'{s:.1f}×' for s in scales)}")

            st.markdown("##### Transaktionskosten")
            tx = st.number_input("TX [€/MWh]", 0.0, step=0.05, format="%.2f", key="tx")
            st.session_state.config["tx_cost"] = tx

            n_sc = len(products) * len(scales) * 5  # 5 Kaufzeitpunkte
            st.info(f"📐 Ca. **{n_sc} Szenarien** (+ DCA + Deals)")
        else:
            st.warning("Forward-Kurven fehlen → bitte im Datenimport hochladen.")

    with tab_dca:
        st.markdown("##### Dollar Cost Averaging")
        st.markdown("> Kauft gleichmäßig über den Beschaffungszeitraum zum jeweiligen Forward-Preis.")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                dca_w = st.selectbox("Fenster", [
                    "Alle Daten", "36 Monate", "24 Monate",
                    "18 Monate", "12 Monate", "6 Monate"], index=0)
                wm = {"Alle": 0, "36": 36, "24": 24, "18": 18, "12": 12, "6": 6}
                st.session_state.config["dca_window_months"] = next(
                    (v for k, v in wm.items() if k in dca_w), 0)
            with c2:
                dca_f = st.selectbox("Frequenz", ["Täglich", "Wöchentlich", "Monatlich"])
                st.session_state.config["dca_freq"] = dca_f

    with tab_deals:
        st.markdown("##### Echte Deals skalieren")
        dd = st.session_state.deals_df
        if dd is not None and not dd.empty:
            st.dataframe(dd[[c for c in ["produkt", "leistung_mw", "preis", "profil", "lieferstart", "lieferende"]
                            if c in dd.columns]], use_container_width=True, hide_index=True)
            st.caption("Im Backtesting werden die Deals auch mit verschiedenen Skalierungsfaktoren getestet.")
        else:
            st.info("Keine Deals hinterlegt.")

    # Produkt-Info
    if fc:
        with st.expander("🔍 Forward-Kurven Übersicht"):
            for name, curve in fc.items():
                s = curve.attrs.get("lieferstart", "?")
                e = curve.attrs.get("lieferende", "?")
                p = curve.attrs.get("profil", "?")
                st.markdown(f"**{name}** · {p} · Lieferung {s} → {e} · "
                            f"{len(curve)} Handelstage · Ø {curve.price.mean():.2f} €/MWh · "
                            f"Min {curve.price.min():.2f} · Max {curve.price.max():.2f}")


# ═══════════════════════════════════════════════
# PAGE: ANALYSE
# ═══════════════════════════════════════════════

elif page == "🔬 Analyse":
    st.header("🔬 Analyse")

    ld = st.session_state.load_df
    sd = st.session_state.spot_df
    fc = st.session_state.forward_curves
    dd = st.session_state.deals_df

    if ld is None or sd is None:
        st.warning("⚠️ Last + Spot nötig → 📥 Datenimport")
        st.stop()
    if fc is None and dd is None:
        st.warning("⚠️ Forward-Kurven oder Deals nötig → 📥 Datenimport")
        st.stop()

    cfg = st.session_state.config

    # ═════════ INTERAKTIVER SLIDER ═════════
    st.subheader("🎚️ Interaktive Schnellbewertung")
    st.caption("Skalieren Sie die Terminbeschaffung sofort – ohne Backtesting.")

    with st.container(border=True):
        if dd is not None and not dd.empty:
            slider_val = st.slider("**Deal-Skalierung**", 0.0, 3.0, 1.0, 0.1,
                                   format="%.1f×", key="int_slider",
                                   help="1.0 = echte Deals, 0.5 = halbe Leistung, 0 = 100% Spot")

            scaled = scale_deals(dd, slider_val)
            r = evaluate_strategy(ld, sd, scaled, f"Deals ×{slider_val:.1f}")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Skalierung", f"{slider_val:.1f}×")
            c2.metric("Gesamt", f"{r['total_cost']:,.0f} €", f"{r['pnl']:+,.0f} € vs Spot")
            c3.metric("Ø Preis", f"{r['avg_price']:.2f} €/MWh")
            c4.metric("Terminanteil", f"{r['termin_pct']:.1f} %")
            c5.metric("100% Spot", f"{r['total_spot_cost']:,.0f} €")

            # Lieferprofil
            st.plotly_chart(fig_delivery_profile(r["merged"]), use_container_width=True)

            # Deal-Details
            if not r["deal_details"].empty:
                with st.expander("📋 Deal-Details"):
                    st.dataframe(r["deal_details"].style.format({
                        "leistung_mw": "{:.1f}", "preis": "{:.2f}",
                        "mwh_geliefert": "{:,.0f}", "kosten": "{:,.0f}",
                    }), use_container_width=True, hide_index=True)
        else:
            st.info("Deals hinterlegen für interaktive Skalierung, oder Forward-Kurven für Simulation nutzen.")

    st.divider()

    # ═════════ FULL BACKTEST ═════════
    st.subheader("🔬 Vollständiges Backtesting")

    if st.button("🚀 Backtesting starten", type="primary", use_container_width=True):
        prog = st.progress(0, "Starte …")
        try:
            bt = run_full_backtest(ld, sd, fc, dd, cfg, prog)
        except ValueError as e:
            prog.empty()
            st.error(f"❌ {e}")
            st.stop()
        prog.empty()
        st.session_state.bt = bt
        st.success(f"✅ **{len(bt['results'])} Strategien** bewertet")
        st.rerun()

    bt = st.session_state.bt
    if bt is not None:
        R = bt["results"]
        TS = bt["total_spot"]
        TD = bt["demand"]
        AS = bt["avg_spot"]
        CUM = bt["cum"]
        details = bt.get("details", {})
        merged_ref = bt["merged"]

        best = R.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Beste", best.Strategie, f"{best['PnL vs Spot [€]']:+,.0f} €")
        c2.metric("Kosten", f"{best['Gesamt [€]']:,.0f} €", f"{best['Ersparnis [%]']:+.1f}%")
        c3.metric("100% Spot", f"{TS:,.0f} €", f"Ø {AS:.2f} €/MWh")
        c4.metric("Bedarf", f"{TD:,.0f} MWh")

        # Echte Deals hervorheben
        if "📝 Echte Deals" in R.Strategie.values:
            deal_row = R[R.Strategie == "📝 Echte Deals"].iloc[0]
            rank = R.index[R.Strategie == "📝 Echte Deals"].tolist()[0]
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns(5)
                icon = "🟢" if deal_row["PnL vs Spot [€]"] > 0 else "🔴"
                c1.metric(f"{icon} Echte Deals", f"{deal_row['Gesamt [€]']:,.0f} €")
                c2.metric("Ersparnis", f"{deal_row['PnL vs Spot [€]']:+,.0f} €")
                c3.metric("Terminanteil", f"{deal_row['Terminanteil [%]']:.1f}%")
                c4.metric("Ø Termin", f"{deal_row['Ø Termin [€/MWh]']:.2f} €/MWh")
                c5.metric("Ranking", f"Platz {rank}", f"von {len(R)}")

        st.markdown("##### 🏆 Top 5")
        st.dataframe(R.head(5)[["Strategie", "Gesamt [€]", "Ø Preis [€/MWh]",
                                "Terminanteil [%]", "PnL vs Spot [€]", "Ersparnis [%]",
                                "Überdeckung [MWh]"]].style.format({
            "Gesamt [€]": "{:,.0f}", "Ø Preis [€/MWh]": "{:.2f}",
            "Terminanteil [%]": "{:.1f}", "PnL vs Spot [€]": "{:+,.0f}",
            "Ersparnis [%]": "{:+.1f}%", "Überdeckung [MWh]": "{:,.0f}",
        }), use_container_width=True, hide_index=True)

        with st.expander("📋 Alle Ergebnisse"):
            st.dataframe(R, use_container_width=True)

        # Charts
        st.divider()
        max_s = st.slider("Max. Balken", 5, len(R), min(15, len(R)), key="ms")
        show = R.head(max_s)

        tabs = st.tabs(["💰 Kosten", "📊 Ersparnis", "📈 Kumulativ",
                        "🔵 Forward-Kurven", "📦 Lieferprofil"])

        with tabs[0]:
            st.plotly_chart(fig_costs(show, TS), use_container_width=True)
        with tabs[1]:
            st.plotly_chart(fig_savings(show), use_container_width=True)
        with tabs[2]:
            dk = [k for k in CUM if k.startswith("📝")]
            if not dk:
                dk = list(CUM.keys())[:5]
            sel = st.multiselect("Strategien", list(CUM.keys()), default=dk[:5])
            if sel and merged_ref is not None:
                st.plotly_chart(fig_cum(merged_ref, CUM, sel), use_container_width=True)
        with tabs[3]:
            if fc:
                st.plotly_chart(fig_forward_curves(fc, dd), use_container_width=True)
        with tabs[4]:
            # Lieferprofil der besten / echten Strategie
            strat_choice = st.selectbox("Strategie wählen",
                                        [r["name"] for r in details.values()])
            if strat_choice in details:
                det = details[strat_choice]
                st.plotly_chart(fig_delivery_profile(det["merged"]), use_container_width=True)
                if not det["deal_details"].empty:
                    st.dataframe(det["deal_details"].style.format({
                        "leistung_mw": "{:.1f}", "preis": "{:.2f}",
                        "mwh_geliefert": "{:,.0f}", "kosten": "{:,.0f}",
                    }), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.header("📊 Dashboard & Empfehlung")

    bt = st.session_state.bt
    if bt is None:
        st.warning("⚠️ Erst Analyse durchführen.")
        st.stop()

    R = bt["results"]
    TS = bt["total_spot"]
    TD = bt["demand"]
    AS = bt["avg_spot"]
    details = bt.get("details", {})

    best, worst = R.iloc[0], R.iloc[-1]

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Optimum", best.Strategie)
        c2.metric("Kosten", f"{best['Gesamt [€]']:,.0f} €", f"{best['PnL vs Spot [€]']:+,.0f} €")
        c3.metric("Schlechteste", worst.Strategie, f"{worst['PnL vs Spot [€]']:+,.0f} €")
        c4.metric("Spanne", f"{worst['Gesamt [€]'] - best['Gesamt [€]']:,.0f} €")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Bedarf", f"{TD:,.0f} MWh")
        c2.metric("100% Spot", f"{TS:,.0f} €")
        c3.metric("Ø Spot", f"{AS:.2f} €/MWh")

    # Echte Deals
    if "📝 Echte Deals" in details:
        st.divider()
        st.subheader("⭐ Bewertung Ihrer Deals")
        det = details["📝 Echte Deals"]
        rank = R.index[R.Strategie == "📝 Echte Deals"].tolist()
        rank_pos = rank[0] if rank else "?"

        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns(5)
            icon = "🟢" if det["pnl"] > 0 else "🔴"
            c1.metric(f"{icon} Kosten", f"{det['total_cost']:,.0f} €")
            c2.metric("Ersparnis vs Spot", f"{det['pnl']:+,.0f} €", f"{det['pct']:+.1f}%")
            c3.metric("Terminanteil", f"{det['termin_pct']:.1f}%",
                      f"{det['termin_vol']:,.0f} MWh")
            c4.metric("Ø Terminpreis", f"{det['avg_termin']:.2f} €/MWh")
            c5.metric("Ranking", f"Platz {rank_pos}", f"von {len(R)}")

        if det["über_vol"] > 0:
            st.warning(f"⚠️ Überdeckung: {det['über_vol']:,.0f} MWh mussten am Spot verkauft werden "
                       f"(Erlös: {det['über_erlös']:,.0f} €)")

        if not det["deal_details"].empty:
            st.markdown("**Deal-Details:**")
            st.dataframe(det["deal_details"][[
                "produkt", "lieferstart", "lieferende", "leistung_mw",
                "preis", "profil", "perioden", "mwh_geliefert", "kosten"
            ]].style.format({
                "leistung_mw": "{:.1f} MW", "preis": "{:.2f} €/MWh",
                "mwh_geliefert": "{:,.0f} MWh", "kosten": "{:,.0f} €",
            }), use_container_width=True, hide_index=True)

    st.divider()

    # Risiko
    st.subheader("📉 Risiko")
    costs = R["Gesamt [€]"].values
    pnl = R["PnL vs Spot [€]"].values
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min.", f"{costs.min():,.0f} €")
        c2.metric("Max.", f"{costs.max():,.0f} €")
        c3.metric("Std.-Abw.", f"{costs.std():,.0f} €")
        if len(pnl) > 2:
            c4.metric("VaR 95%", f"{np.percentile(pnl, 5):+,.0f} €")

    st.divider()

    # Empfehlung
    st.subheader("💡 Empfehlung")
    if best["PnL vs Spot [€]"] > 0:
        st.success(
            f"**Optimale Strategie: {best.Strategie}**\n\n"
            f"- Kosten: **{best['Gesamt [€]']:,.0f} €** (statt {TS:,.0f} € Spot)\n"
            f"- Ersparnis: **{best['PnL vs Spot [€]']:+,.0f} €** ({best['Ersparnis [%]']:+.1f}%)\n"
            f"- Terminanteil: **{best['Terminanteil [%]']:.1f}%**")
    else:
        st.info(f"**100% Spot wäre am günstigsten gewesen** ({TS:,.0f} €, Ø {AS:.2f} €/MWh)")

    st.divider()
    st.download_button("📥 CSV", R.to_csv(index=False, sep=";", decimal=",").encode("utf-8"),
                       "ergebnisse.csv", "text/csv", use_container_width=True)

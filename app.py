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
PEAK_START, PEAK_END = 8, 20  # Peak: Mo-Fr 08:00-20:00

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
    forward_curves=None,  # FIX #4: Dict[name -> {"df":..., "start":..., "end":..., "type":..., "profile":...}]
    deals_df=None,
    bt=None,
    config=dict(
        sim_products=[],
        sim_shares=[0.5, 1.0, 1.5, 2.0, 3.0],
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
    if not name:
        return None, None, ""
    s = name.strip().upper().replace("/", "-").replace(" ", "-")

    def _yr(y):
        y = int(y)
        return y + 2000 if y < 100 else y

    m = re.search(r"(?:CAL|YEAR)-?(\d{2,4})", s)
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
    return "Peak" if "peak" in name.lower() else "Base"


def is_peak_hour(dt) -> bool:
    """Mo-Fr 08:00-20:00 = Peak."""
    if dt.weekday() >= 5:
        return False
    return PEAK_START <= dt.hour < PEAK_END


def peak_hours_in_day(dt) -> float:
    """Wie viele Peak-Stunden hat dieser Tag?"""
    if dt.weekday() >= 5:
        return 0.0
    return float(PEAK_END - PEAK_START)  # 12h


# ═══════════════════════════════════════════════
# DEAL PARSER
# ═══════════════════════════════════════════════

def parse_deals(raw_df) -> Optional[pd.DataFrame]:
    if raw_df is None or raw_df.empty:
        return None
    cols = {c: c.lower().replace("_", " ").replace("-", " ") for c in raw_df.columns}
    r = pd.DataFrame()

    pc = next((o for o, l in cols.items()
               if any(k in l for k in ("produkt", "product", "name", "kontrakt"))), None)
    r["produkt"] = raw_df[pc].astype(str).str.strip() if pc else raw_df.iloc[:, 0].astype(str).str.strip()

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

    kd = next((o for o, l in cols.items()
               if any(k in l for k in ("kaufdatum", "kauf", "buy", "trade", "handel"))), None)
    if kd:
        r["kaufdatum"] = pd.to_datetime(raw_df[kd], dayfirst=True, errors="coerce")

    # FIX #2: Leistung (MW) UND Menge (MWh) erkennen
    mw_c = next((o for o, l in cols.items()
                 if any(k in l for k in ("leistung", "power", "kapaz"))
                 or (l.strip() == "mw" and "mwh" not in l)), None)
    if mw_c:
        v = raw_df[mw_c]
        if v.dtype == object:
            v = v.astype(str).str.replace(",", ".")
        r["leistung_mw"] = pd.to_numeric(v, errors="coerce")

    mwh_c = next((o for o, l in cols.items()
                  if any(k in l for k in ("menge", "volume", "energy", "mwh"))), None)
    if mwh_c:
        v = raw_df[mwh_c]
        if v.dtype == object:
            v = v.astype(str).str.replace(",", ".")
        r["menge_mwh"] = pd.to_numeric(v, errors="coerce")

    p_c = next((o for o, l in cols.items()
                if any(k in l for k in ("preis", "price", "eur", "€"))), None)
    if p_c:
        v = raw_df[p_c]
        if v.dtype == object:
            v = v.astype(str).str.replace(",", ".").str.replace("€", "").str.strip()
        r["preis"] = pd.to_numeric(v, errors="coerce")
    else:
        return None

    pf = next((o for o, l in cols.items()
               if any(k in l for k in ("profil", "profile", "typ", "type"))), None)
    r["profil"] = raw_df[pf].astype(str).str.strip() if pf else r["produkt"].apply(detect_profile)

    return r.dropna(subset=["preis"]).reset_index(drop=True)


# ═══════════════════════════════════════════════
# FORWARD-KURVEN PARSER
# FIX #4: Metadata als dict, NICHT DataFrame.attrs
# ═══════════════════════════════════════════════

def parse_forward_curves(raw_df) -> Optional[Dict]:
    """Returns Dict[name -> {"df": DataFrame, "start": date, "end": date, "type": str, "profile": str}]"""
    if raw_df is None or raw_df.empty:
        return None
    dt_col = find_datetime_col(raw_df)
    if dt_col is None:
        return None
    dates = pd.to_datetime(raw_df[dt_col], dayfirst=True, errors="coerce")
    other_cols = [c for c in raw_df.columns if c != dt_col]

    prod_col = next((c for c in other_cols
                     if any(k in c.lower() for k in ("produkt", "product", "kontrakt", "contract"))), None)
    price_col = next((c for c in other_cols
                      if any(k in c.lower() for k in ("preis", "price", "eur", "€", "close", "settle"))), None)

    curves = {}
    if prod_col and price_col:
        for prod_name, grp in raw_df.groupby(prod_col):
            name = str(prod_name).strip()
            sub = pd.DataFrame({
                "datetime": pd.to_datetime(grp[dt_col], dayfirst=True, errors="coerce"),
                "price": pd.to_numeric(
                    grp[price_col].astype(str).str.replace(",", ".").str.replace("€", "").str.strip(),
                    errors="coerce"),
            }).dropna().sort_values("datetime").reset_index(drop=True)
            if len(sub) > 0:
                s, e, pt = parse_product_period(name)
                curves[name] = {"df": sub, "start": s, "end": e, "type": pt, "profile": detect_profile(name)}
    else:
        for col in other_cols:
            vals = raw_df[col]
            if vals.dtype == object:
                vals = pd.to_numeric(vals.astype(str).str.replace(",", ".").str.replace("€", "").str.strip(),
                                     errors="coerce")
            else:
                vals = pd.to_numeric(vals, errors="coerce")
            if vals.notna().mean() < 0.3:
                continue
            sub = pd.DataFrame({"datetime": dates, "price": vals}).dropna().sort_values("datetime").reset_index(drop=True)
            if len(sub) > 0:
                name = str(col).strip()
                s, e, pt = parse_product_period(name)
                curves[name] = {"df": sub, "start": s, "end": e, "type": pt, "profile": detect_profile(name)}

    return curves if curves else None


def get_curve_df(curves: Dict, name: str) -> pd.DataFrame:
    """Hilfsfunktion: sicher auf curve["df"] zugreifen."""
    entry = curves.get(name, {})
    if isinstance(entry, dict):
        return entry.get("df", pd.DataFrame())
    return entry  # Fallback für alte Struktur


def get_curve_meta(curves: Dict, name: str, key: str, default=None):
    """Hilfsfunktion: sicher auf curve-Metadaten zugreifen."""
    entry = curves.get(name, {})
    if isinstance(entry, dict):
        return entry.get(key, default)
    return default


# ═══════════════════════════════════════════════
# BEWERTUNGS-ENGINE – KORRIGIERT
# ═══════════════════════════════════════════════

def compute_delivery_profile(
    deals: pd.DataFrame, load_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Periodengenau: Terminlieferung pro Deal, Restlast, Überdeckung.

    FIX #2: Unterstützt sowohl leistung_mw als auch menge_mwh.
    FIX #3: Peak-Produkte korrekt bei täglicher Granularität.
    FIX #10: Warnung bei Produktüberlappung.

    Returns: (profile_df, deal_details_df, warnings_list)
    """
    result = load_df[["datetime", "load_mwh"]].copy().sort_values("datetime").reset_index(drop=True)
    warnings = []

    if len(result) > 1:
        h_per_period = max(result.datetime.diff().dropna().median().total_seconds() / 3600, 0.25)
    else:
        h_per_period = 24.0

    is_hourly = h_per_period <= 1.5
    is_daily = 20 <= h_per_period <= 28

    result["termin_mwh"] = 0.0
    result["termin_cost"] = 0.0

    deal_details = []
    active_products = []  # FIX #10: Track overlapping products

    for idx, deal in deals.iterrows():
        if pd.isna(deal.get("lieferstart")) or pd.isna(deal.get("lieferende")):
            continue
        if pd.isna(deal.get("preis")):
            continue

        ls, le = deal.lieferstart, deal.lieferende
        price = deal.preis
        profil = deal.get("profil", "Base")

        # FIX #2: Leistung ODER Menge verwenden
        has_mw = "leistung_mw" in deal.index and pd.notna(deal.get("leistung_mw"))
        has_mwh = "menge_mwh" in deal.index and pd.notna(deal.get("menge_mwh"))

        if not has_mw and not has_mwh:
            warnings.append(f"⚠️ Deal '{deal.get('produkt', idx)}': Weder MW noch MWh angegeben – übersprungen.")
            continue

        mask = (result.datetime >= ls) & (result.datetime <= le)
        n_periods = int(mask.sum())
        if n_periods == 0:
            warnings.append(f"⚠️ Deal '{deal.get('produkt', idx)}': Kein Überlapp mit Lastdaten.")
            continue

        # FIX #3: Peak/Base korrekt für alle Granularitäten
        if profil == "Peak":
            if is_hourly:
                peak_mask = result.datetime.apply(is_peak_hour)
                mask = mask & peak_mask
                mwh_per_period = deal.leistung_mw * h_per_period if has_mw else None
            elif is_daily:
                # Tägliche Daten: Peak-Stunden pro Tag berechnen
                peak_hours = result.datetime.apply(peak_hours_in_day)
                # Nur Wochentage liefern
                weekday_mask = result.datetime.dt.weekday < 5
                mask = mask & weekday_mask
                if has_mw:
                    mwh_per_period = None  # wird pro Tag berechnet
                    # MW × Peak-Stunden des jeweiligen Tages
                    mwh_series = deal.leistung_mw * peak_hours
                else:
                    mwh_per_period = None
            else:
                mwh_per_period = deal.leistung_mw * h_per_period if has_mw else None
        else:
            # Base: alle Perioden, volle Stunden
            mwh_per_period = deal.leistung_mw * h_per_period if has_mw else None

        n_delivery = int(mask.sum())
        if n_delivery == 0:
            continue

        # Liefermengen berechnen
        if profil == "Peak" and is_daily and has_mw:
            # Spezialfall: tägliche Peak-Lieferung variiert
            delivery = deal.leistung_mw * peak_hours
            result.loc[mask, "termin_mwh"] += delivery[mask].values
            result.loc[mask, "termin_cost"] += (delivery[mask] * price).values
            total_mwh = float(delivery[mask].sum())
        elif has_mw:
            result.loc[mask, "termin_mwh"] += mwh_per_period
            result.loc[mask, "termin_cost"] += mwh_per_period * price
            total_mwh = mwh_per_period * n_delivery
        else:
            # FIX #2: menge_mwh gleichmäßig verteilen
            mwh_per_period_calc = deal.menge_mwh / n_delivery
            result.loc[mask, "termin_mwh"] += mwh_per_period_calc
            result.loc[mask, "termin_cost"] += mwh_per_period_calc * price
            total_mwh = deal.menge_mwh

        # FIX #10: Überlappung prüfen
        deal_range = (ls, le, profil)
        for prev_name, prev_start, prev_end, prev_prof in active_products:
            if prev_prof == profil and max(ls, prev_start) <= min(le, prev_end):
                overlap_start = max(ls, prev_start).strftime("%d.%m.%Y")
                overlap_end = min(le, prev_end).strftime("%d.%m.%Y")
                warnings.append(
                    f"ℹ️ Stacking: '{deal.get('produkt', idx)}' + '{prev_name}' "
                    f"überlappen {overlap_start}–{overlap_end} ({profil}). "
                    f"Leistungen addieren sich."
                )
        active_products.append((deal.get("produkt", f"Deal {idx}"), ls, le, profil))

        deal_details.append({
            "produkt": deal.get("produkt", f"Deal {idx}"),
            "lieferstart": ls, "lieferende": le,
            "leistung_mw": deal.get("leistung_mw", np.nan),
            "menge_mwh": total_mwh,
            "preis": price, "profil": profil,
            "perioden": n_delivery, "kosten": total_mwh * price,
        })

    # Effektive Lieferung begrenzt auf Last
    result["termin_eff_mwh"] = result[["termin_mwh", "load_mwh"]].min(axis=1)
    result["überdeckung_mwh"] = (result.termin_mwh - result.load_mwh).clip(lower=0)
    result["restlast_mwh"] = (result.load_mwh - result.termin_eff_mwh).clip(lower=0)

    ratio = np.where(result.termin_mwh > 0, result.termin_eff_mwh / result.termin_mwh, 0)
    result["termin_eff_cost"] = result.termin_cost * ratio

    return result, pd.DataFrame(deal_details) if deal_details else pd.DataFrame(), warnings


def merge_profile_with_spot(profile: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
    """FIX #6: Robuster Spot-Merge mit klarer Fallback-Logik.

    1. Exakter Datetime-Match
    2. Falls <10% Match: täglicher Fallback
    3. Forward-fill für vereinzelte Lücken
    """
    merged = pd.merge(profile, spot_df[["datetime", "spot_price"]],
                       on="datetime", how="left")
    match_rate = merged.spot_price.notna().mean()

    if match_rate < 0.1:
        # Fast keine Matches → täglicher Fallback
        prof_daily = profile.copy()
        prof_daily["_merge_date"] = prof_daily.datetime.dt.date
        spot_daily = (spot_df.assign(_merge_date=spot_df.datetime.dt.date)
                      .groupby("_merge_date")
                      .agg(spot_price=("spot_price", "mean"))
                      .reset_index())
        merged = pd.merge(prof_daily, spot_daily, on="_merge_date", how="left")
        merged = merged.drop(columns=["_merge_date"])
        match_rate = merged.spot_price.notna().mean()

    # Lücken füllen
    if merged.spot_price.isna().any():
        n_gaps = int(merged.spot_price.isna().sum())
        merged["spot_price"] = merged["spot_price"].ffill().bfill()

    return merged


def evaluate_strategy(
    load_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    deals: pd.DataFrame,
    name: str = "Strategie",
    tx_cost: float = 0.0,  # FIX #1: TX-Kosten als Parameter
) -> Dict:
    """Bewertet eine Strategie periodengenau.

    FIX #1: TX-Kosten werden auf Terminvolumen angewendet.
    FIX #6: Verbesserter Spot-Merge.
    """
    profile, deal_det, warnings = compute_delivery_profile(deals, load_df)
    merged = merge_profile_with_spot(profile, spot_df)

    if merged.spot_price.isna().any():
        warnings.append("⚠️ Spotpreise für einige Perioden nicht verfügbar – aufgefüllt.")

    # Kosten berechnen
    merged["spot_cost"] = merged.restlast_mwh * merged.spot_price
    merged["über_erlös"] = merged.überdeckung_mwh * merged.spot_price

    td = float(merged.load_mwh.sum())
    tv = float(merged.termin_eff_mwh.sum())
    tc_term = float(merged.termin_eff_cost.sum())
    sv = float(merged.restlast_mwh.sum())
    sc = float(merged.spot_cost.sum())
    uv = float(merged.überdeckung_mwh.sum())
    ue = float(merged.über_erlös.sum())

    # FIX #1: TX-Kosten auf Terminvolumen
    tx_total = tv * tx_cost

    tc = tc_term + sc - ue + tx_total

    # Benchmark
    merged["full_spot_cost"] = merged.load_mwh * merged.spot_price
    ts = float(merged.full_spot_cost.sum())
    avg_s = ts / td if td else 0
    avg_p = tc / td if td else 0
    pnl = ts - tc
    pct = pnl / ts * 100 if ts else 0
    tp = tv / td * 100 if td else 0
    at = tc_term / tv if tv > 0 else 0

    merged["period_cost"] = merged.termin_eff_cost + merged.spot_cost - merged.über_erlös
    # TX anteilig auf Perioden mit Terminlieferung verteilen
    if tv > 0:
        merged["period_cost"] += merged.termin_eff_mwh * tx_cost
    merged["cum_cost"] = merged.period_cost.cumsum()
    merged["cum_spot_only"] = merged.full_spot_cost.cumsum()

    return dict(
        name=name, total_demand=td,
        termin_vol=tv, termin_cost=tc_term, termin_pct=tp, avg_termin=at,
        spot_vol=sv, spot_cost=sc, avg_spot=avg_s,
        über_vol=uv, über_erlös=ue, tx_cost=tx_total,
        total_cost=tc, avg_price=avg_p, total_spot_cost=ts,
        pnl=pnl, pct=pct, merged=merged, deal_details=deal_det,
        warnings=warnings,
    )


def scale_deals(deals: pd.DataFrame, factor: float) -> pd.DataFrame:
    s = deals.copy()
    if "leistung_mw" in s.columns:
        s["leistung_mw"] = s["leistung_mw"] * factor
    if "menge_mwh" in s.columns:
        s["menge_mwh"] = s["menge_mwh"] * factor
    return s


def build_sim_deals(forward_curves: Dict, products: List[str], base_mw: float, buy_date_idx: int) -> pd.DataFrame:
    rows = []
    for pn in products:
        curve_entry = forward_curves.get(pn)
        if curve_entry is None:
            continue
        cdf = curve_entry["df"]
        s = curve_entry.get("start")
        e = curve_entry.get("end")
        prof = curve_entry.get("profile", "Base")
        if s is None or e is None:
            continue
        ai = min(buy_date_idx, len(cdf) - 1)
        rows.append({
            "produkt": pn, "kaufdatum": cdf.iloc[ai]["datetime"],
            "lieferstart": pd.Timestamp(s), "lieferende": pd.Timestamp(e),
            "leistung_mw": base_mw, "preis": float(cdf.iloc[ai]["price"]), "profil": prof,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_dca_sim(load_df, forward_curves, products, freq, window_months):
    ds = load_df.datetime.min()
    results = []
    for pn in products:
        entry = forward_curves.get(pn)
        if entry is None:
            continue
        cdf = entry["df"]
        fb = cdf[cdf.datetime < ds].copy()
        if window_months > 0:
            fb = fb[fb.datetime >= ds - pd.DateOffset(months=window_months)]
        if freq == "Wöchentlich":
            fb = fb.set_index("datetime").resample("W").last().dropna().reset_index()
        elif freq == "Monatlich":
            fb = fb.set_index("datetime").resample("ME").last().dropna().reset_index()
        if not fb.empty:
            results.append(dict(
                produkt=pn, n_käufe=len(fb), avg_preis=float(fb.price.mean()),
                min_preis=float(fb.price.min()), max_preis=float(fb.price.max()),
            ))
    return results


def validate_forward_timing(forward_curves: Dict, delivery_start) -> List[str]:
    """FIX #7: Prüft ob Forward-Daten vor Lieferperiode liegen."""
    warnings = []
    for name, entry in forward_curves.items():
        cdf = entry["df"]
        fwd_end = cdf.datetime.max()
        if fwd_end >= delivery_start:
            n_after = int((cdf.datetime >= delivery_start).sum())
            warnings.append(
                f"ℹ️ '{name}': {n_after} Forward-Datenpunkte liegen NACH Lieferbeginn "
                f"({delivery_start.date()}) – nur Daten davor werden verwendet."
            )
        fwd_start = cdf.datetime.min()
        n_before = int((cdf.datetime < delivery_start).sum())
        if n_before == 0:
            warnings.append(f"⚠️ '{name}': KEINE Daten vor Lieferbeginn!")
        elif n_before < 20:
            warnings.append(f"⚠️ '{name}': Nur {n_before} Datenpunkte vor Lieferbeginn.")
    return warnings


def run_full_backtest(load_df, spot_df, forward_curves, deals_df, config, progress=None):
    """FIX #1: TX-Kosten durchgereicht.
    FIX #5: Progress-Bar korrekt berechnet.
    FIX #7: Forward-Timing validiert.
    """
    tx = config.get("tx_cost", 0.0)
    all_warnings = []

    # FIX #7: Validierung
    if forward_curves:
        all_warnings.extend(validate_forward_timing(forward_curves, load_df.datetime.min()))

    # Benchmark
    empty = pd.DataFrame(columns=["produkt", "lieferstart", "lieferende", "leistung_mw", "preis", "profil"])
    bench = evaluate_strategy(load_df, spot_df, empty, "100% Spot", tx_cost=0)

    results = []
    done = 0

    # FIX #5: Korrekte Gesamtanzahl berechnen
    n_deal_scenarios = 0
    n_sim_scenarios = 0
    n_dca_scenarios = 0

    sim_shares = config.get("sim_shares", [1.0])

    if deals_df is not None and not deals_df.empty:
        n_deal_scenarios = 1 + len([f for f in sim_shares if f != 1.0])

    if forward_curves:
        products = config.get("sim_products", list(forward_curves.keys()))
        products = [p for p in products if p in forward_curves]
        if products:
            n_sim_scenarios = len(sim_shares) * 5  # 5 buy timings
            n_dca_scenarios = len(sim_shares)

    total_steps = max(n_deal_scenarios + n_sim_scenarios + n_dca_scenarios, 1)

    def _progress(label=""):
        nonlocal done
        done += 1
        if progress:
            progress.progress(min(done / total_steps, 0.99), f"{done}/{total_steps} {label}")

    # ── Echte Deals ──
    if deals_df is not None and not deals_df.empty:
        r = evaluate_strategy(load_df, spot_df, deals_df, "📝 Echte Deals", tx_cost=tx)
        results.append(r)
        all_warnings.extend(r.get("warnings", []))
        _progress("Echte Deals")

        for factor in sim_shares:
            if factor == 1.0:
                continue
            scaled = scale_deals(deals_df, factor)
            r = evaluate_strategy(load_df, spot_df, scaled, f"Deals ×{factor:.1f}", tx_cost=tx)
            results.append(r)
            _progress(f"Deals ×{factor:.1f}")

    # ── Simulation aus Forward-Kurven ──
    if forward_curves:
        products = config.get("sim_products", list(forward_curves.keys()))
        products = [p for p in products if p in forward_curves]

        if products:
            # Kaufzeitpunkte basierend auf erster Kurve
            first_entry = forward_curves[products[0]]
            ds = load_df.datetime.min()
            fb = first_entry["df"][first_entry["df"].datetime < ds]
            n_dates = len(fb)

            if n_dates > 0:
                buy_idxs = [0, n_dates // 4, n_dates // 2, 3 * n_dates // 4, n_dates - 1]
                buy_labels = ["Früh", "25%", "Mitte", "75%", "Spät"]

                for mw in sim_shares:
                    for bi, bl in zip(buy_idxs, buy_labels):
                        sim = build_sim_deals(forward_curves, products, mw, bi)
                        if not sim.empty:
                            r = evaluate_strategy(load_df, spot_df, sim,
                                                  f"Sim {mw:.1f}MW {bl}", tx_cost=tx)
                            results.append(r)
                        _progress(f"Sim {mw:.1f}MW {bl}")

                # DCA
                dca_r = run_dca_sim(load_df, forward_curves, products,
                                    config.get("dca_freq", "Täglich"),
                                    config.get("dca_window_months", 0))
                if dca_r:
                    dca_deals_list = []
                    for dr in dca_r:
                        entry = forward_curves[dr["produkt"]]
                        dca_deals_list.append({
                            "produkt": f"DCA-{dr['produkt']}",
                            "lieferstart": pd.Timestamp(entry["start"]),
                            "lieferende": pd.Timestamp(entry["end"]),
                            "leistung_mw": 1.0,
                            "preis": dr["avg_preis"],
                            "profil": entry.get("profile", "Base"),
                        })
                    if dca_deals_list:
                        dca_df = pd.DataFrame(dca_deals_list)
                        for factor in sim_shares:
                            sc = scale_deals(dca_df, factor)
                            r = evaluate_strategy(load_df, spot_df, sc,
                                                  f"DCA {factor:.1f}MW", tx_cost=tx)
                            results.append(r)
                            _progress(f"DCA {factor:.1f}MW")

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
            "TX-Kosten [€]": r.get("tx_cost", 0),
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
        results=rdf, merged=bench["merged"],
        total_spot=bench["total_spot_cost"],
        demand=float(load_df.load_mwh.sum()),
        avg_spot=bench["avg_spot"],
        cum=cum_data,
        details={r["name"]: r for r in results},
        warnings=all_warnings,
    )


# ═══════════════════════════════════════════════
# DEMO – FIX #8: Realistische Dimensionierung
# ═══════════════════════════════════════════════

def generate_demo():
    rng = np.random.default_rng(42)
    d25 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    n = len(d25)
    s = 20 * np.sin(np.linspace(0, 2 * np.pi, n))

    # ~100 MWh/Tag ≈ 4.2 MW Grundlast
    ld = pd.DataFrame({"datetime": d25, "load_mwh": np.maximum(100 + s + rng.normal(0, 8, n), 20)})
    sd = pd.DataFrame({"datetime": d25, "spot_price": np.maximum(75 + s * .75 + rng.normal(0, 12, n), 5)})

    d24 = pd.bdate_range("2024-01-02", "2024-12-30")[:250]
    base_fwd = 82 + np.cumsum(rng.normal(-.02, .8, len(d24)))

    # FIX #8: Realistische MW-Werte (Last ≈ 4 MW, Deals < 4 MW)
    curves = {}
    for name, off in [("Cal-25 Base", 0), ("Q1-25 Base", 3), ("Q2-25 Base", -2),
                       ("Q3-25 Base", -5), ("Q4-25 Base", 2)]:
        prices = np.maximum(base_fwd + off + rng.normal(0, 1, len(d24)), 40)
        df = pd.DataFrame({"datetime": d24, "price": prices})
        s_d, e_d, pt = parse_product_period(name)
        curves[name] = {"df": df, "start": s_d, "end": e_d, "type": pt, "profile": "Base"}

    deals = pd.DataFrame({
        "produkt": ["Cal-25 Base", "Q1-25 Base", "Q3-25 Base"],
        "kaufdatum": pd.to_datetime(["2024-03-15", "2024-06-01", "2024-09-10"]),
        "lieferstart": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-07-01"]),
        "lieferende": pd.to_datetime(["2025-12-31", "2025-03-31", "2025-09-30"]),
        "leistung_mw": [2.0, 0.5, 0.3],  # FIX #8: realistisch
        "preis": [82.50, 85.00, 78.20],
        "profil": ["Base", "Base", "Base"],
    })
    return ld, sd, curves, deals


# ═══════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════

def _pc(v):
    return [C["pos"] if x > 0 else C["neg"] if x < 0 else C["neut"] for x in v]


def fig_costs(df, ref, suffix=""):
    fig = go.Figure(go.Bar(
        x=df.Strategie, y=df["Gesamt [€]"],
        marker_color=_pc(df["PnL vs Spot [€]"]),
        text=df["Gesamt [€]"].apply(lambda v: f"{v:,.0f}€"),
        textposition="outside", textfont_size=9))
    fig.add_hline(y=ref, line_dash="dash", line_color="red",
                  annotation_text=f"100% Spot: {ref:,.0f}€")
    fig.update_layout(title=f"Gesamtkosten{suffix}", xaxis_tickangle=-45, height=500, template=TPL)
    return fig


def fig_savings(df, suffix=""):
    fig = go.Figure(go.Bar(
        x=df.Strategie, y=df["PnL vs Spot [€]"],
        marker_color=_pc(df["PnL vs Spot [€]"]),
        text=df["PnL vs Spot [€]"].apply(lambda v: f"{v:+,.0f}€"),
        textposition="outside", textfont_size=9))
    fig.add_hline(y=0, line_color="white")
    fig.update_layout(title=f"Ersparnis vs. Spot{suffix}", xaxis_tickangle=-45, height=450, template=TPL)
    return fig


def fig_cum(merged, cum, keys, suffix=""):
    fig = go.Figure()
    if "cum_spot_only" in merged.columns:
        fig.add_trace(go.Scatter(x=merged.datetime, y=merged.cum_spot_only, mode="lines",
                                 name="100% Spot", line=dict(color="red", width=3, dash="dash")))
    for i, k in enumerate(keys):
        if k in cum:
            fig.add_trace(go.Scatter(x=merged.datetime, y=cum[k], mode="lines",
                                     name=k, line=dict(color=PAL[i % len(PAL)], width=2)))
    fig.update_layout(title=f"Kumulative Kosten{suffix}", height=500, template=TPL,
                      legend=dict(font_size=9))
    return fig


def fig_delivery(merged, suffix=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.datetime, y=merged.load_mwh, mode="lines",
                             name="Last [MWh]", line=dict(color=C["blue"], width=1.5)))
    fig.add_trace(go.Scatter(x=merged.datetime, y=merged.termin_eff_mwh, mode="lines",
                             name="Terminlieferung", fill="tozeroy",
                             fillcolor="rgba(46,204,113,0.3)", line=dict(color=C["pos"], width=1)))
    fig.add_trace(go.Scatter(x=merged.datetime, y=merged.restlast_mwh, mode="lines",
                             name="Restlast → Spot", line=dict(color=C["orange"], width=1, dash="dot")))
    if (merged.überdeckung_mwh > 0).any():
        fig.add_trace(go.Scatter(x=merged.datetime, y=merged.überdeckung_mwh, mode="lines",
                                 name="Überdeckung", line=dict(color=C["neg"], width=1, dash="dash")))
    fig.update_layout(title=f"Lieferprofil{suffix}", height=450, template=TPL)
    return fig


def fig_forward_curves(curves, deals_df=None, suffix=""):
    fig = go.Figure()
    for i, (name, entry) in enumerate(curves.items()):
        cdf = entry["df"]
        fig.add_trace(go.Scatter(x=cdf.datetime, y=cdf.price, mode="lines",
                                 name=name, line=dict(color=PAL[i % len(PAL)], width=2)))
    if deals_df is not None and "kaufdatum" in deals_df.columns:
        for _, d in deals_df.iterrows():
            if pd.notna(d.get("kaufdatum")) and pd.notna(d.get("preis")):
                fig.add_trace(go.Scatter(
                    x=[d.kaufdatum], y=[d.preis], mode="markers+text",
                    marker=dict(color=C["gold"], size=12, symbol="star"),
                    text=[d.get("produkt", "")], textposition="top center",
                    textfont=dict(size=9), showlegend=False))
    fig.update_layout(title=f"Forward-Kurven{suffix}", height=450, template=TPL, yaxis_title="€/MWh")
    return fig


# ═══════════════════════════════════════════════
# UI COMPONENT
# ═══════════════════════════════════════════════

def data_input(title, key, state_key, val_label, val_name, placeholder, unit=""):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if st.session_state[state_key] is not None:
            df = st.session_state[state_key]
            c1, c2 = st.columns([5, 1])
            c1.success(f"✅ **{len(df)}** Eintr. · {df.datetime.min().date()} → {df.datetime.max().date()} · Ø {df[val_name].mean():.1f} {unit}")
            if c2.button("🗑️", key=f"{key}_del"):
                st.session_state[state_key] = None
                st.session_state.pop(f"{key}_raw", None)
                st.rerun()
            return
        tp, tf = st.tabs(["📋 Einfügen", "📁 Datei"])
        with tp:
            txt = st.text_area("Daten einfügen", height=140, key=f"{key}_txt", placeholder=placeholder)
            if st.button("🔄 Verarbeiten", key=f"{key}_go", type="primary",
                         use_container_width=True, disabled=not txt):
                raw = parse_text(txt)
                if raw is not None and len(raw.columns) >= 2:
                    st.session_state[f"{key}_raw"] = raw
                    st.rerun()
                else:
                    st.error("❌ Mind. 2 Spalten nötig")
        with tf:
            f = st.file_uploader("Datei", ["csv", "xlsx", "xls"], key=f"{key}_f", label_visibility="collapsed")
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
                ("forward_curves", "Forward"), ("deals_df", "Deals")]:
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

    with st.container(border=True):
        st.markdown("**3️⃣ Forward-Kurven (mehrere Produkte)**")
        st.caption("Breites Format: `Datum | Cal-25 | Q1-25 | …` — oder langes: `Datum | Produkt | Preis`")

        if st.session_state.forward_curves is not None:
            fc = st.session_state.forward_curves
            c1, c2 = st.columns([5, 1])
            c1.success(f"✅ **{len(fc)} Produkte**: {', '.join(fc.keys())}")
            if c2.button("🗑️", key="fc_del"):
                st.session_state.forward_curves = None
                st.rerun()
            for name, entry in fc.items():
                cdf = entry["df"]
                st.caption(f"  • {name} ({entry.get('profile', '?')}): {len(cdf)} Tage, "
                           f"Ø {cdf.price.mean():.2f} €/MWh, "
                           f"Lieferung {entry.get('start', '?')} → {entry.get('end', '?')}")
        else:
            tp, tf = st.tabs(["📋 Einfügen", "📁 Datei"])
            with tp:
                txt = st.text_area("Forward-Daten", height=180, key="fc_txt",
                                   placeholder="Datum\tCal-25 Base\tQ1-25 Base\n02.01.2024\t82.50\t84.00\n03.01.2024\t82.80\t84.20")
                if st.button("🔄 Verarbeiten", key="fc_go", type="primary",
                             use_container_width=True, disabled=not txt):
                    raw = parse_text(txt)
                    if raw is not None:
                        curves = parse_forward_curves(raw)
                        if curves:
                            st.session_state.forward_curves = curves
                            st.rerun()
                        else:
                            st.error("❌ Produktnamen in Spalten nötig (Cal-25, Q1-25 …)")
            with tf:
                f = st.file_uploader("Forward-Datei", ["csv", "xlsx", "xls"],
                                     key="fc_f", label_visibility="collapsed")
                if f and st.button("📁 Laden", key="fc_fl", use_container_width=True):
                    raw = load_file(f)
                    if raw is not None:
                        curves = parse_forward_curves(raw)
                        if curves:
                            st.session_state.forward_curves = curves
                            st.rerun()

    st.divider()

    with st.container(border=True):
        st.markdown("**4️⃣ Echte Deals (optional)**")
        if st.session_state.deals_df is not None:
            d = st.session_state.deals_df
            c1, c2 = st.columns([5, 1])
            c1.success(f"✅ **{len(d)} Deals**")
            if c2.button("🗑️", key="d_del"):
                st.session_state.deals_df = None
                st.rerun()
            dc = [c for c in ["produkt", "kaufdatum", "lieferstart", "lieferende",
                              "leistung_mw", "menge_mwh", "preis", "profil"] if c in d.columns]
            st.dataframe(d[dc], use_container_width=True, hide_index=True)
        else:
            txt = st.text_area("Deals einfügen", height=150, key="d_txt",
                               placeholder="Produkt\tLeistung_MW\tPreis\nCal-25 Base\t2\t82.50\nQ1-25 Base\t0.5\t85.00\n\nODER mit Menge:\nProdukt\tMenge_MWh\tPreis\nCal-25 Base\t17520\t82.50")
            if st.button("🔄 Verarbeiten", key="d_go", type="primary",
                         use_container_width=True, disabled=not txt):
                raw = parse_text(txt)
                if raw is not None:
                    p = parse_deals(raw)
                    if p is not None and not p.empty and p.lieferstart.notna().sum() > 0:
                        st.session_state.deals_df = p
                        st.rerun()
                    else:
                        st.error("❌ Produkt + Preis + Zeitraum/MW/MWh nötig")

        with st.expander("🔍 Unterstützte Formate"):
            st.markdown("""
| Produktname | Zeitraum | Leistung |
|-------------|----------|----------|
| `Cal-25` | 01.01.–31.12.2025 | MW oder MWh |
| `Q1-25`–`Q4-25` | Quartal | MW oder MWh |
| `H1-25`, `H2-25` | Halbjahr | MW oder MWh |
| `Jan-25`–`Dez-25` | Monat | MW oder MWh |

**Peak**: Wird im Produktnamen erkannt → liefert nur Mo-Fr 8-20 Uhr
            """)


# ═══════════════════════════════════════════════
# PAGE: STRATEGIEN
# ═══════════════════════════════════════════════

elif page == "⚙️ Strategien":
    st.header("⚙️ Strategiekonfiguration")

    tab_sim, tab_dca, tab_deals = st.tabs(["🔄 Simulation", "📊 DCA", "📝 Deals"])

    with tab_sim:
        st.markdown("##### Produkte & Skalierung")
        fc = st.session_state.forward_curves
        if fc:
            products = st.multiselect("Produkte", list(fc.keys()),
                                      default=list(fc.keys()), key="sim_prods")
            st.session_state.config["sim_products"] = products
            scales_txt = st.text_input("MW-Faktoren", "0.5, 1.0, 1.5, 2.0, 3.0", key="sc_in")
            try:
                scales = sorted({float(x.strip()) for x in scales_txt.split(",") if float(x.strip()) > 0})
            except ValueError:
                scales = [0.5, 1.0, 2.0]
            st.session_state.config["sim_shares"] = scales
            st.caption(f"Faktoren: {', '.join(f'{s:.1f}×' for s in scales)}")
            tx = st.number_input("TX [€/MWh]", 0.0, step=0.05, format="%.2f", key="tx_in")
            st.session_state.config["tx_cost"] = tx

            n_sc = len(scales) * 5 + len(scales)  # sim + dca
            st.info(f"📐 Ca. **{n_sc} Simulationsszenarien** + Deals")
        else:
            st.warning("Forward-Kurven fehlen.")

    with tab_dca:
        st.markdown("##### Dollar Cost Averaging")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            dca_w = c1.selectbox("Fenster", ["Alle Daten", "36 Monate", "24 Monate",
                                              "18 Monate", "12 Monate", "6 Monate"], key="dca_w")
            wm = {"Alle": 0, "36": 36, "24": 24, "18": 18, "12": 12, "6": 6}
            st.session_state.config["dca_window_months"] = next(
                (v for k, v in wm.items() if k in dca_w), 0)
            dca_f = c2.selectbox("Frequenz", ["Täglich", "Wöchentlich", "Monatlich"], key="dca_f")
            st.session_state.config["dca_freq"] = dca_f

    with tab_deals:
        st.markdown("##### Echte Deals")
        dd = st.session_state.deals_df
        if dd is not None and not dd.empty:
            dc = [c for c in ["produkt", "leistung_mw", "menge_mwh", "preis",
                              "profil", "lieferstart", "lieferende"] if c in dd.columns]
            st.dataframe(dd[dc], use_container_width=True, hide_index=True)
            st.caption("Deals werden auch skaliert getestet (×0.5, ×1.5, ×2.0 …)")
        else:
            st.info("Keine Deals hinterlegt.")


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
        st.warning("⚠️ Forward-Kurven oder Deals nötig")
        st.stop()

    cfg = st.session_state.config

    # ═════════ INTERAKTIVER SLIDER ═════════
    st.subheader("🎚️ Interaktive Bewertung")

    with st.container(border=True):
        if dd is not None and not dd.empty:
            slider_val = st.slider("**Deal-Skalierung**", 0.0, 3.0, 1.0, 0.1,
                                   format="%.1f×", key="int_slider",
                                   help="1.0 = echte Deals, 0 = 100% Spot")
            scaled = scale_deals(dd, slider_val)
            r = evaluate_strategy(ld, sd, scaled, f"Deals ×{slider_val:.1f}",
                                  tx_cost=cfg.get("tx_cost", 0))

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Skalierung", f"{slider_val:.1f}×")
            c2.metric("Gesamt", f"{r['total_cost']:,.0f} €", f"{r['pnl']:+,.0f} € vs Spot")
            c3.metric("Ø Preis", f"{r['avg_price']:.2f} €/MWh")
            c4.metric("Terminanteil", f"{r['termin_pct']:.1f} %")
            c5.metric("100% Spot", f"{r['total_spot_cost']:,.0f} €")

            # Warnungen anzeigen
            for w in r.get("warnings", []):
                st.caption(w)

            st.plotly_chart(fig_delivery(r["merged"], f" (×{slider_val:.1f})"),
                           use_container_width=True, key="ch_int_del")

            if not r["deal_details"].empty:
                with st.expander("📋 Deal-Details"):
                    fmt = {"leistung_mw": "{:.1f}", "menge_mwh": "{:,.0f}",
                           "preis": "{:.2f}", "kosten": "{:,.0f}"}
                    valid_fmt = {k: v for k, v in fmt.items() if k in r["deal_details"].columns}
                    st.dataframe(r["deal_details"].style.format(valid_fmt),
                                 use_container_width=True, hide_index=True)
        else:
            st.info("Deals hinterlegen für interaktive Bewertung.")

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

        # Warnungen anzeigen
        for w in bt.get("warnings", []):
            st.warning(w) if w.startswith("⚠") else st.info(w)

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

        # Echte Deals
        if "📝 Echte Deals" in R.Strategie.values:
            dr = R[R.Strategie == "📝 Echte Deals"].iloc[0]
            rank = int(R.index[R.Strategie == "📝 Echte Deals"].tolist()[0])
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns(5)
                icon = "🟢" if dr["PnL vs Spot [€]"] > 0 else "🔴"
                c1.metric(f"{icon} Echte Deals", f"{dr['Gesamt [€]']:,.0f} €")
                c2.metric("Ersparnis", f"{dr['PnL vs Spot [€]']:+,.0f} €")
                c3.metric("Terminanteil", f"{dr['Terminanteil [%]']:.1f}%")
                c4.metric("Ø Termin", f"{dr['Ø Termin [€/MWh]']:.2f} €/MWh")
                c5.metric("Platz", f"{rank}", f"von {len(R)}")

        st.markdown("##### 🏆 Top 5")
        display_cols = ["Strategie", "Gesamt [€]", "Ø Preis [€/MWh]", "Terminanteil [%]",
                        "PnL vs Spot [€]", "Ersparnis [%]", "Überdeckung [MWh]"]
        display_cols = [c for c in display_cols if c in R.columns]
        fmt = {"Gesamt [€]": "{:,.0f}", "Ø Preis [€/MWh]": "{:.2f}",
               "Terminanteil [%]": "{:.1f}", "PnL vs Spot [€]": "{:+,.0f}",
               "Ersparnis [%]": "{:+.1f}%", "Überdeckung [MWh]": "{:,.0f}"}
        valid_fmt = {k: v for k, v in fmt.items() if k in display_cols}
        st.dataframe(R.head(5)[display_cols].style.format(valid_fmt),
                     use_container_width=True, hide_index=True)

        with st.expander("📋 Alle Ergebnisse"):
            st.dataframe(R, use_container_width=True)

        # Charts
        st.divider()
        max_s = st.slider("Max. Balken", 5, len(R), min(15, len(R)), key="ms_slider")
        show = R.head(max_s)

        tabs = st.tabs(["💰 Kosten", "📊 Ersparnis", "📈 Kumulativ",
                        "🔵 Forward", "📦 Lieferprofil"])

        with tabs[0]:
            st.plotly_chart(fig_costs(show, TS, " – BT"), use_container_width=True, key="ch_bt_cost")
        with tabs[1]:
            st.plotly_chart(fig_savings(show, " – BT"), use_container_width=True, key="ch_bt_sav")
        with tabs[2]:
            dk = [k for k in CUM if k.startswith("📝")]
            if not dk:
                dk = list(CUM.keys())[:5]
            sel = st.multiselect("Strategien", list(CUM.keys()), default=dk[:5], key="cum_sel")
            if sel and merged_ref is not None:
                st.plotly_chart(fig_cum(merged_ref, CUM, sel, " – BT"),
                               use_container_width=True, key="ch_bt_cum")
        with tabs[3]:
            if fc:
                st.plotly_chart(fig_forward_curves(fc, dd, " – BT"),
                               use_container_width=True, key="ch_bt_fwd")
        with tabs[4]:
            avail = list(details.keys())
            if avail:
                choice = st.selectbox("Strategie", avail, key="del_sel")
                if choice in details:
                    det = details[choice]
                    st.plotly_chart(fig_delivery(det["merged"], f" – {choice}"),
                                   use_container_width=True, key="ch_bt_del")
                    if not det["deal_details"].empty:
                        fmt_dd = {"leistung_mw": "{:.1f}", "menge_mwh": "{:,.0f}",
                                  "preis": "{:.2f}", "kosten": "{:,.0f}"}
                        valid_dd = {k: v for k, v in fmt_dd.items() if k in det["deal_details"].columns}
                        st.dataframe(det["deal_details"].style.format(valid_dd),
                                     use_container_width=True, hide_index=True)
                    for w in det.get("warnings", []):
                        st.caption(w)


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

    if "📝 Echte Deals" in details:
        st.divider()
        st.subheader("⭐ Bewertung Ihrer Deals")
        det = details["📝 Echte Deals"]
        rank_list = R.index[R.Strategie == "📝 Echte Deals"].tolist()
        rank_pos = rank_list[0] if rank_list else "?"

        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns(5)
            icon = "🟢" if det["pnl"] > 0 else "🔴"
            c1.metric(f"{icon} Kosten", f"{det['total_cost']:,.0f} €")
            c2.metric("Ersparnis", f"{det['pnl']:+,.0f} €", f"{det['pct']:+.1f}%")
            c3.metric("Terminanteil", f"{det['termin_pct']:.1f}%")
            c4.metric("Ø Terminpreis", f"{det['avg_termin']:.2f} €/MWh")
            c5.metric("Platz", f"{rank_pos}", f"von {len(R)}")

        if det["über_vol"] > 0:
            st.warning(f"⚠️ Überdeckung: {det['über_vol']:,.0f} MWh verkauft (Erlös: {det['über_erlös']:,.0f} €)")
        if det.get("tx_cost", 0) > 0:
            st.info(f"TX-Kosten: {det['tx_cost']:,.0f} €")

        st.plotly_chart(fig_delivery(det["merged"], " – Deals"),
                       use_container_width=True, key="ch_db_del")

        if not det["deal_details"].empty:
            st.markdown("**Deal-Details:**")
            cols_show = [c for c in ["produkt", "lieferstart", "lieferende", "leistung_mw",
                                     "menge_mwh", "preis", "profil", "perioden", "kosten"]
                         if c in det["deal_details"].columns]
            fmt_d = {"leistung_mw": "{:.1f} MW", "menge_mwh": "{:,.0f} MWh",
                     "preis": "{:.2f} €/MWh", "kosten": "{:,.0f} €"}
            valid_d = {k: v for k, v in fmt_d.items() if k in cols_show}
            st.dataframe(det["deal_details"][cols_show].style.format(valid_d),
                         use_container_width=True, hide_index=True)

        for w in det.get("warnings", []):
            st.caption(w)

    st.divider()

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

    st.subheader("💡 Empfehlung")
    if best["PnL vs Spot [€]"] > 0:
        st.success(
            f"**Optimale Strategie: {best.Strategie}**\n\n"
            f"- Kosten: **{best['Gesamt [€]']:,.0f} €** (statt {TS:,.0f} €)\n"
            f"- Ersparnis: **{best['PnL vs Spot [€]']:+,.0f} €** ({best['Ersparnis [%]']:+.1f}%)\n"
            f"- Terminanteil: **{best['Terminanteil [%]']:.1f}%**")
    else:
        st.info(f"**100% Spot wäre am günstigsten gewesen** ({TS:,.0f} €, Ø {AS:.2f} €/MWh)")

    st.divider()
    st.download_button("📥 CSV Export",
                       R.to_csv(index=False, sep=";", decimal=",").encode("utf-8"),
                       "ergebnisse.csv", "text/csv", use_container_width=True, key="dl_csv")

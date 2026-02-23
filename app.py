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
# CONFIG
# ═══════════════════════════════════════════════

st.set_page_config(page_title="⚡ Energiebeschaffung", layout="wide", page_icon="⚡")

C = dict(pos="#2ecc71", neg="#e74c3c", neut="#95a5a6", blue="#3498db",
         orange="#e67e22", purple="#9b59b6", gold="#f1c40f", teal="#1abc9c")
PAL = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#1abc9c",
       "#f39c12", "#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]
TPL = "plotly_dark"

st.markdown("""<style>
.block-container{padding-top:1rem}
div[data-testid="stMetricValue"]{font-size:1.2rem;font-weight:700}
textarea{font-family:'Courier New',monospace!important;font-size:12px!important}
.stTabs [data-baseweb="tab"]{padding:8px 14px;font-weight:600}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════

_DEF = dict(
    load_df=None, spot_df=None, forward_df=None,
    deals_df=None, schedule_df=None,
    bt=None,  # full backtest results dict
    config=dict(
        forward_shares=[i / 100 for i in range(0, 101, 10)],
        patterns=["Gleichmäßig", "DCA (Dollar Cost Averaging)"],
        n_tranches=6, tx_cost=0.0,
        dca_freq="Täglich", dca_window_months=0,  # 0 = all available
    ),
)
for _k, _v in _DEF.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v.copy() if isinstance(_v, (dict, list)) else _v


# ═══════════════════════════════════════════════
# DATA FUNCTIONS
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


# ═══════════════════════════════════════════════
# PRODUCT PARSER (for deals)
# ═══════════════════════════════════════════════

MONTHS_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "mär": 3, "mrz": 3,
    "apr": 4, "mai": 5, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "okt": 10, "oct": 10,
    "nov": 11, "dez": 12, "dec": 12,
}


def parse_product_period(name: str) -> Tuple[Optional[date], Optional[date]]:
    if not name:
        return None, None
    s = name.strip().upper().replace("/", "-").replace(" ", "-")

    def _yr(y):
        y = int(y)
        return y + 2000 if y < 100 else y

    m = re.search(r"(?:CAL|YEAR|Y|JA)-?(\d{2,4})", s)
    if m:
        yr = _yr(m.group(1))
        return date(yr, 1, 1), date(yr, 12, 31)
    m = re.search(r"Q([1-4])-?(\d{2,4})", s)
    if m:
        q, yr = int(m.group(1)), _yr(m.group(2))
        sm = (q - 1) * 3 + 1
        em = q * 3
        return date(yr, sm, 1), date(yr, em, calendar.monthrange(yr, em)[1])
    m = re.search(r"H([12])-?(\d{2,4})", s)
    if m:
        h, yr = int(m.group(1)), _yr(m.group(2))
        if h == 1:
            return date(yr, 1, 1), date(yr, 6, 30)
        return date(yr, 7, 1), date(yr, 12, 31)
    low = name.strip().lower()
    for mname, mnum in MONTHS_MAP.items():
        m2 = re.search(rf"{mname}\w*[- /]?(\d{{2,4}})", low)
        if m2:
            yr = _yr(m2.group(1))
            return date(yr, mnum, 1), date(yr, mnum, calendar.monthrange(yr, mnum)[1])
    m = re.search(r"M(\d{1,2})-?(\d{2,4})", s)
    if m:
        mn, yr = int(m.group(1)), _yr(m.group(2))
        if 1 <= mn <= 12:
            return date(yr, mn, 1), date(yr, mn, calendar.monthrange(yr, mn)[1])
    return None, None


def parse_deals(raw_df):
    if raw_df is None or raw_df.empty:
        return None
    cols = {c: c.lower().replace("_", " ").replace("-", " ") for c in raw_df.columns}
    r = pd.DataFrame()

    # Produkt
    pcol = next((o for o, l in cols.items()
                 if any(k in l for k in ("produkt", "product", "name", "kontrakt"))), None)
    r["produkt"] = raw_df[pcol].astype(str).str.strip() if pcol else raw_df.iloc[:, 0].astype(str).str.strip()

    starts, ends = zip(*[parse_product_period(p) for p in r["produkt"]])

    # Kaufdatum
    kd = next((o for o, l in cols.items()
               if any(k in l for k in ("kaufdatum", "kauf", "buy", "trade", "handel"))), None)
    if kd:
        r["kaufdatum"] = pd.to_datetime(raw_df[kd], dayfirst=True, errors="coerce")

    # Lieferstart / -ende
    sc = next((o for o, l in cols.items()
               if any(k in l for k in ("lieferstart", "start", "begin", "von", "from"))), None)
    ec = next((o for o, l in cols.items()
               if any(k in l for k in ("lieferende", "ende", "end", "bis", "to"))), None)
    r["lieferstart"] = pd.to_datetime(raw_df[sc], dayfirst=True, errors="coerce") if sc else pd.Series(starts).apply(lambda x: pd.Timestamp(x) if x else pd.NaT)
    r["lieferende"] = pd.to_datetime(raw_df[ec], dayfirst=True, errors="coerce") if ec else pd.Series(ends).apply(lambda x: pd.Timestamp(x) if x else pd.NaT)

    # Leistung
    mw_c = next((o for o, l in cols.items()
                 if any(k in l for k in ("leistung", "power", "kapaz")) or (l.strip() == "mw")), None)
    if mw_c:
        v = raw_df[mw_c]
        if v.dtype == object:
            v = v.astype(str).str.replace(",", ".")
        r["leistung_mw"] = pd.to_numeric(v, errors="coerce")

    # Menge
    mwh_c = next((o for o, l in cols.items()
                  if any(k in l for k in ("menge", "volume", "energy", "mwh"))), None)
    if mwh_c:
        v = raw_df[mwh_c]
        if v.dtype == object:
            v = v.astype(str).str.replace(",", ".")
        r["menge_mwh"] = pd.to_numeric(v, errors="coerce")

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
               if any(k in l for k in ("profil", "profile", "typ", "base", "peak"))), None)
    r["profil"] = raw_df[pf].astype(str).str.strip() if pf else r["produkt"].apply(
        lambda x: "Peak" if "peak" in x.lower() else "Base")

    # Menge berechnen falls nötig
    if "leistung_mw" in r.columns and "menge_mwh" not in r.columns:
        h = [(row.lieferende - row.lieferstart).total_seconds() / 3600 + 24
             if pd.notna(row.lieferstart) and pd.notna(row.lieferende) else np.nan
             for _, row in r.iterrows()]
        r["stunden"] = h
        r["menge_mwh"] = r.get("leistung_mw", 0) * r["stunden"]

    return r.dropna(subset=["preis"]).reset_index(drop=True)


def parse_schedule(raw_df, fwd_df=None):
    if raw_df is None or raw_df.empty:
        return None
    r = pd.DataFrame()
    dt_c = find_datetime_col(raw_df)
    if not dt_c:
        return None
    r["datum"] = pd.to_datetime(raw_df[dt_c], dayfirst=True, errors="coerce")
    cols_l = {c: c.lower().replace("_", " ").replace("-", " ") for c in raw_df.columns}

    ac = next((o for o, l in cols_l.items()
               if any(k in l for k in ("anteil", "share", "prozent", "%", "pct"))), None)
    mc = next((o for o, l in cols_l.items()
               if o != ac and any(k in l for k in ("menge", "volume", "mwh"))), None)
    if ac:
        v = raw_df[ac]
        if v.dtype == object:
            v = v.astype(str).str.replace(",", ".").str.replace("%", "")
        r["anteil_pct"] = pd.to_numeric(v, errors="coerce")
    elif mc:
        v = raw_df[mc]
        if v.dtype == object:
            v = v.astype(str).str.replace(",", ".")
        r["menge_mwh"] = pd.to_numeric(v, errors="coerce")
    else:
        second = [c for c in raw_df.columns if c != dt_c]
        if second:
            v = raw_df[second[0]]
            if v.dtype == object:
                v = v.astype(str).str.replace(",", ".").str.replace("%", "")
            r["anteil_pct"] = pd.to_numeric(v, errors="coerce")

    pc = next((o for o, l in cols_l.items()
               if o not in (dt_c, ac, mc) and any(k in l for k in ("preis", "price", "eur", "€"))), None)
    if pc:
        v = raw_df[pc]
        if v.dtype == object:
            v = v.astype(str).str.replace(",", ".").str.replace("€", "").str.strip()
        r["preis"] = pd.to_numeric(v, errors="coerce")

    # Forward-Lookup
    if fwd_df is not None:
        if "preis" not in r.columns:
            r["preis"] = np.nan
        for i, row in r.iterrows():
            if pd.isna(row.get("preis")) and pd.notna(row["datum"]):
                d = abs(fwd_df.datetime - row["datum"])
                n = d.idxmin()
                if d.loc[n] <= pd.Timedelta(days=7):
                    r.at[i, "preis"] = fwd_df.loc[n, "forward_price"]

    return r.dropna(subset=["datum"]).reset_index(drop=True)


# ═══════════════════════════════════════════════
# DCA & AS-OF FUNCTIONS
# ═══════════════════════════════════════════════

def get_dca_dates(fwd_df, delivery_start, window_months=0, freq="Täglich"):
    """Berechnet DCA-Kaufdaten und Preise."""
    fb = fwd_df[fwd_df.datetime < delivery_start].copy()
    if fb.empty:
        return fb

    if window_months > 0:
        start = delivery_start - pd.DateOffset(months=window_months)
        fb = fb[fb.datetime >= start]

    if freq == "Wöchentlich":
        fb = fb.set_index("datetime").resample("W").last().dropna().reset_index()
    elif freq == "Monatlich":
        fb = fb.set_index("datetime").resample("ME").last().dropna().reset_index()

    return fb.sort_values("datetime").reset_index(drop=True)


def compute_as_of(fwd_before, fwd_share, total_demand, spot_cost_ref):
    """Stichtagsbewertung: erwartete Gesamtkosten an jedem Tag im Beschaffungszeitraum.

    An Tag T:
    - Fixiert (Tage 1..T): Volumen × Ø realisierter Preis
    - Offen (Tage T+1..N): Restvolumen × Forward-Preis am Tag T (MtM)
    - Spot-Anteil: (1-fwd_share) × Forward-Preis am Tag T (Proxy vor Lieferung)
    """
    if fwd_before.empty or fwd_share <= 0:
        return None

    prices = fwd_before["forward_price"].values
    n = len(prices)
    fwd_vol = total_demand * fwd_share
    spot_share = 1.0 - fwd_share

    dates = fwd_before["datetime"].values
    cumavg = np.cumsum(prices) / np.arange(1, n + 1)  # rolling average
    progress = np.arange(1, n + 1) / n  # 0→1

    # Expected average forward price at each as-of date
    # = progress * realized_avg + (1-progress) * current_price
    expected_fwd_avg = progress * cumavg + (1 - progress) * prices

    # Expected total forward cost
    fwd_cost_series = fwd_vol * expected_fwd_avg

    # Spot proxy: use forward price as estimate before delivery starts
    spot_cost_series = total_demand * spot_share * prices

    total_expected = fwd_cost_series + spot_cost_series

    # Average price per MWh
    avg_price_series = total_expected / total_demand if total_demand > 0 else total_expected * 0

    return pd.DataFrame({
        "datetime": dates,
        "progress_pct": progress * 100,
        "realized_avg_fwd": cumavg,
        "current_fwd": prices,
        "expected_avg_fwd": expected_fwd_avg,
        "expected_total": total_expected,
        "avg_price": avg_price_series,
    })


# ═══════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════

def compute_weights(pat, n):
    if "Frontloaded" in pat:
        w = np.linspace(2, 0.5, n)
    elif "Backloaded" in pat:
        w = np.linspace(0.5, 2, n)
    else:
        w = np.ones(n)
    return w / w.sum()


def merge_load_spot(load_df, spot_df):
    m = pd.merge(load_df, spot_df, on="datetime", how="inner")
    if len(m) > 0:
        return m, "exakt"
    ld = load_df.assign(d=load_df.datetime.dt.date).groupby("d").agg(load_mwh=("load_mwh", "sum")).reset_index()
    sd = spot_df.assign(d=spot_df.datetime.dt.date).groupby("d").agg(spot_price=("spot_price", "mean")).reset_index()
    m = pd.merge(ld, sd, on="d", how="inner")
    if len(m) > 0:
        m["datetime"] = pd.to_datetime(m["d"])
        return m.drop(columns="d"), "täglich"
    return pd.DataFrame(), "kein Überlapp"


def calc_strategy_cost(merged, total_demand, total_spot, fwd_share, wavg_fwd, tx_cost=0):
    """Berechnet Kosten für eine fwd_share / wavg_fwd Kombination."""
    ss = 1.0 - fwd_share
    fv = total_demand * fwd_share
    fc = fv * wavg_fwd
    tx = fv * tx_cost
    sc = float((merged.load_mwh * ss * merged.spot_price).sum())
    tc = fc + sc + tx
    avg_spot_eff = sc / (total_demand * ss) if ss > 0 else 0
    ap = tc / total_demand if total_demand else 0
    pnl = total_spot - tc
    pct = pnl / total_spot * 100 if total_spot else 0
    period_cost = merged.load_mwh * fwd_share * (wavg_fwd + tx_cost) + merged.load_mwh * ss * merged.spot_price
    return dict(
        fwd_cost=fc, spot_cost=sc, tx=tx, total=tc,
        avg_price=ap, pnl=pnl, pct=pct, avg_spot_eff=avg_spot_eff,
        cum=period_cost.cumsum().values,
    )


def calc_deal_cost(deals_df, merged, total_demand, total_spot):
    if deals_df is None or deals_df.empty:
        return None
    h_per = max((merged.datetime.diff().dropna().median().total_seconds() / 3600), 0.25) if len(merged) > 1 else 24
    mc = merged.copy()
    mc["deal_mwh"] = 0.0
    mc["deal_cost"] = 0.0
    for _, d in deals_df.iterrows():
        if pd.isna(d.get("lieferstart")) or pd.isna(d.get("lieferende")) or pd.isna(d.get("preis")):
            continue
        mask = (mc.datetime >= d.lieferstart) & (mc.datetime <= d.lieferende)
        if "leistung_mw" in d and pd.notna(d.leistung_mw):
            mwh_p = d.leistung_mw * h_per
        elif "menge_mwh" in d and pd.notna(d.menge_mwh):
            mwh_p = d.menge_mwh / mask.sum() if mask.sum() > 0 else 0
        else:
            continue
        mc.loc[mask, "deal_mwh"] += mwh_p
        mc.loc[mask, "deal_cost"] += mwh_p * d.preis

    mc["deal_eff"] = mc[["deal_mwh", "load_mwh"]].min(axis=1)
    mc["spot_mwh"] = mc.load_mwh - mc.deal_eff
    ratio = np.where(mc.deal_mwh > 0, mc.deal_eff / mc.deal_mwh, 0)
    dc = float((mc.deal_cost * ratio).sum())
    sc = float((mc.spot_mwh * mc.spot_price).sum())
    dv = float(mc.deal_eff.sum())
    dp = dv / total_demand * 100 if total_demand else 0
    tc = dc + sc
    return dict(
        name=f"📝 Deals ({dp:.0f}%T/{100 - dp:.0f}%S)", muster="Deals",
        fwd_share=dp, fwd_vol=dv, fwd_cost=dc,
        spot_vol=total_demand - dv, spot_cost=sc, tx=0,
        total=tc, avg_price=tc / total_demand if total_demand else 0,
        avg_fwd=dc / dv if dv > 0 else 0,
        pnl=total_spot - tc, pct=(total_spot - tc) / total_spot * 100 if total_spot else 0,
        cum=(mc.deal_cost * ratio + mc.spot_mwh * mc.spot_price).cumsum().values,
    )


def run_backtest(load_df, spot_df, fwd_df, cfg, deals_df=None, schedule_df=None, progress=None):
    delivery_start = load_df.datetime.min()
    fwd_before = fwd_df[fwd_df.datetime < delivery_start].copy()
    if fwd_before.empty:
        raise ValueError("Keine Forward-Daten VOR Lieferbeginn!")

    merged, mode = merge_load_spot(load_df, spot_df)
    if merged.empty:
        raise ValueError("Kein zeitlicher Überlapp Last ↔ Spot!")

    merged["cost_spot"] = merged.load_mwh * merged.spot_price
    td = float(merged.load_mwh.sum())
    ts = float(merged.cost_spot.sum())
    avg_s = ts / td if td else 0
    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged["cum_spot"] = merged.cost_spot.cumsum()

    fwd_vals = fwd_before.forward_price.values
    n_fwd = len(fwd_before)

    # DCA dates
    dca_dates = get_dca_dates(fwd_before, delivery_start,
                              cfg.get("dca_window_months", 0),
                              cfg.get("dca_freq", "Täglich"))
    dca_avg = float(dca_dates.forward_price.mean()) if not dca_dates.empty else float(fwd_vals.mean())

    n_total = len(cfg["patterns"]) * len(cfg["forward_shares"])
    rows, cum_data = [], {}
    done = 0

    for pat in cfg["patterns"]:
        if "DCA" in pat:
            wavg = dca_avg
            pn = "DCA"
        else:
            pn = pat.split("(")[0].strip()
            n_t = min(cfg["n_tranches"], n_fwd)
            idx = np.round(np.linspace(0, n_fwd - 1, n_t)).astype(int)
            w = compute_weights(pat, n_t)
            wavg = float(np.dot(w, fwd_vals[idx]))

        for fs in cfg["forward_shares"]:
            ss = 1 - fs
            r = calc_strategy_cost(merged, td, ts, fs, wavg, cfg["tx_cost"])
            name = f"{pn} ({fs * 100:.0f}%T/{ss * 100:.0f}%S)"
            rows.append({
                "Strategie": name, "Muster": pn,
                "Terminanteil [%]": fs * 100, "Spotanteil [%]": ss * 100,
                "Ø Forward [€/MWh]": wavg,
                "Terminvol. [MWh]": td * fs, "Terminkosten [€]": r["fwd_cost"],
                "Spotvol. [MWh]": td * ss, "Ø Spot [€/MWh]": avg_s,
                "Spotkosten [€]": r["spot_cost"], "TX [€]": r["tx"],
                "Gesamt [€]": r["total"], "Ø Preis [€/MWh]": r["avg_price"],
                "PnL vs Spot [€]": r["pnl"], "Ersparnis [%]": r["pct"],
            })
            cum_data[name] = r["cum"]
            done += 1
            if progress:
                progress.progress(done / max(n_total, 1), f"{done}/{n_total}")

    # Deals
    dr = calc_deal_cost(deals_df, merged, td, ts) if deals_df is not None else None
    if dr:
        rows.append({
            "Strategie": dr["name"], "Muster": dr["muster"],
            "Terminanteil [%]": dr["fwd_share"], "Spotanteil [%]": 100 - dr["fwd_share"],
            "Ø Forward [€/MWh]": dr["avg_fwd"],
            "Terminvol. [MWh]": dr["fwd_vol"], "Terminkosten [€]": dr["fwd_cost"],
            "Spotvol. [MWh]": dr["spot_vol"], "Ø Spot [€/MWh]": avg_s,
            "Spotkosten [€]": dr["spot_cost"], "TX [€]": 0,
            "Gesamt [€]": dr["total"], "Ø Preis [€/MWh]": dr["avg_price"],
            "PnL vs Spot [€]": dr["pnl"], "Ersparnis [%]": dr["pct"],
        })
        cum_data[dr["name"]] = dr["cum"]

    # Schedule
    if schedule_df is not None and not schedule_df.empty:
        valid = schedule_df.dropna(subset=["datum"])
        prices, weights = [], []
        for _, row in valid.iterrows():
            p = row.get("preis")
            if pd.isna(p):
                continue
            w = row.get("anteil_pct", 100 / len(valid)) / 100
            prices.append(p)
            weights.append(w)
        if prices:
            wa = np.array(weights)
            tsh = min(wa.sum(), 1.0)
            pa = np.array(prices)
            wavg_s = float(np.dot(wa / wa.sum(), pa))
            r = calc_strategy_cost(merged, td, ts, tsh, wavg_s, cfg["tx_cost"])
            sname = f"📅 Plan ({tsh * 100:.0f}%T/{(1 - tsh) * 100:.0f}%S)"
            rows.append({
                "Strategie": sname, "Muster": "Plan",
                "Terminanteil [%]": tsh * 100, "Spotanteil [%]": (1 - tsh) * 100,
                "Ø Forward [€/MWh]": wavg_s,
                "Terminvol. [MWh]": td * tsh, "Terminkosten [€]": r["fwd_cost"],
                "Spotvol. [MWh]": td * (1 - tsh), "Ø Spot [€/MWh]": avg_s,
                "Spotkosten [€]": r["spot_cost"], "TX [€]": r["tx"],
                "Gesamt [€]": r["total"], "Ø Preis [€/MWh]": r["avg_price"],
                "PnL vs Spot [€]": r["pnl"], "Ersparnis [%]": r["pct"],
            })
            cum_data[sname] = r["cum"]

    rdf = pd.DataFrame(rows).sort_values("Gesamt [€]").reset_index(drop=True)
    rdf.index += 1

    return dict(
        results=rdf, merged=merged, total_spot=ts, demand=td,
        avg_spot=avg_s, cum=cum_data, fwd_before=fwd_before,
        merge_mode=mode, dca_avg=dca_avg, dca_dates=dca_dates,
    )


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
                  annotation_text=f"Spot: {ref:,.0f}€")
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


def fig_cum(merged, cum, keys):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.datetime, y=merged.cum_spot, mode="lines",
                             name="100% Spot", line=dict(color="red", width=3, dash="dash")))
    for i, k in enumerate(keys):
        if k in cum:
            fig.add_trace(go.Scatter(x=merged.datetime, y=cum[k], mode="lines",
                                     name=k, line=dict(color=PAL[i % len(PAL)], width=2)))
    fig.update_layout(title="Kumulative Kosten", height=500, template=TPL, legend=dict(font_size=9))
    return fig


def fig_sensitivity(df, pat, ref):
    sub = df[df.Muster == pat].sort_values("Terminanteil [%]")
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


def fig_as_of(as_of_data: Dict[str, pd.DataFrame], fwd_before):
    """Stichtagsbewertung: erwartete Kosten über den Beschaffungszeitraum."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Forward-Preis als Hintergrund
    fig.add_trace(go.Scatter(
        x=fwd_before.datetime, y=fwd_before.forward_price,
        mode="lines", name="Forward-Preis",
        line=dict(color=C["blue"], width=1, dash="dot"),
        opacity=0.4,
    ), secondary_y=True)

    for i, (label, df) in enumerate(as_of_data.items()):
        fig.add_trace(go.Scatter(
            x=df.datetime, y=df.avg_price,
            mode="lines", name=label,
            line=dict(color=PAL[i % len(PAL)], width=2.5),
            hovertemplate="%{x|%d.%m.%Y}<br>Ø Preis: %{y:.2f} €/MWh<extra>" + label + "</extra>",
        ), secondary_y=False)

    fig.update_layout(
        title="Stichtagsbewertung: Erwarteter Ø-Beschaffungspreis im Zeitverlauf",
        height=550, template=TPL,
        legend=dict(font_size=10),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Ø Beschaffungspreis [€/MWh]", secondary_y=False)
    fig.update_yaxes(title_text="Forward-Preis [€/MWh]", secondary_y=True)
    return fig


def fig_dca_vs_tranche(fwd_before, n_tranches, dca_dates):
    """Vergleich: DCA-Kaufpunkte vs. Tranchen auf der Forward-Kurve."""
    n = len(fwd_before)
    idx_t = np.round(np.linspace(0, n - 1, min(n_tranches, n))).astype(int)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fwd_before.datetime, y=fwd_before.forward_price,
                             mode="lines", name="Forward-Kurve",
                             line=dict(color=C["blue"], width=2),
                             fill="tozeroy", fillcolor="rgba(52,152,219,0.06)"))
    # Tranchen
    fig.add_trace(go.Scatter(
        x=fwd_before.iloc[idx_t].datetime, y=fwd_before.iloc[idx_t].forward_price,
        mode="markers", name=f"Tranchen ({n_tranches}×)",
        marker=dict(color=C["neg"], size=10, symbol="triangle-up")))
    # DCA
    if not dca_dates.empty:
        sample = dca_dates.iloc[::max(1, len(dca_dates) // 60)]  # max 60 points for readability
        fig.add_trace(go.Scatter(
            x=sample.datetime, y=sample.forward_price,
            mode="markers", name=f"DCA ({len(dca_dates)} Käufe)",
            marker=dict(color=C["pos"], size=5, symbol="circle", opacity=0.6)))

    avg_t = fwd_before.iloc[idx_t].forward_price.mean()
    fig.add_hline(y=avg_t, line_dash="dash", line_color=C["neg"],
                  annotation_text=f"Ø Tranchen: {avg_t:.2f}")
    if not dca_dates.empty:
        avg_d = dca_dates.forward_price.mean()
        fig.add_hline(y=avg_d, line_dash="dash", line_color=C["pos"],
                      annotation_text=f"Ø DCA: {avg_d:.2f}")

    fig.update_layout(title="Forward-Kurve: Tranchen vs. DCA", height=450, template=TPL)
    return fig


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
    fd = pd.DataFrame({"datetime": d24, "forward_price": np.maximum(82 + np.cumsum(rng.normal(-.02, .8, len(d24))), 40)})
    dd = pd.DataFrame({
        "produkt": ["Cal-25 Base", "Q1-25 Base", "Q3-25 Base"],
        "kaufdatum": pd.to_datetime(["2024-03-15", "2024-06-01", "2024-09-10"]),
        "lieferstart": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-07-01"]),
        "lieferende": pd.to_datetime(["2025-12-31", "2025-03-31", "2025-09-30"]),
        "leistung_mw": [10.0, 5.0, 3.0], "preis": [82.50, 85.00, 78.20],
        "profil": ["Base"] * 3,
        "stunden": [8760., 2160., 2208.], "menge_mwh": [87600., 10800., 6624.],
    })
    return ld, sd, fd, dd


# ═══════════════════════════════════════════════
# UI COMPONENT: DATA INPUT
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
        tp, tf = st.tabs(["📋 Einfügen (Strg+V)", "📁 Datei"])
        with tp:
            txt = st.text_area("", height=150, key=f"{key}_txt", placeholder=placeholder)
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
            tc = c1.selectbox("📅 Datum", cols, index=cols.index(at) if at in cols else 0, key=f"{key}_tc")
            vc = c2.selectbox(f"📊 {val_label}", cols, index=cols.index(av) if av in cols else min(1, len(cols) - 1), key=f"{key}_vc")
            if c3.button("✅", key=f"{key}_ok", type="primary", use_container_width=True):
                df = clean_data(raw, tc, vc, val_name)
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
for sk, lbl in [("load_df", "Last"), ("spot_df", "Spot"), ("forward_df", "Forward"),
                ("deals_df", "Deals"), ("schedule_df", "Plan")]:
    d = st.session_state[sk]
    st.sidebar.success(f"✅ {lbl} ({len(d)})") if d is not None else st.sidebar.caption(f"⏳ {lbl}")

st.sidebar.divider()
sb1, sb2 = st.sidebar.columns(2)
if sb1.button("🗑️ Reset", use_container_width=True):
    for k, v in _DEF.items():
        st.session_state[k] = v.copy() if isinstance(v, (dict, list)) else v
    st.rerun()
if sb2.button("🎲 Demo", use_container_width=True):
    ld, sd, fd, dd = generate_demo()
    st.session_state.update(load_df=ld, spot_df=sd, forward_df=fd, deals_df=dd)
    st.rerun()


# ═══════════════════════════════════════════════
# PAGE: DATENIMPORT
# ═══════════════════════════════════════════════

if page == "📥 Datenimport":
    st.header("📥 Datenimport")
    st.caption("Strg+C → Strg+V aus Excel oder Datei hochladen.")

    data_input("1️⃣  Lastprofil (Lieferperiode)", "load", "load_df",
               "Verbrauch [MWh]", "load_mwh",
               "Datum\tMWh\n01.01.2025\t120.5\n02.01.2025\t115.3", "MWh")
    data_input("2️⃣  Spotpreise (Lieferperiode)", "spot", "spot_df",
               "Preis [€/MWh]", "spot_price",
               "Datum\tEUR\n01.01.2025\t85.20\n02.01.2025\t92.10", "€/MWh")
    data_input("3️⃣  Terminmarktpreise (VOR Lieferung)", "fwd", "forward_df",
               "Forward [€/MWh]", "forward_price",
               "Datum\tForward\n02.01.2024\t78.50\n03.01.2024\t79.20", "€/MWh")

    if all(st.session_state[k] is not None for k in ("load_df", "spot_df", "forward_df")):
        st.success("✅ Pflichtdaten komplett → weiter zu **⚙️ Strategien**")


# ═══════════════════════════════════════════════
# PAGE: STRATEGIEN
# ═══════════════════════════════════════════════

elif page == "⚙️ Strategien":
    st.header("⚙️ Strategien & Beschaffung")

    tab_sim, tab_dca, tab_plan, tab_deals = st.tabs([
        "🔄 Simulation", "📊 DCA-Konfiguration", "📅 Beschaffungsplan", "📝 Echte Deals"
    ])

    # ─── SIMULATION ───
    with tab_sim:
        st.markdown("##### Terminanteile & Muster")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                mode = st.radio("", ["Standard 0–100%", "Eigene"], horizontal=True, label_visibility="collapsed")
                if mode.startswith("Standard"):
                    shares = [i / 100 for i in range(0, 101, 10)]
                else:
                    txt = st.text_input("% kommagetrennt", "0,10,20,30,40,50,60,70,80,90,100")
                    try:
                        shares = sorted({max(0., min(1., float(x) / 100)) for x in txt.split(",")})
                    except ValueError:
                        shares = [i / 100 for i in range(0, 101, 10)]
                st.caption(", ".join(f"{s:.0%}" for s in shares))
            with c2:
                patterns = st.multiselect("Muster", [
                    "Gleichmäßig", "Frontloaded (früh mehr)", "Backloaded (spät mehr)",
                    "DCA (Dollar Cost Averaging)",
                ], default=["Gleichmäßig", "DCA (Dollar Cost Averaging)"], label_visibility="collapsed")
                if not patterns:
                    patterns = ["Gleichmäßig"]
                n_tr = st.slider("Tranchen (für Nicht-DCA)", 2, 36, 6)
                tx = st.number_input("TX [€/MWh]", 0.0, step=0.05, format="%.2f")

        st.session_state.config.update(
            forward_shares=shares, patterns=patterns, n_tranches=n_tr, tx_cost=tx)
        st.info(f"📐 **{len(shares) * len(patterns)} Szenarien**")

    # ─── DCA KONFIGURATION ───
    with tab_dca:
        st.markdown("##### Dollar Cost Averaging – Einstellungen")
        st.markdown("""
> **DCA** kauft gleichmäßig über den gesamten Beschaffungszeitraum
> ein n-tel der Terminmenge an jedem Kauftag zum jeweiligen Forward-Preis.
> Dies glättet das Preisrisiko und vermeidet Market-Timing.
        """)
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                dca_w = st.selectbox("Beschaffungsfenster", [
                    "Alle verfügbaren Forward-Daten",
                    "36 Monate vor Lieferung", "24 Monate vor Lieferung",
                    "18 Monate vor Lieferung", "12 Monate vor Lieferung",
                    "6 Monate vor Lieferung",
                ], index=0)
                window_map = {"Alle": 0, "36": 36, "24": 24, "18": 18, "12": 12, "6": 6}
                wm = 0
                for k, v in window_map.items():
                    if k in dca_w:
                        wm = v
                        break
            with c2:
                dca_f = st.selectbox("Kauffrequenz", ["Täglich", "Wöchentlich", "Monatlich"])

            st.session_state.config.update(dca_window_months=wm, dca_freq=dca_f)

        # Vorschau
        if st.session_state.forward_df is not None:
            ld = st.session_state.load_df
            fd = st.session_state.forward_df
            if ld is not None:
                dca_d = get_dca_dates(fd, ld.datetime.min(), wm, dca_f)
                if not dca_d.empty:
                    st.success(
                        f"✅ DCA: **{len(dca_d)} Kaufzeitpunkte** · "
                        f"{dca_d.datetime.min().date()} → {dca_d.datetime.max().date()} · "
                        f"Ø Forward: **{dca_d.forward_price.mean():.2f} €/MWh**"
                    )
                    with st.expander("🔍 DCA-Kaufpunkte anzeigen"):
                        st.dataframe(dca_d.assign(
                            Datum=dca_d.datetime.dt.date,
                            Preis=dca_d.forward_price.round(2)
                        )[["Datum", "Preis"]].head(100), use_container_width=True, hide_index=True)

    # ─── BESCHAFFUNGSPLAN ───
    with tab_plan:
        st.markdown("##### Eigene Kaufzeitpunkte")
        st.caption("Datum + Anteil (%) + optionaler Preis. Leerer Preis → Forward-Lookup.")
        with st.container(border=True):
            if st.session_state.schedule_df is not None:
                s = st.session_state.schedule_df
                c1, c2 = st.columns([5, 1])
                c1.success(f"✅ **{len(s)} Kaufzeitpunkte**")
                if c2.button("🗑️", key="sd"):
                    st.session_state.schedule_df = None
                    st.rerun()
                st.dataframe(s, use_container_width=True, hide_index=True)
            else:
                txt = st.text_area("Plan einfügen", height=180, key="s_txt",
                                   placeholder="Datum\tAnteil_%\tPreis\n15.01.2024\t10\t\n15.03.2024\t15\t79.50\n15.05.2024\t20\t\n15.07.2024\t25\t\n15.09.2024\t20\t\n15.11.2024\t10\t")
                if st.button("🔄 Plan verarbeiten", key="s_go", type="primary",
                             use_container_width=True, disabled=not txt):
                    raw = parse_text(txt)
                    if raw is not None:
                        p = parse_schedule(raw, st.session_state.forward_df)
                        if p is not None and not p.empty:
                            st.session_state.schedule_df = p
                            st.rerun()
                        else:
                            st.error("❌ Mind. Datum + Anteil/Menge nötig.")
                    else:
                        st.error("❌ Format nicht erkannt.")

    # ─── ECHTE DEALS ───
    with tab_deals:
        st.markdown("##### Reale Beschaffungsgeschäfte")
        st.caption("Produkte wie Cal-25, Q1-25, Jan-25 werden automatisch erkannt.")
        with st.container(border=True):
            if st.session_state.deals_df is not None:
                d = st.session_state.deals_df
                c1, c2 = st.columns([5, 1])
                c1.success(f"✅ **{len(d)} Deals** · Σ {d.get('menge_mwh', pd.Series([0])).sum():,.0f} MWh")
                if c2.button("🗑️", key="dd"):
                    st.session_state.deals_df = None
                    st.rerun()
                dc = [c for c in ["produkt", "kaufdatum", "lieferstart", "lieferende",
                                  "leistung_mw", "menge_mwh", "preis", "profil"] if c in d.columns]
                st.dataframe(d[dc], use_container_width=True, hide_index=True)
            else:
                txt = st.text_area("Deals einfügen", height=200, key="d_txt",
                                   placeholder="Produkt\tKaufdatum\tLeistung_MW\tPreis\nCal-25 Base\t15.03.2024\t10\t82.50\nQ1-25 Base\t01.06.2024\t5\t85.00\nQ3-25 Peak\t15.07.2024\t3\t95.20")
                if st.button("🔄 Deals verarbeiten", key="d_go", type="primary",
                             use_container_width=True, disabled=not txt):
                    raw = parse_text(txt)
                    if raw is not None:
                        p = parse_deals(raw)
                        if p is not None and not p.empty and p.lieferstart.notna().sum() > 0:
                            st.session_state.deals_df = p
                            st.rerun()
                        else:
                            st.error("❌ Produkt + Preis nötig. Produktnamen: Cal-25, Q1-25, Jan-25 etc.")
                    else:
                        st.error("❌ Format nicht erkannt.")

        with st.expander("🔍 Unterstützte Produktformate"):
            st.markdown("""
| Eingabe | Zeitraum |
|---------|----------|
| `Cal-25`, `Y25` | 01.01.–31.12.2025 |
| `Q1-25` … `Q4-25` | Quartal |
| `H1-25`, `H2-25` | Halbjahr |
| `Jan-25` … `Dez-25` | Monat |
            """)


# ═══════════════════════════════════════════════
# PAGE: ANALYSE
# ═══════════════════════════════════════════════

elif page == "🔬 Analyse":
    st.header("🔬 Analyse & Interaktive Exploration")

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

    # ═══════════════════════════════════════
    # INTERAKTIVER SLIDER
    # ═══════════════════════════════════════

    st.subheader("🎚️ Interaktive Schnellbewertung")
    st.caption("Sofortige Berechnung – ohne Backtesting-Start nötig.")

    with st.container(border=True):
        # Pre-compute merged and basics
        merged_quick, _ = merge_load_spot(ld, sd)
        if merged_quick.empty:
            st.error("Kein Überlapp Last ↔ Spot")
            st.stop()

        td_q = float(merged_quick.load_mwh.sum())
        ts_q = float((merged_quick.load_mwh * merged_quick.spot_price).sum())
        as_q = ts_q / td_q if td_q else 0

        delivery_start = ld.datetime.min()
        fb_q = fd[fd.datetime < delivery_start]
        if fb_q.empty:
            st.error("Keine Forward-Daten vor Lieferbeginn!")
            st.stop()

        # DCA avg
        dca_d = get_dca_dates(fd, delivery_start, cfg.get("dca_window_months", 0), cfg.get("dca_freq", "Täglich"))
        dca_avg_q = float(dca_d.forward_price.mean()) if not dca_d.empty else float(fb_q.forward_price.mean())

        # Tranche avg
        n_t = min(cfg["n_tranches"], len(fb_q))
        idx_t = np.round(np.linspace(0, len(fb_q) - 1, n_t)).astype(int)
        tranche_avg_q = float(fb_q.iloc[idx_t].forward_price.mean())

        c1, c2 = st.columns([3, 1])
        with c1:
            slider_fs = st.slider("**Terminanteil**", 0, 100, 50, 1,
                                  format="%d %%", key="interactive_slider")
        with c2:
            slider_mode = st.radio("Preisbasis", ["DCA", "Tranchen"], horizontal=True)

        fs = slider_fs / 100
        wavg = dca_avg_q if slider_mode == "DCA" else tranche_avg_q
        r = calc_strategy_cost(merged_quick, td_q, ts_q, fs, wavg, cfg["tx_cost"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Terminanteil", f"{slider_fs} %")
        c2.metric("Gesamt", f"{r['total']:,.0f} €",
                  f"{r['pnl']:+,.0f} € vs Spot")
        c3.metric("Ø Preis", f"{r['avg_price']:.2f} €/MWh",
                  f"{r['avg_price'] - as_q:+.2f} vs Spot")
        c4.metric("Ø Forward", f"{wavg:.2f} €/MWh")
        c5.metric("100% Spot", f"{ts_q:,.0f} €")

        # Cost breakdown bar
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Terminkosten", x=["Strategie"],
                                 y=[r["fwd_cost"]], marker_color=C["blue"]))
        fig_bar.add_trace(go.Bar(name="Spotkosten", x=["Strategie"],
                                 y=[r["spot_cost"]], marker_color=C["orange"]))
        if r["tx"] > 0:
            fig_bar.add_trace(go.Bar(name="TX", x=["Strategie"],
                                     y=[r["tx"]], marker_color=C["neut"]))
        fig_bar.add_trace(go.Bar(name="100% Spot", x=["Benchmark"],
                                 y=[ts_q], marker_color="rgba(231,76,60,0.5)"))
        fig_bar.update_layout(barmode="stack", height=300, template=TPL,
                              title="Kostenaufschlüsselung", showlegend=True)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ═══════════════════════════════════════
    # FULL BACKTEST
    # ═══════════════════════════════════════

    st.subheader("🔬 Vollständiges Backtesting")

    n_sim = len(cfg["forward_shares"]) * len(cfg["patterns"])
    extras = []
    if dd is not None:
        extras.append(f"📝 {len(dd)} Deals")
    if sch is not None:
        extras.append(f"📅 {len(sch)} Plan")

    st.info(f"**{n_sim + len(extras)} Strategien** werden berechnet" +
            (f" inkl. {', '.join(extras)}" if extras else ""))

    if st.button("🚀 Backtesting starten", type="primary", use_container_width=True):
        prog = st.progress(0, "Starte …")
        try:
            bt = run_backtest(ld, sd, fd, cfg, dd, sch, prog)
        except ValueError as e:
            prog.empty()
            st.error(f"❌ {e}")
            st.stop()
        prog.empty()
        st.session_state.bt = bt
        st.success(f"✅ **{len(bt['results'])}** Szenarien (Merge: {bt['merge_mode']})")
        st.rerun()

    # ── RESULTS ──
    bt = st.session_state.bt
    if bt is not None:
        R = bt["results"]
        M = bt["merged"]
        TS = bt["total_spot"]
        TD = bt["demand"]
        AS = bt["avg_spot"]
        CUM = bt["cum"]
        FB = bt["fwd_before"]
        DCA_D = bt.get("dca_dates", pd.DataFrame())
        DCA_A = bt.get("dca_avg", 0)

        best = R.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Beste", best.Strategie.split("(")[-1].rstrip(")"),
                  f"{best['PnL vs Spot [€]']:+,.0f} €")
        c2.metric("Kosten", f"{best['Gesamt [€]']:,.0f} €", f"{best['Ersparnis [%]']:+.1f}%")
        c3.metric("100% Spot", f"{TS:,.0f} €", f"Ø {AS:.2f} €/MWh")
        c4.metric("Bedarf", f"{TD:,.0f} MWh")

        # Special strategies highlight
        special = R[R.Muster.isin(["Deals", "Plan"])]
        if not special.empty:
            st.markdown("##### ⭐ Eigene Strategien")
            st.dataframe(special[["Strategie", "Gesamt [€]", "Ø Preis [€/MWh]",
                                  "PnL vs Spot [€]", "Ersparnis [%]"]].style.format({
                "Gesamt [€]": "{:,.0f}", "Ø Preis [€/MWh]": "{:.2f}",
                "PnL vs Spot [€]": "{:+,.0f}", "Ersparnis [%]": "{:+.1f}%"}),
                use_container_width=True, hide_index=True)

        st.markdown("##### 🏆 Top 5")
        st.dataframe(R.head(5)[["Strategie", "Gesamt [€]", "Ø Preis [€/MWh]",
                                "PnL vs Spot [€]", "Ersparnis [%]"]].style.format({
            "Gesamt [€]": "{:,.0f}", "Ø Preis [€/MWh]": "{:.2f}",
            "PnL vs Spot [€]": "{:+,.0f}", "Ersparnis [%]": "{:+.1f}%"}),
            use_container_width=True, hide_index=True)

        with st.expander("📋 Alle Ergebnisse"):
            st.dataframe(R, use_container_width=True)

        # ── Charts ──
        st.divider()
        max_s = st.slider("Max. Balken", 5, len(R), min(15, len(R)), key="ms")
        show = R.head(max_s)

        tabs = st.tabs(["💰 Kosten", "📊 Ersparnis", "📈 Kumulativ",
                        "🔵 Forward/DCA", "🎯 Sensitivität"])
        with tabs[0]:
            st.plotly_chart(fig_costs(show, TS), use_container_width=True)
        with tabs[1]:
            st.plotly_chart(fig_savings(show), use_container_width=True)
        with tabs[2]:
            dk = [k for k in list(CUM.keys()) if k.startswith(("📝", "📅"))]
            if not dk:
                dk = list(CUM.keys())[:5]
            sel = st.multiselect("Strategien", list(CUM.keys()), default=dk[:5])
            if sel:
                st.plotly_chart(fig_cum(M, CUM, sel), use_container_width=True)
        with tabs[3]:
            st.plotly_chart(fig_dca_vs_tranche(FB, cfg["n_tranches"], DCA_D), use_container_width=True)
        with tabs[4]:
            for pat in cfg["patterns"]:
                pn = "DCA" if "DCA" in pat else pat.split("(")[0].strip()
                f = fig_sensitivity(R, pn, TS)
                if f:
                    st.plotly_chart(f, use_container_width=True)

        # ═══════════════════════════════════════
        # STICHTAGSBEWERTUNG (AS-OF)
        # ═══════════════════════════════════════

        st.divider()
        st.subheader("📅 Stichtagsbewertung (As-of Analyse)")
        st.markdown("""
> An jedem Tag im Beschaffungszeitraum wird berechnet:
> - **Bereits fixiert:** bisherige Käufe zum realisierten Ø-Preis
> - **Noch offen:** Restvolumen bewertet zum aktuellen Forward-Preis (Mark-to-Market)
> - **Ergebnis:** Erwarteter Ø-Beschaffungspreis an jedem Stichtag
>
> → Zeigt, ob frühes oder spätes Fixieren vorteilhaft gewesen wäre.
        """)

        with st.container(border=True):
            as_of_shares = st.multiselect(
                "Terminanteile für Stichtagsbewertung",
                [20, 30, 40, 50, 60, 70, 80],
                default=[30, 50, 70],
            )

            if as_of_shares and not FB.empty:
                as_of_data = {}
                for sh in sorted(as_of_shares):
                    aof = compute_as_of(FB, sh / 100, TD, TS)
                    if aof is not None:
                        as_of_data[f"{sh}% Termin"] = aof

                if as_of_data:
                    # Add reference lines
                    fig_aof = fig_as_of(as_of_data, FB)

                    # Add actual spot average as horizontal line
                    fig_aof.add_hline(y=AS, line_dash="dash", line_color="red",
                                     annotation_text=f"Ø Spot (realisiert): {AS:.2f}",
                                     secondary_y=False)

                    # Add DCA average
                    if DCA_A > 0:
                        fig_aof.add_hline(y=DCA_A, line_dash="dot", line_color=C["pos"],
                                          annotation_text=f"DCA Ø: {DCA_A:.2f}",
                                          secondary_y=False)

                    st.plotly_chart(fig_aof, use_container_width=True)

                    # Key insights
                    st.markdown("##### Erkenntnisse")
                    for label, aof_df in as_of_data.items():
                        best_date = aof_df.loc[aof_df.avg_price.idxmin()]
                        worst_date = aof_df.loc[aof_df.avg_price.idxmax()]
                        final = aof_df.iloc[-1]
                        st.markdown(
                            f"**{label}:** Bester Stichtag {pd.Timestamp(best_date.datetime).strftime('%d.%m.%Y')} "
                            f"(Ø {best_date.avg_price:.2f} €/MWh) · "
                            f"Schlechtester {pd.Timestamp(worst_date.datetime).strftime('%d.%m.%Y')} "
                            f"(Ø {worst_date.avg_price:.2f} €/MWh) · "
                            f"Final: Ø {final.avg_price:.2f} €/MWh"
                        )
                else:
                    st.warning("Keine Stichtagsbewertung möglich.")

        # ── Marktphasen ──
        st.divider()
        st.subheader("📉 Marktphasen-Analyse")
        st.caption("Wie hat sich die Forward-Kurve im Beschaffungszeitraum verhalten?")

        if not FB.empty and len(FB) > 30:
            fb_c = FB.copy()
            fb_c["rolling_avg_30"] = fb_c.forward_price.rolling(30, min_periods=10).mean()
            fb_c["rolling_std_30"] = fb_c.forward_price.rolling(30, min_periods=10).std()
            fb_c["pct_change_30d"] = fb_c.forward_price.pct_change(30) * 100

            fig_mp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.6, 0.4], vertical_spacing=0.08,
                                   subplot_titles=["Forward-Preis + 30-Tage-Durchschnitt",
                                                   "30-Tage-Volatilität & Preisänderung"])
            fig_mp.add_trace(go.Scatter(x=fb_c.datetime, y=fb_c.forward_price,
                                        mode="lines", name="Forward", line=dict(color=C["blue"], width=1.5)), row=1, col=1)
            fig_mp.add_trace(go.Scatter(x=fb_c.datetime, y=fb_c.rolling_avg_30,
                                        mode="lines", name="30d Ø", line=dict(color="orange", width=2, dash="dot")), row=1, col=1)
            fig_mp.add_trace(go.Bar(x=fb_c.datetime, y=fb_c.rolling_std_30,
                                    name="30d Volatilität", marker_color="rgba(155,89,182,0.5)"), row=2, col=1)
            fig_mp.add_trace(go.Scatter(x=fb_c.datetime, y=fb_c.pct_change_30d,
                                        mode="lines", name="30d Δ%", line=dict(color=C["orange"], width=1)), row=2, col=1)
            fig_mp.add_hline(y=0, line_color="white", line_width=0.5, row=2, col=1)
            fig_mp.update_layout(height=600, template=TPL, showlegend=True)
            st.plotly_chart(fig_mp, use_container_width=True)


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
    FB = bt["fwd_before"]
    DCA_A = bt.get("dca_avg", 0)
    cfg = st.session_state.config

    best, worst = R.iloc[0], R.iloc[-1]

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Optimum", f"{best['Terminanteil [%]']:.0f}%T / {best['Spotanteil [%]']:.0f}%S")
        c2.metric("Kosten", f"{best['Gesamt [€]']:,.0f} €", f"{best['PnL vs Spot [€]']:+,.0f} € vs Spot")
        c3.metric("Schlechteste", f"{worst['Terminanteil [%]']:.0f}%T / {worst['Spotanteil [%]']:.0f}%S",
                  f"{worst['PnL vs Spot [€]']:+,.0f} €")
        c4.metric("Spanne", f"{worst['Gesamt [€]'] - best['Gesamt [€]']:,.0f} €")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bedarf", f"{TD:,.0f} MWh")
        c2.metric("100% Spot", f"{TS:,.0f} €")
        c3.metric("Ø Spot", f"{AS:.2f} €/MWh")
        if FB is not None and not FB.empty:
            af = FB.forward_price.mean()
            c4.metric("Ø Forward", f"{af:.2f} €/MWh", f"{af - AS:+.2f} vs Spot")

    # DCA vs Tranche comparison
    if DCA_A > 0:
        n_t = min(cfg["n_tranches"], len(FB))
        idx = np.round(np.linspace(0, len(FB) - 1, n_t)).astype(int)
        tranche_avg = float(FB.iloc[idx].forward_price.mean())
        st.divider()
        st.subheader("📊 DCA vs. Tranchen-Beschaffung")
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("DCA Ø-Preis", f"{DCA_A:.2f} €/MWh")
            c2.metric(f"Tranchen ({n_t}×) Ø-Preis", f"{tranche_avg:.2f} €/MWh")
            diff = DCA_A - tranche_avg
            c3.metric("Differenz", f"{diff:+.2f} €/MWh",
                      "DCA günstiger" if diff < 0 else "Tranchen günstiger")

    # Special strategies
    special = R[R.Muster.isin(["Deals", "Plan"])]
    if not special.empty:
        st.divider()
        st.subheader("⭐ Eigene Strategien vs. Simulation")
        for _, row in special.iterrows():
            rank = R.index[R.Strategie == row.Strategie].tolist()
            rp = rank[0] if rank else "?"
            icon = "🟢" if row["PnL vs Spot [€]"] > 0 else "🔴"
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"{icon} {row.Strategie.split('(')[0].strip()}", f"{row['Gesamt [€]']:,.0f} €")
                c2.metric("Ersparnis", f"{row['PnL vs Spot [€]']:+,.0f} €", f"{row['Ersparnis [%]']:+.1f}%")
                c3.metric("Ø Preis", f"{row['Ø Preis [€/MWh]']:.2f} €/MWh")
                c4.metric("Ranking", f"Platz {rp}", f"von {len(R)}")

    st.divider()

    # Risk
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
            f"- Ø Preis: **{best['Ø Preis [€/MWh]']:.2f} €/MWh** (statt {AS:.2f})")
    else:
        st.info(
            f"**100% Spot wäre am günstigsten gewesen.**\n\n"
            f"- Spot: **{TS:,.0f} €** (Ø {AS:.2f} €/MWh)\n"
            f"- Nächstbeste: {best.Strategie} → {best['Gesamt [€]']:,.0f} €")

    # Muster
    muster = R.Muster.unique()
    if len(muster) > 1:
        st.divider()
        st.subheader("🗺️ Vergleich nach Muster")
        for m in muster:
            sub = R[R.Muster == m].sort_values("Terminanteil [%]")
            if not sub.empty:
                with st.expander(f"**{m}** ({len(sub)})"):
                    st.dataframe(sub[["Terminanteil [%]", "Gesamt [€]", "Ø Preis [€/MWh]",
                                     "PnL vs Spot [€]", "Ersparnis [%]"]].style.format({
                        "Gesamt [€]": "{:,.0f}", "Ø Preis [€/MWh]": "{:.2f}",
                        "PnL vs Spot [€]": "{:+,.0f}", "Ersparnis [%]": "{:+.1f}%"}),
                        use_container_width=True, hide_index=True)

    st.divider()
    st.download_button("📥 CSV Export", R.to_csv(index=False, sep=";", decimal=",").encode("utf-8"),
                       "ergebnisse.csv", "text/csv", use_container_width=True)

"""
Energiebeschaffungs-Simulation & Backtesting App
=================================================
Eine professionelle Streamlit-Web-App zur Simulation und Backtesting
von Energiebeschaffungsstrategien.

Autor: DeepAgent
Version: 1.0.0
Deployment: Streamlit Cloud Ready

Datenstruktur-Beispiele:
------------------------
Lastprofil (load_profiles):
    timestamp, value, unit
    2024-01-01 00:00, 150.5, MWh
    2024-01-01 00:15, 148.2, MWh

Spotpreise (spot_prices):
    timestamp, price
    2024-01-01 00:00, 85.50
    2024-01-01 01:00, 82.30

Forward-Preise (forward_prices):
    date, product, delivery_period, price
    2024-01-01, Base, 2025, 95.00
    2024-01-01, Peak, Q1-2025, 105.50
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
import io
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURATION & KONSTANTEN
# ============================================================================

APP_TITLE = "⚡ Energiebeschaffungs-Backtesting"
APP_ICON = "⚡"
APP_VERSION = "2.0.0"
DB_PATH = "energy_backtesting.db"
ENCRYPTION_SALT = b'energy_backtesting_salt_v1'

# Strategietypen (erweitert)
STRATEGY_TYPES = {
    "gleichmäßig": "Gleichmäßige Verteilung über den Zeitraum",
    "frontloaded": "Höhere Fixierung zu Beginn (70/30)",
    "backloaded": "Höhere Fixierung zum Ende (30/70)",
    "regelbasiert": "Basierend auf Preisschwellen (kauft mehr unter Schwelle)",
    "gleitender_durchschnitt": "Kauft wenn Preis unter gleitendem Durchschnitt",
    "custom": "Benutzerdefinierte Verteilung"
}

# Produkte
ENERGY_PRODUCTS = ["Base", "Peak", "Jahresband", "Quartalsband", "Monatsband"]

# ============================================================================
# VERSCHLÜSSELUNG
# ============================================================================

def get_encryption_key(password: str = "default_key") -> bytes:
    """Generiert einen Verschlüsselungsschlüssel aus einem Passwort."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=ENCRYPTION_SALT,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt_data(data: str, key: bytes) -> str:
    """Verschlüsselt Daten mit Fernet."""
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str, key: bytes) -> str:
    """Entschlüsselt Daten mit Fernet."""
    try:
        f = Fernet(key)
        return f.decrypt(encrypted_data.encode()).decode()
    except Exception:
        return encrypted_data  # Fallback für unverschlüsselte Daten

# ============================================================================
# DATENBANK-FUNKTIONEN
# ============================================================================

def init_database():
    """Initialisiert die SQLite-Datenbank mit allen Tabellen."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Lastprofile
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS load_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT DEFAULT 'MWh',
            profile_name TEXT DEFAULT 'default',
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Spotpreise
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS spot_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            price REAL NOT NULL,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Forward-Preise
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forward_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            product TEXT NOT NULL,
            delivery_period TEXT NOT NULL,
            price REAL NOT NULL,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Simulationen
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            strategy_config TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Indizes für Performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_load_timestamp ON load_profiles(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_spot_timestamp ON spot_prices(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_forward_date ON forward_prices(date)')
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Erstellt eine Datenbankverbindung."""
    return sqlite3.connect(DB_PATH)

def save_load_profile(df: pd.DataFrame, profile_name: str = "default"):
    """Speichert Lastprofil in der Datenbank."""
    conn = get_db_connection()
    df_to_save = df.copy()
    df_to_save['profile_name'] = profile_name
    df_to_save['uploaded_at'] = datetime.now().isoformat()
    df_to_save.to_sql('load_profiles', conn, if_exists='append', index=False)
    conn.close()

def save_spot_prices(df: pd.DataFrame):
    """Speichert Spotpreise in der Datenbank."""
    conn = get_db_connection()
    df_to_save = df.copy()
    df_to_save['uploaded_at'] = datetime.now().isoformat()
    df_to_save.to_sql('spot_prices', conn, if_exists='append', index=False)
    conn.close()

def save_forward_prices(df: pd.DataFrame):
    """Speichert Forward-Preise in der Datenbank."""
    conn = get_db_connection()
    df_to_save = df.copy()
    df_to_save['uploaded_at'] = datetime.now().isoformat()
    df_to_save.to_sql('forward_prices', conn, if_exists='append', index=False)
    conn.close()

def save_simulation(name: str, strategy_config: dict, results: dict):
    """Speichert Simulationsergebnisse in der Datenbank."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO simulations (name, strategy_config, results_json, created_at)
        VALUES (?, ?, ?, ?)
    ''', (name, json.dumps(strategy_config), json.dumps(results), datetime.now().isoformat()))
    conn.commit()
    conn.close()

def load_data_from_db(table: str) -> pd.DataFrame:
    """Lädt Daten aus der Datenbank."""
    conn = get_db_connection()
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

def load_simulations() -> pd.DataFrame:
    """Lädt alle Simulationen aus der Datenbank."""
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM simulations ORDER BY created_at DESC", conn)
    conn.close()
    return df

def clear_table(table: str):
    """Löscht alle Daten aus einer Tabelle."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()

def get_db_stats() -> dict:
    """Gibt Statistiken über die Datenbank zurück."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    stats = {}
    for table in ['load_profiles', 'spot_prices', 'forward_prices', 'simulations']:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cursor.fetchone()[0]
    
    conn.close()
    return stats

# ============================================================================
# DATEN-UPLOAD & VERARBEITUNG
# ============================================================================

def detect_delimiter(content: str) -> str:
    """Erkennt automatisch das Trennzeichen."""
    delimiters = [';', ',', '\t', '|']
    counts = {d: content.count(d) for d in delimiters}
    return max(counts, key=counts.get)

def parse_uploaded_data(content: str, file_type: str = 'csv') -> pd.DataFrame:
    """Parst hochgeladene Daten aus verschiedenen Formaten."""
    try:
        if file_type == 'csv':
            delimiter = detect_delimiter(content)
            df = pd.read_csv(io.StringIO(content), sep=delimiter)
        elif file_type == 'xlsx':
            df = pd.read_excel(io.BytesIO(content))
        else:
            delimiter = detect_delimiter(content)
            df = pd.read_csv(io.StringIO(content), sep=delimiter)
        return df
    except Exception as e:
        st.error(f"Fehler beim Parsen: {str(e)}")
        return pd.DataFrame()

def validate_load_profile(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validiert ein Lastprofil mit erweiterten Prüfungen."""
    required_cols = ['timestamp', 'value']
    
    # Grundlegende Spaltenprüfung
    for col in required_cols:
        if col not in df.columns:
            return False, f"Spalte '{col}' fehlt"
    
    # Leere DataFrame
    if df.empty:
        return False, "Keine Daten vorhanden"
    
    # Prüfung auf NULL-Werte
    null_count = df['value'].isnull().sum()
    if null_count > 0:
        return False, f"Verbrauchswerte enthalten {null_count} leere Einträge"
    
    # Timestamp-Parsing
    try:
        timestamps = pd.to_datetime(df['timestamp'])
        # Prüfung auf ungültige Timestamps
        if timestamps.isnull().any():
            return False, "Einige Zeitstempel konnten nicht geparst werden"
    except Exception as e:
        return False, f"Zeitstempel-Fehler: {str(e)}"
    
    # Numerische Werte prüfen
    try:
        values = pd.to_numeric(df['value'], errors='coerce')
        if values.isnull().any():
            return False, "Einige Verbrauchswerte sind nicht numerisch"
        if (values < 0).any():
            return False, "Negative Verbrauchswerte gefunden"
    except Exception as e:
        return False, f"Werte-Fehler: {str(e)}"
    
    # Statistiken
    stats = f"✓ {len(df)} Einträge, {values.min():.1f} - {values.max():.1f} MWh"
    return True, f"Validierung erfolgreich: {stats}"

def validate_spot_prices(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validiert Spotpreise mit erweiterten Prüfungen."""
    required_cols = ['timestamp', 'price']
    
    # Grundlegende Spaltenprüfung
    for col in required_cols:
        if col not in df.columns:
            return False, f"Spalte '{col}' fehlt"
    
    # Leere DataFrame
    if df.empty:
        return False, "Keine Daten vorhanden"
    
    # Prüfung auf NULL-Werte
    null_count = df['price'].isnull().sum()
    if null_count > 0:
        return False, f"Preise enthalten {null_count} leere Einträge"
    
    # Timestamp-Parsing
    try:
        timestamps = pd.to_datetime(df['timestamp'])
        if timestamps.isnull().any():
            return False, "Einige Zeitstempel konnten nicht geparst werden"
    except Exception as e:
        return False, f"Zeitstempel-Fehler: {str(e)}"
    
    # Numerische Werte prüfen
    try:
        prices = pd.to_numeric(df['price'], errors='coerce')
        if prices.isnull().any():
            return False, "Einige Preise sind nicht numerisch"
        # Negative Preise sind bei Strom möglich (können vorkommen)
        if (prices < -500).any() or (prices > 5000).any():
            return False, "Unrealistische Preise gefunden (außerhalb -500 bis 5000 €/MWh)"
    except Exception as e:
        return False, f"Preis-Fehler: {str(e)}"
    
    # Statistiken
    stats = f"✓ {len(df)} Einträge, Ø {prices.mean():.2f} €/MWh"
    return True, f"Validierung erfolgreich: {stats}"

def validate_forward_prices(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validiert Forward-Preise mit erweiterten Prüfungen."""
    required_cols = ['date', 'product', 'delivery_period', 'price']
    
    # Grundlegende Spaltenprüfung
    for col in required_cols:
        if col not in df.columns:
            return False, f"Spalte '{col}' fehlt"
    
    # Leere DataFrame
    if df.empty:
        return False, "Keine Daten vorhanden"
    
    # Prüfung auf NULL-Werte in price
    null_count = df['price'].isnull().sum()
    if null_count > 0:
        return False, f"Preise enthalten {null_count} leere Einträge"
    
    # Datum-Parsing
    try:
        dates = pd.to_datetime(df['date'])
        if dates.isnull().any():
            return False, "Einige Datumsangaben konnten nicht geparst werden"
    except Exception as e:
        return False, f"Datums-Fehler: {str(e)}"
    
    # Numerische Werte prüfen
    try:
        prices = pd.to_numeric(df['price'], errors='coerce')
        if prices.isnull().any():
            return False, "Einige Preise sind nicht numerisch"
    except Exception as e:
        return False, f"Preis-Fehler: {str(e)}"
    
    # Statistiken
    products = df['product'].unique()
    stats = f"✓ {len(df)} Einträge, {len(products)} Produkte"
    return True, f"Validierung erfolgreich: {stats}"

def detect_time_interval(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> str:
    """Erkennt das Zeitintervall der Daten."""
    try:
        timestamps = pd.to_datetime(df[timestamp_col]).sort_values()
        if len(timestamps) < 2:
            return "unbekannt"
        
        diff = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds() / 60
        
        if diff <= 15:
            return "15-Minuten"
        elif diff <= 60:
            return "stündlich"
        elif diff <= 1440:
            return "täglich"
        else:
            return "unbekannt"
    except Exception:
        return "unbekannt"

# ============================================================================
# BACKTESTING-ENGINE (ERWEITERT)
# ============================================================================

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Berechnet die Sharpe Ratio."""
    if returns.std() == 0:
        return 0.0
    excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
    return (excess_returns / returns.std()) * np.sqrt(252)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Berechnet die Sortino Ratio (nur Downside-Volatilität)."""
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    excess_returns = returns.mean() - (risk_free_rate / 252)
    return (excess_returns / downside_returns.std()) * np.sqrt(252)

def calculate_max_drawdown(cumulative_returns: pd.Series) -> Tuple[float, int]:
    """Berechnet den Maximum Drawdown und dessen Dauer."""
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    max_dd = drawdown.min()
    
    # Dauer des max Drawdowns
    in_drawdown = drawdown < 0
    drawdown_groups = (~in_drawdown).cumsum()
    drawdown_lengths = in_drawdown.groupby(drawdown_groups).sum()
    max_dd_duration = int(drawdown_lengths.max()) if len(drawdown_lengths) > 0 else 0
    
    return abs(max_dd) * 100, max_dd_duration

def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Berechnet Value at Risk."""
    return abs(np.percentile(returns, (1 - confidence) * 100))

def calculate_fixing_schedule(
    total_volume: float,
    fixing_quota: float,
    strategy_type: str,
    periods: int,
    custom_weights: Optional[List[float]] = None,
    price_threshold: Optional[float] = None,
    prices: Optional[pd.Series] = None
) -> List[float]:
    """Berechnet den Fixierungsplan basierend auf der Strategie."""
    
    fixed_volume = total_volume * (fixing_quota / 100)
    
    if strategy_type == "gleichmäßig":
        weights = [1/periods] * periods
    
    elif strategy_type == "frontloaded":
        # 70% in der ersten Hälfte, 30% in der zweiten
        mid = periods // 2
        first_half = [0.7 / mid] * mid if mid > 0 else [0.7]
        second_half = [0.3 / (periods - mid)] * (periods - mid) if (periods - mid) > 0 else [0.3]
        weights = first_half + second_half
    
    elif strategy_type == "backloaded":
        # 30% in der ersten Hälfte, 70% in der zweiten
        mid = periods // 2
        first_half = [0.3 / mid] * mid if mid > 0 else [0.3]
        second_half = [0.7 / (periods - mid)] * (periods - mid) if (periods - mid) > 0 else [0.7]
        weights = first_half + second_half
    
    elif strategy_type == "regelbasiert" and price_threshold and prices is not None:
        # Kaufe mehr wenn Preis unter Schwelle, weniger wenn darüber
        weights = []
        avg_price = prices.mean()
        for i, price in enumerate(prices[:periods] if len(prices) >= periods else prices):
            if price < price_threshold:
                weights.append(1.5)  # Mehr kaufen
            elif price > avg_price * 1.1:
                weights.append(0.5)  # Weniger kaufen
            else:
                weights.append(1.0)
        # Auffüllen wenn nötig
        while len(weights) < periods:
            weights.append(1.0)
    
    elif strategy_type == "gleitender_durchschnitt" and prices is not None:
        # Kaufe wenn Preis unter gleitendem Durchschnitt
        ma = prices.rolling(window=min(20, len(prices)//4)).mean()
        weights = []
        for i in range(periods):
            idx = min(i * (len(prices) // periods), len(prices) - 1)
            if pd.notna(ma.iloc[idx]) and prices.iloc[idx] < ma.iloc[idx]:
                weights.append(1.3)
            else:
                weights.append(0.8)
    
    elif strategy_type == "custom" and custom_weights:
        total_weight = sum(custom_weights)
        weights = [w / total_weight for w in custom_weights] if total_weight > 0 else [1/periods] * periods
    
    else:
        weights = [1/periods] * periods
    
    # Normalisierung
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1/periods] * periods
    
    return [fixed_volume * w for w in weights]

def run_backtest(
    load_profile: pd.DataFrame,
    spot_prices: pd.DataFrame,
    forward_prices: pd.DataFrame,
    fixing_quota: float,
    strategy_type: str,
    start_date: datetime,
    end_date: datetime,
    transaction_costs: float = 0.0,
    custom_weights: Optional[List[float]] = None,
    price_threshold: Optional[float] = None
) -> dict:
    """Führt das Backtesting für eine Strategie durch mit erweiterten Metriken."""
    
    try:
        # Daten filtern
        load_df = load_profile.copy()
        load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])
        load_df = load_df[(load_df['timestamp'] >= start_date) & (load_df['timestamp'] <= end_date)]
        
        spot_df = spot_prices.copy()
        spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'])
        spot_df = spot_df[(spot_df['timestamp'] >= start_date) & (spot_df['timestamp'] <= end_date)]
        
        if load_df.empty or spot_df.empty:
            return {"error": "Keine Daten im gewählten Zeitraum"}
        
        # Gesamtvolumen berechnen
        total_volume = load_df['value'].sum()
        
        # Fixiertes vs. Spot-Volumen
        fixed_volume = total_volume * (fixing_quota / 100)
        spot_volume = total_volume - fixed_volume
        
        # Durchschnittlicher Forward-Preis
        if not forward_prices.empty:
            forward_df = forward_prices.copy()
            forward_df['date'] = pd.to_datetime(forward_df['date'])
            forward_df = forward_df[(forward_df['date'] >= start_date) & (forward_df['date'] <= end_date)]
            avg_forward_price = forward_df['price'].mean() if not forward_df.empty else spot_df['price'].mean()
        else:
            avg_forward_price = spot_df['price'].mean()
        
        # Spotpreis-Statistiken
        avg_spot_price = spot_df['price'].mean()
        min_spot_price = spot_df['price'].min()
        max_spot_price = spot_df['price'].max()
        spot_volatility = spot_df['price'].std()
        
        # Tägliche Renditen für erweiterte Metriken
        spot_df_daily = spot_df.set_index('timestamp').resample('D')['price'].mean().dropna()
        price_returns = spot_df_daily.pct_change().dropna()
        
        # Kosten berechnen
        fixed_costs = fixed_volume * avg_forward_price
        spot_costs = spot_volume * avg_spot_price
        transaction_cost_total = total_volume * transaction_costs
        
        total_costs = fixed_costs + spot_costs + transaction_cost_total
        avg_price = total_costs / total_volume if total_volume > 0 else 0
        
        # Benchmark: 100% Spot-Beschaffung
        benchmark_costs = total_volume * avg_spot_price + transaction_cost_total
        cost_savings_vs_spot = benchmark_costs - total_costs
        cost_savings_percent = (cost_savings_vs_spot / benchmark_costs * 100) if benchmark_costs > 0 else 0
        
        # Mark-to-Market
        current_price = spot_df['price'].iloc[-1] if not spot_df.empty else avg_spot_price
        mtm_value = total_volume * current_price
        mtm_pnl = mtm_value - total_costs
        mtm_pnl_percent = (mtm_pnl / total_costs * 100) if total_costs > 0 else 0
        
        # Erweiterte Risikometriken
        sharpe_ratio = calculate_sharpe_ratio(price_returns) if len(price_returns) > 0 else 0
        sortino_ratio = calculate_sortino_ratio(price_returns) if len(price_returns) > 0 else 0
        
        cumulative_returns = (1 + price_returns).cumprod() if len(price_returns) > 0 else pd.Series([1])
        max_drawdown, max_dd_duration = calculate_max_drawdown(cumulative_returns)
        
        var_95 = calculate_var(price_returns, 0.95) * 100 if len(price_returns) > 0 else 0
        var_99 = calculate_var(price_returns, 0.99) * 100 if len(price_returns) > 0 else 0
        
        # Monatliche Kostenzuordnung
        load_df['month'] = load_df['timestamp'].dt.to_period('M')
        monthly_costs = []
        
        for month, group in load_df.groupby('month'):
            month_volume = group['value'].sum()
            month_fixed = month_volume * (fixing_quota / 100)
            month_spot = month_volume - month_fixed
            
            month_start = month.start_time
            month_end = month.end_time
            month_spot_prices = spot_df[(spot_df['timestamp'] >= month_start) & (spot_df['timestamp'] <= month_end)]
            month_avg_spot = month_spot_prices['price'].mean() if not month_spot_prices.empty else avg_spot_price
            month_min_spot = month_spot_prices['price'].min() if not month_spot_prices.empty else min_spot_price
            month_max_spot = month_spot_prices['price'].max() if not month_spot_prices.empty else max_spot_price
            
            month_cost = month_fixed * avg_forward_price + month_spot * month_avg_spot
            monthly_costs.append({
                'month': str(month),
                'volume': month_volume,
                'cost': month_cost,
                'avg_price': month_cost / month_volume if month_volume > 0 else 0,
                'spot_avg': month_avg_spot,
                'spot_min': month_min_spot,
                'spot_max': month_max_spot
            })
        
        return {
            "strategy_type": strategy_type,
            "fixing_quota": fixing_quota,
            "total_volume": total_volume,
            "fixed_volume": fixed_volume,
            "spot_volume": spot_volume,
            "avg_forward_price": avg_forward_price,
            "avg_spot_price": avg_spot_price,
            "min_spot_price": min_spot_price,
            "max_spot_price": max_spot_price,
            "fixed_costs": fixed_costs,
            "spot_costs": spot_costs,
            "transaction_costs": transaction_cost_total,
            "total_costs": total_costs,
            "avg_price": avg_price,
            "spot_volatility": spot_volatility,
            "mtm_value": mtm_value,
            "mtm_pnl": mtm_pnl,
            "mtm_pnl_percent": mtm_pnl_percent,
            # Erweiterte Metriken
            "benchmark_costs": benchmark_costs,
            "cost_savings_vs_spot": cost_savings_vs_spot,
            "cost_savings_percent": cost_savings_percent,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_dd_duration,
            "var_95": var_95,
            "var_99": var_99,
            "monthly_costs": monthly_costs,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
    
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

def compare_strategies(results_list: List[dict]) -> pd.DataFrame:
    """Vergleicht mehrere Strategien und erstellt ein Ranking."""
    
    if not results_list:
        return pd.DataFrame()
    
    comparison_data = []
    for result in results_list:
        if "error" not in result:
            comparison_data.append({
                "Strategie": f"{result['strategy_type']} ({result['fixing_quota']}%)",
                "Gesamtkosten (€)": result['total_costs'],
                "Ø Preis (€/MWh)": result['avg_price'],
                "Fixierungskosten (€)": result['fixed_costs'],
                "Spotkosten (€)": result['spot_costs'],
                "Volatilität": result['spot_volatility'],
                "MtM PnL (€)": result['mtm_pnl'],
                "MtM PnL (%)": result['mtm_pnl_percent']
            })
    
    df = pd.DataFrame(comparison_data)
    if not df.empty:
        df = df.sort_values("Gesamtkosten (€)")
        df["Rang"] = range(1, len(df) + 1)
    
    return df

def generate_recommendation(comparison_df: pd.DataFrame, detailed_results: Optional[List[dict]] = None) -> str:
    """Generiert eine erweiterte Handlungsempfehlung basierend auf den Ergebnissen."""
    
    if comparison_df.empty:
        return "Keine ausreichenden Daten für eine Empfehlung."
    
    best = comparison_df.iloc[0]
    worst = comparison_df.iloc[-1]
    
    cost_savings = worst["Gesamtkosten (€)"] - best["Gesamtkosten (€)"]
    cost_savings_percent = (cost_savings / worst["Gesamtkosten (€)"]) * 100 if worst["Gesamtkosten (€)"] > 0 else 0
    
    recommendation = f"""
### 📊 Handlungsempfehlung

**🏆 Beste Strategie:** {best['Strategie']}
- Durchschnittspreis: {best['Ø Preis (€/MWh)']:.2f} €/MWh
- Gesamtkosten: {best['Gesamtkosten (€)']:,.0f} €

**💰 Kosteneinsparung gegenüber schlechtester Strategie:**
- Absolut: {cost_savings:,.0f} €
- Prozentual: {cost_savings_percent:.1f}%
"""
    
    # Erweiterte Risikometriken wenn verfügbar
    if detailed_results:
        best_result = next((r for r in detailed_results if "error" not in r), None)
        if best_result:
            recommendation += f"""
**📈 Risikometriken:**
- Max. Drawdown: {best_result.get('max_drawdown', 0):.2f}%
- Value at Risk (95%): {best_result.get('var_95', 0):.2f}%
- Sharpe Ratio: {best_result.get('sharpe_ratio', 0):.2f}
- Sortino Ratio: {best_result.get('sortino_ratio', 0):.2f}
"""
    
    recommendation += "\n**🎯 Empfehlung:**\n"
    
    # Analyse der besten Strategie
    strat_lower = best['Strategie'].lower()
    if "gleichmäßig" in strat_lower:
        recommendation += "✅ Eine gleichmäßige Fixierungsstrategie bietet das beste Risiko-Rendite-Verhältnis. "
        recommendation += "Diese Strategie glättet Preisschwankungen effektiv und reduziert das Timing-Risiko."
    elif "frontloaded" in strat_lower:
        recommendation += "✅ Eine frühe Fixierung (Frontloaded) war historisch vorteilhaft. "
        recommendation += "Dies deutet auf steigende Marktpreise im Betrachtungszeitraum hin. "
        recommendation += "Empfehlung: Bei erwarteten Preissteigerungen frühzeitig absichern."
    elif "backloaded" in strat_lower:
        recommendation += "✅ Eine späte Fixierung (Backloaded) war historisch vorteilhaft. "
        recommendation += "Dies deutet auf fallende Marktpreise im Betrachtungszeitraum hin. "
        recommendation += "Empfehlung: Bei erwarteten Preisrückgängen abwarten kann sich lohnen."
    elif "regelbasiert" in strat_lower:
        recommendation += "✅ Eine regelbasierte Strategie mit Preisschwellen hat optimal performt. "
        recommendation += "Empfehlung: Setzen Sie klare Preisziele und handeln Sie systematisch."
    elif "gleitender" in strat_lower or "durchschnitt" in strat_lower:
        recommendation += "✅ Die Strategie basierend auf gleitendem Durchschnitt war erfolgreich. "
        recommendation += "Technische Analysetools können bei der Beschaffung helfen."
    else:
        recommendation += "✅ Die gewählte Strategie hat die besten Ergebnisse erzielt."
    
    # Volatilitätshinweis
    avg_volatility = comparison_df["Volatilität"].mean()
    if avg_volatility > 20:
        recommendation += "\n\n⚠️ **Risikohinweis:** Hohe Marktvolatilität erkannt! "
        recommendation += "Eine höhere Fixierungsquote (>70%) könnte zur Risikoreduktion sinnvoll sein."
    elif avg_volatility > 15:
        recommendation += "\n\n⚠️ **Hinweis:** Moderate Volatilität. Eine ausgewogene Fixierungsquote (50-70%) wird empfohlen."
    else:
        recommendation += "\n\n✅ **Hinweis:** Niedrige Volatilität. Mehr Flexibilität bei der Beschaffung möglich."
    
    # Optimale Quote bestimmen
    if 'fixing_quota' in comparison_df.columns or any('25%' in str(s) or '50%' in str(s) for s in comparison_df['Strategie']):
        recommendation += "\n\n**💡 Optimierungstipp:** "
        recommendation += "Führen Sie eine Sensitivitätsanalyse durch, um die optimale Fixierungsquote zu finden."
    
    return recommendation

# ============================================================================
# VISUALISIERUNGEN
# ============================================================================

def plot_price_history(spot_df: pd.DataFrame, forward_df: pd.DataFrame) -> go.Figure:
    """Erstellt ein Liniendiagramm der historischen Preisentwicklung."""
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if not spot_df.empty:
        spot_df = spot_df.copy()
        spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'])
        fig.add_trace(
            go.Scatter(
                x=spot_df['timestamp'],
                y=spot_df['price'],
                name="Spotpreis",
                line=dict(color="#1f77b4", width=1),
                opacity=0.7
            ),
            secondary_y=False
        )
    
    if not forward_df.empty:
        forward_df = forward_df.copy()
        forward_df['date'] = pd.to_datetime(forward_df['date'])
        # Gruppierung nach Datum für Durchschnitt
        daily_forward = forward_df.groupby('date')['price'].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=daily_forward['date'],
                y=daily_forward['price'],
                name="Forward-Preis (Ø)",
                line=dict(color="#ff7f0e", width=2)
            ),
            secondary_y=False
        )
    
    fig.update_layout(
        title="📈 Historische Preisentwicklung",
        xaxis_title="Datum",
        yaxis_title="Preis (€/MWh)",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )
    
    return fig

def plot_load_profile(load_df: pd.DataFrame) -> go.Figure:
    """Erstellt eine Visualisierung des Lastprofils."""
    
    load_df = load_df.copy()
    load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=load_df['timestamp'],
            y=load_df['value'],
            fill='tozeroy',
            name="Verbrauch",
            line=dict(color="#2ecc71"),
            fillcolor="rgba(46, 204, 113, 0.3)"
        )
    )
    
    fig.update_layout(
        title="⚡ Lastprofil (Verbrauch)",
        xaxis_title="Zeitstempel",
        yaxis_title="Verbrauch (MWh)",
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_cost_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Erstellt ein Balkendiagramm zum Kostenvergleich."""
    
    if comparison_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=comparison_df['Strategie'],
            y=comparison_df['Gesamtkosten (€)'],
            marker_color=px.colors.qualitative.Set2[:len(comparison_df)],
            text=comparison_df['Gesamtkosten (€)'].apply(lambda x: f'{x:,.0f} €'),
            textposition='outside'
        )
    )
    
    fig.update_layout(
        title="💰 Kostenvergleich der Strategien",
        xaxis_title="Strategie",
        yaxis_title="Gesamtkosten (€)",
        template="plotly_white",
        height=500,
        showlegend=False
    )
    
    return fig

def plot_cost_heatmap(results_matrix: List[List[dict]], quotas: List[int], periods: List[str]) -> go.Figure:
    """Erstellt eine Heatmap für Fixierungsquoten × Zeiträume."""
    
    # Matrix erstellen
    z_values = []
    for row in results_matrix:
        z_row = []
        for result in row:
            if result and "error" not in result:
                z_row.append(result.get('avg_price', 0))
            else:
                z_row.append(None)
        z_values.append(z_row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=periods,
        y=[f"{q}%" for q in quotas],
        colorscale="RdYlGn_r",
        text=[[f"{v:.2f}" if v else "" for v in row] for row in z_values],
        texttemplate="%{text}",
        hovertemplate="Quote: %{y}<br>Zeitraum: %{x}<br>Ø Preis: %{z:.2f} €/MWh<extra></extra>"
    ))
    
    fig.update_layout(
        title="🗺️ Heatmap: Fixierungsquoten × Zeiträume → Ø Preis",
        xaxis_title="Zeitraum",
        yaxis_title="Fixierungsquote",
        template="plotly_white",
        height=500
    )
    
    return fig

def plot_volatility_analysis(comparison_df: pd.DataFrame) -> go.Figure:
    """Erstellt Boxplots für die Volatilitätsanalyse."""
    
    if comparison_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=comparison_df['Strategie'],
            y=comparison_df['Volatilität'],
            marker_color='#e74c3c',
            name="Volatilität (Std.)"
        )
    )
    
    fig.update_layout(
        title="📊 Volatilitätsanalyse der Strategien",
        xaxis_title="Strategie",
        yaxis_title="Volatilität (Standardabweichung)",
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_pnl_development(results: dict) -> go.Figure:
    """Erstellt ein Diagramm der PnL-Entwicklung über Zeit."""
    
    if not results.get('monthly_costs'):
        return go.Figure()
    
    monthly = pd.DataFrame(results['monthly_costs'])
    
    # Kumulierte Kosten und PnL
    monthly['cumulative_cost'] = monthly['cost'].cumsum()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=monthly['month'],
            y=monthly['cost'],
            name="Monatliche Kosten",
            fill='tozeroy',
            line=dict(color="#3498db")
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly['month'],
            y=monthly['cumulative_cost'],
            name="Kumulierte Kosten",
            line=dict(color="#e74c3c", dash='dash')
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="📈 Kostenentwicklung über Zeit",
        xaxis_title="Monat",
        template="plotly_white",
        height=450
    )
    
    fig.update_yaxes(title_text="Monatliche Kosten (€)", secondary_y=False)
    fig.update_yaxes(title_text="Kumulierte Kosten (€)", secondary_y=True)
    
    return fig

def plot_sensitivity_analysis(base_result: dict, sensitivity_results: List[dict]) -> go.Figure:
    """Erstellt eine Sensitivitätsanalyse."""
    
    if not sensitivity_results:
        return go.Figure()
    
    quotas = [r.get('fixing_quota', 0) for r in sensitivity_results if 'error' not in r]
    costs = [r.get('total_costs', 0) for r in sensitivity_results if 'error' not in r]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=quotas,
            y=costs,
            mode='lines+markers',
            name="Gesamtkosten",
            line=dict(color="#9b59b6", width=3),
            marker=dict(size=10)
        )
    )
    
    # Optimum markieren
    if costs:
        min_idx = costs.index(min(costs))
        fig.add_annotation(
            x=quotas[min_idx],
            y=costs[min_idx],
            text=f"Optimum: {quotas[min_idx]}%",
            showarrow=True,
            arrowhead=2
        )
    
    fig.update_layout(
        title="🎯 Sensitivitätsanalyse: Fixierungsquote vs. Kosten",
        xaxis_title="Fixierungsquote (%)",
        yaxis_title="Gesamtkosten (€)",
        template="plotly_white",
        height=450
    )
    
    return fig

def plot_benchmark_comparison(result: dict) -> go.Figure:
    """Erstellt einen Benchmark-Vergleich (Strategie vs. 100% Spot)."""
    
    if not result or "error" in result:
        return go.Figure()
    
    categories = ['Strategie', 'Benchmark (100% Spot)']
    values = [result.get('total_costs', 0), result.get('benchmark_costs', 0)]
    colors = ['#3498db', '#e74c3c']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{v:,.0f} €' for v in values],
        textposition='outside'
    ))
    
    # Einsparungs-Annotation
    savings = result.get('cost_savings_vs_spot', 0)
    savings_pct = result.get('cost_savings_percent', 0)
    
    fig.add_annotation(
        x=0.5,
        y=max(values) * 1.1,
        text=f"Ersparnis: {savings:,.0f} € ({savings_pct:.1f}%)" if savings > 0 else f"Mehrkosten: {abs(savings):,.0f} € ({abs(savings_pct):.1f}%)",
        showarrow=False,
        font=dict(size=14, color='green' if savings > 0 else 'red')
    )
    
    fig.update_layout(
        title="📊 Benchmark-Vergleich: Strategie vs. 100% Spot",
        yaxis_title="Gesamtkosten (€)",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig

def plot_risk_metrics_radar(result: dict) -> go.Figure:
    """Erstellt ein Radar-Chart für Risikometriken."""
    
    if not result or "error" in result:
        return go.Figure()
    
    # Metriken normalisieren (0-100 Skala)
    sharpe = min(max((result.get('sharpe_ratio', 0) + 2) * 25, 0), 100)  # -2 bis 2 -> 0-100
    sortino = min(max((result.get('sortino_ratio', 0) + 2) * 25, 0), 100)
    max_dd = 100 - min(result.get('max_drawdown', 0), 100)  # Invertiert (weniger ist besser)
    var = 100 - min(result.get('var_95', 0) * 5, 100)  # Invertiert
    cost_eff = min(max(result.get('cost_savings_percent', 0) + 50, 0), 100)  # -50% bis 50% -> 0-100
    
    categories = ['Sharpe Ratio', 'Sortino Ratio', 'Drawdown-Schutz', 'VaR-Schutz', 'Kosteneffizienz']
    values = [sharpe, sortino, max_dd, var, cost_eff]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Schließen des Polygons
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.3)',
        line=dict(color='#3498db', width=2),
        name='Strategie'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        title="🎯 Risikoprofil der Strategie",
        template="plotly_white",
        height=450,
        showlegend=False
    )
    
    return fig

# ============================================================================
# EXPORT-FUNKTIONEN
# ============================================================================

def export_to_csv(df: pd.DataFrame) -> str:
    """Exportiert DataFrame als CSV."""
    return df.to_csv(index=False)

def export_to_excel(df: pd.DataFrame) -> bytes:
    """Exportiert DataFrame als Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Ergebnisse')
    return output.getvalue()

def export_db_backup() -> bytes:
    """Erstellt ein Backup der Datenbank."""
    with open(DB_PATH, 'rb') as f:
        return f.read()

def import_db_backup(uploaded_file):
    """Importiert ein Datenbank-Backup."""
    try:
        with open(DB_PATH, 'wb') as f:
            f.write(uploaded_file.read())
        return True, "Datenbank erfolgreich wiederhergestellt"
    except Exception as e:
        return False, f"Fehler: {str(e)}"

# ============================================================================
# CUSTOM CSS
# ============================================================================

def load_custom_css():
    """Lädt benutzerdefiniertes CSS für professionelles Design."""
    st.markdown("""
    <style>
        /* Hauptcontainer */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Header Styling */
        h1 {
            color: #1a5276;
            font-weight: 700;
            border-bottom: 3px solid #3498db;
            padding-bottom: 0.5rem;
        }
        
        h2, h3 {
            color: #2c3e50;
        }
        
        /* Metric Cards */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        [data-testid="metric-container"] label {
            color: rgba(255, 255, 255, 0.9) !important;
        }
        
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: white !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a5276 0%, #2980b9 100%);
        }
        
        [data-testid="stSidebar"] .stRadio > label {
            color: white !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
        }
        
        /* Download Button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        }
        
        /* Info Boxes */
        .stAlert {
            border-radius: 10px;
        }
        
        /* DataFrames */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #3498db;
            color: white;
        }
        
        /* Card Style */
        .card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        /* Loading Spinner */
        .stSpinner > div {
            border-color: #3498db;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SEITENINHALT - DASHBOARD
# ============================================================================

def render_dashboard():
    """Rendert das Haupt-Dashboard."""
    
    st.header("📊 Dashboard")
    
    # DB Stats laden
    stats = get_db_stats()
    
    # KPI Metriken
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📁 Lastprofile",
            value=f"{stats['load_profiles']:,}",
            help="Anzahl der gespeicherten Lastprofile"
        )
    
    with col2:
        st.metric(
            label="💹 Spotpreise",
            value=f"{stats['spot_prices']:,}",
            help="Anzahl der Spotpreis-Datenpunkte"
        )
    
    with col3:
        st.metric(
            label="📈 Forward-Preise",
            value=f"{stats['forward_prices']:,}",
            help="Anzahl der Forward-Preis-Datenpunkte"
        )
    
    with col4:
        st.metric(
            label="🧪 Simulationen",
            value=f"{stats['simulations']:,}",
            help="Anzahl durchgeführter Simulationen"
        )
    
    st.divider()
    
    # Letzte Simulationen
    st.subheader("🕒 Letzte Simulationen")
    
    simulations_df = load_simulations()
    
    if not simulations_df.empty:
        # Parse results für Anzeige
        display_data = []
        for _, row in simulations_df.head(10).iterrows():
            try:
                results = json.loads(row['results_json'])
                config = json.loads(row['strategy_config'])
                display_data.append({
                    "Name": row['name'],
                    "Strategie": config.get('strategy_type', 'N/A'),
                    "Quote": f"{config.get('fixing_quota', 0)}%",
                    "Gesamtkosten": f"{results.get('total_costs', 0):,.0f} €",
                    "Ø Preis": f"{results.get('avg_price', 0):.2f} €/MWh",
                    "Erstellt": row['created_at'][:16]
                })
            except:
                continue
        
        if display_data:
            st.dataframe(
                pd.DataFrame(display_data),
                use_container_width=True,
                hide_index=True
            )
            
            # Beste Strategie hervorheben
            if len(display_data) > 1:
                st.success(f"✅ **Beste Strategie:** {display_data[0]['Name']} mit {display_data[0]['Ø Preis']}")
        else:
            st.info("Noch keine auswertbaren Simulationen vorhanden.")
    else:
        st.info("Noch keine Simulationen durchgeführt. Starten Sie unter 'Simulation konfigurieren'.")
    
    # Quick Stats wenn Daten vorhanden
    if stats['spot_prices'] > 0:
        st.divider()
        st.subheader("📈 Marktübersicht")
        
        spot_df = load_data_from_db('spot_prices')
        spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Aktueller Spotpreis",
                value=f"{spot_df['price'].iloc[-1]:.2f} €/MWh"
            )
        
        with col2:
            avg_price = spot_df['price'].mean()
            st.metric(
                label="Ø Spotpreis",
                value=f"{avg_price:.2f} €/MWh"
            )
        
        with col3:
            volatility = spot_df['price'].std()
            st.metric(
                label="Volatilität",
                value=f"{volatility:.2f}"
            )
        
        # Mini-Chart
        fig = plot_price_history(spot_df, pd.DataFrame())
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SEITENINHALT - DATEN HOCHLADEN
# ============================================================================

def render_data_upload():
    """Rendert das Daten-Upload-Modul."""
    
    st.header("📁 Daten hochladen")
    
    # Tabs für verschiedene Datentypen
    tab1, tab2, tab3 = st.tabs(["⚡ Lastprofile", "💹 Spotpreise", "📈 Forward-Preise"])
    
    with tab1:
        render_upload_section(
            "Lastprofil",
            "load_profiles",
            ["timestamp", "value", "unit"],
            validate_load_profile,
            save_load_profile
        )
    
    with tab2:
        render_upload_section(
            "Spotpreise",
            "spot_prices",
            ["timestamp", "price"],
            validate_spot_prices,
            save_spot_prices
        )
    
    with tab3:
        render_upload_section(
            "Forward-Preise",
            "forward_prices",
            ["date", "product", "delivery_period", "price"],
            validate_forward_prices,
            save_forward_prices
        )

def render_upload_section(name: str, table: str, required_cols: List[str], 
                          validate_func, save_func):
    """Generische Upload-Sektion für verschiedene Datentypen."""
    
    st.subheader(f"📤 {name} hochladen")
    
    # Upload-Methode wählen
    upload_method = st.radio(
        "Upload-Methode:",
        ["📎 CSV-Datei", "📊 Excel-Datei", "📋 Copy-Paste"],
        horizontal=True,
        key=f"upload_method_{table}"
    )
    
    df = pd.DataFrame()
    
    if upload_method == "📎 CSV-Datei":
        uploaded_file = st.file_uploader(
            "CSV-Datei wählen",
            type=['csv'],
            key=f"csv_upload_{table}",
            help="Unterstützte Trennzeichen: Komma, Semikolon, Tab"
        )
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            df = parse_uploaded_data(content, 'csv')
    
    elif upload_method == "📊 Excel-Datei":
        uploaded_file = st.file_uploader(
            "Excel-Datei wählen",
            type=['xlsx', 'xls'],
            key=f"excel_upload_{table}"
        )
        
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
    
    else:  # Copy-Paste
        paste_content = st.text_area(
            "Daten hier einfügen (Tab- oder Komma-getrennt):",
            height=200,
            key=f"paste_{table}",
            help="Kopieren Sie Ihre Daten direkt aus Excel oder einer anderen Quelle"
        )
        
        if paste_content:
            df = parse_uploaded_data(paste_content, 'paste')
    
    # Wenn Daten vorhanden
    if not df.empty:
        st.success(f"✅ {len(df)} Zeilen erkannt")
        
        # Zeitintervall erkennen
        if 'timestamp' in df.columns or any('time' in c.lower() for c in df.columns):
            time_col = 'timestamp' if 'timestamp' in df.columns else [c for c in df.columns if 'time' in c.lower()][0]
            interval = detect_time_interval(df, time_col)
            st.info(f"📅 Erkanntes Zeitintervall: **{interval}**")
        
        # Spalten-Mapping
        st.markdown("#### Spaltenzuordnung")
        
        col_mapping = {}
        cols = st.columns(len(required_cols))
        
        for i, req_col in enumerate(required_cols):
            with cols[i]:
                # Automatische Erkennung versuchen
                default_idx = 0
                for j, col in enumerate(df.columns):
                    if req_col.lower() in col.lower():
                        default_idx = j
                        break
                
                col_mapping[req_col] = st.selectbox(
                    f"{req_col}:",
                    options=df.columns.tolist(),
                    index=default_idx,
                    key=f"map_{table}_{req_col}"
                )
        
        # DataFrame mit gemappten Spalten
        mapped_df = df[[col_mapping[c] for c in required_cols]].copy()
        mapped_df.columns = required_cols
        
        # Preview
        st.markdown("#### Vorschau")
        st.dataframe(mapped_df.head(20), use_container_width=True, hide_index=True)
        
        # Validierung
        is_valid, validation_msg = validate_func(mapped_df)
        
        if is_valid:
            st.success(f"✅ {validation_msg}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"💾 {name} speichern", key=f"save_{table}", type="primary"):
                    with st.spinner("Speichere Daten..."):
                        try:
                            if table == 'load_profiles':
                                profile_name = st.session_state.get(f"profile_name_{table}", "default")
                                save_func(mapped_df, profile_name)
                            else:
                                save_func(mapped_df)
                            st.success(f"✅ {len(mapped_df)} Einträge erfolgreich gespeichert!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"❌ Fehler beim Speichern: {str(e)}")
            
            with col2:
                if st.button(f"🗑️ Bestehende {name} löschen", key=f"clear_{table}"):
                    clear_table(table)
                    st.warning(f"⚠️ Alle {name} wurden gelöscht.")
        else:
            st.error(f"❌ Validierungsfehler: {validation_msg}")
    
    # Bestehende Daten anzeigen
    with st.expander("📊 Gespeicherte Daten anzeigen"):
        existing_df = load_data_from_db(table)
        if not existing_df.empty:
            st.dataframe(existing_df.tail(100), use_container_width=True, hide_index=True)
            st.caption(f"Zeige letzte 100 von {len(existing_df)} Einträgen")
        else:
            st.info("Keine Daten vorhanden.")

# ============================================================================
# SEITENINHALT - SIMULATION
# ============================================================================

def render_simulation():
    """Rendert das Simulations-Konfigurationsmodul."""
    
    st.header("⚙️ Simulation konfigurieren")
    
    # Prüfen ob Daten vorhanden
    stats = get_db_stats()
    
    if stats['load_profiles'] == 0 or stats['spot_prices'] == 0:
        st.warning("⚠️ Bitte laden Sie zuerst Lastprofile und Spotpreise hoch!")
        return
    
    # Daten laden
    load_df = load_data_from_db('load_profiles')
    spot_df = load_data_from_db('spot_prices')
    forward_df = load_data_from_db('forward_prices')
    
    # Zeitstempel konvertieren
    load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])
    spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'])
    
    # Zeitraum bestimmen
    min_date = max(load_df['timestamp'].min(), spot_df['timestamp'].min()).date()
    max_date = min(load_df['timestamp'].max(), spot_df['timestamp'].max()).date()
    
    st.markdown("### 📅 Zeitraum")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Startdatum",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="sim_start"
        )
    with col2:
        end_date = st.date_input(
            "Enddatum",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="sim_end"
        )
    
    st.divider()
    
    # Strategie-Konfiguration
    st.markdown("### 🎯 Strategie-Konfiguration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy_type = st.selectbox(
            "Strategietyp",
            options=list(STRATEGY_TYPES.keys()),
            format_func=lambda x: f"{x} - {STRATEGY_TYPES[x]}",
            key="strategy_type"
        )
        
        fixing_quota = st.slider(
            "Fixierungsquote (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            key="fixing_quota",
            help="Prozentsatz des Volumens, das über Terminkontrakte abgesichert wird"
        )
    
    with col2:
        transaction_costs = st.number_input(
            "Transaktionskosten (€/MWh)",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1,
            key="transaction_costs",
            help="Zusätzliche Kosten pro MWh (Spreads, Gebühren)"
        )
        
        simulation_name = st.text_input(
            "Simulationsname",
            value=f"Simulation_{datetime.now().strftime('%Y%m%d_%H%M')}",
            key="sim_name"
        )
    
    # Custom Weights für benutzerdefinierte Strategie
    custom_weights = None
    if strategy_type == "custom":
        st.markdown("#### 📊 Benutzerdefinierte Gewichtung")
        num_periods = st.number_input("Anzahl Perioden", min_value=2, max_value=12, value=4)
        
        cols = st.columns(int(num_periods))
        custom_weights = []
        for i, col in enumerate(cols):
            with col:
                w = st.number_input(f"P{i+1}", min_value=0.0, max_value=100.0, value=100/num_periods, key=f"w_{i}")
                custom_weights.append(w)
    
    st.divider()
    
    # Simulation starten
    st.markdown("### 🚀 Simulation ausführen")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_single = st.button("▶️ Einzelne Simulation", type="primary", use_container_width=True)
    
    with col2:
        run_comparison = st.button("📊 Strategievergleich", use_container_width=True)
    
    with col3:
        run_sensitivity = st.button("🎯 Sensitivitätsanalyse", use_container_width=True)
    
    # Einzelne Simulation
    if run_single:
        with st.spinner("Führe Simulation durch..."):
            result = run_backtest(
                load_df, spot_df, forward_df,
                fixing_quota, strategy_type,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
                transaction_costs,
                custom_weights
            )
            
            if "error" not in result:
                # Ergebnis speichern
                save_simulation(
                    simulation_name,
                    {
                        "strategy_type": strategy_type,
                        "fixing_quota": fixing_quota,
                        "transaction_costs": transaction_costs
                    },
                    result
                )
                
                st.session_state['last_result'] = result
                st.success("✅ Simulation abgeschlossen!")
                
                # Ergebnisse anzeigen
                st.markdown("### 📊 Ergebnisse")
                
                # Hauptmetriken
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Gesamtkosten", f"{result['total_costs']:,.0f} €")
                with col2:
                    st.metric("Ø Preis", f"{result['avg_price']:.2f} €/MWh")
                with col3:
                    delta_val = f"{result['cost_savings_percent']:.1f}%" if result.get('cost_savings_percent', 0) != 0 else None
                    st.metric("vs. Benchmark", f"{result.get('cost_savings_vs_spot', 0):,.0f} €", delta=delta_val)
                with col4:
                    st.metric("Volatilität", f"{result['spot_volatility']:.2f}")
                
                # Erweiterte Risikometriken
                st.markdown("### 📈 Erweiterte Risikometriken")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', 0):.2f}",
                             help="Risikoadjustierte Rendite (>1 gut)")
                with col2:
                    st.metric("Max Drawdown", f"{result.get('max_drawdown', 0):.2f}%",
                             help="Maximaler Preisrückgang")
                with col3:
                    st.metric("VaR (95%)", f"{result.get('var_95', 0):.2f}%",
                             help="Value at Risk: Max. Verlust mit 95% Konfidenz")
                with col4:
                    st.metric("Sortino Ratio", f"{result.get('sortino_ratio', 0):.2f}",
                             help="Risikoadjustierte Rendite (nur Downside)")
                
                # Volumen-Aufschlüsselung
                st.markdown("### 📦 Volumen-Aufschlüsselung")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Gesamtvolumen", f"{result['total_volume']:,.0f} MWh")
                with col2:
                    st.metric("Fixiertes Volumen", f"{result['fixed_volume']:,.0f} MWh")
                with col3:
                    st.metric("Spot-Volumen", f"{result['spot_volume']:,.0f} MWh")
                
                # Preisinformationen
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Ø Forward-Preis", f"{result['avg_forward_price']:.2f} €/MWh")
                with col2:
                    st.metric("Ø Spot-Preis", f"{result['avg_spot_price']:.2f} €/MWh")
                with col3:
                    st.metric("Min Spot-Preis", f"{result.get('min_spot_price', 0):.2f} €/MWh")
                with col4:
                    st.metric("Max Spot-Preis", f"{result.get('max_spot_price', 0):.2f} €/MWh")
                
                # Charts in Tabs
                chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📈 Kostenentwicklung", "📊 Benchmark-Vergleich", "🎯 Risikoprofil"])
                
                with chart_tab1:
                    fig = plot_pnl_development(result)
                    st.plotly_chart(fig, use_container_width=True, key="single_sim_pnl")
                
                with chart_tab2:
                    fig = plot_benchmark_comparison(result)
                    st.plotly_chart(fig, use_container_width=True, key="benchmark_comp")
                
                with chart_tab3:
                    fig = plot_risk_metrics_radar(result)
                    st.plotly_chart(fig, use_container_width=True, key="risk_radar")
            else:
                st.error(f"❌ Fehler: {result['error']}")
                if 'traceback' in result:
                    with st.expander("Details"):
                        st.code(result['traceback'])
    
    # Strategievergleich
    if run_comparison:
        with st.spinner("Vergleiche Strategien..."):
            results_list = []
            progress_bar = st.progress(0)
            total_combinations = len(STRATEGY_TYPES) * 4  # 4 Quoten
            current = 0
            
            for strat in STRATEGY_TYPES.keys():
                for quota in [25, 50, 75, 100]:
                    result = run_backtest(
                        load_df, spot_df, forward_df,
                        quota, strat,
                        datetime.combine(start_date, datetime.min.time()),
                        datetime.combine(end_date, datetime.max.time()),
                        transaction_costs
                    )
                    if "error" not in result:
                        results_list.append(result)
                    current += 1
                    progress_bar.progress(current / total_combinations)
            
            progress_bar.empty()
            
            if results_list:
                comparison_df = compare_strategies(results_list)
                st.session_state['comparison_df'] = comparison_df
                st.session_state['comparison_results'] = results_list
                
                st.success(f"✅ {len(results_list)} Strategien verglichen!")
                
                # Top 3 Strategien hervorheben
                st.markdown("### 🥇🥈🥉 Top 3 Strategien")
                top3_cols = st.columns(3)
                for i, (_, row) in enumerate(comparison_df.head(3).iterrows()):
                    medal = ["🥇", "🥈", "🥉"][i]
                    with top3_cols[i]:
                        st.markdown(f"**{medal} Platz {i+1}**")
                        st.markdown(f"**{row['Strategie']}**")
                        st.metric("Kosten", f"{row['Gesamtkosten (€)']:,.0f} €")
                        st.metric("Ø Preis", f"{row['Ø Preis (€/MWh)']:.2f} €/MWh")
                
                # Ranking anzeigen
                st.markdown("### 🏆 Vollständiges Strategie-Ranking")
                st.dataframe(
                    comparison_df.style.format({
                        "Gesamtkosten (€)": "{:,.0f}",
                        "Ø Preis (€/MWh)": "{:.2f}",
                        "Fixierungskosten (€)": "{:,.0f}",
                        "Spotkosten (€)": "{:,.0f}",
                        "Volatilität": "{:.2f}",
                        "MtM PnL (€)": "{:,.0f}",
                        "MtM PnL (%)": "{:.1f}"
                    }).background_gradient(subset=["Gesamtkosten (€)"], cmap="RdYlGn_r"),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualisierung
                fig = plot_cost_comparison(comparison_df)
                st.plotly_chart(fig, use_container_width=True, key="comparison_chart")
                
                # Empfehlung mit erweiterten Daten
                recommendation = generate_recommendation(comparison_df, results_list)
                st.markdown(recommendation)
    
    # Sensitivitätsanalyse
    if run_sensitivity:
        with st.spinner("Führe Sensitivitätsanalyse durch..."):
            sensitivity_results = []
            
            for quota in range(0, 101, 10):
                result = run_backtest(
                    load_df, spot_df, forward_df,
                    quota, strategy_type,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                    transaction_costs
                )
                if "error" not in result:
                    sensitivity_results.append(result)
            
            if sensitivity_results:
                st.session_state['sensitivity_results'] = sensitivity_results
                
                st.success("✅ Sensitivitätsanalyse abgeschlossen!")
                
                fig = plot_sensitivity_analysis(sensitivity_results[0], sensitivity_results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Optimale Quote finden
                optimal = min(sensitivity_results, key=lambda x: x['total_costs'])
                st.info(f"🎯 **Optimale Fixierungsquote:** {optimal['fixing_quota']}% "
                       f"(Kosten: {optimal['total_costs']:,.0f} €)")

# ============================================================================
# SEITENINHALT - ERGEBNISSE & VISUALISIERUNGEN
# ============================================================================

def render_results():
    """Rendert die Ergebnisse und Visualisierungen."""
    
    st.header("📈 Ergebnisse & Visualisierungen")
    
    # Tabs für verschiedene Ansichten
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Übersicht", 
        "📈 Charts", 
        "🗺️ Heatmap",
        "📥 Export"
    ])
    
    with tab1:
        render_results_overview()
    
    with tab2:
        render_charts()
    
    with tab3:
        render_heatmap()
    
    with tab4:
        render_export()

def render_results_overview():
    """Rendert die Ergebnisübersicht."""
    
    # Letzte Simulationen laden
    simulations_df = load_simulations()
    
    if simulations_df.empty:
        st.info("Noch keine Simulationen vorhanden.")
        return
    
    # Auswahl der Simulation
    sim_options = [f"{row['name']} ({row['created_at'][:16]})" 
                   for _, row in simulations_df.iterrows()]
    
    selected_sim = st.selectbox(
        "Simulation auswählen:",
        options=range(len(sim_options)),
        format_func=lambda x: sim_options[x]
    )
    
    # Details anzeigen
    sim_row = simulations_df.iloc[selected_sim]
    results = json.loads(sim_row['results_json'])
    config = json.loads(sim_row['strategy_config'])
    
    st.markdown("### 📊 Simulationsdetails")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Konfiguration:**")
        st.json(config)
    
    with col2:
        st.markdown("**Hauptergebnisse:**")
        metrics_df = pd.DataFrame([
            {"Metrik": "Gesamtkosten", "Wert": f"{results.get('total_costs', 0):,.0f} €"},
            {"Metrik": "Ø Preis", "Wert": f"{results.get('avg_price', 0):.2f} €/MWh"},
            {"Metrik": "Fixierungskosten", "Wert": f"{results.get('fixed_costs', 0):,.0f} €"},
            {"Metrik": "Spotkosten", "Wert": f"{results.get('spot_costs', 0):,.0f} €"},
            {"Metrik": "MtM PnL", "Wert": f"{results.get('mtm_pnl', 0):,.0f} €"},
            {"Metrik": "Volatilität", "Wert": f"{results.get('spot_volatility', 0):.2f}"}
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Vergleich wenn vorhanden
    if 'comparison_df' in st.session_state:
        st.markdown("### 🏆 Strategievergleich")
        st.dataframe(
            st.session_state['comparison_df'],
            use_container_width=True,
            hide_index=True
        )

def render_charts():
    """Rendert die interaktiven Charts."""
    
    stats = get_db_stats()
    
    # Preisentwicklung
    st.markdown("### 📈 Historische Preisentwicklung")
    
    if stats['spot_prices'] > 0:
        spot_df = load_data_from_db('spot_prices')
        forward_df = load_data_from_db('forward_prices') if stats['forward_prices'] > 0 else pd.DataFrame()
        
        fig = plot_price_history(spot_df, forward_df)
        st.plotly_chart(fig, use_container_width=True, key="price_history")
    else:
        st.info("Keine Spotpreise vorhanden.")
    
    # Lastprofil
    st.markdown("### ⚡ Lastprofil")
    
    if stats['load_profiles'] > 0:
        load_df = load_data_from_db('load_profiles')
        fig = plot_load_profile(load_df)
        st.plotly_chart(fig, use_container_width=True, key="load_profile")
    else:
        st.info("Keine Lastprofile vorhanden.")
    
    # Kostenvergleich
    if 'comparison_df' in st.session_state and not st.session_state['comparison_df'].empty:
        st.markdown("### 💰 Kostenvergleich")
        fig = plot_cost_comparison(st.session_state['comparison_df'])
        st.plotly_chart(fig, use_container_width=True, key="cost_comparison")
        
        st.markdown("### 📊 Volatilitätsanalyse")
        fig = plot_volatility_analysis(st.session_state['comparison_df'])
        st.plotly_chart(fig, use_container_width=True, key="volatility")
    
    # PnL Entwicklung
    if 'last_result' in st.session_state:
        st.markdown("### 📈 PnL-Entwicklung")
        fig = plot_pnl_development(st.session_state['last_result'])
        st.plotly_chart(fig, use_container_width=True, key="pnl_dev")

def render_heatmap():
    """Rendert die Heatmap-Analyse."""
    
    st.markdown("### 🗺️ Heatmap: Fixierungsquoten × Zeiträume")
    
    stats = get_db_stats()
    
    if stats['load_profiles'] == 0 or stats['spot_prices'] == 0:
        st.info("Laden Sie zuerst Daten hoch, um eine Heatmap zu erstellen.")
        return
    
    # Parameter für Heatmap
    col1, col2 = st.columns(2)
    
    with col1:
        quotas = st.multiselect(
            "Fixierungsquoten (%)",
            options=[0, 25, 50, 75, 100],
            default=[0, 25, 50, 75, 100]
        )
    
    with col2:
        period_type = st.selectbox(
            "Periodeneinteilung",
            options=["Quartale", "Monate", "Jahre"]
        )
    
    if st.button("🗺️ Heatmap erstellen", type="primary"):
        with st.spinner("Berechne Heatmap..."):
            load_df = load_data_from_db('load_profiles')
            spot_df = load_data_from_db('spot_prices')
            forward_df = load_data_from_db('forward_prices')
            
            load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])
            spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'])
            
            # Perioden erstellen
            if period_type == "Quartale":
                periods = load_df['timestamp'].dt.to_period('Q').unique().tolist()
            elif period_type == "Monate":
                periods = load_df['timestamp'].dt.to_period('M').unique().tolist()[:12]
            else:
                periods = load_df['timestamp'].dt.to_period('Y').unique().tolist()
            
            # Matrix berechnen
            results_matrix = []
            
            for quota in quotas:
                row_results = []
                for period in periods:
                    start = period.start_time
                    end = period.end_time
                    
                    result = run_backtest(
                        load_df, spot_df, forward_df,
                        quota, "gleichmäßig",
                        start, end, 0
                    )
                    row_results.append(result if "error" not in result else None)
                
                results_matrix.append(row_results)
            
            # Heatmap erstellen
            fig = plot_cost_heatmap(results_matrix, quotas, [str(p) for p in periods])
            st.plotly_chart(fig, use_container_width=True)

def render_export():
    """Rendert die Export-Funktionen."""
    
    st.markdown("### 📥 Daten exportieren")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Simulationsergebnisse")
        
        simulations_df = load_simulations()
        
        if not simulations_df.empty:
            # CSV Export
            csv_data = export_to_csv(simulations_df)
            st.download_button(
                label="📄 Als CSV herunterladen",
                data=csv_data,
                file_name="simulationen.csv",
                mime="text/csv"
            )
            
            # Excel Export
            excel_data = export_to_excel(simulations_df)
            st.download_button(
                label="📊 Als Excel herunterladen",
                data=excel_data,
                file_name="simulationen.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("Keine Simulationen zum Exportieren.")
    
    with col2:
        st.markdown("#### Vergleichstabelle")
        
        if 'comparison_df' in st.session_state and not st.session_state['comparison_df'].empty:
            csv_data = export_to_csv(st.session_state['comparison_df'])
            st.download_button(
                label="📄 Vergleich als CSV",
                data=csv_data,
                file_name="strategievergleich.csv",
                mime="text/csv"
            )
            
            excel_data = export_to_excel(st.session_state['comparison_df'])
            st.download_button(
                label="📊 Vergleich als Excel",
                data=excel_data,
                file_name="strategievergleich.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("Führen Sie erst einen Strategievergleich durch.")
    
    st.divider()
    
    st.markdown("#### 💡 Tipp: Chart-Export")
    st.info(
        "Plotly-Charts können direkt über das Kamera-Symbol in der Chart-Toolbar "
        "als PNG exportiert werden."
    )

# ============================================================================
# SEITENINHALT - DATENBANK-MANAGEMENT
# ============================================================================

def render_db_management():
    """Rendert das Datenbank-Management."""
    
    st.header("💾 Datenbank-Management")
    
    # Statistiken
    st.markdown("### 📊 Datenbank-Statistiken")
    
    stats = get_db_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Lastprofile", f"{stats['load_profiles']:,}")
    with col2:
        st.metric("Spotpreise", f"{stats['spot_prices']:,}")
    with col3:
        st.metric("Forward-Preise", f"{stats['forward_prices']:,}")
    with col4:
        st.metric("Simulationen", f"{stats['simulations']:,}")
    
    st.divider()
    
    # Backup & Restore
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Backup erstellen")
        
        if st.button("💾 Datenbank-Backup erstellen", type="primary"):
            try:
                backup_data = export_db_backup()
                st.download_button(
                    label="⬇️ Backup herunterladen",
                    data=backup_data,
                    file_name=f"energy_backtesting_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db",
                    mime="application/x-sqlite3"
                )
                st.success("✅ Backup erstellt!")
            except Exception as e:
                st.error(f"❌ Fehler: {str(e)}")
    
    with col2:
        st.markdown("### 📤 Backup wiederherstellen")
        
        uploaded_backup = st.file_uploader(
            "Backup-Datei auswählen",
            type=['db'],
            key="backup_upload"
        )
        
        if uploaded_backup:
            if st.button("🔄 Wiederherstellen", type="secondary"):
                success, message = import_db_backup(uploaded_backup)
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
    
    st.divider()
    
    # Tabellen verwalten
    st.markdown("### 🗑️ Tabellen verwalten")
    
    st.warning("⚠️ **Achtung:** Das Löschen von Daten kann nicht rückgängig gemacht werden!")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Lastprofile löschen"):
            clear_table('load_profiles')
            st.success("Lastprofile gelöscht")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Spotpreise löschen"):
            clear_table('spot_prices')
            st.success("Spotpreise gelöscht")
            st.rerun()
    
    with col3:
        if st.button("🗑️ Forward-Preise löschen"):
            clear_table('forward_prices')
            st.success("Forward-Preise gelöscht")
            st.rerun()
    
    with col4:
        if st.button("🗑️ Simulationen löschen"):
            clear_table('simulations')
            st.success("Simulationen gelöscht")
            st.rerun()
    
    st.divider()
    
    # Datenbank komplett löschen
    st.markdown("### ⚠️ Gefahrenzone")
    
    with st.expander("🔴 Komplette Datenbank löschen"):
        st.error("Diese Aktion löscht ALLE Daten unwiderruflich!")
        
        confirm = st.text_input(
            "Zur Bestätigung 'LÖSCHEN' eingeben:",
            key="confirm_delete"
        )
        
        if st.button("🗑️ ALLES LÖSCHEN", type="primary"):
            if confirm == "LÖSCHEN":
                for table in ['load_profiles', 'spot_prices', 'forward_prices', 'simulations']:
                    clear_table(table)
                st.success("✅ Alle Daten wurden gelöscht.")
                st.rerun()
            else:
                st.error("Bestätigung nicht korrekt.")

# ============================================================================
# HAUPTANWENDUNG
# ============================================================================

def main():
    """Hauptfunktion der Streamlit-App."""
    
    # Seiten-Konfiguration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS laden
    load_custom_css()
    
    # Datenbank initialisieren
    init_database()
    
    # Session State initialisieren
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True
        st.session_state['comparison_df'] = pd.DataFrame()
        st.session_state['last_result'] = {}
    
    # Sidebar Navigation
    with st.sidebar:
        # Lokales Emoji statt externem Bild für Deployment-Sicherheit
        st.markdown("# ⚡ Energiebeschaffungs-Backtesting")
        st.title("Navigation")
        
        page = st.radio(
            "Seite wählen:",
            options=[
                "📊 Dashboard",
                "📁 Daten hochladen",
                "⚙️ Simulation konfigurieren",
                "📈 Ergebnisse & Visualisierungen",
                "💾 Datenbank-Management"
            ],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Quick Stats in Sidebar
        stats = get_db_stats()
        st.markdown("### 📈 Quick Stats")
        st.markdown(f"- **Lastprofile:** {stats['load_profiles']:,}")
        st.markdown(f"- **Spotpreise:** {stats['spot_prices']:,}")
        st.markdown(f"- **Simulationen:** {stats['simulations']:,}")
        
        st.divider()
        
        # Info
        st.markdown("### ℹ️ Info")
        st.markdown(
            f"**Version:** {APP_VERSION}\n\n"
            "Optimieren Sie Ihre Terminfixierungsstrategie durch "
            "historisches Backtesting."
        )
        
        # Hilfe-Button
        with st.expander("❓ Hilfe & FAQ"):
            st.markdown("""
**Schnellstart:**
1. Laden Sie Lastprofile und Spotpreise hoch
2. Konfigurieren Sie Ihre Strategie
3. Führen Sie Backtests durch
4. Analysieren Sie die Ergebnisse

**Strategien:**
- **Gleichmäßig**: Konstante Fixierung
- **Frontloaded**: 70% früh, 30% spät
- **Backloaded**: 30% früh, 70% spät
- **Regelbasiert**: Kauft unter Preisschwelle
- **Gleitender Ø**: Technische Analyse

**Metriken:**
- **Sharpe Ratio**: >1 ist gut
- **Max Drawdown**: Höchster Verlust
- **VaR**: Value at Risk

**Datenformate:**
- CSV (Komma/Semikolon/Tab)
- Excel (.xlsx)
- Copy-Paste aus Tabellen
            """)
    
    # Hauptinhalt basierend auf Navigation
    st.title(APP_TITLE)
    
    if page == "📊 Dashboard":
        render_dashboard()
    
    elif page == "📁 Daten hochladen":
        render_data_upload()
    
    elif page == "⚙️ Simulation konfigurieren":
        render_simulation()
    
    elif page == "📈 Ergebnisse & Visualisierungen":
        render_results()
    
    elif page == "💾 Datenbank-Management":
        render_db_management()
    
    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "⚡ Energiebeschaffungs-Backtesting | "
        "Professionelle Strategieoptimierung"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

"""
Data Validation Module
Validiert Daten und überprüft Datenqualität
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataValidator:
    """Validiert Trading-Daten"""

    @staticmethod
    def validate_price_data(df: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Validiert Preis-Daten

        Args:
            df: DataFrame mit OHLCV Daten
            symbol: Aktiensymbol

        Returns:
            Tuple (is_valid, error_messages)
        """
        errors = []

        # Prüfe erforderliche Spalten
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Fehlende Spalten: {', '.join(missing_columns)}")
            return False, errors

        # Prüfe auf leere Daten
        if df.empty:
            errors.append("DataFrame ist leer")
            return False, errors

        # Prüfe auf minimale Datenpunkte
        if len(df) < 50:
            errors.append(f"Zu wenig Datenpunkte: {len(df)} < 50")

        # Prüfe auf negative Preise
        for col in ['Open', 'High', 'Low', 'Close']:
            if (df[col] <= 0).any():
                errors.append(f"Negative oder Null-Werte in {col}")

        # Prüfe OHLC Logik: High >= Low, High >= Open/Close, Low <= Open/Close
        invalid_high_low = df[df['High'] < df['Low']]
        if len(invalid_high_low) > 0:
            errors.append(f"High < Low in {len(invalid_high_low)} Zeilen")

        invalid_high = df[(df['High'] < df['Open']) | (df['High'] < df['Close'])]
        if len(invalid_high) > 0:
            errors.append(f"High kleiner als Open/Close in {len(invalid_high)} Zeilen")

        invalid_low = df[(df['Low'] > df['Open']) | (df['Low'] > df['Close'])]
        if len(invalid_low) > 0:
            errors.append(f"Low größer als Open/Close in {len(invalid_low)} Zeilen")

        # Prüfe auf negative Volumen
        if (df['Volume'] < 0).any():
            errors.append("Negative Volumen gefunden")

        # Prüfe auf Duplikate im Index
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            errors.append(f"{dup_count} duplizierte Zeitstempel gefunden")

        # Prüfe auf NaN/Inf Werte
        nan_counts = df[required_columns].isnull().sum()
        if nan_counts.any():
            errors.append(f"NaN Werte gefunden: {nan_counts[nan_counts > 0].to_dict()}")

        inf_mask = np.isinf(df[required_columns].select_dtypes(include=[np.number])).any()
        if inf_mask.any():
            errors.append(f"Inf Werte gefunden in: {inf_mask[inf_mask].index.tolist()}")

        # Prüfe auf unrealistische Preissprünge (> 50% pro Tag)
        price_changes = df['Close'].pct_change().abs()
        extreme_changes = price_changes[price_changes > 0.5]
        if len(extreme_changes) > 0:
            errors.append(f"{len(extreme_changes)} extreme Preissprünge (>50%) gefunden")

        # Prüfe auf konstante Preise (Warnung, nicht Fehler)
        if df['Close'].nunique() == 1:
            errors.append("WARNUNG: Alle Close-Preise sind identisch")

        is_valid = len([e for e in errors if not e.startswith("WARNUNG")]) == 0

        if not is_valid:
            logger.warning(f"Datenvalidierung für {symbol} fehlgeschlagen: {len(errors)} Probleme gefunden")
        elif errors:
            logger.info(f"Datenvalidierung für {symbol}: {len(errors)} Warnungen")

        return is_valid, errors

    @staticmethod
    def clean_price_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Bereinigt Preis-Daten

        Args:
            df: DataFrame mit OHLCV Daten
            symbol: Aktiensymbol

        Returns:
            Bereinigtes DataFrame
        """
        original_len = len(df)

        # Entferne Zeilen mit negativen/Null Preisen
        df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]

        # Entferne Zeilen mit negativem Volumen
        df = df[df['Volume'] >= 0]

        # Korrigiere OHLC Logik wenn möglich
        df.loc[df['High'] < df['Low'], 'High'] = df['Low']
        df.loc[df['High'] < df['Open'], 'High'] = df['Open']
        df.loc[df['High'] < df['Close'], 'High'] = df['Close']
        df.loc[df['Low'] > df['Open'], 'Low'] = df['Open']
        df.loc[df['Low'] > df['Close'], 'Low'] = df['Close']

        # Entferne Duplikate
        df = df[~df.index.duplicated(keep='last')]

        # Forward-fill für kleine Lücken (max 3 Tage)
        df = df.sort_index()
        df = df.fillna(method='ffill', limit=3)

        # Entferne verbleibende NaN
        df = df.dropna()

        # Entferne Inf Werte
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"{symbol}: {removed} Zeilen während Bereinigung entfernt")

        return df

    @staticmethod
    def validate_indicators(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validiert technische Indikatoren

        Args:
            df: DataFrame mit Indikatoren

        Returns:
            Tuple (is_valid, error_messages)
        """
        errors = []

        # Prüfe RSI Bereich (0-100)
        if 'RSI' in df.columns:
            invalid_rsi = df[(df['RSI'] < 0) | (df['RSI'] > 100)]
            if len(invalid_rsi) > 0:
                errors.append(f"RSI außerhalb 0-100 Bereich: {len(invalid_rsi)} Werte")

        # Prüfe Bollinger Bands Logik
        if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            invalid_bb = df[(df['BB_Upper'] < df['BB_Middle']) |
                           (df['BB_Middle'] < df['BB_Lower'])]
            if len(invalid_bb) > 0:
                errors.append(f"Ungültige Bollinger Band Ordnung: {len(invalid_bb)} Zeilen")

        # Prüfe Moving Averages
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            # SMAs sollten nicht negativ sein
            if (df['SMA_50'] < 0).any() or (df['SMA_200'] < 0).any():
                errors.append("Negative Moving Averages gefunden")

        # Prüfe MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            # Histogramm sollte Differenz sein
            if 'MACD_Hist' in df.columns:
                calculated_hist = df['MACD'] - df['MACD_Signal']
                hist_diff = (df['MACD_Hist'] - calculated_hist).abs()
                if (hist_diff > 0.001).any():  # Toleranz für Rundungsfehler
                    errors.append("MACD Histogram stimmt nicht mit MACD - Signal überein")

        # Prüfe Confidence Werte (sollten 0-1 sein)
        if 'Confidence' in df.columns:
            invalid_conf = df[(df['Confidence'] < 0) | (df['Confidence'] > 1)]
            if len(invalid_conf) > 0:
                errors.append(f"Confidence außerhalb 0-1 Bereich: {len(invalid_conf)} Werte")

        is_valid = len(errors) == 0
        return is_valid, errors


class TradeValidator:
    """Validiert Trades und Portfolio-Operationen"""

    @staticmethod
    def validate_trade(
        action: str,
        symbol: str,
        price: float,
        shares: int,
        cash_available: float
    ) -> Tuple[bool, str]:
        """
        Validiert einen Trade

        Args:
            action: 'BUY' oder 'SELL'
            symbol: Aktiensymbol
            price: Preis pro Aktie
            shares: Anzahl Aktien
            cash_available: Verfügbares Cash

        Returns:
            Tuple (is_valid, error_message)
        """
        # Prüfe Action
        if action not in ['BUY', 'SELL']:
            return False, f"Ungültige Action: {action}"

        # Prüfe Symbol
        if not symbol or not isinstance(symbol, str):
            return False, "Ungültiges Symbol"

        # Prüfe Preis
        if price <= 0:
            return False, f"Ungültiger Preis: {price}"

        if not np.isfinite(price):
            return False, "Preis ist Inf oder NaN"

        # Prüfe Shares
        if shares <= 0:
            return False, f"Ungültige Anzahl Aktien: {shares}"

        if not isinstance(shares, int):
            return False, "Shares muss Integer sein"

        # Prüfe Cash bei Kauf
        if action == 'BUY':
            required = price * shares
            if required > cash_available:
                return False, f"Nicht genug Cash: ${required:.2f} > ${cash_available:.2f}"

        return True, ""

    @staticmethod
    def validate_portfolio_state(
        cash: float,
        positions: Dict,
        initial_capital: float
    ) -> Tuple[bool, List[str]]:
        """
        Validiert Portfolio-Status

        Args:
            cash: Aktuelles Cash
            positions: Dictionary mit Positionen
            initial_capital: Startkapital

        Returns:
            Tuple (is_valid, warnings)
        """
        warnings = []

        # Prüfe Cash
        if cash < 0:
            warnings.append(f"KRITISCH: Negatives Cash: ${cash:.2f}")

        if not np.isfinite(cash):
            warnings.append("KRITISCH: Cash ist Inf oder NaN")

        # Prüfe Positionen
        for symbol, position in positions.items():
            if position.shares <= 0:
                warnings.append(f"WARNUNG: {symbol} hat ungültige Shares: {position.shares}")

            if position.entry_price <= 0:
                warnings.append(f"WARNUNG: {symbol} hat ungültigen Entry Price: {position.entry_price}")

            if position.current_price <= 0:
                warnings.append(f"WARNUNG: {symbol} hat ungültigen Current Price: {position.current_price}")

        # Prüfe Gesamtwert
        total_position_value = sum(pos.get_value() for pos in positions.values())
        total_value = cash + total_position_value

        if total_value < initial_capital * 0.5:
            warnings.append(f"WARNUNG: Portfolio-Wert unter 50% des Startkapitals: ${total_value:.2f}")

        if total_value < 0:
            warnings.append(f"KRITISCH: Negativer Portfolio-Wert: ${total_value:.2f}")

        # Prüfe Cash-Reserve
        cash_ratio = cash / total_value if total_value > 0 else 0
        if cash_ratio < 0.05:
            warnings.append(f"WARNUNG: Sehr niedrige Cash-Reserve: {cash_ratio*100:.1f}%")

        is_valid = len([w for w in warnings if w.startswith("KRITISCH")]) == 0
        return is_valid, warnings


class ConfigValidator:
    """Validiert Konfigurationsparameter"""

    @staticmethod
    def validate_config(config: Dict) -> Tuple[bool, List[str]]:
        """
        Validiert Trading-Konfiguration

        Args:
            config: Konfigurations-Dictionary

        Returns:
            Tuple (is_valid, errors)
        """
        errors = []

        # Prüfe Portfolio Config
        if 'PORTFOLIO_CONFIG' in config:
            pc = config['PORTFOLIO_CONFIG']

            if pc.get('initial_capital', 0) <= 0:
                errors.append("initial_capital muss > 0 sein")

            if not (0 <= pc.get('max_position_size', 0) <= 1):
                errors.append("max_position_size muss zwischen 0 und 1 sein")

            if not (0 <= pc.get('min_cash_reserve', 0) <= 1):
                errors.append("min_cash_reserve muss zwischen 0 und 1 sein")

            if pc.get('max_positions', 0) < 1:
                errors.append("max_positions muss >= 1 sein")

        # Prüfe Risk Config
        if 'RISK_CONFIG' in config:
            rc = config['RISK_CONFIG']

            if not (0 < rc.get('stop_loss_percent', 0) < 1):
                errors.append("stop_loss_percent muss zwischen 0 und 1 sein")

            if not (0 < rc.get('take_profit_percent', 0) < 2):
                errors.append("take_profit_percent sollte zwischen 0 und 2 sein")

        # Prüfe Strategy Config
        if 'STRATEGY_CONFIG' in config:
            sc = config['STRATEGY_CONFIG']

            if sc.get('short_window', 0) <= 0:
                errors.append("short_window muss > 0 sein")

            if sc.get('long_window', 0) <= sc.get('short_window', 0):
                errors.append("long_window muss > short_window sein")

            if not (0 < sc.get('rsi_period', 0) <= 100):
                errors.append("rsi_period sollte zwischen 1 und 100 sein")

        # Prüfe Watchlist
        if 'WATCHLIST' in config:
            wl = config['WATCHLIST']
            if not isinstance(wl, list) or len(wl) == 0:
                errors.append("WATCHLIST muss nicht-leere Liste sein")

        is_valid = len(errors) == 0
        return is_valid, errors

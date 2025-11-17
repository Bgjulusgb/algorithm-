"""
Data Handler für Yahoo Finance Daten - Verbesserte Version
Mit Caching, Fehlerbehandlung, und erweiterten Indikatoren
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pickle
import os
from pathlib import Path
import time
import logging

from config import DATA_CONFIG, STRATEGY_CONFIG

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCache:
    """Einfaches Caching-System für Yahoo Finance Daten"""
    
    def __init__(self, cache_dir: str = "cache", duration_hours: int = 1):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.duration = timedelta(hours=duration_hours)
    
    def _get_cache_path(self, symbol: str, data_type: str = "history") -> Path:
        """Generiert Cache-Pfad für Symbol"""
        return self.cache_dir / f"{symbol}_{data_type}.pkl"
    
    def get(self, symbol: str, data_type: str = "history") -> Optional[pd.DataFrame]:
        """Holt Daten aus Cache wenn verfügbar und aktuell"""
        if not DATA_CONFIG.get("enable_cache", True):
            return None
        
        cache_path = self._get_cache_path(symbol, data_type)
        
        if not cache_path.exists():
            return None
        
        try:
            # Prüfe Alter der Cache-Datei
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mtime > self.duration:
                logger.debug(f"Cache für {symbol} abgelaufen")
                return None
            
            # Lade aus Cache
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.debug(f"Daten für {symbol} aus Cache geladen")
            return data
        
        except Exception as e:
            logger.warning(f"Fehler beim Laden aus Cache für {symbol}: {e}")
            return None
    
    def set(self, symbol: str, data: pd.DataFrame, data_type: str = "history"):
        """Speichert Daten im Cache"""
        if not DATA_CONFIG.get("enable_cache", True):
            return
        
        cache_path = self._get_cache_path(symbol, data_type)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Daten für {symbol} im Cache gespeichert")
        except Exception as e:
            logger.warning(f"Fehler beim Speichern im Cache für {symbol}: {e}")
    
    def clear(self, symbol: Optional[str] = None):
        """Löscht Cache für Symbol oder alle"""
        if symbol:
            cache_path = self._get_cache_path(symbol)
            if cache_path.exists():
                cache_path.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        logger.info(f"Cache gelöscht für {symbol or 'alle Symbole'}")


class DataHandler:
    """Erweiterter Data Handler mit Caching und besserer Fehlerbehandlung"""
    
    def __init__(self):
        self.cache = DataCache(
            cache_dir=DATA_CONFIG.get("cache_dir", "cache"),
            duration_hours=DATA_CONFIG.get("cache_duration_hours", 1)
        )
        self.rate_limiter = RateLimiter(
            max_calls=DATA_CONFIG.get("rate_limit_calls", 2000),
            window=DATA_CONFIG.get("rate_limit_window", 3600)
        )
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: Optional[str] = None, 
        interval: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Lädt historische Daten mit Caching und Fehlerbehandlung
        
        Args:
            symbol: Aktiensymbol
            period: Zeitraum (z.B. "1y", "2y")
            interval: Intervall (z.B. "1d", "1h")
            use_cache: Cache verwenden
        
        Returns:
            DataFrame mit OHLCV Daten oder None
        """
        period = period or DATA_CONFIG["history_period"]
        interval = interval or DATA_CONFIG["interval"]
        
        # Versuche aus Cache zu laden
        if use_cache:
            cached_data = self.cache.get(symbol, f"history_{period}_{interval}")
            if cached_data is not None:
                return cached_data
        
        # Rate Limiting
        self.rate_limiter.wait_if_needed()
        
        # Lade von Yahoo Finance mit Retry-Logik
        for attempt in range(DATA_CONFIG.get("retry_attempts", 3)):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    logger.warning(f"Keine Daten für {symbol} gefunden")
                    return None
                
                # Validiere Datenqualität
                if not self._validate_data_quality(df, symbol):
                    logger.warning(f"Datenqualität für {symbol} unzureichend")
                    return None
                
                # Bereinige Daten
                df = self._clean_data(df)
                
                # Speichere im Cache
                if use_cache:
                    self.cache.set(symbol, df, f"history_{period}_{interval}")
                
                logger.info(f"✅ Daten für {symbol} geladen: {len(df)} Einträge")
                return df
            
            except Exception as e:
                logger.warning(f"Versuch {attempt + 1}/{DATA_CONFIG.get('retry_attempts', 3)} für {symbol} fehlgeschlagen: {e}")
                
                if attempt < DATA_CONFIG.get("retry_attempts", 3) - 1:
                    time.sleep(DATA_CONFIG.get("retry_delay", 5))
                else:
                    logger.error(f"❌ Konnte keine Daten für {symbol} laden nach {DATA_CONFIG.get('retry_attempts', 3)} Versuchen")
                    return None
    
    def _validate_data_quality(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validiert Datenqualität"""
        if df is None or df.empty:
            return False
        
        # Prüfe Mindestanzahl Datenpunkte
        min_points = DATA_CONFIG.get("min_data_points", 100)
        if len(df) < min_points:
            logger.warning(f"{symbol}: Zu wenig Datenpunkte ({len(df)} < {min_points})")
            return False
        
        # Prüfe fehlende Daten
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        max_missing = DATA_CONFIG.get("max_missing_data", 0.05)
        if missing_pct > max_missing:
            logger.warning(f"{symbol}: Zu viele fehlende Daten ({missing_pct:.2%} > {max_missing:.2%})")
            return False
        
        # Prüfe auf unrealistische Werte
        if (df['Close'] <= 0).any():
            logger.warning(f"{symbol}: Negative oder Null-Preise gefunden")
            return False
        
        return True
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereinigt Daten"""
        # Entferne NaN Werte
        df = df.dropna()
        
        # Sortiere nach Datum
        df = df.sort_index()
        
        # Entferne Duplikate
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Holt aktuellen Preis mit Fallback-Strategie
        
        Args:
            symbol: Aktiensymbol
        
        Returns:
            Aktueller Preis oder None
        """
        try:
            # Versuche 1: Fast Info
            ticker = yf.Ticker(symbol)
            fast_info = ticker.fast_info
            if hasattr(fast_info, 'last_price') and fast_info.last_price:
                return float(fast_info.last_price)
        except:
            pass
        
        try:
            # Versuche 2: 1-Minuten Daten
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except:
            pass
        
        try:
            # Versuche 3: Letzter Schlusskurs
            data = ticker.history(period="5d", interval="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Konnte Preis für {symbol} nicht abrufen: {e}")
        
        return None
    
    def get_multiple_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Holt aktuelle Preise für mehrere Symbole gleichzeitig (effizienter)
        
        Args:
            symbols: Liste von Aktiensymbolen
        
        Returns:
            Dictionary {symbol: price}
        """
        prices = {}
        
        try:
            # Batch-Download mit yfinance
            tickers = yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    fast_info = ticker.fast_info
                    if hasattr(fast_info, 'last_price'):
                        prices[symbol] = float(fast_info.last_price)
                    else:
                        # Fallback
                        data = ticker.history(period="1d")
                        if not data.empty:
                            prices[symbol] = float(data['Close'].iloc[-1])
                except Exception as e:
                    logger.warning(f"Konnte Preis für {symbol} nicht abrufen: {e}")
                    prices[symbol] = None
        
        except Exception as e:
            logger.error(f"Batch-Download fehlgeschlagen: {e}")
            # Fallback: Einzeln laden
            for symbol in symbols:
                prices[symbol] = self.get_current_price(symbol)
        
        return prices
    
    def calculate_sma(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Berechnet Simple Moving Average mit Fehlerbehandlung"""
        try:
            return df['Close'].rolling(window=window, min_periods=window).mean()
        except Exception as e:
            logger.error(f"Fehler bei SMA-Berechnung: {e}")
            return pd.Series(index=df.index, dtype=float)
    
    def calculate_ema(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Berechnet Exponential Moving Average"""
        try:
            return df['Close'].ewm(span=window, adjust=False, min_periods=window).mean()
        except Exception as e:
            logger.error(f"Fehler bei EMA-Berechnung: {e}")
            return pd.Series(index=df.index, dtype=float)
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Berechnet RSI mit verbesserter Stabilität
        
        Args:
            df: DataFrame mit Preisdaten
            period: RSI Periode
        
        Returns:
            Series mit RSI Werten (0-100)
        """
        try:
            delta = df['Close'].diff()
            
            # Berechne Gains und Losses
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            
            # Wilder's Smoothing
            avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            
            # Verhindere Division durch Null
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            # Fülle NaN mit neutralem Wert (50)
            rsi = rsi.fillna(50)
            
            # Clamp auf 0-100
            rsi = rsi.clip(0, 100)
            
            return rsi
        
        except Exception as e:
            logger.error(f"Fehler bei RSI-Berechnung: {e}")
            return pd.Series([50] * len(df), index=df.index)
    
    def calculate_macd(
        self, 
        df: pd.DataFrame, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Berechnet MACD, Signal und Histogram
        
        Returns:
            Tuple (MACD, Signal, Histogram)
        """
        try:
            exp1 = df['Close'].ewm(span=fast, adjust=False, min_periods=fast).mean()
            exp2 = df['Close'].ewm(span=slow, adjust=False, min_periods=slow).mean()
            
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
            histogram = macd - signal_line
            
            return macd, signal_line, histogram
        
        except Exception as e:
            logger.error(f"Fehler bei MACD-Berechnung: {e}")
            empty = pd.Series(index=df.index, dtype=float)
            return empty, empty, empty
    
    def calculate_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Berechnet Bollinger Bands
        
        Returns:
            Tuple (Upper, Middle, Lower)
        """
        try:
            middle = df['Close'].rolling(window=period, min_periods=period).mean()
            std = df['Close'].rolling(window=period, min_periods=period).std()
            
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return upper, middle, lower
        
        except Exception as e:
            logger.error(f"Fehler bei Bollinger Bands-Berechnung: {e}")
            empty = pd.Series(index=df.index, dtype=float)
            return empty, empty, empty
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Berechnet Average True Range (Volatilität)
        
        Args:
            df: DataFrame mit OHLC Daten
            period: ATR Periode
        
        Returns:
            Series mit ATR Werten
        """
        try:
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period, min_periods=period).mean()
            
            return atr
        
        except Exception as e:
            logger.error(f"Fehler bei ATR-Berechnung: {e}")
            return pd.Series(index=df.index, dtype=float)
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Berechnet volumenbasierte Indikatoren"""
        try:
            # Volume Moving Average
            df['Volume_MA'] = df['Volume'].rolling(
                window=STRATEGY_CONFIG.get("volume_ma_period", 20)
            ).mean()
            
            # Volume Ratio (current / average)
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # On-Balance Volume (OBV)
            obv = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.append(obv[-1] + df['Volume'].iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.append(obv[-1] - df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            
            df['OBV'] = obv
            
            return df
        
        except Exception as e:
            logger.error(f"Fehler bei Volume-Indikator-Berechnung: {e}")
            return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt alle technischen Indikatoren hinzu
        
        Args:
            df: DataFrame mit OHLCV Daten
        
        Returns:
            DataFrame mit zusätzlichen Indikator-Spalten
        """
        if df is None or df.empty:
            return df
        
        try:
            # Moving Averages
            df['SMA_50'] = self.calculate_sma(df, STRATEGY_CONFIG["short_window"])
            df['SMA_200'] = self.calculate_sma(df, STRATEGY_CONFIG["long_window"])
            df['EMA_12'] = self.calculate_ema(df, STRATEGY_CONFIG.get("ema_short", 12))
            df['EMA_26'] = self.calculate_ema(df, STRATEGY_CONFIG.get("ema_long", 26))
            
            # RSI
            df['RSI'] = self.calculate_rsi(df, STRATEGY_CONFIG["rsi_period"])
            
            # MACD
            macd, signal, hist = self.calculate_macd(
                df,
                STRATEGY_CONFIG.get("macd_fast", 12),
                STRATEGY_CONFIG.get("macd_slow", 26),
                STRATEGY_CONFIG.get("macd_signal", 9)
            )
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Hist'] = hist
            
            # Bollinger Bands
            upper, middle, lower = self.calculate_bollinger_bands(
                df,
                STRATEGY_CONFIG.get("bb_period", 20),
                STRATEGY_CONFIG.get("bb_std", 2)
            )
            df['BB_Upper'] = upper
            df['BB_Middle'] = middle
            df['BB_Lower'] = lower
            
            # ATR (Volatilität)
            df['ATR'] = self.calculate_atr(df, 14)
            
            # Volume Indicators
            df = self.calculate_volume_indicators(df)
            
            # Additional Indicators
            df['Daily_Return'] = df['Close'].pct_change()
            df['Price_Change'] = df['Close'].diff()
            
            logger.debug(f"Technische Indikatoren hinzugefügt: {len(df.columns)} Spalten")
            
            return df
        
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen technischer Indikatoren: {e}")
            return df
    
    def get_company_info(self, symbol: str) -> Dict:
        """
        Holt Unternehmensinformationen mit Caching
        
        Args:
            symbol: Aktiensymbol
        
        Returns:
            Dict mit Unternehmensdaten
        """
        # Versuche aus Cache
        cached = self.cache.get(symbol, "info")
        if cached is not None:
            return cached
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            company_data = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', None),
                '52w_high': info.get('fiftyTwoWeekHigh', None),
                '52w_low': info.get('fiftyTwoWeekLow', None),
            }
            
            # Speichere im Cache
            self.cache.set(symbol, company_data, "info")
            
            return company_data
        
        except Exception as e:
            logger.warning(f"Konnte Infos für {symbol} nicht laden: {e}")
            return {
                'symbol': symbol,
                'name': symbol,
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 0
            }


class RateLimiter:
    """Einfacher Rate Limiter für API Calls"""
    
    def __init__(self, max_calls: int = 2000, window: int = 3600):
        self.max_calls = max_calls
        self.window = window
        self.calls = []
    
    def wait_if_needed(self):
        """Wartet wenn Rate Limit erreicht"""
        now = time.time()
        
        # Entferne alte Calls außerhalb des Zeitfensters
        self.calls = [call_time for call_time in self.calls if now - call_time < self.window]
        
        # Prüfe ob Limit erreicht
        if len(self.calls) >= self.max_calls:
            wait_time = self.window - (now - self.calls[0]) + 1
            if wait_time > 0:
                logger.warning(f"Rate Limit erreicht. Warte {wait_time:.1f} Sekunden...")
                time.sleep(wait_time)
                self.calls = []
        
        # Füge aktuellen Call hinzu
        self.calls.append(now)

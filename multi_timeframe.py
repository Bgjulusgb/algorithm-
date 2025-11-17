"""
Multi-Timeframe Analysis Module
Analysiert Daten über mehrere Zeitrahmen hinweg
Enthält Caching für bessere Performance
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import pickle
from pathlib import Path
import hashlib
import json

logger = logging.getLogger(__name__)


class DataCache:
    """Caching-System für Market Data"""

    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialisiert Cache

        Args:
            cache_dir: Directory für Cache-Dateien
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = {
            '1m': timedelta(minutes=5),
            '5m': timedelta(minutes=15),
            '15m': timedelta(hours=1),
            '1h': timedelta(hours=4),
            '1d': timedelta(days=1),
            '1wk': timedelta(weeks=1)
        }

    def _get_cache_key(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """Generiert Cache-Key"""
        data_str = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        return hashlib.md5(data_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Gibt Cache-Dateipfad zurück"""
        return self.cache_dir / f"{cache_key}.pkl"

    def get(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Ruft Daten aus Cache ab

        Returns:
            DataFrame wenn im Cache und nicht abgelaufen, sonst None
        """
        cache_key = self._get_cache_key(symbol, timeframe, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {symbol} {timeframe}")
            return None

        try:
            # Lade Cache
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Überprüfe TTL
            cache_time = cache_data['timestamp']
            ttl = self.cache_ttl.get(timeframe, timedelta(hours=1))

            if datetime.now() - cache_time < ttl:
                logger.debug(f"Cache hit: {symbol} {timeframe}")
                return cache_data['data']
            else:
                logger.debug(f"Cache expired: {symbol} {timeframe}")
                cache_path.unlink()  # Lösche abgelaufenen Cache
                return None

        except Exception as e:
            logger.warning(f"Cache-Fehler: {e}")
            return None

    def set(self, symbol: str, timeframe: str, start_date: str, end_date: str, data: pd.DataFrame):
        """Speichert Daten im Cache"""
        cache_key = self._get_cache_key(symbol, timeframe, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)

        try:
            cache_data = {
                'timestamp': datetime.now(),
                'data': data
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            logger.debug(f"Cache gespeichert: {symbol} {timeframe}")

        except Exception as e:
            logger.warning(f"Cache-Speicherfehler: {e}")

    def clear(self, older_than: Optional[timedelta] = None):
        """
        Löscht Cache

        Args:
            older_than: Nur Dateien älter als timedelta löschen, None = alle
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                if older_than:
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if datetime.now() - file_time < older_than:
                        continue

                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Fehler beim Löschen von {cache_file}: {e}")

        logger.info(f"✅ {count} Cache-Dateien gelöscht")


class EnhancedDataFetcher:
    """Verbesserter Data Fetcher mit Error Handling und Retry Logic"""

    def __init__(self, use_cache: bool = True, cache_dir: str = ".cache"):
        """
        Initialisiert Fetcher

        Args:
            use_cache: Cache verwenden
            cache_dir: Cache-Directory
        """
        self.cache = DataCache(cache_dir) if use_cache else None
        self.max_retries = 3
        self.retry_delay = 2  # Sekunden

    def fetch_data(self,
                   symbol: str,
                   timeframe: str = '1d',
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Ruft Market Data ab mit Error Handling und Caching

        Args:
            symbol: Stock Symbol
            timeframe: '1m', '5m', '15m', '1h', '1d', '1wk'
            start_date: Start-Datum (YYYY-MM-DD)
            end_date: End-Datum (YYYY-MM-DD)
            use_cache: Cache verwenden

        Returns:
            DataFrame mit OHLCV Daten oder None bei Fehler
        """
        # Standard-Daten
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        # Cache-Check
        if use_cache and self.cache:
            cached_data = self.cache.get(symbol, timeframe, start_date, end_date)
            if cached_data is not None:
                return cached_data

        # Fetch mit Retry Logic
        for attempt in range(self.max_retries):
            try:
                import yfinance as yf
                import time

                logger.info(f"Fetching {symbol} {timeframe} (Attempt {attempt + 1}/{self.max_retries})...")

                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=timeframe,
                    actions=False
                )

                if df.empty:
                    logger.warning(f"Keine Daten für {symbol} {timeframe}")
                    return None

                # Validierung
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_columns):
                    logger.error(f"Fehlende Spalten in {symbol} Daten")
                    return None

                # Bereinigung
                df = df[required_columns]
                df = df.dropna()

                # Cache speichern
                if use_cache and self.cache:
                    self.cache.set(symbol, timeframe, start_date, end_date, df)

                logger.info(f"✅ {symbol} {timeframe}: {len(df)} Datenpunkte geladen")
                return df

            except Exception as e:
                logger.warning(f"Fetch-Fehler (Attempt {attempt + 1}): {e}")

                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"❌ Fetch fehlgeschlagen nach {self.max_retries} Versuchen: {symbol} {timeframe}")
                    return None

        return None


class MultiTimeframeAnalyzer:
    """Multi-Timeframe Analyse"""

    def __init__(self, use_cache: bool = True):
        """
        Initialisiert Analyzer

        Args:
            use_cache: Cache für Datenabfragen verwenden
        """
        self.fetcher = EnhancedDataFetcher(use_cache=use_cache)
        self.timeframes = ['1d', '1h', '15m']  # Standard-Timeframes

    def fetch_multiple_timeframes(self,
                                  symbol: str,
                                  timeframes: Optional[List[str]] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Lädt Daten für mehrere Timeframes

        Args:
            symbol: Stock Symbol
            timeframes: Liste von Timeframes (default: ['1d', '1h', '15m'])
            start_date: Start-Datum
            end_date: End-Datum

        Returns:
            Dict {timeframe: DataFrame}
        """
        if timeframes is None:
            timeframes = self.timeframes

        data = {}
        for tf in timeframes:
            df = self.fetcher.fetch_data(symbol, tf, start_date, end_date)
            if df is not None:
                data[tf] = df

        logger.info(f"✅ {len(data)}/{len(timeframes)} Timeframes geladen für {symbol}")
        return data

    def calculate_trend_alignment(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        Berechnet Trend-Ausrichtung über mehrere Timeframes

        Wenn alle Timeframes bullish sind → Strong Bullish
        Wenn alle Timeframes bearish sind → Strong Bearish

        Returns:
            Dict mit Trend-Info
        """
        trends = {}

        for tf, df in data_dict.items():
            if len(df) < 50:
                continue

            # Simple Trend: SMA 20 vs SMA 50
            sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
            current_price = df['Close'].iloc[-1]

            if pd.isna(sma_20) or pd.isna(sma_50):
                trends[tf] = 'unknown'
            elif current_price > sma_20 > sma_50:
                trends[tf] = 'bullish'
            elif current_price < sma_20 < sma_50:
                trends[tf] = 'bearish'
            else:
                trends[tf] = 'neutral'

        # Gesamtbewertung
        bullish_count = sum(1 for t in trends.values() if t == 'bullish')
        bearish_count = sum(1 for t in trends.values() if t == 'bearish')
        total = len(trends)

        if bullish_count == total:
            alignment = 'STRONG_BULLISH'
        elif bullish_count > total / 2:
            alignment = 'BULLISH'
        elif bearish_count == total:
            alignment = 'STRONG_BEARISH'
        elif bearish_count > total / 2:
            alignment = 'BEARISH'
        else:
            alignment = 'NEUTRAL'

        return {
            'individual_trends': trends,
            'alignment': alignment,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'confidence': max(bullish_count, bearish_count) / total if total > 0 else 0
        }

    def calculate_higher_timeframe_support_resistance(self,
                                                      data_dict: Dict[str, pd.DataFrame],
                                                      num_levels: int = 3) -> Dict[str, List[float]]:
        """
        Findet Support/Resistance Levels aus höheren Timeframes

        Levels aus höheren Timeframes sind oft signifikanter
        """
        all_support = []
        all_resistance = []

        # Priorität: Längere Timeframes wichtiger
        timeframe_priority = {'1wk': 3, '1d': 2, '1h': 1, '15m': 0.5}

        for tf, df in data_dict.items():
            if len(df) < 20:
                continue

            weight = timeframe_priority.get(tf, 1)

            # Finde Pivots
            highs = df['High'].values
            lows = df['Low'].values

            window = min(10, len(df) // 4)

            for i in range(window, len(df) - window):
                # Resistance
                if highs[i] == max(highs[i-window:i+window+1]):
                    all_resistance.extend([highs[i]] * int(weight))

                # Support
                if lows[i] == min(lows[i-window:i+window+1]):
                    all_support.extend([lows[i]] * int(weight))

        # Cluster ähnliche Levels
        def cluster_levels(levels, tolerance=0.015):
            if not levels:
                return []

            levels = sorted(levels)
            clustered = []
            current_cluster = [levels[0]]

            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                    current_cluster.append(level)
                else:
                    clustered.append(np.mean(current_cluster))
                    current_cluster = [level]

            clustered.append(np.mean(current_cluster))
            return clustered

        support_levels = cluster_levels(all_support)[-num_levels:]
        resistance_levels = cluster_levels(all_resistance)[-num_levels:]

        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels, reverse=True)
        }

    def get_confluence_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        Findet Confluence (Zusammenfluss) von Signalen über Timeframes

        Confluence = Mehrere Timeframes zeigen gleiches Signal
        → Höhere Wahrscheinlichkeit

        Returns:
            Dict mit Confluence-Analyse
        """
        signals = {}

        for tf, df in data_dict.items():
            if len(df) < 50:
                continue

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1]

            # MACD
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal_line = macd.ewm(span=9, adjust=False).mean()
            macd_value = macd.iloc[-1]
            signal_value = signal_line.iloc[-1]

            # Trend (SMA)
            sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
            current_price = df['Close'].iloc[-1]

            # Einzelsignale
            rsi_signal = 'buy' if rsi_value < 30 else ('sell' if rsi_value > 70 else 'neutral')
            macd_signal = 'buy' if macd_value > signal_value else 'sell'
            trend_signal = 'buy' if current_price > sma_50 else 'sell'

            signals[tf] = {
                'rsi': {'value': rsi_value, 'signal': rsi_signal},
                'macd': {'value': macd_value - signal_value, 'signal': macd_signal},
                'trend': {'signal': trend_signal}
            }

        # Confluence-Bewertung
        buy_confluence = 0
        sell_confluence = 0

        for tf_signals in signals.values():
            if tf_signals['rsi']['signal'] == 'buy':
                buy_confluence += 1
            elif tf_signals['rsi']['signal'] == 'sell':
                sell_confluence += 1

            if tf_signals['macd']['signal'] == 'buy':
                buy_confluence += 1
            elif tf_signals['macd']['signal'] == 'sell':
                sell_confluence += 1

            if tf_signals['trend']['signal'] == 'buy':
                buy_confluence += 1
            elif tf_signals['trend']['signal'] == 'sell':
                sell_confluence += 1

        total_signals = (buy_confluence + sell_confluence) or 1
        confluence_strength = max(buy_confluence, sell_confluence) / total_signals

        overall = 'STRONG_BUY' if buy_confluence > total_signals * 0.7 else (
            'BUY' if buy_confluence > sell_confluence else (
                'STRONG_SELL' if sell_confluence > total_signals * 0.7 else (
                    'SELL' if sell_confluence > buy_confluence else 'NEUTRAL'
                )
            )
        )

        return {
            'timeframe_signals': signals,
            'buy_confluence': buy_confluence,
            'sell_confluence': sell_confluence,
            'confluence_strength': confluence_strength,
            'overall_signal': overall
        }

    def analyze_symbol(self,
                      symbol: str,
                      timeframes: Optional[List[str]] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Dict[str, any]:
        """
        Vollständige Multi-Timeframe Analyse

        Returns:
            Dict mit kompletter Analyse
        """
        logger.info(f"📊 Starte Multi-Timeframe Analyse für {symbol}...")

        # Lade Daten
        data_dict = self.fetch_multiple_timeframes(symbol, timeframes, start_date, end_date)

        if not data_dict:
            logger.error(f"❌ Keine Daten für {symbol}")
            return {'error': 'no_data'}

        # Analysen
        trend_alignment = self.calculate_trend_alignment(data_dict)
        support_resistance = self.calculate_higher_timeframe_support_resistance(data_dict)
        confluence = self.get_confluence_signals(data_dict)

        result = {
            'symbol': symbol,
            'analyzed_timeframes': list(data_dict.keys()),
            'trend_alignment': trend_alignment,
            'support_resistance': support_resistance,
            'confluence': confluence,
            'recommendation': self._generate_recommendation(trend_alignment, confluence)
        }

        logger.info(f"✅ Analyse abgeschlossen: {result['recommendation']}")

        return result

    def _generate_recommendation(self, trend_alignment: Dict, confluence: Dict) -> str:
        """Generiert Trading-Empfehlung basierend auf Analyse"""

        trend_signal = trend_alignment['alignment']
        confluence_signal = confluence['overall_signal']
        confluence_strength = confluence['confluence_strength']

        # Beide stark bullish
        if trend_signal in ['STRONG_BULLISH', 'BULLISH'] and confluence_signal in ['STRONG_BUY', 'BUY']:
            if confluence_strength > 0.7:
                return 'STRONG BUY - High Confidence'
            else:
                return 'BUY - Medium Confidence'

        # Beide stark bearish
        elif trend_signal in ['STRONG_BEARISH', 'BEARISH'] and confluence_signal in ['STRONG_SELL', 'SELL']:
            if confluence_strength > 0.7:
                return 'STRONG SELL - High Confidence'
            else:
                return 'SELL - Medium Confidence'

        # Gemischt
        elif trend_signal in ['STRONG_BULLISH', 'BULLISH'] and confluence_signal in ['STRONG_SELL', 'SELL']:
            return 'NEUTRAL - Conflicting Signals (Trend Bull, Indicators Bear)'

        elif trend_signal in ['STRONG_BEARISH', 'BEARISH'] and confluence_signal in ['STRONG_BUY', 'BUY']:
            return 'NEUTRAL - Conflicting Signals (Trend Bear, Indicators Bull)'

        else:
            return 'HOLD - Wait for clearer signals'

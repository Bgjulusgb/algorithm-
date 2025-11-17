"""
Trading Strategien - Verbesserte Version
Mit Konfidenz-Scores, besserer Signal-Qualität und erweiterten Strategien
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging

from config import STRATEGY_CONFIG, RISK_CONFIG

logger = logging.getLogger(__name__)


class Signal:
    """Signal-Klasse mit Konfidenz-Score"""
    
    def __init__(
        self, 
        direction: int,  # 1=Buy, -1=Sell, 0=Hold
        confidence: float = 0.5,  # 0-1
        reason: str = "",
        indicators: Dict = None
    ):
        self.direction = direction
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp 0-1
        self.reason = reason
        self.indicators = indicators or {}
    
    def __repr__(self):
        direction_str = {1: "BUY", -1: "SELL", 0: "HOLD"}[self.direction]
        return f"Signal({direction_str}, conf={self.confidence:.2f}, reason='{self.reason}')"
    
    def is_strong(self, threshold: float = 0.7) -> bool:
        """Prüft ob Signal stark genug ist"""
        return self.confidence >= threshold


class TradingStrategy:
    """Basis-Klasse für Trading-Strategien"""

    def __init__(self, name: str = "Base Strategy", use_filters: bool = True):
        self.name = name
        self.min_confidence = 0.5  # Minimale Konfidenz für Trade
        self.use_filters = use_filters  # Signal-Filter aktivieren

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generiert Buy/Sell Signale mit Konfidenz

        Args:
            df: DataFrame mit Preisdaten und Indikatoren

        Returns:
            DataFrame mit Signal-Spalten
        """
        raise NotImplementedError("Subklasse muss generate_signals implementieren")

    def _add_signal_columns(self, df: pd.DataFrame):
        """Fügt Standard-Signal-Spalten hinzu"""
        if 'Signal' not in df.columns:
            df['Signal'] = 0
        if 'Confidence' not in df.columns:
            df['Confidence'] = 0.0
        if 'Position' not in df.columns:
            df['Position'] = 0
        return df

    def _calculate_signal_change(self, df: pd.DataFrame):
        """Berechnet Positionsänderungen"""
        df['Position'] = df['Signal'].diff()
        return df

    def apply_signal_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wendet mathematische Filter auf Signale an

        Args:
            df: DataFrame mit Signalen

        Returns:
            DataFrame mit gefilterten Signalen
        """
        if not self.use_filters or 'Signal' not in df.columns:
            return df

        try:
            from math_utils import SignalFilters

            # Glatte Signale mit EMA Filter
            df['Signal_Filtered'] = SignalFilters.exponential_moving_average_filter(
                df['Signal'],
                span=3
            )

            # Normalisiere Konfidenz mit Z-Score
            if 'Confidence' in df.columns:
                df['Confidence_Normalized'] = SignalFilters.z_score_normalization(
                    df['Confidence'],
                    window=20
                )
                # Clamp auf 0-1
                df['Confidence_Normalized'] = df['Confidence_Normalized'].clip(0, 1)

            logger.debug(f"Signal-Filter angewendet für {self.name}")

        except ImportError:
            logger.debug("math_utils nicht verfügbar, überspringe Filter")
        except Exception as e:
            logger.warning(f"Fehler beim Anwenden von Signal-Filtern: {e}")

        return df


class SMAStrategy(TradingStrategy):
    """
    Verbesserte SMA Crossover Strategie
    - Golden Cross / Death Cross
    - Trendstärke-Analyse
    - Volume-Bestätigung
    """
    
    def __init__(self):
        super().__init__("SMA Crossover Enhanced")
        self.short_window = STRATEGY_CONFIG["short_window"]
        self.long_window = STRATEGY_CONFIG["long_window"]
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generiert SMA Crossover Signale mit Konfidenz"""
        if df is None or df.empty:
            return df
        
        df = self._add_signal_columns(df)
        
        # Berechne Crossover
        df['SMA_Diff'] = df['SMA_50'] - df['SMA_200']
        df['SMA_Diff_Prev'] = df['SMA_Diff'].shift(1)
        
        for idx in df.index:
            if pd.isna(df.loc[idx, 'SMA_50']) or pd.isna(df.loc[idx, 'SMA_200']):
                continue
            
            current_diff = df.loc[idx, 'SMA_Diff']
            prev_diff = df.loc[idx, 'SMA_Diff_Prev']
            
            if pd.isna(prev_diff):
                continue
            
            # Golden Cross (Bullish)
            if prev_diff <= 0 and current_diff > 0:
                confidence = self._calculate_confidence(df, idx, direction=1)
                df.loc[idx, 'Signal'] = 1
                df.loc[idx, 'Confidence'] = confidence
            
            # Death Cross (Bearish)
            elif prev_diff >= 0 and current_diff < 0:
                confidence = self._calculate_confidence(df, idx, direction=-1)
                df.loc[idx, 'Signal'] = -1
                df.loc[idx, 'Confidence'] = confidence
            
            # Hold
            else:
                df.loc[idx, 'Signal'] = 0
                df.loc[idx, 'Confidence'] = 0.0
        
        df = self._calculate_signal_change(df)
        return df
    
    def _calculate_confidence(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        direction: int
    ) -> float:
        """
        Berechnet Konfidenz basierend auf mehreren Faktoren
        
        Args:
            df: DataFrame
            idx: Index des Signals
            direction: 1 für Buy, -1 für Sell
        
        Returns:
            Konfidenz-Score 0-1
        """
        confidence = 0.5  # Base confidence
        
        try:
            # Faktor 1: Trendstärke (Gap zwischen SMAs)
            sma_diff_pct = abs(df.loc[idx, 'SMA_Diff']) / df.loc[idx, 'SMA_200']
            if sma_diff_pct > 0.02:  # > 2% Gap
                confidence += 0.15
            
            # Faktor 2: Volume Bestätigung
            if 'Volume_Ratio' in df.columns:
                volume_ratio = df.loc[idx, 'Volume_Ratio']
                if volume_ratio > 1.2:  # Überdurchschnittliches Volumen
                    confidence += 0.15
            
            # Faktor 3: RSI unterstützt Signal
            if 'RSI' in df.columns:
                rsi = df.loc[idx, 'RSI']
                if direction == 1 and rsi < 50:  # Buy bei RSI unter 50
                    confidence += 0.10
                elif direction == -1 and rsi > 50:  # Sell bei RSI über 50
                    confidence += 0.10
            
            # Faktor 4: MACD Bestätigung
            if 'MACD_Hist' in df.columns:
                macd_hist = df.loc[idx, 'MACD_Hist']
                if direction == 1 and macd_hist > 0:
                    confidence += 0.10
                elif direction == -1 and macd_hist < 0:
                    confidence += 0.10
        
        except Exception as e:
            logger.warning(f"Fehler bei Konfidenz-Berechnung: {e}")
        
        return min(1.0, confidence)


class RSIStrategy(TradingStrategy):
    """
    Verbesserte RSI Strategie mit dynamischen Levels
    """
    
    def __init__(self):
        super().__init__("RSI Enhanced")
        self.oversold = STRATEGY_CONFIG["rsi_oversold"]
        self.overbought = STRATEGY_CONFIG["rsi_overbought"]
        self.extreme_oversold = STRATEGY_CONFIG.get("rsi_extreme_oversold", 20)
        self.extreme_overbought = STRATEGY_CONFIG.get("rsi_extreme_overbought", 80)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generiert RSI Signale mit Konfidenz"""
        if df is None or df.empty:
            return df
        
        df = self._add_signal_columns(df)
        
        # RSI Divergenz
        df['RSI_Prev'] = df['RSI'].shift(1)
        
        for idx in df.index:
            if pd.isna(df.loc[idx, 'RSI']):
                continue
            
            rsi = df.loc[idx, 'RSI']
            rsi_prev = df.loc[idx, 'RSI_Prev']
            
            if pd.isna(rsi_prev):
                continue
            
            # Buy Signal: RSI kreuzt von unten über oversold
            if rsi_prev <= self.oversold and rsi > self.oversold:
                confidence = self._calculate_buy_confidence(df, idx, rsi)
                df.loc[idx, 'Signal'] = 1
                df.loc[idx, 'Confidence'] = confidence
            
            # Sell Signal: RSI kreuzt von oben unter overbought
            elif rsi_prev >= self.overbought and rsi < self.overbought:
                confidence = self._calculate_sell_confidence(df, idx, rsi)
                df.loc[idx, 'Signal'] = -1
                df.loc[idx, 'Confidence'] = confidence
            
            else:
                df.loc[idx, 'Signal'] = 0
                df.loc[idx, 'Confidence'] = 0.0
        
        df = self._calculate_signal_change(df)
        return df
    
    def _calculate_buy_confidence(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        rsi: float
    ) -> float:
        """Berechnet Buy-Konfidenz für RSI"""
        confidence = 0.5
        
        try:
            # Extremer Oversold = höhere Konfidenz
            if rsi < self.extreme_oversold:
                confidence += 0.2
            
            # Preis nahe Bollinger Lower Band
            if 'BB_Lower' in df.columns:
                price = df.loc[idx, 'Close']
                bb_lower = df.loc[idx, 'BB_Lower']
                if not pd.isna(bb_lower) and price < bb_lower * 1.02:
                    confidence += 0.15
            
            # Positive Divergenz (Preis fällt, RSI steigt)
            if len(df.loc[:idx]) > 10:
                recent_df = df.loc[:idx].tail(10)
                price_trend = recent_df['Close'].iloc[-1] < recent_df['Close'].iloc[0]
                rsi_trend = recent_df['RSI'].iloc[-1] > recent_df['RSI'].iloc[0]
                if price_trend and rsi_trend:  # Bullish Divergenz
                    confidence += 0.15
        
        except Exception as e:
            logger.warning(f"Fehler bei Buy-Konfidenz: {e}")
        
        return min(1.0, confidence)
    
    def _calculate_sell_confidence(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        rsi: float
    ) -> float:
        """Berechnet Sell-Konfidenz für RSI"""
        confidence = 0.5
        
        try:
            # Extremer Overbought = höhere Konfidenz
            if rsi > self.extreme_overbought:
                confidence += 0.2
            
            # Preis nahe Bollinger Upper Band
            if 'BB_Upper' in df.columns:
                price = df.loc[idx, 'Close']
                bb_upper = df.loc[idx, 'BB_Upper']
                if not pd.isna(bb_upper) and price > bb_upper * 0.98:
                    confidence += 0.15
            
            # Negative Divergenz (Preis steigt, RSI fällt)
            if len(df.loc[:idx]) > 10:
                recent_df = df.loc[:idx].tail(10)
                price_trend = recent_df['Close'].iloc[-1] > recent_df['Close'].iloc[0]
                rsi_trend = recent_df['RSI'].iloc[-1] < recent_df['RSI'].iloc[0]
                if price_trend and rsi_trend:  # Bearish Divergenz
                    confidence += 0.15
        
        except Exception as e:
            logger.warning(f"Fehler bei Sell-Konfidenz: {e}")
        
        return min(1.0, confidence)


class MACDStrategy(TradingStrategy):
    """
    Verbesserte MACD Strategie
    """
    
    def __init__(self):
        super().__init__("MACD Enhanced")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generiert MACD Signale"""
        if df is None or df.empty:
            return df
        
        df = self._add_signal_columns(df)
        
        # MACD Crossover
        df['MACD_Cross'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Cross_Prev'] = df['MACD_Cross'].shift(1)
        
        for idx in df.index:
            if pd.isna(df.loc[idx, 'MACD']) or pd.isna(df.loc[idx, 'MACD_Signal']):
                continue
            
            current_cross = df.loc[idx, 'MACD_Cross']
            prev_cross = df.loc[idx, 'MACD_Cross_Prev']
            
            if pd.isna(prev_cross):
                continue
            
            # Bullish Crossover
            if prev_cross <= 0 and current_cross > 0:
                confidence = self._calculate_confidence(df, idx, direction=1)
                df.loc[idx, 'Signal'] = 1
                df.loc[idx, 'Confidence'] = confidence
            
            # Bearish Crossover
            elif prev_cross >= 0 and current_cross < 0:
                confidence = self._calculate_confidence(df, idx, direction=-1)
                df.loc[idx, 'Signal'] = -1
                df.loc[idx, 'Confidence'] = confidence
            
            else:
                df.loc[idx, 'Signal'] = 0
                df.loc[idx, 'Confidence'] = 0.0
        
        df = self._calculate_signal_change(df)
        return df
    
    def _calculate_confidence(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        direction: int
    ) -> float:
        """Berechnet MACD Konfidenz"""
        confidence = 0.5
        
        try:
            # Histogramm-Stärke
            hist = abs(df.loc[idx, 'MACD_Hist'])
            if hist > 0.5:
                confidence += 0.15
            
            # Trend-Richtung
            macd = df.loc[idx, 'MACD']
            if direction == 1 and macd > 0:
                confidence += 0.15
            elif direction == -1 and macd < 0:
                confidence += 0.15
            
            # Volume Bestätigung
            if 'Volume_Ratio' in df.columns:
                volume_ratio = df.loc[idx, 'Volume_Ratio']
                if volume_ratio > 1.2:
                    confidence += 0.10
            
            # Preis über/unter MA
            if 'SMA_50' in df.columns:
                price = df.loc[idx, 'Close']
                sma = df.loc[idx, 'SMA_50']
                if direction == 1 and price > sma:
                    confidence += 0.10
                elif direction == -1 and price < sma:
                    confidence += 0.10
        
        except Exception as e:
            logger.warning(f"Fehler bei MACD-Konfidenz: {e}")
        
        return min(1.0, confidence)


class CombinedStrategy(TradingStrategy):
    """
    Verbesserte kombinierte Strategie
    Verwendet gewichtete Signale von mehreren Strategien
    """
    
    def __init__(self):
        super().__init__("Combined Enhanced")
        self.sma_strategy = SMAStrategy()
        self.rsi_strategy = RSIStrategy()
        self.macd_strategy = MACDStrategy()
        
        # Gewichtungen für verschiedene Strategien
        self.weights = {
            'sma': 0.35,
            'rsi': 0.30,
            'macd': 0.35
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generiert gewichtete kombinierte Signale"""
        if df is None or df.empty:
            return df
        
        # Hole Signale von allen Strategien
        df_sma = self.sma_strategy.generate_signals(df.copy())
        df_rsi = self.rsi_strategy.generate_signals(df.copy())
        df_macd = self.macd_strategy.generate_signals(df.copy())
        
        df = self._add_signal_columns(df)
        
        for idx in df.index:
            # Gewichtete Signale
            sma_signal = df_sma.loc[idx, 'Signal'] * df_sma.loc[idx, 'Confidence'] * self.weights['sma']
            rsi_signal = df_rsi.loc[idx, 'Signal'] * df_rsi.loc[idx, 'Confidence'] * self.weights['rsi']
            macd_signal = df_macd.loc[idx, 'Signal'] * df_macd.loc[idx, 'Confidence'] * self.weights['macd']
            
            # Kombiniertes Signal
            combined_score = sma_signal + rsi_signal + macd_signal
            
            # Bestimme Signal-Richtung
            min_strength = STRATEGY_CONFIG.get("min_signal_strength", 2) * 0.1
            
            if combined_score > min_strength:
                df.loc[idx, 'Signal'] = 1
                df.loc[idx, 'Confidence'] = min(1.0, abs(combined_score))
            elif combined_score < -min_strength:
                df.loc[idx, 'Signal'] = -1
                df.loc[idx, 'Confidence'] = min(1.0, abs(combined_score))
            else:
                df.loc[idx, 'Signal'] = 0
                df.loc[idx, 'Confidence'] = 0.0

        df = self._calculate_signal_change(df)

        # Wende Signal-Filter an
        df = self.apply_signal_filters(df)

        return df


class MeanReversionStrategy(TradingStrategy):
    """
    Mean Reversion Strategie
    Kauft bei Überverkauf, verkauft bei Überkauf
    """
    
    def __init__(self):
        super().__init__("Mean Reversion")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generiert Mean Reversion Signale"""
        if df is None or df.empty:
            return df
        
        df = self._add_signal_columns(df)
        
        for idx in df.index:
            if pd.isna(df.loc[idx, 'BB_Lower']) or pd.isna(df.loc[idx, 'BB_Upper']):
                continue
            
            price = df.loc[idx, 'Close']
            bb_lower = df.loc[idx, 'BB_Lower']
            bb_upper = df.loc[idx, 'BB_Upper']
            bb_middle = df.loc[idx, 'BB_Middle']
            rsi = df.loc[idx, 'RSI']
            
            # Buy: Preis unter Lower Band + RSI oversold
            if price < bb_lower and rsi < 30:
                confidence = 0.6
                # Höhere Konfidenz je tiefer unter dem Band
                distance = (bb_lower - price) / bb_middle
                confidence += min(0.3, distance * 100)
                
                df.loc[idx, 'Signal'] = 1
                df.loc[idx, 'Confidence'] = confidence
            
            # Sell: Preis über Upper Band + RSI overbought
            elif price > bb_upper and rsi > 70:
                confidence = 0.6
                # Höhere Konfidenz je höher über dem Band
                distance = (price - bb_upper) / bb_middle
                confidence += min(0.3, distance * 100)
                
                df.loc[idx, 'Signal'] = -1
                df.loc[idx, 'Confidence'] = confidence
            
            else:
                df.loc[idx, 'Signal'] = 0
                df.loc[idx, 'Confidence'] = 0.0
        
        df = self._calculate_signal_change(df)
        return df


class StrategyFactory:
    """Factory für Trading-Strategien"""
    
    @staticmethod
    def create_strategy(strategy_name: str = "sma") -> TradingStrategy:
        """
        Erstellt eine Trading-Strategie
        
        Args:
            strategy_name: Name der Strategie
        
        Returns:
            TradingStrategy Instanz
        """
        strategies = {
            "sma": SMAStrategy,
            "rsi": RSIStrategy,
            "macd": MACDStrategy,
            "combined": CombinedStrategy,
            "mean_reversion": MeanReversionStrategy,
        }
        
        strategy_class = strategies.get(strategy_name.lower(), SMAStrategy)
        return strategy_class()
    
    @staticmethod
    def list_strategies() -> list:
        """Listet verfügbare Strategien"""
        return [
            "sma - Simple Moving Average Crossover",
            "rsi - Relative Strength Index",
            "macd - Moving Average Convergence Divergence",
            "combined - Kombinierte Multi-Indikator Strategie",
            "mean_reversion - Mean Reversion (Bollinger Bands + RSI)",
        ]

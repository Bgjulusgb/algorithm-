"""
Enhanced Signal Generator - Erweiterte Signal-Generierung
Enthält 50+ verschiedene Trading-Signale aus allen Kategorien
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EnhancedSignalGenerator:
    """
    Erweiterter Signal-Generator mit 50+ verschiedenen Signalen

    Kategorien:
    1. Trend-Signale (10+)
    2. Momentum-Signale (10+)
    3. Volatility-Signale (10+)
    4. Volume-Signale (8+)
    5. Pattern-Signale (10+)
    6. Oscillator-Signale (8+)
    """

    def __init__(self):
        self.signal_weights = {
            'trend': 1.5,        # Trend ist wichtiger
            'momentum': 1.2,     # Momentum auch wichtig
            'volume': 1.0,       # Volume bestätigt
            'volatility': 0.8,   # Volatility informiert
            'pattern': 1.3,      # Patterns sind stark
            'oscillator': 1.1    # Oscillators ergänzen
        }

    def generate_comprehensive_signals(self, df: pd.DataFrame, idx: int) -> Dict:
        """
        Generiert umfassende Signale aus allen Kategorien

        Args:
            df: DataFrame mit Preisdaten und Indikatoren
            idx: Index Position

        Returns:
            Dict mit allen Signalen und aggregiertem Score
        """
        if idx < 200 or idx >= len(df):  # Brauche genug History
            return {'error': 'insufficient_data', 'total_confidence': 0.0}

        signals = {}

        # 1. TREND SIGNALE (10+)
        signals['trend'] = self._generate_trend_signals(df, idx)

        # 2. MOMENTUM SIGNALE (10+)
        signals['momentum'] = self._generate_momentum_signals(df, idx)

        # 3. VOLATILITY SIGNALE (10+)
        signals['volatility'] = self._generate_volatility_signals(df, idx)

        # 4. VOLUME SIGNALE (8+)
        signals['volume'] = self._generate_volume_signals(df, idx)

        # 5. PATTERN SIGNALE (10+)
        signals['pattern'] = self._generate_pattern_signals(df, idx)

        # 6. OSCILLATOR SIGNALE (8+)
        signals['oscillator'] = self._generate_oscillator_signals(df, idx)

        # Aggregiere alle Signale
        aggregated = self._aggregate_signals(signals)

        return {
            'signals_by_category': signals,
            'aggregated_signal': aggregated['signal'],
            'total_confidence': aggregated['confidence'],
            'signal_strength': aggregated['strength'],
            'num_signals': aggregated['num_signals'],
            'bullish_count': aggregated['bullish_count'],
            'bearish_count': aggregated['bearish_count'],
            'details': aggregated['details']
        }

    def _generate_trend_signals(self, df: pd.DataFrame, idx: int) -> Dict:
        """Generiert 10+ Trend-basierte Signale"""
        signals = []
        details = []

        try:
            close = df['Close'].iloc[idx]

            # Signal 1: SMA Crossover (50/200)
            if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
                sma_50 = df['SMA_50'].iloc[idx]
                sma_200 = df['SMA_200'].iloc[idx]
                sma_50_prev = df['SMA_50'].iloc[idx-1]
                sma_200_prev = df['SMA_200'].iloc[idx-1]

                if not pd.isna(sma_50) and not pd.isna(sma_200):
                    # Golden Cross / Death Cross
                    if sma_50_prev < sma_200_prev and sma_50 > sma_200:
                        signals.append(1.0)  # Golden Cross
                        details.append('Golden Cross (SMA 50 > SMA 200)')
                    elif sma_50_prev > sma_200_prev and sma_50 < sma_200:
                        signals.append(-1.0)  # Death Cross
                        details.append('Death Cross (SMA 50 < SMA 200)')
                    elif sma_50 > sma_200:
                        signals.append(0.5)  # Bullish alignment
                        details.append('Bullish SMA Alignment')
                    else:
                        signals.append(-0.5)  # Bearish alignment
                        details.append('Bearish SMA Alignment')

            # Signal 2: EMA Crossover (12/26)
            if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
                ema_12 = df['EMA_12'].iloc[idx]
                ema_26 = df['EMA_26'].iloc[idx]
                ema_12_prev = df['EMA_12'].iloc[idx-1]
                ema_26_prev = df['EMA_26'].iloc[idx-1]

                if not pd.isna(ema_12) and not pd.isna(ema_26):
                    if ema_12_prev < ema_26_prev and ema_12 > ema_26:
                        signals.append(0.8)
                        details.append('EMA Bullish Crossover')
                    elif ema_12_prev > ema_26_prev and ema_12 < ema_26:
                        signals.append(-0.8)
                        details.append('EMA Bearish Crossover')

            # Signal 3: Preis über/unter SMA 200
            if 'SMA_200' in df.columns:
                sma_200 = df['SMA_200'].iloc[idx]
                if not pd.isna(sma_200):
                    pct_from_sma = (close - sma_200) / sma_200
                    if pct_from_sma > 0.05:  # 5% über SMA 200
                        signals.append(0.7)
                        details.append(f'Preis {pct_from_sma*100:.1f}% über SMA 200 (Stark Bullish)')
                    elif pct_from_sma > 0:
                        signals.append(0.4)
                        details.append(f'Preis über SMA 200 (Bullish)')
                    elif pct_from_sma < -0.05:  # 5% unter SMA 200
                        signals.append(-0.7)
                        details.append(f'Preis {abs(pct_from_sma)*100:.1f}% unter SMA 200 (Stark Bearish)')
                    else:
                        signals.append(-0.4)
                        details.append(f'Preis unter SMA 200 (Bearish)')

            # Signal 4: ADX Trend Strength
            if 'ADX' in df.columns:
                adx = df['ADX'].iloc[idx]
                if not pd.isna(adx):
                    if adx > 25:  # Starker Trend
                        # Prüfe Richtung mit +DI und -DI
                        if 'DI_plus' in df.columns and 'DI_minus' in df.columns:
                            di_plus = df['DI_plus'].iloc[idx]
                            di_minus = df['DI_minus'].iloc[idx]
                            if di_plus > di_minus:
                                signals.append(0.6)
                                details.append(f'Starker Aufwärtstrend (ADX: {adx:.1f})')
                            else:
                                signals.append(-0.6)
                                details.append(f'Starker Abwärtstrend (ADX: {adx:.1f})')

            # Signal 5: Parabolic SAR
            if 'PSAR' in df.columns:
                psar = df['PSAR'].iloc[idx]
                if not pd.isna(psar):
                    if close > psar:
                        signals.append(0.6)
                        details.append('PSAR Bullish (Preis über SAR)')
                    else:
                        signals.append(-0.6)
                        details.append('PSAR Bearish (Preis unter SAR)')

            # Signal 6: Supertrend
            if 'Supertrend' in df.columns and 'Supertrend_Direction' in df.columns:
                st_dir = df['Supertrend_Direction'].iloc[idx]
                if not pd.isna(st_dir):
                    if st_dir > 0:
                        signals.append(0.7)
                        details.append('Supertrend Bullish')
                    else:
                        signals.append(-0.7)
                        details.append('Supertrend Bearish')

            # Signal 7: Ichimoku Cloud
            if 'Ichimoku_Signal' in df.columns:
                ichi_signal = df['Ichimoku_Signal'].iloc[idx]
                if not pd.isna(ichi_signal):
                    if ichi_signal > 0:
                        signals.append(0.8)
                        details.append('Ichimoku Bullish (über Cloud)')
                    elif ichi_signal < 0:
                        signals.append(-0.8)
                        details.append('Ichimoku Bearish (unter Cloud)')

            # Signal 8: Linear Regression Slope
            if len(df) >= idx + 1:
                prices_20 = df['Close'].iloc[idx-19:idx+1].values
                if len(prices_20) == 20:
                    x = np.arange(len(prices_20))
                    slope = np.polyfit(x, prices_20, 1)[0]
                    slope_pct = (slope / prices_20[-1]) * 100

                    if slope_pct > 0.5:
                        signals.append(0.6)
                        details.append(f'Steigender Trend ({slope_pct:.2f}%/Tag)')
                    elif slope_pct < -0.5:
                        signals.append(-0.6)
                        details.append(f'Fallender Trend ({slope_pct:.2f}%/Tag)')

            # Signal 9: Aroon Indicator
            if 'Aroon_Up' in df.columns and 'Aroon_Down' in df.columns:
                aroon_up = df['Aroon_Up'].iloc[idx]
                aroon_down = df['Aroon_Down'].iloc[idx]
                if not pd.isna(aroon_up) and not pd.isna(aroon_down):
                    if aroon_up > 70 and aroon_down < 30:
                        signals.append(0.7)
                        details.append('Aroon Starker Aufwärtstrend')
                    elif aroon_down > 70 and aroon_up < 30:
                        signals.append(-0.7)
                        details.append('Aroon Starker Abwärtstrend')

            # Signal 10: VWAP Trend
            if 'VWAP' in df.columns:
                vwap = df['VWAP'].iloc[idx]
                if not pd.isna(vwap):
                    if close > vwap * 1.02:
                        signals.append(0.5)
                        details.append('Preis deutlich über VWAP (Bullish)')
                    elif close < vwap * 0.98:
                        signals.append(-0.5)
                        details.append('Preis deutlich unter VWAP (Bearish)')

        except Exception as e:
            logger.warning(f"Fehler bei Trend-Signalen: {e}")

        return {
            'signals': signals,
            'average': np.mean(signals) if signals else 0.0,
            'count': len(signals),
            'details': details
        }

    def _generate_momentum_signals(self, df: pd.DataFrame, idx: int) -> Dict:
        """Generiert 10+ Momentum-basierte Signale"""
        signals = []
        details = []

        try:
            close = df['Close'].iloc[idx]

            # Signal 1: RSI
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[idx]
                if not pd.isna(rsi):
                    if rsi < 30:
                        signals.append(0.8)  # Überverkauft
                        details.append(f'RSI Überverkauft ({rsi:.1f})')
                    elif rsi < 40:
                        signals.append(0.5)
                        details.append(f'RSI Bullish ({rsi:.1f})')
                    elif rsi > 70:
                        signals.append(-0.8)  # Überkauft
                        details.append(f'RSI Überkauft ({rsi:.1f})')
                    elif rsi > 60:
                        signals.append(-0.5)
                        details.append(f'RSI Bearish ({rsi:.1f})')

            # Signal 2: Stochastic
            if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
                stoch_k = df['Stoch_K'].iloc[idx]
                stoch_d = df['Stoch_D'].iloc[idx]
                if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                    if stoch_k < 20 and stoch_d < 20:
                        signals.append(0.7)
                        details.append(f'Stochastic Überverkauft (K:{stoch_k:.1f})')
                    elif stoch_k > 80 and stoch_d > 80:
                        signals.append(-0.7)
                        details.append(f'Stochastic Überkauft (K:{stoch_k:.1f})')

                    # Crossover
                    if idx > 0:
                        stoch_k_prev = df['Stoch_K'].iloc[idx-1]
                        stoch_d_prev = df['Stoch_D'].iloc[idx-1]
                        if stoch_k_prev < stoch_d_prev and stoch_k > stoch_d:
                            signals.append(0.6)
                            details.append('Stochastic Bullish Crossover')
                        elif stoch_k_prev > stoch_d_prev and stoch_k < stoch_d:
                            signals.append(-0.6)
                            details.append('Stochastic Bearish Crossover')

            # Signal 3: MACD
            if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                macd = df['MACD'].iloc[idx]
                macd_signal = df['MACD_Signal'].iloc[idx]
                macd_hist = df.get('MACD_Hist', pd.Series()).iloc[idx] if 'MACD_Hist' in df.columns else None

                if not pd.isna(macd) and not pd.isna(macd_signal):
                    # MACD Crossover
                    if idx > 0:
                        macd_prev = df['MACD'].iloc[idx-1]
                        signal_prev = df['MACD_Signal'].iloc[idx-1]

                        if macd_prev < signal_prev and macd > macd_signal:
                            signals.append(0.9)
                            details.append('MACD Bullish Crossover')
                        elif macd_prev > signal_prev and macd < macd_signal:
                            signals.append(-0.9)
                            details.append('MACD Bearish Crossover')

                    # MACD Histogram Divergenz
                    if macd_hist is not None and not pd.isna(macd_hist):
                        if macd_hist > 0 and macd > macd_signal:
                            signals.append(0.5)
                            details.append('MACD Bullish Momentum')
                        elif macd_hist < 0 and macd < macd_signal:
                            signals.append(-0.5)
                            details.append('MACD Bearish Momentum')

            # Signal 4: Williams %R
            if 'Williams_R' in df.columns:
                williams = df['Williams_R'].iloc[idx]
                if not pd.isna(williams):
                    if williams < -80:
                        signals.append(0.6)
                        details.append(f'Williams %R Überverkauft ({williams:.1f})')
                    elif williams > -20:
                        signals.append(-0.6)
                        details.append(f'Williams %R Überkauft ({williams:.1f})')

            # Signal 5: Rate of Change (ROC)
            roc_20 = ((close - df['Close'].iloc[idx-20]) / df['Close'].iloc[idx-20]) * 100 if idx >= 20 else 0
            if roc_20 > 10:
                signals.append(0.6)
                details.append(f'ROC(20) Stark Bullish ({roc_20:.1f}%)')
            elif roc_20 > 5:
                signals.append(0.4)
                details.append(f'ROC(20) Bullish ({roc_20:.1f}%)')
            elif roc_20 < -10:
                signals.append(-0.6)
                details.append(f'ROC(20) Stark Bearish ({roc_20:.1f}%)')
            elif roc_20 < -5:
                signals.append(-0.4)
                details.append(f'ROC(20) Bearish ({roc_20:.1f}%)')

            # Signal 6: Momentum (10-day)
            if idx >= 10:
                momentum_10 = close - df['Close'].iloc[idx-10]
                momentum_pct = (momentum_10 / df['Close'].iloc[idx-10]) * 100
                if momentum_pct > 3:
                    signals.append(0.5)
                    details.append(f'10-Day Momentum Bullish ({momentum_pct:.1f}%)')
                elif momentum_pct < -3:
                    signals.append(-0.5)
                    details.append(f'10-Day Momentum Bearish ({momentum_pct:.1f}%)')

            # Signal 7: CCI (Commodity Channel Index)
            if 'CCI' in df.columns:
                cci = df['CCI'].iloc[idx]
                if not pd.isna(cci):
                    if cci < -100:
                        signals.append(0.7)
                        details.append(f'CCI Überverkauft ({cci:.1f})')
                    elif cci > 100:
                        signals.append(-0.7)
                        details.append(f'CCI Überkauft ({cci:.1f})')

            # Signal 8: Ultimate Oscillator
            if 'UO' in df.columns:
                uo = df['UO'].iloc[idx]
                if not pd.isna(uo):
                    if uo < 30:
                        signals.append(0.6)
                        details.append(f'Ultimate Oscillator Überverkauft ({uo:.1f})')
                    elif uo > 70:
                        signals.append(-0.6)
                        details.append(f'Ultimate Oscillator Überkauft ({uo:.1f})')

            # Signal 9: Money Flow Index (MFI)
            if 'MFI' in df.columns:
                mfi = df['MFI'].iloc[idx]
                if not pd.isna(mfi):
                    if mfi < 20:
                        signals.append(0.7)
                        details.append(f'MFI Überverkauft ({mfi:.1f})')
                    elif mfi > 80:
                        signals.append(-0.7)
                        details.append(f'MFI Überkauft ({mfi:.1f})')

            # Signal 10: TSI (True Strength Index)
            if 'TSI' in df.columns:
                tsi = df['TSI'].iloc[idx]
                if not pd.isna(tsi):
                    if tsi > 25:
                        signals.append(0.5)
                        details.append(f'TSI Bullish ({tsi:.1f})')
                    elif tsi < -25:
                        signals.append(-0.5)
                        details.append(f'TSI Bearish ({tsi:.1f})')

        except Exception as e:
            logger.warning(f"Fehler bei Momentum-Signalen: {e}")

        return {
            'signals': signals,
            'average': np.mean(signals) if signals else 0.0,
            'count': len(signals),
            'details': details
        }

    def _generate_volatility_signals(self, df: pd.DataFrame, idx: int) -> Dict:
        """Generiert 10+ Volatility-basierte Signale"""
        signals = []
        details = []

        try:
            close = df['Close'].iloc[idx]

            # Signal 1: Bollinger Bands
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns and 'BB_Middle' in df.columns:
                bb_upper = df['BB_Upper'].iloc[idx]
                bb_lower = df['BB_Lower'].iloc[idx]
                bb_middle = df['BB_Middle'].iloc[idx]

                if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                    bb_width = (bb_upper - bb_lower) / bb_middle

                    # Bollinger Bounce
                    if close <= bb_lower:
                        signals.append(0.8)
                        details.append('Bollinger Bands: Preis am unteren Band (Oversold)')
                    elif close >= bb_upper:
                        signals.append(-0.8)
                        details.append('Bollinger Bands: Preis am oberen Band (Overbought)')

                    # Bollinger Squeeze
                    if idx >= 20:
                        bb_width_20 = df['BB_Upper'].iloc[idx-20:idx].subtract(df['BB_Lower'].iloc[idx-20:idx]).div(df['BB_Middle'].iloc[idx-20:idx]).mean()
                        if bb_width < bb_width_20 * 0.5:
                            signals.append(0.6)  # Squeeze = kommende Bewegung
                            details.append('Bollinger Squeeze (Volatility Compression)')

            # Signal 2: ATR Volatility
            if 'ATR' in df.columns and idx >= 20:
                atr = df['ATR'].iloc[idx]
                atr_20 = df['ATR'].iloc[idx-20:idx].mean()

                if not pd.isna(atr) and not pd.isna(atr_20):
                    if atr > atr_20 * 1.5:
                        signals.append(-0.4)  # Hohe Volatility = Vorsicht
                        details.append(f'Hohe Volatility (ATR {atr:.2f} > Avg)')
                    elif atr < atr_20 * 0.5:
                        signals.append(0.4)  # Niedrige Volatility = möglicher Breakout
                        details.append(f'Niedrige Volatility (ATR {atr:.2f} < Avg)')

            # Signal 3: Keltner Channels
            if 'Keltner_Upper' in df.columns and 'Keltner_Lower' in df.columns:
                kelt_upper = df['Keltner_Upper'].iloc[idx]
                kelt_lower = df['Keltner_Lower'].iloc[idx]

                if not pd.isna(kelt_upper) and not pd.isna(kelt_lower):
                    if close >= kelt_upper:
                        signals.append(-0.6)
                        details.append('Keltner: Preis über oberem Channel')
                    elif close <= kelt_lower:
                        signals.append(0.6)
                        details.append('Keltner: Preis unter unterem Channel')

            # Signal 4: Donchian Channels
            if idx >= 20:
                high_20 = df['High'].iloc[idx-20:idx].max()
                low_20 = df['Low'].iloc[idx-20:idx].min()

                if close >= high_20:
                    signals.append(0.7)
                    details.append('Donchian: Neues 20-Day High (Breakout)')
                elif close <= low_20:
                    signals.append(-0.7)
                    details.append('Donchian: Neues 20-Day Low (Breakdown)')

            # Signal 5: Historical Volatility
            if idx >= 20:
                returns = df['Close'].iloc[idx-20:idx].pct_change().dropna()
                hist_vol = returns.std() * np.sqrt(252)  # Annualisiert

                if hist_vol > 0.5:  # 50% annualisiert
                    signals.append(-0.3)
                    details.append(f'Sehr hohe Volatility ({hist_vol*100:.1f}%)')
                elif hist_vol < 0.15:  # 15% annualisiert
                    signals.append(0.3)
                    details.append(f'Niedrige Volatility ({hist_vol*100:.1f}%)')

            # Signal 6: Bollinger %B
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                bb_upper = df['BB_Upper'].iloc[idx]
                bb_lower = df['BB_Lower'].iloc[idx]

                if not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_upper != bb_lower:
                    bb_pct = (close - bb_lower) / (bb_upper - bb_lower)

                    if bb_pct < 0.2:
                        signals.append(0.6)
                        details.append(f'Bollinger %B niedrig ({bb_pct:.2f})')
                    elif bb_pct > 0.8:
                        signals.append(-0.6)
                        details.append(f'Bollinger %B hoch ({bb_pct:.2f})')

            # Signal 7: Std Dev Breakout
            if idx >= 20:
                std_20 = df['Close'].iloc[idx-20:idx].std()
                mean_20 = df['Close'].iloc[idx-20:idx].mean()

                z_score = (close - mean_20) / std_20 if std_20 > 0 else 0

                if z_score > 2:
                    signals.append(-0.5)
                    details.append(f'Preis >2 Std Dev über Mean (Overbought)')
                elif z_score < -2:
                    signals.append(0.5)
                    details.append(f'Preis >2 Std Dev unter Mean (Oversold)')

            # Signal 8: Volatility Ratio
            if idx >= 50:
                vol_short = df['Close'].iloc[idx-10:idx].std()
                vol_long = df['Close'].iloc[idx-50:idx].std()

                if vol_long > 0:
                    vol_ratio = vol_short / vol_long

                    if vol_ratio > 1.5:
                        signals.append(-0.4)
                        details.append(f'Steigende Volatility (Ratio: {vol_ratio:.2f})')
                    elif vol_ratio < 0.5:
                        signals.append(0.4)
                        details.append(f'Fallende Volatility (Ratio: {vol_ratio:.2f})')

            # Signal 9: Price Channel Breakout
            if idx >= 50:
                high_50 = df['High'].iloc[idx-50:idx].max()
                low_50 = df['Low'].iloc[idx-50:idx].min()
                channel_width = (high_50 - low_50) / low_50

                if close >= high_50 * 0.99:
                    signals.append(0.7)
                    details.append('50-Day Channel Breakout (Bullish)')
                elif close <= low_50 * 1.01:
                    signals.append(-0.7)
                    details.append('50-Day Channel Breakdown (Bearish)')

            # Signal 10: Envelope Indicator
            if 'SMA_50' in df.columns:
                sma_50 = df['SMA_50'].iloc[idx]
                if not pd.isna(sma_50):
                    upper_env = sma_50 * 1.05
                    lower_env = sma_50 * 0.95

                    if close >= upper_env:
                        signals.append(-0.5)
                        details.append('Preis über 5% Envelope (Overbought)')
                    elif close <= lower_env:
                        signals.append(0.5)
                        details.append('Preis unter 5% Envelope (Oversold)')

        except Exception as e:
            logger.warning(f"Fehler bei Volatility-Signalen: {e}")

        return {
            'signals': signals,
            'average': np.mean(signals) if signals else 0.0,
            'count': len(signals),
            'details': details
        }

    def _generate_volume_signals(self, df: pd.DataFrame, idx: int) -> Dict:
        """Generiert 8+ Volume-basierte Signale"""
        signals = []
        details = []

        try:
            volume = df['Volume'].iloc[idx]
            close = df['Close'].iloc[idx]

            if idx >= 20:
                vol_avg_20 = df['Volume'].iloc[idx-20:idx].mean()

                # Signal 1: Volume Spike
                if volume > vol_avg_20 * 2:
                    if idx > 0:
                        price_change = (close - df['Close'].iloc[idx-1]) / df['Close'].iloc[idx-1]
                        if price_change > 0:
                            signals.append(0.7)
                            details.append(f'Volume Spike mit Preisanstieg ({volume/vol_avg_20:.1f}x avg)')
                        else:
                            signals.append(-0.7)
                            details.append(f'Volume Spike mit Preisrückgang ({volume/vol_avg_20:.1f}x avg)')

                # Signal 2: Low Volume
                elif volume < vol_avg_20 * 0.5:
                    signals.append(-0.3)
                    details.append('Niedriges Volume (Schwache Bewegung)')

                # Signal 3: OBV (On-Balance Volume)
                if 'OBV' in df.columns and idx > 0:
                    obv = df['OBV'].iloc[idx]
                    obv_prev = df['OBV'].iloc[idx-1]

                    if not pd.isna(obv) and not pd.isna(obv_prev):
                        if obv > obv_prev and close > df['Close'].iloc[idx-1]:
                            signals.append(0.6)
                            details.append('OBV Bullish (Volume bestätigt Aufwärtstrend)')
                        elif obv < obv_prev and close < df['Close'].iloc[idx-1]:
                            signals.append(-0.6)
                            details.append('OBV Bearish (Volume bestätigt Abwärtstrend)')

                # Signal 4: Volume Trend
                if idx >= 50:
                    vol_trend_recent = df['Volume'].iloc[idx-10:idx].mean()
                    vol_trend_old = df['Volume'].iloc[idx-50:idx-10].mean()

                    if vol_trend_recent > vol_trend_old * 1.3:
                        signals.append(0.5)
                        details.append('Steigendes Volume-Trend (Interesseerhöhung)')
                    elif vol_trend_recent < vol_trend_old * 0.7:
                        signals.append(-0.3)
                        details.append('Fallendes Volume-Trend (Interesserückgang)')

                # Signal 5: VWAP Signal
                if 'VWAP' in df.columns:
                    vwap = df['VWAP'].iloc[idx]
                    if not pd.isna(vwap):
                        if close > vwap and volume > vol_avg_20:
                            signals.append(0.6)
                            details.append('Preis über VWAP mit hohem Volume (Bullish)')
                        elif close < vwap and volume > vol_avg_20:
                            signals.append(-0.6)
                            details.append('Preis unter VWAP mit hohem Volume (Bearish)')

                # Signal 6: Volume Price Trend (VPT)
                if 'VPT' in df.columns:
                    vpt = df['VPT'].iloc[idx]
                    if not pd.isna(vpt) and idx >= 20:
                        vpt_ma = df['VPT'].iloc[idx-20:idx].mean()
                        if vpt > vpt_ma:
                            signals.append(0.5)
                            details.append('VPT Bullish')
                        else:
                            signals.append(-0.5)
                            details.append('VPT Bearish')

                # Signal 7: Accumulation/Distribution
                if 'AD_Line' in df.columns and idx > 0:
                    ad = df['AD_Line'].iloc[idx]
                    ad_prev = df['AD_Line'].iloc[idx-1]

                    if not pd.isna(ad) and not pd.isna(ad_prev):
                        if ad > ad_prev and close > df['Close'].iloc[idx-1]:
                            signals.append(0.6)
                            details.append('A/D Line Bullish (Akkumulation)')
                        elif ad < ad_prev and close < df['Close'].iloc[idx-1]:
                            signals.append(-0.6)
                            details.append('A/D Line Bearish (Distribution)')

                # Signal 8: Chaikin Money Flow
                if 'CMF' in df.columns:
                    cmf = df['CMF'].iloc[idx]
                    if not pd.isna(cmf):
                        if cmf > 0.1:
                            signals.append(0.6)
                            details.append(f'CMF Bullish (Kaufdruck: {cmf:.2f})')
                        elif cmf < -0.1:
                            signals.append(-0.6)
                            details.append(f'CMF Bearish (Verkaufsdruck: {cmf:.2f})')

        except Exception as e:
            logger.warning(f"Fehler bei Volume-Signalen: {e}")

        return {
            'signals': signals,
            'average': np.mean(signals) if signals else 0.0,
            'count': len(signals),
            'details': details
        }

    def _generate_pattern_signals(self, df: pd.DataFrame, idx: int) -> Dict:
        """Generiert 10+ Pattern-basierte Signale"""
        signals = []
        details = []

        try:
            if idx < 3:
                return {'signals': [], 'average': 0.0, 'count': 0, 'details': []}

            # Signal 1-9: Candlestick Patterns (falls vorhanden)
            pattern_columns = [
                ('Doji', 0.5, 'Doji Pattern (Umkehr)'),
                ('Hammer', 0.7, 'Hammer Pattern (Bullish Umkehr)'),
                ('Shooting_Star', -0.7, 'Shooting Star Pattern (Bearish Umkehr)'),
                ('Engulfing_Bull', 0.8, 'Bullish Engulfing Pattern'),
                ('Engulfing_Bear', -0.8, 'Bearish Engulfing Pattern'),
                ('Morning_Star', 0.9, 'Morning Star Pattern (Starke Bullish Umkehr)'),
                ('Evening_Star', -0.9, 'Evening Star Pattern (Starke Bearish Umkehr)'),
                ('Three_White_Soldiers', 0.8, 'Three White Soldiers (Starker Aufwärtstrend)'),
                ('Three_Black_Crows', -0.8, 'Three Black Crows (Starker Abwärtstrend)')
            ]

            for col, signal_val, detail in pattern_columns:
                if col in df.columns:
                    if df[col].iloc[idx] == True or df[col].iloc[idx] == 1:
                        signals.append(signal_val)
                        details.append(detail)

            # Signal 10: Support/Resistance Break
            if 'Support' in df.columns and 'Resistance' in df.columns:
                support = df['Support'].iloc[idx]
                resistance = df['Resistance'].iloc[idx]
                close = df['Close'].iloc[idx]

                if not pd.isna(resistance) and close > resistance:
                    signals.append(0.8)
                    details.append(f'Resistance Breakout (${resistance:.2f})')
                elif not pd.isna(support) and close < support:
                    signals.append(-0.8)
                    details.append(f'Support Breakdown (${support:.2f})')

        except Exception as e:
            logger.warning(f"Fehler bei Pattern-Signalen: {e}")

        return {
            'signals': signals,
            'average': np.mean(signals) if signals else 0.0,
            'count': len(signals),
            'details': details
        }

    def _generate_oscillator_signals(self, df: pd.DataFrame, idx: int) -> Dict:
        """Generiert 8+ Oscillator-basierte Signale"""
        # Dies ist eine Kombination und Zusammenfassung der anderen Signale
        # Wird bereits in Momentum abgedeckt, aber hier nochmal dediziert

        signals = []
        details = []

        try:
            # Bereits in Momentum abgedeckt, füge nur zusätzliche hinzu

            # Awesome Oscillator
            if 'AO' in df.columns and idx > 0:
                ao = df['AO'].iloc[idx]
                ao_prev = df['AO'].iloc[idx-1]

                if not pd.isna(ao) and not pd.isna(ao_prev):
                    if ao_prev < 0 and ao > 0:
                        signals.append(0.7)
                        details.append('Awesome Oscillator Bullish Crossover')
                    elif ao_prev > 0 and ao < 0:
                        signals.append(-0.7)
                        details.append('Awesome Oscillator Bearish Crossover')

            # Know Sure Thing (KST)
            if 'KST' in df.columns and 'KST_Signal' in df.columns and idx > 0:
                kst = df['KST'].iloc[idx]
                kst_signal = df['KST_Signal'].iloc[idx]
                kst_prev = df['KST'].iloc[idx-1]
                signal_prev = df['KST_Signal'].iloc[idx-1]

                if not pd.isna(kst) and not pd.isna(kst_signal):
                    if kst_prev < signal_prev and kst > kst_signal:
                        signals.append(0.7)
                        details.append('KST Bullish Crossover')
                    elif kst_prev > signal_prev and kst < kst_signal:
                        signals.append(-0.7)
                        details.append('KST Bearish Crossover')

        except Exception as e:
            logger.warning(f"Fehler bei Oscillator-Signalen: {e}")

        return {
            'signals': signals,
            'average': np.mean(signals) if signals else 0.0,
            'count': len(signals),
            'details': details
        }

    def _aggregate_signals(self, signals_by_category: Dict) -> Dict:
        """Aggregiert alle Signale zu einem finalen Signal"""
        all_signals = []
        all_weights = []
        all_details = []
        category_summary = {}

        bullish_count = 0
        bearish_count = 0

        for category, data in signals_by_category.items():
            if data['count'] > 0:
                weight = self.signal_weights.get(category, 1.0)
                avg_signal = data['average']

                all_signals.append(avg_signal)
                all_weights.append(weight)
                all_details.extend(data['details'])

                category_summary[category] = {
                    'average': avg_signal,
                    'count': data['count'],
                    'weight': weight
                }

                if avg_signal > 0.1:
                    bullish_count += data['count']
                elif avg_signal < -0.1:
                    bearish_count += data['count']

        if all_signals:
            # Gewichteter Durchschnitt
            weighted_avg = np.average(all_signals, weights=all_weights)

            # Confidence basierend auf Anzahl der Signale und Konsistenz
            std_dev = np.std(all_signals)
            num_signals = sum(data['count'] for data in signals_by_category.values())

            # Confidence: hoch wenn viele Signale und konsistent
            consistency_factor = 1.0 - min(std_dev, 1.0)  # Je weniger Streuung, desto besser
            volume_factor = min(num_signals / 30.0, 1.0)  # Max bei 30+ Signalen

            confidence = (abs(weighted_avg) * 0.5 + consistency_factor * 0.3 + volume_factor * 0.2)
            confidence = min(max(confidence, 0.0), 1.0)  # Clamp 0-1

            # Final Signal
            if weighted_avg > 0.3:
                final_signal = 'BUY'
                strength = 'STRONG' if weighted_avg > 0.6 else 'MODERATE'
            elif weighted_avg < -0.3:
                final_signal = 'SELL'
                strength = 'STRONG' if weighted_avg < -0.6 else 'MODERATE'
            else:
                final_signal = 'HOLD'
                strength = 'NEUTRAL'

            return {
                'signal': final_signal,
                'strength': strength,
                'confidence': confidence,
                'raw_score': weighted_avg,
                'num_signals': num_signals,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'category_summary': category_summary,
                'details': all_details
            }
        else:
            return {
                'signal': 'HOLD',
                'strength': 'NEUTRAL',
                'confidence': 0.0,
                'raw_score': 0.0,
                'num_signals': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'category_summary': {},
                'details': []
            }

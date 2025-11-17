"""
Pattern Recognition Module
Erkennt Chart-Muster (Candlestick Patterns und Chart Formations)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CandlestickPatterns:
    """Candlestick Pattern Erkennung"""

    @staticmethod
    def doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """
        Doji: Open ≈ Close (indecision pattern)

        Args:
            threshold: Max difference between Open and Close as % of range
        """
        body = abs(df['Close'] - df['Open'])
        range_hl = df['High'] - df['Low']
        return (body / range_hl.replace(0, 1)) < threshold

    @staticmethod
    def hammer(df: pd.DataFrame) -> pd.Series:
        """
        Hammer: Bullish reversal
        - Small body at top
        - Long lower shadow (2x body)
        - Little/no upper shadow
        """
        body = abs(df['Close'] - df['Open'])
        lower_shadow = pd.concat([df['Open'], df['Close']], axis=1).min(axis=1) - df['Low']
        upper_shadow = df['High'] - pd.concat([df['Open'], df['Close']], axis=1).max(axis=1)

        return (lower_shadow > 2 * body) & (upper_shadow < body * 0.5)

    @staticmethod
    def shooting_star(df: pd.DataFrame) -> pd.Series:
        """
        Shooting Star: Bearish reversal
        - Small body at bottom
        - Long upper shadow (2x body)
        - Little/no lower shadow
        """
        body = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - pd.concat([df['Open'], df['Close']], axis=1).max(axis=1)
        lower_shadow = pd.concat([df['Open'], df['Close']], axis=1).min(axis=1) - df['Low']

        return (upper_shadow > 2 * body) & (lower_shadow < body * 0.5)

    @staticmethod
    def engulfing_bullish(df: pd.DataFrame) -> pd.Series:
        """
        Bullish Engulfing:
        - Previous: bearish candle
        - Current: bullish candle that engulfs previous
        """
        prev_bearish = df['Close'].shift(1) < df['Open'].shift(1)
        curr_bullish = df['Close'] > df['Open']
        engulfing = (df['Open'] < df['Close'].shift(1)) & (df['Close'] > df['Open'].shift(1))

        return prev_bearish & curr_bullish & engulfing

    @staticmethod
    def engulfing_bearish(df: pd.DataFrame) -> pd.Series:
        """
        Bearish Engulfing:
        - Previous: bullish candle
        - Current: bearish candle that engulfs previous
        """
        prev_bullish = df['Close'].shift(1) > df['Open'].shift(1)
        curr_bearish = df['Close'] < df['Open']
        engulfing = (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))

        return prev_bullish & curr_bearish & engulfing

    @staticmethod
    def morning_star(df: pd.DataFrame) -> pd.Series:
        """
        Morning Star (3-candle bullish reversal):
        1. Long bearish candle
        2. Small body (star)
        3. Long bullish candle
        """
        # First candle: bearish
        candle1_bearish = df['Close'].shift(2) < df['Open'].shift(2)
        candle1_body = abs(df['Close'].shift(2) - df['Open'].shift(2))

        # Second candle: small body (doji-like)
        candle2_body = abs(df['Close'].shift(1) - df['Open'].shift(1))
        candle2_small = candle2_body < (candle1_body * 0.3)

        # Third candle: bullish
        candle3_bullish = df['Close'] > df['Open']
        candle3_body = abs(df['Close'] - df['Open'])

        # Star gaps down, then gaps up
        gap_down = df['High'].shift(1) < df['Close'].shift(2)
        recovery = df['Close'] > (df['Open'].shift(2) + df['Close'].shift(2)) / 2

        return candle1_bearish & candle2_small & candle3_bullish & recovery

    @staticmethod
    def evening_star(df: pd.DataFrame) -> pd.Series:
        """
        Evening Star (3-candle bearish reversal):
        1. Long bullish candle
        2. Small body (star)
        3. Long bearish candle
        """
        # First candle: bullish
        candle1_bullish = df['Close'].shift(2) > df['Open'].shift(2)
        candle1_body = abs(df['Close'].shift(2) - df['Open'].shift(2))

        # Second candle: small body
        candle2_body = abs(df['Close'].shift(1) - df['Open'].shift(1))
        candle2_small = candle2_body < (candle1_body * 0.3)

        # Third candle: bearish
        candle3_bearish = df['Close'] < df['Open']

        # Star gaps up, then gaps down
        gap_up = df['Low'].shift(1) > df['Close'].shift(2)
        decline = df['Close'] < (df['Open'].shift(2) + df['Close'].shift(2)) / 2

        return candle1_bullish & candle2_small & candle3_bearish & decline

    @staticmethod
    def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
        """
        Three White Soldiers (strong bullish reversal):
        - Three consecutive long bullish candles
        - Each opens within previous body
        - Each closes higher
        """
        # All three bullish
        bull1 = df['Close'].shift(2) > df['Open'].shift(2)
        bull2 = df['Close'].shift(1) > df['Open'].shift(1)
        bull3 = df['Close'] > df['Open']

        # Long bodies
        body1 = df['Close'].shift(2) - df['Open'].shift(2)
        body2 = df['Close'].shift(1) - df['Open'].shift(1)
        body3 = df['Close'] - df['Open']

        # Each opens within previous body
        open_within1 = (df['Open'].shift(1) > df['Open'].shift(2)) & (df['Open'].shift(1) < df['Close'].shift(2))
        open_within2 = (df['Open'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))

        # Consecutive higher closes
        higher_closes = (df['Close'].shift(1) > df['Close'].shift(2)) & (df['Close'] > df['Close'].shift(1))

        return bull1 & bull2 & bull3 & open_within1 & open_within2 & higher_closes

    @staticmethod
    def three_black_crows(df: pd.DataFrame) -> pd.Series:
        """
        Three Black Crows (strong bearish reversal):
        - Three consecutive long bearish candles
        - Each opens within previous body
        - Each closes lower
        """
        # All three bearish
        bear1 = df['Close'].shift(2) < df['Open'].shift(2)
        bear2 = df['Close'].shift(1) < df['Open'].shift(1)
        bear3 = df['Close'] < df['Open']

        # Each opens within previous body
        open_within1 = (df['Open'].shift(1) < df['Open'].shift(2)) & (df['Open'].shift(1) > df['Close'].shift(2))
        open_within2 = (df['Open'] < df['Open'].shift(1)) & (df['Open'] > df['Close'].shift(1))

        # Consecutive lower closes
        lower_closes = (df['Close'].shift(1) < df['Close'].shift(2)) & (df['Close'] < df['Close'].shift(1))

        return bear1 & bear2 & bear3 & open_within1 & open_within2 & lower_closes


class ChartPatterns:
    """Chart Formation Pattern Erkennung"""

    @staticmethod
    def find_support_resistance(df: pd.DataFrame, window: int = 20, num_levels: int = 5) -> Dict[str, List[float]]:
        """
        Findet Support und Resistance Levels

        Returns:
            Dict mit 'support' und 'resistance' Listen
        """
        highs = df['High'].values
        lows = df['Low'].values

        # Lokale Maxima und Minima
        resistance_levels = []
        support_levels = []

        for i in range(window, len(df) - window):
            # Resistance: Lokales Maximum
            if highs[i] == max(highs[i-window:i+window+1]):
                resistance_levels.append(highs[i])

            # Support: Lokales Minimum
            if lows[i] == min(lows[i-window:i+window+1]):
                support_levels.append(lows[i])

        # Cluster ähnliche Levels (innerhalb 1%)
        def cluster_levels(levels, tolerance=0.01):
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

        support_levels = cluster_levels(support_levels)[-num_levels:]
        resistance_levels = cluster_levels(resistance_levels)[-num_levels:]

        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels, reverse=True)
        }

    @staticmethod
    def head_and_shoulders(df: pd.DataFrame, window: int = 5) -> Tuple[bool, Optional[float]]:
        """
        Head and Shoulders Pattern (bearish reversal)

        Returns:
            (pattern_found, neckline_price)
        """
        if len(df) < window * 5:
            return False, None

        highs = df['High'].values
        lows = df['Low'].values

        # Suche nach 3 Peaks (left shoulder, head, right shoulder)
        peaks = []
        for i in range(window, len(df) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                peaks.append((i, highs[i]))

        if len(peaks) < 3:
            return False, None

        # Überprüfe letzten 3 Peaks
        recent_peaks = peaks[-3:]
        left_shoulder = recent_peaks[0]
        head = recent_peaks[1]
        right_shoulder = recent_peaks[2]

        # Head sollte höher sein als beide Shoulders
        if head[1] > left_shoulder[1] and head[1] > right_shoulder[1]:
            # Shoulders sollten ähnliche Höhe haben (±5%)
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
            if shoulder_diff < 0.05:
                # Finde Neckline (Tiefs zwischen Peaks)
                neckline_idx1 = np.argmin(lows[left_shoulder[0]:head[0]]) + left_shoulder[0]
                neckline_idx2 = np.argmin(lows[head[0]:right_shoulder[0]]) + head[0]
                neckline = (lows[neckline_idx1] + lows[neckline_idx2]) / 2

                return True, neckline

        return False, None

    @staticmethod
    def inverse_head_and_shoulders(df: pd.DataFrame, window: int = 5) -> Tuple[bool, Optional[float]]:
        """
        Inverse Head and Shoulders Pattern (bullish reversal)

        Returns:
            (pattern_found, neckline_price)
        """
        if len(df) < window * 5:
            return False, None

        lows = df['Low'].values
        highs = df['High'].values

        # Suche nach 3 Troughs (valleys)
        troughs = []
        for i in range(window, len(df) - window):
            if lows[i] == min(lows[i-window:i+window+1]):
                troughs.append((i, lows[i]))

        if len(troughs) < 3:
            return False, None

        # Überprüfe letzten 3 Troughs
        recent_troughs = troughs[-3:]
        left_shoulder = recent_troughs[0]
        head = recent_troughs[1]
        right_shoulder = recent_troughs[2]

        # Head sollte tiefer sein als beide Shoulders
        if head[1] < left_shoulder[1] and head[1] < right_shoulder[1]:
            # Shoulders sollten ähnliche Tiefe haben
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
            if shoulder_diff < 0.05:
                # Finde Neckline (Hochs zwischen Troughs)
                neckline_idx1 = np.argmax(highs[left_shoulder[0]:head[0]]) + left_shoulder[0]
                neckline_idx2 = np.argmax(highs[head[0]:right_shoulder[0]]) + head[0]
                neckline = (highs[neckline_idx1] + highs[neckline_idx2]) / 2

                return True, neckline

        return False, None

    @staticmethod
    def double_top(df: pd.DataFrame, window: int = 10, tolerance: float = 0.02) -> Tuple[bool, Optional[float]]:
        """
        Double Top Pattern (bearish reversal)

        Returns:
            (pattern_found, support_level)
        """
        if len(df) < window * 3:
            return False, None

        highs = df['High'].values
        lows = df['Low'].values

        # Finde Peaks
        peaks = []
        for i in range(window, len(df) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                peaks.append((i, highs[i]))

        if len(peaks) < 2:
            return False, None

        # Überprüfe letzte 2 Peaks
        last_two = peaks[-2:]
        peak1, peak2 = last_two[0], last_two[1]

        # Peaks sollten ähnliche Höhe haben
        if abs(peak1[1] - peak2[1]) / peak1[1] < tolerance:
            # Finde Trough zwischen Peaks
            trough_idx = np.argmin(lows[peak1[0]:peak2[0]]) + peak1[0]
            support = lows[trough_idx]

            # Mindestabstand zwischen Peaks
            if peak2[0] - peak1[0] > window:
                return True, support

        return False, None

    @staticmethod
    def double_bottom(df: pd.DataFrame, window: int = 10, tolerance: float = 0.02) -> Tuple[bool, Optional[float]]:
        """
        Double Bottom Pattern (bullish reversal)

        Returns:
            (pattern_found, resistance_level)
        """
        if len(df) < window * 3:
            return False, None

        lows = df['Low'].values
        highs = df['High'].values

        # Finde Troughs
        troughs = []
        for i in range(window, len(df) - window):
            if lows[i] == min(lows[i-window:i+window+1]):
                troughs.append((i, lows[i]))

        if len(troughs) < 2:
            return False, None

        # Überprüfe letzte 2 Troughs
        last_two = troughs[-2:]
        trough1, trough2 = last_two[0], last_two[1]

        # Troughs sollten ähnliche Tiefe haben
        if abs(trough1[1] - trough2[1]) / trough1[1] < tolerance:
            # Finde Peak zwischen Troughs
            peak_idx = np.argmax(highs[trough1[0]:trough2[0]]) + trough1[0]
            resistance = highs[peak_idx]

            # Mindestabstand zwischen Troughs
            if trough2[0] - trough1[0] > window:
                return True, resistance

        return False, None

    @staticmethod
    def triangle_pattern(df: pd.DataFrame, window: int = 20) -> Dict[str, any]:
        """
        Triangle Pattern Detection (Ascending, Descending, Symmetrical)

        Returns:
            Dict mit pattern_type und trendlines
        """
        if len(df) < window * 2:
            return {'type': None}

        highs = df['High'].iloc[-window:].values
        lows = df['Low'].iloc[-window:].values
        indices = np.arange(len(highs))

        # Lineare Regression für Highs und Lows
        high_slope, high_intercept = np.polyfit(indices, highs, 1)
        low_slope, low_intercept = np.polyfit(indices, lows, 1)

        # Ascending Triangle: Flache Resistance, steigende Support
        if abs(high_slope) < 0.01 and low_slope > 0.01:
            return {
                'type': 'ascending',
                'upper_line': (high_slope, high_intercept),
                'lower_line': (low_slope, low_intercept),
                'signal': 'bullish_breakout_expected'
            }

        # Descending Triangle: Fallende Resistance, flache Support
        elif high_slope < -0.01 and abs(low_slope) < 0.01:
            return {
                'type': 'descending',
                'upper_line': (high_slope, high_intercept),
                'lower_line': (low_slope, low_intercept),
                'signal': 'bearish_breakdown_expected'
            }

        # Symmetrical Triangle: Beide konvergieren
        elif high_slope < -0.01 and low_slope > 0.01:
            # Überprüfe ob sie konvergieren
            apex_x = (low_intercept - high_intercept) / (high_slope - low_slope)
            if apex_x > 0 and apex_x < window * 2:
                return {
                    'type': 'symmetrical',
                    'upper_line': (high_slope, high_intercept),
                    'lower_line': (low_slope, low_intercept),
                    'signal': 'breakout_direction_uncertain'
                }

        return {'type': None}


class PatternAnalyzer:
    """Kombinierte Pattern-Analyse"""

    @staticmethod
    def analyze_all_patterns(df: pd.DataFrame) -> Dict[str, any]:
        """
        Analysiert alle Patterns im DataFrame

        Returns:
            Dict mit allen erkannten Patterns
        """
        results = {
            'candlestick_patterns': {},
            'chart_patterns': {},
            'support_resistance': {},
            'overall_signal': 'NEUTRAL'
        }

        try:
            # Candlestick Patterns
            df_copy = df.copy()
            results['candlestick_patterns'] = {
                'doji': CandlestickPatterns.doji(df_copy).iloc[-1] if len(df_copy) > 0 else False,
                'hammer': CandlestickPatterns.hammer(df_copy).iloc[-1] if len(df_copy) > 0 else False,
                'shooting_star': CandlestickPatterns.shooting_star(df_copy).iloc[-1] if len(df_copy) > 0 else False,
                'engulfing_bullish': CandlestickPatterns.engulfing_bullish(df_copy).iloc[-1] if len(df_copy) > 1 else False,
                'engulfing_bearish': CandlestickPatterns.engulfing_bearish(df_copy).iloc[-1] if len(df_copy) > 1 else False,
                'morning_star': CandlestickPatterns.morning_star(df_copy).iloc[-1] if len(df_copy) > 2 else False,
                'evening_star': CandlestickPatterns.evening_star(df_copy).iloc[-1] if len(df_copy) > 2 else False,
                'three_white_soldiers': CandlestickPatterns.three_white_soldiers(df_copy).iloc[-1] if len(df_copy) > 2 else False,
                'three_black_crows': CandlestickPatterns.three_black_crows(df_copy).iloc[-1] if len(df_copy) > 2 else False,
            }

            # Chart Patterns
            hs_found, hs_neckline = ChartPatterns.head_and_shoulders(df_copy)
            ihs_found, ihs_neckline = ChartPatterns.inverse_head_and_shoulders(df_copy)
            dt_found, dt_support = ChartPatterns.double_top(df_copy)
            db_found, db_resistance = ChartPatterns.double_bottom(df_copy)
            triangle = ChartPatterns.triangle_pattern(df_copy)

            results['chart_patterns'] = {
                'head_and_shoulders': {'found': hs_found, 'neckline': hs_neckline},
                'inverse_head_and_shoulders': {'found': ihs_found, 'neckline': ihs_neckline},
                'double_top': {'found': dt_found, 'support': dt_support},
                'double_bottom': {'found': db_found, 'resistance': db_resistance},
                'triangle': triangle
            }

            # Support/Resistance
            results['support_resistance'] = ChartPatterns.find_support_resistance(df_copy)

            # Overall Signal (Mehrheitsentscheidung)
            bullish_count = sum([
                results['candlestick_patterns']['hammer'],
                results['candlestick_patterns']['engulfing_bullish'],
                results['candlestick_patterns']['morning_star'],
                results['candlestick_patterns']['three_white_soldiers'],
                results['chart_patterns']['inverse_head_and_shoulders']['found'],
                results['chart_patterns']['double_bottom']['found'],
                triangle['type'] == 'ascending'
            ])

            bearish_count = sum([
                results['candlestick_patterns']['shooting_star'],
                results['candlestick_patterns']['engulfing_bearish'],
                results['candlestick_patterns']['evening_star'],
                results['candlestick_patterns']['three_black_crows'],
                results['chart_patterns']['head_and_shoulders']['found'],
                results['chart_patterns']['double_top']['found'],
                triangle['type'] == 'descending'
            ])

            if bullish_count > bearish_count + 1:
                results['overall_signal'] = 'BULLISH'
            elif bearish_count > bullish_count + 1:
                results['overall_signal'] = 'BEARISH'
            else:
                results['overall_signal'] = 'NEUTRAL'

            logger.info(f"Pattern-Analyse: {bullish_count} Bullish, {bearish_count} Bearish → {results['overall_signal']}")

        except Exception as e:
            logger.error(f"Fehler bei Pattern-Analyse: {e}")

        return results

    @staticmethod
    def add_pattern_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt Pattern-Erkennungs-Spalten zum DataFrame hinzu
        """
        try:
            df['Pattern_Doji'] = CandlestickPatterns.doji(df)
            df['Pattern_Hammer'] = CandlestickPatterns.hammer(df)
            df['Pattern_ShootingStar'] = CandlestickPatterns.shooting_star(df)
            df['Pattern_BullishEngulfing'] = CandlestickPatterns.engulfing_bullish(df)
            df['Pattern_BearishEngulfing'] = CandlestickPatterns.engulfing_bearish(df)
            df['Pattern_MorningStar'] = CandlestickPatterns.morning_star(df)
            df['Pattern_EveningStar'] = CandlestickPatterns.evening_star(df)
            df['Pattern_ThreeWhiteSoldiers'] = CandlestickPatterns.three_white_soldiers(df)
            df['Pattern_ThreeBlackCrows'] = CandlestickPatterns.three_black_crows(df)

            logger.info("✅ Pattern-Spalten hinzugefügt")
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Pattern-Spalten: {e}")

        return df

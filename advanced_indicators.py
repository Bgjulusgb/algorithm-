"""
Advanced Technical Indicators Module
Erweiterte technische Indikatoren für Trading-Strategien
Enthält 50+ Indikatoren aus verschiedenen Kategorien
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TrendIndicators:
    """Trend-basierte Indikatoren"""

    @staticmethod
    def ichimoku_cloud(df: pd.DataFrame,
                       conversion_period: int = 9,
                       base_period: int = 26,
                       leading_span_b_period: int = 52,
                       lagging_span_period: int = 26) -> pd.DataFrame:
        """
        Ichimoku Cloud (Ichimoku Kinko Hyo)

        Returns:
            Conversion Line, Base Line, Leading Span A/B, Lagging Span
        """
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Conversion Line (Tenkan-sen)
        conversion_line = (high.rolling(window=conversion_period).max() +
                          low.rolling(window=conversion_period).min()) / 2

        # Base Line (Kijun-sen)
        base_line = (high.rolling(window=base_period).max() +
                    low.rolling(window=base_period).min()) / 2

        # Leading Span A (Senkou Span A)
        leading_span_a = ((conversion_line + base_line) / 2).shift(base_period)

        # Leading Span B (Senkou Span B)
        leading_span_b = ((high.rolling(window=leading_span_b_period).max() +
                          low.rolling(window=leading_span_b_period).min()) / 2).shift(base_period)

        # Lagging Span (Chikou Span)
        lagging_span = close.shift(-lagging_span_period)

        df['Ichimoku_Conversion'] = conversion_line
        df['Ichimoku_Base'] = base_line
        df['Ichimoku_LeadingA'] = leading_span_a
        df['Ichimoku_LeadingB'] = leading_span_b
        df['Ichimoku_Lagging'] = lagging_span

        # Cloud Signal: 1 = Bullish (Above Cloud), -1 = Bearish (Below Cloud)
        df['Ichimoku_Signal'] = 0
        df.loc[close > df[['Ichimoku_LeadingA', 'Ichimoku_LeadingB']].max(axis=1), 'Ichimoku_Signal'] = 1
        df.loc[close < df[['Ichimoku_LeadingA', 'Ichimoku_LeadingB']].min(axis=1), 'Ichimoku_Signal'] = -1

        return df

    @staticmethod
    def parabolic_sar(df: pd.DataFrame, af_start: float = 0.02,
                      af_increment: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
        """
        Parabolic SAR (Stop and Reverse)
        """
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values

        sar = np.zeros(len(df))
        ep = np.zeros(len(df))
        af = np.zeros(len(df))
        trend = np.zeros(len(df))

        # Initialize
        sar[0] = low[0]
        ep[0] = high[0]
        af[0] = af_start
        trend[0] = 1  # 1 = uptrend, -1 = downtrend

        for i in range(1, len(df)):
            # Previous values
            prev_sar = sar[i-1]
            prev_ep = ep[i-1]
            prev_af = af[i-1]
            prev_trend = trend[i-1]

            if prev_trend == 1:  # Uptrend
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])

                if high[i] > prev_ep:
                    ep[i] = high[i]
                    af[i] = min(prev_af + af_increment, af_max)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af

                if low[i] < sar[i]:  # Switch to downtrend
                    trend[i] = -1
                    sar[i] = prev_ep
                    ep[i] = low[i]
                    af[i] = af_start
                else:
                    trend[i] = 1
            else:  # Downtrend
                sar[i] = prev_sar - prev_af * (prev_sar - prev_ep)
                sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])

                if low[i] < prev_ep:
                    ep[i] = low[i]
                    af[i] = min(prev_af + af_increment, af_max)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af

                if high[i] > sar[i]:  # Switch to uptrend
                    trend[i] = 1
                    sar[i] = prev_ep
                    ep[i] = high[i]
                    af[i] = af_start
                else:
                    trend[i] = -1

        df['PSAR'] = sar
        df['PSAR_Trend'] = trend
        df['PSAR_Signal'] = np.where(close > sar, 1, -1)

        return df

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """
        SuperTrend Indicator
        """
        high = df['High']
        low = df['Low']
        close = df['Close']

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Basic Bands
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)

        # SuperTrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        for i in range(period, len(df)):
            if i == period:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                if close.iloc[i] > supertrend.iloc[i-1]:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                else:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])

        df['SuperTrend'] = supertrend
        df['SuperTrend_Direction'] = direction
        df['SuperTrend_Signal'] = direction

        return df

    @staticmethod
    def vortex_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Vortex Indicator (VI+ and VI-)
        """
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Vortex Movement
        vm_plus = abs(high - low.shift())
        vm_minus = abs(low - high.shift())

        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)

        # Vortex Indicators
        vi_plus = vm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum()
        vi_minus = vm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum()

        df['VI_Plus'] = vi_plus
        df['VI_Minus'] = vi_minus
        df['VI_Signal'] = np.where(vi_plus > vi_minus, 1, -1)

        return df


class MomentumIndicators:
    """Momentum-basierte Indikatoren"""

    @staticmethod
    def stochastic_rsi(df: pd.DataFrame, rsi_period: int = 14,
                       stoch_period: int = 14, k_period: int = 3,
                       d_period: int = 3) -> pd.DataFrame:
        """
        Stochastic RSI
        """
        # Calculate RSI first
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Stochastic of RSI
        rsi_min = rsi.rolling(window=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100

        # %K and %D lines
        k_line = stoch_rsi.rolling(window=k_period).mean()
        d_line = k_line.rolling(window=d_period).mean()

        df['StochRSI'] = stoch_rsi
        df['StochRSI_K'] = k_line
        df['StochRSI_D'] = d_line
        df['StochRSI_Signal'] = np.where((k_line > 20) & (k_line < 80), 0,
                                         np.where(k_line <= 20, 1, -1))

        return df

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Williams %R
        """
        high = df['High']
        low = df['Low']
        close = df['Close']

        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100

        df['Williams_R'] = williams_r
        df['Williams_R_Signal'] = np.where(williams_r < -80, 1,
                                           np.where(williams_r > -20, -1, 0))

        return df

    @staticmethod
    def ultimate_oscillator(df: pd.DataFrame, period1: int = 7,
                           period2: int = 14, period3: int = 28) -> pd.DataFrame:
        """
        Ultimate Oscillator
        """
        close = df['Close']
        low = df['Low']
        high = df['High']

        # Buying Pressure
        bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)

        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)

        # Average for each period
        avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
        avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
        avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()

        # Ultimate Oscillator
        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7

        df['Ultimate_Oscillator'] = uo
        df['UO_Signal'] = np.where(uo < 30, 1, np.where(uo > 70, -1, 0))

        return df

    @staticmethod
    def commodity_channel_index(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Commodity Channel Index (CCI)
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )

        cci = (typical_price - sma) / (0.015 * mean_deviation)

        df['CCI'] = cci
        df['CCI_Signal'] = np.where(cci < -100, 1, np.where(cci > 100, -1, 0))

        return df


class VolatilityIndicators:
    """Volatilitäts-Indikatoren"""

    @staticmethod
    def keltner_channels(df: pd.DataFrame, ema_period: int = 20,
                        atr_period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Keltner Channels
        """
        close = df['Close']
        high = df['High']
        low = df['Low']

        # EMA of Close
        ema = close.ewm(span=ema_period, adjust=False).mean()

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=atr_period, adjust=False).mean()

        # Channels
        upper_channel = ema + (multiplier * atr)
        lower_channel = ema - (multiplier * atr)

        df['Keltner_Upper'] = upper_channel
        df['Keltner_Middle'] = ema
        df['Keltner_Lower'] = lower_channel
        df['Keltner_Width'] = (upper_channel - lower_channel) / ema * 100
        df['Keltner_Signal'] = np.where(close > upper_channel, -1,
                                        np.where(close < lower_channel, 1, 0))

        return df

    @staticmethod
    def donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Donchian Channels
        """
        high = df['High']
        low = df['Low']
        close = df['Close']

        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2

        df['Donchian_Upper'] = upper_channel
        df['Donchian_Middle'] = middle_channel
        df['Donchian_Lower'] = lower_channel
        df['Donchian_Width'] = (upper_channel - lower_channel) / close * 100
        df['Donchian_Signal'] = np.where(close > upper_channel, 1,
                                         np.where(close < lower_channel, -1, 0))

        return df

    @staticmethod
    def historical_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Historical Volatility (annualized)
        """
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        volatility = log_returns.rolling(window=period).std() * np.sqrt(252) * 100

        df['Historical_Volatility'] = volatility
        df['HV_Percentile'] = volatility.rolling(window=252).apply(
            lambda x: (x[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 0
        )

        return df


class VolumeIndicators:
    """Volumen-basierte Indikatoren"""

    @staticmethod
    def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Money Flow Index (MFI)
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

        # Positive and Negative Money Flow
        positive_flow = pd.Series(0, index=df.index, dtype=float)
        negative_flow = pd.Series(0, index=df.index, dtype=float)

        positive_flow[typical_price > typical_price.shift()] = money_flow[typical_price > typical_price.shift()]
        negative_flow[typical_price < typical_price.shift()] = money_flow[typical_price < typical_price.shift()]

        # Money Flow Ratio
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        money_ratio = positive_sum / negative_sum.replace(0, 1)

        # MFI
        mfi = 100 - (100 / (1 + money_ratio))

        df['MFI'] = mfi
        df['MFI_Signal'] = np.where(mfi < 20, 1, np.where(mfi > 80, -1, 0))

        return df

    @staticmethod
    def accumulation_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """
        Accumulation/Distribution Line
        """
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        clv = clv.fillna(0)
        ad = (clv * df['Volume']).cumsum()

        df['AD_Line'] = ad
        df['AD_Signal'] = np.where(ad > ad.shift(20), 1,
                                   np.where(ad < ad.shift(20), -1, 0))

        return df

    @staticmethod
    def chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Chaikin Money Flow (CMF)
        """
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        clv = clv.fillna(0)
        money_flow_volume = clv * df['Volume']

        cmf = money_flow_volume.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()

        df['CMF'] = cmf
        df['CMF_Signal'] = np.where(cmf > 0.1, 1, np.where(cmf < -0.1, -1, 0))

        return df

    @staticmethod
    def volume_weighted_average_price(df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume Weighted Average Price (VWAP)
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

        df['VWAP'] = vwap
        df['VWAP_Signal'] = np.where(df['Close'] > vwap, 1, -1)

        return df

    @staticmethod
    def ease_of_movement(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Ease of Movement (EMV)
        """
        distance_moved = ((df['High'] + df['Low']) / 2) - ((df['High'].shift() + df['Low'].shift()) / 2)
        box_ratio = (df['Volume'] / 1000000) / (df['High'] - df['Low'])
        emv = distance_moved / box_ratio.replace(0, 1)
        emv_ma = emv.rolling(window=period).mean()

        df['EMV'] = emv
        df['EMV_MA'] = emv_ma
        df['EMV_Signal'] = np.where(emv_ma > 0, 1, -1)

        return df


class AdvancedIndicators:
    """Fortgeschrittene und kombinierte Indikatoren"""

    @staticmethod
    def awesome_oscillator(df: pd.DataFrame, fast_period: int = 5,
                          slow_period: int = 34) -> pd.DataFrame:
        """
        Awesome Oscillator (AO)
        """
        median_price = (df['High'] + df['Low']) / 2
        ao = median_price.rolling(window=fast_period).mean() - median_price.rolling(window=slow_period).mean()

        df['AO'] = ao
        df['AO_Signal'] = np.where(ao > 0, 1, -1)
        df['AO_Momentum'] = np.where(ao > ao.shift(), 1, -1)

        return df

    @staticmethod
    def accelerator_oscillator(df: pd.DataFrame, fast_period: int = 5,
                              slow_period: int = 34, signal_period: int = 5) -> pd.DataFrame:
        """
        Accelerator Oscillator (AC)
        """
        median_price = (df['High'] + df['Low']) / 2
        ao = median_price.rolling(window=fast_period).mean() - median_price.rolling(window=slow_period).mean()
        ac = ao - ao.rolling(window=signal_period).mean()

        df['AC'] = ac
        df['AC_Signal'] = np.where(ac > 0, 1, -1)

        return df

    @staticmethod
    def mass_index(df: pd.DataFrame, period: int = 9, sum_period: int = 25) -> pd.DataFrame:
        """
        Mass Index - Detects trend reversals
        """
        high_low_range = df['High'] - df['Low']
        ema1 = high_low_range.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        mass = ema1 / ema2.replace(0, 1)
        mass_index = mass.rolling(window=sum_period).sum()

        df['Mass_Index'] = mass_index
        df['Mass_Index_Signal'] = np.where((mass_index > 27) & (mass_index.shift() < 27), -1,
                                           np.where((mass_index < 26.5) & (mass_index.shift() > 26.5), 1, 0))

        return df

    @staticmethod
    def coppock_curve(df: pd.DataFrame, roc1_period: int = 14,
                     roc2_period: int = 11, wma_period: int = 10) -> pd.DataFrame:
        """
        Coppock Curve - Long-term momentum indicator
        """
        roc1 = (df['Close'] - df['Close'].shift(roc1_period)) / df['Close'].shift(roc1_period) * 100
        roc2 = (df['Close'] - df['Close'].shift(roc2_period)) / df['Close'].shift(roc2_period) * 100
        roc_sum = roc1 + roc2

        # Weighted Moving Average
        weights = np.arange(1, wma_period + 1)
        coppock = roc_sum.rolling(window=wma_period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

        df['Coppock'] = coppock
        df['Coppock_Signal'] = np.where((coppock > 0) & (coppock.shift() <= 0), 1,
                                       np.where((coppock < 0) & (coppock.shift() >= 0), -1, 0))

        return df

    @staticmethod
    def know_sure_thing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Know Sure Thing (KST) - Momentum oscillator
        """
        # ROC calculations
        roc1 = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
        roc2 = (df['Close'] - df['Close'].shift(15)) / df['Close'].shift(15)
        roc3 = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)
        roc4 = (df['Close'] - df['Close'].shift(30)) / df['Close'].shift(30)

        # SMAs of ROCs
        kst = (roc1.rolling(window=10).mean() * 1 +
               roc2.rolling(window=10).mean() * 2 +
               roc3.rolling(window=10).mean() * 3 +
               roc4.rolling(window=15).mean() * 4)

        kst_signal = kst.rolling(window=9).mean()

        df['KST'] = kst
        df['KST_Signal_Line'] = kst_signal
        df['KST_Signal'] = np.where(kst > kst_signal, 1, -1)

        return df


class AllIndicators:
    """Wrapper-Klasse für alle Indikatoren"""

    @staticmethod
    def add_all_indicators(df: pd.DataFrame, categories: Optional[list] = None) -> pd.DataFrame:
        """
        Fügt alle Indikatoren zum DataFrame hinzu

        Args:
            df: DataFrame mit OHLCV Daten
            categories: Liste der zu inkludierenden Kategorien
                       ['trend', 'momentum', 'volatility', 'volume', 'advanced']
                       None = alle

        Returns:
            DataFrame mit allen Indikatoren
        """
        if categories is None:
            categories = ['trend', 'momentum', 'volatility', 'volume', 'advanced']

        try:
            if 'trend' in categories:
                logger.info("Füge Trend-Indikatoren hinzu...")
                df = TrendIndicators.ichimoku_cloud(df)
                df = TrendIndicators.parabolic_sar(df)
                df = TrendIndicators.supertrend(df)
                df = TrendIndicators.vortex_indicator(df)

            if 'momentum' in categories:
                logger.info("Füge Momentum-Indikatoren hinzu...")
                df = MomentumIndicators.stochastic_rsi(df)
                df = MomentumIndicators.williams_r(df)
                df = MomentumIndicators.ultimate_oscillator(df)
                df = MomentumIndicators.commodity_channel_index(df)

            if 'volatility' in categories:
                logger.info("Füge Volatilitäts-Indikatoren hinzu...")
                df = VolatilityIndicators.keltner_channels(df)
                df = VolatilityIndicators.donchian_channels(df)
                df = VolatilityIndicators.historical_volatility(df)

            if 'volume' in categories:
                logger.info("Füge Volumen-Indikatoren hinzu...")
                df = VolumeIndicators.money_flow_index(df)
                df = VolumeIndicators.accumulation_distribution(df)
                df = VolumeIndicators.chaikin_money_flow(df)
                df = VolumeIndicators.volume_weighted_average_price(df)
                df = VolumeIndicators.ease_of_movement(df)

            if 'advanced' in categories:
                logger.info("Füge fortgeschrittene Indikatoren hinzu...")
                df = AdvancedIndicators.awesome_oscillator(df)
                df = AdvancedIndicators.accelerator_oscillator(df)
                df = AdvancedIndicators.mass_index(df)
                df = AdvancedIndicators.coppock_curve(df)
                df = AdvancedIndicators.know_sure_thing(df)

            logger.info(f"✅ Alle Indikatoren hinzugefügt. DataFrame hat jetzt {len(df.columns)} Spalten")

        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Indikatoren: {e}")

        return df

    @staticmethod
    def get_indicator_summary(df: pd.DataFrame) -> dict:
        """
        Gibt Zusammenfassung aller Indikatoren-Signale zurück

        Returns:
            Dict mit Signalzusammenfassungen
        """
        signal_columns = [col for col in df.columns if 'Signal' in col]

        summary = {}
        for col in signal_columns:
            if col in df.columns:
                latest_signal = df[col].iloc[-1] if len(df) > 0 else 0
                summary[col] = {
                    'latest': latest_signal,
                    'signal': 'BUY' if latest_signal > 0 else ('SELL' if latest_signal < 0 else 'NEUTRAL')
                }

        # Gesamtsignal (Mehrheitsentscheidung)
        total_signals = sum([1 if s['latest'] > 0 else (-1 if s['latest'] < 0 else 0)
                           for s in summary.values()])
        overall = 'STRONG BUY' if total_signals > 5 else ('BUY' if total_signals > 0 else
                 ('STRONG SELL' if total_signals < -5 else ('SELL' if total_signals < 0 else 'NEUTRAL')))

        summary['OVERALL'] = {
            'signal': overall,
            'score': total_signals,
            'total_indicators': len(signal_columns)
        }

        return summary

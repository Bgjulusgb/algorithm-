"""
Standalone Test für Enhanced Signal Generator
Testet ohne yfinance-Abhängigkeit
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_sma(df, period):
    """Berechnet Simple Moving Average"""
    return df['Close'].rolling(window=period).mean()


def calculate_ema(df, period):
    """Berechnet Exponential Moving Average"""
    return df['Close'].ewm(span=period, adjust=False).mean()


def calculate_rsi(df, period=14):
    """Berechnet RSI"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Berechnet MACD"""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal

    return macd, macd_signal, macd_hist


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Berechnet Bollinger Bands"""
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()

    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)

    return upper, sma, lower


def calculate_atr(df, period=14):
    """Berechnet Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_stochastic(df, period=14, smooth_k=3, smooth_d=3):
    """Berechnet Stochastic Oscillator"""
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()

    stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    stoch_k = stoch_k.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=smooth_d).mean()

    return stoch_k, stoch_d


def calculate_williams_r(df, period=14):
    """Berechnet Williams %R"""
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()

    williams = -100 * (high_max - df['Close']) / (high_max - low_min)
    return williams


def calculate_cci(df, period=20):
    """Berechnet Commodity Channel Index"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

    cci = (typical_price - sma_tp) / (0.015 * mad)
    return cci


def calculate_mfi(df, period=14):
    """Berechnet Money Flow Index"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()

    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    return mfi


def calculate_adx(df, period=14):
    """Berechnet Average Directional Index"""
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()

    pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    atr = calculate_atr(df, period)

    pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
    neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)

    dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
    adx = dx.rolling(window=period).mean()

    return adx, pos_di, neg_di


def calculate_parabolic_sar(df, af_start=0.02, af_max=0.2):
    """Vereinfachter Parabolic SAR"""
    # Vereinfachte Version - nutzt High/Low
    sar = df['Low'].rolling(window=5).min()
    return sar


def add_basic_indicators(df):
    """Fügt VIELE Indikatoren hinzu (ohne yfinance)"""
    # Moving Averages
    df['SMA_50'] = calculate_sma(df, 50)
    df['SMA_200'] = calculate_sma(df, 200)
    df['EMA_12'] = calculate_ema(df, 12)
    df['EMA_26'] = calculate_ema(df, 26)

    # RSI
    df['RSI'] = calculate_rsi(df, 14)

    # MACD
    macd, signal, hist = calculate_macd(df)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist

    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    # ATR
    df['ATR'] = calculate_atr(df)

    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(df)
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d

    # Williams %R
    df['Williams_R'] = calculate_williams_r(df)

    # CCI
    df['CCI'] = calculate_cci(df)

    # MFI
    df['MFI'] = calculate_mfi(df)

    # ADX
    adx, pos_di, neg_di = calculate_adx(df)
    df['ADX'] = adx
    df['DI_plus'] = pos_di
    df['DI_minus'] = neg_di

    # Parabolic SAR
    df['PSAR'] = calculate_parabolic_sar(df)

    # Volume-basierte Indikatoren
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()

    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # VWAP (Volume Weighted Average Price)
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # A/D Line
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    clv = clv.fillna(0)
    df['AD_Line'] = (clv * df['Volume']).cumsum()

    # CMF (Chaikin Money Flow)
    money_flow_volume = clv * df['Volume']
    df['CMF'] = money_flow_volume.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # VPT (Volume Price Trend)
    df['VPT'] = (df['Volume'] * df['Close'].pct_change()).cumsum()

    # Keltner Channels
    ema_20 = calculate_ema(df, 20)
    df['Keltner_Upper'] = ema_20 + (2 * df['ATR'])
    df['Keltner_Lower'] = ema_20 - (2 * df['ATR'])

    logger.info(f"✅ VIELE Indikatoren hinzugefügt: {len(df.columns)} Spalten")
    logger.info(f"   Anzahl Indikatoren: {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])}")

    return df


def create_test_data(days=300, trend='bullish'):
    """Erstellt Test-Daten"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    base_price = 100
    prices = [base_price]

    if trend == 'bullish':
        drift = 0.002
        volatility = 0.015
    elif trend == 'bearish':
        drift = -0.002
        volatility = 0.015
    else:
        drift = 0.0
        volatility = 0.01

    for i in range(1, days):
        change = drift + np.random.normal(0, volatility)
        prices.append(prices[-1] * (1 + change))

    df = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
        'High': [p * (1 + np.random.uniform(0.005, 0.02)) for p in prices],
        'Low': [p * (1 + np.random.uniform(-0.02, -0.005)) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(1000000, 5000000) for _ in range(days)]
    })
    df.set_index('Date', inplace=True)

    return df


def test_enhanced_generator():
    """Testet Enhanced Signal Generator"""
    logger.info("\n" + "="*80)
    logger.info("🔍 TESTE ENHANCED SIGNAL GENERATOR")
    logger.info("="*80)

    from enhanced_signal_generator import EnhancedSignalGenerator

    gen = EnhancedSignalGenerator()
    logger.info("✅ Enhanced Signal Generator erstellt")

    scenarios = [
        ('bullish', 'Bullish Trend'),
        ('bearish', 'Bearish Trend'),
        ('sideways', 'Seitwärtsbewegung')
    ]

    all_passed = True

    for trend, description in scenarios:
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 SZENARIO: {description}")
        logger.info(f"{'='*80}")

        # Erstelle Test-Daten
        df = create_test_data(days=300, trend=trend)
        logger.info(f"Test-Daten: {len(df)} Tage")
        logger.info(f"Start: ${df['Close'].iloc[0]:.2f}, End: ${df['Close'].iloc[-1]:.2f}")
        logger.info(f"Änderung: {(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:+.2f}%")

        # Füge Indikatoren hinzu
        df = add_basic_indicators(df)

        # Teste Signal-Generierung
        result = gen.generate_comprehensive_signals(df, len(df)-1)

        if 'error' in result:
            logger.warning(f"⚠️ Fehler: {result['error']}")
            continue

        # Zeige Ergebnisse
        logger.info(f"\n📈 ERGEBNISSE:")
        logger.info(f"   Signal: {result['aggregated_signal']} ({result['signal_strength']})")
        logger.info(f"   Konfidenz: {result['total_confidence']*100:.1f}%")
        logger.info(f"   Anzahl Signale: {result['num_signals']}")
        logger.info(f"   Bullish: {result['bullish_count']}, Bearish: {result['bearish_count']}")

        logger.info(f"\n📊 NACH KATEGORIE:")
        for category, data in result['signals_by_category'].items():
            if data['count'] > 0:
                logger.info(f"   {category.upper()}: {data['count']} Signale "
                          f"(Avg: {data['average']:+.2f})")

        logger.info(f"\n📋 DETAILS (erste 15):")
        for i, detail in enumerate(result['details'][:15], 1):
            logger.info(f"   {i}. {detail}")

        if len(result['details']) > 15:
            logger.info(f"   ... und {len(result['details']) - 15} weitere")

        # Validierung
        logger.info(f"\n✅ VALIDIERUNG:")
        passed = True

        if trend == 'bullish' and result['aggregated_signal'] != 'BUY':
            logger.warning(f"   ⚠️ Erwartet BUY für Bullish, bekommen {result['aggregated_signal']}")
            passed = False
        elif trend == 'bearish' and result['aggregated_signal'] != 'SELL':
            logger.warning(f"   ⚠️ Erwartet SELL für Bearish, bekommen {result['aggregated_signal']}")
            passed = False
        else:
            logger.info(f"   ✅ Korrektes Signal für {description}")

        if result['total_confidence'] > 0.4:
            logger.info(f"   ✅ Gute Konfidenz: {result['total_confidence']*100:.1f}%")
        else:
            logger.warning(f"   ⚠️ Niedrige Konfidenz: {result['total_confidence']*100:.1f}%")

        if result['num_signals'] >= 10:
            logger.info(f"   ✅ Viele Signale: {result['num_signals']}")
        else:
            logger.warning(f"   ⚠️ Wenige Signale: {result['num_signals']}")
            passed = False

        all_passed = all_passed and passed

    return all_passed


def test_signal_count():
    """Testet ob genug Signale generiert werden"""
    logger.info("\n" + "="*80)
    logger.info("🔢 TESTE SIGNAL-ANZAHL")
    logger.info("="*80)

    from enhanced_signal_generator import EnhancedSignalGenerator

    gen = EnhancedSignalGenerator()

    # Erstelle reichhaltige Test-Daten
    df = create_test_data(days=300, trend='bullish')
    df = add_basic_indicators(df)

    idx = len(df) - 1

    # Teste jede Kategorie
    categories = {
        'Trend': gen._generate_trend_signals,
        'Momentum': gen._generate_momentum_signals,
        'Volatility': gen._generate_volatility_signals,
        'Volume': gen._generate_volume_signals,
        'Pattern': gen._generate_pattern_signals,
        'Oscillator': gen._generate_oscillator_signals
    }

    total_signals = 0
    category_results = {}

    for cat_name, cat_func in categories.items():
        logger.info(f"\n📊 {cat_name}:")
        result = cat_func(df, idx)

        logger.info(f"   Anzahl: {result['count']}")
        if result['count'] > 0:
            logger.info(f"   Durchschnitt: {result['average']:+.2f}")
            logger.info(f"   Details (erste 3):")
            for detail in result['details'][:3]:
                logger.info(f"      - {detail}")

        total_signals += result['count']
        category_results[cat_name] = result['count']

    logger.info(f"\n{'='*80}")
    logger.info(f"📈 GESAMTSIGNALE: {total_signals}")
    logger.info(f"{'='*80}")

    for cat, count in category_results.items():
        logger.info(f"   {cat}: {count}")

    if total_signals >= 25:
        logger.info(f"\n✅ Hervorragend! {total_signals} Signale generiert")
        return True
    elif total_signals >= 15:
        logger.info(f"\n✅ Gut! {total_signals} Signale generiert")
        return True
    elif total_signals >= 10:
        logger.warning(f"\n⚠️ Ausreichend: {total_signals} Signale")
        return True
    else:
        logger.error(f"\n❌ Zu wenige Signale: {total_signals}")
        return False


def main():
    """Hauptfunktion"""
    logger.info("\n" + "="*80)
    logger.info("🧪 STARTE STANDALONE SIGNAL TESTS")
    logger.info("="*80)

    results = []

    # Test 1: Signal Count
    try:
        results.append(("Signal Count", test_signal_count()))
    except Exception as e:
        logger.error(f"❌ Signal Count Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Signal Count", False))

    # Test 2: Enhanced Generator
    try:
        results.append(("Enhanced Generator", test_enhanced_generator()))
    except Exception as e:
        logger.error(f"❌ Enhanced Generator Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Enhanced Generator", False))

    # Zusammenfassung
    logger.info("\n" + "="*80)
    logger.info("📋 TEST-ZUSAMMENFASSUNG")
    logger.info("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ BESTANDEN" if result else "❌ FEHLGESCHLAGEN"
        logger.info(f"{status}: {name}")

    logger.info(f"\n📊 Ergebnis: {passed}/{total} Tests bestanden")
    logger.info("="*80 + "\n")

    if passed == total:
        logger.info("🎉 ALLE TESTS BESTANDEN!")
        return 0
    else:
        logger.error(f"❌ {total - passed} Test(s) fehlgeschlagen")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

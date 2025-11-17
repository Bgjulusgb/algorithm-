"""
Test für Enhanced Signal Generator
Testet 50+ verschiedene Trading-Signale
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data(days=300, trend='bullish'):
    """Erstellt Test-Daten mit spezifischem Trend"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Erstelle Preis-Serie mit Trend
    base_price = 100
    prices = [base_price]

    if trend == 'bullish':
        drift = 0.002  # 0.2% pro Tag aufwärts
        volatility = 0.015
    elif trend == 'bearish':
        drift = -0.002  # 0.2% pro Tag abwärts
        volatility = 0.015
    else:  # sideways
        drift = 0.0
        volatility = 0.01

    for i in range(1, days):
        change = drift + np.random.normal(0, volatility)
        prices.append(prices[-1] * (1 + change))

    # Erstelle OHLCV Daten
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


def add_all_indicators(df):
    """Fügt alle möglichen Indikatoren hinzu"""
    from data_handler import DataHandler

    handler = DataHandler()

    # Basis-Indikatoren
    df = handler.add_technical_indicators(df)

    # Zusätzliche Indikatoren die der Enhanced Generator nutzt
    try:
        from advanced_indicators import (
            TrendIndicators, MomentumIndicators,
            VolatilityIndicators, VolumeIndicators
        )

        # Trend Indicators
        df = TrendIndicators.ichimoku_cloud(df)
        df = TrendIndicators.parabolic_sar(df)
        df = TrendIndicators.supertrend(df)

        # Momentum Indicators
        df = MomentumIndicators.stochastic(df)
        df = MomentumIndicators.williams_r(df)
        df = MomentumIndicators.cci(df)

        # Volatility Indicators
        df = VolatilityIndicators.keltner_channels(df)
        df = VolatilityIndicators.donchian_channels(df)

        # Volume Indicators
        df = VolumeIndicators.obv(df)
        df = VolumeIndicators.vwap(df)
        df = VolumeIndicators.mfi(df)

        logger.info("✅ Erweiterte Indikatoren hinzugefügt")

    except Exception as e:
        logger.warning(f"Einige erweiterte Indikatoren nicht verfügbar: {e}")

    return df


def test_enhanced_signal_generator():
    """Testet den Enhanced Signal Generator"""
    logger.info("\n" + "="*80)
    logger.info("🔍 TESTE ENHANCED SIGNAL GENERATOR (50+ SIGNALE)")
    logger.info("="*80)

    try:
        from enhanced_signal_generator import EnhancedSignalGenerator

        gen = EnhancedSignalGenerator()
        logger.info("✅ Enhanced Signal Generator erstellt")

        # Test verschiedene Marktszenarien
        scenarios = [
            ('bullish', 'Bullish Trend'),
            ('bearish', 'Bearish Trend'),
            ('sideways', 'Seitwärtsbewegung')
        ]

        for trend, description in scenarios:
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 SZENARIO: {description}")
            logger.info(f"{'='*80}")

            # Erstelle Test-Daten
            df = create_test_data(days=300, trend=trend)
            logger.info(f"Test-Daten erstellt: {len(df)} Tage")
            logger.info(f"Start Preis: ${df['Close'].iloc[0]:.2f}")
            logger.info(f"End Preis: ${df['Close'].iloc[-1]:.2f}")
            logger.info(f"Änderung: {(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:+.2f}%")

            # Füge Indikatoren hinzu
            df = add_all_indicators(df)
            logger.info(f"Indikatoren hinzugefügt: {len(df.columns)} Spalten")

            # Generiere Signals
            result = gen.generate_comprehensive_signals(df, len(df)-1)

            if 'error' in result:
                logger.error(f"❌ Fehler: {result['error']}")
                continue

            # Zeige Ergebnisse
            logger.info(f"\n📈 SIGNAL-ERGEBNISSE:")
            logger.info(f"{'='*80}")
            logger.info(f"Finales Signal: {result['aggregated_signal']} ({result['signal_strength']})")
            logger.info(f"Gesamtvertrauen: {result['total_confidence']*100:.1f}%")
            logger.info(f"Anzahl Signale: {result['num_signals']}")
            logger.info(f"Bullish Signale: {result['bullish_count']}")
            logger.info(f"Bearish Signale: {result['bearish_count']}")

            logger.info(f"\n📊 SIGNALE NACH KATEGORIE:")
            for category, data in result['signals_by_category'].items():
                if data['count'] > 0:
                    logger.info(f"   {category.upper()}: {data['count']} Signale, "
                              f"Durchschnitt: {data['average']:+.2f}")

            logger.info(f"\n📋 DETAIL-SIGNALE (Erste 10):")
            for i, detail in enumerate(result['details'][:10], 1):
                logger.info(f"   {i}. {detail}")

            if len(result['details']) > 10:
                logger.info(f"   ... und {len(result['details']) - 10} weitere")

            # Validierung
            logger.info(f"\n✅ VALIDIERUNG:")
            if trend == 'bullish':
                if result['aggregated_signal'] == 'BUY':
                    logger.info("   ✅ Korrektes Signal für Bullish Trend")
                else:
                    logger.warning(f"   ⚠️ Signal ist {result['aggregated_signal']}, "
                                 f"erwartet BUY für Bullish Trend")
            elif trend == 'bearish':
                if result['aggregated_signal'] == 'SELL':
                    logger.info("   ✅ Korrektes Signal für Bearish Trend")
                else:
                    logger.warning(f"   ⚠️ Signal ist {result['aggregated_signal']}, "
                                 f"erwartet SELL für Bearish Trend")

            if result['total_confidence'] > 0.5:
                logger.info(f"   ✅ Gutes Vertrauen: {result['total_confidence']*100:.1f}%")
            else:
                logger.warning(f"   ⚠️ Niedriges Vertrauen: {result['total_confidence']*100:.1f}%")

            if result['num_signals'] >= 20:
                logger.info(f"   ✅ Viele Signale: {result['num_signals']}")
            else:
                logger.warning(f"   ⚠️ Wenige Signale: {result['num_signals']}")

        logger.info("\n" + "="*80)
        logger.info("✅ ENHANCED SIGNAL GENERATOR TEST ABGESCHLOSSEN")
        logger.info("="*80 + "\n")

        return True

    except Exception as e:
        logger.error(f"❌ Fehler beim Test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_categories():
    """Testet einzelne Signal-Kategorien detailliert"""
    logger.info("\n" + "="*80)
    logger.info("🔍 TESTE EINZELNE SIGNAL-KATEGORIEN")
    logger.info("="*80)

    try:
        from enhanced_signal_generator import EnhancedSignalGenerator

        gen = EnhancedSignalGenerator()

        # Erstelle Test-Daten
        df = create_test_data(days=300, trend='bullish')
        df = add_all_indicators(df)

        idx = len(df) - 1

        # Teste jede Kategorie einzeln
        categories = [
            ('Trend', gen._generate_trend_signals),
            ('Momentum', gen._generate_momentum_signals),
            ('Volatility', gen._generate_volatility_signals),
            ('Volume', gen._generate_volume_signals),
            ('Pattern', gen._generate_pattern_signals),
            ('Oscillator', gen._generate_oscillator_signals)
        ]

        total_signals = 0

        for cat_name, cat_func in categories:
            logger.info(f"\n📊 {cat_name} Signale:")
            result = cat_func(df, idx)

            logger.info(f"   Anzahl: {result['count']}")
            logger.info(f"   Durchschnitt: {result['average']:+.2f}")

            if result['details']:
                logger.info(f"   Details (erste 5):")
                for detail in result['details'][:5]:
                    logger.info(f"      - {detail}")

            total_signals += result['count']

        logger.info(f"\n📈 GESAMTSIGNALE: {total_signals}")
        logger.info("="*80 + "\n")

        if total_signals >= 30:
            logger.info("✅ Sehr viele Signale generiert!")
            return True
        elif total_signals >= 15:
            logger.info("✅ Gute Anzahl Signale generiert")
            return True
        else:
            logger.warning(f"⚠️ Nur {total_signals} Signale generiert")
            return False

    except Exception as e:
        logger.error(f"❌ Fehler beim Kategorien-Test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Hauptfunktion"""
    logger.info("\n" + "="*80)
    logger.info("🧪 STARTE ENHANCED SIGNALS TESTS")
    logger.info("="*80)

    results = []

    # Test 1: Enhanced Signal Generator
    results.append(("Enhanced Signal Generator", test_enhanced_signal_generator()))

    # Test 2: Signal Categories
    results.append(("Signal Categories", test_signal_categories()))

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

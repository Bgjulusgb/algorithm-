"""
Debug-Analyse für Trading Bot
Findet alle Mathe-Fehler und Berechnungsprobleme
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sma_calculation():
    """Testet SMA Berechnung"""
    logger.info("\n" + "="*80)
    logger.info("🔍 TESTE SMA BERECHNUNG")
    logger.info("="*80)

    issues = []

    try:
        from data_handler import DataHandler
        from config import STRATEGY_CONFIG

        # Erstelle Test-Daten
        dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
        test_data = pd.DataFrame({
            'Date': dates,
            'Open': 100 + np.random.randn(300).cumsum(),
            'High': 105 + np.random.randn(300).cumsum(),
            'Low': 95 + np.random.randn(300).cumsum(),
            'Close': 100 + np.random.randn(300).cumsum(),
            'Volume': np.random.randint(1000000, 10000000, 300)
        })
        test_data.set_index('Date', inplace=True)

        logger.info(f"✅ Test-Daten erstellt: {len(test_data)} Tage")

        # Teste DataHandler
        handler = DataHandler()

        # SMA 50 berechnen
        sma_50 = handler.calculate_sma(test_data, 50)
        logger.info(f"\n📊 SMA 50:")
        logger.info(f"   Länge: {len(sma_50)}")
        logger.info(f"   NaN Count: {sma_50.isna().sum()}")
        logger.info(f"   Letzte 5 Werte: {sma_50.tail()}")

        if sma_50.isna().all():
            issues.append("❌ SMA 50: Alle Werte sind NaN!")
        elif sma_50.iloc[-1] == 0 or pd.isna(sma_50.iloc[-1]):
            issues.append(f"❌ SMA 50: Letzter Wert ist {sma_50.iloc[-1]}")
        else:
            logger.info(f"   ✅ SMA 50 funktioniert: {sma_50.iloc[-1]:.2f}")

        # SMA 200 berechnen
        logger.info(f"\n📊 SMA 200 Config: {STRATEGY_CONFIG.get('long_window')}")
        sma_200 = handler.calculate_sma(test_data, STRATEGY_CONFIG.get('long_window', 200))
        logger.info(f"\n📊 SMA 200:")
        logger.info(f"   Länge: {len(sma_200)}")
        logger.info(f"   NaN Count: {sma_200.isna().sum()}")
        logger.info(f"   Erste gültige Position: {sma_200.first_valid_index()}")
        logger.info(f"   Letzte 5 Werte: {sma_200.tail()}")

        if sma_200.isna().all():
            issues.append("❌ SMA 200: Alle Werte sind NaN!")
            issues.append(f"   Grund: Datenlänge ({len(test_data)}) vs Window (200)")
        elif sma_200.iloc[-1] == 0 or pd.isna(sma_200.iloc[-1]):
            issues.append(f"❌ SMA 200: Letzter Wert ist {sma_200.iloc[-1]}")
        else:
            logger.info(f"   ✅ SMA 200 funktioniert: {sma_200.iloc[-1]:.2f}")

        # Teste add_technical_indicators
        df_with_indicators = handler.add_technical_indicators(test_data.copy())

        logger.info(f"\n📊 Indicators hinzugefügt:")
        logger.info(f"   Spalten: {df_with_indicators.columns.tolist()}")

        if 'SMA_200' in df_with_indicators.columns:
            sma_200_col = df_with_indicators['SMA_200']
            logger.info(f"\n   SMA_200 Spalte:")
            logger.info(f"      NaN Count: {sma_200_col.isna().sum()}/{len(sma_200_col)}")
            logger.info(f"      Letzte 5 Werte:\n{sma_200_col.tail()}")

            if sma_200_col.isna().all():
                issues.append("❌ SMA_200 Spalte: Alle Werte sind NaN!")
            elif pd.isna(sma_200_col.iloc[-1]):
                issues.append(f"❌ SMA_200 Spalte: Letzter Wert ist NaN (benötigt 200+ Datenpunkte)")
        else:
            issues.append("❌ SMA_200 Spalte wurde nicht hinzugefügt!")

    except Exception as e:
        issues.append(f"❌ Fehler bei SMA Test: {e}")
        import traceback
        traceback.print_exc()

    return issues


def test_confidence_calculation():
    """Testet Konfidenz-Berechnung"""
    logger.info("\n" + "="*80)
    logger.info("🔍 TESTE KONFIDENZ-BERECHNUNG")
    logger.info("="*80)

    issues = []

    try:
        from continuous_trading_bot import SignalGenerator

        # Test Signal Generator
        signal_gen = SignalGenerator(use_ml=False, use_advanced=False)

        # Mock Market Data mit verschiedenen Signalen
        test_cases = [
            {
                'name': 'STRONG BUY',
                'data': {
                    'recommendation': 'STRONG BUY',
                    'trend_alignment': {'alignment': 'STRONG_BULLISH'},
                    'confluence': {'overall_signal': 'STRONG_BUY'}
                },
                'expected_signal': 'BUY',
                'expected_confidence_min': 0.8
            },
            {
                'name': 'BUY',
                'data': {
                    'recommendation': 'BUY',
                    'trend_alignment': {'alignment': 'BULLISH'},
                    'confluence': {'overall_signal': 'BUY'}
                },
                'expected_signal': 'BUY',
                'expected_confidence_min': 0.6
            },
            {
                'name': 'HOLD',
                'data': {
                    'recommendation': 'HOLD',
                    'trend_alignment': {'alignment': 'NEUTRAL'},
                    'confluence': {'overall_signal': 'HOLD'}
                },
                'expected_signal': 'HOLD',
                'expected_confidence_min': 0.0
            },
            {
                'name': 'STRONG SELL',
                'data': {
                    'recommendation': 'STRONG SELL',
                    'trend_alignment': {'alignment': 'STRONG_BEARISH'},
                    'confluence': {'overall_signal': 'STRONG_SELL'}
                },
                'expected_signal': 'SELL',
                'expected_confidence_min': 0.8
            },
            {
                'name': 'EMPTY',
                'data': {},
                'expected_signal': 'HOLD',
                'expected_confidence_min': 0.0
            }
        ]

        for test_case in test_cases:
            logger.info(f"\n📊 Test Case: {test_case['name']}")

            signal = signal_gen.generate_signal('TEST', test_case['data'])

            logger.info(f"   Signal: {signal['signal']}")
            logger.info(f"   Confidence: {signal['confidence']:.2f}")
            logger.info(f"   Strength: {signal['strength']}")
            logger.info(f"   Raw Score: {signal['raw_score']:.2f}")
            logger.info(f"   Num Signals: {signal['num_signals']}")

            # Prüfe Signal
            if signal['signal'] != test_case['expected_signal']:
                issues.append(
                    f"❌ {test_case['name']}: Erwartetes Signal {test_case['expected_signal']}, "
                    f"bekommen {signal['signal']}"
                )

            # Prüfe Confidence
            if signal['confidence'] < test_case['expected_confidence_min']:
                issues.append(
                    f"❌ {test_case['name']}: Confidence zu niedrig "
                    f"({signal['confidence']:.2f} < {test_case['expected_confidence_min']})"
                )

            # Prüfe ob Confidence berechnet wurde
            if signal['confidence'] == 0.0 and test_case['name'] != 'EMPTY' and test_case['name'] != 'HOLD':
                issues.append(f"❌ {test_case['name']}: Confidence ist 0.0, sollte berechnet sein!")

            if signal['num_signals'] == 0 and test_case['name'] != 'EMPTY':
                issues.append(f"❌ {test_case['name']}: Keine Signale gezählt!")

        logger.info("\n✅ Konfidenz-Tests abgeschlossen")

    except Exception as e:
        issues.append(f"❌ Fehler bei Confidence Test: {e}")
        import traceback
        traceback.print_exc()

    return issues


def test_multi_timeframe_analysis():
    """Testet Multi-Timeframe Analyse"""
    logger.info("\n" + "="*80)
    logger.info("🔍 TESTE MULTI-TIMEFRAME ANALYSE")
    logger.info("="*80)

    issues = []

    try:
        from multi_timeframe import MultiTimeframeAnalyzer

        analyzer = MultiTimeframeAnalyzer(use_cache=False)

        logger.info("✅ MultiTimeframeAnalyzer initialisiert")

        # Teste mit echten Daten (nur wenn verfügbar)
        logger.info("\n📊 Teste Analyse mit AAPL...")

        result = analyzer.analyze_symbol(
            'AAPL',
            timeframes=['1d'],
            end_date=datetime.now().strftime('%Y-%m-%d')
        )

        logger.info(f"\n   Ergebnis Keys: {result.keys()}")

        if 'error' in result:
            logger.warning(f"   ⚠️ Fehler: {result['error']}")
            logger.info("   (Dies ist OK wenn keine Internetverbindung besteht)")
        else:
            logger.info(f"   Recommendation: {result.get('recommendation', 'N/A')}")
            logger.info(f"   Analyzed Timeframes: {len(result.get('analyzed_timeframes', {}))}")

            # Prüfe ob Confluence berechnet wurde
            if 'confluence' in result:
                confluence = result['confluence']
                logger.info(f"\n   Confluence:")
                logger.info(f"      Overall Signal: {confluence.get('overall_signal', 'N/A')}")
                logger.info(f"      Consensus Strength: {confluence.get('consensus_strength', 0):.2f}")

                if confluence.get('consensus_strength', 0) == 0:
                    issues.append("⚠️ Confluence: Consensus Strength ist 0")
            else:
                issues.append("❌ Confluence nicht berechnet!")

            # Prüfe Trend Alignment
            if 'trend_alignment' in result:
                alignment = result['trend_alignment']
                logger.info(f"\n   Trend Alignment:")
                logger.info(f"      Alignment: {alignment.get('alignment', 'N/A')}")
                logger.info(f"      Strength: {alignment.get('strength', 0):.2f}")
            else:
                issues.append("❌ Trend Alignment nicht berechnet!")

    except Exception as e:
        issues.append(f"❌ Fehler bei Multi-Timeframe Test: {e}")
        import traceback
        traceback.print_exc()

    return issues


def test_strategy_signals():
    """Testet Strategy Signal-Generierung"""
    logger.info("\n" + "="*80)
    logger.info("🔍 TESTE STRATEGY SIGNALS")
    logger.info("="*80)

    issues = []

    try:
        from strategy import TradingStrategy
        from data_handler import DataHandler

        # Erstelle Test-Daten mit Trend
        dates = pd.date_range(end=datetime.now(), periods=300, freq='D')

        # Aufwärtstrend
        base_price = 100
        prices = [base_price]
        for i in range(1, 300):
            prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.02)))

        test_data = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 300)
        })
        test_data.set_index('Date', inplace=True)

        logger.info(f"✅ Test-Daten erstellt: {len(test_data)} Tage")
        logger.info(f"   Start Preis: ${test_data['Close'].iloc[0]:.2f}")
        logger.info(f"   End Preis: ${test_data['Close'].iloc[-1]:.2f}")
        logger.info(f"   Trend: {(test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100:+.2f}%")

        # Füge Indikatoren hinzu
        handler = DataHandler()
        test_data = handler.add_technical_indicators(test_data)

        logger.info(f"\n📊 Indikatoren hinzugefügt:")
        logger.info(f"   Spalten: {test_data.columns.tolist()}")

        # Teste Strategy
        strategy = TradingStrategy()
        signals = strategy.generate_signals(test_data)

        logger.info(f"\n📊 Signale generiert:")
        logger.info(f"   Länge: {len(signals)}")
        logger.info(f"   BUY Signale: {(signals == 1).sum()}")
        logger.info(f"   SELL Signale: {(signals == -1).sum()}")
        logger.info(f"   HOLD Signale: {(signals == 0).sum()}")
        logger.info(f"   Letzte 10 Signale:\n{signals.tail(10)}")

        if (signals == 0).all():
            issues.append("❌ Strategy: Alle Signale sind 0 (HOLD)")

        # Teste ob Confidence berechnet wird
        if hasattr(strategy, 'calculate_signal_confidence'):
            for idx in test_data.index[-5:]:
                confidence = strategy.calculate_signal_confidence(test_data, idx)
                logger.info(f"   Confidence für {idx.date()}: {confidence:.2f}")

                if confidence == 0.0:
                    issues.append(f"⚠️ Confidence ist 0.0 für {idx.date()}")

    except Exception as e:
        issues.append(f"❌ Fehler bei Strategy Test: {e}")
        import traceback
        traceback.print_exc()

    return issues


def main():
    """Hauptfunktion"""
    logger.info("\n" + "="*80)
    logger.info("🐛 STARTE DEBUG-ANALYSE")
    logger.info("="*80)

    all_issues = []

    # Test 1: SMA Berechnung
    sma_issues = test_sma_calculation()
    all_issues.extend(sma_issues)

    # Test 2: Konfidenz-Berechnung
    conf_issues = test_confidence_calculation()
    all_issues.extend(conf_issues)

    # Test 3: Multi-Timeframe Analyse
    mtf_issues = test_multi_timeframe_analysis()
    all_issues.extend(mtf_issues)

    # Test 4: Strategy Signals
    strat_issues = test_strategy_signals()
    all_issues.extend(strat_issues)

    # Zusammenfassung
    logger.info("\n" + "="*80)
    logger.info("📋 DEBUG-ZUSAMMENFASSUNG")
    logger.info("="*80)

    if all_issues:
        logger.info(f"\n❌ GEFUNDENE PROBLEME ({len(all_issues)}):\n")
        for i, issue in enumerate(all_issues, 1):
            logger.info(f"{i}. {issue}")
    else:
        logger.info("\n✅ KEINE PROBLEME GEFUNDEN!")

    logger.info("\n" + "="*80 + "\n")

    return all_issues


if __name__ == "__main__":
    issues = main()

    if issues:
        import sys
        sys.exit(1)
    else:
        import sys
        sys.exit(0)

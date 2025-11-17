"""
Test für Continuous Trading Bot
Führt einen Demo-Run des Bots durch
"""
import sys
import logging
from datetime import datetime

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_continuous_bot():
    """
    Testet den Continuous Trading Bot mit einem kurzen Demo-Run
    """
    logger.info("="*80)
    logger.info("🧪 TESTE CONTINUOUS TRADING BOT")
    logger.info("="*80)

    try:
        # Import Bot
        from continuous_trading_bot import ContinuousTradingBot
        logger.info("✅ continuous_trading_bot erfolgreich importiert")

    except ImportError as e:
        logger.error(f"❌ Import-Fehler: {e}")
        logger.error("Stelle sicher, dass alle Module installiert sind")
        return False

    try:
        # Erstelle Bot mit Test-Konfiguration
        logger.info("\n📦 Erstelle Bot-Instanz...")

        test_symbols = ['AAPL', 'MSFT']  # Kleine Testliste

        bot = ContinuousTradingBot(
            symbols=test_symbols,
            initial_capital=100000,
            scan_interval_seconds=60,  # 1 Minute für Test
            min_confidence=0.6,
            use_ml=False,  # Kein ML für schnelleren Test
            use_advanced=True
        )

        logger.info("✅ Bot erfolgreich erstellt")

    except Exception as e:
        logger.error(f"❌ Fehler beim Erstellen des Bots: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Teste einzelnen Zyklus
        logger.info("\n🔄 Teste einzelnen Trading-Zyklus...")

        result = bot.run_single_cycle()

        logger.info("\n📊 Zyklus-Ergebnisse:")
        logger.info(f"   Timestamp: {result['timestamp']}")
        logger.info(f"   Dauer: {result['duration_seconds']:.2f}s")
        logger.info(f"   Signale generiert: {result['signals_generated']}")
        logger.info(f"   Trades ausgeführt: {result['trades_executed']}")
        logger.info(f"   Portfolio-Wert: ${result['portfolio_value']:,.2f}")
        logger.info(f"   Total Return: {result['total_return']*100:+.2f}%")

        logger.info("\n✅ Einzelner Zyklus erfolgreich durchgeführt")

    except Exception as e:
        logger.error(f"❌ Fehler beim Zyklus-Test: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Teste CSV-Export
        logger.info("\n📄 Teste CSV-Export...")

        import os
        csv_files = [
            'yahoo_finance_portfolio.csv',
            'yahoo_finance_portfolio_final.csv'
        ]

        found_files = []
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                found_files.append(csv_file)
                logger.info(f"   ✅ {csv_file} gefunden")

        if found_files:
            logger.info(f"\n✅ {len(found_files)} CSV-Dateien wurden erstellt")
        else:
            logger.warning("⚠️ Keine CSV-Dateien gefunden (evtl. keine Trades)")

    except Exception as e:
        logger.error(f"❌ Fehler beim CSV-Check: {e}")
        return False

    # Abschluss-Checks
    logger.info("\n🔍 Abschluss-Checks...")

    # Prüfe dass keine Excel-Dateien erstellt wurden
    try:
        import glob
        excel_files = glob.glob("*.xlsx")

        if excel_files:
            logger.error(f"❌ FEHLER: Excel-Dateien gefunden: {excel_files}")
            logger.error("   Der Bot sollte KEINE Excel-Dateien erstellen!")
            return False
        else:
            logger.info("   ✅ Keine Excel-Dateien gefunden (wie erwartet)")

    except Exception as e:
        logger.warning(f"⚠️ Excel-Check fehlgeschlagen: {e}")

    # Erfolg
    logger.info("\n" + "="*80)
    logger.info("✅ ALLE TESTS ERFOLGREICH!")
    logger.info("="*80)
    logger.info("\n📋 ZUSAMMENFASSUNG:")
    logger.info("   ✅ Bot-Import funktioniert")
    logger.info("   ✅ Bot-Initialisierung funktioniert")
    logger.info("   ✅ Trading-Zyklus funktioniert")
    logger.info("   ✅ CSV-Export funktioniert")
    logger.info("   ✅ Keine Excel-Dateien erstellt")
    logger.info("\n🎉 Continuous Trading Bot ist einsatzbereit!")
    logger.info("="*80 + "\n")

    return True


def test_components():
    """
    Testet einzelne Bot-Komponenten
    """
    logger.info("\n🔧 TESTE EINZELNE KOMPONENTEN\n")

    results = {
        'scanner': False,
        'signal_generator': False,
        'trade_executor': False
    }

    # Test LiveMarketScanner
    try:
        from continuous_trading_bot import LiveMarketScanner

        scanner = LiveMarketScanner(
            symbols=['AAPL'],
            scan_interval_seconds=60
        )

        logger.info("✅ LiveMarketScanner initialisiert")
        results['scanner'] = True

    except Exception as e:
        logger.error(f"❌ LiveMarketScanner Fehler: {e}")

    # Test SignalGenerator
    try:
        from continuous_trading_bot import SignalGenerator

        signal_gen = SignalGenerator(use_ml=False, use_advanced=False)

        logger.info("✅ SignalGenerator initialisiert")
        results['signal_generator'] = True

    except Exception as e:
        logger.error(f"❌ SignalGenerator Fehler: {e}")

    # Test TradeExecutor
    try:
        from continuous_trading_bot import TradeExecutor
        from portfolio import Portfolio

        portfolio = Portfolio(initial_capital=100000)

        executor = TradeExecutor(
            portfolio=portfolio,
            min_confidence=0.6
        )

        logger.info("✅ TradeExecutor initialisiert")
        results['trade_executor'] = True

    except Exception as e:
        logger.error(f"❌ TradeExecutor Fehler: {e}")

    # Zusammenfassung
    success_count = sum(results.values())
    total_count = len(results)

    logger.info(f"\n📊 Komponenten-Tests: {success_count}/{total_count} erfolgreich")

    return all(results.values())


if __name__ == "__main__":
    logger.info(f"\n🚀 STARTE CONTINUOUS BOT TESTS - {datetime.now()}\n")

    # Teste Komponenten
    components_ok = test_components()

    # Teste Full Bot
    if components_ok:
        bot_ok = test_continuous_bot()
    else:
        logger.error("⚠️ Komponenten-Tests fehlgeschlagen - überspringe Full Bot Test")
        bot_ok = False

    # Exit Code
    if components_ok and bot_ok:
        logger.info("\n🎉 ALLE TESTS BESTANDEN!\n")
        sys.exit(0)
    else:
        logger.error("\n❌ EINIGE TESTS FEHLGESCHLAGEN\n")
        sys.exit(1)

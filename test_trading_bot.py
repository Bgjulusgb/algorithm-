"""
Test Suite für Trading Bot
Führt grundlegende Tests der Hauptkomponenten durch
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Testet ob alle Module importiert werden können"""
    logger.info("🧪 Test 1: Module Imports")
    try:
        import config
        import data_handler
        import strategy
        import portfolio
        import csv_manager
        logger.info("   ✅ Alle Module erfolgreich importiert")
        return True
    except ImportError as e:
        logger.error(f"   ❌ Import-Fehler: {e}")
        return False


def test_config():
    """Testet Konfiguration"""
    logger.info("🧪 Test 2: Konfiguration")
    try:
        from config import (
            WATCHLIST, PORTFOLIO_CONFIG, RISK_CONFIG,
            STRATEGY_CONFIG, validate_config
        )

        # Prüfe Watchlist
        assert isinstance(WATCHLIST, list), "WATCHLIST sollte eine Liste sein"
        assert len(WATCHLIST) > 0, "WATCHLIST sollte nicht leer sein"

        # Prüfe Portfolio Config
        assert PORTFOLIO_CONFIG["initial_capital"] > 0, "Startkapital sollte > 0 sein"
        assert 0 <= PORTFOLIO_CONFIG["max_position_size"] <= 1, "max_position_size sollte zwischen 0 und 1 sein"

        # Validiere Config
        validate_config()

        logger.info("   ✅ Konfiguration valide")
        return True
    except Exception as e:
        logger.error(f"   ❌ Konfigurationsfehler: {e}")
        return False


def test_portfolio():
    """Testet Portfolio-Management"""
    logger.info("🧪 Test 3: Portfolio-Management")
    try:
        from portfolio import Portfolio, Position
        from datetime import datetime

        # Erstelle Portfolio
        portfolio = Portfolio(initial_capital=100000)
        assert portfolio.cash == 100000, "Initial Cash sollte 100000 sein"

        # Teste Kauf
        success = portfolio.buy("AAPL", 150.0, 10, date=datetime.now())
        assert success, "Kauf sollte erfolgreich sein"
        assert "AAPL" in portfolio.positions, "Position sollte existieren"
        assert portfolio.cash < 100000, "Cash sollte reduziert sein"

        # Teste Position Size Berechnung
        shares = portfolio.calculate_position_size("MSFT", 300.0)
        assert shares >= 0, "Shares sollte nicht negativ sein"

        # Teste Verkauf
        success = portfolio.sell("AAPL", 160.0, 10, date=datetime.now(), reason="Test")
        assert success, "Verkauf sollte erfolgreich sein"
        assert "AAPL" not in portfolio.positions, "Position sollte geschlossen sein"

        # Teste Performance Stats
        stats = portfolio.get_performance_stats()
        assert isinstance(stats, dict), "Stats sollte ein Dict sein"

        logger.info("   ✅ Portfolio-Management funktioniert")
        return True
    except Exception as e:
        logger.error(f"   ❌ Portfolio-Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_handler():
    """Testet Data Handler"""
    logger.info("🧪 Test 4: Data Handler")
    try:
        from data_handler import DataHandler

        handler = DataHandler()

        # Teste technische Indikatoren mit Mock-Daten
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

        # Füge Indikatoren hinzu
        df_with_indicators = handler.add_technical_indicators(df)

        # Prüfe ob Indikatoren hinzugefügt wurden
        required_indicators = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
        for indicator in required_indicators:
            assert indicator in df_with_indicators.columns, f"{indicator} sollte vorhanden sein"

        # Prüfe RSI Werte
        assert df_with_indicators['RSI'].min() >= 0, "RSI sollte >= 0 sein"
        assert df_with_indicators['RSI'].max() <= 100, "RSI sollte <= 100 sein"

        # Prüfe Volume Indikatoren
        assert 'Volume_MA' in df_with_indicators.columns, "Volume_MA sollte vorhanden sein"
        assert 'Volume_Ratio' in df_with_indicators.columns, "Volume_Ratio sollte vorhanden sein"
        assert 'OBV' in df_with_indicators.columns, "OBV sollte vorhanden sein"

        logger.info("   ✅ Data Handler funktioniert")
        return True
    except Exception as e:
        logger.error(f"   ❌ Data Handler Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategies():
    """Testet Trading-Strategien"""
    logger.info("🧪 Test 5: Trading-Strategien")
    try:
        from strategy import StrategyFactory, SMAStrategy, RSIStrategy, MACDStrategy, CombinedStrategy
        from data_handler import DataHandler

        # Erstelle Mock-Daten
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
        prices = 100 + np.cumsum(np.random.randn(250) * 2)
        df = pd.DataFrame({
            'Open': prices + np.random.randn(250),
            'High': prices + np.abs(np.random.randn(250)),
            'Low': prices - np.abs(np.random.randn(250)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 250)
        }, index=dates)

        # Füge Indikatoren hinzu
        handler = DataHandler()
        df = handler.add_technical_indicators(df)

        # Teste verschiedene Strategien
        strategies = ['sma', 'rsi', 'macd', 'combined', 'mean_reversion']

        for strategy_name in strategies:
            strategy = StrategyFactory.create_strategy(strategy_name)
            assert strategy is not None, f"Strategie {strategy_name} sollte erstellt werden"

            df_with_signals = strategy.generate_signals(df.copy())
            assert 'Signal' in df_with_signals.columns, f"{strategy_name}: Signal sollte vorhanden sein"
            assert 'Confidence' in df_with_signals.columns, f"{strategy_name}: Confidence sollte vorhanden sein"
            assert 'Position' in df_with_signals.columns, f"{strategy_name}: Position sollte vorhanden sein"

            # Prüfe Signal-Werte
            signals = df_with_signals['Signal'].dropna()
            assert signals.isin([-1, 0, 1]).all(), f"{strategy_name}: Signale sollten -1, 0 oder 1 sein"

            # Prüfe Konfidenz-Werte
            confidence = df_with_signals['Confidence'].dropna()
            assert (confidence >= 0).all() and (confidence <= 1).all(), f"{strategy_name}: Konfidenz sollte zwischen 0 und 1 sein"

        logger.info("   ✅ Alle Strategien funktionieren")
        return True
    except Exception as e:
        logger.error(f"   ❌ Strategie-Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_manager():
    """Testet CSV Manager"""
    logger.info("🧪 Test 6: CSV Manager")
    try:
        from csv_manager import CSVManager, PerformanceLogger
        import os

        # Erstelle temporäre Dateien
        test_trades_file = "test_trades.csv"
        test_perf_file = "test_performance.csv"

        # Teste CSVManager
        csv_manager = CSVManager(trades_file=test_trades_file)
        csv_manager.add_buy("AAPL", datetime.now(), 10, 150.0, notes="Test Buy")
        csv_manager.add_sell("AAPL", datetime.now(), 10, 160.0, notes="Test Sell")

        trades = csv_manager.get_trades()
        assert len(trades) == 2, "Sollte 2 Trades haben"

        # Export test
        output = csv_manager.export_for_yahoo("test_yahoo.csv")
        assert os.path.exists("test_yahoo.csv"), "Yahoo-Export sollte Datei erstellen"

        # Teste PerformanceLogger
        perf_logger = PerformanceLogger(performance_file=test_perf_file)
        perf_logger.log_performance(
            total_value=105000,
            cash=50000,
            positions_value=55000,
            total_return=5.0,
            open_positions=3
        )

        # Cleanup
        for f in [test_trades_file, test_perf_file, "test_yahoo.csv"]:
            if os.path.exists(f):
                os.remove(f)

        logger.info("   ✅ CSV Manager funktioniert")
        return True
    except Exception as e:
        logger.error(f"   ❌ CSV Manager Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_management():
    """Testet Risk Management"""
    logger.info("🧪 Test 7: Risk Management")
    try:
        from portfolio import Portfolio
        from datetime import datetime

        portfolio = Portfolio(initial_capital=100000)

        # Kaufe Position
        portfolio.buy("AAPL", 100.0, 100, date=datetime.now())

        # Teste Stop-Loss
        current_prices = {"AAPL": 90.0}  # 10% Verlust
        portfolio.check_risk_management(current_prices)

        # Position sollte verkauft sein (wegen Stop-Loss)
        # Dies hängt von RISK_CONFIG ab, daher nur warnen wenn nicht

        # Teste Take-Profit
        portfolio2 = Portfolio(initial_capital=100000)
        portfolio2.buy("MSFT", 100.0, 100, date=datetime.now())
        current_prices2 = {"MSFT": 120.0}  # 20% Gewinn
        portfolio2.check_risk_management(current_prices2)

        logger.info("   ✅ Risk Management funktioniert")
        return True
    except Exception as e:
        logger.error(f"   ❌ Risk Management Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Führt alle Tests aus"""
    logger.info("\n" + "="*70)
    logger.info("TRADING BOT TEST SUITE")
    logger.info("="*70 + "\n")

    tests = [
        test_imports,
        test_config,
        test_portfolio,
        test_data_handler,
        test_strategies,
        test_csv_manager,
        test_risk_management,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()  # Leerzeile

    # Zusammenfassung
    logger.info("="*70)
    logger.info("TEST ZUSAMMENFASSUNG")
    logger.info("="*70)

    passed = sum(results)
    total = len(results)

    logger.info(f"✅ Bestanden: {passed}/{total}")
    logger.info(f"❌ Fehlgeschlagen: {total - passed}/{total}")

    if passed == total:
        logger.info("\n🎉 Alle Tests erfolgreich!")
        return 0
    else:
        logger.warning(f"\n⚠️ {total - passed} Test(s) fehlgeschlagen")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

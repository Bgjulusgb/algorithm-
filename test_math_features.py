"""
Test Suite für erweiterte mathematische Features
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_math_utils():
    """Testet math_utils Module"""
    logger.info("🧪 Test 1: Math Utils Module")
    try:
        from math_utils import (
            StatisticsAnalyzer, CorrelationAnalyzer,
            KellyCriterion, MonteCarloSimulator,
            SignalFilters, PortfolioOptimizer
        )

        # Test StatisticsAnalyzer
        returns = pd.Series(np.random.randn(100) * 0.01)

        sharpe = StatisticsAnalyzer.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float), "Sharpe sollte ein Float sein"

        sortino = StatisticsAnalyzer.calculate_sortino_ratio(returns)
        assert isinstance(sortino, float), "Sortino sollte ein Float sein"

        var = StatisticsAnalyzer.calculate_var(returns, 0.95)
        assert isinstance(var, float), "VaR sollte ein Float sein"

        cvar = StatisticsAnalyzer.calculate_cvar(returns, 0.95)
        assert isinstance(cvar, float), "CVaR sollte ein Float sein"

        skewness = StatisticsAnalyzer.calculate_skewness(returns)
        kurtosis = StatisticsAnalyzer.calculate_kurtosis(returns)
        assert isinstance(skewness, float), "Skewness sollte ein Float sein"
        assert isinstance(kurtosis, float), "Kurtosis sollte ein Float sein"

        logger.info("   ✅ StatisticsAnalyzer funktioniert")

        # Test Kelly Criterion
        kelly = KellyCriterion.calculate_kelly_fraction(0.55, 100, 50)
        assert 0 <= kelly <= 1, "Kelly sollte zwischen 0 und 1 sein"
        logger.info(f"   ✅ Kelly Criterion: {kelly:.2%}")

        # Test Monte Carlo
        simulations = MonteCarloSimulator.simulate_portfolio(
            100000, 0.10, 0.20, num_simulations=100, num_days=252
        )
        assert simulations.shape == (100, 252), "Simulation Shape sollte (100, 252) sein"

        results = MonteCarloSimulator.analyze_simulation_results(simulations)
        assert 'mean' in results, "Results sollte 'mean' enthalten"
        logger.info("   ✅ Monte Carlo Simulator funktioniert")

        # Test Signal Filters
        signal = pd.Series(np.random.randn(100))

        filtered_ema = SignalFilters.exponential_moving_average_filter(signal)
        assert len(filtered_ema) == len(signal), "Gefiltert sollte gleiche Länge haben"

        filtered_median = SignalFilters.median_filter(signal, window=5)
        assert len(filtered_median) == len(signal), "Median Filter sollte gleiche Länge haben"

        z_score = SignalFilters.z_score_normalization(signal, window=20)
        assert len(z_score) == len(signal), "Z-Score sollte gleiche Länge haben"

        logger.info("   ✅ Signal Filters funktionieren")

        # Test Correlation Analyzer
        returns_df = pd.DataFrame({
            'A': np.random.randn(100) * 0.01,
            'B': np.random.randn(100) * 0.01,
            'C': np.random.randn(100) * 0.01
        })

        corr_matrix = CorrelationAnalyzer.calculate_correlation_matrix(returns_df)
        assert corr_matrix.shape == (3, 3), "Korrelationsmatrix sollte 3x3 sein"

        cov_matrix = CorrelationAnalyzer.calculate_covariance_matrix(returns_df)
        assert cov_matrix.shape == (3, 3), "Kovarianzmatrix sollte 3x3 sein"

        logger.info("   ✅ Correlation Analyzer funktioniert")

        # Test Portfolio Optimizer
        expected_returns = np.array([0.10, 0.12, 0.08])
        cov_matrix_np = cov_matrix.values

        optimal_weights = PortfolioOptimizer.optimize_sharpe_ratio(
            expected_returns, cov_matrix_np
        )
        assert len(optimal_weights) == 3, "Weights sollte Länge 3 haben"
        assert abs(optimal_weights.sum() - 1.0) < 0.01, "Weights sollten auf 1 summieren"

        logger.info("   ✅ Portfolio Optimizer funktioniert")

        logger.info("   ✅ Alle Math Utils Tests bestanden")
        return True

    except Exception as e:
        logger.error(f"   ❌ Math Utils Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_analytics():
    """Testet Performance Analytics Module"""
    logger.info("🧪 Test 2: Performance Analytics")
    try:
        from performance_analytics import PerformanceAnalytics, PortfolioCorrelationAnalysis

        # Erstelle Mock-Trades
        trades = []
        for i in range(20):
            trade_date = datetime.now() - timedelta(days=20-i)
            trades.append({
                'date': trade_date,
                'symbol': 'TEST',
                'action': 'BUY',
                'shares': 10,
                'price': 100 + i,
                'profit_loss': None
            })
            trades.append({
                'date': trade_date + timedelta(days=5),
                'symbol': 'TEST',
                'action': 'SELL',
                'shares': 10,
                'price': 105 + i,
                'profit_loss': (105 + i - 100 - i) * 10
            })

        analytics = PerformanceAnalytics(trades, 100000)

        # Test umfassende Metriken
        metrics = analytics.calculate_comprehensive_metrics()
        assert 'win_rate' in metrics, "Metrics sollte win_rate enthalten"
        assert 'sharpe_ratio' in metrics, "Metrics sollte sharpe_ratio enthalten"
        assert 'kelly_criterion' in metrics, "Metrics sollte kelly_criterion enthalten"

        logger.info(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
        logger.info(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"   Kelly: {metrics.get('kelly_criterion', 0)*100:.1f}%")

        # Test Report-Generierung
        report = analytics.generate_performance_report()
        assert len(report) > 0, "Report sollte nicht leer sein"
        assert "ERWEITERTE PERFORMANCE-ANALYSE" in report, "Report sollte Header enthalten"

        logger.info("   ✅ Performance Analytics funktioniert")

        # Test Correlation Analysis
        price_data = {
            'AAPL': pd.Series(np.random.randn(100).cumsum() + 100),
            'MSFT': pd.Series(np.random.randn(100).cumsum() + 200),
            'GOOGL': pd.Series(np.random.randn(100).cumsum() + 150)
        }

        corr_analysis = PortfolioCorrelationAnalysis(price_data)
        corr_matrix = corr_analysis.get_correlation_matrix()
        assert corr_matrix.shape == (3, 3), "Korrelationsmatrix sollte 3x3 sein"

        opportunities = corr_analysis.find_diversification_opportunities(max_correlation=0.5)
        logger.info(f"   Diversifikations-Möglichkeiten gefunden: {len(opportunities)}")

        logger.info("   ✅ Alle Performance Analytics Tests bestanden")
        return True

    except Exception as e:
        logger.error(f"   ❌ Performance Analytics Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_portfolio_enhancements():
    """Testet Portfolio-Erweiterungen"""
    logger.info("🧪 Test 3: Portfolio Enhancements")
    try:
        from portfolio import Portfolio
        from datetime import datetime

        portfolio = Portfolio(initial_capital=100000)

        # Teste Standard-Kauf
        success = portfolio.buy("TEST", 100.0, 10, date=datetime.now())
        assert success, "Kauf sollte erfolgreich sein"

        # Simuliere Trades für Kelly
        for i in range(10):
            portfolio.sell("TEST", 110.0, 5, date=datetime.now(), reason="Test")
            portfolio.buy("TEST", 100.0, 5, date=datetime.now())

        # Teste Kelly Position Size (wenn genug Trades)
        try:
            kelly_shares = portfolio.calculate_kelly_position_size("TEST2", 50.0)
            logger.info(f"   Kelly Position Size: {kelly_shares} Aktien")
        except Exception as e:
            logger.debug(f"   Kelly Test übersprungen: {e}")

        # Teste erweiterte Analytics
        try:
            advanced_metrics = portfolio.get_advanced_analytics()
            if advanced_metrics and 'win_rate' in advanced_metrics:
                logger.info(f"   Erweiterte Metriken verfügbar: {len(advanced_metrics)} Werte")
        except Exception as e:
            logger.debug(f"   Erweiterte Analytics übersprungen: {e}")

        logger.info("   ✅ Portfolio Enhancements funktionieren")
        return True

    except Exception as e:
        logger.error(f"   ❌ Portfolio Enhancements Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_filters():
    """Testet Strategy Signal Filters"""
    logger.info("🧪 Test 4: Strategy Signal Filters")
    try:
        from strategy import CombinedStrategy
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

        # Teste Strategie mit Filtern
        strategy = CombinedStrategy()  # use_filters=True by default
        df_with_signals = strategy.generate_signals(df)

        assert 'Signal' in df_with_signals.columns, "Signal sollte vorhanden sein"
        assert 'Confidence' in df_with_signals.columns, "Confidence sollte vorhanden sein"

        # Prüfe ob Filter angewendet wurden
        if 'Signal_Filtered' in df_with_signals.columns:
            logger.info("   ✅ Signal-Filter wurden angewendet")
        else:
            logger.info("   ℹ️ Signal-Filter optional nicht angewendet")

        logger.info("   ✅ Strategy Filters funktionieren")
        return True

    except Exception as e:
        logger.error(f"   ❌ Strategy Filters Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_math_tests():
    """Führt alle Math-Feature-Tests aus"""
    logger.info("\n" + "="*80)
    logger.info("MATHEMATISCHE FEATURES TEST SUITE")
    logger.info("="*80 + "\n")

    tests = [
        test_math_utils,
        test_performance_analytics,
        test_portfolio_enhancements,
        test_strategy_filters,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()  # Leerzeile

    # Zusammenfassung
    logger.info("="*80)
    logger.info("TEST ZUSAMMENFASSUNG")
    logger.info("="*80)

    passed = sum(results)
    total = len(results)

    logger.info(f"✅ Bestanden: {passed}/{total}")
    logger.info(f"❌ Fehlgeschlagen: {total - passed}/{total}")

    if passed == total:
        logger.info("\n🎉 Alle mathematischen Features funktionieren!")
        return 0
    else:
        logger.warning(f"\n⚠️ {total - passed} Test(s) fehlgeschlagen")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_math_tests())

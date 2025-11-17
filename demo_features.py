"""
Demonstration aller neuen Trading Bot Features
Zeigt die Verwendung von Math Utils, Analytics, Visualization und Validation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_math_utils():
    """Demonstriert Math Utils Module"""
    print("\n" + "="*80)
    print("DEMO 1: MATH UTILS MODULE")
    print("="*80 + "\n")

    from math_utils import (
        StatisticsAnalyzer, KellyCriterion, MonteCarloSimulator,
        SignalFilters, PortfolioOptimizer, CorrelationAnalyzer
    )

    # 1. Statistics Analyzer
    print("📊 Statistics Analyzer:")
    returns = pd.Series(np.random.randn(252) * 0.02)  # 1 Jahr Daily Returns

    sharpe = StatisticsAnalyzer.calculate_sharpe_ratio(returns)
    sortino = StatisticsAnalyzer.calculate_sortino_ratio(returns)
    var_95 = StatisticsAnalyzer.calculate_var(returns, 0.95)
    cvar_95 = StatisticsAnalyzer.calculate_cvar(returns, 0.95)

    print(f"   Sharpe Ratio: {sharpe:.3f}")
    print(f"   Sortino Ratio: {sortino:.3f}")
    print(f"   VaR (95%): {var_95*100:.2f}%")
    print(f"   CVaR (95%): {cvar_95*100:.2f}%")

    # 2. Kelly Criterion
    print("\n💰 Kelly Criterion:")
    kelly = KellyCriterion.calculate_kelly_fraction(
        win_rate=0.55,
        avg_win=100,
        avg_loss=50,
        kelly_fraction=0.5  # Half-Kelly
    )
    print(f"   Optimale Position Size: {kelly*100:.2f}% des Kapitals")

    # 3. Monte Carlo Simulation
    print("\n🎲 Monte Carlo Simulation:")
    simulations = MonteCarloSimulator.simulate_portfolio(
        initial_capital=100000,
        expected_return=0.10,
        volatility=0.20,
        num_simulations=100,  # Reduziert für Demo
        num_days=252
    )

    results = MonteCarloSimulator.analyze_simulation_results(simulations)
    print(f"   Mean Final Value: ${results['mean']:,.0f}")
    print(f"   5th Percentile: ${results['5th_percentile']:,.0f}")
    print(f"   95th Percentile: ${results['95th_percentile']:,.0f}")
    print(f"   Probability of Profit: {results['probability_of_profit']*100:.1f}%")

    # 4. Signal Filters
    print("\n🔍 Signal Filters:")
    noisy_signal = pd.Series([1, 1, 0, 1, 1, 0, 0, 1, 1, 1])
    filtered = SignalFilters.exponential_moving_average_filter(noisy_signal, span=3)
    print(f"   Original Signal:  {noisy_signal.values}")
    print(f"   Filtered Signal:  {filtered.round(2).values}")

    # 5. Portfolio Optimizer
    print("\n🎯 Portfolio Optimizer:")
    expected_returns = np.array([0.10, 0.12, 0.08])
    cov_matrix = np.array([
        [0.04, 0.01, 0.02],
        [0.01, 0.06, 0.015],
        [0.02, 0.015, 0.05]
    ])

    optimal_weights = PortfolioOptimizer.optimize_sharpe_ratio(
        expected_returns, cov_matrix
    )
    print(f"   Optimale Gewichte: {(optimal_weights*100).round(1)}%")

    print("\n✅ Math Utils Demo abgeschlossen\n")


def demo_performance_analytics():
    """Demonstriert Performance Analytics"""
    print("="*80)
    print("DEMO 2: PERFORMANCE ANALYTICS")
    print("="*80 + "\n")

    from performance_analytics import PerformanceAnalytics

    # Mock Trades erstellen
    trades = []
    current_date = datetime.now() - timedelta(days=365)

    for i in range(20):
        buy_date = current_date + timedelta(days=i*15)
        sell_date = buy_date + timedelta(days=10)

        entry_price = 100 + np.random.randn() * 5
        exit_price = entry_price + np.random.randn() * 10
        shares = 10

        profit_loss = (exit_price - entry_price) * shares

        trades.append({
            'date': buy_date,
            'symbol': 'TEST',
            'action': 'BUY',
            'shares': shares,
            'price': entry_price,
            'profit_loss': None
        })

        trades.append({
            'date': sell_date,
            'symbol': 'TEST',
            'action': 'SELL',
            'shares': shares,
            'price': exit_price,
            'profit_loss': profit_loss
        })

    # Analytics berechnen
    analytics = PerformanceAnalytics(trades, initial_capital=100000)
    metrics = analytics.calculate_comprehensive_metrics()

    print("📈 Umfassende Performance Metriken:")
    print(f"\n   Basis-Metriken:")
    print(f"   - Total Return: {metrics.get('total_return', 0)*100:.2f}%")
    print(f"   - Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"   - Profit Factor: {metrics.get('profit_factor', 0):.2f}")

    print(f"\n   Risiko-Metriken:")
    print(f"   - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"   - Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
    print(f"   - Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")

    print(f"\n   Trade-Metriken:")
    print(f"   - Expectancy: ${metrics.get('expectancy', 0):.2f}")
    print(f"   - Kelly Criterion: {metrics.get('kelly_criterion', 0)*100:.2f}%")
    print(f"   - Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}")

    # Report generieren
    print("\n📄 Generiere Performance Report...")
    report = analytics.generate_performance_report()
    report_lines = report.split('\n')
    print(f"   Report generiert: {len(report_lines)} Zeilen")

    print("\n✅ Performance Analytics Demo abgeschlossen\n")


def demo_visualization():
    """Demonstriert Visualization Features"""
    print("="*80)
    print("DEMO 3: VISUALIZATION")
    print("="*80 + "\n")

    from visualization import TradingVisualizer

    # Mock Daten erstellen
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(252) * 2)

    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(252),
        'High': close_prices + np.abs(np.random.randn(252)),
        'Low': close_prices - np.abs(np.random.randn(252)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 252),
        'SMA_50': pd.Series(close_prices).rolling(50).mean(),
        'SMA_200': pd.Series(close_prices).rolling(200).mean(),
        'RSI': 50 + np.random.randn(252) * 20,
        'Signal': np.random.choice([0, 1, -1], 252)
    }, index=dates)

    visualizer = TradingVisualizer(output_dir="demo_charts")

    print("📊 Erstelle Visualisierungen...")

    # 1. Price Chart
    print("   1. Price Chart mit Signalen...")
    try:
        chart1 = visualizer.plot_price_with_signals(df, "DEMO", save=True)
        print(f"      ✅ Gespeichert: {chart1}")
    except Exception as e:
        print(f"      ⚠️ Übersprungen: {e}")

    # 2. Portfolio Performance
    print("   2. Portfolio Performance Chart...")
    try:
        equity_curve = pd.Series(
            100000 + np.cumsum(np.random.randn(252) * 500),
            index=dates
        )
        trades = [{'date': dates[i], 'profit_loss': np.random.randn() * 100}
                 for i in range(0, 252, 20)]

        chart2 = visualizer.plot_portfolio_performance(equity_curve, trades, 100000)
        print(f"      ✅ Gespeichert: {chart2}")
    except Exception as e:
        print(f"      ⚠️ Übersprungen: {e}")

    # 3. Correlation Matrix
    print("   3. Korrelationsmatrix...")
    try:
        corr_matrix = pd.DataFrame(
            np.random.rand(5, 5),
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        )
        # Symmetrisch machen
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix.values, 1.0)

        chart3 = visualizer.plot_correlation_matrix(corr_matrix)
        print(f"      ✅ Gespeichert: {chart3}")
    except Exception as e:
        print(f"      ⚠️ Übersprungen: {e}")

    print("\n✅ Visualization Demo abgeschlossen\n")


def demo_validation():
    """Demonstriert Data Validation"""
    print("="*80)
    print("DEMO 4: DATA VALIDATION")
    print("="*80 + "\n")

    from data_validation import DataValidator, TradeValidator, ConfigValidator

    # 1. Price Data Validation
    print("🔍 Data Validator:")

    # Gute Daten
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    good_data = pd.DataFrame({
        'Open': np.random.uniform(95, 105, 100),
        'High': np.random.uniform(100, 110, 100),
        'Low': np.random.uniform(90, 100, 100),
        'Close': np.random.uniform(95, 105, 100),
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    is_valid, errors = DataValidator.validate_price_data(good_data, "TEST")
    print(f"   Gute Daten: {'✅ Valid' if is_valid else '❌ Invalid'}")
    if errors:
        print(f"   Warnungen: {len(errors)}")

    # Schlechte Daten
    bad_data = good_data.copy()
    bad_data.iloc[10, bad_data.columns.get_loc('Close')] = -50  # Negativer Preis
    bad_data.iloc[20, bad_data.columns.get_loc('High')] = 50    # High < Low

    is_valid, errors = DataValidator.validate_price_data(bad_data, "TEST")
    print(f"   Schlechte Daten: {'✅ Valid' if is_valid else '❌ Invalid'}")
    print(f"   Fehler gefunden: {len(errors)}")
    for error in errors[:3]:
        print(f"      - {error}")

    # Data Cleaning
    print("\n🧹 Data Cleaning:")
    cleaned = DataValidator.clean_price_data(bad_data, "TEST")
    print(f"   Original Zeilen: {len(bad_data)}")
    print(f"   Bereinigte Zeilen: {len(cleaned)}")
    print(f"   Entfernt: {len(bad_data) - len(cleaned)}")

    # 2. Trade Validation
    print("\n💼 Trade Validator:")

    is_valid, error = TradeValidator.validate_trade(
        action='BUY',
        symbol='AAPL',
        price=150.0,
        shares=10,
        cash_available=2000.0
    )
    print(f"   Valid Trade: {'✅' if is_valid else '❌'}")

    is_valid, error = TradeValidator.validate_trade(
        action='BUY',
        symbol='AAPL',
        price=150.0,
        shares=100,  # Zu viele Shares
        cash_available=2000.0
    )
    print(f"   Invalid Trade: {'✅ Korrekt erkannt' if not is_valid else '❌'}")
    print(f"   Fehler: {error}")

    # 3. Config Validation
    print("\n⚙️ Config Validator:")

    from config import PORTFOLIO_CONFIG, RISK_CONFIG, STRATEGY_CONFIG, WATCHLIST

    test_config = {
        'PORTFOLIO_CONFIG': PORTFOLIO_CONFIG,
        'RISK_CONFIG': RISK_CONFIG,
        'STRATEGY_CONFIG': STRATEGY_CONFIG,
        'WATCHLIST': WATCHLIST
    }

    is_valid, errors = ConfigValidator.validate_config(test_config)
    print(f"   Config Valid: {'✅' if is_valid else '❌'}")
    if errors:
        print(f"   Fehler: {errors}")

    print("\n✅ Validation Demo abgeschlossen\n")


def demo_portfolio_features():
    """Demonstriert Portfolio Features"""
    print("="*80)
    print("DEMO 5: PORTFOLIO FEATURES")
    print("="*80 + "\n")

    from portfolio import Portfolio

    portfolio = Portfolio(initial_capital=100000)

    # Standard Buy
    print("💼 Standard Portfolio Operations:")
    portfolio.buy("AAPL", 150.0, 10, date=datetime.now())
    portfolio.buy("MSFT", 300.0, 5, date=datetime.now())

    # Simuliere einige Trades für Kelly
    for i in range(10):
        portfolio.sell("AAPL", 160.0 + i, 5, date=datetime.now(), reason="Test")
        portfolio.buy("AAPL", 150.0, 5, date=datetime.now())

    # Kelly Position Size
    print("\n🎯 Kelly Criterion Position Sizing:")
    try:
        kelly_shares = portfolio.calculate_kelly_position_size("GOOGL", 120.0)
        print(f"   Standard Position: ~100 Shares")
        print(f"   Kelly Position: {kelly_shares} Shares")
        print(f"   Differenz: {kelly_shares - 100:+d} Shares")
    except Exception as e:
        print(f"   ⚠️ Kelly berechnung übersprungen: {e}")

    # Advanced Analytics
    print("\n📊 Advanced Analytics:")
    try:
        metrics = portfolio.get_advanced_analytics()
        if metrics:
            print(f"   Metriken verfügbar: {len(metrics)}")
            print(f"   - Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"   - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"   - Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
    except Exception as e:
        print(f"   ⚠️ Analytics übersprungen: {e}")

    print("\n✅ Portfolio Features Demo abgeschlossen\n")


def main():
    """Hauptfunktion - Führt alle Demos aus"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "TRADING BOT FEATURE DEMONSTRATION" + " "*25 + "║")
    print("╚" + "="*78 + "╝")

    try:
        demo_math_utils()
    except Exception as e:
        print(f"⚠️ Math Utils Demo fehlgeschlagen: {e}\n")

    try:
        demo_performance_analytics()
    except Exception as e:
        print(f"⚠️ Performance Analytics Demo fehlgeschlagen: {e}\n")

    try:
        demo_visualization()
    except Exception as e:
        print(f"⚠️ Visualization Demo fehlgeschlagen: {e}\n")

    try:
        demo_validation()
    except Exception as e:
        print(f"⚠️ Validation Demo fehlgeschlagen: {e}\n")

    try:
        demo_portfolio_features()
    except Exception as e:
        print(f"⚠️ Portfolio Features Demo fehlgeschlagen: {e}\n")

    print("="*80)
    print("DEMO ABGESCHLOSSEN")
    print("="*80)
    print("\n✨ Alle neuen Features wurden demonstriert!")
    print("📁 Charts wurden in 'demo_charts/' gespeichert")
    print("📄 Siehe IMPROVEMENTS_ANALYSIS.md für Details\n")


if __name__ == "__main__":
    main()

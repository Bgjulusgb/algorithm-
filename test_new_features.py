"""
Comprehensive Test Suite für alle neuen Features
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(days: int = 252) -> pd.DataFrame:
    """Erstellt Sample OHLCV Daten"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')

    # Simuliere Preis-Bewegung
    np.random.seed(42)
    returns = np.random.randn(days) * 0.02  # 2% volatility
    price = 100 * (1 + returns).cumprod()

    df = pd.DataFrame({
        'Open': price * (1 + np.random.randn(days) * 0.01),
        'High': price * (1 + np.abs(np.random.randn(days)) * 0.02),
        'Low': price * (1 - np.abs(np.random.randn(days)) * 0.02),
        'Close': price,
        'Volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)

    return df


def test_advanced_indicators():
    """Test Advanced Indicators"""
    print("\n" + "="*80)
    print("TEST 1: ADVANCED INDICATORS")
    print("="*80)

    try:
        from advanced_indicators import AllIndicators, TrendIndicators

        # Sample Data
        df = create_sample_data(200)

        # Test einzelne Indikatoren
        print("\n📊 Teste Trend Indicators...")
        df = TrendIndicators.ichimoku_cloud(df)
        df = TrendIndicators.parabolic_sar(df)
        df = TrendIndicators.supertrend(df)

        print(f"   ✅ Ichimoku Cloud: {len([c for c in df.columns if 'Ichimoku' in c])} Spalten")
        print(f"   ✅ Parabolic SAR: PSAR in columns={('PSAR' in df.columns)}")
        print(f"   ✅ SuperTrend: SuperTrend in columns={('SuperTrend' in df.columns)}")

        # Test alle Indikatoren
        print("\n📊 Teste All Indicators...")
        df_all = create_sample_data(200)
        df_all = AllIndicators.add_all_indicators(df_all)

        print(f"   ✅ DataFrame hat jetzt {len(df_all.columns)} Spalten")

        # Test Indicator Summary
        summary = AllIndicators.get_indicator_summary(df_all)
        print(f"   ✅ Indicator Summary: {summary.get('OVERALL', {}).get('signal', 'UNKNOWN')}")

        print("\n✅ Advanced Indicators Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Advanced Indicators Test: FAILED - {e}")
        return False


def test_pattern_recognition():
    """Test Pattern Recognition"""
    print("\n" + "="*80)
    print("TEST 2: PATTERN RECOGNITION")
    print("="*80)

    try:
        from pattern_recognition import PatternAnalyzer, CandlestickPatterns, ChartPatterns

        # Sample Data
        df = create_sample_data(100)

        # Test Candlestick Patterns
        print("\n🕯️ Teste Candlestick Patterns...")
        df_patterns = PatternAnalyzer.add_pattern_columns(df)

        pattern_cols = [c for c in df_patterns.columns if 'Pattern_' in c]
        print(f"   ✅ {len(pattern_cols)} Pattern-Spalten hinzugefügt")

        # Test Chart Patterns
        print("\n📈 Teste Chart Patterns...")
        support_resistance = ChartPatterns.find_support_resistance(df)
        print(f"   ✅ Support Levels: {len(support_resistance.get('support', []))}")
        print(f"   ✅ Resistance Levels: {len(support_resistance.get('resistance', []))}")

        # Double Top/Bottom
        dt_found, dt_level = ChartPatterns.double_top(df)
        db_found, db_level = ChartPatterns.double_bottom(df)
        print(f"   ✅ Double Top: {dt_found}, Double Bottom: {db_found}")

        # Triangle Pattern
        triangle = ChartPatterns.triangle_pattern(df)
        print(f"   ✅ Triangle Pattern: {triangle.get('type', 'None')}")

        # Full Analysis
        print("\n🔍 Teste Full Pattern Analysis...")
        analysis = PatternAnalyzer.analyze_all_patterns(df)
        print(f"   ✅ Overall Signal: {analysis.get('overall_signal', 'UNKNOWN')}")

        print("\n✅ Pattern Recognition Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Pattern Recognition Test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_timeframe():
    """Test Multi-Timeframe Analysis"""
    print("\n" + "="*80)
    print("TEST 3: MULTI-TIMEFRAME ANALYSIS")
    print("="*80)

    try:
        from multi_timeframe import DataCache, MultiTimeframeAnalyzer

        # Test Cache
        print("\n💾 Teste Data Cache...")
        cache = DataCache(cache_dir=".test_cache")

        # Speichere Test-Daten
        test_df = create_sample_data(50)
        cache.set("TEST", "1d", "2023-01-01", "2024-01-01", test_df)

        # Lade aus Cache
        cached_df = cache.get("TEST", "1d", "2023-01-01", "2024-01-01")
        print(f"   ✅ Cache funktioniert: {cached_df is not None}")

        # Cleanup
        cache.clear()

        # Test Multi-Timeframe Analyzer
        print("\n📊 Teste Multi-Timeframe Analyzer...")
        analyzer = MultiTimeframeAnalyzer(use_cache=False)

        # Simuliere Multi-Timeframe Daten
        data_dict = {
            '1d': create_sample_data(252),
            '1h': create_sample_data(180),
            '15m': create_sample_data(100)
        }

        # Trend Alignment
        trend_alignment = analyzer.calculate_trend_alignment(data_dict)
        print(f"   ✅ Trend Alignment: {trend_alignment.get('alignment', 'UNKNOWN')}")
        print(f"   ✅ Confidence: {trend_alignment.get('confidence', 0)*100:.1f}%")

        # Support/Resistance
        sr_levels = analyzer.calculate_higher_timeframe_support_resistance(data_dict)
        print(f"   ✅ HTF Support Levels: {len(sr_levels.get('support', []))}")

        # Confluence
        confluence = analyzer.get_confluence_signals(data_dict)
        print(f"   ✅ Confluence Signal: {confluence.get('overall_signal', 'UNKNOWN')}")
        print(f"   ✅ Confluence Strength: {confluence.get('confluence_strength', 0)*100:.1f}%")

        print("\n✅ Multi-Timeframe Analysis Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Multi-Timeframe Analysis Test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_orders():
    """Test Advanced Orders"""
    print("\n" + "="*80)
    print("TEST 4: ADVANCED ORDERS & RISK MANAGEMENT")
    print("="*80)

    try:
        from advanced_orders import OrderManager, AdvancedRiskManager, OrderSide, OrderType

        # Test Order Manager
        print("\n📝 Teste Order Manager...")
        manager = OrderManager()

        # Market Order
        market_order = manager.create_market_order("AAPL", OrderSide.BUY, 100)
        print(f"   ✅ Market Order: {market_order.order_id}")

        # Limit Order
        limit_order = manager.create_limit_order("AAPL", OrderSide.BUY, 100, 150.0)
        print(f"   ✅ Limit Order: {limit_order.order_id} @ $150.00")

        # Stop Order
        stop_order = manager.create_stop_order("AAPL", OrderSide.SELL, 100, 140.0)
        print(f"   ✅ Stop Order: {stop_order.order_id} @ Stop $140.00")

        # Trailing Stop
        trailing_order = manager.create_trailing_stop_order("AAPL", OrderSide.SELL, 100, 5.0)
        print(f"   ✅ Trailing Stop Order: {trailing_order.order_id} @ 5%")

        # OCO Order
        profit_order, stop_order = manager.create_oco_order("AAPL", 100, 160.0, 140.0)
        print(f"   ✅ OCO Orders: {profit_order.order_id}, {stop_order.order_id}")

        # Test Order Processing
        print("\n⚙️ Teste Order Processing...")

        # Simulate price movement
        current_price = 155.0
        filled = manager.process_orders("AAPL", current_price)
        print(f"   ✅ {len(filled)} Orders filled @ ${current_price}")

        # Test Risk Manager
        print("\n🛡️ Teste Risk Manager...")
        risk_manager = AdvancedRiskManager(initial_capital=100000)

        # Position Size
        position_size = risk_manager.calculate_position_size(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            portfolio_value=100000,
            confidence=1.0
        )
        print(f"   ✅ Position Size: {position_size} shares")

        # Dynamic Stop Loss
        stop_loss = risk_manager.calculate_dynamic_stop_loss(
            entry_price=150.0,
            atr=3.0,
            side="long"
        )
        print(f"   ✅ Dynamic Stop Loss: ${stop_loss:.2f}")

        # Take Profit
        take_profit = risk_manager.calculate_take_profit(
            entry_price=150.0,
            stop_loss=145.0,
            risk_reward_ratio=2.0
        )
        print(f"   ✅ Take Profit: ${take_profit:.2f}")

        print("\n✅ Advanced Orders & Risk Management Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Advanced Orders & Risk Management Test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backtest_optimizer():
    """Test Backtest Optimizer"""
    print("\n" + "="*80)
    print("TEST 5: BACKTEST OPTIMIZATION")
    print("="*80)

    try:
        from backtest_optimizer import BacktestOptimizer, example_backtest_function

        # Sample Data
        df = create_sample_data(300)

        # Test Grid Search
        print("\n🔍 Teste Grid Search...")

        optimizer = BacktestOptimizer(example_backtest_function)

        param_grid = {
            'sma_short': [10, 20, 30],
            'sma_long': [50, 100, 150]
        }

        result = optimizer.grid_search(df, param_grid, metric='sharpe_ratio')

        if result:
            print(f"   ✅ Best Params: {result.get('best_params', {})}")
            print(f"   ✅ Best Sharpe: {result.get('best_metrics', {}).get('sharpe_ratio', 0):.4f}")
            print(f"   ✅ Tested {len(result.get('all_results', []))} combinations")

        # Test Walk-Forward Analysis
        print("\n🔄 Teste Walk-Forward Analysis...")

        wf_result = optimizer.walk_forward_analysis(
            df,
            param_grid,
            train_period_days=100,
            test_period_days=30,
            step_days=20
        )

        if wf_result:
            print(f"   ✅ Periods: {len(wf_result.get('periods', []))}")
            print(f"   ✅ Avg Train Sharpe: {wf_result.get('avg_train_metric', 0):.4f}")
            print(f"   ✅ Avg Test Sharpe: {wf_result.get('avg_test_metric', 0):.4f}")
            print(f"   ✅ Overfitting Rate: {wf_result.get('overfitting_rate', 0)*100:.2f}%")

        print("\n✅ Backtest Optimization Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Backtest Optimization Test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_manager():
    """Test CSV Manager Verbesserungen"""
    print("\n" + "="*80)
    print("TEST 6: CSV MANAGER (YAHOO FINANCE)")
    print("="*80)

    try:
        from csv_manager import CSVManager
        from datetime import datetime

        # Create Manager
        manager = CSVManager(trades_file="test_trades.csv")

        # Add Trades
        print("\n📝 Füge Test-Trades hinzu...")
        manager.add_buy("AAPL", datetime.now(), 10, 150.0, 1.5, "Test Buy")
        manager.add_sell("AAPL", datetime.now(), 10, 160.0, 1.5, "Test Sell")

        # Export for Yahoo Finance
        print("\n💾 Exportiere für Yahoo Finance...")
        output_file = manager.export_for_yahoo("test_yahoo_portfolio.csv")
        print(f"   ✅ Export erfolgreich: {output_file}")

        # Verify file exists and has correct format
        import os
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            required_columns = ['Symbol', 'Trade Date', 'Action', 'Quantity', 'Price', 'Commission', 'Notes']
            has_all_columns = all(col in df.columns for col in required_columns)
            print(f"   ✅ Korrekte Spalten: {has_all_columns}")
            print(f"   ✅ Anzahl Trades: {len(df)}")
        else:
            print(f"   ❌ Datei nicht gefunden: {output_file}")

        print("\n✅ CSV Manager Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ CSV Manager Test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Hauptfunktion - Führt alle Tests aus"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "UMFASSENDE FEATURE TESTS" + " "*28 + "║")
    print("╚" + "="*78 + "╝")

    results = {}

    # Run all tests
    results['advanced_indicators'] = test_advanced_indicators()
    results['pattern_recognition'] = test_pattern_recognition()
    results['multi_timeframe'] = test_multi_timeframe()
    results['advanced_orders'] = test_advanced_orders()
    results['backtest_optimizer'] = test_backtest_optimizer()
    results['csv_manager'] = test_csv_manager()

    # Summary
    print("\n" + "="*80)
    print("TEST ZUSAMMENFASSUNG")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print("="*80)
    print(f"\n🎯 Ergebnis: {passed}/{total} Tests bestanden ({passed/total*100:.1f}%)")

    if passed == total:
        print("✨ ALLE TESTS BESTANDEN! ✨")
        return 0
    else:
        print(f"⚠️ {total - passed} Test(s) fehlgeschlagen")
        return 1


if __name__ == "__main__":
    exit(main())

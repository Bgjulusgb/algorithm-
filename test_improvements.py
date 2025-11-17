"""
Test Suite für neue Verbesserungen
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
    np.random.seed(42)
    returns = np.random.randn(days) * 0.02
    price = 100 * (1 + returns).cumprod()

    df = pd.DataFrame({
        'Open': price * (1 + np.random.randn(days) * 0.01),
        'High': price * (1 + np.abs(np.random.randn(days)) * 0.02),
        'Low': price * (1 - np.abs(np.random.randn(days)) * 0.02),
        'Close': price,
        'Volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)

    return df


def test_execution_simulator():
    """Test Execution Simulator"""
    print("\n" + "="*80)
    print("TEST 1: EXECUTION SIMULATOR (Slippage & Market Impact)")
    print("="*80)

    try:
        from execution_simulator import ExecutionSimulator, MarketCondition, LiquidityAnalyzer

        df = create_sample_data(100)

        # Initialize Simulator
        simulator = ExecutionSimulator(
            base_slippage_bps=5.0,
            market_impact_coefficient=0.1,
            spread_bps=2.0,
            latency_bars=1
        )

        print("\n📊 Teste Buy Execution...")
        buy_exec = simulator.simulate_buy_execution(
            signal_price=100.0,
            signal_time_idx=50,
            shares=1000,
            df=df,
            market_condition=MarketCondition.NORMAL
        )

        if buy_exec['executed']:
            print(f"   ✅ Execution erfolgreich")
            print(f"   Signal Price: ${buy_exec['signal_price']:.2f}")
            print(f"   Final Price: ${buy_exec['final_price']:.2f}")
            print(f"   Slippage: ${buy_exec['slippage']:.4f}")
            print(f"   Market Impact: ${buy_exec['market_impact']:.4f}")
            print(f"   Total Cost: {buy_exec['total_cost_pct']:.4f}%")

        # Test Liquidity Analyzer
        print("\n📊 Teste Liquidity Analyzer...")
        liquidity_score = LiquidityAnalyzer.calculate_liquidity_score(df)
        print(f"   ✅ Liquidity Score berechnet: Durchschnitt = {liquidity_score.mean():.1f}")

        market_condition = LiquidityAnalyzer.classify_market_condition(df, 50)
        print(f"   ✅ Market Condition: {market_condition.value}")

        print("\n✅ Execution Simulator Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Execution Simulator Test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_portfolio_rebalancing():
    """Test Portfolio Rebalancing"""
    print("\n" + "="*80)
    print("TEST 2: PORTFOLIO REBALANCING")
    print("="*80)

    try:
        from portfolio_rebalancing import PortfolioRebalancer, RebalancingStrategy, RebalancingTrigger

        # Sample data
        df1 = create_sample_data(100)
        df2 = create_sample_data(100)
        price_data = {'AAPL': df1, 'MSFT': df2}

        # Test Equal Weight
        print("\n⚖️ Teste Equal Weight Rebalancing...")
        rebalancer = PortfolioRebalancer(
            strategy=RebalancingStrategy.EQUAL_WEIGHT,
            trigger=RebalancingTrigger.TIME_BASED,
            rebalance_frequency_days=30
        )

        current_positions = {'AAPL': 100, 'MSFT': 50}
        target_weights = rebalancer.calculate_target_weights(
            current_positions,
            price_data,
            datetime.now()
        )

        print(f"   ✅ Target Weights: {target_weights}")
        assert len(target_weights) == 2
        assert abs(sum(target_weights.values()) - 1.0) < 0.01

        # Test Risk Parity
        print("\n📊 Teste Risk Parity Rebalancing...")
        rebalancer = PortfolioRebalancer(
            strategy=RebalancingStrategy.RISK_PARITY
        )

        target_weights = rebalancer.calculate_target_weights(
            current_positions,
            price_data,
            datetime.now()
        )

        print(f"   ✅ Risk Parity Weights: {target_weights}")

        # Test Rebalancing Execution
        print("\n🔄 Teste Rebalancing Execution...")
        current_prices = {'AAPL': 150.0, 'MSFT': 300.0}
        total_value = 100 * 150 + 50 * 300  # $30,000

        result = rebalancer.execute_rebalance(
            current_positions,
            current_prices,
            price_data,
            datetime.now(),
            total_value
        )

        if result.get('rebalanced'):
            print(f"   ✅ Rebalancing durchgeführt")
            print(f"   Trades: {len(result.get('trades', {}))}")
            print(f"   Transaction Costs: ${result.get('transaction_costs', 0):.2f}")

        print("\n✅ Portfolio Rebalancing Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Portfolio Rebalancing Test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_integration():
    """Test ML Integration"""
    print("\n" + "="*80)
    print("TEST 3: MACHINE LEARNING INTEGRATION")
    print("="*80)

    try:
        from ml_integration import FeatureEngineer, MLTradingModel

        df = create_sample_data(300)

        # Test Feature Engineering
        print("\n🔧 Teste Feature Engineering...")
        df_features = FeatureEngineer.create_all_features(df)

        print(f"   ✅ Original Columns: {len(df.columns)}")
        print(f"   ✅ Feature Columns: {len(df_features.columns)}")
        print(f"   ✅ New Features: {len(df_features.columns) - len(df.columns)}")

        # Test ML Model Training
        print("\n🤖 Teste ML Model Training...")
        model = MLTradingModel(
            model_type='random_forest',
            prediction_horizon=5,
            threshold=0.01
        )

        train_result = model.train(df, test_size=0.2)

        print(f"   ✅ Model trainiert")
        print(f"   Train Accuracy: {train_result['train_accuracy']*100:.2f}%")
        print(f"   Test Accuracy: {train_result['test_accuracy']*100:.2f}%")
        print(f"   Features: {train_result['num_features']}")

        # Test Prediction
        print("\n🔮 Teste Predictions...")
        predictions = model.predict(df.tail(10))
        proba = model.predict_proba(df.tail(10))

        print(f"   ✅ Predictions: {predictions.values}")
        print(f"   ✅ Probabilities: Mean UP = {proba['prob_up'].mean():.3f}")

        print("\n✅ ML Integration Test: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ ML Integration Test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Hauptfunktion - Führt alle Tests aus"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*22 + "VERBESSERUNGEN TEST SUITE" + " "*31 + "║")
    print("╚" + "="*78 + "╝")

    results = {}

    # Run tests
    results['execution_simulator'] = test_execution_simulator()
    results['portfolio_rebalancing'] = test_portfolio_rebalancing()
    results['ml_integration'] = test_ml_integration()

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

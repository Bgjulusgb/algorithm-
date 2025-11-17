"""
Backtest Optimization Module
Parameter Grid Search und Walk-Forward Analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from itertools import product
from datetime import datetime, timedelta
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

logger = logging.getLogger(__name__)


class BacktestOptimizer:
    """Backtest-Optimierer mit Grid Search"""

    def __init__(self, backtest_function: Callable):
        """
        Initialisiert Optimizer

        Args:
            backtest_function: Funktion die Backtest durchführt
                              Signatur: func(data, **params) -> Dict[metrics]
        """
        self.backtest_function = backtest_function
        self.results = []

    def grid_search(self,
                   data: pd.DataFrame,
                   param_grid: Dict[str, List],
                   metric: str = 'sharpe_ratio',
                   n_jobs: int = 1) -> Dict:
        """
        Grid Search über Parameter-Raum

        Args:
            data: DataFrame mit OHLCV Daten
            param_grid: Dict mit Parameter-Listen
                       z.B. {'sma_short': [10, 20, 30], 'sma_long': [50, 100, 200]}
            metric: Metrik zur Optimierung
            n_jobs: Anzahl paralleler Jobs (1 = sequential)

        Returns:
            Dict mit besten Parametern und Ergebnissen
        """
        logger.info(f"🔍 Starte Grid Search mit {len(param_grid)} Parametern...")

        # Generiere alle Kombinationen
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"📊 {len(combinations)} Kombinationen zu testen")

        results = []

        if n_jobs == 1:
            # Sequential
            for i, combo in enumerate(combinations):
                params = dict(zip(param_names, combo))
                result = self._run_single_backtest(data, params, i, len(combinations))
                if result:
                    results.append(result)
        else:
            # Parallel (nicht implementiert wegen Pickle-Problemen)
            logger.warning("Parallel Processing nicht implementiert, verwende sequential")
            for i, combo in enumerate(combinations):
                params = dict(zip(param_names, combo))
                result = self._run_single_backtest(data, params, i, len(combinations))
                if result:
                    results.append(result)

        if not results:
            logger.error("❌ Keine erfolgreichen Backtests")
            return {}

        # Speichere alle Ergebnisse
        self.results = results

        # Finde beste Parameter
        best_result = max(results, key=lambda x: x['metrics'].get(metric, float('-inf')))

        logger.info(f"✅ Grid Search abgeschlossen")
        logger.info(f"🏆 Beste Parameter: {best_result['params']}")
        logger.info(f"📈 Beste {metric}: {best_result['metrics'].get(metric, 0):.4f}")

        return {
            'best_params': best_result['params'],
            'best_metrics': best_result['metrics'],
            'all_results': results,
            'optimization_metric': metric
        }

    def _run_single_backtest(self, data: pd.DataFrame, params: Dict,
                            index: int, total: int) -> Optional[Dict]:
        """Führt einzelnen Backtest durch"""
        try:
            logger.info(f"Test {index+1}/{total}: {params}")

            # Backtest durchführen
            metrics = self.backtest_function(data, **params)

            if not metrics:
                logger.warning(f"Backtest failed für {params}")
                return None

            return {
                'params': params,
                'metrics': metrics
            }

        except Exception as e:
            logger.error(f"Backtest-Fehler für {params}: {e}")
            return None

    def walk_forward_analysis(self,
                             data: pd.DataFrame,
                             param_grid: Dict[str, List],
                             train_period_days: int = 252,  # 1 Jahr
                             test_period_days: int = 63,     # 3 Monate
                             step_days: int = 21,            # 1 Monat
                             metric: str = 'sharpe_ratio') -> Dict:
        """
        Walk-Forward Analysis

        Trainiert auf einem Zeitraum, testet auf nächstem Zeitraum,
        dann weiter schieben

        Args:
            data: DataFrame mit OHLCV Daten
            param_grid: Parameter-Grid
            train_period_days: Trainings-Periode in Tagen
            test_period_days: Test-Periode in Tagen
            step_days: Schrittweite in Tagen
            metric: Optimierungs-Metrik

        Returns:
            Dict mit Ergebnissen
        """
        logger.info(f"🔄 Starte Walk-Forward Analysis...")
        logger.info(f"📊 Train: {train_period_days} Tage, Test: {test_period_days} Tage, Step: {step_days} Tage")

        if len(data) < train_period_days + test_period_days:
            logger.error("❌ Nicht genug Daten für Walk-Forward Analysis")
            return {}

        results = []
        start_idx = 0

        while start_idx + train_period_days + test_period_days <= len(data):
            train_end_idx = start_idx + train_period_days
            test_end_idx = train_end_idx + test_period_days

            # Split data
            train_data = data.iloc[start_idx:train_end_idx]
            test_data = data.iloc[train_end_idx:test_end_idx]

            logger.info(f"\n📅 Period {len(results)+1}:")
            logger.info(f"   Train: {train_data.index[0]} - {train_data.index[-1]} ({len(train_data)} Tage)")
            logger.info(f"   Test:  {test_data.index[0]} - {test_data.index[-1]} ({len(test_data)} Tage)")

            # Optimize auf Training Data
            train_result = self.grid_search(train_data, param_grid, metric, n_jobs=1)

            if not train_result:
                logger.warning("Optimization failed, skip period")
                start_idx += step_days
                continue

            best_params = train_result['best_params']

            # Test auf Test Data
            try:
                test_metrics = self.backtest_function(test_data, **best_params)

                results.append({
                    'period': len(results) + 1,
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'best_params': best_params,
                    'train_metrics': train_result['best_metrics'],
                    'test_metrics': test_metrics
                })

                logger.info(f"   Train {metric}: {train_result['best_metrics'].get(metric, 0):.4f}")
                logger.info(f"   Test {metric}:  {test_metrics.get(metric, 0):.4f}")

            except Exception as e:
                logger.error(f"Test failed: {e}")

            # Nächste Period
            start_idx += step_days

        if not results:
            logger.error("❌ Walk-Forward Analysis failed")
            return {}

        # Aggregiere Ergebnisse
        avg_train_metric = np.mean([r['train_metrics'].get(metric, 0) for r in results])
        avg_test_metric = np.mean([r['test_metrics'].get(metric, 0) for r in results])

        logger.info(f"\n✅ Walk-Forward Analysis abgeschlossen")
        logger.info(f"📊 Durchschnittlicher Train {metric}: {avg_train_metric:.4f}")
        logger.info(f"📊 Durchschnittlicher Test {metric}:  {avg_test_metric:.4f}")
        logger.info(f"📉 Overfitting-Rate: {(avg_train_metric - avg_test_metric) / avg_train_metric * 100:.2f}%")

        return {
            'periods': results,
            'avg_train_metric': avg_train_metric,
            'avg_test_metric': avg_test_metric,
            'overfitting_rate': (avg_train_metric - avg_test_metric) / avg_train_metric if avg_train_metric != 0 else 0
        }

    def export_results(self, filename: str = "optimization_results.json"):
        """Exportiert Ergebnisse als JSON"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)

            logger.info(f"✅ Ergebnisse exportiert nach: {filename}")

        except Exception as e:
            logger.error(f"Export-Fehler: {e}")


class ParameterSensitivityAnalyzer:
    """Analysiert Parameter-Sensitivität"""

    @staticmethod
    def analyze_parameter_sensitivity(results: List[Dict],
                                     param_name: str,
                                     metric: str = 'sharpe_ratio') -> Dict:
        """
        Analysiert wie sensitiv Ergebnis auf Parameter-Änderung ist

        Args:
            results: Liste von Backtest-Ergebnissen
            param_name: Parameter-Name zu analysieren
            metric: Metrik zur Analyse

        Returns:
            Dict mit Sensitivity-Analyse
        """
        # Gruppiere nach Parameter-Wert
        param_groups = {}

        for result in results:
            param_value = result['params'].get(param_name)
            if param_value is None:
                continue

            if param_value not in param_groups:
                param_groups[param_value] = []

            param_groups[param_value].append(result['metrics'].get(metric, 0))

        # Berechne Statistiken für jeden Wert
        sensitivity_data = []

        for param_value, metrics_list in sorted(param_groups.items()):
            sensitivity_data.append({
                'param_value': param_value,
                'mean': np.mean(metrics_list),
                'std': np.std(metrics_list),
                'min': np.min(metrics_list),
                'max': np.max(metrics_list),
                'count': len(metrics_list)
            })

        # Finde optimalen Wert
        optimal = max(sensitivity_data, key=lambda x: x['mean'])

        # Berechne Stabilität (niedrige Std = stabiler)
        stability_score = 1 / (optimal['std'] + 0.0001)  # Avoid division by zero

        logger.info(f"\n📊 Sensitivity Analysis: {param_name}")
        logger.info(f"   Optimaler Wert: {optimal['param_value']}")
        logger.info(f"   Mean {metric}: {optimal['mean']:.4f} (±{optimal['std']:.4f})")
        logger.info(f"   Stability Score: {stability_score:.2f}")

        return {
            'parameter': param_name,
            'sensitivity_data': sensitivity_data,
            'optimal_value': optimal['param_value'],
            'optimal_mean': optimal['mean'],
            'optimal_std': optimal['std'],
            'stability_score': stability_score
        }


class MonteCarloSimulator:
    """Monte Carlo Simulation für Robustheitstests"""

    @staticmethod
    def simulate_trades(trades: List[Dict],
                       num_simulations: int = 1000,
                       initial_capital: float = 100000) -> Dict:
        """
        Monte Carlo Simulation von Trade-Sequenzen

        Mischt Trades random und berechnet mögliche Ergebnisse

        Args:
            trades: Liste von Trades mit P/L
            num_simulations: Anzahl Simulationen
            initial_capital: Anfangskapital

        Returns:
            Dict mit Simulations-Ergebnissen
        """
        logger.info(f"🎲 Starte Monte Carlo Simulation ({num_simulations} runs)...")

        if not trades:
            logger.error("Keine Trades für Simulation")
            return {}

        # Extrahiere P/L
        pnl_list = [t.get('profit_loss', 0) for t in trades if 'profit_loss' in t]

        if not pnl_list:
            logger.error("Keine P/L-Daten in Trades")
            return {}

        final_values = []

        for sim in range(num_simulations):
            # Shuffle trades
            shuffled_pnl = np.random.choice(pnl_list, size=len(pnl_list), replace=True)

            # Berechne kumulative P/L
            capital = initial_capital
            equity_curve = [capital]

            for pnl in shuffled_pnl:
                capital += pnl
                equity_curve.append(capital)

            final_values.append(capital)

        # Statistiken
        final_values = np.array(final_values)

        percentiles = {
            '5th': np.percentile(final_values, 5),
            '25th': np.percentile(final_values, 25),
            '50th': np.percentile(final_values, 50),
            '75th': np.percentile(final_values, 75),
            '95th': np.percentile(final_values, 95),
        }

        probability_of_profit = (final_values > initial_capital).sum() / num_simulations
        probability_of_ruin = (final_values < initial_capital * 0.5).sum() / num_simulations

        logger.info(f"✅ Monte Carlo abgeschlossen")
        logger.info(f"📊 Median Final Value: ${percentiles['50th']:,.2f}")
        logger.info(f"📊 5th-95th Percentile: ${percentiles['5th']:,.2f} - ${percentiles['95th']:,.2f}")
        logger.info(f"📊 Probability of Profit: {probability_of_profit*100:.1f}%")
        logger.info(f"📊 Probability of Ruin (<50% capital): {probability_of_ruin*100:.1f}%")

        return {
            'num_simulations': num_simulations,
            'initial_capital': initial_capital,
            'final_values': final_values.tolist(),
            'percentiles': percentiles,
            'mean': float(np.mean(final_values)),
            'std': float(np.std(final_values)),
            'probability_of_profit': float(probability_of_profit),
            'probability_of_ruin': float(probability_of_ruin),
            'best_case': float(np.max(final_values)),
            'worst_case': float(np.min(final_values))
        }


def example_backtest_function(data: pd.DataFrame, sma_short: int = 20, sma_long: int = 50, **kwargs) -> Dict:
    """
    Beispiel Backtest-Funktion für Optimizer

    Args:
        data: DataFrame mit OHLCV
        sma_short: Kurzfristige SMA Periode
        sma_long: Langfristige SMA Periode

    Returns:
        Dict mit Metriken
    """
    try:
        # Berechne SMAs
        data['SMA_Short'] = data['Close'].rolling(window=sma_short).mean()
        data['SMA_Long'] = data['Close'].rolling(window=sma_long).mean()

        # Generiere Signale
        data['Signal'] = 0
        data.loc[data['SMA_Short'] > data['SMA_Long'], 'Signal'] = 1
        data.loc[data['SMA_Short'] < data['SMA_Long'], 'Signal'] = -1

        # Berechne Returns
        data['Position'] = data['Signal'].shift(1)
        data['Strategy_Return'] = data['Close'].pct_change() * data['Position']

        # Metriken
        total_return = (1 + data['Strategy_Return']).prod() - 1
        sharpe_ratio = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252)
        max_drawdown = (data['Strategy_Return'].cumsum() - data['Strategy_Return'].cumsum().cummax()).min()

        num_trades = (data['Signal'].diff() != 0).sum() / 2
        win_rate = (data[data['Strategy_Return'] > 0]['Strategy_Return'].count() /
                   data[data['Strategy_Return'] != 0]['Strategy_Return'].count()) if num_trades > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate
        }

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return {}

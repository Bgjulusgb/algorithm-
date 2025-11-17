"""
Performance Analytics Module
Erweiterte Performance-Analysen und Reporting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from math_utils import (
    StatisticsAnalyzer, CorrelationAnalyzer,
    MonteCarloSimulator, KellyCriterion
)

logger = logging.getLogger(__name__)


class PerformanceAnalytics:
    """Umfassende Performance-Analyse für Trading Bot"""

    def __init__(self, trades: List[Dict], initial_capital: float):
        """
        Initialisiert Performance Analytics

        Args:
            trades: Liste von Trade-Dictionaries
            initial_capital: Startkapital
        """
        self.trades = trades
        self.initial_capital = initial_capital
        self.trades_df = self._prepare_trades_dataframe()
        self.stats_analyzer = StatisticsAnalyzer()

    def _prepare_trades_dataframe(self) -> pd.DataFrame:
        """Bereitet Trade-Daten als DataFrame vor"""
        if not self.trades:
            return pd.DataFrame()

        df = pd.DataFrame(self.trades)

        # Konvertiere Datum wenn nötig
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        return df

    def calculate_comprehensive_metrics(self) -> Dict:
        """
        Berechnet umfassende Performance-Metriken

        Returns:
            Dictionary mit allen Metriken
        """
        if len(self.trades_df) == 0:
            return {}

        # Nur Verkaufs-Trades für P&L Analyse
        sell_trades = self.trades_df[self.trades_df['action'] == 'SELL'].copy()

        if len(sell_trades) == 0:
            return {'info': 'Keine abgeschlossenen Trades'}

        # Grundlegende Statistiken
        metrics = self._calculate_basic_metrics(sell_trades)

        # Erweiterte Risk-Metriken
        if 'profit_loss' in sell_trades.columns:
            returns = pd.Series(sell_trades['profit_loss'].values)
            metrics.update(self._calculate_risk_metrics(returns))

        # Kelly Criterion
        kelly = KellyCriterion.calculate_kelly_from_trades(sell_trades)
        metrics['kelly_criterion'] = kelly
        metrics['recommended_position_size'] = f"{kelly*100:.1f}%"

        # Trade-Analyse
        metrics.update(self._analyze_trade_patterns(sell_trades))

        return metrics

    def _calculate_basic_metrics(self, sell_trades: pd.DataFrame) -> Dict:
        """Berechnet grundlegende Metriken"""
        total_trades = len(sell_trades)

        if 'profit_loss' not in sell_trades.columns:
            return {'total_trades': total_trades}

        winning_trades = sell_trades[sell_trades['profit_loss'] > 0]
        losing_trades = sell_trades[sell_trades['profit_loss'] <= 0]

        total_profit = sell_trades['profit_loss'].sum()
        total_win = winning_trades['profit_loss'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 else 0

        avg_win = winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['profit_loss'].mean()) if len(losing_trades) > 0 else 0

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        # Profit Factor
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Average Trade
        avg_trade = sell_trades['profit_loss'].mean()

        # Largest Win/Loss
        largest_win = winning_trades['profit_loss'].max() if len(winning_trades) > 0 else 0
        largest_loss = losing_trades['profit_loss'].min() if len(losing_trades) > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_win': total_win,
            'total_loss': total_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_trade': avg_trade,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
        }

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Berechnet erweiterte Risk-Metriken"""
        metrics = {}

        # Sharpe Ratio
        sharpe = self.stats_analyzer.calculate_sharpe_ratio(returns)
        metrics['sharpe_ratio'] = sharpe

        # Sortino Ratio
        sortino = self.stats_analyzer.calculate_sortino_ratio(returns)
        metrics['sortino_ratio'] = sortino

        # Value at Risk (95%)
        var_95 = self.stats_analyzer.calculate_var(returns, 0.95)
        metrics['var_95'] = var_95

        # CVaR (Expected Shortfall)
        cvar_95 = self.stats_analyzer.calculate_cvar(returns, 0.95)
        metrics['cvar_95'] = cvar_95

        # Skewness & Kurtosis
        skewness = self.stats_analyzer.calculate_skewness(returns)
        kurtosis = self.stats_analyzer.calculate_kurtosis(returns)
        metrics['skewness'] = skewness
        metrics['kurtosis'] = kurtosis

        # Interpretationen
        if skewness > 0:
            metrics['skewness_interpretation'] = "Positive Skew (mehr große Gewinne)"
        else:
            metrics['skewness_interpretation'] = "Negative Skew (mehr große Verluste)"

        if kurtosis > 0:
            metrics['kurtosis_interpretation'] = "Fat Tails (höheres Extremereignis-Risiko)"
        else:
            metrics['kurtosis_interpretation'] = "Thin Tails (geringeres Extremereignis-Risiko)"

        return metrics

    def _analyze_trade_patterns(self, sell_trades: pd.DataFrame) -> Dict:
        """Analysiert Trade-Muster"""
        metrics = {}

        if 'symbol' in sell_trades.columns:
            # Beste und schlechteste Symbole
            symbol_performance = sell_trades.groupby('symbol')['profit_loss'].agg(['sum', 'mean', 'count'])
            symbol_performance = symbol_performance.sort_values('sum', ascending=False)

            if len(symbol_performance) > 0:
                best_symbol = symbol_performance.index[0]
                worst_symbol = symbol_performance.index[-1]

                metrics['best_symbol'] = best_symbol
                metrics['best_symbol_profit'] = symbol_performance.loc[best_symbol, 'sum']
                metrics['worst_symbol'] = worst_symbol
                metrics['worst_symbol_profit'] = symbol_performance.loc[worst_symbol, 'sum']

        # Consecutive Wins/Losses
        if 'profit_loss' in sell_trades.columns:
            is_win = sell_trades['profit_loss'] > 0
            consecutive_wins = self._calculate_max_consecutive(is_win, True)
            consecutive_losses = self._calculate_max_consecutive(is_win, False)

            metrics['max_consecutive_wins'] = consecutive_wins
            metrics['max_consecutive_losses'] = consecutive_losses

        # Average Hold Time (wenn Datum vorhanden)
        if 'date' in sell_trades.index.names or isinstance(sell_trades.index, pd.DatetimeIndex):
            if len(sell_trades) > 1:
                hold_times = sell_trades.index.to_series().diff()
                avg_hold_time = hold_times.mean()
                metrics['avg_hold_time_days'] = avg_hold_time.days if hasattr(avg_hold_time, 'days') else 0

        return metrics

    def _calculate_max_consecutive(self, series: pd.Series, value: bool) -> int:
        """Berechnet maximale aufeinanderfolgende Werte"""
        max_consecutive = 0
        current_consecutive = 0

        for val in series:
            if val == value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def run_monte_carlo_simulation(
        self,
        num_simulations: int = 1000,
        num_days: int = 252,
        confidence_levels: List[float] = [0.05, 0.50, 0.95]
    ) -> Dict:
        """
        Führt Monte Carlo Simulation durch

        Args:
            num_simulations: Anzahl Simulationen
            num_days: Anzahl Tage
            confidence_levels: Konfidenzniveaus

        Returns:
            Simulationsergebnisse
        """
        sell_trades = self.trades_df[self.trades_df['action'] == 'SELL']

        if len(sell_trades) == 0 or 'profit_loss' not in sell_trades.columns:
            return {'error': 'Nicht genug Daten für Simulation'}

        # Berechne historische Statistiken
        returns = sell_trades['profit_loss'] / self.initial_capital
        expected_return = returns.mean() * 252  # Annualisiert
        volatility = returns.std() * np.sqrt(252)  # Annualisiert

        # Simulation
        simulations = MonteCarloSimulator.simulate_portfolio(
            self.initial_capital,
            expected_return,
            volatility,
            num_simulations,
            num_days
        )

        # Analyse
        results = MonteCarloSimulator.analyze_simulation_results(
            simulations,
            confidence_levels
        )

        # Ruin-Wahrscheinlichkeit (unter 50% des Startkapitals)
        ruin_threshold = self.initial_capital * 0.5
        prob_ruin = MonteCarloSimulator.calculate_probability_of_ruin(
            simulations,
            ruin_threshold
        )

        results['probability_of_ruin'] = prob_ruin
        results['ruin_threshold'] = ruin_threshold
        results['num_simulations'] = num_simulations
        results['num_days'] = num_days

        return results

    def generate_performance_report(self) -> str:
        """
        Generiert formatierten Performance-Report

        Returns:
            Report als String
        """
        metrics = self.calculate_comprehensive_metrics()

        if not metrics or 'info' in metrics:
            return "Keine Performance-Daten verfügbar"

        report = []
        report.append("\n" + "="*80)
        report.append("ERWEITERTE PERFORMANCE-ANALYSE")
        report.append("="*80)

        # Grundlegende Statistiken
        report.append("\n📊 TRADING STATISTIKEN:")
        report.append(f"  Total Trades: {metrics.get('total_trades', 0)}")
        report.append(f"  Gewinntrades: {metrics.get('winning_trades', 0)}")
        report.append(f"  Verlusttrades: {metrics.get('losing_trades', 0)}")
        report.append(f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")

        # P&L
        report.append("\n💰 PROFIT & LOSS:")
        report.append(f"  Total Profit: ${metrics.get('total_profit', 0):,.2f}")
        report.append(f"  Durchschn. Gewinn: ${metrics.get('avg_win', 0):,.2f}")
        report.append(f"  Durchschn. Verlust: ${metrics.get('avg_loss', 0):,.2f}")
        report.append(f"  Größter Gewinn: ${metrics.get('largest_win', 0):,.2f}")
        report.append(f"  Größter Verlust: ${metrics.get('largest_loss', 0):,.2f}")

        # Risk-Metriken
        report.append("\n📈 RISK-ADJUSTED PERFORMANCE:")
        report.append(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        report.append(f"  Expectancy: ${metrics.get('expectancy', 0):.2f}")
        report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")

        # VaR
        if 'var_95' in metrics:
            report.append("\n⚠️ VALUE AT RISK:")
            report.append(f"  VaR (95%): ${metrics.get('var_95', 0):,.2f}")
            report.append(f"  CVaR (95%): ${metrics.get('cvar_95', 0):,.2f}")

        # Kelly Criterion
        if 'kelly_criterion' in metrics:
            report.append("\n🎯 POSITION SIZING:")
            report.append(f"  Kelly Criterion: {metrics['recommended_position_size']}")

        # Distribution
        if 'skewness' in metrics:
            report.append("\n📉 RETURN DISTRIBUTION:")
            report.append(f"  Skewness: {metrics.get('skewness', 0):.3f} - {metrics.get('skewness_interpretation', '')}")
            report.append(f"  Kurtosis: {metrics.get('kurtosis', 0):.3f} - {metrics.get('kurtosis_interpretation', '')}")

        # Trade Patterns
        if 'best_symbol' in metrics:
            report.append("\n🏆 SYMBOL PERFORMANCE:")
            report.append(f"  Bestes Symbol: {metrics['best_symbol']} (${metrics['best_symbol_profit']:,.2f})")
            report.append(f"  Schlechtestes: {metrics['worst_symbol']} (${metrics['worst_symbol_profit']:,.2f})")

        if 'max_consecutive_wins' in metrics:
            report.append("\n🔁 TRADE PATTERNS:")
            report.append(f"  Max aufeinanderfolgende Gewinne: {metrics['max_consecutive_wins']}")
            report.append(f"  Max aufeinanderfolgende Verluste: {metrics['max_consecutive_losses']}")

        report.append("\n" + "="*80)

        return "\n".join(report)

    def export_to_excel(self, filename: str = "performance_report.xlsx"):
        """
        Exportiert Performance-Analyse nach Excel

        Args:
            filename: Dateiname
        """
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Trades
                if len(self.trades_df) > 0:
                    self.trades_df.to_excel(writer, sheet_name='Trades')

                # Metrics
                metrics = self.calculate_comprehensive_metrics()
                if metrics:
                    metrics_df = pd.DataFrame([metrics]).T
                    metrics_df.columns = ['Value']
                    metrics_df.to_excel(writer, sheet_name='Metrics')

                # Monte Carlo (wenn genug Daten)
                sell_trades = self.trades_df[self.trades_df['action'] == 'SELL']
                if len(sell_trades) >= 5:
                    mc_results = self.run_monte_carlo_simulation(num_simulations=100)
                    if 'error' not in mc_results:
                        mc_df = pd.DataFrame([mc_results]).T
                        mc_df.columns = ['Value']
                        mc_df.to_excel(writer, sheet_name='Monte Carlo')

            logger.info(f"Performance-Report exportiert nach: {filename}")

        except Exception as e:
            logger.error(f"Fehler beim Excel-Export: {e}")


class PortfolioCorrelationAnalysis:
    """Korrelations-Analyse für Multi-Asset-Portfolios"""

    def __init__(self, price_data: Dict[str, pd.Series]):
        """
        Initialisiert Korrelations-Analyse

        Args:
            price_data: Dictionary {symbol: price_series}
        """
        self.price_data = price_data
        self.returns_df = self._calculate_returns()
        self.corr_analyzer = CorrelationAnalyzer()

    def _calculate_returns(self) -> pd.DataFrame:
        """Berechnet Returns für alle Assets"""
        returns = {}

        for symbol, prices in self.price_data.items():
            returns[symbol] = prices.pct_change().dropna()

        return pd.DataFrame(returns)

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Gibt Korrelationsmatrix zurück"""
        return self.corr_analyzer.calculate_correlation_matrix(self.returns_df)

    def get_covariance_matrix(self) -> pd.DataFrame:
        """Gibt Kovarianzmatrix zurück"""
        return self.corr_analyzer.calculate_covariance_matrix(self.returns_df)

    def find_diversification_opportunities(
        self,
        max_correlation: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """
        Findet Diversifikations-Möglichkeiten

        Args:
            max_correlation: Maximale akzeptable Korrelation

        Returns:
            Liste von Asset-Paaren mit niedriger Korrelation
        """
        corr_matrix = self.get_correlation_matrix()
        return self.corr_analyzer.find_uncorrelated_pairs(corr_matrix, max_correlation)

    def generate_correlation_report(self) -> str:
        """Generiert Korrelations-Report"""
        corr_matrix = self.get_correlation_matrix()

        report = []
        report.append("\n" + "="*80)
        report.append("PORTFOLIO KORRELATIONS-ANALYSE")
        report.append("="*80)

        report.append("\n📊 KORRELATIONS-MATRIX:")
        report.append(str(corr_matrix.round(2)))

        # Durchschnittliche Korrelation
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        report.append(f"\nDurchschnittliche Korrelation: {avg_corr:.3f}")

        # Diversifikations-Möglichkeiten
        opportunities = self.find_diversification_opportunities()
        if opportunities:
            report.append("\n🎯 DIVERSIFIKATIONS-MÖGLICHKEITEN (Korrelation < 0.3):")
            for sym1, sym2, corr in opportunities[:5]:  # Top 5
                report.append(f"  {sym1} - {sym2}: {corr:.3f}")

        report.append("\n" + "="*80)

        return "\n".join(report)

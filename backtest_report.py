"""
Backtest Report Generator
Erstellt umfassende Backtest-Reports mit Analysen und Visualisierungen
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BacktestReportGenerator:
    """Generiert umfassende Backtest-Reports"""

    def __init__(self, output_dir: str = "reports"):
        """
        Initialisiert Report Generator

        Args:
            output_dir: Verzeichnis für Reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_full_report(
        self,
        portfolio_data: Dict,
        trades: List[Dict],
        strategy_name: str,
        watchlist: List[str],
        initial_capital: float,
        config: Dict
    ) -> str:
        """
        Generiert vollständigen Backtest-Report

        Args:
            portfolio_data: Portfolio-Daten
            trades: Liste von Trades
            strategy_name: Name der Strategie
            watchlist: Liste von Symbolen
            initial_capital: Startkapital
            config: Konfiguration

        Returns:
            Pfad zum Report
        """
        report_lines = []

        # Header
        report_lines.append("="*100)
        report_lines.append(f"{'BACKTEST REPORT':^100}")
        report_lines.append("="*100)
        report_lines.append("")

        # Backtest Info
        report_lines.append("📊 BACKTEST INFORMATION")
        report_lines.append("-" * 100)
        report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Strategy: {strategy_name}")
        report_lines.append(f"Symbols Tested: {', '.join(watchlist)}")
        report_lines.append(f"Initial Capital: ${initial_capital:,.2f}")
        report_lines.append("")

        # Performance Summary
        report_lines.extend(self._generate_performance_summary(portfolio_data, initial_capital))

        # Trade Analysis
        if trades:
            report_lines.extend(self._generate_trade_analysis(trades))

        # Risk Metrics
        report_lines.extend(self._generate_risk_metrics(portfolio_data, trades, initial_capital))

        # Strategy Performance by Symbol
        if trades:
            report_lines.extend(self._generate_symbol_analysis(trades))

        # Time-based Analysis
        if trades:
            report_lines.extend(self._generate_time_analysis(trades))

        # Configuration Summary
        report_lines.extend(self._generate_config_summary(config))

        # Recommendations
        report_lines.extend(self._generate_recommendations(portfolio_data, trades))

        # Footer
        report_lines.append("")
        report_lines.append("="*100)
        report_lines.append(f"{'END OF REPORT':^100}")
        report_lines.append("="*100)

        # Speichere Report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"backtest_report_{timestamp}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Backtest Report gespeichert: {filename}")
        return str(filename)

    def _generate_performance_summary(
        self,
        portfolio_data: Dict,
        initial_capital: float
    ) -> List[str]:
        """Generiert Performance-Zusammenfassung"""
        lines = []
        lines.append("💰 PERFORMANCE SUMMARY")
        lines.append("-" * 100)

        final_value = portfolio_data.get('current_value', initial_capital)
        total_return = ((final_value - initial_capital) / initial_capital) * 100

        lines.append(f"Final Portfolio Value: ${final_value:,.2f}")
        lines.append(f"Total Return: {total_return:+.2f}%")
        lines.append(f"Total Profit/Loss: ${final_value - initial_capital:+,.2f}")

        # Additional Metrics
        metrics = portfolio_data.get('metrics', {})

        if 'sharpe_ratio' in metrics:
            lines.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")

        if 'sortino_ratio' in metrics:
            lines.append(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")

        if 'max_drawdown' in metrics:
            lines.append(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")

        if 'calmar_ratio' in metrics:
            lines.append(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")

        lines.append("")
        return lines

    def _generate_trade_analysis(self, trades: List[Dict]) -> List[str]:
        """Generiert Trade-Analyse"""
        lines = []
        lines.append("📈 TRADE ANALYSIS")
        lines.append("-" * 100)

        trades_df = pd.DataFrame(trades)
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']

        lines.append(f"Total Trades: {len(trades)}")
        lines.append(f"Buy Orders: {len(buy_trades)}")
        lines.append(f"Sell Orders: {len(sell_trades)}")

        if len(sell_trades) > 0 and 'profit_loss' in sell_trades.columns:
            winning_trades = sell_trades[sell_trades['profit_loss'] > 0]
            losing_trades = sell_trades[sell_trades['profit_loss'] <= 0]

            win_rate = len(winning_trades) / len(sell_trades) * 100
            lines.append(f"Win Rate: {win_rate:.1f}%")
            lines.append(f"Winning Trades: {len(winning_trades)}")
            lines.append(f"Losing Trades: {len(losing_trades)}")

            if len(winning_trades) > 0:
                avg_win = winning_trades['profit_loss'].mean()
                max_win = winning_trades['profit_loss'].max()
                lines.append(f"Average Win: ${avg_win:,.2f}")
                lines.append(f"Largest Win: ${max_win:,.2f}")

            if len(losing_trades) > 0:
                avg_loss = losing_trades['profit_loss'].mean()
                max_loss = losing_trades['profit_loss'].min()
                lines.append(f"Average Loss: ${avg_loss:,.2f}")
                lines.append(f"Largest Loss: ${max_loss:,.2f}")

            # Profit Factor
            total_wins = winning_trades['profit_loss'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 else 0

            if total_losses > 0:
                profit_factor = total_wins / total_losses
                lines.append(f"Profit Factor: {profit_factor:.2f}")

            # Expectancy
            expectancy = (win_rate/100 * avg_win if len(winning_trades) > 0 else 0) - \
                        ((1 - win_rate/100) * abs(avg_loss) if len(losing_trades) > 0 else 0)
            lines.append(f"Expectancy: ${expectancy:.2f} per trade")

        lines.append("")
        return lines

    def _generate_risk_metrics(
        self,
        portfolio_data: Dict,
        trades: List[Dict],
        initial_capital: float
    ) -> List[str]:
        """Generiert Risk-Metriken"""
        lines = []
        lines.append("⚠️ RISK METRICS")
        lines.append("-" * 100)

        metrics = portfolio_data.get('metrics', {})

        # Value at Risk
        if 'var_95' in metrics:
            lines.append(f"Value at Risk (95%): ${metrics['var_95']:,.2f}")

        if 'cvar_95' in metrics:
            lines.append(f"Conditional VaR (95%): ${metrics['cvar_95']:,.2f}")

        # Distribution Metrics
        if 'skewness' in metrics:
            skew = metrics['skewness']
            lines.append(f"Return Skewness: {skew:.3f}")
            if skew > 0:
                lines.append("  → Positively skewed (more large wins than large losses)")
            else:
                lines.append("  → Negatively skewed (more large losses than large wins)")

        if 'kurtosis' in metrics:
            kurt = metrics['kurtosis']
            lines.append(f"Return Kurtosis: {kurt:.3f}")
            if kurt > 0:
                lines.append("  → Fat tails (higher probability of extreme events)")
            else:
                lines.append("  → Thin tails (lower probability of extreme events)")

        # Kelly Criterion
        if 'kelly_criterion' in metrics:
            kelly = metrics['kelly_criterion'] * 100
            lines.append(f"Optimal Position Size (Kelly): {kelly:.1f}%")

        # Max Consecutive
        if 'max_consecutive_wins' in metrics:
            lines.append(f"Max Consecutive Wins: {metrics['max_consecutive_wins']}")

        if 'max_consecutive_losses' in metrics:
            lines.append(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}")

        lines.append("")
        return lines

    def _generate_symbol_analysis(self, trades: List[Dict]) -> List[str]:
        """Generiert Symbol-basierte Analyse"""
        lines = []
        lines.append("🎯 PERFORMANCE BY SYMBOL")
        lines.append("-" * 100)

        trades_df = pd.DataFrame(trades)

        if 'symbol' in trades_df.columns:
            symbol_stats = trades_df.groupby('symbol').agg({
                'profit_loss': ['count', 'sum', 'mean'],
                'shares': 'sum'
            })

            lines.append(f"{'Symbol':<10} {'Trades':<10} {'Total P/L':<15} {'Avg P/L':<15} {'Total Shares':<15}")
            lines.append("-" * 100)

            for symbol in symbol_stats.index:
                count = int(symbol_stats.loc[symbol, ('profit_loss', 'count')])
                total_pl = symbol_stats.loc[symbol, ('profit_loss', 'sum')]
                avg_pl = symbol_stats.loc[symbol, ('profit_loss', 'mean')]
                total_shares = int(symbol_stats.loc[symbol, ('shares', 'sum')])

                lines.append(f"{symbol:<10} {count:<10} ${total_pl:<14,.2f} ${avg_pl:<14,.2f} {total_shares:<15,}")

        lines.append("")
        return lines

    def _generate_time_analysis(self, trades: List[Dict]) -> List[str]:
        """Generiert zeitbasierte Analyse"""
        lines = []
        lines.append("📅 TIME-BASED ANALYSIS")
        lines.append("-" * 100)

        trades_df = pd.DataFrame(trades)

        if 'date' in trades_df.columns and 'profit_loss' in trades_df.columns:
            try:
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                trades_df['month'] = trades_df['date'].dt.to_period('M')
                trades_df['weekday'] = trades_df['date'].dt.day_name()

                # Monthly Performance
                monthly = trades_df.groupby('month')['profit_loss'].sum()
                if len(monthly) > 0:
                    lines.append("Monthly Performance:")
                    for month, pnl in monthly.items():
                        emoji = "🟢" if pnl > 0 else "🔴"
                        lines.append(f"  {emoji} {month}: ${pnl:+,.2f}")

                # Day of Week Analysis
                weekday = trades_df.groupby('weekday')['profit_loss'].agg(['count', 'mean'])
                if len(weekday) > 0:
                    lines.append("\nPerformance by Day of Week:")
                    for day, stats in weekday.iterrows():
                        lines.append(f"  {day}: {int(stats['count'])} trades, Avg: ${stats['mean']:+,.2f}")

            except Exception as e:
                logger.debug(f"Zeit-Analyse fehlgeschlagen: {e}")

        lines.append("")
        return lines

    def _generate_config_summary(self, config: Dict) -> List[str]:
        """Generiert Konfigurations-Zusammenfassung"""
        lines = []
        lines.append("⚙️ CONFIGURATION")
        lines.append("-" * 100)

        if 'PORTFOLIO_CONFIG' in config:
            pc = config['PORTFOLIO_CONFIG']
            lines.append("Portfolio Settings:")
            lines.append(f"  Max Position Size: {pc.get('max_position_size', 0)*100:.0f}%")
            lines.append(f"  Max Positions: {pc.get('max_positions', 0)}")
            lines.append(f"  Min Cash Reserve: {pc.get('min_cash_reserve', 0)*100:.0f}%")

        if 'RISK_CONFIG' in config:
            rc = config['RISK_CONFIG']
            lines.append("\nRisk Management:")
            lines.append(f"  Stop Loss: {rc.get('stop_loss_percent', 0)*100:.0f}%")
            lines.append(f"  Take Profit: {rc.get('take_profit_percent', 0)*100:.0f}%")
            lines.append(f"  Trailing Stop: {rc.get('trailing_stop', False)}")

        if 'STRATEGY_CONFIG' in config:
            sc = config['STRATEGY_CONFIG']
            lines.append("\nStrategy Parameters:")
            lines.append(f"  SMA Short: {sc.get('short_window', 0)}")
            lines.append(f"  SMA Long: {sc.get('long_window', 0)}")
            lines.append(f"  RSI Period: {sc.get('rsi_period', 0)}")
            lines.append(f"  RSI Oversold: {sc.get('rsi_oversold', 0)}")
            lines.append(f"  RSI Overbought: {sc.get('rsi_overbought', 0)}")

        lines.append("")
        return lines

    def _generate_recommendations(
        self,
        portfolio_data: Dict,
        trades: List[Dict]
    ) -> List[str]:
        """Generiert Empfehlungen basierend auf Ergebnissen"""
        lines = []
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 100)

        metrics = portfolio_data.get('metrics', {})

        # Performance-basierte Empfehlungen
        if metrics.get('total_return', 0) < 0:
            lines.append("⚠️ Negative Returns:")
            lines.append("  → Consider adjusting strategy parameters")
            lines.append("  → Review signal quality and entry/exit criteria")

        # Sharpe Ratio
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < 1.0:
            lines.append("⚠️ Low Sharpe Ratio (<1.0):")
            lines.append("  → Risk-adjusted returns are suboptimal")
            lines.append("  → Consider reducing position sizes or improving signal quality")
        elif sharpe > 2.0:
            lines.append("✅ Excellent Sharpe Ratio (>2.0):")
            lines.append("  → Strong risk-adjusted performance")

        # Win Rate
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 0.4:
            lines.append("⚠️ Low Win Rate (<40%):")
            lines.append("  → Need higher win rate or better win/loss ratio")
            lines.append("  → Consider stricter entry criteria")
        elif win_rate > 0.6:
            lines.append("✅ Good Win Rate (>60%):")
            lines.append("  → Solid trade selection")

        # Profit Factor
        profit_factor = metrics.get('profit_factor', 0)
        if 0 < profit_factor < 1.5:
            lines.append("⚠️ Low Profit Factor (<1.5):")
            lines.append("  → Average wins not sufficiently larger than average losses")
            lines.append("  → Consider letting winners run longer or cutting losses earlier")
        elif profit_factor > 2.0:
            lines.append("✅ Strong Profit Factor (>2.0):")
            lines.append("  → Effective win/loss management")

        # Max Drawdown
        max_dd = metrics.get('max_drawdown', 0)
        if max_dd > 20:
            lines.append("⚠️ High Maximum Drawdown (>20%):")
            lines.append("  → Implement stricter risk management")
            lines.append("  → Consider reducing position sizes")

        # Kelly Criterion
        kelly = metrics.get('kelly_criterion', 0)
        if kelly > 0:
            lines.append(f"\n📊 Position Sizing Recommendation:")
            lines.append(f"  → Use {kelly*100:.1f}% of capital per trade (Kelly Criterion)")

        if not lines[2:]:  # Nur Header, keine Empfehlungen
            lines.append("✅ Overall performance looks solid!")
            lines.append("  → Continue monitoring and refining the strategy")

        lines.append("")
        return lines

    def generate_html_report(
        self,
        portfolio_data: Dict,
        trades: List[Dict],
        strategy_name: str,
        charts_dir: Optional[str] = None
    ) -> str:
        """
        Generiert HTML-Report (wenn möglich)

        Args:
            portfolio_data: Portfolio-Daten
            trades: Trades
            strategy_name: Strategy name
            charts_dir: Verzeichnis mit Charts

        Returns:
            Pfad zum HTML-Report
        """
        try:
            html_lines = []
            html_lines.append("<!DOCTYPE html>")
            html_lines.append("<html><head>")
            html_lines.append("<meta charset='UTF-8'>")
            html_lines.append(f"<title>Backtest Report - {strategy_name}</title>")
            html_lines.append("<style>")
            html_lines.append("body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }")
            html_lines.append("h1, h2 { color: #333; }")
            html_lines.append(".metric { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; }")
            html_lines.append("table { width: 100%; border-collapse: collapse; background: white; }")
            html_lines.append("th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }")
            html_lines.append("th { background: #4CAF50; color: white; }")
            html_lines.append(".positive { color: green; }")
            html_lines.append(".negative { color: red; }")
            html_lines.append("</style>")
            html_lines.append("</head><body>")

            html_lines.append(f"<h1>Backtest Report: {strategy_name}</h1>")
            html_lines.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

            # Performance Metrics
            html_lines.append("<div class='metric'>")
            html_lines.append("<h2>Performance Metrics</h2>")
            metrics = portfolio_data.get('metrics', {})
            html_lines.append("<table>")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    html_lines.append(f"<tr><td>{key}</td><td>{value:.2f}</td></tr>")
            html_lines.append("</table>")
            html_lines.append("</div>")

            # Embed Charts if available
            if charts_dir:
                charts_path = Path(charts_dir)
                for chart_file in charts_path.glob("*.png"):
                    html_lines.append(f"<img src='{chart_file.name}' width='100%' />")

            html_lines.append("</body></html>")

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.output_dir / f"backtest_report_{timestamp}.html"

            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_lines))

            logger.info(f"HTML Report gespeichert: {filename}")
            return str(filename)

        except Exception as e:
            logger.error(f"Fehler beim HTML-Report: {e}")
            return ""

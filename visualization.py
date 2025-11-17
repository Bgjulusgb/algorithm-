"""
Visualization Module für Trading Bot
Erstellt Charts und visuelle Analysen
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TradingVisualizer:
    """Erstellt Trading-Visualisierungen"""

    def __init__(self, output_dir: str = "charts"):
        """
        Initialisiert Visualizer

        Args:
            output_dir: Verzeichnis für Chart-Ausgabe
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_price_with_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        save: bool = True
    ) -> Optional[str]:
        """
        Plottet Preise mit Trading-Signalen

        Args:
            df: DataFrame mit Preisen und Signalen
            symbol: Aktiensymbol
            save: Chart speichern

        Returns:
            Pfad zur gespeicherten Datei oder None
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

            # Plot 1: Preis mit Moving Averages
            ax1.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1.5)

            if 'SMA_50' in df.columns:
                ax1.plot(df.index, df['SMA_50'], label='SMA 50', color='blue', alpha=0.7)
            if 'SMA_200' in df.columns:
                ax1.plot(df.index, df['SMA_200'], label='SMA 200', color='red', alpha=0.7)

            # Bollinger Bands
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'],
                                alpha=0.1, color='gray', label='Bollinger Bands')

            # Trading Signals
            if 'Position' in df.columns:
                buy_signals = df[df['Position'] > 0]
                sell_signals = df[df['Position'] < 0]

                ax1.scatter(buy_signals.index, buy_signals['Close'],
                           marker='^', color='green', s=100, label='Buy Signal', zorder=5)
                ax1.scatter(sell_signals.index, sell_signals['Close'],
                           marker='v', color='red', s=100, label='Sell Signal', zorder=5)

            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.set_title(f'{symbol} - Price Chart with Trading Signals', fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # Plot 2: RSI
            if 'RSI' in df.columns:
                ax2.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=1.5)
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
                ax2.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
                ax2.set_ylabel('RSI', fontsize=12)
                ax2.set_title('Relative Strength Index (RSI)', fontsize=12)
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([0, 100])

            # Plot 3: Volume
            if 'Volume' in df.columns:
                colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red'
                         for i in range(len(df))]
                ax3.bar(df.index, df['Volume'], color=colors, alpha=0.5)

                if 'Volume_MA' in df.columns:
                    ax3.plot(df.index, df['Volume_MA'], color='blue',
                            linewidth=2, label='Volume MA', alpha=0.7)

                ax3.set_ylabel('Volume', fontsize=12)
                ax3.set_xlabel('Date', fontsize=12)
                ax3.set_title('Trading Volume', fontsize=12)
                ax3.legend(loc='best')
                ax3.grid(True, alpha=0.3)

            # Format x-axis
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax3.xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)

            plt.tight_layout()

            if save:
                filename = self.output_dir / f"{symbol}_signals.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                logger.info(f"Chart gespeichert: {filename}")
                plt.close()
                return str(filename)
            else:
                plt.show()
                return None

        except ImportError:
            logger.warning("matplotlib nicht verfügbar")
            return None
        except Exception as e:
            logger.error(f"Fehler beim Plotten: {e}")
            return None

    def plot_portfolio_performance(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        initial_capital: float,
        save: bool = True
    ) -> Optional[str]:
        """
        Plottet Portfolio-Performance über Zeit

        Args:
            equity_curve: Equity Curve Serie
            trades: Liste von Trades
            initial_capital: Startkapital
            save: Chart speichern

        Returns:
            Pfad zur Datei oder None
        """
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

            # Plot 1: Equity Curve
            if not equity_curve.empty:
                ax1.plot(equity_curve.index, equity_curve.values,
                        color='blue', linewidth=2, label='Portfolio Value')
                ax1.axhline(y=initial_capital, color='gray',
                           linestyle='--', alpha=0.5, label='Initial Capital')

                # Drawdown
                running_max = equity_curve.cummax()
                drawdown = (equity_curve - running_max) / running_max * 100
                ax1_twin = ax1.twinx()
                ax1_twin.fill_between(drawdown.index, 0, drawdown.values,
                                     color='red', alpha=0.3, label='Drawdown %')
                ax1_twin.set_ylabel('Drawdown (%)', fontsize=12)
                ax1_twin.legend(loc='lower left')

            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.set_title('Portfolio Performance', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Trade Returns Distribution
            if trades:
                trades_df = pd.DataFrame(trades)
                sell_trades = trades_df[trades_df['action'] == 'SELL']

                if len(sell_trades) > 0 and 'profit_loss' in sell_trades.columns:
                    returns = sell_trades['profit_loss'].values

                    ax2.hist(returns, bins=30, color='steelblue',
                            edgecolor='black', alpha=0.7)
                    ax2.axvline(x=0, color='red', linestyle='--',
                               linewidth=2, label='Break Even')
                    ax2.axvline(x=np.mean(returns), color='green',
                               linestyle='--', linewidth=2, label=f'Mean: ${np.mean(returns):.2f}')

                    ax2.set_xlabel('Profit/Loss ($)', fontsize=12)
                    ax2.set_ylabel('Frequency', fontsize=12)
                    ax2.set_title('Trade Returns Distribution', fontsize=12)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save:
                filename = self.output_dir / "portfolio_performance.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                logger.info(f"Performance Chart gespeichert: {filename}")
                plt.close()
                return str(filename)
            else:
                plt.show()
                return None

        except ImportError:
            logger.warning("matplotlib nicht verfügbar")
            return None
        except Exception as e:
            logger.error(f"Fehler beim Portfolio-Plot: {e}")
            return None

    def plot_correlation_matrix(
        self,
        correlation_matrix: pd.DataFrame,
        save: bool = True
    ) -> Optional[str]:
        """
        Plottet Korrelationsmatrix als Heatmap

        Args:
            correlation_matrix: Korrelationsmatrix
            save: Chart speichern

        Returns:
            Pfad zur Datei oder None
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(10, 8))

            sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                       cmap='coolwarm', center=0, square=True,
                       linewidths=1, cbar_kws={"shrink": 0.8},
                       vmin=-1, vmax=1, ax=ax)

            ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save:
                filename = self.output_dir / "correlation_matrix.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                logger.info(f"Correlation Chart gespeichert: {filename}")
                plt.close()
                return str(filename)
            else:
                plt.show()
                return None

        except ImportError:
            logger.warning("matplotlib oder seaborn nicht verfügbar")
            return None
        except Exception as e:
            logger.error(f"Fehler beim Correlation-Plot: {e}")
            return None

    def plot_monte_carlo_simulation(
        self,
        simulations: np.ndarray,
        percentiles: Dict[str, float],
        save: bool = True
    ) -> Optional[str]:
        """
        Plottet Monte Carlo Simulation Results

        Args:
            simulations: Simulation Array (n_simulations x n_days)
            percentiles: Dictionary mit Percentile-Werten
            save: Chart speichern

        Returns:
            Pfad zur Datei oder None
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(15, 8))

            # Plot einige Simulation-Pfade (nicht alle, zu viel)
            num_paths_to_plot = min(100, simulations.shape[0])
            for i in range(num_paths_to_plot):
                ax.plot(simulations[i], color='gray', alpha=0.05, linewidth=0.5)

            # Plot Percentiles
            median_path = np.median(simulations, axis=0)
            p5_path = np.percentile(simulations, 5, axis=0)
            p95_path = np.percentile(simulations, 95, axis=0)

            ax.plot(median_path, color='blue', linewidth=2.5, label='Median (50th percentile)')
            ax.plot(p5_path, color='red', linewidth=2, linestyle='--', label='5th percentile')
            ax.plot(p95_path, color='green', linewidth=2, linestyle='--', label='95th percentile')

            # Fill between percentiles
            ax.fill_between(range(len(median_path)), p5_path, p95_path,
                           alpha=0.2, color='blue', label='90% Confidence Interval')

            ax.set_xlabel('Trading Days', fontsize=12)
            ax.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax.set_title('Monte Carlo Simulation - Portfolio Projections', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # Add text with final values
            textstr = f'Final Value Estimates:\n'
            textstr += f'5th percentile: ${percentiles.get("percentile_5", 0):,.0f}\n'
            textstr += f'Median: ${percentiles.get("percentile_50", 0):,.0f}\n'
            textstr += f'95th percentile: ${percentiles.get("percentile_95", 0):,.0f}'

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)

            plt.tight_layout()

            if save:
                filename = self.output_dir / "monte_carlo_simulation.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                logger.info(f"Monte Carlo Chart gespeichert: {filename}")
                plt.close()
                return str(filename)
            else:
                plt.show()
                return None

        except ImportError:
            logger.warning("matplotlib nicht verfügbar")
            return None
        except Exception as e:
            logger.error(f"Fehler beim Monte Carlo Plot: {e}")
            return None

    def create_dashboard(
        self,
        portfolio_data: Dict,
        trades: List[Dict],
        symbol_data: Dict[str, pd.DataFrame],
        save: bool = True
    ) -> Optional[str]:
        """
        Erstellt ein umfassendes Trading-Dashboard

        Args:
            portfolio_data: Portfolio-Informationen
            trades: Liste von Trades
            symbol_data: Dictionary mit DataFrames für jedes Symbol
            save: Dashboard speichern

        Returns:
            Pfad zur Datei oder None
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec

            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

            # Top-left: Portfolio Value
            ax1 = fig.add_subplot(gs[0, :2])
            if 'equity_curve' in portfolio_data and portfolio_data['equity_curve'] is not None:
                equity = portfolio_data['equity_curve']
                ax1.plot(equity.index, equity.values, color='blue', linewidth=2)
                ax1.fill_between(equity.index, equity.values, alpha=0.3, color='blue')
                ax1.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Value ($)')
                ax1.grid(True, alpha=0.3)

            # Top-right: Performance Metrics
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.axis('off')
            metrics_text = "Performance Metrics\n" + "="*30 + "\n"
            if 'metrics' in portfolio_data:
                metrics = portfolio_data['metrics']
                metrics_text += f"Total Return: {metrics.get('total_return', 0):.2f}%\n"
                metrics_text += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
                metrics_text += f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%\n"
                metrics_text += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
                metrics_text += f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%\n"
            ax2.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            # Middle-left: Trade Distribution
            ax3 = fig.add_subplot(gs[1, 0])
            if trades:
                trades_df = pd.DataFrame(trades)
                sell_trades = trades_df[trades_df['action'] == 'SELL']
                if len(sell_trades) > 0 and 'profit_loss' in sell_trades.columns:
                    wins = len(sell_trades[sell_trades['profit_loss'] > 0])
                    losses = len(sell_trades[sell_trades['profit_loss'] <= 0])
                    ax3.pie([wins, losses], labels=['Wins', 'Losses'],
                           colors=['green', 'red'], autopct='%1.1f%%',
                           startangle=90)
                    ax3.set_title('Win/Loss Distribution', fontsize=11, fontweight='bold')

            # Middle-center: Returns Distribution
            ax4 = fig.add_subplot(gs[1, 1])
            if trades:
                trades_df = pd.DataFrame(trades)
                sell_trades = trades_df[trades_df['action'] == 'SELL']
                if len(sell_trades) > 0 and 'profit_loss' in sell_trades.columns:
                    ax4.hist(sell_trades['profit_loss'], bins=20, color='steelblue',
                            edgecolor='black', alpha=0.7)
                    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
                    ax4.set_title('Returns Distribution', fontsize=11, fontweight='bold')
                    ax4.set_xlabel('P/L ($)')
                    ax4.set_ylabel('Frequency')
                    ax4.grid(True, alpha=0.3)

            # Middle-right: Monthly Returns
            ax5 = fig.add_subplot(gs[1, 2])
            if trades:
                trades_df = pd.DataFrame(trades)
                if 'date' in trades_df.columns:
                    trades_df['date'] = pd.to_datetime(trades_df['date'])
                    trades_df['month'] = trades_df['date'].dt.to_period('M')
                    monthly = trades_df.groupby('month')['profit_loss'].sum()
                    colors = ['green' if x > 0 else 'red' for x in monthly.values]
                    ax5.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
                    ax5.set_title('Monthly Returns', fontsize=11, fontweight='bold')
                    ax5.set_ylabel('P/L ($)')
                    ax5.grid(True, alpha=0.3)
                    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)

            # Bottom: Symbol Performance Comparison
            ax6 = fig.add_subplot(gs[2, :])
            if symbol_data and len(symbol_data) > 0:
                for symbol, df in symbol_data.items():
                    if 'Close' in df.columns and len(df) > 0:
                        # Normalize to percentage change
                        normalized = (df['Close'] / df['Close'].iloc[0] - 1) * 100
                        ax6.plot(df.index, normalized, label=symbol, linewidth=1.5)

                ax6.set_title('Symbol Performance Comparison (Normalized)', fontsize=11, fontweight='bold')
                ax6.set_ylabel('Return (%)')
                ax6.set_xlabel('Date')
                ax6.legend(loc='best')
                ax6.grid(True, alpha=0.3)
                ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)

            plt.suptitle('Trading Bot Dashboard', fontsize=16, fontweight='bold', y=0.995)

            if save:
                filename = self.output_dir / "trading_dashboard.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                logger.info(f"Dashboard gespeichert: {filename}")
                plt.close()
                return str(filename)
            else:
                plt.show()
                return None

        except ImportError:
            logger.warning("matplotlib nicht verfügbar")
            return None
        except Exception as e:
            logger.error(f"Fehler beim Dashboard erstellen: {e}")
            import traceback
            traceback.print_exc()
            return None

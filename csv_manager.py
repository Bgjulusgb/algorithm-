"""
CSV Manager für Trading-Daten Export
Verwaltet Trading-Logs und Export für Yahoo Finance Portfolio
"""
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import logging

from config import FILE_PATHS

logger = logging.getLogger(__name__)


class CSVManager:
    """Manager für CSV-Export von Trading-Daten"""

    def __init__(self, trades_file: Optional[str] = None):
        """
        Initialisiert CSV Manager

        Args:
            trades_file: Pfad zur Trades-CSV-Datei
        """
        self.trades_file = trades_file or FILE_PATHS.get("trades_csv", "trades.csv")
        self.trades: List[Dict] = []
        self._init_csv_file()

    def _init_csv_file(self):
        """Initialisiert CSV-Datei mit Headers"""
        trades_path = Path(self.trades_file)

        if not trades_path.exists():
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Symbol', 'Trade Date', 'Action', 'Quantity',
                    'Price', 'Commission', 'Notes'
                ])
            logger.info(f"CSV-Datei erstellt: {self.trades_file}")

    def add_buy(
        self,
        symbol: str,
        date: datetime,
        quantity: int,
        price: float,
        commission: float = 0.0,
        notes: str = ""
    ):
        """
        Fügt Kauf-Trade hinzu

        Args:
            symbol: Aktiensymbol
            date: Handelsdatum
            quantity: Anzahl Aktien
            price: Kaufpreis
            commission: Kommission
            notes: Notizen
        """
        trade = {
            'Symbol': symbol,
            'Trade Date': date.strftime('%m/%d/%Y') if isinstance(date, datetime) else str(date),
            'Action': 'Buy',
            'Quantity': quantity,
            'Price': f"{price:.2f}",
            'Commission': f"{commission:.2f}",
            'Notes': notes
        }
        self.trades.append(trade)
        self._append_to_csv(trade)
        logger.debug(f"Trade hinzugefügt: BUY {quantity} {symbol} @ ${price:.2f}")

    def add_sell(
        self,
        symbol: str,
        date: datetime,
        quantity: int,
        price: float,
        commission: float = 0.0,
        notes: str = ""
    ):
        """
        Fügt Verkaufs-Trade hinzu

        Args:
            symbol: Aktiensymbol
            date: Handelsdatum
            quantity: Anzahl Aktien
            price: Verkaufspreis
            commission: Kommission
            notes: Notizen
        """
        trade = {
            'Symbol': symbol,
            'Trade Date': date.strftime('%m/%d/%Y') if isinstance(date, datetime) else str(date),
            'Action': 'Sell',
            'Quantity': quantity,
            'Price': f"{price:.2f}",
            'Commission': f"{commission:.2f}",
            'Notes': notes
        }
        self.trades.append(trade)
        self._append_to_csv(trade)
        logger.debug(f"Trade hinzugefügt: SELL {quantity} {symbol} @ ${price:.2f}")

    def _append_to_csv(self, trade: Dict):
        """Fügt Trade zur CSV-Datei hinzu"""
        try:
            with open(self.trades_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trade.keys())
                writer.writerow(trade)
        except Exception as e:
            logger.error(f"Fehler beim Schreiben in CSV: {e}")

    def export_for_yahoo(self, output_file: Optional[str] = None) -> str:
        """
        Exportiert Trades im Yahoo Finance Format

        Args:
            output_file: Ausgabedatei (optional)

        Returns:
            Pfad zur exportierten Datei
        """
        output_file = output_file or "yahoo_finance_portfolio.csv"

        try:
            if not self.trades:
                logger.warning("Keine Trades zum Exportieren")
                return output_file

            # Konvertiere zu DataFrame
            df = pd.DataFrame(self.trades)

            # Yahoo Finance Format
            df_yahoo = df.rename(columns={
                'Trade Date': 'Trade Date',
                'Action': 'Action',
                'Symbol': 'Symbol',
                'Quantity': 'Quantity',
                'Price': 'Price',
                'Commission': 'Commission',
                'Notes': 'Notes'
            })

            # Exportiere
            df_yahoo.to_csv(output_file, index=False)
            logger.info(f"✅ Trades exportiert nach: {output_file}")

            return output_file

        except Exception as e:
            logger.error(f"Fehler beim Export: {e}")
            return output_file

    def print_summary(self):
        """Gibt Zusammenfassung der Trades aus"""
        if not self.trades:
            logger.info("Keine Trades vorhanden")
            return

        buy_trades = [t for t in self.trades if t['Action'].lower() == 'buy']
        sell_trades = [t for t in self.trades if t['Action'].lower() == 'sell']

        logger.info("\n" + "="*70)
        logger.info("TRADING ZUSAMMENFASSUNG")
        logger.info("="*70)
        logger.info(f"📊 Gesamt Trades: {len(self.trades)}")
        logger.info(f"🟢 Käufe: {len(buy_trades)}")
        logger.info(f"🔴 Verkäufe: {len(sell_trades)}")
        logger.info(f"📁 Gespeichert in: {self.trades_file}")
        logger.info("="*70)

    def get_trades(self) -> List[Dict]:
        """Gibt alle Trades zurück"""
        return self.trades

    def clear_trades(self):
        """Löscht alle Trades"""
        self.trades = []
        self._init_csv_file()
        logger.info("Trades gelöscht")


class PerformanceLogger:
    """Logger für Performance-Metriken"""

    def __init__(self, performance_file: Optional[str] = None):
        """
        Initialisiert Performance Logger

        Args:
            performance_file: Pfad zur Performance-CSV-Datei
        """
        self.performance_file = performance_file or FILE_PATHS.get("performance_csv", "performance.csv")
        self.metrics: List[Dict] = []
        self._init_performance_file()

    def _init_performance_file(self):
        """Initialisiert Performance-CSV-Datei"""
        perf_path = Path(self.performance_file)

        if not perf_path.exists():
            with open(self.performance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'Total Value', 'Cash', 'Positions Value',
                    'Daily Return %', 'Total Return %', 'Sharpe Ratio',
                    'Max Drawdown %', 'Open Positions'
                ])
            logger.info(f"Performance-Datei erstellt: {self.performance_file}")

    def log_performance(
        self,
        total_value: float,
        cash: float,
        positions_value: float,
        daily_return: float = 0.0,
        total_return: float = 0.0,
        sharpe_ratio: float = 0.0,
        max_drawdown: float = 0.0,
        open_positions: int = 0
    ):
        """
        Loggt Performance-Metriken

        Args:
            total_value: Gesamtwert Portfolio
            cash: Cash-Bestand
            positions_value: Wert aller Positionen
            daily_return: Tägliche Rendite
            total_return: Gesamtrendite
            sharpe_ratio: Sharpe Ratio
            max_drawdown: Maximaler Drawdown
            open_positions: Anzahl offener Positionen
        """
        metric = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Total Value': f"{total_value:.2f}",
            'Cash': f"{cash:.2f}",
            'Positions Value': f"{positions_value:.2f}",
            'Daily Return %': f"{daily_return:.2f}",
            'Total Return %': f"{total_return:.2f}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown %': f"{max_drawdown:.2f}",
            'Open Positions': open_positions
        }
        self.metrics.append(metric)
        self._append_to_performance(metric)

    def _append_to_performance(self, metric: Dict):
        """Fügt Metrik zur Performance-Datei hinzu"""
        try:
            with open(self.performance_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metric.keys())
                writer.writerow(metric)
        except Exception as e:
            logger.error(f"Fehler beim Schreiben von Performance-Daten: {e}")

    def get_metrics(self) -> pd.DataFrame:
        """
        Gibt Performance-Metriken als DataFrame zurück

        Returns:
            DataFrame mit Performance-Daten
        """
        try:
            return pd.read_csv(self.performance_file)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Performance-Daten: {e}")
            return pd.DataFrame()

    def plot_performance(self, output_file: str = "performance_chart.png"):
        """
        Erstellt Performance-Chart (benötigt matplotlib)

        Args:
            output_file: Ausgabedatei für Chart
        """
        try:
            import matplotlib.pyplot as plt

            df = self.get_metrics()
            if df.empty:
                logger.warning("Keine Performance-Daten zum Plotten")
                return

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Trading Bot Performance', fontsize=16)

            # Total Value
            axes[0, 0].plot(df['Timestamp'], df['Total Value'])
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Value ($)')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Returns
            axes[0, 1].plot(df['Timestamp'], df['Total Return %'])
            axes[0, 1].set_title('Total Return %')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Return (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Drawdown
            axes[1, 0].plot(df['Timestamp'], df['Max Drawdown %'], color='red')
            axes[1, 0].set_title('Max Drawdown %')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Drawdown (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Sharpe Ratio
            axes[1, 1].plot(df['Timestamp'], df['Sharpe Ratio'], color='green')
            axes[1, 1].set_title('Sharpe Ratio')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(output_file)
            logger.info(f"Performance-Chart gespeichert: {output_file}")

        except ImportError:
            logger.warning("matplotlib nicht installiert - kann keinen Chart erstellen")
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Performance-Charts: {e}")

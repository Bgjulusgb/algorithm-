"""
Trading Bot Hauptprogramm - Verbesserte Version
Mit erweitertem Logging, Error Handling und Performance-Optimierungen
"""
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from typing import Optional, List
import traceback

from config import (
    WATCHLIST, TRADING_MODE, PORTFOLIO_CONFIG, FILE_PATHS,
    LOGGING_CONFIG, BACKTEST_CONFIG, validate_config
)
from data_handler import DataHandler
from strategy import StrategyFactory
from portfolio import Portfolio
from csv_manager import CSVManager, PerformanceLogger


# Setup Logging
def setup_logging():
    """Konfiguriert Logging"""
    log_format = LOGGING_CONFIG.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_level = getattr(logging, LOGGING_CONFIG.get("log_level", "INFO"))
    
    handlers = []
    
    # Console Handler
    if LOGGING_CONFIG.get("log_to_console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)
    
    # File Handler
    if LOGGING_CONFIG.get("log_to_file", True):
        log_file = FILE_PATHS.get("log_file", "trading_bot.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

setup_logging()
logger = logging.getLogger(__name__)


class TradingBot:
    """Erweiterter Trading Bot mit verbessertem Error Handling"""
    
    def __init__(
        self, 
        strategy_name: str = "sma", 
        mode: str = "backtest",
        watchlist: Optional[List[str]] = None
    ):
        """
        Initialisiert den Trading Bot
        
        Args:
            strategy_name: Name der Strategie
            mode: "backtest" oder "live"
            watchlist: Liste von Symbolen (optional)
        """
        self.mode = mode
        self.watchlist = watchlist or WATCHLIST
        
        try:
            self.data_handler = DataHandler()
            self.strategy = StrategyFactory.create_strategy(strategy_name)
            self.portfolio = Portfolio()
            self.csv_manager = CSVManager()
            self.performance_logger = PerformanceLogger()
            
            logger.info("="*70)
            logger.info("🤖 TRADING BOT GESTARTET")
            logger.info("="*70)
            logger.info(f"Modus: {self.mode.upper()}")
            logger.info(f"Strategie: {self.strategy.name}")
            logger.info(f"Watchlist: {', '.join(self.watchlist)}")
            logger.info(f"Startkapital: ${PORTFOLIO_CONFIG['initial_capital']:,.2f}")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"Fehler bei Initialisierung: {e}")
            raise
    
    def backtest(
        self, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None
    ):
        """
        Führt erweiterten Backtest durch
        
        Args:
            start_date: Startdatum
            end_date: Enddatum
        """
        logger.info("📊 Starte Backtest...")
        
        successful_symbols = 0
        failed_symbols = []
        
        for i, symbol in enumerate(self.watchlist, 1):
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"[{i}/{len(self.watchlist)}] Analysiere {symbol}")
                logger.info(f"{'='*70}")
                
                # Lade historische Daten
                df = self.data_handler.get_historical_data(symbol)
                
                if df is None or df.empty:
                    logger.warning(f"⚠️ Überspringe {symbol} - keine Daten")
                    failed_symbols.append(symbol)
                    continue
                
                # Füge technische Indikatoren hinzu
                df = self.data_handler.add_technical_indicators(df)
                
                # Generiere Signale
                df = self.strategy.generate_signals(df)
                
                # Simuliere Trades
                trades_executed = self._simulate_trades(symbol, df)
                
                if trades_executed > 0:
                    successful_symbols += 1
                    logger.info(f"✅ {trades_executed} Trades ausgeführt für {symbol}")
                else:
                    logger.info(f"ℹ️ Keine Trades für {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Fehler bei {symbol}: {e}")
                logger.debug(traceback.format_exc())
                failed_symbols.append(symbol)
                continue
        
        # Zusammenfassung
        logger.info("\n" + "="*70)
        logger.info("🏁 BACKTEST ABGESCHLOSSEN")
        logger.info("="*70)
        logger.info(f"Erfolgreich analysiert: {successful_symbols}/{len(self.watchlist)}")
        
        if failed_symbols:
            logger.warning(f"Fehlgeschlagen: {', '.join(failed_symbols)}")
        
        self._print_final_summary()
    
    def _simulate_trades(self, symbol: str, df: pd.DataFrame) -> int:
        """
        Simuliert Trades mit verbesserter Logik
        
        Args:
            symbol: Aktiensymbol
            df: DataFrame mit Signalen
        
        Returns:
            Anzahl ausgeführter Trades
        """
        trades_executed = 0
        min_confidence = 0.6  # Mindest-Konfidenz für Trades
        
        for idx, row in df.iterrows():
            # Prüfe auf gültiges Signal
            if pd.isna(row.get('Position', 0)) or row.get('Position', 0) == 0:
                continue
            
            signal = row['Position']
            confidence = row.get('Confidence', 0.5)
            current_price = row['Close']
            
            # Prüfe Konfidenz
            if confidence < min_confidence:
                logger.debug(f"Signal für {symbol} übersprungen: Konfidenz zu niedrig ({confidence:.2f} < {min_confidence})")
                continue
            
            # BUY Signal
            if signal > 0:
                # Berechne Positionsgröße
                total_value = self.portfolio.get_total_value({symbol: current_price})
                shares = self.portfolio.calculate_position_size(symbol, current_price)
                
                if shares > 0:
                    success = self.portfolio.buy(symbol, current_price, shares, date=idx)
                    if success:
                        self.csv_manager.add_buy(
                            symbol, idx, shares, current_price,
                            notes=f"{self.strategy.name} Buy (conf={confidence:.2f})"
                        )
                        trades_executed += 1
            
            # SELL Signal
            elif signal < 0:
                if self.portfolio.has_position(symbol):
                    position = self.portfolio.get_position(symbol)
                    shares = position.shares
                    
                    success = self.portfolio.sell(
                        symbol, current_price, shares, date=idx,
                        reason=f"{self.strategy.name} Sell Signal"
                    )
                    if success:
                        self.csv_manager.add_sell(
                            symbol, idx, shares, current_price,
                            notes=f"{self.strategy.name} Sell (conf={confidence:.2f})"
                        )
                        trades_executed += 1
            
            # Risk Management Check
            current_prices = {symbol: current_price}
            self.portfolio.check_risk_management(current_prices)
        
        return trades_executed
    
    def live_trading(self):
        """
        Führt Live-Trading durch (verbessert)
        """
        logger.info("🔴 LIVE TRADING MODUS")
        logger.warning("⚠️ Dies ist eine Simulation. Für echtes Trading müsste eine Broker-API integriert werden.")
        
        try:
            # Hole alle aktuellen Preise auf einmal (effizienter)
            current_prices = self.data_handler.get_multiple_current_prices(self.watchlist)
            
            for symbol in self.watchlist:
                try:
                    logger.info(f"\n{'='*70}")
                    logger.info(f"Prüfe {symbol}")
                    logger.info(f"{'='*70}")
                    
                    # Hole aktuelle Daten
                    df = self.data_handler.get_historical_data(symbol, period="6mo")
                    
                    if df is None or df.empty:
                        logger.warning(f"⚠️ Überspringe {symbol}")
                        continue
                    
                    # Technische Indikatoren
                    df = self.data_handler.add_technical_indicators(df)
                    
                    # Generiere Signale
                    df = self.strategy.generate_signals(df)
                    
                    # Aktueller Preis und Signal
                    current_price = current_prices.get(symbol) or df['Close'].iloc[-1]
                    current_signal = df['Position'].iloc[-1] if not pd.isna(df['Position'].iloc[-1]) else 0
                    current_confidence = df.get('Confidence', pd.Series([0.5] * len(df))).iloc[-1]
                    
                    # Zeige Daten
                    logger.info(f"Aktueller Preis: ${current_price:.2f}")
                    logger.info(f"SMA 50: ${df['SMA_50'].iloc[-1]:.2f}")
                    logger.info(f"SMA 200: ${df['SMA_200'].iloc[-1]:.2f}")
                    logger.info(f"RSI: {df['RSI'].iloc[-1]:.2f}")
                    logger.info(f"Signal Konfidenz: {current_confidence:.2f}")
                    
                    # Trading Entscheidung
                    if current_signal > 0 and current_confidence >= 0.6:
                        logger.info("📈 STARKES BUY SIGNAL!")
                        total_value = self.portfolio.get_total_value(current_prices)
                        shares = self.portfolio.calculate_position_size(symbol, current_price)
                        
                        if shares > 0:
                            success = self.portfolio.buy(symbol, current_price, shares)
                            if success:
                                self.csv_manager.add_buy(
                                    symbol, datetime.now(), shares, current_price,
                                    notes=f"{self.strategy.name} Live Buy (conf={current_confidence:.2f})"
                                )
                    
                    elif current_signal < 0 and current_confidence >= 0.6:
                        logger.info("📉 STARKES SELL SIGNAL!")
                        if self.portfolio.has_position(symbol):
                            position = self.portfolio.get_position(symbol)
                            success = self.portfolio.sell(
                                symbol, current_price, position.shares,
                                reason=f"{self.strategy.name} Live Sell"
                            )
                            if success:
                                self.csv_manager.add_sell(
                                    symbol, datetime.now(), position.shares, current_price,
                                    notes=f"{self.strategy.name} Live Sell (conf={current_confidence:.2f})"
                                )
                    else:
                        logger.info("➡️ HOLD - Kein ausreichend starkes Signal")
                    
                    # Risk Management
                    self.portfolio.check_risk_management({symbol: current_price})
                
                except Exception as e:
                    logger.error(f"Fehler bei {symbol}: {e}")
                    logger.debug(traceback.format_exc())
                    continue
            
            # Zusammenfassung
            self._print_final_summary()
        
        except Exception as e:
            logger.error(f"Kritischer Fehler im Live Trading: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def _print_final_summary(self):
        """Gibt finale Zusammenfassung aus"""
        logger.info("\n" + "="*70)
        logger.info("FINALE ZUSAMMENFASSUNG")
        logger.info("="*70)

        # Portfolio Summary (erweitert wenn verfügbar)
        try:
            self.portfolio.print_advanced_summary()
        except Exception:
            self.portfolio.print_summary()

        # CSV Summary
        self.csv_manager.print_summary()

        # Exportiere für Yahoo Finance
        try:
            output_file = self.csv_manager.export_for_yahoo()
            logger.info(f"\n✅ Trading-Daten exportiert nach: {output_file}")
            logger.info("💡 Diese Datei kann direkt bei Yahoo Finance hochgeladen werden!")
        except Exception as e:
            logger.error(f"Fehler beim CSV-Export: {e}")

        # Erweiterte Performance-Analytics (CSV-Export)
        try:
            self.portfolio.export_performance_report("performance_report")
        except Exception as e:
            logger.debug(f"Performance-Report-Export nicht verfügbar: {e}")

        # Speichere Portfolio-Status
        try:
            self.portfolio.save_state()
            logger.info("✅ Portfolio-Status gespeichert")
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Portfolio-Status: {e}")
    
    def run(self):
        """Startet den Bot im konfigurierten Modus"""
        try:
            if self.mode == "backtest":
                self.backtest()
            elif self.mode == "live":
                self.live_trading()
            else:
                logger.error(f"❌ Unbekannter Modus: {self.mode}")
                return
            
            logger.info("\n✅ Bot erfolgreich beendet!")
        
        except KeyboardInterrupt:
            logger.warning("\n\n⚠️ Bot gestoppt durch Benutzer")
            self._print_final_summary()
        
        except Exception as e:
            logger.error(f"\n❌ Kritischer Fehler: {e}")
            logger.debug(traceback.format_exc())
            raise


def print_banner():
    """Gibt Start-Banner aus"""
    print("\n" + "="*70)
    print(" "*20 + "YAHOO FINANCE TRADING BOT V2")
    print(" "*25 + "Verbesserte Version")
    print("="*70)


def select_strategy() -> str:
    """Interaktive Strategie-Auswahl"""
    print("\n📊 Verfügbare Strategien:")
    strategies = [
        ("1", "sma", "SMA Crossover (Golden Cross / Death Cross)"),
        ("2", "rsi", "RSI (Überkauft/Überverkauft)"),
        ("3", "macd", "MACD (Momentum)"),
        ("4", "combined", "Combined (Mehrere Indikatoren)"),
        ("5", "mean_reversion", "Mean Reversion (Bollinger Bands)"),
    ]
    
    for num, code, desc in strategies:
        print(f"  {num}. {desc}")
    
    print()
    choice = input("Wähle Strategie (1-5) [Standard: 4]: ").strip() or "4"
    
    strategy_map = {s[0]: s[1] for s in strategies}
    return strategy_map.get(choice, "combined")


def select_mode() -> str:
    """Interaktive Modus-Auswahl"""
    print("\n⚙️ Modus:")
    print("  1. Backtest (Historische Daten)")
    print("  2. Live Trading (Aktuelle Daten - Simulation)")
    print()
    
    mode_choice = input("Wähle Modus (1-2) [Standard: 1]: ").strip() or "1"
    return "backtest" if mode_choice == "1" else "live"


def select_watchlist() -> Optional[List[str]]:
    """Interaktive Watchlist-Auswahl"""
    print("\n📈 Watchlist:")
    print(f"  Standard: {', '.join(WATCHLIST)}")
    print()
    
    custom = input("Eigene Watchlist eingeben? (y/n) [n]: ").strip().lower()
    
    if custom == 'y':
        symbols_input = input("Symbole eingeben (Komma-getrennt): ").strip()
        if symbols_input:
            return [s.strip().upper() for s in symbols_input.split(',')]
    
    return None


def main():
    """Hauptfunktion mit verbessertem UI"""
    try:
        # Banner
        print_banner()
        
        # Validiere Konfiguration
        print("\n🔍 Validiere Konfiguration...")
        validate_config()
        
        # Interaktive Auswahl
        strategy = select_strategy()
        mode = select_mode()
        watchlist = select_watchlist()
        
        # Bestätigung
        print("\n" + "="*70)
        print("KONFIGURATION:")
        print("="*70)
        print(f"Strategie: {strategy}")
        print(f"Modus: {mode}")
        print(f"Watchlist: {', '.join(watchlist if watchlist else WATCHLIST)}")
        print(f"Startkapital: ${PORTFOLIO_CONFIG['initial_capital']:,.2f}")
        print("="*70)
        print()
        
        confirm = input("Fortfahren? (y/n) [y]: ").strip().lower() or 'y'
        
        if confirm != 'y':
            print("❌ Abgebrochen")
            return
        
        # Starte Bot
        bot = TradingBot(
            strategy_name=strategy, 
            mode=mode,
            watchlist=watchlist
        )
        bot.run()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Programm abgebrochen")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

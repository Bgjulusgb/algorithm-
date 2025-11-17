"""
Continuous Trading Bot - Automatischer Trading Bot mit Live-Scanning
Scannt kontinuierlich Märkte, generiert Signale und führt Trades aus
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import json

# Imports für Trading Bot Components
from config import WATCHLIST, PORTFOLIO_CONFIG, STRATEGY_CONFIG, RISK_CONFIG
from portfolio import Portfolio
from csv_manager import CSVManager

# Advanced Components
try:
    from advanced_indicators import AllIndicators
    from pattern_recognition import PatternAnalyzer
    from multi_timeframe import MultiTimeframeAnalyzer
    from advanced_orders import OrderManager, AdvancedRiskManager, OrderSide
    from execution_simulator import ExecutionSimulator, MarketCondition
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False

# ML Components (optional)
try:
    from ml_integration import MLTradingModel, EnsembleMLModel
    HAS_ML = True
except ImportError:
    HAS_ML = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveMarketScanner:
    """
    Live Market Scanner
    Scannt kontinuierlich Märkte und sammelt Daten
    """

    def __init__(self,
                 symbols: List[str],
                 scan_interval_seconds: int = 60,
                 use_cache: bool = True):
        """
        Initialisiert Live Market Scanner

        Args:
            symbols: Liste von Symbolen zum Scannen
            scan_interval_seconds: Sekunden zwischen Scans
            use_cache: Cache verwenden für Datenabfrage
        """
        self.symbols = symbols
        self.scan_interval = scan_interval_seconds
        self.use_cache = use_cache
        self.last_scan_time = {}
        self.market_data = {}

        if HAS_ADVANCED:
            self.mtf_analyzer = MultiTimeframeAnalyzer(use_cache=use_cache)

        logger.info(f"🔍 Live Market Scanner initialisiert für {len(symbols)} Symbole")

    def scan_symbol(self, symbol: str, timeframes: List[str] = ['1d']) -> Optional[Dict]:
        """
        Scannt einzelnes Symbol

        Args:
            symbol: Symbol zum Scannen
            timeframes: Liste von Timeframes

        Returns:
            Dict mit Marktdaten und Analyse
        """
        try:
            logger.info(f"📊 Scanne {symbol}...")

            if HAS_ADVANCED:
                # Multi-Timeframe Analyse
                analysis = self.mtf_analyzer.analyze_symbol(
                    symbol,
                    timeframes=timeframes,
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )

                if 'error' in analysis:
                    logger.warning(f"Keine Daten für {symbol}")
                    return None

                # Speichere in Cache
                self.market_data[symbol] = analysis
                self.last_scan_time[symbol] = datetime.now()

                logger.info(f"✅ {symbol}: {analysis.get('recommendation', 'N/A')}")

                return analysis
            else:
                # Fallback: Einfache Datenabfrage
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='1mo', interval='1d')

                if df.empty:
                    return None

                self.market_data[symbol] = {'data': df}
                self.last_scan_time[symbol] = datetime.now()

                return {'data': df}

        except Exception as e:
            logger.error(f"Fehler beim Scannen von {symbol}: {e}")
            return None

    def scan_all_symbols(self, timeframes: List[str] = ['1d', '1h']) -> Dict[str, Dict]:
        """
        Scannt alle Symbole

        Args:
            timeframes: Liste von Timeframes

        Returns:
            Dict mit allen Analysen {symbol: analysis}
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 STARTE MARKET SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")

        results = {}

        for symbol in self.symbols:
            analysis = self.scan_symbol(symbol, timeframes)
            if analysis:
                results[symbol] = analysis

            # Kleine Pause zwischen Symbolen um API Limits zu vermeiden
            time.sleep(1)

        logger.info(f"\n✅ Scan abgeschlossen: {len(results)}/{len(self.symbols)} Symbole erfolgreich")

        return results

    def should_scan(self, symbol: str) -> bool:
        """
        Prüft ob Symbol gescannt werden sollte

        Args:
            symbol: Symbol

        Returns:
            True wenn Scan notwendig
        """
        if symbol not in self.last_scan_time:
            return True

        time_since_scan = (datetime.now() - self.last_scan_time[symbol]).total_seconds()
        return time_since_scan >= self.scan_interval


class SignalGenerator:
    """
    Signal Generator
    Generiert Trading-Signale aus Marktdaten
    """

    def __init__(self, use_ml: bool = False, use_advanced: bool = True):
        """
        Initialisiert Signal Generator

        Args:
            use_ml: ML Models verwenden
            use_advanced: Advanced Indicators & Patterns verwenden
        """
        self.use_ml = use_ml and HAS_ML
        self.use_advanced = use_advanced and HAS_ADVANCED

        if self.use_ml:
            logger.info("🤖 ML Signal Generation aktiviert")

        if self.use_advanced:
            logger.info("📊 Advanced Technical Analysis aktiviert")

    def generate_signal(self, symbol: str, market_data: Dict) -> Dict:
        """
        Generiert Trading-Signal für Symbol

        Args:
            symbol: Symbol
            market_data: Marktdaten und Analyse

        Returns:
            Dict mit Signal und Confidence
        """
        signals = []
        confidences = []

        # 1. Multi-Timeframe Signal
        if 'recommendation' in market_data:
            recommendation = market_data['recommendation']

            if 'STRONG BUY' in recommendation:
                signals.append(1)
                confidences.append(0.9)
            elif 'BUY' in recommendation:
                signals.append(1)
                confidences.append(0.7)
            elif 'STRONG SELL' in recommendation:
                signals.append(-1)
                confidences.append(0.9)
            elif 'SELL' in recommendation:
                signals.append(-1)
                confidences.append(0.7)
            else:
                signals.append(0)
                confidences.append(0.5)

        # 2. Trend Alignment Signal
        if 'trend_alignment' in market_data:
            alignment = market_data['trend_alignment']['alignment']

            if 'STRONG_BULLISH' in alignment:
                signals.append(1)
                confidences.append(0.85)
            elif 'BULLISH' in alignment:
                signals.append(1)
                confidences.append(0.65)
            elif 'STRONG_BEARISH' in alignment:
                signals.append(-1)
                confidences.append(0.85)
            elif 'BEARISH' in alignment:
                signals.append(-1)
                confidences.append(0.65)

        # 3. Confluence Signal
        if 'confluence' in market_data:
            confluence = market_data['confluence']['overall_signal']

            if 'STRONG_BUY' in confluence:
                signals.append(1)
                confidences.append(0.9)
            elif 'BUY' in confluence:
                signals.append(1)
                confidences.append(0.7)
            elif 'STRONG_SELL' in confluence:
                signals.append(-1)
                confidences.append(0.9)
            elif 'SELL' in confluence:
                signals.append(-1)
                confidences.append(0.7)

        # Aggregate Signals
        if signals:
            # Gewichteter Durchschnitt
            avg_signal = np.average(signals, weights=confidences)
            avg_confidence = np.mean(confidences)

            # Final Signal
            if avg_signal > 0.3:
                final_signal = 'BUY'
                strength = 'STRONG' if avg_signal > 0.6 else 'MODERATE'
            elif avg_signal < -0.3:
                final_signal = 'SELL'
                strength = 'STRONG' if avg_signal < -0.6 else 'MODERATE'
            else:
                final_signal = 'HOLD'
                strength = 'NEUTRAL'

            return {
                'symbol': symbol,
                'signal': final_signal,
                'strength': strength,
                'confidence': avg_confidence,
                'raw_score': avg_signal,
                'num_signals': len(signals),
                'timestamp': datetime.now()
            }
        else:
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'strength': 'NEUTRAL',
                'confidence': 0.0,
                'raw_score': 0.0,
                'num_signals': 0,
                'timestamp': datetime.now()
            }


class TradeExecutor:
    """
    Trade Executor
    Führt Trades basierend auf Signalen automatisch aus
    """

    def __init__(self,
                 portfolio: Portfolio,
                 order_manager: Optional['OrderManager'] = None,
                 risk_manager: Optional['AdvancedRiskManager'] = None,
                 csv_manager: Optional[CSVManager] = None,
                 min_confidence: float = 0.6):
        """
        Initialisiert Trade Executor

        Args:
            portfolio: Portfolio Instance
            order_manager: Order Manager (optional)
            risk_manager: Risk Manager (optional)
            csv_manager: CSV Manager für Exports
            min_confidence: Mindest-Confidence für Trade-Ausführung
        """
        self.portfolio = portfolio
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        self.csv_manager = csv_manager or CSVManager()
        self.min_confidence = min_confidence

        logger.info(f"⚙️ Trade Executor initialisiert (Min Confidence: {min_confidence*100:.0f}%)")

    def execute_signal(self, signal: Dict, current_price: float) -> Optional[Dict]:
        """
        Führt Trade basierend auf Signal aus

        Args:
            signal: Signal Dict
            current_price: Aktueller Preis

        Returns:
            Dict mit Trade-Details oder None
        """
        symbol = signal['symbol']
        action = signal['signal']
        confidence = signal['confidence']

        # Prüfe Confidence
        if confidence < self.min_confidence:
            logger.info(f"⚠️ {symbol}: Confidence zu niedrig ({confidence*100:.0f}% < {self.min_confidence*100:.0f}%)")
            return None

        # BUY Signal
        if action == 'BUY':
            # Prüfe ob bereits Position
            if symbol in self.portfolio.positions:
                logger.info(f"⚠️ {symbol}: Position bereits vorhanden")
                return None

            # Berechne Position Size
            if self.risk_manager and HAS_ADVANCED:
                # Dynamische Stop-Loss berechnung
                atr = current_price * 0.02  # Approximation: 2% ATR
                stop_loss = self.risk_manager.calculate_dynamic_stop_loss(
                    entry_price=current_price,
                    atr=atr,
                    side='long'
                )

                shares = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    portfolio_value=self.portfolio.get_total_value(),
                    confidence=confidence
                )
            else:
                # Standard Position Size
                shares = self.portfolio.calculate_position_size(symbol, current_price)

            if shares > 0:
                # Ausführen
                success = self.portfolio.buy(
                    symbol=symbol,
                    price=current_price,
                    shares=shares,
                    date=datetime.now(),
                    reason=f"Auto: {signal['strength']} BUY ({confidence*100:.0f}%)"
                )

                if success:
                    # CSV Manager aktualisieren
                    commission = PORTFOLIO_CONFIG.get('commission', 0.001) * current_price * shares
                    self.csv_manager.add_buy(
                        symbol=symbol,
                        date=datetime.now(),
                        quantity=shares,
                        price=current_price,
                        commission=commission,
                        notes=f"Confidence: {confidence*100:.0f}%"
                    )

                    logger.info(f"✅ TRADE EXECUTED: BUY {shares} {symbol} @ ${current_price:.2f}")

                    return {
                        'action': 'BUY',
                        'symbol': symbol,
                        'shares': shares,
                        'price': current_price,
                        'confidence': confidence,
                        'timestamp': datetime.now()
                    }

        # SELL Signal
        elif action == 'SELL':
            # Prüfe ob Position vorhanden
            if symbol not in self.portfolio.positions:
                logger.info(f"⚠️ {symbol}: Keine Position zum Verkaufen")
                return None

            position = self.portfolio.positions[symbol]

            # Verkaufen
            success = self.portfolio.sell(
                symbol=symbol,
                price=current_price,
                shares=None,  # Alle Shares
                date=datetime.now(),
                reason=f"Auto: {signal['strength']} SELL ({confidence*100:.0f}%)"
            )

            if success:
                # CSV Manager aktualisieren
                commission = PORTFOLIO_CONFIG.get('commission', 0.001) * current_price * position.shares
                self.csv_manager.add_sell(
                    symbol=symbol,
                    date=datetime.now(),
                    quantity=position.shares,
                    price=current_price,
                    commission=commission,
                    notes=f"Confidence: {confidence*100:.0f}%"
                )

                logger.info(f"✅ TRADE EXECUTED: SELL {position.shares} {symbol} @ ${current_price:.2f}")

                return {
                    'action': 'SELL',
                    'symbol': symbol,
                    'shares': position.shares,
                    'price': current_price,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                }

        return None


class ContinuousTradingBot:
    """
    Continuous Trading Bot
    Hauptklasse für kontinuierlichen Trading Bot
    """

    def __init__(self,
                 symbols: List[str],
                 initial_capital: float = 100000,
                 scan_interval_seconds: int = 300,  # 5 Minuten
                 min_confidence: float = 0.6,
                 use_ml: bool = False,
                 use_advanced: bool = True):
        """
        Initialisiert Continuous Trading Bot

        Args:
            symbols: Liste von Symbolen
            initial_capital: Anfangskapital
            scan_interval_seconds: Sekunden zwischen Scans
            min_confidence: Mindest-Confidence für Trades
            use_ml: ML verwenden
            use_advanced: Advanced Features verwenden
        """
        self.symbols = symbols
        self.scan_interval = scan_interval_seconds
        self.running = False

        # Initialize Components
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.csv_manager = CSVManager()

        if use_advanced and HAS_ADVANCED:
            self.order_manager = OrderManager()
            self.risk_manager = AdvancedRiskManager(initial_capital=initial_capital)
        else:
            self.order_manager = None
            self.risk_manager = None

        self.scanner = LiveMarketScanner(
            symbols=symbols,
            scan_interval_seconds=scan_interval_seconds
        )

        self.signal_generator = SignalGenerator(
            use_ml=use_ml,
            use_advanced=use_advanced
        )

        self.trade_executor = TradeExecutor(
            portfolio=self.portfolio,
            order_manager=self.order_manager,
            risk_manager=self.risk_manager,
            csv_manager=self.csv_manager,
            min_confidence=min_confidence
        )

        self.trade_log = []

        logger.info(f"\n{'='*80}")
        logger.info(f"🤖 CONTINUOUS TRADING BOT INITIALISIERT")
        logger.info(f"{'='*80}")
        logger.info(f"Symbole: {len(symbols)}")
        logger.info(f"Kapital: ${initial_capital:,.2f}")
        logger.info(f"Scan Interval: {scan_interval_seconds}s")
        logger.info(f"Min Confidence: {min_confidence*100:.0f}%")
        logger.info(f"{'='*80}\n")

    def run_single_cycle(self) -> Dict:
        """
        Führt einen einzelnen Trading-Zyklus durch

        Returns:
            Dict mit Zyklus-Ergebnissen
        """
        cycle_start = datetime.now()

        # 1. Market Scan
        market_data = self.scanner.scan_all_symbols(timeframes=['1d', '1h'])

        # 2. Signal Generation
        signals = {}
        for symbol, data in market_data.items():
            signal = self.signal_generator.generate_signal(symbol, data)
            signals[symbol] = signal

            logger.info(f"📡 {symbol}: {signal['signal']} ({signal['strength']}) - "
                       f"Confidence: {signal['confidence']*100:.0f}%")

        # 3. Trade Execution
        executed_trades = []

        for symbol, signal in signals.items():
            if signal['signal'] in ['BUY', 'SELL']:
                # Hole aktuellen Preis
                if symbol in market_data and 'analyzed_timeframes' in market_data[symbol]:
                    timeframes = market_data[symbol]['analyzed_timeframes']
                    if timeframes:
                        # Nehme 1d Daten
                        # Approximiere aktuellen Preis
                        current_price = 100.0  # Placeholder - in Real würde man aktuellen Preis holen

                        trade = self.trade_executor.execute_signal(signal, current_price)

                        if trade:
                            executed_trades.append(trade)
                            self.trade_log.append(trade)

        # 4. Export zu Yahoo Finance
        self.csv_manager.export_for_yahoo('yahoo_finance_portfolio.csv')

        # 5. Portfolio Status
        total_value = self.portfolio.get_total_value()
        total_return = (total_value - self.portfolio.initial_capital) / self.portfolio.initial_capital

        cycle_duration = (datetime.now() - cycle_start).total_seconds()

        logger.info(f"\n{'='*80}")
        logger.info(f"📊 ZYKLUS ABGESCHLOSSEN")
        logger.info(f"{'='*80}")
        logger.info(f"Dauer: {cycle_duration:.1f}s")
        logger.info(f"Signale: {len(signals)}")
        logger.info(f"Trades: {len(executed_trades)}")
        logger.info(f"Portfolio Wert: ${total_value:,.2f}")
        logger.info(f"Total Return: {total_return*100:+.2f}%")
        logger.info(f"{'='*80}\n")

        return {
            'timestamp': cycle_start,
            'duration_seconds': cycle_duration,
            'signals_generated': len(signals),
            'trades_executed': len(executed_trades),
            'portfolio_value': total_value,
            'total_return': total_return
        }

    def run_continuous(self, max_cycles: Optional[int] = None):
        """
        Startet kontinuierlichen Trading Bot

        Args:
            max_cycles: Maximale Anzahl Zyklen (None = unendlich)
        """
        self.running = True
        cycle_count = 0

        logger.info(f"🚀 STARTE KONTINUIERLICHEN TRADING BOT")
        logger.info(f"Scan Interval: {self.scan_interval}s")

        if max_cycles:
            logger.info(f"Max Zyklen: {max_cycles}")

        logger.info(f"Drücke CTRL+C zum Stoppen\n")

        try:
            while self.running:
                cycle_count += 1

                if max_cycles and cycle_count > max_cycles:
                    logger.info(f"✅ Maximale Zyklen ({max_cycles}) erreicht")
                    break

                logger.info(f"\n{'#'*80}")
                logger.info(f"# ZYKLUS {cycle_count}")
                logger.info(f"{'#'*80}\n")

                # Run Cycle
                result = self.run_single_cycle()

                # Warte bis nächster Zyklus
                if not max_cycles or cycle_count < max_cycles:
                    logger.info(f"⏳ Warte {self.scan_interval}s bis nächster Scan...\n")
                    time.sleep(self.scan_interval)

        except KeyboardInterrupt:
            logger.info(f"\n\n⚠️ Benutzer-Unterbrechung erkannt")
            self.stop()

        except Exception as e:
            logger.error(f"\n\n❌ Fehler im Trading Bot: {e}")
            import traceback
            traceback.print_exc()
            self.stop()

    def stop(self):
        """Stoppt Trading Bot"""
        self.running = False

        logger.info(f"\n{'='*80}")
        logger.info(f"🛑 TRADING BOT GESTOPPT")
        logger.info(f"{'='*80}")

        # Final Summary
        self.portfolio.print_summary()

        # Export Final CSV
        self.csv_manager.export_for_yahoo('yahoo_finance_portfolio_final.csv')
        logger.info(f"\n✅ Yahoo Finance Portfolio exportiert: yahoo_finance_portfolio_final.csv")

        # Trade Log
        if self.trade_log:
            logger.info(f"\n📋 TRADE LOG ({len(self.trade_log)} Trades):")
            for i, trade in enumerate(self.trade_log[-10:], 1):  # Letzte 10
                logger.info(f"   {i}. {trade['action']} {trade['shares']} {trade['symbol']} "
                          f"@ ${trade['price']:.2f} ({trade['confidence']*100:.0f}%)")

        logger.info(f"\n{'='*80}\n")


def main():
    """
    Hauptfunktion - Startet Continuous Trading Bot
    """
    # Konfiguration
    SYMBOLS = WATCHLIST  # Aus config.py
    INITIAL_CAPITAL = PORTFOLIO_CONFIG.get('initial_capital', 100000)
    SCAN_INTERVAL = 300  # 5 Minuten
    MIN_CONFIDENCE = 0.6  # 60% Mindest-Confidence
    MAX_CYCLES = 10  # Für Demo - None für unendlich

    # Bot erstellen
    bot = ContinuousTradingBot(
        symbols=SYMBOLS,
        initial_capital=INITIAL_CAPITAL,
        scan_interval_seconds=SCAN_INTERVAL,
        min_confidence=MIN_CONFIDENCE,
        use_ml=False,  # ML optional (braucht sklearn)
        use_advanced=True
    )

    # Bot starten
    bot.run_continuous(max_cycles=MAX_CYCLES)


if __name__ == "__main__":
    main()

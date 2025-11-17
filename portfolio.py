"""
Portfolio Management - Verbesserte Version
Mit Stop-Loss, Take-Profit, Trailing-Stop und erweitertem Risk-Management
"""
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import logging
import json

from config import PORTFOLIO_CONFIG, RISK_CONFIG

logger = logging.getLogger(__name__)


class Position:
    """Erweiterte Position-Klasse mit Risk-Management"""
    
    def __init__(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        entry_date: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.symbol = symbol
        self.shares = shares
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.current_price = entry_price
        
        # Risk Management
        self.stop_loss = stop_loss or self._calculate_stop_loss(entry_price)
        self.take_profit = take_profit or self._calculate_take_profit(entry_price)
        self.trailing_stop = None
        self.highest_price = entry_price
        
        # Performance Tracking
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
    
    def _calculate_stop_loss(self, entry_price: float) -> float:
        """Berechnet initialen Stop-Loss"""
        if RISK_CONFIG.get("use_stop_loss", True):
            stop_pct = RISK_CONFIG.get("stop_loss_percent", 0.05)
            return entry_price * (1 - stop_pct)
        return 0.0
    
    def _calculate_take_profit(self, entry_price: float) -> float:
        """Berechnet initialen Take-Profit"""
        if RISK_CONFIG.get("use_take_profit", True):
            tp_pct = RISK_CONFIG.get("take_profit_percent", 0.15)
            return entry_price * (1 + tp_pct)
        return float('inf')
    
    def update(self, current_price: float):
        """
        Aktualisiert Position mit aktuellem Preis
        
        Args:
            current_price: Aktueller Marktpreis
        """
        self.current_price = current_price
        
        # Update Höchstpreis für Trailing Stop
        if current_price > self.highest_price:
            self.highest_price = current_price
            
            # Update Trailing Stop
            if RISK_CONFIG.get("trailing_stop", False):
                trailing_pct = RISK_CONFIG.get("trailing_stop_percent", 0.03)
                self.trailing_stop = current_price * (1 - trailing_pct)
        
        # Berechne Unrealized P&L
        self.unrealized_pnl = (current_price - self.entry_price) * self.shares
        self.unrealized_pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
    
    def should_stop_loss(self) -> bool:
        """Prüft ob Stop-Loss getriggert wurde"""
        if not RISK_CONFIG.get("use_stop_loss", True):
            return False
        
        # Prüfe Trailing Stop falls aktiv
        if self.trailing_stop and self.current_price <= self.trailing_stop:
            logger.info(f"🛑 Trailing Stop getriggert für {self.symbol}: ${self.current_price:.2f} <= ${self.trailing_stop:.2f}")
            return True
        
        # Prüfe regulären Stop-Loss
        if self.current_price <= self.stop_loss:
            logger.info(f"🛑 Stop-Loss getriggert für {self.symbol}: ${self.current_price:.2f} <= ${self.stop_loss:.2f}")
            return True
        
        return False
    
    def should_take_profit(self) -> Tuple[bool, float]:
        """
        Prüft ob Take-Profit getriggert wurde
        
        Returns:
            Tuple (should_exit, amount_to_sell_pct)
        """
        if not RISK_CONFIG.get("use_take_profit", True):
            return False, 0.0
        
        if self.current_price >= self.take_profit:
            logger.info(f"🎯 Take-Profit getriggert für {self.symbol}: ${self.current_price:.2f} >= ${self.take_profit:.2f}")
            return True, 1.0  # Verkaufe 100%
        
        # Partial Take Profit
        if RISK_CONFIG.get("partial_take_profit", False):
            partial_tp_pct = RISK_CONFIG.get("partial_tp_percent", 0.10)
            partial_amount = RISK_CONFIG.get("partial_tp_amount", 0.50)
            
            partial_tp_price = self.entry_price * (1 + partial_tp_pct)
            if self.current_price >= partial_tp_price:
                logger.info(f"🎯 Partial Take-Profit für {self.symbol}: Verkaufe {partial_amount*100:.0f}%")
                return True, partial_amount
        
        return False, 0.0
    
    def get_value(self) -> float:
        """Berechnet aktuellen Wert der Position"""
        return self.shares * self.current_price
    
    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary"""
        return {
            'symbol': self.symbol,
            'shares': self.shares,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date.isoformat() if isinstance(self.entry_date, datetime) else self.entry_date,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
        }


class Portfolio:
    """Erweitertes Portfolio-Management mit Risk-Management"""
    
    def __init__(self, initial_capital: Optional[float] = None):
        """
        Initialisiert das Portfolio
        
        Args:
            initial_capital: Startkapital
        """
        self.initial_capital = initial_capital or PORTFOLIO_CONFIG["initial_capital"]
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        # Risk Tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.peak_value = self.initial_capital
        self.max_drawdown = 0.0
        
        logger.info(f"Portfolio initialisiert mit ${self.initial_capital:,.2f}")
    
    def get_total_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Berechnet Gesamtwert des Portfolios
        
        Args:
            current_prices: Dict mit aktuellen Preisen {symbol: price}
        
        Returns:
            Gesamtwert
        """
        if current_prices:
            for symbol, price in current_prices.items():
                if symbol in self.positions:
                    self.positions[symbol].update(price)
        
        position_value = sum(pos.get_value() for pos in self.positions.values())
        total = self.cash + position_value
        
        # Update Drawdown
        if total > self.peak_value:
            self.peak_value = total
        current_drawdown = (self.peak_value - total) / self.peak_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        return total
    
    def get_position_value(self, symbol: str) -> float:
        """Berechnet Wert einer Position"""
        if symbol not in self.positions:
            return 0.0
        return self.positions[symbol].get_value()
    
    def check_risk_limits(self) -> Tuple[bool, str]:
        """
        Prüft ob Risk-Limits überschritten sind
        
        Returns:
            Tuple (ok, reason)
        """
        total_value = self.get_total_value()
        
        # Max Drawdown
        if self.max_drawdown > RISK_CONFIG.get("max_drawdown", 0.15):
            return False, f"Max Drawdown überschritten: {self.max_drawdown:.1%}"
        
        # Max Daily Loss
        daily_loss_pct = abs(self.daily_pnl / self.initial_capital)
        if self.daily_pnl < 0 and daily_loss_pct > RISK_CONFIG.get("max_daily_loss", 0.03):
            return False, f"Max Daily Loss überschritten: {daily_loss_pct:.1%}"
        
        # Max Positions
        max_positions = PORTFOLIO_CONFIG.get("max_positions", 8)
        if len(self.positions) >= max_positions:
            return False, f"Max Anzahl Positionen erreicht: {len(self.positions)}"
        
        return True, ""
    
    def can_buy(
        self, 
        symbol: str, 
        price: float, 
        shares: int
    ) -> Tuple[bool, str]:
        """
        Prüft ob Kauf möglich ist
        
        Args:
            symbol: Aktiensymbol
            price: Kaufpreis
            shares: Anzahl Aktien
        
        Returns:
            Tuple (can_buy, reason)
        """
        # Kosten berechnen
        commission = PORTFOLIO_CONFIG.get("commission", 0)
        slippage = PORTFOLIO_CONFIG.get("slippage", 0.001)
        effective_price = price * (1 + slippage)
        cost = effective_price * shares + commission
        
        # Prüfe Cash verfügbar
        if cost > self.cash:
            return False, f"Nicht genug Cash: ${cost:.2f} > ${self.cash:.2f}"
        
        # Prüfe Cash-Reserve
        min_cash = self.initial_capital * PORTFOLIO_CONFIG.get("min_cash_reserve", 0.10)
        if self.cash - cost < min_cash:
            return False, f"Würde Cash-Reserve unterschreiten: ${self.cash - cost:.2f} < ${min_cash:.2f}"
        
        # Prüfe Positionsgröße
        total_value = self.get_total_value()
        max_position = total_value * PORTFOLIO_CONFIG.get("max_position_size", 0.20)
        current_position_value = self.get_position_value(symbol)
        
        if current_position_value + cost > max_position:
            return False, f"Würde max Position Size überschreiten"
        
        # Prüfe Risk Limits
        ok, reason = self.check_risk_limits()
        if not ok:
            return False, f"Risk Limit: {reason}"
        
        return True, ""
    
    def calculate_position_size(
        self, 
        symbol: str, 
        price: float,
        risk_pct: Optional[float] = None
    ) -> int:
        """
        Berechnet optimale Positionsgröße basierend auf Risk-Management
        
        Args:
            symbol: Aktiensymbol
            price: Aktueller Preis
            risk_pct: Risiko-Prozent (optional)
        
        Returns:
            Anzahl Aktien zu kaufen
        """
        total_value = self.get_total_value()
        
        # Methode 1: Fixed Percentage
        max_position_value = total_value * PORTFOLIO_CONFIG.get("max_position_size", 0.20)
        min_position_value = total_value * PORTFOLIO_CONFIG.get("min_position_size", 0.02)
        
        # Methode 2: Risk-based (wenn risk_pct gegeben)
        if risk_pct and RISK_CONFIG.get("use_stop_loss", True):
            risk_amount = total_value * RISK_CONFIG.get("max_portfolio_risk", 0.02)
            stop_loss_pct = RISK_CONFIG.get("stop_loss_percent", 0.05)
            risk_based_size = risk_amount / (price * stop_loss_pct)
            max_position_value = min(max_position_value, risk_based_size * price)
        
        # Berücksichtige existierende Position
        current_position_value = self.get_position_value(symbol)
        available_for_position = max_position_value - current_position_value
        
        # Respektiere Cash-Reserve
        min_cash = self.initial_capital * PORTFOLIO_CONFIG.get("min_cash_reserve", 0.10)
        available_cash = self.cash - min_cash
        
        # Nutze das Minimum
        max_buy_value = min(available_cash, available_for_position)
        
        if max_buy_value < min_position_value:
            return 0
        
        # Berechne Shares (mit Slippage und Commission)
        slippage = PORTFOLIO_CONFIG.get("slippage", 0.001)
        commission = PORTFOLIO_CONFIG.get("commission", 0)
        effective_price = price * (1 + slippage)

        # Berücksichtige Commission in der Berechnung
        shares = int((max_buy_value - commission) / effective_price)
        return max(0, shares)
    
    def buy(
        self, 
        symbol: str, 
        price: float, 
        shares: int, 
        date: Optional[datetime] = None
    ) -> bool:
        """
        Führt Kauf aus
        
        Args:
            symbol: Aktiensymbol
            price: Kaufpreis
            shares: Anzahl Aktien
            date: Datum des Trades
        
        Returns:
            True wenn erfolgreich
        """
        can_buy, reason = self.can_buy(symbol, price, shares)
        if not can_buy:
            logger.warning(f"❌ Kauf nicht möglich für {symbol}: {reason}")
            return False
        
        # Berechne effektive Kosten
        commission = PORTFOLIO_CONFIG.get("commission", 0)
        slippage = PORTFOLIO_CONFIG.get("slippage", 0.001)
        effective_price = price * (1 + slippage)
        cost = effective_price * shares + commission
        
        # Aktualisiere Cash
        self.cash -= cost
        
        # Erstelle oder update Position
        if symbol in self.positions:
            # Average up/down
            old_pos = self.positions[symbol]
            total_shares = old_pos.shares + shares
            avg_price = ((old_pos.shares * old_pos.entry_price) + (shares * effective_price)) / total_shares
            
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=total_shares,
                entry_price=avg_price,
                entry_date=old_pos.entry_date
            )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                entry_price=effective_price,
                entry_date=date or datetime.now()
            )
        
        # Speichere Trade
        trade = {
            'date': date or datetime.now(),
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'effective_price': effective_price,
            'commission': commission,
            'slippage': effective_price - price,
            'total': cost,
            'cash_after': self.cash,
        }
        self.trades.append(trade)
        
        logger.info(f"✅ KAUF: {shares} x {symbol} @ ${price:.2f} (eff: ${effective_price:.2f}) | Kosten: ${cost:.2f}")
        return True
    
    def sell(
        self, 
        symbol: str, 
        price: float, 
        shares: Optional[int] = None, 
        date: Optional[datetime] = None,
        reason: str = "Manual"
    ) -> bool:
        """
        Führt Verkauf aus
        
        Args:
            symbol: Aktiensymbol
            price: Verkaufspreis
            shares: Anzahl Aktien (None = alle)
            date: Datum
            reason: Grund für Verkauf
        
        Returns:
            True wenn erfolgreich
        """
        if symbol not in self.positions:
            logger.warning(f"⚠️ Keine Position in {symbol} zum Verkaufen")
            return False
        
        position = self.positions[symbol]
        
        # Wenn shares nicht angegeben, verkaufe alle
        if shares is None:
            shares = position.shares
        
        # Prüfe ob genug Aktien vorhanden
        if shares > position.shares:
            logger.warning(f"⚠️ Nicht genug Aktien von {symbol}")
            return False
        
        # Berechne Erlös
        commission = PORTFOLIO_CONFIG.get("commission", 0)
        slippage = PORTFOLIO_CONFIG.get("slippage", 0.001)
        effective_price = price * (1 - slippage)
        proceeds = effective_price * shares - commission
        
        # Aktualisiere Cash
        self.cash += proceeds
        
        # Berechne P&L
        cost_basis = position.entry_price * shares
        profit_loss = proceeds - cost_basis
        profit_pct = (profit_loss / cost_basis) * 100
        
        # Update Position
        if shares == position.shares:
            # Vollständiger Verkauf
            del self.positions[symbol]
        else:
            # Teilverkauf
            position.shares -= shares
        
        # Speichere Trade
        trade = {
            'date': date or datetime.now(),
            'symbol': symbol,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'effective_price': effective_price,
            'commission': commission,
            'slippage': price - effective_price,
            'total': proceeds,
            'profit_loss': profit_loss,
            'profit_pct': profit_pct,
            'reason': reason,
            'cash_after': self.cash,
        }
        self.trades.append(trade)
        
        # Update daily P&L
        self.daily_pnl += profit_loss
        
        emoji = "🟢" if profit_loss > 0 else "🔴"
        logger.info(f"{emoji} VERKAUF: {shares} x {symbol} @ ${price:.2f} | Erlös: ${proceeds:.2f} | P/L: ${profit_loss:.2f} ({profit_pct:+.2f}%) | Grund: {reason}")
        
        return True
    
    def check_risk_management(self, current_prices: Dict[str, float]):
        """
        Prüft alle Positionen auf Stop-Loss und Take-Profit
        
        Args:
            current_prices: Dict mit aktuellen Preisen
        """
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            position.update(current_price)
            
            # Prüfe Stop-Loss
            if position.should_stop_loss():
                positions_to_close.append((symbol, current_price, "Stop-Loss"))
            
            # Prüfe Take-Profit
            should_tp, amount_pct = position.should_take_profit()
            if should_tp:
                shares_to_sell = int(position.shares * amount_pct)
                if shares_to_sell > 0:
                    reason = "Take-Profit" if amount_pct == 1.0 else f"Partial Take-Profit ({amount_pct*100:.0f}%)"
                    positions_to_close.append((symbol, current_price, reason, shares_to_sell))
        
        # Führe Verkäufe aus
        for item in positions_to_close:
            if len(item) < 3:
                logger.warning(f"Ungültiges Item in positions_to_close: {item}")
                continue

            symbol, price, reason = item[0], item[1], item[2]
            shares = item[3] if len(item) > 3 else None

            # Prüfe ob Position noch existiert (könnte bereits verkauft worden sein)
            if symbol not in self.positions:
                logger.debug(f"Position {symbol} existiert nicht mehr, überspringe")
                continue

            self.sell(symbol, price, shares=shares, reason=reason)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Gibt Position für Symbol zurück"""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Prüft ob Position existiert"""
        return symbol in self.positions
    
    def get_performance_stats(self) -> Dict:
        """Berechnet detaillierte Performance-Statistiken"""
        if not self.trades:
            return {}
        
        total_trades = len(self.trades)
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        
        if not sell_trades:
            return {
                'total_trades': total_trades,
                'open_positions': len(self.positions),
            }
        
        winning_trades = [t for t in sell_trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('profit_loss', 0) <= 0]
        
        total_profit = sum(t.get('profit_loss', 0) for t in sell_trades)
        total_win = sum(t.get('profit_loss', 0) for t in winning_trades)
        total_loss = sum(t.get('profit_loss', 0) for t in losing_trades)
        
        avg_win = total_win / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        
        # Profit Factor
        profit_factor = abs(total_win / total_loss) if total_loss != 0 else float('inf')
        
        # Sharpe Ratio (vereinfacht)
        returns = [t.get('profit_pct', 0) / 100 for t in sell_trades]
        avg_return = sum(returns) / len(returns) if returns else 0
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5 if len(returns) > 1 else 0
        sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return != 0 else 0
        
        current_value = self.get_total_value()
        total_return = ((current_value - self.initial_capital) / self.initial_capital) * 100
        
        stats = {
            'total_trades': total_trades,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(sell_trades) * 100) if sell_trades else 0,
            'total_profit': total_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown * 100,
            'current_value': current_value,
            'open_positions': len(self.positions),
        }
        
        return stats
    
    def save_state(self, filename: str = "portfolio_state.json"):
        """Speichert Portfolio-Status"""
        state = {
            'cash': self.cash,
            'initial_capital': self.initial_capital,
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'trades': self.trades,
            'daily_pnl': self.daily_pnl,
            'peak_value': self.peak_value,
            'max_drawdown': self.max_drawdown,
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Portfolio-Status gespeichert in {filename}")
    
    def print_summary(self):
        """Gibt Portfolio-Zusammenfassung aus"""
        print("\n" + "="*70)
        print("PORTFOLIO ZUSAMMENFASSUNG")
        print("="*70)
        
        current_value = self.get_total_value()
        total_return = ((current_value - self.initial_capital) / self.initial_capital) * 100
        
        print(f"💰 Cash: ${self.cash:,.2f}")
        print(f"📊 Startkapital: ${self.initial_capital:,.2f}")
        print(f"💎 Aktueller Wert: ${current_value:,.2f}")
        print(f"📈 Total Return: {total_return:+.2f}%")
        print(f"📉 Max Drawdown: {self.max_drawdown*100:.2f}%")
        
        if self.positions:
            print(f"\n📈 Aktuelle Positionen ({len(self.positions)}):")
            for symbol, pos in self.positions.items():
                value = pos.get_value()
                print(f"  {symbol}: {pos.shares} @ ${pos.entry_price:.2f} | "
                      f"Aktuell: ${pos.current_price:.2f} | "
                      f"P/L: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:+.2f}%) | "
                      f"SL: ${pos.stop_loss:.2f}")
        else:
            print("\n📈 Keine offenen Positionen")
        
        stats = self.get_performance_stats()
        if stats and 'total_trades' in stats:
            print(f"\n📊 Trading Statistiken:")
            print(f"  Trades gesamt: {stats['total_trades']}")
            print(f"  Verkäufe: {stats['sell_trades']}")
            print(f"  Gewinner: {stats['winning_trades']}")
            print(f"  Verlierer: {stats['losing_trades']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            if stats.get('profit_factor'):
                print(f"  Profit Factor: {stats['profit_factor']:.2f}")
            if stats.get('sharpe_ratio'):
                print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
            print(f"  Gesamt P/L: ${stats['total_profit']:,.2f}")
        
        print("="*70 + "\n")

    def calculate_kelly_position_size(
        self,
        symbol: str,
        price: float,
        kelly_fraction: float = 0.5
    ) -> int:
        """
        Berechnet Position Size basierend auf Kelly Criterion

        Args:
            symbol: Aktiensymbol
            price: Aktueller Preis
            kelly_fraction: Kelly-Multiplikator (Standard: 0.5 = Half-Kelly)

        Returns:
            Anzahl Aktien basierend auf Kelly
        """
        try:
            from math_utils import KellyCriterion
            import pandas as pd

            # Verwende historische Trades für Kelly-Berechnung
            if not self.trades:
                return self.calculate_position_size(symbol, price)

            sell_trades = [t for t in self.trades if t['action'] == 'SELL']
            if len(sell_trades) < 5:  # Mindestens 5 Trades für sinnvolle Berechnung
                return self.calculate_position_size(symbol, price)

            trades_df = pd.DataFrame(sell_trades)
            kelly_pct = KellyCriterion.calculate_kelly_from_trades(trades_df, kelly_fraction)

            # Berechne Position Size basierend auf Kelly
            total_value = self.get_total_value()
            kelly_amount = total_value * kelly_pct

            # Respektiere Min/Max Limits
            max_position = total_value * PORTFOLIO_CONFIG.get("max_position_size", 0.20)
            min_position = total_value * PORTFOLIO_CONFIG.get("min_position_size", 0.02)

            kelly_amount = max(min_position, min(kelly_amount, max_position))

            # Berücksichtige Cash-Reserve
            min_cash = self.initial_capital * PORTFOLIO_CONFIG.get("min_cash_reserve", 0.10)
            available_cash = self.cash - min_cash

            kelly_amount = min(kelly_amount, available_cash)

            # Berechne Shares
            slippage = PORTFOLIO_CONFIG.get("slippage", 0.001)
            commission = PORTFOLIO_CONFIG.get("commission", 0)
            effective_price = price * (1 + slippage)

            shares = int((kelly_amount - commission) / effective_price)

            logger.debug(f"Kelly Criterion für {symbol}: {kelly_pct*100:.1f}% → {shares} Aktien")

            return max(0, shares)

        except ImportError:
            logger.warning("math_utils nicht verfügbar, verwende Standard Position Sizing")
            return self.calculate_position_size(symbol, price)
        except Exception as e:
            logger.error(f"Fehler bei Kelly-Berechnung: {e}")
            return self.calculate_position_size(symbol, price)

    def get_advanced_analytics(self) -> Dict:
        """
        Gibt erweiterte Portfolio-Analysen zurück

        Returns:
            Dictionary mit erweiterten Metriken
        """
        try:
            from performance_analytics import PerformanceAnalytics

            analytics = PerformanceAnalytics(self.trades, self.initial_capital)
            return analytics.calculate_comprehensive_metrics()

        except ImportError:
            logger.warning("performance_analytics nicht verfügbar")
            return self.get_performance_stats()
        except Exception as e:
            logger.error(f"Fehler bei erweiterten Analysen: {e}")
            return self.get_performance_stats()

    def print_advanced_summary(self):
        """Gibt erweiterte Portfolio-Zusammenfassung aus"""
        try:
            from performance_analytics import PerformanceAnalytics

            analytics = PerformanceAnalytics(self.trades, self.initial_capital)
            report = analytics.generate_performance_report()
            print(report)

        except ImportError:
            logger.warning("performance_analytics nicht verfügbar, verwende Standard-Summary")
            self.print_summary()
        except Exception as e:
            logger.error(f"Fehler bei erweiterten Report: {e}")
            self.print_summary()

    def export_performance_report(self, base_filename: str = "performance_report"):
        """
        Exportiert erweiterten Performance-Report als CSV-Dateien (kein Excel!)

        Args:
            base_filename: Basis-Dateiname für CSV-Exporte (ohne .csv)
        """
        try:
            from performance_analytics import PerformanceAnalytics

            analytics = PerformanceAnalytics(self.trades, self.initial_capital)
            exported_files = analytics.export_to_csv(base_filename)

            if exported_files:
                logger.info(f"📊 Performance-Report exportiert: {len(exported_files)} CSV-Dateien")
            else:
                logger.warning("⚠️ Keine Performance-Daten zum Exportieren")

        except ImportError:
            logger.warning("performance_analytics nicht verfügbar")
        except Exception as e:
            logger.error(f"Fehler beim Export: {e}")

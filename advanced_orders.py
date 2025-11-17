"""
Advanced Order Management System
Erweiterte Order-Typen (Limit, Stop, Trailing Stop, OCO, etc.)
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order-Typen"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "one_cancels_other"  # One-Cancels-Other


class OrderSide(Enum):
    """Order-Seite"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order-Status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class Order:
    """Basis-Order-Klasse"""

    def __init__(self,
                 order_id: str,
                 symbol: str,
                 order_type: OrderType,
                 side: OrderSide,
                 quantity: int,
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 trailing_percent: Optional[float] = None,
                 time_in_force: str = "GTC",  # GTC, DAY, IOC, FOK
                 created_at: Optional[datetime] = None):
        """
        Initialisiert Order

        Args:
            order_id: Eindeutige Order-ID
            symbol: Stock Symbol
            order_type: Order-Typ
            side: BUY oder SELL
            quantity: Anzahl Shares
            price: Limit-Preis (für LIMIT Orders)
            stop_price: Stop-Preis (für STOP Orders)
            trailing_percent: Trailing Percent (für TRAILING_STOP)
            time_in_force: Gültigkeitsdauer
            created_at: Erstellungszeit
        """
        self.order_id = order_id
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.filled_quantity = 0
        self.price = price
        self.stop_price = stop_price
        self.trailing_percent = trailing_percent
        self.trailing_high = None  # Für Trailing Stop
        self.trailing_low = None
        self.time_in_force = time_in_force
        self.status = OrderStatus.PENDING
        self.created_at = created_at or datetime.now()
        self.filled_at = None
        self.avg_fill_price = None
        self.reason = ""

    def __repr__(self):
        return (f"Order({self.order_id}, {self.symbol}, {self.order_type.value}, "
                f"{self.side.value}, {self.quantity}, status={self.status.value})")

    def is_fillable(self, current_price: float) -> bool:
        """
        Überprüft ob Order ausgeführt werden kann

        Args:
            current_price: Aktueller Marktpreis

        Returns:
            True wenn Order ausführbar
        """
        if self.status != OrderStatus.PENDING:
            return False

        if self.order_type == OrderType.MARKET:
            return True

        elif self.order_type == OrderType.LIMIT:
            if self.side == OrderSide.BUY:
                return current_price <= self.price
            else:  # SELL
                return current_price >= self.price

        elif self.order_type == OrderType.STOP:
            if self.side == OrderSide.BUY:
                return current_price >= self.stop_price
            else:  # SELL
                return current_price <= self.stop_price

        elif self.order_type == OrderType.STOP_LIMIT:
            # Erst Stop triggern
            if self.side == OrderSide.BUY:
                if current_price >= self.stop_price:
                    # Dann Limit prüfen
                    return current_price <= self.price
            else:  # SELL
                if current_price <= self.stop_price:
                    return current_price >= self.price
            return False

        elif self.order_type == OrderType.TRAILING_STOP:
            return self._check_trailing_stop(current_price)

        return False

    def _check_trailing_stop(self, current_price: float) -> bool:
        """Überprüft Trailing Stop Bedingung"""
        if self.trailing_percent is None:
            return False

        if self.side == OrderSide.SELL:
            # Update trailing high
            if self.trailing_high is None or current_price > self.trailing_high:
                self.trailing_high = current_price

            # Check if stop triggered
            stop_price = self.trailing_high * (1 - self.trailing_percent / 100)
            return current_price <= stop_price

        else:  # BUY (trailing stop für short positions)
            # Update trailing low
            if self.trailing_low is None or current_price < self.trailing_low:
                self.trailing_low = current_price

            # Check if stop triggered
            stop_price = self.trailing_low * (1 + self.trailing_percent / 100)
            return current_price >= stop_price

    def fill(self, price: float, quantity: Optional[int] = None):
        """
        Führt Order aus

        Args:
            price: Ausführungspreis
            quantity: Ausgeführte Menge (None = vollständig)
        """
        if quantity is None:
            quantity = self.quantity - self.filled_quantity

        self.filled_quantity += quantity
        self.avg_fill_price = price  # Vereinfacht, könnte gewichteter Durchschnitt sein

        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now()
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

        logger.info(f"✅ Order {self.order_id} filled: {quantity} @ ${price:.2f}")

    def cancel(self, reason: str = ""):
        """Storniert Order"""
        self.status = OrderStatus.CANCELLED
        self.reason = reason
        logger.info(f"❌ Order {self.order_id} cancelled: {reason}")


class OrderManager:
    """Order Management System"""

    def __init__(self):
        """Initialisiert Order Manager"""
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        self.filled_orders: List[Order] = []

    def create_market_order(self, symbol: str, side: OrderSide, quantity: int) -> Order:
        """
        Erstellt Market Order

        Args:
            symbol: Stock Symbol
            side: BUY oder SELL
            quantity: Anzahl Shares

        Returns:
            Order-Objekt
        """
        order_id = self._generate_order_id()
        order = Order(
            order_id=order_id,
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity
        )

        self.orders[order_id] = order
        logger.info(f"📝 Market Order erstellt: {order}")
        return order

    def create_limit_order(self, symbol: str, side: OrderSide,
                          quantity: int, price: float) -> Order:
        """
        Erstellt Limit Order

        Args:
            symbol: Stock Symbol
            side: BUY oder SELL
            quantity: Anzahl Shares
            price: Limit-Preis

        Returns:
            Order-Objekt
        """
        order_id = self._generate_order_id()
        order = Order(
            order_id=order_id,
            symbol=symbol,
            order_type=OrderType.LIMIT,
            side=side,
            quantity=quantity,
            price=price
        )

        self.orders[order_id] = order
        logger.info(f"📝 Limit Order erstellt: {order} @ ${price:.2f}")
        return order

    def create_stop_order(self, symbol: str, side: OrderSide,
                         quantity: int, stop_price: float) -> Order:
        """
        Erstellt Stop Order

        Args:
            symbol: Stock Symbol
            side: BUY oder SELL
            quantity: Anzahl Shares
            stop_price: Stop-Preis

        Returns:
            Order-Objekt
        """
        order_id = self._generate_order_id()
        order = Order(
            order_id=order_id,
            symbol=symbol,
            order_type=OrderType.STOP,
            side=side,
            quantity=quantity,
            stop_price=stop_price
        )

        self.orders[order_id] = order
        logger.info(f"📝 Stop Order erstellt: {order} @ Stop ${stop_price:.2f}")
        return order

    def create_trailing_stop_order(self, symbol: str, side: OrderSide,
                                  quantity: int, trailing_percent: float) -> Order:
        """
        Erstellt Trailing Stop Order

        Args:
            symbol: Stock Symbol
            side: BUY oder SELL
            quantity: Anzahl Shares
            trailing_percent: Trailing Prozent (z.B. 5 für 5%)

        Returns:
            Order-Objekt
        """
        order_id = self._generate_order_id()
        order = Order(
            order_id=order_id,
            symbol=symbol,
            order_type=OrderType.TRAILING_STOP,
            side=side,
            quantity=quantity,
            trailing_percent=trailing_percent
        )

        self.orders[order_id] = order
        logger.info(f"📝 Trailing Stop Order erstellt: {order} @ {trailing_percent}%")
        return order

    def create_oco_order(self, symbol: str, quantity: int,
                        profit_target: float, stop_loss: float) -> Tuple[Order, Order]:
        """
        Erstellt One-Cancels-Other (OCO) Order Pair

        Wenn eine Order ausgeführt wird, wird die andere storniert

        Args:
            symbol: Stock Symbol
            quantity: Anzahl Shares
            profit_target: Take-Profit Preis
            stop_loss: Stop-Loss Preis

        Returns:
            (profit_order, stop_order)
        """
        profit_order = self.create_limit_order(symbol, OrderSide.SELL, quantity, profit_target)
        stop_order = self.create_stop_order(symbol, OrderSide.SELL, quantity, stop_loss)

        # Link orders
        profit_order.linked_order_id = stop_order.order_id
        stop_order.linked_order_id = profit_order.order_id

        logger.info(f"📝 OCO Order erstellt: Profit ${profit_target:.2f}, Stop ${stop_loss:.2f}")

        return profit_order, stop_order

    def process_orders(self, symbol: str, current_price: float) -> List[Order]:
        """
        Verarbeitet alle Orders für ein Symbol

        Args:
            symbol: Stock Symbol
            current_price: Aktueller Preis

        Returns:
            Liste der ausgeführten Orders
        """
        filled_orders = []

        for order in list(self.orders.values()):
            if order.symbol != symbol or order.status != OrderStatus.PENDING:
                continue

            if order.is_fillable(current_price):
                # Execute order
                order.fill(current_price)

                # Handle OCO
                if hasattr(order, 'linked_order_id') and order.linked_order_id in self.orders:
                    linked_order = self.orders[order.linked_order_id]
                    linked_order.cancel("OCO: Other order filled")

                filled_orders.append(order)
                self.filled_orders.append(order)

        return filled_orders

    def cancel_order(self, order_id: str, reason: str = "User cancelled"):
        """Storniert Order"""
        if order_id in self.orders:
            self.orders[order_id].cancel(reason)

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Gibt alle pending Orders zurück"""
        orders = [o for o in self.orders.values() if o.status == OrderStatus.PENDING]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Gibt Order-Historie zurück"""
        if symbol:
            return [o for o in self.filled_orders if o.symbol == symbol]
        return self.filled_orders

    def _generate_order_id(self) -> str:
        """Generiert eindeutige Order-ID"""
        self.order_counter += 1
        return f"ORD-{datetime.now().strftime('%Y%m%d')}-{self.order_counter:06d}"


class AdvancedRiskManager:
    """Erweitertes Risk Management"""

    def __init__(self, initial_capital: float):
        """
        Initialisiert Risk Manager

        Args:
            initial_capital: Anfangskapital
        """
        self.initial_capital = initial_capital
        self.max_position_size_pct = 0.15  # Max 15% pro Position
        self.max_portfolio_risk_pct = 0.02  # Max 2% Risiko pro Trade
        self.max_daily_loss_pct = 0.05  # Max 5% Verlust pro Tag
        self.max_drawdown_pct = 0.20  # Max 20% Drawdown
        self.correlation_limit = 0.7  # Max Korrelation zwischen Positionen

        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.peak_portfolio_value = initial_capital

    def calculate_position_size(self,
                               symbol: str,
                               entry_price: float,
                               stop_loss: float,
                               portfolio_value: float,
                               confidence: float = 1.0) -> int:
        """
        Berechnet optimale Positionsgröße basierend auf Risk Management

        Args:
            symbol: Stock Symbol
            entry_price: Einstiegspreis
            stop_loss: Stop-Loss Preis
            portfolio_value: Aktueller Portfolio-Wert
            confidence: Konfidenz in Trade (0-1)

        Returns:
            Anzahl Shares
        """
        # Risk per Share
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            logger.warning("Invalid risk_per_share: {risk_per_share}")
            return 0

        # Max Risk Amount (2% des Portfolios)
        max_risk_amount = portfolio_value * self.max_portfolio_risk_pct

        # Shares basierend auf Risk
        shares_by_risk = int(max_risk_amount / risk_per_share)

        # Max Position Value (15% des Portfolios)
        max_position_value = portfolio_value * self.max_position_size_pct
        shares_by_position_limit = int(max_position_value / entry_price)

        # Nehme das Minimum
        shares = min(shares_by_risk, shares_by_position_limit)

        # Anpassung basierend auf Confidence
        shares = int(shares * confidence)

        logger.info(f"Position Size für {symbol}: {shares} Shares "
                   f"(Risk: ${shares * risk_per_share:.2f}, "
                   f"Value: ${shares * entry_price:.2f})")

        return max(0, shares)

    def calculate_dynamic_stop_loss(self,
                                   entry_price: float,
                                   atr: float,
                                   side: str = "long",
                                   atr_multiplier: float = 2.0) -> float:
        """
        Berechnet dynamischen Stop-Loss basierend auf ATR

        Args:
            entry_price: Einstiegspreis
            atr: Average True Range
            side: 'long' oder 'short'
            atr_multiplier: ATR Multiplier (default: 2.0)

        Returns:
            Stop-Loss Preis
        """
        if side == "long":
            stop_loss = entry_price - (atr * atr_multiplier)
        else:  # short
            stop_loss = entry_price + (atr * atr_multiplier)

        logger.info(f"Dynamic Stop-Loss: ${stop_loss:.2f} (ATR: ${atr:.2f})")

        return stop_loss

    def calculate_take_profit(self,
                            entry_price: float,
                            stop_loss: float,
                            risk_reward_ratio: float = 2.0,
                            side: str = "long") -> float:
        """
        Berechnet Take-Profit Level

        Args:
            entry_price: Einstiegspreis
            stop_loss: Stop-Loss Preis
            risk_reward_ratio: Risk/Reward Ratio
            side: 'long' oder 'short'

        Returns:
            Take-Profit Preis
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio

        if side == "long":
            take_profit = entry_price + reward
        else:  # short
            take_profit = entry_price - reward

        logger.info(f"Take-Profit: ${take_profit:.2f} (R:R = 1:{risk_reward_ratio})")

        return take_profit

    def check_daily_loss_limit(self, current_portfolio_value: float) -> bool:
        """
        Überprüft ob Daily Loss Limit erreicht

        Returns:
            True wenn Limit erreicht, False sonst
        """
        daily_loss_pct = (self.initial_capital - current_portfolio_value) / self.initial_capital

        if daily_loss_pct >= self.max_daily_loss_pct:
            logger.warning(f"⚠️ Daily Loss Limit erreicht: {daily_loss_pct*100:.2f}%")
            return True

        return False

    def check_max_drawdown(self, current_portfolio_value: float) -> bool:
        """
        Überprüft Max Drawdown

        Returns:
            True wenn Limit erreicht, False sonst
        """
        # Update Peak
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value

        # Calculate Drawdown
        drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value

        if drawdown >= self.max_drawdown_pct:
            logger.warning(f"⚠️ Max Drawdown erreicht: {drawdown*100:.2f}%")
            return True

        return False

    def should_reduce_risk(self, current_portfolio_value: float, num_losses: int) -> bool:
        """
        Entscheidet ob Risk reduziert werden sollte

        Args:
            current_portfolio_value: Aktueller Portfolio-Wert
            num_losses: Anzahl aufeinanderfolgender Verluste

        Returns:
            True wenn Risk reduziert werden sollte
        """
        # Bei 3+ aufeinanderfolgenden Verlusten
        if num_losses >= 3:
            logger.warning("⚠️ 3+ aufeinanderfolgende Verluste → Risk reduzieren")
            return True

        # Bei Drawdown > 10%
        drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
        if drawdown > 0.10:
            logger.warning(f"⚠️ Drawdown > 10% ({drawdown*100:.1f}%) → Risk reduzieren")
            return True

        return False

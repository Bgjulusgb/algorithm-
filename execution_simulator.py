"""
Execution Simulator Module
Realistische Simulation von Order-Ausführungen mit Slippage, Market Impact, etc.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Marktbedingungen"""
    NORMAL = "normal"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"
    CRISIS = "crisis"


class ExecutionSimulator:
    """
    Simuliert realistische Order-Ausführungen

    Berücksichtigt:
    - Slippage (fixer und variabler Anteil)
    - Market Impact (abhängig von Order-Größe vs. Volumen)
    - Spread Costs (Bid-Ask Spread)
    - Latenz (Zeitverzögerung zwischen Signal und Ausführung)
    """

    def __init__(self,
                 base_slippage_bps: float = 5.0,
                 market_impact_coefficient: float = 0.1,
                 spread_bps: float = 2.0,
                 latency_bars: int = 0):
        """
        Initialisiert Execution Simulator

        Args:
            base_slippage_bps: Basis-Slippage in Basispunkten (1 bps = 0.01%)
            market_impact_coefficient: Market Impact Koeffizient
            spread_bps: Durchschnittlicher Bid-Ask Spread in Basispunkten
            latency_bars: Anzahl Bars Verzögerung zwischen Signal und Ausführung
        """
        self.base_slippage_bps = base_slippage_bps
        self.market_impact_coefficient = market_impact_coefficient
        self.spread_bps = spread_bps
        self.latency_bars = latency_bars

    def calculate_slippage(self,
                          price: float,
                          volatility: float,
                          market_condition: MarketCondition = MarketCondition.NORMAL) -> float:
        """
        Berechnet Slippage basierend auf Marktbedingungen

        Args:
            price: Aktueller Preis
            volatility: Historische Volatilität
            market_condition: Marktbedingung

        Returns:
            Slippage in absoluten Preis-Einheiten
        """
        # Basis-Slippage
        base_slippage = price * (self.base_slippage_bps / 10000)

        # Volatilitäts-Komponente
        volatility_slippage = price * volatility * 0.5

        # Marktbedingungen-Multiplikator
        condition_multiplier = {
            MarketCondition.NORMAL: 1.0,
            MarketCondition.VOLATILE: 2.0,
            MarketCondition.ILLIQUID: 3.0,
            MarketCondition.CRISIS: 5.0
        }

        multiplier = condition_multiplier.get(market_condition, 1.0)

        total_slippage = (base_slippage + volatility_slippage) * multiplier

        return total_slippage

    def calculate_market_impact(self,
                               order_size: float,
                               daily_volume: float,
                               price: float) -> float:
        """
        Berechnet Market Impact basierend auf Order-Größe vs. Volumen

        Market Impact = k * sqrt(order_size / daily_volume) * price

        Args:
            order_size: Order-Größe in Shares
            daily_volume: Durchschnittliches Tagesvolumen
            price: Aktueller Preis

        Returns:
            Market Impact in absoluten Preis-Einheiten
        """
        if daily_volume <= 0:
            return 0.0

        # Anteil des Tagesvolumens
        volume_fraction = order_size / daily_volume

        # Square-Root Market Impact Modell (Almgren-Chriss)
        impact = self.market_impact_coefficient * np.sqrt(volume_fraction) * price

        # Cap bei 10% des Preises
        impact = min(impact, price * 0.10)

        return impact

    def calculate_spread_cost(self, price: float) -> float:
        """
        Berechnet Bid-Ask Spread Kosten

        Args:
            price: Aktueller Mid-Price

        Returns:
            Spread-Kosten (halber Spread)
        """
        return price * (self.spread_bps / 10000) / 2

    def simulate_buy_execution(self,
                              signal_price: float,
                              signal_time_idx: int,
                              shares: int,
                              df: pd.DataFrame,
                              market_condition: MarketCondition = MarketCondition.NORMAL) -> Dict:
        """
        Simuliert Kauf-Ausführung mit allen Kosten

        Args:
            signal_price: Preis zum Zeitpunkt des Signals
            signal_time_idx: Index des Signal-Zeitpunkts
            shares: Anzahl zu kaufende Shares
            df: DataFrame mit OHLCV Daten
            market_condition: Marktbedingung

        Returns:
            Dict mit Ausführungs-Details
        """
        execution_idx = signal_time_idx + self.latency_bars

        if execution_idx >= len(df):
            logger.warning(f"Execution index {execution_idx} außerhalb DataFrame")
            return {'executed': False}

        # Ausführungspreis (mit Latenz)
        execution_price = df.iloc[execution_idx]['Close']

        # Volatilität (20-Tage ATR)
        if execution_idx >= 20:
            recent_returns = df.iloc[execution_idx-20:execution_idx]['Close'].pct_change()
            volatility = recent_returns.std()
        else:
            volatility = 0.02  # Default 2%

        # Tagesvolumen
        avg_volume = df.iloc[max(0, execution_idx-20):execution_idx]['Volume'].mean()

        # Berechne Kosten
        slippage = self.calculate_slippage(execution_price, volatility, market_condition)
        market_impact = self.calculate_market_impact(shares, avg_volume, execution_price)
        spread_cost = self.calculate_spread_cost(execution_price)

        # Total Kosten (BUY: alle Kosten erhöhen den Preis)
        total_cost_per_share = slippage + market_impact + spread_cost

        # Finaler Ausführungspreis
        final_price = execution_price + total_cost_per_share

        return {
            'executed': True,
            'signal_price': signal_price,
            'execution_price': execution_price,
            'final_price': final_price,
            'slippage': slippage,
            'market_impact': market_impact,
            'spread_cost': spread_cost,
            'total_cost': total_cost_per_share,
            'total_cost_pct': (total_cost_per_share / execution_price) * 100,
            'shares': shares,
            'execution_time_idx': execution_idx
        }

    def simulate_sell_execution(self,
                               signal_price: float,
                               signal_time_idx: int,
                               shares: int,
                               df: pd.DataFrame,
                               market_condition: MarketCondition = MarketCondition.NORMAL) -> Dict:
        """
        Simuliert Verkaufs-Ausführung mit allen Kosten

        Args:
            signal_price: Preis zum Zeitpunkt des Signals
            signal_time_idx: Index des Signal-Zeitpunkts
            shares: Anzahl zu verkaufende Shares
            df: DataFrame mit OHLCV Daten
            market_condition: Marktbedingung

        Returns:
            Dict mit Ausführungs-Details
        """
        execution_idx = signal_time_idx + self.latency_bars

        if execution_idx >= len(df):
            logger.warning(f"Execution index {execution_idx} außerhalb DataFrame")
            return {'executed': False}

        # Ausführungspreis (mit Latenz)
        execution_price = df.iloc[execution_idx]['Close']

        # Volatilität
        if execution_idx >= 20:
            recent_returns = df.iloc[execution_idx-20:execution_idx]['Close'].pct_change()
            volatility = recent_returns.std()
        else:
            volatility = 0.02

        # Tagesvolumen
        avg_volume = df.iloc[max(0, execution_idx-20):execution_idx]['Volume'].mean()

        # Berechne Kosten
        slippage = self.calculate_slippage(execution_price, volatility, market_condition)
        market_impact = self.calculate_market_impact(shares, avg_volume, execution_price)
        spread_cost = self.calculate_spread_cost(execution_price)

        # Total Kosten (SELL: alle Kosten reduzieren den erhaltenen Preis)
        total_cost_per_share = slippage + market_impact + spread_cost

        # Finaler Ausführungspreis
        final_price = execution_price - total_cost_per_share

        return {
            'executed': True,
            'signal_price': signal_price,
            'execution_price': execution_price,
            'final_price': final_price,
            'slippage': slippage,
            'market_impact': market_impact,
            'spread_cost': spread_cost,
            'total_cost': total_cost_per_share,
            'total_cost_pct': (total_cost_per_share / execution_price) * 100,
            'shares': shares,
            'execution_time_idx': execution_idx
        }

    def analyze_execution_quality(self, executions: list) -> Dict:
        """
        Analysiert Qualität der Ausführungen

        Args:
            executions: Liste von Execution Dicts

        Returns:
            Dict mit Analyse-Ergebnissen
        """
        if not executions:
            return {}

        executed = [e for e in executions if e.get('executed', False)]

        if not executed:
            return {}

        # Durchschnittliche Kosten
        avg_slippage = np.mean([e['slippage'] for e in executed])
        avg_market_impact = np.mean([e['market_impact'] for e in executed])
        avg_spread_cost = np.mean([e['spread_cost'] for e in executed])
        avg_total_cost = np.mean([e['total_cost'] for e in executed])
        avg_total_cost_pct = np.mean([e['total_cost_pct'] for e in executed])

        # Kosten-Verteilung
        cost_breakdown = {
            'slippage': avg_slippage,
            'market_impact': avg_market_impact,
            'spread': avg_spread_cost,
            'total': avg_total_cost
        }

        # Kosten-Anteil
        total = sum(cost_breakdown.values())
        cost_percentages = {k: (v/total*100) if total > 0 else 0 for k, v in cost_breakdown.items()}

        logger.info(f"\n📊 Execution Quality Analysis:")
        logger.info(f"   Total Executions: {len(executed)}")
        logger.info(f"   Avg Total Cost: {avg_total_cost_pct:.4f}%")
        logger.info(f"   Cost Breakdown:")
        logger.info(f"     - Slippage: {cost_percentages['slippage']:.1f}%")
        logger.info(f"     - Market Impact: {cost_percentages['market_impact']:.1f}%")
        logger.info(f"     - Spread: {cost_percentages['spread']:.1f}%")

        return {
            'num_executions': len(executed),
            'avg_slippage': avg_slippage,
            'avg_market_impact': avg_market_impact,
            'avg_spread_cost': avg_spread_cost,
            'avg_total_cost': avg_total_cost,
            'avg_total_cost_pct': avg_total_cost_pct,
            'cost_breakdown': cost_breakdown,
            'cost_percentages': cost_percentages
        }


class PartialFillSimulator:
    """
    Simuliert teilweise Ausführungen (Partial Fills)
    Wichtig für große Orders in illiquiden Märkten
    """

    def __init__(self, max_volume_participation: float = 0.10):
        """
        Initialisiert Partial Fill Simulator

        Args:
            max_volume_participation: Max Anteil des Volumens pro Bar (z.B. 0.10 = 10%)
        """
        self.max_volume_participation = max_volume_participation

    def simulate_partial_fill(self,
                             desired_shares: int,
                             df: pd.DataFrame,
                             start_idx: int,
                             side: str = 'BUY') -> Dict:
        """
        Simuliert schrittweise Ausführung einer großen Order

        Args:
            desired_shares: Gewünschte Anzahl Shares
            df: DataFrame mit OHLCV Daten
            start_idx: Start-Index für Ausführung
            side: 'BUY' oder 'SELL'

        Returns:
            Dict mit Partial Fill Details
        """
        fills = []
        remaining_shares = desired_shares
        current_idx = start_idx

        while remaining_shares > 0 and current_idx < len(df):
            # Verfügbares Volumen
            bar_volume = df.iloc[current_idx]['Volume']
            max_fillable = int(bar_volume * self.max_volume_participation)

            # Shares für diesen Bar
            fill_shares = min(remaining_shares, max_fillable)

            if fill_shares > 0:
                fills.append({
                    'idx': current_idx,
                    'price': df.iloc[current_idx]['Close'],
                    'shares': fill_shares,
                    'timestamp': df.index[current_idx]
                })

                remaining_shares -= fill_shares

            current_idx += 1

        # Durchschnittlicher Ausführungspreis (VWAP)
        if fills:
            total_value = sum(f['price'] * f['shares'] for f in fills)
            total_shares = sum(f['shares'] for f in fills)
            avg_price = total_value / total_shares if total_shares > 0 else 0

            filled_pct = (desired_shares - remaining_shares) / desired_shares * 100

            return {
                'fully_filled': remaining_shares == 0,
                'filled_shares': desired_shares - remaining_shares,
                'remaining_shares': remaining_shares,
                'filled_pct': filled_pct,
                'avg_price': avg_price,
                'num_fills': len(fills),
                'fills': fills,
                'bars_required': current_idx - start_idx
            }
        else:
            return {
                'fully_filled': False,
                'filled_shares': 0,
                'remaining_shares': desired_shares,
                'filled_pct': 0.0,
                'avg_price': 0.0,
                'num_fills': 0,
                'fills': [],
                'bars_required': 0
            }


class LiquidityAnalyzer:
    """
    Analysiert Markt-Liquidität
    """

    @staticmethod
    def calculate_liquidity_score(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Berechnet Liquiditäts-Score basierend auf Volumen und Spread

        Args:
            df: DataFrame mit OHLCV Daten
            window: Rolling Window für Durchschnitt

        Returns:
            Series mit Liquiditäts-Scores (0-100)
        """
        # Volumen-Komponente (höheres Volumen = besser)
        avg_volume = df['Volume'].rolling(window=window).mean()
        volume_percentile = df['Volume'].rolling(window=252).apply(
            lambda x: (x[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50
        )

        # Spread-Komponente (kleinerer Spread = besser)
        # Approximiere Spread als (High - Low) / Close
        spread_pct = (df['High'] - df['Low']) / df['Close'] * 100
        avg_spread = spread_pct.rolling(window=window).mean()

        # Inverser Spread-Score (niedriger Spread = höherer Score)
        spread_score = 100 - (avg_spread / avg_spread.max() * 100).fillna(50)

        # Kombinierter Score (60% Volumen, 40% Spread)
        liquidity_score = (volume_percentile * 0.6 + spread_score * 0.4).fillna(50)

        return liquidity_score

    @staticmethod
    def classify_market_condition(df: pd.DataFrame, idx: int, window: int = 20) -> MarketCondition:
        """
        Klassifiziert Marktbedingung basierend auf Volatilität und Liquidität

        Args:
            df: DataFrame mit OHLCV Daten
            idx: Index des aktuellen Bars
            window: Rolling Window

        Returns:
            MarketCondition
        """
        if idx < window:
            return MarketCondition.NORMAL

        # Volatilität
        returns = df.iloc[idx-window:idx]['Close'].pct_change()
        volatility = returns.std()

        # Volumen vs. Durchschnitt
        current_volume = df.iloc[idx]['Volume']
        avg_volume = df.iloc[idx-window:idx]['Volume'].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Klassifikation
        if volatility > 0.05:  # > 5% Tages-Volatilität
            return MarketCondition.CRISIS
        elif volatility > 0.03:  # > 3%
            return MarketCondition.VOLATILE
        elif volume_ratio < 0.5:  # < 50% durchschnittliches Volumen
            return MarketCondition.ILLIQUID
        else:
            return MarketCondition.NORMAL


class RealisticBacktester:
    """
    Backtester mit realistischer Ausführungs-Simulation
    """

    def __init__(self,
                 execution_simulator: ExecutionSimulator,
                 use_partial_fills: bool = False,
                 max_volume_participation: float = 0.10):
        """
        Initialisiert Realistic Backtester

        Args:
            execution_simulator: ExecutionSimulator Instance
            use_partial_fills: Partial Fills simulieren
            max_volume_participation: Max Volumen-Anteil pro Bar
        """
        self.execution_simulator = execution_simulator
        self.use_partial_fills = use_partial_fills
        self.partial_fill_simulator = PartialFillSimulator(max_volume_participation) if use_partial_fills else None
        self.executions = []

    def backtest_with_realistic_execution(self,
                                        df: pd.DataFrame,
                                        signals: pd.Series,
                                        position_size_func: callable) -> Dict:
        """
        Führt Backtest mit realistischer Ausführung durch

        Args:
            df: DataFrame mit OHLCV Daten
            signals: Series mit Signalen (1 = BUY, -1 = SELL, 0 = HOLD)
            position_size_func: Funktion die Position Size berechnet

        Returns:
            Dict mit Backtest-Ergebnissen
        """
        self.executions = []
        position = 0
        cash = 100000  # Start Cash
        equity_curve = []

        for i in range(len(df)):
            signal = signals.iloc[i]

            if signal == 1 and position == 0:  # BUY Signal
                # Berechne Position Size
                shares = position_size_func(cash, df.iloc[i]['Close'])

                # Klassifiziere Marktbedingung
                market_condition = LiquidityAnalyzer.classify_market_condition(df, i)

                # Simuliere Ausführung
                execution = self.execution_simulator.simulate_buy_execution(
                    df.iloc[i]['Close'], i, shares, df, market_condition
                )

                if execution['executed']:
                    cost = execution['final_price'] * shares
                    if cost <= cash:
                        cash -= cost
                        position = shares
                        self.executions.append(execution)

            elif signal == -1 and position > 0:  # SELL Signal
                market_condition = LiquidityAnalyzer.classify_market_condition(df, i)

                execution = self.execution_simulator.simulate_sell_execution(
                    df.iloc[i]['Close'], i, position, df, market_condition
                )

                if execution['executed']:
                    cash += execution['final_price'] * position
                    position = 0
                    self.executions.append(execution)

            # Equity Curve
            current_value = cash + (position * df.iloc[i]['Close'] if position > 0 else 0)
            equity_curve.append(current_value)

        # Analyse
        execution_analysis = self.execution_simulator.analyze_execution_quality(self.executions)

        final_value = equity_curve[-1] if equity_curve else 100000
        total_return = (final_value - 100000) / 100000

        return {
            'final_value': final_value,
            'total_return': total_return,
            'equity_curve': equity_curve,
            'num_trades': len(self.executions),
            'execution_analysis': execution_analysis
        }

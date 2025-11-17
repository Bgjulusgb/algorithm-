"""
Portfolio Rebalancing Module
Intelligentes Portfolio-Rebalancing mit verschiedenen Strategien
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RebalancingStrategy(Enum):
    """Rebalancing-Strategien"""
    FIXED_WEIGHT = "fixed_weight"  # Fixe Gewichtung
    EQUAL_WEIGHT = "equal_weight"  # Gleichgewichtung
    RISK_PARITY = "risk_parity"    # Risiko-Parität
    MIN_VARIANCE = "min_variance"  # Minimum Varianz
    MAX_SHARPE = "max_sharpe"      # Maximum Sharpe Ratio
    MOMENTUM = "momentum"          # Momentum-basiert
    MEAN_REVERSION = "mean_reversion"  # Mean-Reversion


class RebalancingTrigger(Enum):
    """Rebalancing-Trigger"""
    TIME_BASED = "time_based"      # Zeitbasiert (monatlich, quarterly, etc.)
    DRIFT_BASED = "drift_based"    # Drift-basiert (>X% Abweichung)
    VOLATILITY_BASED = "volatility_based"  # Volatilitäts-basiert
    MOMENTUM_BASED = "momentum_based"      # Momentum-basiert


class PortfolioRebalancer:
    """
    Portfolio Rebalancer mit mehreren Strategien
    """

    def __init__(self,
                 strategy: RebalancingStrategy = RebalancingStrategy.EQUAL_WEIGHT,
                 trigger: RebalancingTrigger = RebalancingTrigger.TIME_BASED,
                 rebalance_frequency_days: int = 30,
                 drift_threshold: float = 0.05,
                 transaction_cost_bps: float = 10.0):
        """
        Initialisiert Portfolio Rebalancer

        Args:
            strategy: Rebalancing-Strategie
            trigger: Rebalancing-Trigger
            rebalance_frequency_days: Tage zwischen Rebalances (für TIME_BASED)
            drift_threshold: Drift-Schwelle für DRIFT_BASED (z.B. 0.05 = 5%)
            transaction_cost_bps: Transaktionskosten in Basispunkten
        """
        self.strategy = strategy
        self.trigger = trigger
        self.rebalance_frequency_days = rebalance_frequency_days
        self.drift_threshold = drift_threshold
        self.transaction_cost_bps = transaction_cost_bps
        self.last_rebalance_date = None
        self.rebalance_history = []

    def calculate_target_weights(self,
                                current_positions: Dict[str, float],
                                price_data: Dict[str, pd.DataFrame],
                                current_date: datetime) -> Dict[str, float]:
        """
        Berechnet Ziel-Gewichtungen basierend auf Strategie

        Args:
            current_positions: Aktuelle Positionen {symbol: shares}
            price_data: Preis-Daten {symbol: DataFrame}
            current_date: Aktuelles Datum

        Returns:
            Dict mit Ziel-Gewichtungen {symbol: weight}
        """
        symbols = list(current_positions.keys())

        if self.strategy == RebalancingStrategy.EQUAL_WEIGHT:
            return self._equal_weight(symbols)

        elif self.strategy == RebalancingStrategy.RISK_PARITY:
            return self._risk_parity(symbols, price_data, current_date)

        elif self.strategy == RebalancingStrategy.MIN_VARIANCE:
            return self._min_variance(symbols, price_data, current_date)

        elif self.strategy == RebalancingStrategy.MAX_SHARPE:
            return self._max_sharpe(symbols, price_data, current_date)

        elif self.strategy == RebalancingStrategy.MOMENTUM:
            return self._momentum_based(symbols, price_data, current_date)

        elif self.strategy == RebalancingStrategy.MEAN_REVERSION:
            return self._mean_reversion(symbols, price_data, current_date)

        else:  # FIXED_WEIGHT default
            return self._equal_weight(symbols)

    def _equal_weight(self, symbols: List[str]) -> Dict[str, float]:
        """Gleichgewichtung aller Assets"""
        n = len(symbols)
        return {symbol: 1.0 / n for symbol in symbols}

    def _risk_parity(self,
                    symbols: List[str],
                    price_data: Dict[str, pd.DataFrame],
                    current_date: datetime) -> Dict[str, float]:
        """
        Risk Parity: Gleiches Risiko-Beitrag pro Asset

        Gewichtung: w_i = (1/σ_i) / Σ(1/σ_j)
        """
        volatilities = {}

        for symbol in symbols:
            if symbol in price_data:
                df = price_data[symbol]
                # Letzten 60 Tage
                recent_data = df[df.index <= current_date].tail(60)
                if len(recent_data) > 1:
                    returns = recent_data['Close'].pct_change().dropna()
                    vol = returns.std() * np.sqrt(252)  # Annualisiert
                    volatilities[symbol] = vol if vol > 0 else 0.01
                else:
                    volatilities[symbol] = 0.01
            else:
                volatilities[symbol] = 0.01

        # Inverse Volatilität
        inv_vol = {symbol: 1.0 / vol for symbol, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())

        weights = {symbol: inv_vol[symbol] / total_inv_vol for symbol in symbols}

        logger.info(f"Risk Parity Weights: {weights}")

        return weights

    def _min_variance(self,
                     symbols: List[str],
                     price_data: Dict[str, pd.DataFrame],
                     current_date: datetime) -> Dict[str, float]:
        """
        Minimum Variance Portfolio

        Minimiert Portfolio-Varianz unter Nebenbedingung dass Summe = 1
        """
        try:
            from scipy.optimize import minimize

            # Returns-Matrix erstellen
            returns_dict = {}
            for symbol in symbols:
                if symbol in price_data:
                    df = price_data[symbol]
                    recent_data = df[df.index <= current_date].tail(60)
                    if len(recent_data) > 1:
                        returns = recent_data['Close'].pct_change().dropna()
                        returns_dict[symbol] = returns

            if not returns_dict:
                return self._equal_weight(symbols)

            # Align returns
            returns_df = pd.DataFrame(returns_dict).fillna(0)
            cov_matrix = returns_df.cov().values

            # Optimierung
            n = len(symbols)
            init_weights = np.array([1.0 / n] * n)

            def portfolio_variance(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))

            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n))

            result = minimize(
                portfolio_variance,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                optimal_weights = dict(zip(symbols, result.x))
                logger.info(f"Min Variance Weights: {optimal_weights}")
                return optimal_weights
            else:
                return self._equal_weight(symbols)

        except Exception as e:
            logger.warning(f"Min Variance Optimierung fehlgeschlagen: {e}")
            return self._equal_weight(symbols)

    def _max_sharpe(self,
                   symbols: List[str],
                   price_data: Dict[str, pd.DataFrame],
                   current_date: datetime) -> Dict[str, float]:
        """
        Maximum Sharpe Ratio Portfolio
        """
        try:
            from scipy.optimize import minimize

            # Returns-Matrix erstellen
            returns_dict = {}
            for symbol in symbols:
                if symbol in price_data:
                    df = price_data[symbol]
                    recent_data = df[df.index <= current_date].tail(60)
                    if len(recent_data) > 1:
                        returns = recent_data['Close'].pct_change().dropna()
                        returns_dict[symbol] = returns

            if not returns_dict:
                return self._equal_weight(symbols)

            returns_df = pd.DataFrame(returns_dict).fillna(0)
            mean_returns = returns_df.mean().values * 252  # Annualisiert
            cov_matrix = returns_df.cov().values * 252

            n = len(symbols)
            init_weights = np.array([1.0 / n] * n)

            def negative_sharpe(weights):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                sharpe = portfolio_return / portfolio_std if portfolio_std > 0 else 0
                return -sharpe  # Negative für Minimierung

            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n))

            result = minimize(
                negative_sharpe,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                optimal_weights = dict(zip(symbols, result.x))
                logger.info(f"Max Sharpe Weights: {optimal_weights}")
                return optimal_weights
            else:
                return self._equal_weight(symbols)

        except Exception as e:
            logger.warning(f"Max Sharpe Optimierung fehlgeschlagen: {e}")
            return self._equal_weight(symbols)

    def _momentum_based(self,
                       symbols: List[str],
                       price_data: Dict[str, pd.DataFrame],
                       current_date: datetime) -> Dict[str, float]:
        """
        Momentum-basierte Gewichtung

        Höheres Gewicht für Assets mit stärkerem Momentum
        """
        momentums = {}

        for symbol in symbols:
            if symbol in price_data:
                df = price_data[symbol]
                recent_data = df[df.index <= current_date].tail(60)
                if len(recent_data) >= 2:
                    # 60-Tage Momentum
                    momentum = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0]) - 1
                    momentums[symbol] = max(momentum, 0)  # Nur positive Momentums
                else:
                    momentums[symbol] = 0
            else:
                momentums[symbol] = 0

        total_momentum = sum(momentums.values())

        if total_momentum > 0:
            weights = {symbol: mom / total_momentum for symbol, mom in momentums.items()}
        else:
            weights = self._equal_weight(symbols)

        logger.info(f"Momentum Weights: {weights}")

        return weights

    def _mean_reversion(self,
                       symbols: List[str],
                       price_data: Dict[str, pd.DataFrame],
                       current_date: datetime) -> Dict[str, float]:
        """
        Mean-Reversion basierte Gewichtung

        Höheres Gewicht für Assets die unter ihrem Mittelwert sind
        """
        deviations = {}

        for symbol in symbols:
            if symbol in price_data:
                df = price_data[symbol]
                recent_data = df[df.index <= current_date].tail(100)
                if len(recent_data) >= 2:
                    current_price = recent_data['Close'].iloc[-1]
                    mean_price = recent_data['Close'].mean()
                    # Abweichung vom Mittelwert (negativ = unterbewertet)
                    deviation = (current_price / mean_price) - 1
                    # Inverse: mehr Gewicht bei negativer Abweichung
                    deviations[symbol] = max(-deviation, 0)
                else:
                    deviations[symbol] = 0
            else:
                deviations[symbol] = 0

        total_deviation = sum(deviations.values())

        if total_deviation > 0:
            weights = {symbol: dev / total_deviation for symbol, dev in deviations.items()}
        else:
            weights = self._equal_weight(symbols)

        logger.info(f"Mean Reversion Weights: {weights}")

        return weights

    def should_rebalance(self,
                        current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        current_date: datetime,
                        volatility: Optional[float] = None) -> bool:
        """
        Entscheidet ob Rebalancing notwendig ist

        Args:
            current_weights: Aktuelle Gewichtungen
            target_weights: Ziel-Gewichtungen
            current_date: Aktuelles Datum
            volatility: Portfolio-Volatilität (für VOLATILITY_BASED)

        Returns:
            True wenn Rebalancing durchgeführt werden soll
        """
        if self.trigger == RebalancingTrigger.TIME_BASED:
            if self.last_rebalance_date is None:
                return True

            days_since_rebalance = (current_date - self.last_rebalance_date).days
            return days_since_rebalance >= self.rebalance_frequency_days

        elif self.trigger == RebalancingTrigger.DRIFT_BASED:
            # Berechne maximalen Drift
            max_drift = 0
            for symbol in target_weights:
                current_w = current_weights.get(symbol, 0)
                target_w = target_weights.get(symbol, 0)
                drift = abs(current_w - target_w)
                max_drift = max(max_drift, drift)

            return max_drift > self.drift_threshold

        elif self.trigger == RebalancingTrigger.VOLATILITY_BASED:
            # Rebalance wenn Volatilität hoch ist
            if volatility is None:
                return False
            return volatility > 0.25  # > 25% annualisierte Vol

        else:
            return False

    def calculate_rebalancing_trades(self,
                                    current_positions: Dict[str, float],
                                    current_prices: Dict[str, float],
                                    target_weights: Dict[str, float],
                                    total_value: float) -> Dict[str, Dict]:
        """
        Berechnet notwendige Trades für Rebalancing

        Args:
            current_positions: Aktuelle Positionen {symbol: shares}
            current_prices: Aktuelle Preise {symbol: price}
            target_weights: Ziel-Gewichtungen {symbol: weight}
            total_value: Gesamtwert des Portfolios

        Returns:
            Dict mit Trade-Details {symbol: {'action': 'BUY/SELL', 'shares': X, 'value': Y}}
        """
        # Aktuelle Werte
        current_values = {symbol: shares * current_prices.get(symbol, 0)
                         for symbol, shares in current_positions.items()}

        # Ziel-Werte
        target_values = {symbol: weight * total_value
                        for symbol, weight in target_weights.items()}

        # Notwendige Trades
        trades = {}

        for symbol in set(list(current_positions.keys()) + list(target_weights.keys())):
            current_value = current_values.get(symbol, 0)
            target_value = target_values.get(symbol, 0)
            diff_value = target_value - current_value

            if abs(diff_value) > 1:  # Nur Trades > $1
                current_price = current_prices.get(symbol, 0)
                if current_price > 0:
                    shares_to_trade = int(diff_value / current_price)

                    if shares_to_trade != 0:
                        trades[symbol] = {
                            'action': 'BUY' if shares_to_trade > 0 else 'SELL',
                            'shares': abs(shares_to_trade),
                            'value': abs(diff_value),
                            'current_weight': current_value / total_value if total_value > 0 else 0,
                            'target_weight': target_weights.get(symbol, 0)
                        }

        return trades

    def execute_rebalance(self,
                         current_positions: Dict[str, float],
                         current_prices: Dict[str, float],
                         price_data: Dict[str, pd.DataFrame],
                         current_date: datetime,
                         total_value: float) -> Dict:
        """
        Führt vollständiges Rebalancing durch

        Args:
            current_positions: Aktuelle Positionen
            current_prices: Aktuelle Preise
            price_data: Historische Preis-Daten
            current_date: Aktuelles Datum
            total_value: Portfolio-Gesamtwert

        Returns:
            Dict mit Rebalancing-Details
        """
        # Berechne aktuelle Gewichte
        current_weights = {}
        for symbol, shares in current_positions.items():
            value = shares * current_prices.get(symbol, 0)
            current_weights[symbol] = value / total_value if total_value > 0 else 0

        # Berechne Ziel-Gewichte
        target_weights = self.calculate_target_weights(
            current_positions, price_data, current_date
        )

        # Prüfe ob Rebalancing notwendig
        if not self.should_rebalance(current_weights, target_weights, current_date):
            logger.info("Kein Rebalancing notwendig")
            return {'rebalanced': False}

        # Berechne Trades
        trades = self.calculate_rebalancing_trades(
            current_positions, current_prices, target_weights, total_value
        )

        # Berechne Transaktionskosten
        total_turnover = sum(t['value'] for t in trades.values())
        transaction_costs = total_turnover * (self.transaction_cost_bps / 10000)

        # Speichere in Historie
        rebalance_record = {
            'date': current_date,
            'strategy': self.strategy.value,
            'current_weights': current_weights.copy(),
            'target_weights': target_weights.copy(),
            'trades': trades.copy(),
            'transaction_costs': transaction_costs,
            'turnover': total_turnover
        }

        self.rebalance_history.append(rebalance_record)
        self.last_rebalance_date = current_date

        logger.info(f"\n🔄 Rebalancing ausgeführt:")
        logger.info(f"   Strategie: {self.strategy.value}")
        logger.info(f"   Anzahl Trades: {len(trades)}")
        logger.info(f"   Turnover: ${total_turnover:,.2f}")
        logger.info(f"   Transaktionskosten: ${transaction_costs:,.2f}")

        return {
            'rebalanced': True,
            'trades': trades,
            'target_weights': target_weights,
            'transaction_costs': transaction_costs,
            'turnover': total_turnover
        }

    def get_rebalancing_stats(self) -> Dict:
        """
        Gibt Statistiken über Rebalancing-Historie zurück

        Returns:
            Dict mit Statistiken
        """
        if not self.rebalance_history:
            return {}

        num_rebalances = len(self.rebalance_history)
        total_costs = sum(r['transaction_costs'] for r in self.rebalance_history)
        total_turnover = sum(r['turnover'] for r in self.rebalance_history)
        avg_turnover = total_turnover / num_rebalances

        return {
            'num_rebalances': num_rebalances,
            'total_transaction_costs': total_costs,
            'total_turnover': total_turnover,
            'avg_turnover_per_rebalance': avg_turnover,
            'avg_cost_per_rebalance': total_costs / num_rebalances
        }

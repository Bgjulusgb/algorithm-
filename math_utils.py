"""
Erweiterte Mathematik- und Statistik-Module für Trading Bot
Enthält fortgeschrittene mathematische Analysen und Berechnungen
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from scipy import stats
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class StatisticsAnalyzer:
    """Erweiterte statistische Analysen für Trading-Daten"""

    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """
        Berechnet logarithmische Returns

        Args:
            prices: Preis-Serie

        Returns:
            Log-Returns
        """
        return np.log(prices / prices.shift(1))

    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 30, annualize: bool = True) -> pd.Series:
        """
        Berechnet rollende Volatilität

        Args:
            returns: Return-Serie
            window: Fenster für rollende Berechnung
            annualize: Annualisieren (252 Trading-Tage)

        Returns:
            Volatilitäts-Serie
        """
        vol = returns.rolling(window=window, min_periods=window).std()
        if annualize:
            vol = vol * np.sqrt(252)
        return vol

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Berechnet Sharpe Ratio

        Args:
            returns: Return-Serie
            risk_free_rate: Risikofreier Zins (annualisiert)
            periods_per_year: Perioden pro Jahr

        Returns:
            Sharpe Ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
        return float(sharpe)

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Berechnet Sortino Ratio (nur Downside-Volatilität)

        Args:
            returns: Return-Serie
            risk_free_rate: Risikofreier Zins
            periods_per_year: Perioden pro Jahr

        Returns:
            Sortino Ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = np.sqrt(periods_per_year) * (excess_returns.mean() / downside_returns.std())
        return float(sortino)

    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Berechnet maximalen Drawdown

        Args:
            prices: Preis-Serie

        Returns:
            Tuple (max_drawdown, start_date, end_date)
        """
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        end_date = drawdown.idxmin()
        start_date = cumulative[:end_date].idxmax()

        return float(max_dd), start_date, end_date

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, max_drawdown: float) -> float:
        """
        Berechnet Calmar Ratio (Return / Max Drawdown)

        Args:
            returns: Return-Serie
            max_drawdown: Maximaler Drawdown

        Returns:
            Calmar Ratio
        """
        if max_drawdown == 0:
            return 0.0

        annual_return = (1 + returns.mean()) ** 252 - 1
        calmar = annual_return / abs(max_drawdown)
        return float(calmar)

    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Berechnet Value at Risk (VaR)

        Args:
            returns: Return-Serie
            confidence_level: Konfidenzniveau (z.B. 0.95 für 95%)

        Returns:
            VaR
        """
        if len(returns) == 0:
            return 0.0

        var = np.percentile(returns, (1 - confidence_level) * 100)
        return float(var)

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Berechnet Conditional Value at Risk (CVaR / Expected Shortfall)

        Args:
            returns: Return-Serie
            confidence_level: Konfidenzniveau

        Returns:
            CVaR
        """
        if len(returns) == 0:
            return 0.0

        var = StatisticsAnalyzer.calculate_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        return float(cvar)

    @staticmethod
    def calculate_skewness(returns: pd.Series) -> float:
        """Berechnet Schiefe der Return-Verteilung"""
        if len(returns) < 3:
            return 0.0
        return float(stats.skew(returns.dropna()))

    @staticmethod
    def calculate_kurtosis(returns: pd.Series) -> float:
        """Berechnet Kurtosis (Wölbung) der Return-Verteilung"""
        if len(returns) < 4:
            return 0.0
        return float(stats.kurtosis(returns.dropna()))

    @staticmethod
    def calculate_information_ratio(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Berechnet Information Ratio

        Args:
            portfolio_returns: Portfolio-Returns
            benchmark_returns: Benchmark-Returns

        Returns:
            Information Ratio
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std()

        if tracking_error == 0:
            return 0.0

        ir = active_returns.mean() / tracking_error
        return float(ir)


class CorrelationAnalyzer:
    """Korrelations- und Kovarianz-Analysen"""

    @staticmethod
    def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet Korrelationsmatrix

        Args:
            returns_df: DataFrame mit Returns für verschiedene Assets

        Returns:
            Korrelationsmatrix
        """
        return returns_df.corr()

    @staticmethod
    def calculate_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet Kovarianzmatrix

        Args:
            returns_df: DataFrame mit Returns

        Returns:
            Kovarianzmatrix
        """
        return returns_df.cov()

    @staticmethod
    def find_uncorrelated_pairs(
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """
        Findet Paare mit niedriger Korrelation (für Diversifikation)

        Args:
            correlation_matrix: Korrelationsmatrix
            threshold: Maximale Korrelation

        Returns:
            Liste von (symbol1, symbol2, correlation)
        """
        uncorrelated = []
        symbols = correlation_matrix.columns

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                corr = correlation_matrix.loc[sym1, sym2]
                if abs(corr) <= threshold:
                    uncorrelated.append((sym1, sym2, float(corr)))

        return sorted(uncorrelated, key=lambda x: abs(x[2]))

    @staticmethod
    def calculate_portfolio_variance(
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> float:
        """
        Berechnet Portfolio-Varianz

        Args:
            weights: Asset-Gewichte
            covariance_matrix: Kovarianzmatrix

        Returns:
            Portfolio-Varianz
        """
        return float(weights @ covariance_matrix @ weights.T)

    @staticmethod
    def calculate_portfolio_volatility(
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        annualize: bool = True
    ) -> float:
        """
        Berechnet Portfolio-Volatilität

        Args:
            weights: Asset-Gewichte
            covariance_matrix: Kovarianzmatrix
            annualize: Annualisieren

        Returns:
            Portfolio-Volatilität
        """
        variance = CorrelationAnalyzer.calculate_portfolio_variance(weights, covariance_matrix)
        volatility = np.sqrt(variance)

        if annualize:
            volatility *= np.sqrt(252)

        return float(volatility)


class KellyCriterion:
    """Kelly Criterion für optimales Position Sizing"""

    @staticmethod
    def calculate_kelly_fraction(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.5
    ) -> float:
        """
        Berechnet optimale Position Size nach Kelly Criterion

        Args:
            win_rate: Gewinnwahrscheinlichkeit (0-1)
            avg_win: Durchschnittlicher Gewinn
            avg_loss: Durchschnittlicher Verlust (positiv)
            kelly_fraction: Kelly-Multiplikator (0.5 = Half-Kelly, konservativ)

        Returns:
            Optimale Position Size als Anteil des Kapitals (0-1)
        """
        if avg_loss == 0 or win_rate >= 1 or win_rate <= 0:
            return 0.0

        # Kelly Formula: f* = (p * b - q) / b
        # p = win_rate, q = 1 - win_rate, b = avg_win / avg_loss
        b = avg_win / avg_loss
        q = 1 - win_rate

        kelly = (win_rate * b - q) / b

        # Clamp zwischen 0 und 1, multipliziere mit kelly_fraction
        kelly = max(0.0, min(1.0, kelly)) * kelly_fraction

        return float(kelly)

    @staticmethod
    def calculate_kelly_from_trades(
        trades: pd.DataFrame,
        kelly_fraction: float = 0.5
    ) -> float:
        """
        Berechnet Kelly aus historischen Trades

        Args:
            trades: DataFrame mit 'profit_loss' Spalte
            kelly_fraction: Kelly-Multiplikator

        Returns:
            Optimale Position Size
        """
        if len(trades) == 0 or 'profit_loss' not in trades.columns:
            return 0.0

        profits = trades[trades['profit_loss'] > 0]['profit_loss']
        losses = trades[trades['profit_loss'] < 0]['profit_loss'].abs()

        if len(profits) == 0 or len(losses) == 0:
            return 0.0

        win_rate = len(profits) / len(trades)
        avg_win = profits.mean()
        avg_loss = losses.mean()

        return KellyCriterion.calculate_kelly_fraction(
            win_rate, avg_win, avg_loss, kelly_fraction
        )


class MonteCarloSimulator:
    """Monte Carlo Simulation für Risk Analysis"""

    @staticmethod
    def simulate_portfolio(
        initial_capital: float,
        expected_return: float,
        volatility: float,
        num_simulations: int = 1000,
        num_days: int = 252
    ) -> np.ndarray:
        """
        Simuliert Portfolio-Entwicklung mit Monte Carlo

        Args:
            initial_capital: Startkapital
            expected_return: Erwartete annualisierte Rendite
            volatility: Annualisierte Volatilität
            num_simulations: Anzahl Simulationen
            num_days: Anzahl Tage

        Returns:
            Array mit Simulationsergebnissen (num_simulations x num_days)
        """
        # Konvertiere annualisierte Werte zu täglich
        daily_return = expected_return / 252
        daily_volatility = volatility / np.sqrt(252)

        # Generiere Zufallsreturns
        random_returns = np.random.normal(
            daily_return,
            daily_volatility,
            (num_simulations, num_days)
        )

        # Berechne kumulative Returns
        portfolio_values = initial_capital * (1 + random_returns).cumprod(axis=1)

        return portfolio_values

    @staticmethod
    def analyze_simulation_results(
        simulations: np.ndarray,
        confidence_levels: List[float] = [0.05, 0.50, 0.95]
    ) -> Dict[str, float]:
        """
        Analysiert Monte Carlo Simulationsergebnisse

        Args:
            simulations: Simulationsergebnisse
            confidence_levels: Konfidenzniveaus

        Returns:
            Dictionary mit Statistiken
        """
        final_values = simulations[:, -1]

        results = {
            'mean': float(np.mean(final_values)),
            'median': float(np.median(final_values)),
            'std': float(np.std(final_values)),
            'min': float(np.min(final_values)),
            'max': float(np.max(final_values)),
        }

        for level in confidence_levels:
            percentile = level * 100
            results[f'percentile_{int(percentile)}'] = float(
                np.percentile(final_values, percentile)
            )

        return results

    @staticmethod
    def calculate_probability_of_ruin(
        simulations: np.ndarray,
        ruin_threshold: float
    ) -> float:
        """
        Berechnet Wahrscheinlichkeit, unter Schwellwert zu fallen

        Args:
            simulations: Simulationsergebnisse
            ruin_threshold: Ruinschwelle

        Returns:
            Wahrscheinlichkeit (0-1)
        """
        min_values = simulations.min(axis=1)
        num_ruined = np.sum(min_values < ruin_threshold)
        probability = num_ruined / len(simulations)
        return float(probability)


class SignalFilters:
    """Mathematische Filter für Trading-Signale"""

    @staticmethod
    def exponential_moving_average_filter(
        signal: pd.Series,
        span: int = 5
    ) -> pd.Series:
        """
        Glättet Signal mit Exponential Moving Average

        Args:
            signal: Signal-Serie
            span: EMA Span

        Returns:
            Geglättetes Signal
        """
        return signal.ewm(span=span, adjust=False).mean()

    @staticmethod
    def kalman_filter(
        signal: pd.Series,
        process_variance: float = 0.01,
        measurement_variance: float = 0.1
    ) -> pd.Series:
        """
        Wendet Kalman Filter auf Signal an

        Args:
            signal: Signal-Serie
            process_variance: Prozessvarianz
            measurement_variance: Messvarianz

        Returns:
            Gefiltertes Signal
        """
        n = len(signal)
        filtered = np.zeros(n)

        # Initialisierung
        x_est = signal.iloc[0]
        p_est = 1.0

        for i, measurement in enumerate(signal):
            # Prediction
            x_pred = x_est
            p_pred = p_est + process_variance

            # Update
            kalman_gain = p_pred / (p_pred + measurement_variance)
            x_est = x_pred + kalman_gain * (measurement - x_pred)
            p_est = (1 - kalman_gain) * p_pred

            filtered[i] = x_est

        return pd.Series(filtered, index=signal.index)

    @staticmethod
    def median_filter(signal: pd.Series, window: int = 5) -> pd.Series:
        """
        Wendet Median-Filter an (reduziert Ausreißer)

        Args:
            signal: Signal-Serie
            window: Fenster-Größe

        Returns:
            Gefiltertes Signal
        """
        return signal.rolling(window=window, center=True).median().fillna(signal)

    @staticmethod
    def butterworth_filter(
        signal: pd.Series,
        cutoff_freq: float = 0.1,
        order: int = 2
    ) -> pd.Series:
        """
        Wendet Butterworth Low-Pass Filter an

        Args:
            signal: Signal-Serie
            cutoff_freq: Grenzfrequenz (0-1)
            order: Filter-Ordnung

        Returns:
            Gefiltertes Signal
        """
        try:
            from scipy.signal import butter, filtfilt

            b, a = butter(order, cutoff_freq, btype='low')
            filtered = filtfilt(b, a, signal.values)

            return pd.Series(filtered, index=signal.index)

        except ImportError:
            logger.warning("scipy nicht verfügbar, verwende EMA Filter")
            return SignalFilters.exponential_moving_average_filter(signal)

    @staticmethod
    def z_score_normalization(signal: pd.Series, window: int = 20) -> pd.Series:
        """
        Normalisiert Signal mit Z-Score

        Args:
            signal: Signal-Serie
            window: Rollierendes Fenster

        Returns:
            Normalisiertes Signal
        """
        rolling_mean = signal.rolling(window=window, min_periods=1).mean()
        rolling_std = signal.rolling(window=window, min_periods=1).std()

        z_score = (signal - rolling_mean) / (rolling_std + 1e-10)
        return z_score

    @staticmethod
    def apply_threshold(
        signal: pd.Series,
        lower_threshold: float = -1.0,
        upper_threshold: float = 1.0
    ) -> pd.Series:
        """
        Wendet Schwellwerte auf Signal an

        Args:
            signal: Signal-Serie
            lower_threshold: Unterer Schwellwert
            upper_threshold: Oberer Schwellwert

        Returns:
            Schwellwert-gefiltertes Signal
        """
        return signal.clip(lower=lower_threshold, upper=upper_threshold)


class PortfolioOptimizer:
    """Portfolio-Optimierung mit Modern Portfolio Theory"""

    @staticmethod
    def calculate_efficient_frontier(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Efficient Frontier

        Args:
            expected_returns: Erwartete Returns
            cov_matrix: Kovarianzmatrix
            num_points: Anzahl Punkte auf der Frontier

        Returns:
            Tuple (returns, volatilities)
        """
        n_assets = len(expected_returns)

        def portfolio_stats(weights):
            ret = np.sum(expected_returns * weights)
            vol = np.sqrt(weights @ cov_matrix @ weights.T)
            return ret, vol

        target_returns = np.linspace(
            expected_returns.min(),
            expected_returns.max(),
            num_points
        )

        frontier_volatilities = []

        for target in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.sum(expected_returns * w) - target}
            ]
            bounds = tuple((0, 1) for _ in range(n_assets))

            result = minimize(
                lambda w: np.sqrt(w @ cov_matrix @ w.T),
                x0=np.array([1/n_assets] * n_assets),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                _, vol = portfolio_stats(result.x)
                frontier_volatilities.append(vol)
            else:
                frontier_volatilities.append(np.nan)

        return target_returns, np.array(frontier_volatilities)

    @staticmethod
    def optimize_sharpe_ratio(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> np.ndarray:
        """
        Optimiert Portfolio für maximale Sharpe Ratio

        Args:
            expected_returns: Erwartete Returns
            cov_matrix: Kovarianzmatrix
            risk_free_rate: Risikofreier Zins

        Returns:
            Optimale Gewichte
        """
        n_assets = len(expected_returns)

        def negative_sharpe(weights):
            ret = np.sum(expected_returns * weights)
            vol = np.sqrt(weights @ cov_matrix @ weights.T)
            sharpe = (ret - risk_free_rate) / vol
            return -sharpe

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))

        result = minimize(
            negative_sharpe,
            x0=np.array([1/n_assets] * n_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x if result.success else np.array([1/n_assets] * n_assets)

    @staticmethod
    def optimize_minimum_variance(
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Optimiert Portfolio für minimale Varianz

        Args:
            cov_matrix: Kovarianzmatrix

        Returns:
            Optimale Gewichte
        """
        n_assets = cov_matrix.shape[0]

        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights.T

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))

        result = minimize(
            portfolio_variance,
            x0=np.array([1/n_assets] * n_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x if result.success else np.array([1/n_assets] * n_assets)

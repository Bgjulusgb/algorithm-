
"""
Konfigurationsdatei für den Trading Bot - Verbesserte Version
"""
import os
from datetime import datetime

# ============================================================================
# TRADING SYMBOLE
# ============================================================================

# Hauptwatchlist
WATCHLIST = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "TSLA",   # Tesla
]

# Alternative Watchlists für verschiedene Strategien
WATCHLISTS = {
    "tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC"],
    "finance": ["JPM", "BAC", "GS", "MS", "C"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO"],
    "dividend": ["KO", "PEP", "JNJ", "PG", "MCD"],
}

# ============================================================================
# STRATEGIE PARAMETER
# ============================================================================

STRATEGY_CONFIG = {
    # Moving Average Parameter
    "short_window": 50,        # Kurzer Moving Average (Tage)
    "long_window": 200,        # Langer Moving Average (Tage)
    "ema_short": 12,           # Kurzer Exponentieller MA
    "ema_long": 26,            # Langer Exponentieller MA
    
    # RSI Parameter
    "rsi_period": 14,          # RSI Periode
    "rsi_oversold": 30,        # RSI Überverkauft Level
    "rsi_overbought": 70,      # RSI Überkauft Level
    "rsi_extreme_oversold": 20,  # Extrem überverkauft
    "rsi_extreme_overbought": 80,  # Extrem überkauft
    
    # MACD Parameter
    "macd_fast": 12,           # MACD Fast Period
    "macd_slow": 26,           # MACD Slow Period
    "macd_signal": 9,          # MACD Signal Period
    
    # Bollinger Bands
    "bb_period": 20,           # Bollinger Band Periode
    "bb_std": 2,               # Standard Deviations
    
    # Volume Parameter
    "volume_ma_period": 20,    # Volume Moving Average
    "volume_spike_threshold": 1.5,  # Volumen-Spike (1.5x average)
    
    # Signal Bestätigung
    "min_signal_strength": 2,  # Mindestens N Indikatoren müssen übereinstimmen
    "confirmation_periods": 2,  # Signalbestätigung über N Perioden
}

# ============================================================================
# PORTFOLIO PARAMETER
# ============================================================================

PORTFOLIO_CONFIG = {
    # Kapital
    "initial_capital": 100000.0,  # Startkapital in USD
    "min_cash_reserve": 0.10,     # Mindestens 10% Cash-Reserve
    
    # Position Sizing
    "max_position_size": 0.20,    # Maximal 20% des Kapitals pro Position
    "min_position_size": 0.02,    # Minimal 2% pro Position
    "max_positions": 8,           # Maximale Anzahl gleichzeitiger Positionen
    
    # Kosten
    "commission": 0.0,            # Kommission pro Trade (0 für Robinhood/modern brokers)
    "slippage": 0.001,            # 0.1% Slippage-Annahme
}

# ============================================================================
# RISIKO-MANAGEMENT
# ============================================================================

RISK_CONFIG = {
    # Stop Loss & Take Profit
    "use_stop_loss": True,
    "stop_loss_percent": 0.05,      # 5% Stop Loss
    "trailing_stop": True,           # Aktiviere Trailing Stop
    "trailing_stop_percent": 0.03,   # 3% Trailing Stop
    
    "use_take_profit": True,
    "take_profit_percent": 0.15,     # 15% Take Profit
    "partial_take_profit": True,     # Teilverkäufe bei Gewinn
    "partial_tp_percent": 0.10,      # 10% Partial Take Profit
    "partial_tp_amount": 0.50,       # Verkaufe 50% der Position
    
    # Portfolio Risk
    "max_portfolio_risk": 0.02,      # Max 2% Risiko pro Trade
    "max_daily_loss": 0.03,          # Max 3% Tagesverlust
    "max_weekly_loss": 0.08,         # Max 8% Wochenverlust
    "max_drawdown": 0.15,            # Max 15% Drawdown
    
    # Position Management
    "scale_in": True,                # Scale in Positionen
    "scale_in_steps": 3,             # In 3 Schritten einsteigen
    "scale_out": True,               # Scale out Positionen
    "scale_out_steps": 2,            # In 2 Schritten aussteigen
    
    # Diversifikation
    "max_sector_exposure": 0.40,     # Max 40% in einem Sektor
    "max_correlation": 0.70,         # Max Korrelation zwischen Positionen
}

# ============================================================================
# DATEN PARAMETER
# ============================================================================

DATA_CONFIG = {
    # Historische Daten
    "history_period": "2y",          # Historische Daten: 2 Jahre
    "interval": "1d",                # Tägliche Daten
    
    # Cache
    "enable_cache": True,            # Aktiviere Daten-Caching
    "cache_duration_hours": 1,       # Cache für 1 Stunde
    "cache_dir": "cache",            # Cache-Verzeichnis
    
    # API Limits
    "rate_limit_calls": 2000,        # Max API Calls pro Stunde
    "rate_limit_window": 3600,       # Zeitfenster in Sekunden
    "retry_attempts": 3,             # Wiederholungsversuche bei Fehler
    "retry_delay": 5,                # Sekunden zwischen Versuchen
    
    # Datenqualität
    "min_data_points": 100,          # Mindestanzahl Datenpunkte
    "max_missing_data": 0.05,        # Max 5% fehlende Daten erlaubt
}

# ============================================================================
# DATEI-PFADE
# ============================================================================

FILE_PATHS = {
    "trades_csv": "trades.csv",
    "portfolio_csv": "portfolio.csv",
    "performance_csv": "performance.csv",
    "log_file": "trading_bot.log",
    "positions_json": "positions.json",
    "alerts_log": "alerts.log",
}

# ============================================================================
# TRADING MODUS
# ============================================================================

# Modus: "backtest", "live", "paper"
TRADING_MODE = "backtest"

# Backtest Einstellungen
BACKTEST_CONFIG = {
    "start_date": None,              # None = automatisch basierend auf history_period
    "end_date": None,                # None = heute
    "initial_capital": 100000.0,
    "include_dividends": False,      # Dividenden einbeziehen
    "realistic_execution": True,     # Realistische Order-Ausführung
    "execution_price": "open",       # "open", "close", "average"
}

# Live Trading Einstellungen
LIVE_CONFIG = {
    "check_interval": 300,           # Prüfe alle 5 Minuten
    "market_hours_only": True,       # Nur während Handelszeiten
    "pre_market": False,             # Pre-Market Trading
    "after_hours": False,            # After-Hours Trading
    "max_orders_per_day": 20,        # Max Orders pro Tag
}

# ============================================================================
# LOGGING & BENACHRICHTIGUNGEN
# ============================================================================

LOGGING_CONFIG = {
    "log_level": "INFO",             # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "log_to_file": True,
    "log_to_console": True,
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_rotation": True,            # Rotate Logs
    "max_log_size_mb": 10,          # Max 10 MB pro Log-Datei
    "backup_count": 5,               # Behalte 5 alte Logs
}

NOTIFICATION_CONFIG = {
    "enable_notifications": False,
    "email": {
        "enabled": False,
        "smtp_server": "",
        "smtp_port": 587,
        "sender": "",
        "receiver": "",
        "password": "",
    },
    "webhook": {
        "enabled": False,
        "url": "",
    },
    "notify_on": {
        "trade_execution": True,
        "large_loss": True,           # Bei großem Verlust
        "large_gain": True,           # Bei großem Gewinn
        "stop_loss_hit": True,
        "take_profit_hit": True,
        "daily_summary": True,
    }
}

# ============================================================================
# PERFORMANCE & OPTIMIERUNG
# ============================================================================

PERFORMANCE_CONFIG = {
    "parallel_processing": True,      # Parallele Verarbeitung mehrerer Symbole
    "max_workers": 4,                 # Anzahl paralleler Worker
    "batch_size": 10,                 # Batch-Größe für Datenabfragen
}

# ============================================================================
# ERWEITERTE FEATURES
# ============================================================================

ADVANCED_FEATURES = {
    "sentiment_analysis": False,      # Sentiment-Analyse aktivieren
    "news_integration": False,        # News-Integration
    "earnings_calendar": False,       # Earnings berücksichtigen
    "sector_rotation": False,         # Sektor-Rotation Strategie
    "ml_predictions": False,          # Machine Learning Vorhersagen
}

# ============================================================================
# MATHEMATISCHE ANALYSEN
# ============================================================================

MATH_CONFIG = {
    # Kelly Criterion
    "use_kelly_criterion": False,     # Kelly Criterion für Position Sizing
    "kelly_fraction": 0.5,            # Half-Kelly (konservativ)

    # Signal-Filter
    "use_signal_filters": True,       # Mathematische Signal-Filter aktivieren
    "filter_type": "ema",             # ema, kalman, median, butterworth
    "filter_window": 3,               # Filter-Fenster

    # Performance Analytics
    "calculate_var": True,            # Value at Risk berechnen
    "var_confidence": 0.95,           # VaR Konfidenzniveau
    "calculate_sharpe": True,         # Sharpe Ratio berechnen
    "calculate_sortino": True,        # Sortino Ratio berechnen

    # Monte Carlo
    "run_monte_carlo": False,         # Monte Carlo Simulation ausführen
    "mc_simulations": 1000,           # Anzahl Simulationen
    "mc_days": 252,                   # Simulationszeitraum (Tage)

    # Correlation Analysis
    "analyze_correlation": True,      # Korrelations-Analyse für Portfolio
    "max_correlation": 0.7,           # Max Korrelation zwischen Positionen

    # Portfolio Optimization
    "optimize_portfolio": False,      # Portfolio-Optimierung aktivieren
    "optimization_method": "sharpe",  # sharpe, min_variance, max_return
}

# ============================================================================
# VALIDIERUNG
# ============================================================================

def validate_config():
    """Validiert die Konfiguration und gibt Warnungen aus"""
    warnings = []
    
    # Prüfe Portfolio-Limits
    if PORTFOLIO_CONFIG["max_position_size"] > 0.30:
        warnings.append("⚠️ WARNUNG: max_position_size > 30% ist sehr riskant!")
    
    if PORTFOLIO_CONFIG["min_cash_reserve"] < 0.05:
        warnings.append("⚠️ WARNUNG: min_cash_reserve < 5% könnte zu Liquiditätsproblemen führen!")
    
    # Prüfe Risk-Management
    if RISK_CONFIG["max_daily_loss"] > 0.05:
        warnings.append("⚠️ WARNUNG: max_daily_loss > 5% ist sehr hoch!")
    
    if not RISK_CONFIG["use_stop_loss"]:
        warnings.append("⚠️ WARNUNG: Stop-Loss ist deaktiviert! Sehr riskant!")
    
    # Prüfe Watchlist
    if len(WATCHLIST) < 3:
        warnings.append("⚠️ WARNUNG: Weniger als 3 Symbole in Watchlist - geringe Diversifikation!")
    
    if len(WATCHLIST) > 20:
        warnings.append("⚠️ WARNUNG: Mehr als 20 Symbole - könnte Performance beeinträchtigen!")
    
    # Zeige Warnungen
    if warnings:
        print("\n" + "="*70)
        print("KONFIGURATIONS-WARNUNGEN:")
        print("="*70)
        for warning in warnings:
            print(warning)
        print("="*70 + "\n")
    
    return len(warnings) == 0

# ============================================================================
# ENVIRONMENT VARIABLES (Optional)
# ============================================================================

# Überschreibe mit Umgebungsvariablen falls vorhanden
if os.getenv("INITIAL_CAPITAL"):
    PORTFOLIO_CONFIG["initial_capital"] = float(os.getenv("INITIAL_CAPITAL"))

if os.getenv("TRADING_MODE"):
    TRADING_MODE = os.getenv("TRADING_MODE")

if os.getenv("WATCHLIST"):
    WATCHLIST = os.getenv("WATCHLIST").split(",")

# Validiere Konfiguration beim Import
if __name__ != "__main__":
    validate_config()

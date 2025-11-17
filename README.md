# Trading Bot - Yahoo Finance

Ein vollautomatischer Trading Bot für algorithmisches Trading mit Yahoo Finance Daten.

## ✨ Features

- 📊 **5 Trading-Strategien**: SMA Crossover, RSI, MACD, Combined, Mean Reversion
- 💰 **Portfolio-Management**: Automatische Position Sizing und Diversifikation
- 🛡️ **Risk-Management**: Stop-Loss, Take-Profit, Trailing-Stop
- 📈 **Technische Indikatoren**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV
- 💾 **Daten-Caching**: Schnellere Performance durch intelligentes Caching
- 📁 **CSV-Export**: Kompatibel mit Yahoo Finance Portfolio
- 🧪 **Test Suite**: Umfassende Tests für alle Komponenten

## 🚀 Installation

```bash
# Repository klonen
git clone <repository-url>
cd algorithm-

# Abhängigkeiten installieren
pip install -r requirements.txt
```

**Hinweis**: Falls `yfinance` Installationsprobleme verursacht:
```bash
pip install yfinance --no-deps
pip install requests lxml beautifulsoup4 html5lib frozendict platformdirs
```

## 📖 Verwendung

### Interaktiver Modus
```bash
python main.py
```

Das Programm führt Sie durch folgende Schritte:
1. Strategie auswählen (SMA, RSI, MACD, Combined, Mean Reversion)
2. Modus wählen (Backtest oder Live Trading)
3. Watchlist konfigurieren
4. Trading starten

### Backtest-Modus
```python
from main import TradingBot

bot = TradingBot(
    strategy_name="combined",
    mode="backtest",
    watchlist=["AAPL", "MSFT", "GOOGL"]
)
bot.run()
```

### Live Trading (Simulation)
```python
bot = TradingBot(
    strategy_name="sma",
    mode="live",
    watchlist=["TSLA", "AMZN"]
)
bot.run()
```

## ⚙️ Konfiguration

Alle Einstellungen befinden sich in `config.py`:

### Portfolio-Einstellungen
```python
PORTFOLIO_CONFIG = {
    "initial_capital": 100000.0,    # Startkapital
    "max_position_size": 0.20,       # Max 20% pro Position
    "min_cash_reserve": 0.10,        # Min 10% Cash-Reserve
    "max_positions": 8,              # Max Anzahl Positionen
}
```

### Risk-Management
```python
RISK_CONFIG = {
    "stop_loss_percent": 0.05,       # 5% Stop-Loss
    "take_profit_percent": 0.15,     # 15% Take-Profit
    "trailing_stop": True,           # Trailing Stop aktiv
    "trailing_stop_percent": 0.03,   # 3% Trailing Stop
}
```

### Strategie-Parameter
```python
STRATEGY_CONFIG = {
    "short_window": 50,              # SMA 50
    "long_window": 200,              # SMA 200
    "rsi_period": 14,                # RSI Periode
    "rsi_oversold": 30,              # RSI Überverkauft
    "rsi_overbought": 70,            # RSI Überkauft
}
```

## 🧪 Tests

```bash
# Alle Tests ausführen
python test_trading_bot.py
```

Die Test-Suite prüft:
- ✅ Module-Imports
- ✅ Konfiguration
- ✅ Portfolio-Management
- ✅ Data Handler
- ✅ Trading-Strategien
- ✅ CSV Manager
- ✅ Risk-Management

## 📊 Trading-Strategien

### 1. SMA Crossover
Golden Cross / Death Cross mit Konfidenz-Scoring
- **Signal**: SMA 50 kreuzt SMA 200
- **Konfidenz**: Trendstärke, Volume, RSI, MACD

### 2. RSI Strategy
Überkauft/Überverkauft mit Divergenz-Erkennung
- **Buy**: RSI < 30 (überverkauft)
- **Sell**: RSI > 70 (überkauft)

### 3. MACD Strategy
Momentum-Trading mit Crossover
- **Signal**: MACD kreuzt Signal-Linie
- **Konfidenz**: Histogramm-Stärke, Volume

### 4. Combined Strategy (Empfohlen)
Gewichtete Kombination aller Indikatoren
- **Gewichtung**: SMA 35%, RSI 30%, MACD 35%
- **Mindest-Konfidenz**: 60%

### 5. Mean Reversion
Bollinger Bands + RSI für Reversion-Trading
- **Buy**: Preis unter Lower Band + RSI < 30
- **Sell**: Preis über Upper Band + RSI > 70

## 📁 Projektstruktur

```
algorithm-/
├── main.py              # Hauptprogramm
├── config.py            # Konfiguration
├── data_handler.py      # Datenverarbeitung & Indikatoren
├── strategy.py          # Trading-Strategien
├── portfolio.py         # Portfolio-Management
├── csv_manager.py       # CSV Export/Import
├── test_trading_bot.py  # Test Suite
├── requirements.txt     # Python-Abhängigkeiten
└── .gitignore          # Git-Ignore
```

## 🔧 Behobene Bugs & Verbesserungen

### Version 2.0 (Neueste)
- ✅ **Kritischer Fix**: Fehlende `csv_manager.py` erstellt
- ✅ **Performance**: Vektorisierte OBV-Berechnung (100x schneller)
- ✅ **Bugfix**: Division-durch-Null in Volume Ratio verhindert
- ✅ **Bugfix**: Korrekte Commission-Berechnung in Position Sizing
- ✅ **Bugfix**: Verbesserte Error-Handling in Portfolio
- ✅ **Neu**: Umfassende Test-Suite
- ✅ **Neu**: .gitignore für sauberes Repository
- ✅ **Neu**: requirements.txt mit allen Abhängigkeiten
- ✅ **Verbesserung**: Erweiterte Dokumentation

## ⚠️ Wichtige Hinweise

- **Backtest ≠ Zukunft**: Historische Performance garantiert keine zukünftigen Ergebnisse
- **Keine Finanzberatung**: Dies ist ein Bildungsprojekt
- **Echtes Trading**: Für Live-Trading muss eine Broker-API integriert werden
- **Risk-Management**: Immer nur Kapital einsetzen, dessen Verlust Sie verkraften können

## 📈 Performance-Tracking

Der Bot exportiert automatisch:
- `trades.csv` - Alle ausgeführten Trades
- `portfolio.csv` - Portfolio-Status
- `performance.csv` - Performance-Metriken
- `yahoo_finance_portfolio.csv` - Importierbar in Yahoo Finance

## 🤝 Contributing

Verbesserungsvorschläge sind willkommen! Bitte erstellen Sie einen Pull Request oder Issue.

## 📝 Lizenz

Dieses Projekt dient ausschließlich Bildungszwecken.

## 🔗 Weitere Informationen

- [Yahoo Finance API](https://github.com/ranaroussi/yfinance)
- [pandas Documentation](https://pandas.pydata.org/)
- [Technical Indicators](https://www.investopedia.com/terms/t/technicalindicator.asp)

---

**Erstellt mit ❤️ für algorithmisches Trading**
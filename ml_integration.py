"""
Machine Learning Integration Module
Feature Engineering und ML-Modelle für Trading Predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature Engineering für ML-Modelle
    Erstellt Trading-Features aus OHLCV Daten
    """

    @staticmethod
    def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt preis-basierte Features

        Args:
            df: DataFrame mit OHLCV Daten

        Returns:
            DataFrame mit neuen Features
        """
        df = df.copy()

        # Returns
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        df['return_20d'] = df['Close'].pct_change(20)

        # Moving Averages
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()

        # Price vs. SMA
        df['price_vs_sma5'] = df['Close'] / df['sma_5'] - 1
        df['price_vs_sma20'] = df['Close'] / df['sma_20'] - 1
        df['price_vs_sma50'] = df['Close'] / df['sma_50'] - 1

        # SMA Crossovers
        df['sma5_vs_sma20'] = df['sma_5'] / df['sma_20'] - 1
        df['sma20_vs_sma50'] = df['sma_20'] / df['sma_50'] - 1

        return df

    @staticmethod
    def create_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt Momentum-Features"""
        df = df.copy()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Stochastic
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['stochastic'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)

        # ROC (Rate of Change)
        df['roc_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        df['roc_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100

        return df

    @staticmethod
    def create_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt Volatilitäts-Features"""
        df = df.copy()

        # Historical Volatility
        df['volatility_10'] = df['Close'].pct_change().rolling(10).std() * np.sqrt(252)
        df['volatility_20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)

        # ATR
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        df['atr_pct'] = df['atr'] / df['Close']

        # Bollinger Bands
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    @staticmethod
    def create_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt Volumen-Features"""
        df = df.copy()

        # Volume Moving Averages
        df['volume_sma_5'] = df['Volume'].rolling(5).mean()
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()

        # Volume Ratio
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

        # On-Balance Volume
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        df['obv_sma'] = df['obv'].rolling(20).mean()

        # Volume-Price Trend
        df['vpt'] = ((df['Close'] - df['Close'].shift()) / df['Close'].shift() * df['Volume']).cumsum()

        return df

    @staticmethod
    def create_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt Pattern-basierte Features"""
        df = df.copy()

        # Candle body and shadows
        df['candle_body'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - pd.concat([df['Open'], df['Close']], axis=1).max(axis=1)
        df['lower_shadow'] = pd.concat([df['Open'], df['Close']], axis=1).min(axis=1) - df['Low']

        # Candle body ratio
        df['body_ratio'] = df['candle_body'] / (df['High'] - df['Low']).replace(0, np.nan)

        # Consecutive ups/downs
        df['consecutive_up'] = (df['Close'] > df['Close'].shift()).astype(int).rolling(5).sum()
        df['consecutive_down'] = (df['Close'] < df['Close'].shift()).astype(int).rolling(5).sum()

        # Gap
        df['gap'] = df['Open'] - df['Close'].shift()
        df['gap_pct'] = df['gap'] / df['Close'].shift()

        return df

    @classmethod
    def create_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt alle Features

        Args:
            df: DataFrame mit OHLCV Daten

        Returns:
            DataFrame mit allen Features
        """
        logger.info("Erstelle ML Features...")

        df = cls.create_price_features(df)
        df = cls.create_momentum_features(df)
        df = cls.create_volatility_features(df)
        df = cls.create_volume_features(df)
        df = cls.create_pattern_features(df)

        # Remove NaN
        df = df.fillna(method='ffill').fillna(method='bfill')

        logger.info(f"✅ {len(df.columns)} Features erstellt")

        return df


class MLTradingModel:
    """
    Machine Learning Trading Model
    """

    def __init__(self,
                 model_type: str = 'random_forest',
                 prediction_horizon: int = 5,
                 threshold: float = 0.01):
        """
        Initialisiert ML Trading Model

        Args:
            model_type: 'random_forest' oder 'gradient_boosting'
            prediction_horizon: Tage in die Zukunft für Prediction
            threshold: Mindest-Return für BUY Signal (z.B. 0.01 = 1%)
        """
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Erstellt Labels für Classification

        Label = 1 wenn Preis in N Tagen > threshold steigt
        Label = 0 sonst

        Args:
            df: DataFrame mit OHLCV

        Returns:
            Series mit Labels
        """
        future_returns = df['Close'].shift(-self.prediction_horizon) / df['Close'] - 1
        labels = (future_returns > self.threshold).astype(int)
        return labels

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Bereitet Trainingsdaten vor

        Args:
            df: DataFrame mit Features

        Returns:
            (X, y) Tuple
        """
        # Features erstellen
        df_features = FeatureEngineer.create_all_features(df)

        # Labels erstellen
        labels = self.create_labels(df)

        # Feature-Spalten (exclude OHLCV)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]

        X = df_features[feature_cols]
        y = labels

        # Remove rows with NaN in labels
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        self.feature_names = feature_cols

        return X, y

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Trainiert ML Model

        Args:
            df: DataFrame mit OHLCV Daten
            test_size: Anteil Test-Set

        Returns:
            Dict mit Training-Ergebnissen
        """
        logger.info(f"🤖 Trainiere {self.model_type} Model...")

        # Prepare data
        X, y = self.prepare_training_data(df)

        # Train/Test Split (chronological)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Train
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)

        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        self.is_trained = True

        logger.info(f"✅ Training abgeschlossen:")
        logger.info(f"   Train Accuracy: {train_score*100:.2f}%")
        logger.info(f"   Test Accuracy: {test_score*100:.2f}%")
        logger.info(f"   CV Mean Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
        logger.info(f"   Top 3 Features: {[f[0] for f in top_features[:3]]}")

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'top_features': top_features,
            'num_samples': len(X_train),
            'num_features': len(self.feature_names)
        }

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Macht Predictions

        Args:
            df: DataFrame mit OHLCV Daten

        Returns:
            Series mit Predictions (0 oder 1)
        """
        if not self.is_trained:
            raise ValueError("Model muss erst trainiert werden")

        # Features erstellen
        df_features = FeatureEngineer.create_all_features(df)

        # Extract features
        X = df_features[self.feature_names]

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)

        return pd.Series(predictions, index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gibt Prediction Probabilities zurück

        Args:
            df: DataFrame mit OHLCV Daten

        Returns:
            DataFrame mit Probabilities für beide Klassen
        """
        if not self.is_trained:
            raise ValueError("Model muss erst trainiert werden")

        # Features erstellen
        df_features = FeatureEngineer.create_all_features(df)

        # Extract features
        X = df_features[self.feature_names]

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        proba = self.model.predict_proba(X_scaled)

        return pd.DataFrame(
            proba,
            columns=['prob_down', 'prob_up'],
            index=df.index
        )


class EnsembleMLModel:
    """
    Ensemble von mehreren ML Models
    """

    def __init__(self,
                 models: Optional[List[MLTradingModel]] = None,
                 voting: str = 'soft'):
        """
        Initialisiert Ensemble Model

        Args:
            models: Liste von ML Models
            voting: 'hard' (majority vote) oder 'soft' (average probabilities)
        """
        if models is None:
            # Default: Random Forest + Gradient Boosting
            models = [
                MLTradingModel('random_forest', prediction_horizon=5),
                MLTradingModel('gradient_boosting', prediction_horizon=5)
            ]

        self.models = models
        self.voting = voting

    def train_all(self, df: pd.DataFrame) -> List[Dict]:
        """
        Trainiert alle Models im Ensemble

        Args:
            df: DataFrame mit OHLCV Daten

        Returns:
            Liste mit Training-Ergebnissen
        """
        logger.info(f"🎯 Trainiere Ensemble mit {len(self.models)} Models...")

        results = []
        for i, model in enumerate(self.models):
            logger.info(f"\n  Model {i+1}/{len(self.models)}: {model.model_type}")
            result = model.train(df)
            results.append(result)

        avg_test_acc = np.mean([r['test_accuracy'] for r in results])
        logger.info(f"\n✅ Ensemble Training abgeschlossen")
        logger.info(f"   Durchschnittliche Test Accuracy: {avg_test_acc*100:.2f}%")

        return results

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Ensemble Prediction

        Args:
            df: DataFrame mit OHLCV Daten

        Returns:
            Series mit Ensemble Predictions
        """
        if self.voting == 'hard':
            # Majority vote
            predictions = []
            for model in self.models:
                pred = model.predict(df)
                predictions.append(pred)

            predictions_df = pd.DataFrame(predictions).T
            ensemble_pred = predictions_df.mode(axis=1)[0]

            return ensemble_pred

        else:  # soft
            # Average probabilities
            proba_list = []
            for model in self.models:
                proba = model.predict_proba(df)
                proba_list.append(proba['prob_up'])

            avg_proba = pd.DataFrame(proba_list).T.mean(axis=1)
            ensemble_pred = (avg_proba > 0.5).astype(int)

            return ensemble_pred

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """
        Gibt durchschnittliche Probabilities zurück

        Args:
            df: DataFrame mit OHLCV Daten

        Returns:
            Series mit durchschnittlichen UP-Probabilities
        """
        proba_list = []
        for model in self.models:
            proba = model.predict_proba(df)
            proba_list.append(proba['prob_up'])

        avg_proba = pd.DataFrame(proba_list).T.mean(axis=1)

        return avg_proba

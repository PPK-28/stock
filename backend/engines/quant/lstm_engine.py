"""
lstm_engine.py — Deep Learning Sequential Forecaster
======================================================
Uses TensorFlow/Keras to build an LSTM neural network that learns
sequential time-series patterns for price direction prediction.

Approach:
  1. Use last 60 days of OHLCV + technical features to predict next 5 days.
  2. Walk-forward training/validation split.
  3. Predict probability of upward price movement.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class LSTMEngine:
    """
    LSTM deep learning engine for directional movement.
    Falls back to a momentum heuristic if tf is not installed.
    """
    def __init__(self):
        self._has_tf = self._check_tf()

    def _check_tf(self) -> bool:
        try:
            import tensorflow  # noqa
            from tensorflow.keras.models import Sequential  # noqa
            return True
        except ImportError:
            return False

    def predict(self, df: pd.DataFrame, target_col: str = "target_5d_direction") -> Dict:
        if not self._has_tf or len(df) < 150:
            return self._heuristic_prediction(df)
            
        try:
            return self._run_lstm(df, target_col)
        except Exception as e:
            print(f"[LSTM Engine] TensorFlow run failed: {e}")
            return self._heuristic_prediction(df)

    def _run_lstm(self, df: pd.DataFrame, target_col: str) -> Dict:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.preprocessing import RobustScaler

        # Ensure reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        # 1. Feature Prep
        exclude = {"Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "target_5d_direction", "target_5d_return"}
        feature_cols = [c for c in df.columns if c not in exclude and not pd.isna(df[c]).all()]

        if target_col not in df.columns or len(feature_cols) < 5:
            return self._heuristic_prediction(df)

        data = df[feature_cols + [target_col]].dropna()
        if len(data) < 100:
            return self._heuristic_prediction(df)

        X_raw = data[feature_cols].values
        y_raw = data[target_col].values

        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # 2. Sequence Creation (Lookback = 30 days)
        lookback = 30
        X_seq, y_seq = [], []
        for i in range(lookback, len(X_scaled)):
            X_seq.append(X_scaled[i - lookback:i])
            y_seq.append(y_raw[i])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        if len(X_seq) < 50:
            return self._heuristic_prediction(df)

        # Split Train/Test (Walk forward)
        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        # 3. Model Building
        model = Sequential()
        model.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=16))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train (Silent)
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)

        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        # 4. Predict Current State
        # The current state is the LAST lookback window in the entire dataset
        # We need the last `lookback` rows of X_scaled
        X_current = X_scaled[-lookback:].reshape(1, lookback, X_train.shape[2])
        prob = float(model.predict(X_current, verbose=0)[0][0])

        direction = "BULLISH" if prob > 0.5 else "BEARISH"
        confidence = self._calculate_confidence(prob, accuracy)

        return {
            "method": "LSTM Neural Net",
            "direction": direction,
            "probability": round(prob, 3),
            "test_accuracy": round(accuracy * 100, 1),
            "confidence": round(confidence, 1),
            "score": int(prob * 100)
        }

    def _calculate_confidence(self, prob: float, accuracy: float) -> float:
        decisiveness = abs(prob - 0.5) * 2
        base = 50 + (decisiveness * 30)
        acc_penalty = max(0, (0.6 - accuracy) * 100) if accuracy < 0.6 else min(10, (accuracy - 0.6) * 50)
        return min(95, max(10, base + acc_penalty))

    def _heuristic_prediction(self, df: pd.DataFrame) -> Dict:
        """Fallback momentum heuristic if TF fails or data is insufficient."""
        if df.empty or "Close" not in df.columns:
            return self._neutral_response()

        close = df["Close"]
        ret_10 = close.pct_change(10).iloc[-1] if len(close) > 10 else 0
        score = 0.5 + ret_10 * 2
        score = max(0.1, min(0.9, score))
        direction = "BULLISH" if score > 0.5 else "BEARISH"

        return {
            "method": "LSTM Fallback (Mom)",
            "direction": direction,
            "probability": round(score, 3),
            "test_accuracy": 50.0,
            "confidence": 50.0,
            "score": int(score * 100)
        }

    def _neutral_response(self) -> Dict:
        return {
            "method": "N/A", "direction": "NEUTRAL",
            "probability": 0.5, "test_accuracy": 50.0,
            "confidence": 50.0, "score": 50
        }

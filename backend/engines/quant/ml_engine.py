"""
ml_engine.py — XGBoost/Random Forest Price Direction Classifier
===============================================================
Trains and runs an ensemble of ML classifiers to predict whether
a stock will be HIGHER in 5 days.

Strategy Used:
  1. Use FeatureEngine to create 50+ features
  2. Train Random Forest + XGBoost on historical data (walk-forward)
  3. Generate probability-calibrated direction predictions
  4. Use ensemble agreement as a confidence signal

Why XGBoost + Random Forest:
  RF: High variance reducer, robust to noisy features, less prone to overfit
  XGBoost: Gradient boosting = sequential error correction, handles non-linearity

Walk-Forward Evaluation:
  Train on first 80% of data, test on last 20%.
  This mimics real-world deployment (no look-ahead bias).
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings


class MLDirectionEngine:
    """
    Ensemble ML classifier for 5-day direction prediction.
    Falls back gracefully if sklearn/xgboost not available.
    """

    def __init__(self):
        self._has_sklearn = self._check_sklearn()
        self._has_xgb = self._check_xgb()

    def _check_sklearn(self) -> bool:
        try:
            import sklearn  # noqa
            return True
        except ImportError:
            return False

    def _check_xgb(self) -> bool:
        try:
            import xgboost  # noqa
            return True
        except ImportError:
            return False

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Main prediction entry point.

        Args:
            df: Feature DataFrame from FeatureEngine.build_features()

        Returns:
            Prediction dict with direction, probability, and confidence score.
        """
        if not self._has_sklearn or len(df) < 100:
            return self._heuristic_prediction(df)

        return self._ml_predict(df)

    def _ml_predict(self, df: pd.DataFrame) -> Dict:
        """
        Full ML pipeline: feature selection → train/test split → ensemble prediction.
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics import accuracy_score
        import warnings
        warnings.filterwarnings("ignore")

        # Feature columns (exclude OHLCV, target, metadata)
        exclude = {"Open", "High", "Low", "Close", "Volume", "Dividends",
                   "Stock Splits", "target_5d_direction", "target_5d_return"}
        feature_cols = [c for c in df.columns if c not in exclude
                        and not pd.isna(df[c]).all()]

        target_col = "target_5d_direction"
        if target_col not in df.columns:
            return self._heuristic_prediction(df)

        data = df[feature_cols + [target_col]].dropna()
        if len(data) < 80:
            return self._heuristic_prediction(df)

        X = data[feature_cols].values
        y = data[target_col].values

        # Walk-forward split: 80% train, 20% test (no shuffle — time series!)
        split = int(len(data) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Robust scaling (handles outliers better than StandardScaler)
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {}
        probas = {}

        # ── Random Forest ──
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        rf.fit(X_train_s, y_train)
        rf_acc = accuracy_score(y_test, rf.predict(X_test_s))
        models["rf"] = (rf, rf_acc)

        # Predict on last row (current state — what will happen in 5d)
        X_current = scaler.transform(X[[-1]])
        probas["rf"] = rf.predict_proba(X_current)[0][1]  # P(direction=UP)

        # ── Gradient Boosting ──
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train_s, y_train)
        gb_acc = accuracy_score(y_test, gb.predict(X_test_s))
        models["gb"] = (gb, gb_acc)
        probas["gb"] = gb.predict_proba(X_current)[0][1]

        # ── XGBoost (if available) ──
        xgb_acc = None
        if self._has_xgb:
            try:
                from xgboost import XGBClassifier
                xgb = XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=42,
                )
                xgb.fit(X_train_s, y_train)
                xgb_acc = accuracy_score(y_test, xgb.predict(X_test_s))
                models["xgb"] = (xgb, xgb_acc)
                probas["xgb"] = xgb.predict_proba(X_current)[0][1]
            except Exception:
                pass

        # ── Accuracy-Weighted Ensemble ──
        total_acc = sum(acc for _, acc in models.values())
        weighted_prob = sum(
            probas[name] * acc / total_acc
            for name, (_, acc) in models.items()
        )

        # Ensemble agreement = how much do models agree?
        proba_values = list(probas.values())
        agreement = 1 - np.std(proba_values)  # 0-1; higher = more agreement

        direction = "BULLISH" if weighted_prob > 0.5 else "BEARISH"
        confidence = self._prob_to_confidence(weighted_prob, agreement)

        # Feature importance from RF (most influential predictors)
        importances = rf.feature_importances_
        top_features = [
            feature_cols[i]
            for i in np.argsort(importances)[::-1][:5]
        ]

        return {
            "method": "RF+GB" + ("+XGB" if "xgb" in probas else ""),
            "direction": direction,
            "probability": round(float(weighted_prob), 3),
            "ensemble_agreement": round(float(agreement), 3),
            "confidence": round(confidence, 1),
            "rf_accuracy": round(rf_acc * 100, 1),
            "gb_accuracy": round(gb_acc * 100, 1),
            "xgb_accuracy": round(xgb_acc * 100, 1) if xgb_acc else None,
            "top_features": top_features,
            "score": self._to_score(weighted_prob),
        }

    def _heuristic_prediction(self, df: pd.DataFrame) -> Dict:
        """
        Fallback: momentum-based heuristic when sklearn is not available
        or insufficient training data.
        """
        if df.empty or "Close" not in df.columns:
            return self._neutral_response()

        close = df["Close"]

        # 3-factor momentum heuristic (academically validated factors)
        ret_5 = close.pct_change(5).iloc[-1] if len(close) > 5 else 0
        ret_21 = close.pct_change(21).iloc[-1] if len(close) > 21 else 0
        ret_63 = close.pct_change(63).iloc[-1] if len(close) > 63 else 0

        # Momentum consensus (Jegadeesh-Titman factor)
        score = 0.5 + (ret_5 * 2 + ret_21 * 1 + ret_63 * 0.5) / 3
        score = max(0.1, min(0.9, score))

        direction = "BULLISH" if score > 0.5 else "BEARISH"
        confidence = 50 + abs(score - 0.5) * 60

        return {
            "method": "Momentum Heuristic",
            "direction": direction,
            "probability": round(score, 3),
            "ensemble_agreement": 0.6,
            "confidence": round(min(confidence, 75), 1),
            "rf_accuracy": None,
            "gb_accuracy": None,
            "xgb_accuracy": None,
            "top_features": ["ret_5d", "ret_21d", "ret_63d"],
            "score": self._to_score(score),
        }

    def _prob_to_confidence(self, prob: float, agreement: float) -> float:
        """Map probability + agreement to a human-readable confidence %."""
        decisiveness = abs(prob - 0.5) * 2   # 0-1; how decisive is the signal
        base_confidence = 50 + decisiveness * 35
        agreement_boost = agreement * 10      # More agreement = more confident
        return min(92, base_confidence + agreement_boost)

    def _to_score(self, prob: float) -> int:
        """Convert 0-1 probability to 0-100 ensemble score."""
        return int(prob * 100)

    def _neutral_response(self) -> Dict:
        return {
            "method": "N/A",
            "direction": "NEUTRAL",
            "probability": 0.5,
            "ensemble_agreement": 0.5,
            "confidence": 50.0,
            "rf_accuracy": None,
            "gb_accuracy": None,
            "xgb_accuracy": None,
            "top_features": [],
            "score": 50,
        }

"""
arima_engine.py — ARIMA/SARIMA Time-Series Forecasting
=======================================================
Implements ARIMA for short-term price direction forecasting.
Used by quant funds for event-driven and mean-reversion strategies.

Approach:
  1. Work on log-returns (guarantees stationarity — ARIMA requirement)
  2. Auto-select p,d,q parameters using AIC grid search
  3. Forecast 5-day directional bias
  4. Report confidence intervals
  5. Combine with the ML ensemble for final verdict

Key Insight:
  ARIMA forecasts LOG RETURNS, not prices. This is academically correct.
  The model captures serial autocorrelation (momentum/mean-reversion) in returns.
"""
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional, Tuple


class ARIMAEngine:
    """
    ARIMA-based short-term return forecaster.
    Uses manual ARIMA implementation to avoid dependency on statsmodels
    if not available, with graceful fallback to exponential smoothing.
    """

    def __init__(self):
        self._has_statsmodels = self._check_statsmodels()

    def _check_statsmodels(self) -> bool:
        try:
            import statsmodels.api  # noqa
            return True
        except ImportError:
            return False

    def forecast(self, prices: pd.Series, horizon: int = 5) -> Dict:
        """
        Generate ARIMA forecast for the given price series.

        Args:
            prices: Daily close price series (at least 60 observations)
            horizon: Forecast horizon in days (default: 5 trading days)

        Returns:
            Dict with forecast direction, confidence, and score.
        """
        if len(prices) < 60:
            return self._insufficient_data_response()

        # ARIMA operates on log returns (stationary series)
        log_returns = np.log(prices / prices.shift(1)).dropna()

        if self._has_statsmodels:
            return self._arima_forecast(log_returns, prices, horizon)
        else:
            return self._exponential_smoothing_forecast(log_returns, prices, horizon)

    def _arima_forecast(self, log_returns: pd.Series, prices: pd.Series, horizon: int) -> Dict:
        """Full ARIMA with AIC-based parameter selection."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            import warnings
            warnings.filterwarnings("ignore")

            best_aic = np.inf
            best_order = (1, 0, 1)
            best_model = None

            # Grid search over (p=0..2, d=0..1, q=0..2)
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(log_returns, order=(p, d, q))
                            result = model.fit()
                            if result.aic < best_aic:
                                best_aic = result.aic
                                best_order = (p, d, q)
                                best_model = result
                        except Exception:
                            continue

            if best_model is None:
                return self._exponential_smoothing_forecast(log_returns, prices, horizon)

            # Forecast log returns for `horizon` days
            forecast_result = best_model.get_forecast(steps=horizon)
            fc_mean = forecast_result.predicted_mean
            fc_ci = forecast_result.conf_int(alpha=0.05)  # 95% CI

            cumulative_return = fc_mean.sum()
            ci_lower_total = fc_ci.iloc[:, 0].sum()
            ci_upper_total = fc_ci.iloc[:, 1].sum()

            current_price = float(prices.iloc[-1])
            forecast_price = current_price * np.exp(cumulative_return)
            forecast_lower = current_price * np.exp(ci_lower_total)
            forecast_upper = current_price * np.exp(ci_upper_total)

            direction = "BULLISH" if cumulative_return > 0 else "BEARISH"
            confidence = self._calc_confidence(cumulative_return, fc_ci)

            return {
                "method": f"ARIMA{best_order}",
                "direction": direction,
                "forecast_return_pct": round(cumulative_return * 100, 2),
                "forecast_price": round(forecast_price, 2),
                "forecast_lower": round(forecast_lower, 2),
                "forecast_upper": round(forecast_upper, 2),
                "confidence": round(confidence, 1),
                "horizon_days": horizon,
                "aic": round(best_aic, 2),
                "score": self._to_score(direction, confidence),
            }

        except Exception as e:
            return self._exponential_smoothing_forecast(log_returns, prices, horizon)

    def _exponential_smoothing_forecast(
        self, log_returns: pd.Series, prices: pd.Series, horizon: int
    ) -> Dict:
        """
        Holt's Double Exponential Smoothing — level + trend.
        Suitable fallback when statsmodels is unavailable.
        """
        alpha = 0.3   # Level smoothing
        beta = 0.1    # Trend smoothing

        data = log_returns.values
        level = data[0]
        trend = data[1] - data[0] if len(data) > 1 else 0.0

        for val in data[1:]:
            prev_level = level
            level = alpha * val + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        # Forecast: level + h*trend
        forecasts = [level + h * trend for h in range(1, horizon + 1)]
        cumulative_return = sum(forecasts)

        current_price = float(prices.iloc[-1])
        forecast_price = current_price * np.exp(cumulative_return)

        # Simple CI based on historical return std
        std_daily = log_returns.tail(63).std()
        std_n = std_daily * np.sqrt(horizon)
        forecast_lower = current_price * np.exp(cumulative_return - 1.96 * std_n)
        forecast_upper = current_price * np.exp(cumulative_return + 1.96 * std_n)

        direction = "BULLISH" if cumulative_return > 0 else "BEARISH"
        # Confidence: how decisive is the signal vs. noise?
        confidence = min(90, 50 + abs(cumulative_return) / (std_n + 1e-6) * 20)

        return {
            "method": "Holt-EWS",
            "direction": direction,
            "forecast_return_pct": round(cumulative_return * 100, 2),
            "forecast_price": round(forecast_price, 2),
            "forecast_lower": round(forecast_lower, 2),
            "forecast_upper": round(forecast_upper, 2),
            "confidence": round(confidence, 1),
            "horizon_days": horizon,
            "aic": None,
            "score": self._to_score(direction, confidence),
        }

    def _calc_confidence(self, point_estimate: float, conf_int: pd.DataFrame) -> float:
        """
        Confidence = how tightly the CI bounds the direction.
        If lower bound and upper bound are both positive → high confidence bullish.
        """
        lower_sum = conf_int.iloc[:, 0].sum()
        upper_sum = conf_int.iloc[:, 1].sum()

        # Both agree on direction
        if point_estimate > 0 and lower_sum > 0:
            return 85.0
        if point_estimate < 0 and upper_sum < 0:
            return 85.0

        # CI crosses zero — uncertain
        ci_width = upper_sum - lower_sum
        abs_signal = abs(point_estimate)
        confidence = 50 + (abs_signal / (ci_width / 2 + 1e-6)) * 20
        return min(80, confidence)

    def _to_score(self, direction: str, confidence: float) -> int:
        """Convert ARIMA signal to 0-100 score for ensemble use."""
        if direction == "BULLISH":
            return int(50 + (confidence - 50) * 0.8)
        else:
            return int(50 - (confidence - 50) * 0.8)

    def _insufficient_data_response(self) -> Dict:
        return {
            "method": "N/A",
            "direction": "NEUTRAL",
            "forecast_return_pct": 0.0,
            "forecast_price": 0.0,
            "forecast_lower": 0.0,
            "forecast_upper": 0.0,
            "confidence": 50.0,
            "horizon_days": 5,
            "aic": None,
            "score": 50,
        }

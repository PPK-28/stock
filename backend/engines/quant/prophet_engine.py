"""
prophet_engine.py — Facebook Prophet Time-Series Forecaster
===========================================================
Uses Prophet for additive regression time-series forecasting.
Excellent for capturing weekly/yearly seasonality and trend shifts.
"""
import pandas as pd
import numpy as np
from typing import Dict

class ProphetForecaster:
    """Prophet engine for 5-day price forecasting."""
    def __init__(self):
        self._has_prophet = self._check_prophet()

    def _check_prophet(self) -> bool:
        try:
            from prophet import Prophet  # noqa
            return True
        except ImportError:
            return False

    def forecast(self, prices: pd.Series, horizon: int = 5) -> Dict:
        if not self._has_prophet or len(prices) < 60:
            return self._heuristic(prices, horizon)

        try:
            return self._run_prophet(prices, horizon)
        except Exception as e:
            print(f"[Prophet Engine] Prophet run failed: {e}")
            return self._heuristic(prices, horizon)

    def _run_prophet(self, prices: pd.Series, horizon: int) -> Dict:
        from prophet import Prophet
        import logging
        
        # Suppress Prophet logging
        logger = logging.getLogger('cmdstanpy')
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)

        # Prepare data (Prophet requires 'ds' and 'y')
        df = pd.DataFrame({'ds': prices.index.tz_localize(None), 'y': prices.values})

        # Fit Model
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False, changepoint_prior_scale=0.05)
        m.fit(df)

        # Predict future
        future = m.make_future_dataframe(periods=horizon, freq='B') # Business days
        forecast = m.predict(future)

        # Extract results
        last_actual = prices.iloc[-1]
        future_prices = forecast['yhat'].tail(horizon).values
        final_forecast = future_prices[-1]
        
        forecast_lower = forecast['yhat_lower'].iloc[-1]
        forecast_upper = forecast['yhat_upper'].iloc[-1]

        return_pct = ((final_forecast - last_actual) / last_actual) * 100
        direction = "BULLISH" if return_pct > 0 else "BEARISH"
        
        # Confidence based on bound spread and return size
        spread_pct = (forecast_upper - forecast_lower) / last_actual
        confidence = 50 + abs(return_pct) * 5 - spread_pct * 100
        confidence = max(10, min(90, confidence))

        return {
            "method": "Prophet",
            "direction": direction,
            "forecast_return_pct": round(return_pct, 2),
            "forecast_price": round(final_forecast, 2),
            "forecast_lower": round(forecast_lower, 2),
            "forecast_upper": round(forecast_upper, 2),
            "confidence": round(confidence, 1),
            "score": int(50 + (confidence - 50) if direction == "BULLISH" else 50 - (confidence - 50))
        }

    def _heuristic(self, prices: pd.Series, horizon: int) -> Dict:
        """Linear drift fallback."""
        if len(prices) < 20:
             return self._neutral()
             
        drift = (prices.iloc[-1] - prices.iloc[-20]) / 20
        forecast_price = prices.iloc[-1] + drift * horizon
        rtn = ((forecast_price - prices.iloc[-1]) / prices.iloc[-1]) * 100
        
        return {
            "method": "Linear Fallback",
            "direction": "BULLISH" if rtn > 0 else "BEARISH",
            "forecast_return_pct": round(rtn, 2),
            "forecast_price": round(forecast_price, 2),
            "forecast_lower": round(forecast_price * 0.95, 2),
            "forecast_upper": round(forecast_price * 1.05, 2),
            "confidence": 50.0,
            "score": int(50 + rtn * 5)
        }

    def _neutral(self) -> Dict:
        return {
            "method": "N/A", "direction": "NEUTRAL",
            "forecast_return_pct": 0.0, "forecast_price": 0.0,
            "forecast_lower": 0.0, "forecast_upper": 0.0,
            "confidence": 50.0, "score": 50
        }

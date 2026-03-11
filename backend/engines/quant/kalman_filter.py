"""
kalman_filter.py — Kalman Filter for Price Smoothing
=====================================================
Implements a Linear Kalman Filter to extract the true "latent price"
from noisy market observations. Used by professional quant desks to:
  - Reduce false signals from noisy raw prices
  - Provide smoother input to MACD and momentum calculations
  - Detect regime changes more accurately

Mathematical Model:
  State: [price, velocity(drift)]
  Observation: [Close price]

Hyperparameters:
  R — observation noise (market noise); higher = smoother
  Q — process noise (how fast the true price can change)
"""
import numpy as np
import pandas as pd
from typing import Tuple


class KalmanPriceFilter:
    """
    Kalman Filter for adaptive price estimation.

    The filter outputs a "smoothed price" with a confidence band,
    which is used as a cleaner signal for MACD and trend engines.
    """

    def __init__(self, R: float = 0.05, Q: float = 1e-5):
        """
        Args:
            R: Observation noise covariance (measurement uncertainty).
               Higher R  ➜  more smoothing, slower response.
            Q: Process noise covariance (state uncertainty).
               Higher Q  ➜  follows price more aggressively.
        """
        self.R = R               # Observation noise
        self.Q = Q               # Process noise
        self.state_dim = 2       # [price, drift]
        self.obs_dim = 1

    def filter(self, prices: pd.Series) -> pd.DataFrame:
        """
        Apply Kalman filter to a price series.

        Returns:
            DataFrame with columns:
              - kalman_price   : filtered/smoothed price estimate
              - kalman_upper   : +1σ uncertainty band
              - kalman_lower   : -1σ uncertainty band
              - kalman_gain    : adaptive gain (how much filter trusts new observation)
        """
        n = len(prices)
        obs = prices.values.astype(float)

        # State transition matrix (constant velocity model)
        F = np.array([[1, 1],
                      [0, 1]])

        # Observation matrix: we observe only price, not drift
        H = np.array([[1, 0]])

        # Process noise covariance
        Q = np.array([[self.Q, 0],
                      [0, self.Q]])

        # Observation noise covariance
        R = np.array([[self.R]])

        # State estimate and covariance (initialized from first observation)
        x = np.array([[obs[0]], [0.0]])  # [price, drift=0]
        P = np.eye(self.state_dim) * 1.0

        kalman_prices = np.zeros(n)
        kalman_vars = np.zeros(n)
        kalman_gains = np.zeros(n)

        for i, z in enumerate(obs):
            # ── PREDICT ──
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # ── UPDATE (Measurement) ──
            z_vec = np.array([[z]])
            innovation = z_vec - H @ x_pred
            S = H @ P_pred @ H.T + R           # Innovation covariance
            K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

            x = x_pred + K @ innovation
            P = (np.eye(self.state_dim) - K @ H) @ P_pred

            kalman_prices[i] = x[0, 0]
            kalman_vars[i] = P[0, 0]
            kalman_gains[i] = K[0, 0]

        std_dev = np.sqrt(np.abs(kalman_vars))
        result = pd.DataFrame(index=prices.index, data={
            "kalman_price": kalman_prices,
            "kalman_upper": kalman_prices + std_dev,
            "kalman_lower": kalman_prices - std_dev,
            "kalman_gain": kalman_gains,
            "kalman_std": std_dev,
        })
        return result

    def get_signal(self, prices: pd.Series) -> dict:
        """
        High-level signal extraction from Kalman filter output.

        Returns:
            Dictionary with:
              - smoothed_price   : Latest Kalman estimated price
              - trend_direction  : 'UP', 'DOWN', or 'FLAT'
              - trend_acceleration: Positive = accelerating, negative = decelerating
              - noise_level      : Ratio of Kalman uncertainty to price (0-1)
        """
        kf_df = self.filter(prices)
        latest = kf_df.iloc[-1]
        prev5 = kf_df.iloc[-6] if len(kf_df) > 6 else kf_df.iloc[0]

        # Drift (velocity): is the smoothed price trending up or down?
        drift = latest["kalman_price"] - prev5["kalman_price"]
        drift_pct = drift / (prev5["kalman_price"] + 1e-9) * 100

        if drift_pct > 0.5:
            direction = "UP"
        elif drift_pct < -0.5:
            direction = "DOWN"
        else:
            direction = "FLAT"

        # Acceleration (2nd derivative): trend gaining or losing momentum?
        mid = kf_df.iloc[-3] if len(kf_df) > 3 else kf_df.iloc[0]
        accel = (latest["kalman_price"] - mid["kalman_price"]) - \
                (mid["kalman_price"] - prev5["kalman_price"])

        # Noise level: how much does the raw price deviate from Kalman estimate?
        noise_level = latest["kalman_std"] / (latest["kalman_price"] + 1e-9)

        return {
            "smoothed_price": round(latest["kalman_price"], 2),
            "kalman_upper": round(latest["kalman_upper"], 2),
            "kalman_lower": round(latest["kalman_lower"], 2),
            "trend_direction": direction,
            "trend_5d_pct": round(drift_pct, 2),
            "trend_acceleration": "ACCELERATING" if accel > 0 else "DECELERATING",
            "noise_level": round(noise_level * 100, 2),
            "score_contribution": self._score(direction, drift_pct, noise_level),
        }

    def _score(self, direction: str, drift_pct: float, noise_level: float) -> int:
        """Convert Kalman signal to 0-100 score for ensemble integration."""
        base = 50
        if direction == "UP":
            base += min(30, abs(drift_pct) * 3)
        elif direction == "DOWN":
            base -= min(30, abs(drift_pct) * 3)

        # Penalize high noise (noisy signals are less reliable)
        noise_penalty = min(15, noise_level * 100)
        return int(max(0, min(100, base - noise_penalty)))

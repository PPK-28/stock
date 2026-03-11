"""
feature_engine.py — Advanced Feature Engineering for ML Models
==============================================================
Creates academically validated predictive features from raw OHLCV data.

Features implemented:
  - Lagged returns (1, 2, 3, 5, 10, 21 days)
  - Log returns (better statistical properties)
  - Volatility clusters (realized vol in different windows)
  - Volume anomalies (Z-score, relative volume)
  - Momentum features (ROC, RSI-based)
  - Price distance from moving averages (mean-reversion signals)
  - Market microstructure (High-Low range, Close position in range)
  - Trend features (slope of EMA200, ADX surrogate)
"""
import numpy as np
import pandas as pd
from typing import Optional


class FeatureEngine:
    """
    Transforms raw OHLCV DataFrames into rich ML feature matrices.
    All features are normalized to be stationary and scale-invariant.
    """

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Master feature builder. Accepts a yfinance OHLCV df.
        Returns a DataFrame with all engineered features, dropping NaN rows.
        """
        feat = df.copy()
        feat = self._add_return_features(feat)
        feat = self._add_volatility_features(feat)
        feat = self._add_volume_features(feat)
        feat = self._add_momentum_features(feat)
        feat = self._add_price_structure_features(feat)
        feat = self._add_trend_features(feat)
        feat = self._add_target(feat)
        feat.dropna(inplace=True)
        return feat

    # ── 1. Return Features ──────────────────────────────────────────────────
    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        log_r = np.log(close / close.shift(1))

        for lag in [1, 2, 3, 5, 10, 21]:
            df[f"ret_{lag}d"] = close.pct_change(lag)
            df[f"log_ret_{lag}d"] = close.apply(np.log).diff(lag)

        # Cumulative excess return vs simple drift
        df["ret_cum_5d"] = close.pct_change(5)
        df["ret_cum_21d"] = close.pct_change(21)

        # Overnight gap (Open vs prior Close)
        if "Open" in df.columns:
            df["overnight_gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        return df

    # ── 2. Volatility Features ───────────────────────────────────────────────
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        log_r = np.log(df["Close"] / df["Close"].shift(1))

        # Realized volatility at multiple windows (annualized %)
        for w in [5, 10, 21, 63]:
            df[f"rvol_{w}d"] = log_r.rolling(w).std() * np.sqrt(252) * 100

        # Volatility of volatility (2nd order — GARCH proxy)
        df["vol_of_vol"] = df["rvol_21d"].rolling(10).std()

        # Parkinson volatility estimator — uses High-Low range (more efficient)
        if "High" in df.columns and "Low" in df.columns:
            log_hl = np.log(df["High"] / df["Low"])
            df["parkinson_vol"] = np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2)).rolling(21).mean() * np.sqrt(252) * 100

        # Volatility regime: current vol vs 63-day average vol
        df["vol_regime"] = df["rvol_21d"] / (df["rvol_63d"] + 1e-9)  # >1 = expanding vol
        return df

    # ── 3. Volume Anomaly Features ────────────────────────────────────────────
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Volume" not in df.columns:
            return df
        vol = df["Volume"].replace(0, np.nan).ffill()
        vol_20_mean = vol.rolling(20).mean()
        vol_20_std = vol.rolling(20).std()

        df["vol_zscore"] = (vol - vol_20_mean) / (vol_20_std + 1e-9)
        df["vol_ratio_5d"] = vol / vol.rolling(5).mean()  # Short-term relative volume
        df["vol_ratio_20d"] = vol / vol_20_mean            # Medium-term relative volume

        # Volume-price relationship
        close_ret = df["Close"].pct_change()
        df["vol_price_corr"] = close_ret.rolling(10).corr(vol.pct_change())

        # OBV momentum (rate of change of OBV)
        obv = (np.sign(close_ret) * vol).cumsum()
        df["obv_roc_5d"] = obv.pct_change(5)
        return df

    # ── 4. Momentum Features ─────────────────────────────────────────────────
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]

        # Rate of Change (ROC) — standard momentum factor
        for period in [5, 10, 21, 63, 126]:
            df[f"roc_{period}d"] = close.pct_change(period) * 100

        # RSI (14) — normalized to 0-1 for ML
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["rsi_norm"] = (100 - (100 / (1 + rs))) / 100  # Normalized 0-1

        # RSI divergence: price made new high but RSI didn't (bearish divergence)
        price_high_20 = close.rolling(20).max()
        rsi_high_20 = (100 - 100 / (1 + rs)).rolling(20).max()
        df["rsi_divergence"] = ((close / price_high_20) - ((100 - 100 / (1 + rs)) / (rsi_high_20 + 1e-9)))

        # Williams %R (overbought/oversold in trend context)
        if "High" in df.columns and "Low" in df.columns:
            highest_high = df["High"].rolling(14).max()
            lowest_low = df["Low"].rolling(14).min()
            df["williams_r"] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)
        return df

    # ── 5. Price Structure Features ────────────────────────────────────────────
    def _add_price_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]

        # Distance from EMAs (mean-reversion / trend-following)
        for span in [9, 21, 50, 200]:
            ema = close.ewm(span=span, adjust=False).mean()
            df[f"dist_ema{span}"] = (close - ema) / (ema + 1e-9)  # % deviation

        # Bollinger Band Position (0=lower, 1=upper, 0.5=middle)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
        df["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-9)  # Volatility proxy

        # Price position within day's range (0=low, 1=high)
        if "High" in df.columns and "Low" in df.columns:
            df["intraday_position"] = (close - df["Low"]) / (df["High"] - df["Low"] + 1e-9)

        # Fibonacci retracement levels (recent 52-week high-low)
        high_52w = close.rolling(252).max()
        low_52w = close.rolling(252).min()
        fib_range = high_52w - low_52w
        df["fib_0236"] = (close - (high_52w - 0.236 * fib_range)) / (fib_range + 1e-9)
        df["fib_0382"] = (close - (high_52w - 0.382 * fib_range)) / (fib_range + 1e-9)
        df["fib_0618"] = (close - (high_52w - 0.618 * fib_range)) / (fib_range + 1e-9)
        df["fib_position"] = (close - low_52w) / (fib_range + 1e-9)  # 0-1 in annual range
        return df

    # ── 6. Trend Features ────────────────────────────────────────────────────
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]

        # EMA slope (proxy for trend strength and direction)
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        df["ema50_slope"] = ema50.diff(5) / (ema50.shift(5) + 1e-9)   # 5-day slope
        df["ema200_slope"] = ema200.diff(10) / (ema200.shift(10) + 1e-9)

        # Golden/Death cross signal as numeric
        df["golden_cross"] = (ema50 > ema200).astype(int)
        df["price_vs_ema200"] = (close > ema200).astype(int)

        # MACD histogram sign and magnitude
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist_norm"] = (macd - signal) / (close + 1e-9)
        df["macd_sign_change"] = np.sign(macd - signal).diff().abs() / 2  # 1 = crossover
        return df

    # ── 7. Target Variable ────────────────────────────────────────────────────
    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Binary classification target: 1 if price is higher in 5 days, else 0.
        Also add regression target: actual 5-day forward return.
        """
        close = df["Close"]
        df["target_5d_direction"] = (close.shift(-5) > close).astype(int)  # Classification
        df["target_5d_return"] = close.shift(-5).pct_change(5).shift(-5) * 100   # Regression
        return df

    def get_feature_names(self) -> list:
        """Return list of all feature column names (excluding OHLCV and targets)."""
        return [
            # Returns
            *[f"ret_{l}d" for l in [1, 2, 3, 5, 10, 21]],
            *[f"log_ret_{l}d" for l in [1, 2, 3, 5, 10, 21]],
            "ret_cum_5d", "ret_cum_21d", "overnight_gap",
            # Volatility
            *[f"rvol_{w}d" for w in [5, 10, 21, 63]],
            "vol_of_vol", "parkinson_vol", "vol_regime",
            # Volume
            "vol_zscore", "vol_ratio_5d", "vol_ratio_20d", "vol_price_corr", "obv_roc_5d",
            # Momentum
            *[f"roc_{p}d" for p in [5, 10, 21, 63, 126]],
            "rsi_norm", "rsi_divergence", "williams_r",
            # Structure
            *[f"dist_ema{s}" for s in [9, 21, 50, 200]],
            "bb_pct", "bb_width", "intraday_position",
            "fib_0236", "fib_0382", "fib_0618", "fib_position",
            # Trend
            "ema50_slope", "ema200_slope", "golden_cross", "price_vs_ema200",
            "macd_hist_norm", "macd_sign_change",
        ]

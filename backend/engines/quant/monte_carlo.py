"""
monte_carlo.py — Monte Carlo Simulation for Price Distribution
==============================================================
Simulates the distribution of possible future prices using
Geometric Brownian Motion (GBM) — the mathematical foundation
of the Black-Scholes option pricing model.

Used for:
  - Probability that price reaches target
  - Value at Risk (VaR) calculation
  - Expected return under uncertainty
  - Distribution of outcomes (not just a single price forecast)

GBM Model:
  dS = μ·S·dt + σ·S·dW

Where:
  S = stock price
  μ = expected daily log return (drift)
  σ = daily volatility (log returns)
  dW = Wiener process increment (random shock)
"""
import numpy as np
import pandas as pd
from typing import Dict


class MonteCarloEngine:
    """
    Monte Carlo simulator for price path generation.
    Runs N simulations to produce a probability distribution
    of future prices over a given horizon.
    """

    def __init__(self, n_simulations: int = 5000, horizon_days: int = 126):
        """
        Args:
            n_simulations: Number of paths to simulate (5000 is professional standard)
            horizon_days: Forecast horizon (126 = 6 months of trading days)
        """
        self.n_simulations = n_simulations
        self.horizon = horizon_days

    def simulate(self, prices: pd.Series, target_price: float = None) -> Dict:
        """
        Run Monte Carlo simulation on historical price series.

        Args:
            prices: Historical close price series
            target_price: Optional target to compute probability of reaching

        Returns:
            Simulation results with price distribution metrics.
        """
        if len(prices) < 30:
            return self._insufficient_data()

        log_returns = np.log(prices / prices.shift(1)).dropna()

        # GBM parameters (using recent 63 days for regime-aware estimation)
        lookback = min(63, len(log_returns))
        recent_returns = log_returns.tail(lookback)

        mu_daily = recent_returns.mean()         # Mean log return
        sigma_daily = recent_returns.std()       # Daily log return std

        # Drift-adjusted: GBM exact solution's drift term
        drift = mu_daily - 0.5 * sigma_daily ** 2
        current_price = float(prices.iloc[-1])

        # ══ Run Simulations ══
        # Shape: (n_simulations, horizon)
        np.random.seed(None)  # Use current time as seed (non-deterministic)
        random_shocks = np.random.normal(
            loc=drift,
            scale=sigma_daily,
            size=(self.n_simulations, self.horizon)
        )
        # Cumulative sum of log returns → multiply by initial price
        cumulative_log_returns = random_shocks.cumsum(axis=1)
        price_paths = current_price * np.exp(cumulative_log_returns)

        # Final prices at simulation end
        final_prices = price_paths[:, -1]

        # Price at 5d, 21d, 63d checkpoints (for short/medium/long term)
        horizon_5d = min(5, self.horizon - 1)
        horizon_21d = min(21, self.horizon - 1)
        horizon_63d = min(63, self.horizon - 1)

        prices_5d = price_paths[:, horizon_5d]
        prices_21d = price_paths[:, horizon_21d]
        prices_63d = price_paths[:, horizon_63d]

        # ── Distribution Statistics ──
        percentiles = np.percentile(final_prices, [5, 10, 25, 50, 75, 90, 95])

        # Expected Return
        expected_price = final_prices.mean()
        expected_return_pct = (expected_price / current_price - 1) * 100

        # Risk Metrics
        var_95 = np.percentile(final_prices, 5)   # Value at Risk (95%)
        var_99 = np.percentile(final_prices, 1)   # Value at Risk (99%)
        cvar_95 = final_prices[final_prices <= var_95].mean()  # Conditional VaR (Expected Shortfall)

        # Probability of profit
        prob_positive = (final_prices > current_price).mean() * 100
        prob_5pct_up = (final_prices > current_price * 1.05).mean() * 100
        prob_10pct_up = (final_prices > current_price * 1.10).mean() * 100
        prob_5pct_down = (final_prices < current_price * 0.95).mean() * 100

        # Sharpe-like ratio for simulated paths
        annualized_return = (expected_price / current_price) ** (252 / self.horizon) - 1
        annualized_vol = sigma_daily * np.sqrt(252)
        simulated_sharpe = (annualized_return - 0.065) / (annualized_vol + 1e-9)  # 6.5% = India Rf rate

        # Target probability
        prob_target = None
        if target_price and target_price > 0:
            prob_target = (final_prices >= target_price).mean() * 100

        # Short-term checkpoint returns
        st_5d_rets = (prices_5d / current_price - 1) * 100
        st_21d_rets = (prices_21d / current_price - 1) * 100
        st_63d_rets = (prices_63d / current_price - 1) * 100

        return {
            "current_price": round(current_price, 2),
            "simulations": self.n_simulations,
            "horizon_days": self.horizon,

            # Price scenarios at end of horizon
            "scenario_bear": round(percentiles[0], 2),         # 5th percentile
            "scenario_pessimist": round(percentiles[1], 2),    # 10th
            "scenario_base_low": round(percentiles[2], 2),     # 25th
            "scenario_median": round(percentiles[3], 2),       # 50th (median)
            "scenario_base_high": round(percentiles[4], 2),    # 75th
            "scenario_optimist": round(percentiles[5], 2),     # 90th
            "scenario_bull": round(percentiles[6], 2),         # 95th

            # Checkpoint prices (short-term outlook)
            "price_5d_median": round(np.median(prices_5d), 2),
            "price_21d_median": round(np.median(prices_21d), 2),
            "price_63d_median": round(np.median(prices_63d), 2),

            # Return expectations
            "expected_price": round(expected_price, 2),
            "expected_return_pct": round(expected_return_pct, 2),

            # Risk metrics
            "var_95_price": round(var_95, 2),
            "var_99_price": round(var_99, 2),
            "cvar_95_price": round(cvar_95, 2),
            "max_loss_95pct": round((var_95 / current_price - 1) * 100, 2),

            # Probability metrics
            "prob_profit": round(prob_positive, 1),
            "prob_5pct_up": round(prob_5pct_up, 1),
            "prob_10pct_up": round(prob_10pct_up, 1),
            "prob_5pct_down": round(prob_5pct_down, 1),
            "prob_reach_target": round(prob_target, 1) if prob_target is not None else None,

            # GBM Parameters
            "daily_drift": round(mu_daily * 100, 4),
            "daily_vol": round(sigma_daily * 100, 4),
            "annual_vol": round(annualized_vol * 100, 2),
            "simulated_sharpe": round(simulated_sharpe, 2),

            # Score for ensemble (based on probability of profit)
            "score": int(prob_positive),
        }

    def get_confidence_band(self, prices: pd.Series) -> Dict:
        """
        Quick 30-day forward confidence band for UI display.
        Returns upper/lower band for chart overlay.
        """
        if len(prices) < 20:
            return {}

        log_returns = np.log(prices / prices.shift(1)).dropna()
        mu = log_returns.tail(21).mean()
        sigma = log_returns.tail(21).std()
        current = float(prices.iloc[-1])

        days = 30
        t = np.arange(1, days + 1)
        drift = mu - 0.5 * sigma ** 2
        central = current * np.exp(drift * t)
        upper_1s = current * np.exp(drift * t + sigma * np.sqrt(t))
        lower_1s = current * np.exp(drift * t - sigma * np.sqrt(t))
        upper_2s = current * np.exp(drift * t + 2 * sigma * np.sqrt(t))
        lower_2s = current * np.exp(drift * t - 2 * sigma * np.sqrt(t))

        return {
            "days": list(range(1, days + 1)),
            "central": [round(x, 2) for x in central],
            "upper_68": [round(x, 2) for x in upper_1s],
            "lower_68": [round(x, 2) for x in lower_1s],
            "upper_95": [round(x, 2) for x in upper_2s],
            "lower_95": [round(x, 2) for x in lower_2s],
        }

    def _insufficient_data(self) -> Dict:
        return {
            "error": "Insufficient data for Monte Carlo simulation",
            "simulations": 0,
            "prob_profit": 50.0,
            "score": 50,
        }

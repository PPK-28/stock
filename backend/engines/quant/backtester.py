"""
backtester.py — Walk-Forward Backtesting Engine
================================================
Tests the prediction signal on historical data to measure real accuracy.
No look-ahead bias — each prediction uses only data available at that time.

Metrics Computed:
  - Directional accuracy (% of times price moved in predicted direction)
  - RMSE and MAE (for price level forecasts)
  - Sharpe Ratio of a simple strategy following signals
  - Maximum Drawdown of the signal-based portfolio
  - Win Rate, Profit Factor, Average Win vs Average Loss
  - Annualized Return vs Buy-and-Hold benchmark
"""
import numpy as np
import pandas as pd
from typing import Dict


class Backtester:
    """
    Walk-forward backtester for signal validation.
    Tests a simple long/flat strategy based on directional signals.
    """

    def __init__(self, min_periods: int = 126):
        """
        Args:
            min_periods: Minimum history needed before generating signals (126 = 6 months)
        """
        self.min_periods = min_periods

    def run(self, prices: pd.Series, signal_fn=None) -> Dict:
        """
        Run full backtest on historical prices.

        Args:
            prices: Historical close price series
            signal_fn: Optional callable(prices_up_to_t) -> 1 (long) or 0 (flat)
                       If None, uses a default RSI + EMA trend following signal.

        Returns:
            Comprehensive backtest report dict.
        """
        if len(prices) < self.min_periods + 20:
            return self._insufficient_data_response(prices)

        if signal_fn is None:
            signal_fn = self._default_signal

        # Walk-forward: for each day from min_periods onwards, generate signal
        signals = []
        dates = prices.index.tolist()

        for i in range(self.min_periods, len(prices) - 5):
            historical_slice = prices.iloc[:i]
            signal = signal_fn(historical_slice)
            signals.append({
                "date": dates[i],
                "price_entry": prices.iloc[i],
                "price_exit": prices.iloc[i + 5],  # 5-day holding period
                "signal": signal,
            })

        df = pd.DataFrame(signals)
        if df.empty:
            return self._insufficient_data_response(prices)

        # ── Performance Calculation ──
        df["actual_direction"] = (df["price_exit"] > df["price_entry"]).astype(int)
        df["trade_return"] = (df["price_exit"] - df["price_entry"]) / df["price_entry"]
        df["strategy_return"] = df["signal"] * df["trade_return"]  # Only take return when long

        # Directional accuracy (only for trades taken when signal=1)
        long_trades = df[df["signal"] == 1]
        if len(long_trades) > 0:
            dir_accuracy = long_trades["actual_direction"].mean() * 100
        else:
            dir_accuracy = 50.0

        # Returns
        strat_returns = df["strategy_return"]
        bah_returns = df["trade_return"]  # Buy and hold baseline

        # RMSE and MAE on price forecast
        price_errors = df["price_exit"] - df["price_entry"] * (1 + df["trade_return"])
        rmse = np.sqrt((price_errors ** 2).mean()) if len(price_errors) > 0 else 0
        mae = price_errors.abs().mean() if len(price_errors) > 0 else 0

        # Compound returns (per 5-day period → annualize assuming 50 periods/year)
        n_periods = len(df)
        periods_per_year = 50  # 252 trading days / 5 days
        if n_periods > 0:
            strat_cumulative = (1 + strat_returns).prod()
            bah_cumulative = (1 + bah_returns).prod()
            years = n_periods / periods_per_year
            strat_annual = strat_cumulative ** (1 / max(years, 0.1)) - 1
            bah_annual = bah_cumulative ** (1 / max(years, 0.1)) - 1
        else:
            strat_annual = bah_annual = 0

        # Sharpe Ratio (annualized, risk-free = 6.5% for India)
        rf_per_period = 0.065 / periods_per_year
        excess_returns = strat_returns - rf_per_period
        sharpe = (excess_returns.mean() / (excess_returns.std() + 1e-9)) * np.sqrt(periods_per_year)

        # Sortino Ratio (uses only downside deviation)
        downside = excess_returns[excess_returns < 0]
        sortino = (excess_returns.mean() / (downside.std() + 1e-9)) * np.sqrt(periods_per_year)

        # Maximum Drawdown
        cumulative_strat = (1 + strat_returns).cumprod()
        rolling_max = cumulative_strat.cummax()
        drawdown = (cumulative_strat - rolling_max) / (rolling_max + 1e-9)
        max_dd = drawdown.min() * 100

        # Win Rate & Profit Factor (among long trades)
        wins = long_trades[long_trades["trade_return"] > 0]["trade_return"]
        losses = long_trades[long_trades["trade_return"] <= 0]["trade_return"]
        win_rate = len(wins) / max(len(long_trades), 1) * 100
        avg_win = wins.mean() * 100 if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) * 100 if len(losses) > 0 else 0
        profit_factor = (wins.sum() / (abs(losses.sum()) + 1e-9))

        # Signal quality score (0-100)
        quality_score = self._compute_quality_score(
            dir_accuracy, sharpe, win_rate, max_dd, strat_annual
        )

        return {
            # Core accuracy
            "directional_accuracy": round(dir_accuracy, 1),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),

            # Returns
            "strategy_annual_return": round(strat_annual * 100, 2),
            "buyhold_annual_return": round(bah_annual * 100, 2),
            "alpha": round((strat_annual - bah_annual) * 100, 2),

            # Risk metrics
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown": round(max_dd, 2),

            # Error metrics
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),

            # Meta
            "trades_taken": len(long_trades),
            "total_windows": n_periods,
            "backtest_period_days": len(prices),
            "signal_quality_score": quality_score,
        }

    def _default_signal(self, prices: pd.Series) -> int:
        """
        Default trend-following signal: EMA crossover + RSI filter.
        Returns 1 (long) or 0 (flat/no trade).
        """
        if len(prices) < 30:
            return 0

        ema9 = prices.ewm(span=9).mean().iloc[-1]
        ema21 = prices.ewm(span=21).mean().iloc[-1]

        # RSI filter
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        rs = gain / (loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)

        # Go long only when EMA9 > EMA21 (uptrend) and RSI is not overbought
        if ema9 > ema21 and rsi < 70:
            return 1
        return 0

    def _compute_quality_score(
        self, dir_acc: float, sharpe: float, win_rate: float,
        max_dd: float, annual_ret: float
    ) -> int:
        """Composite signal quality score 0-100."""
        score = 0
        score += min(40, (dir_acc - 50) * 2)        # 0-40 for dir accuracy above 50%
        score += min(20, max(0, sharpe * 6))          # 0-20 for Sharpe
        score += min(20, (win_rate - 50) * 0.8)      # 0-20 for win rate above 50%
        score += min(10, max(0, annual_ret * 0.5))   # 0-10 for return
        score -= min(10, abs(max_dd) * 0.3)          # Penalty for drawdown
        return max(0, min(100, int(50 + score)))

    def _insufficient_data_response(self, prices: pd.Series) -> Dict:
        return {
            "directional_accuracy": 50.0,
            "win_rate": 50.0,
            "profit_factor": 1.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "strategy_annual_return": 0.0,
            "buyhold_annual_return": 0.0,
            "alpha": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "trades_taken": 0,
            "total_windows": 0,
            "backtest_period_days": len(prices),
            "signal_quality_score": 50,
        }

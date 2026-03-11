
import yfinance as yf
import numpy as np

class FundamentalEngine:
    """
    Fundamental Analysis Engine — MarketSmith CAN SLIM inspired.
    Scores on: Valuation (25%), Profitability (25%), Financial Health (20%),
    Growth & EPS (20%), Cash Flow (10%).
    """
    def analyze(self, ticker_info):
        try:
            # 1. Core Metrics
            pe = ticker_info.get('trailingPE', 0) or 0
            peg = ticker_info.get('pegRatio', 0) or 0
            roe = ticker_info.get('returnOnEquity', 0) or 0
            eps = ticker_info.get('trailingEps', 0) or 0
            forward_eps = ticker_info.get('forwardEps', 0) or 0
            pb = ticker_info.get('priceToBook', 0) or 0
            de = ticker_info.get('debtToEquity', 0) or 0
            profit_margin = ticker_info.get('profitMargins', 0) or 0
            rev_growth = ticker_info.get('revenueGrowth', 0) or 0
            earnings_growth = ticker_info.get('earningsGrowth', 0) or 0
            fcf = ticker_info.get('freeCashflow', 0) or 0
            
            score = 0
            details = []
            
            # ── 2. VALUATION (Max 25 pts) ──
            valuation_status = "FAIRLY VALUED"
            
            # P/E (0-12 pts)
            if pe <= 0:
                score += 0
                details.append(f"P/E: Negative or N/A")
            elif pe < 12:
                score += 12
                details.append(f"Deep Value P/E ({round(pe, 1)})")
                valuation_status = "UNDERVALUED"
            elif pe < 20:
                score += 8
                details.append(f"Reasonable P/E ({round(pe, 1)})")
            elif pe < 30:
                score += 4
                details.append(f"Growth P/E ({round(pe, 1)})")
            else:
                score += 0
                details.append(f"Expensive P/E ({round(pe, 1)})")
                valuation_status = "OVERVALUED"
            
            # PEG Ratio (0-8 pts) — Key CAN SLIM metric
            if 0 < peg < 0.8:
                score += 8
                details.append(f"Excellent PEG ({round(peg, 2)}) — Cheap growth")
                valuation_status = "UNDERVALUED"
            elif 0.8 <= peg <= 1.5:
                score += 5
            elif peg > 2.5:
                score -= 3
                valuation_status = "OVERVALUED"
            
            # P/B Ratio (0-5 pts)
            if 0 < pb < 2:
                score += 5
                details.append(f"Low P/B ({round(pb, 2)})")
            elif pb > 5:
                score -= 2
            
            # ── 3. PROFITABILITY & QUALITY (Max 25 pts) ──
            quality_rating = "MEDIUM"
            
            # ROE (0-12 pts) — Buffett's favorite
            if roe > 0.20:
                score += 12
                quality_rating = "HIGH"
                details.append(f"Excellent ROE ({round(roe*100, 1)}%)")
            elif roe > 0.15:
                score += 9
                quality_rating = "HIGH"
                details.append(f"Strong ROE ({round(roe*100, 1)}%)")
            elif roe > 0.10:
                score += 6
            elif roe > 0:
                score += 3
            else:
                score -= 3
                quality_rating = "LOW"
                details.append("Negative ROE ⚠️")
            
            # Net Profit Margins (0-8 pts)
            if profit_margin > 0.20:
                score += 8
                details.append(f"Superior Margins ({round(profit_margin*100, 1)}%)")
            elif profit_margin > 0.12:
                score += 5
                details.append(f"Good Margins ({round(profit_margin*100, 1)}%)")
            elif profit_margin > 0.05:
                score += 3
            else:
                score -= 2
            
            # ── 4. FINANCIAL HEALTH (Max 20 pts) ──
            # D/E Ratio (0-12 pts)
            if de < 30:
                score += 12
                details.append("Fortress Balance Sheet (Debt < 30% Equity)")
            elif de < 80:
                score += 8
                details.append("Healthy Debt Levels")
            elif de < 150:
                score += 4
            else:
                score -= 5
                details.append(f"High Leverage (D/E: {round(de, 0)})")
            
            # Free Cash Flow (0-8 pts)
            if fcf > 0:
                score += 8
                details.append("Positive Free Cash Flow ✓")
            else:
                score -= 2
                details.append("Negative Free Cash Flow ⚠️")
            
            # ── 5. GROWTH & EPS (Max 20 pts) — CAN SLIM Core ──
            # EPS Growth (forward vs trailing) — 0-12 pts
            eps_growth_pct = 0
            if eps > 0 and forward_eps > 0:
                eps_growth_pct = ((forward_eps - eps) / eps) * 100
                if eps_growth_pct > 25:
                    score += 12
                    details.append(f"Strong EPS Growth (+{round(eps_growth_pct, 1)}%)")
                elif eps_growth_pct > 10:
                    score += 8
                    details.append(f"Moderate EPS Growth (+{round(eps_growth_pct, 1)}%)")
                elif eps_growth_pct > 0:
                    score += 4
                else:
                    score -= 3
                    details.append(f"Declining EPS ({round(eps_growth_pct, 1)}%)")
            
            # Revenue Growth (0-8 pts)
            if rev_growth > 0.25:
                score += 8
                details.append(f"High Revenue Growth ({round(rev_growth*100, 1)}%)")
            elif rev_growth > 0.10:
                score += 5
            elif rev_growth > 0:
                score += 2
            else:
                score -= 2
            
            # ── 6. FREE CASH FLOW YIELD (Bonus: Max 10 pts) ──
            market_cap = ticker_info.get('marketCap', 0) or 1
            if fcf > 0 and market_cap > 0:
                fcf_yield = (fcf / market_cap) * 100
                if fcf_yield > 5:
                    score += 10
                    details.append(f"Excellent FCF Yield ({round(fcf_yield, 1)}%)")
                elif fcf_yield > 3:
                    score += 6
            
            # Score Normalization
            final_score = max(0, min(100, score))
            
            # Graham Fair Value Estimate
            growth_rate_est = max(rev_growth, 0.05) if rev_growth > 0 else 0.05
            future_eps = max(eps, forward_eps) * ((1 + growth_rate_est) ** 5)
            fair_pe = 18  # Conservative multiplier
            fair_value_5y = round(future_eps * fair_pe, 2)
            
            return {
                "score": final_score,
                "valuation": valuation_status,
                "quality": quality_rating,
                "fair_value_5y": fair_value_5y,
                "eps_growth": round(eps_growth_pct, 1),
                "details": details
            }
        except Exception as e:
            return {
                "score": 50, "valuation": "UNKNOWN", "quality": "UNKNOWN",
                "fair_value_5y": "N/A", "eps_growth": 0, "details": ["Data unavailable"]
            }

class RiskEngine:
    """
    Risk Management Engine — Quantitative Grade.
    Computes:
      - ATR-based dynamic stop loss
      - Computed Beta vs Nifty50 (not yfinance static value)
      - Sharpe Ratio (annualized, vs 6.5% India risk-free rate)
      - Sortino Ratio (downside deviation only)
      - EWMA Volatility (GARCH-approximate: gives more weight to recent volatility)
      - Value at Risk (Historical VaR 95%)
      - Calmar Ratio (return vs max drawdown)
    """
    RISK_FREE_RATE = 0.065  # 6.5% India 10-year G-Sec rate

    def analyze(self, ticker_info, hist_data, market_beta=1.0):
        try:
            import yfinance as yf
            current_price = float(hist_data['Close'].iloc[-1])

            # ── 1. LOG RETURNS (better statistical properties) ──
            log_returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1)).dropna()
            daily_rf = self.RISK_FREE_RATE / 252

            # ── 2. EWMA VOLATILITY (GARCH-approximate) ──
            # λ=0.94 is the RiskMetrics standard decay factor
            lam = 0.94
            ewma_var = log_returns.ewm(com=(1 - lam) / lam, min_periods=10).var()
            garch_vol_annualized = float(np.sqrt(ewma_var.iloc[-1]) * np.sqrt(252) * 100)

            # Also compute simple historical vol for comparison
            hist_vol_20 = float(log_returns.tail(20).std() * np.sqrt(252) * 100)
            hist_vol_63 = float(log_returns.tail(63).std() * np.sqrt(252) * 100)

            # Use the HIGHER of EWMA and 20d realized vol (conservative)
            effective_vol = max(garch_vol_annualized, hist_vol_20)

            # ── 3. COMPUTED BETA vs NIFTY50 ──
            # Download Nifty 1y simultaneously (same period)
            beta_computed = self._compute_beta(hist_data['Close'], ticker_info)

            # ── 4. SHARPE RATIO (annualized) ──
            excess_returns = log_returns - daily_rf
            sharpe = (excess_returns.mean() / (log_returns.std() + 1e-9)) * np.sqrt(252)

            # ── 5. SORTINO RATIO (only penalizes downside) ──
            downside_returns = excess_returns[excess_returns < 0]
            sortino = (excess_returns.mean() / (downside_returns.std() + 1e-9)) * np.sqrt(252)

            # ── 6. HISTORICAL VaR 95% (1-day) ──
            var_95 = float(np.percentile(log_returns, 5)) * 100  # 5th percentile = 95% VaR
            var_99 = float(np.percentile(log_returns, 1)) * 100

            # ── 7. MAX DRAWDOWN ──
            cumulative = (1 + log_returns).cumprod()
            rolling_max = cumulative.cummax()
            drawdown = (cumulative - rolling_max) / (rolling_max + 1e-9)
            max_dd_total = float(drawdown.min() * 100)
            max_dd_20d   = float(drawdown.tail(20).min() * 100)

            # ── 8. CALMAR RATIO ──
            annual_ret = float(log_returns.mean() * 252 * 100)
            calmar = annual_ret / (abs(max_dd_total) + 1e-9)

            # ── 9. ATR-BASED STOP LOSS (Industry Standard) ──
            tr1 = hist_data['High'] - hist_data['Low']
            tr2 = (hist_data['High'] - hist_data['Close'].shift()).abs()
            tr3 = (hist_data['Low'] - hist_data['Close'].shift()).abs()
            tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_14 = float(tr.rolling(14).mean().iloc[-1])

            stop_tight = current_price - (2.0 * atr_14)
            stop_wide  = current_price - (3.0 * atr_14)

            recent_lows = hist_data['Low'].tail(126)
            support_50d = float(recent_lows.tail(50).min())
            support_126d = float(recent_lows.min())

            stop_loss_rec = max(stop_tight, support_50d)

            # ── 10. COMPOSITE RISK SCORE ──
            score = 0

            # Volatility component (0-35 pts)
            if effective_vol > 50:   score += 35
            elif effective_vol > 35: score += 25
            elif effective_vol > 20: score += 15
            else:                     score += 5

            # Beta component (0-20 pts)
            beta_use = beta_computed if beta_computed and beta_computed > 0 else (ticker_info.get('beta', 1.0) or 1.0)
            if beta_use > 1.8:   score += 20
            elif beta_use > 1.3: score += 12
            elif beta_use > 1.0: score += 5

            # 52-week range position (0-20 pts)
            high_52 = ticker_info.get('fiftyTwoWeekHigh', current_price)
            low_52  = ticker_info.get('fiftyTwoWeekLow', current_price)
            range_pos = (current_price - low_52) / (high_52 - low_52 + 1e-9)
            if range_pos > 0.95:   score += 20
            elif range_pos > 0.80: score += 10
            elif range_pos < 0.10: score += 15

            # Drawdown severity (0-15 pts)
            if max_dd_20d < -15: score += 15
            elif max_dd_20d < -8: score += 8

            # Sharpe penalty: negative Sharpe = add risk
            if sharpe < 0:         score += 10
            elif sharpe < 0.5:     score += 5

            final_risk = min(100, max(0, score))

            if final_risk > 75:
                risk_label = "VERY HIGH"
                pos_size_rec = "2-3% (Speculative)"
            elif final_risk > 55:
                risk_label = "HIGH"
                pos_size_rec = "3-5% (Standard)"
            elif final_risk > 35:
                risk_label = "MEDIUM"
                pos_size_rec = "5-8% (Core)"
            else:
                risk_label = "LOW"
                pos_size_rec = "8-10% (High Conviction)"

            upside   = current_price * 0.15
            downside = max(current_price - stop_loss_rec, 1)
            rr_ratio = round(upside / downside, 1)

            return {
                "level":            risk_label,
                "score":            final_risk,
                "volatility_score": round(effective_vol, 1),
                "garch_vol":        round(garch_vol_annualized, 1),
                "hist_vol_20d":     round(hist_vol_20, 1),
                "hist_vol_63d":     round(hist_vol_63, 1),
                "beta":             round(float(beta_use), 2),
                "beta_computed":    round(float(beta_computed), 2) if beta_computed else None,
                "sharpe":           round(sharpe, 2),
                "sortino":          round(sortino, 2),
                "calmar":           round(calmar, 2),
                "var_95_1d":        round(var_95, 2),
                "var_99_1d":        round(var_99, 2),
                "atr":              round(atr_14, 2),
                "max_drawdown_20d": round(max_dd_20d, 1),
                "max_drawdown_all": round(max_dd_total, 1),
                "stop_buffer_pct":  round((current_price - stop_loss_rec) / current_price, 3),
                "stop_loss_price":  round(stop_loss_rec, 2),
                "stop_tight":       round(stop_tight, 2),
                "stop_wide":        round(stop_wide, 2),
                "pos_size_rec":     pos_size_rec,
                "risk_reward":      f"1:{rr_ratio}",
                "supports":         [round(support_50d, 2), round(support_126d, 2)],
                "risks": [
                    f"GARCH Vol: {round(garch_vol_annualized,1)}% (EWMA λ=0.94)",
                    f"Beta (computed): {round(float(beta_use),2)} vs Nifty50",
                    f"Sharpe: {round(sharpe,2)} | Sortino: {round(sortino,2)}",
                    f"VaR 95% (1d): {round(var_95,2)}% | VaR 99%: {round(var_99,2)}%",
                    f"Max DD(20d): {round(max_dd_20d,1)}% | All-time: {round(max_dd_total,1)}%",
                    f"ATR(14): ₹{round(atr_14,2)} | Stop: ₹{round(stop_loss_rec,2)}",
                ]
            }
        except Exception as e:
            import traceback; traceback.print_exc()
            return {"level": "MEDIUM", "score": 50, "volatility_score": 20, "garch_vol": 20,
                    "hist_vol_20d": 20, "hist_vol_63d": 20, "beta": 1.0, "beta_computed": None,
                    "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "var_95_1d": -2.0, "var_99_1d": -3.0,
                    "atr": 0, "max_drawdown_20d": 0, "max_drawdown_all": 0, "stop_buffer_pct": 0.05,
                    "stop_loss_price": 0, "stop_tight": 0, "stop_wide": 0,
                    "pos_size_rec": "5%", "risk_reward": "1:2", "supports": [], "risks": [str(e)]}

    def _compute_beta(self, stock_prices: pd.Series, ticker_info: dict) -> float:
        """
        Compute Beta as Cov(Stock, Market) / Var(Market) using historical returns.
        Downloads Nifty50 for the same period as the stock.
        Falls back to yfinance static beta if download fails.
        """
        try:
            import yfinance as yf
            nifty = yf.download("^NSEI", period="1y", progress=False, threads=False)
            if nifty.empty or len(nifty) < 30:
                return float(ticker_info.get('beta', 1.0) or 1.0)

            nifty_ret = np.log(nifty['Close'] / nifty['Close'].shift(1)).dropna()
            stock_ret = np.log(stock_prices / stock_prices.shift(1)).dropna()

            # Align on common dates
            common_idx = stock_ret.index.intersection(nifty_ret.index)
            if len(common_idx) < 30:
                return float(ticker_info.get('beta', 1.0) or 1.0)

            s = stock_ret.loc[common_idx]
            m = nifty_ret.loc[common_idx]
            cov = np.cov(s, m)[0][1]
            var_m = np.var(m)
            beta = cov / (var_m + 1e-9)
            return float(np.clip(beta, -3.0, 5.0))  # Clamp to reasonable range

        except Exception:
            return float(ticker_info.get('beta', 1.0) or 1.0)




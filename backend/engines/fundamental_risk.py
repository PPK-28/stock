
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
    Risk Management Engine — Uses ATR-based stop losses and max drawdown.
    """
    def analyze(self, ticker_info, hist_data, market_beta=1.0):
        try:
            current_price = hist_data['Close'].iloc[-1]
            
            # ── 1. VOLATILITY ANALYSIS ──
            daily_returns = hist_data['Close'].pct_change()
            hist_vol = daily_returns.tail(20).std() * (252 ** 0.5) * 100
            
            # Max drawdown (20-day worst case)
            rolling_max = hist_data['Close'].rolling(20).max()
            drawdown = ((hist_data['Close'] - rolling_max) / rolling_max) * 100
            max_dd_20d = drawdown.tail(60).min()
            
            # Beta
            beta = ticker_info.get('beta', 1.0)
            if beta is None: beta = 1.0
            
            # 52-Week Range Position
            high_52 = ticker_info.get('fiftyTwoWeekHigh', current_price)
            low_52 = ticker_info.get('fiftyTwoWeekLow', current_price)
            range_pos = (current_price - low_52) / (high_52 - low_52) if high_52 != low_52 else 0.5
            
            # ── 2. ATR-BASED STOP LOSS (Industry Standard) ──
            # Calculate ATR(14)
            tr1 = hist_data['High'] - hist_data['Low']
            tr2 = (hist_data['High'] - hist_data['Close'].shift()).abs()
            tr3 = (hist_data['Low'] - hist_data['Close'].shift()).abs()
            import pandas as pd
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean().iloc[-1]
            
            # Stop loss = Price - 2×ATR (tight) or 3×ATR (wide)
            stop_tight = current_price - (2 * atr_14)
            stop_wide = current_price - (3 * atr_14)
            
            # Also check swing low
            recent_lows = hist_data['Low'].tail(126)
            support_1 = recent_lows.tail(50).min()
            support_2 = recent_lows.min()
            
            # Use the higher of ATR stop or swing low (more protective)
            stop_loss_rec = max(stop_tight, support_1)
            
            # ── 3. COMPOSITE RISK SCORE ──
            score = 0
            
            # Volatility (0-35 pts)
            if hist_vol > 50: score += 35
            elif hist_vol > 35: score += 25
            elif hist_vol > 20: score += 15
            else: score += 5
            
            # Beta (0-20 pts)
            if beta > 1.8: score += 20
            elif beta > 1.3: score += 12
            elif beta > 1.0: score += 5
            
            # Range position (0-20 pts)
            if range_pos > 0.95: score += 20  # Near ATH
            elif range_pos > 0.8: score += 10
            elif range_pos < 0.1: score += 15  # Falling knife risk
            
            # Max drawdown severity (0-15 pts)
            if max_dd_20d < -15: score += 15
            elif max_dd_20d < -8: score += 8
            
            final_risk = min(100, max(0, score))
            
            risk_label = "MEDIUM"
            pos_size_rec = "5%"
            
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
            
            # Risk/Reward
            upside = current_price * 0.15  # Conservative 15% target
            downside = current_price - stop_loss_rec
            rr_ratio = round(upside / downside, 1) if downside > 0 else 1.0
            
            return {
                "level": risk_label,
                "score": final_risk,
                "volatility_score": round(hist_vol, 1),
                "beta": round(beta, 2),
                "atr": round(atr_14, 2),
                "max_drawdown_20d": round(max_dd_20d, 1),
                "stop_buffer_pct": round((current_price - stop_loss_rec)/current_price, 3),
                "stop_loss_price": round(stop_loss_rec, 2),
                "pos_size_rec": pos_size_rec,
                "risk_reward": f"1:{rr_ratio}",
                "supports": [round(support_1, 2), round(support_2, 2)],
                "risks": [
                    f"Volatility: {round(hist_vol, 1)}%" + (" (High)" if hist_vol > 30 else " (Normal)"),
                    f"Beta: {round(beta, 2)} (Market Sensitivity)",
                    f"Max Drawdown (20d): {round(max_dd_20d, 1)}%",
                    f"ATR(14): ₹{round(atr_14, 2)}"
                ]
            }
        except Exception:
            return {"level": "MEDIUM", "score": 50, "volatility_score": 20, "beta": 1.0, "atr": 0,
                    "max_drawdown_20d": 0, "stop_buffer_pct": 0.05, "stop_loss_price": 0,
                    "pos_size_rec": "5%", "risk_reward": "1:2", "supports": [], "risks": []}

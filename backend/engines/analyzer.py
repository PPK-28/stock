from backend.engines.technical import TechnicalEngine
from backend.engines.sentiment import SentimentEngine
from backend.engines.fundamental_risk import FundamentalEngine, RiskEngine
from backend.engines.futures import FuturesEngine
from backend.engines.trust_score import TrustScoreCalculator
import yfinance as yf
import pandas as pd
import numpy as np

class StockAnalyzer:
    def __init__(self):
        self.tech = TechnicalEngine()
        self.sent = SentimentEngine()
        self.fund = FundamentalEngine()
        self.risk = RiskEngine()
        self.futures = FuturesEngine()

    def analyze_stock(self, symbol):
        """
        Executes the 7-Phase Analysis Framework with 5-pillar composite scoring.
        """
        try:
            symbol = symbol.upper().strip()
            
            # Tata Motors demerger mapping
            if symbol.replace(".NS", "").replace(".BO", "") == "TATAMOTORS":
                symbol = "TMPV.NS"
            
            if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
                symbol += ".NS"

            # --- PHASE 0: DATA COLLECTION ---
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            info = ticker.info
            
            if hist.empty or len(hist) < 10:
                print(f"Insufficient data for {symbol} (Found {len(hist)} days)")
                return None

            # Calculate Technical Indicators
            tech_df = self.tech.calculate_indicators(hist)
            last_row = tech_df.iloc[-1]
            current_price = last_row['Close']
            
            # --- PHASE 1: TECHNICAL ANALYSIS (10-Indicator Consensus) ---
            tech_report = self.tech.get_technical_report(tech_df)
            tech_score = tech_report['score']
            votes = tech_report.get('votes', {})
            tech_details = [tech_report['report_html']]
            
            # Signal from consensus
            tech_signal = "NEUTRAL"
            if tech_score >= 60: tech_signal = "BULLISH"
            elif tech_score <= 40: tech_signal = "BEARISH"
            
            # --- PHASE 2: FUNDAMENTAL ANALYSIS (CAN SLIM) ---
            f_data = self.fund.analyze(info)
            fund_score = f_data['score']
            
            if fund_score >= 80: fund_signal = "EXCELLENT"
            elif fund_score >= 60: fund_signal = "GOOD"
            elif fund_score >= 45: fund_signal = "MODERATE"
            else: fund_signal = "WEAK"
            
            # --- PHASE 3: SENTIMENT & F&O ---
            vol_spike = last_row.get('Vol_Spike', 1.0)
            rsi_val = last_row.get('RSI', 50)
            
            if len(tech_df) >= 5:
                price_5d_ago = tech_df.iloc[-5]['Close']
                price_change_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100
            else:
                price_change_5d = 0
            
            avg_vol_ratio = vol_spike  # Already a ratio
            
            futures_data = self.futures.analyze_derivatives(
                symbol, current_price, vol_spike * 100,
                rsi=rsi_val, price_change_5d=price_change_5d, avg_volume_ratio=avg_vol_ratio
            )
            
            s_data = self.sent.analyze_sentiment(info, futures_data, vol_spike)
            sent_score = s_data['score']
            
            if sent_score >= 75: sent_signal = "EXTREME BULLISH"
            elif sent_score >= 55: sent_signal = "BULLISH"
            elif sent_score >= 45: sent_signal = "NEUTRAL"
            elif sent_score >= 30: sent_signal = "BEARISH"
            else: sent_signal = "EXTREME BEARISH"

            # --- PHASE 4: RISK ASSESSMENT ---
            r_data = self.risk.analyze(info, hist)
            risk_lvl = r_data['level']
            risk_score = r_data['score']
            
            # --- PHASE 5: MOMENTUM SCORE ---
            # Relative Strength: How does this stock perform vs Nifty?
            mom_6m = last_row.get('Momentum_6M', 0)
            mom_3m = last_row.get('Momentum_3M', 0)
            mom_1m = last_row.get('Momentum_1M', 0)
            
            # Convert momentum to a 0-100 score
            # +30% in 6m = 100, 0% = 50, -30% = 0
            momentum_score = max(0, min(100, 50 + (mom_6m * 1.5)))
            
            # --- PHASE 6: COMPOSITE TRUST SCORE (5-Pillar) ---
            trust_data = TrustScoreCalculator.calculate_final_score(
                tech_score, fund_score, sent_score, risk_score, momentum_score
            )
            
            confidence = trust_data['trust_score']
            verdict = trust_data['verdict']
            
            # --- PHASE 7: PREDICTIVE FORECASTING ---
            volatility_factor = r_data['volatility_score'] / 100
            target_mean = info.get('targetMeanPrice', 0) or 0
            
            # Smart target: use analyst target when available, else calculate
            if target_mean > 0 and target_mean > current_price:
                # Analyst target is above current price — use it
                base_target = target_mean
            elif verdict in ('BUY', 'STRONG BUY', 'BUY (Contrarian)'):
                # No analyst target but bullish — use 10-15% upside
                uplift = 0.15 if tech_signal == 'BULLISH' else 0.10
                base_target = current_price * (1 + uplift)
            elif verdict in ('SELL', 'STRONG SELL', 'AVOID'):
                # Bearish — target is support level (downside)
                base_target = max(r_data['stop_loss_price'], current_price * 0.92)
            else:
                # HOLD — modest 5% target
                base_target = current_price * 1.05
            
            bull_target = current_price * (1 + max(volatility_factor * 0.8, 0.10))
            if target_mean > bull_target: bull_target = target_mean
            
            bear_target = current_price * (1 - max(volatility_factor * 0.8, 0.05))
            if r_data['stop_loss_price'] > 0 and bear_target > r_data['stop_loss_price']:
                bear_target = r_data['stop_loss_price'] * 0.98
            
            ev_target = (bull_target * 0.2) + (base_target * 0.6) + (bear_target * 0.2)
            
            # Ensure target is always >= entry for BUY verdicts
            display_target = base_target
            if verdict in ('BUY', 'STRONG BUY', 'BUY (Contrarian)'):
                display_target = max(base_target, current_price * 1.05)
            
            # --- PILLAR BREAKDOWN for UI ---
            pillars = trust_data.get('pillar_breakdown', {})
            
            # --- REPORT GENERATION ---
            html_report = f"""
            <div class='analysis-section'>
                <b>📊 PHASE 1: TECHNICALS ({tech_signal})</b><br>
                Score: {tech_score}/100 | 
                Buy: {votes.get('BUY',0)} · Sell: {votes.get('SELL',0)} · Neutral: {votes.get('NEUTRAL',0)}<br>
                {tech_details[0]}
            </div>
            <br>
            <div class='analysis-section'>
                <b>🏢 PHASE 2: FUNDAMENTALS ({fund_signal})</b><br>
                Score: {fund_score}/100 | Valuation: {f_data['valuation']}<br>
                EPS Growth: {f_data.get('eps_growth', 'N/A')}%<br>
                Fair Value (5Y): ₹{f_data['fair_value_5y']}<br>
                {'<br>'.join(f'• {d}' for d in f_data['details'])}
            </div>
            <br>
            <div class='analysis-section'>
                <b>🧠 PHASE 3: SENTIMENT ({sent_signal})</b><br>
                Score: {sent_score}/100<br>
                • Analyst: {s_data['analyst_rating']}<br>
                • Inst. Ownership: {s_data['inst_hold']}%<br>
                • Retail Interest: {s_data['retail_interest']}
            </div>
            <br>
            <div class='analysis-section'>
                <b>📈 PHASE 4: MOMENTUM</b><br>
                Score: {round(momentum_score)}/100<br>
                • 1-Month: {round(mom_1m, 1)}%<br>
                • 3-Month: {round(mom_3m, 1)}%<br>
                • 6-Month: {round(mom_6m, 1)}%
            </div>
            <br>
            <div class='analysis-section'>
                <b>⚠️ PHASE 5: RISK ({risk_lvl})</b><br>
                Risk Score: {r_data['score']}/100<br>
                • Stop Loss: ₹{r_data['stop_loss_price']} 
                  (-{round(r_data['stop_buffer_pct']*100,1)}%)<br>
                • ATR(14): ₹{r_data.get('atr', 'N/A')}<br>
                • Max Drawdown (20d): {r_data.get('max_drawdown_20d', 'N/A')}%<br>
                • Position Size: {r_data['pos_size_rec']}<br>
                • Risk/Reward: {r_data['risk_reward']}
            </div>
            <br>
            <div class='analysis-section' style='background:rgba(0,212,170,0.05); border:1px solid rgba(0,212,170,0.15);'>
                <b>🎯 COMPOSITE VERDICT: {verdict}</b><br>
                <div style='font-size:11px; margin-top:5px;'>
                    {trust_data['reasoning']}
                </div>
                <div style='margin-top:8px; display:grid; grid-template-columns:1fr 1fr 1fr 1fr 1fr; gap:4px; text-align:center; font-size:10px;'>
                    <div>
                        <div style='color:var(--text-muted);'>Tech</div>
                        <div style='font-weight:700;'>{round(tech_score)}</div>
                    </div>
                    <div>
                        <div style='color:var(--text-muted);'>Fund</div>
                        <div style='font-weight:700;'>{round(fund_score)}</div>
                    </div>
                    <div>
                        <div style='color:var(--text-muted);'>Sent</div>
                        <div style='font-weight:700;'>{round(sent_score)}</div>
                    </div>
                    <div>
                        <div style='color:var(--text-muted);'>Mom</div>
                        <div style='font-weight:700;'>{round(momentum_score)}</div>
                    </div>
                    <div>
                        <div style='color:var(--text-muted);'>Risk</div>
                        <div style='font-weight:700;'>{round(100-risk_score)}</div>
                    </div>
                </div>
            </div>
            <br>
            <div class='analysis-section' style='background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.15);'>
                <b>🔮 PHASE 7: FORECASTS (6-Month Outlook)</b><br>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:5px;">
                    <div>
                        <b style="color:#10b981;">🐂 BULL (20%)</b><br>
                        ₹{round(bull_target, 1)}
                    </div>
                    <div>
                        <b style="color:#f59e0b;">🏠 BASE (60%)</b><br>
                        ₹{round(base_target, 1)}
                    </div>
                    <div>
                        <b style="color:#ef4444;">🐻 BEAR (20%)</b><br>
                        ₹{round(bear_target, 1)}
                    </div>
                    <div style="border-left:2px solid var(--accent-primary); padding-left:8px;">
                        <b>EV Target</b><br>
                        ₹{round(ev_target, 1)}
                    </div>
                </div>
            </div>
            """
            
            return {
                "symbol": symbol,
                "price": round(current_price, 2),
                "trust_score": round(confidence, 1),
                "verdict": verdict,
                "reasoning": html_report,
                "risk": risk_lvl,
                "category": "Blue Chip" if info.get('marketCap', 0) > 200000000000 else "Small Cap",
                "advisory": {
                    "entry": f"₹{round(current_price, 2)}",
                    "target": f"{round(display_target, 1)}",
                    "stop_loss": f"{round(r_data['stop_loss_price'], 2)}",
                    "analyst_rating": f"{verdict} (Conf: {round(confidence)}%)"
                },
                "futures_data": futures_data
            }

        except Exception as e:
            print(f"Analysis Error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "symbol": symbol,
                "price": 0,
                "trust_score": 0,
                "verdict": "ERROR", 
                "reasoning": f"Could not analyze: {str(e)}",
                "risk": "UNKNOWN",
                "category": "Unknown",
                "advisory": {"entry":"-","target":"-","stop_loss":"-","analyst_rating":"Error"},
                "futures_data": {}
            }

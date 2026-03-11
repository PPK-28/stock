"""
analyzer.py — Upgraded 9-Phase Hybrid Prediction Engine
=========================================================
Integrates:
  Phase 0:  Data collection (yfinance)
  Phase 1:  Technical Analysis — 13-indicator consensus (upgraded)
  Phase 2:  Fundamental Analysis — CAN SLIM inspired
  Phase 3:  Sentiment & F&O
  Phase 4:  Risk Assessment — Sharpe, Sortino, GARCH vol, computed Beta
  Phase 5:  Momentum Score
  Phase 6:  Composite Trust Score (5-pillar weighted ensemble)

  ★ NEW ★
  Phase 7:  Kalman Filter — noise-smoothed price + trend direction
  Phase 8:  ARIMA Forecast — time-series log-return prediction
  Phase 9:  ML Ensemble — RF + GBM + XGBoost direction classifier
  Phase 10: Monte Carlo — price distribution & probability of profit
  Phase 11: Backtesting — historical signal quality validation
  Phase 12: Hybrid Score — weighted ensemble of all models
"""

from backend.engines.technical import TechnicalEngine
from backend.engines.sentiment import SentimentEngine
from backend.engines.fundamental_risk import FundamentalEngine, RiskEngine
from backend.engines.futures import FuturesEngine
from backend.engines.trust_score import TrustScoreCalculator
from backend.engines.quant.kalman_filter import KalmanPriceFilter
from backend.engines.quant.arima_engine import ARIMAEngine
from backend.engines.quant.ml_engine import MLDirectionEngine
from backend.engines.quant.lstm_engine import LSTMEngine
from backend.engines.quant.prophet_engine import ProphetForecaster
from backend.engines.quant.monte_carlo import MonteCarloEngine
from backend.engines.quant.backtester import Backtester
from backend.engines.quant.feature_engine import FeatureEngine
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class StockAnalyzer:
    """
    9-Phase Quantitative Stock Analysis Engine.
    Combines technical, fundamental, sentiment, quant models,
    and ML into a single confidence-calibrated trading signal.
    """

    def __init__(self):
        self.tech      = TechnicalEngine()
        self.sent      = SentimentEngine()
        self.fund      = FundamentalEngine()
        self.risk      = RiskEngine()
        self.futures   = FuturesEngine()
        self.kalman    = KalmanPriceFilter(R=0.05, Q=1e-5)
        self.arima     = ARIMAEngine()
        self.prophet   = ProphetForecaster()
        self.ml        = MLDirectionEngine()
        self.lstm      = LSTMEngine()
        self.mc        = MonteCarloEngine(n_simulations=3000, horizon_days=63)
        self.backtester = Backtester(min_periods=126)
        self.features  = FeatureEngine()

    def analyze_stock(self, symbol: str) -> dict:
        """
        Full 9-phase analysis pipeline. Returns a rich dict consumed by the API.
        """
        try:
            symbol = symbol.upper().strip()

            # Tata Motors demerger mapping
            if symbol.replace(".NS", "").replace(".BO", "") == "TATAMOTORS":
                symbol = "TMPV.NS"

            if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
                symbol += ".NS"

            # ── PHASE 0: DATA COLLECTION ──
            ticker = yf.Ticker(symbol)
            hist   = ticker.history(period="2y")  # Extended to 2y for ML training
            info   = ticker.info

            if hist.empty or len(hist) < 10:
                print(f"[Analyzer] Insufficient data for {symbol} ({len(hist)} days)")
                return None

            # ── PHASE 1: TECHNICAL ANALYSIS (13-Indicator Consensus) ──
            tech_df   = self.tech.calculate_indicators(hist)
            last_row  = tech_df.iloc[-1]
            current_price = float(last_row['Close'])

            tech_report = self.tech.get_technical_report(tech_df)
            tech_score  = tech_report['score']
            votes       = tech_report.get('votes', {})
            fib_levels  = tech_report.get('fib_levels', {})

            tech_signal = "NEUTRAL"
            if tech_score >= 60: tech_signal = "BULLISH"
            elif tech_score <= 40: tech_signal = "BEARISH"

            # ── PHASE 2: FUNDAMENTAL ANALYSIS (CAN SLIM) ──
            f_data     = self.fund.analyze(info)
            fund_score = f_data['score']

            if fund_score >= 80:   fund_signal = "EXCELLENT"
            elif fund_score >= 60: fund_signal = "GOOD"
            elif fund_score >= 45: fund_signal = "MODERATE"
            else:                   fund_signal = "WEAK"

            # ── PHASE 3: SENTIMENT & F&O ──
            vol_spike        = float(last_row.get('Vol_Spike', 1.0))
            rsi_val          = float(last_row.get('RSI', 50))
            price_5d_ago     = float(tech_df.iloc[-5]['Close']) if len(tech_df) >= 5 else current_price
            price_change_5d  = ((current_price - price_5d_ago) / price_5d_ago) * 100

            futures_data = self.futures.analyze_derivatives(
                symbol, current_price, vol_spike * 100,
                rsi=rsi_val, price_change_5d=price_change_5d, avg_volume_ratio=vol_spike
            )
            s_data     = self.sent.analyze_sentiment(info, futures_data, vol_spike)
            sent_score = s_data['score']

            if sent_score >= 75:   sent_signal = "EXTREME BULLISH"
            elif sent_score >= 55: sent_signal = "BULLISH"
            elif sent_score >= 45: sent_signal = "NEUTRAL"
            elif sent_score >= 30: sent_signal = "BEARISH"
            else:                   sent_signal = "EXTREME BEARISH"

            # ── PHASE 4: RISK ASSESSMENT (Upgraded) ──
            r_data   = self.risk.analyze(info, hist)
            risk_lvl = r_data['level']
            risk_score = r_data['score']

            # ── PHASE 5: MOMENTUM SCORE ──
            mom_6m = float(last_row.get('Momentum_6M', 0) or 0)
            mom_3m = float(last_row.get('Momentum_3M', 0) or 0)
            mom_1m = float(last_row.get('Momentum_1M', 0) or 0)
            momentum_score = max(0, min(100, 50 + (mom_6m * 1.5)))

            # ── PHASE 6: COMPOSITE TRUST SCORE ──
            trust_data = TrustScoreCalculator.calculate_final_score(
                tech_score, fund_score, sent_score, risk_score, momentum_score
            )
            traditional_trust = trust_data['trust_score']
            verdict           = trust_data['verdict']

            # ══════════════════════════════════════════════
            # PHASES 7-10: QUANTITATIVE MODEL LAYER (NEW)
            # ══════════════════════════════════════════════

            # ── PHASE 7: KALMAN FILTER ──
            kalman_signal = self._safe_run(
                lambda: self.kalman.get_signal(hist['Close']),
                default={"smoothed_price": current_price, "trend_direction": "FLAT",
                         "trend_5d_pct": 0, "trend_acceleration": "FLAT",
                         "noise_level": 10, "score_contribution": 50}
            )
            kalman_score = kalman_signal.get('score_contribution', 50)

            # ── PHASE 8: ARIMA FORECAST ──
            arima_result = self._safe_run(
                lambda: self.arima.forecast(hist['Close'], horizon=5),
                default={"method": "N/A", "direction": "NEUTRAL", "forecast_return_pct": 0,
                         "forecast_price": current_price, "confidence": 50, "score": 50}
            )
            arima_score = arima_result.get('score', 50)

            # ── PHASE 8.5: PROPHET FORECAST ──
            prophet_result = self._safe_run(
                lambda: self.prophet.forecast(hist['Close'], horizon=5),
                default={"method": "N/A", "direction": "NEUTRAL", "forecast_return_pct": 0,
                         "forecast_price": current_price, "confidence": 50, "score": 50}
            )
            prophet_score = prophet_result.get('score', 50)

            # ── PHASE 9: ML ENSEMBLE ──
            ml_result = self._safe_run(
                lambda: self._run_ml(hist),
                default={"method": "N/A", "direction": "NEUTRAL", "probability": 0.5,
                         "confidence": 50, "score": 50}
            )
            ml_score = ml_result.get('score', 50)

            # ── PHASE 9.5: LSTM FORECAST ──
            lstm_result = self._safe_run(
                lambda: self.lstm.predict(self.features.build_features(hist.copy())),
                default={"method": "N/A", "direction": "NEUTRAL", "probability": 0.5,
                         "confidence": 50, "score": 50}
            )
            lstm_score = lstm_result.get('score', 50)

            # ── PHASE 10: MONTE CARLO ──
            target_price = float(current_price * 1.15)
            mc_result = self._safe_run(
                lambda: self.mc.simulate(hist['Close'], target_price=target_price),
                default={"prob_profit": 50, "expected_price": current_price,
                         "scenario_median": current_price, "var_95_price": current_price * 0.95,
                         "simulated_sharpe": 0, "score": 50}
            )
            mc_score = mc_result.get('score', 50)

            # ── PHASE 11: BACKTESTING (CACHED — only run for fresh analysis) ──
            bt_result = self._safe_run(
                lambda: self.backtester.run(hist['Close']),
                default={"directional_accuracy": 50, "sharpe_ratio": 0,
                         "win_rate": 50, "signal_quality_score": 50}
            )
            bt_quality_score = bt_result.get('signal_quality_score', 50)

            # ── PHASE 12: HYBRID ENSEMBLE SCORE ──
            # Weight breakdown (professional quant desk approach):
            # Traditional (technicals + fundamentals): 45%
            # Time-series (Kalman + ARIMA + Prophet): 25%
            # ML Ensemble (RF/XGB + LSTM): 20%
            # Monte Carlo (probabilistic): 10%
            hybrid_score = (
                traditional_trust  * 0.45 +
                ((kalman_score + arima_score + prophet_score) / 3) * 0.25 +
                ((ml_score + lstm_score) / 2)           * 0.20 +
                mc_score           * 0.10
            )
            hybrid_score = round(min(100, max(0, hybrid_score)), 1)

            # Ensemble agreement: std of all model scores (lower = more agreement)
            all_scores = [traditional_trust, kalman_score, arima_score, prophet_score, ml_score, lstm_score, mc_score]
            ensemble_std = float(np.std(all_scores))
            ensemble_agreement = max(0, 100 - ensemble_std * 1.5)

            # Confidence interval for final score
            margin = ensemble_std * 0.5
            ci_low  = round(max(0, hybrid_score - margin), 1)
            ci_high = round(min(100, hybrid_score + margin), 1)

            # Override verdict if hybrid disagrees strongly with traditional
            final_verdict = self._reconcile_verdict(
                verdict, hybrid_score, arima_result.get('direction', 'NEUTRAL'),
                ml_result.get('direction', 'NEUTRAL'), ensemble_agreement
            )

            # ── PREDICTIVE FORECASTING (DISPLAY TARGETS) ──
            target_mean       = info.get('targetMeanPrice', 0) or 0
            volatility_factor = r_data['volatility_score'] / 100

            display_target = self._compute_display_target(
                final_verdict, current_price, target_mean, tech_signal,
                arima_result, prophet_result, mc_result
            )

            bull_target = max(
                current_price * (1 + max(volatility_factor * 0.8, 0.10)),
                target_mean if target_mean > current_price else 0,
                mc_result.get('scenario_optimist', current_price * 1.20)
            )
            bear_target = min(
                current_price * (1 - max(volatility_factor * 0.8, 0.05)),
                mc_result.get('scenario_pessimist', current_price * 0.85),
                r_data['stop_loss_price'] * 0.98 if r_data['stop_loss_price'] > 0 else current_price * 0.90
            )
            ev_target = (bull_target * 0.20) + (display_target * 0.60) + (bear_target * 0.20)

            # ── REPORT GENERATION ──
            html_report = self._build_report(
                tech_signal, tech_score, votes, tech_report['report_html'],
                fund_signal, fund_score, f_data,
                sent_signal, sent_score, s_data,
                mom_1m, mom_3m, mom_6m, momentum_score,
                risk_lvl, r_data,
                final_verdict, trust_data,
                kalman_signal, arima_result, prophet_result, ml_result, lstm_result, mc_result, bt_result,
                hybrid_score, ensemble_agreement, ci_low, ci_high,
                bull_target, display_target, bear_target, ev_target,
                fib_levels
            )

            return {
                "symbol":      symbol,
                "price":       round(current_price, 2),
                "trust_score": hybrid_score,
                "verdict":     final_verdict,
                "reasoning":   html_report,
                "risk":        risk_lvl,
                "category":    "Blue Chip" if info.get('marketCap', 0) > 200_000_000_000 else "Small Cap",
                "advisory": {
                    "entry":          f"₹{round(current_price, 2)}",
                    "target":         f"{round(display_target, 1)}",
                    "stop_loss":      f"{round(r_data['stop_loss_price'], 2)}",
                    "analyst_rating": f"{final_verdict} (Conf: {round(hybrid_score)}%)",
                },
                "futures_data":  futures_data,
                "quant_data": {
                    "kalman":       kalman_signal,
                    "arima":        arima_result,
                    "prophet":      prophet_result,
                    "ml":           ml_result,
                    "lstm":         lstm_result,
                    "monte_carlo":  mc_result,
                    "backtesting":  bt_result,
                    "fib_levels":   fib_levels,
                    "sharpe":       r_data.get('sharpe', 0),
                    "sortino":      r_data.get('sortino', 0),
                    "garch_vol":    r_data.get('garch_vol', 20),
                    "beta_computed":r_data.get('beta_computed'),
                    "hybrid_score": hybrid_score,
                    "ensemble_agreement": round(ensemble_agreement, 1),
                    "confidence_interval": [ci_low, ci_high],
                }
            }

        except Exception as e:
            print(f"[Analyzer] Error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "symbol": symbol, "price": 0, "trust_score": 0, "verdict": "ERROR",
                "reasoning": f"Could not analyze: {str(e)}", "risk": "UNKNOWN",
                "category": "Unknown",
                "advisory": {"entry": "-", "target": "-", "stop_loss": "-", "analyst_rating": "Error"},
                "futures_data": {}, "quant_data": {}
            }

    # ── Helper Methods ──────────────────────────────────────────────────────

    def _safe_run(self, fn, default=None):
        """Run a function with error isolation — quant modules should never crash the main pipeline."""
        try:
            return fn()
        except Exception as e:
            print(f"[Analyzer] Quant module error (non-fatal): {e}")
            return default or {}

    def _run_ml(self, hist: pd.DataFrame) -> dict:
        """Run ML pipeline: build features then predict."""
        feat_df = self.features.build_features(hist.copy())
        if feat_df.empty or len(feat_df) < 50:
            return {"method": "N/A", "direction": "NEUTRAL", "probability": 0.5, "score": 50}
        return self.ml.predict(feat_df)

    def _reconcile_verdict(
        self, traditional_verdict: str, hybrid_score: float,
        arima_dir: str, ml_dir: str, agreement: float
    ) -> str:
        """
        When multiple quant models strongly disagree with traditional verdict,
        adjust the verdict to be more conservative.
        """
        # High confidence from all models → keep traditional
        if agreement > 70:
            return traditional_verdict

        # Both ARIMA and ML say BEARISH but traditional says BUY → downgrade
        if arima_dir == "BEARISH" and ml_dir == "BEARISH":
            if traditional_verdict in ("STRONG BUY", "BUY"):
                return "HOLD"

        # Both ARIMA and ML say BULLISH but traditional says SELL → upgrade
        if arima_dir == "BULLISH" and ml_dir == "BULLISH":
            if traditional_verdict in ("STRONG SELL", "SELL"):
                return "HOLD"

        return traditional_verdict

    def _compute_display_target(
        self, verdict: str, price: float, target_mean: float,
        tech_signal: str, arima_result: dict, prophet_result: dict, mc_result: dict
    ) -> float:
        """Compute display target price using multiple model inputs."""
        arima_price = arima_result.get('forecast_price', 0) or 0
        prophet_price = prophet_result.get('forecast_price', 0) or 0
        mc_median   = mc_result.get('price_63d_median', 0) or 0

        if verdict in ('SELL', 'STRONG SELL', 'AVOID'):
            downside = mc_result.get('scenario_bear', price * 0.92)
            return max(downside, price * 0.92)

        elif verdict in ('BUY', 'STRONG BUY', 'BUY (Contrarian)'):
            candidates = [t for t in [target_mean, arima_price, prophet_price, mc_median] if t > price * 1.03]
            if candidates:
                display = sum(candidates) / len(candidates)  # Average bullish targets
            else:
                display = price * 1.10

            # Safety: BUY target must be at least 5% above entry
            return max(display, price * 1.05)
        else:
            # HOLD
            candidates = [t for t in [target_mean, mc_median] if t > price * 1.02]
            display = sum(candidates) / len(candidates) if candidates else price * 1.05
            return max(display, price * 1.03)

    def _build_report(
        self, tech_signal, tech_score, votes, tech_html,
        fund_signal, fund_score, f_data,
        sent_signal, sent_score, s_data,
        mom_1m, mom_3m, mom_6m, momentum_score,
        risk_lvl, r_data,
        final_verdict, trust_data,
        kalman_signal, arima_result, prophet_result, ml_result, lstm_result, mc_result, bt_result,
        hybrid_score, ensemble_agreement, ci_low, ci_high,
        bull_target, display_target, bear_target, ev_target,
        fib_levels
    ) -> str:
        """Build the full HTML analysis report."""
        fib_html = ""
        if fib_levels and fib_levels.get('fib_618'):
            fib_html = f"""
            <br>
            <div class='analysis-section' style='background:rgba(139,92,246,0.06); border:1px solid rgba(139,92,246,0.15);'>
                <b>🌀 FIBONACCI RETRACEMENTS (52W: ₹{fib_levels.get('low_52w','—')} → ₹{fib_levels.get('high_52w','—')})</b><br>
                <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:4px; font-size:10px; margin-top:5px;">
                    <div>23.6%: ₹{fib_levels.get('fib_236','—')}</div>
                    <div>38.2%: ₹{fib_levels.get('fib_382','—')}</div>
                    <div>50.0%: ₹{fib_levels.get('fib_500','—')}</div>
                    <div>61.8% 🟡: ₹{fib_levels.get('fib_618','—')}</div>
                    <div>78.6%: ₹{fib_levels.get('fib_786','—')}</div>
                </div>
            </div>"""

        kalman_icon = "🔼" if kalman_signal.get('trend_direction') == "UP" else ("🔽" if kalman_signal.get('trend_direction') == "DOWN" else "➡️")
        arima_icon  = "🟢" if arima_result.get('direction') == "BULLISH" else "🔴"
        prophet_icon = "🟢" if prophet_result.get('direction') == "BULLISH" else "🔴"
        ml_icon     = "🟢" if ml_result.get('direction') == "BULLISH" else "🔴"
        lstm_icon   = "🟢" if lstm_result.get('direction') == "BULLISH" else "🔴"
        mc_icon     = "🟢" if mc_result.get('prob_profit', 50) > 55 else "🔴"

        ml_acc = ml_result.get('rf_accuracy')
        ml_acc_str = f"(RF Acc: {ml_acc}%)" if ml_acc else "(Heuristic)"

        return f"""
        <div class='analysis-section'>
            <b>📊 PHASE 1: TECHNICALS ({tech_signal})</b><br>
            Score: {tech_score}/100 |
            Buy: {votes.get('BUY',0)} · Sell: {votes.get('SELL',0)} · Neutral: {votes.get('NEUTRAL',0)}<br>
            {tech_html}
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
            • GARCH Vol (EWMA): {r_data.get('garch_vol', '—')}%<br>
            • Beta vs Nifty50: {r_data.get('beta_computed', r_data.get('beta', '—'))}<br>
            • Sharpe: {r_data.get('sharpe', '—')} | Sortino: {r_data.get('sortino', '—')}<br>
            • VaR 95% (1d): {r_data.get('var_95_1d', '—')}%<br>
            • Stop Loss: ₹{r_data['stop_loss_price']} (-{round(r_data['stop_buffer_pct']*100,1)}%)<br>
            • ATR(14): ₹{r_data.get('atr', '—')}<br>
            • Max Drawdown (20d): {r_data.get('max_drawdown_20d', '—')}%<br>
            • Position Size: {r_data['pos_size_rec']}<br>
            • Risk/Reward: {r_data['risk_reward']}
        </div>
        {fib_html}
        <br>
        <div class='analysis-section' style='background:rgba(6,182,212,0.06); border:1px solid rgba(6,182,212,0.2);'>
            <b>🤖 PHASE 6-9: QUANTITATIVE ENGINE</b><br>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; font-size:11px;">
                <div style="background:rgba(255,255,255,0.04); padding:8px; border-radius:6px;">
                    <div style="color:var(--text-muted); font-size:9px; text-transform:uppercase;">Kalman Filter</div>
                    <div style="font-weight:700;">{kalman_icon} {kalman_signal.get('trend_direction','—')} ({kalman_signal.get('trend_acceleration','—')})</div>
                    <div style="font-size:10px; color:var(--text-muted);">Smoothed: ₹{kalman_signal.get('smoothed_price','—')} | Noise: {kalman_signal.get('noise_level','—')}%</div>
                </div>
                <div style="background:rgba(255,255,255,0.04); padding:8px; border-radius:6px;">
                    <div style="color:var(--text-muted); font-size:9px; text-transform:uppercase;">Time-Series Forecast</div>
                    <div style="font-weight:700;">ARIMA: {arima_icon} {arima_result.get('direction','—')}</div>
                    <div style="font-weight:700;">PROPHET: {prophet_icon} {prophet_result.get('direction','—')}</div>
                    <div style="font-size:10px; color:var(--text-muted);">Conf: A={arima_result.get('confidence',50):.0f}% P={prophet_result.get('confidence',50):.0f}%</div>
                </div>
                <div style="background:rgba(255,255,255,0.04); padding:8px; border-radius:6px;">
                    <div style="color:var(--text-muted); font-size:9px; text-transform:uppercase;">ML Direction Ens. {ml_acc_str}</div>
                    <div style="font-weight:700;">XGB/RF: {ml_icon} {ml_result.get('direction','N/A')}</div>
                    <div style="font-weight:700;">LSTM: {lstm_icon} {lstm_result.get('direction','N/A')}</div>
                    <div style="font-size:10px; color:var(--text-muted);">Agree: {round(ml_result.get('ensemble_agreement',0)*100)}%</div>
                </div>
                <div style="background:rgba(255,255,255,0.04); padding:8px; border-radius:6px;">
                    <div style="color:var(--text-muted); font-size:9px; text-transform:uppercase;">Monte Carlo (3000 paths)</div>
                    <div style="font-weight:700;">{mc_icon} P(Profit): {mc_result.get('prob_profit',50):.1f}%</div>
                    <div style="font-size:10px; color:var(--text-muted);">Sharpe: {mc_result.get('simulated_sharpe',0):.2f} | VaR95: ₹{mc_result.get('var_95_price','—')}</div>
                </div>
            </div>
            <div style="margin-top:8px; font-size:10px; background:rgba(255,255,255,0.03); padding:6px; border-radius:4px;">
                📊 Backtest ({bt_result.get('backtest_period_days','—')} days): Dir. Acc: <b>{bt_result.get('directional_accuracy',50):.1f}%</b> |
                Sharpe: <b>{bt_result.get('sharpe_ratio',0):.2f}</b> |
                Win Rate: <b>{bt_result.get('win_rate',50):.1f}%</b> |
                Alpha vs B&H: <b>{bt_result.get('alpha',0):+.1f}%/yr</b>
            </div>
        </div>
        <br>
        <div class='analysis-section' style='background:rgba(0,212,170,0.05); border:1px solid rgba(0,212,170,0.15);'>
            <b>🎯 HYBRID VERDICT: {final_verdict}</b><br>
            <div style="font-size:11px; margin-top:5px; color:var(--text-muted);">
                {trust_data['reasoning']}
            </div>
            <div style="margin-top:8px; display:grid; grid-template-columns:1fr 1fr 1fr 1fr 1fr; gap:4px; text-align:center; font-size:10px;">
                <div><div style='color:var(--text-muted);'>Tech</div><div style='font-weight:700;'>{round(tech_score)}</div></div>
                <div><div style='color:var(--text-muted);'>Fund</div><div style='font-weight:700;'>{round(fund_score)}</div></div>
                <div><div style='color:var(--text-muted);'>Sent</div><div style='font-weight:700;'>{round(sent_score)}</div></div>
                <div><div style='color:var(--text-muted);'>Mom</div><div style='font-weight:700;'>{round(momentum_score)}</div></div>
                <div><div style='color:var(--text-muted);'>Risk</div><div style='font-weight:700;'>{round(100-r_data['score'])}</div></div>
            </div>
            <div style="margin-top:6px; font-size:10px; color:var(--text-muted); text-align:center;">
                Hybrid Score: <b style="color:var(--accent);">{hybrid_score}/100</b>
                [CI: {ci_low}–{ci_high}] | Ensemble Agreement: {round(ensemble_agreement)}%
            </div>
        </div>
        <br>
        <div class='analysis-section' style='background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.15);'>
            <b>🔮 PROBABILISTIC FORECASTS (6-Month Outlook)</b><br>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:5px;">
                <div>
                    <b style="color:#10b981;">🐂 BULL (20%)</b><br>₹{round(bull_target, 1)}
                    <div style="font-size:9px; color:var(--text-muted);">P(Reach): {mc_result.get('prob_10pct_up','—')}%</div>
                </div>
                <div>
                    <b style="color:#f59e0b;">🏠 BASE (60%)</b><br>₹{round(display_target, 1)}
                    <div style="font-size:9px; color:var(--text-muted);">EV: ₹{round(ev_target, 1)}</div>
                </div>
                <div>
                    <b style="color:#ef4444;">🐻 BEAR (20%)</b><br>₹{round(bear_target, 1)}
                    <div style="font-size:9px; color:var(--text-muted);">P(5% drop): {mc_result.get('prob_5pct_down','—')}%</div>
                </div>
                <div style="border-left:2px solid var(--accent); padding-left:8px;">
                    <b>MC Median (63d)</b><br>₹{mc_result.get('price_63d_median', round(display_target, 1))}
                    <div style="font-size:9px; color:var(--text-muted);">Annual Vol: {r_data.get('garch_vol','—')}% (GARCH)</div>
                </div>
            </div>
        </div>
        """

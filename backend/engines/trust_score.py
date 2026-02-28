class TrustScoreCalculator:
    """
    Trust Score Engine — MarketSmith Composite Rating inspired.
    
    Combines ALL signal pillars into a single actionable score:
    - Technical (30%): Consensus vote from 10 indicators
    - Fundamental (25%): CAN SLIM-style quality + valuation
    - Sentiment (20%): Analyst + Institutional + Retail + Options
    - Momentum (15%): Price performance relative to market
    - Risk-Adjusted (10%): Lower risk = higher confidence
    """
    
    @staticmethod
    def calculate_final_score(tech_score, fund_score, sent_score, risk_score, momentum_score=50):
        """
        All inputs are 0-100 scores. risk_score is inverted (high risk = low contribution).
        Returns composite score, verdict, reasoning, and risk level.
        """
        # Weighted composite
        risk_adjusted = max(0, 100 - risk_score)  # Invert: low risk = high score
        
        composite = (
            (tech_score * 0.30) +
            (fund_score * 0.25) +
            (sent_score * 0.20) +
            (momentum_score * 0.15) +
            (risk_adjusted * 0.10)
        )
        
        # ── Verdict Logic ──
        verdict = "HOLD"
        reasoning = ""
        risk_level = "Medium"
        
        # Strong signals
        if composite >= 75:
            verdict = "STRONG BUY"
            reasoning = "All pillars aligned: strong technicals, solid fundamentals, positive sentiment, and good momentum."
            risk_level = "Low"
        elif composite >= 60:
            verdict = "BUY"
            reasoning = "Majority of signals are bullish. Technical and fundamental support is present."
            risk_level = "Medium-Low"
        elif composite >= 45:
            verdict = "HOLD"
            reasoning = "Mixed signals. Some indicators are bullish but others are cautionary."
            risk_level = "Medium"
        elif composite >= 30:
            verdict = "SELL"
            reasoning = "Majority of signals are bearish. Weakening technicals and/or fundamentals."
            risk_level = "Medium-High"
        else:
            verdict = "STRONG SELL"
            reasoning = "Broad weakness across all pillars. Avoid new positions."
            risk_level = "High"
        
        # ── Trap Detection ──
        # Retail Trap: High sentiment but weak technicals + fundamentals
        if sent_score > 70 and tech_score < 40 and fund_score < 40:
            verdict = "AVOID"
            reasoning = "⚠️ RETAIL TRAP: Social hype without technical or fundamental backing."
            risk_level = "High"
            composite *= 0.75  # Penalty
        
        # Contrarian Buy: Strong technicals but depressed sentiment (smart money accumulation)
        if tech_score > 75 and sent_score < 30 and fund_score > 50:
            verdict = "BUY (Contrarian)"
            reasoning = "📊 Contrarian setup: Strong technicals + good fundamentals despite negative sentiment. Potential smart money accumulation."
            risk_level = "Medium"
        
        # Value Trap: Great fundamentals but terrible technicals (broken chart)
        if fund_score > 70 and tech_score < 25:
            if verdict == "BUY" or verdict == "STRONG BUY":
                verdict = "HOLD"
                reasoning = "⚠️ VALUE TRAP RISK: Good fundamentals but chart is broken. Wait for technical reversal."
                risk_level = "Medium-High"
        
        # Momentum play: Weak fundamentals but everything else is bullish
        if fund_score < 40 and tech_score > 70 and momentum_score > 70:
            if verdict == "STRONG BUY":
                verdict = "BUY"
                reasoning = "📈 Momentum play: Technicals are strong but fundamentals are weak. Use tight stop losses."
                risk_level = "Medium-High"
        
        return {
            "trust_score": round(composite, 2),
            "verdict": verdict,
            "reasoning": reasoning,
            "risk_level": risk_level,
            "pillar_breakdown": {
                "technical": round(tech_score, 1),
                "fundamental": round(fund_score, 1),
                "sentiment": round(sent_score, 1),
                "momentum": round(momentum_score, 1),
                "risk_adj": round(risk_adjusted, 1)
            }
        }

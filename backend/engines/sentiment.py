class SentimentEngine:
    """
    Market Sentiment Analyst - 5-Pillar Analysis.
    Weights: Analyst(30%) + Retail(20%) + Inst(30%) + Earnings(10%) + Options(10%)
    """
    
    def analyze_sentiment(self, info: dict, fno_data: dict, vol_spike: float):
        try:
            # 1. ANALYST CONSENSUS (30%)
            rec_key = info.get('recommendationKey', 'none').lower()
            target_mean = info.get('targetMeanPrice', 0)
            current_price = info.get('currentPrice', 0)
            
            analyst_score = 50
            if 'strong_buy' in rec_key: analyst_score = 100
            elif 'buy' in rec_key: analyst_score = 80
            elif 'underperform' in rec_key: analyst_score = 10
            elif 'sell' in rec_key: analyst_score = 20
            
            # Upside potential check
            if current_price > 0 and target_mean > current_price:
                upside = (target_mean - current_price) / current_price
                if upside > 0.2: analyst_score += 10 # Extra points for >20% upside
                
            # 2. RETAIL SENTIMENT (20%) - Proxied by Volume Spike & Momentum
            # Real social listening requires external APIs (Twitter/Reddit)
            retail_score = 50
            if vol_spike > 2.0:
                retail_score = 80 # High interest/hype
            elif vol_spike < 0.6:
                retail_score = 40 # Disinterest
                
            # 3. INSTITUTIONAL FLOWS (30%)
            inst_hold = info.get('heldPercentInstitutions', 0)
            inst_score = 50
            if inst_hold > 0.6: inst_score = 90
            elif inst_hold > 0.4: inst_score = 70
            elif inst_hold < 0.1: inst_score = 30
            
            # 4. EARNINGS REVISION (10%) - Simplified via Forward PE comparison
            # If Forward PE < Trailing PE, broadly implies growing earnings expectation
            trailing_eps = info.get('trailingEps', 0)
            forward_eps = info.get('forwardEps', 0)
            earn_score = 50
            if forward_eps > trailing_eps: earn_score = 75
            elif forward_eps < trailing_eps: earn_score = 25
            
            # 5. OPTIONS MARKET (10%)
            # Use data from FuturesEngine (PCR, Sentiment)
            opt_score = 50
            if fno_data:
                pcr = fno_data.get('pcr', 1.0)
                if pcr > 1.2: opt_score = 70 # (Bullish Put Writing)
                elif pcr < 0.6: opt_score = 30 # (Bearish Call Writing)
            
            # COMPOSITE SENTIMENT SCORE
            # Score: (Analyst×0.3) + (Retail×0.2) + (Institutional×0.3) + (Earnings×0.1) + (Options×0.1)
            final_sentiment = (
                (analyst_score * 0.3) +
                (retail_score * 0.2) +
                (inst_score * 0.3) +
                (earn_score * 0.1) +
                (opt_score * 0.1)
            )
            
            return {
                "score": round(final_sentiment, 1),
                "analyst_rating": rec_key.upper().replace("_", " "),
                "inst_hold": round(inst_hold*100, 1) if inst_hold else 0,
                "retail_interest": "High" if retail_score > 60 else "Normal"
            }
            
        except Exception:
            return {"score": 50, "analyst_rating": "NEUTRAL", "inst_hold": 0, "retail_interest": "Unknown"}

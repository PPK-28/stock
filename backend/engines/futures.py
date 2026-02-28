"""
Futures & Options Analysis Engine
---------------------------------
Derives F&O signals from actual stock data (volume, volatility, price action)
instead of random data. Uses volume profile and price momentum as proxies
for OI buildup and PCR since real-time F&O data requires paid APIs.
"""

class FuturesEngine:
    # NSE F&O stocks list (most actively traded)
    FNO_STOCKS = {
        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'TATAMOTORS',
        'ITC', 'BHARTIARTL', 'LICI', 'MARUTI', 'M&M', 'ASIANPAINT', 'LT',
        'AXISBANK', 'KOTAKBANK', 'WIPRO', 'HCLTECH', 'TATASTEEL', 'NTPC',
        'ONGC', 'POWERGRID', 'ULTRACEMCO', 'BAJFINANCE', 'BAJAJFINSV',
        'SUNPHARMA', 'TITAN', 'INDUSINDBK', 'TECHM', 'ADANIPORTS',
        'DIVISLAB', 'NESTLEIND', 'DRREDDY', 'CIPLA', 'APOLLOHOSP',
        'HEROMOTOCO', 'EICHERMOT', 'JSWSTEEL', 'GRASIM', 'BPCL',
        'HINDALCO', 'COALINDIA', 'YESBANK', 'IDEA', 'TMPV',
        'SILVERBEES', 'GOLDBEES'
    }
    
    def analyze_derivatives(self, symbol, current_price, volume_spike_pct, 
                           rsi=50, price_change_5d=0, avg_volume_ratio=1.0):
        """
        Analyze derivatives positioning using price-derived proxy signals.
        
        Instead of random data, we derive F&O sentiment from:
        1. Volume spike (proxy for OI change)
        2. RSI momentum (proxy for directional conviction)
        3. 5-day price change (proxy for trend)
        4. Volume ratio (proxy for institutional activity)
        """
        base_sym = symbol.replace('.NS', '').replace('.BO', '')
        is_fno = base_sym in self.FNO_STOCKS
        
        if not is_fno:
            return {
                "is_fno": False,
                "status": "Cash Only",
                "sentiment": "Neutral",
                "oi_signal": "N/A",
                "pcr": 0,
                "pcr_signal": "N/A",
                "reasoning": f"{base_sym} is not in the F&O segment."
            }
        
        # ── Derive OI Signal from Volume + Price Action ──
        # Logic: Volume spike + price up = Long Buildup
        #        Volume spike + price down = Short Buildup
        #        Volume drop + price up = Short Covering
        #        Volume drop + price down = Long Unwinding
        
        high_volume = volume_spike_pct > 20 or avg_volume_ratio > 1.3
        price_up = price_change_5d > 0
        
        if high_volume and price_up:
            oi_signal = "Long Buildup"
            sentiment = "Bullish"
        elif high_volume and not price_up:
            oi_signal = "Short Buildup"
            sentiment = "Bearish"
        elif not high_volume and price_up:
            oi_signal = "Short Covering"
            sentiment = "Mildly Bullish"
        else:
            oi_signal = "Long Unwinding"
            sentiment = "Bearish"
        
        # ── Derive PCR from RSI and volume ──
        # PCR > 1.0 = More puts = Bullish (support)
        # PCR < 0.7 = More calls = Bearish (resistance)
        # Derived: RSI > 50 and rising volume suggests put writers are active (bullish PCR)
        
        if rsi > 60 and high_volume:
            pcr = round(1.1 + (rsi - 60) * 0.01, 2)  # 1.1 - 1.5 range
        elif rsi > 50:
            pcr = round(0.85 + (rsi - 50) * 0.02, 2)  # 0.85 - 1.05 range
        elif rsi > 40:
            pcr = round(0.7 + (rsi - 40) * 0.015, 2)  # 0.7 - 0.85 range
        else:
            pcr = round(0.5 + rsi * 0.005, 2)  # 0.5 - 0.7 range
        
        # PCR signal
        if pcr > 1.2:
            pcr_signal = "Very Bullish"
        elif pcr > 0.9:
            pcr_signal = "Neutral"
        elif pcr > 0.7:
            pcr_signal = "Mildly Bearish"
        else:
            pcr_signal = "Bearish"
        
        # Override sentiment for extreme RSI
        if rsi > 75:
            sentiment = "Overbought Risk"
        elif rsi < 25:
            sentiment = "Oversold Opportunity"
        
        # Build reasoning
        reasoning = f"{oi_signal} detected. "
        if oi_signal == "Long Buildup":
            reasoning += "Strong buying with rising volume confirms bullish positioning. "
        elif oi_signal == "Short Buildup":
            reasoning += "Bears are adding positions aggressively. Caution advised. "
        elif oi_signal == "Short Covering":
            reasoning += "Shorts are covering — rally may lack fresh buying conviction. "
        else:
            reasoning += "Bulls are booking profits and exiting. "
        
        reasoning += f"PCR at {pcr} suggests {pcr_signal} sentiment."
        
        return {
            "is_fno": True,
            "status": "Active",
            "sentiment": sentiment,
            "oi_signal": oi_signal,
            "pcr": pcr,
            "pcr_signal": pcr_signal,
            "reasoning": reasoning
        }

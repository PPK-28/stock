import pandas as pd
import numpy as np

class TechnicalEngine:
    """
    Technical Analysis Engine — TradingView-grade.
    Uses consensus voting across 10+ indicators:
    ADX, RSI, Stochastic RSI, MACD, SuperTrend, OBV,
    EMA 9/21 Cross, SMA 50/200, Bollinger Bands, VWAP.
    """
    
    def calculate_indicators(self, df: pd.DataFrame):
        """Calculate all technical indicators on the price DataFrame."""
        try:
            # ── Moving Averages ──
            df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
            df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            df['SMA200'] = df['Close'].rolling(window=200).mean()
            df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
            
            # VWAP Rolling Approximation
            df['Rolling_VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).rolling(20).sum() / df['Volume'].rolling(20).sum()

            # ── RSI (14) ──
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # ── Stochastic RSI ──
            rsi_min = df['RSI'].rolling(14).min()
            rsi_max = df['RSI'].rolling(14).max()
            df['StochRSI_K'] = ((df['RSI'] - rsi_min) / (rsi_max - rsi_min)) * 100
            df['StochRSI_D'] = df['StochRSI_K'].rolling(3).mean()
            
            # ── MACD ──
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal']
            
            # ── ADX (Average Directional Index — Trend Strength) ──
            high_diff = df['High'].diff()
            low_diff = -df['Low'].diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift()).abs()
            tr3 = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()
            
            plus_di = 100 * (plus_dm.rolling(14).mean() / df['ATR'])
            minus_di = 100 * (minus_dm.rolling(14).mean() / df['ATR'])
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            df['ADX'] = dx.rolling(14).mean()
            df['Plus_DI'] = plus_di
            df['Minus_DI'] = minus_di
            
            # ── SuperTrend ──
            hl2 = (df['High'] + df['Low']) / 2
            atr = df['ATR']
            upper_band = hl2 + (2 * atr)
            lower_band = hl2 - (2 * atr)
            
            supertrend = pd.Series(index=df.index, dtype=float)
            direction = pd.Series(index=df.index, dtype=int)
            
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > upper_band.iloc[i-1]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1  # Bullish
                elif df['Close'].iloc[i] < lower_band.iloc[i-1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1  # Bearish
                else:
                    if i > 0 and not pd.isna(supertrend.iloc[i-1]):
                        supertrend.iloc[i] = supertrend.iloc[i-1]
                        direction.iloc[i] = direction.iloc[i-1]
                    else:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1
                        
            df['SuperTrend'] = supertrend
            df['ST_Direction'] = direction
            
            # ── OBV (On-Balance Volume) ──
            obv = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.append(obv[-1] + df['Volume'].iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.append(obv[-1] - df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            df['OBV'] = obv
            df['OBV_EMA20'] = pd.Series(obv).ewm(span=20, adjust=False).mean().values
            
            # ── Bollinger Bands ──
            df['BB_Mid'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
            df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])

            # ── Volume Analysis ──
            df['Vol_Avg_20'] = df['Volume'].rolling(window=20).mean()
            df['Vol_Spike'] = df['Volume'] / df['Vol_Avg_20']
            
            # ── Momentum (Relative Strength vs self) ──
            df['Momentum_1M'] = df['Close'].pct_change(21) * 100
            df['Momentum_3M'] = df['Close'].pct_change(63) * 100
            df['Momentum_6M'] = df['Close'].pct_change(126) * 100
            
            return df
        except Exception as e:
            print(f"Technical Indicator Error: {e}")
            return pd.DataFrame()

    def get_technical_report(self, df):
        """
        TradingView-style consensus voting report.
        Each indicator votes BUY / SELL / NEUTRAL.
        Final verdict = majority vote.
        """
        last = df.iloc[-1]
        prev = df.iloc[-2]
        price = last['Close']
        
        votes = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
        rows = []
        
        # ── 1. RSI (14) ──
        rsi = last['RSI']
        if rsi > 70:
            votes["SELL"] += 1
            status = "Overbought"
            color = "sell"
        elif rsi < 30:
            votes["BUY"] += 1
            status = "Oversold → Buy"
            color = "buy"
        elif rsi > 50:
            votes["BUY"] += 1
            status = "Bullish Momentum"
            color = "buy"
        else:
            votes["SELL"] += 1
            status = "Bearish Momentum"
            color = "sell"
        rows.append(("RSI", f"{round(rsi,1)}", status, color))
        
        # ── 2. Stochastic RSI ──
        stoch_k = last.get('StochRSI_K', 50)
        stoch_d = last.get('StochRSI_D', 50)
        if stoch_k < 20:
            votes["BUY"] += 1
            status = "Oversold → Reversal"
            color = "buy"
        elif stoch_k > 80:
            votes["SELL"] += 1
            status = "Overbought"
            color = "sell"
        elif stoch_k > stoch_d:
            votes["BUY"] += 1
            status = "K crossed above D"
            color = "buy"
        else:
            votes["SELL"] += 1
            status = "K below D"
            color = "sell"
        rows.append(("Stoch RSI", f"K:{round(stoch_k,0)} D:{round(stoch_d,0)}", status, color))
        
        # ── 3. MACD ──
        macd_hist = last.get('MACD_Hist', 0)
        prev_hist = prev.get('MACD_Hist', 0)
        if last['MACD'] > last['Signal']:
            if macd_hist > prev_hist:
                votes["BUY"] += 1
                status = "Bullish + Accelerating"
                color = "buy"
            else:
                votes["BUY"] += 1
                status = "Bullish Crossover"
                color = "buy"
        else:
            if macd_hist < prev_hist:
                votes["SELL"] += 1
                status = "Bearish + Accelerating"
                color = "sell"
            else:
                votes["SELL"] += 1
                status = "Bearish Crossover"
                color = "sell"
        rows.append(("MACD", f"Hist: {round(macd_hist, 2)}", status, color))
        
        # ── 4. ADX (Trend Strength) ──
        adx = last.get('ADX', 20)
        plus_di = last.get('Plus_DI', 25)
        minus_di = last.get('Minus_DI', 25)
        if adx > 25:
            if plus_di > minus_di:
                votes["BUY"] += 1
                status = f"Strong Uptrend (ADX {round(adx,0)})"
                color = "buy"
            else:
                votes["SELL"] += 1
                status = f"Strong Downtrend (ADX {round(adx,0)})"
                color = "sell"
        else:
            votes["NEUTRAL"] += 1
            status = f"Weak/No Trend (ADX {round(adx,0)})"
            color = "neutral"
        rows.append(("ADX", f"{round(adx,1)}", status, color))
        
        # ── 5. SuperTrend ──
        st_dir = last.get('ST_Direction', 0)
        if st_dir == 1:
            votes["BUY"] += 1
            status = "Bullish (Price > SuperTrend)"
            color = "buy"
        elif st_dir == -1:
            votes["SELL"] += 1
            status = "Bearish (Price < SuperTrend)"
            color = "sell"
        else:
            votes["NEUTRAL"] += 1
            status = "Neutral"
            color = "neutral"
        rows.append(("SuperTrend", "2×ATR", status, color))
        
        # ── 6. EMA 9/21 Crossover ──
        if last['EMA9'] > last['EMA21']:
            votes["BUY"] += 1
            status = "EMA9 > EMA21 (Short-Term Bull)"
            color = "buy"
        else:
            votes["SELL"] += 1
            status = "EMA9 < EMA21 (Short-Term Bear)"
            color = "sell"
        rows.append(("EMA Cross", "9/21", status, color))
        
        # ── 7. SMA 50/200 (Golden/Death Cross) ──
        if price > last['SMA50'] and last['SMA50'] > last['SMA200']:
            votes["BUY"] += 1
            status = "Golden Cross (Strong Bull)"
            color = "buy"
        elif price < last['SMA50'] and last['SMA50'] < last['SMA200']:
            votes["SELL"] += 1
            status = "Death Cross (Strong Bear)"
            color = "sell"
        elif price > last['SMA200']:
            votes["BUY"] += 1
            status = "Above SMA200 (Long-Term Bull)"
            color = "buy"
        else:
            votes["SELL"] += 1
            status = "Below SMA200 (Long-Term Bear)"
            color = "sell"
        rows.append(("SMA 50/200", "Golden/Death", status, color))
        
        # ── 8. Bollinger Bands ──
        bb_width = (last['BB_Upper'] - last['BB_Lower']) / last['BB_Mid']
        if price > last['BB_Upper']:
            votes["BUY"] += 1
            status = "Upper Breakout"
            color = "buy"
        elif price < last['BB_Lower']:
            votes["BUY"] += 1
            status = "Oversold (Below Lower Band)"
            color = "buy"
        else:
            votes["NEUTRAL"] += 1
            status = f"Inside Bands (Width: {round(bb_width*100,1)}%)"
            color = "neutral"
        rows.append(("Bollinger", f"Width: {round(bb_width*100,1)}%", status, color))
        
        # ── 9. OBV (Volume Confirmation) ──
        obv = last.get('OBV', 0)
        obv_ema = last.get('OBV_EMA20', 0)
        if obv > obv_ema:
            votes["BUY"] += 1
            status = "Accumulation (OBV > EMA)"
            color = "buy"
        else:
            votes["SELL"] += 1
            status = "Distribution (OBV < EMA)"
            color = "sell"
        rows.append(("OBV", "Volume Flow", status, color))
        
        # ── 10. VWAP ──
        vwap = last.get('Rolling_VWAP', price)
        if price > vwap:
            votes["BUY"] += 1
            status = "Above VWAP (Institutional Buy)"
            color = "buy"
        else:
            votes["SELL"] += 1
            status = "Below VWAP (Institutional Sell)"
            color = "sell"
        rows.append(("VWAP", f"₹{round(vwap, 1)}", status, color))
        
        # ── Consensus Score ──
        total_votes = votes["BUY"] + votes["SELL"] + votes["NEUTRAL"]
        score = round((votes["BUY"] / total_votes) * 100) if total_votes > 0 else 50
        
        # Build HTML report
        color_map = {"buy": "#10b981", "sell": "#ef4444", "neutral": "#f59e0b"}
        report_html = f"""
        <div style="margin-bottom:8px; font-size:11px; color:var(--text-muted);">
            Consensus: <b style="color:#10b981;">{votes['BUY']} Buy</b> · 
            <b style="color:#ef4444;">{votes['SELL']} Sell</b> · 
            <b style="color:#f59e0b;">{votes['NEUTRAL']} Neutral</b>
        </div>
        <table style="width:100%; border-collapse:collapse; font-size:11px;">
            <tr style="background:rgba(255,255,255,0.06);">
                <th style="text-align:left; padding:4px;">Indicator</th>
                <th style="text-align:left; padding:4px;">Value</th>
                <th style="text-align:left; padding:4px;">Signal</th>
            </tr>
        """
        for name, value, status, color in rows:
            c = color_map.get(color, "#94a3b8")
            report_html += f"""
            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                <td style="padding:4px;">{name}</td>
                <td style="padding:4px; font-family:monospace;">{value}</td>
                <td style="padding:4px; color:{c};">{status}</td>
            </tr>
            """
        report_html += "</table>"
        
        return {
            "score": score,
            "votes": votes,
            "report_html": report_html
        }

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
            
            # ── SuperTrend (Fully Vectorized) ──
            hl2 = (df['High'] + df['Low']) / 2
            atr = df['ATR']
            raw_upper = hl2 + (2 * atr)
            raw_lower = hl2 - (2 * atr)

            # Vectorized SuperTrend via NumPy iteration (no iloc penalty)
            close_arr = df['Close'].values
            upper_arr = raw_upper.values
            lower_arr = raw_lower.values
            supertrend_arr = np.full(len(df), np.nan)
            direction_arr = np.zeros(len(df), dtype=int)

            # Seed first valid value
            first_valid = np.argmax(~np.isnan(upper_arr))
            supertrend_arr[first_valid] = lower_arr[first_valid]
            direction_arr[first_valid] = 1

            for i in range(first_valid + 1, len(df)):
                prev_st = supertrend_arr[i - 1]
                if np.isnan(prev_st):
                    supertrend_arr[i] = lower_arr[i]
                    direction_arr[i] = 1
                elif close_arr[i] > prev_st and direction_arr[i - 1] == -1:
                    supertrend_arr[i] = lower_arr[i]  # Bullish flip
                    direction_arr[i] = 1
                elif close_arr[i] < prev_st and direction_arr[i - 1] == 1:
                    supertrend_arr[i] = upper_arr[i]  # Bearish flip
                    direction_arr[i] = -1
                else:
                    direction_arr[i] = direction_arr[i - 1]
                    supertrend_arr[i] = (
                        min(lower_arr[i], prev_st) if direction_arr[i] == 1
                        else max(upper_arr[i], prev_st)
                    )

            df['SuperTrend'] = supertrend_arr
            df['ST_Direction'] = direction_arr

            # ── OBV (On-Balance Volume) — Vectorized via np.where + cumsum ──
            close_diff = df['Close'].diff()
            volume_sign = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
            obv_values = (volume_sign * df['Volume'].values).cumsum()
            df['OBV'] = obv_values
            df['OBV_EMA20'] = pd.Series(obv_values, index=df.index).ewm(span=20, adjust=False).mean()
            
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

            # ── Fibonacci Retracement Levels (52-week high-low) ──
            HIGH_52W = df['High'].rolling(252, min_periods=50).max()
            LOW_52W  = df['Low'].rolling(252, min_periods=50).min()
            fib_range = HIGH_52W - LOW_52W
            df['Fib_236'] = HIGH_52W - 0.236 * fib_range   # 23.6% retracement
            df['Fib_382'] = HIGH_52W - 0.382 * fib_range   # 38.2% retracement
            df['Fib_500'] = HIGH_52W - 0.500 * fib_range   # 50.0% retracement
            df['Fib_618'] = HIGH_52W - 0.618 * fib_range   # 61.8% retracement (golden ratio)
            df['Fib_786'] = HIGH_52W - 0.786 * fib_range   # 78.6% retracement
            df['Fib_High_52w'] = HIGH_52W
            df['Fib_Low_52w']  = LOW_52W

            # ── Williams %R (overbought/oversold oscillator) ──
            highest_high = df['High'].rolling(14).max()
            lowest_low   = df['Low'].rolling(14).min()
            df['Williams_R'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low + 1e-9)

            # ── Chaikin Money Flow (21-day) ──
            mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9)
            df['CMF'] = (mfm * df['Volume']).rolling(21).sum() / df['Volume'].rolling(21).sum()

            return df
        except Exception as e:
            print(f"Technical Indicator Error: {e}")
            import traceback; traceback.print_exc()
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
        
        # ── 11. Fibonacci Retracement ──
        fib_618 = last.get('Fib_618', price)
        fib_382 = last.get('Fib_382', price)
        fib_236 = last.get('Fib_236', price)
        fib_high = last.get('Fib_High_52w', price)
        if not pd.isna(fib_618) and not pd.isna(fib_382):
            if price >= fib_236:
                votes["BUY"] += 1
                status = f"Above 23.6% Fib (₹{round(fib_236,1)}) — Uptrend zone"
                color = "buy"
            elif price >= fib_382:
                votes["NEUTRAL"] += 1
                status = f"38.2% Fib Support (₹{round(fib_382,1)}) — Watch zone"
                color = "neutral"
            elif price >= fib_618:
                votes["NEUTRAL"] += 1
                status = f"Golden Ratio 61.8% (₹{round(fib_618,1)}) — Key support"
                color = "neutral"
            else:
                votes["SELL"] += 1
                status = f"Below 61.8% Fib — Downtrend, next support ₹{round(last.get('Fib_Low_52w', price),1)}"
                color = "sell"
            rows.append(("Fibonacci", f"52W: ₹{round(fib_high,1)}", status, color))

        # ── 12. Williams %R ──
        williams_r = last.get('Williams_R', -50)
        if not pd.isna(williams_r):
            if williams_r < -80:
                votes["BUY"] += 1
                status = f"Oversold ({round(williams_r,1)}) → Reversal potential"
                color = "buy"
            elif williams_r > -20:
                votes["SELL"] += 1
                status = f"Overbought ({round(williams_r,1)}) → Profit taking risk"
                color = "sell"
            elif williams_r > -50:
                votes["BUY"] += 1
                status = f"Bullish zone ({round(williams_r,1)})"
                color = "buy"
            else:
                votes["SELL"] += 1
                status = f"Bearish zone ({round(williams_r,1)})"
                color = "sell"
            rows.append(("Williams %R", f"{round(williams_r,1)}", status, color))

        # ── 13. Chaikin Money Flow ──
        cmf = last.get('CMF', 0)
        if not pd.isna(cmf):
            if cmf > 0.1:
                votes["BUY"] += 1
                status = f"Strong buying pressure (CMF: {round(cmf,2)})"
                color = "buy"
            elif cmf < -0.1:
                votes["SELL"] += 1
                status = f"Selling pressure (CMF: {round(cmf,2)})"
                color = "sell"
            else:
                votes["NEUTRAL"] += 1
                status = f"Neutral money flow (CMF: {round(cmf,2)})"
                color = "neutral"
            rows.append(("CMF", f"{round(cmf,3)}", status, color))

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

        # Fibonacci levels for UI
        fib_levels = {
            "fib_236": round(float(fib_236), 2) if not pd.isna(fib_236) else None,
            "fib_382": round(float(fib_382), 2) if not pd.isna(fib_382) else None,
            "fib_500": round(float(last.get('Fib_500', price)), 2),
            "fib_618": round(float(fib_618), 2) if not pd.isna(fib_618) else None,
            "fib_786": round(float(last.get('Fib_786', price)), 2),
            "high_52w": round(float(fib_high), 2) if not pd.isna(fib_high) else None,
            "low_52w": round(float(last.get('Fib_Low_52w', price)), 2),
        }
        
        return {
            "score": score,
            "votes": votes,
            "report_html": report_html,
            "fib_levels": fib_levels,
        }
